"""
WhisperX SRT Generation Node for ComfyUI
Converts aligned audio-text to SRT subtitle format
Based on align_to_srt.py script functionality
"""

import os
import json
import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Union, Optional

# Configure Hugging Face mirror for better download speed
from .hf_config import auto_configure_mirror, get_mirror_status
from .i18n_utils import (
    get_localized_input_types, 
    get_localized_return_names, 
    get_localized_category,
    get_localized_display_name
)

# Auto-configure HF mirror on import
auto_configure_mirror()

# Import ComfyUI's folder_paths for model directory access
try:
    import folder_paths
except ImportError:
    # Fallback if running outside ComfyUI
    class folder_paths:
        @staticmethod
        def models_dir():
            return "./models"

def convert_audio_to_whisperx_format(audio: Dict[str, Any]) -> Tuple[np.ndarray, float]:
    """
    Convert ComfyUI official audio format to WhisperX format.

    Args:
        audio: Audio dict from ComfyUI's official audio loader
               Expected format: {"waveform": tensor, "sample_rate": int}

    Returns:
        Tuple of (numpy array in WhisperX format (16kHz mono), duration in seconds)
    """
    waveform = audio["waveform"]
    sample_rate = audio["sample_rate"]

    print(f"[DEBUG] Original audio shape: {waveform.shape}, sample_rate: {sample_rate}")

    # Convert to torch tensor if needed
    if isinstance(waveform, np.ndarray):
        waveform = torch.from_numpy(waveform)

    # Ensure tensor is float32
    if waveform.dtype != torch.float32:
        waveform = waveform.float()

    # Handle different audio formats
    if waveform.dim() == 3:
        # Remove batch dimension if present
        waveform = waveform.squeeze(0)
    
    if waveform.dim() == 2:
        # Convert to mono if stereo (take mean of channels)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=False)
        else:
            waveform = waveform.squeeze(0)
    
    # Ensure we have a 1D tensor
    if waveform.dim() > 1:
        waveform = waveform.flatten()

    print(f"[DEBUG] After processing shape: {waveform.shape}")

    # Calculate original duration
    original_duration = waveform.shape[0] / sample_rate
    print(f"[DEBUG] Original duration: {original_duration:.2f} seconds")

    # Resample to 16kHz if needed (WhisperX requirement)
    if sample_rate != 16000:
        print(f"Converting audio from {sample_rate}Hz to WhisperX format (16kHz mono)")
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)

    # Convert to numpy
    audio_array = waveform.numpy()
    
    # Calculate final duration at 16kHz
    final_duration = len(audio_array) / 16000
    print(f"[DEBUG] Final audio shape: {audio_array.shape}, duration: {final_duration:.2f} seconds")

    if final_duration == 0:
        raise ValueError("Audio duration is 0 seconds. Please check your audio input.")

    return audio_array, final_duration

def select_device(choice: str) -> str:
    """Select device based on choice and availability."""
    if choice == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return choice

def flush_line(buffer: List[Dict], subtitles: List[Tuple[float, float, str]]):
    """Flush current buffer to subtitles list."""
    if not buffer:
        return
    start = buffer[0]["start"]
    end = buffer[-1]["end"]
    text = "".join(item["word"] for item in buffer).strip()
    if text:
        subtitles.append((start, end, text))
    buffer.clear()

def create_srt_content(subtitles: List[Tuple[float, float, str]]) -> str:
    """Create SRT format content from subtitle list."""
    srt_lines = []
    
    for idx, (start, end, text) in enumerate(subtitles, 1):
        # Convert seconds to SRT time format (HH:MM:SS,mmm)
        def seconds_to_srt_time(seconds):
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            milliseconds = int((seconds % 1) * 1000)
            return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"
        
        start_time = seconds_to_srt_time(start)
        end_time = seconds_to_srt_time(end)
        
        srt_lines.append(f"{idx}")
        srt_lines.append(f"{start_time} --> {end_time}")
        srt_lines.append(text)
        srt_lines.append("")  # Empty line between subtitles
    
    return "\n".join(srt_lines)

# Define WhisperX models directory
WHISPERX_MODELS_DIR = os.path.join(folder_paths.models_dir, "whisperx")

# Language to model folder name mapping
LANGUAGE_MODEL_MAP = {
    "en": "wav2vec2-large-xlsr-53-english",
    "zh": "wav2vec2-large-xlsr-53-chinese-zh-cn",
    "fr": "wav2vec2-large-xlsr-53-french",
    "de": "wav2vec2-large-xlsr-53-german",
    "es": "wav2vec2-large-xlsr-53-spanish",
    "it": "wav2vec2-large-xlsr-53-italian",
    "pt": "wav2vec2-large-xlsr-53-portuguese",
    "ja": "wav2vec2-large-xlsr-53-japanese",
    "nl": "wav2vec2-large-xlsr-53-dutch",
}

def load_align_model(language_code: str, device: str) -> Tuple[Any, Any]:
    """
    Load alignment model from ComfyUI models directory.
    
    Args:
        language_code: Language code for the alignment model
        device: Device to load model on
        
    Returns:
        Tuple of (model, metadata)
    """
    try:
        import whisperx
    except ImportError:
        raise ImportError(
            "WhisperX is not installed. Please install it using:\n"
            "pip install git+https://github.com/m-bain/whisperx.git"
        )
    
    # Try to load from local directory first
    if language_code in LANGUAGE_MODEL_MAP:
        model_folder = LANGUAGE_MODEL_MAP[language_code]
        model_path = os.path.join(WHISPERX_MODELS_DIR, model_folder)
        
        if os.path.exists(model_path):
            print(f"[INFO] Loading alignment model from local path: {model_path}")
            try:
                from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
                model = Wav2Vec2ForCTC.from_pretrained(model_path).to(device)
                processor = Wav2Vec2Processor.from_pretrained(model_path)
                metadata = {
                    "language": language_code,
                    "model_path": model_path,
                    "model_name": model_folder,
                }
                return model, processor
            except Exception as e:
                print(f"[WARNING] Failed to load local model: {e}")
    
    # Fallback to WhisperX's default loading (downloads from HuggingFace)
    print(f"[INFO] Loading alignment model from HuggingFace for language: {language_code}")
    return whisperx.load_align_model(language_code=language_code, device=device)

class WhisperXSRTNode:
    """
    A ComfyUI node that aligns text with audio and generates SRT subtitles.
    Based on the align_to_srt.py script functionality.
    """
    align_model = None
    metadata = None
    current_language = None

    def __init__(self):
        print("[INFO] WhisperX SRT initialized.")

    @classmethod
    def INPUT_TYPES(cls):
        return get_localized_input_types()

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("SRT字幕内容 / SRT Content", "对齐信息 / Alignment Info")
    FUNCTION = "align_to_srt"
    CATEGORY = "音频处理 / Audio"

    def align_to_srt(
        self,
        audio: Dict[str, Any],
        text: str,
        language: str,
        max_sec: float,
        max_ch: int,
        punct: str,
        device: str = "auto",
        unload_model: bool = False
    ) -> Tuple[str, str]:
        """
        Align text with audio and generate SRT subtitles.
        
        Args:
            audio: Audio data dictionary from ComfyUI audio loader
            text: Text to align with audio
            language: Language code for alignment model
            max_sec: Maximum duration per subtitle line (seconds)
            max_ch: Maximum characters per subtitle line
            punct: Punctuation marks that trigger line breaks
            device: Device to run on (auto, cuda, or cpu)
            
        Returns:
            Tuple of (srt_content, alignment_info)
        """
        try:
            import whisperx
            from whisperx.audio import SAMPLE_RATE
        except ImportError:
            raise ImportError(
                "WhisperX is not installed. Please install it using:\n"
                "pip install git+https://github.com/m-bain/whisperx.git"
            )

        # Validate inputs
        if not audio or not isinstance(audio, dict):
            raise ValueError("Audio input must be a dictionary from ComfyUI audio loader")

        if "waveform" not in audio or "sample_rate" not in audio:
            raise ValueError("Audio data must contain 'waveform' and 'sample_rate' keys")

        if not text or not text.strip():
            raise ValueError("Text input cannot be empty")

        # Select device
        device = select_device(device)
        print(f"[INFO] Using device: {device}")
        print(f"[INFO] Alignment language: {language}")

        # Convert audio to WhisperX format
        audio_array, audio_duration = convert_audio_to_whisperx_format(audio)
        print(f"[INFO] Audio duration: {audio_duration:.2f} seconds")

        # Clean text
        txt = text.strip()
        print(f"[INFO] Text length: {len(txt)} characters")

        # Load alignment model if needed
        if WhisperXSRTNode.align_model is None or WhisperXSRTNode.current_language != language:
            print("[INFO] Loading alignment model...")
            WhisperXSRTNode.align_model, WhisperXSRTNode.metadata = load_align_model(
                language_code=language, 
                device=device
            )
            WhisperXSRTNode.current_language = language
            torch.cuda.empty_cache()
            print("[INFO] Alignment model loaded successfully")

        # Create segments for alignment
        segments = [{"text": txt, "start": 0.0, "end": audio_duration}]
        print(f"[INFO] Submitting {len(segments)} segments for alignment")

        # Perform forced alignment
        print("[INFO] Starting forced alignment...")
        result = whisperx.align(
            segments,
            WhisperXSRTNode.align_model,
            WhisperXSRTNode.metadata,
            audio_array,
            device,
            return_char_alignments=False,
        )
        print("[INFO] Alignment completed")

        # Extract words with valid timing
        words = []
        for seg in result["segments"]:
            for word in seg.get("words", []):
                if (
                    word.get("word", "").strip()
                    and word.get("start") is not None
                    and word.get("end") is not None
                ):
                    words.append(word)

        print(f"[INFO] Successfully aligned words: {len(words)}")
        
        # If no words were aligned, provide helpful error information
        if len(words) == 0:
            error_msg = (
                f"Failed to align any words. This could be due to:\n"
                f"1. Audio duration too short ({audio_duration:.2f}s) - try longer audio\n"
                f"2. Audio quality issues - ensure clear speech\n"
                f"3. Language mismatch - audio language should match selected language ({language})\n"
                f"4. Text-audio content mismatch - ensure text matches spoken content\n"
                f"5. Audio format issues - try different audio format\n"
                f"Text length: {len(txt)} characters\n"
                f"Audio duration: {audio_duration:.2f} seconds"
            )
            print(f"[WARNING] {error_msg}")
            
            # Still return something useful even if alignment failed
            alignment_info = {
                "language": language,
                "device": device,
                "audio_duration_seconds": audio_duration,
                "text_length": len(txt),
                "aligned_words": 0,
                "subtitle_lines": 0,
                "max_duration_per_line": max_sec,
                "max_characters_per_line": max_ch,
                "punctuation_triggers": punct,
                "error": "No words could be aligned",
                "suggestions": [
                    "Check audio quality and clarity",
                    "Ensure text matches audio content",
                    "Try longer audio clips",
                    "Verify language selection matches audio"
                ]
            }
            
            return (
                "# No SRT content generated\n# Alignment failed - see alignment_info for details",
                json.dumps(alignment_info, indent=2, ensure_ascii=False)
            )

        # Group words into subtitle lines
        subtitles = []
        buffer = []
        
        for word in words:
            buffer.append(word)
            duration = buffer[-1]["end"] - buffer[0]["start"]
            text_line = "".join(item["word"] for item in buffer)
            end_with_punct = word["word"] in punct

            if end_with_punct or duration >= max_sec or len(text_line) >= max_ch:
                flush_line(buffer, subtitles)

        # Flush remaining buffer
        flush_line(buffer, subtitles)
        print(f"[INFO] Generated subtitle lines: {len(subtitles)}")

        # Create SRT content
        srt_content = create_srt_content(subtitles)

        # Create alignment info
        alignment_info = {
            "language": language,
            "device": device,
            "audio_duration_seconds": audio_duration,
            "text_length": len(txt),
            "aligned_words": len(words),
            "subtitle_lines": len(subtitles),
            "max_duration_per_line": max_sec,
            "max_characters_per_line": max_ch,
            "punctuation_triggers": punct,
        }

        print("[INFO] SRT generation completed successfully")
        if unload_model:
            print("[INFO] Unloading models from VRAM")
            WhisperXSRTNode.align_model = None
            WhisperXSRTNode.metadata = None
            WhisperXSRTNode.current_language = None
            torch.cuda.empty_cache()

        return (
            srt_content,
            json.dumps(alignment_info, indent=2, ensure_ascii=False)
        )

# Node class mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "WhisperX SRT Generator": WhisperXSRTNode,
}

# Node display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "WhisperX SRT Generator": "WhisperX SRT字幕生成器 / SRT Generator",
}