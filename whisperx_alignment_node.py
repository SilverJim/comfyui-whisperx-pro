"""
WhisperX Alignment Node for ComfyUI
Provides accurate word-level timestamps through forced alignment
"""

import os
import json
import re
import torch
import torchaudio
import numpy as np
from typing import Dict, List, Tuple, Any, Union


def convert_audio_to_whisperx_format(audio: Dict[str, Any]) -> np.ndarray:
    """
    Convert ComfyUI official audio format to WhisperX format.

    Args:
        audio: Audio dict from ComfyUI's official audio loader
               Expected format: {"waveform": tensor, "sample_rate": int}

    Returns:
        numpy array in WhisperX format (16kHz mono)
    """
    waveform = audio["waveform"]
    sample_rate = audio["sample_rate"]

    # Convert to torch tensor if needed
    if isinstance(waveform, np.ndarray):
        waveform = torch.from_numpy(waveform)

    # Ensure tensor is float32
    if waveform.dtype != torch.float32:
        waveform = waveform.float()

    # Convert to mono if stereo (take mean of channels)
    if waveform.dim() > 1 and waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Remove channel dimension if present
    if waveform.dim() > 1:
        waveform = waveform.squeeze(0)

    # Resample to 16kHz if needed (WhisperX requirement)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)

    # Convert to numpy
    audio_array = waveform.numpy()

    return audio_array


def segment_text(text: str, max_chars: int = 200, language: str = "en") -> List[str]:
    """
    Segment text into smaller chunks based on punctuation and max character limit.

    Args:
        text: Input text to segment
        max_chars: Maximum characters per segment
        language: Language code for language-specific segmentation

    Returns:
        List of text segments
    """
    if not text or not text.strip():
        return []

    # Clean up text
    text = text.strip()

    # Define sentence ending punctuation for different languages
    if language in ["zh", "ja"]:
        # Chinese and Japanese sentence delimiters
        sentence_endings = r'[。！？!?；;]'
    else:
        # English and other Latin-based languages
        sentence_endings = r'[.!?;]'

    # First, try to split by sentence endings
    sentences = re.split(f'({sentence_endings})', text)

    # Recombine sentences with their punctuation
    combined_sentences = []
    for i in range(0, len(sentences) - 1, 2):
        if i + 1 < len(sentences):
            combined_sentences.append(sentences[i] + sentences[i + 1])
        else:
            combined_sentences.append(sentences[i])

    # Handle last item if it doesn't have punctuation
    if len(sentences) % 2 != 0 and sentences[-1].strip():
        combined_sentences.append(sentences[-1])

    # If no sentences were found, treat entire text as one segment
    if not combined_sentences:
        combined_sentences = [text]

    # Further split segments that exceed max_chars
    final_segments = []
    for sentence in combined_sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(sentence) <= max_chars:
            final_segments.append(sentence)
        else:
            # Split long sentences by commas, spaces, or character limit
            if language in ["zh", "ja"]:
                # For Chinese/Japanese, split by commas or character count
                sub_parts = re.split(r'[，,、]', sentence)
            else:
                # For English, split by commas, semicolons, or "and"
                sub_parts = re.split(r'[,;]|\s+and\s+', sentence)

            current_segment = ""
            for part in sub_parts:
                part = part.strip()
                if not part:
                    continue

                # Check if adding this part exceeds limit
                test_segment = current_segment + (" " if current_segment else "") + part

                if len(test_segment) <= max_chars:
                    current_segment = test_segment
                else:
                    # Save current segment and start new one
                    if current_segment:
                        final_segments.append(current_segment)

                    # If single part is too long, force split by character limit
                    if len(part) > max_chars:
                        for i in range(0, len(part), max_chars):
                            chunk = part[i:i + max_chars].strip()
                            if chunk:
                                final_segments.append(chunk)
                        current_segment = ""
                    else:
                        current_segment = part

            # Add remaining segment
            if current_segment:
                final_segments.append(current_segment)

    return [seg.strip() for seg in final_segments if seg.strip()]


class WhisperXAlignmentNode:
    """
    A ComfyUI node for aligning text transcripts with audio using WhisperX.
    Provides accurate word-level timestamps through phoneme-based forced alignment.
    """

    def __init__(self):
        self.model_a = None
        self.metadata = None
        self.current_language = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "input_type": (["plain_text", "json"], {
                    "default": "plain_text"
                }),
                "text_input": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "Input plain text or JSON segments. For plain text: just type your text. For JSON: [{'text': 'Hello', 'start': 0.0, 'end': 1.0}]"
                }),
                "language": (["en", "fr", "de", "es", "it", "pt", "nl", "ja", "zh", "auto"], {
                    "default": "en"
                }),
                "auto_segment": ("BOOLEAN", {
                    "default": True,
                    "label_on": "Auto segment text",
                    "label_off": "Use as is"
                }),
                "max_chars_per_segment": ("INT", {
                    "default": 200,
                    "min": 50,
                    "max": 1000,
                    "step": 10,
                    "display": "number"
                }),
                "return_char_alignments": ("BOOLEAN", {
                    "default": False
                }),
            },
            "optional": {
                "device": (["auto", "cuda", "cpu"], {
                    "default": "auto"
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("aligned_segments", "word_segments", "alignment_info")
    FUNCTION = "align_audio_text"
    CATEGORY = "audio/whisperx"

    def align_audio_text(
        self,
        audio: Dict[str, Any],
        input_type: str,
        text_input: str,
        language: str,
        auto_segment: bool,
        max_chars_per_segment: int,
        return_char_alignments: bool,
        device: str = "auto"
    ) -> Tuple[str, str, str]:
        """
        Align transcription segments with audio to get accurate word-level timestamps.

        Args:
            audio: Audio data dictionary from LoadAudioNode
            input_type: Type of input (plain_text or json)
            text_input: Text input (plain text or JSON string)
            language: Language code for alignment model
            auto_segment: Whether to automatically segment the text
            max_chars_per_segment: Maximum characters per segment when auto_segment is True
            return_char_alignments: Whether to return character-level alignments
            device: Device to run on (auto, cuda, or cpu)

        Returns:
            Tuple of (aligned_segments_json, word_segments_json, alignment_info_json)
        """
        try:
            import whisperx
        except ImportError:
            raise ImportError(
                "WhisperX is not installed. Please install it using:\n"
                "pip install git+https://github.com/m-bain/whisperx.git"
            )

        # Validate audio input
        if not audio or not isinstance(audio, dict):
            raise ValueError("Audio input must be a dictionary from ComfyUI audio loader")

        if "waveform" not in audio or "sample_rate" not in audio:
            raise ValueError("Audio data must contain 'waveform' and 'sample_rate' keys")

        # Convert audio to WhisperX format (16kHz mono numpy array)
        print(f"Converting audio from {audio['sample_rate']}Hz to WhisperX format (16kHz mono)")
        audio_array = convert_audio_to_whisperx_format(audio)

        # Validate text input
        if not text_input or not text_input.strip():
            raise ValueError("Text input cannot be empty")

        # Process input based on input_type
        segments = []
        if input_type == "json":
            # Parse JSON input
            try:
                segments = json.loads(text_input)
                if not isinstance(segments, list):
                    raise ValueError("JSON input must be an array of segment objects")
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in text_input: {e}")
        else:
            # Plain text input
            plain_text = text_input.strip()

            if auto_segment:
                # Auto-segment the text
                print(f"Auto-segmenting text with max_chars_per_segment={max_chars_per_segment}")
                text_segments = segment_text(plain_text, max_chars_per_segment, language)
                print(f"Created {len(text_segments)} segments from plain text")

                # Create segment objects without timestamps (WhisperX will generate them)
                segments = [{"text": seg} for seg in text_segments]
            else:
                # Use entire text as one segment
                segments = [{"text": plain_text}]

        # Determine device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Processing audio: {audio_array.shape[0] / 16000:.2f}s at 16kHz")

        # Auto-detect language if needed
        if language == "auto":
            if segments and len(segments) > 0 and "language" in segments[0]:
                language = segments[0]["language"]
            else:
                language = "en"  # Default to English
                print(f"Warning: Could not auto-detect language, defaulting to 'en'")

        # Load alignment model
        print(f"Loading alignment model for language: {language}")
        if self.model_a is None or self.current_language != language:
            self.model_a, self.metadata = whisperx.load_align_model(
                language_code=language,
                device=device
            )
            self.current_language = language
            print(f"Alignment model loaded successfully on device: {device}")

        # Perform alignment
        print(f"Aligning {len(segments)} segments...")
        result = whisperx.align(
            segments,
            self.model_a,
            self.metadata,
            audio_array,
            device,
            return_char_alignments=return_char_alignments
        )

        # Extract aligned segments
        aligned_segments = result.get("segments", [])

        # Extract word-level segments
        word_segments = []
        for segment in aligned_segments:
            if "words" in segment:
                for word in segment["words"]:
                    word_segments.append(word)

        # Create alignment info
        alignment_info = {
            "language": language,
            "device": device,
            "input_type": input_type,
            "auto_segment": auto_segment,
            "max_chars_per_segment": max_chars_per_segment if auto_segment else None,
            "num_segments": len(aligned_segments),
            "num_words": len(word_segments),
            "return_char_alignments": return_char_alignments,
            "audio_duration_seconds": audio_array.shape[0] / 16000,
        }

        # Add timing statistics if available
        if word_segments:
            times = [(w.get("start", 0), w.get("end", 0)) for w in word_segments if "start" in w and "end" in w]
            if times:
                alignment_info["total_duration"] = max([t[1] for t in times]) if times else 0
                alignment_info["average_word_duration"] = sum([t[1] - t[0] for t in times]) / len(times)

        print(f"Alignment complete! Processed {len(aligned_segments)} segments and {len(word_segments)} words")

        # Return as JSON strings
        return (
            json.dumps(aligned_segments, indent=2, ensure_ascii=False),
            json.dumps(word_segments, indent=2, ensure_ascii=False),
            json.dumps(alignment_info, indent=2, ensure_ascii=False)
        )


# Node class mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "WhisperX Alignment": WhisperXAlignmentNode,
}

# Node display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "WhisperX Alignment": "WhisperX Alignment (Text-Audio Align)",
}
