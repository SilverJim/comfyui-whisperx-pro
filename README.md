# ComfyUI WhisperX Pro

A professional ComfyUI custom node package that provides accurate audio transcription and text-audio alignment using [WhisperX](https://github.com/m-bain/whisperX).

## Features

- **WhisperX Transcribe Node**: Fast audio transcription with batching support
- **WhisperX Alignment Node**: Accurate word-level timestamp alignment
- Support for multiple languages (en, fr, de, es, it, pt, nl, ja, zh)
- GPU acceleration support
- Character-level and word-level alignment options

## Installation

### 1. Install to ComfyUI Custom Nodes

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/loockluo/comfyui-whisperx-pro.git
cd comfyui-whisperx-pro
pip install -r requirements.txt
```

### 2. Manual Installation

If the automatic installation fails, install WhisperX manually:

```bash
pip install git+https://github.com/m-bain/whisperx.git
```

## Nodes

### WhisperX Transcribe

Transcribes audio files to text with high accuracy.

**Inputs:**
- `audio_path` (STRING): Path to your audio file (wav, mp3, etc.)
- `model_name` (DROPDOWN): Model size (tiny, base, small, medium, large-v2, large-v3)
- `language` (DROPDOWN): Language code or 'auto' for auto-detection
- `batch_size` (INT): Batch size for processing (default: 16)
- `device` (DROPDOWN): Device to use (auto, cuda, cpu)
- `compute_type` (DROPDOWN): Precision (int8, float16, float32)

**Outputs:**
- `segments` (STRING): JSON array of transcription segments
- `full_text` (STRING): Complete transcription text
- `transcription_info` (STRING): Metadata about the transcription

**Example Segment Output:**
```json
[
  {
    "start": 0.5,
    "end": 2.3,
    "text": "Hello world"
  }
]
```

### WhisperX Alignment

Aligns existing transcripts with audio to get accurate word-level timestamps.

**Inputs:**
- `audio_path` (STRING): Path to your audio file
- `transcript_json` (STRING): JSON array of transcript segments
- `language` (DROPDOWN): Language code for alignment model
- `return_char_alignments` (BOOLEAN): Return character-level alignments
- `device` (DROPDOWN): Device to use (auto, cuda, cpu)

**Outputs:**
- `aligned_segments` (STRING): Segments with accurate timestamps
- `word_segments` (STRING): Individual words with timestamps
- `alignment_info` (STRING): Alignment statistics and metadata

**Example Input Transcript:**
```json
[
  {
    "text": "Hello world",
    "start": 0.0,
    "end": 2.0
  }
]
```

**Example Word Segments Output:**
```json
[
  {
    "word": "Hello",
    "start": 0.52,
    "end": 0.89,
    "score": 0.95
  },
  {
    "word": "world",
    "start": 1.05,
    "end": 1.67,
    "score": 0.97
  }
]
```

## Workflow Examples

### Basic Transcription Workflow

1. Add "WhisperX Transcribe" node
2. Set `audio_path` to your audio file
3. Select model size and language
4. Run to get transcription

### Transcription + Alignment Workflow

1. Add "WhisperX Transcribe" node for initial transcription
2. Connect output to "WhisperX Alignment" node
3. Connect the same audio file to alignment node
4. Get accurate word-level timestamps

### Using with External Transcripts

1. Add "WhisperX Alignment" node
2. Provide your own transcript in JSON format
3. Set audio path and language
4. Get precise timing alignment

## Supported Languages

- English (en)
- French (fr)
- German (de)
- Spanish (es)
- Italian (it)
- Portuguese (pt)
- Dutch (nl)
- Japanese (ja)
- Chinese (zh)
- Auto-detection (auto)

## Performance Tips

1. **GPU Acceleration**: Use CUDA for faster processing
2. **Batch Size**: Increase batch size for longer audio files (if you have enough VRAM)
3. **Model Selection**:
   - Use `tiny` or `base` for quick testing
   - Use `medium` or `large-v3` for best accuracy
4. **Compute Type**: Use `float16` on GPU for best speed/accuracy balance

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)
- ComfyUI

## Troubleshooting

### WhisperX Installation Issues

If you encounter installation issues:

```bash
# Install PyTorch first
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# Then install WhisperX
pip install git+https://github.com/m-bain/whisperx.git
```

### Out of Memory Errors

- Reduce batch_size
- Use a smaller model
- Switch to CPU (slower but uses less memory)

### Audio File Not Found

- Use absolute paths for audio files
- Ensure the file exists and is accessible
- Check file permissions

## Credits

- [WhisperX](https://github.com/m-bain/whisperx) by Max Bain
- Based on [OpenAI Whisper](https://github.com/openai/whisper)

## License

MIT License

## Support

For issues and feature requests, please visit the [GitHub repository](https://github.com/loockluo/comfyui-whisperx-pro/issues).
