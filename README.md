# ComfyUI WhisperX Pro

English | [简体中文](README_CN.md)

A professional ComfyUI custom node for accurate text-audio alignment using [WhisperX](https://github.com/m-bain/whisperX).

## Features

- **WhisperX Alignment Node**: Accurate word-level timestamp alignment
- Plain text and JSON input support
- Automatic text segmentation with customizable sentence splitting
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

**Important:** This node requires audio data from ComfyUI's official audio loading nodes (e.g., "Load Audio" from ComfyUI-Advanced-Audio or similar). Connect the AUDIO output from those nodes to the WhisperX Alignment node.

### WhisperX Alignment

Aligns text transcripts with audio to get accurate word-level timestamps. Supports both plain text and JSON input with automatic text segmentation.

**Inputs:**
- `audio` (AUDIO): Audio data from ComfyUI's official audio loader
- `input_type` (DROPDOWN): Input format - "plain_text" or "json"
- `text_input` (STRING): Your text content (plain text or JSON segments)
- `language` (DROPDOWN): Language code for alignment model
- `auto_segment` (BOOLEAN): Automatically segment text into smaller chunks (default: True)
- `max_chars_per_segment` (INT): Maximum characters per segment when auto_segment is enabled (default: 200, range: 50-1000)
- `return_char_alignments` (BOOLEAN): Return character-level alignments (default: False)
- `model_name` (STRING, optional): Specific model name to use (default: "auto" - auto-select by language)
- `device` (DROPDOWN): Device to use (auto, cuda, cpu)

**Outputs:**
- `aligned_segments` (STRING): Segments with accurate timestamps
- `word_segments` (STRING): Individual words with timestamps
- `alignment_info` (STRING): Alignment statistics and metadata

**Example Plain Text Input:**
```
Hello world. How are you today? I'm doing great!
```

**Example JSON Input:**
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

### Prerequisites

You need a ComfyUI audio loading node first. Install one of these:
- **ComfyUI-Advanced-Audio**: Provides "Load Audio" node
- Or any other ComfyUI audio loader that outputs AUDIO type

### Plain Text Alignment

Perfect for when you already have a transcript and just need timing:

1. Add ComfyUI's "Load Audio" node and set audio file path
2. Add "WhisperX Alignment" node
3. Connect `audio` output from Load Audio to Alignment node
4. Set `input_type` to "plain_text"
5. Paste your transcript in `text_input`
6. Enable `auto_segment` for automatic sentence splitting
7. Adjust `max_chars_per_segment` if needed (recommended: 100-300)
8. Set language
9. Run to get precise word-level timestamps

**Example:**
```
Input text: "Hello everyone. Today we're going to talk about WhisperX. It's an amazing tool for speech recognition and alignment."
Output: Automatically split into 3 segments with accurate word timings
```

### Using with External Transcripts (JSON)

1. Add ComfyUI's "Load Audio" node and set audio file path
2. Add "WhisperX Alignment" node
3. Connect `audio` from Load Audio to Alignment
4. Set `input_type` to "json"
5. Provide your own transcript in JSON format in `text_input`
6. Set language
7. Run to get precise timing alignment

### Chinese/Japanese Text Alignment

The auto-segmentation supports CJK languages:

1. Add ComfyUI's "Load Audio" node with your audio file
2. Add "WhisperX Alignment" node and connect audio
3. Set `language` to "zh" or "ja"
4. Use `plain_text` input type
5. Enable `auto_segment`
6. The segmenter will use appropriate punctuation (。！？etc.)

**Example Chinese:**
```
今天天气很好。我们去公园散步吧。你觉得怎么样？
→ Automatically segmented by Chinese sentence endings
```

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

## Advanced Usage

### Manual Model Selection

By default, the alignment model is automatically selected based on the language you choose. However, you can manually specify a model if needed:

1. Leave `model_name` as "auto" for automatic selection (recommended)
2. Or specify a model name like "WAV2VEC2_ASR_BASE_960H" to force-load a specific model

**When to use manual model selection:**
- Testing different alignment models
- Using custom fine-tuned models
- Debugging alignment issues

**Example:**
- Language: "en"
- Model Name: "auto" → Automatically loads the best English alignment model
- Model Name: "WAV2VEC2_ASR_BASE_960H" → Forces this specific model

## Performance Tips

1. **GPU Acceleration**: Use CUDA for faster processing
2. **Model Selection**:
   - Leave `model_name` as "auto" for best results
   - Manual model selection is for advanced users only
3. **Text Segmentation**:
   - Adjust `max_chars_per_segment` (100-300 recommended)
   - Shorter segments = more accurate alignment but slower processing

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
