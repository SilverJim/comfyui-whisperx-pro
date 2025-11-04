# ComfyUI WhisperX Pro

English | [简体中文](README_CN.md)

A professional ComfyUI custom node for accurate audio-text alignment and SRT subtitle generation using [WhisperX](https://github.com/m-bain/whisperx).

## ✨ Features

- 🎯 **Precise Alignment**: Word-level timestamp alignment using WhisperX
- 🎬 **SRT Subtitle Generation**: Automatically generate SRT subtitle files with customizable formatting
- 🌍 **Multi-language Support**: Support for 9 languages (English, Chinese, French, German, Spanish, Italian, Portuguese, Dutch, Japanese)
- 🌐 **Bilingual UI**: Node interface automatically switches between English and Chinese based on system locale
- 📦 **Local Model Loading**: Load alignment models from ComfyUI/models directory - no automatic downloads
- ⚡ **GPU Acceleration**: CUDA support for faster processing
- 🎛️ **Flexible Configuration**: Customizable line duration, character limits, and punctuation triggers
- 🔧 **Easy Integration**: Seamlessly integrates with ComfyUI's audio loading nodes

## 📋 Table of Contents

- [Installation](#-installation)
- [Model Setup](#-model-setup)
- [Node Overview](#-node-overview)
- [Usage Examples](#-usage-examples)
- [Supported Languages](#-supported-languages)
- [Configuration Tips](#-configuration-tips)
- [Troubleshooting](#-troubleshooting)
- [Credits](#-credits)

## 🚀 Installation

### Step 1: Install the Custom Node

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/loockluo/comfyui-whisperx-pro.git
cd comfyui-whisperx-pro
pip install -r requirements.txt
```

### Step 2: Install WhisperX (if needed)

If the automatic installation fails, install WhisperX manually:

```bash
pip install git+https://github.com/m-bain/whisperx.git
```

For CUDA GPU support, ensure PyTorch is installed with CUDA:

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Step 3: Restart ComfyUI

After installation, restart ComfyUI to load the new node.

## 📦 Model Setup

### Model Directory Structure

Models must be placed in: `ComfyUI/models/whisperx/[model_folder_name]/`

The node will automatically load the appropriate alignment model based on the selected language.

### Download Pre-trained Alignment Models

#### Option 1: HuggingFace (Worldwide)

Download the complete model folder (all files including `config.json`, `pytorch_model.bin`, `preprocessor_config.json`, `tokenizer_config.json`, `vocab.json`):

- **English**: [wav2vec2-large-xlsr-53-english](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-english)
- **Chinese**: [wav2vec2-large-xlsr-53-chinese-zh-cn](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn)
- **French**: [wav2vec2-large-xlsr-53-french](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-french)
- **German**: [wav2vec2-large-xlsr-53-german](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-german)
- **Spanish**: [wav2vec2-large-xlsr-53-spanish](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-spanish)
- **Italian**: [wav2vec2-large-xlsr-53-italian](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-italian)
- **Portuguese**: [wav2vec2-large-xlsr-53-portuguese](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-portuguese)
- **Japanese**: [wav2vec2-large-xlsr-53-japanese](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-japanese)
- **Dutch**: [wav2vec2-large-xlsr-53-dutch](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-dutch)

#### Option 2: ModelScope (Faster for China)

- **English**: [wav2vec2-large-xlsr-53-english](https://modelscope.cn/models/jonatasgrosman/wav2vec2-large-xlsr-53-english)
- **Chinese**: [wav2vec2-large-xlsr-53-chinese-zh-cn](https://modelscope.cn/models/jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn)
- Other languages available similarly

### Example: Installing Chinese Model

```bash
cd ComfyUI/models/whisperx

# Download all files from the model page and place them in:
# wav2vec2-large-xlsr-53-chinese-zh-cn/
#   ├── config.json
#   ├── pytorch_model.bin
#   ├── preprocessor_config.json
#   ├── tokenizer_config.json
#   └── vocab.json
```

## 📦 Node Overview

### WhisperX SRT Generator

This node aligns text with audio and generates SRT subtitle format output with precise word-level timestamps.

#### Input Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `audio` | AUDIO | Required | - | Audio data from ComfyUI's official audio loader |
| `text` | STRING | Required | - | Text content to align with audio |
| `language` | DROPDOWN | `zh` | en/fr/de/es/it/pt/nl/ja/zh | Language code for alignment model |
| `max_sec` | FLOAT | 4.5 | 1.0 - 10.0 | Maximum duration per subtitle line (seconds) |
| `max_ch` | INT | 28 | 10 - 100 | Maximum characters per subtitle line |
| `punct` | STRING | `，。！？；、,.!?;…` | - | Punctuation marks that trigger line breaks |
| `device` | DROPDOWN | `auto` | auto/cuda/cpu | Device to use for processing |

#### Output Results

| Output | Type | Description |
|--------|------|-------------|
| `SRT Content` | STRING | Generated SRT subtitle content with timestamps |
| `Alignment Info` | STRING | JSON metadata about the alignment process |

#### Example Outputs

**SRT Content:**
```srt
1
00:00:00,520 --> 00:00:02,350
Hello everyone.

2
00:00:02,350 --> 00:00:04,180
Today we're going to talk

3
00:00:04,180 --> 00:00:06,890
about WhisperX.
```

**Alignment Info:**
```json
{
  "language": "en",
  "device": "cuda",
  "audio_duration_seconds": 10.5,
  "text_length": 87,
  "aligned_words": 15,
  "subtitle_lines": 8,
  "max_duration_per_line": 4.5,
  "max_characters_per_line": 28,
  "punctuation_triggers": ",.!?;…"
}
```

## 🎯 Usage Examples

### Prerequisites

You need a ComfyUI audio loading node first. Install one of these:
- **ComfyUI-Advanced-Audio**: Provides "Load Audio" node
- Or any other ComfyUI audio loader that outputs AUDIO type

### Basic Workflow

1. **Add Audio Loading Node**
   - Add "Load Audio" node (from ComfyUI-Advanced-Audio or similar)
   - Set the path to your audio file

2. **Add WhisperX SRT Generator Node**
   - Add "WhisperX SRT Generator" node to your workflow
   - Connect the `audio` output from the audio loader to the node

3. **Configure Parameters**
   - Set `language` to match your audio (e.g., "en" for English, "zh" for Chinese)
   - Paste your transcript in the `text` field
   - Adjust timing parameters:
     - `max_sec`: Control how long each subtitle line can be
     - `max_ch`: Control how many characters per line
     - `punct`: Specify which punctuation marks trigger line breaks

4. **Run and Export**
   - Execute the workflow
   - The node will output SRT-formatted subtitles
   - Save the output to a `.srt` file using a text output node

### Example 1: English Video Subtitles

**Scenario**: You have an English video and want to generate subtitles

```
Settings:
- Language: en
- Max Duration: 4.5 seconds
- Max Characters: 28
- Punctuation: ,.!?;…

Input Text:
"Hello everyone. Today we're going to talk about WhisperX. It's an amazing tool for speech recognition and alignment. Let me show you how it works."

Output:
Automatically generates 4-6 subtitle lines with precise word-level timing
```

### Example 2: Chinese Audio Subtitles

**Scenario**: You have a Chinese podcast and want to generate subtitles

```
Settings:
- Language: zh
- Max Duration: 4.5 seconds
- Max Characters: 28
- Punctuation: ，。！？；、

Input Text:
"大家好。今天我们来讨论一下WhisperX这个工具。它是一个非常强大的语音识别和对齐工具。让我来给大家演示一下它的使用方法。"

Output:
Automatically generates subtitle lines respecting Chinese punctuation rules
```

### Example 3: Multi-language Content

For content mixing multiple languages, use the dominant language for the `language` parameter, or process each language segment separately.

## 🌍 Supported Languages

| Language | Code | Model Required |
|----------|------|----------------|
| English | `en` | wav2vec2-large-xlsr-53-english |
| Chinese | `zh` | wav2vec2-large-xlsr-53-chinese-zh-cn |
| French | `fr` | wav2vec2-large-xlsr-53-french |
| German | `de` | wav2vec2-large-xlsr-53-german |
| Spanish | `es` | wav2vec2-large-xlsr-53-spanish |
| Italian | `it` | wav2vec2-large-xlsr-53-italian |
| Portuguese | `pt` | wav2vec2-large-xlsr-53-portuguese |
| Dutch | `nl` | wav2vec2-large-xlsr-53-dutch |
| Japanese | `ja` | wav2vec2-large-xlsr-53-japanese |

## ⚙️ Configuration Tips

### Optimizing Subtitle Line Length

**Short Lines (max_ch: 15-20, max_sec: 2-3)**
- ✅ Good for: Social media videos, mobile viewing
- ✅ Pros: Easy to read, good for fast-paced content
- ❌ Cons: Many subtitle switches, can be distracting

**Medium Lines (max_ch: 25-35, max_sec: 3-5)**
- ✅ Good for: Most standard videos, presentations
- ✅ Pros: Balanced readability and subtitle frequency
- ⭐ **Recommended default**

**Long Lines (max_ch: 40-60, max_sec: 5-8)**
- ✅ Good for: Documentaries, lectures, slow-paced content
- ✅ Pros: Fewer subtitle switches, more context visible
- ❌ Cons: Can be hard to read, especially on small screens

### Punctuation Configuration

**For English:**
```
Recommended: ,.!?;…
```

**For Chinese:**
```
Recommended: ，。！？；、
Include English punctuation if content is mixed: ，。！？；、,.!?
```

**For Japanese:**
```
Recommended: 。！？、
```

### Device Selection

- **auto**: Automatically selects CUDA if available, otherwise CPU (recommended)
- **cuda**: Force GPU processing (faster, requires NVIDIA GPU with CUDA)
- **cpu**: Force CPU processing (slower, works on all systems)

**Performance Comparison:**
- CUDA (GPU): ~10-30x faster than CPU
- CPU: Slower but works everywhere, good for testing

## 🔧 Troubleshooting

### Issue: "WhisperX is not installed"

**Solution:**
```bash
pip install git+https://github.com/m-bain/whisperx.git
```

If this fails, try installing PyTorch first:
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install git+https://github.com/m-bain/whisperx.git
```

### Issue: "Failed to align any words"

**Possible causes and solutions:**

1. **Audio too short**: Use audio clips longer than 2-3 seconds
2. **Audio quality issues**: Ensure clear speech without excessive background noise
3. **Language mismatch**: Make sure selected language matches the audio
4. **Text-audio mismatch**: Verify that the text content matches what is spoken
5. **Incorrect audio format**: Try converting audio to WAV format (16kHz, mono)

### Issue: "Out of memory" error

**Solutions:**
1. Switch to CPU: Set `device` to "cpu"
2. Process shorter audio segments
3. Close other applications to free up memory
4. Upgrade GPU memory (for CUDA users)

### Issue: Model not loading from local directory

**Check:**
1. Model files are in correct directory: `ComfyUI/models/whisperx/[model_folder_name]/`
2. All required files are present: `config.json`, `pytorch_model.bin`, `preprocessor_config.json`, `tokenizer_config.json`, `vocab.json`
3. Model folder name matches the language code mapping (e.g., "wav2vec2-large-xlsr-53-english" for English)

### Issue: SRT timestamps are inaccurate

**Solutions:**
1. Ensure text exactly matches spoken content
2. Try adjusting `max_sec` and `max_ch` parameters
3. Check audio quality and clarity
4. Verify language selection is correct

## 📝 Requirements

- **Python**: 3.8 or higher
- **PyTorch**: 2.0 or higher
- **ComfyUI**: Latest version recommended
- **CUDA**: Optional, for GPU acceleration (CUDA 11.7 or higher)

### Disk Space Requirements

- Base installation: ~500 MB
- Each language model: ~1.2 GB
- Typical installation with 2-3 languages: ~3-4 GB

## 🙏 Credits

- [WhisperX](https://github.com/m-bain/whisperx) by Max Bain - Advanced audio-text alignment
- [OpenAI Whisper](https://github.com/openai/whisper) - Foundation speech recognition model
- [Wav2Vec2](https://huggingface.co/transformers/model_doc/wav2vec2.html) - Alignment models by Facebook AI
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) - Powerful node-based UI for Stable Diffusion

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details

## 🤝 Support

- **Issues**: [GitHub Issues](https://github.com/loockluo/comfyui-whisperx-pro/issues)
- **Discussions**: [GitHub Discussions](https://github.com/loockluo/comfyui-whisperx-pro/discussions)
- **Pull Requests**: Contributions are welcome!

## 🗺️ Roadmap

- [ ] Add batch processing support
- [ ] Support for additional subtitle formats (VTT, ASS)
- [ ] Real-time preview of subtitle timing
- [ ] Custom model fine-tuning support
- [ ] Automatic punctuation restoration

---

If you find this project helpful, please consider giving it a ⭐ on GitHub!
