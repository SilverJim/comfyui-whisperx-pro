# ComfyUI WhisperX Pro

[English](README.md) | 简体中文

ComfyUI 专业自定义节点，使用 [WhisperX](https://github.com/m-bain/whisperx) 提供精确的文本-音频对齐功能。

## 功能特性

- **WhisperX 对齐节点**：精确的词级和句子级时间戳对齐
- **多语言界面**：节点界面根据 ComfyUI 的语言设置自动在中英文之间切换
- **多级输出结构**：提供段落级、句子级（可配置约30字）和词级时间戳
- 支持纯文本和 JSON 输入
- 自动文本分段，可自定义句子拆分
- 支持多种语言（中文、英语、法语、德语、西班牙语、意大利语、葡萄牙语、荷兰语、日语）
- GPU 加速支持
- 字符级和词级对齐选项

## 安装

### 1. 安装到 ComfyUI 自定义节点目录

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/loockluo/comfyui-whisperx-pro.git
cd comfyui-whisperx-pro
pip install -r requirements.txt
```

### 2. 手动安装

如果自动安装失败，可以手动安装 WhisperX：

```bash
pip install git+https://github.com/m-bain/whisperx.git
```

## 节点说明

**重要提示：** 此节点需要来自 ComfyUI 官方音频加载节点的音频数据（例如 ComfyUI-Advanced-Audio 中的 "Load Audio" 节点）。将这些节点的 AUDIO 输出连接到 WhisperX Alignment 节点。

### WhisperX Alignment（文本-音频对齐）

将文本转录与音频对齐，获取精确的词级时间戳。支持纯文本和 JSON 输入，具有自动文本分段功能。

**输入参数：**
- `audio`（AUDIO）：来自 ComfyUI 官方音频加载器的音频数据
- `input_type`（下拉选项）：输入格式 - "plain_text"（纯文本）或 "json"
- `text_input`（字符串）：您的文本内容（纯文本或 JSON 片段）
- `language`（下拉选项）：对齐模型的语言代码
- `auto_segment`（布尔值）：自动将文本分段为较小的块（默认：True）
- `max_chars_per_segment`（整数）：启用 auto_segment 时每段的最大字符数（默认：200，范围：50-1000）
- `max_chars_per_sentence`（整数）：句子级输出的每句最大字符数（默认：30，范围：10-200）
- `return_char_alignments`（布尔值）：返回字符级对齐（默认：False）
- `model_name`（字符串，可选）：指定要使用的模型名称（默认："auto" - 根据语言自动选择）
- `device`（下拉选项）：使用的设备（auto、cuda、cpu）

**输出结果：**
- `aligned_segments`（字符串）：带有精确时间戳的片段（基于 max_chars_per_segment 的较大块）
- `word_segments`（字符串）：带有时间戳的单个词
- `sentence_segments`（字符串）：句子级片段，带有时间戳（每句约 max_chars_per_sentence 字符）
- `alignment_info`（字符串）：对齐统计信息和元数据

**纯文本输入示例：**
```
Hello world. How are you today? I'm doing great!
```

**JSON 输入示例：**
```json
[
  {
    "text": "Hello world",
    "start": 0.0,
    "end": 2.0
  }
]
```

**词级片段输出示例：**
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

**句子级片段输出示例：**
```json
[
  {
    "text": "Hello world. How are you today?",
    "start": 0.52,
    "end": 3.45,
    "words": [
      {"word": "Hello", "start": 0.52, "end": 0.89, "score": 0.95},
      {"word": "world", "start": 1.05, "end": 1.67, "score": 0.97},
      ...
    ]
  },
  {
    "text": "I'm doing great!",
    "start": 3.45,
    "end": 5.20,
    "words": [...]
  }
]
```

## 工作流示例

### 前置要求

您首先需要一个 ComfyUI 音频加载节点。安装以下任一选项：
- **ComfyUI-Advanced-Audio**：提供 "Load Audio" 节点
- 或任何其他输出 AUDIO 类型的 ComfyUI 音频加载器

### 纯文本对齐

当您已有转录文本，只需要获取时间戳时：

1. 添加 ComfyUI 的 "Load Audio" 节点并设置音频文件路径
2. 添加 "WhisperX Alignment" 节点
3. 将 Load Audio 的 `audio` 输出连接到 Alignment 节点
4. 将 `input_type` 设置为 "plain_text"
5. 在 `text_input` 中粘贴您的转录文本
6. 启用 `auto_segment` 以自动分句
7. 根据需要调整 `max_chars_per_segment`（推荐：100-300）
8. 设置语言
9. 运行以获取精确的词级时间戳

**示例：**
```
输入文本："Hello everyone. Today we're going to talk about WhisperX. It's an amazing tool for speech recognition and alignment."
输出：自动分成 3 个片段，带有精确的词级时间戳
```

### 使用外部转录（JSON 格式）

1. 添加 ComfyUI 的 "Load Audio" 节点并设置音频文件路径
2. 添加 "WhisperX Alignment" 节点
3. 将 Load Audio 的 `audio` 连接到 Alignment
4. 将 `input_type` 设置为 "json"
5. 在 `text_input` 中提供 JSON 格式的转录文本
6. 设置语言
7. 运行以获取精确的时间对齐

### 中文/日文文本对齐

自动分段支持中日韩语言：

1. 添加 ComfyUI 的 "Load Audio" 节点并设置音频文件
2. 添加 "WhisperX Alignment" 节点并连接音频
3. 将 `language` 设置为 "zh"（中文）或 "ja"（日文）
4. 使用 `plain_text` 输入类型
5. 启用 `auto_segment`
6. 分段器将使用适当的标点符号（。！？等）

**中文示例：**
```
今天天气很好。我们去公园散步吧。你觉得怎么样？
→ 自动按中文句号分段
```

## 支持的语言

- 中文 (zh)
- 英语 (en)
- 法语 (fr)
- 德语 (de)
- 西班牙语 (es)
- 意大利语 (it)
- 葡萄牙语 (pt)
- 荷兰语 (nl)
- 日语 (ja)
- 自动检测 (auto)

## 高级用法

### 手动选择模型

默认情况下，对齐模型会根据您选择的语言自动选择。但是，您也可以在需要时手动指定模型：

1. 将 `model_name` 保留为 "auto" 以自动选择（推荐）
2. 或指定模型名称如 "WAV2VEC2_ASR_BASE_960H" 以强制加载特定模型

**何时使用手动模型选择：**
- 测试不同的对齐模型
- 使用自定义微调模型
- 调试对齐问题

**示例：**
- 语言：英语 "en"
- 模型名称："auto" → 自动加载最佳英语对齐模型
- 模型名称："WAV2VEC2_ASR_BASE_960H" → 强制使用此特定模型

## 性能优化建议

1. **GPU 加速**：使用 CUDA 以获得更快的处理速度
2. **模型选择**：
   - 将 `model_name` 保留为 "auto" 以获得最佳结果
   - 手动模型选择仅适用于高级用户
3. **文本分段**：
   - 调整 `max_chars_per_segment`（推荐 100-300）
   - 更短的分段 = 更准确的对齐，但处理速度较慢

## 系统要求

- Python 3.8+
- PyTorch 2.0+
- CUDA（可选，用于 GPU 加速）
- ComfyUI

## 常见问题

### WhisperX 安装问题

如果遇到安装问题：

```bash
# 首先安装 PyTorch
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# 然后安装 WhisperX
pip install git+https://github.com/m-bain/whisperx.git
```

### 内存不足错误

- 减少 batch_size
- 使用更小的模型
- 切换到 CPU（速度较慢但使用内存更少）

### 找不到音频文件

- 使用音频文件的绝对路径
- 确保文件存在且可访问
- 检查文件权限

## 致谢

- [WhisperX](https://github.com/m-bain/whisperx) by Max Bain
- 基于 [OpenAI Whisper](https://github.com/openai/whisper)

## 许可证

MIT License

## 支持

如有问题或功能请求，请访问 [GitHub 仓库](https://github.com/loockluo/comfyui-whisperx-pro/issues)。
