#!/usr/bin/env python3
"""Align a transcript to audio and export SRT subtitles.

依赖：`torch`、`whisperx`（需要系统安装 `ffmpeg`）以及 `pysubs2`。

pip install torch torchaudio
pip install whisperx pysubs2 jieba

原理：加载 WhisperX 的强制对齐模型（基于 wav2vec 2.0），使用提供的全文字幕作为
      对齐输入，估计每个词的起止时间，并按标点和时长规则组合成 SRT 字幕行。
提示：首次运行会从 HuggingFace 下载对齐模型，请确保可联网或已提前缓存。
用法：`python scripts/align_audio/align_to_srt.py -a your.wav -t your.txt -o your.srt`
      未显式传参时会使用示例 WAV/TXT 文件并输出到 `scripts/align_audio/output.srt`。
"""

import argparse
from pathlib import Path

import torch
import whisperx
from whisperx.audio import SAMPLE_RATE
import pysubs2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用 WhisperX 将文本与音频对齐，并导出 SRT 字幕"
    )
    parser.add_argument(
        "--audio",
        "-a",
        default="scripts/align_audio/asr_example_zh.wav",
        help="输入音频文件路径（wav、mp3 等 ffmpeg 支持的格式）",
    )
    parser.add_argument(
        "--text",
        "-t",
        default="scripts/align_audio/asr_example_zh.txt",
        help="输入文本文件路径（UTF-8 编码）",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="scripts/align_audio/output.srt",
        help="输出 SRT 文件路径，默认 scripts/align_audio/output.srt",
    )
    parser.add_argument(
        "--language",
        "-l",
        default="zh",
        help="对齐模型语言码，默认 zh",
    )
    parser.add_argument(
        "--device",
        "-d",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="执行对齐使用的设备，默认自动检测",
    )
    parser.add_argument(
        "--max-sec",
        type=float,
        default=4.5,
        help="单条字幕最长持续时间（秒），默认 4.5",
    )
    parser.add_argument(
        "--max-ch",
        type=int,
        default=28,
        help="单条字幕最多字符数，默认 28",
    )
    parser.add_argument(
        "--punct",
        default="，。！？；、,.!?;…",
        help="触发换行的标点符号集合",
    )
    return parser.parse_args()


def select_device(choice: str) -> str:
    if choice == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return choice


def flush_line(buffer, subtitles):
    if not buffer:
        return
    start = buffer[0]["start"]
    end = buffer[-1]["end"]
    text = "".join(item["word"] for item in buffer).strip()
    if text:
        subtitles.append((start, end, text))
    buffer.clear()


def align_to_srt(args: argparse.Namespace) -> None:
    audio_path = Path(args.audio)
    text_path = Path(args.text)
    output_path = Path(args.output)

    if not audio_path.exists():
        raise FileNotFoundError(f"音频文件不存在: {audio_path}")
    if not text_path.exists():
        raise FileNotFoundError(f"文本文件不存在: {text_path}")

    device = select_device(args.device)
    print(f"[INFO] 使用设备: {device}")
    print(f"[INFO] 对齐语言: {args.language}")
    print(f"[INFO] 读取音频: {audio_path}")

    audio = whisperx.load_audio(str(audio_path))
    audio_duration = len(audio) / SAMPLE_RATE
    print(f"[INFO] 音频时长: {audio_duration:.2f} 秒")

    print(f"[INFO] 载入文本: {text_path}")
    txt = text_path.read_text(encoding="utf-8").strip()
    if not txt:
        raise ValueError("文本内容为空，无法对齐")
    print(f"[INFO] 文本长度: {len(txt)} 字符")

    print("[INFO] 加载对齐模型...")
    align_model, metadata = whisperx.load_align_model(language_code=args.language, device=device)

    segments = [{"text": txt, "start": 0.0, "end": audio_duration}]
    print(f"[INFO] 向对齐器提交 {len(segments)} 个段落")

    print("[INFO] 开始执行强制对齐...")
    result = whisperx.align(
        segments,
        align_model,
        metadata,
        audio,
        device,
        return_char_alignments=False,
    )
    print("[INFO] 对齐完成")

    words = []
    for seg in result["segments"]:
        for word in seg.get("words", []):
            if (
                word.get("word", "").strip()
                and word.get("start") is not None
                and word.get("end") is not None
            ):
                words.append(word)

    print(f"[INFO] 成功对齐的词数: {len(words)}")

    subtitles = []
    buffer = []
    for word in words:
        buffer.append(word)
        duration = buffer[-1]["end"] - buffer[0]["start"]
        text_line = "".join(item["word"] for item in buffer)
        end_with_punct = word["word"] in args.punct

        if end_with_punct or duration >= args.max_sec or len(text_line) >= args.max_ch:
            flush_line(buffer, subtitles)

    flush_line(buffer, subtitles)
    print(f"[INFO] 生成字幕行数: {len(subtitles)}")

    ss = pysubs2.SSAFile()
    for idx, (start, end, text) in enumerate(subtitles, 1):
        event = pysubs2.SSAEvent(
            start=pysubs2.make_time(s=start),
            end=pysubs2.make_time(s=end),
            text=text,
        )
        ss.events.append(event)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ss.save(str(output_path))
    print(f"[INFO] 已保存 SRT: {output_path}")


def main() -> None:
    args = parse_args()
    align_to_srt(args)


if __name__ == "__main__":
    main()
