"""
Internationalization utilities for WhisperX Pro nodes
Provides language detection and text localization
"""

import os
import locale
from typing import Dict, Any, Optional

def detect_system_language() -> str:
    """
    Detect system language and return language code.
    
    Returns:
        str: Language code ('zh' for Chinese, 'en' for English, etc.)
    """
    try:
        # Try to get system locale
        system_locale = locale.getdefaultlocale()[0]
        if system_locale:
            # Extract language code (first 2 characters)
            lang_code = system_locale.lower()[:2]
            return lang_code
    except:
        pass
    
    # Fallback: check environment variables
    for env_var in ['LANG', 'LANGUAGE', 'LC_ALL', 'LC_MESSAGES']:
        env_value = os.environ.get(env_var, '')
        if env_value:
            lang_code = env_value.lower()[:2]
            if lang_code in ['zh', 'en', 'ja', 'ko', 'fr', 'de', 'es', 'it', 'pt', 'nl']:
                return lang_code
    
    # Default to English
    return 'en'

def is_chinese_environment() -> bool:
    """Check if current environment is Chinese."""
    lang = detect_system_language()
    return lang == 'zh'

class I18nText:
    """Internationalization text container."""
    
    def __init__(self, zh: str, en: str):
        self.zh = zh
        self.en = en
    
    def get(self, force_lang: Optional[str] = None) -> str:
        """Get localized text based on system language or forced language."""
        if force_lang:
            return self.zh if force_lang == 'zh' else self.en
        
        return self.zh if is_chinese_environment() else self.en
    
    def __str__(self) -> str:
        return self.get()

# Localized text definitions
TEXTS = {
    # Input tooltips
    'audio_tooltip': I18nText(
        zh="输入音频文件",
        en="Input audio file"
    ),
    'text_tooltip': I18nText(
        zh="需要与音频对齐的文本内容，确保文本与音频内容匹配",
        en="Text content to align with audio, ensure text matches audio content"
    ),
    'text_placeholder': I18nText(
        zh="输入要与音频对齐的文本内容",
        en="Enter the text to align with audio"
    ),
    'language_tooltip': I18nText(
        zh="选择音频语言\nen=英语, fr=法语, de=德语, es=西班牙语, it=意大利语, pt=葡萄牙语, nl=荷兰语, ja=日语, zh=中文",
        en="Select audio language\nen=English, fr=French, de=German, es=Spanish, it=Italian, pt=Portuguese, nl=Dutch, ja=Japanese, zh=Chinese"
    ),
    'max_sec_tooltip': I18nText(
        zh="单行字幕最大时长（秒）",
        en="Maximum duration per subtitle line (seconds)"
    ),
    'max_ch_tooltip': I18nText(
        zh="单行字幕最大字符数",
        en="Maximum characters per subtitle line"
    ),
    'punct_tooltip': I18nText(
        zh="当遇到这些标点符号时会强制换行",
        en="Force line break when these punctuation marks are encountered"
    ),
    'punct_placeholder': I18nText(
        zh="触发换行的标点符号",
        en="Punctuation marks that trigger line breaks"
    ),
    'device_tooltip': I18nText(
        zh="选择计算设备\nauto=自动选择, cuda=GPU, cpu=CPU",
        en="Select compute device\nauto=auto select, cuda=GPU, cpu=CPU"
    ),
    
    # Output names
    'srt_content_name': I18nText(
        zh="SRT字幕内容",
        en="SRT Content"
    ),
    'alignment_info_name': I18nText(
        zh="对齐信息",
        en="Alignment Info"
    ),
    
    # Category and display names
    'category_name': I18nText(
        zh="音频处理",
        en="Audio"
    ),
    'node_display_name': I18nText(
        zh="WhisperX SRT字幕生成器",
        en="WhisperX SRT Generator"
    ),
}

def get_localized_input_types() -> Dict[str, Any]:
    """Get localized INPUT_TYPES configuration."""
    return {
        "required": {
            "audio": ("AUDIO", {
                "tooltip": TEXTS['audio_tooltip'].get()
            }),
            "text": ("STRING", {
                "default": "",
                "multiline": True,
                "placeholder": TEXTS['text_placeholder'].get(),
                "tooltip": TEXTS['text_tooltip'].get()
            }),
            "language": (["en", "fr", "de", "es", "it", "pt", "nl", "ja", "zh"], {
                "default": "zh",
                "tooltip": TEXTS['language_tooltip'].get()
            }),
            "max_sec": ("FLOAT", {
                "default": 4.5,
                "min": 1.0,
                "max": 10.0,
                "step": 0.1,
                "display": "number",
                "tooltip": TEXTS['max_sec_tooltip'].get()
            }),
            "max_ch": ("INT", {
                "default": 28,
                "min": 10,
                "max": 100,
                "step": 1,
                "display": "number",
                "tooltip": TEXTS['max_ch_tooltip'].get()
            }),
            "punct": ("STRING", {
                "default": "，。！？；、,.!?;…",
                "multiline": False,
                "placeholder": TEXTS['punct_placeholder'].get(),
                "tooltip": TEXTS['punct_tooltip'].get()
            }),
            "unload_model": ("BOOLEAN", {
                "default": False,
                "tooltip": "Unload models from VRAM after processing"
            }),
        },
        "optional": {
            "device": (["auto", "cuda", "cpu"], {
                "default": "auto",
                "tooltip": TEXTS['device_tooltip'].get()
            }),
        }
    }

def get_localized_return_names() -> tuple:
    """Get localized return names."""
    return (
        TEXTS['srt_content_name'].get(),
        TEXTS['alignment_info_name'].get()
    )

def get_localized_category() -> str:
    """Get localized category name."""
    return TEXTS['category_name'].get()

def get_localized_display_name() -> str:
    """Get localized node display name."""
    return TEXTS['node_display_name'].get()