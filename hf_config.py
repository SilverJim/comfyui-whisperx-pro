"""
Hugging Face Mirror Configuration for WhisperX Pro
Provides centralized configuration for HF model downloads
"""

import os
from typing import Dict, Optional

# Available Hugging Face mirrors
HF_MIRRORS: Dict[str, str] = {
    "default": "https://huggingface.co",
    "hf-mirror": "https://hf-mirror.com",
    "modelscope": "https://www.modelscope.cn",
    "openi": "https://openi.pcl.ac.cn",
    "gitee": "https://ai.gitee.com",
}

def get_current_mirror() -> str:
    """Get currently configured HF mirror."""
    return os.environ.get('HF_ENDPOINT', HF_MIRRORS["hf-mirror"])

def set_hf_mirror(mirror_name: str) -> bool:
    """
    Set Hugging Face mirror for model downloads.
    
    Args:
        mirror_name: Mirror name to use (see HF_MIRRORS keys)
        
    Returns:
        bool: True if mirror was set successfully, False otherwise
    """
    if mirror_name not in HF_MIRRORS:
        print(f"[ERROR] Unknown mirror '{mirror_name}'. Available mirrors: {list(HF_MIRRORS.keys())}")
        return False
    
    mirror_url = HF_MIRRORS[mirror_name]
    os.environ['HF_ENDPOINT'] = mirror_url
    print(f"[INFO] Hugging Face mirror set to: {mirror_name} ({mirror_url})")
    return True

def auto_configure_mirror() -> str:
    """
    Automatically configure the best mirror based on location/network.
    
    Returns:
        str: The mirror name that was configured
    """
    # If already configured, don't change
    if 'HF_ENDPOINT' in os.environ:
        current = get_current_mirror()
        for name, url in HF_MIRRORS.items():
            if url == current:
                print(f"[INFO] HF mirror already configured: {name} ({current})")
                return name
        print(f"[INFO] HF mirror already configured: custom ({current})")
        return "custom"
    
    # Default to hf-mirror for better connectivity worldwide
    # Users can override by setting HF_ENDPOINT environment variable
    mirror_name = "hf-mirror"
    
    set_hf_mirror(mirror_name)
    return mirror_name

def list_available_mirrors() -> Dict[str, str]:
    """List all available mirrors."""
    return HF_MIRRORS.copy()

def test_mirror_connectivity(mirror_name: str, timeout: float = 10.0) -> bool:
    """
    Test connectivity to a specific mirror.
    
    Args:
        mirror_name: Mirror name to test
        timeout: Timeout in seconds
        
    Returns:
        bool: True if mirror is accessible, False otherwise
    """
    if mirror_name not in HF_MIRRORS:
        return False
    
    try:
        import requests
        url = HF_MIRRORS[mirror_name]
        response = requests.get(url, timeout=timeout)
        success = response.status_code == 200
        if success:
            print(f"[INFO] Mirror '{mirror_name}' is accessible")
        else:
            print(f"[WARNING] Mirror '{mirror_name}' returned status {response.status_code}")
        return success
    except Exception as e:
        print(f"[ERROR] Failed to connect to mirror '{mirror_name}': {e}")
        return False

def get_mirror_status() -> Dict[str, any]:
    """Get current mirror configuration status."""
    current_endpoint = get_current_mirror()
    current_mirror = "custom"
    
    for name, url in HF_MIRRORS.items():
        if url == current_endpoint:
            current_mirror = name
            break
    
    return {
        "current_mirror": current_mirror,
        "current_endpoint": current_endpoint,
        "available_mirrors": list(HF_MIRRORS.keys()),
        "environment_variable": "HF_ENDPOINT" in os.environ
    }

# Auto-configure on import
if __name__ != "__main__":
    auto_configure_mirror()