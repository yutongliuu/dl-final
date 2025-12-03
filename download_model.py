#!/usr/bin/env python
# coding=utf-8
"""Download InstructPix2Pix model from HuggingFace Hub using mirror site."""

import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

# 设置镜像站点（默认使用 HF-Mirror）
MIRROR_SITES = {
    "hf-mirror": "https://hf-mirror.com",
    "official": "https://huggingface.co"
}

def download_model(local_dir=None, use_mirror=True, mirror_name="hf-mirror"):
    """Download the InstructPix2Pix model using mirror site.
    
    Args:
        local_dir: 本地保存目录
        use_mirror: 是否使用镜像站点
        mirror_name: 镜像站点名称 (hf-mirror 或其他)
    """
    if local_dir is None:
        local_dir = "/kongweiwen/lyt/dl-final/models/instruct-pix2pix"
    
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置镜像站点环境变量
    if use_mirror and mirror_name in MIRROR_SITES:
        mirror_url = MIRROR_SITES[mirror_name]
        os.environ["HF_ENDPOINT"] = mirror_url
        print(f"Using mirror site: {mirror_url}")
    else:
        print("Using official HuggingFace site")
    
    print(f"Downloading timbrooks/instruct-pix2pix to {local_dir}...")
    print("This may take a while (model size ~2.4 GB)...")
    print("Please be patient...\n")
    
    try:
        snapshot_download(
            repo_id="timbrooks/instruct-pix2pix",
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            resume_download=True
        )
        print(f"\n✓ Model downloaded successfully to: {local_dir}")
        print(f"\nYou can now use it with:")
        print(f"  --pretrained_model_name_or_path {local_dir}")
        return True
    except Exception as e:
        print(f"\n✗ Error downloading model: {e}")
        print("\nPossible solutions:")
        print("1. Check your internet connection")
        print("2. Try a different mirror site")
        print("3. Manually download from: https://huggingface.co/timbrooks/instruct-pix2pix")
        return False

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download InstructPix2Pix model")
    parser.add_argument("--local-dir", type=str, default=None, help="Local directory to save model")
    parser.add_argument("--no-mirror", action="store_true", help="Don't use mirror site")
    parser.add_argument("--mirror", type=str, default="hf-mirror", choices=list(MIRROR_SITES.keys()), 
                       help="Mirror site to use")
    args = parser.parse_args()
    
    download_model(
        local_dir=args.local_dir,
        use_mirror=not args.no_mirror,
        mirror_name=args.mirror
    )

