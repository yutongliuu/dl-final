#!/bin/bash
# Setup HuggingFace mirror site

echo "Setting up HuggingFace mirror site..."

# 设置 HF-Mirror 镜像站点（推荐，速度快且稳定）
export HF_ENDPOINT=https://hf-mirror.com

# 可选：其他镜像站点
# export HF_ENDPOINT=https://huggingface.co  # 官方站点

echo "HF_ENDPOINT set to: $HF_ENDPOINT"
echo ""
echo "To make this permanent, add to your ~/.bashrc:"
echo "  export HF_ENDPOINT=https://hf-mirror.com"
echo ""
echo "Available mirror sites:"
echo "  1. https://hf-mirror.com (HF-Mirror, recommended)"
echo "  2. https://huggingface.co (Official site)"

