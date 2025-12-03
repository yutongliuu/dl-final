# InstructPix2Pix 模型下载说明

## 需要下载的模型

训练脚本需要从 HuggingFace Hub 下载 **`timbrooks/instruct-pix2pix`** 模型。

这个模型包含以下组件：

### 1. 核心组件（必需）

- **scheduler/** - 扩散调度器配置
  - `scheduler_config.json`
  
- **tokenizer/** - CLIP 文本分词器
  - `tokenizer_config.json`
  - `vocab.json`
  - `merges.txt`
  
- **text_encoder/** - CLIP 文本编码器
  - `config.json`
  - `model.safetensors` 或 `pytorch_model.bin`
  
- **vae/** - VAE 编码器/解码器
  - `config.json`
  - `diffusion_pytorch_model.safetensors` 或 `diffusion_pytorch_model.bin`
  
- **unet/** - UNet 模型（这是训练的主要部分）
  - `config.json`
  - `diffusion_pytorch_model.safetensors` 或 `diffusion_pytorch_model.bin`

### 2. 其他文件

- `model_index.json` - 模型索引文件
- `README.md` - 模型说明文档

## 模型大小估算

- text_encoder: ~500 MB
- vae: ~300 MB
- unet: ~1.6 GB
- 其他文件: ~10 MB

**总计约 2.4 GB**

## 使用镜像站点（推荐）

### 设置镜像站点环境变量

```bash
# 使用 HF-Mirror 镜像站点（推荐，速度快）
export HF_ENDPOINT=https://hf-mirror.com

# 或者使用官方站点
export HF_ENDPOINT=https://huggingface.co
```

### 永久设置（添加到 ~/.bashrc）

```bash
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
source ~/.bashrc
```

### 可用的镜像站点

1. **HF-Mirror** (https://hf-mirror.com) - 推荐，速度快且稳定
2. **官方站点** (https://huggingface.co) - 原始站点

## 下载方式

### 方式 1: 使用我创建的下载脚本（推荐，已配置镜像）

```bash
cd /kongweiwen/lyt/dl-final
source /kongweiwen/kongweiwen/miniconda3/etc/profile.d/conda.sh
conda activate dl-final

# 使用镜像站点下载（默认使用 hf-mirror）
python download_model.py

# 或者指定本地目录
python download_model.py --local-dir /path/to/save/model

# 不使用镜像站点（使用官方站点）
python download_model.py --no-mirror
```

### 方式 2: 使用 huggingface-cli

```bash
# 先设置镜像站点
export HF_ENDPOINT=https://hf-mirror.com

# 安装 huggingface-cli
pip install huggingface_hub[cli]

# 下载整个模型
huggingface-cli download timbrooks/instruct-pix2pix --local-dir /path/to/local/model

# 或者只下载特定文件
huggingface-cli download timbrooks/instruct-pix2pix --local-dir /path/to/local/model --include "*.json" "*.safetensors" "*.bin"
```

### 方式 3: 使用 Python 脚本

```python
from huggingface_hub import snapshot_download

# 下载整个模型
snapshot_download(
    repo_id="timbrooks/instruct-pix2pix",
    local_dir="/path/to/local/model",
    local_dir_use_symlinks=False
)
```

### 方式 4: 使用 git-lfs（如果有 git）

```bash
git lfs install
git clone https://huggingface.co/timbrooks/instruct-pix2pix /path/to/local/model
```

### 方式 5: 手动下载（通过浏览器）

访问：https://huggingface.co/timbrooks/instruct-pix2pix/tree/main

手动下载所有文件到本地目录。

## 使用本地模型

下载完成后，修改训练脚本中的模型路径：

```bash
python train_instruct_pix2pix.py \
    --pretrained_model_name_or_path /path/to/local/model \
    --train_data_dir /kongweiwen/lyt/dl-final/dataset_root/test_data \
    ...
```

## 检查模型是否已下载

模型文件通常缓存在：
- Linux: `~/.cache/huggingface/hub/models--timbrooks--instruct-pix2pix/`

检查缓存：
```bash
ls -lh ~/.cache/huggingface/hub/models--timbrooks--instruct-pix2pix/
```

## 网络问题解决方案

如果无法直接访问 HuggingFace Hub，可以：

1. **使用镜像站点**（如果有）
2. **配置代理**
3. **使用 VPN**
4. **手动下载后放到本地路径**

