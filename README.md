# InstructPix2Pix 模型训练项目

本项目用于训练 InstructPix2Pix 模型，该模型可以根据文本指令对图像进行编辑。

## 🗂️ 代码与大文件拆分策略

> GitHub 仓库仅提交可复现的代码与配置；所有数据集、模型权重、训练输出统一托管在外部存储（Hugging Face Hub / OSS / 百度网盘），以避免 100GB 级别资产被推送到 GitHub。

- `.gitignore` 已忽略以下目录/文件：`dataset_root/`、`dataset_mini/`、`processed_dataset_seq/`、`models/`、`output_video_model/`、`test_output/`、`dataset_root.zip` 等。开发者本地可照常使用这些目录，但在 Git 提交前无需关心其状态。
- 若需要恢复或分享数据，请按照下面任一渠道下载，然后解压/覆写到对应路径即可。

### 外部资产汇总

| 资产 | 描述 | 推荐存储 | 备注 |
| --- | --- | --- | --- |
| `dataset_root` | 完整原始视频数据（train/val/test） | Hugging Face Datasets：`https://huggingface.co/datasets/<org>/dl-final-dataset` | 上传 `dataset_root.zip` 并在 README 中记录版本 |
| `processed_dataset_seq` | Arrow 多帧序列数据集 | 同上（也可放 OSS） | 运行 `make_dataset.py` 后打包上传 |
| `models/instruct-pix2pix`、`models/instruct-pix2pix-video-20frames` | 预训练 + 84 通道定制 UNet | Hugging Face Models：`https://huggingface.co/<org>/dl-final-models` | 也可使用 `python download_model.py` 重新下载 |
| `output_video_model/`、`test_output/` | 训练 checkpoint、TensorBoard 日志 | OSS：`oss://<bucket>/dl-final/checkpoints/` | 仅保留最近若干版本 |
| `dataset_mini/`、`dataset_root.zip` | 调试用迷你数据 & 数据压缩包 | 百度网盘分享链接 | 方便对外协作或无法访问 HF/OSS 的同学 |

> 将 `<org>`、`<bucket>`、`<share-id>` 等占位符替换为团队实际值；当外部链接变更时，请同步更新本节内容。

### 下载方式示例

#### 1. Hugging Face Hub（主渠道）

```bash
pip install huggingface_hub[cli]
huggingface-cli login  # 使用拥有 <org>/dl-final-* 权限的账号
export HF_ENDPOINT=https://hf-mirror.com  # 如需加速，可改为官方站点

# 数据集
huggingface-cli download <org>/dl-final-dataset \
  --repo-type dataset \
  --local-dir ./dataset_root_sync
rsync -a dataset_root_sync/dataset_root ./dataset_root

# 模型 / 检查点
huggingface-cli download <org>/dl-final-models \
  --local-dir ./models_sync \
  --include "models/**" "output_video_model/**"
rsync -a models_sync/models ./models
rsync -a models_sync/output_video_model ./output_video_model
```

更多 Hugging Face 相关说明可参考 [`MODEL_DOWNLOAD.md`](MODEL_DOWNLOAD.md) 与 `download_model.py`。

#### 2. 阿里云 OSS（内网备份）

```bash
# 首次使用需执行 ossutil config，填入 <bucket> 的 endpoint 与 AK 信息
ossutil cp -r oss://<bucket>/dl-final/dataset_root ./dataset_root
ossutil cp -r oss://<bucket>/dl-final/processed_dataset_seq ./processed_dataset_seq
ossutil cp -r oss://<bucket>/dl-final/output_video_model ./output_video_model
```

#### 3. 百度网盘（便捷分享）

```
链接：https://pan.baidu.com/s/<share-id>
提取码：<code>
内容：dataset_root.zip、processed_dataset_seq.tar、output_video_model-checkpoint-***.zip
```

上述链接适用于无法访问 HF/OSS 的协作者，可根据需要替换为最新分享地址。

## 📊 项目进展

### ✅ 已完成

- [x] **环境配置**
  - Conda 环境已创建并配置（dl-final）
  - 所有依赖包已安装（requirements.txt）
  - Protobuf 兼容性问题已修复

- [x] **模型准备**
  - 预训练模型已下载（使用 HF-Mirror 镜像站点）
  - 模型大小：约 2.4GB（包含 UNet, VAE, Text Encoder 等）
  - 模型路径：`dl-final/models/instruct-pix2pix`

- [x] **数据集准备**
  - 数据集已解压并整理（dataset_root.zip）
  - 训练集：1500 个样本（3个任务类型：drop_object, cover_object, move_object）
  - 验证集：已准备
  - 测试数据：已转换为训练格式（test_data/）
  - 迷你测试数据集：已创建（dataset_mini/），用于快速验证数据流程
  - ✅ 数据集转换脚本：`make_dataset.py`（支持多帧输入序列，存储帧路径字符串）
  - ✅ 新数据集格式：`processed_dataset_seq`（HuggingFace Arrow 数据集，`input_frames` 序列元素为帧路径）

- [x] **训练脚本**
  - 主训练脚本：`train_instruct_pix2pix.py`（已修复版本检查问题）
  - 测试训练脚本：`test_training.sh`
  - 数据准备脚本：
    - `make_dataset.py`：将原始数据转换为 HuggingFace Dataset 格式（支持多帧输入序列）
    - `prepare_test_data.py`：准备测试数据（从 test_video 转换）
    - `create_mini_data.py`：创建迷你测试数据集（生成随机测试图片）
  - 模型下载脚本：`download_model.py`（支持镜像站点）
  - ✅ 新增 `run_video_training.py`：纯 Python 入口，无需命令行即可启动多 GPU 训练

- [x] **测试验证**
  - ✅ 数据格式验证通过（quick_test.py）
  - ✅ 测试训练成功完成（5步训练）
  - ✅ 训练损失正常下降（0.626 → 0.0134）
  - ✅ 检查点保存成功（checkpoint-5）
  - ✅ 模型可以正常加载和训练

### 🚧 进行中

- [ ] **完整训练**
  - 使用完整数据集进行训练
  - 调整超参数以获得最佳效果
  - 监控训练过程（TensorBoard/WandB）

### 📋 待完成

- [ ] **模型评估**
  - 在验证集上评估模型性能
  - 生成样本图像进行可视化
  - 计算评估指标（如 FID, IS 等）

- [ ] **模型优化**
  - 超参数调优
  - 尝试不同的训练策略
  - 模型压缩和优化

- [ ] **文档完善**
  - 训练结果分析
  - 最佳实践总结
  - 使用案例和示例

### 📈 测试训练结果

**测试训练统计**（2025-12-01）：
- 训练步数：5 步
- 训练时间：~29 秒
- 初始损失：0.626
- 最终损失：0.0134
- 学习率：1e-4
- 批次大小：1
- 分辨率：256x256
- 状态：✅ 成功

**检查点信息**：
- 保存位置：`test_output/checkpoint-5/`
- 包含组件：UNet, Optimizer, Scheduler, Scaler
- 文件大小：~6.5GB（包含优化器状态）

## 📋 目录结构

```
dl-final/
├── README.md                    # 项目说明文档（本文件）
├── requirements.txt             # Python 依赖包列表
├── MODEL_DOWNLOAD.md           # 模型下载详细说明文档
│
├── 训练相关脚本/
│   ├── train_instruct_pix2pix.py   # 主训练脚本（HuggingFace 官方）
│   ├── run_train.sh               # 完整训练启动脚本
│   ├── test_training.sh            # 测试训练脚本（快速验证）
│   └── test_training_output.log    # 测试训练日志输出
│
├── 数据准备脚本/
│   ├── prepare_test_data.py        # 准备测试数据（从 test_video 转换）
│   ├── make_dataset.py             # 数据集制作脚本
│   ├── create_mini_data.py         # 创建迷你测试数据集（生成随机测试图片）
│   └── quick_test.py               # 快速数据格式验证脚本
│
├── 工具脚本/
│   ├── download_model.py           # 模型下载脚本（支持镜像站点）
│   └── setup_mirror.sh             # HuggingFace 镜像站点设置脚本
│
├── src/                          # 源代码目录（可选）
│   ├── dataset.py                 # 数据集处理模块
│   └── train.py                   # 训练模块
│
├── dataset_root/                 # 数据集根目录
│   ├── metadata.json              # 数据集元数据（包含所有样本信息）
│   ├── dataset_root.zip           # 数据集压缩包（原始数据）
│   │
│   ├── train/                     # 训练数据集
│   │   ├── drop_object/           # 放置物体任务（~400个视频）
│   │   │   └── video_*/           # 每个视频一个目录
│   │   │       ├── 00.jpg         # 原始图像
│   │   │       ├── 01.jpg         # 编辑后图像
│   │   │       ├── 02.jpg         # 中间帧（如有）
│   │   │       └── ...
│   │   ├── cover_object/           # 覆盖物体任务（~400个视频）
│   │   │   └── video_*/           # 视频目录结构同上
│   │   └── move_object/           # 移动物体任务（~400个视频）
│   │       └── video_*/           # 视频目录结构同上
│   │
│   ├── val/                       # 验证数据集
│   │   ├── drop_object/           # 放置物体验证集
│   │   ├── cover_object/          # 覆盖物体验证集
│   │   └── move_object/           # 移动物体验证集
│   │
│   ├── test_video/                # 测试视频数据（原始）
│   │   ├── 00.jpg                 # 测试原始图像
│   │   ├── 01.jpg                 # 测试编辑后图像
│   │   └── test_video.json        # 测试视频元数据
│   │
│   └── test_data/                 # 处理后的测试数据（用于快速验证）
│       ├── input_image.jpg        # 输入图像（来自 test_video/00.jpg）
│       ├── edited_image.jpg        # 编辑后图像（来自 test_video/01.jpg）
│       └── metadata.jsonl         # 测试数据元数据（HuggingFace 格式）
│
├── dataset_mini/                 # 迷你测试数据集（程序生成）
│   ├── metadata.json             # 数据集元数据
│   └── train/                    # 训练数据
│       └── video_test_01/        # 测试视频目录
│           ├── 00.jpg            # 第1帧（输入）
│           ├── 01.jpg            # 第2帧
│           ├── ...               # 中间帧
│           └── 20.jpg            # 第21帧（目标输出）
│
├── processed_dataset_seq/        # 处理后的数据集（HuggingFace 格式，支持多帧输入）
│   ├── dataset_dict.json        # 数据集字典配置
│   ├── train/                   # 训练集
│   │   ├── dataset_info.json    # 数据集信息（包含 features 定义）
│   │   ├── state.json           # 状态信息
│   │   └── data-*.arrow         # Arrow 格式数据文件
│   └── test/                    # 测试集（如果数据量足够）
│
├── processed_dataset_mini_test/ # 旧版处理后的数据集（单帧输入格式）
│   └── train/                   # 训练集
│
├── models/                       # 模型存储目录
│   └── instruct-pix2pix/         # InstructPix2Pix 预训练模型
│       ├── README.md              # 模型说明文档
│       ├── model_index.json       # 模型索引文件
│       │
│       ├── unet/                  # UNet 模型（核心组件，~3.3GB）
│       │   ├── config.json        # UNet 配置
│       │   └── diffusion_pytorch_model.safetensors
│       │
│       ├── vae/                   # VAE 编码器/解码器（~300MB）
│       │   ├── config.json
│       │   └── diffusion_pytorch_model.safetensors
│       │
│       ├── text_encoder/          # CLIP 文本编码器（~500MB）
│       │   ├── config.json
│       │   └── model.safetensors
│       │
│       ├── tokenizer/             # CLIP 分词器
│       │   ├── tokenizer_config.json
│       │   ├── vocab.json
│       │   └── merges.txt
│       │
│       ├── scheduler/             # 扩散调度器
│       │   └── scheduler_config.json
│       │
│       ├── safety_checker/        # 安全检查器（可选）
│       │   └── model.safetensors
│       │
│       └── feature_extractor/     # 特征提取器
│           └── preprocessor_config.json
│
└── test_output/                  # 测试训练输出目录
    ├── checkpoint-5/              # 训练检查点（第5步）
    │   ├── unet/                  # 训练后的 UNet
    │   ├── optimizer.bin          # 优化器状态
    │   ├── scheduler.bin          # 学习率调度器状态
    │   └── scaler.pt              # 混合精度缩放器
    │
    ├── logs/                      # 训练日志
    │   └── instruct-pix2pix/      # TensorBoard 日志
    │
    ├── unet/                      # 最终训练后的 UNet
    ├── vae/                       # VAE（从预训练模型复制）
    ├── text_encoder/              # 文本编码器（从预训练模型复制）
    ├── tokenizer/                 # 分词器（从预训练模型复制）
    ├── scheduler/                 # 调度器（从预训练模型复制）
    └── model_index.json           # 模型索引文件
```

### 关键目录说明

- **dataset_root/train/**: 包含三个任务类型的训练数据，每个任务约400个视频样本
- **dataset_root/val/**: 验证集数据，结构与训练集相同
- **dataset_root/test_data/**: 用于快速验证的测试数据（1个样本）
- **dataset_mini/**: 迷你测试数据集，包含程序生成的随机测试图片（21张），用于快速验证数据流程
- **processed_dataset_seq/**: 处理后的 HuggingFace Dataset 格式数据集，支持多帧输入序列（`input_frames: List[str]`，元素为帧路径）
- **processed_dataset_mini_test/**: 旧版处理后的数据集（单帧输入格式，已废弃）
- **models/instruct-pix2pix/**: 预训练模型，总大小约 2.4GB
- **test_output/**: 测试训练的输出，包含检查点和最终模型

## 🚀 快速开始

### 1. 环境准备

#### 使用 Conda（推荐）

```bash
# 如果没有环境，创建新环境
conda create -n dl-final python=3.10
conda activate dl-final
```

#### 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 下载预训练模型

#### 方式 1: 使用下载脚本（推荐，已配置镜像）

```bash
# 使用 HF-Mirror 镜像站点（国内速度快）
export HF_ENDPOINT=https://hf-mirror.com
python download_model.py
```

#### 方式 2: 手动下载

详见 [MODEL_DOWNLOAD.md](MODEL_DOWNLOAD.md)

### 3. 准备测试数据

#### 方式 1: 创建迷你测试数据集并转换为 HuggingFace 格式（推荐）

```bash
# 步骤 1: 生成迷你测试数据集（包含21张随机生成的测试图片）
python create_mini_data.py

# 步骤 2: 转换为 HuggingFace Dataset 格式（支持多帧输入序列）
python make_dataset.py
```

这会创建：
- `dataset_mini/`: 原始测试数据（21张图片，00.jpg 到 20.jpg）
- `processed_dataset_seq/`: HuggingFace Dataset 格式数据集
  - 特征：`input_frames` (List[Image], 20帧), `edit_prompt` (string), `edited_image` (Image)

**注意**：`make_dataset.py` 支持多帧输入序列，将 00.jpg-19.jpg 作为输入帧，20.jpg 作为目标输出。

#### 方式 2: 准备真实测试数据

```bash
# 准备测试数据（从 test_video 目录）
python prepare_test_data.py
```

### 4. 运行测试训练

```bash
# 运行快速测试（5步训练，验证环境是否正确）
bash test_training.sh
```

### 5. 使用 Python 启动视频训练（推荐）

无需手动输入 `accelerate launch ...`，可直接运行：

```bash
python run_video_training.py
```

说明：
- 脚本内部调用 `accelerate.notebook_launcher`，默认启动 4 个进程（可通过 `VIDEO_TRAIN_PROCESSES` 环境变量调整，如 `export VIDEO_TRAIN_PROCESSES=1`）。
- 所有训练超参数集中在 `run_video_training.py` 的 `TRAINING_ARG_LIST` 中，可按需修改。
- 启动时会强制把多进程模式切换为 `spawn`，避免 CUDA fork 子进程时报 “Cannot re-initialize CUDA in forked subprocess”。
- 若需要恢复命令行用法，`train_video_ip2p.py` 仍然支持 `accelerate launch`。

### 6. 完整训练

```bash
# 使用完整数据集进行训练
python train_instruct_pix2pix.py \
    --pretrained_model_name_or_path dl-final/models/instruct-pix2pix \
    --train_data_dir dl-final/dataset_root/train \
    --original_image_column input_image \
    --edit_prompt_column edit_prompt \
    --edited_image_column edited_image \
    --resolution 256 \
    --train_batch_size 4 \
    --num_train_epochs 100 \
    --learning_rate 1e-4 \
    --output_dir dl-final/output \
    --mixed_precision fp16 \
    --checkpointing_steps 500
```

## 📦 依赖说明

主要依赖包：

- **accelerate**: 分布式训练支持
- **diffusers**: HuggingFace 扩散模型库
- **transformers**: HuggingFace 模型库
- **torch**: PyTorch 深度学习框架
- **datasets**: HuggingFace 数据集库
- **wandb**: 实验跟踪（可选）

完整依赖列表见 [requirements.txt](requirements.txt)

## 🔧 配置说明

### 使用镜像站点（加速下载）

```bash
# 设置 HF-Mirror 镜像（推荐）
export HF_ENDPOINT=https://hf-mirror.com

# 或使用官方站点
export HF_ENDPOINT=https://huggingface.co
```

### 修复 Protobuf 兼容性问题

```bash
# 设置环境变量
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# 或降级 protobuf
pip install "protobuf<=3.20.3"
```

## 📊 数据集格式

### 原始数据格式

训练数据需要遵循以下格式：

#### 目录结构

```
dataset_root/
├── metadata.jsonl    # 元数据文件（必需）
├── image1.jpg        # 图像文件
├── image2.jpg
└── ...
```

#### metadata.jsonl 格式（单帧输入）

每行一个 JSON 对象，包含以下字段：

```json
{
  "file_name": "input_image.jpg",
  "input_image": "input_image.jpg",
  "edit_prompt": "dropping betel onto the chair",
  "edited_image": "edited_image.jpg"
}
```

- `file_name`: 主图像文件名（imagefolder 格式要求）
- `input_image`: 原始图像文件名
- `edit_prompt`: 编辑指令文本
- `edited_image`: 编辑后的图像文件名

### 多帧输入序列格式（新格式）

对于支持多帧输入的数据（如视频序列），使用 `make_dataset.py` 转换为 HuggingFace Dataset 格式：

#### 原始数据目录结构

```
dataset_mini/
├── metadata.json              # 元数据文件
└── train/
    └── video_test_01/         # 视频目录
        ├── 00.jpg             # 第1帧（输入）
        ├── 01.jpg             # 第2帧（输入）
        ├── ...
        ├── 19.jpg             # 第20帧（输入）
        └── 20.jpg             # 第21帧（目标输出）
```

#### metadata.json 格式（多帧输入）

```json
[
  {
    "video_path": "train/video_test_01",
    "instruction": "move the object to the right"
  }
]
```

- `video_path`: 视频目录的相对路径
- `instruction`: 编辑指令文本

#### 转换后的 HuggingFace Dataset 格式

使用 `make_dataset.py` 转换后，生成的数据集包含以下特征（帧以路径字符串存储，训练时再读取图片）：

```python
Features({
    "input_frames": Sequence(Value("string")),  # 多帧输入（20帧：00.jpg-19.jpg路径）
    "edit_prompt": Value("string"),             # 编辑指令
    "edited_image": Image(),                    # 目标输出（20.jpg）
})
```

**注意**：`input_frames` 的元素是帧路径字符串（Sequence(Value("string"))），训练脚本会在读取 batch 时用 PIL 打开并转换。

## 🎯 训练参数说明

### 主要参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--pretrained_model_name_or_path` | 预训练模型路径 | 必需 |
| `--train_data_dir` | 训练数据目录 | 必需 |
| `--output_dir` | 输出目录 | `instruct-pix2pix-model` |
| `--resolution` | 图像分辨率 | `256` |
| `--train_batch_size` | 批次大小 | `16` |
| `--num_train_epochs` | 训练轮数 | `100` |
| `--learning_rate` | 学习率 | `1e-4` |
| `--mixed_precision` | 混合精度 | `fp16` |
| `--checkpointing_steps` | 检查点保存步数 | `500` |

### 数据列名参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--original_image_column` | 原始图像列名 | `input_image` |
| `--edit_prompt_column` | 编辑指令列名 | `edit_prompt` |
| `--edited_image_column` | 编辑后图像列名 | `edited_image` |

完整参数列表：

```bash
python train_instruct_pix2pix.py --help
```

## 🧪 测试验证

### 创建迷你测试数据集并转换

```bash
# 步骤 1: 生成迷你测试数据集（用于快速验证数据流程）
python create_mini_data.py

# 步骤 2: 转换为 HuggingFace Dataset 格式
python make_dataset.py
```

**说明**：
- `create_mini_data.py` 生成：
  - 21 张随机测试图片（00.jpg 到 20.jpg）
  - 图片尺寸：128x128 像素
  - 图片类型：随机噪点图（程序自动生成，非真实视频帧）
  - 输出目录：`dataset_mini/`
  - 包含 `metadata.json` 索引文件

- `make_dataset.py` 转换：
  - 输入：`dataset_mini/`（原始数据）
  - 输出：`processed_dataset_seq_test/`（HuggingFace Dataset 格式）
  - 特征：`input_frames` (20帧), `edit_prompt`, `edited_image`
  - 配置：可在脚本中修改 `DATA_ROOT` 和 `OUTPUT_DIR`

### 验证数据集格式

```bash
# 验证转换后的数据集
python -c "from datasets import load_from_disk; ds = load_from_disk('processed_dataset_seq'); print('数据集大小:', len(ds['train'])); print('特征:', ds['train'].features)"
```

### 快速测试数据格式

```bash
python quick_test.py
```

### 运行测试训练

```bash
bash test_training.sh
```

测试训练会：
- 使用测试数据（1个样本）
- 训练 5 步
- 验证所有组件是否正常工作
- 保存检查点到 `test_output/`

## 📝 训练日志

训练日志保存在：

- TensorBoard: `{output_dir}/logs/`
- WandB: 如果配置了 `--report_to wandb`

查看 TensorBoard：

```bash
tensorboard --logdir dl-final/test_output/logs
```

## 🔍 常见问题

### 1. 模型下载失败

**问题**: 无法从 HuggingFace Hub 下载模型

**解决方案**:
- 使用镜像站点：`export HF_ENDPOINT=https://hf-mirror.com`
- 手动下载后使用本地路径
- 检查网络连接

### 2. Protobuf 版本冲突

**问题**: `TypeError: Descriptors cannot not be created directly`

**解决方案**:
```bash
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
pip install "protobuf<=3.20.3"
```

### 3. CUDA 内存不足

**解决方案**:
- 减小 `--train_batch_size`
- 启用 `--gradient_checkpointing`
- 使用 `--mixed_precision fp16` 或 `bf16`
- 安装 xformers: `pip install xformers` 并使用 `--enable_xformers_memory_efficient_attention`

### 4. 数据格式错误

**问题**: `ValueError: file_name must be present`

**解决方案**:
- 确保 `metadata.jsonl` 包含 `file_name` 字段
- 检查图像文件路径是否正确
- 运行 `python quick_test.py` 验证数据格式

### 5. 数据集转换问题

**问题**: `FileNotFoundError: No such file or directory`

**解决方案**:
- 确保使用绝对路径（`make_dataset.py` 已自动处理）
- 检查 `dataset_mini/metadata.json` 中的路径是否正确
- 确保所有帧文件（00.jpg-20.jpg）都存在
- 运行 `python make_dataset.py` 查看详细错误信息

**问题**: 数据集 features 中显示 `List` 而不是 `Sequence`

**说明**:
- JSON 序列化时 `Sequence(Value("string"))` 会以 `List` 形式出现
- `List` 和 `Sequence` 在 HuggingFace datasets 中功能等价
- 可以通过 `ds['train'].features` 验证实际类型，并确认 `input_frames` 的元素是字符串

## 📚 参考资料

- [InstructPix2Pix 论文](https://arxiv.org/abs/2211.09800)
- [HuggingFace Diffusers 文档](https://huggingface.co/docs/diffusers)
- [模型下载说明](MODEL_DOWNLOAD.md)

## 📄 许可证

本项目基于 HuggingFace Diffusers 的 InstructPix2Pix 训练脚本，遵循 Apache 2.0 许可证。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📧 联系方式

如有问题，请提交 Issue 或联系项目维护者。

---

**最后更新**: 2025-12-02

**更新内容**:
- ✅ 添加多帧输入序列支持（`make_dataset.py`，帧路径存储为字符串序列）
- ✅ 新增 `run_video_training.py`，支持纯 Python 启动训练
- ✅ 更新数据集转换文档和使用说明

