# InstructPix2Pix 视频训练交付文档

> 交付对象：下一任负责模型训练与迭代的同学  
> 路径：`dl-final`  
> 最新同步：2025-12-03

## 1. 项目概览

- 目标：在定制的视频编辑数据集上微调 InstructPix2Pix，使模型能够依据文本指令输出下一帧/目标帧。
- 当前成果：
  - 环境、依赖与镜像脚本均配置完毕。
  - 预训练模型与 84 通道定制 UNet 已生成并可复现（`convert_model.py`）。
  - 数据集完成多帧（20 帧输入 + 1 帧目标）转换，存储为 HuggingFace Arrow。
  - `train_video_ip2p.py` 针对多帧输入进行了适配，`run_video_training.py` 可直接以 Python 方式拉起多 GPU 训练。
  - `test_training.sh` 已通过 5 step sanity check，输出位于 `test_output/`。
- 未完成事项：完整大规模训练、验证指标（FID/IS 等）、可视化汇报与超参搜索。

## 2. 关键目录速览

| 路径 | 说明 |
| --- | --- |
| `dl-final/README.md` | 官方使用说明，概述已完成/待办及快速开始流程。|
| `dl-final/models/instruct-pix2pix` | HF 预训练权重（~2.4GB）。|
| `dl-final/models/instruct-pix2pix-video-20frames` | 84 通道定制 UNet（由 `convert_model.py` 生成）。|
| `dl-final/dataset_root` | 原始多任务视频数据全集（train/val/test）。|
| `dl-final/dataset_mini` | 程序生成的 1 条迷你样本，方便快速验证管线。|
| `dl-final/processed_dataset_seq` | 主训练集（Arrow），`input_frames` 为 20 帧路径序列。|
| `dl-final/output_video_model` | 最近一次正式训练的检查点（示例：checkpoint-100）。|
| `dl-final/test_output` | 5 step 测试训练产物 + TensorBoard 日志。|

## 3. 环境与依赖

1. **Conda 建议**
   ```bash
   conda create -n dl-final python=3.10
   conda activate dl-final
   pip install -r requirements.txt
   ```
2. **镜像/下载加速**
   ```bash
   export HF_ENDPOINT=https://hf-mirror.com     # 国内推荐
   # 如需恢复官方
   # export HF_ENDPOINT=https://huggingface.co
   ```
3. **Protobuf 兼容**
   ```bash
   export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
   # 或 pip install "protobuf<=3.20.3"
   ```
4. **可选依赖**
   - `xformers`：显存优化（与 `--enable_xformers_memory_efficient_attention` 配合）。
   - `wandb`：实验追踪（默认仅记录 TensorBoard）。

## 4. 数据资产与管线

1. **原始数据**（`dataset_root`）
   - 三类动作：`drop_object`、`cover_object`、`move_object`，每类约 400 个视频文件夹。
   - `metadata.json` / `metadata.jsonl` 描述帧路径与文本指令。

2. **迷你数据**（`dataset_mini`）
   - 通过 `python create_mini_data.py` 生成 21 张随机图片 + `metadata.json`。
   - 用于快速验证脚本兼容性。

3. **HuggingFace Arrow 多帧格式**（`processed_dataset_seq`）
   - 由 `python make_dataset.py` 生成，特征：
     ```python
     Features({
         "input_frames": Sequence(Value("string"), length=20),
         "edit_prompt": Value("string"),
         "edited_image": Image(),
     })
     ```
   - `input_frames` 内部保存绝对路径字符串，训练时在 `train_video_ip2p.py` 中按需读取并 resize。
   - 数据完整性检查：`python quick_test.py` 或
     ```bash
     python -c "from datasets import load_from_disk; ds = load_from_disk('processed_dataset_seq'); print(len(ds['train']), ds['train'].features)"
     ```

4. **处理流程建议**
   1. （若需）`python create_mini_data.py`
   2. `python make_dataset.py`（如需重新划分 train/test 或切换数据源，修改脚本顶部配置）
   3. 运行 `python quick_test.py` 确认 features / 文件路径无误
   4. 再进入训练阶段

## 5. 模型与权重管理

- 预训练模型：`dl-final/models/instruct-pix2pix`（含 VAE、Tokenizer、CLIP Text Encoder 等）。
- 定制 UNet：`dl-final/models/instruct-pix2pix-video-20frames/unet`
  - `convert_model.py` 将 `conv_in` 通道数改为 `4 + 20*4 = 84`。
  - 初始化策略：保留噪声 latent 与最新条件帧权重，历史 19 帧初始化为 0，训练中逐步学习时间上下文。
- 轻量输出：
  - `test_output/checkpoint-5/`：sanity check，loss 由 0.626 → 0.0134。
  - `output_video_model/checkpoint-100/`：示例正式训练产物（含 optimizer/scheduler 状态）。

## 6. 训练入口与脚本说明

1. **方式 A：Python 一键入口（推荐）**
   - 直接运行 `python run_video_training.py`。
   - 所有参数集中在 `RUN_VIDEO_TRAINING.py` 顶部的 `TRAINING_ARG_LIST`，把需要的字符串值写进去即可（例如改 batch size、步数、输出目录）。
   - 默认使用 4 张 GPU：可用 `export VIDEO_TRAIN_PROCESSES=<n>` 调整。
   - 若需单卡调试，可先 `export VIDEO_TRAIN_PROCESSES=1` 再执行脚本。
   - 脚本内部：
     - 自动设置 `multiprocessing.set_start_method("spawn")`，避免 CUDA fork 报错。
     - 调用 `accelerate.notebook_launcher`，行为等价于 `accelerate launch`，但无需额外命令行。
   - 适合在 Jupyter/VSCode/纯 Python 环境中快速复现；改完参数后只要 `python run_video_training.py` 即可重新拉起训练。

2. **方式 B：显式 `accelerate launch` 命令**
   - 当需要脚本化（如 shell 脚本、集群任务）或想自行控制 `accelerate` 配置文件时，可直接调用 `train_video_ip2p.py`。
   - 使用前先运行一次 `accelerate config`（若尚未生成配置）。
   - 示例命令：
     ```bash
     accelerate launch --num_processes=4 train_video_ip2p.py \
       --pretrained_model_name_or_path dl-final/models/instruct-pix2pix \
       --custom_unet_path dl-final/models/instruct-pix2pix-video-20frames \
       --train_data_dir dl-final/processed_dataset_seq \
       --train_batch_size 2 \
       --gradient_accumulation_steps 4 \
       --max_train_steps 500 \
       --mixed_precision fp16 \
       --output_dir dl-final/output_video_model
     ```
   - 需要改参数时直接在命令中调整；如需 checkpoint 恢复可追加 `--resume_from_checkpoint <path>`。
   - 该脚本的核心逻辑：
     - `--custom_unet_path` 强制加载 84 通道 UNet。
     - `with_transform` 将 20 帧输入堆叠并同步做裁剪/归一化，`collate_fn` 输出 `(batch, 20, 3, H, W)`。
     - 训练循环与 diffusers 官方实现保持一致，兼容 xFormers、gradient checkpointing 等开关。

3. **基线图像版脚本：`train_instruct_pix2pix.py`**
   - 如果需要对单帧任务做对照实验，可继续使用官方 diffusers 脚本（已修复版本检查问题）。

4. **测试/调试**
   - `bash test_training.sh`：默认 5 step，输出日志到 `test_training_output.log`。
   - `python quick_test.py`：仅验证数据 schema。
   - `python run_train.sh`：完整 shell 入口，可按需调整。

## 7. 日志与监控

- TensorBoard：`test_output/logs/instruct-pix2pix`（标准格式，可指向任意 `output_dir/logs`）。
  ```bash
  tensorboard --logdir dl-final/test_output/logs
  ```
- WandB：若在训练命令中添加 `--report_to wandb` 并提前 `wandb login`。
- Shell 日志：`test_training_output.log`、`output_video_model/logs/events.*`。

## 8. 推荐工作流（交付后首轮）

1. **确认环境**：激活 `dl-final`，检查 `nvidia-smi`/CUDA 驱动。
2. **校验数据**：若更换数据源，重新运行 `make_dataset.py` 并 spot-check `processed_dataset_seq/train`。
3. **同步模型**：必要时重跑 `convert_model.py`（确保 UNet 与数据通道一致）。
4. **低步数热身**：先将 `TRAINING_ARG_LIST` 中 `max_train_steps` 降到 50，观察 loss 曲线稳定性。
5. **正式训练**：恢复计划步数（例如 20k step），必要时在 `run_video_training.py` 中开启 `--checkpointing_steps 500`。
6. **评估与可视化**：将新 checkpoint 导入推理脚本（可基于 diffusers pipeline 简单编写）并保存对比图。

## 9. 已知问题与规避建议

| 问题 | 表现 | 解决方案 |
| --- | --- | --- |
| HF 下载慢/失败 | `ConnectionError` | 使用 `setup_mirror.sh` 或手动设置 `HF_ENDPOINT`。|
| Protobuf 报错 | `Descriptors cannot not be created directly` | 设置 `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`。|
| CUDA OOM | 训练中崩溃 | 减小 `--train_batch_size`、开启 `--gradient_checkpointing`、安装 `xformers`。|
| 数据缺帧 | `FileNotFoundError` 或 convert_to_np 失败 | 检查 `dataset_root/.../video_xx/` 是否完整 00-20.jpg，缺失样本会被 `make_dataset.py` 跳过。|
| 多进程启动失败 | `Cannot re-initialize CUDA in forked subprocess` | 仅通过 `run_video_training.py` 或手动调用 `mp.set_start_method("spawn")`。|

## 10. 下一步建议

1. **完整训练计划**：根据显存预算，建议设置 `train_batch_size=1-2`、`gradient_accumulation_steps=8`，目标 10k~30k steps。
2. **验证与指标**：编写推理脚本从 checkpoint 恢复，输出与 GT 对比；计算简单像素指标或 FID。
3. **数据扩展**：若训练不足，可将 `dataset_root` 中 val/test 合并进 train，再保留少量样本作为 hold-out。
4. **超参实验**：优先尝试学习率 (`5e-5` vs `1e-4`)、EMA、以及是否冻结 Text Encoder。
5. **监控**：引入 WandB 或自建日志面板，记录 loss、LR、显存占用。
6. **推理 Demo**：后续交付时最好附上推理脚本 + 样例视频帧展示。

## 11. 支持信息

- 若需更多上下文，详见 `README.md`、`MODEL_DOWNLOAD.md`。
- 重要脚本：`convert_model.py`、`make_dataset.py`、`train_video_ip2p.py`、`run_video_training.py`。
- 遇到未覆盖的问题，建议：
  - 检查 `requirements.txt` 指定版本；
  - 关注 `accelerate` 的警告（尤其是进程数与设备匹配）；
  - 记录实验配置，方便后续同学追溯。

祝训练顺利，如需进一步问题排查可直接在本目录新增 `NOTES-<date>.md` 留下实验记录，方便协作滚动交接。

