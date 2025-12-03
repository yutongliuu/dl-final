#!/usr/bin/env python
"""
从 Python 里直接启动视频版 InstructPix2Pix 的训练，无需手动输入长命令。
可以按需编辑 `TRAINING_ARG_LIST` 或通过环境变量 `VIDEO_TRAIN_PROCESSES`
控制使用的 GPU 数量；脚本会在启动前强制切换到 spawn start method，
避免 CUDA 在 fork 进程中初始化失败。
"""

import multiprocessing as mp
import os

from accelerate import notebook_launcher

from train_video_ip2p import main as train_video_main, parse_args


# 将原来命令行里的参数转成列表，后续可以在这里集中调节
TRAINING_ARG_LIST = [
    "--pretrained_model_name_or_path",
    "models/instruct-pix2pix",
    "--custom_unet_path",
    "models/instruct-pix2pix-video-20frames",
    "--train_data_dir",
    "processed_dataset_seq",
    "--original_image_column",
    "input_frames",
    "--edit_prompt_column",
    "edit_prompt",
    "--edited_image_column",
    "edited_image",
    "--resolution",
    "256",
    "--train_batch_size",
    "2",
    "--gradient_accumulation_steps",
    "4",
    "--max_train_steps",
    "500",
    "--learning_rate",
    "5e-5",
    "--mixed_precision",
    "fp16",
    "--output_dir",
    "output_video_model",
    "--checkpointing_steps",
    "100",
    "--num_train_epochs",
    "100",
]


def _build_args():
    # 复制一份，避免修改全局列表影响后续调用
    return TRAINING_ARG_LIST.copy()


def _training_worker():
    args = parse_args(_build_args())
    train_video_main(args)


def launch_training(num_processes=None):
    """
    通过 accelerate.notebook_launcher 启动多进程训练，行为等价于 accelerate launch。
    `num_processes` 默认读取环境变量 VIDEO_TRAIN_PROCESSES（默认为 4）。
    """
    if num_processes is None:
        num_processes = int(os.environ.get("VIDEO_TRAIN_PROCESSES", "4"))

    # 保证子进程以 spawn 方式启动，避免 CUDA fork 初始化报错
    mp.set_start_method("spawn", force=True)

    notebook_launcher(_training_worker, args=(), num_processes=num_processes)


if __name__ == "__main__":
    launch_training()

