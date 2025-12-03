#!/usr/bin/env python
# coding=utf-8
import argparse
import logging
import math
import os
import shutil
from pathlib import Path
from contextlib import nullcontext

import accelerate
import datasets
import numpy as np
import PIL
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset, load_from_disk
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionInstructPix2PixPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

# 禁用 WandB 如果不需要
if is_wandb_available():
    import wandb

logger = get_logger(__name__, log_level="INFO")

def parse_args():
    parser = argparse.ArgumentParser(description="Script to fine-tune InstructPix2Pix for Video Prediction (20 frames).")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="原始 IP2P 模型路径 (用于加载 VAE, Tokenizer, TextEncoder)",
    )
    parser.add_argument(
        "--custom_unet_path",
        type=str,
        default=None,
        required=True,
        help="【关键】你修改后的 84 通道 UNet 的路径",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help="处理后的 Arrow 数据集路径 (processed_dataset_seq)",
    )
    parser.add_argument(
        "--original_image_column",
        type=str,
        default="input_frames", # 默认改为 input_frames
        help="包含20帧输入的列名",
    )
    parser.add_argument(
        "--edited_image_column",
        type=str,
        default="edited_image",
        help="目标帧列名",
    )
    parser.add_argument(
        "--edit_prompt_column",
        type=str,
        default="edit_prompt",
        help="文本指令列名",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="instruct-pix2pix-video-model",
        help="模型输出目录",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help="训练分辨率",
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help="是否中心裁剪",
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="是否随机水平翻转",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help='The scheduler type to use.',
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="Number of subprocesses to use for data loading.",
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"])
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--checkpoints_total_limit", type=int, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args

def convert_to_np(image, resolution):
    # 如果是 list，说明传进来的是 Sequence，需要递归处理
    if isinstance(image, list):
        return [convert_to_np(img, resolution) for img in image]
        
    if isinstance(image, str):
        image = PIL.Image.open(image)
    image = image.convert("RGB").resize((resolution, resolution))
    return np.array(image).transpose(2, 0, 1)

def main():
    args = parse_args()
    logging_dir = os.path.join(args.output_dir, "logs")
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None

    # Logging setup
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # ================= Load Models =================
    # 1. Load Scheduler, Tokenizer, VAE, TextEncoder from ORIGINAL IP2P
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    
    # 2. Load CUSTOM UNet (84 Channels)
    # 我们不从 original path 加载 UNet，而是从 custom path 加载
    logger.info(f"Loading Custom UNet from: {args.custom_unet_path}")
    unet = UNet2DConditionModel.from_pretrained(args.custom_unet_path, subfolder="unet")

    # 验证通道数
    expected_in_channels = 84
    if unet.config.in_channels != expected_in_channels:
        raise ValueError(f"UNet channel mismatch! Expected {expected_in_channels}, but got {unet.config.in_channels}. Please re-run convert_model.py.")
    
    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    
    # Enable xformers
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available.")
            
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Optimizer
    if args.use_8bit_adam:
        import bitsandbytes as bnb
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # ================= Data Loader =================
    # Load dataset from disk (Arrow format)
    logger.info(f"Loading dataset from {args.train_data_dir}")
    dataset = load_from_disk(args.train_data_dir)

    column_names = dataset["train"].column_names
    input_frames_col = args.original_image_column # "input_frames"
    edited_image_col = args.edited_image_column
    edit_prompt_col = args.edit_prompt_column

    # Tokenize captions
    def tokenize_captions(captions):
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    # Transformations
    train_transforms = transforms.Compose(
        [
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
        ]
    )

    # ================= Preprocessing Logic (Modified) =================
    def preprocess_train(examples):
        # inputs 是 list of lists: [[img0...img19], [img0...img19]]
        inputs = examples[input_frames_col] 
        targets = examples[edited_image_col]
        captions = list(examples[edit_prompt_col])

        all_original_pixel_values = []
        all_edited_pixel_values = []

        # 遍历 Batch 中的每一条数据
        for i in range(len(inputs)):
            input_seq = inputs[i] # 20 images
            target_img = targets[i] # 1 image

            # 1. 转换为 Numpy: (20, 3, H, W) 和 (1, 3, H, W)
            # 注意: convert_to_np 会把 H,W resize 到 args.resolution
            input_np_list = [convert_to_np(img, args.resolution) for img in input_seq]
            target_np = convert_to_np(target_img, args.resolution)
            
            # 2. 拼接所有帧 (20 + 1) 以保证 Transform 一致性
            # shape: (21, 3, H, W)
            combined_np = np.stack(input_np_list + [target_np])
            
            # 3. 转 Tensor 并归一化 [-1, 1]
            combined_tensor = torch.tensor(combined_np).float()
            combined_tensor = 2 * (combined_tensor / 255.0) - 1.0
            
            # 4. Apply Transforms (Crop/Flip)
            # transforms 会把 (Batch, C, H, W) 当作 Batch 处理，对每一张做同样的变换(如果是 RandomCrop，位置可能不同?)
            # 修正: 为了保证时间一致性(crop位置相同)，我们需要先把 time 和 channel 合并? 
            # PyTorch transforms.RandomCrop 对 batch 的处理通常是独立的。
            # 这里的简单做法: 假设已经 resize 到了 resolution，如果不需要 random crop，就这样。
            # 如果需要 Random Crop 且保证位置一致，比较麻烦。此处暂且假设是 Resize + CenterCrop (or Resize only)
            transformed = train_transforms(combined_tensor)

            # 5. 拆分回去
            # input: 前20帧, target: 最后一帧
            original_frames = transformed[:20] # (20, 3, H, W)
            edited_frame = transformed[20]     # (3, H, W)

            all_original_pixel_values.append(original_frames)
            all_edited_pixel_values.append(edited_frame)

        # Stack into batch tensors
        examples["original_pixel_values"] = all_original_pixel_values # List of (20, 3, H, W)
        examples["edited_pixel_values"] = all_edited_pixel_values     # List of (3, H, W)
        examples["input_ids"] = tokenize_captions(captions)
        
        return examples

    with accelerator.main_process_first():
        train_dataset = dataset["train"].with_transform(preprocess_train)

    def collate_fn(examples):
        # Stack inputs: (Batch, 20, 3, H, W)
        original_pixel_values = torch.stack([ex["original_pixel_values"] for ex in examples])
        
        # Stack targets: (Batch, 3, H, W)
        edited_pixel_values = torch.stack([ex["edited_pixel_values"] for ex in examples])
        
        input_ids = torch.stack([ex["input_ids"] for ex in examples])
        
        return {
            "original_pixel_values": original_pixel_values,
            "edited_pixel_values": edited_pixel_values,
            "input_ids": input_ids,
        }

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Prepare logic
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    # ================= Training Loop =================
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Batch size = {args.train_batch_size}")
    logger.info(f"  Total steps = {args.max_train_steps}")

    global_step = 0
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    
    for epoch in range(args.num_train_epochs):
        unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # 1. Encode Targets (Edited Image) -> Latents
                # Batch["edited_pixel_values"]: (B, 3, H, W)
                latents = vae.encode(batch["edited_pixel_values"].to(weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor # (B, 4, h, w)

                # 2. Sample Noise
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()
                
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # 3. Get Text Embeddings
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # 4. Encode Condition Images (20 Frames)
                # Input: (B, 20, 3, H, W)
                original_pixels = batch["original_pixel_values"].to(weight_dtype)
                
                # Reshape to (B*20, 3, H, W) for batch encoding
                b, f, c, h, w = original_pixels.shape
                flattened_pixels = original_pixels.view(b * f, c, h, w)
                
                # VAE Encode (use mode for condition)
                # 使用 no_grad 节省显存，Condition 不需要梯度传回 VAE
                with torch.no_grad():
                    original_image_embeds = vae.encode(flattened_pixels).latent_dist.mode()
                
                # Reshape back: (B*20, 4, h_latent, w_latent) -> (B, 20*4, h_latent, w_latent)
                # VAE downsamples by 8 usually
                _, c_latent, h_latent, w_latent = original_image_embeds.shape
                original_image_embeds = original_image_embeds.view(b, f * c_latent, h_latent, w_latent)
                
                # Now original_image_embeds is (B, 80, h, w)

                # 5. Concatenate
                # noisy_latents: (B, 4, h, w)
                # original_image_embeds: (B, 80, h, w)
                # result: (B, 84, h, w)
                concatenated_noisy_latents = torch.cat([noisy_latents, original_image_embeds], dim=1)

                # 6. Predict
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                
                model_pred = unet(concatenated_noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": loss.detach().item()}, step=global_step)
                
                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved checkpoint to {save_path}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break
    
    # Save final model
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            text_encoder=text_encoder,
            vae=vae,
            unet=accelerator.unwrap_model(unet),
            tokenizer=tokenizer,
        )
        pipeline.save_pretrained(args.output_dir)
        logger.info(f"Training finished. Model saved to {args.output_dir}")

    accelerator.end_training()

if __name__ == "__main__":
    main()