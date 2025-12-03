import torch
import os
from diffusers import UNet2DConditionModel

# ================= 配置区域 =================
# 1. 原始预训练模型路径
# 请修改为你实际存放 instruct-pix2pix 的路径
PRETRAINED_MODEL_PATH = "models/instruct-pix2pix"

# 2. 修改后的模型保存路径
OUTPUT_MODEL_PATH = "models/instruct-pix2pix-video-20frames"

# 3. 维度配置
# 原始模型输入通道 = 8 (4 noisy + 4 conditional)
# 新模型输入通道 = 4 (noisy) + (20 frames * 4 channels) = 84
NEW_IN_CHANNELS = 84 
# ===========================================

def main():
    print(f"正在加载原始模型: {PRETRAINED_MODEL_PATH} ...")
    try:
        unet = UNet2DConditionModel.from_pretrained(
            PRETRAINED_MODEL_PATH, 
            subfolder="unet", 
            torch_dtype=torch.float32
        )
    except OSError:
        print(f"❌ 错误: 无法加载模型，请检查路径 {PRETRAINED_MODEL_PATH}")
        return

    print("原始模型加载成功。")
    print(f"原始输入通道数: {unet.config.in_channels}") # 应该是 8

    # 获取原始的 conv_in 层
    old_conv_in = unet.conv_in
    
    # 创建一个新的 conv_in 层
    # 参数: (新通道数, 输出通道数, 卷积核大小, padding)
    new_conv_in = torch.nn.Conv2d(
        NEW_IN_CHANNELS,
        old_conv_in.out_channels,
        kernel_size=old_conv_in.kernel_size,
        padding=old_conv_in.padding
    )

    print(f"\n开始修改权重 (Shape: {new_conv_in.weight.shape})...")

    # ================= 权重初始化策略 (关键!) =================
    # 策略: 
    # 1. Noisy Latents (前4通道): 完全复制原始权重。
    # 2. Conditional Latents (后80通道): 
    #    我们有 20 帧。
    #    - 第 20 帧 (最新的帧): 复制原始模型的 conditional 权重。
    #      这让模型一开始觉得"我只是在根据最新的一帧做编辑"，像原来的 IP2P 一样。
    #    - 第 1~19 帧 (历史帧): 初始化为 0。
    #      这让模型初始时忽略历史信息，随着训练进行，慢慢学会利用它们。
    
    with torch.no_grad():
        # 1. 复制 Noisy Latents 权重 (前4通道 -> 前4通道)
        new_conv_in.weight[:, :4, :, :] = old_conv_in.weight[:, :4, :, :]
        
        # 2. 复制 Bias (偏置)
        new_conv_in.bias = old_conv_in.bias

        # 3. 处理条件帧权重
        # 原始模型的后4通道是 image latents 权重
        original_cond_weight = old_conv_in.weight[:, 4:, :, :] # shape [320, 4, 3, 3]

        # 初始化后 80 个通道为 0
        new_conv_in.weight[:, 4:, :, :] = 0

        # 将原始条件权重复制到最后 4 个通道 (代表第 20 帧)
        # 索引范围: 从 80 到 84 (也就是最后的 4 个通道)
        new_conv_in.weight[:, -4:, :, :] = original_cond_weight
        
        print("权重初始化完成：")
        print("- Noisy 通道 (0-3): 已复制")
        print("- 历史帧通道 (4-79): 已置零")
        print("- 当前帧通道 (80-83): 已复制原始条件权重")

    # 替换模型中的层
    unet.conv_in = new_conv_in
    
    # 更新配置 (非常重要，否则加载时会报错)
    unet.config.in_channels = NEW_IN_CHANNELS

    # 保存修改后的 UNet
    print(f"\n正在保存修改后的模型到: {OUTPUT_MODEL_PATH} ...")
    unet.save_pretrained(os.path.join(OUTPUT_MODEL_PATH, "unet"))
    
    # 同时也需要把其他组件(VAE, Tokenizer等)复制过去，为了方便直接加载
    # 这里我们只保存了 UNet，训练脚本里可以分别加载
    print("✅ UNet 修改完成！")
    print(f"下一步: 在训练脚本中，将 --pretrained_model_name_or_path 指向原始模型")
    print(f"但需要修改代码让它加载这个新的 UNet。")

if __name__ == "__main__":
    main()