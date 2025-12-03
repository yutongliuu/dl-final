import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class VideoFrameDataset(Dataset):
    def __init__(self, use_mock=True, resolution=128, length=100):
        """
        Args:
            use_mock (bool): 如果为 True，生成随机噪声数据用于测试环境。
            resolution (int): 图片分辨率，作业要求至少 96x96 。
        """
        self.use_mock = use_mock
        self.resolution = resolution
        self.length = length
        
        # 预定义的作业任务指令 [cite: 30, 35]
        self.prompts = [
            "Moving something from left to right",
            "Dropping something onto something",
            "Covering something with something"
        ]

    def __len__(self):
        # 如果是 Mock 模式，返回预设长度；否则返回真实数据长度
        return self.length if self.use_mock else len(self.real_data_list)

    def __getitem__(self, idx):
        if self.use_mock:
            # === Mock Data 分支 (无需真实数据即可运行) ===
            # 模拟第 20 帧 (输入条件图)
            input_image = torch.randn(3, self.resolution, self.resolution)
            # 模拟第 21 帧 (Ground Truth 目标图)
            target_image = torch.randn(3, self.resolution, self.resolution)
            # 模拟文本指令
            text_prompt = self.prompts[idx % len(self.prompts)]
            
            return {
                "pixel_values": target_image,      # 目标图 (GT)
                "condition_pixel_values": input_image, # 条件图 (Input)
                "input_ids": text_prompt           # 文本 (稍后在训练循环中tokenize)
            }
        else:
            # === 真实数据分支 (留给队友填空) ===
            # TODO: 读取真实的 .jpg 文件并转换为 Tensor
            pass