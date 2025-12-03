import os
import json
from PIL import Image
import numpy as np

# === 配置 ===
MINI_ROOT = "dataset_mini"
VIDEO_NAME = "video_test_01"
NUM_FRAMES = 21  # 00-19做输入，20做输出

def create_dummy_data():
    # 1. 创建目录结构 dataset_mini/train/video_test_01
    video_dir = os.path.join(MINI_ROOT, "train", VIDEO_NAME)
    os.makedirs(video_dir, exist_ok=True)
    print(f"创建目录: {video_dir}")

    # 2. 生成 21 张测试图片 (00.jpg 到 20.jpg)
    # 生成一张纯色图作为测试
    for i in range(NUM_FRAMES):
        file_name = f"{i:02d}.jpg"
        file_path = os.path.join(video_dir, file_name)
        
        # 创建一个 128x128 的随机噪点图，方便肉眼区分
        arr = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        img.save(file_path)
    print(f"已生成 {NUM_FRAMES} 张测试图片")

    # 3. 生成 metadata.json
    metadata = [
        {
            "video_path": f"train/{VIDEO_NAME}",  # 相对路径
            "instruction": "move the object to the right"
        }
    ]
    
    json_path = os.path.join(MINI_ROOT, "metadata.json")
    with open(json_path, "w", encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    print(f"已生成索引文件: {json_path}")

if __name__ == "__main__":
    create_dummy_data()
    print("\n✅ 迷你数据集准备完毕！")
