from datasets import load_from_disk
from PIL import Image

# 1. 加载刚才生成的数据集
dataset = load_from_disk("processed_dataset_seq")

# 2. 取第一条数据
sample = dataset["train"][0]

# 3. 检查 input_frames 的类型
frames = sample["input_frames"]
print(f"Input frames 类型: {type(frames)}")
print(f"列表长度: {len(frames)}")
print(f"列表里第一个元素的类型: {type(frames[0])}")

# 4. 判断是否成功
if isinstance(frames[0], Image.Image):
    print("\n✅ 成功！Hugging Face 已经自动把列表里的路径转成了图片对象。")
else:
    print("\n❌ 失败！列表里仍然是字符串路径，没有自动解码。")