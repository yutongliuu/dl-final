import json
import os
import shutil
from datasets import Dataset, Features, Image, Value, Sequence

# ================= 配置区域 =================
# 获取脚本所在目录，确保路径解析正确
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. 数据根目录 
# (测试时建议指向 dataset_mini，正式训练时改为 dataset_root)
DATA_ROOT = os.path.join(BASE_DIR, "dataset_mini")

# 2. 索引文件路径
METADATA_PATH = os.path.join(DATA_ROOT, "metadata.json")

# 3. 输出处理后的数据集路径 (Arrow 格式)
OUTPUT_DIR = os.path.join(BASE_DIR, "processed_dataset_seq")

# 4. 帧数配置
NUM_INPUT_FRAMES = 20          # 输入帧数 (00.jpg ~ 19.jpg)
TARGET_FRAME_INDEX = 20        # 目标帧索引 (20.jpg)
# ===========================================

def generate_examples():
    """
    生成器函数：读取数据并组装成 HuggingFace Dataset 需要的格式。
    """
    # 1. 读取 JSON 索引
    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(f"找不到索引文件: {METADATA_PATH}")
    
    with open(METADATA_PATH, 'r', encoding='utf-8') as f:
        metadata_list = json.load(f)

    print(f"正在处理 {len(metadata_list)} 条视频数据...")
    
    # 2. 遍历每一条数据
    for idx, item in enumerate(metadata_list):
        # 获取视频文件夹路径
        # 兼容 dataset_mini 生成的 key ("video_path") 和原始数据可能的 key
        video_rel_path = item.get("video_path", item.get("input_frames_folder", ""))
        video_folder = os.path.join(DATA_ROOT, video_rel_path)
        
        # 获取文本指令
        instruction = item.get("instruction", item.get("edit_prompt", ""))

        # -------------------------------------------------
        # 关键逻辑：循环生成 20 帧输入的路径列表
        # -------------------------------------------------
        input_frames_paths = []
        missing_frame = False

        # 收集 00.jpg 到 19.jpg
        for i in range(NUM_INPUT_FRAMES):
            frame_name = f"{i:02d}.jpg" 
            frame_path = os.path.join(video_folder, frame_name)
            
            if not os.path.exists(frame_path):
                # print(f"缺失帧: {frame_path}") # 调试用
                missing_frame = True
                break
            input_frames_paths.append(frame_path)

        # 获取目标帧 (20.jpg)
        target_frame_name = f"{TARGET_FRAME_INDEX:02d}.jpg"
        target_img_path = os.path.join(video_folder, target_frame_name)

        # 检查完整性
        if missing_frame or not os.path.exists(target_img_path):
            print(f"[警告] 跳过不完整的样本: {video_folder}")
            continue

        # 3. yield 数据字典
        # "input_frames" 是一个包含 20 个路径字符串的 List
        # HF Dataset 会根据 Feature 定义自动把这些路径加载为图片对象
        yield {
            "input_frames": input_frames_paths,  
            "edit_prompt": instruction,
            "edited_image": target_img_path,     
        }


def _patch_sequence_info_file(info_path, column_name, expected_length, expected_feature_spec):
    if not os.path.exists(info_path):
        return False

    with open(info_path, "r", encoding="utf-8") as f:
        info = json.load(f)

    features = info.get("features", {})
    column = features.get(column_name)
    if not isinstance(column, dict):
        return False

    updated = False

    if column.get("_type") != "Sequence":
        column["_type"] = "Sequence"
        updated = True

    if expected_length is not None and column.get("length") != expected_length:
        column["length"] = expected_length
        updated = True

    feature_spec = column.get("feature")
    if not isinstance(feature_spec, dict):
        column["feature"] = dict(expected_feature_spec or {})
        updated = True
    else:
        for key, value in (expected_feature_spec or {}).items():
            if feature_spec.get(key) != value:
                feature_spec[key] = value
                updated = True

    if updated:
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
        print(f"已修复 {info_path} 中 `{column_name}` 的元信息。")

    return updated


def ensure_sequence_metadata(
    dataset_dir,
    column_name="input_frames",
    expected_length=None,
    expected_feature_spec=None,
):
    dataset_dict_path = os.path.join(dataset_dir, "dataset_dict.json")
    splits = []

    if os.path.exists(dataset_dict_path):
        with open(dataset_dict_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                splits = data.get("splits", [])
            except json.JSONDecodeError:
                splits = []

    target_dirs = []
    if splits:
        for split in splits:
            split_dir = os.path.join(dataset_dir, split)
            if os.path.isdir(split_dir):
                target_dirs.append(split_dir)
    else:
        target_dirs.append(dataset_dir)

    patched_any = False
    for split_dir in target_dirs:
        info_path = os.path.join(split_dir, "dataset_info.json")
        if _patch_sequence_info_file(info_path, column_name, expected_length, expected_feature_spec or {}):
            patched_any = True

    if patched_any:
        print("✅ 已确保 Sequence 元信息与训练脚本预期一致。")

def main():
    # 0. 清理旧数据 (防止缓存导致 Metadata 不更新)
    if os.path.exists(OUTPUT_DIR):
        print(f"发现旧数据目录 {OUTPUT_DIR}，正在删除以确保生成新结构...")
        shutil.rmtree(OUTPUT_DIR)

    # 1. 定义数据结构 (Schema)
    # 存储输入帧的文件路径字符串，避免 pyarrow 对 Sequence(Image) 的限制
    input_sequence_feature = Value("string")
    features = Features(
        {
            "input_frames": Sequence(feature=input_sequence_feature, length=NUM_INPUT_FRAMES),
            "edit_prompt": Value("string"),
            "edited_image": Image(),
        }
    )

    # 2. 从生成器创建数据集
    print("开始转换数据...")
    try:
        dataset = Dataset.from_generator(generate_examples, features=features)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"生成数据集失败: {e}")
        return

    # 3. 划分训练集和验证集
    # 如果数据量太少(比如只有1条)，test_size=0.1 会导致报错，这里做个保护
    if len(dataset) > 1:
        print("划分 10% 验证集...")
        dataset = dataset.train_test_split(test_size=0.1)
    else:
        print("数据量过少，不划分验证集，全部用于训练...")
        # 为了保持 DatasetDict 结构一致，手动构造 dict
        from datasets import DatasetDict
        dataset = DatasetDict({"train": dataset, "test": dataset})

    # 4. 保存到硬盘
    dataset.save_to_disk(OUTPUT_DIR)
    ensure_sequence_metadata(
        OUTPUT_DIR,
        column_name="input_frames",
        expected_length=NUM_INPUT_FRAMES,
        expected_feature_spec={"_type": "Value", "dtype": "string"},
    )
    
    print(f"\n✅ 数据集构建成功！")
    print(f"输出目录: {os.path.abspath(OUTPUT_DIR)}")
    print("现在你可以去检查 dataset_info.json 里的 features 是否包含 Sequence 了。")

if __name__ == "__main__":
    main()