#!/usr/bin/env python
# coding=utf-8
"""Prepare test data for InstructPix2Pix training."""

import json
import os
import shutil
from pathlib import Path

def prepare_test_data():
    """Convert test_video data to format required by training script."""
    base_dir = Path("/kongweiwen/lyt/dl-final/dataset_root")
    test_video_dir = base_dir / "test_video"
    test_data_dir = base_dir / "test_data"
    
    # Create test_data directory
    test_data_dir.mkdir(exist_ok=True)
    
    # Read test_video.json or use default from metadata.json
    test_info = None
    if (test_video_dir / "test_video.json").exists():
        try:
            with open(test_video_dir / "test_video.json", "r") as f:
                content = f.read().strip()
                if content:
                    test_info = json.loads(content)
        except:
            pass
    
    # If test_video.json is empty or doesn't exist, use default from metadata.json
    if test_info is None:
        # Read from metadata.json to find the corresponding entry
        with open(base_dir / "metadata.json", "r") as f:
            metadata = json.load(f)
        # Find entry for video_76 (based on test_video.json structure)
        for entry in metadata:
            if "video_76" in entry.get("video_path", ""):
                test_info = {
                    "instruction": entry["instruction"],
                    "task_type": entry["task_type"]
                }
                break
        
        # If still not found, use a default
        if test_info is None:
            test_info = {
                "instruction": "dropping betel onto the chair",
                "task_type": "drop_object"
            }
    
    # Copy images to test_data directory with unique names
    # For imagefolder format, we need file_name field, but InstructPix2Pix needs multiple images
    # We'll create a structure where each entry has all three images
    shutil.copy(test_video_dir / "00.jpg", test_data_dir / "input_image.jpg")
    shutil.copy(test_video_dir / "01.jpg", test_data_dir / "edited_image.jpg")
    
    # Create metadata.jsonl for HuggingFace imagefolder format
    # The format needs file_name, but we also need input_image, edit_prompt, edited_image
    # We'll use file_name as a reference and include all fields
    entry = {
        "file_name": "input_image.jpg",  # Required by imagefolder format
        "input_image": "input_image.jpg",
        "edit_prompt": test_info["instruction"],
        "edited_image": "edited_image.jpg"
    }
    
    metadata_file = test_data_dir / "metadata.jsonl"
    with open(metadata_file, "w") as f:
        f.write(json.dumps(entry) + "\n")
    
    print(f"Test data prepared in: {test_data_dir}")
    print(f"Metadata file: {metadata_file}")
    print(f"Original image: input_image.jpg")
    print(f"Edited image: edited_image.jpg")
    print(f"Instruction: {test_info['instruction']}")
    print(f"\nYou can now run training with:")
    print(f"  --train_data_dir {test_data_dir}")
    print(f"  --original_image_column input_image")
    print(f"  --edit_prompt_column edit_prompt")
    print(f"  --edited_image_column edited_image")

if __name__ == "__main__":
    prepare_test_data()

