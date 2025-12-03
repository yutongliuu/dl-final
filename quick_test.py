#!/usr/bin/env python
# coding=utf-8
"""Quick test to verify data format and model loading."""

import json
from pathlib import Path
from PIL import Image

def test_data_format():
    """Test if the data format is correct."""
    test_data_dir = Path("/kongweiwen/lyt/dl-final/dataset_root/test_data")
    
    print("Testing data format...")
    print(f"Test data directory: {test_data_dir}")
    
    # Check if files exist
    metadata_file = test_data_dir / "metadata.jsonl"
    input_image = test_data_dir / "input_image.jpg"
    edited_image = test_data_dir / "edited_image.jpg"
    
    print(f"\nChecking files:")
    print(f"  metadata.jsonl: {metadata_file.exists()}")
    print(f"  input_image.jpg: {input_image.exists()}")
    print(f"  edited_image.jpg: {edited_image.exists()}")
    
    if not all([metadata_file.exists(), input_image.exists(), edited_image.exists()]):
        print("ERROR: Some files are missing!")
        return False
    
    # Read metadata
    with open(metadata_file, "r") as f:
        line = f.readline().strip()
        if line:
            metadata = json.loads(line)
            print(f"\nMetadata content:")
            print(f"  input_image: {metadata.get('input_image')}")
            print(f"  edit_prompt: {metadata.get('edit_prompt')}")
            print(f"  edited_image: {metadata.get('edited_image')}")
            
            # Check if required keys exist
            required_keys = ["input_image", "edit_prompt", "edited_image"]
            if all(key in metadata for key in required_keys):
                print("\n✓ All required keys present in metadata")
            else:
                print(f"\n✗ Missing keys: {set(required_keys) - set(metadata.keys())}")
                return False
    
    # Check images
    try:
        img1 = Image.open(input_image)
        img2 = Image.open(edited_image)
        print(f"\nImage info:")
        print(f"  Input image size: {img1.size}, mode: {img1.mode}")
        print(f"  Edited image size: {img2.size}, mode: {img2.mode}")
        print("\n✓ Images can be loaded successfully")
    except Exception as e:
        print(f"\n✗ Error loading images: {e}")
        return False
    
    print("\n" + "="*50)
    print("✓ Data format test PASSED!")
    print("="*50)
    print("\nYou can now run training with:")
    print(f"  bash test_training.sh")
    print("\nOr manually:")
    print(f"  python train_instruct_pix2pix.py \\")
    print(f"    --pretrained_model_name_or_path timbrooks/instruct-pix2pix \\")
    print(f"    --train_data_dir {test_data_dir} \\")
    print(f"    --original_image_column input_image \\")
    print(f"    --edit_prompt_column edit_prompt \\")
    print(f"    --edited_image_column edited_image \\")
    print(f"    --max_train_steps 5 \\")
    print(f"    --output_dir /kongweiwen/lyt/dl-final/test_output")
    
    return True

if __name__ == "__main__":
    test_data_format()

