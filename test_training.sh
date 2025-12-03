#!/bin/bash
# Test training script to verify the model can train successfully

# Activate conda environment
source /kongweiwen/kongweiwen/miniconda3/etc/profile.d/conda.sh
conda activate dl-final

# Fix protobuf compatibility issue
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
# Also downgrade protobuf if needed
pip install "protobuf<=3.20.3" --quiet 2>/dev/null || true

# Use HuggingFace mirror site for faster downloads
export HF_ENDPOINT=https://hf-mirror.com

# Set variables
# Use local model if available, otherwise use HuggingFace Hub
if [ -d "/kongweiwen/lyt/dl-final/models/instruct-pix2pix" ]; then
    PRETRAINED_MODEL="/kongweiwen/lyt/dl-final/models/instruct-pix2pix"
    echo "Using local model: $PRETRAINED_MODEL"
else
    PRETRAINED_MODEL="timbrooks/instruct-pix2pix"
    echo "Using HuggingFace Hub model: $PRETRAINED_MODEL"
fi
TRAIN_DATA_DIR="/kongweiwen/lyt/dl-final/dataset_root/test_data"
OUTPUT_DIR="/kongweiwen/lyt/dl-final/test_output"
MAX_TRAIN_STEPS=5  # Very few steps for quick test
RESOLUTION=256
BATCH_SIZE=1

echo "Starting test training..."
echo "Pretrained model: $PRETRAINED_MODEL"
echo "Train data dir: $TRAIN_DATA_DIR"
echo "Output dir: $OUTPUT_DIR"
echo "Max train steps: $MAX_TRAIN_STEPS"

python train_instruct_pix2pix.py \
    --pretrained_model_name_or_path="$PRETRAINED_MODEL" \
    --train_data_dir="$TRAIN_DATA_DIR" \
    --original_image_column="input_image" \
    --edit_prompt_column="edit_prompt" \
    --edited_image_column="edited_image" \
    --resolution=$RESOLUTION \
    --random_flip \
    --train_batch_size=$BATCH_SIZE \
    --max_train_steps=$MAX_TRAIN_STEPS \
    --checkpointing_steps=$MAX_TRAIN_STEPS \
    --learning_rate=1e-4 \
    --max_grad_norm=1.0 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --output_dir="$OUTPUT_DIR" \
    --mixed_precision="fp16" \
    --report_to="tensorboard" \
    --seed=42

echo "Test training completed!"
echo "Check output directory: $OUTPUT_DIR"

