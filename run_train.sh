export MODEL_NAME="timbrooks/instruct-pix2pix"
export DATASET_DIR="processed_dataset"  # 刚才生成的文件夹

accelerate launch --mixed_precision="fp16" train_instruct_pix2pix.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_DIR \
  --use_ema \
  --enable_xformers_memory_efficient_attention \
  --resolution=128 \
  --random_flip \
  --train_batch_size=4 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=15000 \
  --checkpointing_steps=5000 \
  --learning_rate=1e-5 \
  --lr_warmup_steps=0 \
  --mixed_precision=fp16 \
  --val_image_url="dataset_root/train/move_object/video_001/20.jpg" \
  --validation_prompt="Moving something up" \
  --output_dir="experiment_output"