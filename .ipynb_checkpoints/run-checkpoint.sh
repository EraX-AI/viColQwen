#!/bin/bash

# Set environment variables
export ACCELERATE_LOG_LEVEL=info
export CUDA_VISIBLE_DEVICES=0,1,2,3,4  # Adjust based on your GPU setup
export PYTHONPATH="$PYTHONPATH:$(pwd)"

# Configuration parameters
MODEL="Qwen/Qwen2-VL-2B-Instruct"
EMBED_DIM=1024
BATCH_SIZE=32
EVAL_BATCH_SIZE=16
GRAD_ACCUM=4
LR=1e-4
EPOCHS=3
OUTPUT_DIR="./qwen2vl2b_colpali_checkpoints"

# Model max length
MAX_LENGTH=8192

# Dataset parameter (replace with your actual dataset path)
TRAIN_DATASET="/root/test_dataset_TRAIN.jsonl"  # Single pre-prepared dataset
EVAL_DATASET="/root/test_dataset_EVAL.jsonl"  # Single pre-prepared dataset
IMAGE_BASE_PATH="/root/Qwen2VL/OCR/"  # Set this to your image directory

# Evaluation settings
EVAL_STRATEGY="steps"
EVAL_STEPS=100

# Optimization settings
WEIGHT_DECAY=0.001
WARMUP_RATIO=0.05
MAX_GRAD_NORM=10.0  # Increased from default 1.0 to handle high gradients

# Loss settings
TEMPERATURE=0.07
CONTRASTIVE_MARGIN=0.2
ADAPTIVE_ALPHA=0.5

# Include the use_adaptive_loss flag if needed
# USE_ADAPTIVE_LOSS="--use_adaptive_loss"
USE_ADAPTIVE_LOSS=""

# FSDP specific settings
FSDP_TRANSFORMER_LAYER="Qwen2VLDecoderLayer"

# Create output directory
mkdir -p $OUTPUT_DIR
#   --force_ddp \

# Launch distributed training with accelerate
accelerate launch \
  --config_file qwen2VL2B.yaml \
  train.py \
  --model $MODEL \
  --embed_dim $EMBED_DIM \
  --batch_size $BATCH_SIZE \
  --eval_batch_size $EVAL_BATCH_SIZE \
  --grad_accum $GRAD_ACCUM \
  --lr $LR \
  --epochs $EPOCHS \
  --weight_decay $WEIGHT_DECAY \
  --warmup_ratio $WARMUP_RATIO \
  --max_grad_norm $MAX_GRAD_NORM \
  --train_dataset $TRAIN_DATASET \
  --eval_dataset $EVAL_DATASET \
  --image_base_path $IMAGE_BASE_PATH \
  --evaluation_strategy $EVAL_STRATEGY \
  --eval_steps $EVAL_STEPS \
  --temperature $TEMPERATURE \
  --contrastive_margin $CONTRASTIVE_MARGIN \
  --adaptive_alpha $ADAPTIVE_ALPHA \
  $USE_ADAPTIVE_LOSS \
  --use_fsdp \
  --fsdp_transformer_layer_cls_to_wrap $FSDP_TRANSFORMER_LAYER \
  --output_dir $OUTPUT_DIR \
  --max_length $MAX_LENGTH \
  --seed 42