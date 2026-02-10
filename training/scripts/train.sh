#!/bin/bash

# GRPO Training Script for AutoWebWorld
# This script trains a vision-language model using GRPO (Generalized Reward-based Policy Optimization)

export DEBUG_MODE="true"

# Paths configuration
export DATA_PATH=data/train_data                    # Path to training data directory
export CKPT_PATH=models/Qwen2.5-VL-3B              # Path to base model checkpoint
export SAVE_PATH=outputs/grpo_model                # Path to save trained model
export LOG_PATH=${SAVE_PATH}"/debug_log.txt"
export Train_PATH=${SAVE_PATH}"/train.log"
export FAILED_LOG_PATH=${SAVE_PATH}"/failed_cases.txt"
export FAILED_IMG_DIR=${SAVE_PATH}"/failed_cases_images"
mkdir -p $SAVE_PATH

# Training with DeepSpeed Zero-3
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12346" \
    -m training.grpo_train \
    --output_dir ${SAVE_PATH} \
    --model_name_or_path ${CKPT_PATH} \
    --data_file_paths ${DATA_PATH}/train.json \
    --image_folders ${DATA_PATH}/train_imgs \
    --dataset_name ${DATA_PATH} \
    --deepspeed training/configs/zero3.json \
    --max_prompt_length 1024 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --logging_steps 1 \
    --bf16 \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --max_pixels 12845056 \
    --num_train_epochs 1 \
    --run_name GRPO_AutoWebWorld_train \
    --save_strategy steps \
    --save_steps 100 \
    --save_only_model true \
    --num_generations 8 \
    >> $Train_PATH 2>&1
