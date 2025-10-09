#!/usr/bin/env bash

# Prevent tokenizer parallelism issues
export TOKENIZERS_PARALLELISM=false

# Model Configuration
MODEL_ARGS=(
    --model_path "THUDM/CogVideoX-5b-I2V"
    --model_name "cogvideox-i2v"  # ["cogvideox-t2v"]
    --model_type "i2v"
    --training_type "lora"
)

# Output Configuration
OUTPUT_ARGS=(
    --output_dir "output/hug_i2v_sft"
    --report_to "tensorboard"
)

# Data Configuration
DATA_ARGS=(
    --data_root "train_data/hug_debug"
    --caption_column "prompt.txt"
    --video_column "videos.txt"
    --train_resolution "49x480x720"
)

# Training Configuration
TRAIN_ARGS=(
    --train_epochs 1000 # number of training epochs
    --seed 42 # random seed
    #########   Please keep consistent with deepspeed config file ##########
    --batch_size 2
    --gradient_accumulation_steps 1
    --mixed_precision "bf16"  # ["no", "fp16"] Only CogVideoX-2B supports fp16 training
    ########################################################################
)

# System Configuration
SYSTEM_ARGS=(
    --num_workers 4
    --pin_memory True
    --nccl_timeout 1800
)

# Checkpointing Configuration
CHECKPOINT_ARGS=(
    --checkpointing_steps 250 # save checkpoint every x steps
    --checkpointing_limit 2 # maximum number of checkpoints to keep, after which the oldest one is deleted
    --resume_from_checkpoint "ckpt"  # if you want to resume from a checkpoint, otherwise, comment this line
)

# Validation Configuration
VALIDATION_ARGS=(
    --do_validation true
    --validation_dir "val_data/hug_debug"
    --validation_steps 250
    --validation_prompts "prompts.txt"
    --validation_images "images.txt"
    --gen_fps 16
)

# Combine all arguments and launch training
accelerate launch --config_file accelerate_config.yaml train.py \
    "${MODEL_ARGS[@]}" \
    "${OUTPUT_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${TRAIN_ARGS[@]}" \
    "${SYSTEM_ARGS[@]}" \
    "${CHECKPOINT_ARGS[@]}" \
    "${VALIDATION_ARGS[@]}"
