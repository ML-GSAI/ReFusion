#!/bin/bash

# Default arguments
NNODES=1
NODE_RANK=0
MASTER_ADDR="127.0.0.1"

# Parse arguments: -n (nodes), -r (rank), -m (master addr)
while getopts "n:r:m:" opt; do
  case ${opt} in
    n ) NNODES=$OPTARG ;;
    r ) NODE_RANK=$OPTARG ;;
    m ) MASTER_ADDR=$OPTARG ;;
    \? ) echo "Invalid option: -$OPTARG" >&2; exit 1 ;;
    : ) echo "Option -$OPTARG requires an argument." >&2; exit 1 ;;
  esac
done

# Print configuration
echo "====================== TRAINING CONFIG ======================"
echo "Start Time: $(date)"
echo "NNODES: ${NNODES}"
echo "NODE_RANK: ${NODE_RANK}"
echo "MASTER_ADDR: ${MASTER_ADDR}"
echo "============================================================="


datasets=(
    "training_data_path_1.json"
    "training_data_path_2.json"
    "training_data_path_3.json"
)
datasets_str=$(IFS=,; echo "${datasets[*]}")

CUDA_VISIBLE_DEVICES=7,6,5,4,3,2,1,0 torchrun \
        --nproc_per_node=8 \
        --nnodes=${NNODES} \
        --node_rank=${NODE_RANK} \
        --master_addr=${MASTER_ADDR} \
        --master_port=20213 \
        train.py  \
        --model_name_or_path Qwen/Qwen3-8B \
        --bf16 True \
        --output_dir ./output_checkpoints \
        --model_max_length 4096 \
        --use_flash_attn True \
        --data_path "$datasets_str" \
        --low_rank_training False \
        --num_train_epochs 4  \
        --per_device_train_batch_size 4     \
        --gradient_accumulation_steps 1    \
        --save_strategy "epoch"     \
        --save_total_limit 5     \
        --learning_rate 2e-5     \
        --weight_decay 0.0     \
        --warmup_ratio 0.0     \
        --lr_scheduler_type "cosine"     \
        --logging_steps 10    \
        --deepspeed ds_configs/stage2.json \
        --report_to "tensorboard" \
        --tf32 True

# Usage example:
# ./train.sh -n 2 -r 0 -m 192.168.1.1