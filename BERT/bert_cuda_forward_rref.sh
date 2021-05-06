#!/bin/bash

export MASTER_PORT=29500
export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
export CUDA_VISIBLE_DEVICES=${SLURM_LOCALID}
export UNUSED_GPUS=1

python -u bert_cuda_forward_rref.py \
        --world_size=$((SLURM_NTASKS-UNUSED_GPUS)) \
        --rank=${SLURM_PROCID} \
        --master_addr=${MASTER_ADDR} \
        --master_port=${MASTER_PORT}
