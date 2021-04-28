#!/bin/bash

export MASTER_PORT=29500
export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
export CUDA_VISIBLE_DEVICES=${SLURM_LOCALID}

# echo "SLURM_NODELIST = "${SLURM_NODELIST}
# echo "SLURM_NTASKS = "${SLURM_NTASKS}
# echo "SLURM_PROCID = "${SLURM_PROCID}
# echo "MASTER_ADDR = "${MASTER_ADDR}
# echo "MASTER_PORT = "${MASTER_PORT}
# echo "CUDA_VISIBLE_DEVICES = "${CUDA_VISIBLE_DEVICES}

python -u mnist1_cpu.py \
        --world_size=${SLURM_NTASKS} \
        --rank=${SLURM_PROCID} \
        --master_addr=${MASTER_ADDR} \
        --master_port=${MASTER_PORT}

