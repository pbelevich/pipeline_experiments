#!/bin/bash

#SBATCH --job-name=mnist1_cpu_sbatch

#SBATCH --partition=q2

#SBATCH --nodes=4

#SBATCH --gpus-per-node=2

#SBATCH --gpus-per-task=1

#SBATCH --time=10:00

# export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
# export MASTER_PORT=29500
srun --label mnist1_cpu.sh

