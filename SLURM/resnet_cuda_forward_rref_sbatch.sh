#!/bin/bash

#SBATCH --job-name=resnet_cuda_forward_rref_sbatch

#SBATCH --partition=q2

#SBATCH --nodes=4

#SBATCH --gpus-per-node=2

#SBATCH --gpus-per-task=1

#SBATCH --time=1:00:00

srun --label resnet_cuda_forward_rref.sh
