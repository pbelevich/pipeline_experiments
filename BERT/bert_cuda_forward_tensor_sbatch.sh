#!/bin/bash

#SBATCH --job-name=bert_cuda_forward_tensor_sbatch

#SBATCH  --partition=q3

#SBATCH --ntasks=104

#SBATCH --nodes=13

#SBATCH --gpus-per-node=8

#SBATCH --gpus-per-task=1

#SBATCH --time=24:00:00

srun --label bert_cuda_forward_tensor.sh
