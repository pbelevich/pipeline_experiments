#!/bin/bash

export USE_TQDM=1

srun --label \
	--job-name=bert_cuda_forward_tensor_interactive \
	--ntasks=52 \
	--partition=q2 \
	--nodes=13 \
	--gpus-per-node=4 \
	--gpus-per-task=1 \
	--time=12:00:00 \
	bert_cuda_forward_tensor.sh
