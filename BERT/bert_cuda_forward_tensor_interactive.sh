#!/bin/bash

export USE_TQDM=1

srun --label \
	--job-name=bert_cuda_forward_tensor_interactive \
	--ntasks=104 \
	--partition=q3 \
	--nodes=13 \
	--gpus-per-node=8 \
	--gpus-per-task=1 \
	--time=24:00:00 \
	bert_cuda_forward_tensor.sh
