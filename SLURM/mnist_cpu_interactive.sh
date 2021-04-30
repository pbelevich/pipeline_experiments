#!/bin/bash

export USE_TQDM=1

srun --label \
	--job-name=mnist_cpu_interactive \
	--ntasks=8 \
	--partition=q2 \
	--nodes=4 \
	--gpus-per-node=2 \
	--gpus-per-task=1 \
	--time=1:00:00 \
	mnist_cpu.sh
