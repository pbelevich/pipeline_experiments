#!/bin/bash

srun --label \
	--job-name=mnist_cpu_interactive \
	--ntasks=8 \
	--partition=q2 \
	--nodes=4 \
	--gpus-per-node=2 \
	--gpus-per-task=1 \
	--time=10:00 \
	mnist_cpu.sh
