#!/bin/bash

srun --label \
	--job-name=mnist1_cpu_interactive \
	--ntasks=8 \
	--partition=q2 \
	--nodes=4 \
	--gpus-per-node=2 \
	--gpus-per-task=1 \
	--time=01:00:00 \
	resnet_cpu.sh

