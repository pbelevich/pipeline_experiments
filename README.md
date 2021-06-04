# pipeline_experiments
PyTorch pipeline parallelism experiments
## Local pipeline

	To run one-machine with 8 gpus locally:
	```
	python bert_local_pipeline.py --nlayers=24 --emsize=2048 --nhid=8192 --nhead=16 --log-interval=1 --batch_size=32 --gpus=8
	```

## CPU RPC forward(rref)

	To run one-machine with 8 gpus locally:
	```
	python bert_cpu.py --nlayers=24 --emsize=2048 --nhid=8192 --nhead=16 --log-interval=1 --batch_size=32 --world_size=9
	```

	To run multi-machine SLURM job in interactive mode edit bert_cpu_forward_rref.sh and bert_cpu_forward_rref_interactive.sh and run:
	```
	./bert_cpu_forward_rref_interactive.sh
	```

	To run multi-machine SLURM job in batch mode edit bert_cpu_forward_rref.sh and bert_cpu_forward_rref_sbatch.sh and run:
	```
	sbatch bert_cpu_forward_rref_sbatch.sh
	```

## CUDA RPC forward(rref)

	To run one-machine with 4 gpus locally:
	```
	python bert_cuda_forward_rref.py --nlayers=24 --emsize=2048 --nhid=8192 --nhead=16 --log-interval=1 --batch_size=32 --world_size=5
	```

	To run multi-machine SLURM job in interactive mode edit bert_cuda_forward_rref.sh and bert_cuda_forward_rref_interactive.sh and run:
	```
	./bert_cuda_forward_rref_interactive.sh
	```

	To run multi-machine SLURM job in batch mode edit bert_cuda_forward_rref.sh and bert_cuda_forward_rref_sbatch.sh and run:
	```
	sbatch bert_cuda_forward_rref_sbatch.sh
	```

## CUDA RPC forward(tensor)

	To run one-machine with 2 gpus locally:
	```
	python bert_cuda_forward_tensor.py --nlayers=24 --emsize=2048 --nhid=8192 --nhead=16 --log-interval=1 --batch_size=32 --world_size=3
	```

	To run multi-machine SLURM job in interactive mode edit bert_cuda_forward_tensor.sh and bert_cuda_forward_tensor_interactive.sh and run:
	```
	./bert_cuda_forward_tensor_interactive.sh
	```

	To run multi-machine SLURM job in batch mode edit bert_cuda_forward_tensor.sh and bert_cuda_forward_tensor_sbatch.sh and run:
	```
	sbatch bert_cuda_forward_tensor_sbatch.sh
	```
