#!/bin/bash
#SBATCH --job-name=pipeline
#SBATCH --output=logs.out.%s.%t
#SBATCH --error=logs.err.%s.%t
#SBATCH --gres gpu:8
#SBATCH --nodes 1
#SBATCH --partition=q3
#SBATCH --ntasks-per-node 1
#SBATCH --time=6:00:00


export MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)

echo $MASTER_ADDR

set -x
#srun python3 -m BERT.run_pipeline --batch_size 64 --nlayers=36 --emsize=2048 --nhid=10240 --nhead=32 --world_size=8 --num_workers=2 --num_batch=30 --epochs=1
#srun -u python3 -m BERT.run_pipeline --batch_size 100 --nlayers=96 --emsize=12288 --nhid=49152 --nhead=16 --world_size=104 --num_workers=13 --num_batch=30 --epochs=1
srun python3 -m BERT.run_pipeline --batch_size 16 --nlayers=5 --emsize=12288 --nhid=49152 --nhead=16 --world_size=8 --num_workers=1 --num_batch=1 --epochs=1
