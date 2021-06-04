#!/bin/bash

chunks=2
ws=4
nodes=$(((ws+3)/4))
##nodes=$(((ws+3)/4))
bs=128
#model=BERT1.2
model=GPT-TEST
#model=GPT175
#model=GPT88
#model=BERT88
#model=test

model_params="--batch_size $bs --num_chunks=$chunks --world_size=$ws --num_workers=$nodes  --num_batch=20 --epochs=1"

echo srun -u python3 -m BERT.run_pipeline $model_params --nlayers=96 --emsize=12288 --nhid=49152 --nhead=16 --ep_embedding --ep_head --ep_noop
sbatch <<EOT
#!/bin/bash

#SBATCH --job-name=pipeline-$model-bs-$bs-ws-$ws-chunks-$chunks
#SBATCH --output=logs/0/logs.%x.%t.out
#SBATCH --error=logs/0/logs.%x.%t.err
#SBATCH --gres gpu:4
#SBATCH --nodes $nodes
#SBATCH --partition=dev
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 32
#SBATCH --time=1:00:00


export MASTER_ADDR=\$(srun --ntasks=1 hostname 2>&1 | tail -n1)

echo \$MASTER_ADDR

set -x

#GPT1NODE
srun -u python3 -m BERT.run_pipeline $model_params --nlayers=1 --emsize=12288 --nhid=49152 --nhead=16 --ep_embedding --ep_head --ep_noop

#GPT175
#echo srun -u python3 -m BERT.run_pipeline $model_params --nlayers=96 --emsize=12288 --nhid=49152 --nhead=16 --ep_embedding --ep_head --ep_noop

#GPT88
#srun -u python3 -m BERT.run_pipeline $model_params --nlayers=61 --emsize=12288 --nhid=49152 --nhead=16 --ep_embedding --ep_head --ep_noop

#BERT 1.2B:
#srun -u python3 -m BERT.run_pipeline $model_params --nlayers=24 --emsize=2048 --nhid=8192 --nhead=16

#BERT 88B:
#srun -u python3 -m BERT.run_pipeline $model_params --nlayers=1750 --emsize=2048 --nhid=8192 --nhead=16 --ep_head --ep_noop

#TEST
#srun -u python3 -m BERT.run_pipeline $model_params --nlayers=4 --emsize=256 --nhid=1024 --nhead=16

EOT

squeue
