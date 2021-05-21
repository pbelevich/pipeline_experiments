#!/bin/bash

chunks=128
ws=104
nodes=$(((ws+7)/8))
bs=128
#model=BERT1.2
model=GPT175
#model=GPT88
#model=BERT88
#model=test

model_params="--batch_size $bs --num_chunks=$chunks --world_size=$ws --num_workers=$nodes  --num_batch=20 --epochs=1"

sbatch <<EOT
#!/bin/bash

#SBATCH --job-name=pipeline-$model-bs-$bs-ws-$ws-chunks-$chunks
#SBATCH --output=logs/logs.%x.%t.out
#SBATCH --error=logs/logs.%x.%t.err
#SBATCH --gres gpu:8
#SBATCH --nodes $nodes
#SBATCH --partition=q3
#SBATCH --ntasks-per-node 1
#SBATCH --time=2:00:00


export MASTER_ADDR=\$(srun --ntasks=1 hostname 2>&1 | tail -n1)

echo \$MASTER_ADDR

set -x

#GPT175
srun -u python3 -m BERT.run_pipeline $model_params --nlayers=96 --emsize=12288 --nhid=49152 --nhead=16 --ep_embedding --ep_head --ep_noop

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
