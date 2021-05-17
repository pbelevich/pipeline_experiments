#!/bin/bash

chunks=8
ws=8
nodes=$(((ws+7)/8))
bs=256
model=BERT1.2
#model=GPT175

model_params="--batch_size $bs --num_chunks=$chunks --world_size=$ws --num_workers=$nodes  --num_batch=30 --epochs=1"

sbatch <<EOT
#!/bin/bash

#SBATCH --job-name=pipeline-$model-bs-$bs-ws-$ws-chunks-$chunks
#SBATCH --output=logs/logs.%x.%t.out
#SBATCH --error=logs/logs.%x.%t.err
#SBATCH --gres gpu:8
#SBATCH --nodes $nodes
#SBATCH --partition=q3
#SBATCH --ntasks-per-node 1
#SBATCH --time=3:00:00


export MASTER_ADDR=\$(srun --ntasks=1 hostname 2>&1 | tail -n1)

echo \$MASTER_ADDR

set -x
#srun -u python3 -m BERT.run_pipeline $model_params --nlayers=36 --emsize=2048 --nhid=10240 --nhead=32
#srun -u python3 -m BERT.run_pipeline $model_params --nlayers=96 --emsize=12288 --nhid=49152 --nhead=16 --ep_embedding --ep_head --ep_noop
#srun -u python3 -m BERT.run_pipeline $model_params  --nlayers=5 --emsize=12288 --nhid=49152 --nhead=16

#BERT 1.2B:
srun -u python3 -m BERT.run_pipeline $model_params --nlayers=24 --emsize=2048 --nhid=8192 --nhead=16

EOT

squeue
