local machines = {
  local template = {
  },

  dev: template {
  	queue: 'dev',
    gpu_per_task: 4,
    cpu_per_task: 32,
  },

  train: template {
  	queue: 'train',
    gpu_per_task: 8,
    cpu_per_task: 96,
  },
};


local models = {
  local template = {
    local model = self,
    chunks: std.floor(self.ws / 2),
	shards: 1,
    ws: error 'not provided',
	ws_perm: [i for i in std.range(0, self.ws-1)],
    bs: 128,
    name: error 'not provided',
    params: {
      batch_size: model.bs,
      num_chunks: model.chunks,
	  nshards: model.shards
    },
  },

  gpt175: template {
    local model = self,
    ws: model.layers + 3,
    layers: 20, //96,
	shards: 1,
    name: 'GPT175',
    params+: {
      nlayers: model.layers,
      emsize: 12288,
      nhid: 49152,
      nhead: 16,
      ep_embedding: true,
      ep_head: true,
      ep_noop: true,
    },
  },

  gpt4gpu: $.gpt175 {
    layers: 4,
	chunks: 4,
	shards: 4,
	ws: 8,
	ws_perm: [0,4,5,6,7,1,2,3],
	bs: 64,
    params+: {
      emsize: 12288 / 2,
      nhid: 49152 / 2,
      nhead: 16,
	 },
     name: 'GPT-TEST',
  },


  sharded_gpt4gpu: $.gpt175 {
    layers: 1,
	chunks: 2,
	shards: 2,
	ws: 4,
	bs: 16,
    params+: {
      emsize: 12288 / 2,
      nhid: 49152 / 2,
      nhead: 16,
	  ep_noop: null,
	 },
     name: 'GPT-TEST',
  },
};

local config_tmpl = {
  local config = self,
  local params = $.model.params {
    num_workers: std.ceil(config.model.ws / config.machine_gpus),
	ws_perm: std.join(',', [''+p for p in config.model.ws_perm]),
    world_size: params.num_workers * config.machine_gpus,
    num_batch: 20,
    epochs: 1,
  },
  command: std.join(' ', [{
    local v = params[x],
    result: if v == null then '' else if v == true then '--' + x else '--' + x + '=' + v,
  }.result for x in std.objectFields(params)]),
  machine_gpus: $.machine.gpu_per_task,
  machine_cpus: $.machine.cpu_per_task,
  machine_num: params.num_workers,
  machine_queue: $.machine.queue,
  job_name: 'pipeline-v2-%s-%s-bs-%s-ws-%s-chunks-%s-shards-%s' % [config.model.name, $.machine.queue, $.model.bs, params.world_size, $.model.chunks, $.model.shards],
};


local config = config_tmpl {
  machine: machines.dev,
  model: models.gpt4gpu,
};

|||
  #!/bin/bash

  #SBATCH --job-name=%(job_name)s
  #SBATCH --output=logs/0/logs.%%x.%%t.out
  #SBATCH --error=logs/0/logs.%%x.%%t.err
  #SBATCH --gres gpu:%(machine_gpus)s
  #SBATCH --nodes %(machine_num)s
  #SBATCH --partition=%(machine_queue)s
  #SBATCH --ntasks-per-node 1
  #SBATCH --cpus-per-task %(machine_cpus)s
  #SBATCH --time=0:15:00


  export MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)

  echo $MASTER_ADDR

  set -x

  srun -u python3 -m BERT.run_pipeline  %(command)s

||| % config
