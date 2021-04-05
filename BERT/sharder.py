import torch.nn as nn
from model import MLMTaskEmbedding, MLMTaskEncoder, MLMTaskHead, MLMTask2
from torch.distributed.rpc import RRef

class LocalPipelineImpl(nn.Module):
    def __init__(self, seq):
        super().__init__()
        self.seq = seq

    def forward(self, x):
        for gpu, module in enumerate(self.seq):
            device = next(module.parameters()).device
            x = x.to(device)
            x = module(x)
        return x.cpu()

    def parameter_rrefs(self):
        return [RRef(p) for p in self.seq.parameters()]


def MLMTaskSharder(devices, config, ntoken, ninp, nhead, nhid, dropout):
    gpus = len(devices)
    add_embedding = config.get('include_embeddings', False)
    n_encoders_on_first_gpu = config.get('n_encoders_on_first_gpu', 0)
    add_head = config.get('include_head', False)
    n_encoders_on_last_gpu = config.get('n_encoders_on_last_gpu', 0)
    n_encoders = config.get('n_encoders', 0)

    assert(n_encoders_on_first_gpu >= 0)
    assert(n_encoders_on_last_gpu >= 0)
    layers = []
    if gpus == 1:
        sublayers = []
        if add_embedding:
            sublayers.append(MLMTaskEmbedding(ntoken, ninp))
        if n_encoders > 0:
            sublayers.append(MLMTaskEncoder(ninp, nhead, nhid, n_encoders, dropout))
        if add_head:
            sublayers.append(MLMTaskHead(ntoken, ninp))
        if sublayers:
            layers.append(nn.Sequential(*sublayers).to(devices[0]))
    elif gpus == 2:
        if n_encoders_on_first_gpu == 0 and n_encoders_on_last_gpu == 0:
            assert(n_encoders % 2 == 0)
            n_encoders_on_first_gpu = n_encoders // 2
            n_encoders_on_last_gpu = n_encoders // 2
        elif n_encoders_on_first_gpu == 0:
            n_encoders_on_first_gpu = n_encoders - n_encoders_on_last_gpu
        elif n_encoders_on_last_gpu == 0:
            n_encoders_on_last_gpu = n_encoders - n_encoders_on_first_gpu
        else:
            assert(n_encoders == n_encoders_on_first_gpu + n_encoders_on_last_gpu)

        sublayers = []
        if add_embedding:
            sublayers.append(MLMTaskEmbedding(ntoken, ninp))
        if n_encoders_on_first_gpu > 0:
            sublayers.append(MLMTaskEncoder(ninp, nhead, nhid, n_encoders_on_first_gpu, dropout))
        if sublayers:
            layers.append(nn.Sequential(*sublayers).to(devices[0]))

        sublayers = []
        if n_encoders_on_last_gpu > 0:
            sublayers.append(MLMTaskEncoder(ninp, nhead, nhid, n_encoders_on_last_gpu, dropout))
        if add_head:
            sublayers.append(MLMTaskHead(ntoken, ninp))
        if sublayers:
            layers.append(nn.Sequential(*sublayers).to(devices[1]))
        
    else:
        n_encoders_on_middle_gpus = n_encoders - n_encoders_on_first_gpu - n_encoders_on_last_gpu
        middle_gpus = gpus - int(add_embedding) - int(add_head)
        assert(n_encoders_on_middle_gpus % middle_gpus == 0)
        n_encoders_per_middle_gpu = n_encoders_on_middle_gpus // middle_gpus

        sublayers = []
        if add_embedding:
            sublayers.append(MLMTaskEmbedding(ntoken, ninp))
        if n_encoders_on_first_gpu > 0:
            sublayers.append(MLMTaskEncoder(ninp, nhead, nhid, n_encoders_on_first_gpu, dropout))
        if sublayers:
            layers.append(nn.Sequential(*sublayers).to(devices[0]))

        for gpu in range(int(add_embedding), int(add_embedding) + middle_gpus):
            layers.append(MLMTaskEncoder(ninp, nhead, nhid, n_encoders_per_middle_gpu, dropout).to(devices[gpu]))

        sublayers = []
        if n_encoders_on_last_gpu > 0:
            sublayers.append(MLMTaskEncoder(ninp, nhead, nhid, n_encoders_on_last_gpu, dropout))
        if add_head:
            sublayers.append(MLMTaskHead(ntoken, ninp))
        if sublayers:
            layers.append(nn.Sequential(*sublayers).to(devices[gpus - 1]))

    assert len(layers) == gpus #, f"gpus = {gpus}, layers = {layers}"

    return LocalPipelineImpl(nn.Sequential(*layers))


# def get_shards(workers, gpus, ntoken, ninp, nhead, nhid, nlayers, dropout):
#     args = (gpus, ntoken, ninp, nhead, nhid, nlayers, dropout)
#     if workers == 1:
#         return {"worker1": (LocalPipeline, args, {'add_embedding': True, 'n_encoders': nlayers, 'add_head': True})}
#     elif workers == 2:
#         assert(nlayers % 2 == 0) # TODO: support odd number of layers
#         return {
#             "worker1": (LocalPipeline, args, {'add_embedding': True, 'n_encoders': nlayers // 2}),
#             "worker2": (LocalPipeline, args, {'n_encoders': nlayers // 2, 'add_head': True}),
#             }
#     elif workers > 3:
#         all_gpus = workers * gpus
#         encoders_gpus = all_gpus - 2
#         assert(nlayers % encoders_gpus == 0)
#         encoders_per_gpu = nlayers // encoders_gpus
#         return {
#             "worker1": (LocalPipeline, args, {'add_embedding': True, 'n_encoders': (gpus - 1)*encoders_per_gpu}),
#             **{f"worker{i}": (LocalPipeline, args, {'n_encoders': (workers-2)*gpus}) for i in range(2, workers-1)},
#             f"worker{workers}": (LocalPipeline, args, {'n_encoders': (gpus - 1)*encoders_per_gpu, 'add_head': True}),
#             }
