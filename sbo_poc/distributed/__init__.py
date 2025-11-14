import os
from dataclasses import dataclass

import torch

from sbo_poc.distributed.deep_ep import (
    DeepEPInitParameters,
    destroy_deepep_wrapper,
    init_deepep_wrapper,
)

_GLOBAL_PROCESS_GROUP = None


@dataclass
class DistributedEnvironmentInitParameters:
    master_addr: str
    master_port: int
    ep_rank: int
    ep_size: int
    tp_size: int
    local_rank: int
    num_experts: int
    hidden_size: int
    max_generate_batch_size: int


def init_distributed_environment_once(params: DistributedEnvironmentInitParameters):
    # Initialize global process group
    if not torch.distributed.is_initialized():
        os.environ["TORCH_DIST_INIT_BARRIER"] = "1"
        torch.distributed.init_process_group(
            backend="nccl",
            init_method=f"tcp://{params.master_addr}:{params.master_port}",
            world_size=params.ep_size,
            rank=params.ep_rank,
            device_id=torch.device(f"cuda:{params.local_rank}"),
            timeout=None,
        )
    # Set default device
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device(f"cuda:{params.local_rank}")
    torch.cuda.set_device(params.local_rank)
    # Initialize deep_ep
    deepep_init_params = DeepEPInitParameters(
        ep_rank=params.ep_rank,
        ep_size=params.ep_size,
        tp_size=params.tp_size,
        num_experts=params.num_experts,
        hidden_size=params.hidden_size,
        max_generate_batch_size=params.max_generate_batch_size,
    )
    global _GLOBAL_PROCESS_GROUP
    _GLOBAL_PROCESS_GROUP = torch.distributed.new_group(list(range(params.ep_size)))
    init_deepep_wrapper(_GLOBAL_PROCESS_GROUP, deepep_init_params)


def destroy_distributed_environment_once():
    # Destroy deep_ep
    destroy_deepep_wrapper()
    # Destroy global process group
    global _GLOBAL_PROCESS_GROUP
    if _GLOBAL_PROCESS_GROUP is not None:
        torch.distributed.destroy_process_group(_GLOBAL_PROCESS_GROUP)
        _GLOBAL_PROCESS_GROUP = None
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
