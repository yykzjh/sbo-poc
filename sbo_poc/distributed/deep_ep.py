import gc
from enum import IntEnum, auto
from dataclasses import dataclass
from typing import Optional, Tuple

from deep_ep import Buffer as DeepEPBuffer
from deep_ep import Config as DeepEPConfig
from torch.distributed import ProcessGroup

__all__ = [
    "DeepEPBuffer",
    "DeepEPConfig",
    "DeepEPInitParameters",
    "init_deepep_wrapper",
    "get_deepep_wrapper",
    "destroy_deepep_wrapper",
]


@dataclass
class DeepEPInitParameters:
    ep_rank: int
    ep_size: int
    tp_size: int
    num_experts: int
    hidden_size: int
    max_generate_batch_size: int


class DeepEPMode(IntEnum):
    """
    The mode of deep_ep.
    """

    NORMAL = auto()
    LOW_LATENCY = auto()
    LOW_LATENCY_M2N = auto()


class DeepEPWrapper:
    """
    A wrapper for deep_ep.
    """

    _num_rdma_bytes: int = 0
    _ll_num_max_token_per_rank: int = 0
    _mode: DeepEPMode = DeepEPMode.NORMAL
    _buffer: Optional[DeepEPBuffer] = None

    def __init__(self, group: ProcessGroup, params: DeepEPInitParameters) -> None:
        self._ep_rank = params.ep_rank
        self._ep_size = params.ep_size
        self._tp_size = params.tp_size
        self._num_experts = params.num_experts
        self._hidden_size = params.hidden_size
        self._max_generate_batch_size = params.max_generate_batch_size
        self._mode, self._buffer = self._init_deepep_buffer(group, params)

    @property
    def buffer(self) -> DeepEPBuffer:
        assert self._buffer is not None, "deep_ep buffer is not initialized"
        return self._buffer

    @property
    def mode(self) -> DeepEPMode:
        return self._mode

    @property
    def num_rdma_bytes(self) -> int:
        return self._num_rdma_bytes

    @property
    def ll_num_max_token_per_rank(self) -> int:
        return self._ll_num_max_token_per_rank

    def _init_deepep_buffer(self, group: ProcessGroup, params: DeepEPInitParameters) -> Tuple[DeepEPMode, DeepEPBuffer]:
        # init deep_ep buffer
        return DeepEPMode.LOW_LATENCY, self._init_low_latency_buffer(group, params)

    def _calc_low_latency_max_token_per_rank(self, max_generate_batch_size: int, tp_size: int) -> int:
        ll_num_max_token_per_rank = (max_generate_batch_size + tp_size - 1) // tp_size

        matched_tokens = [
            16,
            24,
            32,
            40,
            48,
            56,
            64,
            72,
            80,
            88,
            96,
            104,
            112,
            120,
            128,
        ]
        if ll_num_max_token_per_rank > 128:
            ll_num_max_token_per_rank = ((ll_num_max_token_per_rank + 127) // 128) * 128
            return ll_num_max_token_per_rank
        for t in matched_tokens:
            if ll_num_max_token_per_rank <= t:
                ll_num_max_token_per_rank = t
                return ll_num_max_token_per_rank
        return 128

    def _init_low_latency_buffer(self, group: ProcessGroup, params: DeepEPInitParameters) -> DeepEPBuffer:
        max_generate_batch_size: int = params.max_generate_batch_size
        tp_size: int = params.tp_size
        assert max_generate_batch_size > 0 and tp_size > 0, "max_generate_batch_size and tp_size must be set"
        self._ll_num_max_token_per_rank = self._calc_low_latency_max_token_per_rank(max_generate_batch_size, tp_size)

        hidden_size: int = params.hidden_size
        ep_size: int = params.ep_size
        num_experts: int = params.num_experts
        num_qps_per_rank = num_experts / ep_size
        assert hidden_size > 0 and ep_size > 0 and num_experts > 0, "hidden_size, ep_size and num_experts must be set"
        self._num_rdma_bytes = DeepEPBuffer.get_low_latency_rdma_size_hint(
            self._ll_num_max_token_per_rank,
            hidden_size,
            ep_size,
            num_experts,
        )

        init_kwargs = {
            "group": group,
            "num_rdma_bytes": self._num_rdma_bytes,
            "low_latency_mode": True,
            "num_qps_per_rank": num_qps_per_rank,
            "allow_nvlink_for_low_latency_mode": True,
            "explicitly_destroy": True,
            "allow_mnnvl": False,
            "enable_shrink": False,
        }
        return DeepEPBuffer(**init_kwargs)  # type: ignore

    def destroy_deepep_buffer(self) -> None:
        if self._buffer is not None:
            self._buffer.destroy()
            del self._buffer
            self._buffer = None
        gc.collect()


_DEEP_EP: Optional[DeepEPWrapper] = None


def get_deepep_wrapper() -> DeepEPWrapper:
    assert _DEEP_EP is not None, "deep_ep wrapper is not initialized"
    return _DEEP_EP


def init_deepep_wrapper(group: ProcessGroup, params: DeepEPInitParameters) -> None:
    global _DEEP_EP
    if _DEEP_EP is not None:
        return
    _DEEP_EP = DeepEPWrapper(group, params)  # pyright: ignore[reportConstantRedefinition]


def destroy_deepep_wrapper() -> None:
    global _DEEP_EP
    if _DEEP_EP:
        _DEEP_EP.destroy_deepep_buffer()
    _DEEP_EP = None  # pyright: ignore[reportConstantRedefinition]
    gc.collect()
