from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
from deep_gemm import m_grouped_fp8_gemm_nt_masked

from sbo_poc.modules import utils
from sbo_poc.distributed.deep_ep import get_deepep_wrapper
from sbo_poc.modules.ops.activation import silu_mul_fp8_quant_deep_gemm_masked
from sbo_poc.modules.ops.deep_gemm_wrapper import configure_deep_gemm_num_sms


class FusedMoE(nn.Module):
    """Fused MoE module"""

    def __init__(
        self,
        ep_rank: int,
        ep_size: int,
        tp_rank: int,
        tp_size: int,
        num_topk: int,
        num_experts: int,
        hidden_size: int,
        inter_size: int,
        max_generate_batch_size: int,
        num_combine_sms: int = 3,
        max_block_n: int = 256,
    ):
        """Initialize the FusedMoE module

        Args:
            ep_rank (int): the rank of the expert parallel group
            ep_size (int): the size of the expert parallel group
            tp_rank (int): the rank of the tensor parallel group
            tp_size (int): the size of the tensor parallel group
            num_topk (int): the number of top-k experts to select
            num_experts (int): the total number of experts
            hidden_size (int): the hidden size of the input tokens
            inter_size (int): the intermediate size of the intermediate tokens
            max_generate_batch_size (int): the maximum number of tokens per dp rank
            num_combine_sms (int): the number of SMs to use for the combine operation
            max_block_n (int): the maximum block size for the deep_gemm operation
        """
        super(FusedMoE, self).__init__()
        # Initialize common parameters
        self.ep_rank = ep_rank
        self.ep_size = ep_size
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.num_topk = num_topk
        self.num_experts = num_experts
        self.num_local_experts = self.num_experts // self.ep_size
        self.hidden_size = hidden_size
        self.inter_size = inter_size
        self.max_generate_batch_size = max_generate_batch_size
        self.num_combine_sms = num_combine_sms
        self.max_block_n = max_block_n
        # Calculate DeepGemm SMs
        total_num_sms = torch.cuda.get_device_properties(
            device="cuda"
        ).multi_processor_count
        self.num_deep_gemm_sms = total_num_sms - self.num_combine_sms
        # Initialize deep_ep buffer
        self.buffer = get_deepep_wrapper().buffer
        self.ll_num_max_token_per_rank = get_deepep_wrapper().ll_num_max_token_per_rank
        # Initialize Up and Down GEMMs
        self.M = self.ep_size * self.ll_num_max_token_per_rank
        self.K = self.hidden_size
        self.N = self.inter_size
        self._w1 = torch.randn((self.num_local_experts, self.N, self.K), dtype=torch.float32, device="cuda").to(
            torch.float8_e4m3fn
        )
        self._w1_scale = torch.randn(
            (self.num_local_experts, self.N // 128, self.K // 128), dtype=torch.float32, device="cuda"
        )
        self._w2 = torch.randn((self.num_local_experts, self.K, self.N // 2), dtype=torch.float32, device="cuda").to(
            torch.float8_e4m3fn
        )
        self._w2_scale = torch.randn(
            (self.num_local_experts, self.K // 128, self.N // 2 // 128), dtype=torch.float32, device="cuda"
        )
        self.workspace = torch.empty((self.num_local_experts, self.M, self.N), dtype=torch.bfloat16, device="cuda")
        self.output = torch.empty((self.num_local_experts, self.M, self.K), dtype=torch.bfloat16, device="cuda")
        self.comp_signal = torch.empty(
            (self.num_local_experts * utils.ceil_div(self.M, 64)), dtype=torch.int32, device="cuda"
        )
        self.expect_m = self.M
        # Initialize combine send steam
        self.combine_send_event = torch.cuda.Event()
        self.combine_send_stream = torch.cuda.Stream()

    def op_dispatch_send(
        self, hidden_states: torch.Tensor, topk_idx: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple, Callable]:
        recv_x, recv_count, handle, _, hook = self.buffer.low_latency_dispatch(
            hidden_states,
            topk_idx,
            self.ll_num_max_token_per_rank,
            self.num_experts,
            use_fp8=True,
            async_finish=False,
            return_recv_hook=True,
        )
        self.expect_m = max(int(hidden_states.shape[0] * self.ep_size * self.num_topk / self.num_experts), 1)
        return recv_x, recv_count, handle, hook

    def op_dispatch_recv(
        self, recv_x: torch.Tensor, recv_count: torch.Tensor, handle: Tuple, hook: Callable
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple]:
        hook()
        return recv_x, recv_count, handle

    def op_up_gemm(
        self, expert_x: torch.Tensor, expert_x_scale: torch.Tensor, packed_recv_count: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        m_grouped_fp8_gemm_nt_masked(
            (expert_x, expert_x_scale),
            (self._w1, self._w1_scale),
            self.workspace,
            packed_recv_count,
            self.expect_m,
            compiled_dims="nk",
            disable_ue8m0_cast=True,
        )
        return self.workspace, packed_recv_count

    def op_activation(
        self, workspace: torch.Tensor, packed_recv_count: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        a2q, a2q_scale = silu_mul_fp8_quant_deep_gemm_masked(
            workspace,
            packed_recv_count,
            group_size=128,
            use_ue8m0=False,
            eps=1e-10,
        )
        return a2q, a2q_scale

    def op_down_gemm(
        self,
        a2q: torch.Tensor,
        a2q_scale: torch.Tensor,
        packed_recv_count: torch.Tensor,
        overlap: bool = False,
    ) -> Tuple[torch.Tensor, int, int]:
        if overlap:
            self.comp_signal.zero_()
            self.combine_send_event.record(torch.cuda.current_stream())
            with configure_deep_gemm_num_sms(self.num_deep_gemm_sms):
                block_m, signal_threshold = m_grouped_fp8_gemm_nt_masked(
                    (a2q, a2q_scale),
                    (self._w2, self._w2_scale),
                    self.output,
                    packed_recv_count,
                    self.expect_m,
                    compiled_dims="nk",
                    disable_ue8m0_cast=True,
                    max_block_n=self.max_block_n,
                    enable_overlap=True,
                    signal=self.comp_signal,
                )
            return self.output, block_m, signal_threshold, self.comp_signal
        else:
            m_grouped_fp8_gemm_nt_masked(
                (a2q, a2q_scale),
                (self._w2, self._w2_scale),
                self.output,
                packed_recv_count,
                self.expect_m,
                compiled_dims="nk",
                enable_overlap=False,
            )
            return self.output

    def op_combine_send(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        packed_recv_count: torch.Tensor,
        handle: Tuple,
        overlap: bool = False,
        comp_signal: Optional[torch.Tensor] = None,
        block_m: int = 64,
        threshold: int = 0,
    ) -> Tuple[torch.Tensor, Callable]:
        if overlap:
            self.combine_send_stream.wait_event(self.combine_send_event)
            with torch.cuda.stream(self.combine_send_stream):
                combined_x, _, hook = self.buffer.low_latency_combine(
                    hidden_states,
                    topk_idx,
                    topk_weights,
                    handle,
                    overlap=overlap,
                    packed_recv_count=packed_recv_count,
                    comp_signal=comp_signal,
                    block_m=block_m,
                    threshold=threshold,
                    num_sms=self.num_combine_sms,
                    use_logfmt=False,
                    async_finish=False,
                    zero_copy=False,
                    return_recv_hook=True,
                )
        else:
            combined_x, _, hook = self.buffer.low_latency_combine(
                hidden_states,
                topk_idx,
                topk_weights,
                handle,
                overlap=False,
                packed_recv_count=packed_recv_count,
                use_logfmt=False,
                async_finish=False,
                zero_copy=False,
                return_recv_hook=True,
            )
        return combined_x, hook

    def op_combine_recv(self, combined_x: torch.Tensor, hook: Callable) -> torch.Tensor:
        hook()
        torch.cuda.current_stream().wait_stream(self.combine_send_stream)
        return combined_x
