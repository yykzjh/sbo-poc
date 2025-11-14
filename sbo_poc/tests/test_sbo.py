import atexit
import os
import random
import signal
import sys
from functools import partial
from types import SimpleNamespace

import torch
import torch.distributed as dist
import deep_ep

from sbo_poc.distributed import (
    DistributedEnvironmentInitParameters,
    destroy_distributed_environment_once,
    init_distributed_environment_once,
)
from sbo_poc.modules.fused_moe import FusedMoE
from sbo_poc.distributed.deep_ep import get_deepep_wrapper
from sbo_poc.modules import utils


separator = "=" * 80


def parse_environment_variables() -> SimpleNamespace:
    """
    Parse all environment variables required for testing.

    Returns:
        A SimpleNamespace object containing all configuration parameters, supporting attribute access
    """
    # Testing configuration
    torch_cuda_profiler_dir_path = os.getenv("TORCH_CUDA_PROFILER_DIR_PATH", "./")

    # Distributed configuration
    num_nodes = int(os.getenv("NUM_NODES", "1"))
    node_rank = int(os.getenv("NODE_RANK", "0"))
    master_addr = os.getenv("MASTER_ADDR", "127.0.0.1")
    master_port = int(os.getenv("MASTER_PORT", "8361"))
    num_processes = int(os.getenv("NUM_PROCESSES", "8"))

    # MoE model configuration
    num_topk = int(os.getenv("NUM_TOPK", "8"))
    num_experts = int(os.getenv("NUM_EXPERTS", "160"))
    hidden_size = int(os.getenv("HIDDEN_SIZE", "6144"))
    inter_size = int(os.getenv("INTER_SIZE", "5120"))
    max_generate_batch_size = int(os.getenv("MAX_GENERATE_BATCH_SIZE", "128"))
    num_combine_sms = int(os.getenv("NUM_COMBINE_SMS", "3"))
    max_block_n = int(os.getenv("MAX_BLOCK_N", "256"))

    return SimpleNamespace(
        torch_cuda_profiler_dir_path=torch_cuda_profiler_dir_path,
        num_nodes=num_nodes,
        node_rank=node_rank,
        master_addr=master_addr,
        master_port=master_port,
        num_processes=num_processes,
        num_topk=num_topk,
        num_experts=num_experts,
        hidden_size=hidden_size,
        inter_size=inter_size,
        max_generate_batch_size=max_generate_batch_size,
        num_combine_sms=num_combine_sms,
        max_block_n=max_block_n,
    )


def test_main(
    ep_rank: int,
    ep_size: int,
    num_topk: int,
    num_experts: int,
    num_tokens: int,
    hidden_size: int,
    fused_moe: FusedMoE,
    torch_cuda_profiler_dir_path: str,
) -> None:
    # Set random seed
    torch.manual_seed(42 + ep_rank)
    random.seed(42 + ep_rank)
    # Assert
    assert num_experts % ep_size == 0
    num_local_experts = num_experts // ep_size
    rank_offset = 128
    assert ep_size - rank_offset < 257, "Too many ranks (exceeding test precision limit)"

    # Generate random data
    hidden_states_check = torch.ones((num_tokens, hidden_size), dtype=torch.bfloat16, device="cuda") * (
        ep_rank - rank_offset
    )
    hidden_states_check[:, -128:] = torch.arange(num_tokens, device="cuda").to(torch.bfloat16).view(-1, 1)
    hidden_states_random = torch.randn((num_tokens, hidden_size), dtype=torch.bfloat16, device="cuda") * 0.1
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device="cuda").abs() + 1
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=True)[1]
    topk_idx = topk_idx.to(deep_ep.topk_idx_t)
    topk_weights = torch.randn((num_tokens, num_topk), dtype=torch.float32, device="cuda").abs()
    # Gather topk_idx of all ep_ranks
    all_topk_idx = torch.empty((ep_size, num_tokens, num_topk), dtype=topk_idx.dtype, device="cuda")
    dist.all_gather_into_tensor(all_topk_idx, topk_idx)

    # Get deep_ep buffer
    buffer = get_deepep_wrapper().buffer

    # Dispatch checkedhidden states to experts
    packed_recv_x, packed_recv_count, handle, hook = fused_moe.op_dispatch_send(hidden_states_check, topk_idx)
    packed_recv_x, packed_recv_count, handle = fused_moe.op_dispatch_recv(
        packed_recv_x, packed_recv_count, handle, hook
    )
    packed_recv_x = (packed_recv_x[0], packed_recv_x[1].contiguous())
    # Cast fp8 back to bf16
    simulated_gemm_x = utils.per_token_cast_back(
        packed_recv_x[0].view(-1, hidden_size), packed_recv_x[1].view(-1, hidden_size // 128)
    ).view(packed_recv_x[0].shape)
    # Check received hidden states
    for i in range(num_local_experts):
        # Get data of current local expert
        expert_id = ep_rank * num_local_experts + i
        recv_x_per_expert = utils.per_token_cast_back(packed_recv_x[0][i], packed_recv_x[1][i])
        recv_count_per_expert, recv_src_info_per_expert, recv_layout_range_per_expert = (
            packed_recv_count[i],
            # Received token index from each rank
            handle[0][i],
            # Layout range of received tokens from each rank, such as: [[begin_idx, count], [begin_idx, count], ...]
            handle[1][i],
        )
        # Mask for received token count per rank
        int_mask = (2**32) - 1
        num_valid_tokens = recv_count_per_expert.item()
        # Check the number of received tokens is equal to the number of tokens in the layout range
        assert (
            num_valid_tokens == (recv_layout_range_per_expert & int_mask).sum().item()
        ), f"{num_valid_tokens} != {recv_layout_range_per_expert & int_mask}.sum().item()"
        # If no tokens are received, skip
        if num_valid_tokens == 0:
            continue
        # Check received data
        recv_x_per_expert = recv_x_per_expert[:num_valid_tokens]
        recv_x_amin = recv_x_per_expert[:, :-128].amin(dim=-1)
        src_token_idx = recv_src_info_per_expert[:num_valid_tokens] & int_mask
        # Check the minimum and maximum are equal per received token
        assert torch.equal(recv_x_amin, recv_x_per_expert[:, :-128].amax(dim=-1))
        # Check the received token index is correct
        assert (recv_x_per_expert[:, -128:] - src_token_idx.view(-1, 1) % num_tokens).sum().item() == 0
        for j in range(ep_size):
            begin_idx, count = (recv_layout_range_per_expert[j] >> 32).item(), (
                recv_layout_range_per_expert[j] & int_mask
            ).item()
            # Check the number of tokens received from specific rank is equal to the number of tokens sent to current expert
            assert (recv_x_amin == j - rank_offset).sum().item() == (all_topk_idx[j] == expert_id).sum().item()
            # Check the tokens values of specific layout range of current rank are correct
            assert (recv_x_per_expert[begin_idx : begin_idx + count, :-128] - j + rank_offset).sum().item() == 0

    # Check combine correctness
    for overlap in (False, True):
        if overlap:
            # Initialize data for sbo
            block_m, threshold = 64, 10
            total_num_per_expert = utils.ceil_div(fused_moe.ll_num_max_token_per_rank * ep_size, block_m)
            comp_signal = torch.zeros(num_local_experts * total_num_per_expert, dtype=torch.int32, device="cuda")
            # Fill comp_signal with threshold for valid tokens, simulate the gemm completion signal
            for i in range(num_local_experts):
                vaild_num = utils.ceil_div(packed_recv_count[i], block_m)
                comp_signal[i * total_num_per_expert : i * total_num_per_expert + vaild_num] = threshold
            combined_x, hook = fused_moe.op_combine_send(
                simulated_gemm_x,
                topk_idx,
                topk_weights,
                packed_recv_count,
                handle,
                overlap=True,
                comp_signal=comp_signal,
                block_m=block_m,
                threshold=threshold,
            )
        else:
            combined_x, hook = fused_moe.op_combine_send(
                simulated_gemm_x,
                topk_idx,
                topk_weights,
                packed_recv_count,
                handle,
                overlap=False,
            )
        fused_moe.op_combine_recv(combined_x, hook)
        # Check the combined hidden states are not nan
        assert torch.isnan(combined_x).sum().item() == 0
        # Check the combined hidden states are correct
        diff = utils.calc_diff(
            hidden_states_check * topk_weights.masked_fill(topk_idx == -1, 0).sum(dim=1).view(-1, 1), combined_x
        )
        assert diff < 9e-4, f"Error: {diff=}"

    # Execute preliminary steps
    recv_x, recv_count, handle, hook = fused_moe.op_dispatch_send(hidden_states_random, topk_idx)
    recv_x, recv_count, handle = fused_moe.op_dispatch_recv(recv_x, recv_count, handle, hook)
    recv_x = (recv_x[0], recv_x[1].contiguous())
    workspace, packed_recv_count = fused_moe.op_up_gemm(recv_x[0], recv_x[1], recv_count)
    a2q, a2q_scale = fused_moe.op_activation(workspace, packed_recv_count)
    # Initialize compute data in hook
    mat_0 = torch.randn((256, 256), dtype=torch.float)
    mat_1 = torch.randn((256, 256), dtype=torch.float)

    # Define test function
    def test_func(overlap: bool = True):
        if overlap:
            output, block_m, signal_threshold, comp_signal = fused_moe.op_down_gemm(
                a2q, a2q_scale, packed_recv_count, overlap=True
            )
            combined_x, hook = fused_moe.op_combine_send(
                output,
                topk_idx,
                topk_weights,
                packed_recv_count,
                handle,
                overlap=True,
                comp_signal=comp_signal,
                block_m=block_m,
                threshold=signal_threshold,
            )
        else:
            output = fused_moe.op_down_gemm(a2q, a2q_scale, packed_recv_count, overlap=False)
            combined_x, hook = fused_moe.op_combine_send(
                output,
                topk_idx,
                topk_weights,
                packed_recv_count,
                handle,
                overlap=False,
            )
        # mat_0 @ mat_1
        fused_moe.op_combine_recv(combined_x, hook)

    # Calculate bandwidth
    num_combine_comm_bytes = 0
    for i in range(num_tokens):
        num_selections = (topk_idx[i] != -1).sum().item()
        num_combine_comm_bytes += hidden_size * 2 * num_selections
    # Benchmark test function
    for overlap in (False, True):
        avg_t, min_t, max_t = utils.bench(partial(test_func, overlap=overlap), num_warmups=50, num_tests=30)
        print(
            f"[rank {ep_rank}] combine bandwidth (overlap={overlap}): {num_combine_comm_bytes / 1e9 / avg_t:.2f} GB/s, "
            f"avg_t={avg_t * 1e6:.2f} us, min_t={min_t * 1e6:.2f} us, max_t={max_t * 1e6:.2f} us",
            flush=True,
        )
    # Profile timeline
    for overlap in (False, True):
        dist.barrier()
        combine_t = utils.bench_kineto(
            partial(test_func, overlap=overlap),
            kernel_names=("combine",),
            num_tests=30,
            suppress_kineto_output=True,
            trace_path=os.path.join(
                torch_cuda_profiler_dir_path,
                f"sbo_poc_trace_overlap-{overlap}_num_tokens-{num_tokens}_rank-{ep_rank}.json",
            ),
            barrier_comm_profiling=True,
            num_kernels_per_period=2,
        )
        print(
            f"[rank {ep_rank}] Combine send/recv time (overlap={overlap}): {combine_t[0][0] * 1e6:.2f} + {combine_t[0][1] * 1e6:.2f} us | ",
            flush=True,
        )


def test_loop(local_rank: int, num_local_ranks: int, args: SimpleNamespace) -> None:
    # Calculate distributed info
    num_nodes = args.num_nodes
    node_rank = args.node_rank
    ep_rank = node_rank * num_local_ranks + local_rank
    ep_size = num_nodes * num_local_ranks
    # Get test config data
    torch_cuda_profiler_dir_path = args.torch_cuda_profiler_dir_path
    num_topk = args.num_topk
    num_experts = args.num_experts
    hidden_size = args.hidden_size
    inter_size = args.inter_size
    max_generate_batch_size = args.max_generate_batch_size
    num_combine_sms = args.num_combine_sms
    max_block_n = args.max_block_n
    # 注册清理函数，确保在程序退出时执行资源释放
    cleanup_registered = False

    def cleanup_resources():
        """清理资源的函数"""
        nonlocal cleanup_registered
        if cleanup_registered:
            return
        cleanup_registered = True
        try:
            if torch.distributed.is_initialized():
                destroy_distributed_environment_once()
        except Exception as e:
            if local_rank == 0:
                print(f"Error during cleanup: {e}", flush=True, file=sys.stderr)

    # 注册 atexit 处理函数
    atexit.register(cleanup_resources)

    # 注册信号处理器（处理 SIGTERM, SIGINT 等）
    def signal_handler(signum, frame):
        if local_rank == 0:
            print(f"Received signal {signum}, cleaning up resources...", flush=True)
        cleanup_resources()
        sys.exit(1)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    try:
        # Initialize distributed environment
        init_distributed_environment_once(
            DistributedEnvironmentInitParameters(
                master_addr=args.master_addr,
                master_port=args.master_port,
                ep_rank=ep_rank,
                ep_size=ep_size,
                tp_size=1,
                local_rank=local_rank,
                num_experts=args.num_experts,
                hidden_size=args.hidden_size,
                max_generate_batch_size=args.max_generate_batch_size,
            )
        )
        # Initialize FusedMoE module
        fused_moe = FusedMoE(
            ep_rank=ep_rank,
            ep_size=ep_size,
            tp_rank=0,
            tp_size=1,
            num_topk=num_topk,
            num_experts=num_experts,
            hidden_size=hidden_size,
            inter_size=inter_size,
            max_generate_batch_size=max_generate_batch_size,
            num_combine_sms=num_combine_sms,
            max_block_n=max_block_n,
        )
        # Run test main
        for num_tokens in range(1, max_generate_batch_size + 1, 64):
            if local_rank == 0:
                print(separator, f"Running test with num_tokens={num_tokens}", separator, flush=True)
            test_main(
                ep_rank,
                ep_size,
                num_topk,
                num_experts,
                num_tokens,
                hidden_size,
                fused_moe,
                torch_cuda_profiler_dir_path,
            )
            dist.barrier()
            if local_rank == 0:
                print(separator, f"Test with num_tokens={num_tokens} completed", separator, flush=True)
    except Exception as e:
        # 捕获所有异常，确保资源被释放
        if local_rank == 0:
            print(f"Error occurred: {e}", flush=True, file=sys.stderr)
        raise  # 重新抛出异常
    finally:
        # 无论是否发生异常，都执行资源清理
        cleanup_resources()


if __name__ == "__main__":
    # Parse environment variables
    args = parse_environment_variables()

    # Launch multi processes
    num_processes = args.num_processes
    torch.multiprocessing.spawn(test_loop, args=(num_processes, args), nprocs=num_processes)
