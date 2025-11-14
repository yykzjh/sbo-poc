#!/bin/bash

export PYTHONPATH="./"
# export NCCL_DEBUG="INFO"
# export NVSHMEM_DEBUG="INFO"

export NUM_NODES=1
export NODE_RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=8361
export NUM_PROCESSES=8
export NUM_TOPK=8
export NUM_EXPERTS=160
export HIDDEN_SIZE=6144
export INTER_SIZE=5120
export MAX_GENERATE_BATCH_SIZE=192
export NUM_COMBINE_SMS=5
export MAX_BLOCK_N=256

export TORCH_CUDA_PROFILER_DIR_PATH="./trace_files/"

python sbo_poc/tests/test_sbo.py
