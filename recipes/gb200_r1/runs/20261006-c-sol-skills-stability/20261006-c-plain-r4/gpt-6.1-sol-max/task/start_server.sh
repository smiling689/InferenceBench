#!/usr/bin/env bash
set -euo pipefail

cd /home/agent/task

export NCCL_NVLS_ENABLE=0
export TORCH_SYMM_MEM_DISABLE_MULTICAST=1
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-${INFERENCE_BENCH_SERVER_PORT:-30080}}"

exec python3 -m sglang.launch_server \
    --model-path /models/deepseek-r1 \
    --tokenizer-path /models/deepseek-r1 \
    --served-model-name /models/deepseek-r1 \
    --host "$HOST" \
    --port "$PORT" \
    --load-format dummy \
    --dtype bfloat16 \
    --kv-cache-dtype bf16 \
    --tp-size 4 \
    --attention-backend trtllm_mla \
    --moe-runner-backend flashinfer_trtllm \
    --fp8-gemm-backend flashinfer_trtllm \
    --mem-fraction-static 0.95 \
    --context-length 16384 \
    --max-running-requests 128 \
    --max-total-tokens 131072 \
    --chunked-prefill-size 8192 \
    --scheduler-recv-interval 128 \
    --cuda-graph-max-bs 64 \
    --disable-radix-cache \
    --enforce-disable-flashinfer-allreduce-fusion \
    --disable-custom-all-reduce
