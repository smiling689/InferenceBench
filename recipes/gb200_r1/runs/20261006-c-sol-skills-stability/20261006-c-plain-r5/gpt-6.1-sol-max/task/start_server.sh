#!/usr/bin/env bash
set -euo pipefail

TASK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CACHE_ROOT="$TASK_DIR/.cache"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-${INFERENCE_BENCH_SERVER_PORT:-30080}}"

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export SGLANG_SHARED_EXPERT_TP1=0
export NCCL_NVLS_ENABLE=0
export TORCH_SYMM_MEM_DISABLE_MULTICAST=1
export OMP_NUM_THREADS=8
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0
export GLOO_SOCKET_IFNAME=lo
export SGLANG_CACHE_DIR="$CACHE_ROOT/sglang"
export SGLANG_DG_CACHE_DIR="$CACHE_ROOT/deep_gemm"
export FLASHINFER_WORKSPACE_BASE="$TASK_DIR"
export TRITON_CACHE_DIR="$CACHE_ROOT/triton"
export TVM_FFI_CACHE_DIR="$CACHE_ROOT/tvm-ffi"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_ROOT/torchinductor"
export XDG_CACHE_HOME="$CACHE_ROOT"

exec python3 -m sglang.launch_server \
    --model-path /models/deepseek-r1 \
    --tokenizer-path /models/deepseek-r1 \
    --served-model-name deepseek-ai/DeepSeek-R1 \
    --load-format dummy \
    --quantization fp8 \
    --dtype bfloat16 \
    --kv-cache-dtype bf16 \
    --tp-size 4 \
    --ep-size 2 \
    --context-length 16384 \
    --mem-fraction-static 0.94 \
    --max-total-tokens 131072 \
    --max-running-requests 64 \
    --chunked-prefill-size 4096 \
    --attention-backend trtllm_mla \
    --moe-runner-backend flashinfer_trtllm \
    --page-size 64 \
    --cuda-graph-backend-decode full \
    --cuda-graph-max-bs-decode 64 \
    --cuda-graph-bs-decode 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 \
    --cuda-graph-backend-prefill tc_piecewise \
    --cuda-graph-tc-compiler eager \
    --cuda-graph-max-bs-prefill 4096 \
    --cuda-graph-bs-prefill 832 896 960 1024 2048 4096 \
    --disable-radix-cache \
    --stream-interval 16 \
    --enforce-disable-flashinfer-allreduce-fusion \
    --disable-custom-all-reduce \
    --random-seed 42 \
    --host "$HOST" \
    --port "$PORT"
