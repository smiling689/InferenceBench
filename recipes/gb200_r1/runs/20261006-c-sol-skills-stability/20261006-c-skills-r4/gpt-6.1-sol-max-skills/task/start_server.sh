#!/usr/bin/env bash
set -euo pipefail

cd /home/agent/task
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
export NCCL_NVLS_ENABLE=0
export TORCH_SYMM_MEM_DISABLE_MULTICAST=1
export SGLANG_SHARED_EXPERT_TP1=0
export SGLANG_PROFILE_V2=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4

HOST="${INFERENCE_BENCH_SERVER_HOST:-${HOST:-127.0.0.1}}"
PORT="${INFERENCE_BENCH_SERVER_PORT:-${PORT:-30080}}"

exec python3 -m sglang.launch_server \
    --model-path /models/deepseek-r1 \
    --tokenizer-path /models/deepseek-r1 \
    --served-model-name deepseek-ai/DeepSeek-R1 \
    --load-format dummy \
    --quantization fp8 \
    --dtype bfloat16 \
    --kv-cache-dtype bf16 \
    --tp-size 4 \
    --ep-size 1 \
    --context-length 16384 \
    --mem-fraction-static 0.94 \
    --max-running-requests 64 \
    --chunked-prefill-size 4096 \
    --attention-backend trtllm_mla \
    --fp8-gemm-backend flashinfer_trtllm \
    --moe-runner-backend flashinfer_trtllm \
    --page-size 64 \
    --cuda-graph-backend-decode full \
    --cuda-graph-max-bs-decode 64 \
    --cuda-graph-bs-decode 1 2 4 8 12 16 20 24 28 32 40 48 56 64 \
    --cuda-graph-backend-prefill disabled \
    --disable-radix-cache \
    --enforce-disable-flashinfer-allreduce-fusion \
    --disable-custom-all-reduce \
    --random-seed 42 \
    --host "$HOST" \
    --port "$PORT"
