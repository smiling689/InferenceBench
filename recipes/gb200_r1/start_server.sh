#!/usr/bin/env bash
set -euo pipefail

# Preparation profile: full 61-layer R1, one TP4 replica, actual GPU computation.
# This launcher belongs inside the Docker container created by docker.sh.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export SGLANG_SHARED_EXPERT_TP1=0
export PYTHONDONTWRITEBYTECODE=1
export NCCL_NVLS_ENABLE=0

exec python3 -m sglang.launch_server \
    --model-path /models/deepseek-r1 \
    --served-model-name deepseek-ai/DeepSeek-R1 \
    --load-format dummy \
    --quantization fp8 \
    --dtype bfloat16 \
    --kv-cache-dtype bf16 \
    --tp-size 4 \
    --ep-size 1 \
    --context-length 16384 \
    --mem-fraction-static 0.94 \
    --max-total-tokens 65536 \
    --max-running-requests 64 \
    --chunked-prefill-size 4096 \
    --attention-backend trtllm_mla \
    --moe-runner-backend flashinfer_trtllm \
    --page-size 64 \
    --disable-cuda-graph \
    --disable-radix-cache \
    --disable-overlap-schedule \
    --enforce-disable-flashinfer-allreduce-fusion \
    --disable-custom-all-reduce \
    --random-seed 42 \
    --host 127.0.0.1 \
    --port "${BENCH_PORT:-30080}"
