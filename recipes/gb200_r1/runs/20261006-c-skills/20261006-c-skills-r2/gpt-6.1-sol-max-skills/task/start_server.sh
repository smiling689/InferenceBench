#!/usr/bin/env bash
set -euo pipefail

TASK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${TASK_DIR}/server_config.env"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export NCCL_NVLS_ENABLE=0
export TORCH_SYMM_MEM_DISABLE_MULTICAST=1
export SGLANG_SHARED_EXPERT_TP1=0
export SGLANG_PROFILE_V2=0
export SGLANG_JIT_DEEPGEMM_FAST_WARMUP=1
export GLOO_SOCKET_IFNAME=lo

HOST="${HOST:-${INFERENCE_BENCH_SERVER_HOST:-127.0.0.1}}"
PORT="${PORT:-${INFERENCE_BENCH_SERVER_PORT:-30080}}"

exec python3 -m sglang.launch_server \
    --model-path /models/deepseek-r1 \
    --tokenizer-path /models/deepseek-r1 \
    --served-model-name deepseek-ai/DeepSeek-R1 \
    --load-format dummy \
    --quantization fp8 \
    --dtype bfloat16 \
    --kv-cache-dtype bf16 \
    --tp-size 4 \
    --ep-size "${EP_SIZE}" \
    --context-length "${CONTEXT_LENGTH}" \
    --mem-fraction-static "${MEM_FRACTION}" \
    --max-total-tokens "${MAX_TOTAL_TOKENS}" \
    --max-running-requests "${MAX_RUNNING_REQUESTS}" \
    --chunked-prefill-size "${CHUNKED_PREFILL_SIZE}" \
    --attention-backend "${ATTENTION_BACKEND}" \
    --moe-runner-backend "${MOE_BACKEND}" \
    --page-size 64 \
    --cuda-graph-backend-decode "${DECODE_GRAPH}" \
    --cuda-graph-backend-prefill "${PREFILL_GRAPH}" \
    --disable-radix-cache \
    --enforce-disable-flashinfer-allreduce-fusion \
    --disable-custom-all-reduce \
    --random-seed 42 \
    --host "${HOST}" \
    --port "${PORT}" \
    "${EXTRA_FLAGS[@]}"
