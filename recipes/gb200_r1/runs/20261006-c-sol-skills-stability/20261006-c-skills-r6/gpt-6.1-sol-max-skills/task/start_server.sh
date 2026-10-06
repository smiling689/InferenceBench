#!/usr/bin/env bash
set -euo pipefail

TASK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-30080}"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_NVLS_ENABLE=0
export TORCH_SYMM_MEM_DISABLE_MULTICAST=1
export SGLANG_SHARED_EXPERT_TP1=0
export SGLANG_PROFILE_V2=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export SGLANG_CACHE_DIR="${TASK_DIR}/runtime_cache/sglang"
export FLASHINFER_WORKSPACE_BASE="${TASK_DIR}/runtime_cache/flashinfer"
export TRITON_CACHE_DIR="${TASK_DIR}/runtime_cache/triton"
mkdir -p "${SGLANG_CACHE_DIR}" "${FLASHINFER_WORKSPACE_BASE}" "${TRITON_CACHE_DIR}"

exec python3 -m sglang.launch_server \
  --model-path /models/deepseek-r1 \
  --tokenizer-path /models/deepseek-r1 \
  --served-model-name deepseek-ai/DeepSeek-R1 \
  --host "${HOST}" --port "${PORT}" \
  --load-format dummy --quantization fp8 \
  --dtype bfloat16 --kv-cache-dtype bf16 \
  --tp-size 4 --ep-size 1 \
  --attention-backend trtllm_mla \
  --moe-runner-backend flashinfer_trtllm \
  --fp8-gemm-backend flashinfer_trtllm \
  --random-seed 549333021 \
  --context-length 16384 \
  --mem-fraction-static 0.94 \
  --max-running-requests 64 \
  --max-total-tokens 131072 \
  --chunked-prefill-size 4096 \
  --max-prefill-tokens 8192 \
  --scheduler-recv-interval 16 \
  --cuda-graph-backend-decode full \
  --cuda-graph-backend-prefill disabled \
  --cuda-graph-bs-decode 1 2 4 8 16 24 32 48 64 \
  --disable-radix-cache \
  --enforce-disable-flashinfer-allreduce-fusion \
  --disable-custom-all-reduce
