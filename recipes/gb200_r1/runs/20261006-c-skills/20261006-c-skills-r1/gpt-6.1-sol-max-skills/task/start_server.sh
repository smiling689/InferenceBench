#!/usr/bin/env bash
set -euo pipefail

cd /home/agent/task
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-30080}"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_NVLS_ENABLE=0
export TORCH_SYMM_MEM_DISABLE_MULTICAST=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export SGLANG_PROFILE_V2=0

exec python3 -m sglang.launch_server \
  --model-path /models/deepseek-r1 \
  --tokenizer-path /models/deepseek-r1 \
  --served-model-name deepseek-ai/DeepSeek-R1 \
  --load-format dummy \
  --dtype bfloat16 \
  --quantization fp8 \
  --kv-cache-dtype bf16 \
  --tp-size 4 \
  --random-seed 951794684 \
  --host "$HOST" --port "$PORT" \
  --context-length 4096 \
  --mem-fraction-static 0.94 \
  --max-running-requests 64 \
  --chunked-prefill-size 4096 \
  --max-prefill-tokens 8192 \
  --attention-backend trtllm_mla \
  --moe-runner-backend flashinfer_trtllm \
  --fp8-gemm-backend deep_gemm \
  --stream-interval 1 \
  --cuda-graph-bs-decode 1 2 4 8 16 24 32 48 64 \
  --cuda-graph-backend-prefill disabled \
  --disable-radix-cache \
  --enforce-disable-flashinfer-allreduce-fusion \
  --disable-custom-all-reduce
