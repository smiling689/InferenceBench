#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-30080}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NCCL_NVLS_ENABLE=0
unset NCCL_PROTO
export TORCH_SYMM_MEM_DISABLE_MULTICAST=1
export SGLANG_PROFILE_V2=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export SGLANG_SIMULATE_ACC_LEN=-1

exec python3 -m sglang.launch_server \
    --model-path /models/deepseek-r1 \
    --tokenizer-path /models/deepseek-r1 \
    --served-model-name deepseek-ai/DeepSeek-R1 \
    --host "$HOST" --port "$PORT" \
    --load-format dummy \
    --dtype bfloat16 \
    --quantization fp8 \
    --kv-cache-dtype bf16 \
    --tp-size 4 \
    --attention-backend trtllm_mla \
    --moe-runner-backend flashinfer_trtllm \
    --fp8-gemm-backend flashinfer_trtllm \
    --mem-fraction-static 0.95 \
    --context-length 4096 \
    --max-running-requests 64 \
    --max-total-tokens 131072 \
    --chunked-prefill-size 8192 \
    --cuda-graph-max-bs-decode 64 \
    --cuda-graph-bs-decode 1 2 4 8 12 16 24 32 40 48 56 64 \
    --cuda-graph-backend-prefill disabled \
    --schedule-policy fcfs \
    --disable-radix-cache \
    --random-seed 1062112617 \
    --speculative-algorithm NEXTN \
    --speculative-draft-load-format dummy \
    --speculative-num-steps 7 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 8 \
    --speculative-attention-mode decode \
    --speculative-use-rejection-sampling \
    --enforce-disable-flashinfer-allreduce-fusion \
    --disable-custom-all-reduce
