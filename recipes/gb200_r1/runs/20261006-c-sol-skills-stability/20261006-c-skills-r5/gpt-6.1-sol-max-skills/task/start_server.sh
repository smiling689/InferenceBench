#!/usr/bin/env bash
set -euo pipefail

cd /home/agent/task
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-30080}"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_NVLS_ENABLE=0
export TORCH_SYMM_MEM_DISABLE_MULTICAST=1
unset NCCL_ALGO NCCL_PROTO NCCL_DEBUG_SUBSYS
export NCCL_DEBUG=WARN
export GLOO_SOCKET_IFNAME=lo
export R1_PREFILL_CLOCK_PATCH=1
export PYTHONPATH=/home/agent/task/runtime_patches${PYTHONPATH:+:$PYTHONPATH}
export SGLANG_PROFILE_V2=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=8
export SGLANG_CACHE_DIR=/home/agent/task/runtime_cache/sglang
export SGLANG_DG_CACHE_DIR=/home/agent/task/runtime_cache/deep_gemm
export FLASHINFER_WORKSPACE_BASE=/home/agent/task/runtime_cache/flashinfer
export TRITON_CACHE_DIR=/home/agent/task/runtime_cache/triton

test -f /home/agent/task/runtime_patches/sitecustomize.py
test -f /home/agent/task/runtime_patches/r1_prefill_clock.py
mkdir -p "$SGLANG_CACHE_DIR" "$SGLANG_DG_CACHE_DIR" "$FLASHINFER_WORKSPACE_BASE" "$TRITON_CACHE_DIR"

server_args=(
    --model-path /models/deepseek-r1
    --tokenizer-path /models/deepseek-r1
    --served-model-name deepseek-ai/DeepSeek-R1
    --load-format dummy
    --dtype bfloat16
    --quantization fp8
    --kv-cache-dtype bf16
    --tp-size 4
    --random-seed 327017732
    --host "$HOST"
    --port "$PORT"
    --context-length 4096
    --mem-fraction-static 0.95
    --max-running-requests 128
    --chunked-prefill-size 4096
    --enable-prefill-delayer
    --prefill-delayer-queue-min-ratio 0.15
    --prefill-delayer-max-delay-ms 350
    --attention-backend trtllm_mla
    --moe-runner-backend flashinfer_trtllm
    --fp8-gemm-backend flashinfer_trtllm
    --cuda-graph-backend-decode full
    --cuda-graph-backend-prefill disabled
    --cuda-graph-bs-decode 1 2 4 8 16 32 48 64
    --disable-radix-cache
    --enforce-disable-flashinfer-allreduce-fusion
    --disable-custom-all-reduce
    --log-level info
)

printf 'Launch:'
printf ' %q' python3 -m sglang.launch_server "${server_args[@]}"
printf '\n'
exec python3 -m sglang.launch_server "${server_args[@]}"
