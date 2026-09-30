#!/usr/bin/env bash
set -euo pipefail
cd /home/agent/task

HOST="${HOST:-${INFERENCE_BENCH_SERVER_HOST:-127.0.0.1}}"
PORT="${PORT:-${INFERENCE_BENCH_SERVER_PORT:-30080}}"

export NCCL_NVLS_ENABLE=0
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export TORCH_SYMM_MEM_DISABLE_MULTICAST=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export SGLANG_ENABLE_JIT_DEEPGEMM=false
export SGLANG_CACHE_DIR=/home/agent/task/runtime_cache/sglang
export TRITON_CACHE_DIR=/home/agent/task/runtime_cache/triton
export TORCHINDUCTOR_CACHE_DIR=/home/agent/task/runtime_cache/torchinductor
export CUDA_CACHE_PATH=/home/agent/task/runtime_cache/cuda
export FLASHINFER_WORKSPACE_BASE=/home/agent/task/runtime_cache/flashinfer
export TVM_FFI_CACHE_DIR=/home/agent/task/runtime_cache/tvm-ffi
mkdir -p "$SGLANG_CACHE_DIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" \
    "$CUDA_CACHE_PATH" "$FLASHINFER_WORKSPACE_BASE" "$TVM_FFI_CACHE_DIR"
export PIP_NO_INDEX=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
python3 -m pip install --no-deps --no-build-isolation -e /home/agent/task/diagnostics_plugin
export SGLANG_PLUGINS=r1_sampling_fastpaths
unset R1_DIAGNOSTICS_ENABLED SGLANG_ENABLE_ASYNC_ASSERT SGLANG_SANITIZE_NAN_LOGITS
unset SGLANG_SIMULATE_ACC_LEN SGLANG_SIMULATE_ACC_METHOD SGLANG_SIMULATE_ACC_TOKEN_MODE

exec python3 -m sglang.launch_server \
    --model-path /models/deepseek-r1 \
    --tokenizer-path /models/deepseek-r1 \
    --served-model-name deepseek-ai/DeepSeek-R1 \
    --host "$HOST" \
    --port "$PORT" \
    --load-format dummy \
    --sampling-defaults openai \
    --dtype bfloat16 \
    --kv-cache-dtype bf16 \
    --tp-size 4 \
    --random-seed 42 \
    --context-length 4096 \
    --attention-backend trtllm_mla \
    --moe-runner-backend flashinfer_trtllm \
    --fp8-gemm-backend flashinfer_trtllm \
    --mem-fraction-static 0.94 \
    --max-total-tokens 92160 \
    --max-running-requests 40 \
    --chunked-prefill-size 8192 \
    --max-prefill-tokens 8192 \
    --cuda-graph-max-bs-decode 40 \
    --cuda-graph-bs-decode 1 2 4 8 16 32 40 \
    --scheduler-recv-interval 8 \
    --disable-prefill-cuda-graph \
    --disable-radix-cache \
    --stream-interval 32 \
    --speculative-algorithm NEXTN \
    --speculative-draft-load-format dummy \
    --speculative-num-steps 31 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 32 \
    --speculative-use-rejection-sampling \
    --enforce-disable-flashinfer-allreduce-fusion \
    --disable-custom-all-reduce \
    --watchdog-timeout 900
