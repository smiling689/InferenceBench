#!/usr/bin/env bash
set -euo pipefail
cd /home/agent/task

# Full 61-layer DeepSeek-R1, with dummy FP8 weights and BF16 state.
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export NCCL_NVLS_ENABLE=0
export TORCH_SYMM_MEM_DISABLE_MULTICAST=1
export SGLANG_SHARED_EXPERT_TP1=0
export SGLANG_ENABLE_JIT_DEEPGEMM=0
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1
export FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=/home/agent/task/runtime_cache/fa4
python3 /home/agent/task/prepare_graph_overlay.py
export PYTHONPATH="/home/agent/task/graph_overlay${PYTHONPATH:+:$PYTHONPATH}"
export SGLANG_CACHE_DIR=/home/agent/task/runtime_cache/sglang
export TRITON_CACHE_DIR=/home/agent/task/runtime_cache/triton
export TORCHINDUCTOR_CACHE_DIR=/home/agent/task/runtime_cache/torchinductor
export FLASHINFER_WORKSPACE_BASE=/home/agent/task/runtime_cache
export SGLANG_DG_CACHE_DIR=/home/agent/task/runtime_cache/deep_gemm
export CUDA_CACHE_PATH=/home/agent/task/runtime_cache/cuda
mkdir -p "$SGLANG_CACHE_DIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$CUDA_CACHE_PATH"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-30080}"

exec python3 -m sglang.launch_server \
  --model-path /models/deepseek-r1 \
  --tokenizer-path /models/deepseek-r1 \
  --served-model-name deepseek-ai/DeepSeek-R1 \
  --load-format dummy \
  --dtype bfloat16 \
  --kv-cache-dtype bf16 \
  --host "$HOST" --port "$PORT" \
  --tp-size 4 \
  --context-length 16384 \
  --mem-fraction-static 0.95 \
  --max-total-tokens 131072 \
  --max-running-requests 128 \
  --chunked-prefill-size 2048 \
  --prefill-max-requests 2 \
  --cuda-graph-max-bs-decode 64 \
  --cuda-graph-bs-decode 1 2 4 8 12 16 24 32 40 48 56 64 \
  --cuda-graph-backend-prefill breakable \
  --cuda-graph-bs-prefill 64 1024 2048 \
  --cuda-graph-max-bs-prefill 2048 \
  --cuda-graph-tc-compiler eager \
  --enable-return-hidden-states \
  --attention-backend trtllm_mla \
  --prefill-attention-backend fa4 \
  --moe-runner-backend flashinfer_trtllm \
  --fp8-gemm-backend flashinfer_trtllm \
  --speculative-algorithm EAGLE \
  --speculative-draft-load-format dummy \
  --speculative-num-steps 11 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 12 \
  --speculative-use-rejection-sampling \
  --speculative-attention-mode decode \
  --scheduler-recv-interval 4 \
  --stream-interval 8 \
  --decode-log-interval 200 \
  --random-seed 1 \
  --enforce-disable-flashinfer-allreduce-fusion \
  --disable-custom-all-reduce
