#!/usr/bin/env bash
set -euo pipefail

TASK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
HOST="${HOST:-${INFERENCE_BENCH_SERVER_HOST:-127.0.0.1}}"
PORT="${PORT:-${INFERENCE_BENCH_SERVER_PORT:-30080}}"
DP_SIZE="${SGLANG_DP_SIZE:-1}"
EXTRA_ARGS=()
if [ "${DP_SIZE}" -gt 1 ]; then
    EXTRA_ARGS+=(--enable-dp-attention)
    DEFAULT_RECV_INTERVAL=1
else
    DEFAULT_RECV_INTERVAL=8
fi
if [ -n "${SGLANG_CUDA_GRAPH_BATCH_SIZES:-}" ]; then
    read -r -a GRAPH_BATCH_SIZES <<< "${SGLANG_CUDA_GRAPH_BATCH_SIZES}"
    EXTRA_ARGS+=(--cuda-graph-bs-decode "${GRAPH_BATCH_SIZES[@]}")
fi

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export NCCL_NVLS_ENABLE=0
export TORCH_SYMM_MEM_DISABLE_MULTICAST=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export SGLANG_ENABLE_JIT_DEEPGEMM=0
export SGLANG_SET_CPU_AFFINITY=1
export GLOO_SOCKET_IFNAME=lo
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export PYTHONUNBUFFERED=1
export HF_HOME="${TASK_DIR}/runtime_cache/huggingface"
export TORCHINDUCTOR_CACHE_DIR="${TASK_DIR}/runtime_cache/torchinductor"
export TRITON_CACHE_DIR="${TASK_DIR}/runtime_cache/triton"
export SGLANG_CACHE_DIR="${TASK_DIR}/runtime_cache/sglang"
export FLASHINFER_WORKSPACE_BASE="${TASK_DIR}/runtime_cache/flashinfer"
mkdir -p "${HF_HOME}" "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}" "${SGLANG_CACHE_DIR}" "${FLASHINFER_WORKSPACE_BASE}"

exec python3 -m sglang.launch_server \
    --model-path /models/deepseek-r1 \
    --tokenizer-path /models/deepseek-r1 \
    --served-model-name "${INFERENCE_BENCH_BASE_MODEL:-deepseek-ai/DeepSeek-R1}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --load-format dummy \
    --dtype bfloat16 \
    --kv-cache-dtype bf16 \
    --tp-size 4 \
    --ep-size "${SGLANG_EP_SIZE:-1}" \
    --dp-size "${DP_SIZE}" \
    --moe-a2a-backend none \
    --random-seed "${SGLANG_RANDOM_SEED:-595750618}" \
    --context-length 16384 \
    --mem-fraction-static "${SGLANG_MEM_FRACTION:-0.94}" \
    --max-running-requests "${SGLANG_MAX_RUNNING_REQUESTS:-128}" \
    --chunked-prefill-size "${SGLANG_CHUNKED_PREFILL_SIZE:-8192}" \
    --max-prefill-tokens "${SGLANG_MAX_PREFILL_TOKENS:-8192}" \
    --cuda-graph-max-bs-decode "${SGLANG_CUDA_GRAPH_MAX_BS:-64}" \
    --attention-backend "${SGLANG_ATTENTION_BACKEND:-trtllm_mla}" \
    --moe-runner-backend "${SGLANG_MOE_BACKEND:-flashinfer_trtllm}" \
    --fp8-gemm-backend "${SGLANG_FP8_GEMM_BACKEND:-flashinfer_trtllm}" \
    --stream-interval "${SGLANG_STREAM_INTERVAL:-16}" \
    --scheduler-recv-interval "${SGLANG_SCHEDULER_RECV_INTERVAL:-${DEFAULT_RECV_INTERVAL}}" \
    --decode-log-interval 200 \
    --disable-radix-cache \
    --enforce-disable-flashinfer-allreduce-fusion \
    --disable-custom-all-reduce \
    "${EXTRA_ARGS[@]}" \
    "$@"
