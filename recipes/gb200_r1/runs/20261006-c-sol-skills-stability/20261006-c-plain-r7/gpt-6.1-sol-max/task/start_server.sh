#!/usr/bin/env bash
set -euo pipefail

TASK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="/models/deepseek-r1"
HOST="${HOST:-${INFERENCE_BENCH_SERVER_HOST:-127.0.0.1}}"
PORT="${PORT:-${INFERENCE_BENCH_SERVER_PORT:-30080}}"
SPEC_STEPS="${SPEC_STEPS:-7}"
CUDA_GRAPH_BATCHES="${CUDA_GRAPH_BATCHES:-1 2 4 8 16 32 48 64}"
MIN_FREE_SLOTS_DELAY="${MIN_FREE_SLOTS_DELAY:-4}"
PREFILL_QUEUE_MIN_RATIO="${PREFILL_QUEUE_MIN_RATIO:-0.25}"

export NCCL_NVLS_ENABLE=0
export TORCH_SYMM_MEM_DISABLE_MULTICAST=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export SGLANG_SHARED_EXPERT_TP1=0
export SGLANG_SET_CPU_AFFINITY=1
export GLOO_SOCKET_IFNAME=lo
export NCCL_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1
export SGLANG_JIT_DEEPGEMM_FAST_WARMUP=1
export CUDA_CACHE_PATH="${TASK_DIR}/runtime_cache/cuda"
export TRITON_CACHE_DIR="${TASK_DIR}/runtime_cache/triton"
export TORCHINDUCTOR_CACHE_DIR="${TASK_DIR}/runtime_cache/torchinductor"
export SGLANG_CACHE_DIR="${TASK_DIR}/runtime_cache/sglang"
export SGLANG_DG_CACHE_DIR="${TASK_DIR}/runtime_cache/deep_gemm"
export FLASHINFER_WORKSPACE_BASE="${TASK_DIR}/runtime_cache"
export FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1
export FLASH_ATTENTION_CUTE_DSL_CACHE_DIR="${TASK_DIR}/runtime_cache/fa4"
mkdir -p "${CUDA_CACHE_PATH}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${SGLANG_CACHE_DIR}" "${SGLANG_DG_CACHE_DIR}"

EXTRA_ARGS=()
if [[ -n "${CUDA_GRAPH_BATCHES:-}" ]]; then
    read -r -a DECODE_GRAPH_BATCHES <<< "${CUDA_GRAPH_BATCHES}"
    EXTRA_ARGS+=(--cuda-graph-bs-decode "${DECODE_GRAPH_BATCHES[@]}")
fi
if [[ -n "${PREFILL_ATTENTION_BACKEND:-}" ]]; then
    EXTRA_ARGS+=(--prefill-attention-backend "${PREFILL_ATTENTION_BACKEND}")
fi
if [[ -n "${PREFILL_GRAPH_BACKEND:-}" ]]; then
    EXTRA_ARGS+=(
        --cuda-graph-backend-prefill "${PREFILL_GRAPH_BACKEND}"
        --cuda-graph-max-bs-prefill "${PREFILL_GRAPH_MAX_TOKENS:-4096}"
        --cuda-graph-tc-compiler "${PREFILL_GRAPH_COMPILER:-eager}"
    )
    if [[ -n "${PREFILL_GRAPH_BUCKETS:-}" ]]; then
        read -r -a PREFILL_GRAPH_TOKEN_BUCKETS <<< "${PREFILL_GRAPH_BUCKETS}"
        EXTRA_ARGS+=(--cuda-graph-bs-prefill "${PREFILL_GRAPH_TOKEN_BUCKETS[@]}")
    fi
fi
if [[ "${PREFILL_QUEUE_MIN_RATIO}" != 0 ]]; then
    EXTRA_ARGS+=(
        --enable-prefill-delayer
        --prefill-delayer-queue-min-ratio "${PREFILL_QUEUE_MIN_RATIO}"
        --prefill-delayer-max-delay-ms "${PREFILL_MAX_DELAY_MS:-1000}"
        --prefill-delayer-max-delay-passes "${PREFILL_MAX_DELAY_PASSES:-32}"
    )
fi
if [[ "${DP_SIZE:-1}" != 1 ]]; then
    EXTRA_ARGS+=(--enable-dp-attention --dp-size "${DP_SIZE}" --enable-dp-lm-head)
fi
if [[ "${MIXED_CHUNK:-0}" == 1 ]]; then
    EXTRA_ARGS+=(--enable-mixed-chunk)
fi
if [[ "${DISABLE_RADIX_CACHE:-0}" == 1 ]]; then
    EXTRA_ARGS+=(--disable-radix-cache)
fi
if [[ -n "${MIN_FREE_SLOTS_DELAY:-}" ]]; then
    EXTRA_ARGS+=(--min-free-slots-delay "${MIN_FREE_SLOTS_DELAY}")
fi
if [[ "${SPEC_STEPS:-0}" != 0 ]]; then
    EXTRA_ARGS+=(
        --speculative-algorithm EAGLE
        --speculative-draft-model-path "${MODEL_PATH}"
        --speculative-draft-load-format dummy
        --speculative-draft-model-quantization fp8
        --speculative-moe-runner-backend "${MOE_RUNNER_BACKEND:-flashinfer_trtllm}"
        --speculative-eagle-topk 1
        --speculative-num-steps "${SPEC_STEPS}"
        --speculative-num-draft-tokens "${SPEC_DRAFT_TOKENS:-$((SPEC_STEPS + 1))}"
        --speculative-attention-mode decode
        --speculative-use-rejection-sampling
    )
    if [[ -n "${SPEC_ADAPTIVE_CONFIG:-}" ]]; then
        EXTRA_ARGS+=(--speculative-adaptive --speculative-adaptive-config "${SPEC_ADAPTIVE_CONFIG}")
    fi
fi

exec python3 -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --tokenizer-path "${MODEL_PATH}" \
    --served-model-name deepseek-ai/DeepSeek-R1 \
    --host "${HOST}" \
    --port "${PORT}" \
    --load-format dummy \
    --dtype bfloat16 \
    --quantization fp8 \
    --kv-cache-dtype bf16 \
    --tp-size 4 \
    --ep-size "${EP_SIZE:-1}" \
    --context-length 16384 \
    --mem-fraction-static "${MEM_FRACTION_STATIC:-0.94}" \
    --max-running-requests "${MAX_RUNNING_REQUESTS:-64}" \
    --chunked-prefill-size "${CHUNKED_PREFILL_SIZE:-8192}" \
    --fp8-gemm-backend "${FP8_GEMM_BACKEND:-auto}" \
    --moe-runner-backend "${MOE_RUNNER_BACKEND:-flashinfer_trtllm}" \
    --attention-backend "${ATTENTION_BACKEND:-trtllm_mla}" \
    --cuda-graph-max-bs-decode "${CUDA_GRAPH_MAX_BS:-64}" \
    --stream-interval "${STREAM_INTERVAL:-8}" \
    --sampling-defaults "${SAMPLING_DEFAULTS:-openai}" \
    --enforce-disable-flashinfer-allreduce-fusion \
    --disable-custom-all-reduce \
    --random-seed 42 \
    --log-level info \
    "${EXTRA_ARGS[@]}" \
    "$@"
