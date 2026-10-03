#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

HOST="${HOST:-${INFERENCE_BENCH_SERVER_HOST:-127.0.0.1}}"
PORT="${PORT:-${INFERENCE_BENCH_SERVER_PORT:-30080}}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
export NCCL_NVLS_ENABLE=0
export TORCH_SYMM_MEM_DISABLE_MULTICAST=1
export SGLANG_SHARED_EXPERT_TP1="${SGLANG_SHARED_EXPERT_TP1:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM=false
export SGLANG_SET_CPU_AFFINITY="${SGLANG_SET_CPU_AFFINITY:-1}"
export SGLANG_JIT_DEEPGEMM_PRECOMPILE="${SGLANG_JIT_DEEPGEMM_PRECOMPILE:-0}"
export SGLANG_CACHE_DIR="$PWD/runtime_cache/sglang"
export SGLANG_DG_CACHE_DIR="$PWD/runtime_cache/deep_gemm"
export TRITON_CACHE_DIR="$PWD/runtime_cache/triton"
export FLASHINFER_WORKSPACE_BASE="$PWD/runtime_cache"
export CUDA_CACHE_PATH="$PWD/runtime_cache/cuda"
mkdir -p "$SGLANG_CACHE_DIR" "$SGLANG_DG_CACHE_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH"

if [[ "${R1_ENABLE_LOCAL_PREFILL_BATCHER:-1}" == 1 ]]; then
    export R1_ENABLE_LOCAL_PREFILL_BATCHER=1
    export R1_PREFILL_BATCH_MAX_WAIT_STEPS="${R1_PREFILL_BATCH_MAX_WAIT_STEPS:-80}"
    export R1_PREFILL_BATCH_TARGET_DIVISOR="${R1_PREFILL_BATCH_TARGET_DIVISOR:-4}"
    export PYTHONPATH="$PWD/inference_tuning:${PYTHONPATH:-}"
fi

args=(
    --model-path /models/deepseek-r1
    --tokenizer-path /models/deepseek-r1
    --served-model-name deepseek-ai/DeepSeek-R1
    --load-format dummy
    --quantization fp8
    --dtype bfloat16
    --kv-cache-dtype bf16
    --tp-size 4
    --ep-size "${R1_EP_SIZE:-1}"
    --moe-a2a-backend none
    --dp-size "${R1_DP_SIZE:-1}"
    --load-balance-method "${R1_LOAD_BALANCE_METHOD:-auto}"
    --context-length "${R1_CONTEXT_LENGTH:-16384}"
    --mem-fraction-static "${R1_MEM_FRACTION_STATIC:-0.94}"
    --max-total-tokens "${R1_MAX_TOTAL_TOKENS:-147456}"
    --max-running-requests "${R1_MAX_RUNNING_REQUESTS:-64}"
    --chunked-prefill-size "${R1_CHUNKED_PREFILL_SIZE:-8192}"
    --attention-backend "${R1_ATTENTION_BACKEND:-trtllm_mla}"
    --moe-runner-backend "${R1_MOE_RUNNER_BACKEND:-flashinfer_trtllm}"
    --fp8-gemm-backend "${R1_FP8_GEMM_BACKEND:-flashinfer_trtllm}"
    --page-size 64
    --stream-interval "${R1_STREAM_INTERVAL:-16}"
    --num-continuous-decode-steps "${R1_CONTINUOUS_DECODE_STEPS:-1}"
    --disable-radix-cache
    --enforce-disable-flashinfer-allreduce-fusion
    --disable-custom-all-reduce
    --random-seed 42
    --host "$HOST"
    --port "$PORT"
)

if [[ "${R1_ENABLE_CUDA_GRAPH:-1}" == 1 ]]; then
    args+=(--cuda-graph-backend-decode full --cuda-graph-max-bs-decode "${R1_CUDA_GRAPH_MAX_BS:-64}" --cuda-graph-backend-prefill "${R1_PREFILL_GRAPH_BACKEND:-disabled}")
    if [[ -n "${R1_CUDA_GRAPH_BS:-1 2 4 8 16 24 32 48 64}" ]]; then
        read -r -a graph_sizes <<< "${R1_CUDA_GRAPH_BS:-1 2 4 8 16 24 32 48 64}"
        args+=(--cuda-graph-bs-decode "${graph_sizes[@]}")
    fi
else
    args+=(--disable-cuda-graph)
fi

if [[ "${R1_ENABLE_OVERLAP:-1}" != 1 ]]; then
    args+=(--disable-overlap-schedule)
fi

if [[ "${R1_ENABLE_MIXED_CHUNK:-0}" == 1 ]]; then
    args+=(--enable-mixed-chunk)
fi

if [[ "${R1_ENABLE_DP_ATTENTION:-0}" == 1 ]]; then
    args+=(--enable-dp-attention --enable-dp-attention-local-control-broadcast)
fi

if [[ "${R1_ENABLE_DP_LM_HEAD:-0}" == 1 ]]; then
    args+=(--enable-dp-lm-head)
fi

if [[ "${R1_ENABLE_PREFILL_DELAYER:-0}" == 1 ]]; then
    args+=(
        --enable-prefill-delayer
        --prefill-delayer-queue-min-ratio "${R1_PREFILL_QUEUE_MIN_RATIO:-0.25}"
        --prefill-delayer-max-delay-passes "${R1_PREFILL_MAX_DELAY_PASSES:-30}"
        --prefill-delayer-max-delay-ms "${R1_PREFILL_MAX_DELAY_MS:-400}"
    )
fi

if [[ -n "${R1_MIN_FREE_SLOTS_DELAY:-}" ]]; then
    args+=(--min-free-slots-delay "$R1_MIN_FREE_SLOTS_DELAY")
fi

printf 'Launching full DeepSeek-R1:'
printf ' %q' python3 -m sglang.launch_server "${args[@]}"
printf '\n'
exec python3 -m sglang.launch_server "${args[@]}"
