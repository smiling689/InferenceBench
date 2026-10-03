#!/usr/bin/env bash
set -euo pipefail

cd /home/agent/task

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export SGLANG_SHARED_EXPERT_TP1=0
export PYTHONDONTWRITEBYTECODE=1
export NCCL_NVLS_ENABLE=0
export TORCH_SYMM_MEM_DISABLE_MULTICAST=1
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export MALLOC_ARENA_MAX=2
export GLOO_SOCKET_IFNAME=lo
export NCCL_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1
export SGLANG_JIT_DEEPGEMM_FAST_WARMUP=1
export SGLANG_CACHE_DIR=/home/agent/task/runtime_cache/sglang
export SGLANG_DG_CACHE_DIR=/home/agent/task/runtime_cache/deep_gemm
export TRITON_CACHE_DIR=/home/agent/task/runtime_cache/triton
export TORCH_EXTENSIONS_DIR=/home/agent/task/runtime_cache/torch_extensions
export FLASHINFER_WORKSPACE_BASE=/home/agent/task/runtime_cache

HOST="${HOST:-${INFERENCE_BENCH_SERVER_HOST:-127.0.0.1}}"
PORT="${PORT:-${INFERENCE_BENCH_SERVER_PORT:-30080}}"
GRAPH_MAX_BS="${R1_GRAPH_MAX_BS:-64}"
mapfile -t GRAPH_BATCHES < <(seq 1 "$GRAPH_MAX_BS")
PARALLEL_ARGS=()
RECV_INTERVAL="${R1_RECV_INTERVAL:-8}"
DP_SIZE="${R1_DP_SIZE:-1}"
PREFILL_GRAPH="${R1_PREFILL_GRAPH:-tc_piecewise}"
if (( DP_SIZE > 1 )); then
    RECV_INTERVAL=1
    PREFILL_GRAPH="${R1_PREFILL_GRAPH:-disabled}"
    PARALLEL_ARGS+=(
        --dp-size "$DP_SIZE"
        --enable-dp-attention
        --enable-dp-attention-local-control-broadcast
        --load-balance-method total_requests
        --load-snapshot-publish-interval 4
    )
    if [[ "${R1_DP_LM_HEAD:-0}" == 1 ]]; then
        PARALLEL_ARGS+=(--enable-dp-lm-head)
    fi
    if [[ "${R1_DP_PREFILL_DELAY:-0}" == 1 ]]; then
        PARALLEL_ARGS+=(
            --enable-prefill-delayer
            --prefill-delayer-queue-min-ratio 0.1
            --prefill-delayer-max-delay-ms 50
            --prefill-delayer-max-delay-passes 8
        )
    fi
fi

exec python3 -m sglang.launch_server \
    --model-path /models/deepseek-r1 \
    --tokenizer-path /models/deepseek-r1 \
    --served-model-name deepseek-ai/DeepSeek-R1 \
    --load-format dummy \
    --quantization fp8 \
    --dtype bfloat16 \
    --kv-cache-dtype bf16 \
    --tp-size 4 \
    --ep-size "${R1_EP_SIZE:-1}" \
    "${PARALLEL_ARGS[@]}" \
    --context-length 16384 \
    --mem-fraction-static "${R1_MEM_FRACTION:-0.94}" \
    --max-total-tokens "${R1_KV_TOKENS:-147456}" \
    --max-running-requests "${R1_MAX_RUNNING:-64}" \
    --chunked-prefill-size "${R1_PREFILL_CHUNK:-4096}" \
    --attention-backend "${R1_ATTENTION_BACKEND:-trtllm_mla}" \
    --moe-runner-backend flashinfer_trtllm \
    --page-size 64 \
    --cuda-graph-backend-decode full \
    --cuda-graph-backend-prefill "$PREFILL_GRAPH" \
    --cuda-graph-bs-decode "${GRAPH_BATCHES[@]}" \
    --cuda-graph-bs-prefill 1024 2048 4096 \
    --cuda-graph-tc-compiler eager \
    --disable-radix-cache \
    --stream-interval "${R1_STREAM_INTERVAL:-16}" \
    --scheduler-recv-interval "$RECV_INTERVAL" \
    --schedule-conservativeness "${R1_SCHEDULE_CONSERVATIVENESS:-1.0}" \
    --enforce-disable-flashinfer-allreduce-fusion \
    --disable-custom-all-reduce \
    --random-seed 42 \
    --host "$HOST" \
    --port "$PORT"
