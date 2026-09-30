#!/usr/bin/env bash
# Source this file from the Docker management wrapper on the selected GB200 node.
BENCH_IMAGE="${BENCH_IMAGE:-lmsysorg/sglang@sha256:00c53fe4c31bf22d7b37537f28bbdfd924c02de13cdfb4bff7378c9c34d75ab2}"
BENCH_CONTAINER="${BENCH_CONTAINER:-inferencebench-r1-preparation}"
BENCH_GPUS="${BENCH_GPUS:-0,1,2,3}"
BENCH_PORT="${BENCH_PORT:-30080}"
BENCH_CPUS="${BENCH_CPUS:-32}"
BENCH_MEMORY="${BENCH_MEMORY:-768g}"
BENCH_MODEL_ID="deepseek-ai/DeepSeek-R1"
BENCH_MODEL_DIR="${BENCH_MODEL_DIR:-${BENCH_REPO_ROOT}/data/model_metadata/deepseek-ai_DeepSeek-R1}"
BENCH_MODEL_REVISION="56d4cbbb4d29f4355bab4b9a39ccb717a14ad5ad"
