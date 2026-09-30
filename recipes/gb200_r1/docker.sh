#!/usr/bin/env bash
set -euo pipefail

BENCH_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${BENCH_REPO_ROOT}/recipes/gb200_r1/settings.sh"
action="${1:-help}"
if (( $# )); then shift; fi

docker_cmd=(sudo -n docker)
recipe_label="inferencebench.preparation"
recipe_value="gb200-r1"

require_node() {
    if [[ "$(uname -m)" != aarch64 ]]; then
        echo "Run this wrapper on the ARM64 GB200 node, after the approved source sync." >&2
        exit 2
    fi
    "${docker_cmd[@]}" image inspect "${BENCH_IMAGE}" >/dev/null
}

require_owned_container() {
    local owner
    owner="$("${docker_cmd[@]}" inspect --format '{{index .Config.Labels "inferencebench.preparation"}}' "${BENCH_CONTAINER}")"
    if [[ "${owner}" != "${recipe_value}" ]]; then
        echo "Container ${BENCH_CONTAINER} is not owned by this preparation recipe." >&2
        exit 2
    fi
}

client_args=(run --rm --runtime runc --network host --cpus 2 --memory 4g
    --env PYTHONDONTWRITEBYTECODE=1
    --mount "type=bind,src=${BENCH_REPO_ROOT},dst=/workspace,readonly"
    --workdir /workspace --entrypoint python3)

case "${action}" in
    check)
        require_node
        "${docker_cmd[@]}" run --rm --network none --gpus "\"device=${BENCH_GPUS}\"" \
            --cpus 4 --memory 8g --env PYTHONDONTWRITEBYTECODE=1 \
            --mount "type=bind,src=${BENCH_REPO_ROOT},dst=/workspace,readonly" \
            --mount "type=bind,src=${BENCH_MODEL_DIR},dst=/models/deepseek-r1,readonly" \
            --workdir /workspace --entrypoint python3 "${BENCH_IMAGE}" \
            recipes/gb200_r1/probe.py environment
        ;;
    start)
        require_node
        if [[ "${BENCH_GPUS}" != "0,1,2,3" ]]; then
            echo "This full-R1 recipe reserves all four GPUs (0,1,2,3) on one node." >&2
            exit 2
        fi
        if "${docker_cmd[@]}" container inspect "${BENCH_CONTAINER}" >/dev/null 2>&1; then
            echo "Container ${BENCH_CONTAINER} already exists; inspect it or use the stop action first." >&2
            exit 2
        fi
        "${docker_cmd[@]}" run -d --name "${BENCH_CONTAINER}" \
            --label "${recipe_label}=${recipe_value}" \
            --device /dev/nvidia-caps-imex-channels/channel0 \
            --gpus "\"device=${BENCH_GPUS}\"" --network host --shm-size 32g \
            --cpus "${BENCH_CPUS}" --memory "${BENCH_MEMORY}" \
            --ulimit memlock=-1 --ulimit stack=67108864 \
            --env "BENCH_PORT=${BENCH_PORT}" --env HF_HUB_OFFLINE=1 \
            --env TRANSFORMERS_OFFLINE=1 --env PYTHONDONTWRITEBYTECODE=1 \
            --mount "type=bind,src=${BENCH_REPO_ROOT},dst=/workspace,readonly" \
            --mount "type=bind,src=${BENCH_MODEL_DIR},dst=/models/deepseek-r1,readonly" \
            --workdir /workspace --entrypoint bash "${BENCH_IMAGE}" \
            recipes/gb200_r1/start_server.sh
        ;;
    smoke)
        require_node
        require_owned_container
        "${docker_cmd[@]}" "${client_args[@]}" "${BENCH_IMAGE}" \
            recipes/gb200_r1/probe.py server --server-url "http://127.0.0.1:${BENCH_PORT}" "$@"
        ;;
    logs)
        require_node
        require_owned_container
        "${docker_cmd[@]}" logs "${BENCH_CONTAINER}" "$@"
        ;;
    stop)
        require_node
        require_owned_container
        "${docker_cmd[@]}" stop --time 45 "${BENCH_CONTAINER}"
        "${docker_cmd[@]}" rm "${BENCH_CONTAINER}"
        ;;
    api-models|api-chat)
        require_node
        # Read the external credential JSON from stdin; never put the key in argv or a mount.
        "${docker_cmd[@]}" "${client_args[@]}" -i "${BENCH_IMAGE}" \
            recipes/gb200_r1/probe.py "${action}" "$@"
        ;;
    tests)
        require_node
        "${docker_cmd[@]}" "${client_args[@]}" "${BENCH_IMAGE}" \
            -m pytest -q -p no:cacheprovider tests
        ;;
    *)
        echo "Usage: bash recipes/gb200_r1/docker.sh {check|start|smoke|logs|stop|api-models|api-chat|tests}"
        echo "API actions read external JSON credentials from stdin. api-chat also requires --model."
        [[ "${action}" == help ]]
        ;;
esac
