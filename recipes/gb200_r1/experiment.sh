#!/usr/bin/env bash
# Host-side Docker management only. All Python, tests, agents, and GPU work run in containers.
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
image="${INFERENCE_BENCH_IMAGE:-inferencebench-gb200:20260930}"
action="${1:?action required}"
shift
run_id="${INFERENCE_BENCH_RUN_ID:-20260930-c-r1}"
[[ "$run_id" =~ ^[a-zA-Z0-9_.-]+$ ]] || exit 2
result_root="$repo/results/gb200_r1/$run_id"
model_dir="$repo/data/model_metadata/deepseek-ai_DeepSeek-R1"
docker=(sudo -n docker)
gpu_args=(--gpus all --device /dev/nvidia-caps-imex-channels/channel0)
proxy_args=(--env "HTTP_PROXY=${HTTP_PROXY:-}" --env "HTTPS_PROXY=${HTTPS_PROXY:-}" --env NO_PROXY=localhost,127.0.0.1,::1)
common=(--network host --shm-size 32g --cpus 32 --memory 768g --ulimit memlock=-1 --ulimit stack=67108864
        --mount "type=bind,src=$repo/src,dst=/opt/inferencebench/src,readonly"
        --mount "type=bind,src=$repo/agents,dst=/opt/inferencebench/agents,readonly"
        --mount "type=bind,src=$repo/recipes/gb200_r1,dst=/opt/inferencebench/recipes/gb200_r1,readonly"
        --mount "type=bind,src=$model_dir,dst=/models/deepseek-r1,readonly" "${proxy_args[@]}")
mkdir -p "$result_root/shared"

case "$action" in
    build)
        "${docker[@]}" build --network host --build-arg "HTTP_PROXY=${HTTP_PROXY:-}" --build-arg "HTTPS_PROXY=${HTTPS_PROXY:-}" \
            -f "$repo/recipes/gb200_r1/Dockerfile" -t "$image" "$repo"
        ;;
    prepare)
        "${docker[@]}" run --rm --runtime runc "${common[@]}" \
            --mount "type=bind,src=$result_root/shared,dst=/bench-inputs" "$image" prepare
        ;;
    tests)
        "${docker[@]}" run --rm --runtime runc "${common[@]}" \
            --mount "type=bind,src=$repo/tests,dst=/opt/inferencebench/tests,readonly" \
            --entrypoint python3 "$image" -m pytest -q -p no:cacheprovider tests
        ;;
    baseline)
        mkdir -p "$result_root/preflight"
        "${docker[@]}" run --rm --runtime runc "${common[@]}" \
            --mount "type=bind,src=$result_root/shared,dst=/bench-inputs,readonly" \
            --mount "type=bind,src=$result_root/preflight,dst=/home/agent/task" "$image" baseline
        ;;
    pipeline)
        if [[ ! -f "$result_root/preflight/baseline_summary.json" ]]; then
            echo "Full scenario C baseline preflight has not passed; refusing to start optimization timers." >&2
            exit 2
        fi
        "${docker[@]}" run --rm --runtime runc --network none \
            --mount "type=bind,src=$result_root/preflight,dst=/preflight,readonly" \
            --entrypoint python3 "$image" -c \
            'import json; d=json.load(open("/preflight/baseline_summary.json")); assert d.get("performance_passed") is True; assert set(d["profiles"]) == {"burst","poisson","constant"}; assert all(p["success_count"] == p["request_count"] == 256 for p in d["profiles"].values())'
        agent_model="${1:?agent model required}"
        method="${2:?mechanical method required}"
        [[ "$agent_model" == gpt-6.1-sol || "$agent_model" == gpt-6-astra ]] || exit 2
        [[ "$method" == random || "$method" == smac ]] || exit 2
        # The caller supplies one JSON line over SSH stdin; never save it to a file or image config.
        IFS= read -r credentials
        agent_dir="$result_root/${agent_model}-max"
        hpo_dir="$result_root/$method"
        mkdir -p "$agent_dir/task" "$agent_dir/codex" "$hpo_dir/task"
        name="ib-${run_id}-${agent_model}"
        printf '%s\n' agent > "$result_root/pipeline_stage.txt"
        set +e
        printf '%s\n' "$credentials" | "${docker[@]}" run -i --name "$name" --label "inferencebench.run=$run_id" \
            "${gpu_args[@]}" "${common[@]}" \
            --mount "type=bind,src=$result_root/shared/requests_21.jsonl,dst=/bench-inputs/requests_21.jsonl,readonly" \
            --mount "type=bind,src=$agent_dir/task,dst=/home/agent/task" \
            --mount "type=bind,src=$agent_dir/codex,dst=/root/.codex" \
            "$image" agent --model "$agent_model" > "$agent_dir/container.log" 2>&1
        agent_rc=$?
        unset credentials
        set -e
        printf '%s\n' "$agent_rc" > "$agent_dir/container_exit_code"
        printf '%s\n' final_evaluation > "$result_root/pipeline_stage.txt"
        snapshot="inferencebench-final:${run_id}-${agent_model}"
        # The API key only existed in the agent process, so docker commit cannot capture it in Config.Env.
        "${docker[@]}" commit "$name" "$snapshot" > "$agent_dir/image_id.txt"
        set +e
        "${docker[@]}" run --name "${name}-final" --label "inferencebench.run=$run_id" "${gpu_args[@]}" "${common[@]}" \
            --mount "type=bind,src=$result_root/shared,dst=/bench-inputs,readonly" \
            --mount "type=bind,src=$agent_dir/task,dst=/home/agent/task" \
            "$snapshot" final > "$agent_dir/final.log" 2>&1
        final_rc=$?
        set -e
        printf '%s\n' "$final_rc" > "$agent_dir/final_exit_code"
        printf '%s\n' "$method" > "$result_root/pipeline_stage.txt"
        set +e
        "${docker[@]}" run --name "ib-${run_id}-${method}" --label "inferencebench.run=$run_id" "${gpu_args[@]}" "${common[@]}" \
            --mount "type=bind,src=$result_root/shared,dst=/bench-inputs,readonly" \
            --mount "type=bind,src=$hpo_dir/task,dst=/home/agent/task" \
            "$image" hpo --method "$method" > "$hpo_dir/container.log" 2>&1
        hpo_rc=$?
        set -e
        printf '%s\n' "$hpo_rc" > "$hpo_dir/container_exit_code"
        printf '%s\n' finished > "$result_root/pipeline_stage.txt"
        ;;
    status)
        "${docker[@]}" ps -a --filter "label=inferencebench.run=$run_id" --format '{{.Names}}\t{{.Status}}'
        if [[ -f "$result_root/pipeline_stage.txt" ]]; then cat "$result_root/pipeline_stage.txt"; fi
        ;;
    *) echo "Usage: $0 {build|prepare|tests|baseline|pipeline MODEL METHOD|status}" >&2; exit 2 ;;
esac
