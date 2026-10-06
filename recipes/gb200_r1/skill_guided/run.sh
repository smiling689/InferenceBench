#!/usr/bin/env bash
# Node host: Docker management only. Project Python runs in containers.
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
action="${1:?action required}"
run_id="${2:?independent run id required}"
[[ "$run_id" =~ ^[a-zA-Z0-9_.-]+$ ]] || exit 2
image="inferencebench-gb200:20260930"
vendor="$repo/results/gb200_r1/20261006-c-skills-preparation/vendor"
reference="$repo/results/gb200_r1/20260930-c-r1"
result="$repo/results/gb200_r1/$run_id/gpt-6.1-sol-max-skills"
docker=(sudo -n docker)
common=(--network host --shm-size 32g --cpus 32 --memory 768g
    --ulimit memlock=-1 --ulimit stack=67108864
    --mount "type=bind,src=$repo/src,dst=/opt/inferencebench/src,readonly"
    --mount "type=bind,src=$repo/agents,dst=/opt/inferencebench/agents,readonly"
    --mount "type=bind,src=$repo/recipes/gb200_r1,dst=/opt/inferencebench/recipes/gb200_r1,readonly"
    --mount "type=bind,src=$vendor,dst=/opt/ai-infra-skills,readonly"
    --mount "type=bind,src=$repo/data/model_metadata/deepseek-ai_DeepSeek-R1,dst=/models/deepseek-r1,readonly"
    --env NO_PROXY=localhost,127.0.0.1,::1 --env no_proxy=localhost,127.0.0.1,::1)
gpu=(--gpus all --device /dev/nvidia-caps-imex-channels/channel0)
case "$action" in
    inspect)
        "${docker[@]}" run --rm --runtime runc "${common[@]}" \
            --mount "type=bind,src=$repo,dst=/frozen-reference,readonly" --entrypoint python3 "$image" \
            /opt/inferencebench/recipes/gb200_r1/skill_guided/preflight.py
        ;;
    run)
        # A fresh task and Codex home are required; never resume another run.
        [[ ! -e "$result" ]] || { echo "Refusing to reuse $result" >&2; exit 2; }
        bash "${BASH_SOURCE[0]}" inspect "$run_id"
        [[ -f "$reference/preflight/baseline_summary.json" ]] || exit 2
        "${docker[@]}" run --rm --runtime runc --network none \
            --mount "type=bind,src=$reference/preflight,dst=/preflight,readonly" \
            --entrypoint python3 "$image" -c \
            'import json; d=json.load(open("/preflight/baseline_summary.json")); assert d["performance_passed"] is True; assert set(d["profiles"])=={"burst","poisson","constant"}; assert all(p["request_count"]==p["success_count"]==256 for p in d["profiles"].values())'
        mkdir -p "$result/task" "$result/codex"
        IFS= read -r credentials
        name="ib-${run_id}-sol-skills"
        date -u --iso-8601=seconds > "$result/pipeline_started_at.txt"
        printf '%s\n' optimizing > "$result/pipeline_stage.txt"
        set +e
        printf '%s\n' "$credentials" | "${docker[@]}" run -i --name "$name" \
            --label "inferencebench.run=$run_id" "${gpu[@]}" "${common[@]}" \
            --mount "type=bind,src=$reference/shared/requests_21.jsonl,dst=/bench-inputs/requests_21.jsonl,readonly" \
            --mount "type=bind,src=$result/task,dst=/home/agent/task" \
            --mount "type=bind,src=$result/codex,dst=/root/.codex" \
            --entrypoint python3 "$image" /opt/inferencebench/recipes/gb200_r1/skill_guided/agent.py \
            --model gpt-6.1-sol --hours 2 > "$result/container.log" 2>&1
        agent_rc=$?
        unset credentials
        set -e
        printf '%s\n' "$agent_rc" > "$result/container_exit_code"
        printf '%s\n' final_evaluation > "$result/pipeline_stage.txt"
        snapshot="inferencebench-final:${run_id}-sol-skills"
        "${docker[@]}" commit "$name" "$snapshot" > "$result/image_id.txt"
        "${docker[@]}" diff "$name" > "$result/container_diff.txt"
        set +e
        "${docker[@]}" run --name "${name}-final" --label "inferencebench.run=$run_id" \
            "${gpu[@]}" "${common[@]}" \
            --mount "type=bind,src=$reference/shared,dst=/bench-inputs,readonly" \
            --mount "type=bind,src=$result/task,dst=/home/agent/task" \
            --entrypoint python3 "$snapshot" /opt/inferencebench/recipes/gb200_r1/experiment.py final \
            > "$result/final.log" 2>&1
        final_rc=$?
        set -e
        printf '%s\n' "$final_rc" > "$result/final_exit_code"
        date -u --iso-8601=seconds > "$result/pipeline_finished_at.txt"
        if (( final_rc == 0 )); then
            printf '%s\n' finished > "$result/pipeline_stage.txt"
        else
            printf '%s\n' failed > "$result/pipeline_stage.txt"
        fi
        exit "$final_rc"
        ;;
    *) echo "Usage: $0 {inspect|run} RUN_ID" >&2; exit 2 ;;
esac
