#!/usr/bin/env bash
# Frozen solver and treatment; only scheduling and metadata collection are added.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../../.." && pwd)"
run_id="${1:?run id required}"
treatment="${2:?plain or skills required}"
[[ "$run_id" =~ ^[a-zA-Z0-9_.-]+$ ]] || exit 2
[[ "$treatment" == plain || "$treatment" == skills ]] || exit 2
image="inferencebench-gb200:20260930"
reference="$repo/results/gb200_r1/20260930-c-r1"
method=gpt-6.1-sol-max
[[ "$treatment" == plain ]] || method+=-skills
result="$repo/results/gb200_r1/$run_id/$method"
[[ ! -e "$result" ]] || { echo "Refusing to reuse $result" >&2; exit 2; }
IFS= read -r credentials
docker=(sudo -n docker)
common=(--network host --shm-size 32g --cpus 32 --memory 768g
    --ulimit memlock=-1 --ulimit stack=67108864
    --mount "type=bind,src=$repo/src,dst=/opt/inferencebench/src,readonly"
    --mount "type=bind,src=$repo/agents,dst=/opt/inferencebench/agents,readonly"
    --mount "type=bind,src=$repo/recipes/gb200_r1,dst=/opt/inferencebench/recipes/gb200_r1,readonly"
    --mount "type=bind,src=$repo/data/model_metadata/deepseek-ai_DeepSeek-R1,dst=/models/deepseek-r1,readonly"
    --env NO_PROXY=localhost,127.0.0.1,::1 --env no_proxy=localhost,127.0.0.1,::1)
entry=(/opt/inferencebench/recipes/gb200_r1/experiment.py agent --model gpt-6.1-sol --hours 2)
if [[ "$treatment" == skills ]]; then
    common+=(--mount "type=bind,src=$repo/results/gb200_r1/20261006-c-skills-preparation/vendor,dst=/opt/ai-infra-skills,readonly")
    entry=(/opt/inferencebench/recipes/gb200_r1/skill_guided/agent.py --model gpt-6.1-sol --hours 2)
fi
gpu=(--gpus all --device /dev/nvidia-caps-imex-channels/channel0)
mkdir -p "$result/task" "$result/codex"
name="ib-${run_id}-sol-${treatment}"
date -u --iso-8601=seconds > "$result/pipeline_started_at.txt"
printf '%s\n' optimizing > "$result/pipeline_stage.txt"
set +e
printf '%s\n' "$credentials" | "${docker[@]}" run -i --name "$name" \
    --label "inferencebench.run=$run_id" "${gpu[@]}" "${common[@]}" \
    --mount "type=bind,src=$reference/shared/requests_21.jsonl,dst=/bench-inputs/requests_21.jsonl,readonly" \
    --mount "type=bind,src=$result/task,dst=/home/agent/task" \
    --mount "type=bind,src=$result/codex,dst=/root/.codex" \
    --entrypoint python3 "$image" "${entry[@]}" > "$result/container.log" 2>&1
agent_rc=$?
unset credentials
set -e
printf '%s\n' "$agent_rc" > "$result/container_exit_code"
printf '%s\n' final_evaluation > "$result/pipeline_stage.txt"
snapshot="inferencebench-final:${run_id}-sol-${treatment}"
"${docker[@]}" commit "$name" "$snapshot" > "$result/image_id.txt"
"${docker[@]}" diff "$name" > "$result/container_diff.txt"
"${docker[@]}" run -d --name "${name}-final" --label "inferencebench.run=$run_id" \
    "${gpu[@]}" "${common[@]}" \
    --mount "type=bind,src=$reference/shared,dst=/bench-inputs,readonly" \
    --mount "type=bind,src=$result/task,dst=/home/agent/task" \
    --entrypoint python3 "$snapshot" /opt/inferencebench/recipes/gb200_r1/experiment.py final \
    > "$result/final_container_id.txt"
"${docker[@]}" exec "${name}-final" python3 /opt/inferencebench/recipes/gb200_r1/repetitions/capture_formal.py \
    > "$result/capture.log" 2>&1 &
capture_pid=$!
"${docker[@]}" wait "${name}-final" > "$result/final_exit_code"
set +e
wait "$capture_pid"
printf '%s\n' "$?" > "$result/capture_exit_code"
set -e
"${docker[@]}" logs "${name}-final" > "$result/final.log" 2>&1
final_rc="$(cat "$result/final_exit_code")"
date -u --iso-8601=seconds > "$result/pipeline_finished_at.txt"
if [[ "$final_rc" == 0 ]]; then
    printf '%s\n' finished > "$result/pipeline_stage.txt"
else
    printf '%s\n' failed > "$result/pipeline_stage.txt"
fi
exit "$final_rc"
