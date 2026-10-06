#!/usr/bin/env bash
# Host-side scheduling and Docker management; no project workloads on the host.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../../.." && pwd)"
node="${1:?node required}"
[[ "$node" == gb200-1 || "$node" == gb200-2 ]] || exit 2
batch="$repo/results/gb200_r1/20261006-c-sol-skills-stability/$node"
mkdir -p "$batch"
exec 9>"$batch/queue.lock"
flock -n 9 || { echo "Queue already running" >&2; exit 2; }
[[ ! -e "$batch/started_at.txt" ]] || { echo "Refusing to replay queue" >&2; exit 2; }
IFS= read -r credentials
date -u --iso-8601=seconds > "$batch/started_at.txt"
printf '%s\n' running > "$batch/stage.txt"
while IFS=$'\t' read -r assigned run_id treatment; do
    [[ "$assigned" == "$node" ]] || continue
    printf '%s\n' "$run_id" > "$batch/current_run.txt"
    printf '[queue] %s starting %s %s\n' "$(date -u --iso-8601=seconds)" "$run_id" "$treatment"
    set +e
    printf '%s\n' "$credentials" | bash "$here/run.sh" "$run_id" "$treatment"
    rc=$?
    set -e
    printf '%s\t%s\t%s\n' "$run_id" "$treatment" "$rc" >> "$batch/completed.tsv"
    printf '[queue] %s completed %s rc=%s\n' "$(date -u --iso-8601=seconds)" "$run_id" "$rc"
done < "$here/schedule.tsv"
unset credentials
date -u --iso-8601=seconds > "$batch/finished_at.txt"
printf '%s\n' finished > "$batch/stage.txt"
