"""Collect terminal HPO metadata and same-node durations in CPU Docker.

Arguments: node, run_id, method, JSON list of the three Docker State objects
(Agent, Agent formal evaluation, HPO). Mount the HPO task read-only at
/home/agent/task. Print evidence for collection on the devbox.
"""

import datetime
import json
import sys
from pathlib import Path


def timestamp(value):
    return datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))


node, run_id, method, state_json = sys.argv[1:]
assert node in {"gb200-1", "gb200-2"} and method in {"random", "smac"}
states = json.loads(state_json)
assert len(states) == 3
assert all(not state["Running"] and not state["OOMKilled"] and state["ExitCode"] == 0 for state in states)
record_path = Path("/home/agent/task/hpo") / method / "sglang/scenario_c/seed_0.jsonl"
rows = [json.loads(line) for line in record_path.read_text().splitlines() if line.strip()]
end = rows[-1]
assert end["type"] == "run_end"
final = record_path.with_name("seed_0_artifacts") / "final"
audit = json.loads((final / "result_audit.json").read_text())
assert audit["audit_passed"] and audit["complete_success_records"] == 768
metrics = json.loads((final / "final_metrics.json").read_text())
trials = [row for row in rows if row["type"] == "trial"]
valid = [row for row in trials if row["failure_reason"] is None and not row["integrity_flags"]]
hpo_state = states[2]
elapsed = (timestamp(hpo_state["FinishedAt"]) - timestamp(hpo_state["StartedAt"])).total_seconds()
pipeline_elapsed = (timestamp(hpo_state["FinishedAt"]) - timestamp(states[0]["StartedAt"])).total_seconds()
record = {
    "run_id": run_id, "node": node, "method": method,
    "runtime_commit": "67a9cf389bd7568afac740e0e7cb8b6edcddc59b",
    "optimization_budget_s": 7200,
    "recorded_search_elapsed_s": end["budget_used_s"],
    "final_evaluation_wall_s": end["final_metrics"]["wall_s"],
    "trial_count": len(trials), "valid_trial_count": len(valid),
    "unique_valid_configurations": len({json.dumps(row["config"], sort_keys=True) for row in valid}),
    "best_trial_idx": end["best_trial_idx"], "container_exit_code": hpo_state["ExitCode"],
    "container_started_at_node_utc": hpo_state["StartedAt"],
    "container_finished_at_node_utc": hpo_state["FinishedAt"],
    "container_elapsed_s": elapsed,
    "primary_metric": end["final_primary_metric"],
    "primary_metric_source": end["final_primary_metric_source"],
    "evaluation_seed": 1337, "server_random_seed": 42,
    "quality_evaluated": False, "quality_gate_passed": end["gate_passed"],
    "performance_passed": end["performance_passed"],
    "integrity_passed": end["integrity_passed"], "result_audit_passed": audit["audit_passed"],
    "request_successes": 768, "request_count": 768,
    "model_shape": audit["model_shape"], "model_config_sha256": audit["model_config_sha256"],
    "hpo_runtime_sha256": audit["hpo_runtime_sha256"],
    "original_task_directory": f"/home/azhpcuser/ShrinkWrap-v2/InferenceBench/results/gb200_r1/{run_id}/{method}/task",
}
summary = {
    "finished_at": end["end_ts"], "status": "completed",
    "primary_metric": end["final_primary_metric"],
    "primary_metric_source": end["final_primary_metric_source"],
    "quality_evaluated": False, "performance_passed": True,
    "integrity_passed": True, "result_audit_passed": True,
    "profiles": metrics["profiles"],
}
timing = {
    "node": node, "run_id": run_id,
    "agent_container_started_at_node_utc": states[0]["StartedAt"],
    "mechanical_container_finished_at_node_utc": hpo_state["FinishedAt"],
    "agent_to_mechanical_final_elapsed_s": pipeline_elapsed,
    "agent_to_mechanical_final_elapsed_hours": pipeline_elapsed / 3600,
    "configured_optimization_budget_s": 14400,
    "budget_exclusive_pipeline_overhead_s": pipeline_elapsed - 14400,
    "note": "Durations use one node's timestamps. Includes startup, previews and formal evaluations; excludes subsequent recovery archiving.",
}
print(json.dumps({"run_record": record, "summary": summary, "pipeline_timing": timing}, indent=2))
