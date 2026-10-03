"""Collect completed-search evidence while original HPO final evaluation runs.

Run inside the existing container with random or smac as the argument.
The files under search_evidence are observations, not HPO inputs or scores
from the held-out evaluation. They do not change the optimization workflow.
"""

import datetime
import hashlib
import json
import math
import sys
from pathlib import Path


method = sys.argv[1]
assert method in {"random", "smac"}
task = Path("/home/agent/task")
record_path = task / "hpo" / method / "sglang/scenario_c/seed_0.jsonl"
records = [json.loads(line) for line in record_path.read_text().splitlines() if line.strip()]
start = records[0]
assert start["type"] == "run_start" and start["method"] == method
assert start["seed_pair"] == [21, 1337]
trials = [row for row in records if row["type"] == "trial"]
assert [row["trial_idx"] for row in trials] == list(range(len(trials)))
valid = [row for row in trials if row["failure_reason"] is None and not row["integrity_flags"]]
assert valid and all(row["quick_metric"] > 0 for row in valid)
best = max(valid, key=lambda row: row["quick_metric"])
final = record_path.with_name("seed_0_artifacts") / "final"
assert (final / "server.log").exists()
info = json.loads((final / "formal_server_info.json").read_text())
args = info.get("server_args", info)
assert not args.get("api_key")
assert args["load_format"] == "dummy" and args["quantization"] == "fp8"
assert args["tp_size"] == 4 and args["ep_size"] == args["dp_size"] == 1
assert args["dtype"] == "bfloat16" and args["kv_cache_dtype"] in {"bf16", "bfloat16"}
assert args["disable_cuda_graph"] is True
assert args["cuda_graph_config"]["decode"]["backend"] == "disabled"
assert args["cuda_graph_config"]["prefill"]["backend"] == "disabled"
assert args["enable_torch_compile"] is False
assert args["speculative_algorithm"] is None
for key in ("max_running_requests", "chunked_prefill_size", "mem_fraction_static", "schedule_policy"):
    assert args[key] == best["config"][key], key

runtime = Path("/opt/inferencebench/src/eval/inference/hpo_search_baselines.py")
runtime_hash = hashlib.sha256(runtime.read_bytes()).hexdigest()
assert runtime_hash == "be9b857c8b1ef0b036e12af406439d250ca8b6d3da01dc329ae62eb1875c75e8"
quick = json.loads(Path(best["metrics_path"]).read_text())
assert quick["performance_check"]["pass"] is True
assert quick["speed_eval"]["requests_source"] == "/bench-inputs/requests_21.jsonl"
assert set(quick["profiles"]) == {"burst", "poisson", "constant"}
for profile in quick["profiles"].values():
    assert profile["request_count"] == profile["success_count"] == 4
    assert profile["failure_count"] == profile["empty_output_count"] == 0
quick_score = math.exp(sum(math.log(p["request_throughput_req_per_s"]) for p in quick["profiles"].values()) / 3)
assert math.isclose(quick_score, best["quick_metric"], rel_tol=1e-12)

output = task / "search_evidence"
output.mkdir(exist_ok=True)
summary = {
    "status": "search_complete_formal_server_configuration_verified",
    "method": method,
    "optimization_budget_s": 7200,
    "hpo_budget_started_at_node_utc": start["start_ts"],
    "last_trial_teardown_at_node_utc": trials[-1]["teardown_ts"],
    "collected_at_node_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "total_trials": len(trials),
    "valid_quick_trials": len(valid),
    "unique_valid_configurations": len({json.dumps(row["config"], sort_keys=True) for row in valid}),
    "failed_trials": [{"trial_idx": row["trial_idx"], "reason": row["failure_reason"]} for row in trials if row["failure_reason"] is not None],
    "best_trial_idx": best["trial_idx"],
    "selected_quick_metric": quick_score,
    "selected_quick_metric_source": "geomean_request_throughput_req_per_s",
    "quick_requests_per_profile": 4,
    "development_seed": 21,
    "evaluation_seed": 1337,
    "optimizer_seeds": {"random": 21} if method == "random" else {"configspace": 21, "scenario": 0},
    "hpo_runtime_sha256": runtime_hash,
    "actual_formal_server_configuration_matches_selected_trial": True,
    "quality_evaluated": False,
    "note": "Quick score is development feedback. Official score awaits the full held-out evaluation; budget_used_s will come from run_end.",
}
for name, payload in (("search_summary.json", summary), ("selected_config.json", best["config"]), ("selected_quick_metrics.json", quick)):
    (output / name).write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps({key: summary[key] for key in ("method", "total_trials", "valid_quick_trials", "unique_valid_configurations", "best_trial_idx", "selected_quick_metric", "actual_formal_server_configuration_matches_selected_trial")}))
