"""Audit a completed original HPO run inside a CPU-only node container.

Mount its task at /home/agent/task, inputs at /bench-inputs, model metadata
at /models/deepseek-r1, and frozen source at /opt/inferencebench/src.
Pass random or smac as the first argument. This writes only audit evidence.
"""

import hashlib
import json
import math
import sys
from pathlib import Path


def read_json(path):
    return json.loads(path.read_text())


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


method = sys.argv[1]
assert method in {"random", "smac"}
task = Path("/home/agent/task")
record_path = task / "hpo" / method / "sglang/scenario_c/seed_0.jsonl"
records = read_jsonl(record_path)
assert sum(row["type"] == "run_start" for row in records) == 1
assert sum(row["type"] == "run_end" for row in records) == 1
assert not any(row["type"] == "run_abort" for row in records)
end = records[-1]
assert end["type"] == "run_end"
trials = [row for row in records if row["type"] == "trial"]
assert len(trials) == end["total_trials"]
assert [row["trial_idx"] for row in trials] == list(range(len(trials)))
valid = [row for row in trials if row["failure_reason"] is None and not row["integrity_flags"]]
assert valid and all(row["quick_metric"] > 0 for row in valid)
best = max(valid, key=lambda row: row["quick_metric"])
assert best["trial_idx"] == end["best_trial_idx"]
assert best["config"] == end["final_config"]
assert end["performance_passed"] is True and end["integrity_passed"] is True
assert end["integrity_flags"] == [] and end["quality_evaluated"] is False

final = record_path.with_name("seed_0_artifacts") / "final"
metrics_path = final / "final_metrics.json"
metrics = read_json(metrics_path)
assert metrics == end["final_metrics"]
assert metrics["scenario"] == "C" and metrics["model_id"] == "/models/deepseek-r1"
assert metrics["performance_check"]["pass"] is True and not metrics.get("error")
assert metrics["quality_evaluated"] is False
assert metrics["quality_check"]["note"] == "dummy weights: performance only"

input_hashes = {
    "requests_21.jsonl": "e3329da09722133ae0d3fd368b3438b2c8f230815ab32ee32cb77773e71f3ba5",
    "requests_1337.jsonl": "08fb8644ae4fee2939187086103a9a310866be4b9bd5f50b9c2aef2a343d6cae",
}
for name, expected_hash in input_hashes.items():
    assert sha256(Path("/bench-inputs") / name) == expected_hash
source = Path("/bench-inputs/requests_1337.jsonl")
assert metrics["speed_eval"]["requests_source"] == str(source)
expected = read_jsonl(source)
used_path = Path(metrics["speed_eval"]["requests_used_file"])
assert used_path == final / "requests_used_speed.jsonl"
used = read_jsonl(used_path)
assert len(expected) == len(used) == 256
fields = ("sample_id", "messages", "max_new_tokens", "temperature", "ignore_eos", "input_token_count")
for original, actual in zip(expected, used):
    assert all(original.get(key) == actual.get(key) for key in fields), original["sample_id"]
    assert actual["temperature"] == 0.3 and actual["ignore_eos"] is True
    assert 820 <= actual["input_token_count"] <= 1024
    assert 820 <= actual["max_new_tokens"] <= 1024
assert len({row["sample_id"] for row in used}) == 256
assert sum(row["max_new_tokens"] for row in used) == 235820

profiles = metrics["profiles"]
assert set(profiles) == {"burst", "poisson", "constant"}
for profile in profiles.values():
    assert profile["request_count"] == profile["success_count"] == 256
    assert profile["failure_count"] == profile["empty_output_count"] == 0
    assert profile["request_throughput_req_per_s"] > 0
generation_path = Path(metrics["speed_eval"]["generation_log_file"])
assert generation_path == final / "speed_generations.jsonl"
rows = read_jsonl(generation_path)
assert len(rows) == 768
for block in range(3):
    batch = rows[block * 256 : (block + 1) * 256]
    assert {row["request_index"] for row in batch} == set(range(256))
    for row in batch:
        request = used[row["request_index"]]
        assert row["sample_id"] == request["sample_id"]
        assert row["success"] is True and not row["empty_output"] and not row["error"]
        assert isinstance(row["model_output"], str) and row["model_output"].strip()
        assert row["output_tokens"] == row["tokens"] == request["max_new_tokens"]
        assert row["input_tokens"] == request["input_token_count"]

model_path = Path("/models/deepseek-r1/config.json")
model = read_json(model_path)
shape = {key: model[key] for key in (
    "num_hidden_layers", "hidden_size", "n_routed_experts",
    "num_experts_per_tok", "num_attention_heads",
)}
assert shape == {
    "num_hidden_layers": 61, "hidden_size": 7168, "n_routed_experts": 256,
    "num_experts_per_tok": 8, "num_attention_heads": 128,
}
assert sha256(model_path) == "79ddea672a62e95d3f0be27be434375538e1988975971cffcf87c96c1de84a65"
info = read_json(final / "formal_server_info.json")
args = info.get("server_args", info)
assert args["model_path"] == "/models/deepseek-r1"
assert args["load_format"] == "dummy" and args["quantization"] == "fp8"
assert args["dtype"] == "bfloat16" and args["kv_cache_dtype"] in {"bf16", "bfloat16"}
assert args["tp_size"] == 4 and args["ep_size"] == args["dp_size"] == 1
assert args["disable_cuda_graph"] is True
assert args["cuda_graph_config"]["decode"]["backend"] == "disabled"
assert args["cuda_graph_config"]["prefill"]["backend"] == "disabled"
assert args["enable_torch_compile"] is False
assert args["speculative_algorithm"] is None
assert args["disable_custom_all_reduce"] is True
assert args["enforce_disable_flashinfer_allreduce_fusion"] is True
for key in ("max_running_requests", "chunked_prefill_size", "mem_fraction_static", "schedule_policy"):
    assert args[key] == end["final_config"][key], key

runtime = Path("/opt/inferencebench/src/eval/inference/hpo_search_baselines.py")
runtime_hash = "be9b857c8b1ef0b036e12af406439d250ca8b6d3da01dc329ae62eb1875c75e8"
assert sha256(runtime) == runtime_hash
score = math.exp(sum(math.log(profile["request_throughput_req_per_s"]) for profile in profiles.values()) / 3)
assert math.isclose(score, end["final_primary_metric"], rel_tol=1e-12)
result = {
    "audit_passed": True,
    "method": method,
    "official_evaluation_seed": 1337,
    "request_file_sha256": input_hashes,
    "requests_per_profile": 256,
    "profiles": 3,
    "generation_records": len(rows),
    "complete_success_records": len(rows),
    "input_output_token_counts_exact": True,
    "request_fields_unchanged": list(fields),
    "model_shape": shape,
    "model_config_sha256": sha256(model_path),
    "weights_format": "fp8 dummy",
    "activation_dtype": "bfloat16",
    "kv_dtype": args["kv_cache_dtype"],
    "cuda_graph_backends": {key: args["cuda_graph_config"][key]["backend"] for key in ("decode", "prefill")},
    "primary_metric": score,
    "profile_throughputs": {key: profile["request_throughput_req_per_s"] for key, profile in profiles.items()},
    "total_requested_output_tokens_per_profile": 235820,
    "generation_log_sha256": sha256(generation_path),
    "requests_used_sha256": sha256(used_path),
    "hpo_runtime_sha256": runtime_hash,
    "trial_count": len(trials),
    "valid_quick_trials": len(valid),
    "failed_quick_trials": len(trials) - len(valid),
    "best_trial_idx": best["trial_idx"],
    "budget_used_s": end["budget_used_s"],
    "quality_evaluated": False,
}
(final / "result_audit.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
