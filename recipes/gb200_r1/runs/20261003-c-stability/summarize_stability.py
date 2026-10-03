"""Compute official-score statistics inside a CPU-only node container.

Pass a JSON object of source records as argv[1]. Each record identifies its
method, official metric and the SHA-256 of the source evidence. This script
prints JSON for collection on the devbox; it does not run new evaluations.
"""

import json
import math
import statistics
import sys


records = json.loads(sys.argv[1])
assert set(records) == {"sol", "random", "smac", "astra"}
expected_counts = {"sol": 3, "random": 2, "smac": 2, "astra": 1}
output = {
    "status": "completed",
    "metric": "geomean_request_throughput_req_per_s",
    "evaluation_seed": 1337,
    "individual_runs": records,
}
for method, runs in records.items():
    assert len(runs) == expected_counts[method]
    for run in runs:
        assert run["request_count"] == run["request_successes"] == 768
        assert run["evaluation_seed"] == 1337
        assert len(run["evidence_sha256"]) == 64 and run["evidence_file"]
        int(run["evidence_sha256"], 16)
        assert math.isfinite(run["score"]) and run["score"] > 0
        profiles = run["profile_throughputs"]
        assert set(profiles) == {"burst", "poisson", "constant"}
        assert all(math.isfinite(value) and value > 0 for value in profiles.values())
        score = math.exp(sum(math.log(value) for value in profiles.values()) / 3)
        assert math.isclose(score, run["score"], rel_tol=1e-12)
    values = [run["score"] for run in runs]
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) > 1 else None
    output[method] = {
        "n": len(values),
        "scores": values,
        "relative_change_vs_first_percent": [(value / values[0] - 1) * 100 for value in values],
        "mean": mean,
        "sample_sd": sd,
        "cv_percent": sd / mean * 100 if sd is not None else None,
        "min": min(values),
        "max": max(values),
        "max_over_min": max(values) / min(values),
        "median": statistics.median(values),
        "complete_runs": len(values),
        "failed_runs": 0,
        "request_successes_per_run": 768,
    }
output["random"]["optimizer_seed"] = 21
output["random"]["same_selected_configuration"] = records["random"][1]["same_selected_configuration_as_first"]
output["smac"].update(configspace_seed=21, scenario_seed=0)
output["smac"]["same_selected_configuration"] = records["smac"][1]["same_selected_configuration_as_first"]
output["astra"]["additional_runs"] = 0
if len(sys.argv) == 3:
    timings = json.loads(sys.argv[2])
    assert len(timings) == 2 and {row["node"] for row in timings} == {"gb200-1", "gb200-2"}
    output["additional_round_pipeline_timings"] = timings
    output["additional_round_pipeline_node_hours"] = sum(row["agent_to_mechanical_final_elapsed_hours"] for row in timings)
    output["additional_round_configured_optimization_budget_node_hours"] = 8
output["limitations"] = [
    "Small sample sizes; no statistical significance claim.",
    "Mechanical repeats keep original optimizer seeds; this is not cross-seed search stability.",
    "Different nodes and service configurations contribute to variation.",
    "Dummy weights affect routing and speculative acceptance; no accuracy evaluation.",
    "Agent and mechanical methods have different permitted search spaces.",
    "Random repeats select the identical full service configuration but have different measured throughput; execution/measurement variation is present.",
]
print(json.dumps(output, indent=2))
