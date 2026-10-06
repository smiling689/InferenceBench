"""Produce a final comparison only from terminal, audited repetition evidence."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import statistics


def load(path):
    return json.loads(path.read_text())


def stats(values):
    if not values:
        return {"valid_runs": 0, "individual_scores": []}
    mean = statistics.mean(values)
    standard = statistics.stdev(values) if len(values) > 1 else None
    return {"valid_runs": len(values), "mean": mean, "median": statistics.median(values),
            "sample_std": standard, "sample_cv_pct": 100 * standard / mean if standard is not None else None,
            "min": min(values), "max": max(values), "individual_scores": values}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True,
                        help="Evidence tree copied from the canonical devbox")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = load(args.inputs / "run_manifest.json")
    entries = []
    for queue in manifest["queues"]:
        for position, run in enumerate(queue["runs"], 1):
            method = "gpt-6.1-sol-max" + ("-skills" if run["treatment"] == "skills" else "")
            directory = args.inputs / run["run_id"] / method
            stage = (directory / "pipeline_stage.txt").read_text().strip()
            assert stage in {"finished", "failed"}, f"Run is not terminal: {run['run_id']}"
            state = load(directory / "task/run_status.json")
            assert state["method"] == "gpt-6.1-sol" and state["reasoning_effort"] == "max"
            assert state["budget_seconds"] == 7200 and state["status"] == "optimization_finished"
            remaining = re.findall(r"remaining_seconds=(\d+)", (directory / "container.log").read_text())
            assert remaining and int(remaining[-1]) == 0, f"Budget was not consumed: {run['run_id']}"
            entry = {**run, "node": queue["node"], "queue_position": position, "stage": stage,
                     "pipeline_started_at": (directory / "pipeline_started_at.txt").read_text().strip(),
                     "pipeline_finished_at": (directory / "pipeline_finished_at.txt").read_text().strip(),
                     "container_exit_code": int((directory / "container_exit_code").read_text()),
                     "final_exit_code": int((directory / "final_exit_code").read_text()),
                     "run_status": state, "solver_budget_consumed": True, "score": None}
            if entry["final_exit_code"] == 0:
                audit_file = directory / "task/result_audit.json"
                audit = load(audit_file)
                assert audit["audit_passed"] is True and audit["solver_budget_consumed"] is True
                assert audit["generation_records"] == 768 and audit["optimization_budget_seconds"] == 7200
                assert audit["skills_treatment"] == (run["treatment"] == "skills")
                assert audit["effective_configuration_source"] == "live formal server endpoint"
                entry.update(score=audit["primary_metric"], profile_throughputs=audit["profile_throughputs"],
                             launcher_sha256=audit["launcher_sha256"],
                             audit_sha256=hashlib.sha256(audit_file.read_bytes()).hexdigest(),
                             effective_server_args=audit["effective_server_args"],
                             skill_evidence=audit["skill_evidence"])
            entries.append(entry)
    assert len(entries) == len({entry["run_id"] for entry in entries}) == 8
    groups = {name: stats([entry["score"] for entry in entries
                          if entry["treatment"] == name and entry["score"] is not None])
              for name in ("plain", "skills")}
    for name in groups:
        assert sum(entry["treatment"] == name for entry in entries) == 4
        groups[name]["failed_runs"] = 4 - groups[name]["valid_runs"]
    historical = load(args.inputs / "historical_runs.json")
    assert len(historical["plain"]) == 3 and len(historical["skills"]) == 2
    historical_stats = {name: stats([row["score"] for row in historical[name]]) for name in groups}
    combined = {name: stats([row["score"] for row in historical[name]] + groups[name]["individual_scores"])
                for name in groups}
    by_node = {node: {name: stats([entry["score"] for entry in entries
                                 if entry["node"] == node and entry["treatment"] == name and entry["score"] is not None])
                     for name in groups} for node in ("gb200-1", "gb200-2")}
    mean_change = (groups["skills"]["mean"] / groups["plain"]["mean"] - 1) * 100 if all(
        groups[name]["valid_runs"] for name in groups) else None
    result = {"status": "completed", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
              "metric": "geomean_request_throughput_req_per_s", "new_runs": entries,
              "new_run_statistics": groups, "new_skills_mean_change_pct": mean_change,
              "new_run_statistics_by_node": by_node, "historical_runs": historical,
              "historical_statistics": historical_stats, "combined_statistics": combined,
              "quality_evaluated": False,
              "limitations": ["Four new repetitions per treatment; descriptive comparison with small samples.",
                              "Independent search outcomes include selected-configuration and runtime variation.",
                              "Identical workload seeds; no claim of generalization to other traffic or lengths.",
                              "Historical samples are shown separately and are not balanced by node or date.",
                              "Skill artifact counts require manual interpretation; they do not prove complete compliance.",
                              "Full dummy weights measure performance, not model quality."]}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "comparison_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    fields = ["run_id", "treatment", "node", "queue_position", "score", "burst", "poisson", "constant",
              "final_exit_code", "pipeline_started_at", "pipeline_finished_at"]
    with (args.output / "individual_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for entry in sorted(entries, key=lambda item: (item["treatment"], item["run_id"])):
            row = {key: entry.get(key) for key in fields}
            row.update(entry.get("profile_throughputs", {}))
            writer.writerow(row)
    lines = ["# Sol/max: four new repetitions per treatment", "",
             f"Completed all eight planned experiments. New skills mean change relative to ordinary: {mean_change:+.2f}%." if mean_change is not None else "Completed all eight planned experiments; no valid comparison score.", "",
             "Each score is the formal geometric mean of burst/Poisson/constant request throughput, in req/s.",
             "Every valid new score passed all 768 held-out generation records and exact token counts.", "",
             "| Treatment | Valid / planned | Mean | Median | Sample CV | Range |",
             "| --- | --- | --- | --- | --- | --- |"]
    for name, group in groups.items():
        if group["valid_runs"]:
            cv = f"{group['sample_cv_pct']:.2f}%" if group["sample_cv_pct"] is not None else "n/a"
            lines.append(f"| {name} | {group['valid_runs']}/4 | {group['mean']:.6f} | {group['median']:.6f} | {cv} | {group['min']:.6f}–{group['max']:.6f} |")
        else:
            lines.append(f"| {name} | 0/4 | n/a | n/a | n/a | n/a |")
    lines.extend(["", "## Every new experiment", "",
                  "| Run | Node | Burst | Poisson | Constant | Score |",
                  "| --- | --- | --- | --- | --- | --- |"])
    for entry in sorted(entries, key=lambda item: (item["treatment"], item["run_id"])):
        if entry["score"] is None:
            values = "failed | failed | failed | failed"
        else:
            profile = entry["profile_throughputs"]
            values = " | ".join(f"{value:.6f}" for value in [profile["burst"], profile["poisson"], profile["constant"], entry["score"]])
        lines.append(f"| {entry['run_id']} | {entry['node']} | {values} |")
    lines.extend(["", "## Historical context", "",
                  "Historical runs remain separate from the balanced new four-versus-four comparison.", "",
                  "| Treatment | Earlier scores | Combined n | Combined mean | Combined sample CV |",
                  "| --- | --- | --- | --- | --- |"])
    for name in groups:
        earlier = ", ".join(f"{row['score']:.6f}" for row in historical[name])
        group = combined[name]
        lines.append(f"| {name} | {earlier} | {group['valid_runs']} | {group['mean']:.6f} | {group['sample_cv_pct']:.2f}% |")
    lines.extend(["", "## Scope and limits", "", *[f"- {item}" for item in result["limitations"]], "",
                  "The original scenario C, FP8 dummy full DeepSeek-R1, BF16 activations/KV, four GB200 GPUs and 7200-second optimization budgets are retained.",
                  "Each node ran two ordinary and two skills runs in opposite alternating orders. Previews and fresh-container formal evaluation add elapsed time.",
                  "See comparison_summary.json for per-node statistics, effective configurations, timing and actual skill evidence; individual_results.csv preserves every new score.", ""])
    (args.output / "README.md").write_text("\n".join(lines))
    print(json.dumps({"new_run_statistics": groups, "new_skills_mean_change_pct": mean_change,
                      "combined_statistics": combined}, indent=2))


if __name__ == "__main__":
    main()
