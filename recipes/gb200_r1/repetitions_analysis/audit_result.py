"""Audit completed run records inside a CPU Docker container on its node."""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime, timedelta
import hashlib
import gzip
import json
import math
from pathlib import Path
import re


def load(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def inspect_skills(task, codex, deadline):
    trials = []
    for path in sorted(task.glob("agent/run_*/trials/trial_*/trial.json")):
        trial = load(path)
        started = datetime.fromisoformat(trial["started_at"])
        analysis = path.parent / "analysis.md"
        trials.append({"trial": str(path.relative_to(task)), "started_at": trial["started_at"],
                       "during_optimization": started < deadline,
                       "status": trial.get("status"), "passed": trial.get("performance_passed"),
                       "analysis_present": analysis.is_file(),
                       "analysis_bytes": analysis.stat().st_size if analysis.is_file() else 0})
    calls = []
    contexts = Counter()
    sessions = []
    for session in sorted(codex.glob("sessions/**/*.jsonl")):
        sessions.append({"file": str(session.relative_to(codex)), "sha256": digest(session)})
        for row in rows(session):
            payload = row.get("payload", {})
            if row.get("type") == "turn_context":
                contexts[(payload.get("model"), payload.get("effort", payload.get("reasoning_effort")))] += 1
            if payload.get("type") != "function_call":
                continue
            arguments = payload.get("arguments", "")
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments)
            paths = sorted(set(re.findall(r"/opt/ai-infra-skills/[A-Za-z0-9_./-]+", arguments)))
            if paths:
                calls.append({"timestamp": row.get("timestamp"), "call_id": payload.get("call_id"),
                              "tool": payload.get("name"), "toolkit_paths": paths})
    traces = []
    for path in task.rglob("*"):
        if path.is_file() and (path.name.endswith(".trace.json.gz") or path.name.endswith(".trace.json")):
            evidence = {"path": str(path.relative_to(task)), "bytes": path.stat().st_size,
                        "sha256": digest(path)}
            opener = gzip.open if path.name.endswith(".gz") else open
            try:
                with opener(path, "rt") as handle:
                    trace = json.load(handle)
                events = trace.get("traceEvents", [])
                evidence["gpu_kernel_events"] = sum(
                    "kernel" in str(event.get("cat", "")).lower() and event.get("dur", 0) > 0
                    for event in events)
                evidence["nonzero_gpu_events"] = evidence["gpu_kernel_events"] > 0
                del trace, events
            except (OSError, ValueError) as error:
                evidence["trace_error"] = str(error)
            traces.append(evidence)
    in_budget = [trial for trial in trials if trial["during_optimization"]]
    return {"optimization_evaluations": len(in_budget),
            "optimization_evaluations_with_analysis": sum(t["analysis_present"] for t in in_budget),
            "post_budget_evaluations": len(trials) - len(in_budget), "trials": trials,
            "actual_toolkit_function_calls": calls, "codex_sessions": sessions,
            "model_contexts": [{"model": model, "effort": effort, "count": count}
                               for (model, effort), count in contexts.items()],
            "trace_artifacts": traces,
            "final_selection_files": [str(p.relative_to(task)) for p in task.glob("agent/run_*/final_selection.md")],
            "interpretation": "Presence and tool-call evidence only; manually audit per-configuration analysis and representative profiling."}


def formal_args(task):
    info = task / "formal_server_info.json"
    if info.is_file():
        value = load(info)
        return value.get("server_args", value), "live formal server endpoint"
    # Older runs did not collect the live endpoint. Preserve that distinction.
    for line in (task / "final_server.log").read_text().splitlines():
        if "server_args=ServerArgs(" in line:
            expression = line.split("server_args=", 1)[1]
            call = ast.parse(expression, mode="eval").body
            assert isinstance(call, ast.Call)
            parsed = {}
            for keyword in call.keywords:
                try:
                    parsed[keyword.arg] = ast.literal_eval(keyword.value)
                except (ValueError, TypeError):
                    pass
            return parsed, "formal server startup log (live endpoint not captured)"
    raise AssertionError("No effective formal server configuration evidence")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=Path, default=Path("/home/agent/task"))
    parser.add_argument("--codex", type=Path, default=Path("/evidence/codex"))
    args = parser.parse_args()
    task = args.task
    metrics = load(task / "final_metrics.json")
    expected_file = Path("/bench-inputs/requests_1337.jsonl")
    expected_hash = "08fb8644ae4fee2939187086103a9a310866be4b9bd5f50b9c2aef2a343d6cae"
    assert digest(expected_file) == expected_hash
    expected = rows(expected_file)
    used_file = Path(metrics["speed_eval"]["requests_used_file"])
    used = rows(used_file)
    assert len(used) == len(expected) == 256
    fields = ("sample_id", "messages", "max_new_tokens", "temperature", "ignore_eos", "input_token_count")
    for left, right in zip(expected, used):
        assert all(left.get(key) == right.get(key) for key in fields), left["sample_id"]
        assert right["temperature"] == .3 and right["ignore_eos"] is True
        assert 820 <= right["input_token_count"] <= 1024
        assert 820 <= right["max_new_tokens"] <= 1024
    assert metrics["speed_eval"]["requests_source"] == str(expected_file)
    assert metrics["model_id"] == "deepseek-ai/DeepSeek-R1"
    profiles = metrics["profiles"]
    assert set(profiles) == {"burst", "poisson", "constant"}
    assert metrics["performance_check"]["pass"] is True and not metrics.get("error")
    for profile in profiles.values():
        assert profile["request_count"] == profile["success_count"] == 256
        assert profile["failure_count"] == profile["empty_output_count"] == 0
        assert profile["request_throughput_req_per_s"] > 0
    generation_file = Path(metrics["speed_eval"]["generation_log_file"])
    generations = rows(generation_file)
    assert len(generations) == 768
    for block in range(3):
        batch = generations[block * 256:(block + 1) * 256]
        assert {row["request_index"] for row in batch} == set(range(256))
        for row in batch:
            request = used[row["request_index"]]
            assert row["sample_id"] == request["sample_id"]
            assert row["success"] is True and not row["empty_output"] and not row["error"]
            assert isinstance(row["model_output"], str) and row["model_output"].strip()
            assert row["output_tokens"] == row["tokens"] == request["max_new_tokens"]
            assert row["input_tokens"] == request["input_token_count"]
    model = load("/models/deepseek-r1/config.json")
    shape = {key: model[key] for key in ("num_hidden_layers", "hidden_size", "n_routed_experts",
                                         "num_experts_per_tok", "num_attention_heads")}
    assert shape == {"num_hidden_layers": 61, "hidden_size": 7168, "n_routed_experts": 256,
                     "num_experts_per_tok": 8, "num_attention_heads": 128}
    server, configuration_source = formal_args(task)
    assert server["load_format"] == "dummy" and server["quantization"] == "fp8"
    assert server["dtype"] == "bfloat16" and server["kv_cache_dtype"] in ("bf16", "bfloat16")
    assert server["tp_size"] == 4
    assert server["disable_custom_all_reduce"] is True
    assert server["enforce_disable_flashinfer_allreduce_fusion"] is True
    score = math.exp(sum(math.log(profile["request_throughput_req_per_s"]) for profile in profiles.values()) / 3)
    assert math.isclose(score, load(task / "summary.json")["primary_metric"], rel_tol=1e-12)
    state = load(task / "run_status.json")
    assert state["method"] == "gpt-6.1-sol" and state["reasoning_effort"] == "max"
    assert state["budget_seconds"] == 7200 and state["status"] == "optimization_finished"
    solver_log = (task.parent / "container.log").read_text()
    assert "total_timeout_seconds=7200" in solver_log
    remaining = re.findall(r"remaining_seconds=(\d+)", solver_log)
    assert remaining and int(remaining[-1]) == 0, "Original solver did not consume its full optimization budget"
    deadline = datetime.fromisoformat(state["started_at"]) + timedelta(seconds=7200)
    skill_evidence = inspect_skills(task, args.codex, deadline)
    result = {"audit_passed": True, "primary_metric": score, "official_evaluation_seed": 1337,
              "request_source_sha256": expected_hash, "requests_per_profile": 256,
              "profiles": 3, "generation_records": 768, "input_output_token_counts_exact": True,
              "request_fields_unchanged": list(fields), "model_shape": shape,
              "weights_format": "fp8 dummy", "activation_dtype": "bfloat16", "kv_dtype": server["kv_cache_dtype"],
              "quality_evaluated": False, "effective_server_args": server,
              "effective_configuration_source": configuration_source,
              "model_config_sha256": digest("/models/deepseek-r1/config.json"),
              "generation_log_sha256": digest(generation_file), "requests_used_sha256": digest(used_file),
              "launcher_sha256": digest(task / "start_server.sh"),
              "profile_throughputs": {name: value["request_throughput_req_per_s"] for name, value in profiles.items()},
              "optimization_budget_seconds": 7200, "solver_exit_code": state["solver_exit_code"],
              "solver_budget_consumed": True, "last_remaining_seconds": int(remaining[-1]),
              "skills_treatment": (task / "skill_manifest.json").is_file(), "skill_evidence": skill_evidence}
    (task / "result_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"audit_passed": True, "primary_metric": score,
                      "generation_records": 768, "configuration_source": configuration_source,
                      "optimization_evaluations": skill_evidence["optimization_evaluations"],
                      "analysis_present": skill_evidence["optimization_evaluations_with_analysis"],
                      "toolkit_calls": len(skill_evidence["actual_toolkit_function_calls"])}))


if __name__ == "__main__":
    main()
