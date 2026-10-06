#!/usr/bin/env python3
"""Check treatment dependencies without starting a server or an agent timer."""
from __future__ import annotations

import ast
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
VENDOR = Path("/opt/ai-infra-skills")


def check_trial_archives() -> None:
    import feedback
    from src.eval.inference import runner
    with tempfile.TemporaryDirectory() as temporary:
        task = Path(temporary)
        trials = task / "trials"
        output = task / "preview_metrics.json"
        generation = task / "shared_generations.jsonl"
        requests = task / "shared_requests.jsonl"
        (task / "start_server.sh").write_text("#!/bin/bash\nexec server\n")
        args = ["evaluate.py", "--quick", "--json-output-file", str(output)]
        seen = []

        def successful(scenario, parsed):
            assert scenario == feedback.SCENARIO
            assert parsed.quick and parsed.request_limit is None
            assert parsed.quick_request_limit == 4
            seen.append(parsed.json_output_file)
            generation.write_text('{"successful_request": 1}\n')
            requests.write_text('{"input": 1}\n')
            return {"performance_check": {"pass": True}, "profiles": {},
                    "speed_eval": {"generation_log_file": str(generation), "requests_used_file": str(requests)}}

        def failing(scenario, parsed):
            output.write_text('{"error": "expected failure", "profiles": {}}')
            raise RuntimeError("expected failure")

        with patch.object(feedback, "TASK", task), patch.object(feedback, "server_info", return_value={}), \
                patch.dict(feedback.os.environ, {"INFERENCE_BENCH_SKILL_TRIALS": str(trials),
                                                 "INFERENCE_BENCH_QUICK_REQUEST_LIMIT": "4"}), \
                patch.object(sys, "argv", args), patch.object(sys, "stdout", io.StringIO()):
            with patch.object(runner, "run_evaluation", side_effect=successful):
                feedback.main()
            generation.write_text('{"different_request": 2}\n')
            assert (trials / "trial_0001/generation_log_file.jsonl").read_text() == '{"successful_request": 1}\n'
            assert seen == [str(output)]
            with patch.object(runner, "run_evaluation", side_effect=failing):
                try:
                    feedback.main()
                except RuntimeError as exc:
                    assert str(exc) == "expected failure"
                else:
                    raise AssertionError("The original evaluator failure must remain a failure")
            failed = json.loads((trials / "trial_0002/trial.json").read_text())
            assert failed["status"] == "failed" and not failed["performance_passed"]
            assert json.loads((trials / "trial_0002/metrics.json").read_text())["error"] == "expected failure"


def main() -> None:
    for path in HERE.glob("*.py"):
        ast.parse(path.read_text(), filename=str(path))
    subprocess.run(["bash", "-n", str(HERE / "run.sh")], check=True)
    reference = Path("/frozen-reference")
    expected = json.loads((HERE / "runtime_git_blobs.json").read_text())
    matched = 0
    for relative, digest in expected.items():
        if relative == "recipes/gb200_r1/PREFLIGHT_20260930.md":
            continue  # This non-runtime document was never deployed.
        data = (reference / relative).read_bytes()
        actual = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        if actual != digest:
            raise RuntimeError(f"Frozen source mismatch: {relative}")
        matched += 1
    assert matched == 96
    sys.path.insert(0, str(HERE))
    from agent import SKILL_NAMES, load_experiment, register_skills
    original = load_experiment()
    assert original.SCENARIO.name == "inference_scenario_c_high_load"
    scenario = json.loads((original.SCENARIO / "scenario.json").read_text())
    assert scenario["synthetic"] == {"input_len": 1024, "output_len": 1024, "range_ratio": 0.8, "ignore_eos": True}
    assert [(p["name"], p["concurrency"]) for p in scenario["profiles"]] == [("burst", 64), ("poisson", 32), ("constant", 16)]
    model = json.loads(Path("/models/deepseek-r1/config.json").read_text())
    assert [model[k] for k in ("num_hidden_layers", "hidden_size", "n_routed_experts", "num_experts_per_tok")] == [61, 7168, 256, 8]
    skills = register_skills()
    assert len(skills) == len(SKILL_NAMES) + 1
    assert (VENDOR / "docs/upstream-source-contracts.md").is_file()
    assert (VENDOR / "model-pr-optimization-history/sglang/deepseek-v3-r1/README.zh.md").is_file()
    check_trial_archives()
    checks = [
        [sys.executable, str(VENDOR / "skills/llm-torch-profiler-analysis/scripts/analyze_llm_torch_profile.py"), "--help"],
        [sys.executable, str(VENDOR / "skills/llm-serving-capacity-planner/scripts/capacity_analyzer.py"), "--help"],
        [sys.executable, str(VENDOR / "model-pr-optimization-history/scripts/query.py"), "--framework", "sglang", "--model", "deepseek-v3-r1", "--paths-only"],
    ]
    for command in checks:
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        if result.returncode:
            raise RuntimeError(result.stderr)
        assert result.stdout.strip()
    result = {"status": "passed", "trial_archive_checks": "passed", "frozen_runtime_files_matched": matched, "registered_skills": skills,
              "codex_version": subprocess.check_output(["codex", "--version"], text=True).strip(),
              "runtime_files": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in (HERE.parent / "experiment.py", ROOT / "agents/codex/solve.sh", ROOT / "src/eval/inference/runner.py")},
              "scenario_sha256": hashlib.sha256((original.SCENARIO / "scenario.json").read_bytes()).hexdigest()}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
