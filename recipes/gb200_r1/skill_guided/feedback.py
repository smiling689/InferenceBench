#!/usr/bin/env python3
"""Archive trial evidence around the unchanged scenario C evaluator."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request

TASK = Path("/home/agent/task")
ROOT = Path("/opt/inferencebench")
SCENARIO = ROOT / "src/eval/tasks/inference_scenario_c_high_load"
VENDOR = Path("/opt/ai-infra-skills")


def save(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def server_info(url: str) -> dict:
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(url.rstrip("/") + "/server_info", timeout=3) as response:
            return json.load(response)
    except Exception as exc:
        return {"capture_error": type(exc).__name__ + ": " + str(exc)}


def new_trial(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    index = 1
    while True:
        trial = root / f"trial_{index:04d}"
        try:
            trial.mkdir()
            return trial
        except FileExistsError:
            index += 1


def archive_files(trial: Path, metrics: dict) -> dict:
    artifacts = {}
    speed = metrics.get("speed_eval") or {}
    for field in ("generation_log_file", "requests_used_file"):
        raw = speed.get(field)
        if raw and Path(raw).is_file():
            source = Path(raw)
            destination = trial / (field + source.suffix)
            shutil.copyfile(source, destination)
            artifacts[field] = {"original": raw, "archived": str(destination),
                                "sha256": hashlib.sha256(destination.read_bytes()).hexdigest()}
    return artifacts


def analyze_capacity(trial: Path) -> dict:
    raw = os.environ.get("INFERENCE_BENCH_SKILL_SERVER_LOG")
    source = Path(raw) if raw else TASK / "server.log"
    if not source.is_file():
        return {"status": "no_log", "expected_log": str(source)}
    log = trial / "server.log"
    shutil.copyfile(source, log)
    command = [sys.executable, str(VENDOR / "skills/llm-serving-capacity-planner/scripts/capacity_analyzer.py"),
               "--log-file", str(log), "--config-json", "/models/deepseek-r1/config.json",
               "--request-tokens", "1640,1840,2048", "--format", "json"]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=20)
        (trial / "capacity_analysis.json").write_text(result.stdout)
        (trial / "capacity_analysis_stderr.txt").write_text(result.stderr)
        return {"status": "completed" if result.returncode == 0 else "failed",
                "returncode": result.returncode, "command": command, "log_source": str(source)}
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "command": command, "log_source": str(source)}


def main() -> None:
    from src.eval.inference.runner import build_parser, run_evaluation
    args = build_parser().parse_args()
    trial = new_trial(Path(os.environ["INFERENCE_BENCH_SKILL_TRIALS"]))
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    launcher = TASK / "start_server.sh"
    if launcher.is_file():
        shutil.copyfile(launcher, trial / "start_server.sh")
    url = args.server_url or os.environ["INFERENCE_BENCH_SERVER_URL"]
    save(trial / "server_info_before.json", server_info(url))
    save(trial / "trial.json", {"started_at": started, "status": "evaluating", "argv": sys.argv[1:]})
    metrics = {}
    error = None
    try:
        metrics = run_evaluation(SCENARIO, args)
    except Exception as exc:
        error = type(exc).__name__ + ": " + str(exc)
        output = Path(args.json_output_file) if args.json_output_file else None
        if output and output.is_file():
            try:
                metrics = json.loads(output.read_text())
            except (OSError, ValueError):
                metrics = {"error": error}
        else:
            metrics = {"error": error}
        raise
    finally:
        finished = dt.datetime.now(dt.timezone.utc).isoformat()
        save(trial / "metrics.json", metrics)
        evidence = {
            "started_at": started, "evaluation_finished_at": finished,
            "status": "failed" if error else "completed", "error": error,
            "argv": sys.argv[1:], "quick": bool(args.quick),
            "performance_passed": metrics.get("performance_check", {}).get("pass") is True,
            "artifacts": archive_files(trial, metrics), "capacity_analysis": analyze_capacity(trial),
            "analysis_note": str(trial / "analysis.md"),
        }
        save(trial / "trial.json", evidence)
        print(f"[skills] Trial evidence: {trial}; analyze it before selecting the next configuration.", flush=True)


if __name__ == "__main__":
    main()
