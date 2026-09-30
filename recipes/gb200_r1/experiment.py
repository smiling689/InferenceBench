#!/usr/bin/env python3
"""Container entrypoints for the original InferenceBench workflow on GB200."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
TASK = Path("/home/agent/task")
SCENARIO = ROOT / "src/eval/tasks/inference_scenario_c_high_load"
MODEL = "/models/deepseek-r1"


def save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def environment(seed: int = 21) -> None:
    values = {
        "INFERENCE_BENCH_BASE_MODEL": MODEL,
        "INFERENCE_BENCH_SCENARIO": SCENARIO.name,
        "INFERENCE_BENCH_PERFORMANCE_ONLY": "1",
        "INFERENCE_BENCH_TOKENIZER_BACKEND": "sglang",
        "INFERENCE_BENCH_DATASET_SEED": str(seed),
        "INFERENCE_BENCH_REQUESTS_DIR": "/bench-inputs",
        "INFERENCE_BENCH_MAX_MODEL_LEN": "16384",
        "INFERENCE_BENCH_SERVER_HOST": "127.0.0.1",
        "INFERENCE_BENCH_SERVER_PORT": "30080",
        "INFERENCE_BENCH_SERVER_URL": "http://127.0.0.1:30080",
        "INFERENCE_BENCH_SERVER_WAIT_S": "1200",
        "INFERENCE_BENCH_METRICS_PATH": str(TASK / "preview_metrics.json"),
        "INFERENCE_BENCH_DISABLE_RUNTIME_SHIMS": "1",
        "INFERENCE_BENCH_RUNTIME_CACHE_DIR": str(TASK / "runtime_cache"),
        "INFERENCE_BENCH_QUICK_REQUEST_LIMIT": "4",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1", "SGLANG_SHARED_EXPERT_TP1": "0",
        "NCCL_NVLS_ENABLE": "0",
        "TORCH_SYMM_MEM_DISABLE_MULTICAST": "1",
        "HOST": "127.0.0.1", "PORT": "30080",
        "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(ROOT),
        "NO_PROXY": "localhost,127.0.0.1,::1", "no_proxy": "localhost,127.0.0.1,::1",
    }
    os.environ.update(values)
    TASK.mkdir(parents=True, exist_ok=True)


def generate_requests(tokenizer, seed: int) -> list[dict]:
    from src.eval.inference.runner import _count_chat_tokens, _truncate_messages
    in_rng = random.Random(f"{seed}:input:1024:1024:0.8")
    out_rng = random.Random(f"{seed}:output:1024:1024:0.8")
    content_rng = random.Random(f"{seed}:synthetic-content")
    words = "the a engine model request memory server compute system token batch schedule network process storage matrix layer expert number time data input output GPU parallel cache test result load node device tensor kernel measurement sequence attention value index performance experiment inference generation throughput latency report allocation capacity".split()
    rows = []
    for i in range(256):
        target = in_rng.randint(820, 1024)
        text = " ".join(content_rng.choices(words, k=2000))
        messages = _truncate_messages([{"role": "user", "content": text}], tokenizer, target, keep="head")
        count = _count_chat_tokens(messages, tokenizer)
        # Ordinary words are stable under decode/encode; pad any boundary shortfall.
        for _ in range(8):
            if count >= target:
                break
            messages[0]["content"] += " a"
            count = _count_chat_tokens(messages, tokenizer)
        if count != target:
            raise ValueError(f"Could not construct exact input length {target}: {count}")
        rows.append({"sample_id": f"synthetic-{seed}-{i}", "messages": messages,
                     "max_new_tokens": out_rng.randint(820, 1024), "temperature": 0.3,
                     "ignore_eos": True, "input_token_count": count,
                     "target_input_token_count": target, "sampling_range_ratio": 0.8})
        if (i + 1) % 64 == 0:
            print(f"[prepare] seed={seed} requests={i + 1}/256", flush=True)
    return rows


def prepare() -> None:
    from src.eval.inference.runner import _get_tokenizer
    environment()
    tokenizer = _get_tokenizer(MODEL)
    out = Path("/bench-inputs")
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    for seed in (21, 1337):
        path = out / f"requests_{seed}.jsonl"
        rows = generate_requests(tokenizer, seed)
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        files[path.name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "count": len(rows), "seed": seed}
    manifest = {"created_at": now(), "upstream_commit": "24cdf88f6a4e14ed85d665aa132cecccb3ee95ef",
                "scenario": "C", "model": "deepseek-ai/DeepSeek-R1", "load_format": "dummy",
                "quality_evaluated": False, "datasets_downloaded": False, "files": files,
                "tokenizer_backend": "sglang", "tokenizer_class": type(tokenizer).__name__,
                "scenario_sha256": hashlib.sha256((SCENARIO / "scenario.json").read_bytes()).hexdigest(),
                "search_space_sha256": hashlib.sha256((ROOT / "recipes/gb200_r1/sglang.json").read_bytes()).hexdigest(),
                "budget_seconds_per_method": 7200, "hpo_quick_request_limit": 4}
    save(out / "manifest.json", manifest)
    print(json.dumps(manifest), flush=True)


def setup_agent(model: str, hours: float) -> None:
    environment()
    os.environ.update(AGENT_CONFIG=model + "-max", NUM_HOURS=str(hours))
    shared = ROOT / "src/eval/tasks/_shared/task_context"
    for name in ("start_server.sh", "test_server.sh"):
        shutil.copy2(shared / name, TASK / name)
        (TASK / name).chmod(0o755)
    (TASK / "evaluate.py").write_text(
        "#!/usr/bin/env python3\nfrom pathlib import Path\n"
        "from src.eval.inference.runner import build_parser, run_evaluation\n"
        f"run_evaluation(Path({str(SCENARIO)!r}), build_parser().parse_args())\n")
    (TASK / "inference").symlink_to(ROOT / "src/eval/inference", target_is_directory=True)
    prompt = subprocess.check_output([
        sys.executable, str(ROOT / "src/eval/general/get_prompt.py"), "--agent", "codex",
        "--base-model", "deepseek-ai/DeepSeek-R1", "--scenario-id", SCENARIO.name,
        "--num-hours", str(hours), "--starting-point", "default"], cwd=ROOT, text=True)
    prompt = prompt.replace("NVIDIA H100 80GB HBM3 (1x)", "NVIDIA GB200 189471 MiB (4x), one node")
    prompt = prompt.replace("RAM: 180 GB", "RAM: 768 GiB container limit")
    prompt = prompt.replace("You must verify that quantization does not degrade model accuracy below acceptable benchmarks (monitor 'evaluate.py' results for quality drops/hallucinations).", "This dummy-weight experiment evaluates performance only; model accuracy is not evaluated.")
    prompt += f"""

## GB200 / full R1 environment
The complete 61-layer, 256-routed-expert DeepSeek-R1 configuration and tokenizer are mounted
read-only at {MODEL}. Use this local path for model and tokenizer loading. Weights and
datasets must NOT be downloaded. Use dummy weights (SGLang --load-format dummy) and FP8
weights with BF16 activations/KV. Preserve all model dimensions, layers, experts, and top-k.
Do not skip model computation, replace the model with a smaller model, return cached outputs,
or fabricate server-reported token counts. Each request must compute every requested token.
SGLang 0.5.15.post1, CUDA 13, and PyTorch are preinstalled; inspect their installed CLI/docs.
All four GPUs are exclusively available. This is an isolated Docker container.
The rack's multicast path failed preparation: NCCL NVLS hangs in cuMulticastBindMem,
and FlashInfer MNNVL allreduce fusion failed under sustained R1 load. Keep
NCCL_NVLS_ENABLE=0, TORCH_SYMM_MEM_DISABLE_MULTICAST=1, and use SGLang --enforce-disable-flashinfer-allreduce-fusion
--disable-custom-all-reduce so all four methods use the ordinary NCCL path.
The original evaluate.py protocol is available here, with the same scenario C input/output
length distribution, traffic profiles, and request-throughput objective. Request contents
are fixed synthetic text, since datasets are excluded from this experiment. The development
seed is 21. Final evaluation uses a separate held-out request set (seed 1337).
Quality is reported as not evaluated. Incomplete or failed requests invalidate a measurement.
The original --quick mode uses 4 requests per profile; use full evaluation to measure high load.
The objective is the geometric mean of request throughput over burst, poisson, and constant.
The evaluator and input files are read-only. Keep all your changes and notes under this task
directory. Installed dependencies are captured for the fresh final-evaluation container.
"""
    prompt_path = Path("/home/agent/prompt.txt")
    prompt_path.write_text(prompt)
    shutil.copy2(prompt_path, TASK / "prompt.txt")
    os.environ["PROMPT_FILE"] = str(prompt_path)


def agent(args) -> None:
    credentials = json.load(sys.stdin)
    os.environ["CODEX_API_KEY"] = credentials["api_key"]
    os.environ["INFERENCE_BENCH_CODEX_BASE_URL"] = credentials["base_url"]
    setup_agent(args.model, args.hours)
    state = {"method": args.model, "reasoning_effort": "max", "started_at": now(),
             "budget_seconds": int(args.hours * 3600), "workflow": "upstream agents/codex/solve.sh",
             "quality_evaluated": False, "status": "optimizing"}
    save(TASK / "run_status.json", state)
    result = subprocess.run(["bash", str(ROOT / "agents/codex/solve.sh")], cwd=TASK)
    state.update(ended_at=now(), solver_exit_code=result.returncode, status="optimization_finished")
    save(TASK / "run_status.json", state)


def evaluate(args) -> None:
    from src.eval.inference.runner import build_parser, run_evaluation
    from src.eval.inference.hpo_search_baselines import primary_metric
    environment(1337)
    path = TASK / ("baseline_metrics.json" if args.action == "baseline" else "final_metrics.json")
    proc = None
    log = None
    try:
        if args.action == "final":
            log = (TASK / "final_server.log").open("w")
            proc = subprocess.Popen(["bash", str(ROOT / "src/eval/inference/bin/launch_supervised_server.sh"), str(TASK / "start_server.sh")], cwd=TASK, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        if proc is not None:
            deadline = time.monotonic() + 1200
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    raise RuntimeError(f"Final server exited before readiness: {proc.returncode}")
                try:
                    with urllib.request.urlopen("http://127.0.0.1:30080/v1/models", timeout=3) as response:
                        if response.status == 200:
                            break
                except Exception:
                    pass
                time.sleep(2)
            else:
                raise RuntimeError("Final server readiness deadline expired")
        parsed = build_parser().parse_args(["--server-url", "http://127.0.0.1:30080", "--model", "",
                                           "--seed", "1337", "--server-wait-s", "1200",
                                           "--request-timeout-s", "600", "--json-output-file", str(path)])
        metrics = run_evaluation(SCENARIO, parsed)
        metric, source = primary_metric(metrics, SCENARIO.name)
        valid = metrics.get("performance_check", {}).get("pass") is True and not metrics.get("error")
        summary = {"finished_at": now(), "status": "completed" if valid else "failed",
                   "primary_metric": metric if valid else None, "primary_metric_source": source,
                   "quality_evaluated": False, "performance_passed": valid,
                   "metrics_path": str(path), "error": metrics.get("error"),
                   "profiles": metrics.get("profiles", {})}
        save(TASK / ("baseline_summary.json" if args.action == "baseline" else "summary.json"), summary)
        print(json.dumps(summary), flush=True)
        if not valid:
            raise RuntimeError("Final performance validation failed; see saved metrics")
    finally:
        if proc is not None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=15)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        if log:
            log.close()


def hpo(args) -> None:
    environment()
    command = [sys.executable, "-u", "-m", "src.eval.inference.hpo_search_baselines",
               "--method", args.method, "--engine", "sglang", "--scenario", "c",
               "--base-model", MODEL, "--max-model-len", "16384", "--seed-dev", "21",
               "--seed-eval", "1337", "--budget-s", str(args.hours * 3600),
               "--search-space", str(ROOT / "recipes/gb200_r1/sglang.json"),
               "--allow-unsafe-search-space", "--out-root", str(TASK / "hpo"),
               "--first-server-start-timeout-s", "1200", "--server-start-timeout-s", "1200",
               "--final-server-start-timeout-s", "1200", "--quick-eval-timeout-s", "600",
               "--final-eval-timeout-s", "3600"]
    if args.max_trials is not None:
        command += ["--max-trials", str(args.max_trials)]
    state = {"method": args.method, "started_at": now(), "budget_seconds": args.hours * 3600,
             "status": "optimizing", "quality_evaluated": False, "command": command}
    save(TASK / "run_status.json", state)
    result = subprocess.run(command, cwd=ROOT)
    state.update(ended_at=now(), exit_code=result.returncode, status="finished" if result.returncode == 0 else "failed")
    save(TASK / "run_status.json", state)
    if result.returncode:
        raise SystemExit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "agent", "final", "baseline", "hpo"))
    parser.add_argument("--model", choices=("gpt-6.1-sol", "gpt-6-astra"))
    parser.add_argument("--method", choices=("random", "smac"))
    parser.add_argument("--hours", type=float, default=2)
    parser.add_argument("--max-trials", type=int)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    elif args.action == "agent":
        if not args.model:
            parser.error("agent requires --model")
        agent(args)
    elif args.action == "hpo":
        if not args.method:
            parser.error("hpo requires --method")
        hpo(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
