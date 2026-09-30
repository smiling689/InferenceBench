#!/usr/bin/env python3
"""Unscored preparation checks; execute in a Docker container on the GB200 node."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

import requests


def environment() -> dict:
    import torch

    manifest = json.loads(Path(__file__).with_name("model_metadata.json").read_text())
    model_dir = Path("/models/deepseek-r1")
    verified = []
    for name, expected in manifest["files"].items():
        digest = hashlib.sha256((model_dir / name).read_bytes()).hexdigest()
        if digest != expected["sha256"]:
            raise ValueError(f"Metadata checksum mismatch: {name}")
        verified.append(name)
    config = json.loads((model_dir / "config.json").read_text())
    if config["num_hidden_layers"] != 61 or config["n_routed_experts"] != 256:
        raise ValueError("Expected the full 61-layer, 256-expert DeepSeek-R1 configuration")
    if platform.machine() != "aarch64" or torch.cuda.device_count() != 4:
        raise ValueError("Expected an ARM64 container with four visible GPUs")
    devices = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        if "GB200" not in properties.name:
            raise ValueError(f"Unexpected GPU: {properties.name}")
        devices.append({"index": index, "name": properties.name, "memory_bytes": properties.total_memory})
    return {
        "check": "environment",
        "status": "passed",
        "architecture": platform.machine(),
        "packages": {name: importlib.metadata.version(name) for name in ("sglang", "torch", "transformers", "pytest")},
        "cuda": torch.version.cuda,
        "devices": devices,
        "model": manifest["repository"],
        "revision": manifest["revision"],
        "metadata_files_verified": verified,
        "num_hidden_layers": config["num_hidden_layers"],
        "load_format": "dummy",
        "quality_evaluated": False,
        "nvidia_smi": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.used", "--format=csv,noheader"], text=True
        ).strip().splitlines(),
    }


def server(args: argparse.Namespace) -> dict:
    deadline = time.monotonic() + args.wait_seconds
    base = args.server_url.rstrip("/")
    session = requests.Session()
    session.trust_env = False
    while True:
        try:
            response = session.get(base + "/v1/models", timeout=5)
            response.raise_for_status()
            models = [item["id"] for item in response.json()["data"]]
            if "deepseek-ai/DeepSeek-R1" not in models:
                raise ValueError("Server is not advertising the expected DeepSeek-R1 model")
            break
        except requests.RequestException:
            if time.monotonic() >= deadline:
                raise TimeoutError("The local inference server did not become ready") from None
            time.sleep(2)
    payload = {
        "model": "deepseek-ai/DeepSeek-R1",
        "messages": [{"role": "user", "content": "Reply with a short greeting."}],
        "max_tokens": 16,
        "temperature": 0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    started = time.monotonic()
    usage = None
    has_text = False
    finished = False
    with session.post(base + "/v1/chat/completions", json=payload, stream=True, timeout=(10, 300)) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                finished = True
                break
            chunk = json.loads(data)
            if chunk.get("error"):
                raise ValueError("Server returned an error in its SSE stream")
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                delta = choice.get("delta") or {}
                has_text |= bool(delta.get("content") or delta.get("reasoning_content"))
    if not finished or not has_text or not usage or usage.get("completion_tokens") != 16:
        raise ValueError("Streaming smoke failed: expected text, [DONE], and usage reporting exactly 16 output tokens")
    return {
        "check": "server",
        "status": "passed",
        "server_url": base,
        "model": payload["model"],
        "streaming": True,
        "usage": usage,
        "elapsed_seconds": time.monotonic() - started,
        "load_format": "dummy",
        "scored": False,
        "quality_evaluated": False,
    }


def api(args: argparse.Namespace) -> dict:
    credentials = json.load(sys.stdin)
    base = str(credentials["base_url"]).rstrip("/")
    key = str(credentials["api_key"])
    parsed = urlsplit(base)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("The external provider must have a credential-free HTTP(S) base URL")
    if not key:
        raise ValueError("Missing API key")
    session = requests.Session()
    session.headers["Authorization"] = "Bearer " + key
    try:
        if args.action == "api-models":
            response = session.get(base + "/models", timeout=(10, 45), allow_redirects=False)
        else:
            response = session.post(
                base + "/chat/completions",
                json={"model": args.model, "messages": [{"role": "user", "content": "Reply only with OK."}], "max_tokens": 32, "stream": False},
                timeout=(10, 90),
                allow_redirects=False,
            )
        if response.status_code != 200:
            return {"check": args.action, "status": "failed", "http_status": response.status_code, "base_url": base}
        payload = response.json()
        if args.action == "api-models":
            models = sorted(str(item["id"]) for item in payload["data"])
            return {"check": args.action, "status": "passed", "base_url": base, "models": models, "model_count": len(models)}
        choices = payload.get("choices") or []
        if not choices or not choices[0].get("message"):
            raise ValueError("Provider returned no chat completion")
        return {"check": args.action, "status": "passed", "base_url": base, "model": args.model, "usage": payload.get("usage"), "tool_calling_verified": False}
    except requests.RequestException as exc:
        # Do not print headers, credential input, or provider error response bodies.
        return {"check": args.action, "status": "failed", "base_url": base, "error_type": type(exc).__name__}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("environment", "server", "api-models", "api-chat"))
    parser.add_argument("--server-url", default="http://127.0.0.1:30080")
    parser.add_argument("--wait-seconds", type=float, default=1200)
    parser.add_argument("--model")
    args = parser.parse_args()
    if args.action == "api-chat" and not args.model:
        parser.error("api-chat requires an explicit --model from the provider's model list")
    try:
        result = environment() if args.action == "environment" else server(args) if args.action == "server" else api(args)
    except Exception as exc:
        result = {"check": args.action, "status": "failed", "error_type": type(exc).__name__}
        if args.action in {"environment", "server"}:
            result["error"] = str(exc)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
