"""Capture formal server metadata without changing the original HPO workflow.

Run via docker exec -i <active-container> python3 - <random|smac>, with
this file on stdin. The final/server.log sentinel separates the formal
server from quick trials. No additional generation requests are sent.
"""

import datetime
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


method = sys.argv[1]
assert method in {"random", "smac"}
final = Path("/home/agent/task/hpo") / method / "sglang/scenario_c/seed_0_artifacts/final"
info_path = final / "formal_server_info.json"
assert not info_path.exists(), "Preserve existing metadata rather than overwrite it."

while True:
    if (final / "server.log").exists():
        for process in Path("/proc").iterdir():
            if not process.name.isdigit():
                continue
            try:
                argv = process.joinpath("cmdline").read_bytes().decode().rstrip("\x00").split("\x00")
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                continue
            if "sglang.launch_server" not in argv or "--port" not in argv:
                continue
            port = int(argv[argv.index("--port") + 1])
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/get_server_info", timeout=10) as response:
                    info = json.load(response)
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
                continue
            args = info.get("server_args", info)
            assert args["port"] == port and args["load_format"] == "dummy"
            assert args["quantization"] == "fp8" and args["tp_size"] == 4
            assert not args.get("api_key"), "Do not archive an authenticated server's key."
            captured_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
            observation = {
                "method": method,
                "captured_at_node_utc": captured_at,
                "process_id": int(process.name),
                "server_url": f"http://127.0.0.1:{port}",
                "command": argv,
                "formal_server_log": str(final / "server.log"),
            }
            temporary = info_path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(info, indent=2) + "\n")
            os.replace(temporary, info_path)
            (final / "formal_observation.json").write_text(json.dumps(observation, indent=2) + "\n")
            print(json.dumps({"captured": True, "method": method, "port": port, "captured_at_node_utc": captured_at}), flush=True)
            sys.exit(0)
    time.sleep(30)
