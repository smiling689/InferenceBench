"""Read formal server metadata inside its container, without generation traffic."""
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import urllib.request

task = Path("/home/agent/task")
target = task / "formal_server_info.json"
assert not target.exists(), "Preserve existing metadata"
deadline = time.monotonic() + 1250
while time.monotonic() < deadline:
    try:
        with urllib.request.urlopen("http://127.0.0.1:30080/get_server_info", timeout=5) as response:
            info = json.load(response)
    except (OSError, ValueError):
        time.sleep(3)
        continue
    args = info.get("server_args", info)
    assert not args.get("api_key"), "Do not archive authentication keys"
    processes = []
    for process in Path("/proc").iterdir():
        if not process.name.isdigit():
            continue
        try:
            command = (process / "cmdline").read_bytes().decode().rstrip("\x00").split("\x00")
        except (OSError, UnicodeError):
            continue
        if "sglang.launch_server" in command:
            processes.append({"pid": int(process.name), "command": command})
    assert processes, "The observed server must belong to this formal container"
    target.write_text(json.dumps(info, indent=2) + "\n")
    (task / "formal_observation.json").write_text(json.dumps({
        "captured_at_node_utc": datetime.now(timezone.utc).isoformat(),
        "processes": processes, "generation_requests_sent": 0,
    }, indent=2) + "\n")
    print("formal server metadata captured", flush=True)
    break
else:
    raise TimeoutError("Formal server metadata was unavailable")
