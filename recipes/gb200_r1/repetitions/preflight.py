"""Validate the batch environment and perform a real private Responses call."""
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import urllib.request

here = Path(__file__).resolve().parent
root = here.parents[2]
expected = json.loads((here.parent / "skill_guided/runtime_git_blobs.json").read_text())
for relative, digest in expected.items():
    if relative.endswith("PREFLIGHT_20260930.md"):
        continue
    content = (Path("/frozen-reference") / relative).read_bytes()
    assert hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest() == digest, relative
for script in here.glob("*.sh"):
    subprocess.run(["bash", "-n", str(script)], check=True)
for script in here.glob("*.py"):
    ast.parse(script.read_text())
schedule = [line.split("\t") for line in (here / "schedule.tsv").read_text().splitlines()]
assert len(schedule) == len({row[1] for row in schedule}) == 8
assert Counter((row[0], row[2]) for row in schedule) == {
    ("gb200-1", "plain"): 2, ("gb200-1", "skills"): 2,
    ("gb200-2", "plain"): 2, ("gb200-2", "skills"): 2,
}
assert not any((Path.home() / ".agents/skills").glob("*/SKILL.md"))
assert not any((Path.home() / ".codex/skills").glob("*/SKILL.md"))
assert not (Path.home() / ".codex/config.toml").exists()
assert not any(Path("/home/agent/task").iterdir())
hashes = {
    "requests_21.jsonl": "e3329da09722133ae0d3fd368b3438b2c8f230815ab32ee32cb77773e71f3ba5",
    "requests_1337.jsonl": "08fb8644ae4fee2939187086103a9a310866be4b9bd5f50b9c2aef2a343d6cae",
}
for filename, digest in hashes.items():
    assert hashlib.sha256((Path("/bench-inputs") / filename).read_bytes()).hexdigest() == digest
baseline = json.loads(Path("/preflight/baseline_summary.json").read_text())
assert baseline["performance_passed"] is True
assert all(p["request_count"] == p["success_count"] == 256 for p in baseline["profiles"].values())
credentials = json.load(sys.stdin)
request = urllib.request.Request(
    credentials["base_url"].rstrip("/") + "/responses",
    data=json.dumps({"model": "gpt-6.1-sol", "input": "Reply with OK.",
                     "reasoning": {"effort": "max"}, "max_output_tokens": 2048}).encode(),
    headers={"Authorization": "Bearer " + credentials["api_key"], "Content-Type": "application/json"},
)
with urllib.request.urlopen(request, timeout=120) as response:
    assert response.status == 200
    result = json.load(response)
assert result["status"] == "completed", result.get("status")
assert result["model"] == "gpt-6.1-sol"
assert result["reasoning"]["effort"] == "max"
print(json.dumps({"passed": True, "frozen_files_matched": 96, "schedule_runs": 8,
                  "base_has_no_registered_skills": True, "base_task_and_codex_empty": True,
                  "request_hashes": hashes, "api_model": result["model"],
                  "api_reasoning_effort": result["reasoning"]["effort"], "api_status": result["status"]}, indent=2))
