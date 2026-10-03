"""Archive completed runs in a CPU Docker container after GPU evaluation.

Mount the node repository read-only at /repo and the persistent archive
directory at /archives. Docker image saves must already exist there.
Arguments: node run_id mechanical_method agent_image_id base_image_id.
No Codex home or credentials are archived.
"""

import datetime
import hashlib
import json
import sys
import tarfile
from pathlib import Path


def digest(stream):
    value = hashlib.sha256()
    while chunk := stream.read(8 * 1024 * 1024):
        value.update(chunk)
    return value.hexdigest()


def file_digest(path):
    with path.open("rb") as stream:
        return digest(stream)


def verify_image(path, image_id):
    with tarfile.open(path) as archive:
        manifest = json.load(archive.extractfile("manifest.json"))
        assert len(manifest) == 1
        config = archive.extractfile(manifest[0]["Config"]).read()
        assert "sha256:" + hashlib.sha256(config).hexdigest() == image_id
    return {
        "path": str(Path("/srv/shrinkwrap-data/inferencebench/images") / path.name),
        "bytes": path.stat().st_size,
        "sha256": file_digest(path),
        "image_id": image_id,
        "manifest_config_verified": True,
    }


node, run_id, method, agent_image_id, base_image_id = sys.argv[1:]
assert node in {"gb200-1", "gb200-2"} and method in {"random", "smac"}
assert run_id in {"20261003-c-r2", "20261003-c-r3"}
repo = Path("/repo")
run = repo / "results/gb200_r1" / run_id
metadata = repo / "data/model_metadata/deepseek-ai_DeepSeek-R1"
archives = Path("/archives")
agent_task = run / "gpt-6.1-sol-max/task"
hpo_task = run / method / "task"
hpo_final = hpo_task / "hpo" / method / "sglang/scenario_c/seed_0_artifacts/final"
assert json.loads((agent_task / "result_audit.json").read_text())["audit_passed"]
assert json.loads((hpo_final / "result_audit.json").read_text())["audit_passed"]
assert (run / method / "container_exit_code").read_text().strip() == "0"

task_path = archives / f"inferencebench-tasks-{run_id}-{node}.tar.gz"
assert not task_path.exists(), "Refuse to overwrite an existing recovery archive"
partial = task_path.with_suffix(task_path.suffix + ".partial")
assert not partial.exists(), "Inspect an interrupted archive before retrying"


def include(member):
    parts = Path(member.name).parts
    if any(part in {"codex", ".codex", "__pycache__", ".git"} for part in parts):
        return None
    if member.name.endswith(("/auth.json", "/smiling-api.json")):
        return None
    assert not Path(member.name).is_absolute() and ".." not in parts
    return member


with tarfile.open(partial, "w:gz", compresslevel=1) as archive:
    for source in (run, metadata):
        archive.add(source, arcname=str(source.relative_to(repo)), filter=include)

verified_files = 0
with tarfile.open(partial, "r:gz") as archive:
    names = set()
    for member in archive:
        names.add(member.name)
        assert not any(part in {"codex", ".codex"} for part in Path(member.name).parts)
        if member.isfile():
            assert digest(archive.extractfile(member)) == file_digest(repo / member.name), member.name
            verified_files += 1
    for source in (
        agent_task / "start_server.sh", agent_task / "result_audit.json",
        hpo_final / "result_audit.json", hpo_final / "final_metrics.json",
        run / "shared/requests_21.jsonl", run / "shared/requests_1337.jsonl",
        metadata / "config.json", metadata / "tokenizer.json",
    ):
        assert str(source.relative_to(repo)) in names
partial.rename(task_path)

result = {
    "node": node,
    "run_id": run_id,
    "verified_at_node_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "runtime_commit": "67a9cf389bd7568afac740e0e7cb8b6edcddc59b",
    "base_image": verify_image(archives / "inferencebench-gb200-20260930.tar", base_image_id),
    "agent_image": verify_image(archives / f"inferencebench-final-{run_id}-gpt-6.1-sol.tar", agent_image_id),
    "task_archive": {
        "path": str(Path("/srv/shrinkwrap-data/inferencebench/images") / task_path.name),
        "bytes": task_path.stat().st_size,
        "sha256": file_digest(task_path),
        "methods": ["gpt-6.1-sol-max", method],
        "verified_regular_files": verified_files,
        "all_archived_regular_files_match_source": True,
        "codex_home_excluded": True,
    },
    "mechanical_runtime": "Use archived base image and task-local runtime_cache; ephemeral compiler caches can regenerate.",
    "restore_note": "Restore archive paths relative to the repository root alongside frozen source. Archives contain dummy model metadata and tokenizer, no model weights or datasets.",
}
(archives / f"recovery-{run_id}.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
