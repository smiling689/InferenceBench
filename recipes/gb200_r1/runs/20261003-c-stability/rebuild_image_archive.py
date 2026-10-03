"""Recover an exact committed image when native Docker save cannot read a base layer.

Run in a CPU container after the node's benchmarks finish. Mount the existing
archive directory at /archives, Docker image/overlay2 read-only at /docker-image,
and Docker overlay2 read-only at /docker-layers. Arguments: run_id, image_id,
base_image_id. Reuse archived base layers and reconstruct the one new layer
from Docker's tar-split records. No Docker storage files are modified.
"""

import base64
import gzip
import hashlib
import io
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


def add_bytes(archive, name, payload):
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    info.mode = 0o644
    archive.addfile(info, io.BytesIO(payload))


run_id, image_id, base_image_id = sys.argv[1:]
assert run_id in {"20261003-c-r2", "20261003-c-r3"}
archives = Path("/archives")
image_store = Path("/docker-image")
config = (image_store / "imagedb/content/sha256" / image_id.removeprefix("sha256:")).read_bytes()
assert "sha256:" + hashlib.sha256(config).hexdigest() == image_id
image = json.loads(config)
diff_ids = image["rootfs"]["diff_ids"]
base_path = archives / "inferencebench-gb200-20260930.tar"
with tarfile.open(base_path) as archive:
    base_manifest = json.load(archive.extractfile("manifest.json"))
    assert len(base_manifest) == 1
    base_manifest = base_manifest[0]
    base_config = archive.extractfile(base_manifest["Config"]).read()
    assert "sha256:" + hashlib.sha256(base_config).hexdigest() == base_image_id
    base_ids = json.loads(base_config)["rootfs"]["diff_ids"]
assert diff_ids[:-1] == base_ids
assert len(base_manifest["Layers"]) == len(base_ids)

chain = diff_ids[0]
for diff_id in diff_ids[1:]:
    chain = "sha256:" + hashlib.sha256((chain + " " + diff_id).encode()).hexdigest()
layerdb = image_store / "layerdb/sha256" / chain.removeprefix("sha256:")
cache_id = (layerdb / "cache-id").read_text().strip()
assert cache_id.isalnum()
layer_dir = Path("/docker-layers") / cache_id / "diff"
delta = archives / f"inferencebench-final-{run_id}-gpt-6.1-sol.delta.tar"
partial_delta = delta.with_suffix(".tar.partial")
assert not delta.exists() and not partial_delta.exists()
with gzip.open(layerdb / "tar-split.json.gz", "rt") as records, partial_delta.open("xb") as output:
    for position, line in enumerate(records):
        entry = json.loads(line)
        assert entry["position"] == position
        if entry["type"] == 2:
            output.write(base64.b64decode(entry["payload"]))
        else:
            assert entry["type"] == 1
            size = entry.get("size", 0)
            if size:
                name = Path(entry["name"])
                assert not name.is_absolute() and ".." not in name.parts
                source = layer_dir / name
                assert source.stat().st_size == size, str(name)
                with source.open("rb") as stream:
                    remaining = size
                    while remaining:
                        chunk = stream.read(min(8 * 1024 * 1024, remaining))
                        assert chunk
                        output.write(chunk)
                        remaining -= len(chunk)
assert "sha256:" + file_digest(partial_delta) == diff_ids[-1]
partial_delta.rename(delta)

image_path = archives / f"inferencebench-final-{run_id}-gpt-6.1-sol.tar"
partial_image = image_path.with_suffix(".tar.partial")
assert not image_path.exists() and not partial_image.exists()
config_name = "blobs/sha256/" + image_id.removeprefix("sha256:")
delta_name = "blobs/sha256/" + diff_ids[-1].removeprefix("sha256:")
manifest = [{
    "Config": config_name,
    "RepoTags": [f"inferencebench-final:{run_id}-gpt-6.1-sol"],
    "Layers": base_manifest["Layers"] + [delta_name],
}]
with tarfile.open(base_path) as source, tarfile.open(partial_image, "w") as output:
    for name in dict.fromkeys(base_manifest["Layers"]):
        member = source.getmember(name)
        assert member.isfile()
        output.addfile(member, source.extractfile(member))
    output.add(delta, arcname=delta_name)
    add_bytes(output, config_name, config)
    add_bytes(output, "manifest.json", json.dumps(manifest).encode())

verified = {}
with tarfile.open(partial_image) as archive:
    assert json.load(archive.extractfile("manifest.json")) == manifest
    assert "sha256:" + digest(archive.extractfile(config_name)) == image_id
    for name, expected in zip(manifest[0]["Layers"], diff_ids):
        if name not in verified:
            stream = archive.extractfile(name)
            magic = stream.read(2)
            stream.seek(0)
            if magic == b"\x1f\x8b":
                stream = gzip.GzipFile(fileobj=stream)
            verified[name] = "sha256:" + digest(stream)
        assert verified[name] == expected, name
partial_image.rename(image_path)
result = {
    "run_id": run_id,
    "original_image_id": image_id,
    "recovered_image_id": image_id,
    "image_id_unchanged": True,
    "base_image_id": base_image_id,
    "base_archive": str(base_path),
    "archive": str(image_path),
    "bytes": image_path.stat().st_size,
    "sha256": file_digest(image_path),
    "rootfs_layers": len(diff_ids),
    "all_layer_diff_ids_verified": True,
    "new_layer_diff_id": diff_ids[-1],
    "new_layer_tar_bytes": delta.stat().st_size,
    "method": "archived base layers plus exact tar-split reconstruction of committed delta",
    "docker_storage_modified": False,
}
(archives / f"image-rebuild-{run_id}.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
