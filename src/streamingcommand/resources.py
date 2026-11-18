"""Pinned model downloads and offline checks; no model download during pip/import."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from filelock import FileLock
from ._runtime.paths import ASSETS, CACHE


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ModelStore:
    def __init__(self, cache_dir=None, *, offline=False):
        self.root = Path(cache_dir or CACHE) / "models"
        self.offline = offline
        self.manifest = json.loads((ASSETS / "models.json").read_text(encoding="utf-8"))

    def ensure(self, name, override=None):
        if override:
            path = Path(override).absolute()
            if not path.is_dir():
                raise FileNotFoundError(f"Model directory missing: {path}")
            return path
        packaged = ASSETS / "models" / name
        if packaged.is_dir():
            integrity = json.loads((packaged / "integrity.json").read_text())
            for filename, expected in integrity["weights"].items():
                if sha256(packaged / filename) != expected:
                    raise ValueError(f"Packaged checkpoint checksum mismatch: {name}/{filename}")
            return packaged
        spec = self.manifest[name]
        path = self.root / name
        path.mkdir(parents=True, exist_ok=True)
        with FileLock(str(path / ".download.lock")):
            for filename, expected in spec["files"].items():
                destination = path / filename
                if destination.is_file() and destination.stat().st_size == expected["size"] and sha256(destination) == expected["sha256"]:
                    continue
                if self.offline:
                    raise FileNotFoundError(f"Offline model is missing or corrupt: {name}/{filename}")
                from huggingface_hub import hf_hub_download
                source = hf_hub_download(repo_id=spec["repo_id"], revision=spec["revision"], filename=spec.get("prefix", "") + filename, cache_dir=str(self.root.parent / "hub"))
                if sha256(source) != expected["sha256"]:
                    raise ValueError(f"Downloaded model checksum mismatch: {name}/{filename}")
                destination.parent.mkdir(parents=True, exist_ok=True)
                fd, temporary = tempfile.mkstemp(dir=destination.parent)
                os.close(fd)
                try:
                    shutil.copyfile(source, temporary)
                    os.replace(temporary, destination)
                finally:
                    if Path(temporary).exists():
                        Path(temporary).unlink()
        return path
