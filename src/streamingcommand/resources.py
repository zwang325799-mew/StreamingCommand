"""Pinned model downloads and offline checks; no model download during pip/import."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from filelock import FileLock
from .config import BUNDLED_CHECKPOINTS, CHECKPOINTS, PUBLISHED_MODEL
from ._runtime.paths import ASSETS, CACHE, model_path


def sha256(path):
    """Hash a file incrementally so multi-gigabyte model weights need not fit in memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ModelStore:
    def __init__(self, cache_dir=None, *, offline=False):
        """Read the bundled model manifest and record the cache root and offline policy."""
        self.root = Path(cache_dir or CACHE) / "models"
        self.offline = offline
        self.manifest = json.loads((ASSETS / "models.json").read_text(encoding="utf-8"))
        self._sources = {}

    @classmethod
    def from_config(cls, config):
        """Use the same resource selection for startup, doctor and download."""
        store = cls(config.cache_dir, offline=config.offline)
        store._sources = {name: dict(entry) for name, entry in config.models.items()}
        asr, cmd = config.asr, config.command
        legacy_paths = {"asr": asr.model_dir, "asr-base": asr.alternate_model_dir,
                        "language": asr.language_model_dir, "speech": asr.speech_model_dir,
                        "llm-base": cmd.base_model_dir, cmd.model: cmd.checkpoint}
        for name, path in legacy_paths.items():
            if path:
                if name in store._sources:
                    raise ValueError(f"Choose either models.{name} or its legacy local path")
                store._sources[name] = {"source": "local", "path": str(Path(path).absolute())}
        # Older workers may select a bundled checkpoint through STREAMINGCOMMAND_MODELS.
        for name in BUNDLED_CHECKPOINTS:
            path = model_path(name)
            if name not in store._sources and path != ASSETS / "models" / name:
                store._sources[name] = {"source": "local", "path": str(path.absolute())}
        return store

    def _default_source(self, name):
        """Resolve a known resource to its pinned upstream entry or its packaged checkpoint."""
        if name in self.manifest:
            spec = self.manifest[name]
            entry = {"source": "huggingface", "url": "https://huggingface.co/" + spec["repo_id"],
                     "revision": spec["revision"]}
            if spec.get("prefix"):
                entry["subfolder"] = spec["prefix"].rstrip("/")
            if "expected_step" in spec:
                entry["expected_step"] = spec["expected_step"]
            return entry
        if name in BUNDLED_CHECKPOINTS:
            return {"source": "package", "path": f"assets/models/{name}"}
        raise ValueError(f"Unknown model resource: {name}")

    def source(self, name):
        """Effective source without starting workers or downloading weights."""
        return dict(self._sources.get(name) or self._default_source(name))

    def ensure(self, name, override=None):
        """Materialize a configured resource and check the required files and optional checkpoint step."""
        entry = self.source(name) if not override else {"source": "local", "path": override}
        path = self._resolve(name, entry)
        if name in CHECKPOINTS:
            files = ["experiment.json", "head.safetensors"]
            adapter = path if name == PUBLISHED_MODEL else path / "adapter"
            required = [adapter / "adapter_config.json", adapter / "adapter_model.safetensors"]
            if name != PUBLISHED_MODEL:
                files.append("clarify_head.pt")
            required.extend(path / filename for filename in files)
            for file in required:
                if not file.is_file():
                    raise FileNotFoundError(f"Incompatible checkpoint for {name}: missing {file}")
            if "expected_step" in entry:
                metadata = json.loads((path / "experiment.json").read_text(encoding="utf-8"))
                step = metadata.get("global_step" if name == PUBLISHED_MODEL else "step")
                if step != entry["expected_step"]:
                    raise ValueError(f"{name}: expected step {entry['expected_step']}, found {step} in {path}")
        return path

    def _resolve(self, name, entry):
        """Resolve directories, bundled files, or pinned downloads with atomic cache replacement."""
        if entry["source"] == "local":
            path = Path(entry["path"]).expanduser().absolute()
            if not path.is_dir():
                raise FileNotFoundError(f"Model directory missing for {name}: {path}")
            return path
        if entry["source"] == "package":
            packaged = ASSETS / "models" / name
            integrity = json.loads((packaged / "integrity.json").read_text())
            for filename, expected in integrity["weights"].items():
                if sha256(packaged / filename) != expected:
                    raise ValueError(f"Packaged checkpoint checksum mismatch: {name}/{filename}")
            return packaged
        default = self._default_source(name)
        if (entry["url"], entry["revision"], entry.get("subfolder", "")) != (
                default.get("url"), default.get("revision"), default.get("subfolder", "")):
            from huggingface_hub import snapshot_download
            folder = entry.get("subfolder", "")
            snapshot = snapshot_download(
                repo_id=entry["url"].removeprefix("https://huggingface.co/"),
                revision=entry["revision"], cache_dir=str(self.root.parent / "hub"),
                allow_patterns=f"{folder}/*" if folder else None, local_files_only=self.offline)
            path = Path(snapshot) / folder
            if not path.is_dir():
                raise FileNotFoundError(f"Missing {name} directory in pinned snapshot: {path}")
            return path
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
