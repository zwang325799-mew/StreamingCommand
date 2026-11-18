"""Serializable configuration; independent of application paths and GPU UUIDs."""
from dataclasses import dataclass, field, fields
from pathlib import Path
import json


@dataclass
class ASRConfig:
    backend: str = "native"
    device: str = "auto"
    profile: str = "v15-canto-commit"
    python: str | None = None
    model_dir: str | None = None
    alternate_model_dir: str | None = None
    language_model_dir: str | None = None
    speech_model_dir: str | None = None
    expected_gpu_uuid: str | None = None
    warmup: bool = True

    def __post_init__(self):
        if self.backend not in {"native", "vllm", "vllm-builtin", "fake"}:
            raise ValueError("ASR backend must be native, vllm, vllm-builtin or fake")
        if self.profile == "v15-speech-aware-dual":
            raise ValueError("Parallel ASR engines are unsupported; use v15-speech-aware")


@dataclass
class CommandConfig:
    enabled: bool = True
    model: str = "clarification-new"
    device: str = "auto"
    python: str | None = None
    checkpoint: str | None = None
    base_model_dir: str | None = None
    candidates_file: str | None = None
    recognition_logic: str = "original"
    use_cuda_graphs: bool = True
    expected_gpu_uuid: str | None = None


@dataclass
class StreamingConfig:
    asr: ASRConfig = field(default_factory=ASRConfig)
    command: CommandConfig = field(default_factory=CommandConfig)
    interval_ms: int = 800
    endpoint_ms: int = 240
    max_segment_ms: int = 16000
    event_queue_size: int = 256
    max_pending_audio_ms: int = 30000
    cache_dir: str | None = None
    logs_dir: str | None = None
    offline: bool = False

    def __post_init__(self):
        if self.interval_ms < 20 or self.endpoint_ms < 20 or self.max_segment_ms < self.interval_ms:
            raise ValueError("Invalid streaming intervals")
        if self.event_queue_size < 1 or self.max_pending_audio_ms < self.max_segment_ms:
            raise ValueError("Queue capacities must be positive and cover one maximum segment")

    @classmethod
    def load(cls, path):
        path = Path(path)
        value = json.loads(path.read_text(encoding="utf-8"))
        allowed = {f.name for f in fields(cls)}
        if set(value) - allowed:
            raise ValueError(f"Unknown configuration keys: {sorted(set(value)-allowed)}")
        value["asr"] = ASRConfig(**value.get("asr", {}))
        value["command"] = CommandConfig(**value.get("command", {}))
        # User paths are relative to their config, never to the current cwd.
        for obj, names in [(value["asr"], ("python", "model_dir", "alternate_model_dir", "language_model_dir", "speech_model_dir")),
                           (value["command"], ("python", "checkpoint", "base_model_dir", "candidates_file"))]:
            for name in names:
                item = getattr(obj, name)
                if item and not Path(item).is_absolute():
                    setattr(obj, name, str((path.parent / item).absolute()))
        for name in ("cache_dir", "logs_dir"):
            if value.get(name) and not Path(value[name]).is_absolute():
                value[name] = str((path.parent / value[name]).absolute())
        return cls(**value)
