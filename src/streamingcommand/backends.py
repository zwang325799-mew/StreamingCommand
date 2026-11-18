"""Factory boundary used by both the public API and Demo integration."""
import os
from pathlib import Path
import sys
from .config import StreamingConfig
from .resources import ModelStore


async def build_backends(config: StreamingConfig):
    """Resolve resources off the audio loop, then construct unloaded backends."""
    import asyncio
    from ._runtime import paths
    from ._runtime.device_policy import resolve_device
    store = ModelStore(config.cache_dir, offline=config.offline)
    spec = config.asr
    if spec.backend == "fake":
        from ._runtime.asr_backend import FakeASR
        asr = FakeASR()
    else:
        resources = {}
        for name, override in [("asr", spec.model_dir), ("asr-base", spec.alternate_model_dir),
                               ("language", spec.language_model_dir), ("speech", spec.speech_model_dir)]:
            resources[name] = await asyncio.to_thread(store.ensure, name, override)
        # These model locations are resolved once for the single resident runtime.
        paths._overrides.update({name: str(value) for name, value in resources.items()})
        from ._runtime import acoustic_speech
        acoustic_speech.MODEL_DIR = resources["speech"]
        from ._runtime.asr_defaults import build_asr
        device, _ = resolve_device(spec.device)
        if spec.backend.startswith("vllm") and not device.startswith("cuda:"):
            raise ValueError("vLLM requires a compatible NVIDIA CUDA device; select native for CPU")
        asr = build_asr(spec.profile, device=device)
        if hasattr(asr, "_continuity_asr") and not asr.status.get("asr_engine_count") == 1:
            from ._runtime.shared_asr import configure_shared_engine
            configure_shared_engine(asr)
        asr.model_dir = resources["asr"]
        asr.alternate_model_dir = resources["asr-base"] if hasattr(asr, "_continuity_asr") else None
        if hasattr(asr, "_continuity_asr"):
            asr._continuity_asr.model_dir = resources["asr-base"]
        asr.python = Path(spec.python or sys.executable).absolute()
        if spec.backend == "vllm":
            # Explicit legacy SDK backend. It runs in a separately configured interpreter.
            asr.worker_environment["STREAMINGCOMMAND_RUNTIME_CACHE"] = str(store.root.parent / "runtime")
        else:
            asr.worker = paths.WORKERS / "portable_worker.py"
            asr.worker_environment.update(STREAMINGCOMMAND_ASR_BACKEND=spec.backend)
        asr.warmup = spec.warmup
        asr.gpu_uuid = spec.expected_gpu_uuid or ""
        asr.status.update(device=device, backend=spec.backend, model_dir=str(asr.model_dir), implementation="streamingcommand")
    if config.logs_dir:
        asr.logs_dir = Path(config.logs_dir)
    command = None
    if config.command.enabled:
        llm = config.command
        base = await asyncio.to_thread(store.ensure, "llm-base", llm.base_model_dir)
        from ._runtime.selectable_commands import SelectableCommandParser
        command = SelectableCommandParser(model=llm.model, device=llm.device,
            python=llm.python, base_model_dir=str(base), candidates_file=llm.candidates_file,
            checkpoint_overrides={llm.model: llm.checkpoint} if llm.checkpoint else None,
            use_cuda_graphs=llm.use_cuda_graphs, logs_root=config.logs_dir, expected_gpu_uuid=llm.expected_gpu_uuid)
    return asr, command
