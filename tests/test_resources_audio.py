import json
from pathlib import Path
import numpy as np
import pytest
from streamingcommand.audio import audio_file
from streamingcommand.config import StreamingConfig
from streamingcommand.resources import ModelStore, sha256


@pytest.mark.asyncio
async def test_resampling_preserves_duration_across_blocks(tmp_path):
    import soundfile as sf
    rate = 44100
    audio = np.sin(np.arange(rate * 2) * 2 * np.pi * 440 / rate).astype(np.float32) * .2
    source = tmp_path / "stereo.wav"
    sf.write(source, np.column_stack([audio, audio]), rate)
    chunks = [chunk async for chunk in audio_file(source, realtime=False, block_ms=17)]
    result = np.frombuffer(b"".join(chunks), dtype="<i2")
    assert len(result) == 32000
    assert result.max() > 5000


def test_offline_rejects_corrupt_resources(tmp_path):
    store = ModelStore(tmp_path, offline=True)
    with pytest.raises(FileNotFoundError, match="Offline"):
        store.ensure("asr")


def test_packaged_adapters_are_complete_and_verified():
    store = ModelStore(offline=True)
    for name in ("clarification-new", "clarification-latest"):
        path = store.ensure(name)
        assert (path / "adapter/adapter_model.safetensors").stat().st_size > 1000000
        assert (path / "clarify_head.pt").is_file()


def test_config_paths_and_unknown_fields(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"asr": {"model_dir": "models/asr"}, "command": {"candidates_file": "scrub.json"}}))
    config = StreamingConfig.load(path)
    assert config.asr.model_dir == str(tmp_path / "models/asr")
    assert config.command.candidates_file == str(tmp_path / "scrub.json")
    path.write_text('{"unknown": true}')
    with pytest.raises(ValueError, match="Unknown"):
        StreamingConfig.load(path)
