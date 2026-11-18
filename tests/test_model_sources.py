import json

import pytest

from streamingcommand import CommandConfig, StreamingConfig
from streamingcommand._runtime.paths import ASSETS
from streamingcommand.cli import main
from streamingcommand.config import ASR_RESOURCES, PUBLISHED_MODEL
from streamingcommand.resources import ModelStore


def test_presets_declare_all_active_models():
    folder = ASSETS / "presets"
    local = StreamingConfig.load(folder / "local-latest.json")
    assert local.command.model == "clarification-latest"
    assert local.models[local.command.model]["expected_step"] == 936
    assert local.command.recognition_logic == "semantic-pair" and local.offline
    assert set(local.models) == {*ASR_RESOURCES, "llm-base", local.command.model}
    assert all(s["source"] == "local" and s["path"] for s in local.models.values())
    release = StreamingConfig.load(folder / "huggingface-release.json")
    assert release.command.model == PUBLISHED_MODEL
    assert release.command.recognition_logic == "original"
    assert set(release.models) == {*ASR_RESOURCES, "llm-base", PUBLISHED_MODEL}
    for name, source in release.models.items():
        assert source == ModelStore().source(name)
        assert source["source"] == "huggingface"


@pytest.mark.parametrize("entry", [
    {"source": "local"},
    {"source": "local", "path": ".", "expected_step": 3},
    {"source": "huggingface", "url": "org/repo", "revision": "1" * 40},
    {"source": "huggingface", "url": "https://huggingface.co/org/repo", "revision": "main"},
    {"source": "huggingface", "url": "https://huggingface.co/org/repo", "revision": "1" * 40,
     "subfolder": "../model"},
])
def test_invalid_sources(entry):
    with pytest.raises(ValueError):
        StreamingConfig(models={"asr": entry})


def test_relative_local_path_conflicts_and_missing_files(tmp_path, monkeypatch):
    directory = tmp_path / "models"
    directory.mkdir()
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"models": {"asr": {"source": "local", "path": "models"}}}))
    monkeypatch.chdir(tmp_path.parent)
    config = StreamingConfig.load(path)
    store = ModelStore.from_config(config)
    assert store.ensure("asr") == directory
    assert store.source("asr") == {"source": "local", "path": str(directory)}
    config.asr.model_dir = str(directory)
    with pytest.raises(ValueError, match="either models.asr"):
        ModelStore.from_config(config)
    with pytest.raises(FileNotFoundError, match="Incompatible checkpoint"):
        store.ensure(PUBLISHED_MODEL, override=directory)


def test_local_checkpoint_step_is_checked():
    name = "clarification-latest"
    checkpoint = ASSETS / "models" / name
    store = ModelStore.from_config(StreamingConfig(models={
        name: {"source": "local", "path": str(checkpoint), "expected_step": 936},
    }))
    assert store.ensure(name) == checkpoint
    wrong = ModelStore.from_config(StreamingConfig(models={
        name: {"source": "local", "path": str(checkpoint), "expected_step": 3003},
    }))
    with pytest.raises(ValueError, match="expected step 3003, found 936"):
        wrong.ensure(name)


def test_explicit_source_takes_precedence_over_worker_model_path(tmp_path, monkeypatch):
    from streamingcommand._runtime import paths

    name = "clarification-new"
    monkeypatch.setitem(paths._overrides, name, str(tmp_path))
    assert ModelStore.from_config(StreamingConfig()).source(name) == {"source": "local", "path": str(tmp_path)}
    source = {"source": "package", "path": "assets/models/" + name}
    assert ModelStore.from_config(StreamingConfig(models={name: source})).source(name) == source


def test_custom_snapshot_uses_configured_revision_and_subfolder(tmp_path, monkeypatch):
    import huggingface_hub

    calls = []

    def snapshot_download(**kwargs):
        calls.append(kwargs)
        snapshot = tmp_path / kwargs["revision"]
        (snapshot / "vad").mkdir(parents=True)
        return str(snapshot)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    for rev in ("a" * 40, "b" * 40):
        store = ModelStore.from_config(StreamingConfig(cache_dir=str(tmp_path), offline=True, models={
            "speech": {"source": "huggingface", "url": "https://huggingface.co/owner/repo",
                       "revision": rev, "subfolder": "vad"},
        }))
        assert store.ensure("speech") == tmp_path / rev / "vad"
        assert calls[-1] == dict(repo_id="owner/repo", revision=rev, local_files_only=True,
                                 allow_patterns="vad/*", cache_dir=str(tmp_path / "hub"))


def test_release_cli_shows_model_without_loading_and_downloads_active_resources(tmp_path, monkeypatch, capsys):
    path = ASSETS / "presets" / "huggingface-release.json"
    calls = []

    def ensure(store, name):
        calls.append((name, store.source(name)))
        return tmp_path / name

    monkeypatch.setattr(ModelStore, "ensure", ensure)
    main(["--config", str(path), "doctor"])
    report = json.loads(capsys.readouterr().out)
    assert not calls
    assert report["command_model"] == PUBLISHED_MODEL and report["asr_profile"] == "v15-speech-aware"
    main(["--config", str(path), "download"])
    assert dict(calls) == report["models"]
    with pytest.raises(ValueError, match="original"):
        StreamingConfig(command=CommandConfig(model=PUBLISHED_MODEL, recognition_logic="semantic-pair"))


@pytest.mark.parametrize(("command", "target", "status"), [
    (0, 1, "NON_COMMAND"), (1, 0, "UNRESOLVED"), (1, 1, "EXECUTE"),
])
def test_release_decodes_noncommand_unresolved_and_business_none(command, target, status):
    import torch
    from streamingcommand._runtime.llm_ranker import QwenEncoder
    from streamingcommand._runtime.published_command_runtime import PublishedCommandRuntime

    class Tokenizer:
        pad_token_id = 1

        def encode(self, text, add_special_tokens=False):
            return [1]

    runtime = PublishedCommandRuntime.__new__(PublishedCommandRuntime)
    runtime.device = "cpu"
    runtime.encoder = QwenEncoder(Tokenizer())

    def forward(**kwargs):
        return {"command": torch.tensor([[1 - command, command]]),
                "slots": torch.tensor([[[0, 1], [1 - target, target], [0, 1]]])}

    runtime.model = forward
    result = runtime.parse_turn("Bend.", {"I": ["bend"], "T": ["none"], "P": ["none"]}, {})
    assert result["status"] == status
    assert result["execute"] == (status == "EXECUTE")
    assert result["slots"]["T"] == ("none" if command and target else None)
    assert not result["clarification_required"]
