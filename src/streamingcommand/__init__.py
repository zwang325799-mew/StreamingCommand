"""Public API. Importing StreamingCommand does not load models or initialize CUDA."""
from .config import ASRConfig, CommandConfig, StreamingConfig
from .pipeline import StreamingASR, StreamingCommand
from .audio import microphone, audio_file

__version__ = "0.1.0"
__all__ = ["StreamingASR", "StreamingCommand", "ASRConfig", "CommandConfig",
           "StreamingConfig", "microphone", "audio_file", "CandidateProfiles"]


def __getattr__(name):
    if name == "CandidateProfiles":
        from ._runtime.candidate_configs import CandidateConfigs
        return CandidateConfigs
    raise AttributeError(name)
