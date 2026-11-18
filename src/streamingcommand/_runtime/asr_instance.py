"""One ASR owner per user, shared between independent installs and Demo adapters."""
import json
import os
from pathlib import Path
from filelock import FileLock, Timeout
from platformdirs import user_cache_path


class ASRInstanceLease:
    def __init__(self, device, *, lock_path=None, **kwargs):
        self.device = device
        self.path = Path(lock_path or Path(user_cache_path("StreamingCommand")) / "asr.lock")
        self._lock = None

    def acquire(self):
        if self._lock is not None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock = FileLock(str(self.path))
        try:
            lock.acquire(timeout=0)
        except Timeout as exc:
            raise RuntimeError("An ASR instance is already active; close it before starting another") from exc
        self._lock = lock

    def release(self):
        if self._lock is not None:
            self._lock.release()
            self._lock = None
