"""Async adapter for the resident, trained one-forward semantic parser."""
from __future__ import annotations

import asyncio
import copy
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

from .paths import candidate_root
DEFAULT_CANDIDATES = candidate_root() / "command_candidates.json"


def validate_candidates(value):
    if not isinstance(value, dict) or set(value) != {"I", "T", "P"}:
        raise ValueError("candidates must contain independent I/T/P lists")
    for slot, values in value.items():
        if (not isinstance(values, list) or (not values and slot != "P") or
                any(not isinstance(v, str) or not v.strip() or v == "__unresolved__" for v in values) or
                len(set(values)) != len(values)):
            raise ValueError(f"Invalid {slot} candidate list")
    if not value["P"]:
        if not set(value["I"]) <= {"give", "stop", "reset"}:
            raise ValueError("Empty P requires give/stop/reset candidates")
    elif any("none" not in value[s] for s in ("T", "P")):
        raise ValueError("T/P candidate lists must include business value none")
    return copy.deepcopy(value)


def validate_history(value):
    if value is None:
        return {}
    if not isinstance(value, dict) or set(value)-{"H1", "H2"} or any(not isinstance(v, str) for v in value.values()):
        raise ValueError("history must map H1/H2 to text")
    return {key: value[key] for key in ("H2", "H1") if value.get(key)}

