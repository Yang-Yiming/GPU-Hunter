"""Shared argument/field parsers: sizes, GPU id lists, script interpreter completion."""
from __future__ import annotations

import math
import re
import shlex
import sys
from pathlib import Path

_UNITS = {
    "": 2**20, "B": 1,
    "K": 2**10, "KB": 2**10, "KIB": 2**10,
    "M": 2**20, "MB": 2**20, "MIB": 2**20,
    "G": 2**30, "GB": 2**30, "GIB": 2**30,
    "T": 2**40, "TB": 2**40, "TIB": 2**40,
}


def parse_size_mb(text: str) -> int:
    """'500' -> 500MB, '40G'/'40GiB' -> 40960MB; invalid input raises ValueError."""
    if not isinstance(text, str):
        raise ValueError(f"size must be a string: {text!r}")
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([A-Za-z]*)\s*", text)
    if not m:
        raise ValueError(f"cannot parse size: {text!r} (examples: 500 / 24000M / 40G)")
    unit = m.group(2).upper()
    if unit not in _UNITS:
        raise ValueError(f"unknown size unit: {m.group(2)!r}")
    return math.ceil(float(m.group(1)) * _UNITS[unit] / 2**20)


def parse_int_list(value) -> list[int]:
    """'[2,3]' or "2,3" -> [2,3]; de-duplicates while preserving order; invalid raises ValueError."""
    items: list[int] = []
    tokens: list[str]
    if isinstance(value, str):
        tokens = value.split(",")
    elif isinstance(value, (list, tuple)):
        tokens = [str(v) for v in value]
    else:
        raise ValueError(f"invalid GPU id list: {value!r}")
    for tok in tokens:
        tok = tok.strip()
        if not tok:
            continue
        try:
            v = int(tok)
        except ValueError:
            raise ValueError(f"invalid GPU id: {tok!r}") from None
        if v < 0:
            raise ValueError(f"GPU id must not be negative: {v}")
        if v not in items:
            items.append(v)
    if not items:
        raise ValueError("GPU id list is empty")
    return items


def ensure_interpreter(cmd: list[str]) -> list[str]:
    """Add an interpreter for a lone xx.py/xx.sh; return anything else unchanged."""
    if len(cmd) == 1:
        s = cmd[0]
        lower = s.lower()
        if lower.endswith(".py"):
            return [sys.executable, s]
        if lower.endswith(".sh"):
            return ["bash", s]
    return cmd


def coerce_cmd(value, ctx: str) -> list[str]:
    """cmd field: a string split with shlex, or a list of strings; empty is an error."""
    if isinstance(value, str):
        cmd = shlex.split(value)
    elif isinstance(value, (list, tuple)) and all(isinstance(v, str) for v in value):
        cmd = list(value)
    else:
        raise ValueError(f"{ctx}: cmd must be a string or a list of strings")
    if not cmd:
        raise ValueError(f"{ctx}: cmd must not be empty")
    return cmd


def coerce_path(value, ctx: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{ctx}: path must be a non-empty string")
    return Path(value).expanduser()
