"""共享的参数/字段解析:容量、GPU 编号列表、脚本解释器补全。"""
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
    """'500'→500MB,'40G'/'40GiB'→40960MB;非法输入抛 ValueError。"""
    if not isinstance(text, str):
        raise ValueError(f"容量必须是字符串:{text!r}")
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([A-Za-z]*)\s*", text)
    if not m:
        raise ValueError(f"无法解析容量:{text!r}(示例 500 / 24000M / 40G)")
    unit = m.group(2).upper()
    if unit not in _UNITS:
        raise ValueError(f"未知容量单位:{m.group(2)!r}")
    return math.ceil(float(m.group(1)) * _UNITS[unit] / 2**20)


def parse_int_list(value) -> list[int]:
    """'[2,3]' 或 \"2,3\" → [2,3];去重保序;非法抛 ValueError。"""
    items: list[int] = []
    tokens: list[str]
    if isinstance(value, str):
        tokens = value.split(",")
    elif isinstance(value, (list, tuple)):
        tokens = [str(v) for v in value]
    else:
        raise ValueError(f"GPU 编号列表格式非法:{value!r}")
    for tok in tokens:
        tok = tok.strip()
        if not tok:
            continue
        try:
            v = int(tok)
        except ValueError:
            raise ValueError(f"非法 GPU 编号:{tok!r}") from None
        if v < 0:
            raise ValueError(f"GPU 编号不能为负:{v}")
        if v not in items:
            items.append(v)
    if not items:
        raise ValueError("GPU 编号列表为空")
    return items


def ensure_interpreter(cmd: list[str]) -> list[str]:
    """单个 xx.py/xx.sh 自动补解释器,其余原样返回。"""
    if len(cmd) == 1:
        s = cmd[0]
        if s.endswith(".py"):
            return [sys.executable, s]
        if s.endswith(".sh"):
            return ["bash", s]
    return cmd


def coerce_cmd(value, ctx: str) -> list[str]:
    """cmd 字段:字符串按 shlex 切分,或字符串列表;空则报错。"""
    if isinstance(value, str):
        cmd = shlex.split(value)
    elif isinstance(value, (list, tuple)) and all(isinstance(v, str) for v in value):
        cmd = list(value)
    else:
        raise ValueError(f"{ctx}: cmd 必须是字符串或字符串列表")
    if not cmd:
        raise ValueError(f"{ctx}: cmd 不能为空")
    return cmd


def coerce_path(value, ctx: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{ctx}: 路径必须是非空字符串")
    return Path(value)
