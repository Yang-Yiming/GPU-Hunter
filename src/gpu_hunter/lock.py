"""进程级单实例锁:防止同时开多个 gpu-hunter 双倍占卡。"""
from __future__ import annotations

import fcntl
import os
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def instance_lock(name: str = "gpu-hunter"):
    cache = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    d = Path(cache) / name
    d.mkdir(parents=True, exist_ok=True)
    fh = open(d / "instance.lock", "w")
    try:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"检测到另一个 {name} 实例正在运行(锁文件:{d / 'instance.lock'})")
        fh.write(str(os.getpid()))
        fh.flush()
        fh.truncate()
        yield
    finally:
        try:
            fcntl.flock(fh, fcntl.LOCK_UN)
        finally:
            fh.close()
