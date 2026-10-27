"""Process-level single-instance lock: prevents running multiple gpu-hunter copies."""
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
            raise SystemExit(
                f"another {name} instance is already running "
                f"(lock file: {d / 'instance.lock'})"
            )
        fh.write(str(os.getpid()))
        fh.flush()
        fh.truncate()
        yield
    finally:
        try:
            fcntl.flock(fh, fcntl.LOCK_UN)
        finally:
            fh.close()
