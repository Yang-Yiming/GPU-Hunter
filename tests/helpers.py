"""Shared test fixtures: fake probe, GpuInfo builder, HuntConfig factory, output collector."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gpu_hunter.probe import GpuInfo  # noqa: E402
from gpu_hunter.watcher import HuntConfig  # noqa: E402


def gpu(index, used=100, util=0, pids=(), total=81920):
    return GpuInfo(
        index=index,
        name=f"fake-{index}",
        mem_used_mb=used,
        mem_total_mb=total,
        util_percent=util,
        compute_pids=tuple(pids) if pids is not None else None,
    )


class FakeProbe:
    """Return preset frames in order, then stay on the last frame."""

    def __init__(self, frames):
        self._frames = list(frames)
        self.calls = 0

    def snapshot(self):
        frame = self._frames[min(self.calls, len(self._frames) - 1)]
        self.calls += 1
        return frame


def cfg(**kw):
    base = dict(interval_s=0.005, stable_rounds=2, max_polls=200)
    base.update(kw)
    return HuntConfig(**base)


class Collector(list):
    def __call__(self, msg):
        self.append(msg)
