"""Polling state machine: sliding windows decide "stably idle" and rank GPUs.

AvailabilityTracker keeps the raw sample window per GPU; idle predicates
(strict or with a memory requirement) are applied on read, so the single-job
Hunter and the multi-job Scheduler share the same logic.
"""
from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from typing import Callable, Iterable, Protocol, Sequence

from .probe import GpuInfo


class Probe(Protocol):
    def snapshot(self) -> dict[int, GpuInfo]: ...


@dataclass
class HuntConfig:
    need: int = 1                        # how many GPUs must be free at the same time
    mem_per_gpu_mb: int | None = None    # free memory needed per GPU; None means fully idle
    prefer: tuple[int, ...] = ()         # preference order, e.g. (2, 3)
    allow: frozenset[int] | None = None  # only pick from these GPUs; None means all
    idle_mem_mb: int = 500               # used memory below this counts as idle
    idle_util: int = 5                   # utilization below this counts as idle
    interval_s: float = 5.0              # polling interval
    stable_rounds: int = 3               # consecutive idle rounds required for stability
    max_polls: int | None = None         # poll limit (for tests; None means unlimited)

    def __post_init__(self) -> None:
        if self.need < 1:
            raise ValueError("need must be >= 1")
        if self.interval_s <= 0:
            raise ValueError("interval_s must be > 0")
        if self.stable_rounds < 1:
            raise ValueError("stable_rounds must be >= 1")


def passes(
    info: GpuInfo,
    *,
    idle_mem_mb: int,
    idle_util: int,
    need_mb: int | None = None,
) -> bool:
    """Idle predicate for one sample: compute processes always mean busy, then util and memory."""
    if info.compute_pids:
        return False
    if info.util_percent >= 0 and info.util_percent > idle_util:
        return False
    if need_mb is not None:
        return info.mem_free_mb >= need_mb
    return info.mem_used_mb <= idle_mem_mb


def is_idle(info: GpuInfo, cfg: HuntConfig) -> bool:
    return passes(
        info,
        idle_mem_mb=cfg.idle_mem_mb,
        idle_util=cfg.idle_util,
        need_mb=cfg.mem_per_gpu_mb,
    )


def rank_candidates(ids: Iterable[int], prefer: Sequence[int]) -> list[int]:
    """GPUs in the preference list come first (in given order), the rest by ascending id."""
    order = {g: i for i, g in enumerate(prefer)}
    return sorted(set(ids), key=lambda g: (order.get(g, len(order)), g))


class AvailabilityTracker:
    """Keep a raw sliding window per GPU; occupied GPUs clear their history (must restabilize)."""

    def __init__(
        self,
        idle_mem_mb: int = 500,
        idle_util: int = 5,
        stable_rounds: int = 3,
    ) -> None:
        self.idle_mem_mb = idle_mem_mb
        self.idle_util = idle_util
        self.stable_rounds = stable_rounds
        self.windows: dict[int, deque[GpuInfo]] = {}

    def _ok(self, info: GpuInfo, need_mb: int | None) -> bool:
        return passes(
            info,
            idle_mem_mb=self.idle_mem_mb,
            idle_util=self.idle_util,
            need_mb=need_mb,
        )

    def update(
        self,
        snap: dict[int, GpuInfo],
        occupied: Iterable[int] = (),
    ) -> None:
        occ = set(occupied)
        self.windows = {
            g: w for g, w in self.windows.items() if g in snap and g not in occ
        }
        for gid, info in snap.items():
            if gid in occ:
                continue
            w = self.windows.setdefault(gid, deque(maxlen=self.stable_rounds))
            w.append(info)

    def stable(self, mem_per_gpu_mb: int | None = None) -> set[int]:
        """GPUs that satisfied the idle conditions in all of the last stable_rounds rounds."""
        return {
            g
            for g, w in self.windows.items()
            if len(w) == self.stable_rounds
            and all(self._ok(s, mem_per_gpu_mb) for s in w)
        }


class Hunter:
    """Single-job hunt: block until enough stably idle GPUs are available."""

    def __init__(
        self,
        probe: Probe,
        cfg: HuntConfig,
        emit: Callable[[str], None] | None = None,
        stop_event: threading.Event | None = None,
    ) -> None:
        self.probe = probe
        self.cfg = cfg
        self.emit = emit if emit is not None else print
        self.stop = stop_event if stop_event is not None else threading.Event()

    def hunt(self) -> list[int] | None:
        """Block until enough stable idle GPUs are found and return their ids; None if aborted."""
        cfg = self.cfg
        tracker = AvailabilityTracker(cfg.idle_mem_mb, cfg.idle_util, cfg.stable_rounds)
        polls = 0
        last_status = ""
        err_streak = 0
        warned_refs = False

        while not self.stop.is_set():
            try:
                snap = self.probe.snapshot()
                err_streak = 0
            except Exception as e:
                err_streak += 1
                if err_streak == 1 or err_streak % 12 == 0:
                    self.emit(f"warning: failed to read GPU state ({e}), retrying")
                snap = {}

            tracker.update(snap)
            eligible = [g for g in sorted(snap) if cfg.allow is None or g in cfg.allow]

            if snap and not warned_refs:
                ghost = sorted((set(cfg.prefer) | set(cfg.allow or ())) - set(snap))
                if ghost:
                    self.emit(f"warning: configuration references nonexistent GPU ids {ghost}, ignored")
                warned_refs = True

            stable = tracker.stable(cfg.mem_per_gpu_mb)
            ranked = rank_candidates((g for g in eligible if g in stable), cfg.prefer)
            if len(ranked) >= cfg.need:
                chosen = ranked[: cfg.need]
                self.emit(f"matched idle GPU(s) {chosen}")
                return chosen

            status = self._render(eligible, snap, tracker, cfg)
            if status != last_status:
                self.emit(status)
                last_status = status

            polls += 1
            if cfg.max_polls is not None and polls >= cfg.max_polls:
                return None
            self.stop.wait(cfg.interval_s)
        return None

    @staticmethod
    def _render(
        eligible: Sequence[int],
        snap: dict[int, GpuInfo],
        tracker: AvailabilityTracker,
        cfg: HuntConfig,
    ) -> str:
        segs = []
        for gid in eligible:
            w = tracker.windows.get(gid)
            streak = f"{sum(is_idle(s, cfg) for s in w)}/{len(w)}" if w else f"0/{cfg.stable_rounds}"
            state = "ok" if is_idle(snap[gid], cfg) else "--"
            info = snap[gid]
            segs.append(
                f"{gid}:{state}({streak},{info.mem_used_mb}MB,u{info.util_percent}%)"
            )
        body = " ".join(segs) if segs else "(no visible GPUs)"
        return f"waiting | {body} | need {cfg.need} GPU(s) x {cfg.stable_rounds} stable rounds"
