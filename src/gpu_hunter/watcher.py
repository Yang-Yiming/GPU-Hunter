"""轮询状态机:滑动窗口判定"稳定空闲",按优先级挑卡。

AvailabilityTracker 维护每张卡的原始采样窗口;空闲谓词(严格/带显存需求)
在读取时套用,供单任务 Hunter 与多任务 Scheduler 共用。
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
    need: int = 1                        # 需要几张卡(同时空闲才算命中)
    mem_per_gpu_mb: int | None = None    # 单卡显存需求;None 表示要求卡完全空闲
    prefer: tuple[int, ...] = ()         # 优先顺序,如 (2, 3)
    allow: frozenset[int] | None = None  # 只允许在这些卡里挑;None 表示全部
    idle_mem_mb: int = 500               # 占用显存低于该值视为空闲
    idle_util: int = 5                   # 利用率低于该值视为空闲
    interval_s: float = 5.0              # 轮询间隔
    stable_rounds: int = 3               # 连续多少轮空闲才认为稳定
    max_polls: int | None = None         # 轮询上限(测试用;None 表示不限)

    def __post_init__(self) -> None:
        if self.need < 1:
            raise ValueError("need 必须 >= 1")
        if self.interval_s <= 0:
            raise ValueError("interval_s 必须 > 0")
        if self.stable_rounds < 1:
            raise ValueError("stable_rounds 必须 >= 1")


def passes(
    info: GpuInfo,
    *,
    idle_mem_mb: int,
    idle_util: int,
    need_mb: int | None = None,
) -> bool:
    """单个采样的空闲判定:有计算进程一律占用;再按利用率与显存条件。"""
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
    """偏好列表中的卡排前面(按给定顺序),其余按编号升序跟在后面。"""
    order = {g: i for i, g in enumerate(prefer)}
    return sorted(set(ids), key=lambda g: (order.get(g, len(order)), g))


class AvailabilityTracker:
    """按卡维护原始采样滑窗;occupied 的卡清空历史(释放后需重新稳定)。"""

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
        """最近 stable_rounds 轮全部满足空闲条件的卡。"""
        return {
            g
            for g, w in self.windows.items()
            if len(w) == self.stable_rounds
            and all(self._ok(s, mem_per_gpu_mb) for s in w)
        }


class Hunter:
    """单任务守候:阻塞到凑齐稳定空闲卡为止。"""

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
        """阻塞直到凑齐稳定空闲的卡并返回卡号;被中止/超限时返回 None。"""
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
                    self.emit(f"警告:读取 GPU 状态失败({e}),继续重试")
                snap = {}

            tracker.update(snap)
            eligible = [g for g in sorted(snap) if cfg.allow is None or g in cfg.allow]

            if snap and not warned_refs:
                ghost = sorted((set(cfg.prefer) | set(cfg.allow or ())) - set(snap))
                if ghost:
                    self.emit(f"警告:配置引用了不存在的 GPU 编号 {ghost},已忽略")
                warned_refs = True

            stable = tracker.stable(cfg.mem_per_gpu_mb)
            ranked = rank_candidates((g for g in eligible if g in stable), cfg.prefer)
            if len(ranked) >= cfg.need:
                chosen = ranked[: cfg.need]
                self.emit(f"命中空闲 GPU {chosen}")
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
        body = " ".join(segs) if segs else "(无可见 GPU)"
        return f"等待中 | {body} | 需 {cfg.need} 张 x 连续{cfg.stable_rounds}轮"
