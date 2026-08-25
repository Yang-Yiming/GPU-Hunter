"""轮询状态机:滑动窗口判定"稳定空闲",按优先级挑卡,凑齐即出手。"""
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


def is_idle(info: GpuInfo, cfg: HuntConfig) -> bool:
    """有计算进程一律视为占用;再按利用率/显存阈值判定。"""
    if info.compute_pids:
        return False
    if info.util_percent >= 0 and info.util_percent > cfg.idle_util:
        return False
    if cfg.mem_per_gpu_mb is not None:
        return info.mem_free_mb >= cfg.mem_per_gpu_mb
    return info.mem_used_mb <= cfg.idle_mem_mb


def rank_candidates(ids: Iterable[int], prefer: Sequence[int]) -> list[int]:
    """偏好列表中的卡排前面(按给定顺序),其余按编号升序跟在后面。"""
    order = {g: i for i, g in enumerate(prefer)}
    return sorted(set(ids), key=lambda g: (order.get(g, len(order)), g))


class Hunter:
    def __init__(
        self,
        probe: Probe,
        cfg: HuntConfig,
        emit: Callable[[str], None] | None = None,
        stop_event: threading.Event | None = None,
    ) -> None:
        self.probe = probe
        self.cfg = cfg
        self.emit = emit or print
        self.stop = stop_event or threading.Event()

    def hunt(self) -> list[int] | None:
        """阻塞直到凑齐稳定空闲的卡并返回卡号;被中止/超限时返回 None。"""
        cfg = self.cfg
        windows: dict[int, deque[bool]] = {}
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

            windows = {g: w for g, w in windows.items() if g in snap}
            eligible = [g for g in sorted(snap) if cfg.allow is None or g in cfg.allow]

            if snap and not warned_refs:
                ghost = sorted((set(cfg.prefer) | set(cfg.allow or ())) - set(snap))
                if ghost:
                    self.emit(f"警告:配置引用了不存在的 GPU 编号 {ghost},已忽略")
                warned_refs = True

            for gid in eligible:
                w = windows.setdefault(gid, deque(maxlen=cfg.stable_rounds))
                w.append(is_idle(snap[gid], cfg))

            stable = [
                g
                for g in eligible
                if len(windows[g]) == cfg.stable_rounds and all(windows[g])
            ]
            ranked = rank_candidates(stable, cfg.prefer)
            if len(ranked) >= cfg.need:
                chosen = ranked[: cfg.need]
                self.emit(f"命中空闲 GPU {chosen}")
                return chosen

            status = self._render(eligible, snap, windows, cfg)
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
        windows: dict[int, deque[bool]],
        cfg: HuntConfig,
    ) -> str:
        segs = []
        for gid in eligible:
            w = windows.get(gid)
            streak = f"{sum(w)}/{len(w)}" if w else f"0/{cfg.stable_rounds}"
            state = "ok" if is_idle(snap[gid], cfg) else "--"
            info = snap[gid]
            segs.append(
                f"{gid}:{state}({streak},{info.mem_used_mb}MB,u{info.util_percent}%)"
            )
        body = " ".join(segs) if segs else "(无可见 GPU)"
        return f"等待中 | {body} | 需 {cfg.need} 张 x 连续{cfg.stable_rounds}轮"
