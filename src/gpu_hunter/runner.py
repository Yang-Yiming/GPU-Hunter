"""任务启动与运行期管理:CUDA 设备绑定、抢占校验、进程组信号转发。"""
from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path
from typing import IO, TYPE_CHECKING, Protocol, Sequence

if TYPE_CHECKING:
    from .probe import GpuInfo

CLAIM_CONFIRMED = "confirmed"
CLAIM_LOST = "lost"
CLAIM_UNKNOWN = "unknown"


class ProbeLike(Protocol):
    def snapshot(self) -> dict[int, "GpuInfo"]: ...


def build_child_env(gpus: Sequence[int], extra: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ)
    if extra:
        env.update(extra)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
    return env


def launch(
    cmd: Sequence[str],
    gpus: Sequence[int],
    log_path: Path | None = None,
    cwd: str | None = None,
    extra_env: dict[str, str] | None = None,
) -> tuple[subprocess.Popen, IO[bytes] | None]:
    """以独立会话启动子进程;给了 log_path 就把输出重定向进去,否则透传终端。"""
    fout: IO[bytes] | None = None
    if log_path is not None:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fout = open(log_path, "ab")
    try:
        proc = subprocess.Popen(
            [str(c) for c in cmd],
            env=build_child_env(gpus, extra_env),
            stdout=fout if fout else None,
            stderr=subprocess.STDOUT if fout else None,
            stdin=subprocess.DEVNULL,
            cwd=str(cwd) if cwd else None,
            start_new_session=True,
        )
    except OSError:
        if fout:
            fout.close()
        raise
    return proc, fout


def close_log(fout: IO[bytes] | None) -> None:
    if fout is not None:
        fout.close()


def _send_group(proc: subprocess.Popen, sig: signal.Signals) -> None:
    if proc.poll() is not None:
        return
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    try:
        os.killpg(pgid, sig)
    except (ProcessLookupError, PermissionError):
        proc.send_signal(sig)


def terminate_tree(proc: subprocess.Popen, term_wait: float = 10.0) -> None:
    """先 TERM 后 KILL,作用于整个进程组(覆盖 sh 包一层脚本的情况)。"""
    for sig, wait_s in ((signal.SIGTERM, term_wait), (signal.SIGKILL, 5.0)):
        _send_group(proc, sig)
        deadline = time.monotonic() + wait_s
        while proc.poll() is None and time.monotonic() < deadline:
            time.sleep(0.1)
        if proc.poll() is not None:
            return


def verify_claim(
    probe: ProbeLike,
    child_pid: int,
    gpus: Sequence[int],
    *,
    grace_s: float = 30.0,
    poll_s: float = 2.0,
    foreign_stable_s: float = 8.0,
    stop=None,
) -> str:
    """宽限期内确认子进程确实占到了卡。

    confirmed: 任一目标卡上看到子进程 pid(多进程任务如 torchrun 的
               worker pid 视为"自己人"的误伤源,因此见到自家 pid 即放行);
    lost:      目标卡上持续出现别人的计算进程超过 foreign_stable_s 秒,
               且始终未见本进程 —— 判定被抢;
    unknown:   超时仍没看到任何进程(任务可能尚未初始化 CUDA,或容器内 pid 不可见)。
    """
    gpus = tuple(gpus)
    deadline = time.monotonic() + max(grace_s, 0.0)
    foreign_since: float | None = None
    while True:
        if stop is not None and stop.is_set():
            return CLAIM_UNKNOWN
        try:
            snap = probe.snapshot()
        except Exception:
            snap = {}
        mine = False
        foreign = False
        for g in gpus:
            gi = snap.get(g)
            if gi is None:
                continue
            if child_pid in gi.compute_pids:
                mine = True
            elif any(p != child_pid for p in gi.compute_pids):
                foreign = True
        now = time.monotonic()
        if mine:
            return CLAIM_CONFIRMED
        if foreign:
            if foreign_since is None:
                foreign_since = now
            elif now - foreign_since >= foreign_stable_s:
                return CLAIM_LOST
        else:
            foreign_since = None
        if now >= deadline:
            return CLAIM_UNKNOWN
        remaining = deadline - now
        nap = min(poll_s, max(0.05, remaining))
        if stop is not None:
            stop.wait(nap)
        else:
            time.sleep(nap)


class SignalBridge:
    """把 SIGINT/SIGTERM 转发给所有在管子进程组;没有子进程时转为 KeyboardInterrupt。

    通过 attach()/detach() 登记子进程;5 秒内连按两次同一个信号视为
    "急停",对全部进程组 SIGKILL。
    """

    ESCALATE_WINDOW = 5.0

    def __init__(self) -> None:
        self.procs: list[subprocess.Popen] = []
        self._prev: dict[signal.Signals, object] = {}
        self._last_sig: signal.Signals | None = None
        self._last_t = 0.0

    def attach(self, proc: subprocess.Popen) -> None:
        if proc not in self.procs:
            self.procs.append(proc)

    def detach(self, proc: subprocess.Popen) -> None:
        if proc in self.procs:
            self.procs.remove(proc)

    def install(self) -> None:
        for sig in (signal.SIGINT, signal.SIGTERM):
            self._prev[sig] = signal.signal(sig, self._handle)

    def restore(self) -> None:
        for sig, handler in self._prev.items():
            signal.signal(sig, handler)
        self._prev.clear()

    def terminate_all(self, term_wait: float = 10.0) -> None:
        """ orderly 杀掉全部在管子进程(Ctrl-C 清理用)。"""
        for proc in list(self.procs):
            terminate_tree(proc, term_wait=term_wait)
            self.detach(proc)

    def follow(self, proc: subprocess.Popen, status_fn=None, heartbeat_s: float = 120.0) -> int:
        """跟随单个子进程直到退出(单任务模式);期间按心跳周期输出状态。"""
        self.attach(proc)
        last_beat = time.monotonic()
        try:
            while True:
                rc = proc.poll()
                if rc is not None:
                    return rc
                now = time.monotonic()
                if status_fn is not None and now - last_beat >= heartbeat_s:
                    last_beat = now
                    try:
                        status_fn()
                    except Exception:
                        pass
                time.sleep(0.5)
        finally:
            self.detach(proc)

    def _handle(self, signum, _frame) -> None:
        live = [p for p in self.procs if p.poll() is None]
        if not live:
            raise KeyboardInterrupt
        now = time.monotonic()
        if signum == self._last_sig and now - self._last_t < self.ESCALATE_WINDOW:
            for p in live:
                _send_group(p, signal.SIGKILL)
            return
        self._last_sig = signum
        self._last_t = now
        for p in live:
            _send_group(p, signum)
