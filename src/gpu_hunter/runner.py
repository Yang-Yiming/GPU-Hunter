"""Job launch and runtime management: CUDA device binding, claim verification, process-group signals."""
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
    """Start the child in its own session; redirect output to log_path when given."""
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
    """Send TERM, then KILL, to the whole process group (covers a wrapping shell)."""
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
    give_up=None,
) -> str:
    """Confirm within the grace period that the child really got the GPUs.

    confirmed: the child pid shows up on any target GPU (multi-process jobs such
               as torchrun have different worker pids, so seeing our own pid is
               enough to avoid false preemption);
    lost:      another process shows up continuously on the target GPUs for
               longer than foreign_stable_s while ours is never seen;
    unknown:   nothing was seen before the timeout (the job may not have
               initialized CUDA yet, or container pids are invisible), or
               give_up() returned true (e.g. the child already exited).
    """
    gpus = tuple(gpus)
    deadline = time.monotonic() + max(grace_s, 0.0)
    foreign_since: float | None = None
    while True:
        if stop is not None and stop.is_set():
            return CLAIM_UNKNOWN
        if give_up is not None and give_up():
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
    """Forward SIGINT/SIGTERM to every managed child process group.

    Children are registered via attach()/detach(); two presses of the same
    signal within 5 seconds count as an emergency stop and SIGKILL every
    process group.
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
        """Kill every managed child in order (used for Ctrl-C cleanup)."""
        for proc in list(self.procs):
            terminate_tree(proc, term_wait=term_wait)
            self.detach(proc)

    def follow(self, proc: subprocess.Popen, status_fn=None, heartbeat_s: float = 120.0) -> int:
        """Follow a single child until it exits (single-job mode), emitting status per heartbeat."""
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
