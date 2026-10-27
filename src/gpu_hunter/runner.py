"""Job launch and runtime management: CUDA device binding, claim verification, process-group signals."""
from __future__ import annotations

import logging
import os
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import IO, TYPE_CHECKING, Callable, Protocol, Sequence

if TYPE_CHECKING:
    from .probe import GpuInfo

log = logging.getLogger("gpu-hunter")

CLAIM_CONFIRMED = "confirmed"
CLAIM_LOST = "lost"
CLAIM_UNKNOWN = "unknown"


def normalize_rc(rc: int | None) -> int | None:
    """Return a shell-style exit code (negative signal rc -> 128 + signo)."""
    if rc is None:
        return None
    return 128 + (-rc) if rc < 0 else rc


class ProbeLike(Protocol):
    def snapshot(self) -> dict[int, "GpuInfo"]: ...


def build_child_env(gpus: Sequence[int], extra: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ)
    if extra:
        env.update(extra)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
    # NVML/nvidia-smi indexes follow PCI bus order.  CUDA's default device
    # order may be FASTEST_FIRST, which would make "2" mean a different card.
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
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
    # Remember the identity of the process we started so a later pid reuse
    # cannot make terminate_tree signal an unrelated process.
    try:
        proc._gpu_hunter_start_time = _proc_start_time(proc.pid)  # type: ignore[attr-defined]
    except Exception:
        pass
    return proc, fout


def close_log(fout: IO[bytes] | None) -> None:
    if fout is not None:
        fout.close()


# ---------- ownership / safety checks ----------

def _proc_uid(pid: int) -> int | None:
    """Return the owner uid of ``pid``; None when ownership cannot be verified.

    Every signal we send must pass this check first: a stale pid, a pid from a
    container namespace or a bug must never let us terminate another user's job.
    Linux exposes this via /proc; on systems without it we fail closed.
    """
    try:
        return os.stat(f"/proc/{pid}").st_uid
    except OSError:
        return None


def is_owned_by_current_user(pid: int) -> bool:
    uid = _proc_uid(pid)
    return uid is not None and uid == os.getuid()


def _proc_start_time(pid: int) -> int | None:
    """Kernel start time of ``pid`` (field 22 of /proc/<pid>/stat).

    Used together with the uid check to detect pid reuse before sending a
    signal: even a process owned by us must not be killed if it is not the
    process we originally launched.
    """
    try:
        with open(f"/proc/{pid}/stat", "rb") as fh:
            data = fh.read()
    except OSError:
        return None
    rparen = data.rfind(b")")
    if rparen < 0:
        return None
    fields = data[rparen + 2 :].split()
    if len(fields) < 20 or not fields[19].isdigit():
        return None
    return int(fields[19])


def _group_members(pgid: int) -> list[tuple[int, int | None]] | None:
    """Return (pid, uid) pairs in process group ``pgid``; None if /proc is unreadable."""
    try:
        entries = os.listdir("/proc")
    except OSError:
        return None
    members: list[tuple[int, int | None]] = []
    for entry in entries:
        if not entry.isdigit():
            continue
        pid = int(entry)
        try:
            with open(f"/proc/{pid}/stat", "rb") as fh:
                data = fh.read()
        except OSError:
            continue
        rparen = data.rfind(b")")
        if rparen < 0:
            continue
        fields = data[rparen + 2 :].split()
        # fields = state, ppid, pgrp, session, ...
        if len(fields) < 3 or not fields[2].isdigit():
            continue
        if int(fields[2]) == pgid:
            members.append((pid, _proc_uid(pid)))
    return members


def _group_owned_by_current_user(pgid: int) -> bool:
    members = _group_members(pgid)
    if members is None:
        # Fail closed: without /proc we can only vouch for the group leader.
        return is_owned_by_current_user(pgid)
    return bool(members) and all(uid == os.getuid() for _, uid in members)


def _group_alive(pgid: int) -> bool:
    members = _group_members(pgid)
    if members is None:
        try:
            os.killpg(pgid, 0)
            return True
        except OSError:
            return False
    return bool(members)


def _send_group(proc: subprocess.Popen, sig: signal.Signals) -> bool:
    """Signal the child's process group only when every member belongs to us.

    Returns True when a signal was delivered (or attempted against a live
    child), False when the process/group could not be verified as ours.
    """
    pid = proc.pid
    recorded = hasattr(proc, "_gpu_hunter_start_time")
    expected_start = getattr(proc, "_gpu_hunter_start_time", None)
    actual_start = _proc_start_time(pid)
    if recorded:
        if expected_start is None:
            log.warning(
                "refusing to signal pid=%d: process identity was not recorded at launch", pid
            )
            return False
        if actual_start is not None and actual_start != expected_start:
            log.warning(
                "refusing to signal pid=%d: process identity changed (pid reuse?)", pid
            )
            return False
    try:
        pgid = os.getpgid(pid)
    except ProcessLookupError:
        # The child may already have been reaped while descendants remain in
        # its session/process group (start_new_session=True makes pid == pgid).
        pgid = pid

    if pgid <= 0 or pgid == os.getpgrp():
        if is_owned_by_current_user(pid):
            try:
                proc.send_signal(sig)
                return True
            except OSError:
                return False
        log.warning("refusing to signal pid=%d: not owned by the current user", pid)
        return False

    if not _group_owned_by_current_user(pgid):
        log.warning(
            "refusing to signal process group %d: pid=%d or one of its members "
            "is not owned by the current user",
            pgid,
            pid,
        )
        return False

    try:
        os.killpg(pgid, sig)
        return True
    except (ProcessLookupError, PermissionError):
        if is_owned_by_current_user(pid):
            try:
                proc.send_signal(sig)
                return True
            except OSError:
                pass
        return False


def terminate_tree(proc: subprocess.Popen, term_wait: float = 10.0) -> None:
    """Send TERM, then KILL, to the whole process group (covers a wrapping shell).

    Signals are only sent after verifying that the target processes belong to
    the current user.  If the direct child exits while background descendants
    remain, the group is still drained before KILLing it.
    """
    for sig, wait_s in ((signal.SIGTERM, term_wait), (signal.SIGKILL, 5.0)):
        if not _send_group(proc, sig):
            return
        deadline = time.monotonic() + wait_s
        while time.monotonic() < deadline:
            if proc.poll() is not None and not _group_alive(proc.pid):
                return
            time.sleep(0.1)


def _session_of(pid: int) -> int | None:
    try:
        return os.getsid(pid)
    except OSError:
        return None


def _belongs_to_child(pid: int, child_pid: int, child_sid: int) -> bool | None:
    """True/False when ownership is knowable, None when the pid is not visible.

    Children are launched with start_new_session=True, so the child is the
    session leader and bash wrappers, torchrun workers and other descendants
    inherit its session id.  This is what keeps a wrapper's children from being
    mistaken for a foreign preemption.
    """
    if pid == child_pid:
        return True
    sid = _session_of(pid)
    if sid is None:
        return None
    return sid == child_sid


def verify_claim(
    probe: ProbeLike,
    child_pid: int,
    gpus: Sequence[int],
    *,
    grace_s: float = 30.0,
    poll_s: float = 2.0,
    foreign_stable_s: float = 8.0,
    stop: threading.Event | None = None,
    give_up: Callable[[], bool] | None = None,
) -> str:
    """Confirm within the grace period that the child really got the GPUs.

    confirmed: the child or any process in its session shows up on a target
               GPU (multi-process jobs such as torchrun and bash wrappers have
               different worker pids but share the child's session);
    lost:      a process that is verifiably *not* ours shows up continuously
               on the target GPUs for longer than foreign_stable_s while ours
               is never seen;
    unknown:   nothing was seen before the timeout (the job may not have
               initialized CUDA yet, or container pids are invisible), a pid
               could not be attributed safely, or give_up() returned true
               (e.g. the child already exited).
    """
    gpus = tuple(gpus)
    child_sid = _session_of(child_pid)
    if child_sid is None:
        child_sid = child_pid
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
            for pid in gi.compute_pids or ():
                owned = _belongs_to_child(pid, child_pid, child_sid)
                if owned is True:
                    mine = True
                elif owned is False:
                    foreign = True
                # owned is None: pid invisible / unattributable.  Never treat
                # that as preemption; container pids are the common case.
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
    process group.  The optional stop event is set on the first signal so a
    scheduler stops taking new jobs instead of silently continuing the queue.
    """

    ESCALATE_WINDOW = 5.0

    def __init__(self, stop_event: threading.Event | None = None) -> None:
        self.procs: list[subprocess.Popen] = []
        self.stop = stop_event
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
        # Stop scheduling new work even if children are still alive and can be
        # asked to terminate gracefully.
        if self.stop is not None:
            self.stop.set()
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
