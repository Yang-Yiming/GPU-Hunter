"""Reservation ("hold") mode: hold GPUs for later use.

``reserve`` finds stably idle GPUs, starts one placeholder process per card
(see :mod:`gpu_hunter.hog`) that allocates the requested amount of device
memory, and records the placeholders in a small JSON state file.  The
reservation can then be checked (``status``), released (``release``) or handed
over to an arbitrary command -- including one that cannot be described as a
gpu-hunter job.

The placeholder processes run in their own sessions and are *not* attached to
the signal bridge, so they survive an abnormal exit of the hunter process.  The
state file, the per-hog self-expiry (``--hold-max``) and ``status`` make a
reservation recoverable instead of silently leaking.
"""
from __future__ import annotations

import fcntl
import json
import os
import shlex
import signal
import socket
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Iterator

from .runner import (
    close_log,
    launch,
    normalize_rc,
    process_is_live,
    process_start_time,
    signal_group_checked,
    terminate_tree,
)
from .watcher import HuntConfig, Hunter

__all__ = [
    "AcquireOptions",
    "HogRecord",
    "Reservation",
    "ReserveError",
    "StateGone",
    "Supervisor",
    "acquire",
    "hog_command",
    "load_state",
    "release",
    "state_path",
    "status_report",
]

STATE_VERSION = 1
HOG_MODULE = "gpu_hunter.hog"
DEFAULT_HOLD_MAX_S = 12 * 3600.0
FULL_MARGIN_MB = 512          # leave this much free when reserving a whole card
HOLD_CONFIRM_S = 30.0         # how long a placeholder gets to prove it allocated

Emit = Callable[[str], None]
Sleep = Callable[[float], None]


class ReserveError(RuntimeError):
    """Reservation operation failed (no state, unverifiable state, ...)."""


class StateGone(ReserveError):
    """The state file disappeared (e.g. another process released it)."""


# ---------------------------------------------------------------------------
# state file
# ---------------------------------------------------------------------------

def state_dir() -> Path:
    override = os.environ.get("GPU_HUNTER_STATE_DIR")
    if override:
        return Path(override).expanduser()
    base = os.environ.get("XDG_STATE_HOME") or str(Path.home() / ".local" / "state")
    return Path(base) / "gpu-hunter"


def state_path() -> Path:
    return state_dir() / f"reservation-{os.getuid()}.json"


def _lock_path() -> Path:
    return state_dir() / f"reservation-{os.getuid()}.lock"


@contextmanager
def _state_lock() -> Iterator[None]:
    state_dir().mkdir(parents=True, exist_ok=True)
    fh = open(_lock_path(), "w", encoding="utf-8")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fh, fcntl.LOCK_UN)
        finally:
            fh.close()


@dataclass
class HogRecord:
    index: int
    pid: int
    start_time: int | None
    mem_mb: int
    used_before_mb: int = 0
    name: str = ""
    log: str = ""

    @property
    def alive(self) -> bool:
        return process_is_live(self.pid, self.start_time)

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "pid": self.pid,
            "start_time": self.start_time,
            "mem_mb": self.mem_mb,
            "used_before_mb": self.used_before_mb,
            "name": self.name,
            "log": self.log,
        }

    @classmethod
    def from_dict(cls, data) -> "HogRecord":
        if not isinstance(data, dict):
            raise ValueError("gpu entries must be mappings")
        return cls(
            index=int(data["index"]),
            pid=int(data["pid"]),
            start_time=None if data.get("start_time") is None else int(data["start_time"]),
            mem_mb=int(data.get("mem_mb", 0)),
            used_before_mb=int(data.get("used_before_mb", 0)),
            name=str(data.get("name", "")),
            log=str(data.get("log", "")),
        )


@dataclass
class Reservation:
    mem_mb: int
    gpus: list[HogRecord]
    created_at: float
    expires_at: float | None
    hostname: str
    uid: int
    full: bool = False
    version: int = STATE_VERSION

    @property
    def gpu_ids(self) -> list[int]:
        return [rec.index for rec in self.gpus]

    def remaining_s(self, now: float | None = None) -> float | None:
        if self.expires_at is None:
            return None
        return max(0.0, self.expires_at - (time.time() if now is None else now))

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "uid": self.uid,
            "hostname": self.hostname,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "mem_mb": self.mem_mb,
            "full": self.full,
            "gpus": [rec.to_dict() for rec in self.gpus],
        }

    @classmethod
    def from_dict(cls, data) -> "Reservation":
        if not isinstance(data, dict):
            raise ValueError("reservation state must be a mapping")
        raw_gpus = data.get("gpus")
        if not isinstance(raw_gpus, list) or not raw_gpus:
            raise ValueError("gpus must be a non-empty list")
        expires = data.get("expires_at")
        return cls(
            mem_mb=int(data.get("mem_mb", 0)),
            gpus=[HogRecord.from_dict(item) for item in raw_gpus],
            created_at=float(data["created_at"]),
            expires_at=None if expires is None else float(expires),
            hostname=str(data.get("hostname", "")),
            uid=int(data["uid"]),
            full=bool(data.get("full", False)),
            version=int(data.get("version", STATE_VERSION)),
        )


def _save_state_locked(state: Reservation) -> None:
    directory = state_dir()
    directory.mkdir(parents=True, exist_ok=True)
    tmp = directory / f".reservation-{os.getuid()}.{os.getpid()}.tmp"
    payload = json.dumps(state.to_dict(), indent=2, sort_keys=True) + "\n"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, state_path())
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass


def load_state() -> Reservation | None:
    """Read the reservation state (None when absent); raises on corruption."""
    try:
        raw = state_path().read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as e:
        raise ReserveError(f"cannot read reservation state: {e}") from e
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ReserveError(f"reservation state is corrupt ({state_path()}): {e}") from e
    try:
        state = Reservation.from_dict(data)
    except (KeyError, TypeError, ValueError) as e:
        raise ReserveError(f"reservation state is malformed ({state_path()}): {e}") from e
    if state.version != STATE_VERSION:
        raise ReserveError(
            f"unsupported reservation state version {state.version} (expected {STATE_VERSION})"
        )
    if state.uid != os.getuid():
        raise ReserveError(
            f"reservation state belongs to uid {state.uid}, not {os.getuid()}"
        )
    return state


def clear_state() -> None:
    try:
        state_path().unlink()
    except FileNotFoundError:
        pass


# ---------------------------------------------------------------------------
# placeholder launch
# ---------------------------------------------------------------------------

def hog_command(template: str | None, mem_mb: int, gpu: int, timeout_s: int) -> list[str]:
    """Build the placeholder command for one GPU.

    ``template`` (``--hold-cmd`` / ``$GPU_HUNTER_HOG_CMD``) may contain
    ``{mem_mb}``, ``{gpu}`` and ``{timeout}`` placeholders.
    """
    text = template or os.environ.get("GPU_HUNTER_HOG_CMD")
    if text:
        tokens = []
        for token in shlex.split(text):
            tokens.append(
                token.replace("{mem_mb}", str(mem_mb))
                .replace("{gpu}", str(gpu))
                .replace("{timeout}", str(timeout_s))
            )
        if not tokens:
            raise ReserveError("placeholder command (--hold-cmd) is empty")
        return tokens
    cmd = [sys.executable, "-m", HOG_MODULE, "--mem-mb", str(mem_mb)]
    if timeout_s > 0:
        cmd += ["--timeout", str(timeout_s)]
    return cmd


def _hog_log_path(gpu: int) -> Path:
    directory = state_dir()
    directory.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return directory / f"hog-gpu{gpu}-{stamp}-{os.getpid()}.log"


def _snapshot(probe) -> dict | None:
    try:
        return probe.snapshot()
    except Exception:
        return None


def _wait_hold(
    probe,
    proc,
    gpu: int,
    used_before_mb: int,
    mem_mb: int,
    timeout_s: float,
    stop: threading.Event,
    log_path: Path,
) -> tuple[bool, str]:
    """Wait until NVML shows the placeholder's memory on the target GPU."""
    deadline = time.monotonic() + max(timeout_s, 0.1)
    needed = max(1, int(mem_mb * 0.9))
    while True:
        rc = proc.poll()
        if rc is not None:
            return False, (
                f"placeholder exited rc={normalize_rc(rc)} before holding memory "
                f"(log: {log_path})"
            )
        if stop.is_set():
            return False, "aborted"
        snap = _snapshot(probe)
        if snap is not None and gpu in snap:
            used = snap[gpu].mem_used_mb
            if used - used_before_mb >= needed:
                return True, "held"
        now = time.monotonic()
        if now >= deadline:
            return False, (
                f"placeholder memory was not visible within {timeout_s:.0f}s "
                f"(log: {log_path})"
            )
        wait = min(0.25, max(0.05, deadline - now))
        if stop.wait(wait):
            return False, "aborted"


class _HoldFailed(ReserveError):
    pass


def _hold_gpus(probe, gpus, opts: "AcquireOptions", *, emit: Emit, stop: threading.Event) -> Reservation:
    snap = _snapshot(probe)
    if snap is None:
        raise _HoldFailed("cannot read GPU state to start placeholders")
    launched: list = []
    records: list[HogRecord] = []
    timeout_s = opts.hold_max_s if opts.hold_max_s and opts.hold_max_s > 0 else 0
    try:
        for gpu in gpus:
            gi = snap.get(gpu)
            used_before = gi.mem_used_mb if gi is not None else 0
            if opts.full:
                if gi is None:
                    raise _HoldFailed(f"cannot read GPU{gpu} state for a full reservation")
                mem_mb = gi.mem_free_mb - FULL_MARGIN_MB
                if mem_mb <= 0:
                    raise _HoldFailed(f"GPU{gpu} has almost no free memory to hold")
            else:
                if not opts.mem_mb:
                    raise _HoldFailed("no memory amount configured")
                mem_mb = int(opts.mem_mb)
            cmd = hog_command(opts.hold_cmd, mem_mb, gpu, int(timeout_s))
            log_path = _hog_log_path(gpu)
            try:
                proc, fout = launch(cmd, [gpu], log_path=log_path)
            except OSError as e:
                raise _HoldFailed(f"GPU{gpu}: cannot start placeholder: {e}") from e
            close_log(fout)
            launched.append(proc)
            start_time = process_start_time(proc.pid)
            if start_time is None:
                raise _HoldFailed(
                    f"GPU{gpu}: cannot record placeholder identity (pid {proc.pid})"
                )
            ok, reason = _wait_hold(
                probe, proc, gpu, used_before, mem_mb,
                opts.confirm_s, stop, log_path,
            )
            if not ok:
                raise _HoldFailed(f"GPU{gpu}: {reason}")
            emit(f"holding GPU{gpu}: pid {proc.pid}, {mem_mb}MB")
            records.append(
                HogRecord(
                    index=gpu,
                    pid=proc.pid,
                    start_time=start_time,
                    mem_mb=mem_mb,
                    used_before_mb=used_before,
                    name=gi.name if gi is not None else "",
                    log=str(log_path),
                )
            )
    except BaseException:
        for proc in launched:
            terminate_tree(proc, term_wait=5.0)
        raise
    mem_total = max(rec.mem_mb for rec in records)
    created = time.time()
    expires = created + opts.hold_max_s if opts.hold_max_s and opts.hold_max_s > 0 else None
    return Reservation(
        mem_mb=mem_total,
        gpus=records,
        created_at=created,
        expires_at=expires,
        hostname=socket.gethostname(),
        uid=os.getuid(),
        full=opts.full,
    )


# ---------------------------------------------------------------------------
# acquisition
# ---------------------------------------------------------------------------

@dataclass
class AcquireOptions:
    need: int = 1
    mem_mb: int | None = None        # free memory required, also the amount to hold
    full: bool = False               # hold all free memory instead of mem_mb
    prefer: tuple[int, ...] = ()
    allow: frozenset[int] | None = None
    idle_mem_mb: int = 500
    idle_util: int = 5
    interval_s: float = 5.0
    stable_rounds: int = 3
    hold_max_s: float = DEFAULT_HOLD_MAX_S
    hold_cmd: str | None = None
    confirm_s: float = HOLD_CONFIRM_S
    max_polls: int | None = None     # tests

    def hunt_config(self) -> HuntConfig:
        return HuntConfig(
            need=self.need,
            mem_per_gpu_mb=None if self.full else self.mem_mb,
            prefer=self.prefer,
            allow=self.allow,
            idle_mem_mb=self.idle_mem_mb,
            idle_util=self.idle_util,
            interval_s=self.interval_s,
            stable_rounds=self.stable_rounds,
            max_polls=self.max_polls,
        )


def _acquire_once(
    probe,
    opts: AcquireOptions,
    *,
    emit: Emit,
    stop: threading.Event,
    on_attempt: Callable[[bool, str], None] | None = None,
) -> Reservation:
    """Hunt and hold until a full reservation is in place (state not written)."""
    hunter = Hunter(probe, opts.hunt_config(), emit=emit, stop_event=stop, on_attempt=on_attempt)
    backoff = max(opts.interval_s, 2.0)
    while not stop.is_set():
        gpus = hunter.hunt()
        if gpus is None:
            raise ReserveError("reserve aborted while waiting for idle GPUs")
        try:
            return _hold_gpus(probe, gpus, opts, emit=emit, stop=stop)
        except _HoldFailed as e:
            emit(f"warning: could not hold GPU(s) {gpus}: {e}; retrying in {backoff:.0f}s")
            if stop.wait(backoff):
                raise ReserveError("reserve aborted while retrying placeholder launch")
            backoff = min(backoff * 2, 60.0)
    raise ReserveError("reserve aborted")


def acquire(
    probe,
    opts: AcquireOptions,
    *,
    emit: Emit = print,
    stop: threading.Event | None = None,
    on_attempt: Callable[[bool, str], None] | None = None,
) -> Reservation:
    """Hunt for idle GPUs, hold them and persist the reservation."""
    stop = stop if stop is not None else threading.Event()
    with _state_lock():
        existing = load_state()
        if existing is not None:
            raise ReserveError(
                f"a reservation is already active (GPU {existing.gpu_ids}); "
                "release it first with: gpu-hunter --release"
            )
        state = _acquire_once(probe, opts, emit=emit, stop=stop, on_attempt=on_attempt)
        _save_state_locked(state)
    return state


# ---------------------------------------------------------------------------
# release
# ---------------------------------------------------------------------------

def _wait_processes_gone(records: list[HogRecord], timeout: float) -> list[HogRecord]:
    deadline = time.monotonic() + max(timeout, 0.0)
    remaining = list(records)
    while remaining and time.monotonic() < deadline:
        remaining = [rec for rec in remaining if rec.alive]
        if remaining:
            time.sleep(0.1)
    return remaining


def _kill_processes(records: list[HogRecord], emit: Emit) -> None:
    for rec in records:
        if not rec.alive:
            continue
        if signal_group_checked(rec.pid, rec.start_time, signal.SIGKILL):
            emit(f"warning: placeholder on GPU {rec.index} did not exit; sent SIGKILL")
        else:
            emit(
                f"warning: cannot verify pid {rec.pid} on GPU {rec.index}; "
                "refusing to signal it"
            )


def _wait_gpus_free(probe, gpus: list[int], *, timeout: float, emit: Emit) -> None:
    deadline = time.monotonic() + max(timeout, 0.0)
    while True:
        snap = _snapshot(probe)
        if snap is None:
            emit("warning: cannot verify the release through NVML; continuing")
            return
        busy = [
            gpu
            for gpu in gpus
            if gpu in snap and (snap[gpu].compute_pids is None or snap[gpu].compute_pids)
        ]
        if not busy:
            return
        if time.monotonic() >= deadline:
            emit(f"warning: GPU(s) {busy} still show compute processes after release")
            return
        time.sleep(0.2)


def _release_locked(
    probe,
    state: Reservation,
    *,
    force: bool,
    emit: Emit,
    term_wait: float,
    free_wait: float,
) -> None:
    if state.hostname and state.hostname != socket.gethostname():
        if not force:
            raise ReserveError(
                f"reservation was created on host {state.hostname!r} "
                f"(this host is {socket.gethostname()!r}); refusing to touch it "
                "(use --force to clear it here)"
            )
        emit(
            f"warning: reservation belongs to host {state.hostname!r}; "
            "clearing the state without signalling anything"
        )
        return

    pending: list[HogRecord] = []
    for rec in state.gpus:
        if rec.alive:
            if signal_group_checked(rec.pid, rec.start_time, signal.SIGTERM):
                emit(f"releasing GPU {rec.index}: SIGTERM -> pid {rec.pid}")
                pending.append(rec)
            else:
                emit(
                    f"warning: could not verify pid {rec.pid} on GPU {rec.index}; "
                    "leaving it alone"
                )
        else:
            emit(f"GPU {rec.index}: placeholder pid {rec.pid} is already gone")

    try:
        remaining = _wait_processes_gone(pending, term_wait)
        if remaining:
            _kill_processes(remaining, emit)
            _wait_processes_gone(remaining, 5.0)
        if probe is not None:
            _wait_gpus_free(probe, [rec.index for rec in state.gpus], timeout=free_wait, emit=emit)
    except KeyboardInterrupt:
        _kill_processes(pending, emit)
        raise


def release(
    probe,
    state: Reservation | None = None,
    *,
    force: bool = False,
    emit: Emit = print,
    term_wait: float = 15.0,
    free_wait: float = 15.0,
) -> list[int]:
    """Terminate the placeholders, confirm the GPUs are free and clear the state.

    Returns the released GPU ids.  Raises ReserveError when there is nothing to
    release or the on-disk state cannot be handled.
    """
    with _state_lock():
        disk = None
        unreadable = False
        try:
            disk = load_state()
        except ReserveError as e:
            if not force:
                raise
            emit(f"warning: ignoring unreadable reservation state ({e})")
            clear_state()
            unreadable = True
        if disk is not None:
            state = disk
        if state is None:
            if unreadable:
                return []
            raise ReserveError("no active reservation")
        _release_locked(
            probe, state, force=force, emit=emit,
            term_wait=term_wait, free_wait=free_wait,
        )
        # Only drop the state once the placeholders are gone: an interrupted or
        # failed release must remain retryable.
        try:
            clear_state()
        except OSError as e:
            emit(f"warning: could not remove the reservation state file: {e}")
        return sorted(set(rec.index for rec in state.gpus))


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

@dataclass
class StatusReport:
    text: str
    exit_code: int
    state: Reservation | None = None
    healthy: bool = True


def _fmt_dur(seconds: float) -> str:
    seconds = int(max(0, seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _fmt_time(ts: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))


def status_report(probe=None) -> StatusReport:
    """Describe the active reservation; exit code 0 healthy, 1 degraded, 2 none."""
    try:
        state = load_state()
    except ReserveError as e:
        return StatusReport(f"reservation state error: {e}", 1)
    if state is None:
        return StatusReport("no active reservation", 2, None, True)

    now = time.time()
    snap = _snapshot(probe) if probe is not None else None
    lines = [
        f"reservation active: {len(state.gpus)} GPU(s), "
        f"{state.mem_mb}MB each" + (" (full card)" if state.full else "")
    ]
    lines.append(f"  host: {state.hostname or '?'}  uid: {state.uid}")
    created = f"  created: {_fmt_time(state.created_at)}"
    if state.expires_at is None:
        created += "  expires: never"
    else:
        created += f"  expires: {_fmt_time(state.expires_at)} ({_fmt_dur(state.expires_at - now)} left)"
    lines.append(created)

    healthy = True
    if state.hostname and state.hostname != socket.gethostname():
        lines.append(f"  warning: created on host {state.hostname!r}; this is {socket.gethostname()!r}")
        healthy = False
    if state.expires_at is not None and now >= state.expires_at:
        lines.append("  warning: the reservation has expired")
        healthy = False

    for rec in sorted(state.gpus, key=lambda item: item.index):
        alive = rec.alive
        held: bool | None = None
        if snap is not None and rec.index in snap:
            delta = snap[rec.index].mem_used_mb - rec.used_before_mb
            held = delta >= max(1, int(rec.mem_mb * 0.5))
        ok = alive and held is not False
        if not ok:
            healthy = False
        detail = f"pid {rec.pid}" if alive else f"pid {rec.pid} (dead)"
        if held is True:
            detail += ", memory held"
        elif held is False:
            detail += ", memory not held"
        label = "ok" if ok else "LOST"
        lines.append(
            f"  GPU{rec.index:<2} {rec.name or '?':<18} {rec.mem_mb}MB  "
            f"{detail}  [{label}]"
        )
    if probe is None:
        lines.append("  warning: GPU probing unavailable; only process state was checked")
    return StatusReport("\n".join(lines), 0 if healthy else 1, state, healthy)


# ---------------------------------------------------------------------------
# foreground supervisor
# ---------------------------------------------------------------------------

class Supervisor:
    """Keep a foreground reservation healthy and release it when asked.

    Each tick verifies that every placeholder is still alive and still holds its
    memory.  Lost placeholders are dropped and the missing GPUs are re-acquired
    so the reservation degrades as briefly as possible.
    """

    def __init__(
        self,
        probe,
        state: Reservation,
        opts: AcquireOptions,
        *,
        emit: Emit = print,
        stop: threading.Event | None = None,
        on_attempt: Callable[[bool, str], None] | None = None,
        interval_s: float | None = None,
        heartbeat_s: float = 60.0,
        sleep: Sleep = time.sleep,
        now: Callable[[], float] = time.time,
    ) -> None:
        self.probe = probe
        self.state = state
        self.opts = opts
        self.emit = emit
        self.stop = stop if stop is not None else threading.Event()
        self.on_attempt = on_attempt
        self.interval_s = interval_s if interval_s is not None else max(opts.interval_s, 2.0)
        self.heartbeat_s = heartbeat_s
        self._sleep = sleep
        self._now = now
        self._last_beat = now()

    def run(self) -> str:
        """Tick forever; returns "interrupted", "expired", "missing" or "error"."""
        while True:
            if self.stop.is_set():
                return "interrupted"
            verdict = self.tick()
            if verdict != "continue":
                return verdict
            self._sleep(self.interval_s)

    def tick(self) -> str:
        if not state_path().exists():
            return "missing"
        try:
            state = load_state()
        except ReserveError as e:
            self.emit(f"warning: {e}")
            return "error"
        if state is None:
            return "missing"
        self.state = state
        if state.expires_at is not None and self._now() >= state.expires_at:
            return "expired"

        snap = _snapshot(self.probe)
        lost = [rec for rec in state.gpus if not self._record_ok(rec, snap)]
        if lost:
            for rec in lost:
                self.emit(f"warning: reservation on GPU {rec.index} was lost (pid {rec.pid})")
                if rec.alive:
                    signal_group_checked(rec.pid, rec.start_time, signal.SIGTERM)
            state.gpus = [rec for rec in state.gpus if rec not in lost]
            self.emit(f"trying to re-acquire {len(lost)} GPU(s) to restore the reservation")
            try:
                extra = _acquire_once(
                    self.probe, replace(self.opts, need=len(lost),
                                        mem_mb=self.opts.mem_mb, full=self.opts.full),
                    emit=self.emit, stop=self.stop, on_attempt=self.on_attempt,
                )
            except ReserveError as e:
                self.emit(f"warning: could not restore the reservation: {e}")
            else:
                state.gpus.extend(extra.gpus)
                try:
                    self._save()
                except StateGone:
                    return "missing"
                self.emit(f"reservation restored on GPU {[rec.index for rec in extra.gpus]}")

        if self._now() - self._last_beat >= self.heartbeat_s:
            self._last_beat = self._now()
            self.emit(self._describe())
        return "continue"

    # -- helpers -------------------------------------------------------
    def _record_ok(self, rec: HogRecord, snap: dict | None) -> bool:
        if not rec.alive:
            return False
        if snap is not None and rec.index in snap:
            delta = snap[rec.index].mem_used_mb - rec.used_before_mb
            if delta < max(1, int(rec.mem_mb * 0.5)):
                return False
        return True

    def _save(self) -> None:
        with _state_lock():
            try:
                current = load_state()
            except ReserveError as e:
                raise StateGone(f"reservation state became unreadable: {e}") from e
            if current is None:
                raise StateGone("reservation state disappeared")
            _save_state_locked(self.state)

    def _describe(self) -> str:
        ids = sorted(rec.index for rec in self.state.gpus)
        left = "never"
        if self.state.expires_at is not None:
            left = _fmt_dur(self.state.expires_at - self._now())
        return (
            f"holding GPU {','.join(map(str, ids))} "
            f"({len(ids)} x {self.state.mem_mb}MB, {left} left)"
        )
