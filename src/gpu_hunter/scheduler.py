"""Multi-job scheduler: greedy (default) or strictly sequential, until the queue drains.

Every tick: probe -> update sliding windows -> launch whatever fits under the
strategy -> reap exited jobs. The ``claimed`` ledger records GPUs this
scheduler has handed out, keeping concurrent jobs disjoint.
"""
from __future__ import annotations

import logging
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, IO, Sequence

from .probe import Probe, describe
from .runner import (
    CLAIM_LOST,
    CLAIM_UNKNOWN,
    SignalBridge,
    close_log,
    launch,
    normalize_rc,
    terminate_tree,
    verify_claim,
)
from .spec import JobSpec, WorkflowSpec
from .watcher import AvailabilityTracker, rank_candidates

log = logging.getLogger("gpu-hunter")

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_name(name: str) -> str:
    """Keep a job name usable inside a generated log filename."""
    cleaned = _SAFE_NAME.sub("_", name).strip("._-")
    return cleaned or "job"


@dataclass
class JobResult:
    name: str
    rc: int | None
    gpus: tuple[int, ...]
    started_at: float
    ended_at: float
    attempts: int = 1

    @property
    def elapsed_s(self) -> float:
        return self.ended_at - self.started_at

    @property
    def ok(self) -> bool:
        return self.rc == 0


@dataclass
class _Running:
    spec: JobSpec
    proc: subprocess.Popen
    fout: IO[bytes] | None
    gpus: tuple[int, ...]
    started_at: float
    attempts: int


def format_summary(results: Sequence[JobResult]) -> str:
    if not results:
        return "(no jobs recorded)"
    rows = [
        (
            r.name,
            ",".join(map(str, r.gpus)) or "-",
            f"{r.elapsed_s:.0f}s",
            "?" if r.rc is None else str(r.rc),
            "OK" if r.ok else "FAIL",
            str(r.attempts),
        )
        for r in results
    ]
    head = ("name", "gpus", "time", "rc", "verdict", "try")
    width = [
        max(len(head[i]), max(len(row[i]) for row in rows)) for i in range(len(head))
    ]
    lines = ["  ".join(h.ljust(width[i]) for i, h in enumerate(head))]
    lines.append("-" * (sum(width) + 2 * (len(width) - 1)))
    for row in rows:
        lines.append("  ".join(v.ljust(width[i]) for i, v in enumerate(row)))
    return "\n".join(lines)


class Scheduler:
    def __init__(
        self,
        probe: Probe,
        wf: WorkflowSpec,
        emit: Callable[[str], None] | None = None,
        stop_event: threading.Event | None = None,
        bridge: SignalBridge | None = None,
        verify_fn: Callable | None = None,
        max_ticks: int | None = None,
        notify: Callable[[str, str], None] | None = None,
    ) -> None:
        self.probe = probe
        self.wf = wf
        self.emit = emit if emit is not None else print
        self.notify = notify
        self.stop = stop_event if stop_event is not None else threading.Event()
        self.bridge = bridge
        self.max_ticks = max_ticks
        self.tracker = AvailabilityTracker(
            wf.idle_mem_mb, idle_util=wf.idle_util, stable_rounds=wf.stable_rounds
        )
        self._verify = verify_fn or self._default_verify
        self.pending: list[JobSpec] = list(wf.jobs)
        self.running: list[_Running] = []
        self.claimed: dict[int, str] = {}   # gpu -> job name
        self.results: list[JobResult] = []
        self._attempts: dict[str, int] = {}
        self._ready_at: dict[str, float] = {}
        self._auto_log_seq = 0
        self._err_streak = 0
        self._warned_refs = False
        self._attempted_tick = False
        self._launched_tick = False

    def _notify(self, kind: str, message: str = "") -> None:
        if self.notify is not None:
            self.notify(kind, message)

    # ---------- main loop ----------

    def run(self) -> list[JobResult]:
        last_beat = time.monotonic()
        ticks = 0
        while (self.pending or self.running) and not self.stop.is_set():
            self._attempted_tick = False
            self._launched_tick = False
            snap = self._safe_snapshot()
            if snap is not None:
                self.tracker.update(snap, occupied=self.claimed.keys())
                self._warn_missing_refs(snap)
                if self.wf.schedule == "sequential":
                    self._tick_sequential()
                else:
                    self._tick_greedy()

            self._reap()

            waiting_for_slot = (
                snap is not None
                and bool(self.pending)
                and not self.stop.is_set()
                and not self._attempted_tick
                and not (self.wf.schedule == "sequential" and self.running)
            )
            if waiting_for_slot:
                self._notify("attempt", "")

            now = time.monotonic()
            if self.running and now - last_beat >= self.wf.heartbeat_s:
                last_beat = now
                self._heartbeat()

            ticks += 1
            if self.max_ticks is not None and ticks >= self.max_ticks:
                break
            if self.pending or self.running:
                self.stop.wait(self.wf.interval_s)

        if self.stop.is_set():
            self._abort_running()
        return self.results

    # ---------- launch strategies ----------

    def _tick_greedy(self) -> None:
        now = time.monotonic()
        for spec in list(self.pending):
            if self.stop.is_set():
                break
            if now < self._ready_at.get(spec.name, 0.0):
                continue
            gpus = self._pick(spec)
            if gpus:
                self.pending.remove(spec)
                self._launch(spec, gpus)

    def _tick_sequential(self) -> None:
        if self.running or not self.pending:
            return
        spec = self.pending[0]
        if time.monotonic() < self._ready_at.get(spec.name, 0.0):
            return
        gpus = self._pick(spec)
        if gpus:
            self.pending.remove(spec)
            self._launch(spec, gpus)

    def _pick(self, spec: JobSpec) -> tuple[int, ...]:
        stable = self.tracker.stable(spec.mem_per_gpu_mb)
        cands = [
            g
            for g in stable
            if g not in self.claimed
            and (spec.allow is None or g in spec.allow)
        ]
        ranked = rank_candidates(cands, spec.prefer)
        if len(ranked) < spec.gpu_num:
            return ()
        return tuple(ranked[: spec.gpu_num])

    # ---------- launch and confirmation ----------

    def _launch(self, spec: JobSpec, gpus: tuple[int, ...]) -> None:
        self._attempted_tick = True
        attempts = self._attempts[spec.name] = self._attempts.get(spec.name, 0) + 1
        log_path = spec.log
        if log_path is None and not spec.passthrough_log:
            self._auto_log_seq += 1
            ts = time.strftime("%Y%m%d-%H%M%S")
            log_path = Path("logs") / f"{_safe_name(spec.name)}-{ts}-{self._auto_log_seq}.log"

        started = time.monotonic()
        try:
            proc, fout = launch(
                spec.cmd, gpus, log_path=log_path, cwd=spec.cwd, extra_env=spec.env or None
            )
        except OSError as e:
            ended = time.monotonic()
            message = f"[{spec.name}] failed to start: {e}"
            self.emit(message)
            self._notify("error", message)
            self.results.append(
                JobResult(
                    name=spec.name,
                    rc=127,
                    gpus=tuple(gpus),
                    started_at=started,
                    ended_at=ended,
                    attempts=attempts,
                )
            )
            return

        if self.bridge is not None:
            self.bridge.attach(proc)
        started_message = (
            f"[{spec.name}] started pid={proc.pid} -> GPU {list(gpus)}"
            + (f" (log: {log_path})" if log_path else "")
        )
        self.emit(started_message)

        try:
            verdict = self._verify(proc, gpus)
        except Exception as e:  # never let a probe/verify bug leak the child or fds
            message = f"[{spec.name}] claim verification failed ({e}); continuing to monitor"
            self.emit(message)
            self._notify("warning", message)
            verdict = CLAIM_UNKNOWN

        if proc.poll() is not None:
            self._retire(spec, proc, fout, gpus, started, attempts)
            return
        if self.stop.is_set():
            self._rollback(spec, proc, fout, gpus, requeue=False, started=started)
            return
        if verdict == CLAIM_LOST:
            self.emit(f"[{spec.name}] GPU {list(gpus)} was preempted; rolling back")
            self._rollback(spec, proc, fout, gpus, requeue=True, started=started)
            return
        if verdict == CLAIM_UNKNOWN:
            self.emit(
                f"[{spec.name}] claim not confirmed within the grace period "
                "(the job may not have initialized CUDA yet); continuing to monitor"
            )
        for g in gpus:
            self.claimed[g] = spec.name
        self.running.append(
            _Running(
                spec=spec,
                proc=proc,
                fout=fout,
                gpus=gpus,
                started_at=started,
                attempts=attempts,
            )
        )
        self._launched_tick = True
        self._notify("launch", started_message)

    def _default_verify(self, proc, gpus: tuple[int, ...]):
        return verify_claim(
            self.probe, proc.pid, gpus,
            grace_s=self.wf.grace_s,
            foreign_stable_s=self.wf.claim_stable_s,
            stop=self.stop,
            give_up=lambda: proc.poll() is not None,
        )

    # ---------- reaping and cleanup ----------

    def _reap(self) -> None:
        for run in list(self.running):
            if run.proc.poll() is not None:
                self._retire(
                    run.spec, run.proc, run.fout, run.gpus, run.started_at, run.attempts
                )

    def _retire(self, spec, proc, fout, gpus, started: float, attempts: int) -> None:
        for run in list(self.running):
            if run.proc is proc:
                self.running.remove(run)
                break
        if self.bridge is not None:
            self.bridge.detach(proc)
        close_log(fout)
        for g in gpus:
            self.claimed.pop(g, None)
        rc = normalize_rc(proc.returncode)
        ended = time.monotonic()
        self.results.append(
            JobResult(
                name=spec.name,
                rc=rc,
                gpus=tuple(gpus),
                started_at=started,
                ended_at=ended,
                attempts=attempts,
            )
        )
        message = f"[{spec.name}] finished rc={rc} after {ended - started:.0f}s"
        self.emit(message)
        self._notify("finish", message)

    def _rollback(self, spec, proc, fout, gpus, *, requeue: bool, started: float) -> None:
        terminate_tree(proc)
        if self.bridge is not None:
            self.bridge.detach(proc)
        close_log(fout)
        # A GPU that just had a foreign process must restabilize before reuse.
        self.tracker.invalidate(gpus)
        ended = time.monotonic()
        attempts = self._attempts.get(spec.name, 1)
        if requeue and attempts < self.wf.max_attempts:
            delay = min(
                max(self.wf.retry_delay_s, 0.0) * (2 ** min(attempts - 1, 5)), 60.0
            )
            self._ready_at[spec.name] = ended + delay
            self.pending.insert(0, spec)
            self.emit(
                f"[{spec.name}] requeued (attempt {attempts}/{self.wf.max_attempts}, "
                f"retry in {delay:.0f}s)"
            )
            self._notify("retry", "")
            return
        if requeue:
            message = (
                f"[{spec.name}] giving up after {attempts} attempt(s); "
                "the claim was repeatedly lost"
            )
            self.emit(message)
            self._notify("error", message)
        else:
            self._notify("abort", f"[{spec.name}] launch aborted")
        self.results.append(
            JobResult(
                name=spec.name,
                rc=None,
                gpus=tuple(gpus),
                started_at=started,
                ended_at=ended,
                attempts=attempts,
            )
        )

    def _abort_running(self) -> None:
        for run in list(self.running):
            message = f"[{run.spec.name}] scheduler aborted; terminating child process group"
            self.emit(message)
            self._notify("abort", message)
            terminate_tree(run.proc)
            if self.bridge is not None:
                self.bridge.detach(run.proc)
            close_log(run.fout)
            for g in run.gpus:
                self.claimed.pop(g, None)
            self.running.remove(run)
            self.results.append(
                JobResult(
                    name=run.spec.name,
                    rc=None,
                    gpus=run.gpus,
                    started_at=run.started_at,
                    ended_at=time.monotonic(),
                    attempts=run.attempts,
                )
            )
        if self.bridge is not None:
            self.bridge.terminate_all()

    def _heartbeat(self) -> None:
        try:
            snap = self.probe.snapshot()
        except Exception:
            return
        parts = []
        for run in self.running:
            infos = [describe(snap[g]) for g in run.gpus if g in snap]
            if infos:
                parts.append(f"[{run.spec.name}] " + " ".join(infos))
        queue = ",".join(s.name for s in self.pending) or "-"
        parts.append(f"queued: {queue}")
        self.emit("running | " + " | ".join(parts))

    def _warn_missing_refs(self, snap: dict) -> None:
        if self._warned_refs or not snap:
            return
        visible = set(snap)
        for spec in self.wf.jobs:
            ghost = sorted((set(spec.prefer) | set(spec.allow or ())) - visible)
            if ghost:
                message = (
                    f"[{spec.name}] warning: configuration references nonexistent "
                    f"GPU ids {ghost}, ignored"
                )
                self.emit(message)
                self._notify("warning", message)
        self._warned_refs = True

    def _safe_snapshot(self) -> dict | None:
        try:
            snap = self.probe.snapshot()
            self._err_streak = 0
            return snap
        except Exception as e:
            self._err_streak += 1
            if self._err_streak == 1 or self._err_streak % 12 == 0:
                message = f"warning: failed to read GPU state ({e}), retrying"
                self.emit(message)
                self._notify("warning", message)
            # None means "probe failed": do not clear every stability window.
            return None
