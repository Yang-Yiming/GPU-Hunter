"""Command-line entry point: gpu-hunter [options] -- <your command>"""
from __future__ import annotations

import argparse
import logging
import shlex
import sys
import threading
import time
from pathlib import Path
from typing import Sequence

from . import __version__, parsing
from .lock import LockError, instance_lock
from .progress import Progress
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
from .watcher import HuntConfig, Hunter

log = logging.getLogger("gpu-hunter")


class _ProgressLogHandler(logging.StreamHandler):
    """Keep WARNING/ERROR lines from being glued onto a pending x-line."""

    def __init__(self, progress: Progress) -> None:
        super().__init__(sys.stderr)
        self.progress = progress

    def emit(self, record: logging.LogRecord) -> None:
        self.progress.break_line()
        super().emit(record)


# ---------- argument parsing helpers ----------

def parse_size_mb(text: str) -> int:
    try:
        return parsing.parse_size_mb(text)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from None


def parse_int_list(text: str) -> list[int]:
    try:
        return parsing.parse_int_list(text)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from None


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"must be an integer, got {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {value}")
    return value


def _nonneg_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"must be an integer, got {text!r}") from None
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0, got {value}")
    return value


def _positive_float(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"must be a number, got {text!r}") from None
    if value <= 0:
        raise argparse.ArgumentTypeError(f"must be > 0, got {value}")
    return value


def _nonneg_float(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"must be a number, got {text!r}") from None
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0, got {value}")
    return value


def _path_arg(text: str) -> Path:
    return Path(text).expanduser()


def split_argv(argv: Sequence[str]) -> tuple[list[str], list[str] | None]:
    argv = list(argv)
    if "--" in argv:
        i = argv.index("--")
        return argv[:i], argv[i + 1 :]
    return argv, None


# ---------- CLI definition ----------

_EPILOG = """\
Examples:
  gpu-hunter --gpu-num 2 --prefer 2,3 -- python train.py --epochs 100
  gpu-hunter --mem-per-gpu 40G --log runs/train.log -- bash run.sh
  gpu-hunter -v --interval 3 --stable-rounds 4 -- ./long_job.sh
  gpu-hunter workflow.yaml            # multi-job queue (greedy / sequential)

  # hold 2 x 12G for later use, Ctrl-C releases (or hand over a command)
  gpu-hunter --reserve --gpu-num 2 --mem-per-gpu 12G
  gpu-hunter --reserve --gpu-num 2 --mem-per-gpu 12G --detach
  gpu-hunter --reserve-status
  gpu-hunter --release --run -- bash

Tip: a lone xx.py / xx.sh gets an interpreter automatically; output goes to the
terminal by default, so pair this with nohup/tmux when running in the background.
Press Ctrl-C twice within 5 seconds to force-kill every job group.
"""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gpu-hunter",
        description="Poll GPU state and launch the given job once enough cards are stably free.",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--gpu-num", type=_positive_int, default=1, metavar="N",
                   help="number of simultaneously free GPUs required (default: 1)")
    p.add_argument("--mem-per-gpu", type=parse_size_mb, default=None, metavar="SIZE",
                   help="free memory required per GPU, e.g. 40G / 24000M; "
                        "default means the card must be completely idle")
    p.add_argument("--prefer", type=parse_int_list, default=None, metavar="LIST",
                   help="preferred GPU order, e.g. 2,3 (try 2 first, then 3)")
    p.add_argument("--allow", type=parse_int_list, default=None, metavar="LIST",
                   help="only pick from these GPUs, e.g. 0,4,5")
    p.add_argument("--interval", type=_positive_float, default=5.0, metavar="SEC",
                   help="polling interval in seconds (default: 5)")
    p.add_argument("--stable-rounds", type=_positive_int, default=3, metavar="K",
                   help="consecutive idle rounds required for stability (default: 3)")
    p.add_argument("--idle-mem", dest="idle_mem_mb", type=parse_size_mb, default=500,
                   metavar="SIZE",
                   help="used memory below this value counts as idle (default: 500M)")
    p.add_argument("--idle-util", type=_nonneg_int, default=5, metavar="PCT",
                   help="utilization at or below this value counts as idle (default: 5)")
    p.add_argument("--grace", type=_nonneg_float, default=45.0, metavar="SEC",
                   help="grace period in seconds for confirming the claim after launch "
                        "(default: 45)")
    p.add_argument("--claim-stable", type=_nonneg_float, default=8.0, metavar="SEC",
                   help="how long a foreign process must persist before the claim is "
                        "considered lost (default: 8)")
    p.add_argument("--heartbeat", type=_positive_float, default=120.0, metavar="SEC",
                   help="heartbeat interval in seconds for status output while the job runs "
                        "(default: 120)")
    p.add_argument("--log", type=_path_arg, default=None, metavar="FILE",
                   help="write job output to this file; default streams to the terminal")
    p.add_argument("--cwd", type=_path_arg, default=None, metavar="DIR",
                   help="working directory for the child process (default: current directory)")
    # ---- reservation (hold) mode ----
    p.add_argument("--reserve", action="store_true",
                   help="hold the requested GPUs for later use instead of launching a job; "
                        "runs in the foreground until Ctrl-C releases the reservation "
                        "(needs --mem-per-gpu or --reserve-full)")
    p.add_argument("--reserve-status", dest="reserve_status", action="store_true",
                   help="show the active reservation, its GPUs and remaining hold time")
    p.add_argument("--release", action="store_true",
                   help="release the active reservation; with --run, hand the GPUs over "
                        "to the command after \"--\"")
    p.add_argument("--detach", action="store_true",
                   help="with --reserve: acquire the GPUs and exit, keeping the "
                        "reservation for a later --release")
    p.add_argument("--hold-max", dest="hold_max", type=_nonneg_float,
                   default=12 * 3600.0, metavar="SEC",
                   help="how long a reservation may be held before its placeholders "
                        "self-expire; 0 means never (default: 43200)")
    p.add_argument("--reserve-full", dest="reserve_full", action="store_true",
                   help="with --reserve: hold all remaining free memory per GPU instead "
                        "of exactly --mem-per-gpu")
    p.add_argument("--hold-cmd", dest="hold_cmd", default=None, metavar="CMD",
                   help="custom placeholder command for --reserve; {mem_mb}, {gpu} and "
                        "{timeout} are substituted (default: the built-in CUDA hog)")
    p.add_argument("--run", action="store_true",
                   help="with --release or --reserve: run the command after \"--\" once "
                        "the reservation has been released")
    p.add_argument("--force", action="store_true",
                   help="with --release: clear the reservation even when it cannot be "
                        "fully verified (foreign host, dead placeholders)")
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    p.add_argument("workflow", nargs="?", default=None, metavar="WORKFLOW",
                   help="YAML workflow file (.yaml/.yml); mutually exclusive with \"-- command\". "
                        "When given, enters multi-job queue mode and ignores the other "
                        "single-job options")
    return p


def _wrap_script(cmd: list[str]) -> list[str]:
    """Add an interpreter when the user passes a lone script file."""
    wrapped = parsing.ensure_interpreter(cmd)
    if wrapped != cmd:
        log.info("script argument detected, adding interpreter automatically: %s", shlex.join(wrapped))
    return wrapped


def _fmt_dur(sec: float) -> str:
    sec = int(sec)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _beat(probe, gpus: Sequence[int]):
    def fn() -> None:
        try:
            snap = probe.snapshot()
        except Exception:
            return
        from .probe import describe

        parts = [describe(snap[g]) for g in gpus if g in snap]
        if parts:
            log.info("running | %s", " ".join(parts))

    return fn


def _launch_and_follow(probe, gpus, cmd: list[str], ns, stop: threading.Event,
                       bridge: SignalBridge, progress: Progress | None = None) -> tuple[int, str]:
    """Launch *cmd* on *gpus*, verify the claim and wait for the job to finish.

    Returns ``(exit_code, verdict)`` where verdict is one of ``finished``,
    ``exited``, ``lost`` or ``launch_error``.  ``lost`` means the claim was
    preempted; the caller may retry on other GPUs.
    """
    try:
        proc, fout = launch(cmd, gpus, ns.log, ns.cwd)
    except OSError as e:
        log.error("failed to launch %s: %s", shlex.join(cmd), e)
        return 127, "launch_error"
    bridge.attach(proc)
    bound = f"CUDA_VISIBLE_DEVICES={','.join(map(str, gpus))}"
    where = f"log: {ns.log}" if ns.log else "output streams to the terminal"
    log.info("launched pid=%d -> %s (%s)", proc.pid, bound, where)
    if progress is not None:
        progress.say(f"launched pid={proc.pid} -> {bound} ({where})")

    verdict = verify_claim(
        probe, proc.pid, gpus,
        grace_s=ns.grace,
        foreign_stable_s=ns.claim_stable,
        stop=stop,
        give_up=lambda p=proc: p.poll() is not None,
    )
    if proc.poll() is not None:
        rc = normalize_rc(proc.returncode)
        log.error("job exited on its own before the claim was confirmed, rc=%s", rc)
        bridge.detach(proc)
        close_log(fout)
        return (rc if rc is not None else 1), "exited"
    if verdict == CLAIM_LOST:
        log.warning("GPU %s was claimed by another process; rolling back this launch "
                    "and continuing to wait", list(gpus))
        terminate_tree(proc)
        bridge.detach(proc)
        close_log(fout)
        if progress is not None:
            progress.failed()
        return 1, "lost"
    if verdict == CLAIM_UNKNOWN:
        log.info("child pid=%d was not seen on the GPU within the %.0fs grace period; "
                 "it may not have initialized CUDA yet or may be inside a container, "
                 "continuing to monitor as usual", proc.pid, ns.grace)

    t0 = time.monotonic()
    rc_raw = bridge.follow(proc, status_fn=_beat(probe, gpus),
                           heartbeat_s=ns.heartbeat)
    elapsed = _fmt_dur(time.monotonic() - t0)
    rc = normalize_rc(rc_raw) if rc_raw is not None else "?"
    log.info("job finished rc=%s after %s", rc, elapsed)
    if progress is not None:
        progress.say(f"job finished rc={rc} after {elapsed}")
    close_log(fout)
    rc = normalize_rc(proc.returncode)
    return (rc if rc is not None else 0), "finished"


def _hunt_and_run(probe, cfg: HuntConfig, ns, cmd: list[str], stop: threading.Event,
                  bridge: SignalBridge, progress: Progress | None = None) -> int:
    def hunter_emit(message: str) -> None:
        if progress is None:
            log.info("%s", message)
        elif message.startswith("warning:"):
            progress.say(message)

    def on_attempt(ok: bool, message: str) -> None:
        if progress is None:
            return
        if ok:
            progress.say(message)
        else:
            progress.failed()

    hunter = Hunter(
        probe, cfg,
        emit=hunter_emit,
        stop_event=stop,
        on_attempt=on_attempt if progress is not None else None,
    )

    mem_desc = (
        f"needs >= {cfg.mem_per_gpu_mb}MB free per GPU"
        if cfg.mem_per_gpu_mb is not None
        else "must be completely idle"
    )
    allow_desc = ",".join(map(str, sorted(cfg.allow))) if cfg.allow else "all"
    prefer_desc = ",".join(map(str, cfg.prefer)) or "-"
    log.info("target: %d GPU(s) (%s) | candidates: %s | preference: %s",
             cfg.need, mem_desc, allow_desc, prefer_desc)
    log.info("strategy: poll every %.0fs, require %d consecutive idle rounds; "
             "back off and retry if the claim is lost",
             cfg.interval_s, cfg.stable_rounds)
    log.info("command: %s", shlex.join(cmd))

    attempt = 0
    backoff = max(cfg.interval_s, 2.0)
    while True:
        attempt += 1
        log.info("waiting for idle GPUs (attempt %d)...", attempt)
        gpus = hunter.hunt()
        if gpus is None:
            log.info("wait aborted, exiting")
            return 130
        rc, verdict = _launch_and_follow(probe, gpus, cmd, ns, stop, bridge, progress)
        if verdict == "lost":
            stop.wait(backoff)
            backoff = min(backoff * 2, 60.0)
            continue
        return rc


def _run_workflow(probe, wf, ns, stop: threading.Event, bridge: SignalBridge,
                  progress: Progress | None = None) -> int:
    from .scheduler import Scheduler, format_summary

    log.info("workflow: %s | %d job(s) | schedule: %s", wf.path, len(wf.jobs), wf.schedule)
    log.info("tick %.0fs x %d stable rounds; grace period %.0fs",
             wf.interval_s, wf.stable_rounds, wf.grace_s)
    for j in wf.jobs:
        need = f"{j.gpu_num} GPU(s)" + (
            f" >= {j.mem_per_gpu_mb}MB" if j.mem_per_gpu_mb else ""
        )
        log.info("  - %s (%s): %s", j.name, need, shlex.join(j.cmd))

    def notify(kind: str, message: str) -> None:
        if progress is None:
            return
        if kind in ("attempt", "retry"):
            progress.failed()
        elif message:
            progress.say(message)

    sched = Scheduler(
        probe, wf,
        emit=(lambda m: log.info("%s", m)) if progress is None else (lambda m: None),
        stop_event=stop,
        bridge=bridge,
        notify=notify if progress is not None else None,
    )
    interrupted = False
    try:
        results = sched.run()
    except KeyboardInterrupt:
        interrupted = True
        stop.set()
        sched._abort_running()
        results = list(sched.results)

    def report(message: str) -> None:
        if progress is None:
            log.info("%s", message)
        else:
            progress.say(message)

    if interrupted or stop.is_set():
        bridge.terminate_all()
        if results:
            report("partial results:\n" + format_summary(results))
        report("workflow aborted")
        return 130

    fails = sum(1 for r in results if not r.ok)
    report(f"all jobs finished: {len(results) - fails} succeeded / {fails} failed")
    report(format_summary(results))
    return min(fails, 125)


# ---------- reservation (hold) mode ----------

def _setup_logging(ns) -> Progress | None:
    if ns.verbose:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s %(levelname)-7s %(message)s",
            datefmt="%H:%M:%S",
            stream=sys.stderr,
        )
        return None
    progress = Progress()
    handler = _ProgressLogHandler(progress)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S"
    ))
    root = logging.getLogger()
    if not root.handlers:
        root.addHandler(handler)
    root.setLevel(logging.WARNING)
    return progress


def _resolve_mode(ns, cmd_tokens, parser) -> str:
    has_cmd = bool(cmd_tokens)
    hold_modes = [bool(ns.reserve), bool(ns.reserve_status), bool(ns.release)]
    if sum(hold_modes) > 1:
        parser.error("--reserve, --reserve-status and --release are mutually exclusive")
    if ns.detach and not ns.reserve:
        parser.error("--detach is only valid with --reserve")
    if ns.force and not ns.release:
        parser.error("--force is only valid with --release")
    if ns.run and not (ns.reserve or ns.release):
        parser.error("--run is only valid with --reserve or --release")
    if ns.detach and ns.run:
        parser.error("--run cannot be combined with --detach")
    if ns.workflow:
        if any(hold_modes):
            parser.error(
                "a workflow file cannot be combined with "
                "--reserve/--reserve-status/--release"
            )
        if has_cmd:
            parser.error("WORKFLOW and \"-- command\" are mutually exclusive")
        return "workflow"
    if ns.reserve:
        if ns.run and not has_cmd:
            parser.error('--run requires a command after "--"')
        if has_cmd and not ns.run:
            parser.error('--reserve accepts a command only together with --run')
        if not ns.mem_per_gpu and not ns.reserve_full:
            parser.error("--reserve requires --mem-per-gpu SIZE or --reserve-full")
        return "reserve"
    if ns.reserve_status:
        if has_cmd:
            parser.error("--reserve-status does not accept a command")
        return "reserve-status"
    if ns.release:
        if ns.run and not has_cmd:
            parser.error('--run requires a command after "--"')
        if has_cmd and not ns.run:
            parser.error('--release accepts a command only together with --run')
        return "release"
    if has_cmd:
        return "single"
    parser.error('missing command or workflow file: '
                 "gpu-hunter workflow.yaml or gpu-hunter [options] -- python train.py")
    raise AssertionError("unreachable")


def _report(progress: Progress | None, message: str) -> None:
    if progress is None:
        log.info("%s", message)
    else:
        progress.say(message)


def _reserve_emit(progress: Progress | None):
    def emit(message: str) -> None:
        if progress is None:
            log.info("%s", message)
        elif message.startswith(("warning:", "holding")):
            progress.say(message)

    return emit


def _reserve_on_attempt(progress: Progress | None):
    if progress is None:
        return None

    def on_attempt(ok: bool, message: str) -> None:
        if ok:
            progress.say(message)
        else:
            progress.failed()

    return on_attempt


def _reserve_options(ns):
    from .reserve import AcquireOptions

    return AcquireOptions(
        need=ns.gpu_num,
        mem_mb=ns.mem_per_gpu,
        full=ns.reserve_full,
        prefer=tuple(ns.prefer or ()),
        allow=frozenset(ns.allow) if ns.allow else None,
        idle_mem_mb=ns.idle_mem_mb,
        idle_util=ns.idle_util,
        interval_s=ns.interval,
        stable_rounds=ns.stable_rounds,
        hold_max_s=ns.hold_max,
        hold_cmd=ns.hold_cmd,
    )


def _reserve_summary(state) -> str:
    ids = ",".join(str(gpu) for gpu in state.gpu_ids)
    left = "never"
    if state.expires_at is not None:
        left = _fmt_dur(max(0.0, state.expires_at - time.time()))
    amount = "full card" if state.full else f"{state.mem_mb}MB"
    return f"reserved GPU {ids} ({len(state.gpu_ids)} x {amount}); hold time left: {left}"


def _handoff(progress: Progress | None, gpus: list[int]) -> None:
    visible = ",".join(str(gpu) for gpu in gpus)
    _report(progress, f"released GPU {visible}")
    _report(progress, f"hand-off: export CUDA_VISIBLE_DEVICES={visible}")


def _probe_for_hold(required: bool):
    from .probe import NVMLProbe

    try:
        return NVMLProbe()
    except Exception as e:
        if required:
            raise
        log.warning("GPU probing unavailable: %s", e)
        return None


def _run_reserve_status(progress: Progress | None) -> int:
    from . import reserve as reserve_mod

    probe = _probe_for_hold(required=False)
    try:
        report = reserve_mod.status_report(probe)
    finally:
        if probe is not None:
            probe.close()
    if progress is None:
        for line in report.text.splitlines():
            log.info("%s", line)
    else:
        progress.say(report.text)
    return report.exit_code


def _run_reserve(ns, cmd: list[str], progress: Progress | None,
                 bridge: SignalBridge) -> int:
    from . import reserve as reserve_mod
    from .probe import NVMLProbe

    try:
        probe = NVMLProbe()
    except Exception as e:
        log.error("%s", e)
        return 2
    emit = _reserve_emit(progress)
    on_attempt = _reserve_on_attempt(progress)
    opts = _reserve_options(ns)
    with probe:
        try:
            try:
                if ns.detach:
                    with instance_lock():
                        state = reserve_mod.acquire(
                            probe, opts, emit=emit, on_attempt=on_attempt
                        )
                    _report(progress, _reserve_summary(state))
                    _report(progress, "the placeholders keep holding the GPUs in the "
                                      "background; release with: gpu-hunter --release")
                    return 0
                with instance_lock():
                    state = reserve_mod.acquire(
                        probe, opts, emit=emit, on_attempt=on_attempt
                    )
            except KeyboardInterrupt:
                # The interrupt may have arrived right after the state file was
                # written; never leave a half-created foreground reservation.
                try:
                    reserve_mod.release(
                        probe, None, emit=emit, term_wait=5.0, free_wait=5.0
                    )
                except reserve_mod.ReserveError:
                    pass
                log.info("interrupted while waiting for idle GPUs, exiting")
                return 130
            _report(progress, _reserve_summary(state))
            target = "the command" if cmd else "your own task"
            _report(progress, f"holding for {target}; Ctrl-C releases the reservation"
                              + (" and runs the command" if cmd else ""))
            supervisor = reserve_mod.Supervisor(
                probe, state, opts, emit=emit, on_attempt=on_attempt,
                interval_s=max(opts.interval_s, 2.0),
            )
            try:
                reason = supervisor.run()
            except KeyboardInterrupt:
                reason = "interrupted"
            state = supervisor.state
        except reserve_mod.ReserveError as e:
            log.error("%s", e)
            return 2
        except KeyboardInterrupt:
            log.info("interrupted while waiting for idle GPUs, exiting")
            return 130

        if reason == "missing":
            _report(progress, "the reservation was released externally")
        elif reason == "expired":
            _report(progress, "the reservation expired; releasing it now")
        try:
            gpus = reserve_mod.release(probe, state, force=ns.force, emit=emit)
        except reserve_mod.ReserveError as e:
            log.error("failed to release the reservation: %s", e)
            return 1
        except KeyboardInterrupt:
            log.warning("interrupted during release; check with: gpu-hunter --reserve-status")
            return 130
        _handoff(progress, gpus)
        if not cmd:
            return 130 if reason == "interrupted" else 0
        run_stop = threading.Event()
        try:
            with instance_lock():
                rc, _verdict = _launch_and_follow(
                    probe, gpus, cmd, ns, run_stop, bridge, progress
                )
        except LockError as e:
            log.error("%s", e)
            return 2
        return rc


def _run_release(ns, cmd: list[str], progress: Progress | None,
                 bridge: SignalBridge) -> int:
    from . import reserve as reserve_mod

    emit = _reserve_emit(progress)
    try:
        probe = _probe_for_hold(required=bool(cmd))
    except Exception as e:
        log.error("%s", e)
        return 2
    try:
        try:
            state = reserve_mod.load_state()
        except reserve_mod.ReserveError as e:
            if not ns.force:
                log.error("%s", e)
                return 1
            log.warning("clearing unreadable reservation state: %s", e)
            reserve_mod.clear_state()
            _report(progress, "cleared the stale reservation state")
            return 0
        if state is None:
            if ns.force:
                _report(progress, "no active reservation")
                return 0
            log.error("no active reservation (nothing to release)")
            return 2
        try:
            gpus = reserve_mod.release(probe, state, force=ns.force, emit=emit)
        except reserve_mod.ReserveError as e:
            log.error("failed to release the reservation: %s", e)
            return 1
        except KeyboardInterrupt:
            log.warning("interrupted during release; check with: gpu-hunter --reserve-status")
            return 130
        _handoff(progress, gpus)
        if not cmd:
            return 0
        run_stop = threading.Event()
        try:
            with instance_lock():
                rc, _verdict = _launch_and_follow(
                    probe, gpus, cmd, ns, run_stop, bridge, progress
                )
        except LockError as e:
            log.error("%s", e)
            return 2
        return rc
    finally:
        if probe is not None:
            probe.close()


def main(argv: Sequence[str] | None = None) -> int:
    opts, cmd_tokens = split_argv(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    ns = parser.parse_args(opts)
    progress = _setup_logging(ns)
    mode = _resolve_mode(ns, cmd_tokens, parser)

    wf = None
    cmd: list[str] = [str(t) for t in (cmd_tokens or [])]
    if mode == "workflow":
        from .spec import SpecError, load_workflow

        try:
            wf = load_workflow(ns.workflow)
        except SpecError as e:
            log.error("%s", e)
            return 2
    elif mode == "single":
        cmd = _wrap_script(cmd)

    hold_mode = mode in ("reserve", "reserve-status", "release")
    stop = threading.Event()
    # In hold mode Ctrl-C must interrupt the foreground hold/release path
    # instead of being forwarded to a child (there is none until a --run
    # hand-off, after which the bridge behaves normally again).
    bridge = SignalBridge(None if hold_mode else stop)
    bridge.install()
    try:
        if mode == "reserve-status":
            return _run_reserve_status(progress)
        if mode == "release":
            return _run_release(ns, cmd, progress, bridge)
        if mode == "reserve":
            return _run_reserve(ns, cmd, progress, bridge)

        try:
            from .probe import NVMLProbe
        except Exception as e:
            log.error("%s", e)
            return 2

        cfg = None
        if mode == "single":
            try:
                cfg = HuntConfig(
                    need=ns.gpu_num,
                    mem_per_gpu_mb=ns.mem_per_gpu,
                    prefer=tuple(ns.prefer) if ns.prefer else (),
                    allow=frozenset(ns.allow) if ns.allow else None,
                    idle_mem_mb=ns.idle_mem_mb,
                    idle_util=ns.idle_util,
                    interval_s=ns.interval,
                    stable_rounds=ns.stable_rounds,
                )
            except ValueError as e:
                parser.error(str(e))
        with instance_lock():
            with NVMLProbe() as probe:
                if mode == "workflow":
                    return _run_workflow(probe, wf, ns, stop, bridge, progress)
                return _hunt_and_run(probe, cfg, ns, cmd, stop, bridge, progress)
    except LockError as e:
        log.error("%s", e)
        return 2
    except KeyboardInterrupt:
        bridge.terminate_all()
        log.info("interrupted, exiting")
        return 130
    except Exception as e:
        from .probe import ProbeError

        if isinstance(e, ProbeError):
            log.error("%s", e)
            return 2
        raise
    finally:
        if progress is not None:
            progress.finish()
        # Never let a managed child survive an unexpected exit path.
        bridge.terminate_all()
        bridge.restore()


if __name__ == "__main__":
    sys.exit(main())
