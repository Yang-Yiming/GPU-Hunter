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

        try:
            proc, fout = launch(cmd, gpus, ns.log, ns.cwd)
        except OSError as e:
            log.error("failed to launch %s: %s", shlex.join(cmd), e)
            return 127
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
            return rc if rc is not None else 1
        if verdict == CLAIM_LOST:
            log.warning("GPU %s was claimed by another process; rolling back this launch "
                        "and continuing to wait", list(gpus))
            terminate_tree(proc)
            bridge.detach(proc)
            close_log(fout)
            if progress is not None:
                progress.failed()
            stop.wait(backoff)
            backoff = min(backoff * 2, 60.0)
            continue
        if verdict == CLAIM_UNKNOWN:
            log.info("child pid=%d was not seen on the GPU within the %.0fs grace period; "
                     "it may not have initialized CUDA yet or may be inside a container, "
                     "continuing to monitor as usual",
                     proc.pid, ns.grace)

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
        return rc if rc is not None else 0


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


def main(argv: Sequence[str] | None = None) -> int:
    opts, cmd_tokens = split_argv(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    ns = parser.parse_args(opts)

    if ns.verbose:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s %(levelname)-7s %(message)s",
            datefmt="%H:%M:%S",
            stream=sys.stderr,
        )
        progress = None
    else:
        progress = Progress()
        handler = _ProgressLogHandler(progress)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S"
        ))
        root = logging.getLogger()
        if not root.handlers:
            root.addHandler(handler)
        root.setLevel(logging.WARNING)

    has_cmd = bool(cmd_tokens)
    if ns.workflow and has_cmd:
        parser.error("WORKFLOW and \"-- command\" are mutually exclusive")

    wf = None
    cmd: list[str] = []
    if ns.workflow:
        from .spec import SpecError, load_workflow

        try:
            wf = load_workflow(ns.workflow)
        except SpecError as e:
            log.error("%s", e)
            return 2
    else:
        cmd = [str(t) for t in (cmd_tokens or [])]
        if not cmd:
            parser.error('missing command or workflow file: '
                         "gpu-hunter workflow.yaml or gpu-hunter [options] -- python train.py")
        cmd = _wrap_script(cmd)

    try:
        from .probe import NVMLProbe
    except Exception as e:
        log.error("%s", e)
        return 2

    cfg = None
    if wf is None:
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
    stop = threading.Event()
    bridge = SignalBridge(stop)
    bridge.install()

    try:
        with instance_lock():
            with NVMLProbe() as probe:
                if wf is not None:
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
