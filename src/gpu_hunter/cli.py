"""Command-line entry point: gpu-hunter [options] -- <your command>"""
from __future__ import annotations

import argparse
import logging
import shlex
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Sequence

from . import __version__, parsing
from .lock import instance_lock
from .runner import (
    CLAIM_LOST,
    CLAIM_UNKNOWN,
    SignalBridge,
    close_log,
    launch,
    terminate_tree,
    verify_claim,
)
from .watcher import HuntConfig, Hunter

log = logging.getLogger("gpu-hunter")


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
    p.add_argument("--gpu-num", type=int, default=1, metavar="N",
                   help="number of simultaneously free GPUs required (default: 1)")
    p.add_argument("--mem-per-gpu", type=parse_size_mb, default=None, metavar="SIZE",
                   help="free memory required per GPU, e.g. 40G / 24000M; "
                        "default means the card must be completely idle")
    p.add_argument("--prefer", type=parse_int_list, default=None, metavar="LIST",
                   help="preferred GPU order, e.g. 2,3 (try 2 first, then 3)")
    p.add_argument("--allow", type=parse_int_list, default=None, metavar="LIST",
                   help="only pick from these GPUs, e.g. 0,4,5")
    p.add_argument("--interval", type=float, default=5.0, metavar="SEC",
                   help="polling interval in seconds (default: 5)")
    p.add_argument("--stable-rounds", type=int, default=3, metavar="K",
                   help="consecutive idle rounds required for stability (default: 3)")
    p.add_argument("--idle-mem", dest="idle_mem_mb", type=parse_size_mb, default=500,
                   metavar="SIZE",
                   help="used memory below this value counts as idle (default: 500M)")
    p.add_argument("--grace", type=float, default=45.0, metavar="SEC",
                   help="grace period in seconds for confirming the claim after launch "
                        "(default: 45)")
    p.add_argument("--heartbeat", type=float, default=120.0, metavar="SEC",
                   help="heartbeat interval in seconds for status output while the job runs "
                        "(default: 120)")
    p.add_argument("--log", type=Path, default=None, metavar="FILE",
                   help="write job output to this file; default streams to the terminal")
    p.add_argument("--cwd", default=None, metavar="DIR",
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


def _norm_rc(rc: int) -> int:
    return 128 + (-rc) if rc < 0 else rc


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
                  bridge: SignalBridge) -> int:
    hunter = Hunter(probe, cfg, emit=lambda m: log.info("%s", m), stop_event=stop)

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

        proc, fout = launch(cmd, gpus, ns.log, ns.cwd)
        bridge.attach(proc)
        bound = f"CUDA_VISIBLE_DEVICES={','.join(map(str, gpus))}"
        where = f"log: {ns.log}" if ns.log else "output streams to the terminal"
        log.info("launched pid=%d -> %s (%s)", proc.pid, bound, where)

        verdict = verify_claim(
            probe, proc.pid, gpus,
            grace_s=ns.grace, stop=stop,
            give_up=lambda p=proc: p.poll() is not None,
        )
        if proc.poll() is not None:
            rc = _norm_rc(proc.returncode)
            log.error("job exited on its own before the claim was confirmed, rc=%d", rc)
            close_log(fout)
            return rc
        if verdict == CLAIM_LOST:
            log.warning("GPU %s was claimed by another process; rolling back this launch "
                        "and continuing to wait", list(gpus))
            terminate_tree(proc)
            close_log(fout)
            stop.wait(backoff)
            backoff = min(backoff * 2, 60.0)
            continue
        if verdict == CLAIM_UNKNOWN:
            log.warning("child pid=%d was not seen on the GPU within the %.0fs grace period; "
                        "it may not have initialized CUDA yet or may be inside a container, "
                        "continuing to monitor as usual",
                        proc.pid, ns.grace)

        t0 = time.monotonic()
        rc_raw = bridge.follow(proc, status_fn=_beat(probe, gpus),
                               heartbeat_s=ns.heartbeat)
        log.info("job finished rc=%s after %s",
                 _norm_rc(rc_raw) if rc_raw is not None else "?", _fmt_dur(time.monotonic() - t0))
        close_log(fout)
        return _norm_rc(proc.returncode)


def _run_workflow(probe, wf, ns, stop: threading.Event, bridge: SignalBridge) -> int:
    from .scheduler import Scheduler, format_summary

    log.info("workflow: %s | %d job(s) | schedule: %s", wf.path, len(wf.jobs), wf.schedule)
    log.info("tick %.0fs x %d stable rounds; grace period %.0fs",
             wf.interval_s, wf.stable_rounds, wf.grace_s)
    for j in wf.jobs:
        need = f"{j.gpu_num} GPU(s)" + (
            f" >= {j.mem_per_gpu_mb}MB" if j.mem_per_gpu_mb else ""
        )
        log.info("  - %s (%s): %s", j.name, need, shlex.join(j.cmd))

    sched = Scheduler(
        probe, wf,
        emit=lambda m: log.info("%s", m),
        stop_event=stop,
        bridge=bridge,
    )
    interrupted = False
    try:
        results = sched.run()
    except KeyboardInterrupt:
        interrupted = True
        results = []

    if interrupted or stop.is_set():
        bridge.terminate_all()
        log.info("workflow aborted")
        return 130

    fails = sum(1 for r in results if not r.ok)
    log.info("all jobs finished: %d succeeded / %d failed", len(results) - fails, fails)
    log.info("\n%s", format_summary(results))
    return min(fails, 125)


def main(argv: Sequence[str] | None = None) -> int:
    opts, cmd_tokens = split_argv(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    ns = parser.parse_args(opts)

    logging.basicConfig(
        level=logging.DEBUG if ns.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )

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
        cfg = HuntConfig(
            need=ns.gpu_num,
            mem_per_gpu_mb=ns.mem_per_gpu,
            prefer=tuple(ns.prefer) if ns.prefer else (),
            allow=frozenset(ns.allow) if ns.allow else None,
            idle_mem_mb=ns.idle_mem_mb,
            interval_s=ns.interval,
            stable_rounds=ns.stable_rounds,
        )
    stop = threading.Event()
    bridge = SignalBridge()
    bridge.install()

    try:
        with instance_lock():
            with NVMLProbe() as probe:
                if wf is not None:
                    return _run_workflow(probe, wf, ns, stop, bridge)
                return _hunt_and_run(probe, cfg, ns, cmd, stop, bridge)
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
        bridge.restore()


if __name__ == "__main__":
    sys.exit(main())
