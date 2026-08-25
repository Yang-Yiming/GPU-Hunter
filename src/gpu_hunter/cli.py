"""命令行入口:gpu-hunter [选项] -- <你的命令>"""
from __future__ import annotations

import argparse
import logging
import math
import re
import shlex
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Sequence

from . import __version__
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


# ---------- 参数解析辅助 ----------

_UNITS = {
    "": 2**20, "B": 1,
    "K": 2**10, "KB": 2**10, "KIB": 2**10,
    "M": 2**20, "MB": 2**20, "MIB": 2**20,
    "G": 2**30, "GB": 2**30, "GIB": 2**30,
    "T": 2**40, "TB": 2**40, "TIB": 2**40,
}


def parse_size_mb(text: str) -> int:
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([A-Za-z]*)\s*", text)
    if not m:
        raise argparse.ArgumentTypeError(f"无法解析容量:{text!r}(示例 500 / 24000M / 40G)")
    unit = m.group(2).upper()
    if unit not in _UNITS:
        raise argparse.ArgumentTypeError(f"未知容量单位:{m.group(2)!r}")
    return math.ceil(float(m.group(1)) * _UNITS[unit] / 2**20)


def parse_int_list(text: str) -> list[int]:
    out: list[int] = []
    for tok in text.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            v = int(tok)
        except ValueError:
            raise argparse.ArgumentTypeError(f"非法 GPU 编号:{tok!r}") from None
        if v < 0:
            raise argparse.ArgumentTypeError(f"GPU 编号不能为负:{v}")
        if v not in out:
            out.append(v)
    if not out:
        raise argparse.ArgumentTypeError("列表为空")
    return out


def split_argv(argv: Sequence[str]) -> tuple[list[str], list[str] | None]:
    argv = list(argv)
    if "--" in argv:
        i = argv.index("--")
        return argv[:i], argv[i + 1 :]
    return argv, None


# ---------- CLI 定义 ----------

_EPILOG = """\
示例:
  gpu-hunter --gpus 2 --prefer 2,3 -- python train.py --epochs 100
  gpu-hunter --mem-per-gpu 40G --log runs/train.log -- bash run.sh
  gpu-hunter -v --interval 3 --stable-rounds 4 -- ./long_job.sh

提示: 单个 xx.py / xx.sh 会自动补解释器;输出默认透传终端,后台用建议配合 nohup/tmux。
5 秒内连按两次 Ctrl-C 强杀任务组。
"""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gpu-hunter",
        description="轮询显卡状态,出现满足条件的稳定空闲卡后自动启动指定任务。",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--gpus", type=int, default=1, metavar="N",
                   help="需要几张同时空闲的卡(默认 1)")
    p.add_argument("--mem-per-gpu", type=parse_size_mb, default=None, metavar="SIZE",
                   help="单卡显存需求,如 40G / 24000M;缺省表示要求卡完全空闲")
    p.add_argument("--prefer", type=parse_int_list, default=None, metavar="LIST",
                   help="优先卡序,如 2,3(先挑 2 再挑 3)")
    p.add_argument("--allow", type=parse_int_list, default=None, metavar="LIST",
                   help="只在这些卡里挑,如 0,4,5")
    p.add_argument("--interval", type=float, default=5.0, metavar="SEC",
                   help="轮询间隔秒(默认 5)")
    p.add_argument("--stable-rounds", type=int, default=3, metavar="K",
                   help="连续几轮空闲才算稳定(默认 3)")
    p.add_argument("--idle-mem", dest="idle_mem_mb", type=parse_size_mb, default=500,
                   metavar="SIZE",
                   help="占用低于该值视为空闲(默认 500M)")
    p.add_argument("--grace", type=float, default=45.0, metavar="SEC",
                   help="启动后用于确认拿到卡的宽限期秒(默认 45)")
    p.add_argument("--heartbeat", type=float, default=120.0, metavar="SEC",
                   help="任务运行中输出状态的心跳周期秒(默认 120)")
    p.add_argument("--log", type=Path, default=None, metavar="FILE",
                   help="把任务输出写入该文件;缺省透传到当前终端")
    p.add_argument("--cwd", default=None, metavar="DIR",
                   help="子进程工作目录(默认继承当前目录)")
    p.add_argument("-v", "--verbose", action="store_true", help="调试日志")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


def _wrap_script(cmd: list[str]) -> list[str]:
    """用户按最初设想直接给一个脚本文件时,自动补上解释器。"""
    if len(cmd) == 1:
        s = cmd[0]
        if s.endswith(".py"):
            log.info("检测到 .py 脚本,自动改为 %s 执行", sys.executable)
            return [sys.executable, s]
        if s.endswith(".sh"):
            log.info("检测到 .sh 脚本,自动使用 bash 执行")
            return ["bash", s]
    return cmd


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
            log.info("运行中 | %s", " ".join(parts))

    return fn


def _hunt_and_run(probe, cfg: HuntConfig, ns, cmd: list[str], stop: threading.Event,
                  bridge: SignalBridge) -> int:
    hunter = Hunter(probe, cfg, emit=lambda m: log.info("%s", m), stop_event=stop)

    mem_desc = f"每卡需≥{cfg.mem_per_gpu_mb}MB 显存" if cfg.mem_per_gpu_mb is not None else "要求完全空闲"
    allow_desc = ",".join(map(str, sorted(cfg.allow))) if cfg.allow else "全部"
    prefer_desc = ",".join(map(str, cfg.prefer)) or "-"
    log.info("目标:%d 张卡(%s)| 候选范围:%s | 偏好顺序:%s",
             cfg.need, mem_desc, allow_desc, prefer_desc)
    log.info("策略:每 %.0fs 轮询 × 连续 %d 轮空闲才出手;抢卡失败退避重试",
             cfg.interval_s, cfg.stable_rounds)
    log.info("命令:%s", shlex.join(cmd))

    attempt = 0
    backoff = max(cfg.interval_s, 2.0)
    while True:
        attempt += 1
        log.info("第 %d 次守候空闲卡…", attempt)
        gpus = hunter.hunt()
        if gpus is None:
            log.info("等待被中止,退出")
            return 130

        proc, fout = launch(cmd, gpus, ns.log, ns.cwd)
        bridge.proc = proc
        bound = f"CUDA_VISIBLE_DEVICES={','.join(map(str, gpus))}"
        where = f"日志:{ns.log}" if ns.log else "输出透传终端"
        log.info("已启动 pid=%d → %s(%s)", proc.pid, bound, where)

        verdict = verify_claim(
            probe, proc.pid, gpus,
            grace_s=ns.grace, stop=stop,
        )
        if proc.poll() is not None:
            rc = _norm_rc(proc.returncode)
            log.error("任务在确认期前自行退出 rc=%d", rc)
            close_log(fout)
            return rc
        if verdict == CLAIM_LOST:
            log.warning("GPU %s 被其他进程抢占,回滚本次启动并继续等待", list(gpus))
            terminate_tree(proc)
            close_log(fout)
            stop.wait(backoff)
            backoff = min(backoff * 2, 60.0)
            continue
        if verdict == CLAIM_UNKNOWN:
            log.warning("宽限期 %.0fs 内未在 GPU 上看到本进程(pid=%d);"
                        "可能是任务尚未初始化 CUDA 或处于容器内,继续照常监控",
                        ns.grace, proc.pid)

        t0 = time.monotonic()
        try:
            rc_raw = bridge.follow(proc, status_fn=_beat(probe, gpus),
                                   heartbeat_s=ns.heartbeat)
        finally:
            bridge.proc = None
        log.info("任务结束 rc=%s 用时 %s",
                 _norm_rc(rc_raw) if rc_raw is not None else "?", _fmt_dur(time.monotonic() - t0))
        close_log(fout)
        return _norm_rc(proc.returncode)


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

    cmd = [str(t) for t in (cmd_tokens or [])]
    if not cmd:
        parser.error('缺少待执行命令:请用 "--" 分隔选项与命令,'
                     "例如 gpu-hunter --gpus 1 -- python train.py")
    cmd = _wrap_script(cmd)

    try:
        from .probe import NVMLProbe
    except Exception as e:
        log.error("%s", e)
        return 2

    cfg = HuntConfig(
        need=ns.gpus,
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
                return _hunt_and_run(probe, cfg, ns, cmd, stop, bridge)
    except KeyboardInterrupt:
        log.info("收到中断,退出")
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
