"""工作流 YAML 的加载与校验:WorkflowSpec / JobSpec。"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .parsing import coerce_cmd, coerce_path, ensure_interpreter, parse_int_list, parse_size_mb

log = logging.getLogger("gpu-hunter")


class SpecError(ValueError):
    pass


_SCHEDULES = ("greedy", "sequential")

_JOB_KEYS = {"name", "cmd", "gpu-num", "mem-per-gpu", "prefer", "allow", "env", "log", "cwd"}
_DEFAULT_KEYS = _JOB_KEYS - {"name", "cmd"}


@dataclass
class JobSpec:
    name: str
    cmd: list[str]
    gpu_num: int = 1
    mem_per_gpu_mb: int | None = None
    prefer: tuple[int, ...] = ()
    allow: frozenset[int] | None = None
    env: dict[str, str] = field(default_factory=dict)
    log: Path | None = None          # None → 调度器自动生成;- 透传终端
    cwd: Path | None = None


@dataclass
class WorkflowSpec:
    path: Path
    schedule: str = "greedy"
    interval_s: float = 5.0
    stable_rounds: int = 3
    idle_mem_mb: int = 500
    grace_s: float = 45.0
    heartbeat_s: float = 120.0
    jobs: list[JobSpec] = field(default_factory=list)


def _err(ctx: str, msg: str) -> SpecError:
    return SpecError(f"{ctx}: {msg}")


def _expect_mapping(value, ctx: str) -> dict:
    if not isinstance(value, dict):
        raise _err(ctx, "必须是映射(table)")
    return value


def _opt_int(value, ctx: str, default: int | None = None, *, minimum: int | None = None) -> int | None:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise _err(ctx, f"必须是整数,得到 {value!r}")
    if minimum is not None and value < minimum:
        raise _err(ctx, f"必须 ≥ {minimum},得到 {value}")
    return value


def _opt_float(value, ctx: str, default: float, *, minimum: float) -> float:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _err(ctx, f"必须是数字,得到 {value!r}")
    v = float(value)
    if v < minimum:
        raise _err(ctx, f"必须 ≥ {minimum},得到 {v}")
    return v


def _opt_size(value, ctx: str) -> int | None:
    if value is None:
        return None
    try:
        return parse_size_mb(value) if isinstance(value, str) else _opt_int(value, ctx, minimum=1)
    except ValueError as e:
        raise _err(ctx, str(e)) from None


def _opt_ids(value, ctx: str):
    if value is None:
        return None
    try:
        ids = parse_int_list(value)
    except ValueError as e:
        raise _err(ctx, str(e)) from None
    return tuple(ids) if ctx.endswith("prefer") else frozenset(ids)


def _opt_env(value, ctx: str) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise _err(ctx, "env 必须是字符串到字符串的映射")
    out = {}
    for k, v in value.items():
        if isinstance(v, (dict, list)):
            raise _err(f"{ctx}.env[{k}]", "值必须是标量")
        out[str(k)] = str(v)
    return out


def _build_job(raw, index: int, defaults: dict) -> JobSpec:
    raw = _expect_mapping(raw, f"jobs[{index}]")
    unknown = set(raw) - _JOB_KEYS
    if unknown:
        raise _err(f"jobs[{index}]", f"未知字段 {sorted(unknown)},可用:{sorted(_JOB_KEYS)}")
    ctx = f"jobs[{index}]"

    name = raw.get("name")
    if name is not None and (not isinstance(name, str) or not name.strip()):
        raise _err(f"{ctx}.name", "必须是非空字符串")

    cmd_value = raw.get("cmd", defaults.get("cmd"))
    if cmd_value is None:
        raise _err(ctx, "缺少必填字段 cmd")
    try:
        cmd = ensure_interpreter(coerce_cmd(cmd_value, ctx))
    except ValueError as e:
        raise _err(ctx, str(e)) from None

    gpu_num = _opt_int(
        raw.get("gpu-num", defaults.get("gpu-num")), f"{ctx}.gpu-num", 1, minimum=1
    )
    mem = _opt_size(raw.get("mem-per-gpu", defaults.get("mem-per-gpu")), f"{ctx}.mem-per-gpu")
    prefer = _opt_ids(raw.get("prefer", defaults.get("prefer")), f"{ctx}.prefer") or ()
    allow = _opt_ids(raw.get("allow", defaults.get("allow")), f"{ctx}.allow")

    env = _opt_env(defaults.get("env"), "defaults.env")
    env.update(_opt_env(raw.get("env"), f"{ctx}.env"))

    log_v = raw.get("log", defaults.get("log"))
    log_path = None
    if log_v is not None:
        if isinstance(log_v, str) and log_v.strip() == "-":
            log_path = None
        else:
            log_path = coerce_path(log_v, f"{ctx}.log")

    cwd_v = raw.get("cwd", defaults.get("cwd"))
    cwd = coerce_path(cwd_v, f"{ctx}.cwd") if cwd_v is not None else None

    return JobSpec(
        name=name or f"job-{index + 1}",
        cmd=cmd,
        gpu_num=gpu_num,
        mem_per_gpu_mb=mem,
        prefer=tuple(prefer),
        allow=allow,
        env=env,
        log=log_path,
        cwd=cwd,
    )


def load_workflow(path: str | Path) -> WorkflowSpec:
    path = Path(path)
    if not path.is_file():
        raise SpecError(f"工作流文件不存在:{path}")
    try:
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise SpecError(f"YAML 语法错误({path}):{e}") from None
    root = _expect_mapping(doc, str(path))

    unknown = set(root) - {
        "schedule", "interval", "stable-rounds", "idle-mem", "grace",
        "heartbeat", "defaults", "jobs",
    }
    if unknown:
        raise _err(str(path), f"未知顶层字段 {sorted(unknown)}")

    schedule = root.get("schedule", "greedy")
    if schedule not in _SCHEDULES:
        raise _err("schedule", f"必须是 {'/'.join(map(repr, _SCHEDULES))},得到 {schedule!r}")

    jobs_raw = root.get("jobs")
    if not jobs_raw or not isinstance(jobs_raw, list):
        raise _err("jobs", "必须是非空列表")

    defaults_raw = root.get("defaults") or {}
    defaults_raw = _expect_mapping(defaults_raw, "defaults")
    unknown_d = set(defaults_raw) - _DEFAULT_KEYS
    if unknown_d:
        raise _err("defaults", f"未知字段 {sorted(unknown_d)},可用:{sorted(_DEFAULT_KEYS)}")

    wf = WorkflowSpec(
        path=path,
        schedule=schedule,
        interval_s=_opt_float(root.get("interval"), "interval", 5.0, minimum=0.1),
        stable_rounds=_opt_int(root.get("stable-rounds"), "stable-rounds", 3, minimum=1),
        idle_mem_mb=_opt_size(root.get("idle-mem"), "idle-mem") or 500,
        grace_s=_opt_float(root.get("grace"), "grace", 45.0, minimum=0.0),
        heartbeat_s=_opt_float(root.get("heartbeat"), "heartbeat", 120.0, minimum=1.0),
    )
    wf.jobs = [_build_job(j, i, defaults_raw) for i, j in enumerate(jobs_raw)]

    names = [j.name for j in wf.jobs]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise _err("jobs", f"name 重复:{dupes}")
    return wf
