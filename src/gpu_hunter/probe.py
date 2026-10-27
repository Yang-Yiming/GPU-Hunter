"""GPU state probing: talks to NVML directly (the library behind nvidia-smi), no text parsing."""
from __future__ import annotations

from dataclasses import dataclass

try:
    import pynvml
except Exception:
    pynvml = None


class ProbeError(RuntimeError):
    pass


@dataclass(frozen=True)
class GpuInfo:
    index: int
    name: str
    mem_used_mb: int
    mem_total_mb: int
    util_percent: int
    compute_pids: tuple[int, ...]

    @property
    def mem_free_mb(self) -> int:
        return self.mem_total_mb - self.mem_used_mb


def describe(info: GpuInfo) -> str:
    pids = ",".join(str(p) for p in info.compute_pids) or "-"
    return (
        f"GPU{info.index}[{info.name}] "
        f"{info.mem_used_mb}/{info.mem_total_mb}MB util={info.util_percent}% pids={pids}"
    )


def _name_of(handle) -> str:
    try:
        n = pynvml.nvmlDeviceGetName(handle)
    except pynvml.NVMLError:
        return "unknown"
    return n.decode() if isinstance(n, bytes) else str(n)


def _compute_pids(handle) -> tuple[int, ...]:
    for attr in (
        "nvmlDeviceGetComputeRunningProcesses_v3",
        "nvmlDeviceGetComputeRunningProcesses_v2",
        "nvmlDeviceGetComputeRunningProcesses",
    ):
        fn = getattr(pynvml, attr, None)
        if fn is None:
            continue
        try:
            return tuple(int(p.pid) for p in fn(handle))
        except pynvml.NVMLError:
            continue
    return ()


class NVMLProbe:
    """Each snapshot() call returns {gpu_index: GpuInfo}."""

    def __init__(self) -> None:
        if pynvml is None:
            raise ProbeError("missing dependency nvidia-ml-py: pip install nvidia-ml-py")
        try:
            pynvml.nvmlInit()
        except pynvml.NVMLError as e:
            raise ProbeError(
                f"NVML initialization failed: {e} "
                "(driver not loaded or no NVIDIA device present?)"
            ) from e
        try:
            count = pynvml.nvmlDeviceGetCount()
            self._handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(count)]
        except pynvml.NVMLError as e:
            self.close()
            raise ProbeError(f"failed to enumerate GPUs: {e}") from e
        self._names = [_name_of(h) for h in self._handles]

    def snapshot(self) -> dict[int, GpuInfo]:
        out: dict[int, GpuInfo] = {}
        for i, h in enumerate(self._handles):
            try:
                mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                used, total = int(mem.used), int(mem.total)
                try:
                    util = int(pynvml.nvmlDeviceGetUtilizationRates(h).gpu)
                except pynvml.NVMLError:
                    util = -1
            except pynvml.NVMLError as e:
                raise ProbeError(f"failed to read state of GPU{i}: {e}") from e
            out[i] = GpuInfo(
                index=i,
                name=self._names[i],
                mem_used_mb=used // 1024**2,
                mem_total_mb=total // 1024**2,
                util_percent=util,
                compute_pids=_compute_pids(h),
            )
        return out

    def close(self) -> None:
        if pynvml is not None:
            try:
                pynvml.nvmlShutdown()
            except pynvml.NVMLError:
                pass

    def __enter__(self) -> "NVMLProbe":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
