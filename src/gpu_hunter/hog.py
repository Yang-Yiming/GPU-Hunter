"""Placeholder ("hog") process used by reserve mode to hold GPU memory.

The process allocates a fixed amount of device memory through the CUDA driver
API (``libcuda.so.1``) and then idles until it is asked to stop or its own
timeout expires.  It deliberately avoids a torch/tensorflow dependency: the
driver library is always present on a machine with a working NVIDIA driver.

The device is always CUDA device 0 because the parent launches one hog per GPU
with ``CUDA_VISIBLE_DEVICES`` pointing at exactly that card.
"""
from __future__ import annotations

import argparse
import ctypes
import os
import signal
import sys
import threading
import time
from typing import Callable

__all__ = ["CudaDriver", "HogError", "hold", "main"]

_DRIVER_LIBRARIES = ("libcuda.so.1", "libcuda.so", "libcuda.dylib")


class HogError(RuntimeError):
    pass


class CudaDriver:
    """Minimal ctypes binding for the CUDA driver API calls we need."""

    def __init__(self, lib) -> None:
        self._lib = lib
        self._ctx: ctypes.c_void_p | None = None
        self._ptr: ctypes.c_void_p | None = None

    # -- loading -------------------------------------------------------
    @classmethod
    def load(cls) -> "CudaDriver":
        errors = []
        for name in _DRIVER_LIBRARIES:
            try:
                return cls(ctypes.CDLL(name))
            except OSError as e:  # pragma: no cover - depends on the host
                errors.append(f"{name}: {e}")
        raise HogError(
            "cannot load the NVIDIA driver library "
            f"({_DRIVER_LIBRARIES[0]}); tried: {'; '.join(errors)}"
        )

    def _symbol(self, *names: str):
        for name in names:
            fn = getattr(self._lib, name, None)
            if fn is not None:
                return fn
        raise HogError(f"CUDA driver is missing required symbol {names[0]!r}")

    def _check(self, rc: int, what: str) -> None:
        if rc == 0:
            return
        name = None
        try:
            get_name = self._symbol("cuGetErrorName")
            text = ctypes.c_char_p()
            get_name.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
            get_name.restype = ctypes.c_int
            if get_name(rc, ctypes.byref(text)) == 0 and text.value:
                name = text.value.decode("ascii", "replace")
        except Exception:
            name = None
        raise HogError(f"{what} failed (CUresult={rc}{': ' + name if name else ''})")

    # -- context / memory ---------------------------------------------
    def init(self) -> None:
        cu_init = self._symbol("cuInit")
        cu_init.argtypes = [ctypes.c_uint]
        cu_init.restype = ctypes.c_int
        self._check(cu_init(0), "cuInit")

    def allocate(self, mem_bytes: int) -> None:
        """Create a context on device 0 and allocate *mem_bytes* of device memory."""
        cu_device_get = self._symbol("cuDeviceGet")
        cu_device_get.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        cu_device_get.restype = ctypes.c_int
        device = ctypes.c_int()
        self._check(cu_device_get(ctypes.byref(device), 0), "cuDeviceGet")

        cu_ctx_create = self._symbol("cuCtxCreate_v2", "cuCtxCreate")
        cu_ctx_create.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_uint,
            ctypes.c_int,
        ]
        cu_ctx_create.restype = ctypes.c_int
        ctx = ctypes.c_void_p()
        self._check(cu_ctx_create(ctypes.byref(ctx), 0, device.value), "cuCtxCreate")
        self._ctx = ctypes.c_void_p(ctx.value)

        cu_mem_alloc = self._symbol("cuMemAlloc_v2", "cuMemAlloc")
        cu_mem_alloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        cu_mem_alloc.restype = ctypes.c_int
        ptr = ctypes.c_void_p()
        self._check(cu_mem_alloc(ctypes.byref(ptr), ctypes.c_size_t(mem_bytes)), "cuMemAlloc")
        self._ptr = ctypes.c_void_p(ptr.value)

    def release(self) -> None:
        if self._ptr is not None:
            try:
                cu_mem_free = self._symbol("cuMemFree_v2", "cuMemFree")
                cu_mem_free.argtypes = [ctypes.c_void_p]
                cu_mem_free.restype = ctypes.c_int
                cu_mem_free(self._ptr)
            except Exception:
                pass
            self._ptr = None
        if self._ctx is not None:
            try:
                cu_ctx_destroy = self._symbol("cuCtxDestroy_v2", "cuCtxDestroy")
                cu_ctx_destroy.argtypes = [ctypes.c_void_p]
                cu_ctx_destroy.restype = ctypes.c_int
                cu_ctx_destroy(self._ctx)
            except Exception:
                pass
            self._ctx = None


def hold(
    mem_mb: int,
    *,
    timeout_s: float = 0.0,
    emit: Callable[[str], None] | None = None,
    driver: CudaDriver | None = None,
) -> int:
    """Allocate ``mem_mb`` on device 0 and block until stopped or timed out.

    Returns a process exit code: 0 on a normal stop/expiry, 2 when the driver
    or the allocation is unavailable.
    """
    say = emit if emit is not None else (lambda message: print(message, flush=True))

    if driver is None:
        try:
            driver = CudaDriver.load()
        except HogError as e:
            print(f"gpu-hunter hog: {e}", file=sys.stderr)
            return 2
    try:
        driver.init()
        driver.allocate(int(mem_mb) * 1024 * 1024)
    except HogError as e:
        print(f"gpu-hunter hog: {e}", file=sys.stderr)
        try:
            driver.release()
        except Exception:
            pass
        return 2

    stop = threading.Event()

    def _handler(_signum, _frame) -> None:
        stop.set()

    previous = {}
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            previous[sig] = signal.signal(sig, _handler)
        except (ValueError, OSError):  # not the main thread
            pass

    say(f"hog ready: pid={os.getpid()} mem={mem_mb}MB timeout={timeout_s or 'never'}")
    deadline = time.monotonic() + timeout_s if timeout_s and timeout_s > 0 else None
    try:
        while not stop.is_set():
            wait = 5.0
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    say(f"hog expired: pid={os.getpid()}")
                    break
                wait = min(wait, remaining)
            stop.wait(wait)
    finally:
        try:
            driver.release()
        finally:
            for sig, handler in previous.items():
                try:
                    signal.signal(sig, handler)
                except (ValueError, OSError):
                    pass
    say(f"hog released: pid={os.getpid()}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gpu-hunter.hog",
        description="Hold GPU memory by allocating it through the CUDA driver API.",
    )
    p.add_argument("--mem-mb", type=int, required=True, metavar="MB",
                   help="amount of device memory to allocate in MiB")
    p.add_argument("--timeout", type=float, default=0.0, metavar="SEC",
                   help="release automatically after SEC seconds; 0 means never (default)")
    return p


def main(argv=None) -> int:
    ns = build_parser().parse_args(argv)
    if ns.mem_mb < 1:
        print("gpu-hunter hog: --mem-mb must be >= 1", file=sys.stderr)
        return 2
    if ns.timeout < 0:
        print("gpu-hunter hog: --timeout must be >= 0", file=sys.stderr)
        return 2
    return hold(ns.mem_mb, timeout_s=ns.timeout)


if __name__ == "__main__":
    sys.exit(main())
