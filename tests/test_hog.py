"""Tests for the CUDA placeholder process (no GPU required)."""
import io
import sys
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import gpu_hunter.hog as hog  # noqa: E402


class FakeDriver:
    def __init__(self):
        self.calls = []

    def init(self):
        self.calls.append("init")

    def allocate(self, mem_bytes):
        self.calls.append(("alloc", mem_bytes))

    def release(self):
        self.calls.append("release")


class HoldTest(unittest.TestCase):
    def test_allocates_exact_memory_and_releases_after_timeout(self):
        driver = FakeDriver()
        messages = []
        rc = hog.hold(12 * 1024, timeout_s=0.05, emit=messages.append, driver=driver)
        self.assertEqual(rc, 0)
        self.assertEqual(driver.calls[0], "init")
        self.assertEqual(driver.calls[1], ("alloc", 12 * 1024 * 1024 * 1024))
        self.assertEqual(driver.calls[-1], "release")
        self.assertTrue(any("hog ready" in m for m in messages))
        self.assertTrue(any("hog expired" in m for m in messages))

    def test_returns_error_when_allocation_fails(self):
        class Broken(FakeDriver):
            def allocate(self, mem_bytes):
                self.calls.append(("alloc-failed", mem_bytes))
                raise hog.HogError("CUDA_ERROR_OUT_OF_MEMORY")

        err = io.StringIO()
        driver = Broken()
        with redirect_stderr(err):
            rc = hog.hold(1024, timeout_s=0.01, emit=lambda m: None, driver=driver)
        self.assertEqual(rc, 2)
        self.assertIn("OUT_OF_MEMORY", err.getvalue())
        self.assertEqual(driver.calls[-1], "release")

    def test_returns_error_without_driver(self):
        with patch.object(hog.CudaDriver, "load", side_effect=hog.HogError("no driver")):
            err = io.StringIO()
            with redirect_stderr(err):
                rc = hog.hold(1024, timeout_s=0.01, emit=lambda m: None)
        self.assertEqual(rc, 2)
        self.assertIn("no driver", err.getvalue())


class DriverTest(unittest.TestCase):
    def test_load_fails_without_library(self):
        with patch.object(hog.ctypes, "CDLL", side_effect=OSError("missing")):
            with self.assertRaises(hog.HogError):
                hog.CudaDriver.load()

    def test_missing_symbol_is_reported(self):
        driver = hog.CudaDriver(object())
        with self.assertRaises(hog.HogError):
            driver._symbol("cuInit", "cuInit_v2")


class MainTest(unittest.TestCase):
    def test_rejects_invalid_memory(self):
        err = io.StringIO()
        with redirect_stderr(err):
            self.assertEqual(hog.main(["--mem-mb", "0"]), 2)
        self.assertIn("--mem-mb", err.getvalue())

    def test_rejects_negative_timeout(self):
        err = io.StringIO()
        with redirect_stderr(err):
            self.assertEqual(hog.main(["--mem-mb", "1", "--timeout", "-1"]), 2)
        self.assertIn("--timeout", err.getvalue())

    def test_main_uses_loaded_driver(self):
        driver = FakeDriver()
        with patch.object(hog.CudaDriver, "load", return_value=driver):
            rc = hog.main(["--mem-mb", "1024", "--timeout", "0.01"])
        self.assertEqual(rc, 0)
        self.assertEqual(driver.calls[0], "init")
        self.assertEqual(driver.calls[-1], "release")


if __name__ == "__main__":
    unittest.main()
