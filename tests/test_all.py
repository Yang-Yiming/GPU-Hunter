import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from unittest.mock import patch  # noqa: E402

from helpers import Collector, FakeProbe, cfg, gpu  # noqa: E402

import gpu_hunter.runner as runner_mod  # noqa: E402
from gpu_hunter.cli import parse_int_list, parse_size_mb, split_argv  # noqa: E402
from gpu_hunter.probe import GpuInfo  # noqa: E402
from gpu_hunter.runner import (  # noqa: E402
    CLAIM_CONFIRMED,
    CLAIM_LOST,
    CLAIM_UNKNOWN,
    SignalBridge,
    build_child_env,
    is_owned_by_current_user,
    terminate_tree,
    verify_claim,
)
from gpu_hunter.watcher import HuntConfig, Hunter, is_idle, rank_candidates  # noqa: E402


def spawn_sleeper() -> subprocess.Popen:
    return subprocess.Popen(["sleep", "30"], start_new_session=True)


def kill_tree(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except OSError:
        pass
    try:
        proc.wait(timeout=5)
    except Exception:
        pass


def spawn_bash_with_child() -> tuple[subprocess.Popen, int]:
    """A shell wrapper plus its own (different-pid) child, sharing the session."""
    proc = subprocess.Popen(["bash", "-c", "sleep 30 & wait"], start_new_session=True)
    for _ in range(200):
        try:
            text = Path(f"/proc/{proc.pid}/task/{proc.pid}/children").read_text()
        except OSError:
            text = ""
        children = [int(x) for x in text.split()]
        if children:
            return proc, children[0]
        time.sleep(0.01)
    kill_tree(proc)
    raise RuntimeError("bash did not spawn a child")



class ParseSizeTest(unittest.TestCase):
    def test_units(self):
        self.assertEqual(parse_size_mb("500"), 500)
        self.assertEqual(parse_size_mb("40G"), 40960)
        self.assertEqual(parse_size_mb("40GiB"), 40960)
        self.assertEqual(parse_size_mb("24000M"), 24000)
        self.assertEqual(parse_size_mb("1.5G"), 1536)
        self.assertEqual(parse_size_mb("2T"), 2097152)

    def test_invalid(self):
        from argparse import ArgumentTypeError

        for bad in ("abc", "10Q", "-5G", ""):
            with self.assertRaises(ArgumentTypeError):
                parse_size_mb(bad)


class ParseListTest(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(parse_int_list("2,3"), [2, 3])
        self.assertEqual(parse_int_list(" 1 , 1 ,0 "), [1, 0])

    def test_invalid(self):
        from argparse import ArgumentTypeError

        for bad in ("a,b", "1,-2", ","):
            with self.assertRaises(ArgumentTypeError):
                parse_int_list(bad)


class SplitArgvTest(unittest.TestCase):
    def test_with_marker(self):
        opts, cmd = split_argv(["-v", "--gpu-num", "2", "--", "python", "x.py"])
        self.assertEqual(opts, ["-v", "--gpu-num", "2"])
        self.assertEqual(cmd, ["python", "x.py"])

    def test_without_marker(self):
        opts, cmd = split_argv(["-v"])
        self.assertEqual(opts, ["-v"])
        self.assertIsNone(cmd)


class IsIdleTest(unittest.TestCase):
    def setUp(self):
        self.c = cfg()

    def test_compute_pid_means_busy(self):
        self.assertFalse(is_idle(gpu(0, pids=(9,)), self.c))

    def test_unknown_compute_pids_means_busy(self):
        self.assertFalse(is_idle(gpu(0, pids=None), self.c))

    def test_high_util_means_busy(self):
        self.assertFalse(is_idle(gpu(0, util=90), self.c))

    def test_mem_threshold_strict(self):
        self.assertFalse(is_idle(gpu(0, used=600), self.c))
        self.assertTrue(is_idle(gpu(0, used=100), self.c))

    def test_mem_requirement_relaxed(self):
        c = cfg(mem_per_gpu_mb=40000)
        self.assertTrue(is_idle(gpu(0, used=100), c))
        self.assertFalse(is_idle(gpu(0, used=45000), c))


class TrackerTest(unittest.TestCase):
    def test_invalidate_requires_restabilization(self):
        from gpu_hunter.watcher import AvailabilityTracker

        tracker = AvailabilityTracker(stable_rounds=1)
        tracker.update({0: gpu(0)})
        self.assertEqual(tracker.stable(), {0})
        tracker.invalidate([0])
        tracker.update({0: gpu(0)})
        self.assertEqual(tracker.stable(), {0})
        tracker.invalidate([0])
        self.assertEqual(tracker.stable(), set())


class RankTest(unittest.TestCase):
    def test_prefer_first_then_index(self):
        self.assertEqual(rank_candidates({3, 0, 2}, (2, 3)), [2, 3, 0])
        self.assertEqual(rank_candidates({1, 0}, ()), [0, 1])


class HunterTest(unittest.TestCase):
    def test_flapping_gpu_never_selected(self):
        frames = [{0: gpu(0)}, {0: gpu(0, used=90000)}] * 10
        out = Hunter(FakeProbe(frames), cfg(max_polls=6), emit=Collector()).hunt()
        self.assertIsNone(out)

    def test_on_attempt_reports_failures_then_success(self):
        frames = [{0: gpu(0, used=90000)}, {0: gpu(0)}]
        calls = []
        out = Hunter(
            FakeProbe(frames),
            cfg(stable_rounds=1),
            emit=Collector(),
            on_attempt=lambda ok, msg: calls.append((ok, msg)),
        ).hunt()
        self.assertEqual(out, [0])
        self.assertEqual(calls[0], (False, ""))
        self.assertTrue(calls[-1][0])
        self.assertIn("matched idle GPU(s)", calls[-1][1])

    def test_stable_gpu_selected(self):
        frames = [{0: gpu(0)}, {0: gpu(0)}]
        out = Hunter(FakeProbe(frames), cfg(), emit=Collector()).hunt()
        self.assertEqual(out, [0])

    def test_atomic_need_two(self):
        frames = [
            {0: gpu(0)},
            {0: gpu(0), 1: gpu(1)},
        ]
        out = Hunter(FakeProbe(frames), cfg(need=2, stable_rounds=1), emit=Collector()).hunt()
        self.assertEqual(out, [0, 1])

    def test_not_enough_gpus_waits(self):
        frames = [{0: gpu(0)}]
        out = Hunter(FakeProbe(frames), cfg(need=2), emit=Collector()).hunt()
        self.assertIsNone(out)

    def test_prefer_order_respected(self):
        frames = [{0: gpu(0), 2: gpu(2), 3: gpu(3)}]
        out = Hunter(
            FakeProbe(frames), cfg(need=2, prefer=(3, 2), stable_rounds=1),
            emit=Collector(),
        ).hunt()
        self.assertEqual(out, [3, 2])

    def test_allow_list_filters(self):
        frames = [
            {0: gpu(0), 1: gpu(1, used=90000)},
            {0: gpu(0), 1: gpu(1)},
        ]
        out = Hunter(
            FakeProbe(frames), cfg(allow=frozenset({1}), stable_rounds=1),
            emit=Collector(),
        ).hunt()
        self.assertEqual(out, [1])

    def test_stop_event_aborts(self):
        stop = threading.Event()
        stop.set()
        probe = FakeProbe([{0: gpu(0)}])
        out = Hunter(probe, cfg(stable_rounds=1), emit=Collector(), stop_event=stop).hunt()
        self.assertIsNone(out)
        self.assertEqual(probe.calls, 0)


class VerifyClaimTest(unittest.TestCase):
    def setUp(self):
        self._procs: list[subprocess.Popen] = []

    def tearDown(self):
        for proc in self._procs:
            kill_tree(proc)

    def spawn(self) -> subprocess.Popen:
        proc = spawn_sleeper()
        self._procs.append(proc)
        return proc

    def test_confirmed_when_child_visible(self):
        child = self.spawn()
        probe = FakeProbe([{0: gpu(0, pids=(child.pid,))}])
        v = verify_claim(probe, child.pid, [0], grace_s=5)
        self.assertEqual(v, CLAIM_CONFIRMED)

    def test_confirmed_after_delay(self):
        child = self.spawn()
        probe = FakeProbe([{}, {0: gpu(0)}, {0: gpu(0, pids=(child.pid,))}])
        v = verify_claim(probe, child.pid, [0], grace_s=30, poll_s=0.01)
        self.assertEqual(v, CLAIM_CONFIRMED)

    def test_lost_on_foreign_process(self):
        child = self.spawn()
        foreign = self.spawn()
        probe = FakeProbe([{0: gpu(0, pids=(foreign.pid,))}])
        v = verify_claim(
            probe, child.pid, [0], grace_s=30, foreign_stable_s=0.05, poll_s=0.01
        )
        self.assertEqual(v, CLAIM_LOST)

    def test_unknown_on_timeout(self):
        child = self.spawn()
        probe = FakeProbe([{}])
        v = verify_claim(probe, child.pid, [0], grace_s=0.03, poll_s=0.01)
        self.assertEqual(v, CLAIM_UNKNOWN)

    def test_multi_gpu_all_foreign_loses(self):
        child = self.spawn()
        foreign_a = self.spawn()
        foreign_b = self.spawn()
        snap = {2: gpu(2, pids=(foreign_a.pid,)), 3: gpu(3, pids=(foreign_b.pid,))}
        probe = FakeProbe([snap])
        v = verify_claim(
            probe, child.pid, [2, 3], grace_s=30, foreign_stable_s=0.05, poll_s=0.01
        )
        self.assertEqual(v, CLAIM_LOST)

    def test_mine_on_one_of_two_is_confirmed(self):
        # Multi-process jobs have different worker pids; seeing our own session is enough
        child = self.spawn()
        foreign = self.spawn()
        snap = {2: gpu(2, pids=(child.pid,)), 3: gpu(3, pids=(foreign.pid,))}
        probe = FakeProbe([snap])
        v = verify_claim(probe, child.pid, [2, 3], grace_s=5)
        self.assertEqual(v, CLAIM_CONFIRMED)

    def test_transient_foreign_then_mine_confirmed(self):
        child = self.spawn()
        foreign = self.spawn()
        frames = [
            {0: gpu(0, pids=(foreign.pid,))},
            {},
            {0: gpu(0, pids=(child.pid,))},
        ]
        probe = FakeProbe(frames)
        v = verify_claim(probe, child.pid, [0], grace_s=30, foreign_stable_s=30, poll_s=0.01)
        self.assertEqual(v, CLAIM_CONFIRMED)

    def test_wrapper_grandchild_counts_as_ours(self):
        # gpu-hunter -- bash run.sh: the GPU pid is a child of the launched shell,
        # not the shell pid itself.  Session inheritance must identify it as ours.
        wrapper, grandchild = spawn_bash_with_child()
        self._procs.append(wrapper)
        probe = FakeProbe([{0: gpu(0, pids=(grandchild,))}])
        v = verify_claim(probe, wrapper.pid, [0], grace_s=5, foreign_stable_s=0.01, poll_s=0.01)
        self.assertEqual(v, CLAIM_CONFIRMED)

    def test_invisible_pid_is_not_preemption(self):
        # Container/invisible pids must not cause us to kill our own job.
        child = self.spawn()
        probe = FakeProbe([{0: gpu(0, pids=(99_999_999,))}])
        v = verify_claim(
            probe, child.pid, [0], grace_s=0.05, foreign_stable_s=0.01, poll_s=0.01
        )
        self.assertEqual(v, CLAIM_UNKNOWN)


class EnvTest(unittest.TestCase):
    def test_cuda_visible_devices(self):
        env = build_child_env([2, 3])
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "2,3")
        self.assertEqual(env["CUDA_DEVICE_ORDER"], "PCI_BUS_ID")
        self.assertIn("PATH", env)


class SafetyTest(unittest.TestCase):
    def test_current_user_owned(self):
        self.assertTrue(is_owned_by_current_user(os.getpid()))
        self.assertFalse(is_owned_by_current_user(99_999_999))

    def test_terminate_tree_kills_own_process_group(self):
        proc = spawn_sleeper()
        try:
            terminate_tree(proc, term_wait=1.0)
            self.assertIsNotNone(proc.poll())
        finally:
            kill_tree(proc)

    def test_terminate_tree_refuses_foreign_owner(self):
        proc = spawn_sleeper()
        try:
            with patch.object(runner_mod, "_proc_uid", return_value=os.getuid() + 1):
                with self.assertLogs("gpu-hunter", level="WARNING") as logs:
                    terminate_tree(proc, term_wait=0.2)
            self.assertIn("refusing", logs.output[0])
            self.assertIsNone(proc.poll(), "must not signal a process owned by another uid")
        finally:
            kill_tree(proc)

    def test_terminate_tree_refuses_reused_pid(self):
        proc = spawn_sleeper()
        try:
            start = runner_mod._proc_start_time(proc.pid)
            self.assertIsNotNone(start)
            proc._gpu_hunter_start_time = start + 1
            with self.assertLogs("gpu-hunter", level="WARNING") as logs:
                terminate_tree(proc, term_wait=0.2)
            self.assertIn("identity changed", logs.output[0])
            self.assertIsNone(proc.poll(), "must not signal a different process with a reused pid")
        finally:
            kill_tree(proc)

    def test_signal_bridge_sets_stop_event(self):
        proc = spawn_sleeper()
        stop = threading.Event()
        bridge = SignalBridge(stop)
        bridge.attach(proc)
        try:
            with patch.object(runner_mod, "_send_group", return_value=True):
                bridge._handle(signal.SIGINT, None)
            self.assertTrue(stop.is_set())
        finally:
            kill_tree(proc)


class LaunchSmokeTest(unittest.TestCase):
    def test_launch_writes_log(self):
        from gpu_hunter.runner import launch

        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "nested" / "out.log"
            proc, fout = launch(
                [sys.executable, "-c", "print('hunter-ok')"], [2], log_path=log_path
            )
            rc = proc.wait(timeout=30)
            if fout:
                fout.close()
            self.assertEqual(rc, 0)
            self.assertIn(b"hunter-ok", log_path.read_bytes())


if __name__ == "__main__":
    unittest.main()
