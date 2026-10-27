import sys
import tempfile
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from helpers import Collector, FakeProbe, cfg, gpu  # noqa: E402

from gpu_hunter.cli import parse_int_list, parse_size_mb, split_argv  # noqa: E402
from gpu_hunter.probe import GpuInfo  # noqa: E402
from gpu_hunter.runner import (  # noqa: E402
    CLAIM_CONFIRMED,
    CLAIM_LOST,
    CLAIM_UNKNOWN,
    build_child_env,
    verify_claim,
)
from gpu_hunter.watcher import HuntConfig, Hunter, is_idle, rank_candidates  # noqa: E402


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

    def test_high_util_means_busy(self):
        self.assertFalse(is_idle(gpu(0, util=90), self.c))

    def test_mem_threshold_strict(self):
        self.assertFalse(is_idle(gpu(0, used=600), self.c))
        self.assertTrue(is_idle(gpu(0, used=100), self.c))

    def test_mem_requirement_relaxed(self):
        c = cfg(mem_per_gpu_mb=40000)
        self.assertTrue(is_idle(gpu(0, used=100), c))
        self.assertFalse(is_idle(gpu(0, used=45000), c))


class RankTest(unittest.TestCase):
    def test_prefer_first_then_index(self):
        self.assertEqual(rank_candidates({3, 0, 2}, (2, 3)), [2, 3, 0])
        self.assertEqual(rank_candidates({1, 0}, ()), [0, 1])


class HunterTest(unittest.TestCase):
    def test_flapping_gpu_never_selected(self):
        frames = [{0: gpu(0)}, {0: gpu(0, used=90000)}] * 10
        out = Hunter(FakeProbe(frames), cfg(max_polls=6), emit=Collector()).hunt()
        self.assertIsNone(out)

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
    def test_confirmed_when_child_visible(self):
        probe = FakeProbe([{0: gpu(0, pids=(4242,))}])
        v = verify_claim(probe, 4242, [0], grace_s=5)
        self.assertEqual(v, CLAIM_CONFIRMED)

    def test_confirmed_after_delay(self):
        probe = FakeProbe([{}, {0: gpu(0)}, {0: gpu(0, pids=(777,))}])
        v = verify_claim(probe, 777, [0], grace_s=30, poll_s=0.01)
        self.assertEqual(v, CLAIM_CONFIRMED)

    def test_lost_on_foreign_process(self):
        probe = FakeProbe([{0: gpu(0, pids=(999,))}])
        v = verify_claim(probe, 4242, [0], grace_s=30, foreign_stable_s=0.05, poll_s=0.01)
        self.assertEqual(v, CLAIM_LOST)

    def test_unknown_on_timeout(self):
        probe = FakeProbe([{}])
        v = verify_claim(probe, 4242, [0], grace_s=0.03, poll_s=0.01)
        self.assertEqual(v, CLAIM_UNKNOWN)

    def test_multi_gpu_all_foreign_loses(self):
        snap = {2: gpu(2, pids=(555,)), 3: gpu(3, pids=(556,))}
        probe = FakeProbe([snap])
        v = verify_claim(probe, 4242, [2, 3], grace_s=30, foreign_stable_s=0.05, poll_s=0.01)
        self.assertEqual(v, CLAIM_LOST)

    def test_mine_on_one_of_two_is_confirmed(self):
        # Multi-process jobs have different worker pids; seeing our own pid is enough
        snap = {2: gpu(2, pids=(4242,)), 3: gpu(3, pids=(555,))}
        probe = FakeProbe([snap])
        v = verify_claim(probe, 4242, [2, 3], grace_s=5)
        self.assertEqual(v, CLAIM_CONFIRMED)

    def test_transient_foreign_then_mine_confirmed(self):
        frames = [
            {0: gpu(0, pids=(999,))},
            {},
            {0: gpu(0, pids=(4242,))},
        ]
        probe = FakeProbe(frames)
        v = verify_claim(probe, 4242, [0], grace_s=30, foreign_stable_s=30, poll_s=0.01)
        self.assertEqual(v, CLAIM_CONFIRMED)


class EnvTest(unittest.TestCase):
    def test_cuda_visible_devices(self):
        env = build_child_env([2, 3])
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "2,3")
        self.assertIn("PATH", env)


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
