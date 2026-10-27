"""Tests for reservation (hold) mode: state file, acquire/release/status, supervisor.

No real GPU is needed: the probe is faked and the placeholders are ordinary
``sleep`` processes (``--hold-cmd``) whose memory usage is simulated in the
probe frames.  The CUDA allocation itself is covered by test_hog.py.
"""
import json
import os
import shlex
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

warnings.filterwarnings("ignore", category=ResourceWarning)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from helpers import FakeProbe, gpu  # noqa: E402

import gpu_hunter.reserve as reserve_mod  # noqa: E402
import gpu_hunter.runner as runner_mod  # noqa: E402
from gpu_hunter.reserve import (  # noqa: E402
    AcquireOptions,
    HogRecord,
    Reservation,
    ReserveError,
    Supervisor,
    acquire,
    hog_command,
    load_state,
    release,
    status_report,
)


def sleep_cmd(seconds: int = 60) -> str:
    return shlex.join([sys.executable, "-c", f"import time; time.sleep({seconds})"])


def free_frame(gpus=(0, 1), used=100):
    return {g: gpu(g, used=used) for g in gpus}


def held_frame(gpus, mem_mb, used_before=100):
    return {g: gpu(g, used=used_before + mem_mb) for g in gpus}


def record_for(proc: subprocess.Popen, index=0, mem_mb=1024) -> HogRecord:
    start = runner_mod.process_start_time(proc.pid)
    assert start is not None
    return HogRecord(
        index=index, pid=proc.pid, start_time=start, mem_mb=mem_mb, used_before_mb=0
    )


def eager_frames(mem_mb=1024):
    """Frames for an eager 2-GPU reservation: GPU0 first, GPU1 later."""
    free0 = gpu(0, used=100)
    busy1 = gpu(1, used=90000)
    held0 = gpu(0, used=100 + mem_mb, pids=(4242,))
    free1 = gpu(1, used=100)
    held1 = gpu(1, used=100 + mem_mb, pids=(4242,))
    return [
        {0: free0, 1: busy1},   # hunt for GPU0
        {0: free0, 1: busy1},
        {0: free0, 1: busy1},   # baseline for the GPU0 placeholder
        {0: held0, 1: busy1},   # GPU0 hold confirmed
        {0: held0, 1: busy1},   # hunt for GPU1 (history is shared)
        {0: held0, 1: free1},
        {0: held0, 1: free1},   # GPU1 becomes stable
        {0: held0, 1: free1},   # baseline for the GPU1 placeholder
        {0: held0, 1: held1},   # GPU1 hold confirmed
    ]


class ReservationTestBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._saved_env = {
            key: os.environ.get(key)
            for key in ("GPU_HUNTER_STATE_DIR", "XDG_CACHE_HOME")
        }
        os.environ["GPU_HUNTER_STATE_DIR"] = str(Path(self._tmp.name) / "state")
        os.environ["XDG_CACHE_HOME"] = str(Path(self._tmp.name) / "cache")
        self._procs: list[subprocess.Popen] = []
        self.addCleanup(self._cleanup)
        # Placeholder Popen handles are intentionally dropped by reserve mode
        # (they outlive the launching code), which trips ResourceWarning in
        # tests; suppress it only for the duration of the test.
        warnings.filterwarnings("ignore", category=ResourceWarning)

    def _cleanup(self):
        try:
            state = load_state()
        except Exception:
            state = None
        if state is not None:
            for rec in state.gpus:
                try:
                    if rec.alive:
                        os.killpg(rec.pid, signal.SIGKILL)
                except OSError:
                    pass
        for proc in self._procs:
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except OSError:
                    pass
            try:
                proc.wait(timeout=5)
            except Exception:
                pass
        for key, value in self._saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def spawn_sleep(self) -> subprocess.Popen:
        proc = subprocess.Popen(["sleep", "60"], start_new_session=True)
        self._procs.append(proc)
        return proc

    def make_state(self, records, *, mem_mb=1024, expires_in=3600.0, hostname=None):
        now = time.time()
        return Reservation(
            mem_mb=mem_mb,
            gpus=list(records),
            created_at=now,
            expires_at=None if expires_in is None else now + expires_in,
            hostname=hostname or socket.gethostname(),
            uid=os.getuid(),
        )

    def save(self, state: Reservation) -> None:
        reserve_mod._save_state_locked(state)


# ---------------------------------------------------------------------------
# state file
# ---------------------------------------------------------------------------

class StateFileTest(ReservationTestBase):
    def test_roundtrip_and_permissions(self):
        proc = self.spawn_sleep()
        state = self.make_state([record_for(proc, index=2)])
        self.save(state)
        loaded = load_state()
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.gpu_ids, [2])
        self.assertEqual(loaded.gpus[0].pid, proc.pid)
        self.assertEqual(loaded.mem_mb, 1024)
        self.assertEqual(loaded.uid, os.getuid())
        mode = stat.S_IMODE(reserve_mod.state_path().stat().st_mode)
        self.assertEqual(mode, 0o600)

    def test_corrupt_state_raises(self):
        reserve_mod.state_dir().mkdir(parents=True, exist_ok=True)
        reserve_mod.state_path().write_text("{not json", encoding="utf-8")
        with self.assertRaises(ReserveError):
            load_state()

    def test_malformed_state_raises(self):
        reserve_mod.state_dir().mkdir(parents=True, exist_ok=True)
        payload = {"version": 1, "uid": os.getuid(), "created_at": 1.0, "gpus": []}
        reserve_mod.state_path().write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaises(ReserveError):
            load_state()

    def test_foreign_uid_rejected(self):
        proc = self.spawn_sleep()
        state = self.make_state([record_for(proc)])
        payload = state.to_dict()
        payload["uid"] = os.getuid() + 1
        reserve_mod.state_dir().mkdir(parents=True, exist_ok=True)
        reserve_mod.state_path().write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaises(ReserveError):
            load_state()

    def test_acquire_refuses_when_a_reservation_is_active(self):
        proc = self.spawn_sleep()
        self.save(self.make_state([record_for(proc)]))
        probe = FakeProbe([free_frame((0,))])
        opts = AcquireOptions(need=1, mem_mb=1024, interval_s=0.005, stable_rounds=1,
                              hold_cmd=sleep_cmd())
        with self.assertRaises(ReserveError):
            acquire(probe, opts, emit=lambda m: None)


# ---------------------------------------------------------------------------
# placeholder command
# ---------------------------------------------------------------------------

class HogCommandTest(unittest.TestCase):
    def test_builtin_command_holds_through_the_hog_module(self):
        cmd = hog_command(None, 12288, 2, 3600)
        self.assertIn("gpu_hunter.hog", cmd)
        self.assertIn("12288", cmd)
        self.assertIn("3600", cmd)

    def test_template_substitution(self):
        cmd = hog_command("fake {gpu} {mem_mb} {timeout}", 1024, 3, 60)
        self.assertEqual(cmd, ["fake", "3", "1024", "60"])

    def test_environment_override(self):
        with patch.dict(os.environ, {"GPU_HUNTER_HOG_CMD": "hold {mem_mb}"}):
            self.assertEqual(hog_command(None, 512, 0, 0), ["hold", "512"])


# ---------------------------------------------------------------------------
# acquire
# ---------------------------------------------------------------------------

class AcquireTest(ReservationTestBase):
    def test_acquire_and_release_cycle(self):
        # hunt x2 -> baseline/verify for GPU0 -> baseline/verify for GPU1
        frames = [
            free_frame((0, 1)),
            free_frame((0, 1)),
            free_frame((0, 1)),
            held_frame((0, 1), 1024),
            free_frame((0, 1)),
            held_frame((0, 1), 1024),
        ]
        probe = FakeProbe(frames)
        opts = AcquireOptions(
            need=2, mem_mb=1024, interval_s=0.005, stable_rounds=2,
            confirm_s=2.0, hold_cmd=sleep_cmd(),
        )
        state = acquire(probe, opts, emit=lambda m: None)
        self.assertEqual(sorted(state.gpu_ids), [0, 1])
        self.assertTrue(all(rec.alive for rec in state.gpus))
        self.assertIsNotNone(load_state())

        released = release(
            FakeProbe([free_frame((0, 1))]), emit=lambda m: None,
            term_wait=5.0, free_wait=5.0,
        )
        self.assertEqual(released, [0, 1])
        self.assertIsNone(load_state())
        for rec in state.gpus:
            self.assertFalse(rec.alive)

    def test_hold_fails_when_the_placeholder_exits(self):
        probe = FakeProbe([free_frame((0,))])
        opts = AcquireOptions(
            need=1, mem_mb=1024, interval_s=0.005, stable_rounds=1,
            confirm_s=1.0,
            hold_cmd=shlex.join([sys.executable, "-c", "raise SystemExit(3)"]),
        )
        terms = []
        with patch.object(reserve_mod, "terminate_tree", side_effect=lambda p, term_wait=10.0: terms.append(p)):
            with self.assertRaises(reserve_mod._HoldFailed):
                reserve_mod._hold_gpus(
                    probe, [0], opts, emit=lambda m: None, stop=threading.Event()
                )
        self.assertEqual(len(terms), 1)


# ---------------------------------------------------------------------------
# release
# ---------------------------------------------------------------------------

class ReleaseTest(ReservationTestBase):
    def test_release_terminates_placeholders_and_clears_state(self):
        proc = self.spawn_sleep()
        state = self.make_state([record_for(proc)])
        self.save(state)
        messages = []
        released = release(
            FakeProbe([free_frame((0,))]), emit=messages.append,
            term_wait=5.0, free_wait=5.0,
        )
        self.assertEqual(released, [0])
        self.assertIsNotNone(proc.wait(timeout=5))
        self.assertIsNone(load_state())
        self.assertTrue(any("releasing GPU 0" in m for m in messages))

    def test_release_never_signals_an_unverifiable_process(self):
        proc = self.spawn_sleep()
        state = self.make_state([record_for(proc)])
        self.save(state)
        messages = []
        with patch.object(reserve_mod, "signal_group_checked", return_value=False):
            release(FakeProbe([free_frame((0,))]), emit=messages.append,
                    term_wait=0.2, free_wait=0.2)
        self.assertIsNone(proc.poll(), "must not kill a process the safety gate refused")
        self.assertTrue(any("could not verify" in m for m in messages))
        self.assertIsNone(load_state())

    def test_release_refuses_a_reservation_from_another_host(self):
        proc = self.spawn_sleep()
        state = self.make_state([record_for(proc)], hostname="some-other-host")
        self.save(state)
        with self.assertRaises(ReserveError):
            release(FakeProbe([free_frame((0,))]), term_wait=0.1, free_wait=0.1,
                    emit=lambda m: None)
        self.assertIsNone(proc.poll(), "a foreign-host reservation must not be touched")
        self.assertIsNotNone(load_state(), "state must survive a refused release")

    def test_release_force_clears_a_foreign_host_reservation(self):
        proc = self.spawn_sleep()
        state = self.make_state([record_for(proc)], hostname="some-other-host")
        self.save(state)
        messages = []
        released = release(
            FakeProbe([free_frame((0,))]), force=True, emit=messages.append,
            term_wait=0.1, free_wait=0.1,
        )
        self.assertEqual(released, [0])
        self.assertIsNone(load_state())
        self.assertIsNone(proc.poll(), "force on a foreign host must not signal pids")

    def test_release_force_clears_corrupt_state(self):
        reserve_mod.state_dir().mkdir(parents=True, exist_ok=True)
        reserve_mod.state_path().write_text("garbage", encoding="utf-8")
        released = release(FakeProbe([free_frame((0,))]), force=True,
                           emit=lambda m: None, term_wait=0.1, free_wait=0.1)
        self.assertEqual(released, [])
        self.assertIsNone(load_state())

    def test_release_without_state_raises(self):
        with self.assertRaises(ReserveError):
            release(FakeProbe([free_frame((0,))]), emit=lambda m: None,
                    term_wait=0.1, free_wait=0.1)


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

class StatusTest(ReservationTestBase):
    def test_no_reservation(self):
        report = status_report(FakeProbe([free_frame((0,))]))
        self.assertEqual(report.exit_code, 2)
        self.assertIn("no active reservation", report.text)

    def test_healthy_reservation(self):
        proc = self.spawn_sleep()
        self.save(self.make_state([record_for(proc, index=0, mem_mb=1024)]))
        report = status_report(FakeProbe([{0: gpu(0, used=2048)}]))
        self.assertEqual(report.exit_code, 0, report.text)
        self.assertIn("memory held", report.text)
        self.assertIn("[ok]", report.text)

    def test_memory_loss_marks_reservation_degraded(self):
        proc = self.spawn_sleep()
        self.save(self.make_state([record_for(proc, index=0, mem_mb=1024)]))
        report = status_report(FakeProbe([{0: gpu(0, used=100)}]))
        self.assertEqual(report.exit_code, 1)
        self.assertIn("[LOST]", report.text)

    def test_dead_placeholder_marks_reservation_degraded(self):
        state = self.make_state([
            HogRecord(index=0, pid=99_999_999, start_time=123, mem_mb=1024)
        ])
        self.save(state)
        report = status_report(FakeProbe([free_frame((0,))]))
        self.assertEqual(report.exit_code, 1)
        self.assertIn("(dead)", report.text)


# ---------------------------------------------------------------------------
# supervisor
# ---------------------------------------------------------------------------

class SupervisorTest(ReservationTestBase):
    def _opts(self, **kw):
        base = dict(need=1, mem_mb=1024, interval_s=0.005, stable_rounds=2,
                    confirm_s=2.0, hold_cmd=sleep_cmd())
        base.update(kw)
        return AcquireOptions(**base)

    def test_missing_state_file(self):
        state = self.make_state([
            HogRecord(index=0, pid=1, start_time=1, mem_mb=1024)
        ])
        supervisor = Supervisor(FakeProbe([free_frame((0,))]), state, self._opts(),
                                emit=lambda m: None, sleep=lambda s: None)
        self.assertEqual(supervisor.tick(), "missing")

    def test_expired_reservation(self):
        state = self.make_state([
            HogRecord(index=0, pid=1, start_time=1, mem_mb=1024)
        ], expires_in=-5.0)
        self.save(state)
        supervisor = Supervisor(FakeProbe([free_frame((0,))]), state, self._opts(),
                                emit=lambda m: None, sleep=lambda s: None)
        self.assertEqual(supervisor.tick(), "expired")

    def test_reacquires_a_lost_placeholder(self):
        dead = HogRecord(index=0, pid=99_999_999, start_time=123, mem_mb=1024,
                         used_before_mb=100)
        state = self.make_state([dead])
        self.save(state)
        frames = [
            free_frame((0,)),
            free_frame((0,)),
            free_frame((0,)),
            free_frame((0,)),
            held_frame((0,), 1024),
        ]
        probe = FakeProbe(frames)
        messages = []
        supervisor = Supervisor(
            probe, state, self._opts(), emit=messages.append,
            sleep=lambda s: None, heartbeat_s=10 ** 9,
        )
        self.assertEqual(supervisor.tick(), "continue")
        loaded = load_state()
        self.assertIsNotNone(loaded)
        self.assertEqual(len(loaded.gpus), 1)
        self.assertNotEqual(loaded.gpus[0].pid, dead.pid)
        self.assertTrue(loaded.gpus[0].alive)
        self.assertTrue(any("restored" in m for m in messages))

    def test_removed_state_file_is_reported_as_missing(self):
        proc = self.spawn_sleep()
        state = self.make_state([record_for(proc)])
        self.save(state)
        probe = FakeProbe([held_frame((0,), 1024, used_before=0)])
        supervisor = Supervisor(probe, state, self._opts(),
                                emit=lambda m: None, sleep=lambda s: None)
        self.assertEqual(supervisor.tick(), "continue")
        reserve_mod.clear_state()
        self.assertEqual(supervisor.tick(), "missing")


# ---------------------------------------------------------------------------
# eager (partial) acquisition
# ---------------------------------------------------------------------------

class EagerTest(ReservationTestBase):
    def _opts(self, **kw):
        base = dict(need=2, mem_mb=1024, interval_s=0.005, stable_rounds=2,
                    confirm_s=2.0, hold_cmd=sleep_cmd(), eager=True)
        base.update(kw)
        return AcquireOptions(**base)

    def test_holds_cards_one_by_one_and_persists_progress(self):
        probe = FakeProbe(eager_frames())
        messages = []
        saved = []
        real_save = reserve_mod._save_state_locked

        def spy(state):
            saved.append((len(state.gpus), state.target_count, state.eager))
            real_save(state)

        with patch.object(reserve_mod, "_save_state_locked", side_effect=spy):
            state = reserve_mod._eager_acquire(
                probe, self._opts(), emit=messages.append, stop=threading.Event()
            )
        self.assertEqual(sorted(state.gpu_ids), [0, 1])
        self.assertEqual(state.target_count, 2)
        self.assertTrue(state.eager)
        self.assertIn((1, 2, True), saved)
        self.assertEqual(saved[-1], (2, 2, True))
        self.assertTrue(any("eager: held 1/2" in m for m in messages))
        loaded = load_state()
        self.assertIsNotNone(loaded)
        self.assertEqual(sorted(loaded.gpu_ids), [0, 1])
        release(FakeProbe([free_frame((0, 1))]), emit=lambda m: None,
                term_wait=5.0, free_wait=5.0)

    def test_failed_hold_keeps_existing_cards_and_retries(self):
        rec_a = HogRecord(index=0, pid=99_999_997, start_time=1, mem_mb=1024,
                          used_before_mb=100)
        rec_b = HogRecord(index=2, pid=99_999_996, start_time=1, mem_mb=1024,
                          used_before_mb=100)
        failed = reserve_mod._HoldFailed("preempted by another process")

        class _FakeHunter:
            def __init__(self, *args, **kwargs):
                self._queue = [[0], [1], [2]]

            def hunt(self):
                return self._queue.pop(0)

        messages = []
        with patch.object(reserve_mod, "Hunter", _FakeHunter), \
                patch.object(reserve_mod, "_hold_one",
                             side_effect=[rec_a, failed, rec_b]):
            state = reserve_mod._eager_acquire(
                FakeProbe([]), self._opts(), emit=messages.append,
                stop=threading.Event(),
            )
        self.assertEqual([rec.index for rec in state.gpus], [0, 2])
        self.assertTrue(any("keeping 1/2" in m for m in messages))
        reserve_mod.clear_state()

    def test_hold_max_expiry_releases_the_partial_reservation(self):
        rec = HogRecord(index=0, pid=99_999_995, start_time=1, mem_mb=1024)

        class _FakeHunter:
            def __init__(self, *args, **kwargs):
                pass

            def hunt(self):
                return [0]

        def slow_hold(*args, **kwargs):
            time.sleep(0.02)
            return rec

        with patch.object(reserve_mod, "Hunter", _FakeHunter), \
                patch.object(reserve_mod, "_hold_one", side_effect=slow_hold):
            with self.assertRaises(ReserveError) as ctx:
                reserve_mod._eager_acquire(
                    FakeProbe([]), self._opts(hold_max_s=0.01),
                    emit=lambda m: None, stop=threading.Event(),
                )
        self.assertIn("hold-max expired", str(ctx.exception))
        self.assertIsNone(load_state())

    def test_external_release_aborts_a_partial_acquisition(self):
        calls = {"n": 0}

        class _FakeHunter:
            def __init__(self, *args, **kwargs):
                pass

            def hunt(self):
                calls["n"] += 1
                if calls["n"] == 1:
                    return [0]
                reserve_mod.clear_state()
                return [1]

        records = [
            HogRecord(index=0, pid=99_999_994, start_time=1, mem_mb=1024),
            HogRecord(index=1, pid=99_999_993, start_time=1, mem_mb=1024),
        ]
        with patch.object(reserve_mod, "Hunter", _FakeHunter), \
                patch.object(reserve_mod, "_hold_one", side_effect=records):
            with self.assertRaises(ReserveError) as ctx:
                reserve_mod._eager_acquire(
                    FakeProbe([]), self._opts(), emit=lambda m: None,
                    stop=threading.Event(),
                )
        self.assertIn("released externally", str(ctx.exception))
        self.assertIsNone(load_state())

    def test_status_reports_eager_progress(self):
        proc = self.spawn_sleep()
        state = self.make_state([record_for(proc, index=0, mem_mb=1024)])
        state.target_count = 2
        state.eager = True
        self.save(state)
        report = status_report(FakeProbe([{0: gpu(0, used=2048)}]))
        self.assertEqual(report.exit_code, 0, report.text)
        self.assertIn("eager reservation in progress: 1/2", report.text)
        self.assertIn("waiting for 1 more GPU(s)", report.text)


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------

def fake_nvml_probe(frames):
    """A class standing in for gpu_hunter.probe.NVMLProbe in cli.main()."""

    class _FakeProbe:
        def __init__(self):
            self._probe = FakeProbe(frames)

        def snapshot(self):
            return self._probe.snapshot()

        def close(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            self.close()

    return _FakeProbe


class CliReserveTest(ReservationTestBase):
    def test_detach_status_release_and_hand_over(self):
        from gpu_hunter import cli

        frames = [
            free_frame((0, 1)),
            free_frame((0, 1)),
            free_frame((0, 1)),
            held_frame((0, 1), 1024),
            free_frame((0, 1)),
            held_frame((0, 1), 1024),
        ]
        with patch("gpu_hunter.probe.NVMLProbe", fake_nvml_probe(frames)):
            rc = cli.main([
                "--reserve", "--gpu-num", "2", "--mem-per-gpu", "1G",
                "--detach", "--interval", "0.01", "--stable-rounds", "2",
                "--hold-cmd", sleep_cmd(),
            ])
        self.assertEqual(rc, 0)
        state = load_state()
        self.assertIsNotNone(state)
        self.assertEqual(sorted(state.gpu_ids), [0, 1])

        with patch("gpu_hunter.probe.NVMLProbe",
                   fake_nvml_probe([held_frame((0, 1), 1024)])):
            rc = cli.main(["--reserve-status"])
        self.assertEqual(rc, 0)

        with patch("gpu_hunter.probe.NVMLProbe",
                   fake_nvml_probe([free_frame((0, 1))])):
            rc = cli.main([
                "--release", "--run", "--grace", "0.05",
                "--", sys.executable, "-c", "raise SystemExit(0)",
            ])
        self.assertEqual(rc, 0)
        self.assertIsNone(load_state())

    def test_eager_acquire_and_release(self):
        from gpu_hunter import cli

        with patch("gpu_hunter.probe.NVMLProbe", fake_nvml_probe(eager_frames())), \
                patch.object(reserve_mod.Supervisor, "run", return_value="missing"), \
                patch.object(reserve_mod, "_wait_gpus_free"):
            rc = cli.main([
                "--reserve", "--eager", "--gpu-num", "2", "--mem-per-gpu", "1G",
                "--interval", "0.01", "--stable-rounds", "2",
                "--hold-cmd", sleep_cmd(),
            ])
        self.assertEqual(rc, 0)
        self.assertIsNone(load_state())

    def test_status_without_reservation(self):
        from gpu_hunter import cli

        with patch("gpu_hunter.probe.NVMLProbe",
                   fake_nvml_probe([free_frame((0,))])):
            self.assertEqual(cli.main(["--reserve-status"]), 2)

    def test_release_without_reservation_and_force(self):
        from gpu_hunter import cli

        with patch("gpu_hunter.probe.NVMLProbe",
                   fake_nvml_probe([free_frame((0,))])):
            self.assertEqual(cli.main(["--release"]), 2)
            self.assertEqual(cli.main(["--release", "--force"]), 0)


class ModeValidationTest(ReservationTestBase):
    def test_mutually_exclusive_hold_modes(self):
        from gpu_hunter import cli

        with self.assertRaises(SystemExit) as ctx:
            cli.main(["--reserve", "--release", "--mem-per-gpu", "1G"])
        self.assertEqual(ctx.exception.code, 2)

    def test_reserve_requires_memory_amount(self):
        from gpu_hunter import cli

        with self.assertRaises(SystemExit) as ctx:
            cli.main(["--reserve", "--detach"])
        self.assertEqual(ctx.exception.code, 2)

    def test_reserve_without_run_rejects_a_command(self):
        from gpu_hunter import cli

        with self.assertRaises(SystemExit) as ctx:
            cli.main(["--reserve", "--mem-per-gpu", "1G", "--", "true"])
        self.assertEqual(ctx.exception.code, 2)

    def test_run_requires_a_command(self):
        from gpu_hunter import cli

        with self.assertRaises(SystemExit) as ctx:
            cli.main(["--release", "--run"])
        self.assertEqual(ctx.exception.code, 2)

    def test_detach_requires_reserve(self):
        from gpu_hunter import cli

        with self.assertRaises(SystemExit) as ctx:
            cli.main(["--release", "--detach"])
        self.assertEqual(ctx.exception.code, 2)

    def test_detach_rejects_run(self):
        from gpu_hunter import cli

        with self.assertRaises(SystemExit) as ctx:
            cli.main([
                "--reserve", "--mem-per-gpu", "1G", "--detach",
                "--run", "--", "true",
            ])
        self.assertEqual(ctx.exception.code, 2)

    def test_eager_requires_reserve(self):
        from gpu_hunter import cli

        with self.assertRaises(SystemExit) as ctx:
            cli.main(["--eager", "--", "true"])
        self.assertEqual(ctx.exception.code, 2)

    def test_eager_rejects_detach(self):
        from gpu_hunter import cli

        with self.assertRaises(SystemExit) as ctx:
            cli.main(["--reserve", "--eager", "--detach", "--mem-per-gpu", "1G"])
        self.assertEqual(ctx.exception.code, 2)

    def test_reserve_status_rejects_a_command(self):
        from gpu_hunter import cli

        with self.assertRaises(SystemExit) as ctx:
            cli.main(["--reserve-status", "--", "true"])
        self.assertEqual(ctx.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
