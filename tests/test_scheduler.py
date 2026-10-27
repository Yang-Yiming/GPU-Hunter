import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from helpers import FakeProbe, gpu  # noqa: E402

import gpu_hunter.scheduler as sched_mod  # noqa: E402
from gpu_hunter.runner import CLAIM_CONFIRMED, CLAIM_LOST  # noqa: E402
from gpu_hunter.scheduler import JobResult, Scheduler, format_summary  # noqa: E402
from gpu_hunter.spec import JobSpec, WorkflowSpec  # noqa: E402


class FakeProc:
    """Programmable lifecycle: rc_immediate exits at once; done_after_polls exits at poll N."""

    _pid_seq = 1000

    def __init__(self, rc_immediate=None, done_after_polls=None):
        FakeProc._pid_seq += 1
        self.pid = FakeProc._pid_seq
        self.returncode = rc_immediate
        self._polls = 0
        self._done_after = done_after_polls

    def poll(self):
        if self.returncode is not None:
            return self.returncode
        self._polls += 1
        if self._done_after is not None and self._polls >= self._done_after:
            self.returncode = 0
        return self.returncode


def job(name="j", cmd=("/bin/true",), gpu_num=1, **kw):
    return JobSpec(name=name, cmd=list(cmd), gpu_num=gpu_num, **kw)


def wf(jobs, schedule="greedy", **kw):
    base = dict(
        path=Path("t.yaml"),
        schedule=schedule,
        interval_s=0.005,
        stable_rounds=1,
        grace_s=0.01,
        claim_stable_s=0.01,
        heartbeat_s=3600.0,
        retry_delay_s=0.0,
    )
    base.update(kw)
    return WorkflowSpec(jobs=list(jobs), **base)


FREE2 = [{0: gpu(0), 1: gpu(1)}]
FREE1 = [{0: gpu(0)}]


class LaunchRecorder:
    """Replacement for scheduler.launch: records calls and returns scripted fake processes."""

    def __init__(self, procs=None):
        self.calls = []          # (name, gpus, log_path, extra_env)
        self._procs = list(procs or [])
        self.on_launch = None    # hook to mutate external state on launch (e.g. set stop)

    def __call__(self, cmd, gpus, log_path=None, cwd=None, extra_env=None):
        self.calls.append((list(cmd), tuple(gpus), log_path, extra_env))
        if self.on_launch:
            self.on_launch()
        proc = self._procs.pop(0) if self._procs else FakeProc(rc_immediate=0)
        return proc, None


def run_scheduler(probe, workflow, recorder, verify=lambda proc, gpus: CLAIM_CONFIRMED,
                  stop_event=None, max_ticks=50, notify=None):
    with patch.object(sched_mod, "launch", recorder), \
         patch.object(sched_mod, "terminate_tree") as term:
        s = Scheduler(
            probe, workflow,
            emit=lambda m: None,
            verify_fn=verify,
            max_ticks=max_ticks,
            stop_event=stop_event,
            notify=notify,
        )
        results = s.run()
        return s, results, term


class GreedyTest(unittest.TestCase):
    def test_all_launched_and_disjoint_when_free(self):
        rec = LaunchRecorder([FakeProc(done_after_polls=4), FakeProc(done_after_polls=4)])
        s, results, _ = run_scheduler(FakeProbe(FREE2 * 5), wf([job("a"), job("b")]), rec)
        self.assertEqual(len(rec.calls), 2)
        gpus_used = [c[1] for c in rec.calls]
        self.assertEqual(len(set(gpus_used)), 2)          # no overlap
        self.assertEqual([r.name for r in results], ["a", "b"])
        self.assertTrue(all(r.ok for r in results))

    def test_skips_unfittable_head_keeps_scanning(self):
        # A needs 2 GPUs but only 1 is free; B needs 1 -> B should fly first while A waits
        frames = FREE1 * 3 + FREE2 * 6
        rec = LaunchRecorder(
            [FakeProc(done_after_polls=8), FakeProc(done_after_polls=2)]
        )
        s, results, _ = run_scheduler(
            FakeProbe(frames), wf([job("a", gpu_num=2), job("b")]), rec
        )
        by_name = {r.name: r for r in results}
        self.assertTrue(by_name["b"].ok)                  # launched first
        self.assertTrue(by_name["a"].ok)                  # also completes once more GPUs free up
        b_idx = rec.calls.index(next(c for c in rec.calls if len(c[1]) == 1))
        a_idx = next(i for i, c in enumerate(rec.calls) if len(c[1]) == 2)
        self.assertLess(b_idx, a_idx)

    def test_prefer_conflict_resolved_disjoint(self):
        rec = LaunchRecorder(
            [FakeProc(done_after_polls=4), FakeProc(done_after_polls=4)]
        )
        jobs = [job("a", prefer=(0,)), job("b", prefer=(0,))]
        run_scheduler(FakeProbe(FREE2 * 5), wf(jobs), rec)
        self.assertEqual(rec.calls[0][1], (0,))
        self.assertEqual(rec.calls[1][1], (1,))           # a takes 0, b falls back to 1

    def test_env_forwarded(self):
        rec = LaunchRecorder([FakeProc(rc_immediate=0)])
        j = job("a", env={"FOO": "1"})
        run_scheduler(FakeProbe(FREE2 * 3), wf([j]), rec)
        self.assertEqual(rec.calls[0][3], {"FOO": "1"})

    def test_auto_log_generated_only_when_not_passthrough(self):
        rec = LaunchRecorder([FakeProc(rc_immediate=0), FakeProc(rc_immediate=0)])
        jobs = [job("a"), job("b", passthrough_log=True)]
        run_scheduler(FakeProbe(FREE2 * 3), wf([*jobs]), rec)
        lp_a, lp_b = rec.calls[0][2], rec.calls[1][2]
        self.assertIsNotNone(lp_a)
        self.assertIn("logs", str(lp_a))
        self.assertIsNone(lp_b)


class SequentialTest(unittest.TestCase):
    def test_strict_order_blocks_smaller_behind(self):
        # Head job A needs 2 GPUs but only 1 is ever free -> B must not run even though it could
        rec = LaunchRecorder([FakeProc()])
        s, results, _ = run_scheduler(
            FakeProbe(FREE1 * 10),
            wf([job("a", gpu_num=2), job("b")], schedule="sequential"),
            rec,
            max_ticks=6,
        )
        self.assertEqual(rec.calls, [])
        self.assertEqual(results, [])
        self.assertEqual(len(s.pending), 2)

    def test_sequential_runs_in_order_one_at_a_time(self):
        rec = LaunchRecorder(
            [FakeProc(done_after_polls=2) for _ in range(3)]
        )
        jobs = [job("a"), job("b"), job("c")]
        _, results, _ = run_scheduler(
            FakeProbe(FREE2 * 12), wf(jobs, schedule="sequential"), rec
        )
        self.assertEqual([r.name for r in results], ["a", "b", "c"])
        # Strict serial execution: b launches only after a finishes
        starts = [i for i, c in enumerate(rec.calls)]
        self.assertEqual(starts, [0, 1, 2])


class FailureTest(unittest.TestCase):
    def test_failure_recorded_and_queue_continues(self):
        rec = LaunchRecorder([FakeProc(rc_immediate=3), FakeProc(rc_immediate=0)])
        _, results, _ = run_scheduler(FakeProbe(FREE2 * 5), wf([job("bad"), job("good")]), rec)
        by_name = {r.name: r for r in results}
        self.assertFalse(by_name["bad"].ok)
        self.assertEqual(by_name["bad"].rc, 3)
        self.assertTrue(by_name["good"].ok)

    def test_launch_oserror_recorded_and_queue_continues(self):
        with patch.object(
            sched_mod,
            "launch",
            side_effect=[OSError("no such executable"), (FakeProc(rc_immediate=0), None)],
        ), patch.object(sched_mod, "terminate_tree"):
            s = Scheduler(
                FakeProbe(FREE2 * 5),
                wf([job("bad"), job("good")]),
                emit=lambda m: None,
                verify_fn=lambda proc, gpus: CLAIM_CONFIRMED,
                max_ticks=10,
            )
            results = s.run()
        by_name = {r.name: r for r in results}
        self.assertEqual(by_name["bad"].rc, 127)
        self.assertTrue(by_name["good"].ok)


class SnapshotErrorTest(unittest.TestCase):
    def test_failed_snapshot_does_not_reset_stability(self):
        class FlakyProbe:
            def __init__(self):
                self.calls = 0

            def snapshot(self):
                self.calls += 1
                if self.calls == 2:
                    raise RuntimeError("nvml hiccup")
                return {0: gpu(0)}

        rec = LaunchRecorder([FakeProc(rc_immediate=0)])
        _, results, _ = run_scheduler(
            FlakyProbe(),
            wf([job("a")], stable_rounds=2),
            rec,
            max_ticks=3,
        )
        self.assertEqual(len(rec.calls), 1)
        self.assertEqual(results[0].rc, 0)


class PreemptTest(unittest.TestCase):
    def test_lost_rolls_back_and_retries(self):
        seq = iter([CLAIM_LOST, CLAIM_CONFIRMED])
        rec = LaunchRecorder([FakeProc(), FakeProc(done_after_polls=2)])
        verify = lambda proc, gpus: next(seq)  # noqa: E731
        s, results, term = run_scheduler(
            FakeProbe(FREE1 * 8), wf([job("a")]), rec, verify=verify
        )
        self.assertEqual(len(rec.calls), 2)               # tried twice
        term.assert_called_once()                          # first attempt rolled back and killed
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].attempts, 2)
        self.assertEqual(s.pending, [])                    # second attempt left the queue on success

    def test_rollback_releases_claim_for_next_job(self):
        # big is preempted and its GPU history is invalidated; the same GPU must
        # restabilize before another job may use it
        seq = iter([CLAIM_LOST, CLAIM_CONFIRMED, CLAIM_CONFIRMED])
        rec = LaunchRecorder(
            [FakeProc(), FakeProc(rc_immediate=0), FakeProc(done_after_polls=2)]
        )

        def verify(proc, gpus):
            return next(seq)

        _, results, term = run_scheduler(
            FakeProbe(FREE1 * 10),
            wf([job("big", prefer=(0,), gpu_num=1), job("small", prefer=(0,))]),
            rec,
            verify=verify,
        )
        self.assertEqual(len(rec.calls), 3)               # big preempted + small + big retry
        term.assert_called_once()
        by_name = {r.name: r for r in results}
        self.assertTrue(by_name["small"].ok)
        self.assertEqual(by_name["big"].attempts, 2)
        self.assertTrue(by_name["big"].ok)

    def test_retry_gives_up_after_max_attempts(self):
        rec = LaunchRecorder([FakeProc() for _ in range(5)])
        _, results, term = run_scheduler(
            FakeProbe(FREE1 * 20),
            wf([job("a")], max_attempts=2),
            rec,
            verify=lambda proc, gpus: CLAIM_LOST,
        )
        self.assertEqual(len(rec.calls), 2)
        self.assertEqual(len(results), 1)
        self.assertIsNone(results[0].rc)
        self.assertEqual(results[0].attempts, 2)
        self.assertEqual(term.call_count, 2)


class StopTest(unittest.TestCase):
    def test_stop_aborts_and_kills_children(self):
        rec = LaunchRecorder([FakeProc()])                # long job that never exits
        import threading

        stop = threading.Event()

        rec.on_launch = stop.set
        _, results, term = run_scheduler(
            FakeProbe(FREE1 * 5), wf([job("long")]), rec, stop_event=stop
        )
        term.assert_called_once()                         # child group terminated during rollback
        self.assertEqual(len(results), 1)                 # one aborted result recorded
        self.assertIsNone(results[0].rc)

    def test_stop_prevents_next_job_from_starting(self):
        import threading

        stop = threading.Event()
        rec = LaunchRecorder([FakeProc()])
        rec.on_launch = stop.set
        _, results, _ = run_scheduler(
            FakeProbe(FREE1 * 10),
            wf([job("a"), job("b")]),
            rec,
            stop_event=stop,
            max_ticks=10,
        )
        self.assertEqual(len(rec.calls), 1)
        self.assertEqual([r.name for r in results], ["a"])

    def test_stop_while_running_records_aborted_job(self):
        import threading

        stop = threading.Event()
        rec = LaunchRecorder([FakeProc()])                # running job that never exits

        def stop_later():
            threading.Timer(0.05, stop.set).start()

        rec.on_launch = stop_later
        _, results, term = run_scheduler(
            FakeProbe(FREE1 * 20), wf([job("long")]), rec, stop_event=stop, max_ticks=200
        )
        self.assertEqual(len(results), 1)
        self.assertIsNone(results[0].rc)
        term.assert_called_once()


class SummaryTest(unittest.TestCase):
    def test_format_summary_layout(self):
        rows = [
            JobResult("pretrain", 0, (2, 3), 0.0, 65.0, attempts=1),
            JobResult("eval", 3, (4,), 0.0, 7.0, attempts=2),
        ]
        text = format_summary(rows)
        lines = text.splitlines()
        self.assertEqual(len(lines), 4)
        self.assertIn("pretrain", lines[2])
        self.assertIn("OK", lines[2])
        self.assertIn("FAIL", lines[3])
        self.assertIn("2,3", lines[2])
        self.assertEqual(format_summary([]), "(no jobs recorded)")


class NotifyTest(unittest.TestCase):
    def events(self):
        return []

    def test_launch_and_finish_are_reported(self):
        events = []
        rec = LaunchRecorder([FakeProc(done_after_polls=2)])
        run_scheduler(
            FakeProbe(FREE1 * 5), wf([job("a")]), rec,
            notify=lambda kind, msg: events.append((kind, msg)),
        )
        kinds = [kind for kind, _ in events]
        self.assertIn("launch", kinds)
        self.assertIn("finish", kinds)
        self.assertNotIn("attempt", kinds)

    def test_waiting_tick_emits_attempt_x(self):
        events = []
        rec = LaunchRecorder([])
        run_scheduler(
            FakeProbe([{0: gpu(0, used=90000)}] * 5), wf([job("a")]), rec,
            notify=lambda kind, msg: events.append((kind, msg)),
            max_ticks=3,
        )
        self.assertEqual(events, [("attempt", "")] * 3)

    def test_preemption_emits_retry(self):
        events = []
        seq = iter([CLAIM_LOST, CLAIM_CONFIRMED])
        rec = LaunchRecorder([FakeProc(), FakeProc(done_after_polls=2)])
        run_scheduler(
            FakeProbe(FREE1 * 8), wf([job("a")]), rec,
            verify=lambda proc, gpus: next(seq),
            notify=lambda kind, msg: events.append((kind, msg)),
        )
        kinds = [kind for kind, _ in events]
        self.assertEqual(kinds[0], "retry")
        self.assertIn("launch", kinds)


if __name__ == "__main__":
    unittest.main()
