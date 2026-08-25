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
    """可编程生命周期:rc_immediate 立即结束;done_after_polls 在第 N 次 poll 后结束。"""

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
        heartbeat_s=3600.0,
    )
    base.update(kw)
    return WorkflowSpec(jobs=list(jobs), **base)


FREE2 = [{0: gpu(0), 1: gpu(1)}]
FREE1 = [{0: gpu(0)}]


class LaunchRecorder:
    """替换 scheduler.launch:记录调用并按脚本返回假进程。"""

    def __init__(self, procs=None):
        self.calls = []          # (name, gpus, log_path, extra_env)
        self._procs = list(procs or [])
        self.on_launch = None    # 钩子,可在发射时改变外部状态(如 set stop)

    def __call__(self, cmd, gpus, log_path=None, cwd=None, extra_env=None):
        self.calls.append((list(cmd), tuple(gpus), log_path, extra_env))
        if self.on_launch:
            self.on_launch()
        proc = self._procs.pop(0) if self._procs else FakeProc(rc_immediate=0)
        return proc, None


def run_scheduler(probe, workflow, recorder, verify=lambda pid, gpus: CLAIM_CONFIRMED,
                  stop_event=None, max_ticks=50):
    with patch.object(sched_mod, "launch", recorder), \
         patch.object(sched_mod, "terminate_tree") as term:
        s = Scheduler(
            probe, workflow,
            emit=lambda m: None,
            verify_fn=verify,
            max_ticks=max_ticks,
            stop_event=stop_event,
        )
        results = s.run()
        return s, results, term


class GreedyTest(unittest.TestCase):
    def test_all_launched_and_disjoint_when_free(self):
        rec = LaunchRecorder([FakeProc(done_after_polls=4), FakeProc(done_after_polls=4)])
        s, results, _ = run_scheduler(FakeProbe(FREE2 * 5), wf([job("a"), job("b")]), rec)
        self.assertEqual(len(rec.calls), 2)
        gpus_used = [c[1] for c in rec.calls]
        self.assertEqual(len(set(gpus_used)), 2)          # 不重叠
        self.assertEqual([r.name for r in results], ["a", "b"])
        self.assertTrue(all(r.ok for r in results))

    def test_skips_unfittable_head_keeps_scanning(self):
        # A 要 2 张但只有 1 张空闲;B 只要 1 张 → B 应当先飞,A 继续等
        frames = FREE1 * 3 + FREE2 * 6
        rec = LaunchRecorder(
            [FakeProc(done_after_polls=8), FakeProc(done_after_polls=2)]
        )
        s, results, _ = run_scheduler(
            FakeProbe(frames), wf([job("a", gpu_num=2), job("b")]), rec
        )
        by_name = {r.name: r for r in results}
        self.assertTrue(by_name["b"].ok)                  # 先飞
        self.assertTrue(by_name["a"].ok)                  # 后面卡够了也完成
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
        self.assertEqual(rec.calls[1][1], (1,))           # a 拿走 0,b 兜底拿 1

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
        # 队首 A 要 2 张,永远只有 1 张 → B 即使能跑也不许跑
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
        # 严格串行:a 结束后 b 才发射
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


class PreemptTest(unittest.TestCase):
    def test_lost_rolls_back_and_retries(self):
        seq = iter([CLAIM_LOST, CLAIM_CONFIRMED])
        rec = LaunchRecorder([FakeProc(), FakeProc(done_after_polls=2)])
        verify = lambda pid, gpus: next(seq)  # noqa: E731
        s, results, term = run_scheduler(
            FakeProbe(FREE1 * 8), wf([job("a")]), rec, verify=verify
        )
        self.assertEqual(len(rec.calls), 2)               # 尝试了两次
        term.assert_called_once()                          # 第一次被回滚杀掉
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].attempts, 2)
        self.assertEqual(s.pending, [])                    # 第二次成功离队

    def test_rollback_releases_claim_for_next_job(self):
        # big 被抢后释放卡;同节拍继续扫描 → small 立刻飞;small 完成后 big 重试成功
        seq = iter([CLAIM_LOST, CLAIM_CONFIRMED, CLAIM_CONFIRMED])
        rec = LaunchRecorder(
            [FakeProc(), FakeProc(rc_immediate=0), FakeProc(done_after_polls=2)]
        )

        def verify(pid, gpus):
            return next(seq)

        _, results, term = run_scheduler(
            FakeProbe(FREE1 * 10),
            wf([job("big", prefer=(0,), gpu_num=1), job("small", prefer=(0,))]),
            rec,
            verify=verify,
        )
        self.assertEqual(len(rec.calls), 3)               # big 抢占 + small + big 重试
        term.assert_called_once()
        by_name = {r.name: r for r in results}
        self.assertTrue(by_name["small"].ok)
        self.assertEqual(by_name["big"].attempts, 2)
        self.assertTrue(by_name["big"].ok)


class StopTest(unittest.TestCase):
    def test_stop_aborts_and_kills_children(self):
        rec = LaunchRecorder([FakeProc()])                # 永不退出的长任务
        import threading

        stop = threading.Event()

        rec.on_launch = stop.set
        _, results, term = run_scheduler(
            FakeProbe(FREE1 * 5), wf([job("long")]), rec, stop_event=stop
        )
        term.assert_called_once()                         # 回滚时终止子进程组
        self.assertEqual(len(results), 1)                 # 记一条中止结果
        self.assertIsNone(results[0].rc)


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
        self.assertEqual(format_summary([]), "(无任务记录)")


if __name__ == "__main__":
    unittest.main()
