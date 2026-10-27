import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gpu_hunter.parsing import coerce_cmd, ensure_interpreter, parse_int_list, parse_size_mb
from gpu_hunter.spec import SpecError, load_workflow

try:
    import yaml  # noqa: F401
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


def write_wf(tmp: Path, text: str) -> Path:
    p = tmp / "wf.yaml"
    p.write_text(text, encoding="utf-8")
    return p


@unittest.skipUnless(HAS_YAML, "pyyaml not installed")
class WorkflowSpecTest(unittest.TestCase):
    def setUp(self):
        import tempfile

        self._td = tempfile.TemporaryDirectory()
        self.tmp = Path(self._td.name)
        self.addCleanup(self._td.cleanup)

    def load(self, text: str):
        return load_workflow(write_wf(self.tmp, text))

    def test_minimal(self):
        wf = self.load(
            """
            jobs:
              - cmd: python train.py --epochs 100
            """
        )
        self.assertEqual(wf.schedule, "greedy")
        self.assertEqual(len(wf.jobs), 1)
        job = wf.jobs[0]
        self.assertEqual(job.name, "job-1")
        self.assertEqual(job.cmd, ["python", "train.py", "--epochs", "100"])
        self.assertEqual(job.gpu_num, 1)
        self.assertIsNone(job.mem_per_gpu_mb)

    def test_defaults_inheritance_and_override(self):
        wf = self.load(
            """
            defaults:
              gpu-num: 2
              mem-per-gpu: 40G
              prefer: [2, 3]
              env: {FOO: a}
            jobs:
              - name: a
                cmd: x.sh
              - name: b
                cmd: y.py
                gpu-num: 4
                env: {BAR: b, FOO: override}
                allow: [7, 8, 9, 10]
            """
        )
        a, b = wf.jobs
        self.assertEqual(a.gpu_num, 2)
        self.assertEqual(a.mem_per_gpu_mb, 40960)
        self.assertEqual(a.prefer, (2, 3))
        self.assertEqual(a.env, {"FOO": "a"})
        self.assertEqual(b.gpu_num, 4)
        self.assertEqual(b.env, {"FOO": "override", "BAR": "b"})
        self.assertEqual(b.allow, frozenset({7, 8, 9, 10}))
        self.assertEqual(b.cmd, [sys.executable, "y.py"])

    def test_cmd_as_list_and_dash_log_passthrough(self):
        wf = self.load(
            """
            jobs:
              - name: a
                cmd: [python, -u, train.py]
                log: "-"
            """
        )
        self.assertEqual(wf.jobs[0].cmd, ["python", "-u", "train.py"])
        self.assertIsNone(wf.jobs[0].log)

    def test_sequential_schedule_and_runtime_fields(self):
        wf = self.load(
            """
            schedule: sequential
            interval: 2.5
            stable-rounds: 5
            idle-mem: 300M
            idle-util: 7
            grace: 10
            claim-stable: 3
            heartbeat: 30
            max-attempts: 4
            retry-delay: 1.5
            jobs:
              - {name: a, cmd: sleep.sh}
            """
        )
        self.assertEqual(wf.schedule, "sequential")
        self.assertAlmostEqual(wf.interval_s, 2.5)
        self.assertEqual(wf.stable_rounds, 5)
        self.assertEqual(wf.idle_mem_mb, 300)
        self.assertEqual(wf.idle_util, 7)
        self.assertAlmostEqual(wf.grace_s, 10.0)
        self.assertAlmostEqual(wf.claim_stable_s, 3.0)
        self.assertEqual(wf.max_attempts, 4)
        self.assertAlmostEqual(wf.retry_delay_s, 1.5)

    def test_allow_smaller_than_gpu_num_rejected(self):
        with self.assertRaisesRegex(SpecError, "only 1 GPU"):
            self.load("jobs: [{cmd: x, gpu-num: 2, allow: [7]}]\n")

    def test_zero_idle_mem_rejected(self):
        with self.assertRaisesRegex(SpecError, "idle-mem"):
            self.load("idle-mem: 0\njobs: [{cmd: x}]\n")

    def test_max_attempts_must_be_positive(self):
        with self.assertRaisesRegex(SpecError, "max-attempts"):
            self.load("max-attempts: 0\njobs: [{cmd: x}]\n")

    def test_duplicate_names_rejected(self):
        with self.assertRaisesRegex(SpecError, "duplicate name"):
            self.load(
                """
                jobs:
                  - {name: a, cmd: x}
                  - {name: a, cmd: y}
                """
            )

    def test_unknown_top_level_key(self):
        with self.assertRaisesRegex(SpecError, "unknown top-level fields"):
            self.load("jobs: [{cmd: x}]\ngpu_nmu: 1\n")

    def test_unknown_job_key(self):
        with self.assertRaisesRegex(SpecError, "unknown fields"):
            self.load("jobs:\n  - cmd: x\n    gpu_nums: 2\n")

    def test_missing_cmd(self):
        with self.assertRaisesRegex(SpecError, "missing required field cmd"):
            self.load("jobs:\n  - name: a\n")

    def test_empty_jobs(self):
        with self.assertRaisesRegex(SpecError, "non-empty list"):
            self.load("jobs: []\n")

    def test_bad_schedule(self):
        with self.assertRaisesRegex(SpecError, "schedule"):
            self.load("schedule: random\njobs: [{cmd: x}]\n")

    def test_bad_gpu_num(self):
        with self.assertRaisesRegex(SpecError, r"gpu-num"):
            self.load("jobs: [{cmd: x, gpu-num: 0}]\n")

    def test_bad_size(self):
        with self.assertRaisesRegex(SpecError, "size"):
            self.load("jobs: [{cmd: x, mem-per-gpu: 40Q}]\n")

    def test_bad_yaml_syntax(self):
        with self.assertRaisesRegex(SpecError, "YAML syntax error"):
            self.load("jobs: [\n")

    def test_missing_file(self):
        with self.assertRaisesRegex(SpecError, "does not exist"):
            load_workflow(self.tmp / "nope.yaml")


class ParsingSharedTest(unittest.TestCase):
    def test_parse_size_mb_errors_are_value_error(self):
        for bad in ("abc", "10Q", "-5G", "", None):
            with self.assertRaises(ValueError):
                parse_size_mb(bad)

    def test_parse_int_list_accepts_yaml_lists(self):
        self.assertEqual(parse_int_list([3, 1, 1]), [3, 1])
        self.assertEqual(parse_int_list("7"), [7])

    def test_ensure_interpreter(self):
        self.assertEqual(ensure_interpreter(["a.py"]), [sys.executable, "a.py"])
        self.assertEqual(ensure_interpreter(["a.sh"]), ["bash", "a.sh"])
        self.assertEqual(ensure_interpreter(["python", "a.py"]), ["python", "a.py"])

    def test_coerce_cmd(self):
        self.assertEqual(coerce_cmd("a b 'c d'", "t"), ["a", "b", "c d"])
        with self.assertRaises(ValueError):
            coerce_cmd("", "t")
        with self.assertRaises(ValueError):
            coerce_cmd([1, 2], "t")


if __name__ == "__main__":
    unittest.main()
