import io
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gpu_hunter.progress import Progress  # noqa: E402


class ProgressTest(unittest.TestCase):
    def test_failed_attempts_stay_on_one_line(self):
        out = io.StringIO()
        p = Progress(out)
        p.failed()
        p.failed()
        self.assertEqual(out.getvalue(), "xx")

    def test_success_breaks_pending_line(self):
        out = io.StringIO()
        p = Progress(out)
        p.failed()
        p.say("matched idle GPU(s) [2]")
        self.assertEqual(out.getvalue(), "x\nmatched idle GPU(s) [2]\n")

    def test_finish_is_idempotent(self):
        out = io.StringIO()
        p = Progress(out)
        p.failed()
        p.finish()
        p.finish()
        self.assertEqual(out.getvalue(), "x\n")


if __name__ == "__main__":
    unittest.main()
