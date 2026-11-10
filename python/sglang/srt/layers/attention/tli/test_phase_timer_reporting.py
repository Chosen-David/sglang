"""CPU accounting tests for the actual diagnostic timer, not GPU timing tests."""

import ast
import contextlib
import io
import re
import unittest
from pathlib import Path


def timer_class(enabled=True):
    path = Path(__file__).with_name("backend.py")
    tree = ast.parse(path.read_text())
    node = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "_PhaseTimer"
    )
    namespace = {"_TIMING": enabled}
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace
    )
    return namespace["_PhaseTimer"]


def interval(timer, values):
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        for phases in values:
            for name, seconds in phases.items():
                timer.add(name, seconds)
            timer.maybe_report()
    return output.getvalue()


class PhaseTimerReportingTests(unittest.TestCase):
    def test_equal_intervals_have_equal_per_step_values(self):
        timer = timer_class()()
        lines = interval(timer, [{"select": 0.001}] * 192).splitlines()
        self.assertEqual(len(lines), 3)
        self.assertEqual(
            [float(re.search(r"\(([\d.]+)ms/step\)", line)[1]) for line in lines],
            [1.0] * 3,
        )
        self.assertEqual(timer.steps, 192)
        self.assertEqual(timer.acc, {"select": 0.0})

    def test_reports_only_current_interval_phase_totals(self):
        timer = timer_class()()
        interval(timer, [{"select": 0.003, "sparse_attn": 0.007}] * 64)
        line = interval(timer, [{"select": 0.002, "sparse_attn": 0.003}] * 64)
        self.assertIn("steps=128", line)
        self.assertIn("total=320.0ms", line)
        self.assertIn("(5.00ms/step)", line)
        self.assertIn("select=128.0ms", line)
        self.assertIn("sparse_attn=192.0ms", line)

    def test_report_cadence_and_disabled_timer(self):
        timer = timer_class()()
        self.assertEqual(interval(timer, [{"select": 0.001}] * 63), "")
        self.assertIn("steps=64", interval(timer, [{"select": 0.001}]))
        disabled = timer_class(enabled=False)()
        self.assertEqual(interval(disabled, [{"select": 0.001}] * 192), "")
        self.assertEqual(disabled.steps, 0)


if __name__ == "__main__":
    unittest.main()
