#!/usr/bin/env python3
"""Exercise the benchmark's actual timer with harmless stand-in provers."""

import contextlib
import io
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


BENCHMARK = Path(__file__).resolve().parents[1] / "benchmark.sh"
TIMER = re.search(
    r"prove_ms=\$\(python3 [^\n]+ <<'PY'\n(.*?)\nPY\n",
    BENCHMARK.read_text(),
    re.DOTALL,
).group(1)


@unittest.skipUnless(
    sys.platform == "linux" and os.access("/usr/bin/timeout", os.X_OK),
    "requires the benchmark's Linux/GNU timeout environment",
)
class BenchmarkTimingTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output = io.StringIO()

    def run_timer(self, prover):
        (self.root / "run-prover.sh").write_text("#!/bin/bash\n" + prover)
        argv = ["timer", str(self.root), "prover", "seed", str(self.root), "table"]
        with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(
            self.output
        ):
            exec(compile(TIMER, str(BENCHMARK), "exec"), {})

    def test_waits_for_exit_without_timeout_polling(self):
        original_wait = subprocess.Popen._wait

        def direct_wait(process, timeout):
            self.assertIsNone(timeout, "timed process must not use polling wait")
            return original_wait(process, timeout)

        with mock.patch.object(subprocess.Popen, "_wait", direct_wait):
            self.run_timer("exec /bin/sleep 0.01\n")
        self.assertGreaterEqual(float(self.output.getvalue()), 10)

    def test_child_failure_does_not_publish_a_time(self):
        with self.assertRaises(subprocess.CalledProcessError) as caught:
            self.run_timer("exit 23\n")
        self.assertEqual(caught.exception.returncode, 23)
        self.assertEqual(self.output.getvalue(), "")

    def test_deadline_kills_child_and_does_not_publish_a_time(self):
        original_run = subprocess.run

        def shortened_deadline(command, **kwargs):
            self.assertEqual(
                command[:3], ["/usr/bin/timeout", "--signal=KILL", "900s"]
            )
            return original_run(
                [*command[:2], "0.05s", *command[3:]], **kwargs
            )

        with mock.patch.object(subprocess, "run", shortened_deadline):
            with self.assertRaises(subprocess.CalledProcessError) as caught:
                self.run_timer('echo $$ > "$2/pid"\nexec /bin/sleep 5\n')
        self.assertIn(caught.exception.returncode, (-signal.SIGKILL, 137))
        self.assertEqual(self.output.getvalue(), "")
        pid = int((self.root / "pid").read_text())
        # Killed children may briefly await reaping by init.
        status = Path(f"/proc/{pid}/stat")
        try:
            state = status.read_text().split(") ", 1)[1][0]
        except FileNotFoundError:
            pass
        else:
            self.assertEqual(state, "Z")


if __name__ == "__main__":
    unittest.main()
