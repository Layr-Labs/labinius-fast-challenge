#!/usr/bin/env python3
"""Median score selection and the refusal to publish a failed run."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import score  # noqa: E402


HOST = {
    "cpuModel": "test-cpu",
    "nproc": 8,
    "avx512f": True,
    "memTotalKb": 1024,
}


def verifier(modulus: str = "55144219393") -> dict:
    return {
        "modulus": modulus,
        "modulusBits": 36,
        "polynomialSize": 2**31,
        "packedElements": 2**24,
        "field": "GHASH",
    }


def run(time_ms: float, proof_bytes: int, modulus: str = "55144219393") -> dict:
    return {
        "provingTimeMs": time_ms,
        "proofBytes": proof_bytes,
        "verifier": verifier(modulus),
    }


class MedianScoreTest(unittest.TestCase):
    def test_median_time_and_its_proof_size(self) -> None:
        runs = [
            run(10, 100),
            run(30, 300, "99"),
            run(20, 200),
            run(50, 500),
            run(40, 400),
        ]
        document = score.score_document(runs, HOST)
        self.assertEqual(document["score"], 30)
        self.assertEqual(document["metrics"]["provingTimeMs"], 30)
        self.assertEqual(document["metrics"]["proofBytes"], 300)
        self.assertEqual(document["metrics"]["modulus"], "99")
        self.assertEqual(document["metrics"]["field"], "GHASH")
        self.assertEqual(document["metrics"]["cpuModel"], "test-cpu")
        self.assertEqual(
            document["runs"],
            [
                {"run": 1, "provingTimeMs": 10, "proofBytes": 100},
                {"run": 2, "provingTimeMs": 30, "proofBytes": 300},
                {"run": 3, "provingTimeMs": 20, "proofBytes": 200},
                {"run": 4, "provingTimeMs": 50, "proofBytes": 500},
                {"run": 5, "provingTimeMs": 40, "proofBytes": 400},
            ],
        )

    def test_tie_breaks_toward_the_earlier_run(self) -> None:
        runs = [run(5, 50), run(1, 10), run(5, 52), run(5, 53), run(9, 90)]
        chosen = score.median_run(runs)
        self.assertEqual(chosen["provingTimeMs"], 5)
        self.assertEqual(chosen["proofBytes"], 52)

    def test_incomplete_runs_are_rejected(self) -> None:
        with self.assertRaises(SystemExit):
            score.score_document([run(1, 10)] * 4, HOST)

    def test_bad_proof_size_is_rejected(self) -> None:
        runs = [run(1, 10) for _ in range(5)]
        runs[2] = run(1, 0)
        with self.assertRaises(SystemExit):
            score.score_document(runs, HOST)
        runs[2] = run(1, score.MAX_PROOF_BYTES + 1)
        with self.assertRaises(SystemExit):
            score.score_document(runs, HOST)

    def test_bad_verifier_output_is_rejected(self) -> None:
        runs = [run(1, 10) for _ in range(5)]
        runs[4]["verifier"] = {"field": "OTHER", "polynomialSize": 2**31}
        with self.assertRaises(SystemExit):
            score.score_document(runs, HOST)

    def test_cli_does_not_publish_a_failed_set(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs_path = root / "runs.json"
            out_path = root / "score.json"
            runs_path.write_text(json.dumps([run(1, 10)] * 4), encoding="utf-8")
            completed = subprocess.run(
                [
                    sys.executable,
                    str(Path(score.__file__).resolve()),
                    "--runs",
                    str(runs_path),
                    "--cpu-model",
                    "test-cpu",
                    "--nproc",
                    "8",
                    "--avx512f",
                    "true",
                    "--mem-total-kb",
                    "1024",
                    "--out",
                    str(out_path),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertFalse(out_path.exists())


if __name__ == "__main__":
    unittest.main()
