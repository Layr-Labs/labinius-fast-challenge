#!/usr/bin/env python3
"""Publish a leaderboard score from five successful prover runs."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

TRIALS = 5
MAX_PROOF_BYTES = 64 * 1024 * 1024


def _proof_bytes(run: dict) -> int:
    size = run["proofBytes"]
    if isinstance(size, bool) or not isinstance(size, int):
        raise SystemExit("run failed the proof-size requirement")
    if size <= 0 or size > MAX_PROOF_BYTES:
        raise SystemExit("run failed the proof-size requirement")
    return size


def _proving_time_ms(run: dict) -> float:
    time_ms = run["provingTimeMs"]
    if isinstance(time_ms, bool) or not isinstance(time_ms, (int, float)):
        raise SystemExit("invalid proving time")
    if not math.isfinite(time_ms) or time_ms < 0:
        raise SystemExit("invalid proving time")
    return float(time_ms)


def _verifier_metrics(run: dict) -> dict:
    metrics = run.get("verifier")
    if not isinstance(metrics, dict):
        raise SystemExit("run is missing verifier output")
    if metrics.get("field") != "GHASH" or metrics.get("polynomialSize") != 2**31:
        raise SystemExit("checker reported unexpected parameters")
    return metrics


def median_run(runs: list[dict]) -> dict:
    """Select the median run. Ties break toward the earlier execution index."""
    if len(runs) != TRIALS:
        raise SystemExit(f"expected {TRIALS} successful runs, got {len(runs)}")
    for run in runs:
        _proof_bytes(run)
        _proving_time_ms(run)
        _verifier_metrics(run)
    order = sorted(range(len(runs)), key=lambda index: (_proving_time_ms(runs[index]), index))
    return runs[order[len(runs) // 2]]


def score_document(runs: list[dict], host: dict) -> dict:
    chosen = median_run(runs)
    metrics = dict(_verifier_metrics(chosen))
    metrics["provingTimeMs"] = _proving_time_ms(chosen)
    metrics["proofBytes"] = _proof_bytes(chosen)
    metrics["cpuModel"] = host["cpuModel"]
    metrics["nproc"] = host["nproc"]
    metrics["avx512f"] = host["avx512f"]
    metrics["memTotalKb"] = host["memTotalKb"]
    return {
        "score": metrics["provingTimeMs"],
        "runs": [
            {
                "run": index + 1,
                "provingTimeMs": _proving_time_ms(run),
                "proofBytes": _proof_bytes(run),
            }
            for index, run in enumerate(runs)
        ],
        "metrics": metrics,
    }


def write_score(runs: list[dict], host: dict, path: Path) -> str:
    document = score_document(runs, host)
    text = json.dumps(document) + "\n"
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", required=True)
    parser.add_argument("--cpu-model", required=True)
    parser.add_argument("--nproc", required=True, type=int)
    parser.add_argument("--avx512f", required=True, choices=("true", "false"))
    parser.add_argument("--mem-total-kb", required=True, type=int)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    runs = json.loads(Path(args.runs).read_text(encoding="utf-8"))
    text = write_score(
        runs,
        {
            "cpuModel": args.cpu_model,
            "nproc": args.nproc,
            "avx512f": args.avx512f == "true",
            "memTotalKb": args.mem_total_kb,
        },
        Path(args.out),
    )
    sys.stdout.write(text)


if __name__ == "__main__":
    main()
