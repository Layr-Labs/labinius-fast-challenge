#!/usr/bin/env python3
"""Structural regression checks for the protected benchmark workflow adapter."""

import json
from pathlib import Path
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"
WRAPPER_CALL = (
    "Layr-Labs/labinius-fast-challenge/"
    ".github/workflows/benchmark-runner.yml@master"
)
PR_CALL = (
    "Layr-Labs/labinius-fast-challenge/"
    ".github/workflows/benchmark.yml@master"
)
DEFAULT_REPOSITORY = "${{ inputs.solution_repository || github.repository }}"
DEFAULT_REF = "${{ inputs.solution_ref || github.sha }}"


def load_workflow(name: str) -> dict:
    # BaseLoader keeps `on` as a string instead of applying YAML 1.1's
    # surprising on/off boolean coercion.
    with (WORKFLOWS / name).open(encoding="utf-8") as workflow_file:
        return yaml.load(workflow_file, Loader=yaml.BaseLoader)


def step_named(workflow: dict, name: str) -> dict:
    steps = workflow["jobs"]["benchmark"]["steps"]
    return next(step for step in steps if step["name"] == name)


class BenchmarkWorkflowTest(unittest.TestCase):
    def setUp(self) -> None:
        self.wrapper = load_workflow("benchmark.yml")
        self.executor = load_workflow("benchmark-runner.yml")
        self.pull_request = load_workflow("test-benchmark.yml")

    def test_manifest_keeps_original_workflow_filename(self) -> None:
        with (ROOT / "benchmark.json").open(encoding="utf-8") as manifest_file:
            manifest = json.load(manifest_file)
        self.assertEqual(manifest["runner"]["workflow"], "benchmark.yml")

    def test_wrapper_preserves_dispatch_and_call_entrypoints(self) -> None:
        self.assertEqual(
            set(self.wrapper["on"]), {"workflow_dispatch", "workflow_call"}
        )
        for event in ("workflow_dispatch", "workflow_call"):
            inputs = self.wrapper["on"][event]["inputs"]
            self.assertEqual(
                set(inputs), {"solution_repository", "solution_ref"}
            )
            for input_name in inputs:
                self.assertEqual(inputs[input_name]["required"], "false")
                self.assertEqual(inputs[input_name]["type"], "string")

    def test_baseline_and_submission_defaults_use_dispatch_run_identity(self) -> None:
        self.assertEqual(set(self.wrapper["jobs"]), {"benchmark"})
        job = self.wrapper["jobs"]["benchmark"]
        self.assertEqual(job["uses"], WRAPPER_CALL)
        self.assertNotIn("runs-on", job)
        self.assertNotIn("steps", job)
        self.assertEqual(
            job["with"],
            {
                "solution_repository": DEFAULT_REPOSITORY,
                "solution_ref": DEFAULT_REF,
            },
        )
        self.assertEqual(
            self.wrapper["concurrency"],
            {
                "group": (
                    "benchmark-${{ github.repository }}-"
                    "${{ github.workflow }}-${{ github.ref }}-"
                    "${{ inputs.solution_ref || github.sha }}"
                ),
                "cancel-in-progress": "true",
            },
        )

    def test_permissions_remain_read_only(self) -> None:
        expected = {"contents": "read", "actions": "read"}
        self.assertEqual(self.wrapper["permissions"], expected)
        self.assertEqual(self.executor["permissions"], expected)
        self.assertEqual(self.pull_request["permissions"], expected)

    def test_executor_is_the_only_runner_job(self) -> None:
        self.assertEqual(set(self.executor["on"]), {"workflow_call"})
        self.assertEqual(set(self.executor["jobs"]), {"benchmark"})
        job = self.executor["jobs"]["benchmark"]
        self.assertNotIn("uses", job)
        self.assertEqual(
            job["runs-on"],
            {
                "group": "labinius-intel-8488c-avx512-c7i",
                "labels": "labinius-avx512-c7i",
            },
        )
        self.assertEqual(job["timeout-minutes"], "120")
        self.assertNotIn("concurrency", self.executor)

    def test_pr_and_bot_concurrency_keys_do_not_collide(self) -> None:
        template = self.wrapper["concurrency"]["group"]

        def evaluate(workflow: str, ref: str, solution_sha: str) -> str:
            replacements = {
                "${{ github.repository }}": "Layr-Labs/labinius-fast-challenge",
                "${{ github.workflow }}": workflow,
                "${{ github.ref }}": ref,
                "${{ inputs.solution_ref || github.sha }}": solution_sha,
            }
            key = template
            for expression, value in replacements.items():
                key = key.replace(expression, value)
            self.assertNotIn("${{", key)
            return key

        solution_sha = "a" * 40
        bot_key = evaluate(
            "Benchmark", "refs/heads/submissions/example", solution_sha
        )
        repeated_bot_key = evaluate(
            "Benchmark", "refs/heads/submissions/example", solution_sha
        )
        pr_key = evaluate(
            "Test PCS benchmark", "refs/pull/123/merge", solution_sha
        )

        self.assertEqual(bot_key, repeated_bot_key)
        self.assertNotEqual(bot_key, pr_key)

    def test_executor_keeps_trusted_and_solution_checkout_identity(self) -> None:
        trusted = step_named(self.executor, "Checkout trusted harness")
        self.assertEqual(
            trusted["with"]["repository"],
            "Layr-Labs/labinius-fast-challenge",
        )
        self.assertEqual(trusted["with"]["ref"], "${{ job.workflow_sha }}")
        self.assertEqual(trusted["with"]["path"], ".trusted-src")
        self.assertEqual(trusted["with"]["submodules"], "recursive")
        self.assertEqual(trusted["with"]["persist-credentials"], "false")

        solution = step_named(self.executor, "Checkout untrusted solution input")
        self.assertEqual(solution["with"]["repository"], DEFAULT_REPOSITORY)
        self.assertEqual(solution["with"]["ref"], DEFAULT_REF)
        self.assertEqual(solution["with"]["path"], ".untrusted-src")
        self.assertEqual(solution["with"]["fetch-depth"], "1")
        self.assertEqual(solution["with"]["submodules"], "false")
        self.assertEqual(solution["with"]["persist-credentials"], "false")

    def test_executor_keeps_score_artifact_contract(self) -> None:
        upload = step_named(self.executor, "Upload score")
        self.assertEqual(upload["if"], "always()")
        self.assertEqual(upload["with"]["name"], "benchmark-score")
        self.assertEqual(
            upload["with"]["path"],
            (
                "${{ github.workspace }}/.trusted-src/"
                ".yukon/labinius-fast-score.json"
            ),
        )
        self.assertEqual(upload["with"]["if-no-files-found"], "error")

    def test_pull_request_workflow_still_calls_protected_wrapper(self) -> None:
        job = self.pull_request["jobs"]["benchmark"]
        self.assertEqual(job["uses"], PR_CALL)
        self.assertEqual(
            job["with"],
            {
                "solution_repository": (
                    "${{ github.event.pull_request.head.repo.full_name }}"
                ),
                "solution_ref": "${{ github.event.pull_request.head.sha }}",
            },
        )


if __name__ == "__main__":
    unittest.main()
