#!/usr/bin/env bash
# Median selection and refusal to publish a failed or oversized run.
set -euo pipefail
script_dir=$(cd "$(dirname "$0")/.." && pwd)
python3 "$script_dir/tests/test_score.py"
echo "scoring checks passed"
