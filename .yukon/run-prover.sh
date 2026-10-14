#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=.yukon/lib/sandbox.sh
source "$script_dir/lib/sandbox.sh"

if [[ "$#" -ne 5 ]]; then
  echo "usage: run-prover.sh PROVER WORKDIR SEED SANDBOX_PROOF_PATH TABLE" >&2
  exit 2
fi

run_contestant_prover "$1" "$2" "$3" "$4" "$5"
