#!/usr/bin/env bash

# Trusted benchmark orchestration. The score is the median of five sequential
# runs, published only when every run passes verification and the proof-size limit.

set -euo pipefail
umask 077

script_dir=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=.yukon/lib/sandbox.sh
source "$script_dir/lib/sandbox.sh"

# Recover the successful local build when no CI workspace was supplied.
if [[ -z "${RUNNER_TEMP:-}" && -z "${GITHUB_ACTIONS:-}" ]]; then
  rm -f "$script_dir/labinius-fast-score.json"
  state="$script_dir/local-workspace"
  if [[ ! -f "$state" || -L "$state" ]]; then
    echo 'Run yukon setup first.' >&2
    exit 1
  fi
  IFS= read -r RUNNER_TEMP < "$state"
  export RUNNER_TEMP
  if [[ "$RUNNER_TEMP" != /var/tmp/labinius-local.* || ! -d "$RUNNER_TEMP" || -L "$RUNNER_TEMP" || ! -O "$RUNNER_TEMP" ]]; then
    echo 'Local workspace is unavailable. Rerun yukon setup.' >&2
    exit 1
  fi
  root=$(git -C "$script_dir" rev-parse --show-toplevel)
  if [[ $(cat "$RUNNER_TEMP/source-checkout") != "$root" ||
        $(git -C "$RUNNER_TEMP/trusted-checkout" rev-parse HEAD) != $(git -C "$root" rev-parse HEAD) ]]; then
    echo 'Checkout changed. Rerun yukon setup.' >&2
    exit 1
  fi
  unset TRUSTED_DIR CONTESTANT_DIR
fi

: "${RUNNER_TEMP:?RUNNER_TEMP is required}"
trusted="${TRUSTED_DIR:-$RUNNER_TEMP/labinius-trusted}"
contestant="${CONTESTANT_DIR:-$RUNNER_TEMP/labinius-contestant}"
require_runner_temp_path "$trusted"
require_runner_temp_path "$contestant"

score="$script_dir/labinius-fast-score.json"
staged="$script_dir/labinius-fast-score.tmp"
runs_json="$script_dir/labinius-fast-runs.tmp"
rm -f "$score" "$staged" "$runs_json"

command -v bwrap >/dev/null || {
  echo "bubblewrap (bwrap) is required to run the untrusted prover" >&2
  exit 1
}

prover="$contestant/solution/proof_generate/target/release/labinius-fast"
verifier="$trusted/labinius-fast-verify"
max_bytes=$((64 * 1024 * 1024))

if [[ ! -f "$prover" || -L "$prover" ]]; then
  echo "missing regular contestant prover binary" >&2
  exit 1
fi
if [[ ! -f "$verifier" || -L "$verifier" ]]; then
  echo "missing regular trusted verifier binary" >&2
  exit 1
fi

got=$(sha256_file "$verifier")
want=$(awk '{print $1}' "$trusted/verify.sha256")
if [[ "$got" != "$want" ]]; then
  echo "trusted verifier hash changed" >&2
  exit 1
fi

cpu_model=$(awk -F: '/^model name/ { gsub(/^ +/, "", $2); print $2; exit }' /proc/cpuinfo)
nproc_count=$(nproc)
if grep -qw avx512f /proc/cpuinfo; then
  avx512f=true
else
  avx512f=false
fi
mem_total_kb=$(awk '/^MemTotal:/ { print $2; exit }' /proc/meminfo)

workdir=$(mktemp -d "$RUNNER_TEMP/labinius-fast-proof.XXXXXX")
table_dir=$(mktemp -d "$RUNNER_TEMP/labinius-fast-table.XXXXXX")
cleanup() {
  rm -rf "$workdir" "$table_dir"
  rm -f "$staged" "$runs_json"
}
trap cleanup EXIT
proof="$workdir/proof.bin"
table="$table_dir/witness-table.bin"
printf '[]\n' > "$runs_json"

for trial in 1 2 3 4 5; do
  # A fresh seed keeps a prover from reusing a cached proof. The witness table
  # is written before the timer; verification runs after it.
  LABINIUS_SEED=$(openssl rand -hex 32)
  rm -f "$table" "$proof"
  env -i PATH=/usr/bin:/bin LABINIUS_SEED="$LABINIUS_SEED" \
    "$verifier" --write-table "$table"
  chmod 444 "$table"

  prove_ms=$(python3 - "$script_dir" "$prover" "$LABINIUS_SEED" "$workdir" "$table" <<'PY'
import subprocess
import sys
import time

script_dir, prover, seed, workdir, table = sys.argv[1:6]
cmd = [
    "/usr/bin/bash",
    f"{script_dir}/run-prover.sh",
    prover,
    workdir,
    seed,
    "/work/proof.bin",
    table,
]
start = time.perf_counter()
subprocess.run(cmd, timeout=900, check=True)
print((time.perf_counter() - start) * 1000)
PY
)

  proof_bytes=$(regular_file_size "$proof")
  if (( proof_bytes <= 0 || proof_bytes > max_bytes )); then
    echo "empty or oversized proof on run ${trial}" >&2
    exit 1
  fi

  verify_out=$(env -i PATH=/usr/bin:/bin LABINIUS_SEED="$LABINIUS_SEED" python3 - "$verifier" "$proof" <<'PY'
import subprocess
import sys

checked = subprocess.run(
    [sys.argv[1], "--verify", sys.argv[2]],
    timeout=900,
    capture_output=True,
    text=True,
)
if checked.returncode != 0:
    sys.stderr.write(checked.stderr)
    sys.stderr.write(checked.stdout)
    raise SystemExit(checked.returncode)
sys.stdout.write(checked.stdout)
PY
)

  python3 -c '
import json, sys
path, prove_ms, proof_bytes, verifier = sys.argv[1:5]
with open(path, encoding="utf-8") as handle:
    runs = json.load(handle)
runs.append({
    "provingTimeMs": float(prove_ms),
    "proofBytes": int(proof_bytes),
    "verifier": json.loads(verifier),
})
with open(path, "w", encoding="utf-8") as handle:
    json.dump(runs, handle)
' "$runs_json" "$prove_ms" "$proof_bytes" "$verify_out"
done

python3 "$script_dir/lib/score.py" \
  --runs "$runs_json" \
  --cpu-model "$cpu_model" \
  --nproc "$nproc_count" \
  --avx512f "$avx512f" \
  --mem-total-kb "$mem_total_kb" \
  --out "$staged"

mv "$staged" "$score"
