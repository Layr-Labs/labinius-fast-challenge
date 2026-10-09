#!/usr/bin/env bash

# Trusted benchmark orchestration. Only an accepted nonempty proof with valid
# moduli produces a score.

set -euo pipefail
umask 077

script_dir=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=.yukon/lib/sandbox.sh
source "$script_dir/lib/sandbox.sh"

: "${RUNNER_TEMP:?RUNNER_TEMP is required}"
trusted="${TRUSTED_DIR:-$RUNNER_TEMP/labinius-trusted}"
contestant="${CONTESTANT_DIR:-$RUNNER_TEMP/labinius-contestant}"
require_runner_temp_path "$trusted"
require_runner_temp_path "$contestant"

score="$script_dir/labinius-fast-score.json"
staged="$script_dir/labinius-fast-score.tmp"
rm -f "$score" "$staged"

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

# Randomize the problem instance so a prover cannot rely on a cached proof.
LABINIUS_SEED=$(openssl rand -hex 32)
export LABINIUS_SEED

workdir=$(mktemp -d "$RUNNER_TEMP/labinius-fast-proof.XXXXXX")
cleanup() {
  rm -rf "$workdir"
  rm -f "$staged"
}
trap cleanup EXIT
proof="$workdir/proof.bin"

prove_ms=$(python3 - "$script_dir" "$prover" "$LABINIUS_SEED" "$workdir" <<'PY'
import subprocess
import sys
import time

script_dir, prover, seed, workdir = sys.argv[1:5]
cmd = [
    "/usr/bin/bash",
    f"{script_dir}/run-prover.sh",
    prover,
    workdir,
    seed,
    "/work/proof.bin",
]
start = time.perf_counter()
subprocess.run(cmd, timeout=900, check=True)
print((time.perf_counter() - start) * 1000)
PY
)

proof_bytes=$(wc -c < "$proof" | tr -d ' ')
if (( proof_bytes <= 0 || proof_bytes > max_bytes )); then
  echo "empty or oversized proof" >&2
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

verify_out=$(env -i PATH=/usr/bin:/bin python3 - "$verifier" "$proof" <<'PY'
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
metrics = json.loads(sys.argv[1])
if metrics.get("field") != "GHASH" or metrics.get("polynomialSize") != 2**31:
    raise SystemExit("checker reported unexpected parameters")
prove_ms = float(sys.argv[2])
metrics["provingTimeMs"] = prove_ms
metrics["proofBytes"] = int(sys.argv[3])
metrics["cpuModel"] = sys.argv[5]
metrics["nproc"] = int(sys.argv[6])
metrics["avx512f"] = sys.argv[7] == "true"
metrics["memTotalKb"] = int(sys.argv[8])
result = {"score": prove_ms, "metrics": metrics}
text = json.dumps(result) + "\n"
with open(sys.argv[4], "w", encoding="utf-8") as staged:
    staged.write(text)
sys.stdout.write(text)
' "$verify_out" "$prove_ms" "$proof_bytes" "$staged" "$cpu_model" "$nproc_count" "$avx512f" "$mem_total_kb"

mv "$staged" "$score"
