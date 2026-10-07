#!/usr/bin/env bash

# Trusted benchmark orchestration. Only an accepted nonempty proof with 
# valid moduli produces a score.

set -euo pipefail
cd "$(dirname "$0")/.."

score=.yukon/labinius-fast-score.json
staged=.yukon/labinius-fast-score.tmp
rm -f "$score" "$staged"

command -v bwrap >/dev/null || { echo "bubblewrap (bwrap) is required to run the untrusted prover" >&2; exit 1; }

root=$(pwd)
trusted="${TRUSTED_DIR:-${RUNNER_TEMP:-/tmp}/labinius-trusted}"
prover=$root/solution/proof_generate/target/release/labinius-fast
verifier=$trusted/labinius-fast-verify
max_bytes=$((64 * 1024 * 1024))

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

# sample os randomness to seed the transcript. This is to prevent prevent a
# malicious prover from being able to cheat via attacks such as caching prior 
# proofs. This randomness is used to setup the problem instance the prover will 
# see (ie, exact polynomials)
export LABINIUS_SEED
LABINIUS_SEED=$(openssl rand -hex 32)

# Keep the proof directory off /tmp: the prover sandbox mounts a fresh tmpfs there.
workparent="${TMPDIR:-/var/tmp}"
case "$workparent" in
  /tmp|/tmp/*) workparent=/var/tmp ;;
esac
mkdir -p "$workparent"
workdir=$(mktemp -d "$workparent/labinius-fast.XXXXXX")
cleanup() { rm -rf "$workdir"; rm -f "$staged"; }
trap cleanup EXIT
proof=$workdir/proof.bin

prove_ms=$(python3 - "$prover" "$proof" "$LABINIUS_SEED" "$workdir" <<'PY'
import subprocess, sys, time
prover, proof, seed, workdir = sys.argv[1:5]
cmd = [
    "bwrap",
    "--ro-bind", "/", "/",
    "--dev", "/dev",
    "--proc", "/proc",
    "--tmpfs", "/tmp",
    "--bind", workdir, workdir,
    "--unshare-pid", "--unshare-net", "--die-with-parent",
    "--setenv", "LABINIUS_SEED", seed,
    "--chdir", workdir,
    "--", prover, "--prove", proof,
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

verify_out=$(python3 - "$verifier" "$proof" <<'PY'
import subprocess, sys
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
if metrics.get("field") != "GHASH" or metrics.get("polynomialSize") != 2**24:
    raise SystemExit("checker reported unexpected parameters")
prove_ms = float(sys.argv[2])
metrics["provingTimeMs"] = prove_ms
metrics["proofBytes"] = int(sys.argv[3])
metrics["cpuModel"] = sys.argv[5]
metrics["nproc"] = int(sys.argv[6])
metrics["avx512f"] = sys.argv[7] == "true"
result = {"score": prove_ms, "metrics": metrics}
text = json.dumps(result) + "\n"
with open(sys.argv[4], "w", encoding="utf-8") as staged:
    staged.write(text)
sys.stdout.write(text)
' "$verify_out" "$prove_ms" "$proof_bytes" "$staged" "$cpu_model" "$nproc_count" "$avx512f"

mv "$staged" "$score"
