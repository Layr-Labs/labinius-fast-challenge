#!/usr/bin/env bash
set -euo pipefail
umask 077

script_dir=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=.yukon/lib/sandbox.sh
source "$script_dir/lib/sandbox.sh"

: "${RUNNER_TEMP:?RUNNER_TEMP is required}"
: "${TRUSTED_CHECKOUT:?TRUSTED_CHECKOUT is required}"
: "${UNTRUSTED_CHECKOUT:?UNTRUSTED_CHECKOUT is required}"

trusted="${TRUSTED_DIR:-$RUNNER_TEMP/labinius-trusted}"
trusted_cargo="${TRUSTED_CARGO_HOME:-$RUNNER_TEMP/labinius-trusted-cargo}"
trusted_target="${TRUSTED_TARGET_DIR:-$RUNNER_TEMP/labinius-trusted-target}"
contestant="${CONTESTANT_DIR:-$RUNNER_TEMP/labinius-contestant}"
contestant_cargo="${CONTESTANT_CARGO_HOME:-$RUNNER_TEMP/labinius-contestant-cargo}"

require_runner_temp_path "$trusted"
require_runner_temp_path "$trusted_cargo"
require_runner_temp_path "$trusted_target"
require_runner_temp_path "$contestant"
require_runner_temp_path "$contestant_cargo"
require_host_command /opt/cargo/bin/cargo
require_host_command /opt/cargo/bin/rustc
require_host_command /usr/bin/gcc-13
command -v bwrap >/dev/null || {
  echo "bubblewrap (bwrap) is required to build the untrusted solution" >&2
  exit 1
}

if [[ ! -f "$TRUSTED_CHECKOUT/verifier/labinius/crates/pcs/Cargo.toml" ]]; then
  echo "trusted checkout is missing its pinned verifier submodule" >&2
  exit 1
fi
if [[ ! -d "$UNTRUSTED_CHECKOUT/solution" ]]; then
  echo "untrusted checkout has no solution directory" >&2
  exit 1
fi

mkdir -p "$trusted" "$trusted_cargo" "$trusted_target"
chmod 700 "$trusted" "$trusted_cargo" "$trusted_target"

# Build only protected trusted source outside the contestant sandbox. Its Cargo
# home and target are disposable and never share contestant state or host tools.
env -i \
  PATH=/opt/cargo/bin:/usr/bin:/bin \
  HOME="$trusted_cargo" \
  RUSTUP_HOME=/opt/rustup \
  CARGO_HOME="$trusted_cargo" \
  CARGO_TARGET_DIR="$trusted_target" \
  CC=/usr/bin/gcc-13 \
  CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER=/usr/bin/gcc-13 \
  RUSTFLAGS="-C target-cpu=native" \
  /opt/cargo/bin/cargo build \
    --locked \
    --release \
    --bin labinius-fast-verify \
    --manifest-path "$TRUSTED_CHECKOUT/verifier/proof_verify/Cargo.toml"

install -m 755 "$trusted_target/release/labinius-fast-verify" "$trusted/labinius-fast-verify"
sha256_file "$trusted/labinius-fast-verify" > "$trusted/verify.sha256"
chmod 555 "$trusted/labinius-fast-verify"
chmod 444 "$trusted/verify.sha256"
chmod 555 "$trusted"

# Copy data only: reject symlinks, special files, git control data, and paths
# that cannot remain beneath solution/.
python3 "$script_dir/stage-solution.py" \
  "$UNTRUSTED_CHECKOUT/solution" \
  "$contestant/solution"

mkdir -p "$contestant_cargo" "$contestant/solution/proof_generate/target"
chmod 700 "$contestant_cargo"

run_contestant_build \
  "$contestant/solution" \
  "$contestant/solution/proof_generate/target" \
  "$contestant_cargo" \
  fetch \
  /opt/cargo/bin/cargo fetch \
    --locked \
    --manifest-path /workspace/solution/proof_generate/Cargo.toml

run_contestant_build \
  "$contestant/solution" \
  "$contestant/solution/proof_generate/target" \
  "$contestant_cargo" \
  offline \
  /opt/cargo/bin/cargo build \
    --locked \
    --offline \
    --release \
    --bin labinius-fast \
    --manifest-path /workspace/solution/proof_generate/Cargo.toml

prover="$contestant/solution/proof_generate/target/release/labinius-fast"
if [[ ! -f "$prover" || -L "$prover" ]]; then
  echo "contestant build did not produce a regular prover binary" >&2
  exit 1
fi
