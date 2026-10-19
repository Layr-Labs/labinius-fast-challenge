#!/usr/bin/env bash
set -euo pipefail
umask 077

script_dir=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=.yukon/lib/sandbox.sh
source "$script_dir/lib/sandbox.sh"

# GitHub supplies these paths. Prepare them here for local CLI use.
local_setup=false
if [[ -z "${RUNNER_TEMP:-}" && -z "${GITHUB_ACTIONS:-}" ]]; then
  local_setup=true
  rm -f "$script_dir/local-workspace" "$script_dir/labinius-fast-score.json"
  if [[ $(uname -s) != Linux || $(uname -m) != x86_64 ]]; then
    echo 'Local setup requires Ubuntu 24.04 on x86-64.' >&2
    exit 1
  fi
  source /etc/os-release
  if [[ ${ID:-} != ubuntu || ${VERSION_ID:-} != 24.04 ]]; then
    echo 'Local setup requires Ubuntu 24.04.' >&2
    exit 1
  fi
  grep -qw avx512f /proc/cpuinfo || { echo 'AVX-512F is required.' >&2; exit 1; }

  packages=(build-essential gcc-13 bubblewrap python3 openssl libssl-dev pkg-config curl ca-certificates git)
  missing=()
  for package in "${packages[@]}"; do
    if [[ $(dpkg-query -W -f='${Status}' "$package" 2>/dev/null || true) != 'install ok installed' ]]; then
      missing+=("$package")
    fi
  done
  if (( ${#missing[@]} )); then
    sudo apt-get update
    sudo apt-get install -y "${missing[@]}"
  fi

  if [[ ! -e /opt/cargo && ! -L /opt/cargo && ! -e /opt/rustup && ! -L /opt/rustup ]]; then
    installer=$(mktemp)
    trap 'rm -f "$installer"' EXIT
    curl --proto '=https' --tlsv1.2 -fsS https://sh.rustup.rs -o "$installer"
    sudo env CARGO_HOME=/opt/cargo RUSTUP_HOME=/opt/rustup sh "$installer" \
      -y --profile minimal --default-toolchain stable --no-modify-path
    rm -f "$installer"
    trap - EXIT
  fi
  env CARGO_HOME=/opt/cargo RUSTUP_HOME=/opt/rustup /opt/cargo/bin/rustc --version

  export UNTRUSTED_CHECKOUT="$(git -C "$script_dir" rev-parse --show-toplevel)"
  revision=$(git -C "$UNTRUSTED_CHECKOUT" rev-parse HEAD)
  export RUNNER_TEMP="$(mktemp -d /var/tmp/labinius-local.XXXXXX)"
  export TRUSTED_CHECKOUT="$RUNNER_TEMP/trusted-checkout"
  printf 'Local workspace: %s\n' "$RUNNER_TEMP"
  git clone --no-hardlinks --no-checkout "$UNTRUSTED_CHECKOUT" "$TRUSTED_CHECKOUT"
  git -C "$TRUSTED_CHECKOUT" checkout --detach "$revision"
  git -C "$TRUSTED_CHECKOUT" submodule update --init --recursive
  printf '%s\n' "$UNTRUSTED_CHECKOUT" > "$RUNNER_TEMP/source-checkout"
  unset TRUSTED_DIR TRUSTED_CARGO_HOME TRUSTED_TARGET_DIR CONTESTANT_DIR CONTESTANT_CARGO_HOME
fi

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
# that cannot remain beneath solution/. This private RUNNER_TEMP copy is the
# contestant's only writable source tree for native build-system outputs.
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
  /usr/bin/bash -eu -c '
    manifest=./Cargo.toml
    if [[ ! -x /workspace || ! -x /workspace/solution || ! -r "$manifest" ]]; then
      echo "staged contestant manifest is not readable inside the sandbox" >&2
      exit 1
    fi
    exec /opt/cargo/bin/cargo fetch --locked --manifest-path "$manifest"
  '

run_contestant_build \
  "$contestant/solution" \
  "$contestant/solution/proof_generate/target" \
  "$contestant_cargo" \
  offline \
  /usr/bin/bash -eu -c '
    manifest=./Cargo.toml
    if [[ ! -x /workspace || ! -x /workspace/solution || ! -r "$manifest" ]]; then
      echo "staged contestant manifest is not readable inside the sandbox" >&2
      exit 1
    fi
    exec /opt/cargo/bin/cargo build \
      --locked --offline --release --bin labinius-fast --manifest-path "$manifest"
  '

prover="$contestant/solution/proof_generate/target/release/labinius-fast"
if [[ ! -f "$prover" || -L "$prover" ]]; then
  echo "contestant build did not produce a regular prover binary" >&2
  exit 1
fi

# Publish state only after both builds succeed. Each setup uses a fresh workspace.
if $local_setup; then
  state=$(mktemp "$script_dir/local-workspace.XXXXXX")
  printf '%s\n' "$RUNNER_TEMP" > "$state"
  mv -f "$state" "$script_dir/local-workspace"
  echo 'Setup complete. Run yukon run; rerun yukon setup after solution edits.'
fi
