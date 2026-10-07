#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
command -v cargo >/dev/null || { echo "Install Rust with rustup before setup" >&2; exit 1; }
command -v bwrap >/dev/null || { echo "bubblewrap (bwrap) is required to build the untrusted solution" >&2; exit 1; }

# verifier/labinius is a gitlink. A normal checkout leaves that directory empty.
if [[ ! -f verifier/labinius/crates/pcs/Cargo.toml ]]; then
  git submodule update --init verifier/labinius
fi

# Trusted checker first, before any contestant build runs.
(cd verifier/proof_verify && cargo build --locked --release --bin labinius-fast-verify)

trusted="${TRUSTED_DIR:-${RUNNER_TEMP:-/tmp}/labinius-trusted}"
mkdir -p "$trusted"
chmod u+w "$trusted"
install -m 755 verifier/proof_verify/target/release/labinius-fast-verify "$trusted/labinius-fast-verify"
if command -v sha256sum >/dev/null 2>&1; then
  sha256sum "$trusted/labinius-fast-verify" | awk '{print $1}' > "$trusted/verify.sha256"
else
  shasum -a 256 "$trusted/labinius-fast-verify" | awk '{print $1}' > "$trusted/verify.sha256"
fi
chmod 555 "$trusted/labinius-fast-verify"
chmod 444 "$trusted/verify.sha256"
chmod 555 "$trusted"

# Contestant build sees the repo read-only except solution/ and the cargo home.
root=$(pwd)
export CARGO_HOME="${CARGO_HOME:-$HOME/.cargo}"
mkdir -p "$CARGO_HOME"
bwrap \
  --ro-bind / / \
  --dev /dev \
  --proc /proc \
  --tmpfs /tmp \
  --bind "$root/solution" "$root/solution" \
  --bind "$CARGO_HOME" "$CARGO_HOME" \
  --unshare-pid \
  --die-with-parent \
  --chdir "$root/solution/proof_generate" \
  -- \
  cargo build --locked --release --bin labinius-fast
