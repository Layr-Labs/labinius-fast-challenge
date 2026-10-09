# Dedicated runner security boundary

The benchmark job targets the organization runner group and label
`labinius-avx512-c7i`. That group is an external admission boundary:
organization settings must keep it closed until review, then restrict it to the
protected reusable workflow reference
`Layr-Labs/labinius-fast-challenge/.github/workflows/benchmark.yml@master`.
Repository labels alone do not prevent another workflow from requesting a
self-hosted runner.

The reusable workflow checks out its trusted harness at `job.workflow_sha`,
the exact revision that supplied the called workflow. The solution repository
and immutable commit are checked out separately. Only regular files and real
directories beneath `solution/` are copied into disposable runner state;
symlinks, special files, `.git` control data, and path escapes are rejected. No
workflow or Yukon script from the solution checkout is executed.

The host must provide the preinstalled, read-only toolchain paths
`/opt/cargo/bin`, `/opt/rustup`, `/usr/bin/gcc-13`, plus `bwrap`, Python 3,
OpenSSL, and standard GNU userland. The trusted verifier and contestant use
separate disposable Cargo homes and target directories. A restricted
`cargo fetch --locked` phase has networking; the subsequent offline compilation
and prover execution have no network. The prover sandbox sees only its binary,
runtime libraries, `/proc`, `/dev`, a fresh `/tmp`, and its proof output
directory.

Bubblewrap reduces what the contestant process can see and write on the host;
it is not a VM reset or a claim of complete isolation. The runner image,
metadata firewall, restrictive bubblewrap/AppArmor policy, non-sudo runner
account, runner-group admission, and post-job disposal/cleanup remain required
layers. Do not place application keys or long-lived secrets in the host
repository or job environment.

On a disposable runner image clone, run the regression test:

```bash
export RUNNER_TEMP
RUNNER_TEMP=$(mktemp -d /var/tmp/labinius-runner-test.XXXXXX)
bash .yukon/tests/test-sandbox.sh
rm -rf "$RUNNER_TEMP"
```

It requires the same toolchain paths and packages as production. It proves that
solution symlinks are rejected, an inherited marker variable is cleared, an
arbitrary host path is not visible, `/opt/cargo/bin` is not writable, and a
legitimate dependency-free Rust crate still builds.

Because the current base includes the generic-modulus prover/verifier change,
run a genuine full benchmark smoke test rather than relying on an older proof:

```bash
export RUNNER_TEMP TRUSTED_CHECKOUT UNTRUSTED_CHECKOUT
export TRUSTED_DIR TRUSTED_CARGO_HOME TRUSTED_TARGET_DIR
export CONTESTANT_DIR CONTESTANT_CARGO_HOME
RUNNER_TEMP=$(mktemp -d /var/tmp/labinius-full-test.XXXXXX)
TRUSTED_CHECKOUT=/path/to/protected-harness-checkout
UNTRUSTED_CHECKOUT=/path/to/solution-input-checkout
git -C "$TRUSTED_CHECKOUT" submodule update --init --recursive
TRUSTED_DIR=$RUNNER_TEMP/labinius-trusted
TRUSTED_CARGO_HOME=$RUNNER_TEMP/labinius-trusted-cargo
TRUSTED_TARGET_DIR=$RUNNER_TEMP/labinius-trusted-target
CONTESTANT_DIR=$RUNNER_TEMP/labinius-contestant
CONTESTANT_CARGO_HOME=$RUNNER_TEMP/labinius-contestant-cargo
bash "$TRUSTED_CHECKOUT/.yukon/setup.sh"
bash "$TRUSTED_CHECKOUT/.yukon/benchmark.sh"
test -s "$TRUSTED_CHECKOUT/.yukon/labinius-fast-score.json"
rm -rf "$RUNNER_TEMP"
```
