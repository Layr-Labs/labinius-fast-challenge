# Dedicated runner security boundary

The benchmark job targets organization runner group
`labinius-intel-8488c-avx512-c7i` and compatibility label
`labinius-avx512-c7i`. The group is an external admission boundary:
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
separate disposable Cargo homes and target directories. Native build systems may
write only within the validated private staged solution under `RUNNER_TEMP`; the
original solution checkout, trusted checkout, and host toolchain remain outside
that writable tree. A restricted `cargo fetch --locked` phase has networking;
the subsequent offline compilation and prover execution have no network. The
prover sandbox sees only its binary,
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
arbitrary host path is not visible, proof FIFOs/symlinks are rejected,
`/opt/cargo/bin` is not writable, a native build can write only in the private
staged solution, and a legitimate dependency-free Rust crate still builds.

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

## Disposable-clone validation

The implementation through commit
`84644416716c75b5db9e8f95bcadb160f8ddcb9b` was validated on 2026-10-09 using
a disposable `c7i.2xlarge` clone of private AMI
`ami-05be5ff6eebeffbd7` in account `089601768650`, with an Intel Xeon Platinum
8488C, 8 CPUs, and 16 GiB RAM. The source archive SHA-256 was
`26b9cfb8d20d5e4c68ad3aa0cdbd4a130a5cd4689388b355d078b08f16f2b82d`.

Observed results from that one run:

- sandbox regression, setup, and benchmark all exited 0;
- score/proving time: `2972.2717100000295 ms`;
- proof size: `13,958,612` bytes; modulus: `55144219393` (36 bits);
- benchmark wall time: `5.05 s`; peak RSS: `540,496 KiB`;
- reported `avx512f: true`, `nproc: 8`, and `MemTotal: 16,076,996 KiB`;
- the temporary encrypted S3 transfer was deleted before untrusted execution,
  then the clone and its EBS volume were deleted with no cleanup errors.

This is a functional validation and one performance observation, not a stable
performance baseline or proof of complete isolation. It did not test GitHub
runner-group admission: the group must remain closed until its selected-workflow
restriction explicitly allows the protected reusable workflow reference. The
routing/docs commit after the tested implementation changes no sandbox, setup,
benchmark, verifier, or solution code.
