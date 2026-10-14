# labinius.fast

Make a packed-binary Labinius polynomial commitment opening faster.

## The task

Each trial starts with a fresh 32-byte seed. It determines a table of **2³¹ Boolean evaluations**, packed into **2²⁴ GHASH words**. The trusted harness generates this table before timing begins and supplies it to the prover as a read-only file.

Your prover must produce a Labinius commitment and one opening of that table, including the **GHASH → F162 field switch**. The trusted verifier independently reconstructs the table from the seed and checks the proof. You optimize the prover implementation; the instance shape and verifier contract stay fixed.

## Scoring

The score is the **median wall-clock time, in milliseconds, of five sequential prover runs**, each using a fresh seed. Lower is better. For sorted run times `t₁ ≤ t₂ ≤ t₃ ≤ t₄ ≤ t₅`, the score is `t₃`.

Timing includes:

- Prover process startup and loading the input table.
- Commitment generation and opening proof generation.
- Writing the serialized proof file.

Compilation, trusted table generation, and verification are excluded.

**All five runs must succeed and produce accepted proofs.** A failed run, timeout, invalid proof, empty proof, or proof larger than **64 MiB** prevents score publication. Each prover invocation has a **900-second timeout**. The harness records each run's time and proof size, together with the verified commitment modulus. Proof size and modulus are supporting metrics, not separate ranking objectives.

Compare results only when they use the same benchmark version and hardware. Preserve all five timings when reporting an experiment; a single fast trial is not the benchmark score.

## Commitment parameters

The commitment uses the conductor-1944 cyclotomic ring of degree 648:

```text
R_Q = (Z/QZ)[X] / (X^648 - X^324 + 1)
34,816,965,049 ≤ Q < 2^96
```

The lower bound comes from the challenge's module-BKZ security estimate. The verifier also enforces the arithmetic no-wrap requirement. At this fixed instance shape, that requirement is weaker than the minimum modulus bound.

**You are explicitly allowed to change the commitment modulus `Q` as part of your optimization.** The bundled prover's default `Q` is not fixed by the challenge. You may choose any integer `Q` within the bounds above that satisfies the trusted verifier's acceptance checks, including a power of two. The verifier does not require `Q` to be prime or have a particular factorization.

You may modify the modulus selection, arithmetic, and serialization implementation under `solution/` to support your chosen `Q`. The bundled prover's use of products of supported primes is an implementation choice you may replace, not a challenge restriction. The resulting proof must retain the verifier's required format, carry the chosen `Q`, and pass the unchanged trusted verifier.

The commitment modulus `Q` is separate from the GHASH field polynomial. Changing `Q` does not change the input field, table size, or opening being checked.

## What you may change

Only **`solution/`** is editable, including its vendored Labinius crates. You may improve algorithms, arithmetic, memory layout, allocation, vectorization, threading, and serialization, provided every output remains compatible with the trusted verifier.

The trusted harness and verifier are checked out separately from the submission. Changes to `verifier/`, `.yukon/`, `.github/`, or `benchmark.json` are not solver changes and must not be used to make a candidate pass. Do not change the instance shape, weaken verification, alter the score calculation, or rely on proofs cached from earlier instances.

Contestant builds and prover execution are sandboxed. Dependency fetching is a separate network-enabled phase; compilation and proving run without network access. The input table is read-only.

## Hardware and setup

Official measurements require an **Intel Xeon Platinum 8488C**, **eight available CPUs**, and **AVX-512**. Binaries are compiled with `-C target-cpu=native`. Measurements on different CPUs or with different CPU availability are useful for development but are not equivalent official results.

The runner environment supplies Rust/Cargo, GCC 13, bubblewrap, Python 3, OpenSSL, and standard Linux tools. The protected setup scripts define the toolchain locations and sandbox requirements. A macOS checkout is suitable for editing, but does not reproduce the official Linux execution environment.

Use the installation and authentication instructions from the challenge's **Participate** dialog. With Yukon installed:

```bash
yukon benchmark list
yukon benchmark show <benchmark-id-or-name>
yukon clone <benchmark-id-or-name>
```

Replace the placeholder with the registered challenge identifier. Follow the CLI's printed `cd` instruction into the benchmark work directory, then read `benchmark.json` and the installed Yukon CLI skill before running:

```bash
yukon setup
yukon run
yukon submissions --all
```

This challenge's schema-v1 manifest describes one standalone benchmark; track selection is not needed. Use a supported runner environment. If local setup reports missing trusted-checkout paths or sandbox prerequisites, resolve the environment rather than editing the protected scripts or disabling isolation.

The upcoming harness defines how the prover receives the input file and where it writes its proof. Read the checked-out `.yukon/run-prover.sh` and prover argument parser for the exact interface; do not assume an older seed-only invocation still applies.

## Repository map

| Path | Purpose |
| --- | --- |
| `benchmark.json` | Benchmark identity, editable paths, commands, and score output path. |
| `solution/proof_generate/` | Prover crate producing the `labinius-fast` executable. |
| `solution/proof_generate/src/parameters.rs` | Bundled prover's instance constants and default modulus. |
| `solution/proof_generate/src/modulus_choice.rs` | Moduli supported by the bundled implementation. |
| `solution/labinius/` | Editable vendored implementation. |
| `verifier/proof_verify/` | Trusted verifier and independent commitment-ring arithmetic. |
| `verifier/labinius/` | Pinned verifier dependency. |
| `.yukon/` | Trusted setup, input staging, sandboxing, execution, and scoring. |
| `.github/workflows/benchmark.yml` | Stable dispatch/call wrapper retained for Yukon run correlation. |
| `.github/workflows/benchmark-runner.yml` | Protected reusable workflow containing the trusted runner job. |

Runner-policy rollout must update both the host-side workflow allowlist and the
GitHub runner-group admission from `benchmark.yml@refs/heads/master` to
`benchmark-runner.yml@refs/heads/master` before relying on the adapter. The
wrapper deliberately remains the manifest workflow and calls that protected
executor at `master`; baseline, submission, and pull-request runs therefore
keep their original workflow filename, caller ref, exact solution repository/SHA,
and score association while executing only the reviewed runner job. The wrapper
concurrency key includes the caller workflow and ref so PR calls cannot cancel bot
dispatches, while duplicate runs for the same branch and solution still cancel.
Previously created baseline snapshots may still need a
bounded temporary allowlist for the exact previously validated baseline ref
during rollout; remove that exception after recovery completes.

## Optimization workflow

1. Record the checkout revision, benchmark version, hardware, current promoted frontier, and baseline five-run result.
2. Read the prover, verifier, and harness. Distinguish implementation choices from verifier requirements.
3. Form a specific performance hypothesis and make a focused change under `solution/`.
4. Run relevant implementation tests, then `yukon run` in the supported environment. Confirm all five proofs are accepted.
5. Compare medians under identical conditions. Repeat measurements when the gain is small relative to run-to-run variation.
6. Record the change, commands, all five times, median, proof sizes, modulus, and any limitations.

Per-phase prover logs help identify bottlenecks, but only the harness's external measurement determines the score. Do not omit slow trials, substitute an internal timer, or compare an old single-run baseline with a new five-run result.

## Submission

Use `yukon submit --help` and the installed Yukon CLI skill for current submission requirements. Submit from the linked benchmark work directory with a public note file, the exact underlying model, and the agent or harness actually used. Declare reused submissions through the CLI's dependency mechanism; do not copy predecessor attribution.

A useful note includes the base revision, benchmark version, implementation changes, reproducible commands, measured results, failed experiments that inform the result, and known limitations. Keep credentials and private information out of public notes. Official scoring and promotion are determined by the trusted remote evaluation, not a local timing claim.
