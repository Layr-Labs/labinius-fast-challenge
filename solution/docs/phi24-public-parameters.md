# Fixed public Phi24 parameters

The packed file `labinius/crates/pcs/src/simd/phi24-digits6.bin` contains only an encoding of the fixed public Ajtai matrix. It never contains a witness table, a commitment, an opening, or an instance seed. The matrix comes from the unchanged public seed in `proof_generate/src/rq.rs`, with Q=35039150080 and4096 ring elements of648 coefficients.

Let M=55*65536 and Y=X^81. Then Y^8-Y^4+1=0. The six weights are the existing numeric pairs interpreted as u+vY. Each octet has an exact48-dimensional modular-kernel encoding. The reduced integer basis, weighted Gram-Schmidt data and deterministic beam encoder are in `phi24_lattice_core.rs`. The fixed balancing scales are1016,1012,999,1009,994,971; beam width4096, generated-state budget4000000. An index-derived BLAKE3 rotation by a power of Y distributes octet directions and is reversed before storage. None of these values depend on the fresh input table.

Data order is bank, ring element, coefficient. Each signed coefficient uses six little-endian packed bits (two's complement); the shipped maximum absolute coefficient is16. Packed SHA256: `a3017efcd78c7f6f9a7142a32dbd763314d2e01e8812edd6a51f1752a49651b1`. The decoded signed-byte image SHA256 is `c9505482745d2f059ca608f7880a1dd076bd31b19f9ddceab810fb9256232257`. Public key construction verifies every coefficient congruence modulo M. Its ordinary P=9721 stream supplies the other CRT residue. For other matrices, the codec performs an actual generic lattice encoding and checks the result.

The asset can be regenerated offline from source. From `solution/`, set `RUSTFLAGS="-C target-cpu=native"` and run:

```sh
cargo run --locked --offline --release --manifest-path proof_generate/Cargo.toml --bin labinius-rebuild-phi24-public -- regenerated.bin
```

Add `--smoke` before the output filename to encode four rings. Full regeneration is expensive; the generator is never invoked by the prover or timed harness. The published asset was generated with Rust1.99.1 on the benchmark Xeon8488C. Output equality across different floating-point compiler implementations should be checked; every encoding must independently pass congruence, range, moment and proof validation.

The public first/second moments and7.75-sigma recovery certificate are recomputed at key build. Exact per-coefficient source-word bounds are proved from the actual assembly and all possible prefix lookup domains, then checked against every public MAC stripe. The class20 bitmap uses20u64 words per public polynomial, four bit classes interleaved by word index. Its variable shifts and phase permutation consume the original input words without a GFNI transpose. The bit63 carry case is handled explicitly.

A matching parity short-circuits a correction only under the guarded existing recovery model. Mismatches, noncentered residues and cuts below1024 retain the full exact dot. The included rational proof bounds additional outside-window even-wrap misses below2^-85 under the same independent-Boolean model; this is not an unconditional bound for adversarial tables. Trusted verification, transcript, instance shape and proof encoding are unchanged. Performance must be established with the complete unchanged five-run harness.
