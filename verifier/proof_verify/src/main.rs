//! Fixed checker for one packed-binary PCS instance.
//!
//! The table is 2^31 bits, packed into 2^24 GHASH words. The prover commits to
//! that table, evaluates it at a GHASH point, and switches the claim into a
//! single F162 opening. This binary rebuilds the table from `LABINIUS_SEED`,
//! recomputes the commitment, and checks the switch and the clear opening.
//! Verifier time is not scored.
//!
//! `--prove` is a reference prover used by the round-trip tests. The scored
//! prover is `solution/proof_generate`, which must emit this same proof layout.
use bincode::Options;
use labinius::fields::{
    crossfield as cf,
    scalar::{B128, F162},
};
use labinius::ring::Modulus;
use labinius::scheme::*;
use labinius::{wire, Transcript};
use serde::{Deserialize, Serialize};

/// Boolean table length. A solution cannot change this.
const LOG_VARS: usize = 31;
/// Packed GHASH words. Each word holds `2^LOG_PACK` bits, and `LOG_PACK` is 7.
const LOG_PACKED: usize = LOG_VARS - labinius::fields::crossfield::LOG_PACK;
/// Column variables of the PCS witness. The row length is `LOG_PACKED - COLUMN_LOG`.
const COLUMN_LOG: u32 = 10;
/// Reject proofs larger than this before deserialization. Matches the scorer.
const LIMIT: u64 = 64 * 1024 * 1024;

/// Smallest modulus reaching 100 bits of module-BKZ security at this instance's shape:
/// ring degree 648, clear opening, length bound 8*12*sqrt(3)*sqrt(2^22 * 648 * 28 * 3.85) = 9.00e7.
/// 34816965048 gives 99.6691 bits and 34816965049 gives 100.2194; no modulus takes a value
/// between the two, so the threshold is one integer wide.
const MIN_MODULUS: u128 = 34_816_965_049;
/// The threshold above is a function of the shape, so it only holds for this one.
const _: () = assert!(LOG_VARS == 31 && COLUMN_LOG == 10);

/// Product of the ordered factors carried by the proof.
fn commitment_modulus(primes: &[u16]) -> Result<u128, String> {
    if primes.is_empty() {
        return Err("proof carries no commitment modulus".into());
    }
    let mut product = 1u128;
    for (i, q) in primes.iter().enumerate() {
        if primes[..i].contains(q) {
            return Err("duplicate commitment prime".into());
        }
        product = product
            .checked_mul(u128::from(*q))
            .ok_or_else(|| "modulus overflow".to_string())?;
    }
    Ok(product)
}

/// The proof's modulus passes when it is at least `MIN_MODULUS`.
fn accept_modulus(primes: &[u16]) -> Result<u128, String> {
    let product = commitment_modulus(primes)?;
    if product < MIN_MODULUS {
        return Err("commitment modulus is below the security bound".into());
    }
    Ok(product)
}

fn parse_primes(value: &str) -> Result<Vec<u16>, String> {
    let primes: Vec<u16> = value
        .split(',')
        .map(|s| s.trim().parse().map_err(|_| "invalid prime list".to_string()))
        .collect::<Result<_, _>>()?;
    accept_modulus(&primes)?;
    Ok(primes)
}

/// The bytes the solution prover must write. Field order is the bincode layout.
#[derive(Serialize, Deserialize)]
struct Proof {
    version: u32,
    /// Ordered commitment-prime factors. The first is the base limb.
    primes: Vec<u16>,
    /// Commitment of the seed's table, not a prover-chosen table.
    commitment: Vec<u8>,
    /// One GHASH partial per bit of a packed word (128).
    partials: Vec<u128>,
    /// One sumcheck message per packed coordinate (24). Each message is two F162 elements.
    rounds: Vec<[[u64; 3]; 2]>,
    /// The F162 value the switch reduces the GHASH claim to.
    opened: [u64; 3],
    /// Row evaluation at the column point, in the shared wire codec.
    row: Vec<u8>,
    /// Folded witness, entropy-coded by the shared wire codec.
    folded: Vec<u8>,
}
/// Bincode layout shared with the solution prover: fixed-width integers, no trailing bytes.
fn codec() -> impl Options {
    bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .with_limit(LIMIT)
        .reject_trailing_bytes()
}
/// Public parameters for `primes`. Shape and clear opening stay fixed; only the modulus varies.
fn params(primes: &[u16]) -> Params {
    accept_modulus(primes).expect("invalid commitment modulus");
    let ms: Vec<_> = primes
        .iter()
        .map(|q| {
            Modulus::ALL
                .into_iter()
                .find(|m| m.prime() == *q)
                .expect("commitment modulus has no opening arithmetic")
        })
        .collect();
    Params::with_base(
        LOG_PACKED as u32,
        COLUMN_LOG,
        ms[0],
        ms[1..].to_vec(),
        Opening::Clear,
    )
    .unwrap()
}
fn seed() -> [u8; 32] {
    let s = std::env::var("LABINIUS_SEED").expect("LABINIUS_SEED is required");
    assert_eq!(s.len(), 64);
    core::array::from_fn(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).expect("invalid seed"))
}
/// The instance table: a BLAKE3 XOF keyed by the seed. Both sides derive it, so the
/// prover does not get to choose the polynomial.
fn trace(seed: &[u8; 32]) -> Vec<B128> {
    let mut h = blake3::Hasher::new_keyed(seed);
    h.update(b"labinius-fast/binary-table/v1");
    let mut b = vec![0; (1 << LOG_VARS) / 8];
    h.finalize_xof().fill(&mut b);
    b.chunks_exact(16)
        .map(|v| B128(u128::from_le_bytes(v.try_into().unwrap())))
        .collect()
}
/// Fiat–Shamir state. Absorbing the seed, the shape, the ordered primes, and the
/// commitment binds every later challenge to this instance.
fn transcript(seed: &[u8; 32], primes: &[u16], commitment: &[u8]) -> Transcript {
    let mut t = Transcript::new(b"labinius-fast/v1");
    t.absorb_bytes(seed);
    t.absorb_u64(LOG_VARS as u64);
    t.absorb_u64(COLUMN_LOG as u64);
    t.absorb_u64(primes.len() as u64);
    for q in primes {
        t.absorb_u64(*q as u64);
    }
    t.absorb_bytes(commitment);
    t
}
/// The GHASH evaluation point. The low 7 coordinates select a bit inside a word;
/// the high `LOG_PACKED` coordinates select which packed word.
fn point(t: &mut Transcript) -> (Vec<B128>, Vec<B128>) {
    (
        t.sample_b128(b"r-lo", 7),
        t.sample_b128(b"r-hi", LOG_PACKED),
    )
}
/// Combine the 128 partial evaluations with `eq` on the low point to get the GHASH claim.
fn claim(partials: &[B128], lo: &[B128]) -> B128 {
    partials
        .iter()
        .zip(cf::eq_expand_b128(lo))
        .fold(B128::ZERO, |a, (&v, e)| a + v * e)
}
/// An F162 element is three limbs, and the top limb has only 34 bits. A larger
/// encoding would be a non-canonical representative of the same field element.
fn checked_f162(x: [u64; 3]) -> F162 {
    assert_eq!(x[2] >> 34, 0, "noncanonical F162 element");
    F162(x)
}
/// Reference proof of `trace(seed)` under `primes`. The transcript messages here
/// are what `verify` expects, in the same order.
fn prove(seed: &[u8; 32], primes: &[u16]) -> Proof {
    let p = params(primes);
    let pp = PublicParameters::from_seed(p.clone(), [0x5a; 32]);
    let mut prover = Prover::new(&pp);
    let verifier = Verifier::new(&pp);
    let values = trace(seed);
    let w = Witness::lifted(&p, &values).unwrap();
    let (commitment, opening) = prover.commit(&w);
    let commitment = commitment.to_bytes();
    let mut t = transcript(seed, primes, &commitment);
    let (lo, hi) = point(&mut t);
    let mut ws = cf::Workspace::new(LOG_PACKED, 1);
    let partials = ws.partial_evals(0, &values, &hi);
    let value = claim(&partials, &lo);
    t.absorb_b128(&[value]);
    t.absorb_b128(&partials);
    let batch = cf::eq_expand_f162(&t.sample_f162(b"switch-batch", 7));
    let mut sw = cf::SwitchProver::from_workspace(ws, &values, &[F162::ONE], &batch);
    let mut rounds = Vec::new();
    let mut coords = Vec::new();
    for _ in 0..LOG_PACKED {
        let msg = sw.msg();
        t.absorb_f162(&msg);
        let r = t.sample_f162(b"switch-round", 1)[0];
        rounds.push(msg.map(|v| v.0));
        coords.push(r);
        sw.fold(r);
    }
    let opened = sw.final_eval();
    t.absorb_f162(&[opened]);
    let point = EvaluationPoint::msb_first(&p, &coords);
    let row = w.row_evaluate(&point);
    let challenges = verifier.derive_folding_challenges(&mut t, &row);
    let folded = prover.fold(opening, &challenges);
    Proof {
        version: 1,
        primes: primes.to_vec(),
        commitment,
        partials: partials.iter().map(|v| v.0).collect(),
        rounds,
        opened: opened.0,
        row: wire::pack_row_evaluation(&row),
        folded: wire::encode(&folded, p.base.prime()),
    }
}
fn check_proof_shape(proof: &Proof) {
    assert_eq!(proof.version, 1);
    assert_eq!(proof.partials.len(), 1 << 7, "128 partial evaluations");
    assert_eq!(
        proof.rounds.len(),
        LOG_PACKED,
        "one switch round per packed coordinate"
    );
}

/// Recompute the commitment of the seed's table. An opening of any other table is rejected
/// here, before the switch or the PCS checks run.
fn commitment_for_table(
    pp: &PublicParameters,
    p: &Params,
    values: &[B128],
    bytes: &[u8],
) -> Commitment {
    let witness = Witness::lifted(p, values).unwrap();
    let expected = Prover::new(pp).commit(&witness).0;
    assert_eq!(
        bytes,
        expected.to_bytes(),
        "commitment does not match instance"
    );
    Commitment::from_bytes(p, bytes).unwrap()
}

/// Check the GHASH claim and the sumcheck that switches it into one F162 value.
///
/// The claim is computed from the seed's table. The proof's partials must recombine
/// to that claim. Each round message is absorbed before its challenge is sampled,
/// and the final check binds `opened` to the high point.
fn check_field_switch(
    transcript: &mut Transcript,
    values: &[B128],
    proof: &Proof,
) -> (F162, Vec<F162>) {
    let (lo, hi) = point(transcript);
    let mut workspace = cf::Workspace::new(LOG_PACKED, 1);
    let trusted = workspace.partial_evals(0, values, &hi);
    let value = claim(&trusted, &lo);
    let partials: Vec<_> = proof.partials.iter().copied().map(B128).collect();
    transcript.absorb_b128(&[value]);
    transcript.absorb_b128(&partials);
    let batch = cf::eq_expand_f162(&transcript.sample_f162(b"switch-batch", 7));
    let mut switch = cf::SwitchVerifier::start(&partials, value, &lo, &batch).unwrap();
    let mut coords = Vec::with_capacity(proof.rounds.len());
    for raw in &proof.rounds {
        let msg = raw.map(checked_f162);
        transcript.absorb_f162(&msg);
        let challenge = transcript.sample_f162(b"switch-round", 1)[0];
        switch.round(msg, challenge);
        coords.push(challenge);
    }
    let opened = checked_f162(proof.opened);
    switch.finish(&hi, &coords, &batch, opened).unwrap();
    transcript.absorb_f162(&[opened]);
    (opened, coords)
}

/// Clear opening of `opened` at the sumcheck point.
///
/// The row dotted with `eq` on the column point must equal `opened`. Folding
/// challenges then come from the transcript and that row. The folded witness must
/// match the folded commitment on every prime and match the folded row as bits.
fn check_opening(
    verifier: &Verifier,
    p: &Params,
    transcript: &mut Transcript,
    commitment: &Commitment,
    opened: F162,
    coords: &[F162],
    proof: &Proof,
) {
    let point = EvaluationPoint::msb_first(p, coords);
    let row = wire::unpack_row_evaluation(&proof.row, p.columns()).unwrap();
    let folded = wire::decode(&proof.folded).unwrap();
    let challenges = verifier.derive_folding_challenges(transcript, &row);
    verifier.verify_evaluation(&point, &opened, &row).unwrap();
    let folded_commitment = verifier.fold_commitment(commitment, &challenges);
    let folded_row_value = verifier.fold_row_evaluation(&row, &challenges);
    verifier
        .verify_opening(
            commitment,
            &challenges,
            &point,
            OpeningMessage::Clear {
                folded_commitment: &folded_commitment,
                folded_witness: &folded,
                folded_row_value: &folded_row_value,
            },
        )
        .unwrap();
}

/// Accept `proof` only if it opens `trace(seed)` under the primes it carries.
fn verify(seed: &[u8; 32], proof: &Proof) {
    check_proof_shape(proof);
    let p = params(&proof.primes);
    let pp = PublicParameters::from_seed(p.clone(), [0x5a; 32]);
    let verifier = Verifier::new(&pp);
    let values = trace(seed);
    let commitment = commitment_for_table(&pp, &p, &values, &proof.commitment);
    let mut transcript = transcript(seed, &proof.primes, &proof.commitment);
    let (opened, coords) = check_field_switch(&mut transcript, &values, proof);
    check_opening(
        &verifier,
        &p,
        &mut transcript,
        &commitment,
        opened,
        &coords,
        proof,
    );
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert!(
        args.len() == 3 || args.len() == 5,
        "usage: labinius-fast-verify --prove FILE --primes q1,q2,... | --verify FILE"
    );
    let seed = seed();
    match args[1].as_str() {
        "--prove" => {
            assert_eq!(args.len(), 5, "the reference prover takes --primes");
            assert_eq!(args[3], "--primes", "unknown prover option");
            let primes = parse_primes(&args[4]).expect("invalid commitment modulus");
            let proof = prove(&seed, &primes);
            std::fs::write(&args[2], codec().serialize(&proof).unwrap()).unwrap();
        }
        "--verify" => {
            assert_eq!(
                args.len(),
                3,
                "verification reads the modulus from the proof"
            );
            assert!(
                std::fs::metadata(&args[2]).unwrap().len() <= LIMIT,
                "proof too large"
            );
            let proof: Proof = codec()
                .deserialize(&std::fs::read(&args[2]).unwrap())
                .unwrap();
            // The modulus is the one this proof was built with. Check it against the
            // threshold, then check the opening under that schema.
            let modulus = accept_modulus(&proof.primes).expect("invalid commitment modulus");
            verify(&seed, &proof);
            println!(
                "{{\"modulus\":\"{modulus}\",\"modulusPrimes\":{:?},\"polynomialSize\":{},\"packedElements\":{},\"field\":\"GHASH\"}}",
                proof.primes,
                // `1 << LOG_VARS` would default to i32 and overflow at 31 bits.
                1usize << LOG_VARS,
                1usize << LOG_PACKED
            );
        }
        _ => panic!("unknown command"),
    }
}
