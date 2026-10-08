//! Fixed checker for one packed-binary PCS instance.
//!
//! The table is 2^31 bits, packed into 2^24 GHASH words. The prover commits to
//! that table, evaluates it at a GHASH point, and switches the claim into a
//! single F162 opening. This binary rebuilds the table from `LABINIUS_SEED`,
//! and checks the switch and the clear opening. Verifier time is not scored.
//!
//! The commitment modulus is whatever `u128` the proof names, subject to size alone. The
//! opening check is therefore recomputed in [`rq`] — schoolbook arithmetic in
//! `R_Q = Z_Q[X]/(X^648 - X^324 + 1)` — rather than through the library's limb kernels, which
//! instantiate their arithmetic per prime at compile time. Only the F162 side of the proof
//! still runs on the library.
//!
//! `--prove` is a reference prover, generic in the modulus and slow; see [`reference`].
use bincode::Options;
use labinius::challenge::{sample_short_challenge, ShortChallenge, DEFAULT_BOUND, DEFAULT_WEIGHT};
use labinius::eval::eq_table;
use labinius::fields::{
    crossfield as cf,
    scalar::{B128, F162},
};
use labinius::params::N;
use labinius::ring::Representation;
use labinius::scheme::FoldingSource;
use labinius::{wire, FoldedWitness, RingElement648, Transcript};
use serde::{Deserialize, Serialize};

mod reference;
mod rq;

/// Boolean table length. A solution cannot change this.
const LOG_VARS: usize = 31;
/// Packed GHASH words. Each word holds `2^LOG_PACK` bits, and `LOG_PACK` is 7.
const LOG_PACKED: usize = LOG_VARS - labinius::fields::crossfield::LOG_PACK;
/// Column variables of the PCS witness. The row length is `LOG_PACKED - COLUMN_LOG`.
const COLUMN_LOG: u32 = 10;
/// Columns of the witness, which is also the number of folding challenges and of commitment
/// columns.
const COLUMNS: usize = 1 << COLUMN_LOG;
/// Row variables: the index inside one column.
const ROW_LOG: usize = LOG_PACKED - COLUMN_LOG as usize;
/// One column in `F162` elements.
const COLUMN_LEN: usize = 1 << ROW_LOG;
/// Ring elements of one column, and so of the fold: four `F162` make one element of `R_648`.
const RING_ELEMENTS: usize = COLUMN_LEN / 4;
/// Reject proofs larger than this before deserialization. Matches the scorer.
const LIMIT: u64 = 64 * 1024 * 1024;

/// Smallest modulus reaching 100 bits of module-BKZ security at this instance's shape:
/// ring degree 648, clear opening, length bound 8*12*sqrt(3)*sqrt(2^22 * 648 * 28 * 3.85) = 9.00e7.
/// 34816965048 gives 99.6691 bits and 34816965049 gives 100.2194; no modulus takes a value
/// between the two, so the threshold is one integer wide.
const MIN_MODULUS: u128 = 34_816_965_049;
/// The threshold above is a function of the shape, so it only holds for this one.
const _: () = assert!(LOG_VARS == 31 && COLUMN_LOG == 10);

/// The cap on `‖v‖^2` the opening check applies, the library's `Params::fold_cap()` at this
/// shape: `FOLD_CAP` per coefficient over the `witness_len / 4 * N` coefficients of the fold.
/// The fold has `witness_len / (4 * columns)` ring elements and each coefficient is a sum of
/// `28 * columns` signed terms, so the product does not depend on `COLUMN_LOG`.
fn fold_cap() -> u64 {
    (labinius::recursion::FOLD_CAP * ((1usize << LOG_PACKED) / 4 * rq::N) as f64).ceil() as u64
}

/// The largest coefficient the `‖v‖^2` cap leaves room for, `floor(sqrt(fold_cap))`.
fn fold_cap_bound() -> u128 {
    fold_cap().isqrt() as u128
}

/// The proof's modulus passes on size alone: no primality, no factorisation, no smoothness, and
/// a power of two is as welcome as anything else.
///
/// Two conditions. `MIN_MODULUS` is the module-BKZ threshold above. The second is the no-wrap
/// condition that makes the mod-`Q` identity `A v = sum_j c_j C_j` an identity over the
/// integers: the norm check accepts an integer fold with `‖v‖^2 <= fold_cap()`, so every
/// coefficient is at most `fold_cap_bound()` in absolute value and two admissible folds differ
/// coefficientwise by at most twice that. Once `Q` exceeds `2 * fold_cap_bound()` a centered
/// residue class holds at most one admissible integer vector, and the extracted `v` is the one
/// the prover folded. At this shape `2 * fold_cap_bound()` is 1083574, four orders of magnitude
/// below `MIN_MODULUS`, so the condition states the intent rather than filtering anything.
///
/// [`rq::MAX_MODULUS`] is the third, and is an artifact of the checker's `i128` lanes rather
/// than of the scheme; it sits above the product of all seven of the library's primes.
fn accept_modulus(modulus: u128) -> Result<u128, String> {
    if modulus < MIN_MODULUS {
        return Err("commitment modulus is below the security bound".into());
    }
    if modulus <= 2 * fold_cap_bound() {
        return Err("commitment modulus does not clear the no-wrap bound".into());
    }
    if modulus >= rq::MAX_MODULUS {
        return Err("commitment modulus is above what the checker's arithmetic covers".into());
    }
    Ok(modulus)
}

/// The bytes the solution prover must write. Field order is the bincode layout.
#[derive(Serialize, Deserialize)]
struct Proof {
    version: u32,
    /// The commitment modulus. Any `u128` the size conditions of [`accept_modulus`] admit.
    modulus: u128,
    /// Commitment of the seed's table, not a prover-chosen table: the `COLUMNS` columns as
    /// coefficients modulo the modulus, in [`rq::pack_ring_elements`].
    commitment: Vec<u8>,
    /// One GHASH partial per bit of a packed word (128).
    partials: Vec<u128>,
    /// One sumcheck message per packed coordinate (24). Each message is two F162 elements.
    rounds: Vec<[[u64; 3]; 2]>,
    /// The F162 value the switch reduces the GHASH claim to.
    opened: [u64; 3],
    /// Row evaluation at the column point, in the shared wire codec.
    row: Vec<u8>,
    /// Folded witness, one fixed-width little-endian integer per coefficient.
    folded: Vec<u8>,
}
/// Bincode layout shared with the solution prover: fixed-width integers, no trailing bytes.
fn codec() -> impl Options {
    bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .with_limit(LIMIT)
        .reject_trailing_bytes()
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
/// Fiat–Shamir state. Absorbing the seed, the shape, the modulus, and the commitment binds
/// every later challenge to this instance.
fn transcript(seed: &[u8; 32], modulus: u128, commitment: &[u8]) -> Transcript {
    let mut t = Transcript::new(b"labinius-fast/v1");
    t.absorb_bytes(seed);
    t.absorb_u64(LOG_VARS as u64);
    t.absorb_u64(COLUMN_LOG as u64);
    t.absorb_bytes(&modulus.to_le_bytes());
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
/// Bytes one coefficient of the folded witness takes on the wire: a little-endian signed
/// integer per coefficient, element by element and then coefficient by coefficient.
///
/// A fold coefficient is a sum over the `columns` columns of a weight-28 signed challenge
/// against a binary witness, so it lies in `[-28 * 1024, 28 * 1024]` and even an `i16` would
/// hold it. Four bytes is margin on a proof whose size the scorer records rather than scores.
const FOLDED_BYTES: usize = 4;
fn pack_folded(folded: &FoldedWitness) -> Vec<u8> {
    let elements = folded.elements();
    assert!(
        elements
            .iter()
            .all(|e| e.representation == Representation::Coefficients),
        "a fold is in coefficient form"
    );
    let mut out = Vec::with_capacity(FOLDED_BYTES * N * elements.len());
    for e in elements {
        for &c in &e.v {
            out.extend_from_slice(&(c as i32).to_le_bytes());
        }
    }
    out
}
/// The inverse of [`pack_folded`] for a fold of `elements` ring elements, strict: one fold has
/// exactly one encoding, and an accepted byte string is the encoding of the fold it yields.
///
/// Strictness is what keeps the transcript honest. The transcript absorbs the commitment bytes
/// rather than the commitment object, so an encoding with slack would let a prover re-encode one
/// proof many ways and keep whichever encoding drew the challenges it liked. The length here is
/// exact, every word must be a coefficient the ring can hold, and `reject_trailing_bytes` on the
/// codec refuses bytes after the last field — bytes hidden inside this field are refused by the
/// length instead. A fixed width and those checks make the encoding a bijection, so comparing a
/// re-serialization against the proof would be a check that can never fire.
fn unpack_folded(bytes: &[u8], elements: usize) -> Result<FoldedWitness, String> {
    if bytes.len() != FOLDED_BYTES * N * elements {
        return Err("folded witness has the wrong length".into());
    }
    let mut words = bytes.chunks_exact(FOLDED_BYTES);
    let mut fold = Vec::with_capacity(elements);
    for _ in 0..elements {
        let mut e = RingElement648::zero(Representation::Coefficients);
        for slot in e.v.iter_mut() {
            let word = i32::from_le_bytes(words.next().unwrap().try_into().unwrap());
            *slot = i16::try_from(word).map_err(|_| "coefficient out of range".to_string())?;
        }
        fold.push(e);
    }
    Ok(FoldedWitness::of(fold))
}
/// An inner product over `F162`. The library's own kernel for this is crate-private, and the
/// longest one here is 2^14 terms.
fn dot_f162(a: &[F162], b: &[F162]) -> F162 {
    assert_eq!(a.len(), b.len(), "the two operands have different lengths");
    a.iter()
        .zip(b)
        .fold(F162::ZERO, |acc, (&x, &y)| acc + x * y)
}
/// The evaluation point split the way the witness is: `p0` over the row variables, `p1` over the
/// column variables. The sumcheck hands its coordinates back most significant first, and the
/// flat witness index carries the row below the column, so the leading `COLUMN_LOG` coordinates
/// are `p1` reversed and the trailing `ROW_LOG` are `p0` reversed.
fn split_point(coords: &[F162]) -> (Vec<F162>, Vec<F162>) {
    assert_eq!(coords.len(), LOG_PACKED, "one coordinate per packed variable");
    let (high, low) = coords.split_at(COLUMN_LOG as usize);
    (
        low.iter().rev().copied().collect(),
        high.iter().rev().copied().collect(),
    )
}
/// The `4 * v.len()` field elements of `v mod 2`, in the witness's own index order: element
/// `4 m + k` is component `k` of packed ring element `m`, and its bit `p` is the parity of
/// coefficient `4 p + k` of that element.
fn components_mod_2(v: &[RingElement648]) -> Vec<F162> {
    let mut out = vec![F162::ZERO; 4 * v.len()];
    for (m, e) in v.iter().enumerate() {
        for p in 0..rq::N / 4 {
            for k in 0..4 {
                if e.v[4 * p + k] & 1 != 0 {
                    out[4 * m + k].0[p >> 6] |= 1u64 << (p & 63);
                }
            }
        }
    }
    out
}
/// The `COLUMNS` folding challenges: the row evaluation absorbed, then one weight-28 signed
/// subring element per column, sampled exactly as the library's `FoldingChallenges::derive`
/// does.
fn folding_challenges(
    transcript: &mut Transcript,
    row: &labinius::RowEvaluation,
) -> Vec<ShortChallenge> {
    row.absorb(transcript);
    (0..COLUMNS)
        .map(|_| sample_short_challenge(transcript, DEFAULT_WEIGHT, DEFAULT_BOUND).0)
        .collect()
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

/// Clear opening of `opened` at the sumcheck point, recomputed in `R_Q`.
///
/// The row dotted with `eq` on the column point must equal `opened`. Folding challenges then
/// come from the transcript and that row. The fold must be short — `‖v‖^2` within the cap and
/// every coefficient within `(Q-1)/2`, both over the integers — it must satisfy
/// `A v = sum_j c_j C_j` in `R_Q`, and it must match the folded row as bits over `F162`.
fn check_opening(
    modulus: u128,
    transcript: &mut Transcript,
    opened: F162,
    coords: &[F162],
    proof: &Proof,
) {
    let (p0, p1) = split_point(coords);
    let row = wire::unpack_row_evaluation(&proof.row, COLUMNS).expect("malformed row evaluation");
    let folded = unpack_folded(&proof.folded, RING_ELEMENTS).expect("malformed folded witness");
    let challenges = folding_challenges(transcript, &row);
    assert_eq!(
        dot_f162(&eq_table(&p1), row.values()),
        opened,
        "the row evaluation does not reduce to the opened value"
    );

    let v = folded.elements();
    assert_eq!(v.len(), RING_ELEMENTS, "the fold is one column of R_648");
    let mut normsq = 0u64;
    let mut worst = 0u128;
    for e in v {
        assert_eq!(
            e.representation,
            labinius::ring::Representation::Coefficients,
            "the fold is not in coefficient form"
        );
        for &c in &e.v {
            let magnitude = (c as i32).unsigned_abs();
            normsq += u64::from(magnitude) * u64::from(magnitude);
            worst = worst.max(u128::from(magnitude));
        }
    }
    assert!(normsq <= fold_cap(), "the fold is longer than the cap");
    assert!(worst <= (modulus - 1) / 2, "the fold is not centered");

    let columns = rq::unpack_ring_elements(&proof.commitment, COLUMNS * rq::N, modulus)
        .expect("malformed commitment");
    let matrix = rq::ajtai_matrix(modulus, RING_ELEMENTS);
    let coefficients: Vec<i16> = v.iter().flat_map(|e| e.v).collect();
    assert_eq!(
        rq::a_times_v(&matrix, &coefficients, modulus),
        rq::challenge_combination(&columns, &challenges, modulus),
        "the fold does not open the folded commitment"
    );

    let parities: Vec<F162> = challenges.iter().map(|c| c.to_f162()).collect();
    assert_eq!(
        dot_f162(&eq_table(&p0), &components_mod_2(v)),
        dot_f162(&parities, row.values()),
        "the fold does not match the folded row evaluation"
    );
}

/// Accept `proof` only if it opens `trace(seed)` under the modulus it carries.
///
/// The commitment is taken as given rather than recomputed. The instance is pinned by the
/// claim: `check_field_switch` derives the GHASH evaluation from the seed's table, so the
/// sumcheck reduces the true claim, and the opening then forces the committed witness to
/// agree with that table at a point drawn after the commitment was absorbed.
fn verify(seed: &[u8; 32], proof: &Proof) {
    check_proof_shape(proof);
    let modulus = accept_modulus(proof.modulus).expect("invalid commitment modulus");
    let values = trace(seed);
    let mut transcript = transcript(seed, modulus, &proof.commitment);
    let (opened, coords) = check_field_switch(&mut transcript, &values, proof);
    check_opening(modulus, &mut transcript, opened, &coords, proof);
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert!(
        args.len() == 3 || args.len() == 5,
        "usage: labinius-fast-verify --prove FILE --modulus Q | --verify FILE"
    );
    let seed = seed();
    match args[1].as_str() {
        "--prove" => {
            assert_eq!(args.len(), 5, "the reference prover takes --modulus");
            assert_eq!(args[3], "--modulus", "unknown prover option");
            let modulus: u128 = args[4].trim().parse().expect("invalid modulus");
            accept_modulus(modulus).expect("invalid commitment modulus");
            let proof = reference::prove(&seed, modulus);
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
            // size conditions, then check the opening under that schema.
            let modulus = accept_modulus(proof.modulus).expect("invalid commitment modulus");
            verify(&seed, &proof);
            println!(
                "{{\"modulus\":\"{modulus}\",\"modulusBits\":{},\"polynomialSize\":{},\"packedElements\":{},\"field\":\"GHASH\"}}",
                rq::modulus_bits(modulus),
                // `1 << LOG_VARS` would default to i32 and overflow at 31 bits.
                1usize << LOG_VARS,
                1usize << LOG_PACKED
            );
        }
        _ => panic!("unknown command"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use labinius::ring::Modulus;
    use labinius::scheme::{Opening, Params};

    /// The cap the checker applies is the library's, at this shape.
    #[test]
    fn fold_cap_matches_the_library() {
        let params = Params::with_base(
            LOG_PACKED as u32,
            COLUMN_LOG,
            Modulus::BASE,
            vec![],
            Opening::Clear,
        )
        .unwrap();
        assert_eq!(fold_cap(), params.fold_cap());
        assert_eq!(fold_cap(), 293_534_171_136);
        assert_eq!(fold_cap_bound(), 541_787);
    }

    /// The gate takes the security threshold and nothing structural: a power of two above it
    /// passes, the integer below it does not, and the no-wrap bound is slack.
    #[test]
    fn the_gate_is_two_size_conditions() {
        assert!(accept_modulus(1u128 << 36).is_ok());
        assert!(accept_modulus(MIN_MODULUS).is_ok());
        assert!(accept_modulus(MIN_MODULUS - 1).is_err());
        assert!(accept_modulus(55_144_219_393).is_ok());
        assert!(accept_modulus(rq::MAX_MODULUS).is_err());
        assert!(2 * fold_cap_bound() < MIN_MODULUS);
    }
}
