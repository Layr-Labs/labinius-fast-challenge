//! Compare all four ordinary arithmetic combinations on the same witness.
use super::*;
use crate::ntt::neon::{sms, sms_sat84};
use std::time::{Duration, Instant};

const LABELS: [&str; 4] = ["wide", "sat84", "lag81_wide", "lag81_sat84"];
const ORDER: [[usize; 4]; 4] = [[0, 1, 3, 2], [1, 2, 0, 3], [2, 3, 1, 0], [3, 0, 2, 1]];

struct Matrices {
    wide: sms::SmsMatrix,
    sat84: sms_sat84::SmsMatrix,
}

impl Matrices {
    fn new() -> Self {
        let wide = sms::SmsMatrix::from_images(
            &SMS_DATA.0, &SMS_MEANS.0, &SMS_MU.0, &SMS_META.0, &SMS_DIGITS.0,
            &SMS_MEAN_INT.0, &SMS_THRESHOLD.0, RING_ELEMENTS,
        ).expect("the diagnostic requires the embedded mixed key");
        let sat84 = sms_sat84::SmsMatrix {
            elements: wide.elements,
            data: wide.data.clone(),
            mean_residues: wide.mean_residues.clone(),
            mu: wide.mu.clone(),
            digits: wide.digits.clone(),
            mean_int: wide.mean_int.clone(),
            threshold: wide.threshold.clone(),
            margin: wide.margin,
            column_failure: wide.column_failure,
        };
        assert_eq!(wide.data.as_ptr(), sat84.data.as_ptr());
        assert_eq!(wide.mean_residues.as_ptr(), sat84.mean_residues.as_ptr());
        assert_eq!(wide.mu.as_ptr(), sat84.mu.as_ptr());
        assert_eq!(wide.digits.as_ptr(), sat84.digits.as_ptr());
        assert_eq!(wide.mean_int.as_ptr(), sat84.mean_int.as_ptr());
        assert_eq!(wide.threshold.as_ptr(), sat84.threshold.as_ptr());
        Self { wide, sat84 }
    }
}

type CommitFn = fn(&[B128], &Matrices, u128, &mut [u8]);
type FoldFn = fn(&[B128], &[ShortChallenge]) -> Vec<RingElement648>;

#[inline(never)]
fn commit_wide(values: &[B128], matrices: &Matrices, q: u128, out: &mut [u8]) {
    sms::commit_v::<17>(values, &matrices.wide, threads(), q, out, false);
}

#[inline(never)]
fn commit_sat84(values: &[B128], matrices: &Matrices, q: u128, out: &mut [u8]) {
    sms_sat84::commit_v::<17>(values, &matrices.sat84, threads(), q, out, false);
}

#[inline(never)]
fn fold_lag81(values: &[B128], challenges: &[ShortChallenge]) -> Vec<RingElement648> {
    if challenges.iter().map(|c| c.weight).sum::<usize>() < 1 << 15 {
        fold_neon_lag81(values, challenges)
    } else {
        // The ordinary lag81 variant has the same large-weight fallback.
        fold(values, challenges)
    }
}

fn ns(value: Duration) -> u64 {
    value.as_nanos().try_into().expect("phase exceeds u64 nanoseconds")
}

#[inline(never)]
fn execute(
    seed: &[u8;32], modulus: u128, values: &[B128], matrices: &Matrices,
    mut commitment: Vec<u8>, commit_fn: CommitFn, fold_fn: FoldFn,
) -> (ProofParts, [u64;4], u64) {
    let start = Instant::now();
    let mut lap = start;
    let mut phases = [0;4];
    commit_fn(values, matrices, modulus, &mut commitment);
    labinius::par::pool();
    let now = Instant::now();
    phases[0] = ns(now.duration_since(lap));
    lap = now;
    // No output prefault here: the switch that follows runs on every core from its first
    // round, and idle population workers measurably slowed it. The final encode faults the
    // file pages itself, in parallel.
    let mut t = transcript(seed, modulus, &commitment);
    let lo = t.sample_b128(b"r-lo", 7);
    let hi = t.sample_b128(b"r-hi", LOG_PACKED);
    let (partials, rounds, coords, opened) = switch_proof(values, &lo, &hi, &mut t);
    t.absorb_f162(&[opened]);
    let now = Instant::now();
    phases[1] = ns(now.duration_since(lap));
    lap = now;
    let (p0, _) = split_point(&coords);
    let row = RowEvaluation::of(row_evaluate(values, &p0));
    let now = Instant::now();
    phases[2] = ns(now.duration_since(lap));
    lap = now;
    let challenges = folding_challenges(&mut t, &row);
    let folded = FoldedWitness::of(fold_fn(values, &challenges));
    let now = Instant::now();
    phases[3] = ns(now.duration_since(lap));
    let proof = ProofParts {
        modulus,
        commitment,
        partials: partials.iter().map(|v| v.0).collect(),
        rounds,
        opened: opened.0,
        row: wire::pack_row_evaluation(&row),
        folded,
    };
    (proof, phases, ns(start.elapsed()))
}

#[allow(clippy::too_many_arguments)]
fn sample(
    route: usize, class: &str, round: usize, position: usize, seed: &[u8;32],
    modulus: u128, values: &[B128], matrices: &Matrices, expected: Option<&[u8]>,
) -> Option<Vec<u8>> {
    let (commit, fold): (CommitFn, FoldFn) = match route {
        0 => (commit_wide, super::fold),
        1 => (commit_sat84, super::fold),
        2 => (commit_wide, fold_lag81),
        3 => (commit_sat84, fold_lag81),
        _ => unreachable!("proof route"),
    };
    let bytes = COLUMNS * rq::N * rq::coefficient_bytes(modulus);
    let mut commitment = vec![0u8;bytes];
    for page in commitment.chunks_mut(4096) {
        // Force each fresh output page resident before any route clocks it.
        unsafe { core::ptr::write_volatile(page.as_mut_ptr(), 0xa5); }
    }
    let (proof, phases, total) = execute(seed, modulus, values, matrices, commitment, commit, fold);
    let checking = Instant::now();
    let actual = output::encode_vec(&proof);
    assert_eq!(actual.len(), 13_958_612);
    if let Some(want) = expected {
        assert!(actual == want, "proof bytes differ in {} {class}/{round}/{position}", LABELS[route]);
    }
    let check = ns(checking.elapsed());
    let [commit, switch, row, fold] = phases;
    eprintln!(
        "diag_proof sample route={route} name={} class={class} round={round} position={position} total_ns={total} commit_ns={commit} switch_ns={switch} row_ns={row} fold_ns={fold} check_ns={check} bytes={} equal=true",
        LABELS[route], actual.len(),
    );
    if expected.is_none() { Some(actual) } else { None }
}

pub(super) fn run(seed: &[u8;32], modulus: u128, values: &[B128]) -> Vec<u8> {
    let setup = Instant::now();
    let matrices = Matrices::new();
    let remapped = matrices.wide.data.as_ptr().cast::<u8>() != SMS_DATA.0.as_ptr();
    let rotation = usize::from(seed[0]) % ORDER.len();
    eprintln!(
        "diag_proof begin routes=4 workers={} warm_rounds=4 rotation={rotation} hook=false shared_key=true timed=in_memory_prove excluded=buffer_setup,wire_encode,equality,output_write setup_ns={} remapped={remapped}",
        threads(), ns(setup.elapsed()),
    );
    let reference = sample(0, "first_use", 0, 0, seed, modulus, values, &matrices, None).unwrap();
    for route in 1..LABELS.len() {
        sample(route, "first_use", 0, route, seed, modulus, values, &matrices, Some(&reference));
    }
    for i in 0..ORDER.len() {
        for (position, route) in ORDER[(i + rotation)%ORDER.len()].into_iter().enumerate() {
            sample(route, "warmed", i+1, position, seed, modulus, values, &matrices, Some(&reference));
        }
    }
    reference
}

pub(super) fn close(expected: &[u8], proof: &ProofParts, elapsed: Duration) {
    let checking = Instant::now();
    let actual = output::encode_vec(proof);
    assert!(actual == expected, "ordinary closing proof differs from timed samples");
    eprintln!(
        "diag_proof closing route=0 hook=true total_ns={} check_ns={} bytes={} equal=true ordinary_write_next=true",
        ns(elapsed), ns(checking.elapsed()), actual.len(),
    );
}

/// The NEON bit-sliced fold (`fold_bits::neon`): 128-lane blocks, negacyclic slot pairs,
/// plain-op full adders, the elements written in place.
#[cfg(target_arch = "aarch64")]
fn fold_neon_lag81(values: &[B128], challenges: &[ShortChallenge]) -> Vec<RingElement648> {
    use labinius::fold_bits as fb;
    const _: () = assert!(COLUMN_LEN % fb::neon_lag81::LANES == 0 && fb::N == rq::N);
    let cs: Vec<fb::Challenge> = challenges
        .iter()
        .map(|c| fb::Challenge {
            positions: c.positions,
            signs: c.signs,
            weight: c.weight,
        })
        .collect();
    // SAFETY: `B128` is `repr(transparent)` over `u128`.
    let words = unsafe { core::slice::from_raw_parts(values.as_ptr() as *const u128, values.len()) };
    fb::neon_lag81::fold(words, COLUMN_LEN, &cs, threads())
}

