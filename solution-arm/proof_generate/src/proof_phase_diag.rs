//! Full same-input timing. Every sample recomputes all four proving phases.
use super::*;
use crate::ntt::{neon::sms, phase_keep10::sms as keep10};
use std::time::Instant;

const LABELS: [&str; 6] = ["literal", "keep10", "lag81", "keep10_lag81", "trim", "lag81_trim"];
const ORDER: [[usize; 6]; 6] = [[0, 1, 5, 2, 4, 3], [1, 2, 0, 3, 5, 4], [2, 3, 1, 4, 0, 5], [3, 4, 2, 5, 1, 0], [4, 5, 3, 0, 2, 1], [5, 0, 4, 1, 3, 2]];

struct Matrices {
    ordinary: sms::SmsMatrix,
    guarded: keep10::SmsMatrix,
}

impl Matrices {
    fn new() -> Self {
        let ordinary = sms::SmsMatrix::from_images(
            &SMS_DATA.0, &SMS_MEANS.0, &SMS_MU.0, &SMS_META.0, &SMS_DIGITS.0,
            &SMS_MEAN_INT.0, &SMS_THRESHOLD.0, RING_ELEMENTS,
        ).expect("paired timing requires the inherited embedded key");
        let guarded = keep10::SmsMatrix {
            elements: ordinary.elements,
            data: ordinary.data.clone(),
            mean_residues: ordinary.mean_residues.clone(),
            mu: ordinary.mu.clone(),
            digits: ordinary.digits.clone(),
            mean_int: ordinary.mean_int.clone(),
            threshold: ordinary.threshold.clone(),
            margin: ordinary.margin,
            column_failure: ordinary.column_failure,
        };
        assert_eq!(ordinary.data.as_ptr(), guarded.data.as_ptr());
        assert_eq!(ordinary.mean_residues.as_ptr(), guarded.mean_residues.as_ptr());
        assert_eq!(ordinary.mu.as_ptr(), guarded.mu.as_ptr());
        assert_eq!(ordinary.digits.as_ptr(), guarded.digits.as_ptr());
        assert_eq!(ordinary.mean_int.as_ptr(), guarded.mean_int.as_ptr());
        assert_eq!(ordinary.threshold.as_ptr(), guarded.threshold.as_ptr());
        Self { ordinary, guarded }
    }
}

#[derive(Default)]
struct Phases([u64; 4]);

fn ns(d: std::time::Duration) -> u64 {
    d.as_nanos().try_into().expect("phase exceeds u64 nanoseconds")
}

#[inline(never)]
fn execute<const GUARDED: bool, const FOLD: u8>(
    seed: &[u8; 32],
    modulus: u128,
    values: &[B128],
    matrices: &Matrices,
    mut commitment: Vec<u8>,
) -> (ProofParts, Phases, u64) {
    let start = Instant::now();
    let mut lap = start;
    let mut phases = Phases::default();
    if GUARDED {
        keep10::commit_v::<17>(values, &matrices.guarded, threads(), modulus, &mut commitment, false);
    } else {
        sms::commit_v::<17>(values, &matrices.ordinary, threads(), modulus, &mut commitment, false);
    }
    labinius::par::pool();
    let now = Instant::now();
    phases.0[0] = ns(now.duration_since(lap));
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
    phases.0[1] = ns(now.duration_since(lap));
    lap = now;
    let (p0, _) = split_point(&coords);
    let row = RowEvaluation::of(row_evaluate(values, &p0));
    let now = Instant::now();
    phases.0[2] = ns(now.duration_since(lap));
    lap = now;
    let challenges = folding_challenges(&mut t, &row);
    let folded = FoldedWitness::of(match FOLD {
        0 => fold(values, &challenges),
        1 => fold_lag81(values, &challenges),
        2 => fold_trim(values, &challenges),
        3 => fold_lag81_trim(values, &challenges),
        _ => unreachable!("fold route"),
    });
    let now = Instant::now();
    phases.0[3] = ns(now.duration_since(lap));
    
    let proof = ProofParts {
        modulus,
        commitment,
        partials: partials.iter().map(|v| v.0).collect(),
        rounds,
        opened: opened.0,
        row: wire::pack_row_evaluation(&row),
        folded,
    };
    let elapsed = ns(start.elapsed());
    (proof, phases, elapsed)
}

fn sample(
    route: usize, class: &str, round: usize, position: usize, seed: &[u8; 32],
    modulus: u128, values: &[B128], matrices: &Matrices, expected: Option<&[u8]>,
) -> Option<Vec<u8>> {
    let bytes = COLUMNS * rq::N * rq::coefficient_bytes(modulus);
    let mut commitment = vec![0u8; bytes];
    for page in commitment.chunks_mut(4096) {
        // With all pages resident, each route overwrites the entire fresh buffer.
        unsafe { core::ptr::write_volatile(page.as_mut_ptr(), 0xa5); }
    }
    let (proof, phases, elapsed) = match route {
        0 => execute::<false, 0>(seed, modulus, values, matrices, commitment),
        1 => execute::<true, 0>(seed, modulus, values, matrices, commitment),
        2 => execute::<false, 1>(seed, modulus, values, matrices, commitment),
        3 => execute::<true, 1>(seed, modulus, values, matrices, commitment),
        4 => execute::<false, 2>(seed, modulus, values, matrices, commitment),
        5 => execute::<false, 3>(seed, modulus, values, matrices, commitment),
        _ => unreachable!("proof phase route"),
    };
    let checking = Instant::now();
    let actual = output::encode_vec(&proof);
    assert_eq!(actual.len(), 13_958_612);
    if let Some(want) = expected {
        assert!(actual == want, "proof bytes differ in {} {class}/{round}/{position}", LABELS[route]);
    }
    let check_ns = ns(checking.elapsed());
    let [commit, switch, row, fold] = phases.0;
    eprintln!(
        "diag_proof sample route={route} name={} class={class} round={round} position={position} total_ns={elapsed} commit_ns={commit} switch_ns={switch} row_ns={row} fold_ns={fold} check_ns={check_ns} bytes={} equal=true",
        LABELS[route], actual.len()
    );
    if expected.is_none() { Some(actual) } else { None }
}

pub(super) fn run(seed: &[u8; 32], modulus: u128, values: &[B128]) -> Vec<u8> {
    let setup = Instant::now();
    let matrices = Matrices::new();
    let rotation = usize::from(seed[0]) % ORDER.len();
    eprintln!(
        "diag_proof begin routes=6 workers={} warm_rounds=6 rotation={rotation} hook=false shared_key=true timed=in_memory_prove excluded=buffer_setup,wire_encode,equality,output_write setup_ns={}",
        threads(), ns(setup.elapsed())
    );
    let reference = sample(0, "first_use", 0, 0, seed, modulus, values, &matrices, None).unwrap();
    for route in 1..LABELS.len() {
        sample(route, "first_use", 0, route, seed, modulus, values, &matrices, Some(&reference));
    }
    for i in 0..ORDER.len() {
        for (position, route) in ORDER[(i + rotation) % ORDER.len()].into_iter().enumerate() {
            sample(route, "warmed", i + 1, position, seed, modulus, values, &matrices, Some(&reference));
        }
    }
    reference
}

pub(super) fn close(expected: &[u8], proof: &ProofParts, elapsed: std::time::Duration) {
    let checking = Instant::now();
    let actual = output::encode_vec(proof);
    assert!(actual == expected, "literal closing proof differs from in-process matrix");
    eprintln!(
        "diag_proof closing route=0 hook=true total_ns={} check_ns={} bytes={} equal=true ordinary_write_next=true",
        ns(elapsed), ns(checking.elapsed()), actual.len()
    );
}

fn fold_lag81(values: &[B128], challenges: &[ShortChallenge]) -> Vec<RingElement648> {
    // The NEON bit-sliced fold's 16-plane counters hold sums below 2^15.
    #[cfg(target_arch = "aarch64")]
    if challenges.iter().map(|c| c.weight).sum::<usize>() < 1 << 15 {
        return fold_neon_lag81(values, challenges);
    }
    // The scheduled bit-sliced fold's 15-bit counters hold the total weight plus its padding.
    if let Some(out) = fold_scheduled(values, challenges) {
        return out;
    }
    // The bit-sliced fold's 16-plane counters hold sums below 2^15.
    if challenges.iter().map(|c| c.weight).sum::<usize>() < 1 << 15 {
        return fold_bitsliced(values, challenges);
    }
    fold_bytes(values, challenges)
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


fn fold_trim(values: &[B128], challenges: &[ShortChallenge]) -> Vec<RingElement648> {
    // The NEON bit-sliced fold's 16-plane counters hold sums below 2^15.
    #[cfg(target_arch = "aarch64")]
    if challenges.iter().map(|c| c.weight).sum::<usize>() < 1 << 15 {
        return fold_neon_trim(values, challenges);
    }
    // The scheduled bit-sliced fold's 15-bit counters hold the total weight plus its padding.
    if let Some(out) = fold_scheduled(values, challenges) {
        return out;
    }
    // The bit-sliced fold's 16-plane counters hold sums below 2^15.
    if challenges.iter().map(|c| c.weight).sum::<usize>() < 1 << 15 {
        return fold_bitsliced(values, challenges);
    }
    fold_bytes(values, challenges)
}

/// The NEON bit-sliced fold (`fold_bits::neon`): 128-lane blocks, negacyclic slot pairs,
/// plain-op full adders, the elements written in place.
#[cfg(target_arch = "aarch64")]
fn fold_neon_trim(values: &[B128], challenges: &[ShortChallenge]) -> Vec<RingElement648> {
    use labinius::fold_bits as fb;
    const _: () = assert!(COLUMN_LEN % fb::neon_trim::LANES == 0 && fb::N == rq::N);
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
    fb::neon_trim::fold(words, COLUMN_LEN, &cs, threads())
}


fn fold_lag81_trim(values: &[B128], challenges: &[ShortChallenge]) -> Vec<RingElement648> {
    // The NEON bit-sliced fold's 16-plane counters hold sums below 2^15.
    #[cfg(target_arch = "aarch64")]
    if challenges.iter().map(|c| c.weight).sum::<usize>() < 1 << 15 {
        return fold_neon_lag81_trim(values, challenges);
    }
    // The scheduled bit-sliced fold's 15-bit counters hold the total weight plus its padding.
    if let Some(out) = fold_scheduled(values, challenges) {
        return out;
    }
    // The bit-sliced fold's 16-plane counters hold sums below 2^15.
    if challenges.iter().map(|c| c.weight).sum::<usize>() < 1 << 15 {
        return fold_bitsliced(values, challenges);
    }
    fold_bytes(values, challenges)
}

/// The NEON bit-sliced fold (`fold_bits::neon`): 128-lane blocks, negacyclic slot pairs,
/// plain-op full adders, the elements written in place.
#[cfg(target_arch = "aarch64")]
fn fold_neon_lag81_trim(values: &[B128], challenges: &[ShortChallenge]) -> Vec<RingElement648> {
    use labinius::fold_bits as fb;
    const _: () = assert!(COLUMN_LEN % fb::neon_lag81_trim::LANES == 0 && fb::N == rq::N);
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
    fb::neon_lag81_trim::fold(words, COLUMN_LEN, &cs, threads())
}

