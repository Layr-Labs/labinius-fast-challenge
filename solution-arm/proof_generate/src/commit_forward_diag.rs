//! Time each exact forward executor on the same fresh witness.
use crate::ntt::{neon::sms, forward_selective, forward_mixed, forward_paired, forward_mixed_paired, forward_strict};
use crate::{Aligned64, B128, SMS_DATA, SMS_DIGITS, SMS_MEAN_INT, SMS_MEANS, SMS_META,
            SMS_MU, SMS_THRESHOLD};
use std::time::Instant;

static MIXED_DATA: &Aligned64<[u8]> =
    &Aligned64(*include_bytes!(concat!(env!("OUT_DIR"), "/forward-mixed-data.bin")));

pub fn map_mixed_key() {
    forward_mixed::sms::map_key(&MIXED_DATA.0);
}

const ROUTES: usize = 6;
const ROUNDS: usize = 6;
const VARIANTS: [&str; ROUTES] = ["keep10_lag81", "selective", "mixed", "paired", "mixed_paired", "strict"];
const WILLIAMS: [[usize; ROUTES]; ROUNDS] = [[0, 1, 5, 2, 4, 3], [1, 2, 0, 3, 5, 4], [2, 3, 1, 4, 0, 5], [3, 4, 2, 5, 1, 0], [4, 5, 3, 0, 2, 1], [5, 0, 4, 1, 3, 2]];

macro_rules! shared {
    ($module:ident, $key:expr) => {{
        let key = $key;
        let shared = $module::sms::SmsMatrix {
            elements: key.elements,
            data: key.data.clone(),
            mean_residues: key.mean_residues.clone(),
            mu: key.mu.clone(),
            digits: key.digits.clone(),
            mean_int: key.mean_int.clone(),
            threshold: key.threshold.clone(),
            margin: key.margin,
            column_failure: key.column_failure,
        };
        assert_eq!(shared.data.as_ptr(), key.data.as_ptr());
        assert_eq!(shared.mean_residues.as_ptr(), key.mean_residues.as_ptr());
        assert_eq!(shared.mu.as_ptr(), key.mu.as_ptr());
        assert_eq!(shared.digits.as_ptr(), key.digits.as_ptr());
        assert_eq!(shared.mean_int.as_ptr(), key.mean_int.as_ptr());
        assert_eq!(shared.threshold.as_ptr(), key.threshold.as_ptr());
        shared
    }};
}

type Executor = fn(&Matrices<'_>, &[B128], usize, u128, &mut [u8]);

struct Matrices<'a> {
    base: &'a sms::SmsMatrix,
    selective: forward_selective::sms::SmsMatrix,
    mixed: forward_mixed::sms::SmsMatrix,
    paired: forward_paired::sms::SmsMatrix,
    mixed_paired: forward_mixed_paired::sms::SmsMatrix,
    strict: forward_strict::sms::SmsMatrix,
}

impl<'a> Matrices<'a> {
    fn from_keys(base: &'a sms::SmsMatrix, mixed: forward_mixed::sms::SmsMatrix) -> Self {
        let selective = shared!(forward_selective, &base);
        let paired = shared!(forward_paired, &base);
        let mixed_paired = shared!(forward_mixed_paired, &mixed);
        let strict = shared!(forward_strict, &base);
        Self { base, selective, mixed, paired, mixed_paired, strict }
    }

    fn executor(i: usize) -> Executor {
        match i {
            0 => |m, values, workers, q, out| sms::commit_v::<17>(values, m.base, workers, q, out, false),
            1 => |m, values, workers, q, out| forward_selective::sms::commit_v::<17>(values, &m.selective, workers, q, out, false),
            2 => |m, values, workers, q, out| forward_mixed::sms::commit_v::<17>(values, &m.mixed, workers, q, out, false),
            3 => |m, values, workers, q, out| forward_paired::sms::commit_v::<17>(values, &m.paired, workers, q, out, false),
            4 => |m, values, workers, q, out| forward_mixed_paired::sms::commit_v::<17>(values, &m.mixed_paired, workers, q, out, false),
            5 => |m, values, workers, q, out| forward_strict::sms::commit_v::<17>(values, &m.strict, workers, q, out, false),
            _ => unreachable!("forward comparison route"),
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn sample(
    i: usize, class: &str, round: usize, position: usize,
    values: &[B128], matrices: &Matrices<'_>, workers: usize, q: u128,
    out: &mut [u8], reference: Option<&[u8]>,
) -> f64 {
    let execute = Matrices::executor(i);
    let start = Instant::now();
    execute(matrices, values, workers, q, out);
    let ms = start.elapsed().as_secs_f64() * 1000.0;
    if let Some(want) = reference {
        assert!(out == want, "forward comparison mismatch in {} {class} {round} {position}", VARIANTS[i]);
    }
    eprintln!("diag_commit sample variant={i} name={} class={class} round={round} position={position} ms={ms:.6} bytes={} equal=true", VARIANTS[i], out.len());
    ms
}

pub fn run(values: &[B128], base: &sms::SmsMatrix, workers: usize, q: u128, bytes: usize) {
    let mixed = forward_mixed::sms::SmsMatrix::from_images(
        &MIXED_DATA.0, &SMS_MEANS.0, &SMS_MU.0, &SMS_META.0, &SMS_DIGITS.0,
        &SMS_MEAN_INT.0, &SMS_THRESHOLD.0, base.elements,
    ).unwrap_or_else(|| forward_mixed::sms::matrix(base.elements, workers));
    assert_ne!(base.data.as_ptr(), mixed.data.as_ptr());
    let old_remapped = !SMS_DATA.0.is_empty() && base.data.as_ptr().cast::<u8>() != SMS_DATA.0.as_ptr();
    let mixed_remapped = !MIXED_DATA.0.is_empty() && mixed.data.as_ptr().cast::<u8>() != MIXED_DATA.0.as_ptr();
    if !MIXED_DATA.0.is_empty() {
        assert_eq!(base.mean_residues.as_ptr(), mixed.mean_residues.as_ptr());
        assert_eq!(base.mean_int.as_ptr(), mixed.mean_int.as_ptr());
        assert_eq!(base.digits.as_ptr(), mixed.digits.as_ptr());
        assert_eq!(base.mu.as_ptr(), mixed.mu.as_ptr());
        assert_eq!(base.threshold.as_ptr(), mixed.threshold.as_ptr());
    }
    let matrices = Matrices::from_keys(base, mixed);
    eprintln!("diag_commit begin variants={ROUTES} workers={workers} fresh_input=true hooks=false schedule=williams warm_rounds={ROUNDS} external_timer_includes_all_work=true hot_keys=2 old_remapped={old_remapped} mixed_remapped={mixed_remapped}");
    let mut outputs: [Vec<u8>; ROUTES] = std::array::from_fn(|_| {
        let mut out = vec![0; bytes];
        for page in out.chunks_mut(4096) {
            page[0] = 1;
        }
        std::hint::black_box(&mut out);
        out
    });
    let first = sample(0, "first_use", 0, 0, values, &matrices, workers, q, &mut outputs[0], None);
    let reference = outputs[0].clone();
    for (i, out) in outputs.iter_mut().enumerate().skip(1) {
        sample(i, "first_use", 0, i, values, &matrices, workers, q, out, Some(&reference));
    }
    let mut times = [[0.0f64; ROUNDS]; ROUTES];
    for (round, row) in WILLIAMS.iter().enumerate() {
        for (position, &i) in row.iter().enumerate() {
            times[i][round] = sample(i, "warmed", round + 1, position, values, &matrices,
                                     workers, q, &mut outputs[i], Some(&reference));
        }
    }
    let closing = sample(0, "closing_control", ROUNDS + 1, 0, values, &matrices, workers, q, &mut outputs[0], Some(&reference));
    for (i, samples) in times.iter().enumerate() {
        let mut sorted = *samples;
        sorted.sort_by(f64::total_cmp);
        let median = (sorted[ROUNDS / 2 - 1] + sorted[ROUNDS / 2]) * 0.5;
        eprintln!("diag_commit summary variant={i} name={} warmed_median_ms={median:.6} warmed_samples_ms={samples:?}", VARIANTS[i]);
    }
    eprintln!("diag_commit end first_control_ms={first:.6} closing_control_ms={closing:.6} all_equal=true normal_proof_next=true");
}
