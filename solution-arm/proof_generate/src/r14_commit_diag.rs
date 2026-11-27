//! Time the exact forward executors on one fresh table and one public key.
use crate::ntt::{neon::sms, r14_strict23, r14_v93_23, r14_sat84};
use crate::B128;
use std::time::Instant;

const ROUTES: usize = 4;
const ROUNDS: usize = 4;
const VARIANTS: [&str; ROUTES] = ["strict16", "strict23", "v93_23", "sat84"];
const WILLIAMS: [[usize; ROUTES]; ROUNDS] = [[0, 1, 3, 2], [1, 2, 0, 3], [2, 3, 1, 0], [3, 0, 2, 1]];

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

struct Matrices<'a> {
    quarter: &'a sms::SmsMatrix,
    strict23: r14_strict23::sms::SmsMatrix,
    v93_23: r14_v93_23::sms::SmsMatrix,
    sat84: r14_sat84::sms::SmsMatrix,
}

impl<'a> Matrices<'a> {
    fn new(quarter: &'a sms::SmsMatrix) -> Self {
        Self {
            quarter,
            strict23: shared!(r14_strict23, quarter),
            v93_23: shared!(r14_v93_23, quarter),
            sat84: shared!(r14_sat84, quarter),
        }
    }

    fn commit(&self, i: usize, values: &[B128], workers: usize, q: u128, out: &mut [u8]) {
        match i {
            0 => sms::commit_v::<17>(values, self.quarter, workers, q, out, false),
            1 => r14_strict23::sms::commit_v::<17>(values, &self.strict23, workers, q, out, false),
            2 => r14_v93_23::sms::commit_v::<17>(values, &self.v93_23, workers, q, out, false),
            3 => r14_sat84::sms::commit_v::<17>(values, &self.sat84, workers, q, out, false),
            _ => unreachable!("forward comparison route"),
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn sample(
    i: usize, class: &str, round: usize, position: usize,
    values: &[B128], matrices: &Matrices<'_>, workers: usize, q: u128,
    out: &mut [u8], reference: Option<&[u8]>,
) -> f64 {
    let start = Instant::now();
    matrices.commit(i, values, workers, q, out);
    let ms = start.elapsed().as_secs_f64() * 1000.0;
    if let Some(want) = reference {
        assert!(out == want, "forward comparison mismatch in {} {class} {round} {position}", VARIANTS[i]);
    }
    eprintln!("diag_commit sample variant={i} name={} class={class} round={round} position={position} ms={ms:.6} bytes={} equal=true", VARIANTS[i], out.len());
    ms
}

pub fn run(values: &[B128], quarter: &sms::SmsMatrix, workers: usize, q: u128, bytes: usize) {
    let matrices = Matrices::new(quarter);
    eprintln!("diag_commit begin variants={ROUTES} workers={workers} fresh_input=true hooks=false schedule=williams warm_rounds={ROUNDS} external_timer_includes_all_work=true shared_key=true");
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
