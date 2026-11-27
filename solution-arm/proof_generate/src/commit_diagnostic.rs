//! Same-input commitment comparisons. All work is inside the normal prover process.
use crate::{ntt::neon::sms, B128};
use std::time::Instant;

// Add a variant only after its implementation and source record are ready.
const VARIANTS: &[(u8, &str)] = &[
    (4, "leader"),
    (0, "driver_control"),
    (15, "weighted_paired_index_loads"),
    (19, "weighted_front_tuned_tail"),
    (16, "signed_front_tuned_tail"),
    (20, "weighted_front_paired_tail"),
];

fn warmed_rounds(variants: usize) -> usize {
    if variants % 2 == 0 { variants } else { 2 * variants }
}

// First-order balanced Williams rows. Odd orders also need the reversed rows.
fn warmed_variant_index(variants: usize, round: usize, position: usize) -> usize {
    let position = if round < variants { position } else { variants - 1 - position };
    let base_index = match position {
        0 => 0,
        n if n % 2 == 1 => n.div_ceil(2),
        n => variants - n / 2,
    };
    (round + base_index) % variants
}

fn commit(v: u8, values: &[B128], matrix: &sms::SmsMatrix, workers: usize, q: u128, out: &mut [u8]) {
    match v {
        4 => sms::commit_v::<4>(values, matrix, workers, q, out, false),
        0 => sms::commit_candidate::<0>(values, matrix, workers, q, out, false),
        15 => sms::commit_candidate::<15>(values, matrix, workers, q, out, false),
        19 => sms::commit_candidate::<19>(values, matrix, workers, q, out, false),
        16 => sms::commit_candidate::<16>(values, matrix, workers, q, out, false),
        20 => sms::commit_candidate::<20>(values, matrix, workers, q, out, false),
        _ => unreachable!("unregistered diagnostic variant"),
    }
}

fn sample(
    v: u8, name: &str, class: &str, round: usize, position: usize,
    values: &[B128], matrix: &sms::SmsMatrix, workers: usize, q: u128,
    out: &mut [u8], reference: Option<&[u8]>,
) -> f64 {
    let start = Instant::now();
    commit(v, values, matrix, workers, q, out);
    let ms = start.elapsed().as_secs_f64() * 1000.0;
    if let Some(reference) = reference {
        assert!(out == reference, "diagnostic commitment mismatch: variant {v}, class {class}, round {round}, position {position}");
    }
    eprintln!("diag_commit sample variant={v} name={name} class={class} round={round} position={position} ms={ms:.6} bytes={} equal=true", out.len());
    ms
}

pub fn run(values: &[B128], matrix: &sms::SmsMatrix, workers: usize, q: u128, bytes: usize) {
    assert_eq!(VARIANTS[0].0, 4);
    let rounds = warmed_rounds(VARIANTS.len());
    eprintln!("diag_commit begin variants={} workers={workers} fresh_input=true hooks=false schedule=williams warm_rounds={rounds} external_timer_includes_all_work=true", VARIANTS.len());
    let mut outputs: Vec<Vec<u8>> = VARIANTS.iter().map(|_| {
        let mut out = vec![0u8; bytes];
        for page in out.chunks_mut(4096) {
            page[0] = 1;
        }
        if let Some(last) = out.last_mut() { *last = 1; }
        std::hint::black_box(&mut out);
        out
    }).collect();
    // The first leader call establishes this instance's reference. No output is
    // retained across inputs or reused for the final proof.
    let first = sample(4, VARIANTS[0].1, "first_use", 0, 0,
        values, matrix, workers, q, &mut outputs[0], None);
    let reference = outputs[0].clone();
    for i in 1..VARIANTS.len() {
        let (v, name) = VARIANTS[i];
        sample(v, name, "first_use", 0, i,
            values, matrix, workers, q, &mut outputs[i], Some(&reference));
    }
    // The initial calls above are reported separately from the warmed medians.
    let mut times: Vec<Vec<f64>> = VARIANTS.iter().map(|_| Vec::new()).collect();
    for round in 0..rounds {
        for position in 0..VARIANTS.len() {
            let i = warmed_variant_index(VARIANTS.len(), round, position);
            let (v, name) = VARIANTS[i];
            times[i].push(sample(v, name, "warmed", round + 1, position,
                values, matrix, workers, q, &mut outputs[i], Some(&reference)));
        }
    }
    let closing = sample(4, VARIANTS[0].1, "closing_control", rounds + 1, 0,
        values, matrix, workers, q, &mut outputs[0], Some(&reference));
    for (i, samples) in times.iter().enumerate() {
        let mut sorted = samples.clone();
        sorted.sort_by(f64::total_cmp);
        let n = sorted.len();
        let median = if n % 2 == 1 { sorted[n / 2] } else { (sorted[n / 2 - 1] + sorted[n / 2]) * 0.5 };
        let (v, name) = VARIANTS[i];
        eprintln!("diag_commit summary variant={v} name={name} warmed_median_ms={median:.6} warmed_samples_ms={samples:?}");
    }
    eprintln!("diag_commit end first_control_ms={first:.6} closing_control_ms={closing:.6} all_equal=true normal_proof_next=true");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diagnostic_variants_match_leader() {
        for (elements, columns, workers, force_repair) in [
            (5, 3, 1, false),
            (385, 23, 1, false),
            (130, 18, 1, true),
            (401, 41, 3, false),
        ] {
            let mut state = 300;
            let values = (0..columns * 4 * elements).map(|_| {
                let hi = crate::ntt::tests::xorshift(&mut state) as u128;
                let lo = crate::ntt::tests::xorshift(&mut state) as u128;
                B128((hi << 64) | lo)
            }).collect::<Vec<_>>();
            let mut matrix = sms::matrix(elements, workers);
            if force_repair {
                matrix.threshold.to_mut().fill(0);
            }
            let mut leader = vec![0; columns * crate::ntt::N * 5];
            commit(4, &values, &matrix, workers, sms::MODULUS, &mut leader);
            let mut actual = vec![0; leader.len()];
            for &(variant, name) in &VARIANTS[1..] {
                actual.fill(1);
                commit(variant, &values, &matrix, workers, sms::MODULUS, &mut actual);
                assert!(actual == leader, "{name}, {elements} elements, {columns} columns");
            }
        }
    }
}
