//! Compare complete commitments on this invocation's witness table.
use crate::ntt::neon::{keydiag_c128_fixed, keydiag_c144_cold, keydiag_c144_exact, sms};
use crate::{Aligned64, B128, RING_ELEMENTS};
use std::time::Instant;

const VARIANTS: [&str; 4] = ["v84", "fixed128", "fixed144_exact", "fixed144_cold"];
const WILLIAMS: [[usize; 4]; 4] = [[0, 1, 3, 2], [1, 2, 0, 3], [2, 3, 1, 0], [3, 0, 2, 1]];

macro_rules! image {
    ($name:ident, $file:literal) => {
        static $name: &Aligned64<[u8]> = &Aligned64(*include_bytes!(concat!(env!("OUT_DIR"), "/", $file)));
    };
}
image!(DATA, "keydiag-data.bin");
image!(DIGITS, "keydiag-digits.bin");
image!(MEANS128, "keydiag-c128-means.bin");
image!(MU128, "keydiag-c128-mu.bin");
image!(META128, "keydiag-c128-meta.bin");
image!(MEAN128, "keydiag-c128-mean-int.bin");
image!(GATE128, "keydiag-c128-threshold.bin");
image!(MEANS144, "keydiag-c144-means.bin");
image!(MU144, "keydiag-c144-mu.bin");
image!(META144, "keydiag-c144-meta.bin");
image!(MEAN144, "keydiag-c144-mean-int.bin");
image!(GATE144, "keydiag-c144-threshold.bin");

struct Matrices<'a> {
    ordinary: &'a sms::SmsMatrix,
    fixed128: keydiag_c128_fixed::SmsMatrix,
    fixed144_exact: keydiag_c144_exact::SmsMatrix,
    fixed144_cold: keydiag_c144_cold::SmsMatrix,
}

impl<'a> Matrices<'a> {
    fn new(ordinary: &'a sms::SmsMatrix) -> Self {
        let data = sms::keydiag_map_image(&DATA.0);
        let all_digits = &DIGITS.0;
        let digits = &all_digits[..RING_ELEMENTS * sms::DIGITS * crate::ntt::N];
        let fixed128 = keydiag_c128_fixed::SmsMatrix::from_images(data, &MEANS128.0, &MU128.0, &META128.0,
            digits, &MEAN128.0, &GATE128.0, RING_ELEMENTS).expect("fixed C128 diagnostic key");
        let fixed144_exact = keydiag_c144_exact::SmsMatrix::from_images(data, &MEANS144.0, &MU144.0, &META144.0,
            digits, &MEAN144.0, &GATE144.0, RING_ELEMENTS).expect("fixed C144 exact diagnostic key");
        let fixed144_cold = keydiag_c144_cold::SmsMatrix::from_images(data, &MEANS144.0, &MU144.0, &META144.0,
            all_digits, &MEAN144.0, &GATE144.0, RING_ELEMENTS).expect("fixed C144 cold diagnostic key");
        assert!(std::ptr::eq(fixed128.data.as_ptr(), fixed144_exact.data.as_ptr()));
        assert!(std::ptr::eq(fixed128.data.as_ptr(), fixed144_cold.data.as_ptr()));
        assert!(std::ptr::eq(fixed128.digits.as_ptr(), fixed144_exact.digits.as_ptr()));
        assert!(std::ptr::eq(fixed128.digits.as_ptr(), fixed144_cold.digits.as_ptr()));
        Self { ordinary, fixed128, fixed144_exact, fixed144_cold }
    }

    fn commit(&self, i: usize, values: &[B128], workers: usize, q: u128, out: &mut [u8]) {
        match i {
            0 => sms::commit_v::<17>(values, self.ordinary, workers, q, out, false),
            1 => keydiag_c128_fixed::commit_v::<17>(values, &self.fixed128, workers, q, out, false),
            2 => keydiag_c144_exact::commit_v::<17>(values, &self.fixed144_exact, workers, q, out, false),
            3 => keydiag_c144_cold::commit_v::<17>(values, &self.fixed144_cold, workers, q, out, false),
            _ => unreachable!("diagnostic route"),
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
        assert!(out == want, "diagnostic commitment mismatch in {} {class} {round} {position}", VARIANTS[i]);
    }
    eprintln!("diag_commit sample variant={i} name={} class={class} round={round} position={position} ms={ms:.6} bytes={} equal=true", VARIANTS[i], out.len());
    ms
}

pub fn run(values: &[B128], ordinary: &sms::SmsMatrix, workers: usize, q: u128, bytes: usize) {
    let matrices = Matrices::new(ordinary);
    eprintln!("diag_commit begin variants=4 workers={workers} fresh_input=true hooks=false schedule=williams warm_rounds=4 external_timer_includes_all_work=true");
    let mut outputs: [Vec<u8>; 4] = std::array::from_fn(|_| {
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
    let mut times = [[0.0f64; 4]; 4];
    for (round, row) in WILLIAMS.iter().enumerate() {
        for (position, &i) in row.iter().enumerate() {
            times[i][round] = sample(i, "warmed", round + 1, position, values, &matrices,
                                     workers, q, &mut outputs[i], Some(&reference));
        }
    }
    let closing = sample(0, "closing_control", 5, 0, values, &matrices, workers, q, &mut outputs[0], Some(&reference));
    for (i, samples) in times.iter().enumerate() {
        let mut sorted = *samples;
        sorted.sort_by(f64::total_cmp);
        let median = (sorted[1] + sorted[2]) * 0.5;
        eprintln!("diag_commit summary variant={i} name={} warmed_median_ms={median:.6} warmed_samples_ms={samples:?}",VARIANTS[i]);
    }
    eprintln!("diag_commit end first_control_ms={first:.6} closing_control_ms={closing:.6} all_equal=true normal_proof_next=true");
}
