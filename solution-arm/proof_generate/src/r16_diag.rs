//! Compare the four commitments before making the ordinary proof.
use crate::ntt::neon::{
    balanced_exact, balanced_sat84, count_balanced_exact, count_balanced_sat84,
    count_published_exact, count_published_sat84, published_sat84, sms,
};
use crate::{Aligned64, B128, RING_ELEMENTS, SMS_DATA, SMS_DIGITS, SMS_MEANS, SMS_MEAN_INT,
            SMS_META, SMS_MU, SMS_THRESHOLD};
use std::time::Instant;

macro_rules! image {
    ($name:ident, $suffix:literal) => {
        pub(crate) static $name: &Aligned64<[u8]> = &Aligned64(*include_bytes!(concat!(
            env!("OUT_DIR"), "/balanced-", $suffix, ".bin"
        )));
    };
}
image!(BAL_DATA, "data");
image!(BAL_MEANS, "means");
image!(BAL_MU, "mu");
image!(BAL_META, "meta");
image!(BAL_DIGITS, "digits");
image!(BAL_MEAN_INT, "mean-int");
image!(BAL_THRESHOLD, "threshold");

const NAMES: [&str; 4] = ["published_exact", "balanced_exact", "published_sat84", "balanced_sat84"];
const ORDER: [[usize; 4]; 4] = [[0, 1, 3, 2], [1, 2, 0, 3], [2, 3, 1, 0], [3, 0, 2, 1]];

pub(crate) fn map_balanced_key() {
    balanced_exact::map_key(&BAL_DATA.0);
}

pub(crate) fn run(values: &[B128], ordinary: &sms::SmsMatrix, workers: usize, modulus: u128, bytes: usize) {
    let published_data = sms::r16_key_image(&SMS_DATA.0);
    let balanced_data = balanced_exact::r16_key_image(&BAL_DATA.0);
    macro_rules! published {
        ($module:ident) => {{
            let m = $module::SmsMatrix::from_images(
                published_data, &SMS_MEANS.0, &SMS_MU.0, &SMS_META.0, &SMS_DIGITS.0,
                &SMS_MEAN_INT.0, &SMS_THRESHOLD.0, RING_ELEMENTS).unwrap();
            assert_eq!(ordinary.data.as_ptr(), m.data.as_ptr());
            assert_eq!(ordinary.digits.as_ptr(), m.digits.as_ptr());
            m
        }};
    }
    macro_rules! balanced {
        ($module:ident) => {{
            let m = $module::SmsMatrix::from_images(
                balanced_data, &BAL_MEANS.0, &BAL_MU.0, &BAL_META.0, &BAL_DIGITS.0,
                &BAL_MEAN_INT.0, &BAL_THRESHOLD.0, RING_ELEMENTS).unwrap();
            assert_eq!(m.data.as_ptr().cast::<u8>(), balanced_data.as_ptr());
            assert_eq!(m.digits.as_ptr().cast::<u8>(), BAL_DIGITS.0.as_ptr());
            m
        }};
    }
    let p_sat = published!(published_sat84);
    let b_exact = balanced!(balanced_exact);
    let b_sat = balanced!(balanced_sat84);
    let p_remapped = published_data.as_ptr() != SMS_DATA.0.as_ptr();
    let b_remapped = balanced_data.as_ptr() != BAL_DATA.0.as_ptr();
    assert_ne!(published_data.as_ptr(), balanced_data.as_ptr());
    eprintln!("r16 begin workers={workers} published_remapped={p_remapped} balanced_remapped={b_remapped}");
    let apply = |id, out: &mut [u8]| match id {
        0 => sms::commit_v::<17>(values, ordinary, workers, modulus, out, false),
        1 => balanced_exact::commit_v::<17>(values, &b_exact, workers, modulus, out, false),
        2 => published_sat84::commit_v::<17>(values, &p_sat, workers, modulus, out, false),
        3 => balanced_sat84::commit_v::<17>(values, &b_sat, workers, modulus, out, false),
        _ => unreachable!(),
    };
    let mut reference = vec![0; bytes];
    let mut result = vec![0; bytes];
    let start = Instant::now();
    apply(0, &mut reference);
    eprintln!("r16 first-reference ms={:.6}",start.elapsed().as_secs_f64()*1000.);
    for (round, row) in ORDER.iter().enumerate() {
        for (position, &id) in row.iter().enumerate() {
            let start = Instant::now();
            apply(id, &mut result);
            let ms = start.elapsed().as_secs_f64() * 1000.;
            assert_eq!(result, reference, "route {}", NAMES[id]);
            eprintln!("r16 time route={} round={round} position={position} ms={ms:.6} equal=true", NAMES[id]);
        }
    }
    macro_rules! count {
        ($id:literal,$module:ident,$m:expr) => {{
            let matrix = $m;
            $module::commit_v::<17>(values, &matrix, workers, modulus, &mut result, false);
            assert_eq!(result, reference, "counted route {}", NAMES[$id]);
            let [lanes, rows, forward, terminal] = $module::take_counts();
            eprintln!("r16 count route={} lanes={lanes} rows={rows} forward={forward} terminal={terminal} equal=true",NAMES[$id]);
        }};
    }
    count!(0,count_published_exact,published!(count_published_exact));
    count!(1,count_balanced_exact,balanced!(count_balanced_exact));
    count!(2,count_published_sat84,published!(count_published_sat84));
    count!(3,count_balanced_sat84,balanced!(count_balanced_sat84));
    apply(0, &mut result);
    assert_eq!(result, reference, "closing control");
    eprintln!("r16 end ordinary_proof_next=true closing_equal=true");
}
