//! Four same-input routes, before the untouched ordinary proof output.
use crate::ntt::{neon::sms, r17_mixed::sms as mixed};
use crate::{Aligned64, B128, RING_ELEMENTS, SMS_DATA, SMS_DIGITS, SMS_MEANS, SMS_MEAN_INT,
            SMS_META, SMS_MU, SMS_THRESHOLD};
use std::time::Instant;

static MIXED_DATA: &Aligned64<[u8]> = &Aligned64(*include_bytes!(concat!(
    env!("OUT_DIR"), "/r17-mixed-data.bin"
)));

const NAMES: [&str; 4] = ["selective_dense", "selective_padded", "mixed_dense", "mixed_padded"];
const ORDER: [[usize; 4]; 4] = [[0, 1, 3, 2], [1, 2, 0, 3], [2, 3, 1, 0], [3, 0, 2, 1]];

pub(crate) fn map_mixed_key() { mixed::map_key(&MIXED_DATA.0); }

pub(crate) fn run(values: &[B128], dense: &sms::SmsMatrix, workers: usize, q: u128, bytes: usize) {
    let m = mixed::SmsMatrix::from_images(&MIXED_DATA.0, &SMS_MEANS.0, &SMS_MU.0,
        &SMS_META.0, &SMS_DIGITS.0, &SMS_MEAN_INT.0, &SMS_THRESHOLD.0, RING_ELEMENTS).unwrap();
    assert_eq!(dense.data.as_ptr().cast::<u8>(),sms::selected_key(&SMS_DATA.0).as_ptr());
    assert_eq!(m.data.as_ptr().cast::<u8>(),mixed::selected_key(&MIXED_DATA.0).as_ptr());
    assert_ne!(m.data.as_ptr(),dense.data.as_ptr());
    assert_eq!(m.digits.as_ptr(),dense.digits.as_ptr());
    let p_remapped = dense.data.as_ptr().cast::<u8>() != SMS_DATA.0.as_ptr();
    let m_remapped = m.data.as_ptr().cast::<u8>() != MIXED_DATA.0.as_ptr();
    eprintln!("r17 begin workers={workers} selective_remapped={p_remapped} mixed_remapped={m_remapped}");
    let apply = |i, out: &mut [u8]| match i {
        0 => sms::commit_layout::<17,0>(values,dense,workers,q,out,false),
        1 => sms::commit_layout::<17,192>(values,dense,workers,q,out,false),
        2 => mixed::commit_layout::<17,0>(values,&m,workers,q,out,false),
        3 => mixed::commit_layout::<17,192>(values,&m,workers,q,out,false),
        _ => unreachable!(),
    };
    let mut reference=vec![0;bytes];
    let mut result=vec![0;bytes];
    apply(0,&mut reference);
    for id in 1..4 {
        apply(id,&mut result);
        assert_eq!(result,reference,"first use {}",NAMES[id]);
    }
    for (round,row) in ORDER.iter().enumerate() {
        for (pos,&id) in row.iter().enumerate() {
            let t=Instant::now();
            apply(id,&mut result);
            let ms=t.elapsed().as_secs_f64()*1000.;
            assert_eq!(result,reference,"{}",NAMES[id]);
            eprintln!("r17 time route={} round={round} position={pos} ms={ms:.6} equal=true",NAMES[id]);
        }
    }
    apply(0,&mut result);
    assert_eq!(result,reference,"closing control");
    eprintln!("r17 end ordinary_proof_next=true closing_equal=true");
}
