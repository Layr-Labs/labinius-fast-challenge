#[repr(C, align(64))]
pub(crate) struct SmsR3PairIndex32 {
    pub(crate) rows: [[u16; 32]; 81],
}
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,gfni")]
pub(crate) unsafe fn slice_b128_cols16_r3pair_into(
    cols: &[*const crate::fields::scalar::B128; 16],
    out: &mut SmsR3PairIndex32,
) {
    slice_b128_cols16_sms_into(
        cols,
        &mut *(out as *mut SmsR3PairIndex32 as *mut SmsBinaryIndex32),
    );
    let mask = _mm512_set1_epi16(0x15);
    let swap = _mm512_set1_epi64(0x0201080420108040);
    for i in 0..27 {
        let a = _mm512_load_si512(out.rows[i + 27].as_ptr().cast());
        let b = _mm512_load_si512(out.rows[i + 54].as_ptr().cast());
        let q = _mm512_ternarylogic_epi64::<0xe4>(a, b, mask);
        let v = _mm512_ternarylogic_epi64::<0xe4>(b, a, mask);
        let p = _mm512_gf2p8affine_epi64_epi8::<0>(v, swap);
        _mm512_store_si512(out.rows[i + 27].as_mut_ptr().cast(), p);
        _mm512_store_si512(out.rows[i + 54].as_mut_ptr().cast(), q);
    }
}
#[cfg(test)]
mod r3pair_index_guard {
    use super::*;
    #[test]
    fn exhaustive_gfni_and_pair_tags() {
        unsafe {
            for base in (0..256).step_by(64) {
                let x: [u8; 64] = core::array::from_fn(|i| (base + i) as u8);
                let v = _mm512_loadu_si512(x.as_ptr().cast());
                let y =
                    _mm512_gf2p8affine_epi64_epi8::<0>(v, _mm512_set1_epi64(0x0201080420108040));
                let got: [u8; 64] = core::mem::transmute(y);
                for i in 0..64 {
                    assert_eq!(got[i], ((x[i] & 0x55) << 1) | ((x[i] & 0xaa) >> 1));
                }
            }
            for a in 0..64u16 {
                for b in 0..64u16 {
                    let av = _mm512_set1_epi16(a as i16);
                    let bv = _mm512_set1_epi16(b as i16);
                    let m = _mm512_set1_epi16(0x15);
                    let q = _mm512_ternarylogic_epi64::<0xe4>(av, bv, m);
                    let v = _mm512_ternarylogic_epi64::<0xe4>(bv, av, m);
                    let pv = _mm512_gf2p8affine_epi64_epi8::<0>(
                        v,
                        _mm512_set1_epi64(0x0201080420108040),
                    );
                    let pp: [u16; 32] = core::mem::transmute(pv);
                    let qq: [u16; 32] = core::mem::transmute(q);
                    assert!(pp
                        .iter()
                        .all(|&x| x == ((a >> 1) & 0x15) | ((b & 0x15) << 1)));
                    assert!(qq.iter().all(|&x| x == (a & 0x15) | (b & 0x2a)));
                }
            }
        }
    }
}
