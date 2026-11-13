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
    // The row pairing is applied by the row emitter itself (one load of the earlier
    // partner per pair) instead of a separate pass over rows 27..81.
    slice_b128_cols16_sms_r3pair_rows_into(
        cols,
        &mut *(out as *mut SmsR3PairIndex32 as *mut SmsBinaryIndex32),
    );
}
#[cfg(test)]
mod r3pair_index_guard {
    use super::*;
    /// The fused r3pair emitter equals the base constructor followed by the former separate
    /// pairing pass, on one-hot, constant and random columns including zero-padded short groups.
    #[test]
    fn fused_pairing_matches_separate_pass() {
        unsafe {
            let mut state = 0x3c6e_f372_fe94_f82bu128;
            for fixture in 0..520 {
                let mut columns = [[crate::fields::scalar::B128(0); 8]; 16];
                let live = if fixture % 7 == 0 { 1 + fixture % 16 } else { 16 };
                for col in 0..live {
                    for w in 0..8 {
                        columns[col][w].0 = if fixture < 512 {
                            if w == fixture % 8 { 1u128 << (fixture / 4) } else { 0 }
                        } else {
                            state ^= state << 13;
                            state ^= state >> 7;
                            state ^= state << 17;
                            match fixture { 512 => 0, 513 => u128::MAX, _ => state }
                        };
                    }
                }
                let ptrs = core::array::from_fn(|i| columns[i].as_ptr());
                let mut want = SmsR3PairIndex32 { rows: [[0x5a5a; 32]; 81] };
                slice_b128_cols16_sms_into(&ptrs, &mut *(&mut want as *mut SmsR3PairIndex32 as *mut SmsBinaryIndex32));
                let mask = _mm512_set1_epi16(0x15);
                let swap = _mm512_set1_epi64(0x0201080420108040);
                for i in 0..27 {
                    let a = _mm512_load_si512(want.rows[i + 27].as_ptr().cast());
                    let b = _mm512_load_si512(want.rows[i + 54].as_ptr().cast());
                    let q = _mm512_ternarylogic_epi64::<0xe4>(a, b, mask);
                    let v = _mm512_ternarylogic_epi64::<0xe4>(b, a, mask);
                    _mm512_store_si512(want.rows[i + 27].as_mut_ptr().cast(), _mm512_gf2p8affine_epi64_epi8::<0>(v, swap));
                    _mm512_store_si512(want.rows[i + 54].as_mut_ptr().cast(), q);
                }
                let mut got = SmsR3PairIndex32 { rows: [[0xa5a5; 32]; 81] };
                slice_b128_cols16_r3pair_into(&ptrs, &mut got);
                assert!(got.rows == want.rows, "fixture={fixture}");
            }
        }
    }
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
