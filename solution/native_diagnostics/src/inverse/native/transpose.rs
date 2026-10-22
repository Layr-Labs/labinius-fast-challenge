use std::arch::x86_64::*;
#[inline(always)]
pub(crate) unsafe fn transpose8x8_q(r: [__m512i; 8]) -> [__m512i; 8] {
    let a0 = _mm512_unpacklo_epi64(r[0], r[1]);
    let a1 = _mm512_unpackhi_epi64(r[0], r[1]);
    let a2 = _mm512_unpacklo_epi64(r[2], r[3]);
    let a3 = _mm512_unpackhi_epi64(r[2], r[3]);
    let a4 = _mm512_unpacklo_epi64(r[4], r[5]);
    let a5 = _mm512_unpackhi_epi64(r[4], r[5]);
    let a6 = _mm512_unpacklo_epi64(r[6], r[7]);
    let a7 = _mm512_unpackhi_epi64(r[6], r[7]);
    let b0 = _mm512_shuffle_i64x2::<0x88>(a0, a2);
    let b1 = _mm512_shuffle_i64x2::<0xDD>(a0, a2);
    let b2 = _mm512_shuffle_i64x2::<0x88>(a4, a6);
    let b3 = _mm512_shuffle_i64x2::<0xDD>(a4, a6);
    let b4 = _mm512_shuffle_i64x2::<0x88>(a1, a3);
    let b5 = _mm512_shuffle_i64x2::<0xDD>(a1, a3);
    let b6 = _mm512_shuffle_i64x2::<0x88>(a5, a7);
    let b7 = _mm512_shuffle_i64x2::<0xDD>(a5, a7);
    [
        _mm512_shuffle_i64x2::<0x88>(b0, b2),
        _mm512_shuffle_i64x2::<0x88>(b4, b6),
        _mm512_shuffle_i64x2::<0x88>(b1, b3),
        _mm512_shuffle_i64x2::<0x88>(b5, b7),
        _mm512_shuffle_i64x2::<0xDD>(b0, b2),
        _mm512_shuffle_i64x2::<0xDD>(b4, b6),
        _mm512_shuffle_i64x2::<0xDD>(b1, b3),
        _mm512_shuffle_i64x2::<0xDD>(b5, b7),
    ]
}
