//! Byte transposes of 64-word tiles, so that `vgf2p8affineqb` — `out.byte[t].bit[i] =
//! <A.byte[7 - i], x.byte[t]>`, an 8x8x8 block product per qword — sees eight different words in
//! each qword. The byte order inside a tile is the scramble [`word_at`].
use core::arch::x86_64::*;

/// `affine(ident, D)` is the bit transpose of each qword of `D`.
pub const GF_IDENT: i64 = 0x8040_2010_0804_0201u64 as i64;

/// The word at byte position `p` of a transposed tile.
#[inline]
pub fn word_at(p: usize) -> usize {
    (((p >> 5) & 1) << 1) | ((p >> 4) & 1) | ((p & 0xf) << 2)
}

/// `out[k].byte[p]` = byte `k` of word `word_at(p)` of the 64 words at `src`.
///
/// Within each 128-bit lane `L` (`0..4`), `word_at(16*L + r) = 4*r + L`, which is lane `L` of
/// input vector `r` (`0..16`). A 4-round `vpunpcklbw`/`vpunpckhbw` butterfly across strides
/// `8, 4, 2, 1` transposes the 16x16 bytes inside each 128-bit lane in 64 single-uOp instructions
/// with zero index registers.
#[inline(always)]
pub unsafe fn transpose_tile(src: *const u8, out: &mut [__m512i; 16]) {
    let z0 = _mm512_loadu_si512(src as *const __m512i);
    let z1 = _mm512_loadu_si512(src.add(64) as *const __m512i);
    let z2 = _mm512_loadu_si512(src.add(128) as *const __m512i);
    let z3 = _mm512_loadu_si512(src.add(192) as *const __m512i);
    let z4 = _mm512_loadu_si512(src.add(256) as *const __m512i);
    let z5 = _mm512_loadu_si512(src.add(320) as *const __m512i);
    let z6 = _mm512_loadu_si512(src.add(384) as *const __m512i);
    let z7 = _mm512_loadu_si512(src.add(448) as *const __m512i);
    let z8 = _mm512_loadu_si512(src.add(512) as *const __m512i);
    let z9 = _mm512_loadu_si512(src.add(576) as *const __m512i);
    let z10 = _mm512_loadu_si512(src.add(640) as *const __m512i);
    let z11 = _mm512_loadu_si512(src.add(704) as *const __m512i);
    let z12 = _mm512_loadu_si512(src.add(768) as *const __m512i);
    let z13 = _mm512_loadu_si512(src.add(832) as *const __m512i);
    let z14 = _mm512_loadu_si512(src.add(896) as *const __m512i);
    let z15 = _mm512_loadu_si512(src.add(960) as *const __m512i);

    let a0 = _mm512_unpacklo_epi8(z0, z8);
    let a8 = _mm512_unpackhi_epi8(z0, z8);
    let a1 = _mm512_unpacklo_epi8(z1, z9);
    let a9 = _mm512_unpackhi_epi8(z1, z9);
    let a2 = _mm512_unpacklo_epi8(z2, z10);
    let a10 = _mm512_unpackhi_epi8(z2, z10);
    let a3 = _mm512_unpacklo_epi8(z3, z11);
    let a11 = _mm512_unpackhi_epi8(z3, z11);
    let a4 = _mm512_unpacklo_epi8(z4, z12);
    let a12 = _mm512_unpackhi_epi8(z4, z12);
    let a5 = _mm512_unpacklo_epi8(z5, z13);
    let a13 = _mm512_unpackhi_epi8(z5, z13);
    let a6 = _mm512_unpacklo_epi8(z6, z14);
    let a14 = _mm512_unpackhi_epi8(z6, z14);
    let a7 = _mm512_unpacklo_epi8(z7, z15);
    let a15 = _mm512_unpackhi_epi8(z7, z15);

    let b0 = _mm512_unpacklo_epi8(a0, a4);
    let b4 = _mm512_unpackhi_epi8(a0, a4);
    let b1 = _mm512_unpacklo_epi8(a1, a5);
    let b5 = _mm512_unpackhi_epi8(a1, a5);
    let b2 = _mm512_unpacklo_epi8(a2, a6);
    let b6 = _mm512_unpackhi_epi8(a2, a6);
    let b3 = _mm512_unpacklo_epi8(a3, a7);
    let b7 = _mm512_unpackhi_epi8(a3, a7);
    let b8 = _mm512_unpacklo_epi8(a8, a12);
    let b12 = _mm512_unpackhi_epi8(a8, a12);
    let b9 = _mm512_unpacklo_epi8(a9, a13);
    let b13 = _mm512_unpackhi_epi8(a9, a13);
    let b10 = _mm512_unpacklo_epi8(a10, a14);
    let b14 = _mm512_unpackhi_epi8(a10, a14);
    let b11 = _mm512_unpacklo_epi8(a11, a15);
    let b15 = _mm512_unpackhi_epi8(a11, a15);

    let c0 = _mm512_unpacklo_epi8(b0, b2);
    let c2 = _mm512_unpackhi_epi8(b0, b2);
    let c1 = _mm512_unpacklo_epi8(b1, b3);
    let c3 = _mm512_unpackhi_epi8(b1, b3);
    let c4 = _mm512_unpacklo_epi8(b4, b6);
    let c6 = _mm512_unpackhi_epi8(b4, b6);
    let c5 = _mm512_unpacklo_epi8(b5, b7);
    let c7 = _mm512_unpackhi_epi8(b5, b7);
    let c8 = _mm512_unpacklo_epi8(b8, b10);
    let c10 = _mm512_unpackhi_epi8(b8, b10);
    let c9 = _mm512_unpacklo_epi8(b9, b11);
    let c11 = _mm512_unpackhi_epi8(b9, b11);
    let c12 = _mm512_unpacklo_epi8(b12, b14);
    let c14 = _mm512_unpackhi_epi8(b12, b14);
    let c13 = _mm512_unpacklo_epi8(b13, b15);
    let c15 = _mm512_unpackhi_epi8(b13, b15);

    out[0] = _mm512_unpacklo_epi8(c0, c1);
    out[1] = _mm512_unpackhi_epi8(c0, c1);
    out[2] = _mm512_unpacklo_epi8(c2, c3);
    out[3] = _mm512_unpackhi_epi8(c2, c3);
    out[4] = _mm512_unpacklo_epi8(c4, c5);
    out[5] = _mm512_unpackhi_epi8(c4, c5);
    out[6] = _mm512_unpacklo_epi8(c6, c7);
    out[7] = _mm512_unpackhi_epi8(c6, c7);
    out[8] = _mm512_unpacklo_epi8(c8, c9);
    out[9] = _mm512_unpackhi_epi8(c8, c9);
    out[10] = _mm512_unpacklo_epi8(c10, c11);
    out[11] = _mm512_unpackhi_epi8(c10, c11);
    out[12] = _mm512_unpacklo_epi8(c12, c13);
    out[13] = _mm512_unpackhi_epi8(c12, c13);
    out[14] = _mm512_unpacklo_epi8(c14, c15);
    out[15] = _mm512_unpackhi_epi8(c14, c15);
}

/// `planes[c].byte[p]` = byte `c` of word `word_at(p)` becomes `out[g].qword[i].byte[c]` of word
/// `8g + i`.
#[inline(always)]
pub unsafe fn untranspose_planes(planes: [__m512i; 8]) -> [__m512i; 8] {
    let a0 = _mm512_unpacklo_epi8(planes[0], planes[4]);
    let a4 = _mm512_unpackhi_epi8(planes[0], planes[4]);
    let a1 = _mm512_unpacklo_epi8(planes[1], planes[5]);
    let a5 = _mm512_unpackhi_epi8(planes[1], planes[5]);
    let a2 = _mm512_unpacklo_epi8(planes[2], planes[6]);
    let a6 = _mm512_unpackhi_epi8(planes[2], planes[6]);
    let a3 = _mm512_unpacklo_epi8(planes[3], planes[7]);
    let a7 = _mm512_unpackhi_epi8(planes[3], planes[7]);

    let b0 = _mm512_unpacklo_epi8(a0, a2);
    let b2 = _mm512_unpackhi_epi8(a0, a2);
    let b1 = _mm512_unpacklo_epi8(a1, a3);
    let b3 = _mm512_unpackhi_epi8(a1, a3);
    let b4 = _mm512_unpacklo_epi8(a4, a6);
    let b6 = _mm512_unpackhi_epi8(a4, a6);
    let b5 = _mm512_unpacklo_epi8(a5, a7);
    let b7 = _mm512_unpackhi_epi8(a5, a7);

    let c0 = _mm512_unpacklo_epi8(b0, b1);
    let c1 = _mm512_unpackhi_epi8(b0, b1);
    let c2 = _mm512_unpacklo_epi8(b2, b3);
    let c3 = _mm512_unpackhi_epi8(b2, b3);
    let c4 = _mm512_unpacklo_epi8(b4, b5);
    let c5 = _mm512_unpackhi_epi8(b4, b5);
    let c6 = _mm512_unpacklo_epi8(b6, b7);
    let c7 = _mm512_unpackhi_epi8(b6, b7);

    let fix = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);
    [
        _mm512_permutexvar_epi64(fix, c0),
        _mm512_permutexvar_epi64(fix, c1),
        _mm512_permutexvar_epi64(fix, c2),
        _mm512_permutexvar_epi64(fix, c3),
        _mm512_permutexvar_epi64(fix, c4),
        _mm512_permutexvar_epi64(fix, c5),
        _mm512_permutexvar_epi64(fix, c6),
        _mm512_permutexvar_epi64(fix, c7),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tile_transpose_and_back() {
        let words: Vec<u128> = (0..64u128)
            .map(|i| (i * 0x0123_4567_89AB_CDEF_1122_3344_5566_7788u128) ^ (i << 100))
            .collect();
        let mut t = [unsafe { _mm512_setzero_si512() }; 16];
        unsafe { transpose_tile(words.as_ptr() as *const u8, &mut t) };
        let bytes: Vec<[u8; 64]> = t
            .iter()
            .map(|&z| unsafe { core::mem::transmute::<__m512i, [u8; 64]>(z) })
            .collect();
        for k in 0..16 {
            for p in 0..64 {
                assert_eq!(bytes[k][p], words[word_at(p)].to_le_bytes()[k], "k {k} p {p}");
            }
        }
        let planes: [__m512i; 8] = core::array::from_fn(|c| t[c]);
        let back = unsafe { untranspose_planes(planes) };
        for g in 0..8 {
            let q: [u64; 8] = unsafe { core::mem::transmute(back[g]) };
            for i in 0..8 {
                assert_eq!(q[i], words[8 * g + i] as u64, "g {g} i {i}");
            }
        }
    }
}
