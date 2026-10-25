//! CRT lift for ordinary residues modulo 9721 and 2917 and the auxiliary 2048 residue.
//!
//! Inverse transforms supply centered i16 representatives. The ordinary CRT
//! inverse is833 modulo2917; the auxiliary inverse is277 modulo2048.

const N: usize = 648;
const P0: u64 = 9721;
const P1: u64 = 2917;
const PRODUCT: u64 = P0 * P1;
const INV_ORDINARY: u64 = 833;
const INV_TAIL: u64 = 277;
const _: () = assert!(PRODUCT == 28_356_157);
const _: () = assert!(P0 * INV_ORDINARY % P1 == 1);
const _: () = assert!((PRODUCT * INV_TAIL) & 2047 == 1);

/// Inputs are centered: |r0| <= 4860 and |r1| <= 1458. Only eleven tail bits matter.
#[inline(always)]
pub fn lift(r0: i16, r1: i16, tail: u32) -> u64 {
    debug_assert!((-(P0 as i32 / 2)..=P0 as i32 / 2).contains(&(r0 as i32)));
    debug_assert!((-(P1 as i32 / 2)..=P1 as i32 / 2).contains(&(r1 as i32)));
    let a = if r0 < 0 { r0 as i32 + P0 as i32 } else { r0 as i32 };
    let b = if r1 < 0 { r1 as i32 + P1 as i32 } else { r1 as i32 };
    let digit = ((b-a)*INV_ORDINARY as i32).rem_euclid(P1 as i32) as u64;
    let value = a as u64 + P0 * digit;
    let tail_digit = (tail as u64).wrapping_sub(value).wrapping_mul(INV_TAIL) & 2047;
    value + PRODUCT * tail_digit
}

#[cfg(target_arch = "x86_64")]
static LIFT_BYTES: [u8; 64] = {
    let mut t = [0u8; 64];
    let mut k = 0;
    while k < 40 {
        t[k] = ((k / 5) * 8 + k % 5) as u8;
        k += 1;
    }
    t
};

/// Lift and serialize one complete column as five-byte little-endian coefficients.
///
/// # Safety
/// Requires AVX-512 F, BW and VBMI. Residues must satisfy [`lift`]'s centered bounds;
/// both residue slices contain 648 elements and `out` contains 5*648 writable bytes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi")]
pub unsafe fn lift_column(r0: &[i16], r1: &[i16], tail: &[u32; N], out: &mut [u8]) {
    use core::arch::x86_64::*;
    assert!(r0.len() == N && r1.len() == N && out.len() == 5 * N && N % 8 == 0);
    let q0 = _mm512_set1_epi64(P0 as i64);
    let q1 = _mm512_set1_epi64(P1 as i64);
    let inv = _mm512_set1_epi64(INV_ORDINARY as i64);
    let reciprocal = _mm512_set1_epi64(((1u64 << 32) / P1) as i64);
    let offset = _mm512_set1_epi64((4 * P1) as i64);
    let inv_tail = _mm512_set1_epi64(INV_TAIL as i64);
    let product = _mm512_set1_epi64(PRODUCT as i64);
    let low11 = _mm512_set1_epi64(2047);
    let pick = _mm512_loadu_si512(LIFT_BYTES.as_ptr() as *const _);
    let keep: __mmask64 = (1u64 << 40) - 1;
    for j in 0..N / 8 {
        let a = _mm512_cvtepi16_epi64(_mm_loadu_si128(r0.as_ptr().add(8 * j) as *const __m128i));
        let a = _mm512_add_epi64(a, _mm512_and_si512(_mm512_srai_epi64::<63>(a), q0));
        let b = _mm512_cvtepi16_epi64(_mm_loadu_si128(r1.as_ptr().add(8 * j) as *const __m128i));
        let b = _mm512_add_epi64(b, _mm512_and_si512(_mm512_srai_epi64::<63>(b), q1));
        //0<=a<9721,0<=b<2917, hence0<(b-a+4*2917)*833<2^24.
        let x = _mm512_mul_epu32(_mm512_add_epi64(_mm512_sub_epi64(b,a),offset),inv);
        //Floor reciprocal division is low by at most one for x<2^32.
        let quotient = _mm512_srli_epi64::<32>(_mm512_mul_epu32(x,reciprocal));
        let d = _mm512_sub_epi64(x,_mm512_mul_epu32(quotient,q1));
        let d = _mm512_mask_sub_epi64(d,_mm512_cmp_epi64_mask(d,q1,_MM_CMPINT_NLT),d,q1);
        //The corrected digit is canonical, and all low-dword products fit.
        let value = _mm512_add_epi64(a, _mm512_mul_epu32(d, q0));
        let tl = _mm512_cvtepu32_epi64(_mm256_loadu_si256(tail.as_ptr().add(8 * j) as *const __m256i));
        // A negative difference wraps, but discarding bits above bit31 before
        // multiplication cannot change its low eleven bits: 2048 divides 2^32.
        let tail_digit = _mm512_and_si512(_mm512_mul_epu32(_mm512_sub_epi64(tl, value), inv_tail), low11);
        let lifted = _mm512_add_epi64(value, _mm512_mul_epu32(tail_digit, product));
        _mm512_mask_storeu_epi8(out.as_mut_ptr().add(40 * j) as *mut i8, keep, _mm512_permutexvar_epi8(pick, lifted));
    }
}

#[cfg(not(target_arch = "x86_64"))]
pub unsafe fn lift_column(_: &[i16], _: &[i16], _: &[u32; N], _: &mut [u8]) {
    unreachable!("no wide lift off x86-64")
}
