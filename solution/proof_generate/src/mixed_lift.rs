//! CRT lift for ordinary residues modulo 9721 and 4861 and the auxiliary 1024 residue.
//!
//! Inverse transforms supply centered i16 representatives. Since 9721 = 2*4861-1,
//! the ordinary CRT digit is `(a-b) mod 4861`, without a reciprocal multiply.

const N: usize = 648;
const P0: u64 = 9721;
const P1: u64 = 4861;
const PRODUCT: u64 = P0 * P1;
const INV_TAIL: u64 = 573;
const _: () = assert!(PRODUCT == 47_253_781);
const _: () = assert!((PRODUCT * INV_TAIL) & 1023 == 1);

/// Inputs are centered: |r0| <= 4860 and |r1| <= 2430. Only the low ten tail bits matter.
#[inline(always)]
pub fn lift(r0: i16, r1: i16, tail: u32) -> u64 {
    debug_assert!((-(P0 as i32 / 2)..=P0 as i32 / 2).contains(&(r0 as i32)));
    debug_assert!((-(P1 as i32 / 2)..=P1 as i32 / 2).contains(&(r1 as i32)));
    let a = if r0 < 0 { r0 as i32 + P0 as i32 } else { r0 as i32 };
    let b = if r1 < 0 { r1 as i32 + P1 as i32 } else { r1 as i32 };
    let mut digit = a - b;
    if digit < 0 { digit += P1 as i32; }
    if digit >= P1 as i32 { digit -= P1 as i32; }
    let value = a as u64 + P0 * digit as u64;
    let tail_digit = (tail as u64).wrapping_sub(value).wrapping_mul(INV_TAIL) & 1023;
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
    let inv_tail = _mm512_set1_epi64(INV_TAIL as i64);
    let product = _mm512_set1_epi64(PRODUCT as i64);
    let low10 = _mm512_set1_epi64(1023);
    let pick = _mm512_loadu_si512(LIFT_BYTES.as_ptr() as *const _);
    let keep: __mmask64 = (1u64 << 40) - 1;
    for j in 0..N / 8 {
        let a = _mm512_cvtepi16_epi64(_mm_loadu_si128(r0.as_ptr().add(8 * j) as *const __m128i));
        let a = _mm512_add_epi64(a, _mm512_and_si512(_mm512_srai_epi64::<63>(a), q0));
        let b = _mm512_cvtepi16_epi64(_mm_loadu_si128(r1.as_ptr().add(8 * j) as *const __m128i));
        let b = _mm512_add_epi64(b, _mm512_and_si512(_mm512_srai_epi64::<63>(b), q1));
        let d = _mm512_sub_epi64(a, b);
        let d = _mm512_add_epi64(d, _mm512_and_si512(_mm512_srai_epi64::<63>(d), q1));
        let d = _mm512_mask_sub_epi64(d, _mm512_cmp_epi64_mask(d, q1, _MM_CMPINT_NLT), d, q1);
        // 0 <= d < 4861, so these low-dword products retain every input bit.
        let value = _mm512_add_epi64(a, _mm512_mul_epu32(d, q0));
        let tl = _mm512_cvtepu32_epi64(_mm256_loadu_si256(tail.as_ptr().add(8 * j) as *const __m256i));
        // A negative difference wraps, but discarding bits above bit31 before
        // multiplication cannot change its low ten bits: 1024 divides 2^32.
        let tail_digit = _mm512_and_si512(_mm512_mul_epu32(_mm512_sub_epi64(tl, value), inv_tail), low10);
        let lifted = _mm512_add_epi64(value, _mm512_mul_epu32(tail_digit, product));
        _mm512_mask_storeu_epi8(out.as_mut_ptr().add(40 * j) as *mut i8, keep, _mm512_permutexvar_epi8(pick, lifted));
    }
}

#[cfg(not(target_arch = "x86_64"))]
pub unsafe fn lift_column(_: &[i16], _: &[i16], _: &[u32; N], _: &mut [u8]) {
    unreachable!("no wide lift off x86-64")
}
