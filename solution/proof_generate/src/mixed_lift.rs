//! CRT lift for centered residues modulo 9721/2917 and an arbitrary u32 tail.
//!
//! The auxiliary residue is interpreted modulo 1280. All 34 product bits are
//! retained before reduction; wrapping a negative difference modulo 2^32 would
//! change its residue modulo five and is not used here.

const N: usize = 648;
const P0: u64 = 9721;
const P1: u64 = 2917;
const PRODUCT: u64 = P0 * P1;
const AUX: u64 = 1280;
const INV_ORDINARY: u64 = 833;
const INV_TAIL: u64 = 533;
const POSITIVE_OFFSET: u64 = PRODUCT.div_ceil(AUX) * AUX;
const MAX_DELTA: u64 = POSITIVE_OFFSET + AUX - 1;
const MAX_PRODUCT: u64 = MAX_DELTA * INV_TAIL;
const RECIP5_CEIL: u64 = ((1u64 << 32) + 4) / 5;
const _: () = {
    assert!(PRODUCT == 28_356_157);
    assert!(PRODUCT * AUX == 36_295_880_960);
    assert!(P0 * INV_ORDINARY % P1 == 1);
    assert!(PRODUCT * INV_TAIL % AUX == 1);
    assert!(POSITIVE_OFFSET == 28_357_120 && POSITIVE_OFFSET % AUX == 0);
    assert!(POSITIVE_OFFSET >= PRODUCT - 1);
    assert!(MAX_DELTA < (1u64 << 32));
    assert!(MAX_PRODUCT == 15_115_026_667 && MAX_PRODUCT < (1u64 << 34));
    assert!((MAX_PRODUCT >> 8) < (1u64 << 30));
    assert!((u32::MAX as u64 >> 8) < (1u64 << 30));
    assert!(RECIP5_CEIL * 5 == (1u64 << 32) + 4);
    assert!(4 * P1 > P0 - 1);
    assert!((5 * P1 - 1) * INV_ORDINARY < (1u64 << 24));
};

/// Exact reduction for x < 2^38; every production argument is below 2^34.
///
/// Put h=x>>8=5k+r. Multiplication by ceil(2^32/5) gives quotient k:
/// 0 <= r*2^32+4h < 5*2^32, since r<=4 and h<2^30. Therefore
/// q=(h*RECIP5_CEIL)>>32 is floor(h/5), without a correction. The low
/// byte plus 256*(h-5q) is the canonical residue modulo 1280.
#[inline(always)]
pub(crate) fn reduce1280_bounded(x: u64) -> u64 {
    debug_assert!((x >> 8) < (1u64 << 30));
    let h = x >> 8;
    let quotient = (h * RECIP5_CEIL) >> 32;
    let rem5 = h - 5 * quotient;
    (x & 255) + (rem5 << 8)
}

/// Inputs are centered: |r0|<=4860 and |r1|<=1458. Any u32 tail is accepted.
#[inline(always)]
pub fn lift(r0: i16, r1: i16, tail: u32) -> u64 {
    debug_assert!((-(P0 as i32 / 2)..=P0 as i32 / 2).contains(&(r0 as i32)));
    debug_assert!((-(P1 as i32 / 2)..=P1 as i32 / 2).contains(&(r1 as i32)));
    let a = if r0 < 0 { r0 as i32 + P0 as i32 } else { r0 as i32 };
    let b = if r1 < 0 { r1 as i32 + P1 as i32 } else { r1 as i32 };
    let digit = ((b - a) * INV_ORDINARY as i32).rem_euclid(P1 as i32) as u64;
    let value = a as u64 + P0 * digit;
    let canonical_tail = reduce1280_bounded(tail as u64);
    let delta = canonical_tail + POSITIVE_OFFSET - value;
    let tail_digit = reduce1280_bounded(delta * INV_TAIL);
    value + PRODUCT * tail_digit
}

#[cfg(target_arch = "x86_64")]
static LIFT_BYTES: [u8; 64] = {
    let mut table = [0u8; 64];
    let mut k = 0;
    while k < 40 {
        table[k] = ((k / 5) * 8 + k % 5) as u8;
        k += 1;
    }
    table
};

/// Eight exact bounded reductions using low-dword multiplication only after
/// shifting the full u64 input. The shifted value is below 2^30.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn reduce1280_wide(x: core::arch::x86_64::__m512i) -> core::arch::x86_64::__m512i {
    use core::arch::x86_64::*;
    let high = _mm512_srli_epi64::<8>(x);
    let quotient = _mm512_srli_epi64::<32>(_mm512_mul_epu32(high, _mm512_set1_epi64(RECIP5_CEIL as i64)));
    let times5 = _mm512_add_epi64(quotient, _mm512_slli_epi64::<2>(quotient));
    let rem5 = _mm512_sub_epi64(high, times5);
    _mm512_add_epi64(_mm512_and_si512(x, _mm512_set1_epi64(255)), _mm512_slli_epi64::<8>(rem5))
}

/// Lift and serialize a complete column as five-byte little-endian coefficients.
///
/// # Safety
/// AVX-512 F/BW/VBMI; r0/r1 have 648 centered entries and out has 5*648 bytes.
/// Every u32 tail is interpreted modulo 1280, including noncanonical values.
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
    let positive_offset = _mm512_set1_epi64(POSITIVE_OFFSET as i64);
    let product = _mm512_set1_epi64(PRODUCT as i64);
    let pick = _mm512_loadu_si512(LIFT_BYTES.as_ptr() as *const _);
    let keep: __mmask64 = (1u64 << 40) - 1;
    for j in 0..N / 8 {
        let a = _mm512_cvtepi16_epi64(_mm_loadu_si128(r0.as_ptr().add(8 * j) as *const __m128i));
        let a = _mm512_add_epi64(a, _mm512_and_si512(_mm512_srai_epi64::<63>(a), q0));
        let b = _mm512_cvtepi16_epi64(_mm_loadu_si128(r1.as_ptr().add(8 * j) as *const __m128i));
        let b = _mm512_add_epi64(b, _mm512_and_si512(_mm512_srai_epi64::<63>(b), q1));
        // Same ordinary CRT as C16: positive x<2^24 and one reciprocal correction.
        let x = _mm512_mul_epu32(_mm512_add_epi64(_mm512_sub_epi64(b, a), offset), inv);
        let quotient = _mm512_srli_epi64::<32>(_mm512_mul_epu32(x, reciprocal));
        let digit = _mm512_sub_epi64(x, _mm512_mul_epu32(quotient, q1));
        let digit = _mm512_mask_sub_epi64(digit, _mm512_cmp_epi64_mask(digit, q1, _MM_CMPINT_NLT), digit, q1);
        let value = _mm512_add_epi64(a, _mm512_mul_epu32(digit, q0));
        let tail = _mm512_cvtepu32_epi64(_mm256_loadu_si256(tail.as_ptr().add(8 * j) as *const __m256i));
        let tail = reduce1280_wide(tail);
        let delta = _mm512_sub_epi64(_mm512_add_epi64(tail, positive_offset), value);
        // delta<2^32, but delta*533 needs 34 bits. VPMULUDQ retains all64 result bits.
        let scaled = _mm512_mul_epu32(delta, inv_tail);
        let tail_digit = reduce1280_wide(scaled);
        let lifted = _mm512_add_epi64(value, _mm512_mul_epu32(tail_digit, product));
        _mm512_mask_storeu_epi8(out.as_mut_ptr().add(40 * j) as *mut i8, keep, _mm512_permutexvar_epi8(pick, lifted));
    }
}

#[cfg(not(target_arch = "x86_64"))]
pub unsafe fn lift_column(_: &[i16], _: &[i16], _: &[u32; N], _: &mut [u8]) {
    unreachable!("no wide lift off x86-64")
}
