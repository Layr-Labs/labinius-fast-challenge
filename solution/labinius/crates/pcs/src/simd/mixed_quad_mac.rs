//! Rank-three, two-stream quadratic MAC for the 4861 hybrid limb.
//!
//! The binary quadratic NTT stores leaf j in rows 2j and 2j+1. Each
//! vector's adjacent i16 lanes are two ring elements of one of 16 columns.
//! Public key form 2 is centered(k0+k1); this preprocessing is essential
//! to the declared sum-class fold-back period.
use crate::params::{barrett_v, Params, ParamsQ};
use crate::ring::{Batch32, Representation};
use crate::simd::ntt::{bin_quad, gen_quad};
use core::arch::x86_64::*;

const Q: u16 = 4861;
pub const LEAVES: usize = 324;
/// Per LOCK2 step: [leaf][form][base_h0,tail_h0,base_h1,tail_h1].
pub const KEY_WORDS: usize = LEAVES * 3 * 4;
pub const W_BOUND: i64 = bin_quad::output_bound(Q) as i64;
const A_BOUND: i64 = ((Q - 1) / 2) as i64;
const R16C: i32 = (65536 % Q as i32) - Q as i32;
const AFTER_REDUCE: i64 = 32768 * (1 + R16C.unsigned_abs() as i64);
const BIAS: i64 = ((1i64 << 31) / Q as i64) * Q as i64;
/// One pair batch adds two scalar products to each dword lane.
const PER_PAIR_DIAG: i64 = 2 * W_BOUND * A_BOUND;
/// |w0+w1|<=2W and the precentered key sum has |k|<=A.
const PER_PAIR_SUM: i64 = 4 * W_BOUND * A_BOUND;

const fn period(per_pair: i64) -> usize {
    let mut p = 1;
    while AFTER_REDUCE + 2 * p as i64 * per_pair <= BIAS {
        p *= 2;
    }
    p
}
pub const DIAG_PERIOD: usize = period(PER_PAIR_DIAG);
pub const SUM_PERIOD: usize = period(PER_PAIR_SUM);

const _: () = {
    assert!(2 * W_BOUND <= i16::MAX as i64);
    assert!(2 * A_BOUND < i16::MAX as i64);
    assert!(AFTER_REDUCE + DIAG_PERIOD as i64 * PER_PAIR_DIAG <= BIAS);
    assert!(AFTER_REDUCE + SUM_PERIOD as i64 * PER_PAIR_SUM <= BIAS);
    assert!(BIAS <= i32::MAX as i64);
    assert!(DIAG_PERIOD == 16 && SUM_PERIOD == 8);
    assert!(DIAG_PERIOD % 2 == 0 && SUM_PERIOD % 2 == 0);
    assert!(128 % DIAG_PERIOD == 0 && 128 % SUM_PERIOD == 0);
    assert!(6 * (Q as i32) < i16::MAX as i32);
    assert!(A_BOUND <= gen_quad::in_bound(Q) as i64);
    // For bv=7, rounded Barrett's exact remainder is bounded by
    // |7q-32768| + q/2 < 3690 < q over every signed i16 input.
    assert!(barrett_v(Q) == 7);
    assert!(1259 * 2 + (Q as i32) < 2 * (Q as i32));
};

/// [leaf][rank-three form][ordinary/auxiliary stream][16 columns].
#[repr(C, align(128))]
pub struct QuadPairAcc {
    pub v: [[[[i32; 16]; 2]; 3]; LEAVES],
}

const _: () = assert!(core::mem::size_of::<QuadPairAcc>() == LEAVES * 3 * 2 * 64);

impl QuadPairAcc {
    pub fn zero() -> Box<Self> {
        let mut p = Box::<Self>::new_uninit();
        // Every field is an i32 array; all-zero bits initialize every lane.
        unsafe {
            core::ptr::write_bytes(p.as_mut_ptr() as *mut u8, 0, core::mem::size_of::<Self>());
            p.assume_init()
        }
    }
}

/// x = signed_low + 2^16*(signed_high + carry(bit15)); replace
/// 2^16 by the congruent centered residue R16C.
#[inline(always)]
unsafe fn reduce_c(x: __m512i) -> __m512i {
    let word = (1u32 | ((R16C as u16 as u32) << 16)) as i32;
    let y = _mm512_madd_epi16(x, _mm512_set1_epi32(word));
    let carry = _mm512_test_epi32_mask(x, _mm512_set1_epi32(0x8000));
    _mm512_mask_add_epi32(y, carry, y, _mm512_set1_epi32(R16C))
}

#[inline(always)]
unsafe fn dp<const OFF: usize>(mut a: __m512i, w: __m512i, key: *const u32) -> __m512i {
    core::arch::asm!(
        "vpdpwssd {a}, {w}, dword ptr [{key} + {off}]{{1to16}}",
        a = inout(zmm_reg) a,
        w = in(zmm_reg) w,
        key = in(reg) key,
        off = const OFF,
        options(pure, readonly, nostack, preserves_flags)
    );
    a
}

#[inline(always)]
unsafe fn form<const RED: bool>(w0: __m512i, w1: __m512i, key: *const u32, acc: *mut i32) {
    let mut a = _mm512_load_si512(acc as *const __m512i);
    let mut t = _mm512_load_si512(acc.add(16) as *const __m512i);
    a = dp::<0>(a, w0, key);
    t = dp::<4>(t, w0, key);
    a = dp::<8>(a, w1, key);
    t = dp::<12>(t, w1, key);
    if RED {
        a = reduce_c(a);
        t = reduce_c(t);
    }
    _mm512_store_si512(acc as *mut __m512i, a);
    _mm512_store_si512(acc.add(16) as *mut __m512i, t);
}

#[inline(never)]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
unsafe fn accumulate<const RD: bool, const RS: bool>(
    w0: &Batch32, w1: &Batch32, key: *const u32, acc: &mut QuadPairAcc,
) {
    for j in 0..LEAVES {
        let a0 = _mm512_load_si512(w0.v[2 * j].as_ptr() as *const __m512i);
        let a1 = _mm512_load_si512(w0.v[2 * j + 1].as_ptr() as *const __m512i);
        let b0 = _mm512_load_si512(w1.v[2 * j].as_ptr() as *const __m512i);
        let b1 = _mm512_load_si512(w1.v[2 * j + 1].as_ptr() as *const __m512i);
        let k = key.add(12 * j);
        form::<RD>(a0, b0, k, acc.v[j][0].as_mut_ptr() as *mut i32);
        form::<RD>(a1, b1, k.add(4), acc.v[j][1].as_mut_ptr() as *mut i32);
        form::<RS>(
            _mm512_add_epi16(a0, a1), _mm512_add_epi16(b0, b1), k.add(8),
            acc.v[j][2].as_mut_ptr() as *mut i32,
        );
    }
}

/// Accumulate exactly two consecutive pair batches (four rings/column).
///
/// # Safety
/// The host needs AVX512F/BW/VNNI. Both batches are quadratic NTT output
/// for 4861 with |lane|<=W_BOUND. `key` covers KEY_WORDS readable dwords,
/// disjoint from the accumulator. Every signed i16 key component, including
/// the centered sum form, lies in [-2430,2430]. Calls start with zero lanes,
/// advance `b_done` by two, and obey all class-specific fold-back boundaries.
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
pub unsafe fn accumulate_lock2(
    w0: &Batch32, w1: &Batch32, key: *const u32, acc: &mut QuadPairAcc, b_done: usize,
) {
    debug_assert!(b_done > 0 && b_done % 2 == 0);
    debug_assert_eq!(w0.representation, Representation::Ntt);
    debug_assert_eq!(w1.representation, Representation::Ntt);
    if b_done % DIAG_PERIOD == 0 {
        accumulate::<true, true>(w0, w1, key, acc);
    } else if b_done % SUM_PERIOD == 0 {
        accumulate::<false, true>(w0, w1, key, acc);
    } else {
        accumulate::<false, false>(w0, w1, key, acc);
    }
}

const fn qinv16() -> u32 {
    let mut x = Q as u32;
    let mut i = 0;
    while i < 4 { x = x.wrapping_mul(2u32.wrapping_sub(Q as u32 * x)) & 0xffff; i += 1; }
    x
}

/// The existing hybrid red16 construction: for |x|<=BIAS, result is
/// congruent to x and has absolute value strictly below 2Q.
#[inline(always)]
unsafe fn red16(x: __m512i) -> __m512i {
    const WL: u32 = (1u32 << 16) % Q as u32;
    const WH: u32 = ((1u64 << 32) % Q as u64) as u32;
    const W: i32 = (WL | (WH << 16)) as i32;
    const WP: i32 = (((WL * qinv16()) & 0xffff) | (((WH * qinv16()) & 0xffff) << 16)) as i32;
    let y = _mm512_add_epi32(x, _mm512_set1_epi32(BIAS as i32));
    let m = _mm512_mullo_epi16(y, _mm512_set1_epi32(WP));
    let r = _mm512_sub_epi16(
        _mm512_mulhi_epu16(y, _mm512_set1_epi32(W)),
        _mm512_mulhi_epu16(m, _mm512_set1_epi16(Q as i16)),
    );
    _mm512_madd_epi16(r, _mm512_set1_epi16(1))
}

const LEAF_MONT: [[i16; 2]; LEAVES] = {
    let mut t = [[0; 2]; LEAVES];
    let mut j = 0;
    while j < LEAVES {
        let w = Params::<Q>::to_mont(ParamsQ::<Q>::LEAF_C[j]);
        t[j] = [w, Params::<Q>::mont_pre(w)];
        j += 1;
    }
    t
};

#[inline(always)]
unsafe fn center16(x: __m256i) -> __m256i {
    let q = _mm256_set1_epi16(Q as i16);
    let t = _mm256_mulhrs_epi16(x, _mm256_set1_epi16(barrett_v(Q)));
    let r = _mm256_sub_epi16(x, _mm256_mullo_epi16(t, q));
    let hi = _mm256_cmpgt_epi16_mask(r, _mm256_set1_epi16((Q / 2) as i16));
    let r = _mm256_mask_sub_epi16(r, hi, r, q);
    let lo = _mm256_cmplt_epi16_mask(r, _mm256_set1_epi16(-((Q / 2) as i16)));
    _mm256_mask_add_epi16(r, lo, r, q)
}

/// Finish one stream, combine the quadratic product, and clear only its
/// three classes. Write the centered quadratic NTT residues into one half
/// of `out`; the other half is retained for the caller's paired inverse.
///
/// # Safety
/// AVX512F/BW/VL. Accumulation must obey the stated periods and input bounds.
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
pub unsafe fn finish_stream(acc: &mut QuadPairAcc, stream: usize, out: &mut Batch32, half: usize) {
    assert!(stream < 2 && half < 2);
    let zero = _mm512_setzero_si512();
    for j in 0..LEAVES {
        let p0 = acc.v[j][0][stream].as_mut_ptr() as *mut __m512i;
        let p1 = acc.v[j][1][stream].as_mut_ptr() as *mut __m512i;
        let ps = acc.v[j][2][stream].as_mut_ptr() as *mut __m512i;
        let r0 = _mm512_cvtepi32_epi16(red16(_mm512_load_si512(p0)));
        let r1 = _mm512_cvtepi32_epi16(red16(_mm512_load_si512(p1)));
        let rs = _mm512_cvtepi32_epi16(red16(_mm512_load_si512(ps)));
        _mm512_store_si512(p0, zero);
        _mm512_store_si512(p1, zero);
        _mm512_store_si512(ps, zero);
        let [w, wp] = LEAF_MONT[j];
        let m = _mm256_mullo_epi16(r1, _mm256_set1_epi16(wp));
        let c1 = _mm256_sub_epi16(
            _mm256_mulhi_epi16(r1, _mm256_set1_epi16(w)),
            _mm256_mulhi_epi16(m, _mm256_set1_epi16(Q as i16)),
        );
        let y0 = center16(_mm256_add_epi16(r0, c1));
        let y1 = center16(_mm256_sub_epi16(_mm256_sub_epi16(rs, r0), r1));
        _mm256_storeu_si256(out.v[2 * j].as_mut_ptr().add(16 * half) as *mut __m256i, y0);
        _mm256_storeu_si256(out.v[2 * j + 1].as_mut_ptr().add(16 * half) as *mut __m256i, y1);
    }
    out.representation = Representation::Ntt;
}
