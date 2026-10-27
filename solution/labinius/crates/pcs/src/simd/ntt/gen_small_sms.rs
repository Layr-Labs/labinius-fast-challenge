//! Fuse public-mean subtraction and digit accumulation into the inverse's last pass.
use super::*;

const P: u16 = 9721;
const N: usize = 648;

/// For the sum output, KB^-1 = 648. For the difference output,
/// KA^-1 = 324 * (2*zeta6 - 1). Offsets are centered residues.
pub fn sms_inverse_biases(means: &[i16]) -> Vec<i16> {
    assert_eq!(means.len() % N, 0);
    let inverse_ka = 324 * inv_mod(TwI::<P>::DET as u64, P as u64) % P as u64;
    means.iter().enumerate().map(|(j, &mean)| {
        let factor = if j % N < N / 2 { 648 } else { inverse_ka as i64 };
        let r = (-(mean as i64) * factor).rem_euclid(P as i64);
        if r > (P as i64 - 1) / 2 { (r - P as i64) as i16 } else { r as i16 }
    }).collect()
}

/// Scalar regression using the independent lane-wise Montgomery implementation.
/// Exercises every public mean, including extremal pre-normalization lanes.
pub fn sms_affine_scalar_check() -> usize {
    let half = (P as i32 - 1) / 2;
    let limit = 2 * TwI::<P>::BOUND[1];
    let mul = |x: i32, pair: [u32; 2]| -> i32 {
        crate::params::mont_mul_i16(i16::try_from(x).unwrap(), pair[1] as i16,
            pair[0] as i16, P) as i32
    };
    let centered = |x: i32| (x + half).rem_euclid(P as i32) - half;
    let once = |x: i32| {
        assert!(x.abs() <= (3 * P as i32 - 1) / 2);
        let x = if x > half { x - P as i32 } else { x };
        if x < -half { x + P as i32 } else { x }
    };
    let mut state = 0x2539_71abu64;
    let mut count = 0;
    for mean in -half..=half {
        let offsets = sms_inverse_biases(&vec![mean as i16; N]);
        state ^= state << 13; state ^= state >> 7; state ^= state << 17;
        let random = (state % (2 * limit + 1) as u64) as i32 - limit;
        for (s, d) in [(0, 0), (limit, -limit), (-limit, limit),
            (limit, limit), (-limit, -limit), (random, -random)] {
            let expected0 = centered(mul(s, TwI::<P>::KB) + mul(d, TwI::<P>::KC) - mean);
            let expected1 = centered(mul(d, TwI::<P>::KA) - mean);
            let got0 = once(mul(s + offsets[0] as i32, TwI::<P>::KB) + mul(d, TwI::<P>::KC));
            let got1 = once(mul(d + offsets[324] as i32, TwI::<P>::KA));
            assert_eq!(got0, expected0, "sum output, mean {mean}, s {s}, d {d}");
            assert_eq!(got1, expected1, "difference output, mean {mean}, d {d}");
            count += 2;
        }
    }
    count
}

#[cfg(test)]
mod tests {
    #[test]
    fn normalization_offsets_match_centered_reference() {
        assert_eq!(super::sms_affine_scalar_check(), 9721 * 6 * 2);
    }
}

// Before normalization, a sum/difference is at most twice the level-1 bound.
// The added centered offset fits i16. Even bounding both Montgomery products
// using the larger input leaves the sum inside center()'s 3q/2 contract.
const _: () = {
    let biased = 2 * TwI::<P>::BOUND[1] + (P as i32 - 1) / 2;
    assert!(biased <= i16::MAX as i32);
    assert!(2 * mont_bound(biased, P) <= (3 * P as i32 - 1) / 2);
};

#[inline(always)]
unsafe fn mean_offset(lo: *const i16, hi: *const i16, j: usize) -> __m512i {
    _mm512_mask_blend_epi16(0xffff_0000,
        _mm512_set1_epi16(*lo.add(j)), _mm512_set1_epi16(*hi.add(j)))
}

#[inline(always)]
unsafe fn accumulate<const BOTH: bool>(x: __m512i, j: usize, shift: __m128i, total: *mut [i32; 32]) {
    let dst = (*total.add(j)).as_mut_ptr() as *mut __m512i;
    let lo = _mm512_sll_epi32(_mm512_cvtepi16_epi32(_mm512_castsi512_si256(x)), shift);
    _mm512_storeu_si512(dst, _mm512_add_epi32(_mm512_loadu_si512(dst), lo));
    if BOTH {
        let hi = _mm512_sll_epi32(_mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(x)), shift);
        _mm512_storeu_si512(dst.add(1), _mm512_add_epi32(_mm512_loadu_si512(dst.add(1)), hi));
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
unsafe fn final_pass<const BOTH: bool>(p: *mut __m512i, lo: *const i16, hi: *const i16,
    shift: u32, total: *mut [i32; 32]) {
    let q = _mm512_set1_epi32(Tw::<P>::QD as i32);
    let bv = _mm512_set1_epi32(Tw::<P>::BV as i32);
    let half = _mm512_set1_epi32(TwI::<P>::HALF as i32);
    let nhalf = _mm512_set1_epi32(TwI::<P>::NHALF as i32);
    let l1 = TwI::<P>::IL1.as_ptr();
    let (zap, za) = (bc(l1), bc(l1.add(1)));
    let (zbp, zb) = (bc(l1.add(2)), bc(l1.add(3)));
    let ka = TwI::<P>::KA.as_ptr();
    let kb = TwI::<P>::KB.as_ptr();
    let kc = TwI::<P>::KC.as_ptr();
    let shift = _mm_cvtsi32_si128(shift as i32);
    for i in 0..162 {
        let (c0, c1) = ir2(ld(p, i), ld(p, i + 162), zap, za, q);
        let (c2, c3) = ir2(ld(p, i + 324), ld(p, i + 486), zbp, zb, q);
        let c0 = barrett(c0, bv, q);
        let c2 = barrett(c2, bv, q);
        let d0 = sub(c0, c2);
        let s0 = add(c0, c2);
        let a1 = mont(add(d0, mean_offset(lo, hi, i + 324)), bc(ka), bc(ka.add(1)), q);
        let a0 = add(
            mont(add(s0, mean_offset(lo, hi, i)), bc(kb), bc(kb.add(1)), q),
            mont(d0, bc(kc), bc(kc.add(1)), q));
        accumulate::<BOTH>(center(a0, q, half, nhalf), i, shift, total);
        accumulate::<BOTH>(center(a1, q, half, nhalf), i + 324, shift, total);

        let d1 = sub(c1, c3);
        let s1 = add(c1, c3);
        let b1 = mont(add(d1, mean_offset(lo, hi, i + 486)), bc(ka), bc(ka.add(1)), q);
        let b0 = add(
            mont(add(s1, mean_offset(lo, hi, i + 162)), bc(kb), bc(kb.add(1)), q),
            mont(d1, bc(kc), bc(kc.add(1)), q));
        accumulate::<BOTH>(center(b0, q, half, nhalf), i + 162, shift, total);
        accumulate::<BOTH>(center(b1, q, half, nhalf), i + 486, shift, total);
    }
}

/// Add 8^digit times the centered inverse minus its public mean into total.
/// Bias rows come from sms_inverse_biases(). The input is destroyed scratch;
/// no representation or contents of b are promised on return.
///
/// # Safety
/// AVX-512 F/BW/VL. Input lanes obey TwI<9721>::IN_BOUND; the offsets are
/// centered and correspond to the coefficient means to subtract.
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
pub unsafe fn intt_sms_accumulate<const BOTH: bool>(b: &mut Batch32,
    bias_lo: &[i16], bias_hi: &[i16], shift: u32, total: &mut [[i32; 32]]) {
    assert!(bias_lo.len() == N && bias_hi.len() == N && total.len() == N && shift < 32);
    debug_assert_eq!(b.representation, Representation::Ntt);
    let p = b.v.as_mut_ptr() as *mut __m512i;
    for blk in 0..4 {
        for k4 in 6 * blk..6 * blk + 6 {
            ipass_d::<P, false>(p, k4);
            ipass_c5::<P, false>(p, k4);
            ipass_c4::<P, false>(p, k4);
        }
        ipass_b::<P, false>(p, blk);
    }
    final_pass::<BOTH>(p, bias_lo.as_ptr(), bias_hi.as_ptr(), shift, total.as_mut_ptr());
}
