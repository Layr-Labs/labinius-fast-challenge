//! Generic-input NTT on the vertical `Batch32` layout (32 polynomials, one 512-bit vector per
//! coefficient index), in place: forward, `Coefficients -> Ntt`, input lanes `|x| <= q`, and its
//! inverse, `Ntt -> Coefficients` with fully reduced centered output (see "The inverse" below).
//!
//! # Structure (forward)
//!
//! The 7 levels of the tree (see `params`) run as five passes; after the first one everything is
//! depth-first inside a 162-block (10 KB) so the working set stays in L1.
//!
//! | pass | levels | group of vectors                                    | l/s | butterflies |
//! |------|--------|-----------------------------------------------------|-----|-------------|
//! | A    | 0 + 1  | `i, i+162, i+324, i+486` (i < 162)                  | 4   | 4 x radix-2 |
//! | B    | 2 + 3  | `j+27a`, `81+j+27a` (j < 27, a < 3), per 162-block  | 6   | 3 x radix-2, 2 x radix-3 |
//! | C4   | 4      | `i, i+9, i+18` (i < 9), per 27-block                | 3   | 1 x radix-3 |
//! | C5   | 5      | `i, i+3, i+6`, per 9-block                          | 3   | 1 x radix-3 |
//! | D    | 6      | `3g, 3g+1, 3g+2`, per 3-block                       | 3   | 1 x radix-3 |
//!
//! Pass A fuses levels 0 and 1 into one radix-4 pass (level 0 is the Phi_6 split
//! `t = zeta6*a1; y0 = a0 + t; y1 = a0 + a1 - t`, level 1 the two radix-2 halves): 162 iterations
//! of 4 loads, 4 Montgomery multiplications, 4 stores.
//!
//! Deeper fusion was implemented and measured: fusing levels 4+5 into a radix-9 pass over 9
//! registers costs 3.5% (438 vs 423 cycles/poly at q = 3889), 5+6 costs 3.8%, and the full
//! radix-27 tail costs 6.3%. They do cut L1 traffic, but the kernel is port-0-throughput-bound,
//! not load/store-bound, and the longer dependency chains inside a fused group only reduce the
//! number of independent butterflies in flight. Splitting pass B into separate level-2 and
//! level-3 passes also loses (427 vs 423), so B stays fused.
//!
//! # Arithmetic and bounds
//!
//! Twiddle multiplication is the 3-uop signed Montgomery form `mullo(a,w') / mulhi(a,w) /
//! mulhi(m,q) / sub` with `|mont(a,w)| <= |a| q/2^17 + q/2`; twiddle constants live in memory as
//! duplicated `u32` so the broadcast is a pure load (`vpbroadcastd`).  With `c = q/2^17` and all
//! bounds written as multiples of q (input `|x| <= q`), one radix-3 level maps a bound A to
//! `A + 2(1/2 + cA)` on the untwiddled path, so the a0 input needs an occasional `barrett`
//! (`|barrett(x)| <= 0.899q` for 3889, `0.809q` for 9721).  Placement (`BAR_A`, `BAR_L`) and the
//! resulting per-level bounds:
//!
//! ```text
//!                       q = 3889 (budget 8.4276 q)      q = 9721 (budget 3.3708 q)
//!   level 0  (in pass A)   2.5297                          2.5742  (barrett on the a0 half)
//!   level 1  (in pass A)   3.1047                          2.1909
//!   level 2                3.6969                          2.8535
//!   level 3                4.9163                          2.2323  (barrett a0)
//!   level 4                6.2080                          2.1401  (barrett a0)
//!   level 5                2.2676  (barrett a0)            2.1265  (barrett a0)
//!   level 6                3.4022  (output)                2.1244  (output, barrett a0)
//! ```
//!
//! Output bound: `|v| <= 3.4022 q = 13231` for q = 3889 and `|v| <= 2.1244 q = 20652` for q = 9721
//! (`OUTPUT_BOUND`), verified against an exact i32 shadow model in `tests/ntt.rs::gen_small`.
//!
//! # The inverse (`intt_gen_batch32`, 541 / 601)
//!
//! `Ntt -> Coefficients`, the same five passes in the opposite order — levels 6, 5, 4 per
//! 27-block, then levels 3+2 per 162-block, then levels 1+0 over the whole batch — with
//! Gentleman-Sande butterflies, the transposes of the forward ones, which cost exactly what the
//! forward ones cost: `u = omega (y2 - y1)`, then `(y0+y1+y2, (y0-y1+u) zeta^-1,
//! (y0-y2-u) zeta^-2)`, 3 Montgomery products and 7 add/sub for radix 3, and
//! `(y0+y1, (y0-y1) zeta^-1)` for radix 2. The per-level `1/3` and `1/2` are *not* applied:
//! every value reaching level 0 is the true one times `2 * 2 * 3^4 = 324`, and the whole
//! `1/648` corrected by the Phi_6 determinant — `scalar::intt`'s `inv2 / inv3 / det` chain,
//! multiplied out — sits in the three level-0 constants [`TwI::KA`], `KB`, `KC`. The output is
//! fully reduced and centered in `[-(q-1)/2, (q-1)/2]`.
//!
//! What the inverse pays over the forward is entirely what a *sum* costs. Forward, every output
//! is either a Montgomery product (`< 0.75 q`, reduced for free) or `a0 +- t` with an occasional
//! Barrett on `a0`; inverse, the untwiddled output is `y0+y1+y2`, which triples whatever it is
//! given and cannot absorb a constant, so the position classes whose whole ancestry is
//! untwiddled have to be Barretted level after level. Three roughly equal thirds, in ALU uops
//! per polynomial against the forward kernel's 736:
//!
//! ```text
//!   the Phi_6 inverse is a general 2x2 matrix — 3 Montgomery products per level-0
//!     butterfly against pass A's 1 (a1 = KA (Y0-Y1), a0 = KB (Y0+Y1) + KC (Y0-Y1))     +81
//!   Barretts: 984 per batch (3889) / 1836 (9721), against 216 / 1026 forward           +72
//!   centering the 648 outputs (2 mask compares + 2 masked adds each)                   +81
//! ```
//!
//! Placement (`BAR_IN`, `BAR_S6 .. BAR_S1`, each indexed by exactly the loop variable that names
//! the position class) is the cheapest member of that flag set that keeps every intermediate
//! inside i16, found by exhaustive search and replayed by the `const` recursion `inv_bounds`,
//! which also produces `TwI::BOUND` and `TwI::PEAK`; `tests/ntt.rs::gen_small` replays the same
//! schedule in i32 against the kernel. Declared input bound: the binary kernel's 7.5 q / 2.3 q,
//! i.e. every lazily reduced transform this crate produces. Per level, max |lane| afterwards:
//!
//! ```text
//!                          q = 3889 (budget 8.4256 q)     q = 9721 (budget 3.3707 q)
//!   input                     7.5000                         2.3000
//!   level 6  (pass D)         2.6976  (barrett the 3 inputs) 0.8090  (3 inputs and the sum)
//!   level 5  (pass C5)        1.7094  (sum of j = 0)         0.8090  (all sums)
//!   level 4  (pass C4)        2.6976  (sums of i = 1, 2)     0.8090  (all sums)
//!   level 3  (pass B)         0.8992  (all sums)             0.8090  (all sums)
//!   level 2  (pass B)         1.7984                         1.6179
//!   level 1  (pass A)         3.5968                         0.8090  (all sums)
//!   level 0  (pass A)         1.4271  -> centered            1.2400  -> centered
//!   peak intermediate         8.0928                         3.2359
//! ```
//!
//! The three inputs of a level-6 butterfly are Barretted as they are loaded — at 7.5 q even
//! `y1 - y2` leaves i16 — which is 648 of the 984; at the fold's centered input (`|v| <= q/2`)
//! the same search returns 220, worth ~8 %. That would need a second flag set and a second bound
//! recursion for one caller, and the fold's inverse is 0.07 ms either way, so the kernel has one
//! declared input bound. Port balance is 515 p0 against 464 p5 per polynomial (3889), so the
//! shuffle-port lookup Barrett of `ntt::bin_asm` (5 uops, none of them p0, against 3 with 2
//! on p0) would pay for about 270 of the 984 before p0 stops being the constraint — not taken,
//! nor the level-0 form `a0 = KB (Y0+Y1) - a1/2`, which replaces one Montgomery product by a
//! conditional-add-and-shift halving (2 port-0 uops per butterfly) at the price of a serial
//! dependency and an output bound that then depends on level 1's.
use core::arch::x86_64::*;
use labinius::params::{barrett_v, inv_mod, Params};
use labinius::ring::{Batch32, Representation};

/// A 16-bit constant duplicated into a u32 so that `vpbroadcastd m32` fills a whole zmm with it.
const fn dup(w: i16) -> u32 {
    let u = w as u16 as u32;
    u | (u << 16)
}

/// Compile-time twiddle tables, all in the duplicated-u32 broadcast form.
///
/// Radix-2 levels store `[zeta', zeta]` per sub-ring, radix-3 levels store
/// `[zeta', zeta, (zeta^2)', zeta^2]` (primed = the `mont_pre` companion).
pub struct Tw<const Q: u16>;

impl<const Q: u16> Tw<Q> {
    const fn pair(x: u16) -> [u32; 2] {
        let w = Params::<Q>::to_mont(x);
        [dup(Params::<Q>::mont_pre(w)), dup(w)]
    }
    const fn r2<const M: usize>(level: usize, nk: usize) -> [u32; M] {
        let mut t = [0u32; M];
        let mut k = 0;
        while k < nk {
            let p = Self::pair(Params::<Q>::zeta(level, k));
            t[2 * k] = p[0];
            t[2 * k + 1] = p[1];
            k += 1;
        }
        t
    }
    /// Radix-3 twiddle table with both twiddles pre-multiplied by `scale` (1 = plain).
    const fn r3_scaled<const M: usize>(level: usize, nk: usize, scale: u16) -> [u32; M] {
        let mut t = [0u32; M];
        let mut k = 0;
        while k < nk {
            let z = Params::<Q>::zeta(level, k);
            let z2 = (z as u64 * z as u64 % Q as u64) as u16;
            let sc = scale as u64;
            let a = Self::pair((z as u64 * sc % Q as u64) as u16);
            let b = Self::pair((z2 as u64 * sc % Q as u64) as u16);
            t[4 * k] = a[0];
            t[4 * k + 1] = a[1];
            t[4 * k + 2] = b[0];
            t[4 * k + 3] = b[1];
            k += 1;
        }
        t
    }
    const fn r3<const M: usize>(level: usize, nk: usize) -> [u32; M] {
        Self::r3_scaled::<M>(level, nk, 1)
    }
    pub const Z6: [u32; 2] = Self::pair(Params::<Q>::ZETA6);
    pub const OM: [u32; 2] = Self::pair(Params::<Q>::OMEGA);
    pub const QD: u32 = dup(Q as i16);
    pub const BV: u32 = dup(barrett_v(Q));
    pub const L1: [u32; 4] = Self::r2::<4>(1, 2);
    pub const L2: [u32; 8] = Self::r2::<8>(2, 4);
    pub const L3: [u32; 32] = Self::r3::<32>(3, 8);
    pub const L4: [u32; 96] = Self::r3::<96>(4, 24);
    pub const L5: [u32; 288] = Self::r3::<288>(5, 72);
    pub const L6: [u32; 864] = Self::r3::<864>(6, 216);

    /// Barrett the level-0 output `a0 + a1 - zeta6*a1` inside pass A (only 9721 needs it).
    pub const BAR_A: bool = Q == 9721;
    /// `BAR_L[l]` = barrett the untwiddled a0 input of level l (l = 2..6).
    pub const BAR_L: [bool; 7] = if Q == 9721 {
        [false, false, false, true, true, true, true]
    } else {
        [false, false, false, false, false, true, false]
    };
    /// Proven output bound, as `ceil(bound * q)`; see the module comment.
    pub const OUTPUT_BOUND: i32 = if Q == 9721 { 20652 } else { 13231 };
}

#[inline(always)]
unsafe fn bc(p: *const u32) -> __m512i {
    _mm512_set1_epi32(p.read() as i32)
}

#[inline(always)]
unsafe fn ld(p: *const __m512i, j: usize) -> __m512i {
    _mm512_load_si512(p.add(j))
}

#[inline(always)]
unsafe fn st(p: *mut __m512i, j: usize, x: __m512i) {
    _mm512_store_si512(p.add(j), x);
}

/// Signed Montgomery twiddle product: 3 multiply uops, `|r| <= |a| q/2^17 + q/2 < q`.
#[inline(always)]
unsafe fn mont(a: __m512i, wp: __m512i, w: __m512i, q: __m512i) -> __m512i {
    let m = _mm512_mullo_epi16(a, wp);
    let hi = _mm512_mulhi_epi16(a, w);
    _mm512_sub_epi16(hi, _mm512_mulhi_epi16(m, q))
}

/// `vpmulhrsw`-based partial reduction: 2 multiply uops, `|r| <= 0.899q / 0.809q`.
#[inline(always)]
unsafe fn barrett(a: __m512i, v: __m512i, q: __m512i) -> __m512i {
    _mm512_sub_epi16(a, _mm512_mullo_epi16(_mm512_mulhrs_epi16(a, v), q))
}

#[inline(always)]
unsafe fn add(a: __m512i, b: __m512i) -> __m512i {
    _mm512_add_epi16(a, b)
}

#[inline(always)]
unsafe fn sub(a: __m512i, b: __m512i) -> __m512i {
    _mm512_sub_epi16(a, b)
}

/// Radix-3 butterfly `(a0, a1, a2) -> (a0+t1+t2, a0-t2+u, a0-t1-u)`, `t1 = zeta a1`,
/// `t2 = zeta^2 a2`, `u = omega (t1-t2)`: 9 multiply uops + 7 adds.
#[inline(always)]
unsafe fn r3(
    a0: __m512i,
    a1: __m512i,
    a2: __m512i,
    tw: *const u32,
    omp: __m512i,
    om: __m512i,
    q: __m512i,
) -> (__m512i, __m512i, __m512i) {
    let t1 = mont(a1, bc(tw), bc(tw.add(1)), q);
    let t2 = mont(a2, bc(tw.add(2)), bc(tw.add(3)), q);
    let u = mont(sub(t1, t2), omp, om, q);
    (
        add(a0, add(t1, t2)),
        add(sub(a0, t2), u),
        sub(sub(a0, t1), u),
    )
}

/// Levels 0 and 1 fused into one radix-4 pass over the 648 vectors.
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
unsafe fn pass_a<const Q: u16>(p: *mut __m512i) {
    let q = _mm512_set1_epi32(Tw::<Q>::QD as i32);
    let bv = _mm512_set1_epi32(Tw::<Q>::BV as i32);
    let z6p = bc(Tw::<Q>::Z6.as_ptr());
    let z6 = bc(Tw::<Q>::Z6.as_ptr().add(1));
    let l1 = Tw::<Q>::L1.as_ptr();
    let (zap, za) = (bc(l1), bc(l1.add(1)));
    let (zbp, zb) = (bc(l1.add(2)), bc(l1.add(3)));
    for i in 0..162 {
        let a0 = ld(p, i);
        let a1 = ld(p, i + 324);
        let b0 = ld(p, i + 162);
        let b1 = ld(p, i + 486);
        let t = mont(a1, z6p, z6, q);
        let s = mont(b1, z6p, z6, q);
        let c0 = add(a0, t);
        let c1 = add(b0, s);
        let mut c2 = sub(add(a0, a1), t);
        let c3 = sub(add(b0, b1), s);
        if Tw::<Q>::BAR_A {
            c2 = barrett(c2, bv, q);
        }
        let u = mont(c1, zap, za, q);
        let w = mont(c3, zbp, zb, q);
        st(p, i, add(c0, u));
        st(p, i + 162, sub(c0, u));
        st(p, i + 324, add(c2, w));
        st(p, i + 486, sub(c2, w));
    }
}

/// Levels 2 and 3 for one 162-block: 27 groups of 6 vectors, 3 radix-2 + 2 radix-3 each.
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
unsafe fn pass_b<const Q: u16>(p: *mut __m512i, blk: usize) {
    let q = _mm512_set1_epi32(Tw::<Q>::QD as i32);
    let bv = _mm512_set1_epi32(Tw::<Q>::BV as i32);
    let omp = bc(Tw::<Q>::OM.as_ptr());
    let om = bc(Tw::<Q>::OM.as_ptr().add(1));
    let l2 = Tw::<Q>::L2.as_ptr().add(2 * blk);
    let (zp, z) = (bc(l2), bc(l2.add(1)));
    let ta = Tw::<Q>::L3.as_ptr().add(8 * blk);
    let tb = ta.add(4);
    let base = blk * 162;
    for j in 0..27 {
        let b = base + j;
        let x0 = ld(p, b);
        let x1 = ld(p, b + 27);
        let x2 = ld(p, b + 54);
        let t0 = mont(ld(p, b + 81), zp, z, q);
        let t1 = mont(ld(p, b + 108), zp, z, q);
        let t2 = mont(ld(p, b + 135), zp, z, q);
        let mut n0 = add(x0, t0);
        let n1 = add(x1, t1);
        let n2 = add(x2, t2);
        let mut m0 = sub(x0, t0);
        let m1 = sub(x1, t1);
        let m2 = sub(x2, t2);
        if Tw::<Q>::BAR_L[3] {
            n0 = barrett(n0, bv, q);
            m0 = barrett(m0, bv, q);
        }
        let (y0, y1, y2) = r3(n0, n1, n2, ta, omp, om, q);
        let (w0, w1, w2) = r3(m0, m1, m2, tb, omp, om, q);
        st(p, b, y0);
        st(p, b + 27, y1);
        st(p, b + 54, y2);
        st(p, b + 81, w0);
        st(p, b + 108, w1);
        st(p, b + 135, w2);
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
unsafe fn pass_d<const Q: u16>(p: *mut __m512i, k4: usize) {
    let q = _mm512_set1_epi32(Tw::<Q>::QD as i32);
    let bv = _mm512_set1_epi32(Tw::<Q>::BV as i32);
    let omp = bc(Tw::<Q>::OM.as_ptr());
    let om = bc(Tw::<Q>::OM.as_ptr().add(1));
    let t6 = Tw::<Q>::L6.as_ptr().add(36 * k4);
    let base = k4 * 27;
    for g in 0..9 {
        let b = base + 3 * g;
        let mut c0 = ld(p, b);
        if Tw::<Q>::BAR_L[6] {
            c0 = barrett(c0, bv, q);
        }
        let (y0, y1, y2) = r3(c0, ld(p, b + 1), ld(p, b + 2), t6.add(4 * g), omp, om, q);
        st(p, b, y0);
        st(p, b + 1, y1);
        st(p, b + 2, y2);
    }
}

/// Level 4 alone for one 27-block: 3 groups of 3 independent butterflies .
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
unsafe fn pass_c4<const Q: u16>(p: *mut __m512i, k4: usize) {
    let q = _mm512_set1_epi32(Tw::<Q>::QD as i32);
    let bv = _mm512_set1_epi32(Tw::<Q>::BV as i32);
    let omp = bc(Tw::<Q>::OM.as_ptr());
    let om = bc(Tw::<Q>::OM.as_ptr().add(1));
    let t4 = Tw::<Q>::L4.as_ptr().add(4 * k4);
    let base = k4 * 27;
    for c in 0..3 {
        for i in 3 * c..3 * c + 3 {
            let b = base + i;
            let mut c0 = ld(p, b);
            if Tw::<Q>::BAR_L[4] {
                c0 = barrett(c0, bv, q);
            }
            let (y0, y1, y2) = r3(c0, ld(p, b + 9), ld(p, b + 18), t4, omp, om, q);
            st(p, b, y0);
            st(p, b + 9, y1);
            st(p, b + 18, y2);
        }
    }
}

/// Level 5 alone for one 27-block: 3 groups of 9 vectors, 3 butterflies each .
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
unsafe fn pass_c5<const Q: u16>(p: *mut __m512i, k4: usize) {
    let q = _mm512_set1_epi32(Tw::<Q>::QD as i32);
    let bv = _mm512_set1_epi32(Tw::<Q>::BV as i32);
    let omp = bc(Tw::<Q>::OM.as_ptr());
    let om = bc(Tw::<Q>::OM.as_ptr().add(1));
    let t5 = Tw::<Q>::L5.as_ptr().add(12 * k4);
    let base = k4 * 27;
    for bb in 0..3 {
        for j in 0..3 {
            let b = base + 9 * bb + j;
            let mut c0 = ld(p, b);
            if Tw::<Q>::BAR_L[5] {
                c0 = barrett(c0, bv, q);
            }
            let (y0, y1, y2) = r3(c0, ld(p, b + 3), ld(p, b + 6), t5.add(4 * bb), omp, om, q);
            st(p, b, y0);
            st(p, b + 3, y1);
            st(p, b + 6, y2);
        }
    }
}

/// Forward NTT of a batch of 32 polynomials in place: `Coefficients -> Ntt` (tree order).
///
/// Requires `|b.v[j][p]| <= q`. Output satisfies `|b.v[j][p]| <= Tw::<Q>::OUTPUT_BOUND`.
///
/// # Safety
/// The host must have AVX-512 F/BW/VL; `b` must be 64-byte aligned (`Batch32` is).
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
pub unsafe fn ntt_gen_batch32<const Q: u16>(b: &mut Batch32) {
    debug_assert_eq!(b.representation, Representation::Coefficients);
    let p = b.v.as_mut_ptr() as *mut __m512i;
    pass_a::<Q>(p);
    for blk in 0..4 {
        pass_b::<Q>(p, blk);
        for k4 in 6 * blk..6 * blk + 6 {
            pass_c4::<Q>(p, k4);
            pass_c5::<Q>(p, k4);
            pass_d::<Q>(p, k4);
        }
    }
    b.representation = Representation::Ntt;
}

/// Exact existing levels4/5/6. Input is plain partial27 residues, |x|<2q.
/// No new partial inverse: caller follows this with actual full inverse for aux.
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
pub unsafe fn partial_to_full<const Q: u16>(b: &mut Batch32) {
    let p = b.v.as_mut_ptr() as *mut __m512i;
    for k in 0..24 {
        pass_c4::<Q>(p, k);
        pass_c5::<Q>(p, k);
        pass_d::<Q>(p, k);
    }
    b.representation = Representation::Ntt;
}
