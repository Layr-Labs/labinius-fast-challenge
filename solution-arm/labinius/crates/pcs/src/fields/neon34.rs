// Kernels of the i34-9 lean switch (submission 4982e5aa), kept apart from `neon.rs`.
#![allow(dead_code)]
//! Native AArch64 kernels for the field switch.
//!
//! The kernels in [`crossfield`](super::crossfield) and [`sumcheck`](super::sumcheck) are
//! written against AVX-512 GFNI and VPCLMULQDQ, which this target only emulates. These compute
//! the same field elements with NEON and, for the `F162` products, PMULL:
//!
//! - [`partial_evals_rows`]: the `128 x n` by `n x 128` bit product `v[k] = sum_j bit_k(pi0[j])
//!   eq[j]` by the method of four Russians. Per group of eight rows, a 256-entry table of the
//!   XOR combinations of the eight `eq` rows, indexed by the byte that collects bit `k` of the
//!   eight `pi0` rows (an 8 x 8 bit transpose per byte column, three masked shift rounds).
//! - [`poly_from_eq_rows`]: `a[j] = sum_k bit_k(eq[j]) batch[k]`, sixteen lookups per row in
//!   per-byte tables of `batch` combinations, written straight into the word-sliced layout.
//! - [`poly_from_b128_rows`]: the low and high words of each `B128` via de-interleaving loads.
//! - [`msg_range`] and [`fold_share`]: the sumcheck round over the word-sliced polynomials, two
//!   entries per step, Karatsuba over the three 64-bit limbs with `PMULL`/`PMULL2`. Message sums
//!   are accumulated unreduced and reduced once, as the AVX-512 kernel does.
//!
//! Everything here is GF(2)-linear bookkeeping around exact field arithmetic, so the outputs
//! are bit-identical to the emulated kernels; the tests check that against both.
use super::scalar::{B128, F162};
use core::arch::aarch64::*;

/// Whether the `F162` kernels can run: PMULL is the `aes` feature.
#[inline]
pub(super) fn pmull() -> bool {
    std::arch::is_aarch64_feature_detected!("aes")
}

// ---------------------------------------------------------------------------------------------
// partial evaluations
// ---------------------------------------------------------------------------------------------

/// Rows per table group, and groups per pass (eight 4 KiB tables).
const GROUP: usize = 8;
const PASS: usize = 8;

/// The eight rows at `p` as byte columns: `out[16 t + b]` collects bit `8 b + t` of rows
/// `0..8` in bits `0..8`.
#[inline(always)]
unsafe fn patterns(p: *const B128, out: &mut [u8; 128]) {
    let mut v: [uint8x16_t; 8] = core::array::from_fn(|i| vld1q_u8(p.add(i) as *const u8));
    let m1 = vdupq_n_u8(0x55);
    for i in [0usize, 2, 4, 6] {
        let (x, y) = (v[i], v[i + 1]);
        v[i] = vbslq_u8(m1, x, vshlq_n_u8::<1>(y));
        v[i + 1] = vbslq_u8(m1, vshrq_n_u8::<1>(x), y);
    }
    let m2 = vdupq_n_u8(0x33);
    for i in [0usize, 1, 4, 5] {
        let (x, y) = (v[i], v[i + 2]);
        v[i] = vbslq_u8(m2, x, vshlq_n_u8::<2>(y));
        v[i + 2] = vbslq_u8(m2, vshrq_n_u8::<2>(x), y);
    }
    for i in 0..4 {
        let (x, y) = (v[i], v[i + 4]);
        v[i] = vsliq_n_u8::<4>(x, y);
        v[i + 4] = vsriq_n_u8::<4>(y, x);
    }
    for (t, vt) in v.iter().enumerate() {
        vst1q_u8(out.as_mut_ptr().add(16 * t), *vt);
    }
}

/// `t[s]` = the sum of the rows `e[i]` over the set bits `i` of `s`.
#[inline(always)]
unsafe fn combinations(e: *const B128, t: &mut [uint8x16_t; 256]) {
    let rows: [uint8x16_t; 8] = core::array::from_fn(|i| vld1q_u8(e.add(i) as *const u8));
    let mut lo = [vdupq_n_u8(0); 16];
    let mut hi = [vdupq_n_u8(0); 16];
    for s in 1..16usize {
        let b = s.trailing_zeros() as usize;
        lo[s] = veorq_u8(lo[s & (s - 1)], rows[b]);
        hi[s] = veorq_u8(hi[s & (s - 1)], rows[4 + b]);
    }
    for h in 0..16 {
        for l in 0..16 {
            t[16 * h + l] = veorq_u8(hi[h], lo[l]);
        }
    }
}

/// `v[k] = sum_j bit_k(pi0[j]) eq[j]` over all rows of the slices.
pub(super) fn partial_evals_rows(pi0: &[B128], eq: &[B128]) -> Vec<B128> {
    assert_eq!(pi0.len(), eq.len());
    let n = pi0.len();
    let groups = n / GROUP;
    // acc[16 t + b] sums v[8 b + t], the order `patterns` emits.
    let mut acc = [0u128; 128];
    unsafe {
        let mut tabs: Vec<[uint8x16_t; 256]> = vec![[vdupq_n_u8(0); 256]; PASS];
        let mut pats = [[0u8; 128]; PASS];
        let mut sums = [vdupq_n_u8(0); 128];
        let mut g = 0;
        while g < groups {
            let count = (groups - g).min(PASS);
            for x in 0..count {
                let j = GROUP * (g + x);
                combinations(eq.as_ptr().add(j), &mut tabs[x]);
                patterns(pi0.as_ptr().add(j), &mut pats[x]);
            }
            let tabs = &tabs[..count];
            let pats = &pats[..count];
            for (q, sum) in sums.iter_mut().enumerate() {
                let mut a = vdupq_n_u8(0);
                for (t, p) in tabs.iter().zip(pats) {
                    a = veorq_u8(a, t[p[q] as usize]);
                }
                *sum = veorq_u8(*sum, a);
            }
            g += count;
        }
        for (q, s) in sums.iter().enumerate() {
            vst1q_u8(&mut acc[q] as *mut u128 as *mut u8, *s);
        }
    }
    let mut v = vec![B128::ZERO; 128];
    for (q, &a) in acc.iter().enumerate() {
        v[8 * (q % 16) + q / 16] = B128(a);
    }
    for (&p, &e) in pi0[GROUP * groups..].iter().zip(&eq[GROUP * groups..]) {
        for (k, vk) in v.iter_mut().enumerate() {
            if p.bit(k) {
                *vk = *vk + e;
            }
        }
    }
    v
}

// ---------------------------------------------------------------------------------------------
// the sumcheck's polynomials
// ---------------------------------------------------------------------------------------------

/// Per-byte tables of `sum_k bit_k(x) batch[k]`, summed over the claims: entry `256 b + s` is
/// the image of byte value `s` at byte `b`, its low limbs in `lo` and its top limb in `top`.
pub(super) struct ByteTables {
    lo: Vec<u128>,
    top: Vec<u64>,
}

impl ByteTables {
    pub(super) fn new(batch: &[F162]) -> Self {
        assert_eq!(batch.len(), 128);
        let mut lo = vec![0u128; 16 * 256];
        let mut top = vec![0u64; 16 * 256];
        for b in 0..16 {
            let mut t = [F162::ZERO; 256];
            for s in 1..256usize {
                let i = s.trailing_zeros() as usize;
                t[s] = t[s & (s - 1)] + batch[8 * b + i];
            }
            for (s, x) in t.iter().enumerate() {
                lo[256 * b + s] = x.0[0] as u128 | (x.0[1] as u128) << 64;
                top[256 * b + s] = x.0[2];
            }
        }
        Self { lo, top }
    }
}

/// `w[.][j] = sum_c sum_k bit_k(eqs[c][j]) batch_c[k]`, each claim through its [`ByteTables`].
pub(super) fn poly_from_eq_rows(eqs: &[&[B128]], tabs: &[ByteTables], w: [&mut [u64]; 3]) {
    let n = eqs[0].len();
    let [w0, w1, w2] = w;
    assert!(w0.len() == n && w1.len() == n && w2.len() == n);
    assert_eq!(eqs.len(), tabs.len());
    unsafe {
        for j in 0..n {
            let mut lo = vdupq_n_u64(0);
            let mut top = 0u64;
            for (eq, tab) in eqs.iter().zip(tabs) {
                let x = eq.get_unchecked(j).0;
                let (l, t) = (tab.lo.as_ptr(), tab.top.as_ptr());
                for b in 0..16 {
                    let s = 256 * b + ((x >> (8 * b)) & 0xff) as usize;
                    lo = veorq_u64(lo, vld1q_u64(l.add(s) as *const u64));
                    top ^= *t.add(s);
                }
            }
            *w0.get_unchecked_mut(j) = vgetq_lane_u64::<0>(lo);
            *w1.get_unchecked_mut(j) = vgetq_lane_u64::<1>(lo);
            *w2.get_unchecked_mut(j) = top;
        }
    }
}

/// `pi0` lifted: low words, high words, and a zero top limb.
pub(super) fn poly_from_b128_rows(pi0: &[B128], w: [&mut [u64]; 3]) {
    let n = pi0.len();
    let [w0, w1, w2] = w;
    assert!(w0.len() == n && w1.len() == n && w2.len() == n);
    w2.fill(0);
    let pairs = n / 2;
    unsafe {
        let src = pi0.as_ptr() as *const u64;
        for i in 0..pairs {
            let x = vld2q_u64(src.add(4 * i));
            vst1q_u64(w0.as_mut_ptr().add(2 * i), x.0);
            vst1q_u64(w1.as_mut_ptr().add(2 * i), x.1);
        }
    }
    for j in 2 * pairs..n {
        w0[j] = pi0[j].0 as u64;
        w1[j] = (pi0[j].0 >> 64) as u64;
    }
}

// ---------------------------------------------------------------------------------------------
// F162 with PMULL
// ---------------------------------------------------------------------------------------------

/// The limb products of lane 0 and of lane 1.
#[inline(always)]
unsafe fn pmul_lo(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
    vreinterpretq_u64_p128(vmull_p64(vgetq_lane_u64::<0>(a), vgetq_lane_u64::<0>(b)))
}

#[inline(always)]
unsafe fn pmul_hi(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
    vreinterpretq_u64_p128(vmull_high_p64(
        vreinterpretq_p64_u64(a),
        vreinterpretq_p64_u64(b),
    ))
}

#[inline(always)]
unsafe fn x2(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
    veorq_u64(a, b)
}

#[inline(always)]
unsafe fn x3(a: uint64x2_t, b: uint64x2_t, c: uint64x2_t) -> uint64x2_t {
    veorq_u64(veorq_u64(a, b), c)
}

/// Two entries' limbs from `j`.
#[inline(always)]
unsafe fn load2(w: [&[u64]; 3], j: usize) -> [uint64x2_t; 3] {
    [
        vld1q_u64(w[0].as_ptr().add(j)),
        vld1q_u64(w[1].as_ptr().add(j)),
        vld1q_u64(w[2].as_ptr().add(j)),
    ]
}

/// The Karatsuba operands of a limb triple: `[x0, x1, x2, x0 + x1, x0 + x2, x1 + x2]`.
#[inline(always)]
unsafe fn kara(x: [uint64x2_t; 3]) -> [uint64x2_t; 6] {
    [x[0], x[1], x[2], x2(x[0], x[1]), x2(x[0], x[2]), x2(x[1], x[2])]
}

/// `acc[i] += a_i b_i` for both lanes, unreduced: each `acc[i]` is one 128-bit sum.
#[inline(always)]
unsafe fn mac2(acc: &mut [uint64x2_t; 6], a: [uint64x2_t; 3], b: [uint64x2_t; 3]) {
    let (ka, kb) = (kara(a), kara(b));
    for i in 0..6 {
        acc[i] = x3(acc[i], pmul_lo(ka[i], kb[i]), pmul_hi(ka[i], kb[i]));
    }
}

/// Six unreduced Karatsuba products of one element as limbs `p0..p5` of the 323-bit product.
#[inline(always)]
fn assemble(m: [u128; 6]) -> [u64; 6] {
    let [m0, m1, m2, k01, k02, k12] = m;
    let d1 = k01 ^ m0 ^ m1;
    let d2 = k02 ^ m0 ^ m2 ^ m1;
    let d3 = k12 ^ m1 ^ m2;
    let lo = |x: u128| x as u64;
    let hi = |x: u128| (x >> 64) as u64;
    [
        lo(m0),
        hi(m0) ^ lo(d1),
        hi(d1) ^ lo(d2),
        hi(d2) ^ lo(d3),
        hi(d3) ^ lo(m2),
        hi(m2),
    ]
}

/// The reduction modulo `x^162 + x^81 + 1` of [`F162`]'s product, on limbs `p0..p5`.
#[inline(always)]
fn reduce(p: [u64; 6]) -> F162 {
    const M34: u64 = (1 << 34) - 1;
    let h00 = (p[2] >> 34) | (p[3] << 30);
    let h01 = (p[3] >> 34) & ((1 << 17) - 1);
    let h10 = (p[3] >> 51) | (p[4] << 13);
    let h11 = (p[4] >> 51) | (p[5] << 13);
    F162([
        p[0] ^ h00 ^ h10,
        p[1] ^ h01 ^ h11 ^ (h00 << 17),
        (p[2] & M34) ^ (h01 << 17) ^ (h00 >> 47),
    ])
}

#[inline(always)]
unsafe fn to_u128(x: uint64x2_t) -> u128 {
    vgetq_lane_u64::<0>(x) as u128 | (vgetq_lane_u64::<1>(x) as u128) << 64
}

/// The round message over one range: `[sum a0 p0, sum (a0 + a1)(p0 + p1)]`.
///
/// # Safety
/// The CPU must have PMULL ([`pmull`]).
#[target_feature(enable = "aes")]
pub(super) unsafe fn msg_range(
    a0s: [&[u64]; 3],
    a1s: [&[u64]; 3],
    p0s: [&[u64]; 3],
    p1s: [&[u64]; 3],
) -> [F162; 2] {
    let m = a0s[0].len();
    for w in [a0s, a1s, p0s, p1s] {
        assert!(w.iter().all(|x| x.len() == m));
    }
    let mut acc0 = [vdupq_n_u64(0); 6];
    let mut acci = [vdupq_n_u64(0); 6];
    let pairs = m / 2;
    for i in 0..pairs {
        let j = 2 * i;
        let a0 = load2(a0s, j);
        let a1 = load2(a1s, j);
        let p0 = load2(p0s, j);
        let p1 = load2(p1s, j);
        mac2(&mut acc0, a0, p0);
        let sa = [x2(a0[0], a1[0]), x2(a0[1], a1[1]), x2(a0[2], a1[2])];
        let sp = [x2(p0[0], p1[0]), x2(p0[1], p1[1]), x2(p0[2], p1[2])];
        mac2(&mut acci, sa, sp);
    }
    let mut e0 = reduce(assemble(acc0.map(|x| to_u128(x))));
    let mut einf = reduce(assemble(acci.map(|x| to_u128(x))));
    for j in 2 * pairs..m {
        let get = |w: [&[u64]; 3]| F162([w[0][j], w[1][j], w[2][j]]);
        let (a0, a1, p0, p1) = (get(a0s), get(a1s), get(p0s), get(p1s));
        e0 += a0 * p0;
        einf += (a0 + a1) * (p0 + p1);
    }
    [e0, einf]
}

/// `lo[j] <- lo[j] + r (lo[j] + hi[j])` over one share.
///
/// # Safety
/// The CPU must have PMULL ([`pmull`]).
#[target_feature(enable = "aes")]
pub(super) unsafe fn fold_share(lo: [&mut [u64]; 3], hi: [&[u64]; 3], r: F162) {
    let m = hi[0].len();
    assert!(lo.iter().all(|x| x.len() == m) && hi.iter().all(|x| x.len() == m));
    let [l0, l1, l2] = lo;
    let rk = kara([
        vdupq_n_u64(r.0[0]),
        vdupq_n_u64(r.0[1]),
        vdupq_n_u64(r.0[2]),
    ]);
    let m17 = vdupq_n_u64((1 << 17) - 1);
    let m34 = vdupq_n_u64((1 << 34) - 1);
    let pairs = m / 2;
    for i in 0..pairs {
        let j = 2 * i;
        let x0 = [
            vld1q_u64(l0.as_ptr().add(j)),
            vld1q_u64(l1.as_ptr().add(j)),
            vld1q_u64(l2.as_ptr().add(j)),
        ];
        let x1 = load2(hi, j);
        let d = kara([x2(x0[0], x1[0]), x2(x0[1], x1[1]), x2(x0[2], x1[2])]);
        // Products of lane 0 and lane 1, then their low and high words gathered across lanes.
        let mut plo = [vdupq_n_u64(0); 6];
        let mut phi = [vdupq_n_u64(0); 6];
        for k in 0..6 {
            let (a, b) = (pmul_lo(d[k], rk[k]), pmul_hi(d[k], rk[k]));
            plo[k] = vzip1q_u64(a, b);
            phi[k] = vzip2q_u64(a, b);
        }
        let [m0l, m1l, m2l, k01l, k02l, k12l] = plo;
        let [m0h, m1h, m2h, k01h, k02h, k12h] = phi;
        let d1l = x3(k01l, m0l, m1l);
        let d1h = x3(k01h, m0h, m1h);
        let d2l = x2(x3(k02l, m0l, m2l), m1l);
        let d2h = x2(x3(k02h, m0h, m2h), m1h);
        let d3l = x3(k12l, m1l, m2l);
        let d3h = x3(k12h, m1h, m2h);
        let p0 = m0l;
        let p1 = x2(m0h, d1l);
        let p2 = x2(d1h, d2l);
        let p3 = x2(d2h, d3l);
        let p4 = x2(d3h, m2l);
        let p5 = m2h;
        let h00 = vsliq_n_u64::<30>(vshrq_n_u64::<34>(p2), p3);
        let h01 = vandq_u64(vshrq_n_u64::<34>(p3), m17);
        let h10 = vsliq_n_u64::<13>(vshrq_n_u64::<51>(p3), p4);
        let h11 = vsliq_n_u64::<13>(vshrq_n_u64::<51>(p4), p5);
        let r0 = x3(p0, h00, h10);
        let r1 = x2(x3(p1, h01, h11), vshlq_n_u64::<17>(h00));
        let r2 = x3(
            vandq_u64(p2, m34),
            vshlq_n_u64::<17>(h01),
            vshrq_n_u64::<47>(h00),
        );
        vst1q_u64(l0.as_mut_ptr().add(j), x2(x0[0], r0));
        vst1q_u64(l1.as_mut_ptr().add(j), x2(x0[1], r1));
        vst1q_u64(l2.as_mut_ptr().add(j), x2(x0[2], r2));
    }
    for j in 2 * pairs..m {
        let x0 = F162([l0[j], l1[j], l2[j]]);
        let x1 = F162([hi[0][j], hi[1][j], hi[2][j]]);
        let y = x0 + r * (x0 + x1);
        l0[j] = y.0[0];
        l1[j] = y.0[1];
        l2[j] = y.0[2];
    }
}


// ---------------------------------------------------------------------------------------------
// the lean switch: block partial sets and the materialized fold
// ---------------------------------------------------------------------------------------------

/// Groups of eight `eq` rows whose tables are built per pass of [`partial_sets_rows`].
const LPASS: usize = 4;

/// `sets[u][k] = sum_j bit_k(pi0[u * stride + j0 + j]) eq[j]` for `u < blocks`, `j < eq.len()`.
///
/// The four-Russians tables of [`partial_evals_rows`] depend only on the `eq` rows, so each one
/// is built once and read for all `blocks` blocks of `pi0` that share those `eq` rows.
pub(super) fn partial_sets_rows(
    pi0: &[B128],
    stride: usize,
    blocks: usize,
    j0: usize,
    eq: &[B128],
) -> Vec<Vec<B128>> {
    let m = eq.len();
    assert!(blocks >= 1 && (blocks - 1) * stride + j0 + m <= pi0.len());
    let groups = m / GROUP;
    let mut out = vec![vec![B128::ZERO; 128]; blocks];
    unsafe {
        let mut tabs: Vec<[uint8x16_t; 256]> = vec![[vdupq_n_u8(0); 256]; LPASS];
        let mut sums: Vec<[uint8x16_t; 128]> = vec![[vdupq_n_u8(0); 128]; blocks];
        let mut pats = [[0u8; 128]; LPASS];
        let mut g = 0;
        while g < groups {
            let count = (groups - g).min(LPASS);
            for x in 0..count {
                combinations(eq.as_ptr().add(GROUP * (g + x)), &mut tabs[x]);
            }
            let tabs = &tabs[..count];
            for (u, sum) in sums.iter_mut().enumerate() {
                let base = pi0.as_ptr().add(u * stride + j0);
                for x in 0..count {
                    patterns(base.add(GROUP * (g + x)), &mut pats[x]);
                }
                let pats = &pats[..count];
                for (q, s) in sum.iter_mut().enumerate() {
                    let mut a = vdupq_n_u8(0);
                    for (t, p) in tabs.iter().zip(pats) {
                        a = veorq_u8(a, *t.get_unchecked(*p.get_unchecked(q) as usize));
                    }
                    *s = veorq_u8(*s, a);
                }
            }
            g += count;
        }
        for (sum, v) in sums.iter().zip(out.iter_mut()) {
            for (q, s) in sum.iter().enumerate() {
                let mut a = 0u128;
                vst1q_u8(&mut a as *mut u128 as *mut u8, *s);
                v[8 * (q % 16) + q / 16] = B128(a);
            }
        }
    }
    for (u, v) in out.iter_mut().enumerate() {
        let rows = &pi0[u * stride + j0..][..m];
        for (&p, &e) in rows[GROUP * groups..].iter().zip(&eq[GROUP * groups..]) {
            for (k, vk) in v.iter_mut().enumerate() {
                if p.bit(k) {
                    *vk = *vk + e;
                }
            }
        }
    }
    out
}

/// `out[.][j] = sum_u w[u] lift(pi0[u * stride + j0 + j])`, products accumulated unreduced and
/// reduced once per entry.
///
/// # Safety
/// The CPU must have PMULL ([`pmull`]).
#[target_feature(enable = "aes")]
pub(super) unsafe fn lift_combine_rows(
    pi0: &[B128],
    stride: usize,
    w: &[F162],
    j0: usize,
    out: [&mut [u64]; 3],
) {
    let [o0, o1, o2] = out;
    let m = o0.len();
    assert!(o1.len() == m && o2.len() == m);
    assert!(!w.is_empty() && (w.len() - 1) * stride + j0 + m <= pi0.len());
    let src = pi0.as_ptr() as *const u64;
    for j in 0..m {
        let mut acc = [vdupq_n_u64(0); 4];
        for (u, wu) in w.iter().enumerate() {
            let at = 2 * (u * stride + j0 + j);
            let x0 = *src.add(at);
            let x1 = *src.add(at + 1);
            for i in 0..3 {
                let wi = wu.0[i];
                acc[i] = veorq_u64(acc[i], vreinterpretq_u64_p128(vmull_p64(wi, x0)));
                acc[i + 1] = veorq_u64(acc[i + 1], vreinterpretq_u64_p128(vmull_p64(wi, x1)));
            }
        }
        let lo = |x: uint64x2_t| vgetq_lane_u64::<0>(x);
        let hi = |x: uint64x2_t| vgetq_lane_u64::<1>(x);
        let p = [
            lo(acc[0]),
            hi(acc[0]) ^ lo(acc[1]),
            hi(acc[1]) ^ lo(acc[2]),
            hi(acc[2]) ^ lo(acc[3]),
            hi(acc[3]),
            0,
        ];
        let y = reduce(p);
        o0[j] = y.0[0];
        o1[j] = y.0[1];
        o2[j] = y.0[2];
    }
}

