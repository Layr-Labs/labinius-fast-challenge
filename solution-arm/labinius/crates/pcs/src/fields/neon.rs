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

pub(super) fn dot_lifted(a: &[F162], b: &[B128]) -> F162 {
    assert_eq!(a.len(), b.len());
    unsafe {
        let mut acc = [vdupq_n_u64(0); 6];
        let pairs = a.len() / 2;
        for i in 0..pairs {
            let aa = vld3q_u64(a.as_ptr().add(2 * i) as *const u64);
            let bb = vld2q_u64(b.as_ptr().add(2 * i) as *const u64);
            let ka = [aa.0, aa.1, vdupq_n_u64(0), x2(aa.0, aa.1), x2(aa.0, aa.2), x2(aa.1, aa.2)];
            let kb = [bb.0, bb.1, vdupq_n_u64(0), x2(bb.0, bb.1), bb.0, bb.1];
            for k in [0usize, 1, 3, 4, 5] {
                acc[k] = x2(acc[k], pmul_lo(ka[k], kb[k]));
                acc[k] = x2(acc[k], pmul_hi(ka[k], kb[k]));
            }
        }
        let mut out = reduce(assemble(acc.map(|x| to_u128(x))));
        if pairs * 2 < a.len() { out += a[2 * pairs] * F162::from_b128(b[2 * pairs]); }
        out
    }
}

/// [`dot_lifted`] of `a` against two columns at once: `a` (the eq table) is loaded once per pair
/// for both, halving its L2 traffic. Each column's products are summed in the same order as
/// [`dot_lifted`]'s, so each result is the same field element.
pub(super) fn dot_lifted2(a: &[F162], b0: &[B128], b1: &[B128]) -> [F162; 2] {
    assert!(a.len() == b0.len() && a.len() == b1.len());
    unsafe {
        let mut acc0 = [vdupq_n_u64(0); 6];
        let mut acc1 = [vdupq_n_u64(0); 6];
        let pairs = a.len() / 2;
        for i in 0..pairs {
            let aa = vld3q_u64(a.as_ptr().add(2 * i) as *const u64);
            let ka = [aa.0, aa.1, vdupq_n_u64(0), x2(aa.0, aa.1), x2(aa.0, aa.2), x2(aa.1, aa.2)];
            let bb = vld2q_u64(b0.as_ptr().add(2 * i) as *const u64);
            let kb = [bb.0, bb.1, vdupq_n_u64(0), x2(bb.0, bb.1), bb.0, bb.1];
            let cc = vld2q_u64(b1.as_ptr().add(2 * i) as *const u64);
            let kc = [cc.0, cc.1, vdupq_n_u64(0), x2(cc.0, cc.1), cc.0, cc.1];
            for k in [0usize, 1, 3, 4, 5] {
                acc0[k] = x2(acc0[k], pmul_lo(ka[k], kb[k]));
                acc0[k] = x2(acc0[k], pmul_hi(ka[k], kb[k]));
                acc1[k] = x2(acc1[k], pmul_lo(ka[k], kc[k]));
                acc1[k] = x2(acc1[k], pmul_hi(ka[k], kc[k]));
            }
        }
        let mut out0 = reduce(assemble(acc0.map(|x| to_u128(x))));
        let mut out1 = reduce(assemble(acc1.map(|x| to_u128(x))));
        if pairs * 2 < a.len() {
            out0 += a[2 * pairs] * F162::from_b128(b0[2 * pairs]);
            out1 += a[2 * pairs] * F162::from_b128(b1[2 * pairs]);
        }
        [out0, out1]
    }
}

pub(super) fn dot_lifted4(a: &[F162], b: [&[B128]; 4]) -> [F162; 4] {
    assert!(b.iter().all(|c| c.len() == a.len()));
    unsafe {
        let mut acc = [[vdupq_n_u64(0); 6]; 4];
        let pairs = a.len() / 2;
        for i in 0..pairs {
            let aa = vld3q_u64(a.as_ptr().add(2 * i) as *const u64);
            let ka = [aa.0, aa.1, vdupq_n_u64(0), x2(aa.0, aa.1), x2(aa.0, aa.2), x2(aa.1, aa.2)];
            for c in 0..4 {
                let bb = vld2q_u64(b[c].as_ptr().add(2 * i) as *const u64);
                let kb = [bb.0, bb.1, vdupq_n_u64(0), x2(bb.0, bb.1), bb.0, bb.1];
                for k in [0usize, 1, 3, 4, 5] {
                    acc[c][k] = x2(acc[c][k], pmul_lo(ka[k], kb[k]));
                    acc[c][k] = x2(acc[c][k], pmul_hi(ka[k], kb[k]));
                }
            }
        }
        core::array::from_fn(|c| {
            let mut out = reduce(assemble(acc[c].map(|x| to_u128(x))));
            if pairs * 2 < a.len() {
                out += a[2 * pairs] * F162::from_b128(b[c][2 * pairs]);
            }
            out
        })
    }
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

#[inline(always)]
unsafe fn fold2(x0: [uint64x2_t; 3], x1: [uint64x2_t; 3], rk: [uint64x2_t; 6]) -> [uint64x2_t; 3] {
    let m17 = vdupq_n_u64((1 << 17) - 1);
    let m34 = vdupq_n_u64((1 << 34) - 1);
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
    [x2(x0[0], r0), x2(x0[1], r1), x2(x0[2], r2)]
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
    let pairs = m / 2;
    for i in 0..pairs {
        let j = 2 * i;
        let x0 = [
            vld1q_u64(l0.as_ptr().add(j)),
            vld1q_u64(l1.as_ptr().add(j)),
            vld1q_u64(l2.as_ptr().add(j)),
        ];
        let x1 = load2(hi, j);
        let y = fold2(x0, x1, rk);
        vst1q_u64(l0.as_mut_ptr().add(j), y[0]);
        vst1q_u64(l1.as_mut_ptr().add(j), y[1]);
        vst1q_u64(l2.as_mut_ptr().add(j), y[2]);
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

/// `sum[k] += sum_x tabs[x][pats[x][k]]` for a full pass of [`PASS`] tables, eight entries `k`
/// at a time: one 64-bit load per table brings that table's eight index bytes (instead of one
/// `ldrb` per lookup, which made the loop load-bound on Neoverse V2's three load pipes). The
/// XOR tree uses plain `EOR`s: as `EOR3`s it would bind V0, the only pipe that runs `EOR3` on V2.
#[inline(always)]
unsafe fn lookup_pass(tabs: &[[uint8x16_t; 256]], pats: &[[u8; 128]; PASS], sum: &mut [uint8x16_t; 128]) {
    debug_assert!(tabs.len() >= PASS);
    let t: [*const uint8x16_t; PASS] = core::array::from_fn(|x| tabs.get_unchecked(x).as_ptr());
    let p = pats.as_ptr() as *const u8;
    for kb in (0..128).step_by(8) {
        let idx: [u64; PASS] = core::array::from_fn(|x| u64::from_le((p.add(128 * x + kb) as *const u64).read_unaligned()));
        // Constant byte positions (one `ubfx` per index), unrolled by hand: LLVM keeps a `for` over
        // the eight positions rolled, with a variable shift and a mask per index.
        let s = sum.as_mut_ptr().add(kb);
        lookup_one::<0>(&t, &idx, &mut *s);
        lookup_one::<1>(&t, &idx, &mut *s.add(1));
        lookup_one::<2>(&t, &idx, &mut *s.add(2));
        lookup_one::<3>(&t, &idx, &mut *s.add(3));
        lookup_one::<4>(&t, &idx, &mut *s.add(4));
        lookup_one::<5>(&t, &idx, &mut *s.add(5));
        lookup_one::<6>(&t, &idx, &mut *s.add(6));
        lookup_one::<7>(&t, &idx, &mut *s.add(7));
    }
}

/// One entry of [`lookup_pass`]: byte `KK` of every table's index word.
#[inline(always)]
unsafe fn lookup_one<const KK: u32>(t: &[*const uint8x16_t; PASS], idx: &[u64; PASS], s: &mut uint8x16_t) {
    let v = |x: usize| *t[x].add(((idx[x] >> (8 * KK)) & 0xff) as usize);
    // Two three-input `EOR3`s (one per cycle on Neoverse V2, on one pipe) and four plain `EOR`s
    // instead of eight `EOR`s: fewer instructions in a loop that is issue-bound.
    let a = eor3(v(0), v(1), v(2));
    let b = eor3(v(3), v(4), v(5));
    let c = eor_plain(v(6), v(7));
    *s = eor_plain(*s, eor_plain(eor_plain(a, b), c));
}

/// `a ^ b ^ c` as one `EOR3`.
#[inline(always)]
unsafe fn eor3(mut a: uint8x16_t, b: uint8x16_t, c: uint8x16_t) -> uint8x16_t {
    core::arch::asm!(
        "eor3 {a:v}.16b, {a:v}.16b, {b:v}.16b, {c:v}.16b",
        a = inout(vreg) a, b = in(vreg) b, c = in(vreg) c,
        options(pure, nomem, nostack, preserves_flags)
    );
    a
}

/// `a ^ b` as one plain `EOR` that LLVM cannot fuse into `EOR3` (V0-only on Neoverse V2).
#[inline(always)]
unsafe fn eor_plain(mut a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    core::arch::asm!(
        "eor {a:v}.16b, {a:v}.16b, {b:v}.16b",
        a = inout(vreg) a, b = in(vreg) b,
        options(pure, nomem, nostack, preserves_flags)
    );
    a
}

/// `prfm pldl2keep` on every 64-byte line of `[p, p + bytes)` (a hint: no fault, no effect on results).
#[inline(always)]
unsafe fn prefetch_l2(p: *const u8, bytes: usize) {
    for off in (0..bytes).step_by(64) {
        core::arch::asm!("prfm pldl2keep, [{p}]", p = in(reg) p.wrapping_add(off), options(nostack, preserves_flags, readonly));
    }
}

#[cfg(test)]
pub(super) fn partial_evals_sets(words: &[B128], eq: &[B128], start: usize, end: usize) -> Vec<Vec<B128>> {
    partial_evals_sets_tiled(words, eq.len(), start, end, |base, tile| {
        tile.copy_from_slice(&eq[base..base + tile.len()]);
    })
}

pub(super) fn partial_evals_sets_tiled(
    words: &[B128], stride: usize, start: usize, end: usize,
    mut fill: impl FnMut(usize, &mut [B128]),
) -> Vec<Vec<B128>> {
    assert_eq!(words.len() % stride, 0);
    assert!(start <= end && end <= stride && start % GROUP == 0 && end % GROUP == 0);
    let blocks = words.len() / stride;
    unsafe {
        let mut sums = vec![[vdupq_n_u8(0); 128]; blocks];
        let mut tabs = vec![[vdupq_n_u8(0); 256]; PASS];
        let mut pats = [[0u8; 128]; PASS];
        let mut tile = [B128::ZERO; GROUP * PASS];
        for base in (start..end).step_by(GROUP * PASS) {
            let count = ((end - base) / GROUP).min(PASS);
            fill(base, &mut tile[..GROUP * count]);
            for x in 0..count {
                combinations(tile.as_ptr().add(GROUP * x), &mut tabs[x]);
            }
            let next = base + GROUP * PASS < end;
            for (q, sum) in sums.iter_mut().enumerate() {
                // This block's next tile into L2: 64 blocks 4 MiB apart, 1 KiB each per tile, are more
                // streams than the hardware prefetchers follow.
                let row = words.as_ptr().add(q * stride + base) as *const u8;
                if next {
                    prefetch_l2(row.add(16 * GROUP * PASS), 16 * GROUP * PASS);
                }
                for x in 0..count {
                    patterns(words.as_ptr().add(q * stride + base + GROUP * x), &mut pats[x]);
                }
                if count == PASS {
                    lookup_pass(&tabs, &pats, sum);
                    continue;
                }
                for k in 0..128 {
                    let mut a = sum[k];
                    for x in 0..count {
                        a = veorq_u8(a, tabs[x][pats[x][k] as usize]);
                    }
                    sum[k] = a;
                }
            }
        }
        sums.into_iter().map(|sum| {
            let mut out = vec![B128::ZERO; 128];
            for (q, s) in sum.into_iter().enumerate() {
                vst1q_u8(&mut out[8 * (q % 16) + q / 16].0 as *mut u128 as *mut u8, s);
            }
            out
        }).collect()
    }
}

impl ByteTables {
    #[inline(always)]
    pub(super) fn apply(&self, x: B128) -> F162 {
        unsafe {
            let mut lo = vdupq_n_u64(0);
            let mut top = 0u64;
            for b in 0..16 {
                let s = 256 * b + ((x.0 >> (8 * b)) & 255) as usize;
                lo = veorq_u64(lo, vld1q_u64(self.lo.as_ptr().add(s) as *const u64));
                top ^= *self.top.get_unchecked(s);
            }
            F162([vgetq_lane_u64::<0>(lo), vgetq_lane_u64::<1>(lo), top])
        }
    }
}

/// `acc += w * (two words p)`, unreduced (limb product 2 is always zero: B128 has two limbs).
#[inline(always)]
unsafe fn mat_step(acc: &mut [[uint64x2_t; 6]; 2], w: &[uint64x2_t; 6], p: uint64x2x2_t) {
    let k = [p.0, p.1, vdupq_n_u64(0), x2(p.0, p.1), p.0, p.1];
    for j in [0, 1, 3, 4, 5] {
        acc[0][j] = x2(acc[0][j], pmul_lo(w[j], k[j]));
        acc[1][j] = x2(acc[1][j], pmul_hi(w[j], k[j]));
    }
}

#[target_feature(enable = "aes")]
pub(super) unsafe fn materialize_prefix(words: &[B128], weights: &[F162], start: usize, out: [&mut [u64]; 3]) {
    let stride = words.len() / weights.len();
    let [o0, o1, o2] = out;
    let len = o0.len();
    assert!(o1.len() == len && o2.len() == len && start + len <= stride && len % 2 == 0);
    let wk: Vec<_> = weights.iter().map(|w| kara(w.0.map(|x| vdupq_n_u64(x)))).collect();
    const PAIRS: usize = 80;
    let z = vdupq_n_u64(0);
    let mut scratch = vec![[[z; 6]; 2]; PAIRS];
    for first in (0..len).step_by(2 * PAIRS) {
        let pairs = ((len - first) / 2).min(PAIRS);
        let next = first + 2 * PAIRS < len;
        for (group, ws) in wk.chunks(8).enumerate() {
            // This group's rows of the next chunk into L2.
            for g in 0..ws.len() {
                let row = words.as_ptr().add((8 * group + g) * stride + start + first) as *const u8;
                if next {
                    prefetch_l2(row.wrapping_add(64 * PAIRS), 64 * PAIRS);
                }
            }
            // Two pairs (one 64-byte line of each row) per step: the eight rows of a group sit 4 MiB
            // apart, so their lines share one L1 set; reading a line once instead of once per pair
            // halves the L1 refills (a 4-way set cannot hold eight rows' lines between two pairs).
            let mut i = 0;
            while i + 2 <= pairs {
                let mut a = if group == 0 { [[z; 6]; 2] } else { scratch[i] };
                let mut b = if group == 0 { [[z; 6]; 2] } else { scratch[i + 1] };
                for (g, w) in ws.iter().enumerate() {
                    let q = 8 * group + g;
                    let row = words.as_ptr().add(q * stride + start + first + 2 * i) as *const u64;
                    mat_step(&mut a, w, vld2q_u64(row));
                    mat_step(&mut b, w, vld2q_u64(row.add(4)));
                }
                scratch[i] = a;
                scratch[i + 1] = b;
                i += 2;
            }
            if i < pairs {
                let mut acc = if group == 0 { [[z; 6]; 2] } else { scratch[i] };
                for (g, w) in ws.iter().enumerate() {
                    let q = 8 * group + g;
                    mat_step(&mut acc, w, vld2q_u64(words.as_ptr().add(q * stride + start + first + 2 * i) as *const u64));
                }
                scratch[i] = acc;
            }
        }
        for (i, acc) in scratch[..pairs].iter().enumerate() {
            let a = reduce(assemble(acc[0].map(|v| to_u128(v))));
            let b = reduce(assemble(acc[1].map(|v| to_u128(v))));
            let j = first + 2 * i;
            for (dst, k) in [(&mut *o0, 0), (&mut *o1, 1), (&mut *o2, 2)] {
                dst[j] = a.0[k]; dst[j + 1] = b.0[k];
            }
        }
    }
}


#[target_feature(enable = "aes")]
pub(super) unsafe fn linear_message(eq: &[B128], lo: [&[u64]; 3], hi: [&[u64]; 3], l: &ByteTables, lp: &ByteTables) -> [F162; 2] {
    let n = eq.len();
    assert!(lo.iter().chain(hi.iter()).all(|s| s.len() == n));
    let mut acc0 = [vdupq_n_u64(0); 6];
    let mut acci = acc0;
    for j in (0..n / 2 * 2).step_by(2) {
        let a = [l.apply(eq[j]), l.apply(eq[j + 1])];
        let d = [lp.apply(eq[j]), lp.apply(eq[j + 1])];
        let pack = |a: [F162; 2]| core::array::from_fn(|k| vcombine_u64(vcreate_u64(a[0].0[k]), vcreate_u64(a[1].0[k])));
        let (p0, p1) = (load2(lo, j), load2(hi, j));
        mac2(&mut acc0, pack(a), p0);
        mac2(&mut acci, pack(d), core::array::from_fn(|k| x2(p0[k], p1[k])));
    }
    let mut out = [reduce(assemble(acc0.map(|v| to_u128(v)))), reduce(assemble(acci.map(|v| to_u128(v))))];
    if n % 2 != 0 {
        let j = n - 1;
        let p0 = F162(core::array::from_fn(|k| lo[k][j]));
        let p1 = F162(core::array::from_fn(|k| hi[k][j]));
        out[0] += l.apply(eq[j]) * p0;
        out[1] += lp.apply(eq[j]) * (p0 + p1);
    }
    out
}

#[target_feature(enable = "aes")]
pub(super) unsafe fn fold_linear_message(
    eq: &[B128], q0: [&mut [u64]; 3], q1: [&mut [u64]; 3],
    h0: [&[u64]; 3], h1: [&[u64]; 3], r: F162, l: &ByteTables, lp: &ByteTables,
) -> [F162; 2] {
    let n = eq.len();
    assert!(q0.iter().chain(q1.iter()).all(|s| s.len() == n));
    assert!(h0.iter().chain(h1.iter()).all(|s| s.len() == n));
    let rk = kara(r.0.map(|x| vdupq_n_u64(x)));
    let mut acc0 = [vdupq_n_u64(0); 6];
    let mut acci = acc0;
    for j in (0..n / 2 * 2).step_by(2) {
        let a = [l.apply(eq[j]), l.apply(eq[j + 1])];
        let d = [lp.apply(eq[j]), lp.apply(eq[j + 1])];
        let pack = |a: [F162; 2]| core::array::from_fn(|k| vcombine_u64(vcreate_u64(a[0].0[k]), vcreate_u64(a[1].0[k])));
        let get = |q: &[&mut [u64]; 3]| core::array::from_fn(|k| vld1q_u64(q[k].as_ptr().add(j)));
        let p0 = fold2(get(&q0), load2(h0, j), rk);
        let p1 = fold2(get(&q1), load2(h1, j), rk);
        mac2(&mut acc0, pack(a), p0);
        mac2(&mut acci, pack(d), core::array::from_fn(|k| x2(p0[k], p1[k])));
        for k in 0..3 {
            vst1q_u64(q0[k].as_mut_ptr().add(j), p0[k]);
            vst1q_u64(q1[k].as_mut_ptr().add(j), p1[k]);
        }
    }
    let mut out = [reduce(assemble(acc0.map(|v| to_u128(v)))), reduce(assemble(acci.map(|v| to_u128(v))))];
    if n % 2 != 0 {
        let j = n - 1;
        let get = |q: &[&mut [u64]; 3]| F162(core::array::from_fn(|k| q[k][j]));
        let (a, b) = (get(&q0), get(&q1));
        let p0 = a + r * (a + F162(core::array::from_fn(|k| h0[k][j])));
        let p1 = b + r * (b + F162(core::array::from_fn(|k| h1[k][j])));
        for k in 0..3 { q0[k][j] = p0.0[k]; q1[k][j] = p1.0[k]; }
        out[0] += l.apply(eq[j]) * p0;
        out[1] += lp.apply(eq[j]) * (p0 + p1);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u128);
    impl Rng {
        fn next(&mut self) -> u128 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x.wrapping_mul(0x2545F4914F6CDD1D_9E3779B97F4A7C15) ^ (x >> 64)
        }
        fn f162(&mut self) -> F162 {
            let (a, b) = (self.next(), self.next());
            F162([a as u64, (a >> 64) as u64, (b as u64) & ((1 << 34) - 1)])
        }
    }

    #[test]
    fn patterns_transpose_bytes() {
        let mut rng = Rng(0x1234_5678_9abc_def0_0fed_cba9_8765_4321);
        let rows: Vec<B128> = (0..8).map(|_| B128(rng.next())).collect();
        let mut out = [0u8; 128];
        unsafe { patterns(rows.as_ptr(), &mut out) };
        for t in 0..8 {
            for b in 0..16 {
                let k = 8 * b + t;
                let want = (0..8).fold(0u8, |s, i| s | ((rows[i].bit(k) as u8) << i));
                assert_eq!(out[16 * t + b], want, "t {t} b {b}");
            }
        }
    }

    #[test]
    fn partial_evals_match_the_bitwise_sum() {
        let mut rng = Rng(0x9E3779B97F4A7C15_0123_4567_89AB_CDEF);
        for n in [0usize, 1, 7, 8, 9, 63, 64, 65, 127, 1000, 4103] {
            let pi0: Vec<B128> = (0..n).map(|_| B128(rng.next())).collect();
            let eq: Vec<B128> = (0..n).map(|_| B128(rng.next())).collect();
            let mut want = vec![B128::ZERO; 128];
            for (&p, &e) in pi0.iter().zip(&eq) {
                for (k, w) in want.iter_mut().enumerate() {
                    if p.bit(k) {
                        *w = *w + e;
                    }
                }
            }
            assert_eq!(partial_evals_rows(&pi0, &eq), want, "n {n}");
        }
    }

    /// `lookup_pass` before the 64-bit index loads: the reference for the regression test.
    unsafe fn lookup_pass_ref(tabs: &[[uint8x16_t; 256]], pats: &[[u8; 128]; PASS], sum: &mut [uint8x16_t; 128]) {
        let t: [*const uint8x16_t; PASS] = core::array::from_fn(|x| tabs.get_unchecked(x).as_ptr());
        let p = pats.as_ptr() as *const u8;
        for k in 0..128 {
            let v = |x: usize| *t[x].add(*p.add(128 * x + k) as usize);
            let a = veorq_u8(veorq_u8(v(0), v(1)), veorq_u8(v(2), v(3)));
            let b = veorq_u8(veorq_u8(v(4), v(5)), veorq_u8(v(6), v(7)));
            let s = sum.get_unchecked_mut(k);
            *s = veorq_u8(*s, veorq_u8(a, b));
        }
    }

    #[test]
    fn lookup_pass_matches_the_byte_index_version() {
        let mut rng = Rng(0xA303_0002_7AB1_E5ED_0123_4567_89AB_CDEF);
        let rand_v = |rng: &mut Rng| unsafe { vld1q_u8(rng.next().to_le_bytes().as_ptr()) };
        let eq_bytes = |v: uint8x16_t| unsafe { let mut b = [0u8; 16]; vst1q_u8(b.as_mut_ptr(), v); b };
        for round in 0..20 {
            let tabs: Vec<[uint8x16_t; 256]> = (0..PASS).map(|_| core::array::from_fn(|_| rand_v(&mut rng))).collect();
            let mut pats = [[0u8; 128]; PASS];
            for row in pats.iter_mut() {
                for b in row.iter_mut() {
                    // Extremes too: index 0 and 255 in every entry of rounds 0 and 1.
                    *b = match round { 0 => 0, 1 => 255, _ => rng.next() as u8 };
                }
            }
            let init: [uint8x16_t; 128] = core::array::from_fn(|_| rand_v(&mut rng));
            let (mut got, mut want) = (init, init);
            unsafe {
                lookup_pass(&tabs, &pats, &mut got);
                lookup_pass_ref(&tabs, &pats, &mut want);
            }
            for k in 0..128 {
                assert_eq!(eq_bytes(got[k]), eq_bytes(want[k]), "round {round} k {k}");
            }
        }
    }

    #[test]
    fn partial_sets_match_the_bitwise_sum() {
        let mut rng = Rng(0x0A03_0A03_1357_9BDF_2468_ACE0_FDB9_7531);
        for (stride, blocks, start, end) in [(64usize, 1usize, 0usize, 64usize), (128, 3, 0, 128), (200, 2, 64, 200), (1024, 4, 512, 1024), (72, 2, 0, 72)] {
            let words: Vec<B128> = (0..stride * blocks).map(|_| B128(rng.next())).collect();
            let eq: Vec<B128> = (0..stride).map(|_| B128(rng.next())).collect();
            let (start, end) = (start / GROUP * GROUP, end / GROUP * GROUP);
            let got = partial_evals_sets(&words, &eq, start, end);
            for q in 0..blocks {
                let mut want = vec![B128::ZERO; 128];
                for j in start..end {
                    let p = words[q * stride + j];
                    for (k, w) in want.iter_mut().enumerate() {
                        if p.bit(k) {
                            *w = *w + eq[j];
                        }
                    }
                }
                assert_eq!(got[q], want, "stride {stride} block {q}");
            }
        }
    }

    #[test]
    fn byte_tables_match_the_bitwise_sum() {
        let mut rng = Rng(0x0BAD_F00D_DEAD_BEEF_0123_4567_89AB_CDEF);
        let batches: Vec<Vec<F162>> = (0..2).map(|_| (0..128).map(|_| rng.f162()).collect()).collect();
        let tabs: Vec<ByteTables> = batches.iter().map(|b| ByteTables::new(b)).collect();
        for claims in [1usize, 2] {
            for n in [0usize, 1, 2, 3, 17, 64, 130] {
                let eqs: Vec<Vec<B128>> = (0..claims)
                    .map(|_| (0..n).map(|_| B128(rng.next())).collect())
                    .collect();
                let views: Vec<&[B128]> = eqs.iter().map(|e| e.as_slice()).collect();
                let mut w = [vec![7u64; n], vec![7u64; n], vec![7u64; n]];
                let [w0, w1, w2] = &mut w;
                poly_from_eq_rows(&views, &tabs[..claims], [&mut w0[..], &mut w1[..], &mut w2[..]]);
                for j in 0..n {
                    let mut want = F162::ZERO;
                    for c in 0..claims {
                        for k in 0..128 {
                            if eqs[c][j].bit(k) {
                                want += batches[c][k];
                            }
                        }
                    }
                    assert_eq!(F162([w[0][j], w[1][j], w[2][j]]), want, "n {n} j {j}");
                }
                let p: Vec<B128> = (0..n).map(|_| B128(rng.next())).collect();
                let [w0, w1, w2] = &mut w;
                poly_from_b128_rows(&p, [&mut w0[..], &mut w1[..], &mut w2[..]]);
                for j in 0..n {
                    assert_eq!(F162([w[0][j], w[1][j], w[2][j]]), F162::from_b128(p[j]));
                }
            }
        }
    }

    fn view(w: &[Vec<u64>; 3]) -> [&[u64]; 3] {
        [&w[0][..], &w[1][..], &w[2][..]]
    }

    #[test]
    fn tiled_prefix_binding_matches_scalar_products() {
        if !pmull() { return; }
        let mut rng = Rng(0xCA03_B10C_6F12_ABCD_0123_4567_89AB_CDEF);
        for blocks in [1usize, 2, 8, 16, 64] {
            let weights: Vec<_> = (0..blocks).map(|_| rng.f162()).collect();
            for (start, n) in [(0, 2usize), (2, 8), (0, 64), (4, 70), (6, 130)] {
                let words: Vec<_> = (0..blocks * (start + n)).map(|_| B128(rng.next())).collect();
                let mut got = [vec![0; n], vec![0; n], vec![0; n]];
                unsafe {
                    let [a, b, c] = &mut got;
                    materialize_prefix(&words, &weights, start, [a, b, c]);
                }
                let stride = start + n;
                for j in 0..n {
                    let want = weights.iter().enumerate().fold(F162::ZERO, |a, (q, &w)| {
                        a + w * F162::from_b128(words[q * stride + start + j])
                    });
                    assert_eq!(F162(core::array::from_fn(|k| got[k][j])), want,
                        "blocks={blocks} start={start} n={n} j={j}");
                }
            }
        }
    }

    #[test]
    fn five_product_dot_matches_scalar_products() {
        let mut rng = Rng(0x50A1_1280_F162_ABCD_9876_0011_A9E0_FEDC);
        for n in [0usize, 1, 2, 3, 16, 129, 1025] {
            let a: Vec<_> = (0..n).map(|_| rng.f162()).collect();
            let b: Vec<_> = (0..n).map(|_| B128(rng.next())).collect();
            let want = a.iter().zip(&b).fold(F162::ZERO, |sum, (&x, &y)| sum + x * F162::from_b128(y));
            assert_eq!(dot_lifted(&a, &b), want, "n={n}");
            let c: Vec<_> = (0..n).map(|_| B128(rng.next())).collect();
            assert_eq!(dot_lifted2(&a, &b, &c), [dot_lifted(&a, &b), dot_lifted(&a, &c)], "n={n}");
        }
    }

    #[test]
    fn fused_fold_message_matches_separate_passes() {
        if !pmull() { return; }
        let mut rng = Rng(0xF053_DF01_D135_1A9E_6754_3210_ABCD_EF01);
        let batch: Vec<_> = (0..128).map(|_| rng.f162()).collect();
        let batch_p: Vec<_> = (0..128).map(|_| rng.f162()).collect();
        let (l, lp) = (ByteTables::new(&batch), ByteTables::new(&batch_p));
        for n in [0usize, 1, 2, 3, 8, 13, 64, 257] {
            let mut q0: [Vec<u64>; 3] = core::array::from_fn(|k| (0..n).map(|_| rng.f162().0[k]).collect());
            let mut q1: [Vec<u64>; 3] = core::array::from_fn(|k| (0..n).map(|_| rng.f162().0[k]).collect());
            let h0: [Vec<u64>; 3] = core::array::from_fn(|k| (0..n).map(|_| rng.f162().0[k]).collect());
            let h1: [Vec<u64>; 3] = core::array::from_fn(|k| (0..n).map(|_| rng.f162().0[k]).collect());
            let eq: Vec<_> = (0..n).map(|_| B128(rng.next())).collect();
            let (mut want0, mut want1) = (q0.clone(), q1.clone());
            let r = rng.f162();
            let got = unsafe {
                let [a, b, c] = &mut want0;
                fold_share([a, b, c], view(&h0), r);
                let [a, b, c] = &mut want1;
                fold_share([a, b, c], view(&h1), r);
                let [a, b, c] = &mut q0;
                let [d, e, f] = &mut q1;
                fold_linear_message(&eq, [a, b, c], [d, e, f], view(&h0), view(&h1), r, &l, &lp)
            };
            assert_eq!(q0, want0, "lower n={n}");
            assert_eq!(q1, want1, "upper n={n}");
            let want = unsafe { linear_message(&eq, view(&want0), view(&want1), &l, &lp) };
            assert_eq!(got, want, "message n={n}");
        }
    }

    #[test]
    fn sumcheck_round_matches_the_scalar_field() {
        if !pmull() {
            return;
        }
        let mut rng = Rng(0x5555_AAAA_1234_9876_FEDC_BA98_7654_3210);
        for m in [0usize, 1, 2, 3, 8, 13, 100, 257] {
            let poly = |rng: &mut Rng| -> [Vec<u64>; 3] {
                let v: Vec<F162> = (0..m).map(|_| rng.f162()).collect();
                [
                    v.iter().map(|x| x.0[0]).collect(),
                    v.iter().map(|x| x.0[1]).collect(),
                    v.iter().map(|x| x.0[2]).collect(),
                ]
            };
            let (a0, a1, p0, p1) = (poly(&mut rng), poly(&mut rng), poly(&mut rng), poly(&mut rng));
            let got = unsafe { msg_range(view(&a0), view(&a1), view(&p0), view(&p1)) };
            let get = |w: &[Vec<u64>; 3], j: usize| F162([w[0][j], w[1][j], w[2][j]]);
            let mut want = [F162::ZERO; 2];
            for j in 0..m {
                want[0] += get(&a0, j) * get(&p0, j);
                want[1] += (get(&a0, j) + get(&a1, j)) * (get(&p0, j) + get(&p1, j));
            }
            assert_eq!(got, want, "m {m}");
            let r = rng.f162();
            let mut lo = a0.clone();
            {
                let [x, y, z] = &mut lo;
                unsafe { fold_share([&mut x[..], &mut y[..], &mut z[..]], view(&a1), r) };
            }
            for j in 0..m {
                let (x0, x1) = (get(&a0, j), get(&a1, j));
                assert_eq!(get(&lo, j), x0 + r * (x0 + x1), "m {m} j {j}");
            }
        }
    }
}
