//! The hot part of [`super::commit`] on AArch64: eight columns per vector, one `i16` lane each.
//!
//! For a batch of eight columns and one ring element `m`, the binary witness elements become
//! `0 / -1` lane masks per coefficient ([`masks`]), which are the coefficients of `-s`. Each
//! prime's transform then runs on whole vectors, so every butterfly serves eight independent
//! transforms and no lane ever moves: levels 0 and 1 select constants with the masks (no
//! multiplications), level 2 is a radix-2 split and levels 3 to 6 are radix-3 splits whose
//! twiddles are Barrett products by constants (`mul`, `sqrdmulh`, `mls` against a precomputed
//! `round(w 2^15 / p)`). Values grow lazily within the `i16` bounds that
//! [`super::transform_bound`] checks at compile time and are reduced where the prime's
//! [`super::Schedule`] says. The last level's outputs are multiplied straight into the batch's
//! `i32` slot sums with `smlsl` by the matrix slot (subtracting the product with `-s` adds the
//! product with `s`); the sums are reduced every [`super::Schedule::accumulate`] elements.
use super::{for_blocks, primes, schedule, Matrix, Schedule, ALL, FIRST, N, PRIMES, ROW, SCHEDULES};
use core::arch::aarch64::*;
use labinius::fields::scalar::B128;
use std::sync::OnceLock;

/// Columns per vector.
const LANES: usize = 8;
/// Coefficients a binary witness element can have set.
const BITS: usize = 512;

/// One prime's constants in the order the passes read them. A twiddle `w` is stored as the pair
/// `(w, round(w 2^15 / p))`, both centered.
pub struct Kernel {
    /// `[p, round(2^26 / p), eps, eps', 0, 0, 0, 0]`, `eps = zeta^648`.
    k: [i16; 8],
    /// Levels 0 and 1: `[zeta^324, zeta^1620, zeta^162, zeta^486, zeta^810, zeta^486, 0, 0]`.
    a: [i16; 8],
    /// Levels 0 and 1 of a general transform, as pairs: `zeta^324, zeta^1620, zeta^162,
    /// zeta^810`.
    g: [i16; 8],
    /// Levels 2 and 3, per level-1 node `n`: the radix-3 pairs of level-2 nodes `2n`, `2n + 1`,
    /// then the radix-2 pair of node `n`.
    b: [[i16; 16]; 4],
    /// Levels 4 and 5, per level-3 node `n`: its radix-3 pairs, then those of its three
    /// children.
    c: [[i16; 16]; 24],
    /// Level 6, per two level-5 nodes: their radix-3 pairs.
    d: [[i16; 8]; 108],
    /// `round(2^31 / p)` for the reduction of the `i32` sums.
    v32: i32,
}

fn twiddle(p: u32, w: u64) -> [i16; 2] {
    let c = super::centered(w as i64, p) as i64;
    let twisted = (2 * c * 32768 + p as i64).div_euclid(2 * p as i64);
    assert!(twisted.abs() <= 1 << 14);
    [c as i16, twisted as i16]
}

impl Kernel {
    fn new(index: usize) -> Kernel {
        let plan = &primes()[index];
        let p = plan.p;
        let tw = |e: usize| twiddle(p, plan.root(e));
        // a radix-3 node of exponent e: (tau, tau^2), tau = zeta^(e / 3)
        let r3 = |e: u32| -> [i16; 4] {
            let t = e as usize / 3;
            let (x, y) = (tw(t), tw(2 * t));
            [x[0], x[1], y[0], y[1]]
        };
        let eps = tw(648);
        let v = (((1u64 << 27) / p as u64 + 1) / 2) as i16;
        let k = [p as i16, v, eps[0], eps[1], -(p as i16), 0, 0, 0];
        let c = |e: usize| super::centered(plan.root(e) as i64, p);
        let a = [c(324), c(1620), c(162), c(486), c(810), c(486), 0, 0];
        let mut g = [0i16; 8];
        for (i, e) in [324usize, 1620, 162, 810].into_iter().enumerate() {
            g[2 * i..2 * i + 2].copy_from_slice(&tw(e));
        }
        let mut b = [[0i16; 16]; 4];
        for (n, slot) in b.iter_mut().enumerate() {
            slot[..4].copy_from_slice(&r3(plan.exps[2][2 * n]));
            slot[4..8].copy_from_slice(&r3(plan.exps[2][2 * n + 1]));
            slot[8..10].copy_from_slice(&tw(plan.exps[1][n] as usize / 2));
        }
        let mut cc = [[0i16; 16]; 24];
        for (n, slot) in cc.iter_mut().enumerate() {
            slot[..4].copy_from_slice(&r3(plan.exps[3][n]));
            for k in 0..3 {
                slot[4 + 4 * k..8 + 4 * k].copy_from_slice(&r3(plan.exps[4][3 * n + k]));
            }
        }
        let mut d = [[0i16; 8]; 108];
        for (q, slot) in d.iter_mut().enumerate() {
            slot[..4].copy_from_slice(&r3(plan.exps[5][2 * q]));
            slot[4..].copy_from_slice(&r3(plan.exps[5][2 * q + 1]));
        }
        Kernel {
            k,
            a,
            g,
            b,
            c: cc,
            d,
            v32: ((1u64 << 32) / p as u64).div_ceil(2) as i32,
        }
    }
}

pub fn kernels() -> &'static [Kernel; 3] {
    static K: OnceLock<[Kernel; 3]> = OnceLock::new();
    K.get_or_init(|| [Kernel::new(0), Kernel::new(1), Kernel::new(2)])
}

/// Barrett product by the twiddle in lanes `W` (value) and `T` (twisted) of `t`; `k` holds `p` in
/// lane 0. `|result| <= product_bound(p, |a|)`.
#[inline(always)]
unsafe fn mulc<const W: i32, const T: i32>(a: int16x8_t, t: int16x8_t, k: int16x8_t) -> int16x8_t {
    let lo = vmulq_laneq_s16::<W>(a, t);
    let hi = vqrdmulhq_laneq_s16::<T>(a, t);
    // `lo - hi p` as `lo + hi (-p)`: lane 4 of `k` holds `-p`, so this is one `mla` by element
    // whose latency LLVM's scheduler knows (an `mls` intrinsic became `neg` + `mla`; the earlier
    // inline-asm `mls` was opaque to the scheduler). Equal modulo 2^16, so every value is unchanged.
    vmlaq_laneq_s16::<4>(lo, hi, k)
}

/// Barrett reduction of any `i16` lanes to `|result| <= reduced_bound(p)`, congruent mod `p`.
#[inline(always)]
pub unsafe fn reduce(a: int16x8_t, k: int16x8_t) -> int16x8_t {
    let t = vqdmulhq_laneq_s16::<1>(a, k);
    let t = vrshrq_n_s16::<11>(t);
    vmlsq_laneq_s16::<0>(a, t, k)
}

/// `a - round(a / 2^14) p`: one multiplication, `|result| <= cheap_bound(p, |a|)`, for the prime
/// close to `2^14`.
#[inline(always)]
pub unsafe fn reduce_cheap(a: int16x8_t, k: int16x8_t) -> int16x8_t {
    vmlsq_laneq_s16::<0>(a, vrshrq_n_s16::<14>(a), k)
}

/// Reduces the level's inputs the schedule asks for: all of them (the multiplied ones cheaply
/// when `cheap`), or only those at `first`, the butterflies' inputs that are not multiplied by
/// a twiddle.
#[inline(always)]
unsafe fn reduce_inputs<const L: usize>(
    v: &mut [int16x8_t; L],
    mode: u8,
    cheap: bool,
    first: &[usize],
    k: int16x8_t,
) {
    if mode == ALL {
        for (i, y) in v.iter_mut().enumerate() {
            *y = if cheap && !first.contains(&i) { reduce_cheap(*y, k) } else { reduce(*y, k) };
        }
    } else if mode == FIRST {
        for &i in first {
            v[i] = reduce(v[i], k);
        }
    }
}

/// [`radix3`] with the schedule's reduction of `d`: none, Barrett, or the cheap one.
macro_rules! r3 {
    ($s:expr, $t0:literal, $t1:literal, $t2:literal, $t3:literal, $a:expr, $b:expr, $c:expr, $t:expr, $k:expr) => {
        match ($s.reduce_d, $s.cheap) {
            (false, _) => radix3::<$t0, $t1, $t2, $t3, 0>($a, $b, $c, $t, $k),
            (true, false) => radix3::<$t0, $t1, $t2, $t3, 1>($a, $b, $c, $t, $k),
            (true, true) => radix3::<$t0, $t1, $t2, $t3, 2>($a, $b, $c, $t, $k),
        }
    };
}

/// The radix-3 butterfly of a node with `(tau, tau^2)` in lanes `T0..T3` of `t`:
/// `y_k = a + eps^k tau b + eps^2k tau^2 c`, with `eps^2 = -1 - eps`.
#[inline(always)]
unsafe fn radix3<const T0: i32, const T1: i32, const T2: i32, const T3: i32, const RD: u8>(
    a: int16x8_t,
    b: int16x8_t,
    c: int16x8_t,
    t: int16x8_t,
    k: int16x8_t,
) -> (int16x8_t, int16x8_t, int16x8_t) {
    let b1 = mulc::<T0, T1>(b, t, k);
    let c1 = mulc::<T2, T3>(c, t, k);
    let s = vaddq_s16(b1, c1);
    let mut d = vsubq_s16(b1, c1);
    if RD == 1 {
        d = reduce(d, k);
    } else if RD == 2 {
        d = reduce_cheap(d, k);
    }
    let e = mulc::<2, 3>(d, k, k);
    (
        vaddq_s16(a, s),
        vaddq_s16(vsubq_s16(a, c1), e),
        vsubq_s16(vsubq_s16(a, b1), e),
    )
}

/// `out[i]`, lane `c`: `-1` where coefficient `i` of element `m` of column `c` is set, else 0.
/// Coefficient `4 q + k` is bit `q` of word `k`; `words[c]` points at the element's four words.
#[inline(always)]
unsafe fn masks(words: &[*const u8; LANES], out: &mut [int16x8_t; BITS]) {
    let bit: [uint16x8_t; 8] = core::array::from_fn(|i| vdupq_n_u16(1 << i));
    for k in 0..4 {
        let r: [uint8x16_t; 8] = core::array::from_fn(|c| vld1q_u8(words[c].add(16 * k)));
        // three zip rounds transpose the 8 x 16 bytes: t[v] holds byte 2v of every column in
        // its low half and byte 2v + 1 in its high half
        let a: [uint8x16_t; 8] = core::array::from_fn(|i| {
            let (x, y) = (r[2 * (i % 4)], r[2 * (i % 4) + 1]);
            if i < 4 { vzip1q_u8(x, y) } else { vzip2q_u8(x, y) }
        });
        let b: [uint16x8_t; 8] = core::array::from_fn(|i| {
            let h = i / 4; // bytes 0..8 or 8..16
            let (x, y) = (
                vreinterpretq_u16_u8(a[4 * h + 2 * ((i % 4) / 2)]),
                vreinterpretq_u16_u8(a[4 * h + 2 * ((i % 4) / 2) + 1]),
            );
            if i % 2 == 0 { vzip1q_u16(x, y) } else { vzip2q_u16(x, y) }
        });
        // b[4h + 2g + z]: rows 4g..4g+4, bytes 8h + 4z .. 8h + 4z + 4
        let t: [uint8x16_t; 8] = core::array::from_fn(|v| {
            let h = v / 4;
            let z = (v % 4) / 2;
            let (x, y) = (vreinterpretq_u32_u16(b[4 * h + z]), vreinterpretq_u32_u16(b[4 * h + 2 + z]));
            vreinterpretq_u8_u32(if v % 2 == 0 { vzip1q_u32(x, y) } else { vzip2q_u32(x, y) })
        });
        for (v, &pair) in t.iter().enumerate() {
            for (half, column_bytes) in [vmovl_u8(vget_low_u8(pair)), vmovl_high_u8(pair)]
                .into_iter()
                .enumerate()
            {
                let byte = 2 * v + half;
                for i in 0..8 {
                    out[32 * byte + 4 * i + k] = vreinterpretq_s16_u16(vtstq_u16(column_bytes, bit[i]));
                }
            }
        }
    }
}

#[inline(always)]
unsafe fn load(x: &[int16x8_t; N], i: usize) -> int16x8_t {
    *x.get_unchecked(i)
}

#[inline(always)]
unsafe fn store(x: &mut [int16x8_t; N], i: usize, v: int16x8_t) {
    *x.get_unchecked_mut(i) = v;
}

/// Levels 0 and 1 on the masks: four slots per `j < 162`, sums of at most four selected
/// constants (`m & c` is `c` where the coefficient is set).
#[inline(always)]
unsafe fn pass_a(m: &[int16x8_t; BITS], x: &mut [int16x8_t; N], kern: &Kernel) {
    let a = vld1q_s16(kern.a.as_ptr());
    let (cw, cw5) = (vdupq_laneq_s16::<0>(a), vdupq_laneq_s16::<1>(a));
    let (ca, caw) = (vdupq_laneq_s16::<2>(a), vdupq_laneq_s16::<3>(a));
    let (cb, cbw) = (vdupq_laneq_s16::<4>(a), vdupq_laneq_s16::<5>(a));
    // coefficient j + 486 exists (is below 512) only for j < 26
    for j in 0..162 {
        let (m0, m1, m2) = (m[j], m[j + 162], m[j + 324]);
        let u = vsubq_s16(m0, vandq_s16(m2, cw));
        let v = vsubq_s16(m0, vandq_s16(m2, cw5));
        let mut w = vandq_s16(m1, ca);
        let mut w2 = vandq_s16(m1, cb);
        if j < BITS - 486 {
            let m3 = m[j + 486];
            w = vaddq_s16(w, vandq_s16(m3, caw));
            w2 = vaddq_s16(w2, vandq_s16(m3, cbw));
        }
        store(x, j, vsubq_s16(u, w));
        store(x, j + 162, vaddq_s16(u, w));
        store(x, j + 324, vsubq_s16(v, w2));
        store(x, j + 486, vaddq_s16(v, w2));
    }
}

/// Levels 2 and 3: per level-1 node (162 slots) and `j < 27`, the six slots `j + 27 i`. Three
/// `j` per step, loaded before any is stored (arm-big): the loads of a step do not wait on the
/// previous group's stores, and the eighteen values give the scheduler independent butterflies.
#[inline(always)]
unsafe fn pass_b<const S: usize>(x: &mut [int16x8_t; N], kern: &Kernel) {
    let s: Schedule = schedule(S);
    let k = vld1q_s16(kern.k.as_ptr());
    #[inline(always)]
    unsafe fn compute(mut v: [int16x8_t; 6], s: Schedule, t0: int16x8_t, t1: int16x8_t, k: int16x8_t) -> [int16x8_t; 6] {
        // level 2 multiplies v[3..6]
        reduce_inputs(&mut v, s.before[2], s.cheap, &[0, 1, 2], k);
        for i in 0..3 {
            let b1 = mulc::<0, 1>(v[i + 3], t1, k);
            let a = v[i];
            v[i] = vaddq_s16(a, b1);
            v[i + 3] = vsubq_s16(a, b1);
        }
        reduce_inputs(&mut v, s.before[3], s.cheap, &[0, 3], k);
        let (y0, y1, y2) = r3!(s, 0, 1, 2, 3, v[0], v[1], v[2], t0, k);
        let (y3, y4, y5) = r3!(s, 4, 5, 6, 7, v[3], v[4], v[5], t0, k);
        [y0, y1, y2, y3, y4, y5]
    }
    for n in 0..4 {
        let t0 = vld1q_s16(kern.b[n].as_ptr());
        let t1 = vld1q_s16(kern.b[n].as_ptr().add(8));
        let base = 162 * n;
        for j in (0..27).step_by(3) {
            let at = base + j;
            let v0: [int16x8_t; 6] = core::array::from_fn(|i| load(x, at + 27 * i));
            let v1: [int16x8_t; 6] = core::array::from_fn(|i| load(x, at + 1 + 27 * i));
            let v2: [int16x8_t; 6] = core::array::from_fn(|i| load(x, at + 2 + 27 * i));
            let y0 = compute(v0, s, t0, t1, k);
            let y1 = compute(v1, s, t0, t1, k);
            let y2 = compute(v2, s, t0, t1, k);
            for i in 0..6 {
                store(x, at + 27 * i, y0[i]);
                store(x, at + 1 + 27 * i, y1[i]);
                store(x, at + 2 + 27 * i, y2[i]);
            }
        }
    }
}

/// Levels 4 and 5: per level-3 node (27 slots) and `j < 3`, the nine slots `j + 3 i`. All three
/// `j` of a node in one block (arm-big): nine independent butterflies per level.
#[inline(always)]
unsafe fn pass_c<const S: usize>(x: &mut [int16x8_t; N], kern: &Kernel) {
    let s: Schedule = schedule(S);
    let k = vld1q_s16(kern.k.as_ptr());
    #[inline(always)]
    unsafe fn lvl4(x: &[int16x8_t; N], s: Schedule, t0: int16x8_t, k: int16x8_t, at: usize) -> [int16x8_t; 9] {
        let mut v: [int16x8_t; 9] = core::array::from_fn(|i| load(x, at + 3 * i));
        reduce_inputs(&mut v, s.before[4], s.cheap, &[0, 1, 2], k);
        for i in 0..3 {
            let (a, b, c) = r3!(s, 0, 1, 2, 3, v[i], v[i + 3], v[i + 6], t0, k);
            (v[i], v[i + 3], v[i + 6]) = (a, b, c);
        }
        reduce_inputs(&mut v, s.before[5], s.cheap, &[0, 3, 6], k);
        v
    }
    #[inline(always)]
    unsafe fn lvl5(x: &mut [int16x8_t; N], s: Schedule, v: [int16x8_t; 9], t0: int16x8_t, t1: int16x8_t, k: int16x8_t, at: usize) {
        let r = [
            r3!(s, 4, 5, 6, 7, v[0], v[1], v[2], t0, k),
            r3!(s, 0, 1, 2, 3, v[3], v[4], v[5], t1, k),
            r3!(s, 4, 5, 6, 7, v[6], v[7], v[8], t1, k),
        ];
        for (g, (a, b, c)) in r.into_iter().enumerate() {
            store(x, at + 9 * g, a);
            store(x, at + 9 * g + 3, b);
            store(x, at + 9 * g + 6, c);
        }
    }
    for n in 0..24 {
        let t0 = vld1q_s16(kern.c[n].as_ptr());
        let t1 = vld1q_s16(kern.c[n].as_ptr().add(8));
        let base = 27 * n;
        let v0 = lvl4(x, s, t0, k, base);
        let v1 = lvl4(x, s, t0, k, base + 1);
        lvl5(x, s, v0, t0, t1, k, base);
        let v2 = lvl4(x, s, t0, k, base + 2);
        lvl5(x, s, v1, t0, t1, k, base + 1);
        lvl5(x, s, v2, t0, t1, k, base + 2);
    }
}

/// `acc[2 s..2 s + 2] -= a_s * y` for the six slots `s = base..base + 6`, `a_s` in lanes 0..6
/// of `av`.
#[inline(always)]
unsafe fn mac6(acc: &mut [int32x4_t], base: usize, y: &[int16x8_t; 6], av: int16x8_t) {
    macro_rules! slot {
        ($i:literal) => {{
            let lo = acc.get_unchecked_mut(2 * (base + $i));
            *lo = vmlsl_laneq_s16::<$i>(*lo, vget_low_s16(y[$i]), av);
            let hi = acc.get_unchecked_mut(2 * (base + $i) + 1);
            *hi = vmlsl_high_laneq_s16::<$i>(*hi, y[$i], av);
        }};
    }
    slot!(0);
    slot!(1);
    slot!(2);
    slot!(3);
    slot!(4);
    slot!(5);
}

/// Level 6 per two level-5 nodes (six slots); its outputs, within [`super::transform_bound`],
/// are either multiplied into `acc` by the matrix row (`MAC`) or stored.
#[inline(always)]
unsafe fn pass_d<const S: usize, const MAC: bool>(
    x: &mut [int16x8_t; N],
    kern: &Kernel,
    row: &[i16],
    acc: &mut [int32x4_t],
) {
    let s: Schedule = schedule(S);
    let k = vld1q_s16(kern.k.as_ptr());
    debug_assert!(!MAC || (row.len() == ROW && acc.len() == 2 * N));
    #[inline(always)]
    unsafe fn compute(x: &[int16x8_t; N], s: Schedule, kern: &Kernel, k: int16x8_t, q: usize) -> [int16x8_t; 6] {
        let t = vld1q_s16(kern.d[q].as_ptr());
        let mut v: [int16x8_t; 6] = core::array::from_fn(|i| load(x, 6 * q + i));
        reduce_inputs(&mut v, s.before[6], s.cheap, &[0, 3], k);
        let (y0, y1, y2) = r3!(s, 0, 1, 2, 3, v[0], v[1], v[2], t, k);
        let (y3, y4, y5) = r3!(s, 4, 5, 6, 7, v[3], v[4], v[5], t, k);
        [y0, y1, y2, y3, y4, y5]
    }
    // three groups per step, all computed before any is stored (arm-big); 108 = 36 * 3
    for q in (0..N / 6).step_by(3) {
        let ys = [compute(x, s, kern, k, q), compute(x, s, kern, k, q + 1), compute(x, s, kern, k, q + 2)];
        for (d, y) in ys.iter().enumerate() {
            let base = 6 * (q + d);
            if MAC {
                let av = vld1q_s16(row.as_ptr().add(base));
                mac6(acc, base, y, av);
            } else {
                for (i, &v) in y.iter().enumerate() {
                    store(x, base + i, v);
                }
            }
        }
    }
}

/// The transform of `-s` for the eight columns' element whose masks are `m`, in `x`, within
/// [`super::transform_bound`].
#[inline(never)]
#[cfg_attr(not(test), allow(dead_code))]
pub unsafe fn transform<const P: usize>(m: &[int16x8_t; BITS], x: &mut [int16x8_t; N]) {
    let kern = &kernels()[P];
    pass_a(m, x, kern);
    pass_b::<P>(x, kern);
    pass_c::<P>(x, kern);
    pass_d::<P, false>(x, kern, &[], &mut []);
}

/// Levels 0 and 1 of a general transform: Barrett products, every value reduced in between
/// (see [`super::general_transform_bound`]).
#[inline(always)]
unsafe fn pass_a_general(x: &mut [int16x8_t; N], kern: &Kernel) {
    let g = vld1q_s16(kern.g.as_ptr());
    let k = vld1q_s16(kern.k.as_ptr());
    for j in 0..N / 2 {
        let (a, b) = (load(x, j), load(x, j + N / 2));
        store(x, j, reduce(vaddq_s16(a, mulc::<0, 1>(b, g, k)), k));
        store(x, j + N / 2, reduce(vaddq_s16(a, mulc::<2, 3>(b, g, k)), k));
    }
    for (base, j) in (0..162).map(|j| (0, j)).chain((0..162).map(|j| (324, j))) {
        let a = load(x, base + j);
        let b = load(x, base + j + 162);
        let t = if base == 0 { mulc::<4, 5>(b, g, k) } else { mulc::<6, 7>(b, g, k) };
        store(x, base + j, vaddq_s16(a, t));
        store(x, base + j + 162, vsubq_s16(a, t));
    }
}

/// The transform of eight matrix elements at once (lanes), centered residues in, values
/// congruent to the transform out (within [`super::general_transform_bound`]).
pub fn transform_general(prime: usize, x: &mut [[i16; 8]; N]) {
    // SAFETY: `[[i16; 8]; N]` and `[int16x8_t; N]` have the same size and every bit pattern is
    // valid for both; `int16x8_t` needs 16-byte alignment, so the lanes are copied.
    unsafe {
        let mut v: [int16x8_t; N] = core::array::from_fn(|j| vld1q_s16(x[j].as_ptr()));
        let kern = &kernels()[prime];
        pass_a_general(&mut v, kern);
        pass_b::<3>(&mut v, kern);
        pass_c::<3>(&mut v, kern);
        pass_d::<3, false>(&mut v, kern, &[], &mut []);
        for (o, y) in x.iter_mut().zip(&v) {
            vst1q_s16(o.as_mut_ptr(), *y);
        }
    }
}

/// One prime's constants for [`inverse`], as twiddle pairs `(w, round(w 2^15 / p))`.
pub struct InverseKernel {
    /// Per node, levels 6 down to 1 in the order [`inverse`] visits them (216, 72, 24, 8, 4, 2
    /// nodes): `[r^-1, r^-1 tau^-1, r^-1 tau^-2, eps^-1]` for a radix-3 node of exponent `e`,
    /// `tau = zeta^(e / r)`, `eps = zeta^648`; `[2^-1, 2^-1 tau^-1, 0, 0]` for a radix-2 node.
    t: Vec<[i16; 8]>,
    /// The trinomial unsplit: `[inv_split, zeta^324, 0, 0]`.
    u: [i16; 8],
}

impl InverseKernel {
    fn new(index: usize) -> InverseKernel {
        use super::{ORDER, RADICES};
        let plan = &primes()[index];
        let (p, pp) = (plan.p, plan.p as u64);
        let mut t = Vec::new();
        for level in (0..RADICES.len()).rev() {
            let r = RADICES[level];
            let ir = plan.inv_radix[r] as u64;
            for &e in &plan.exps[level] {
                let tau = e as usize / r;
                let mut row = [0i16; 8];
                for i in 0..r {
                    let s = plan.root(ORDER - tau * i % ORDER) * ir % pp;
                    row[2 * i..2 * i + 2].copy_from_slice(&twiddle(p, s));
                }
                if r == 3 {
                    row[6..8].copy_from_slice(&twiddle(p, plan.root(ORDER - ORDER / 3)));
                }
                t.push(row);
            }
        }
        assert_eq!(t.len(), 216 + 72 + 24 + 8 + 4 + 2);
        let mut u = [0i16; 8];
        u[..2].copy_from_slice(&twiddle(p, plan.inv_split as u64));
        u[2..4].copy_from_slice(&twiddle(p, plan.root(super::HALF)));
        InverseKernel { t, u }
    }
}

#[cfg_attr(not(test), allow(dead_code))]
fn inverse_kernels() -> &'static [InverseKernel; 3] {
    static K: OnceLock<[InverseKernel; 3]> = OnceLock::new();
    K.get_or_init(|| [InverseKernel::new(0), InverseKernel::new(1), InverseKernel::new(2)])
}

/// Whether [`inverse`] stays inside `i16` from inputs `|x| <= p`, reducing every level's inputs
/// (`red`) or never (values then shrink on their own: every level ends in Barrett products).
const fn inverse_fits(p: u32, red: bool) -> bool {
    use super::{product_bound, reduced_bound, RADICES};
    const MAX: u32 = i16::MAX as u32;
    let mut b = p;
    let mut level = RADICES.len();
    while level > 0 {
        level -= 1;
        if red {
            b = reduced_bound(p);
        }
        // radix 3: s0 = y0 + y1 + y2, s1/s2 = (y0 - y2/y1) +- eps^-1 (y1 - y2); radix 2: y0 +- y1
        let s = if RADICES[level] == 3 {
            let s1 = 2 * b + product_bound(p, 2 * b);
            if 3 * b > s1 { 3 * b } else { s1 }
        } else {
            2 * b
        };
        if s > MAX {
            return false;
        }
        b = product_bound(p, s);
    }
    if red {
        b = reduced_bound(p);
    }
    // b = (u - v) inv_split, reduced and centered; a = u - w b
    2 * b <= MAX && b + product_bound(p, (p - 1) / 2) <= MAX
}

/// Whether [`inverse`] reduces every level's inputs for prime `p`: only when it must.
const fn inverse_reduces(p: u32) -> bool {
    !inverse_fits(p, false)
}

const _: () = {
    let mut i = 0;
    while i < 3 {
        assert!(inverse_fits(PRIMES[i], inverse_reduces(PRIMES[i])));
        i += 1;
    }
};

/// `a` (`|a| <= p + (p - 1) / 2`) to its centered residue in `[-(p - 1) / 2, (p - 1) / 2]`;
/// `p` in lane 0 of `k`, `h = (p - 1) / 2`.
#[inline(always)]
unsafe fn center(a: int16x8_t, h: int16x8_t, k: int16x8_t) -> int16x8_t {
    let p = vdupq_laneq_s16::<0>(k);
    let a = vsubq_s16(a, vandq_s16(vreinterpretq_s16_u16(vcgtq_s16(a, h)), p));
    vaddq_s16(a, vandq_s16(vreinterpretq_s16_u16(vcltq_s16(a, vnegq_s16(h))), p))
}

/// One radix-3 level of [`inverse`]: per node (pairs in `t`) and `j < len`, `y_k = x[j + k len]`
/// becomes `a_i = r^-1 tau^-i sum_k eps^-(k i) y_k`, with `eps^-2 = -1 - eps^-1`.
#[inline(always)]
unsafe fn inverse_radix3(x: &mut [int16x8_t; N], t: &[[i16; 8]], len: usize, red: bool, k: int16x8_t) {
    debug_assert_eq!(t.len() * 3 * len, N);
    for (node, tw) in t.iter().enumerate() {
        let tv = vld1q_s16(tw.as_ptr());
        let base = 3 * len * node;
        for j in base..base + len {
            let (mut y0, mut y1, mut y2) = (load(x, j), load(x, j + len), load(x, j + 2 * len));
            if red {
                (y0, y1, y2) = (reduce(y0, k), reduce(y1, k), reduce(y2, k));
            }
            let s0 = vaddq_s16(vaddq_s16(y0, y1), y2);
            let e = mulc::<6, 7>(vsubq_s16(y1, y2), tv, k);
            let s1 = vaddq_s16(vsubq_s16(y0, y2), e);
            let s2 = vsubq_s16(vsubq_s16(y0, y1), e);
            store(x, j, mulc::<0, 1>(s0, tv, k));
            store(x, j + len, mulc::<2, 3>(s1, tv, k));
            store(x, j + 2 * len, mulc::<4, 5>(s2, tv, k));
        }
    }
}

/// One radix-2 level of [`inverse`]: `a_0 = (y0 + y1) / 2`, `a_1 = tau^-1 (y0 - y1) / 2`.
#[inline(always)]
unsafe fn inverse_radix2(x: &mut [int16x8_t; N], t: &[[i16; 8]], len: usize, red: bool, k: int16x8_t) {
    debug_assert_eq!(t.len() * 2 * len, N);
    for (node, tw) in t.iter().enumerate() {
        let tv = vld1q_s16(tw.as_ptr());
        let base = 2 * len * node;
        for j in base..base + len {
            let (mut y0, mut y1) = (load(x, j), load(x, j + len));
            if red {
                (y0, y1) = (reduce(y0, k), reduce(y1, k));
            }
            store(x, j, mulc::<0, 1>(vaddq_s16(y0, y1), tv, k));
            store(x, j + len, mulc::<2, 3>(vsubq_s16(y0, y1), tv, k));
        }
    }
}

/// The inverse of the transform modulo prime `P` on eight lanes at once ([`super::Prime::inverse`]
/// per lane): values congruent to the transform with `|x| <= p` in, the coefficients out as
/// centered residues, `|x| <= (p - 1) / 2` exactly. Levels 6 to 1 undo the binomial splits with
/// `1 / r` and `tau^-i` folded into the twiddles, then the trinomial unsplit
/// `b = (u - v) inv_split`, `a = u - w b`. Inside `i16` by [`inverse_fits`].
#[inline(never)]
#[cfg_attr(not(test), allow(dead_code))]
pub unsafe fn inverse<const P: usize>(x: &mut [int16x8_t; N]) {
    const { assert!(P < 3) };
    let red = const { inverse_reduces(PRIMES[P]) };
    let k = vld1q_s16(kernels()[P].k.as_ptr());
    let inv = &inverse_kernels()[P];
    let t = &inv.t[..];
    inverse_radix3(x, &t[..216], 1, red, k);
    inverse_radix3(x, &t[216..288], 3, red, k);
    inverse_radix3(x, &t[288..312], 9, red, k);
    inverse_radix3(x, &t[312..320], 27, red, k);
    inverse_radix2(x, &t[320..324], 81, red, k);
    inverse_radix2(x, &t[324..326], 162, red, k);
    let uv = vld1q_s16(inv.u.as_ptr());
    let h = vdupq_n_s16(((PRIMES[P] - 1) / 2) as i16);
    for j in 0..N / 2 {
        let (mut u, mut v) = (load(x, j), load(x, j + N / 2));
        if red {
            (u, v) = (reduce(u, k), reduce(v, k));
        }
        let b = center(reduce(mulc::<0, 1>(vsubq_s16(u, v), uv, k), k), h, k);
        let a = center(reduce(vsubq_s16(u, mulc::<2, 3>(b, uv, k)), k), h, k);
        store(x, j, a);
        store(x, j + N / 2, b);
    }
}

/// `acc` reduced to `|acc| <= p` lane by lane: `acc - round(acc v / 2^31) p`, `v = round(2^31 / p)`.
#[inline(always)]
unsafe fn reduce_sums(acc: &mut [int32x4_t], p: i32, v: i32) {
    for a in acc.iter_mut() {
        let t = vqrdmulhq_n_s32(*a, v);
        *a = vmlsq_n_s32(*a, t, p);
    }
}

/// One prime's slot sums for a batch: `acc[2 s + h]` lane `l` is column `4 h + l`'s sum at slot
/// `s`, reduced to `|sum| <= p`.
#[inline(never)]
unsafe fn sums<const P: usize>(
    words: &[(*const u8, usize); LANES],
    a: &Matrix,
    m: &mut [int16x8_t; BITS],
    x: &mut [int16x8_t; N],
    acc: &mut [int32x4_t],
) {
    const { assert!(P < 3) };
    let kern = &kernels()[P];
    let s: Schedule = SCHEDULES[P];
    let p = PRIMES[P] as i32;
    acc.fill(vdupq_n_s32(0));
    for e in 0..a.elements {
        let at: [*const u8; LANES] = core::array::from_fn(|c| words[c].0.add(e * words[c].1));
        masks(&at, m);
        pass_a(m, x, kern);
        pass_b::<P>(x, kern);
        pass_c::<P>(x, kern);
        pass_d::<P, true>(x, kern, a.row(e, P), acc);
        if (e + 1) % s.accumulate == 0 {
            reduce_sums(acc, p, kern.v32);
        }
    }
    reduce_sums(acc, p, kern.v32);
}

/// Zeros standing in for the missing columns of a partial batch.
static ZERO_WORDS: [u128; 4] = [0; 4];

/// The commitment of every column of `values` (columns of `4 a.elements` words).
pub fn commit(values: &[B128], a: &Matrix, workers: usize) -> Vec<u128> {
    let len = 4 * a.elements;
    assert_eq!(values.len() % len, 0);
    let columns = values.len() / len;
    let mut out = vec![0u128; columns * N];
    for_blocks(&mut out, N, LANES, workers, |first, block| {
        let mut m = Box::new([unsafe { vdupq_n_s16(0) }; BITS]);
        let mut x = Box::new([unsafe { vdupq_n_s16(0) }; N]);
        let mut acc = vec![unsafe { vdupq_n_s32(0) }; 3 * 2 * N];
        let count = block.len() / N;
        for (b, target) in block.chunks_mut(LANES * N).enumerate() {
            let col0 = first + LANES * b;
            let here = (count - LANES * b).min(LANES);
            // (first word of the column, bytes per element); missing columns read zeros
            let words: [(*const u8, usize); LANES] = core::array::from_fn(|c| {
                if c < here {
                    (values[(col0 + c) * len..].as_ptr() as *const u8, 64)
                } else {
                    (ZERO_WORDS.as_ptr() as *const u8, 0)
                }
            });
            let (a0, rest) = acc.split_at_mut(2 * N);
            let (a1, a2) = rest.split_at_mut(2 * N);
            unsafe {
                sums::<0>(&words, a, &mut m, &mut x, a0);
                sums::<1>(&words, a, &mut m, &mut x, a1);
                sums::<2>(&words, a, &mut m, &mut x, a2);
            }
            // lanes back to columns: reduce to [0, p), invert, recombine
            let lanes: Vec<[i32; 4]> =
                acc.iter().map(|&v| unsafe { core::mem::transmute::<int32x4_t, [i32; 4]>(v) }).collect();
            for (c, column) in target.chunks_exact_mut(N).enumerate() {
                let mut residues = [[0u32; N]; 3];
                for (i, r) in residues.iter_mut().enumerate() {
                    let p = PRIMES[i] as i32;
                    for (slot, value) in r.iter_mut().enumerate() {
                        let v = lanes[i * 2 * N + 2 * slot + c / 4][c % 4];
                        *value = v.rem_euclid(p) as u32;
                    }
                }
                super::finish_column(&residues, column);
            }
        }
    });
    out
}

#[path = "ntt_hybrid.rs"]
pub mod hybrid;

#[path = "ntt_sms.rs"]
pub mod sms;

#[cfg(test)]
mod tests {
    use super::super::{product_bound, reduced_bound, tests::xorshift};
    use super::*;

    fn lanes(v: int16x8_t) -> [i16; 8] {
        unsafe { core::mem::transmute(v) }
    }

    fn every_i16() -> impl Iterator<Item = int16x8_t> {
        (0..1 << 13).map(|i| unsafe {
            let a: [i16; 8] = core::array::from_fn(|l| (i * 8 + l as i32 - 32768) as i16);
            vld1q_s16(a.as_ptr())
        })
    }

    /// The Barrett reduction, over every `i16`: congruent, and within [`reduced_bound`].
    #[test]
    fn reduction_is_exact_and_bounded() {
        for (i, kern) in kernels().iter().enumerate() {
            let p = PRIMES[i] as i32;
            let k = unsafe { vld1q_s16(kern.k.as_ptr()) };
            let mut worst = 0;
            for a in every_i16() {
                let r = lanes(unsafe { reduce(a, k) });
                for (x, y) in lanes(a).into_iter().zip(r) {
                    assert_eq!((x as i32 - y as i32).rem_euclid(p), 0);
                    worst = worst.max((y as i32).abs());
                }
            }
            assert!(worst as u32 <= reduced_bound(p as u32), "prime {p}: {worst}");
        }
    }

    /// The cheap reduction, over every `i16`, for the primes whose schedule uses it: congruent,
    /// and within [`super::super::cheap_bound`] of `|a|`.
    #[test]
    fn cheap_reduction_is_exact_and_bounded() {
        for (i, kern) in kernels().iter().enumerate() {
            if !SCHEDULES[i].cheap {
                continue;
            }
            let p = PRIMES[i] as i32;
            let k = unsafe { vld1q_s16(kern.k.as_ptr()) };
            for a in every_i16() {
                let r = lanes(unsafe { reduce_cheap(a, k) });
                for (x, y) in lanes(a).into_iter().zip(r) {
                    assert_eq!((x as i32 - y as i32).rem_euclid(p), 0);
                    let bound = super::super::cheap_bound(p as u32, x.unsigned_abs() as u32);
                    assert!(y.unsigned_abs() as u32 <= bound, "prime {p}: {x} -> {y}");
                }
            }
        }
    }

    /// Every twiddle's Barrett product, over every `i16`: congruent to `a w` and within
    /// [`product_bound`] of `|a|`.
    #[test]
    fn twiddle_products_are_exact_and_bounded() {
        for (i, kern) in kernels().iter().enumerate() {
            let p = PRIMES[i] as i64;
            let k = unsafe { vld1q_s16(kern.k.as_ptr()) };
            let mut pairs: Vec<[i16; 2]> = vec![[kern.k[2], kern.k[3]]];
            for row in kern.b.iter().chain(kern.c.iter()) {
                pairs.extend(row.chunks_exact(2).map(|c| [c[0], c[1]]));
            }
            for row in kern.d.iter() {
                pairs.extend(row.chunks_exact(2).map(|c| [c[0], c[1]]));
            }
            for [w, wt] in pairs {
                let t = unsafe { vld1q_s16([w, wt, 0, 0, 0, 0, 0, 0].as_ptr()) };
                for a in every_i16() {
                    let r = lanes(unsafe { mulc::<0, 1>(a, t, k) });
                    for (x, y) in lanes(a).into_iter().zip(r) {
                        assert_eq!((x as i64 * w as i64 - y as i64).rem_euclid(p), 0);
                        assert!(y.unsigned_abs() as u32 <= product_bound(p as u32, x.unsigned_abs() as u32));
                    }
                }
            }
        }
    }

    fn run_inverse(i: usize, x: &[[i16; 8]; N]) -> [[i16; 8]; N] {
        unsafe {
            let mut v: [int16x8_t; N] = core::array::from_fn(|j| vld1q_s16(x[j].as_ptr()));
            match i {
                0 => inverse::<0>(&mut v),
                1 => inverse::<1>(&mut v),
                _ => inverse::<2>(&mut v),
            }
            core::array::from_fn(|j| lanes(v[j]))
        }
    }

    /// The vector inverse agrees lane by lane with the scalar inverse, centered, on random
    /// inputs in `[-p, p]` and on the extremes `+-p`.
    #[test]
    fn inverse_matches_scalar() {
        let mut state = 11u64;
        for (i, plan) in primes().iter().enumerate() {
            let p = plan.p as i64;
            for case in 0..8 {
                let x: [[i16; 8]; N] = core::array::from_fn(|_| {
                    core::array::from_fn(|l| match (case, l % 2) {
                        (0, _) => p as i16,
                        (1, _) => -p as i16,
                        (2, 0) | (3, 1) => p as i16,
                        (2, _) | (3, _) => -p as i16,
                        (4, _) => if xorshift(&mut state) & 1 == 0 { p as i16 } else { -p as i16 },
                        _ => ((xorshift(&mut state) % (2 * p + 1) as u64) as i64 - p) as i16,
                    })
                });
                let y = run_inverse(i, &x);
                for c in 0..8 {
                    let mut s: [u32; N] = core::array::from_fn(|j| (x[j][c] as i64).rem_euclid(p) as u32);
                    plan.inverse(&mut s);
                    for j in 0..N {
                        let want = super::super::centered(s[j] as i64, plan.p);
                        assert_eq!(y[j][c], want, "prime {p} case {case} lane {c} coefficient {j}");
                    }
                }
            }
        }
    }

    /// Scalar forward then vector inverse is the identity, centered; the transform values enter
    /// as `[0, p)`, centered, or shifted by `-p`, by lane.
    #[test]
    fn inverse_round_trip() {
        let mut state = 23u64;
        for (i, plan) in primes().iter().enumerate() {
            let p = plan.p as i64;
            for _ in 0..3 {
                let coeffs: [[u32; N]; 8] =
                    core::array::from_fn(|_| core::array::from_fn(|_| (xorshift(&mut state) % p as u64) as u32));
                let mut x = [[0i16; 8]; N];
                for (c, a) in coeffs.iter().enumerate() {
                    let mut y = *a;
                    plan.forward(&mut y);
                    for j in 0..N {
                        let v = y[j] as i64;
                        x[j][c] = match c % 3 {
                            0 => v,
                            1 => super::super::centered(v, plan.p) as i64,
                            _ => v - p,
                        } as i16;
                    }
                }
                let out = run_inverse(i, &x);
                for (c, a) in coeffs.iter().enumerate() {
                    for j in 0..N {
                        assert_eq!(out[j][c], super::super::centered(a[j] as i64, plan.p), "prime {p} lane {c} coefficient {j}");
                    }
                }
            }
        }
    }

    /// Rough timing of one `inverse::<1>` call: `cargo test --release inverse_timing -- --ignored
    /// --nocapture`.
    #[test]
    #[ignore]
    fn inverse_timing() {
        let mut state = 5u64;
        let mut v: [int16x8_t; N] = core::array::from_fn(|_| unsafe {
            let a: [i16; 8] = core::array::from_fn(|_| (xorshift(&mut state) % 9721) as i16);
            vld1q_s16(a.as_ptr())
        });
        unsafe { inverse::<1>(&mut v) };
        let iters = 200_000;
        let start = std::time::Instant::now();
        for _ in 0..iters {
            unsafe { inverse::<1>(core::hint::black_box(&mut v)) };
        }
        let ns = start.elapsed().as_nanos() as f64 / iters as f64;
        println!("inverse::<1>: {ns:.0} ns per call (8 lanes)");
    }

    /// The vector transform of eight binary elements agrees lane by lane with the scalar
    /// transform of `-s`.
    #[test]
    fn transform_matches_scalar() {
        let mut state = 7u64;
        for case in 0..12 {
            // eight columns' words for one element: random, all ones, zero, single bits
            let words: [[u128; 4]; 8] = core::array::from_fn(|c| {
                core::array::from_fn(|k| match (case, c) {
                    (0, _) => u128::MAX,
                    (1, _) => 0,
                    (2, 0) => (k == 0) as u128,
                    (2, 1) => ((k == 3) as u128) << 127,
                    (2, 2) => if k == 1 { 1 << 64 } else { 0 },
                    _ => (xorshift(&mut state) as u128) << 64 | xorshift(&mut state) as u128,
                })
            });
            let ptrs: [*const u8; 8] = core::array::from_fn(|c| words[c].as_ptr() as *const u8);
            let mut m = [unsafe { vdupq_n_s16(0) }; BITS];
            unsafe { masks(&ptrs, &mut m) };
            for (i, plan) in primes().iter().enumerate() {
                let mut x = [unsafe { vdupq_n_s16(0) }; N];
                unsafe {
                    match i {
                        0 => transform::<0>(&m, &mut x),
                        1 => transform::<1>(&m, &mut x),
                        _ => transform::<2>(&m, &mut x),
                    }
                }
                for c in 0..8 {
                    let b: Vec<B128> = words[c].iter().map(|&w| B128(w)).collect();
                    let mut s = [0u32; N];
                    super::super::element_bits(&b, &mut s);
                    plan.forward(&mut s);
                    for (slot, &want) in s.iter().enumerate() {
                        let got = lanes(x[slot])[c] as i64;
                        let bound = super::super::transform_bound(plan.p, SCHEDULES[i]);
                        assert!(got.unsigned_abs() <= bound as u64);
                        assert_eq!(
                            (got + want as i64).rem_euclid(plan.p as i64),
                            0,
                            "case {case} prime {} column {c} slot {slot}",
                            plan.p
                        );
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod pass_timing {
    use super::*;

    fn xs(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    /// Per-pass timing of the binary transform: `cargo test --release pass_timing -- --ignored
    /// --nocapture` (arm-big diagnostics).
    #[test]
    #[ignore]
    fn pass_timing() {
        let mut state = 11u64;
        let words: [[u128; 4]; 8] =
            core::array::from_fn(|_| core::array::from_fn(|_| (xs(&mut state) as u128) << 64 | xs(&mut state) as u128));
        let ptrs: [*const u8; 8] = core::array::from_fn(|c| words[c].as_ptr() as *const u8);
        let mut m = Box::new([unsafe { vdupq_n_s16(0) }; BITS]);
        let mut x = Box::new([unsafe { vdupq_n_s16(0) }; N]);
        let mut acc = vec![unsafe { vdupq_n_s32(0) }; 2 * N];
        let row = vec![1i16; ROW];
        let iters = 100_000;
        macro_rules! time {
            ($name:expr, $body:expr) => {{
                let start = std::time::Instant::now();
                for _ in 0..iters {
                    unsafe { $body };
                    core::hint::black_box(&mut *x);
                }
                let ns = start.elapsed().as_nanos() as f64 / iters as f64;
                println!("{:>10}: {ns:7.0} ns", $name);
            }};
        }
        fn kp<const P: usize>() -> &'static Kernel { &kernels()[P] }
        // peak: 12 independent mulc chains, 1000 rounds each
        {
            let k = unsafe { vld1q_s16(kernels()[1].k.as_ptr()) };
            let t = unsafe { vld1q_s16(kernels()[1].d[3].as_ptr()) };
            let mut v: [int16x8_t; 12] = core::array::from_fn(|i| unsafe { vdupq_n_s16(i as i16 + 3) });
            let start = std::time::Instant::now();
            for _ in 0..iters / 10 {
                for _ in 0..100 {
                    for y in v.iter_mut() {
                        *y = unsafe { mulc::<0, 1>(*y, t, k) };
                    }
                }
                core::hint::black_box(&mut v);
            }
            let ns = start.elapsed().as_nanos() as f64 / (iters as f64 * 10.0 * 12.0);
            println!("mulc peak: {ns:.3} ns per mulc (8 lanes)");
            let start = std::time::Instant::now();
            for _ in 0..iters / 10 {
                for _ in 0..100 {
                    for y in v.iter_mut() {
                        *y = unsafe { vmulq_laneq_s16::<0>(*y, t) };
                    }
                }
                core::hint::black_box(&mut v);
            }
            let ns = start.elapsed().as_nanos() as f64 / (iters as f64 * 10.0 * 12.0);
            println!("mul peak: {ns:.3} ns per vmul (8 lanes)");
        }
        time!("masks", masks(&ptrs, &mut m));
        for p in 0..2 {
            println!("prime {}", PRIMES[p]);
            if p == 0 {
                time!("pass_a", pass_a(&m, &mut x, kp::<0>()));
                time!("pass_b", pass_b::<0>(&mut x, kp::<0>()));
                time!("pass_c", pass_c::<0>(&mut x, kp::<0>()));
                time!("pass_d", pass_d::<0, false>(&mut x, kp::<0>(), &[], &mut []));
                time!("pass_dmac", pass_d::<0, true>(&mut x, kp::<0>(), &row, &mut acc));
            } else {
                time!("pass_a", pass_a(&m, &mut x, kp::<1>()));
                time!("pass_b", pass_b::<1>(&mut x, kp::<1>()));
                time!("pass_c", pass_c::<1>(&mut x, kp::<1>()));
                time!("pass_d", pass_d::<1, false>(&mut x, kp::<1>(), &[], &mut []));
                time!("pass_dmac", pass_d::<1, true>(&mut x, kp::<1>(), &row, &mut acc));
            }
        }
    }
}
