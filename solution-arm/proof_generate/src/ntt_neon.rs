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
//!
//! The SMS commitment ([`sms`]) runs levels 0 to 2 and the level-3 twiddles as lookups instead
//! ([`nibbles`], [`pass_lut`]): per level-1 node, 16-entry `i16` tables indexed by the nibble of
//! the coefficients `j + 162 g`, one `vqtbl2q_u8` per lookup and two lookups per level-3 input,
//! within [`super::lut_transform_bound`] (the library's x86 `bin_asm` does the same).
use super::{for_blocks, primes, schedule, Matrix, Schedule, ALL, FIRST, FIRST13, N, PRIMES, ROW, SCHEDULES};
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
    /// Levels 0 to 2 and the level-3 twiddles of the binary transform as nibble tables
    /// ([`pass_lut`]), per level-1 node `n`: table `t` holds, for each nibble `v` (bit `g` the
    /// coefficient `j + 162 g`), the centered residue of `f_t` times node `n`'s level-1 value of
    /// `-s` ([`pass_a`]'s formula on that nibble), where, with `tau` node `n`'s radix-2 twiddle
    /// and `z_0` the radix-3 twiddle of its level-2 child `2 n`, `f` is
    /// `[1, tau, z_0, z_0 tau, z_0^2, z_0^2 tau]`. The other child's twiddles
    /// are `(1 + eps) z_0` and `eps z_0^2`, so its butterfly shares these lookups.
    lut: [[[i16; 16]; 6]; 4],
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
        let k = [p as i16, v, eps[0], eps[1], 0, 0, 0, 0];
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
        let mut lut = [[[0i16; 16]; 6]; 4];
        for (n, tables) in lut.iter_mut().enumerate() {
            let tau = plan.root(plan.exps[1][n] as usize / 2);
            let z = |h: usize| plan.root(plan.exps[2][2 * n + h] as usize / 3);
            let pp = p as u64;
            let f = [
                1,
                tau,
                z(0),
                z(0) * tau % pp,
                z(0) * z(0) % pp,
                z(0) * z(0) % pp * tau % pp,
            ];
            // Adjacent radix-2 children differ in exponent by 972. Their
            // radix-3 twiddles differ by zeta^324 = 1 + eps, whose square is eps.
            assert_eq!(z(1), z(0) * ((1 + plan.root(648)) % pp) % pp);
            assert_eq!(z(1) * z(1) % pp, z(0) * z(0) % pp * plan.root(648) % pp);
            for v in 0..16 {
                let level1 = lut_level1(&a, n, v) as i64;
                for (t, table) in tables.iter_mut().enumerate() {
                    table[v] = super::centered(level1 * f[t] as i64, p);
                }
            }
        }
        // `lut_transform_bound` takes every entry centered
        assert!(lut.iter().flatten().flatten().all(|e| e.unsigned_abs() as u32 <= (p - 1) / 2));
        Kernel {
            k,
            a,
            g,
            b,
            c: cc,
            d,
            v32: ((1u64 << 32) / p as u64).div_ceil(2) as i32,
            lut,
        }
    }
}

/// [`pass_a`]'s output for level-1 node `n` (slot `j + 162 n`) on the nibble `v`, bit `g` set
/// where coefficient `j + 162 g` is, with the same constants `a` (the field of [`Kernel`]):
/// `m & c` is `c` where the mask is set.
fn lut_level1(a: &[i16; 8], n: usize, v: usize) -> i32 {
    let m = |g: usize| -(((v >> g) & 1) as i32);
    let sel = |g: usize, c: i16| if m(g) != 0 { c as i32 } else { 0 };
    let (cw, cw5, ca, caw, cb, cbw) = (a[0], a[1], a[2], a[3], a[4], a[5]);
    let u = m(0) - sel(2, cw);
    let v5 = m(0) - sel(2, cw5);
    let w = sel(1, ca) + sel(3, caw);
    let w2 = sel(1, cb) + sel(3, cbw);
    [u - w, u + w, v5 - w2, v5 + w2][n]
}

pub fn kernels() -> &'static [Kernel; 3] {
    static K: OnceLock<[Kernel; 3]> = OnceLock::new();
    K.get_or_init(|| [Kernel::new(0), Kernel::new(1), Kernel::new(2)])
}

/// Barrett product by the twiddle in lanes `W` (value) and `T` (twisted) of `t`; `k` holds `p` in
/// lane 0. `|result| <= product_bound(p, |a|)`.
#[inline(always)]
unsafe fn mulc<const W: i32, const T: i32>(a: int16x8_t, t: int16x8_t, k: int16x8_t) -> int16x8_t {
    let mut lo = vmulq_laneq_s16::<W>(a, t);
    let hi = vqrdmulhq_laneq_s16::<T>(a, t);
    // `lo - hi p` as one `mls`: written as the intrinsic, LLVM turns the two products into
    // `mul`, `neg`, `mla`, one instruction more.
    core::arch::asm!(
        "mls {lo:v}.8h, {hi:v}.8h, {k:v}.h[0]",
        lo = inout(vreg) lo,
        hi = in(vreg) hi,
        k = in(vreg_low16) k,
        options(pure, nomem, nostack, preserves_flags)
    );
    lo
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

/// `a - round(a / 2^13) p`: one multiplication, `|result| <= round13_bound(p, |a|)`, for the
/// prime close to `2^13` (the reduction of [`FIRST13`]).
#[inline(always)]
unsafe fn reduce_round13(a: int16x8_t, k: int16x8_t) -> int16x8_t {
    vmlsq_laneq_s16::<0>(a, vrshrq_n_s16::<13>(a), k)
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
    } else if mode == FIRST13 {
        for &i in first {
            v[i] = vmlsq_laneq_s16::<0>(v[i], vrshrq_n_s16::<13>(v[i]), k);
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
    radix3_twiddled::<RD>(a, b1, c1, k)
}

/// The radix-3 butterfly on inputs already multiplied by their twiddles, `b1 = tau b` and
/// `c1 = tau^2 c`: `y0 = a + (b1 + c1)`, `y1 = (a - c1) + e`, `y2 = (a - b1) - e`,
/// `e = eps (b1 - c1)`, with `b1 - c1` reduced per `RD` (none, Barrett, cheap) first.
#[inline(always)]
unsafe fn radix3_twiddled<const RD: u8>(
    a: int16x8_t,
    b1: int16x8_t,
    c1: int16x8_t,
    k: int16x8_t,
) -> (int16x8_t, int16x8_t, int16x8_t) {
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

/// Coefficient groups of the nibble tables: group `j < 162` is the coefficients `j + 162 g`.
const GROUPS: usize = N / 4;

/// `out[j]`, lane `c`: the byte pair `(2 v, 2 v + 1)` of column `c`'s nibble `v` at group `j`
/// (bit `g` set where coefficient `j + 162 g` is; coefficients from 512 on are zero), the
/// `vqtbl2q_u8` index of entry `v` of a 16-entry `i16` table.
#[inline(always)]
unsafe fn nibbles(m: &[int16x8_t; BITS], out: &mut [uint8x16_t; GROUPS]) {
    let (lo, c0) = (vdupq_n_u16(0x0100), vdupq_n_u16(0x0302));
    let (c1, c2, c3) = (vdupq_n_u16(0x0404), vdupq_n_u16(0x0808), vdupq_n_u16(0x1010));
    let mask = |i: usize| vreinterpretq_u16_s16(*m.get_unchecked(i));
    for j in 0..GROUPS {
        let mut r = vbslq_u16(mask(j), c0, lo);
        r = vorrq_u16(r, vandq_u16(mask(j + 162), c1));
        r = vorrq_u16(r, vandq_u16(mask(j + 324), c2));
        if j < BITS - 486 {
            r = vorrq_u16(r, vandq_u16(mask(j + 486), c3));
        }
        *out.get_unchecked_mut(j) = vreinterpretq_u8_u16(r);
    }
}

/// Exchange alternating groups of `D` bits between two bit planes. After the two
/// rounds below, each byte contains the nibbles of positions `r` and `r + 4`.
#[inline(always)]
unsafe fn index_swap<const D: i32>(a: uint8x16_t, b: uint8x16_t, mask: u8) -> (uint8x16_t, uint8x16_t) {
    let mask = vdupq_n_u8(mask);
    (vbslq_u8(mask, a, vshlq_n_u8::<D>(b)), vbslq_u8(mask, vshrq_n_u8::<D>(a), b))
}

/// Sixteen bit positions of word `K`, across all eight columns. Each byte-row
/// vector packs two consecutive bytes, eight columns of each. The sentinel row
/// supplies the zero bits above the end of a word.
#[inline(always)]
unsafe fn index_block<const K: usize, const G: usize>(bytes: &[[uint8x16_t; 9]; 2], idx: &mut [uint8x16_t; GROUPS]) {
    let a = bytes[K / 2][G];
    let other = &bytes[1 - K / 2];
    // j + 162: the other word shifted by 40 bits (K < 2) or 41 bits.
    let mut b = vextq_u8::<8>(other[2 + G], other[3 + G]);
    if K >= 2 {
        b = vorrq_u8(vshrq_n_u8::<1>(b), vshlq_n_u8::<7>(other[3 + G]));
    }
    // j + 324: the same word shifted by 81 bits.
    let c0 = bytes[K / 2][5 + G];
    let c1 = vextq_u8::<8>(c0, bytes[K / 2][6 + G]);
    let c = vorrq_u8(vshrq_n_u8::<1>(c0), vshlq_n_u8::<7>(c1));
    // j + 486 exists only in the first seven (K < 2) or six positions.
    let d = if G == 0 {
        let last = vextq_u8::<8>(other[7], vdupq_n_u8(0));
        if K < 2 { vshrq_n_u8::<1>(last) } else { vshrq_n_u8::<2>(last) }
    } else {
        vdupq_n_u8(0)
    };
    let (a, b) = index_swap::<1>(a, b, 0x55);
    let (c, d) = index_swap::<1>(c, d, 0x55);
    let (a, c) = index_swap::<2>(a, c, 0x33);
    let (b, d) = index_swap::<2>(b, d, 0x33);
    macro_rules! put {
        ($q:expr, $z:expr) => {
            if 4 * (16 * G + $q) + K < GROUPS {
                *idx.get_unchecked_mut(4 * (16 * G + $q) + K) = $z;
            }
        };
    }
    macro_rules! emit {
        ($v:expr, $r:literal) => {{
            let v = $v;
            let lo = vandq_u8(vshlq_n_u8::<1>(v), vdupq_n_u8(0x1e));
            let hi = vandq_u8(vshrq_n_u8::<3>(v), vdupq_n_u8(0x1e));
            // Set the second byte before duplication: one OR serves both halves.
            let lo_odd = vorrq_u8(lo, vdupq_n_u8(1));
            let hi_odd = vorrq_u8(hi, vdupq_n_u8(1));
            put!($r, vzip1q_u8(lo, lo_odd));
            put!($r + 4, vzip1q_u8(hi, hi_odd));
            put!($r + 8, vzip2q_u8(lo, lo_odd));
            put!($r + 12, vzip2q_u8(hi, hi_odd));
        }};
    }
    emit!(a, 0);
    emit!(b, 1);
    emit!(c, 2);
    emit!(d, 3);
}

/// The table indices of one element of eight columns. Coefficient `4 q + k`
/// is bit `q` of word `k`; each result lane holds `(2 p, 2 p + 1)` for the
/// four-bit pattern at `j + 162 t`, with coefficients above 511 zero.
#[inline(never)]
unsafe fn indices_pair<const P: usize>(words: &[*const u8; LANES], idx: &mut [uint8x16_t; GROUPS]) {
    let mut bytes = [[vdupq_n_u8(0); 9]; 2];
    for (i, row) in bytes.iter_mut().enumerate() {
        let k = P + 2 * i;
        let r: [uint8x16_t; 8] = core::array::from_fn(|c| vld1q_u8(words[c].add(16 * k)));
        let a: [uint8x16_t; 8] = core::array::from_fn(|i| {
            let (x, y) = (r[2 * (i % 4)], r[2 * (i % 4) + 1]);
            if i < 4 { vzip1q_u8(x, y) } else { vzip2q_u8(x, y) }
        });
        let b: [uint16x8_t; 8] = core::array::from_fn(|i| {
            let h = i / 4;
            let (x, y) = (
                vreinterpretq_u16_u8(a[4 * h + 2 * ((i % 4) / 2)]),
                vreinterpretq_u16_u8(a[4 * h + 2 * ((i % 4) / 2) + 1]),
            );
            if i % 2 == 0 { vzip1q_u16(x, y) } else { vzip2q_u16(x, y) }
        });
        for v in 0..8 {
            let h = v / 4;
            let z = (v % 4) / 2;
            let (x, y) = (vreinterpretq_u32_u16(b[4 * h + z]), vreinterpretq_u32_u16(b[4 * h + 2 + z]));
            row[v] = vreinterpretq_u8_u32(if v % 2 == 0 { vzip1q_u32(x, y) } else { vzip2q_u32(x, y) });
        }
    }
    macro_rules! word {
        ($k:literal) => {{
            index_block::<$k, 0>(&bytes, idx);
            index_block::<$k, 1>(&bytes, idx);
            index_block::<$k, 2>(&bytes, idx);
        }};
    }
    if P == 0 {
        word!(0);
        word!(2);
    } else {
        word!(1);
        word!(3);
    }
}

/// Process independent word pairs separately, keeping only sixteen transposed
/// byte rows live rather than all thirty-two rows of the four-word input.
#[inline(always)]
unsafe fn indices(words: &[*const u8; LANES], idx: &mut [uint8x16_t; GROUPS]) {
    indices_pair::<0>(words, idx);
    indices_pair::<1>(words, idx);
}

/// The 16-entry `i16` table `t` as the two registers `vqtbl2q_u8` reads.
#[inline(always)]
unsafe fn table(t: &[i16; 16]) -> uint8x16x2_t {
    vld1q_u8_x2(t.as_ptr() as *const u8)
}

/// Entry `v` of table `t` in every lane, `v` from the lane's index pair in `i`.
#[inline(always)]
unsafe fn look(t: uint8x16x2_t, i: uint8x16_t) -> int16x8_t {
    vreinterpretq_s16_u8(vqtbl2q_u8(t, i))
}

/// The second child's radix-3 butterfly from the shared lookups. If `b` and
/// `c` are the differences of the same tables whose sums feed the first child,
/// its multiplied inputs are `(1 + eps) b` and `eps c`. Using eps^2 = -1-eps:
/// y = [a + b + eps(b+c), a - b + c, a - c - eps(b+c)].
/// All inputs are bounded by 2h, so `b+c` is bounded by 4h; the three output
/// bounds are the same 4h+product_bound(p,4h), 6h as in radix3_twiddled.
#[inline(always)]
unsafe fn radix3_sixth<const RD: u8>(
    a: int16x8_t,
    b: int16x8_t,
    c: int16x8_t,
    k: int16x8_t,
) -> (int16x8_t, int16x8_t, int16x8_t) {
    let mut d = vaddq_s16(b, c);
    if RD == 1 {
        d = reduce(d, k);
    } else if RD == 2 {
        d = reduce_cheap(d, k);
    }
    let e = mulc::<2, 3>(d, k, k);
    (
        vaddq_s16(vaddq_s16(a, b), e),
        vaddq_s16(vsubq_s16(a, b), c),
        vsubq_s16(vsubq_s16(a, c), e),
    )
}

/// Levels 0 to 3 from six tables per level-1 node. The two children share every
/// lookup: sums feed the first radix-3 butterfly and differences the second,
/// whose sixth-root twiddle is absorbed by radix3_sixth. Every intermediate
/// stays within the existing lut_transform_bound, and only eps is multiplied.
#[inline(always)]
unsafe fn pass_lut<const S: usize>(idx: &[uint8x16_t; GROUPS], x: &mut [int16x8_t; N], kern: &Kernel) {
    let s: Schedule = schedule(S);
    const { assert!(S < 3 && super::lut_transform_bound(PRIMES[S], SCHEDULES[S]) <= i16::MAX as u32) };
    let k = vld1q_s16(kern.k.as_ptr());
    let ix = |i: usize| *idx.get_unchecked(i);
    for n in 0..4 {
        let l = &kern.lut[n];
        let t: [uint8x16x2_t; 6] = core::array::from_fn(|i| table(&l[i]));
        let base = 162 * n;
        for j in 0..27 {
            let (x0, y0) = (look(t[0], ix(j)), look(t[1], ix(j + 81)));
            let (x1, y1) = (look(t[2], ix(j + 27)), look(t[3], ix(j + 108)));
            let (x2, y2) = (look(t[4], ix(j + 54)), look(t[5], ix(j + 135)));
            let (a0, b0) = (vaddq_s16(x0, y0), vsubq_s16(x0, y0));
            let (a1, b1) = (vaddq_s16(x1, y1), vsubq_s16(x1, y1));
            let (a2, b2) = (vaddq_s16(x2, y2), vsubq_s16(x2, y2));
            let (u, v) = match (s.reduce_d, s.cheap) {
                (false, _) => (radix3_twiddled::<0>(a0, a1, a2, k), radix3_sixth::<0>(b0, b1, b2, k)),
                (true, false) => (radix3_twiddled::<1>(a0, a1, a2, k), radix3_sixth::<1>(b0, b1, b2, k)),
                (true, true) => (radix3_twiddled::<2>(a0, a1, a2, k), radix3_sixth::<2>(b0, b1, b2, k)),
            };
            let at = base + j;
            store(x, at, u.0);
            store(x, at + 27, u.1);
            store(x, at + 54, u.2);
            store(x, at + 81, v.0);
            store(x, at + 108, v.1);
            store(x, at + 135, v.2);
        }
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

/// Level-4 nodes that need no first-input reduction before level 5 after the 9721 LUT front.
/// Proven for all admissible nibble patterns by `lazy_lut_intervals` below.
#[inline(always)]
const fn skip_lut5(n: usize) -> bool {
    const NODES: [usize; 27] = [
        1, 11, 15, 16, 17, 19, 20, 22, 23, 25, 28, 30, 31, 32,
        34, 35, 36, 37, 44, 50, 56, 59, 61, 65, 67, 70, 71,
    ];
    const MASK: u128 = {
        let mut mask = 0;
        let mut i = 0;
        while i < NODES.len() { mask |= 1 << NODES[i]; i += 1; }
        mask
    };
    MASK >> n & 1 != 0
}

/// Levels 4 and 5: per level-3 node (27 slots) and `j < 3`, the nine slots `j + 3 i`. All three
/// `j` of a node in one block (arm-big): nine independent butterflies per level.
/// `S == 4` uses prime 9721's table-front intervals to omit 99 first-input reductions.
#[inline(always)]
unsafe fn pass_c<const S: usize>(x: &mut [int16x8_t; N], kern: &Kernel) {
    let s: Schedule = schedule(if S == 4 { 1 } else { S });
    let k = vld1q_s16(kern.k.as_ptr());
    #[inline(always)]
    unsafe fn lvl4<const SS: usize, const R4: bool, const SKIP: u8>(x: &[int16x8_t; N], s: Schedule, t0: int16x8_t, k: int16x8_t, at: usize) -> [int16x8_t; 9] {
        let mut v: [int16x8_t; 9] = core::array::from_fn(|i| load(x, at + 3 * i));
        if SS != 4 || R4 {
            reduce_inputs(&mut v, s.before[4], s.cheap, &[0, 1, 2], k);
        }
        for i in 0..3 {
            let (a, b, c) = r3!(s, 0, 1, 2, 3, v[i], v[i + 3], v[i + 6], t0, k);
            (v[i], v[i + 3], v[i + 6]) = (a, b, c);
        }
        if SS == 4 {
            for g in 0..3 {
                if SKIP >> g & 1 == 0 {
                    v[3 * g] = vmlsq_laneq_s16::<0>(v[3 * g], vrshrq_n_s16::<13>(v[3 * g]), k);
                }
            }
        } else {
            reduce_inputs(&mut v, s.before[5], s.cheap, &[0, 3, 6], k);
        }
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
    #[inline(always)]
    unsafe fn node<const SS: usize, const R4: bool, const SKIP: u8>(x: &mut [int16x8_t; N], s: Schedule, kern: &Kernel, k: int16x8_t, n: usize) {
        let t0 = vld1q_s16(kern.c[n].as_ptr());
        let t1 = vld1q_s16(kern.c[n].as_ptr().add(8));
        let base = 27 * n;
        let v0 = lvl4::<SS, R4, SKIP>(x, s, t0, k, base);
        let v1 = lvl4::<SS, R4, SKIP>(x, s, t0, k, base + 1);
        lvl5(x, s, v0, t0, t1, k, base);
        let v2 = lvl4::<SS, R4, SKIP>(x, s, t0, k, base + 2);
        lvl5(x, s, v1, t0, t1, k, base + 1);
        lvl5(x, s, v2, t0, t1, k, base + 2);
    }
    if S == 4 {
        // every node with its reductions as constants: none is a run-time branch in the pass
        macro_rules! nodes {
            ($($n:literal),*) => {
                const { assert!([$($n),*].len() == 24) };
                $(node::<S, { lut_node_kind($n).0 }, { lut_node_kind($n).1 }>(x, s, kern, k, $n);)*
            };
        }
        nodes!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23);
    } else {
        for n in 0..24 {
            node::<S, true, 0>(x, s, kern, k, n);
        }
    }
}

/// Node `n`'s reductions in `pass_c::<4>`: whether its level-4 inputs are reduced, and the `g`
/// whose level-5 first input is not ([`skip_lut5`]) as bits.
const fn lut_node_kind(n: usize) -> (bool, u8) {
    let mut skip = 0u8;
    let mut g = 0;
    while g < 3 {
        if skip_lut5(3 * n + g) {
            skip |= 1 << g;
        }
        g += 1;
    }
    (n != 1 && n != 15, skip)
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

/// [`transform`] through the nibble tables: [`nibbles`] and [`pass_lut`] in place of [`pass_a`]
/// and [`pass_b`], values within [`super::lut_transform_bound`].
#[inline(never)]
#[cfg_attr(not(test), allow(dead_code))]
pub unsafe fn transform_lut<const P: usize>(m: &[int16x8_t; BITS], idx: &mut [uint8x16_t; GROUPS], x: &mut [int16x8_t; N]) {
    let kern = &kernels()[P];
    nibbles(m, idx);
    pass_lut::<P>(idx, x, kern);
    if P == 1 { pass_c::<4>(x, kern); } else { pass_c::<P>(x, kern); }
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
    /// [`t`](Self::t) without `r^-1`, for [`inverse_folded`]: `[1, tau^-1, tau^-2, eps^-1]` for a
    /// radix-3 node (lane pair 0 unused), `[1, tau^-1, 0, 0]` for a radix-2 node.
    f: Vec<[i16; 8]>,
}

impl InverseKernel {
    fn new(index: usize) -> InverseKernel {
        use super::{ORDER, RADICES};
        let plan = &primes()[index];
        let (p, pp) = (plan.p, plan.p as u64);
        // `scaled`: with `r^-1` folded in ([`inverse`]) or not ([`inverse_folded`])
        let rows = |scaled: bool| {
            let mut t = Vec::new();
            for level in (0..RADICES.len()).rev() {
                let r = RADICES[level];
                let ir = if scaled { plan.inv_radix[r] as u64 } else { 1 };
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
            t
        };
        let mut u = [0i16; 8];
        u[..2].copy_from_slice(&twiddle(p, plan.inv_split as u64));
        u[2..4].copy_from_slice(&twiddle(p, plan.root(super::HALF)));
        InverseKernel { t: rows(true), u, f: rows(false) }
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

/// The product of the radices of levels 1 to 6, `2^2 3^4`: [`inverse_folded`] is this many times
/// [`inverse`].
pub const FOLD: u32 = {
    let mut f = 1;
    let mut level = 0;
    while level < super::RADICES.len() {
        f *= super::RADICES[level] as u32;
        level += 1;
    }
    f
};
const _: () = assert!(FOLD == 324);

/// Whether [`inverse_folded`] stays inside `i16` from inputs `|x| <= p`, and its trinomial step's
/// two Barrett products land inside [`center`]'s range `|a| <= p + (p - 1) / 2`. Without `1 / r`
/// the first output of a butterfly is the bare sum `s0` (radix 2: `y0 + y1`), reduced by
/// `a - round(a / 2^13) p` ([`reduce_round13`], one multiplication); the others stay Barrett
/// products. `b` bounds every slot after a level.
const fn inverse_folded_fits(p: u32) -> bool {
    use super::{product_bound, round13_bound, RADICES};
    const MAX: u32 = i16::MAX as u32;
    let mut b = p;
    let mut level = RADICES.len();
    while level > 0 {
        level -= 1;
        // radix 3: s0 = y0 + y1 + y2, s1/s2 = (y0 - y2/y1) +- eps^-1 (y1 - y2); radix 2: y0 +- y1
        let (s0, s) = if RADICES[level] == 3 { (3 * b, 2 * b + product_bound(p, 2 * b)) } else { (2 * b, 2 * b) };
        if s0 > MAX || s > MAX {
            return false;
        }
        let a0 = round13_bound(p, s0);
        let a = product_bound(p, s);
        b = if a0 > a { a0 } else { a };
    }
    // b = center((u - v) inv_split), a = center(u - w b) with |b| <= (p - 1) / 2 after center
    let c = p + (p - 1) / 2;
    2 * b <= MAX && product_bound(p, 2 * b) <= c && b + product_bound(p, (p - 1) / 2) <= c
}

// The SMS prime 9721 (`|1 - p / 2^13| < 0.19`): the tightest point is level 3's `s0`, `3 * 10,814 =
// 32,442`. The other two primes are too far from `2^13` and keep [`inverse`].
const _: () = assert!(inverse_folded_fits(PRIMES[1]) && !inverse_folded_fits(PRIMES[0]) && !inverse_folded_fits(PRIMES[2]));

/// [`inverse_radix3`] without `1 / r`: `a_0 = sum_k y_k` (reduced by [`reduce_round13`]),
/// `a_i = tau^-i sum_k eps^-(k i) y_k`.
#[inline(always)]
unsafe fn inverse_radix3_folded(x: &mut [int16x8_t; N], t: &[[i16; 8]], len: usize, k: int16x8_t) {
    debug_assert_eq!(t.len() * 3 * len, N);
    for (node, tw) in t.iter().enumerate() {
        let tv = vld1q_s16(tw.as_ptr());
        let base = 3 * len * node;
        for j in base..base + len {
            let (y0, y1, y2) = (load(x, j), load(x, j + len), load(x, j + 2 * len));
            let s0 = vaddq_s16(vaddq_s16(y0, y1), y2);
            let e = mulc::<6, 7>(vsubq_s16(y1, y2), tv, k);
            let s1 = vaddq_s16(vsubq_s16(y0, y2), e);
            let s2 = vsubq_s16(vsubq_s16(y0, y1), e);
            store(x, j, reduce_round13(s0, k));
            store(x, j + len, mulc::<2, 3>(s1, tv, k));
            store(x, j + 2 * len, mulc::<4, 5>(s2, tv, k));
        }
    }
}

/// [`inverse_radix2`] without `1 / 2`: `a_0 = y0 + y1` (reduced by [`reduce_round13`]),
/// `a_1 = tau^-1 (y0 - y1)`.
#[inline(always)]
unsafe fn inverse_radix2_folded(x: &mut [int16x8_t; N], t: &[[i16; 8]], len: usize, k: int16x8_t) {
    debug_assert_eq!(t.len() * 2 * len, N);
    for (node, tw) in t.iter().enumerate() {
        let tv = vld1q_s16(tw.as_ptr());
        let base = 2 * len * node;
        for j in base..base + len {
            let (y0, y1) = (load(x, j), load(x, j + len));
            store(x, j, reduce_round13(vaddq_s16(y0, y1), k));
            store(x, j + len, mulc::<2, 3>(vsubq_s16(y0, y1), tv, k));
        }
    }
}

/// [`FOLD`] `= 324` times [`inverse`], for inputs that already carry `324^-1`: the SMS key's
/// transformed rows do ([`sms::matrix`]), so the commitment's inverses need no `1 / r`. Values
/// congruent to the transform with `|x| <= p` in, centered residues out, `|x| <= (p - 1) / 2`
/// exactly: the same output as [`inverse`] of `324 x`. The trinomial step's two Barrett
/// reductions are dropped as well, since [`center`] takes both products directly. Only for a
/// prime close to `2^13` ([`inverse_folded_fits`]).
#[inline(never)]
pub unsafe fn inverse_folded<const P: usize>(x: &mut [int16x8_t; N]) {
    const { assert!(P < 3 && inverse_folded_fits(PRIMES[P])) };
    let k = vld1q_s16(kernels()[P].k.as_ptr());
    let inv = &inverse_kernels()[P];
    let t = &inv.f[..];
    inverse_radix3_folded(x, &t[..216], 1, k);
    inverse_radix3_folded(x, &t[216..288], 3, k);
    inverse_radix3_folded(x, &t[288..312], 9, k);
    inverse_radix3_folded(x, &t[312..320], 27, k);
    inverse_radix2_folded(x, &t[320..324], 81, k);
    inverse_radix2_folded(x, &t[324..326], 162, k);
    let uv = vld1q_s16(inv.u.as_ptr());
    let h = vdupq_n_s16(((PRIMES[P] - 1) / 2) as i16);
    for j in 0..N / 2 {
        let (u, v) = (load(x, j), load(x, j + N / 2));
        let b = center(mulc::<0, 1>(vsubq_s16(u, v), uv, k), h, k);
        let a = center(vsubq_s16(u, mulc::<2, 3>(b, uv, k)), h, k);
        store(x, j, a);
        store(x, j + N / 2, b);
    }
}

/// [`InverseKernel`]'s twiddles without the `1 / r` factors, for [`inverse_lazy`]: per radix-3
/// node, levels 6 down to 3 (216, 72, 24, 8 nodes), `[tau^-1, tau^-2, eps^-1]` as pairs; per
/// radix-2 node, levels 2 and 1 (4, 2 nodes), `[tau^-1]`; and the trinomial unsplit's
/// `[inv_split, zeta^324]`.
pub struct LazyInverse {
    r3: Vec<[i16; 8]>,
    r2: Vec<[i16; 8]>,
    u: [i16; 8],
}

impl LazyInverse {
    fn new(index: usize) -> LazyInverse {
        use super::{ORDER, RADICES};
        let plan = &primes()[index];
        let p = plan.p;
        let (mut r3, mut r2) = (Vec::new(), Vec::new());
        for level in (0..RADICES.len()).rev() {
            let r = RADICES[level];
            for &e in &plan.exps[level] {
                let tau = e as usize / r;
                let mut row = [0i16; 8];
                for i in 1..r {
                    row[2 * (i - 1)..2 * i].copy_from_slice(&twiddle(p, plan.root(ORDER - tau * i % ORDER)));
                }
                if r == 3 {
                    row[4..6].copy_from_slice(&twiddle(p, plan.root(ORDER - ORDER / 3)));
                    r3.push(row);
                } else {
                    r2.push(row);
                }
            }
        }
        assert!(r3.len() == 216 + 72 + 24 + 8 && r2.len() == 4 + 2);
        let mut u = [0i16; 8];
        u[..2].copy_from_slice(&twiddle(p, plan.inv_split as u64));
        u[2..4].copy_from_slice(&twiddle(p, plan.root(super::HALF)));
        LazyInverse { r3, r2, u }
    }
}

pub fn lazy_inverses() -> &'static [LazyInverse; 3] {
    static K: OnceLock<[LazyInverse; 3]> = OnceLock::new();
    K.get_or_init(|| [LazyInverse::new(0), LazyInverse::new(1), LazyInverse::new(2)])
}

/// The factor [`inverse_lazy`]'s output carries: `3^4 2^2`, the radices it does not divide by.
pub const LAZY_SCALE: u64 = 324;

/// `a - round(a / 2^13) p`: `|result| <= 0.18664 |a| + p / 2` for `p = 9721`.
#[inline(always)]
unsafe fn r13(a: int16x8_t, k: int16x8_t) -> int16x8_t {
    vmlsq_laneq_s16::<0>(a, vrshrq_n_s16::<13>(a), k)
}

/// The radix-3 step of [`inverse_lazy`] on inputs `|y| <= B`, `t` a node's twiddles:
/// `a = y0 + y1 + y2` (cheaply reduced), `b = tau^-1 (y0 + eps^-1 y1 + eps^-2 y2)`,
/// `c = tau^-2 (y0 + eps^-2 y1 + eps^-1 y2)`, `eps^-2 = -1 - eps^-1`: three times the inverse
/// butterfly, with three Barrett products instead of four.
#[inline(always)]
unsafe fn ib3(y0: int16x8_t, y1: int16x8_t, y2: int16x8_t, t: int16x8_t, k: int16x8_t) -> (int16x8_t, int16x8_t, int16x8_t) {
    let a = r13(vaddq_s16(vaddq_s16(y0, y1), y2), k);
    let e = mulc::<4, 5>(vsubq_s16(y1, y2), t, k);
    let b = mulc::<0, 1>(vaddq_s16(vsubq_s16(y0, y2), e), t, k);
    let c = mulc::<2, 3>(vsubq_s16(vsubq_s16(y0, y1), e), t, k);
    (a, b, c)
}

/// The radix-2 step: `a = y0 + y1` (cheaply reduced), `b = tau^-1 (y0 - y1)`: twice the inverse.
#[inline(always)]
unsafe fn ib2(y0: int16x8_t, y1: int16x8_t, t: int16x8_t, k: int16x8_t) -> (int16x8_t, int16x8_t) {
    (r13(vaddq_s16(y0, y1), k), mulc::<0, 1>(vsubq_s16(y0, y1), t, k))
}

/// The inverse transform times [`LAZY_SCALE`] for the prime 9721, on eight lanes: inputs
/// `|x| <= p`, outputs congruent to `324` times [`inverse`]'s with `|x| <= p` (not centered).
/// Four passes: levels 6 and 5 per level-4 node, level 4, level 3, then levels 2, 1 and the
/// trinomial unsplit per group of eight slots `j + 81 b`, each slot loaded and stored once per
/// pass. Bounds (inputs `<= 9721`, `r13`'s `0.18664 |a| + 4860.5`, [`super::product_bound`]):
/// the radix-3 sums reach 29163, 30912, 31890, 32439 and the outputs 10304, 10630, 10813,
/// 10915; the radix-2 sums 21830 and 17870, outputs 8935 and 8196; the unsplit outputs 7499
/// and 7292. The test `lazy_inverse_matches` checks the congruence and the output bound.
#[inline(never)]
pub unsafe fn inverse_lazy(x: &mut [int16x8_t; N]) {
    let at = x.as_mut_ptr();
    inverse_lazy_io(x, |i| *at.add(i), |i, v| *at.add(i) = v);
}

/// [`inverse_lazy`] with its first pass reading slot `i` from `input(i)` instead of `x`, and its
/// last pass handing slot `i` to `output(i, value)` instead of storing it: `x` is the scratch of
/// the middle passes. `input` must return values `|x| <= p`; `output` sees every slot once.
#[inline(always)]
pub unsafe fn inverse_lazy_io(x: &mut [int16x8_t; N], mut input: impl FnMut(usize) -> int16x8_t, mut output: impl FnMut(usize, int16x8_t)) {
    const { assert!(PRIMES[1] == 9721) };
    let k = vld1q_s16(kernels()[1].k.as_ptr());
    let inv = &lazy_inverses()[1];
    let (r3, r2) = (&inv.r3[..], &inv.r2[..]);
    // levels 6 and 5, per level-4 node m (slots 9 m..9 m + 9)
    for m in 0..72 {
        let base = 9 * m;
        let mut v: [int16x8_t; 9] = core::array::from_fn(|i| input(base + i));
        for g in 0..3 {
            let t = vld1q_s16(r3.get_unchecked(3 * m + g).as_ptr());
            (v[3 * g], v[3 * g + 1], v[3 * g + 2]) = ib3(v[3 * g], v[3 * g + 1], v[3 * g + 2], t, k);
        }
        let t = vld1q_s16(r3.get_unchecked(216 + m).as_ptr());
        for j in 0..3 {
            (v[j], v[j + 3], v[j + 6]) = ib3(v[j], v[j + 3], v[j + 6], t, k);
        }
        for (i, &y) in v.iter().enumerate() {
            store(x, base + i, y);
        }
    }
    // level 4, per level-3 node n (27 slots): (j, j + 9, j + 18)
    for n in 0..24 {
        let t = vld1q_s16(r3.get_unchecked(288 + n).as_ptr());
        let base = 27 * n;
        for j0 in (0..9).step_by(3) {
            let v: [[int16x8_t; 3]; 3] = core::array::from_fn(|q| core::array::from_fn(|i| load(x, base + j0 + q + 9 * i)));
            let y: [(int16x8_t, int16x8_t, int16x8_t); 3] = core::array::from_fn(|q| ib3(v[q][0], v[q][1], v[q][2], t, k));
            for (q, &(a, b, c)) in y.iter().enumerate() {
                store(x, base + j0 + q, a);
                store(x, base + j0 + q + 9, b);
                store(x, base + j0 + q + 18, c);
            }
        }
    }
    // level 3, per level-2 node n (81 slots): (j, j + 27, j + 54)
    for n in 0..8 {
        let t = vld1q_s16(r3.get_unchecked(312 + n).as_ptr());
        let base = 81 * n;
        for j0 in (0..27).step_by(3) {
            let v: [[int16x8_t; 3]; 3] = core::array::from_fn(|q| core::array::from_fn(|i| load(x, base + j0 + q + 27 * i)));
            let y: [(int16x8_t, int16x8_t, int16x8_t); 3] = core::array::from_fn(|q| ib3(v[q][0], v[q][1], v[q][2], t, k));
            for (q, &(a, b, c)) in y.iter().enumerate() {
                store(x, base + j0 + q, a);
                store(x, base + j0 + q + 27, b);
                store(x, base + j0 + q + 54, c);
            }
        }
    }
    // levels 2 and 1 and the unsplit, per j < 81: slots j + 81 i, i < 8
    let t2: [int16x8_t; 4] = core::array::from_fn(|n| vld1q_s16(r2.get_unchecked(n).as_ptr()));
    let t1: [int16x8_t; 2] = core::array::from_fn(|n| vld1q_s16(r2.get_unchecked(4 + n).as_ptr()));
    let uv = vld1q_s16(inv.u.as_ptr());
    for j in 0..81 {
        let mut v: [int16x8_t; 8] = core::array::from_fn(|i| load(x, j + 81 * i));
        // level 2: node n pairs slots 162 n + j and 162 n + j + 81
        for n in 0..4 {
            (v[2 * n], v[2 * n + 1]) = ib2(v[2 * n], v[2 * n + 1], t2[n], k);
        }
        // level 1: node n pairs 324 n + i and 324 n + i + 162, i = j, j + 81
        for n in 0..2 {
            for h in 0..2 {
                (v[4 * n + h], v[4 * n + h + 2]) = ib2(v[4 * n + h], v[4 * n + h + 2], t1[n], k);
            }
        }
        // the unsplit: b = (u - v) inv_split, a = u - zeta^324 b, pairs i and i + 324
        for i in 0..4 {
            let (u, w) = (v[i], v[i + 4]);
            let b = mulc::<0, 1>(vsubq_s16(u, w), uv, k);
            let a = r13(vsubq_s16(u, mulc::<2, 3>(b, uv, k)), k);
            (v[i], v[i + 4]) = (a, b);
        }
        for (i, &y) in v.iter().enumerate() {
            output(j + 81 * i, y);
        }
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

    #[no_mangle]
    #[inline(never)]
    unsafe fn mca_pass_lut(idx: &[uint8x16_t; GROUPS], x: &mut [int16x8_t; N], kern: &Kernel) {
        pass_lut::<1>(idx, x, kern);
    }

    #[no_mangle]
    #[inline(never)]
    unsafe fn mca_indices(words: &[*const u8; LANES], idx: &mut [uint8x16_t; GROUPS]) {
        indices(words, idx);
    }

    /// Every legal table pattern is covered, then conservative signed intervals are
    /// propagated through every integer Barrett rounding and every lazy intermediate.
    #[test]
    fn lazy_lut_intervals() {
        type Range = (i32, i32);
        fn check(a: Range) -> Range {
            assert!(a.0 >= -32768 && a.1 <= 32767, "interval {a:?}");
            a
        }
        fn add(a: Range, b: Range) -> Range { check((a.0 + b.0, a.1 + b.1)) }
        fn sub(a: Range, b: Range) -> Range { check((a.0 - b.1, a.1 - b.0)) }
        fn map(a: Range, f: impl Fn(i32) -> i32) -> Range {
            check(a);
            check((a.0..=a.1).map(f).fold((i32::MAX, i32::MIN), |r, v| (r.0.min(v), r.1.max(v))))
        }
        fn mul(a: Range, t: &[i16]) -> Range {
            map(a, |v| v * t[0] as i32 - ((v * t[1] as i32 + 16384) >> 15) * 9721)
        }
        fn red(a: Range) -> Range { map(a, |v| v - ((v + 4096) >> 13) * 9721) }
        fn bare(v: [Range; 3], k: &Kernel) -> [Range; 3] {
            let [a, b, c] = v;
            let e = mul(sub(b, c), &k.k[2..]);
            [add(a, add(b, c)), add(sub(a, c), e), sub(sub(a, b), e)]
        }
        fn sixth(v: [Range; 3], k: &Kernel) -> [Range; 3] {
            let [a, b, c] = v;
            let e = mul(add(b, c), &k.k[2..]);
            [add(add(a, b), e), add(sub(a, b), c), sub(sub(a, c), e)]
        }
        fn r3(v: [Range; 3], t: &[i16], k: &Kernel) -> [Range; 3] {
            bare([v[0], mul(v[1], t), mul(v[2], &t[2..])], k)
        }
        let k = &kernels()[1];
        let mut x = [(0, 0); N];
        for n in 0..4 {
            for j in 0..27 {
                let look = |t: usize, pos: usize| {
                    (0..16).filter(|&pat| (0..4).all(|b| pat >> b & 1 == 0 || pos + 162 * b < BITS))
                        .map(|pat| k.lut[n][t][pat] as i32)
                        .fold((i32::MAX, i32::MIN), |r, v| (r.0.min(v), r.1.max(v)))
                };
                for h in 0..2 {
                    let v = core::array::from_fn(|r| {
                        let t = 2 * r;
                        let a = look(t, j + 27 * r);
                        let b = look(t + 1, j + 27 * r + 81);
                        if h == 0 { add(a, b) } else { sub(a, b) }
                    });
                    let ys = if h == 0 { bare(v, k) } else { sixth(v, k) };
                    for (i, a) in ys.into_iter().enumerate() {
                        x[162 * n + 81 * h + j + 27 * i] = a;
                    }
                }
            }
        }
        for n in 0..24 {
            for j in 0..9 {
                let pos: [usize; 3] = core::array::from_fn(|i| 27 * n + j + 9 * i);
                let mut v = pos.map(|i| x[i]);
                if n != 1 && n != 15 { v[0] = red(v[0]); }
                for (i, a) in pos.into_iter().zip(r3(v, &k.c[n], k)) { x[i] = a; }
            }
        }
        for n in 0..72 {
            for j in 0..3 {
                let pos: [usize; 3] = core::array::from_fn(|i| 9 * n + j + 3 * i);
                let mut v = pos.map(|i| x[i]);
                if !skip_lut5(n) { v[0] = red(v[0]); }
                let t = if n % 3 == 0 { &k.c[n / 3][4..] } else { &k.c[n / 3][8 + 4 * (n % 3 - 1)..] };
                for (i, a) in pos.into_iter().zip(r3(v, t, k)) { x[i] = a; }
            }
        }
        let bound = super::super::lut_transform_bound(9721, SCHEDULES[1]) as i32;
        for n in 0..216 {
            let mut v = core::array::from_fn(|i| x[3 * n + i]);
            v[0] = red(v[0]);
            let t = &k.d[n / 2][4 * (n % 2)..];
            for a in r3(v, t, k) {
                assert!(a.0 >= -bound && a.1 <= bound, "final {a:?} bound {bound}");
            }
        }
    }

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
    fn lazy_inverse_matches() {
        let p = PRIMES[1] as i64;
        let mut state = 99u64;
        for case in 0..40 {
            let lanes: [[i16; 8]; N] = core::array::from_fn(|j| core::array::from_fn(|l| {
                match case {
                    0 => p as i16,
                    1 => -(p as i16),
                    2 => if (j + l) % 2 == 0 { p as i16 } else { -(p as i16) },
                    _ => ((super::super::tests::xorshift(&mut state) % (2 * p as u64 + 1)) as i64 - p) as i16,
                }
            }));
            let want = run_inverse(1, &lanes);
            let mut x: [int16x8_t; N] = unsafe { core::array::from_fn(|j| vld1q_s16(lanes[j].as_ptr())) };
            unsafe { inverse_lazy(&mut x) };
            for j in 0..N {
                let got = lanes_of(x[j]);
                for l in 0..8 {
                    assert!((got[l] as i64).abs() <= p, "bound case {case} slot {j}");
                    assert_eq!((got[l] as i64 - LAZY_SCALE as i64 * want[j][l] as i64).rem_euclid(p), 0, "case {case} slot {j} lane {l}");
                }
            }
        }
    }

    fn lanes_of(v: int16x8_t) -> [i16; 8] {
        unsafe { core::mem::transmute(v) }
    }

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

    /// The one-multiply reduction of [`FIRST13`] and [`inverse_folded`], over every `i16`, for the
    /// prime close to `2^13`: congruent, and within [`super::super::round13_bound`] of `|a|`.
    #[test]
    fn round13_reduction_is_exact_and_bounded() {
        let p = PRIMES[1] as i32;
        let k = unsafe { vld1q_s16(kernels()[1].k.as_ptr()) };
        for a in every_i16() {
            let r = lanes(unsafe { reduce_round13(a, k) });
            for (x, y) in lanes(a).into_iter().zip(r) {
                assert_eq!((x as i32 - y as i32).rem_euclid(p), 0);
                let bound = super::super::round13_bound(p as u32, x.unsigned_abs() as u32);
                assert!(y.unsigned_abs() as u32 <= bound, "{x} -> {y}");
            }
        }
    }

    /// The folded inverse is [`FOLD`] `= 324` times the scalar inverse, centered, for the SMS
    /// prime, on random inputs in `[-p, p]`, on the extremes `+-p` and on random inputs in the
    /// range the commitment feeds it (`|x| <= 4861`); and it is exactly [`inverse`] of `324 x`.
    #[test]
    fn inverse_folded_matches_scalar() {
        let mut state = 13u64;
        let plan = &primes()[1];
        let p = plan.p as i64;
        for case in 0..16 {
            let x: [[i16; 8]; N] = core::array::from_fn(|_| {
                core::array::from_fn(|l| match (case, l % 2) {
                    (0, _) => p as i16,
                    (1, _) => -p as i16,
                    (2, 0) | (3, 1) => p as i16,
                    (2, _) | (3, _) => -p as i16,
                    (4, _) | (5, _) => if xorshift(&mut state) & 1 == 0 { p as i16 } else { -p as i16 },
                    (6, _) | (7, _) => ((xorshift(&mut state) % 9723) as i64 - 4861) as i16,
                    _ => ((xorshift(&mut state) % (2 * p + 1) as u64) as i64 - p) as i16,
                })
            });
            let y = unsafe {
                let mut v: [int16x8_t; N] = core::array::from_fn(|j| vld1q_s16(x[j].as_ptr()));
                inverse_folded::<1>(&mut v);
                core::array::from_fn::<[i16; 8], N, _>(|j| lanes(v[j]))
            };
            let scaled: [[i16; 8]; N] = core::array::from_fn(|j| {
                core::array::from_fn(|c| super::super::centered(FOLD as i64 * x[j][c] as i64, plan.p))
            });
            assert_eq!(y, run_inverse(1, &scaled), "case {case}: inverse of 324 x");
            for c in 0..8 {
                let mut s: [u32; N] = core::array::from_fn(|j| (x[j][c] as i64).rem_euclid(p) as u32);
                plan.inverse(&mut s);
                for j in 0..N {
                    let want = super::super::centered(FOLD as i64 * s[j] as i64, plan.p);
                    assert_eq!(y[j][c], want, "case {case} lane {c} coefficient {j}");
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
        let start = std::time::Instant::now();
        for _ in 0..iters {
            unsafe { inverse_folded::<1>(core::hint::black_box(&mut v)) };
        }
        let ns = start.elapsed().as_nanos() as f64 / iters as f64;
        println!("inverse_folded::<1>: {ns:.0} ns per call (8 lanes)");
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

    /// Eight columns' words for one element in [`transform_matches_scalar`]'s twelve cases.
    fn case_words(case: usize, state: &mut u64) -> [[u128; 4]; 8] {
        core::array::from_fn(|c| {
            core::array::from_fn(|k| match (case, c) {
                (0, _) => u128::MAX,
                (1, _) => 0,
                (2, 0) => (k == 0) as u128,
                (2, 1) => ((k == 3) as u128) << 127,
                (2, 2) => if k == 1 { 1 << 64 } else { 0 },
                _ => (xorshift(state) as u128) << 64 | xorshift(state) as u128,
            })
        })
    }

    /// Levels 0 to `last` of [`super::super::Prime::forward`] in scalar arithmetic (level 0 the
    /// trinomial split, levels 1.. the binomial splits), residues in `[0, p)`.
    fn forward_levels(plan: &super::super::Prime, x: &mut [u32; N], last: usize) {
        use super::super::{HALF, ORDER, RADICES};
        let p = plan.p as u64;
        let (w, w5) = (plan.root(HALF), plan.root(5 * HALF));
        for j in 0..HALF {
            let (a, b) = (x[j] as u64, x[j + HALF] as u64);
            x[j] = ((a + w * b) % p) as u32;
            x[j + HALF] = ((a + w5 * b) % p) as u32;
        }
        let mut len = HALF;
        for (level, &r) in RADICES.iter().enumerate().take(last) {
            let sub = len / r;
            for (node, &e) in plan.exps[level].iter().enumerate() {
                let tau = e as usize / r;
                for j in 0..sub {
                    let at = node * len + j;
                    let a: Vec<u64> = (0..r).map(|i| x[at + i * sub] as u64 * plan.root(tau * i) % p).collect();
                    for k in 0..r {
                        let acc: u64 = (0..r).map(|i| a[i] * plan.root(k * i * (ORDER / r)) % p).sum();
                        x[at + k * sub] = (acc % p) as u32;
                    }
                }
            }
            len = sub;
        }
    }

    /// Exact byte-pair indices, including every one-hot coefficient in every
    /// column and the mixed patterns used by the transform regression tests.
    #[test]
    fn indices_match_mask_nibbles() {
        let mut state = 7u64;
        for case in 0..BITS + 12 {
            let words = if case < BITS {
                core::array::from_fn(|c| {
                    let bit = (case + 37 * c) % BITS;
                    core::array::from_fn(|k| if k == bit % 4 { 1u128 << (bit / 4) } else { 0 })
                })
            } else {
                case_words(case - BITS, &mut state)
            };
            let ptrs: [*const u8; LANES] = core::array::from_fn(|c| words[c].as_ptr() as *const u8);
            let mut m = [unsafe { vdupq_n_s16(0) }; BITS];
            let mut want = [unsafe { vdupq_n_u8(0) }; GROUPS];
            let mut got = want;
            unsafe {
                masks(&ptrs, &mut m);
                nibbles(&m, &mut want);
                mca_indices(&ptrs, &mut got);
            }
            for j in 0..GROUPS {
                let got: [u8; 16] = unsafe { core::mem::transmute(got[j]) };
                let want: [u8; 16] = unsafe { core::mem::transmute(want[j]) };
                assert_eq!(got, want, "case {case} group {j}");
            }
        }
    }

    /// The table-fed levels 0 to 3 ([`nibbles`], [`pass_lut`]) agree lane by lane with the scalar
    /// levels 0 to 3 of `-s` and with [`pass_a`] and [`pass_b`] modulo `p`, within
    /// [`super::super::lut_level3_bound`]; checked for every prime whose bound admits the tables.
    #[test]
    fn lut_levels_match_scalar() {
        use super::super::{lut_level3_bound, lut_transform_bound};
        let mut state = 7u64;
        fn run<const P: usize>(m: &[int16x8_t; BITS], idx: &mut [uint8x16_t; GROUPS], x: &mut [int16x8_t; N], old: &mut [int16x8_t; N]) {
            let kern = &kernels()[P];
            unsafe {
                nibbles(m, idx);
                if P == 1 { mca_pass_lut(idx, x, kern); } else { pass_lut::<P>(idx, x, kern); }
                pass_a(m, old, kern);
                pass_b::<P>(old, kern);
            }
        }
        for case in 0..12 {
            let words = case_words(case, &mut state);
            let ptrs: [*const u8; 8] = core::array::from_fn(|c| words[c].as_ptr() as *const u8);
            let mut m = [unsafe { vdupq_n_s16(0) }; BITS];
            unsafe { masks(&ptrs, &mut m) };
            // 3889 and 9721: for 17497 the level-3 difference of two table sums, 4 h, exceeds i16
            for (i, plan) in primes().iter().enumerate().take(2) {
                let mut idx = [unsafe { vdupq_n_u8(0) }; GROUPS];
                let mut x = [unsafe { vdupq_n_s16(0) }; N];
                let mut old = [unsafe { vdupq_n_s16(0) }; N];
                match i {
                    0 => run::<0>(&m, &mut idx, &mut x, &mut old),
                    _ => run::<1>(&m, &mut idx, &mut x, &mut old),
                }
                let bound = lut_level3_bound(plan.p, SCHEDULES[i]);
                assert!(lut_transform_bound(plan.p, SCHEDULES[i]) <= i16::MAX as u32);
                let p = plan.p as i64;
                for c in 0..8 {
                    let b: Vec<B128> = words[c].iter().map(|&w| B128(w)).collect();
                    let mut s = [0u32; N];
                    super::super::element_bits(&b, &mut s);
                    forward_levels(plan, &mut s, 3);
                    for (slot, &want) in s.iter().enumerate() {
                        let got = lanes(x[slot])[c] as i64;
                        assert!(got.unsigned_abs() <= bound as u64, "case {case} prime {p} column {c} slot {slot}: {got}");
                        assert_eq!((got + want as i64).rem_euclid(p), 0, "case {case} prime {p} column {c} slot {slot}");
                        assert_eq!((got - lanes(old[slot])[c] as i64).rem_euclid(p), 0, "case {case} prime {p} column {c} slot {slot}");
                    }
                }
            }
        }
    }

    /// The table-fed transform ([`transform_lut`]) of eight binary elements agrees lane by lane
    /// with the scalar transform of `-s`, within [`super::super::lut_transform_bound`].
    #[test]
    fn transform_lut_matches_scalar() {
        let mut state = 7u64;
        for case in 0..12 {
            let words = case_words(case, &mut state);
            let ptrs: [*const u8; 8] = core::array::from_fn(|c| words[c].as_ptr() as *const u8);
            let mut m = [unsafe { vdupq_n_s16(0) }; BITS];
            unsafe { masks(&ptrs, &mut m) };
            for (i, plan) in primes().iter().enumerate().take(2) {
                let mut idx = [unsafe { vdupq_n_u8(0) }; GROUPS];
                let mut x = [unsafe { vdupq_n_s16(0) }; N];
                unsafe {
                    match i {
                        0 => transform_lut::<0>(&m, &mut idx, &mut x),
                        _ => transform_lut::<1>(&m, &mut idx, &mut x),
                    }
                }
                let bound = super::super::lut_transform_bound(plan.p, SCHEDULES[i]);
                for c in 0..8 {
                    let b: Vec<B128> = words[c].iter().map(|&w| B128(w)).collect();
                    let mut s = [0u32; N];
                    super::super::element_bits(&b, &mut s);
                    plan.forward(&mut s);
                    for (slot, &want) in s.iter().enumerate() {
                        let got = lanes(x[slot])[c] as i64;
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
                let mut idx = Box::new([unsafe { vdupq_n_u8(0) }; GROUPS]);
                time!("nibbles", nibbles(&m, &mut idx));
                time!("pass_lut", pass_lut::<1>(&idx, &mut x, kp::<1>()));
            }
        }
        // one element-prime transform of the SMS commit, old (pass_a, pass_b) against the
        // nibble tables, masks included, interleaved
        let mut idx = Box::new([unsafe { vdupq_n_u8(0) }; GROUPS]);
        let (mut old, mut new) = (Vec::new(), Vec::new());
        for _ in 0..15 {
            let start = std::time::Instant::now();
            for _ in 0..iters / 5 {
                unsafe {
                    masks(&ptrs, &mut m);
                    transform::<1>(&m, &mut x);
                }
                core::hint::black_box(&mut *x);
            }
            old.push(start.elapsed().as_nanos() as f64 / (iters / 5) as f64);
            let start = std::time::Instant::now();
            for _ in 0..iters / 5 {
                unsafe {
                    masks(&ptrs, &mut m);
                    transform_lut::<1>(&m, &mut idx, &mut x);
                }
                core::hint::black_box(&mut *x);
            }
            new.push(start.elapsed().as_nanos() as f64 / (iters / 5) as f64);
        }
        old.sort_by(f64::total_cmp);
        new.sort_by(f64::total_cmp);
        println!(
            "masks+transform<1>: old median {:.0} min {:.0} ns, lut median {:.0} min {:.0} ns ({:+.1}% median, {:+.1}% min)",
            old[7], old[0], new[7], new[0], 100.0 * (new[7] / old[7] - 1.0), 100.0 * (new[0] / old[0] - 1.0)
        );
    }
}
