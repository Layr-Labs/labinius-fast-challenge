//! The fold `v = sum_j c_j W_j` straight from the binary words: the x86 track's scheduled
//! bit-sliced kernel (`fold::bitsliced`), on NEON.
//!
//! # The algebra
//!
//! With `Y = X^4`, component `k` of ring element `m` of column `j` is the binary polynomial
//! `sum_q bit_q(word 4m + k) Y^q` (`q < 128`), and `c_j` sits in `R_648` as `c_j(-Y)`. So per
//! lane (a word of the block) the fold is a product in `Z[Y]/(Y^162 - Y^81 + 1)`. It is computed
//! as a convolution in `Z[Y]/(Y^243 + 1)` (243 *slots*), then reduced modulo
//! `Y^162 - Y^81 + 1`; since `(Y^81 + 1)(Y^162 - Y^81 + 1) = Y^243 + 1` this is exact.
//!
//! # The computation
//!
//! A *task* is a quarter (128 lanes, one `uint64x2_t` per bit plane) of a 512-word block. For
//! every column group of [`COL_GROUP`] columns, each column's 128 words are transposed into 128
//! bit planes ([`slice`]), and every slot pair `(2q, 2q + 1)` adds, with a precomputed schedule
//! of `u16` plane indices shared by all tasks ([`plan_group`]), the planes of every term that
//! reaches it into a 15-bit bit-sliced counter ([`accumulate_pair`]): a 64-input carry-save tree
//! keeps the low six bits in registers, and its weight-64 carries are counted in three register
//! bits and rippled into the upper nine bits in memory every seventh chunk.
//!
//! A negative bit `-b` enters as its complement, `-b = (1 - b) - 1`, and the number of negative
//! entries is subtracted at the end (the bias). The complement is never formed: negative
//! entries are scheduled apart from positive ones, and their full adders keep their carries
//! complemented (for `x = a ^ b`, the carry of `s + !a + !b` is `!bsl(x, s ^ x, a)`, as cheap as
//! the direct `bsl(x, s, a)`), with one NOT per tree at its root. Every column's run of 128
//! planes is flanked by zero guard planes, so a slot pair's odd slot reads a zero wherever the
//! term does not reach, with no edge cases in the schedule.
//!
//! The output stage transposes the counters back to `u16` counts, removes the bias, and folds
//! slot `s` with slot `162 + s` (`s < 81`, `Y^162 = Y^81 - 1` twice) or `s + 81` into `R_162`
//! coefficient `s` of every component, checking for `i16` overflow.

use crate::fold_bits::Challenge;

/// Degree of the ring.
pub const N: usize = 648;
/// Lanes (words) per block: 128 ring elements.
pub const LANES: usize = 512;
/// Lanes per task: one quarter of a block, a 128-bit NEON register per plane.
const QL: usize = 128;
/// Quarters per block.
const QUARTERS: usize = LANES / QL;
/// Slots of the convolution modulo `Y^243 + 1`.
const SLOTS: usize = 243;
/// Counter bits: counts are at most 32767.
const BITS: usize = 15;
/// Columns per group: 32 columns of 129 planes of 16 bytes, 66 KB per worker.
pub const COL_GROUP: usize = 32;
/// Planes per column: 128 direct planes and a zero guard (also the next column's leading
/// guard).
const STRIDE: usize = 129;
/// Plane index of column 0's first direct plane; planes 0 and 1 (the padding pair) are zero.
const GUARD: usize = 2;
/// Slot pairs `(2q, 2q + 1)`; the odd slot of the last pair (243) is a dummy.
const PAIRS: usize = SLOTS.div_ceil(2);
/// Entries a group's schedule may add to a slot pair beyond one per term: the padding of its
/// positive and of its negative entries to even lengths.
const PAD_PER_GROUP: usize = 2;
/// Output slots: coefficients of `R_162` per component.
const OUT_SLOTS: usize = 162;
/// Columns ahead whose words the transpose prefetches.
const PF_COLUMNS: usize = 2;

const _: () = assert!(GUARD + COL_GROUP * STRIDE <= u16::MAX as usize);

/// Whether the 15-bit counters cannot overflow: a slot's counter adds at most one entry per
/// challenge term and the padding of every group.
pub fn fits(challenges: &[Challenge]) -> bool {
    let weight: usize = challenges.iter().map(|c| c.weight).sum();
    challenges.iter().all(|c| c.weight <= c.positions.len() && c.positions[..c.weight].iter().all(|&p| p < 162))
        && weight + PAD_PER_GROUP * challenges.len().div_ceil(COL_GROUP) <= i16::MAX as usize
}

/// Whether a shape is whole blocks.
pub fn shape_ok(column_words: usize) -> bool {
    column_words > 0 && column_words % LANES == 0
}

/// A column group's schedule.
struct Group {
    first: usize,
    columns: usize,
    /// `terms[offsets[q] .. mid[q]]` and `terms[mid[q] .. offsets[q + 1]]` are the plane
    /// indices of the even slot of pair `q`'s positive and negative entries (the odd slot reads
    /// the next plane), each padded to an even length with index 0.
    offsets: [usize; PAIRS + 1],
    mid: [usize; PAIRS],
    terms: Vec<u16>,
}

#[inline(always)]
fn negative(c: &Challenge, t: usize) -> bool {
    ((c.signs >> t) & 1 != 0) ^ (c.positions[t] % 2 != 0)
}

/// The schedule of column group `g` (columns `cs`), and per slot pair the number of its
/// negative entries (padding included): each adds one to the bias of both slots.
fn plan_group(g: usize, cs: &[Challenge]) -> (Group, [i32; PAIRS]) {
    // A term's run `[m, end)` of slots meets pair `q` for `q` in `[m / 2, (end - 1) / 2]`; the
    // wrapped run `[0, wrap)` meets pairs `[0, (wrap - 1) / 2]`, with the opposite sign.
    let mut count_diff = [[0i32; PAIRS + 1]; 2];
    for c in cs {
        for t in 0..c.weight {
            let m = c.positions[t] as usize;
            let neg = negative(c, t);
            let end = (m + 128).min(SLOTS);
            count_diff[neg as usize][m / 2] += 1;
            count_diff[neg as usize][(end + 1) / 2] -= 1;
            let wrap = (m + 128).saturating_sub(SLOTS);
            if wrap != 0 {
                count_diff[!neg as usize][0] += 1;
                count_diff[!neg as usize][(wrap + 1) / 2] -= 1;
            }
        }
    }
    let mut group = Group {
        first: g * COL_GROUP,
        columns: cs.len(),
        offsets: [0; PAIRS + 1],
        mid: [0; PAIRS],
        terms: Vec::new(),
    };
    let mut negatives = [0i32; PAIRS];
    let mut count = [0i32; 2];
    for q in 0..PAIRS {
        for (c, diff) in count.iter_mut().zip(&count_diff) {
            *c += diff[q];
        }
        let (pos, neg) = ((count[0] as usize + 1) & !1, (count[1] as usize + 1) & !1);
        group.mid[q] = group.offsets[q] + pos;
        group.offsets[q + 1] = group.mid[q] + neg;
        negatives[q] = neg as i32;
    }
    // Index 0 is the padding (planes 0 and 1 are zero).
    let mut terms = vec![0u16; group.offsets[PAIRS]];
    let mut at: [[usize; PAIRS]; 2] = [group.offsets[..PAIRS].try_into().unwrap(), group.mid];
    for (j, c) in cs.iter().enumerate() {
        let column = GUARD + j * STRIDE;
        for t in 0..c.weight {
            let m = c.positions[t] as usize;
            let neg = negative(c, t);
            let end = (m + 128).min(SLOTS);
            // The plane of slot `2q` for the run starting at slot `m` is `column + 2q - m`: the
            // guard before the run when `2q = m - 1`; the odd slot reads one plane further, the
            // guard after the run when `2q + 1 = end`.
            let at_s = &mut at[neg as usize];
            let mut val = (column + 2 * (m / 2)) - m;
            for q in m / 2..(end + 1) / 2 {
                terms[at_s[q]] = val as u16;
                at_s[q] += 1;
                val += 2;
            }
            let wrap = (m + 128).saturating_sub(SLOTS);
            if wrap != 0 {
                let at_s = &mut at[!neg as usize];
                let mut val = column + (end - m);
                for q in 0..(wrap + 1) / 2 {
                    terms[at_s[q]] = val as u16;
                    at_s[q] += 1;
                    val += 2;
                }
            }
        }
    }
    debug_assert!((0..PAIRS).all(|q| at[0][q] <= group.mid[q] && at[1][q] <= group.offsets[q + 1]));
    group.terms = terms;
    (group, negatives)
}

/// Every group's schedule and the per-slot bias.
fn plan(challenges: &[Challenge], workers: usize) -> (Vec<Group>, [i32; SLOTS]) {
    let jobs = challenges.len().div_ceil(COL_GROUP);
    let group = |g: usize| {
        let first = g * COL_GROUP;
        let last = (first + COL_GROUP).min(challenges.len());
        plan_group(g, &challenges[first..last])
    };
    let parts = if jobs < 4 { 1 } else { workers.clamp(1, jobs) };
    let built: Vec<Vec<(Group, [i32; PAIRS])>> =
        crate::par::map_blocks(jobs, parts, 1, |_, s, e| (s..e).map(group).collect());
    let mut groups = Vec::with_capacity(jobs);
    let mut negatives = [0i32; PAIRS];
    for (g, n) in built.into_iter().flatten() {
        groups.push(g);
        for (dst, x) in negatives.iter_mut().zip(n) {
            *dst += x;
        }
    }
    (groups, core::array::from_fn(|s| negatives[s / 2]))
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use super::*;
    use core::arch::aarch64::*;

    pub(super) type V = uint64x2_t;

    #[inline(always)]
    unsafe fn xor3(a: V, b: V, c: V) -> V {
        #[cfg(target_feature = "sha3")]
        {
            veor3q_u64(a, b, c)
        }
        #[cfg(not(target_feature = "sha3"))]
        {
            veorq_u64(veorq_u64(a, b), c)
        }
    }

    #[inline(always)]
    unsafe fn not(a: V) -> V {
        vreinterpretq_u64_u8(vmvnq_u8(vreinterpretq_u8_u64(a)))
    }

    /// `maj(a, b, c)`: where `a` and `b` differ, `c` decides.
    #[inline(always)]
    unsafe fn maj(a: V, b: V, c: V) -> V {
        vbslq_u64(veorq_u64(a, b), c, a)
    }

    /// Full adder of `s` and two equally weighted planes: `s` becomes the sum, the carry is
    /// returned.
    #[inline(always)]
    unsafe fn csa(s: &mut V, a: V, b: V) -> V {
        let x = veorq_u64(a, b);
        let c = vbslq_u64(x, *s, a);
        *s = veorq_u64(*s, x);
        c
    }

    /// Full adder of `s` and the complements of `a` and `b`: the sum is the same, and the
    /// carry is returned complemented, `!maj(s, !a, !b) = bsl(a ^ b, s', a)` with `s'` the new
    /// sum. Its inputs may themselves be complemented carries of the level below.
    #[inline(always)]
    unsafe fn csa_c(s: &mut V, a: V, b: V) -> V {
        let x = veorq_u64(a, b);
        *s = veorq_u64(*s, x);
        vbslq_u64(x, *s, a)
    }

    #[inline(always)]
    unsafe fn id(a: V) -> V {
        a
    }

    /// The 8 x 8 bit transpose in every byte lane of eight registers: afterwards bit `y` of
    /// byte `k` of register `x` is the old bit `x` of byte `k` of register `y`.
    #[inline(always)]
    pub(super) unsafe fn bit8x8(a: &mut [uint8x16_t; 8]) {
        for r in 0..4 {
            let (x, y) = (a[r], a[r + 4]);
            a[r] = vsliq_n_u8::<4>(x, y);
            a[r + 4] = vsriq_n_u8::<4>(y, x);
        }
        let m = vdupq_n_u8(0x33);
        for r in [0, 1, 4, 5] {
            let (x, y) = (a[r], a[r + 2]);
            a[r] = vbslq_u8(m, x, vshlq_n_u8::<2>(y));
            a[r + 2] = vbslq_u8(m, vshrq_n_u8::<2>(x), y);
        }
        let m = vdupq_n_u8(0x55);
        for r in [0, 2, 4, 6] {
            let (x, y) = (a[r], a[r + 1]);
            a[r] = vbslq_u8(m, x, vshlq_n_u8::<1>(y));
            a[r + 1] = vbslq_u8(m, vshrq_n_u8::<1>(x), y);
        }
    }

    /// 16 x 16 byte transpose: afterwards byte `b` of register `r` is the old byte `r` of
    /// register `b`.
    #[inline(always)]
    unsafe fn byte16x16(m: &mut [uint8x16_t; 16]) {
        for i in (0..16).step_by(2) {
            let (a, b) = (m[i], m[i + 1]);
            m[i] = vtrn1q_u8(a, b);
            m[i + 1] = vtrn2q_u8(a, b);
        }
        for i in [0, 1, 4, 5, 8, 9, 12, 13] {
            let (a, b) = (vreinterpretq_u16_u8(m[i]), vreinterpretq_u16_u8(m[i + 2]));
            m[i] = vreinterpretq_u8_u16(vtrn1q_u16(a, b));
            m[i + 2] = vreinterpretq_u8_u16(vtrn2q_u16(a, b));
        }
        for i in [0, 1, 2, 3, 8, 9, 10, 11] {
            let (a, b) = (vreinterpretq_u32_u8(m[i]), vreinterpretq_u32_u8(m[i + 4]));
            m[i] = vreinterpretq_u8_u32(vtrn1q_u32(a, b));
            m[i + 4] = vreinterpretq_u8_u32(vtrn2q_u32(a, b));
        }
        for i in 0..8 {
            let (a, b) = (vreinterpretq_u64_u8(m[i]), vreinterpretq_u64_u8(m[i + 8]));
            m[i] = vreinterpretq_u8_u64(vtrn1q_u64(a, b));
            m[i + 8] = vreinterpretq_u8_u64(vtrn2q_u64(a, b));
        }
    }

    #[inline(always)]
    pub(super) unsafe fn prefetch(p: *const u8) {
        core::arch::asm!("prfm pldl1keep, [{p}]", p = in(reg) p, options(nostack, readonly, preserves_flags));
    }

    #[inline(always)]
    unsafe fn prefetch_l2(p: *const u8) {
        core::arch::asm!("prfm pldl2keep, [{p}]", p = in(reg) p, options(nostack, readonly, preserves_flags));
    }

    /// The 128 planes of 128 words (`words`, a quarter of a block) to `out[0..128]`: bit `p` of
    /// word `16 r + k` lands on lane `8 k + r` of plane `p` (see [`super::lane_word`]). The
    /// 2 KB at `next` are prefetched.
    #[inline(always)]
    pub(super) unsafe fn slice(words: *const u128, out: *mut V, next: *const u128) {
        let mut tmp = core::mem::MaybeUninit::<[[uint8x16_t; 8]; 16]>::uninit();
        let tmp = tmp.as_mut_ptr() as *mut uint8x16_t;
        for r in 0..8 {
            for l in 0..4 {
                prefetch((next as *const u8).add(256 * r + 64 * l));
            }
            let src = words.add(16 * r) as *const u8;
            let mut m: [uint8x16_t; 16] = core::array::from_fn(|k| vld1q_u8(src.add(16 * k)));
            byte16x16(&mut m);
            // m[i] byte k: byte i of word 16 r + k.
            for i in 0..16 {
                *tmp.add(8 * i + r) = m[i];
            }
        }
        for i in 0..16 {
            let mut a: [uint8x16_t; 8] = core::array::from_fn(|r| *tmp.add(8 * i + r));
            bit8x8(&mut a);
            // a[b] byte k bit r: bit 8 i + b of word 16 r + k.
            for b in 0..8 {
                *out.add(8 * i + b) = vreinterpretq_u64_u8(a[b]);
            }
        }
    }

    /// The 15-plane counter at `acc` as 128 `u16` counts in word order (`out[w]` for word `w`
    /// of the quarter).
    #[inline(always)]
    pub(super) unsafe fn unslice(acc: *const V, out: *mut u16) {
        let mut lo: [uint8x16_t; 8] = core::array::from_fn(|b| vreinterpretq_u8_u64(*acc.add(b)));
        let mut hi: [uint8x16_t; 8] =
            core::array::from_fn(|b| if b < BITS - 8 { vreinterpretq_u8_u64(*acc.add(8 + b)) } else { vdupq_n_u8(0) });
        bit8x8(&mut lo);
        bit8x8(&mut hi);
        // lo[r] byte k: the low byte of lane 8 k + r's count, i.e. word 16 r + k's.
        for r in 0..8 {
            vst1q_u16(out.add(16 * r), vreinterpretq_u16_u8(vzip1q_u8(lo[r], hi[r])));
            vst1q_u16(out.add(16 * r + 8), vreinterpretq_u16_u8(vzip2q_u8(lo[r], hi[r])));
        }
    }

    /// Two consecutive slots at once: every scheduled index is the plane of the even slot,
    /// and the odd slot's plane is the next one. Each slot keeps the six low counter bits of a
    /// 64-input tree in registers, plus a three-bit counter of the trees' weight-64 carries;
    /// every seventh chunk that counter is added into the nine upper counter bits at `acc`.
    /// The `pos` entries add their planes, the `neg` entries their complements.
    #[inline(never)]
    pub(super) unsafe fn accumulate_pair(planes: *const V, pos: &[u16], neg: &[u16], acc: *mut V) {
        let p1 = planes.add(1);
        let (even, odd) = (acc, acc.add(BITS));
        let mut s0: [V; 6] = core::array::from_fn(|b| *even.add(b));
        let mut s1: [V; 6] = core::array::from_fn(|b| *odd.add(b));
        let zero = vdupq_n_u64(0);
        let (mut h0, mut h1) = ([zero; 3], [zero; 3]);
        let mut pending = 0usize;
        macro_rules! flush {
            ($slot:ident, $h:ident) => {{
                let p = $slot.add(6);
                let m6 = *p;
                *p = veorq_u64(m6, $h[0]);
                let mut carry = vandq_u64(m6, $h[0]);
                let p = $slot.add(7);
                let m7 = *p;
                *p = xor3(m7, $h[1], carry);
                carry = maj(m7, $h[1], carry);
                let p = $slot.add(8);
                let m8 = *p;
                *p = xor3(m8, $h[2], carry);
                carry = maj(m8, $h[2], carry);
                for b in 9..BITS - 1 {
                    let p = $slot.add(b);
                    let old = *p;
                    *p = veorq_u64(old, carry);
                    carry = vandq_u64(old, carry);
                }
                let p = $slot.add(BITS - 1);
                *p = veorq_u64(*p, carry);
            }};
        }
        let mut t: *const u16;
        macro_rules! two {
            ($off:expr, $fa:ident) => {{
                let i0 = *t.add($off) as usize;
                let i1 = *t.add($off + 1) as usize;
                let a0 = *planes.add(i0);
                let b0 = *planes.add(i1);
                let a1 = *p1.add(i0);
                let b1 = *p1.add(i1);
                ($fa(&mut s0[0], a0, b0), $fa(&mut s1[0], a1, b1))
            }};
        }
        macro_rules! level {
            ($bit:literal, $lower:ident, $off:expr, $fa:ident) => {{
                let (x0, x1) = $lower!($off, $fa);
                let (y0, y1) = $lower!($off + (1 << $bit), $fa);
                ($fa(&mut s0[$bit], x0, y0), $fa(&mut s1[$bit], x1, y1))
            }};
        }
        macro_rules! four { ($off:expr, $fa:ident) => { level!(1, two, $off, $fa) }; }
        macro_rules! eight { ($off:expr, $fa:ident) => { level!(2, four, $off, $fa) }; }
        macro_rules! sixteen { ($off:expr, $fa:ident) => { level!(3, eight, $off, $fa) }; }
        macro_rules! thirty_two { ($off:expr, $fa:ident) => { level!(4, sixteen, $off, $fa) }; }
        macro_rules! add_carry64 {
            ($c0:expr, $c1:expr) => {{
                let (c0, c1) = ($c0, $c1);
                let u0 = vandq_u64(h0[0], c0);
                let v0 = vandq_u64(h0[1], u0);
                h0[0] = veorq_u64(h0[0], c0);
                h0[1] = veorq_u64(h0[1], u0);
                h0[2] = veorq_u64(h0[2], v0);
                let u1 = vandq_u64(h1[0], c1);
                let v1 = vandq_u64(h1[1], u1);
                h1[0] = veorq_u64(h1[0], c1);
                h1[1] = veorq_u64(h1[1], u1);
                h1[2] = veorq_u64(h1[2], v1);
            }};
        }
        macro_rules! step_pending {
            () => {{
                pending += 1;
                if pending == 7 {
                    flush!(even, h0);
                    flush!(odd, h1);
                    h0 = [zero; 3];
                    h1 = [zero; 3];
                    pending = 0;
                }
            }};
        }
        // A carry of weight `2^$bit` from a partial tree, through half adders at that level and
        // above. The partial trees of a remainder total at most 62 inputs, so the low six bits
        // wrap modulo 64 at most once per lane over the remainder.
        macro_rules! half_adders {
            ($c0:expr, $c1:expr, $($bit:literal),+) => {{
                let (mut c0, mut c1) = ($c0, $c1);
                $(
                    let (u0, u1) = (vandq_u64(s0[$bit], c0), vandq_u64(s1[$bit], c1));
                    s0[$bit] = veorq_u64(s0[$bit], c0);
                    s1[$bit] = veorq_u64(s1[$bit], c1);
                    (c0, c1) = (u0, u1);
                )+
                add_carry64!(c0, c1);
            }};
        }
        macro_rules! entries {
            ($terms:expr, $fa:ident, $fix:ident) => {{
                let terms: &[u16] = $terms;
                for input in terms.chunks_exact(64) {
                    t = input.as_ptr();
                    let (c0, c1) = level!(5, thirty_two, 0, $fa);
                    add_carry64!($fix(c0), $fix(c1));
                    step_pending!();
                }
                let rest = terms.chunks_exact(64).remainder();
                debug_assert!(rest.len() % 2 == 0);
                if !rest.is_empty() {
                    t = rest.as_ptr();
                    let n = rest.len();
                    if n & 32 != 0 {
                        let (c0, c1) = thirty_two!(0, $fa);
                        half_adders!($fix(c0), $fix(c1), 5);
                        t = t.add(32);
                    }
                    if n & 16 != 0 {
                        let (c0, c1) = sixteen!(0, $fa);
                        half_adders!($fix(c0), $fix(c1), 4, 5);
                        t = t.add(16);
                    }
                    if n & 8 != 0 {
                        let (c0, c1) = eight!(0, $fa);
                        half_adders!($fix(c0), $fix(c1), 3, 4, 5);
                        t = t.add(8);
                    }
                    if n & 4 != 0 {
                        let (c0, c1) = four!(0, $fa);
                        half_adders!($fix(c0), $fix(c1), 2, 3, 4, 5);
                        t = t.add(4);
                    }
                    if n & 2 != 0 {
                        let (c0, c1) = two!(0, $fa);
                        half_adders!($fix(c0), $fix(c1), 1, 2, 3, 4, 5);
                    }
                    step_pending!();
                }
            }};
        }
        entries!(pos, csa, id);
        entries!(neg, csa_c, not);
        if pending != 0 {
            flush!(even, h0);
            flush!(odd, h1);
        }
        for b in 0..6 {
            *even.add(b) = s0[b];
            *odd.add(b) = s1[b];
        }
    }

    /// One task's output: the 243 counters at `acc` (slot `s` at `acc + s * BITS`) reduced
    /// modulo `Y^162 - Y^81 + 1` with the bias removed, for the 32 ring elements of the quarter;
    /// element `e`'s coefficient `i` is written to `out + e * stride + i`. Returns whether some
    /// coefficient overflowed `i16`.
    pub(super) unsafe fn write_quarter(
        acc: *const V,
        counts: &mut [[u16; QL]; SLOTS],
        bias: &[i32; SLOTS],
        out: *mut i16,
        stride: usize,
    ) -> bool {
        for s in 0..SLOTS {
            unslice(acc.add(s * BITS), counts[s].as_mut_ptr());
        }
        let mut bad = vdupq_n_u16(0);
        let mut value = |s: usize, i: usize| -> int16x8_t {
            let below = s < 81;
            let t = if below { 162 + s } else { s + 81 };
            let cs = vreinterpretq_s16_u16(vld1q_u16(counts[s].as_ptr().add(8 * i)));
            let ct = vreinterpretq_s16_u16(vld1q_u16(counts[t].as_ptr().add(8 * i)));
            let x = vsubq_s16(cs, vdupq_n_s16(bias[s] as i16));
            let bt = vdupq_n_s16(bias[t] as i16);
            let y = if below { vsubq_s16(bt, ct) } else { vsubq_s16(ct, bt) };
            let v = vaddq_s16(x, y);
            bad = vorrq_u16(bad, vmvnq_u16(vceqq_s16(v, vqaddq_s16(x, y))));
            v
        };
        for s in (0..OUT_SLOTS).step_by(2) {
            for i in 0..QL / 8 {
                // Words 8 i .. 8 i + 8: elements 2 i and 2 i + 1, components 0..4 each.
                let a = vreinterpretq_u64_s16(value(s, i));
                let b = vreinterpretq_u64_s16(value(s + 1, i));
                let e0 = out.add(2 * i * stride + 4 * s);
                vst1q_s16(e0, vreinterpretq_s16_u64(vzip1q_u64(a, b)));
                vst1q_s16(e0.add(stride), vreinterpretq_s16_u64(vzip2q_u64(a, b)));
            }
        }
        vmaxvq_u16(bad) != 0
    }

    pub(super) unsafe fn prefetch_column(p: *const u128) {
        for l in 0..(16 * QL) / 64 {
            prefetch_l2((p as *const u8).add(64 * l));
        }
    }
}

/// The word (within a quarter) whose bits lane `l` of a plane carries.
#[cfg_attr(not(test), allow(dead_code))]
fn lane_word(l: usize) -> usize {
    16 * (l % 8) + l / 8
}

struct SendPtr<T>(*mut T);
unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}

/// The fold of `columns` columns of `column_words` words at `words` against `challenges`:
/// element `e`'s coefficient `i` (`e < column_words / 4`, `i < 648`) is written to
/// `out + e * stride + i`. Tasks (quarters of 512-word blocks) are handed out dynamically to
/// `workers` threads. Panics if a coefficient overflows `i16`.
///
/// # Safety
/// `words` addresses `columns * column_words` readable `u128`; `out` addresses
/// `(column_words / 4 - 1) * stride + 648` writable `i16`. [`fits`] and [`shape_ok`] hold.
#[cfg(target_arch = "aarch64")]
pub unsafe fn fold_into(
    words: *const u128,
    columns: usize,
    column_words: usize,
    challenges: &[Challenge],
    workers: usize,
    out: *mut i16,
    stride: usize,
) {
    use neon::*;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    assert_eq!(challenges.len(), columns, "one challenge per column");
    assert!(fits(challenges) && shape_ok(column_words) && stride >= N);
    let (groups, bias) = plan(challenges, workers);
    assert!(bias.iter().all(|&b| (0..=i16::MAX as i32).contains(&b)));
    let tasks = column_words / LANES * QUARTERS;
    let next = AtomicUsize::new(0);
    let overflow = AtomicBool::new(false);
    let (words, out) = (SendPtr(words as *mut u128), SendPtr(out));
    let (words, out, groups, bias, next, overflow) = (&words, &out, &groups, &bias, &next, &overflow);
    let work = move || {
        let zero = unsafe { core::arch::aarch64::vdupq_n_u64(0) };
        let mut planes: Vec<V> = vec![zero; GUARD + COL_GROUP * STRIDE];
        let mut acc: Vec<V> = vec![zero; 2 * PAIRS * BITS];
        let mut counts: Box<[[u16; QL]; SLOTS]> = vec![[0u16; QL]; SLOTS].into_boxed_slice().try_into().unwrap();
        loop {
            let task = next.fetch_add(1, Ordering::Relaxed);
            if task >= tasks {
                break;
            }
            let (block, quarter) = (task / QUARTERS, task % QUARTERS);
            let base = block * LANES + quarter * QL;
            acc.fill(zero);
            unsafe {
                let col = |j: usize| words.0.add(j * column_words + base) as *const u128;
                for (gi, group) in groups.iter().enumerate() {
                    let last = group.first + group.columns;
                    for j in 0..group.columns {
                        let c = group.first + j;
                        let ahead = c + PF_COLUMNS;
                        let next_src = if ahead < columns { col(ahead) } else { col(c) };
                        slice(col(c), planes.as_mut_ptr().add(GUARD + j * STRIDE), next_src);
                    }
                    // The next group's first columns, while this one accumulates.
                    if gi + 1 < groups.len() {
                        for c in last..(last + PF_COLUMNS).min(columns) {
                            prefetch_column(col(c));
                        }
                    }
                    for q in 0..PAIRS {
                        accumulate_pair(
                            planes.as_ptr(),
                            &group.terms[group.offsets[q]..group.mid[q]],
                            &group.terms[group.mid[q]..group.offsets[q + 1]],
                            acc.as_mut_ptr().add(2 * q * BITS),
                        );
                    }
                }
                let dst = out.0.add((base / 4) * stride);
                if write_quarter(acc.as_ptr(), &mut counts, bias, dst, stride) {
                    overflow.store(true, Ordering::Relaxed);
                }
            }
        }
    };
    let workers = workers.clamp(1, tasks);
    std::thread::scope(|scope| {
        for _ in 1..workers {
            scope.spawn(work);
        }
        work();
    });
    assert!(!overflow.load(Ordering::Relaxed), "a fold coefficient does not fit the wire codec");
}

/// [`fold_into`] into a fresh vector of elements.
///
/// # Safety
/// `words` addresses `columns * column_words` readable `u128`.
#[cfg(target_arch = "aarch64")]
pub unsafe fn fold_all(
    words: *const u128,
    columns: usize,
    column_words: usize,
    challenges: &[Challenge],
    workers: usize,
) -> Vec<[i16; N]> {
    let mut out = vec![[0i16; N]; column_words / 4];
    fold_into(words, columns, column_words, challenges, workers, out.as_mut_ptr() as *mut i16, N);
    out
}

#[cfg(all(test, target_arch = "aarch64"))]
mod tests {
    use super::neon::*;
    use super::*;
    use crate::fold_bits::{self as fb, reference_element};

    fn next(x: &mut u128) -> u128 {
        *x ^= *x << 13;
        *x ^= *x >> 7;
        *x ^= *x << 17;
        x.wrapping_mul(0x2545F4914F6CDD1Du128) ^ (*x >> 64)
    }

    fn challenge(x: &mut u128, weight: usize) -> Challenge {
        let mut taken = [false; 162];
        let mut positions = Vec::new();
        while positions.len() < weight {
            let p = (next(x) % 162) as u8;
            if !taken[p as usize] {
                taken[p as usize] = true;
                positions.push(p);
            }
        }
        positions.sort_unstable();
        let mut c = Challenge { positions: [0; 32], signs: (next(x) as u32) & ((1u64 << weight) - 1) as u32, weight };
        c.positions[..weight].copy_from_slice(&positions);
        c
    }

    fn bit(v: V, l: usize) -> u64 {
        let w: [u64; 2] = unsafe { core::mem::transmute(v) };
        (w[l / 64] >> (l % 64)) & 1
    }

    #[test]
    fn slice_holds_the_words_bits() {
        let mut x = 0x0123_4567_89AB_CDEFu128;
        let words: Vec<u128> = (0..QL).map(|_| next(&mut x)).collect();
        let mut planes = vec![unsafe { core::arch::aarch64::vdupq_n_u64(0) }; 128];
        unsafe { slice(words.as_ptr(), planes.as_mut_ptr(), words.as_ptr()) };
        for p in 0..128 {
            for l in 0..QL {
                assert_eq!(bit(planes[p], l), ((words[lane_word(l)] >> p) & 1) as u64, "plane {p} lane {l}");
            }
        }
    }

    #[test]
    fn unslice_reads_the_counter() {
        let mut x = 0xF00D_1234u128;
        let acc: Vec<V> = (0..BITS).map(|_| unsafe { core::mem::transmute(next(&mut x)) }).collect();
        let mut got = [0u16; QL];
        unsafe { unslice(acc.as_ptr(), got.as_mut_ptr()) };
        for w in 0..QL {
            let l = (0..QL).find(|&l| lane_word(l) == w).unwrap();
            let want: u16 = (0..BITS).map(|b| (bit(acc[b], l) as u16) << b).sum();
            assert_eq!(got[w], want, "word {w}");
        }
    }

    /// The slot-pair kernel on schedules of every length class, in pieces, both signs, up to
    /// the top of the counter range.
    #[test]
    fn pair_kernel_counts_exactly() {
        let mut x = 0x5151_2323_7777u128;
        let zero = unsafe { core::arch::aarch64::vdupq_n_u64(0) };
        let mut planes = vec![zero; 66];
        for p in planes.iter_mut().skip(2) {
            *p = unsafe { core::mem::transmute(next(&mut x) & next(&mut x)) };
        }
        for lens in [
            vec![16usize],
            vec![32],
            vec![48],
            vec![2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62],
            vec![80, 112, 144],
            vec![64 * 7 + 48; 9],
            vec![16; 40],
            vec![16000, 752, 15232],
            vec![0, 64],
        ] {
            let mut acc = vec![zero; 2 * BITS];
            let mut want = [[0usize; QL]; 2];
            for (i, &n) in lens.iter().enumerate() {
                let m = lens[(i + 1) % lens.len()];
                let mut terms = |n: usize| -> Vec<u16> {
                    (0..n).map(|i| if i % 13 == 5 { 0 } else { (2 + next(&mut x) % 63) as u16 }).collect()
                };
                let (pos, neg) = (terms(n), terms(m));
                for (list, complement) in [(&pos, 0), (&neg, 1)] {
                    for &o in list.iter() {
                        let p = o as usize;
                        for l in 0..QL {
                            want[0][l] += (bit(planes[p], l) ^ complement) as usize;
                            want[1][l] += (bit(planes[p + 1], l) ^ complement) as usize;
                        }
                    }
                }
                unsafe { accumulate_pair(planes.as_ptr(), &pos, &neg, acc.as_mut_ptr()) };
            }
            for slot in 0..2 {
                for l in 0..QL {
                    let got: usize = (0..BITS).map(|b| (bit(acc[slot * BITS + b], l) as usize) << b).sum();
                    assert_eq!(got, want[slot][l] & 0x7fff, "lens {lens:?} slot {slot} lane {l}");
                }
            }
        }
    }

    fn check(words: &[u128], columns: usize, column_words: usize, cs: &[Challenge], workers: usize) {
        let got = unsafe { fold_all(words.as_ptr(), columns, column_words, cs, workers) };
        let old = unsafe { fb::fold_all::<fb::V, fb::Portable>(words.as_ptr(), columns, column_words, cs, &fb::Portable) };
        assert_eq!(got.len(), column_words / 4);
        for m in 0..column_words / 4 {
            assert_eq!(got[m][..], old[m][..], "columns {columns} words {column_words} element {m} vs fold_bits");
        }
        for m in (0..column_words / 4).step_by(37).chain([column_words / 4 - 1]) {
            let want = reference_element(words, columns, column_words, cs, m);
            assert_eq!(got[m][..], want[..], "columns {columns} words {column_words} element {m}");
        }
    }

    #[test]
    fn fold_matches_the_reference() {
        let mut x = 0xDEAD_BEEF_CAFE_F00Du128;
        for (columns, column_words, workers) in
            [(1usize, 512usize, 1usize), (2, 512, 2), (33, 1024, 3), (40, 1536, 4), (70, 512, 8), (97, 2048, 5)]
        {
            let words: Vec<u128> = (0..columns * column_words).map(|_| next(&mut x)).collect();
            let cs: Vec<Challenge> = (0..columns)
                .map(|j| challenge(&mut x, if j % 5 == 0 { 28 } else { j % 29 }))
                .collect();
            check(&words, columns, column_words, &cs, workers);
        }
    }

    /// All degrees, both signs, and zero / all-one words.
    #[test]
    fn fold_every_degree_and_sign() {
        for sign in [0u32, 1] {
            let cs: Vec<Challenge> = (0..162u8)
                .map(|m| {
                    let mut c = Challenge { positions: [0; 32], signs: sign, weight: 1 };
                    c.positions[0] = m;
                    c
                })
                .collect();
            let words: Vec<u128> = (0..cs.len() * LANES).map(|i| if i % 7 == 0 { 0 } else { u128::MAX }).collect();
            check(&words, cs.len(), LANES, &cs, 4);
        }
    }

    /// Full-weight challenges on dense words: near the top of the counter range.
    #[test]
    fn fold_full_weight_dense() {
        let mut x = 0x7777_1111_2222u128;
        let columns = 1024;
        let words: Vec<u128> = (0..columns * LANES).map(|_| next(&mut x) | next(&mut x)).collect();
        let cs: Vec<Challenge> = (0..columns).map(|_| challenge(&mut x, 28)).collect();
        assert!(fits(&cs));
        check(&words, columns, LANES, &cs, 8);
    }

    /// Full-size timing (1024 columns of 2^14 words): `cargo test --release -- --ignored bench`.
    #[test]
    #[ignore = "full-size timing"]
    fn bench_full() {
        let mut x = 0x1234_5678u128;
        let (columns, column_words) = (1024usize, 1usize << 14);
        let words: Vec<u128> = (0..columns * column_words).map(|_| next(&mut x)).collect();
        let cs: Vec<Challenge> = (0..columns).map(|_| challenge(&mut x, 28)).collect();
        let workers = crate::par::threads();
        let mut out = vec![[0i16; N]; column_words / 4];
        let mut times = Vec::new();
        for _ in 0..7 {
            let start = std::time::Instant::now();
            unsafe { fold_into(words.as_ptr(), columns, column_words, &cs, workers, out.as_mut_ptr() as *mut i16, N) };
            times.push(start.elapsed().as_micros());
        }
        let want = reference_element(&words, columns, column_words, &cs, 1234);
        assert_eq!(out[1234][..], want[..]);
        eprintln!("fold2 bench workers={workers} us={times:?}");
    }
}
