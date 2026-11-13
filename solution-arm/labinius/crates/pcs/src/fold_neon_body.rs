    use super::{Challenge, N, Q162, WORD_BITS};
    use crate::ring::{Representation, RingElement};
    use core::arch::aarch64::*;

    /// Lanes per block: 128 words, one plane is one `q` register.
    pub const LANES: usize = 128;
    /// Ring elements per block.
    pub const BLOCK_ELEMENTS: usize = LANES / 4;
    /// Slots of `Z[Y] / (Y^243 + 1)`.
    const SLOTS: usize = 243;
    /// Slot pairs; the odd slot of the last pair (243) is a dummy.
    const PAIRS: usize = SLOTS.div_ceil(2);
    /// Zero planes in front of the first column (plane 1 is column 0's leading guard).
    const GUARD: usize = 2;
    /// Planes per column: 128 and the trailing zero guard (the next column's leading guard).
    const STRIDE: usize = WORD_BITS + 1;
    /// The first of the two zero planes and the two all-ones planes after the columns, which
    /// the padding entries of a list's tail point at.
    const PAD_ZERO: usize = GUARD + GROUP * STRIDE;
    const PAD_ONES: usize = PAD_ZERO + 2;
    /// The group's plane buffer.
    const PLANES: usize = PAD_ONES + 2;
    /// Entries of a subtree: lists run in subtrees of eight, the last one padded.
    const SUB: usize = 8;
    /// Counter bits kept in registers during a pair's pass.
    const LO: usize = 6;
    /// Counter bits above them, in memory: counts stay below `2^15`.
    const HI: usize = 9;
    /// Planes added to every offset so that it is non-negative (positions are below 162).
    const BIAS_PLANES: usize = 160;
    const _: () = assert!(16 * (GUARD + (GROUP - 1) * STRIDE + BIAS_PLANES + WORD_BITS) <= Off::MAX as usize);
    const _: () = assert!(16 * (BIAS_PLANES + PLANES) <= Off::MAX as usize);
    const _: () = assert!(GUARD + BIAS_PLANES >= Q162);

    type V = uint8x16_t;

    /// Whether the counters cannot overflow: a slot takes at most one entry per term.
    pub fn fits(challenges: &[Challenge]) -> bool {
        challenges.iter().map(|c| c.weight).sum::<usize>() <= i16::MAX as usize
    }

    /// One column group's share of the schedule.
    pub struct Group {
        first: usize,
        columns: usize,
        /// Per sign (0 positive, 1 negative), the group's entries sorted by position: byte
        /// offsets of the even slot's plane relative to the pair's base pointer.
        list: [Vec<Off>; 2],
        /// Per pair: positive `[lo, hi)`, negative `[lo, hi)`.
        win: Vec<[u16; 4]>,
        /// Per pair and sign, the last `len % SUB` entries of the window, then entries of the
        /// pad planes (zero for positive, all-ones for negative: neither adds anything), as
        /// offsets from that pair's base.
        tail: Vec<[[Off; SUB]; 2]>,
    }

    /// The number of groups of `columns` columns.
    pub fn groups(columns: usize) -> usize {
        columns.div_ceil(GROUP)
    }
    /// The first column of group `g`: the groups split the columns evenly, at most `GROUP`
    /// each.
    fn group_first(columns: usize, g: usize) -> usize {
        g * columns / groups(columns)
    }

    /// Positions of the sorted entries, shifted to be non-negative: real terms `0 .. 162`,
    /// wrapped ones `-127 .. 0`.
    const KEY0: i32 = 127;
    const KEYS: usize = KEY0 as usize + Q162;

    impl Group {
        /// The entries of group `g` (columns `group_first(n, g) ..`) of `challenges`,
        /// bucket-sorted by position, per pair the windows of the terms reaching it, and the
        /// windows' padded tails.
        pub fn of(challenges: &[Challenge], g: usize) -> Group {
            let n = challenges.len();
            let first = group_first(n, g);
            let cs = &challenges[first..group_first(n, g + 1)];
            // Each entry is `(key, offset)`; a term at `m` reaches pair `i` when `[2i, 2i + 1]`
            // meets `[m, m + 127]`, i.e. `2i - 127 <= m <= 2i + 1`.
            let mut start = [[0u16; KEYS + 1]; 2];
            let each = |f: &mut dyn FnMut(usize, usize, Off)| {
                for (jl, c) in cs.iter().enumerate() {
                    assert!(c.weight <= c.positions.len());
                    let col = (GUARD + jl * STRIDE + BIAS_PLANES) as i32;
                    for t in 0..c.weight {
                        let p = c.positions[t] as usize;
                        assert!(p < Q162, "a position is below 162");
                        let negative = ((c.signs >> t) & 1 == 1) ^ (p % 2 == 1);
                        let m = p as i32;
                        f(negative as usize, (m + KEY0) as usize, (16 * (col - m)) as Off);
                        // Slots `p + q >= 243` wrap to `p + q - 243` with the other sign: a
                        // term at position `p - 243`.
                        if p + WORD_BITS > SLOTS {
                            let m = m - SLOTS as i32;
                            f(!negative as usize, (m + KEY0) as usize, (16 * (col - m)) as Off);
                        }
                    }
                }
            };
            each(&mut |sign, key, _| start[sign][key + 1] += 1);
            for st in start.iter_mut() {
                for k in 0..KEYS {
                    st[k + 1] += st[k];
                }
            }
            let mut at = start;
            let mut list = [vec![0 as Off; start[0][KEYS] as usize], vec![0 as Off; start[1][KEYS] as usize]];
            each(&mut |sign, key, off| {
                list[sign][at[sign][key] as usize] = off;
                at[sign][key] += 1;
            });
            let win: Vec<[u16; 4]> = (0..PAIRS)
                .map(|i| {
                    let lo = 2 * i;
                    let hi = (2 * i + 1 + KEY0 as usize + 1).min(KEYS);
                    [start[0][lo], start[0][hi], start[1][lo], start[1][hi]]
                })
                .collect();
            let tail = win
                .iter()
                .enumerate()
                .map(|(i, w)| {
                    core::array::from_fn(|sign| {
                        let (lo, hi) = (w[2 * sign] as usize, w[2 * sign + 1] as usize);
                        let full = lo + (hi - lo) / SUB * SUB;
                        let pad = [PAD_ZERO, PAD_ONES][sign];
                        let pad = (16 * (BIAS_PLANES + pad) - 32 * i) as Off;
                        core::array::from_fn(|k| if full + k < hi { list[sign][full + k] } else { pad })
                    })
                })
                .collect();
            Group { first, columns: cs.len(), list, win, tail }
        }
    }

    /// Per slot, the number of negative entries over the groups (each adds one to both slots
    /// of its pair).
    fn bias<'a>(groups: impl Iterator<Item = &'a Group>) -> [i32; SLOTS + 1] {
        let mut bias = [0i32; SLOTS + 1];
        for g in groups {
            for (i, w) in g.win.iter().enumerate() {
                let n = (w[3] - w[2]) as i32;
                bias[2 * i] += n;
                bias[2 * i + 1] += n;
            }
        }
        bias
    }

    /// A worker's buffers.
    pub struct Workspace {
        planes: Vec<V>,
        lo: Vec<V>,
        hi: Vec<V>,
        tmp: Vec<V>,
        counts: Vec<[uint16x8_t; 16]>,
    }

    impl Workspace {
        pub fn new() -> Workspace {
            unsafe {
                let mut planes = vec![vdupq_n_u8(0); PLANES];
                planes[PAD_ONES] = vdupq_n_u8(0xFF);
                planes[PAD_ONES + 1] = vdupq_n_u8(0xFF);
                Workspace {
                    planes,
                    lo: vec![vdupq_n_u8(0); PAIRS * 2 * LO],
                    hi: vec![vdupq_n_u8(0); PAIRS * 2 * HI],
                    tmp: vec![vdupq_n_u8(0); WORD_BITS],
                    counts: vec![[vdupq_n_u16(0); 16]; SLOTS + 1],
                }
            }
        }
    }

    impl Default for Workspace {
        fn default() -> Self {
            Self::new()
        }
    }

    // -----------------------------------------------------------------------------------------
    // the transposes
    // -----------------------------------------------------------------------------------------

    /// 16 x 16 byte transpose: output register `k` byte `g` is input register `g` byte `k`.
    #[inline(always)]
    unsafe fn t16(r: [V; 16]) -> [V; 16] {
        let mut a = r;
        let mut b = r;
        for g in (0..16).step_by(2) {
            b[g] = vtrn1q_u8(a[g], a[g + 1]);
            b[g + 1] = vtrn2q_u8(a[g], a[g + 1]);
        }
        for g in [0, 1, 4, 5, 8, 9, 12, 13] {
            let (x, y) = (vreinterpretq_u16_u8(b[g]), vreinterpretq_u16_u8(b[g + 2]));
            a[g] = vreinterpretq_u8_u16(vtrn1q_u16(x, y));
            a[g + 2] = vreinterpretq_u8_u16(vtrn2q_u16(x, y));
        }
        for g in [0, 1, 2, 3, 8, 9, 10, 11] {
            let (x, y) = (vreinterpretq_u32_u8(a[g]), vreinterpretq_u32_u8(a[g + 4]));
            b[g] = vreinterpretq_u8_u32(vtrn1q_u32(x, y));
            b[g + 4] = vreinterpretq_u8_u32(vtrn2q_u32(x, y));
        }
        for g in 0..8 {
            let (x, y) = (vreinterpretq_u64_u8(b[g]), vreinterpretq_u64_u8(b[g + 8]));
            a[g] = vreinterpretq_u8_u64(vtrn1q_u64(x, y));
            a[g + 8] = vreinterpretq_u8_u64(vtrn2q_u64(x, y));
        }
        a
    }

    /// The 128 words at `src` as 128 bit planes at `dst`: plane `q` lane `l` is bit `q` of word
    /// `l`. Groups of eight words first exchange the word index with the bit-in-byte index
    /// (nibble, pair and bit swaps), then a byte transpose per bit-in-byte index finishes.
    ///
    /// # Safety
    /// `src` addresses 2048 readable bytes, `dst` and `tmp` 128 writable planes.
    #[inline(always)]
    pub unsafe fn slice(src: *const u8, dst: *mut V, tmp: *mut V) {
        let m33 = vdupq_n_u8(0x33);
        let m55 = vdupq_n_u8(0x55);
        for g in 0..16 {
            let p = src.add(128 * g);
            let mut a: [V; 8] = core::array::from_fn(|i| vld1q_u8(p.add(16 * i)));
            for i in 0..4 {
                let (x, y) = (a[i], a[i + 4]);
                a[i] = vsliq_n_u8::<4>(x, y);
                a[i + 4] = vsriq_n_u8::<4>(y, x);
            }
            for i in [0, 1, 4, 5] {
                let (x, y) = (a[i], a[i + 2]);
                a[i] = vbslq_u8(m33, x, vshlq_n_u8::<2>(y));
                a[i + 2] = vbslq_u8(m33, vshrq_n_u8::<2>(x), y);
            }
            for i in [0, 2, 4, 6] {
                let (x, y) = (a[i], a[i + 1]);
                a[i] = vbslq_u8(m55, x, vshlq_n_u8::<1>(y));
                a[i + 1] = vbslq_u8(m55, vshrq_n_u8::<1>(x), y);
            }
            // Register `i`, byte `k`, bit `j`: bit `8k + i` of word `8g + j`.
            for (i, &x) in a.iter().enumerate() {
                *tmp.add(16 * i + g) = x;
            }
        }
        for i in 0..8 {
            let r: [V; 16] = core::array::from_fn(|g| *tmp.add(16 * i + g));
            let p = t16(r);
            for (k, &x) in p.iter().enumerate() {
                *dst.add(8 * k + i) = x;
            }
        }
    }

    /// `a ^ b` that LLVM cannot fuse into an `EOR3` (V0-only on Neoverse V2).
    #[inline(always)]
    fn eor_plain(mut a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
        unsafe {
            core::arch::asm!(
                "eor {a:v}.16b, {a:v}.16b, {b:v}.16b",
                a = inout(vreg) a, b = in(vreg) b,
                options(pure, nomem, nostack, preserves_flags)
            );
        }
        a
    }

    /// 8 x 8 bit transpose in each 64-bit lane: byte `r` bit `c` goes to byte `c` bit `r`.
    #[inline(always)]
    unsafe fn bits8x8(x: uint64x2_t) -> uint64x2_t {
        let t = vandq_u64(veorq_u64(x, vshrq_n_u64::<7>(x)), vdupq_n_u64(0x00AA_00AA_00AA_00AA));
        let x = eor_plain(veorq_u64(x, t), vshlq_n_u64::<7>(t));
        let t = vandq_u64(veorq_u64(x, vshrq_n_u64::<14>(x)), vdupq_n_u64(0x0000_CCCC_0000_CCCC));
        let x = eor_plain(veorq_u64(x, t), vshlq_n_u64::<14>(t));
        let t = vandq_u64(veorq_u64(x, vshrq_n_u64::<28>(x)), vdupq_n_u64(0x0000_0000_F0F0_F0F0));
        eor_plain(veorq_u64(x, t), vshlq_n_u64::<28>(t))
    }

    /// A 16-plane bit-sliced counter (plane `w` = weight `2^w`) as 128 `u16` counts: register
    /// `k` holds lanes `8k .. 8k + 8`.
    #[inline(always)]
    pub unsafe fn unslice(p: [V; 16]) -> [uint16x8_t; 16] {
        let r = t16(p);
        core::array::from_fn(|k| {
            let x = vreinterpretq_u8_u64(bits8x8(vreinterpretq_u64_u8(r[k])));
            vreinterpretq_u16_u8(vzip1q_u8(x, vextq_u8::<8>(x, x)))
        })
    }

    // -----------------------------------------------------------------------------------------
    // the counters
    // -----------------------------------------------------------------------------------------

    /// `t` through an empty asm statement, so that `s ^ t` is not fused into an `EOR3` with the
    /// XOR that made `t` (one `EOR3` per cycle on V2, against four `EOR`s).
    #[inline(always)]
    fn opaque(mut t: V) -> V {
        unsafe {
            core::arch::asm!("/* {0:v} */", inout(vreg) t, options(pure, nomem, nostack, preserves_flags));
        }
        t
    }

    /// A full adder of `s` and two planes: the new sum in `s`, the carry returned. On
    /// complemented planes and a complemented `s` it computes the complemented sum and carry
    /// (see [`pass`]).
    ///
    /// Three plain ops in an asm block: `t = a ^ b` (into `b`), the carry `BIT(a, s, t)` (into
    /// `a`, so no register copies) and `s ^= t`. Written as intrinsics, LLVM reassociates the
    /// chain of `s ^= t` into `EOR3`s, which Neoverse V2 issues on one pipe only.
    #[inline(always)]
    unsafe fn fa(s: &mut V, mut a: V, b: V) -> V {
        core::arch::asm!(
            "eor {b:v}.16b, {a:v}.16b, {b:v}.16b",
            "bit {a:v}.16b, {s:v}.16b, {b:v}.16b",
            "eor {s:v}.16b, {s:v}.16b, {b:v}.16b",
            a = inout(vreg) a, b = inout(vreg) b => _, s = inout(vreg) *s,
            options(pure, nomem, nostack, preserves_flags)
        );
        a
    }

    /// Bits of the register counter `h` above the `LO` bits (weights `2^6 .. 2^8`), added into
    /// the memory bits at the end of a pass, or within it once the trees have added
    /// `H_MAX - 1` carries.
    const HB: usize = 3;
    /// Carries of weight 64 that `h` holds.
    const H_MAX: usize = (1 << HB) - 1;

    /// `hi += h`: the three-bit counter `h` (weights `2^6 .. 2^8`) into the nine upper counter
    /// bits at `p`.
    #[inline(always)]
    unsafe fn flush(p: *mut V, h: &[V; HB]) {
        let m = *p;
        *p = veorq_u8(m, h[0]);
        let mut c = vandq_u8(m, h[0]);
        for b in 1..HB {
            let m = *p.add(b);
            let t = opaque(veorq_u8(m, h[b]));
            *p.add(b) = veorq_u8(t, c);
            c = vbslq_u8(t, c, m);
        }
        for b in HB..HI {
            let m = *p.add(b);
            *p.add(b) = veorq_u8(m, c);
            c = vandq_u8(m, c);
        }
    }

    /// One pair's pass over one group: the positive then the negative entries. Each list is
    /// `n8` subtrees of eight entries at `pos` / `neg` (whole 32-entry trees, then up to
    /// three more subtrees) and its padded tail; the subtrees' weight-8 carries meet in the
    /// counter bits 3 and 4, and whatever waits there at the end goes into `h` with the tail's,
    /// without branching on the remainder.
    ///
    /// # Safety
    /// `be + o` and `be + o + 16` are readable planes for every offset `o` of the lists and
    /// tails; `lo` and `hi` address the pair's `2 LO` and `2 HI` counter planes.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn pass(
        be: *const u8,
        pos: *const Off,
        pos8: usize,
        neg: *const Off,
        neg8: usize,
        tail: &[[Off; SUB]; 2],
        lo: *mut V,
        hi: *mut V,
    ) {
        let bo = be.add(16);
        let z = vdupq_n_u8(0);
        let ones = vdupq_n_u8(0xFF);
        let mut s0: [V; LO] = core::array::from_fn(|b| *lo.add(b));
        let mut s1: [V; LO] = core::array::from_fn(|b| *lo.add(LO + b));
        // The second bit-0 counters of the two slots, zero at the pass's start.
        let (mut r0, mut r1) = (z, z);
        let mut h0 = [z; HB];
        let mut h1 = [z; HB];
        // Carries in `h`: the trees' carries flush it at `H_MAX - 1`, which leaves room for
        // the one carry of the list's end.
        let mut hn = 0usize;
        let mut t: *const Off;
        macro_rules! add_h {
            ($c0:expr, $c1:expr) => {{
                let (mut c0, mut c1) = ($c0, $c1);
                for b in 0..HB - 1 {
                    let u0 = vandq_u8(h0[b], c0);
                    h0[b] = veorq_u8(h0[b], c0);
                    let u1 = vandq_u8(h1[b], c1);
                    h1[b] = veorq_u8(h1[b], c1);
                    (c0, c1) = (u0, u1);
                }
                h0[HB - 1] = veorq_u8(h0[HB - 1], c0);
                h1[HB - 1] = veorq_u8(h1[HB - 1], c1);
            }};
        }
        macro_rules! two {
            ($off:expr, $neg:literal) => {{
                // u16: four offsets per 64-bit load (one per four entries), split with shifts.
                // u32: the pair's two offsets as one `ldp w, w`.
                let (o0, o1) = if core::mem::size_of::<Off>() == 2 {
                    let w = (t.add($off & !3) as *const u64).read_unaligned();
                    ((w >> (16 * ($off % 4))) as u16 as usize, (w >> (16 * ($off % 4 + 1))) as u16 as usize)
                } else {
                    (*t.add($off) as usize, *t.add($off + 1) as usize)
                };
                let a0 = vld1q_u8(be.add(o0));
                let a1 = vld1q_u8(bo.add(o0));
                let b0 = vld1q_u8(be.add(o1));
                let b1 = vld1q_u8(bo.add(o1));
                // Alternate leaf pairs go to a second bit-0 counter, halving that chain.
                if ($off / 2) % 2 == 0 {
                    (fa(&mut s0[0], a0, b0), fa(&mut s1[0], a1, b1))
                } else {
                    (fa(&mut r0, a0, b0), fa(&mut r1, a1, b1))
                }
            }};
        }
        macro_rules! level {
            ($bit:literal, $lower:ident, $off:expr, $neg:literal) => {{
                let (x0, x1) = $lower!($off, $neg);
                let (y0, y1) = $lower!($off + (1 << $bit), $neg);
                (fa(&mut s0[$bit], x0, y0), fa(&mut s1[$bit], x1, y1))
            }};
        }
        macro_rules! four { ($off:expr, $neg:literal) => { level!(1, two, $off, $neg) }; }
        macro_rules! eight { ($off:expr, $neg:literal) => { level!(2, four, $off, $neg) }; }
        // The true carry of a tree.
        macro_rules! fix {
            ($c:expr, $neg:literal) => {
                if $neg { vmvnq_u8($c) } else { $c }
            };
        }
        // A list: subtrees of eight, their carries paired at bits 3 and 4 (`p3`, `p4` hold the
        // one waiting, or nothing: zero, all-ones for the complemented carries of `NEG`).
        // A weight-32 carry (true value) into counter bit 5, its carry (weight 64) into `h`. In
        // the negative list bit 5 is held complemented: `S ^ e` stays the complemented sum and
        // the true carry is `e & ~S`.
        macro_rules! half5 {
            ($e0:expr, $e1:expr, $neg:literal) => {{
                let (e0, e1) = ($e0, $e1);
                let (c0, c1) = if $neg {
                    (vbicq_u8(e0, s0[5]), vbicq_u8(e1, s1[5]))
                } else {
                    (vandq_u8(e0, s0[5]), vandq_u8(e1, s1[5]))
                };
                s0[5] = veorq_u8(s0[5], e0);
                s1[5] = veorq_u8(s1[5], e1);
                add_h!(c0, c1);
            }};
        }
        macro_rules! entries {
            ($list:expr, $n8:expr, $tail:expr, $neg:literal) => {{
                let none = if $neg { ones } else { z };
                let (mut p30, mut p31) = (none, none);
                let (mut p40, mut p41) = (none, none);
                t = $list;
                // Whole 32-entry trees in pairs: their weight-32 carries meet in counter bit 5,
                // whose carry (weight 64) goes into `h`.
                macro_rules! tree32 {
                    () => {{
                        let (a0, a1) = eight!(0, $neg);
                        let (b0, b1) = eight!(SUB, $neg);
                        let d0 = fa(&mut s0[3], a0, b0);
                        let d1 = fa(&mut s1[3], a1, b1);
                        let (a0, a1) = eight!(2 * SUB, $neg);
                        let (b0, b1) = eight!(3 * SUB, $neg);
                        let f0 = fa(&mut s0[3], a0, b0);
                        let f1 = fa(&mut s1[3], a1, b1);
                        let e = (fa(&mut s0[4], d0, f0), fa(&mut s1[4], d1, f1));
                        t = t.add(4 * SUB);
                        e
                    }};
                }
                macro_rules! check_h {
                    () => {{
                        hn += 1;
                        if hn >= H_MAX - 1 {
                            flush(hi, &h0);
                            flush(hi.add(HI), &h1);
                            h0 = [z; HB];
                            h1 = [z; HB];
                            hn = 0;
                        }
                    }};
                }
                for _ in 0..$n8 / 8 {
                    let (x0, x1) = tree32!();
                    let (y0, y1) = tree32!();
                    let c0 = fa(&mut s0[5], x0, y0);
                    let c1 = fa(&mut s1[5], x1, y1);
                    add_h!(fix!(c0, $neg), fix!(c1, $neg));
                    check_h!();
                }
                if ($n8 / 4) % 2 == 1 {
                    let (e0, e1) = tree32!();
                    half5!(fix!(e0, $neg), fix!(e1, $neg), $neg);
                    check_h!();
                }
                // Up to three more subtrees of eight.
                let left = $n8 % 4;
                if left != 0 {
                    (p30, p31) = eight!(0, $neg);
                    t = t.add(SUB);
                    if left != 1 {
                        let (c0, c1) = eight!(0, $neg);
                        t = t.add(SUB);
                        p40 = fa(&mut s0[3], p30, c0);
                        p41 = fa(&mut s1[3], p31, c1);
                        (p30, p31) = (none, none);
                        if left != 2 {
                            (p30, p31) = eight!(0, $neg);
                        }
                    }
                }
                // The tail, and whatever waits at bits 3 and 4, into `h`.
                t = $tail.as_ptr();
                let (c0, c1) = eight!(0, $neg);
                let d0 = fa(&mut s0[3], p30, c0);
                let d1 = fa(&mut s1[3], p31, c1);
                let e0 = fa(&mut s0[4], p40, d0);
                let e1 = fa(&mut s1[4], p41, d1);
                half5!(fix!(e0, $neg), fix!(e1, $neg), $neg);
                hn += 1;
            }};
        }
        entries!(pos, pos8, tail[0], false);
        // The negative list's planes stand for complements, and so do its carries. With the
        // counter bits complemented too, the positive full adder is exact for it: for
        // `t = a ^ b` it returns `t ? ~s : ~a = ~(t ? s : a)` as the carry and `~s ^ t = ~(s ^ t)`
        // as the sum. The carry does not wait for the new sum, which shortens the
        // dependency chain of every full adder of the list.
        for b in 0..LO {
            s0[b] = vmvnq_u8(s0[b]);
            s1[b] = vmvnq_u8(s1[b]);
        }
        (r0, r1) = (vmvnq_u8(r0), vmvnq_u8(r1));
        entries!(neg, neg8, tail[1], true);
        for b in 0..LO {
            s0[b] = vmvnq_u8(s0[b]);
            s1[b] = vmvnq_u8(s1[b]);
        }
        (r0, r1) = (vmvnq_u8(r0), vmvnq_u8(r1));
        // The second bit-0 counters into the first: a ripple through the `LO` bits, the carry
        // out of the top into `h`.
        let (mut c0, mut c1) = (r0, r1);
        for b in 0..LO {
            let (u0, u1) = (vandq_u8(s0[b], c0), vandq_u8(s1[b], c1));
            s0[b] = veorq_u8(s0[b], c0);
            s1[b] = veorq_u8(s1[b], c1);
            (c0, c1) = (u0, u1);
        }
        if hn >= H_MAX {
            flush(hi, &h0);
            flush(hi.add(HI), &h1);
            h0 = [z; HB];
            h1 = [z; HB];
            hn = 0;
        }
        add_h!(c0, c1);
        hn += 1;
        debug_assert!(hn <= H_MAX);
        flush(hi, &h0);
        flush(hi.add(HI), &h1);
        for b in 0..LO {
            *lo.add(b) = s0[b];
            *lo.add(LO + b) = s1[b];
        }
    }

    #[inline(always)]
    unsafe fn prefetch_l2(p: *const u8) {
        core::arch::asm!("prfm pldl2keep, [{0}]", in(reg) p, options(nostack, preserves_flags, readonly));
    }

    /// The fold of block `block` (ring elements `32 block ..`), written to `out[0 .. 32]`.
    /// `claim` is called once, when the last group starts, for the block to run next (whose
    /// first slices are then prefetched); its answer is returned.
    ///
    /// # Safety
    /// `words` holds the groups' columns of `column_words` words (a multiple of 128) with
    /// `128 block + 128 <= column_words`, and so for any block `claim` returns; `out` addresses
    /// 32 writable (possibly uninitialized) ring elements.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn fold_block(
        words: *const u128,
        column_words: usize,
        groups: &[&Group],
        bias: &[i32; SLOTS + 1],
        block: usize,
        claim: &mut dyn FnMut() -> Option<usize>,
        ws: &mut Workspace,
        out: *mut RingElement,
    ) -> Option<usize> {
        let z = vdupq_n_u8(0);
        ws.lo.fill(z);
        ws.hi.fill(z);
        let planes = ws.planes.as_mut_ptr();
        let (lo, hi, tmp) = (ws.lo.as_mut_ptr(), ws.hi.as_mut_ptr(), ws.tmp.as_mut_ptr());
        let src = |column: usize, block: usize| words.add(column * column_words + block * LANES) as *const u8;
        let mut claimed = None;
        for (gi, g) in groups.iter().enumerate() {
            for jl in 0..g.columns {
                slice(src(g.first + jl, block), planes.add(GUARD + jl * STRIDE), tmp);
            }
            // The next group's 2 KB slices (or the next block's first group's), 64-byte lines
            // spread over the pairs.
            let next = if gi + 1 < groups.len() {
                Some((groups[gi + 1], block))
            } else {
                claimed = claim();
                claimed.map(|b| (groups[0], b))
            };
            let lines = next.map_or(0, |(n, _)| n.columns * 32);
            let per = lines.div_ceil(PAIRS);
            let base = (planes as *const u8).wrapping_sub(16 * BIAS_PLANES);
            for i in 0..PAIRS {
                let w = *g.win.get_unchecked(i);
                let be = base.wrapping_add(32 * i);
                pass(
                    be,
                    g.list[0].as_ptr().add(w[0] as usize),
                    (w[1] - w[0]) as usize / SUB,
                    g.list[1].as_ptr().add(w[2] as usize),
                    (w[3] - w[2]) as usize / SUB,
                    g.tail.get_unchecked(i),
                    lo.add(2 * LO * i),
                    hi.add(2 * HI * i),
                );
                if let Some((n, b)) = next {
                    for l in per * i..(per * (i + 1)).min(lines) {
                        prefetch_l2(src(n.first + l / 32, b).add(64 * (l % 32)));
                    }
                }
            }
        }
        // The counters as integers.
        for s in 0..SLOTS {
            let (i, odd) = (s / 2, s % 2);
            let l = lo.add(2 * LO * i + LO * odd);
            let h = hi.add(2 * HI * i + HI * odd);
            let p: [V; 16] = core::array::from_fn(|b| {
                if b < LO {
                    *l.add(b)
                } else if b < LO + HI {
                    *h.add(b - LO)
                } else {
                    z
                }
            });
            ws.counts[s] = unslice(p);
        }
        // Reduction modulo `Y^162 - Y^81 + 1`, the bias, and the elements.
        for e in 0..BLOCK_ELEMENTS {
            core::ptr::addr_of_mut!((*out.add(e)).representation).write(Representation::Coefficients);
        }
        let mut overflow = vdupq_n_s16(0);
        for s in 0..Q162 {
            let below = s < 81;
            let t = if below { s + 162 } else { s + 81 };
            let bs = vdupq_n_s16(bias[s] as i16);
            let bt = vdupq_n_s16(bias[t] as i16);
            let (cs, ct) = (&ws.counts[s], &ws.counts[t]);
            for r in 0..16 {
                let x = vsubq_s16(vreinterpretq_s16_u16(cs[r]), bs);
                let y = vsubq_s16(vreinterpretq_s16_u16(ct[r]), bt);
                let (v, sat) = if below {
                    (vsubq_s16(x, y), vqsubq_s16(x, y))
                } else {
                    (vaddq_s16(x, y), vqaddq_s16(x, y))
                };
                overflow = vorrq_s16(overflow, veorq_s16(v, sat));
                let d0 = (core::ptr::addr_of_mut!((*out.add(2 * r)).v) as *mut i16).add(4 * s);
                let d1 = (core::ptr::addr_of_mut!((*out.add(2 * r + 1)).v) as *mut i16).add(4 * s);
                vst1_s16(d0, vget_low_s16(v));
                vst1_s16(d1, vget_high_s16(v));
            }
        }
        assert!(
            vmaxvq_u16(vreinterpretq_u16_s16(overflow)) == 0,
            "a fold coefficient does not fit the wire codec"
        );
        let _ = N;
        claimed
    }

    /// `v = sum_j c_j W_j` for every ring element of the `columns` columns of `column_words`
    pub fn fold(words: &[u128], column_words: usize, challenges: &[Challenge], threads: usize) -> Vec<RingElement> {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::OnceLock;
        assert!(column_words % LANES == 0, "a column is whole blocks");
        assert_eq!(words.len(), challenges.len() * column_words, "one challenge per column");
        assert!(fits(challenges), "the counters would overflow");
        let elements = column_words / 4;
        let blocks = column_words / LANES;
        let ngroups = groups(challenges.len());
        let built: Vec<OnceLock<Group>> = (0..ngroups).map(|_| OnceLock::new()).collect();
        let workers = threads.clamp(1, blocks.max(1));
        let pool = crate::par::pool();
        pool.run(ngroups, |g| {
            let _ = built[g].set(Group::of(challenges, g));
        });
        let groups: Vec<&Group> = built.iter().map(|g| g.get().expect("every group is built")).collect();
        let bias = bias(groups.iter().copied());
        let mut out: Vec<RingElement> = Vec::with_capacity(elements);
        let out_addr = out.as_mut_ptr() as usize;
        let words_addr = words.as_ptr() as usize;
        let next = AtomicUsize::new(0);
        let (groups, bias) = (&groups, &bias);
        crate::par::diag(format!("f0={}", crate::par::stamp()));
        pool.run(workers, |w| {
            let mut ws = Workspace::new();
            let mut claim = || {
                let b = next.fetch_add(1, Ordering::Relaxed);
                (b < blocks).then_some(b)
            };
            let mut block = claim();
            while let Some(b) = block {
                // SAFETY: blocks are disjoint runs of 32 elements of `out`'s capacity.
                block = unsafe {
                    fold_block(
                        words_addr as *const u128,
                        column_words,
                        groups,
                        bias,
                        b,
                        &mut claim,
                        &mut ws,
                        (out_addr as *mut RingElement).add(b * BLOCK_ELEMENTS),
                    )
                };
            }
            crate::par::diag(format!("fw{}={}", w, crate::par::stamp()));
        });
        // SAFETY: every block wrote all fields of its 32 elements.
        unsafe { out.set_len(elements) };
        out
    }

    #[cfg(test)]
    mod tests {
        use super::super::{reference_element, tests_support::*, Challenge, WORD_BITS};
        use super::*;

        #[test]
        fn slice_holds_the_words_bits() {
            let mut x = 0x0123_4567_89AB_CDEFu128;
            let words: Vec<u128> = (0..LANES).map(|_| next(&mut x)).collect();
            let mut planes = vec![unsafe { vdupq_n_u8(0) }; WORD_BITS];
            let mut tmp = planes.clone();
            unsafe { slice(words.as_ptr() as *const u8, planes.as_mut_ptr(), tmp.as_mut_ptr()) };
            for q in 0..WORD_BITS {
                let b: [u8; 16] = unsafe { core::mem::transmute(planes[q]) };
                for l in 0..LANES {
                    assert_eq!((b[l / 8] >> (l % 8)) & 1, ((words[l] >> q) & 1) as u8, "q {q} l {l}");
                }
            }
        }

        #[test]
        fn unslice_reads_the_counts() {
            let mut x = 0xF00D_1234u128;
            let p: [V; 16] = core::array::from_fn(|_| unsafe { core::mem::transmute(next(&mut x)) });
            let got: [[u16; 8]; 16] = unsafe { core::mem::transmute(unslice(p)) };
            let pb: [[u8; 16]; 16] = unsafe { core::mem::transmute(p) };
            for l in 0..LANES {
                let want: u16 = (0..16).map(|w| (((pb[w][l / 8] >> (l % 8)) & 1) as u16) << w).sum();
                assert_eq!(got[l / 8][l % 8], want, "lane {l}");
            }
        }

        fn check(columns: usize, column_words: usize, weights: impl Fn(usize) -> usize, dense: u32, seed: u128) {
            let mut x = seed;
            let words: Vec<u128> = (0..columns * column_words)
                .map(|_| match dense {
                    0 => next(&mut x),
                    1 => u128::MAX,
                    _ => next(&mut x) & next(&mut x) & next(&mut x),
                })
                .collect();
            let challenges: Vec<Challenge> = (0..columns).map(|j| challenge(&mut x, weights(j))).collect();
            let got = fold(&words, column_words, &challenges, 3);
            assert_eq!(got.len(), column_words / 4);
            for m in 0..column_words / 4 {
                let want = reference_element(&words, columns, column_words, &challenges, m);
                assert_eq!(got[m].v[..], want[..], "columns {columns} element {m}");
                assert_eq!(got[m].representation, Representation::Coefficients);
            }
        }

        #[test]
        fn neon_fold_matches_the_reference() {
            check(1, 128, |_| 28, 0, 1);
            check(2, 256, |_| 28, 0, 2);
            check(17, 128, |j| if j % 5 == 0 { 28 } else { 1 + (j % 28) }, 0, 3);
            check(40, 384, |j| [0usize, 1, 2, 28, 7, 32][j % 6], 0, 4);
            check(33, 128, |_| 28, 1, 5);
            check(50, 256, |_| 28, 2, 6);
        }

        #[test]
        fn group_windows_match_brute_force() {
            let mut x = 0xA5A5_5A5Au128;
            let challenges: Vec<Challenge> = (0..2 * GROUP + 5)
                .map(|j| challenge(&mut x, [0usize, 1, 2, 28, 7, 32][j % 6]))
                .collect();
            for g in 0..groups(challenges.len()) {
                let group = Group::of(&challenges, g);
                let (first, end) = (group_first(challenges.len(), g), group_first(challenges.len(), g + 1));
                assert!(end - first <= GROUP && group.first == first && group.columns == end - first);
                for i in 0..PAIRS {
                    for sign in 0..2 {
                        let w = group.win[i];
                        let mut got: Vec<u16> = group.list[sign][w[2 * sign] as usize..w[2 * sign + 1] as usize].to_vec();
                        let mut want = Vec::new();
                        for (jl, c) in challenges[first..end].iter().enumerate() {
                            for t in 0..c.weight {
                                let p = c.positions[t] as i32;
                                let negative = (((c.signs >> t) & 1 == 1) ^ (p % 2 == 1)) as usize;
                                let col = (GUARD + jl * STRIDE + BIAS_PLANES) as i32;
                                for (m, sg) in [(p, negative), (p - SLOTS as i32, 1 - negative)] {
                                    let reaches = (2 * i as i32) <= m + 127 && 2 * i as i32 + 1 >= m;
                                    if sg == sign && reaches && (m >= 0 || p as usize + WORD_BITS > SLOTS) {
                                        want.push((16 * (col - m)) as u16);
                                    }
                                }
                            }
                        }
                        let (lo, hi) = (w[2 * sign] as usize, w[2 * sign + 1] as usize);
                        let full = lo + (hi - lo) / SUB * SUB;
                        let pad = (16 * (BIAS_PLANES + [PAD_ZERO, PAD_ONES][sign]) - 32 * i) as u16;
                        for k in 0..SUB {
                            let e = if full + k < hi { group.list[sign][full + k] } else { pad };
                            assert_eq!(group.tail[i][sign][k], e, "tail of group {g} pair {i} sign {sign}");
                        }
                        got.sort_unstable();
                        want.sort_unstable();
                        assert_eq!(got, want, "group {g} pair {i} sign {sign}");
                    }
                }
            }
        }

        /// Whole groups of one effective sign on all-ones words: a pass's list carries far more
        /// than `h` holds, so the in-loop flush must run; the signs alternate by group so that
        /// the coefficients still fit.
        #[test]
        fn neon_fold_long_lists() {
            let (columns, column_words) = (1024usize, 128usize);
            let words = vec![u128::MAX; columns * column_words];
            let n = groups(columns);
            let challenges: Vec<Challenge> = (0..columns)
                .map(|j| {
                    let g = (0..n).rev().find(|&g| group_first(columns, g) <= j).unwrap();
                    let start = (j * 37) % 130;
                    let mut positions = [0u8; 32];
                    for (t, p) in positions.iter_mut().take(31).enumerate() {
                        *p = (start + t) as u8;
                    }
                    let signs = (0..31).fold(0u32, |a, t| a | (((positions[t] as u32 & 1) ^ (g % 2) as u32) << t));
                    Challenge { positions, signs, weight: 31 }
                })
                .collect();
            let got = fold(&words, column_words, &challenges, 3);
            for m in 0..column_words / 4 {
                let want = reference_element(&words, columns, column_words, &challenges, m);
                assert_eq!(got[m].v[..], want[..], "element {m}");
            }
        }

        #[test]
        fn neon_fold_full_width() {
            // 1024 columns of weight 28, one block, dense and all-ones words.
            check(1024, 128, |_| 28, 0, 7);
            check(1024, 128, |_| 28, 1, 8);
        }
    }
