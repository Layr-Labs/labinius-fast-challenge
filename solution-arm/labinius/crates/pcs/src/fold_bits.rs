//! The fold `v = sum_j c_j W_j` straight from the binary words, with bit-sliced counters.
//!
//! # The algebra, in the verifier's terms
//!
//! A challenge `c_j` is a weight-28 signed subring element with positions `p_t` and signs; it
//! sits in `R_648` as `c_j(-X^4)`, whose coefficient of `X^{4p}` is the sign of position `p`
//! times `(-1)^p`. Ring element `m` of column `j` is binary: coefficient `4q + k` is bit `q` of
//! word `4m + k` of the column, and a word has 128 bits, so `q < 128`. The verifier's reference
//! fold computes, per element `m`,
//!
//! ```text
//!     raw[c] = sum_j sum_t sign_{j,t} * bits_{j,m}[c - 4 p_{j,t}]     (0 <= c < 2 * 648 - 1)
//!     for c = 1295 down to 648: raw[c - 324] += raw[c]; raw[c - 648] -= raw[c]
//! ```
//!
//! Index `c = 4 e + k` keeps the component `k`: contributions to `(e, k)` come from bit
//! `e - p_t` of word `4m + k`, for every `(j, t)` with `0 <= e - p_t < 128`, and `e` ranges
//! over `0 .. 289`. So for a fixed *row* `e` and a fixed lane `(m, k)`, the raw value is a signed
//! count of bits, and the reduction is the same shift-and-add on rows `e`.
//!
//! # The computation
//!
//! Lanes are `(m, k)` pairs, 512 of them per block (128 ring elements, the 512 consecutive words
//! `4 m + k` of a column slice). Each column's 512 words are transposed into 128 bit planes of
//! 512 lanes ([`Transposer`]); then, row by row, every contributing plane is pushed into a
//! carry-save accumulator ([`Pending`]: a binary counter of vectors, one full adder per three
//! vectors of equal weight) and the accumulator is flushed into the row's 16-plane bit-sliced
//! counter once per batch of columns. A negative term `-b` enters as `!b`, since `-b = !b - 1`,
//! and the number of negative terms of the row is subtracted at the end. Counts stay below
//! `28 * 1024 < 2^15`.
//!
//! Everything is XOR, AND and majority on 512-bit vectors: about four operations per term
//! against the hundreds of cycles per ring element a transform costs. [`Lanes`] is the vector
//! type — `__m512i` on x86-64, `[u64; 8]` elsewhere, the same algorithm over both — so the whole
//! fold runs and is tested on any machine.
//!
//! The lane order inside a 64-word tile follows [`word_at`], the byte scramble of the library's
//! GFNI tile transpose, so that transpose can feed this directly; the portable transposer uses
//! the same order.

/// Degree of the ring.
pub const N: usize = 648;
/// Rows of the raw fold: `e = q + p`, `q < 128`, `p < 162`.
pub const ROWS: usize = 289;
/// Bits of a word, the nonzero coefficients of a component.
pub const WORD_BITS: usize = 128;
/// Coefficients of a component after reduction.
pub const Q162: usize = 162;
/// Lanes per block: 512 words, 128 ring elements.
pub const LANES: usize = 512;
/// Ring elements per block.
pub const BLOCK_ELEMENTS: usize = LANES / 4;
/// Planes of the counter: counts are below `2^15`.
pub const COUNTER_PLANES: usize = 16;
/// Columns whose planes are resident together: 16 x 128 planes x 64 bytes = 128 KB.
pub const BATCH: usize = 16;

/// A challenge in the sparse form the library samples: `weight` sorted positions below 162, a
/// sign bit per position (`1` = negative).
#[derive(Clone, Copy, Debug)]
pub struct Challenge {
    pub positions: [u8; 32],
    pub signs: u32,
    pub weight: usize,
}

/// The word at byte position `p` of a transposed 64-word tile (the library's `bitmat::word_at`).
#[inline]
pub fn word_at(p: usize) -> usize {
    (((p >> 5) & 1) << 1) | ((p >> 4) & 1) | ((p & 0xf) << 2)
}

/// The word, within a 512-word block, that lane `l` carries.
#[inline]
pub fn lane_word(l: usize) -> usize {
    64 * (l / 64) + word_at(l % 64)
}

// =============================================================================================
// the vector type
// =============================================================================================

/// 512 lanes of one bit.
pub trait Lanes: Copy {
    fn zero() -> Self;
    fn ones() -> Self;
    /// # Safety
    /// `p` addresses 8 readable `u64`.
    unsafe fn load(p: *const u64) -> Self;
    /// # Safety
    /// `p` addresses 8 writable `u64`.
    unsafe fn store(self, p: *mut u64);
    fn xor(self, o: Self) -> Self;
    fn and(self, o: Self) -> Self;
    /// `(a ^ b ^ c, majority(a, b, c))`: a full adder on every lane.
    fn csa(a: Self, b: Self, c: Self) -> (Self, Self);
    /// The lanes as eight `u64`.
    fn words(self) -> [u64; 8];
    fn from_words(w: [u64; 8]) -> Self;
    /// The 16-plane bit-sliced counter at `planes` (plane `w` at `planes + 8 w`) as 512 `i32`
    /// counts.
    ///
    /// # Safety
    /// `planes` addresses `8 * COUNTER_PLANES` readable `u64`, `out` 512 writable `i32`.
    unsafe fn read_counter(planes: *const u64, out: *mut i32);
}

/// [`Lanes::read_counter`] one lane at a time, for the portable type and the tests.
///
/// # Safety
/// As [`Lanes::read_counter`].
pub unsafe fn read_counter_scalar(planes: *const u64, out: *mut i32) {
    for l in 0..LANES {
        let (word, bit) = (l / 64, l % 64);
        let mut v = 0i32;
        for w in 0..COUNTER_PLANES {
            v |= (((*planes.add(8 * w + word) >> bit) & 1) as i32) << w;
        }
        *out.add(l) = v;
    }
}

#[cfg(target_arch = "x86_64")]
mod vec512 {
    use super::Lanes;
    use core::arch::x86_64::*;

    #[derive(Clone, Copy)]
    pub struct V(pub __m512i);

    impl Lanes for V {
        #[inline(always)]
        fn zero() -> Self {
            V(unsafe { _mm512_setzero_si512() })
        }
        #[inline(always)]
        fn ones() -> Self {
            V(unsafe { _mm512_set1_epi64(-1) })
        }
        #[inline(always)]
        unsafe fn load(p: *const u64) -> Self {
            V(_mm512_loadu_si512(p as *const __m512i))
        }
        #[inline(always)]
        unsafe fn store(self, p: *mut u64) {
            _mm512_storeu_si512(p as *mut __m512i, self.0)
        }
        #[inline(always)]
        fn xor(self, o: Self) -> Self {
            V(unsafe { _mm512_xor_si512(self.0, o.0) })
        }
        #[inline(always)]
        fn and(self, o: Self) -> Self {
            V(unsafe { _mm512_and_si512(self.0, o.0) })
        }
        #[inline(always)]
        fn csa(a: Self, b: Self, c: Self) -> (Self, Self) {
            unsafe {
                (
                    V(_mm512_ternarylogic_epi64::<0x96>(a.0, b.0, c.0)),
                    V(_mm512_ternarylogic_epi64::<0xE8>(a.0, b.0, c.0)),
                )
            }
        }
        #[inline(always)]
        fn words(self) -> [u64; 8] {
            unsafe { core::mem::transmute(self.0) }
        }
        #[inline(always)]
        fn from_words(w: [u64; 8]) -> Self {
            V(unsafe { core::mem::transmute(w) })
        }
        unsafe fn read_counter(planes: *const u64, out: *mut i32) {
            // Sixteen lanes at a time: plane `w`'s bits for those lanes are a 16-bit mask, and
            // a masked broadcast of `1 << w` ORed over the planes rebuilds the counts.
            for g in 0..super::LANES / 16 {
                let (word, shift) = (g / 4, 16 * (g % 4));
                let mut acc = _mm512_setzero_si512();
                for w in 0..super::COUNTER_PLANES {
                    let bits = (*planes.add(8 * w + word) >> shift) as u16;
                    acc = _mm512_mask_or_epi32(acc, bits, acc, _mm512_set1_epi32(1 << w));
                }
                _mm512_storeu_si512(out.add(16 * g) as *mut __m512i, acc);
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
pub use vec512::V;

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
mod vec512 {
    use super::Lanes;

    #[derive(Clone, Copy)]
    pub struct V(pub [u64; 8]);

    impl Lanes for V {
        #[inline(always)]
        fn zero() -> Self {
            V([0; 8])
        }
        #[inline(always)]
        fn ones() -> Self {
            V([!0; 8])
        }
        #[inline(always)]
        unsafe fn load(p: *const u64) -> Self {
            V(core::ptr::read_unaligned(p as *const [u64; 8]))
        }
        #[inline(always)]
        unsafe fn store(self, p: *mut u64) {
            core::ptr::write_unaligned(p as *mut [u64; 8], self.0)
        }
        #[inline(always)]
        fn xor(self, o: Self) -> Self {
            V(core::array::from_fn(|i| self.0[i] ^ o.0[i]))
        }
        #[inline(always)]
        fn and(self, o: Self) -> Self {
            V(core::array::from_fn(|i| self.0[i] & o.0[i]))
        }
        #[inline(always)]
        fn csa(a: Self, b: Self, c: Self) -> (Self, Self) {
            (
                V(core::array::from_fn(|i| a.0[i] ^ b.0[i] ^ c.0[i])),
                V(core::array::from_fn(|i| {
                    (a.0[i] & b.0[i]) | (a.0[i] & c.0[i]) | (b.0[i] & c.0[i])
                })),
            )
        }
        #[inline(always)]
        fn words(self) -> [u64; 8] {
            self.0
        }
        #[inline(always)]
        fn from_words(w: [u64; 8]) -> Self {
            V(w)
        }
        unsafe fn read_counter(planes: *const u64, out: *mut i32) {
            super::read_counter_scalar(planes, out)
        }
    }
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub use vec512::V;

/// NEON: four 128-bit registers per 512-lane vector; the full adder's sum is one EOR3 (SHA3)
/// or two EORs, its majority one BSL on `a ^ b` (where `a` and `b` differ, `c` decides).
#[cfg(target_arch = "aarch64")]
mod neon512 {
    use super::Lanes;
    use core::arch::aarch64::*;

    #[derive(Clone, Copy)]
    pub struct V(pub [uint64x2_t; 4]);

    #[inline(always)]
    fn map2(a: V, b: V, f: impl Fn(uint64x2_t, uint64x2_t) -> uint64x2_t) -> V {
        V([f(a.0[0], b.0[0]), f(a.0[1], b.0[1]), f(a.0[2], b.0[2]), f(a.0[3], b.0[3])])
    }

    #[inline(always)]
    fn xor3(a: uint64x2_t, b: uint64x2_t, c: uint64x2_t) -> uint64x2_t {
        #[cfg(target_feature = "sha3")]
        unsafe {
            veor3q_u64(a, b, c)
        }
        #[cfg(not(target_feature = "sha3"))]
        unsafe {
            veorq_u64(veorq_u64(a, b), c)
        }
    }

    impl Lanes for V {
        #[inline(always)]
        fn zero() -> Self {
            unsafe { V([vdupq_n_u64(0); 4]) }
        }
        #[inline(always)]
        fn ones() -> Self {
            unsafe { V([vdupq_n_u64(!0); 4]) }
        }
        #[inline(always)]
        unsafe fn load(p: *const u64) -> Self {
            let x = vld1q_u64_x4(p);
            V([x.0, x.1, x.2, x.3])
        }
        #[inline(always)]
        unsafe fn store(self, p: *mut u64) {
            vst1q_u64_x4(p, uint64x2x4_t(self.0[0], self.0[1], self.0[2], self.0[3]))
        }
        #[inline(always)]
        fn xor(self, o: Self) -> Self {
            map2(self, o, |a, b| unsafe { veorq_u64(a, b) })
        }
        #[inline(always)]
        fn and(self, o: Self) -> Self {
            map2(self, o, |a, b| unsafe { vandq_u64(a, b) })
        }
        #[inline(always)]
        fn csa(a: Self, b: Self, c: Self) -> (Self, Self) {
            let mut s = a;
            let mut m = a;
            for i in 0..4 {
                let (x, y, z) = (a.0[i], b.0[i], c.0[i]);
                s.0[i] = xor3(x, y, z);
                m.0[i] = unsafe { vbslq_u64(veorq_u64(x, y), z, x) };
            }
            (s, m)
        }
        #[inline(always)]
        fn words(self) -> [u64; 8] {
            unsafe { core::mem::transmute(self.0) }
        }
        #[inline(always)]
        fn from_words(w: [u64; 8]) -> Self {
            V(unsafe { core::mem::transmute(w) })
        }
        /// Sixteen lanes at a time: each plane's 16 bits become 0/-1 bytes (a byte broadcast
        /// by TBL, then a bit test), weighted into a low and a high count byte, zipped to
        /// `u16` and widened to `i32`.
        unsafe fn read_counter(planes: *const u64, out: *mut i32) {
            const _: () = assert!(super::COUNTER_PLANES == 16);
            let idx = vcombine_u8(vdup_n_u8(0), vdup_n_u8(1));
            let bits: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
            let bits = vld1q_u8(bits.as_ptr());
            for g in 0..super::LANES / 16 {
                let (word, shift) = (g / 4, 16 * (g % 4));
                let mut lo = vdupq_n_u8(0);
                let mut hi = vdupq_n_u8(0);
                for w in 0..8 {
                    let b = vreinterpretq_u8_u64(vdupq_n_u64(*planes.add(8 * w + word) >> shift));
                    let m = vtstq_u8(vqtbl1q_u8(b, idx), bits);
                    lo = vorrq_u8(lo, vandq_u8(m, vdupq_n_u8(1 << w)));
                    let b = vreinterpretq_u8_u64(vdupq_n_u64(*planes.add(8 * (w + 8) + word) >> shift));
                    let m = vtstq_u8(vqtbl1q_u8(b, idx), bits);
                    hi = vorrq_u8(hi, vandq_u8(m, vdupq_n_u8(1 << w)));
                }
                let a = vreinterpretq_u16_u8(vzip1q_u8(lo, hi));
                let b = vreinterpretq_u16_u8(vzip2q_u8(lo, hi));
                let o = out.add(16 * g);
                vst1q_s32(o, vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(a))));
                vst1q_s32(o.add(4), vreinterpretq_s32_u32(vmovl_high_u16(a)));
                vst1q_s32(o.add(8), vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(b))));
                vst1q_s32(o.add(12), vreinterpretq_s32_u32(vmovl_high_u16(b)));
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
pub use neon512::V;

// =============================================================================================
// the transpose
// =============================================================================================

/// 512 words into 128 bit planes: `out[q]` holds bit `q` of the words, lane `l` carrying word
/// [`lane_word`]`(l)` of the block, or whatever [`Transposer::lane_word`] says.
pub trait Transposer {
    /// # Safety
    /// `words` addresses 512 readable `u128`.
    unsafe fn planes<L: Lanes>(&self, words: *const u128, out: &mut [L; WORD_BITS]);
    /// The word (within the block) lane `l` carries.
    fn lane_word(&self, l: usize) -> usize {
        lane_word(l)
    }
}

/// The portable transpose: eight 64-word tiles, each a 128 x 64 bit matrix transposed with the
/// classic swap network, lanes in [`word_at`] order.
pub struct Portable;

/// The same with any lane order: lane `l` carries word `map(l)` of the block.
pub struct PortableWith(pub fn(usize) -> usize);

/// # Safety
/// `words` addresses 512 readable `u128`.
pub unsafe fn portable_planes<L: Lanes>(
    words: *const u128,
    out: &mut [L; WORD_BITS],
    map: fn(usize) -> usize,
) {
    let mut planes = [[0u64; 8]; WORD_BITS];
    for tile in 0..8 {
        // Gather the tile's 64 words in lane order, as two 64 x 64 bit matrices.
        let mut lo = [0u64; 64];
        let mut hi = [0u64; 64];
        for p in 0..64 {
            let w = *words.add(map(64 * tile + p));
            lo[p] = w as u64;
            hi[p] = (w >> 64) as u64;
        }
        transpose64(&mut lo);
        transpose64(&mut hi);
        for q in 0..64 {
            planes[q][tile] = lo[q];
            planes[64 + q][tile] = hi[q];
        }
    }
    for (o, p) in out.iter_mut().zip(&planes) {
        *o = L::from_words(*p);
    }
}

impl Transposer for Portable {
    unsafe fn planes<L: Lanes>(&self, words: *const u128, out: &mut [L; WORD_BITS]) {
        portable_planes(words, out, lane_word)
    }
}

impl Transposer for PortableWith {
    unsafe fn planes<L: Lanes>(&self, words: *const u128, out: &mut [L; WORD_BITS]) {
        portable_planes(words, out, self.0)
    }
    fn lane_word(&self, l: usize) -> usize {
        (self.0)(l)
    }
}

/// The lane order the GFNI path produces: the `gf2p8affine` by the identity transposes each
/// qword's 8x8 bits with the byte index reversed, so within every group of eight lanes the
/// order of [`word_at`] runs backwards.
#[inline]
pub fn lane_word_reversed8(l: usize) -> usize {
    64 * (l / 64) + word_at(8 * ((l % 64) / 8) + 7 - (l % 8))
}

/// In-place transpose of a 64 x 64 bit matrix held as 64 rows: afterwards row `b` bit `a` is
/// the old row `a` bit `b`.
pub fn transpose64(m: &mut [u64; 64]) {
    let mut j = 32;
    let mut mask = 0x0000_0000_FFFF_FFFFu64;
    while j != 0 {
        let mut k = 0;
        while k < 64 {
            for i in k..k + j {
                let t = ((m[i] >> j) ^ m[i + j]) & mask;
                m[i] ^= t << j;
                m[i + j] ^= t;
            }
            k += 2 * j;
        }
        j >>= 1;
        mask ^= mask << j;
    }
}

// =============================================================================================
// the accumulator
// =============================================================================================

/// A binary counter of vectors: at most two pending vectors per weight `2^w`; a third of the
/// same weight is combined with them by one full adder, whose carry moves up one weight. Used
/// for the few vectors a row visit leaves over, not for the inputs themselves.
struct Pending<L: Lanes> {
    slots: [[L; 2]; COUNTER_PLANES],
    count: [u8; COUNTER_PLANES],
}

impl<L: Lanes> Pending<L> {
    #[inline(always)]
    fn new() -> Self {
        Pending {
            slots: [[L::zero(); 2]; COUNTER_PLANES],
            count: [0; COUNTER_PLANES],
        }
    }

    /// Add `x` at weight `2^w`.
    #[inline(always)]
    fn push(&mut self, mut x: L, mut w: usize) {
        loop {
            match self.count[w] {
                0 => {
                    self.slots[w][0] = x;
                    self.count[w] = 1;
                    return;
                }
                1 => {
                    self.slots[w][1] = x;
                    self.count[w] = 2;
                    return;
                }
                _ => {
                    let (s, c) = L::csa(self.slots[w][0], self.slots[w][1], x);
                    self.slots[w][0] = s;
                    self.count[w] = 1;
                    x = c;
                    w += 1;
                    debug_assert!(w < COUNTER_PLANES, "the pending sum overflows the counter");
                }
            }
        }
    }

    /// Add everything pending into the 16-plane bit-sliced counter at `planes` (plane `w` at
    /// `planes + 8 w`): the pending vectors are first compressed to one digit per weight (a
    /// full adder per weight over the two slots and the carry from below), and that number is
    /// added to the counter with a ripple of full adders. The two carry chains are independent.
    ///
    /// # Safety
    /// `planes` addresses `8 * COUNTER_PLANES` writable `u64`.
    #[inline(always)]
    unsafe fn flush_into(&self, planes: *mut u64) {
        let mut carry_pending = L::zero();
        let mut carry_add = L::zero();
        for w in 0..COUNTER_PLANES {
            let (a, b) = (self.slots[w][0], self.slots[w][1]);
            let (digit, up) = match self.count[w] {
                0 => (carry_pending, L::zero()),
                1 => (a.xor(carry_pending), a.and(carry_pending)),
                _ => L::csa(a, b, carry_pending),
            };
            carry_pending = up;
            let p = planes.add(8 * w);
            let (sum, out) = L::csa(L::load(p), digit, carry_add);
            sum.store(p);
            carry_add = out;
        }
        // Counts stay below 2^15, so nothing is carried out of the top plane.
    }
}

/// The Harley–Seal state: one vector each of weight 1, 2, 4 and 8.
struct Hs<L: Lanes> {
    ones: L,
    twos: L,
    fours: L,
    eights: L,
}

impl<L: Lanes> Hs<L> {
    #[inline(always)]
    fn new() -> Self {
        Hs {
            ones: L::zero(),
            twos: L::zero(),
            fours: L::zero(),
            eights: L::zero(),
        }
    }

    /// Sixteen inputs through fifteen full adders: the weight-16 carry comes back, the rest
    /// stays in the state.
    #[inline(always)]
    fn add16(&mut self, x: &[L; 16]) -> L {
        let (ones, twos_a) = L::csa(self.ones, x[0], x[1]);
        let (ones, twos_b) = L::csa(ones, x[2], x[3]);
        let (twos, fours_a) = L::csa(self.twos, twos_a, twos_b);
        let (ones, twos_a) = L::csa(ones, x[4], x[5]);
        let (ones, twos_b) = L::csa(ones, x[6], x[7]);
        let (twos, fours_b) = L::csa(twos, twos_a, twos_b);
        let (fours, eights_a) = L::csa(self.fours, fours_a, fours_b);
        let (ones, twos_a) = L::csa(ones, x[8], x[9]);
        let (ones, twos_b) = L::csa(ones, x[10], x[11]);
        let (twos, fours_a) = L::csa(twos, twos_a, twos_b);
        let (ones, twos_a) = L::csa(ones, x[12], x[13]);
        let (ones, twos_b) = L::csa(ones, x[14], x[15]);
        let (twos, fours_b) = L::csa(twos, twos_a, twos_b);
        let (fours, eights_b) = L::csa(fours, fours_a, fours_b);
        let (eights, sixteens) = L::csa(self.eights, eights_a, eights_b);
        self.ones = ones;
        self.twos = twos;
        self.fours = fours;
        self.eights = eights;
        sixteens
    }
}

// =============================================================================================
// one block
// =============================================================================================

/// A column's challenge in the form the row loop reads: per term its position and its sign as a
/// mask (`!0` negative, `0` positive) — the embedding `c(-X^4)`: the sign bit times `(-1)^p`.
#[derive(Clone, Copy)]
struct Terms {
    position: [u8; 32],
    negative: [u8; 32],
}

impl Terms {
    fn of(c: &Challenge) -> Terms {
        let mut t = Terms {
            position: [0; 32],
            negative: [0; 32],
        };
        for i in 0..c.weight {
            let p = c.positions[i];
            assert!((p as usize) < Q162, "a position is below 162");
            if i > 0 {
                assert!(p > t.position[i - 1], "positions are sorted and distinct");
            }
            t.position[i] = p;
            t.negative[i] = (((c.signs >> i) & 1 == 1) ^ (p % 2 == 1)) as u8;
        }
        t
    }
}

/// For every column and row, the terms that reach the row: `t` in `lo .. hi` has
/// `0 <= e - p_t < 128`. Positions are sorted, so the terms form a run.
pub struct Reach {
    columns: usize,
    lo_hi: Vec<[u8; 2]>,
}

impl Reach {
    /// Positions are sorted, so with `e` increasing, `hi(e) = #{t : p_t <= e}` and
    /// `lo(e) = #{t : p_t + 128 <= e}` both advance monotonically: one sweep per column.
    pub fn of(challenges: &[Challenge]) -> Reach {
        let mut lo_hi = vec![[0u8; 2]; challenges.len() * ROWS];
        for (j, c) in challenges.iter().enumerate() {
            let (mut lo, mut hi) = (0usize, 0usize);
            for e in 0..ROWS {
                while hi < c.weight && c.positions[hi] as usize <= e {
                    hi += 1;
                }
                while lo < c.weight && c.positions[lo] as usize + WORD_BITS <= e {
                    lo += 1;
                }
                lo_hi[j * ROWS + e] = [lo.min(hi) as u8, hi as u8];
            }
        }
        Reach {
            columns: challenges.len(),
            lo_hi,
        }
    }

    /// The same table term by term, the oracle of the tests.
    #[cfg(test)]
    fn brute(challenges: &[Challenge]) -> Reach {
        let mut lo_hi = vec![[0u8; 2]; challenges.len() * ROWS];
        for (j, c) in challenges.iter().enumerate() {
            for e in 0..ROWS {
                let mut lo = c.weight;
                let mut hi = 0;
                for t in 0..c.weight {
                    let p = c.positions[t] as usize;
                    if p <= e && e - p < WORD_BITS {
                        lo = lo.min(t);
                        hi = t + 1;
                    }
                }
                lo_hi[j * ROWS + e] = [lo.min(hi) as u8, hi as u8];
            }
        }
        Reach {
            columns: challenges.len(),
            lo_hi,
        }
    }
    #[inline(always)]
    fn of_row(&self, j: usize, e: usize) -> (usize, usize) {
        let r = self.lo_hi[j * ROWS + e];
        (r[0] as usize, r[1] as usize)
    }
}

/// The 16-plane counters of the [`ROWS`] rows of one block, plane-major per row.
struct Counters {
    planes: Vec<u64>,
}

impl Counters {
    fn new() -> Counters {
        Counters {
            planes: vec![0u64; ROWS * COUNTER_PLANES * 8],
        }
    }
    #[inline(always)]
    fn row(&mut self, e: usize) -> *mut u64 {
        unsafe { self.planes.as_mut_ptr().add(e * COUNTER_PLANES * 8) }
    }
    /// The count of row `e`, lane `l`.
    #[allow(dead_code)]
    fn count(&self, e: usize, l: usize) -> u32 {
        let base = e * COUNTER_PLANES * 8;
        let (word, bit) = (l / 64, l % 64);
        let mut v = 0u32;
        for w in 0..COUNTER_PLANES {
            v |= (((self.planes[base + 8 * w + word] >> bit) & 1) as u32) << w;
        }
        v
    }
}

/// The fold of one block: ring elements `block * 128 ..` of the fold, `out[i]` being element
/// `block * 128 + i`. `words` is the whole witness, `columns` columns of `column_words` words;
/// `challenges` one per column, `reach` their [`Reach`].
///
/// # Safety
/// AVX-512 F on x86-64 (the vector type), and `words` holds `columns * column_words` words with
/// `column_words` a multiple of 512 and `block * 512 + 512 <= column_words`.
pub unsafe fn fold_block<L: Lanes, T: Transposer>(
    words: *const u128,
    columns: usize,
    column_words: usize,
    challenges: &[Challenge],
    reach: &Reach,
    block: usize,
    transposer: &T,
    out: &mut [[i16; N]],
) {
    assert_eq!(challenges.len(), columns, "one challenge per column");
    assert_eq!(reach.columns, columns, "the reach is of these challenges");
    assert_eq!(out.len(), BLOCK_ELEMENTS, "one block of elements");
    let terms: Vec<Terms> = challenges.iter().map(Terms::of).collect();
    let mut counters = Counters::new();
    let mut negatives = vec![0u32; ROWS];
    // Each column's planes and their complements, so a negative term reads `!b` straight from
    // memory; `base[i][t]` is the address of term `t`'s plane for row 0, and row `e` reads it
    // `e` planes further on (`e - p_t` is the plane index).
    let mut planes: Vec<[L; 2 * WORD_BITS]> = (0..BATCH).map(|_| [L::zero(); 2 * WORD_BITS]).collect();
    let mut base = [[core::ptr::null::<L>(); 32]; BATCH];
    let mut negs = [0u32; BATCH];
    let mut list = [core::ptr::null::<L>(); 2 * BATCH * 32];
    let zero = L::zero();
    let zeros = [L::zero(); 16];
    for batch_start in (0..columns).step_by(BATCH) {
        let batch = BATCH.min(columns - batch_start);
        for i in 0..batch {
            let column = batch_start + i;
            let plane: &mut [L; 2 * WORD_BITS] = &mut planes[i];
            let (direct, complement) = plane.split_at_mut(WORD_BITS);
            let direct: &mut [L; WORD_BITS] = direct.try_into().unwrap();
            transposer.planes(words.add(column * column_words + block * LANES), direct);
            for q in 0..WORD_BITS {
                complement[q] = direct[q].xor(L::ones());
            }
            let t = &terms[column];
            let first = plane.as_ptr();
            for k in 0..32 {
                // Plane `e - p` of the direct or complemented set: `first + (neg ? 128 : 0) - p`,
                // then `+ e` per row. Positions are below 162 < 2^8; the pointer may point below
                // the array for rows the term does not reach, and is never dereferenced there.
                let offset = t.negative[k] as usize * WORD_BITS;
                base[i][k] = first.wrapping_add(offset).wrapping_sub(t.position[k] as usize);
            }
            negs[i] = (0..32).map(|k| t.negative[k] as u32).sum::<u32>();
            let _ = negs;
        }
        for e in 0..ROWS {
            // The planes reaching row `e`, listed.
            let mut n = 0usize;
            let mut negative = 0u32;
            for i in 0..batch {
                let j = batch_start + i;
                let (lo, hi) = reach.of_row(j, e);
                let t = &terms[j];
                let b = &base[i];
                for k in lo..hi {
                    *list.get_unchecked_mut(n) = b.get_unchecked(k).wrapping_add(e);
                    negative += *t.negative.get_unchecked(k) as u32;
                    n += 1;
                }
            }
            let mut hs = Hs::<L>::new();
            let mut pending = Pending::<L>::new();
            let mut at = 0usize;
            while at + 16 <= n {
                let x: [L; 16] = core::array::from_fn(|i| **list.get_unchecked(at + i));
                pending.push(hs.add16(&x), 4);
                at += 16;
            }
            if at < n {
                let mut x = zeros;
                for i in at..n {
                    x[i - at] = **list.get_unchecked(i);
                }
                pending.push(hs.add16(&x), 4);
            }
            let _ = zero;
            pending.push(hs.ones, 0);
            pending.push(hs.twos, 1);
            pending.push(hs.fours, 2);
            pending.push(hs.eights, 3);
            pending.flush_into(counters.row(e));
            negatives[e] += negative;
        }
    }
    // The raw rows as integers, then the verifier's reduction, then the elements.
    let mut raw = vec![0i32; ROWS * LANES];
    for e in 0..ROWS {
        let n = negatives[e] as i32;
        let row = &mut raw[e * LANES..(e + 1) * LANES];
        L::read_counter(counters.row(e), row.as_mut_ptr());
        for x in row.iter_mut() {
            *x -= n;
        }
    }
    for e in (Q162..ROWS).rev() {
        for l in 0..LANES {
            let x = raw[e * LANES + l];
            raw[(e - 81) * LANES + l] += x;
            raw[(e - Q162) * LANES + l] -= x;
        }
    }
    for l in 0..LANES {
        let word = transposer.lane_word(l);
        let (m, k) = (word / 4, word % 4);
        for q in 0..Q162 {
            out[m][4 * q + k] = i16::try_from(raw[q * LANES + l])
                .expect("a fold coefficient does not fit the wire codec");
        }
    }
}

/// Every block of the fold, serially: `out[m]` for every ring element `m` of the witness's
/// `column_words / 4` elements per column.
///
/// # Safety
/// As [`fold_block`].
pub unsafe fn fold_all<L: Lanes, T: Transposer>(
    words: *const u128,
    columns: usize,
    column_words: usize,
    challenges: &[Challenge],
    transposer: &T,
) -> Vec<[i16; N]> {
    assert!(column_words % LANES == 0, "a column is whole blocks");
    let elements = column_words / 4;
    let reach = Reach::of(challenges);
    let mut out = vec![[0i16; N]; elements];
    for (block, chunk) in out.chunks_mut(BLOCK_ELEMENTS).enumerate() {
        fold_block::<L, T>(words, columns, column_words, challenges, &reach, block, transposer, chunk);
    }
    out
}

// =============================================================================================
// the reference, for the tests and the runtime check
// =============================================================================================

/// The verifier's reference fold of one ring element: `v_m = sum_j c_j W_{j,m}` over the
/// integers, by shifted adds of the words' bits and the ring's reduction.
pub fn reference_element(
    words: &[u128],
    columns: usize,
    column_words: usize,
    challenges: &[Challenge],
    m: usize,
) -> [i16; N] {
    let mut raw = vec![0i32; 2 * N - 1];
    let mut bits = [0i32; N];
    for (j, c) in challenges.iter().enumerate() {
        let packed = &words[j * column_words + 4 * m..][..4];
        for (k, x) in packed.iter().enumerate() {
            for p in 0..N / 4 {
                bits[4 * p + k] = if p < WORD_BITS { ((x >> p) & 1) as i32 } else { 0 };
            }
        }
        for t in 0..c.weight {
            let position = c.positions[t] as usize;
            let negative = ((c.signs >> t) & 1 == 1) ^ (position % 2 == 1);
            let window = &mut raw[4 * position..4 * position + N];
            if negative {
                for (slot, &b) in window.iter_mut().zip(&bits) {
                    *slot -= b;
                }
            } else {
                for (slot, &b) in window.iter_mut().zip(&bits) {
                    *slot += b;
                }
            }
        }
    }
    for k in (N..2 * N - 1).rev() {
        let value = raw[k];
        raw[k] = 0;
        raw[k - N / 2] += value;
        raw[k - N] -= value;
    }
    let mut element = [0i16; N];
    for (slot, &value) in element.iter_mut().zip(&raw[..N]) {
        *slot = i16::try_from(value).expect("a fold coefficient does not fit the wire codec");
    }
    let _ = columns;
    element
}

#[cfg(test)]
pub(crate) mod tests_support {
    use super::*;

    pub fn next(x: &mut u128) -> u128 {
        *x ^= *x << 13;
        *x ^= *x >> 7;
        *x ^= *x << 17;
        x.wrapping_mul(0x2545F4914F6CDD1Du128) ^ (*x >> 64)
    }

    pub fn challenge(x: &mut u128, weight: usize) -> Challenge {
        let mut taken = [false; Q162];
        let mut positions = Vec::new();
        while positions.len() < weight {
            let p = (next(x) % Q162 as u128) as u8;
            if !taken[p as usize] {
                taken[p as usize] = true;
                positions.push(p);
            }
        }
        positions.sort_unstable();
        let mut c = Challenge {
            positions: [0; 32],
            signs: next(x) as u32 & (((1u64 << weight) - 1) as u32),
            weight,
        };
        c.positions[..weight].copy_from_slice(&positions);
        c
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    use super::tests_support::*;

    #[test]
    fn transpose64_is_a_transpose() {
        let mut x = 0x9E3779B97F4A7C15u128;
        let m: [u64; 64] = core::array::from_fn(|_| next(&mut x) as u64);
        let mut t = m;
        transpose64(&mut t);
        for a in 0..64 {
            for b in 0..64 {
                assert_eq!((t[b] >> a) & 1, (m[a] >> b) & 1, "({a}, {b})");
            }
        }
        transpose64(&mut t);
        assert_eq!(t, m);
    }

    #[test]
    fn portable_planes_hold_the_words_bits() {
        let mut x = 0x0123_4567_89AB_CDEFu128;
        let words: Vec<u128> = (0..LANES).map(|_| next(&mut x)).collect();
        let mut planes = [V::zero(); WORD_BITS];
        unsafe { Portable.planes(words.as_ptr(), &mut planes) };
        for q in 0..WORD_BITS {
            let w = planes[q].words();
            for l in 0..LANES {
                let bit = (w[l / 64] >> (l % 64)) & 1;
                assert_eq!(bit, ((words[lane_word(l)] >> q) & 1) as u64, "q {q} l {l}");
            }
        }
    }

    #[test]
    fn pending_counts_exactly() {
        let mut x = 0x2545F4914F6CDD1Du128;
        for n in [0usize, 1, 2, 3, 5, 17, 96, 300, 1000] {
            let inputs: Vec<[u64; 8]> = (0..n).map(|_| core::array::from_fn(|_| next(&mut x) as u64)).collect();
            let mut planes = [0u64; COUNTER_PLANES * 8];
            // Pre-load the counter with something, to exercise the ripple.
            for (i, p) in planes.iter_mut().enumerate() {
                *p = if i < 8 { next(&mut x) as u64 } else { 0 };
            }
            let before: Vec<u32> = (0..512)
                .map(|l| (0..COUNTER_PLANES).map(|w| (((planes[8 * w + l / 64] >> (l % 64)) & 1) as u32) << w).sum())
                .collect();
            let mut pending = Pending::<V>::new();
            for inp in &inputs {
                pending.push(V::from_words(*inp), 0);
            }
            unsafe { pending.flush_into(planes.as_mut_ptr()) };
            for l in 0..512 {
                let want: u32 = before[l]
                    + inputs.iter().map(|inp| ((inp[l / 64] >> (l % 64)) & 1) as u32).sum::<u32>();
                let got: u32 = (0..COUNTER_PLANES)
                    .map(|w| (((planes[8 * w + l / 64] >> (l % 64)) & 1) as u32) << w)
                    .sum();
                assert_eq!(got, want, "n {n} lane {l}");
            }
        }
    }

    #[test]
    fn any_lane_order_gives_the_same_fold() {
        let mut x = 0x5151_2323_7777u128;
        let (columns, column_words) = (9usize, 512usize);
        let words: Vec<u128> = (0..columns * column_words).map(|_| next(&mut x)).collect();
        let challenges: Vec<Challenge> = (0..columns).map(|_| challenge(&mut x, 28)).collect();
        let reach = Reach::of(&challenges);
        let mut a = vec![[0i16; N]; BLOCK_ELEMENTS];
        let mut b = vec![[0i16; N]; BLOCK_ELEMENTS];
        unsafe {
            fold_block::<V, Portable>(words.as_ptr(), columns, column_words, &challenges, &reach, 0, &Portable, &mut a);
            let t = PortableWith(lane_word_reversed8);
            fold_block::<V, PortableWith>(words.as_ptr(), columns, column_words, &challenges, &reach, 0, &t, &mut b);
        }
        assert_eq!(a, b);
        assert_eq!(a[3][..], reference_element(&words, columns, column_words, &challenges, 3)[..]);
    }

    #[test]
    fn reach_sweep_matches_brute_force() {
        let mut x = 0xA5A5_5A5Au128;
        let challenges: Vec<Challenge> = (0..40)
            .map(|j| challenge(&mut x, [0usize, 1, 2, 28, 7][j % 5]))
            .collect();
        // Both spell an empty run as some `lo >= hi`; compare the runs they denote.
        let canon = |r: &Reach| -> Vec<(usize, usize)> {
            r.lo_hi
                .iter()
                .map(|&[lo, hi]| if lo < hi { (lo as usize, hi as usize) } else { (0, 0) })
                .collect()
        };
        assert_eq!(canon(&Reach::of(&challenges)), canon(&Reach::brute(&challenges)));
    }

    #[test]
    fn counter_read_out_matches_the_scalar_one() {
        let mut x = 0xF00D_1234u128;
        let planes: Vec<u64> = (0..COUNTER_PLANES * 8).map(|_| next(&mut x) as u64).collect();
        let mut a = [0i32; LANES];
        let mut b = [0i32; LANES];
        unsafe {
            read_counter_scalar(planes.as_ptr(), a.as_mut_ptr());
            V::read_counter(planes.as_ptr(), b.as_mut_ptr());
        }
        assert_eq!(a[..], b[..]);
    }

    #[test]
    fn bit_sliced_fold_matches_the_reference() {
        let mut x = 0xDEAD_BEEF_CAFE_F00Du128;
        for (columns, column_words) in [(2usize, 512usize), (6, 1024), (17, 512), (40, 1536)] {
            let words: Vec<u128> = (0..columns * column_words).map(|_| next(&mut x)).collect();
            let challenges: Vec<Challenge> = (0..columns)
                .map(|j| challenge(&mut x, if j % 5 == 0 { 28 } else { 1 + (j % 28) }))
                .collect();
            let got = unsafe { fold_all::<V, Portable>(words.as_ptr(), columns, column_words, &challenges, &Portable) };
            let elements = column_words / 4;
            assert_eq!(got.len(), elements);
            for m in 0..elements {
                let want = reference_element(&words, columns, column_words, &challenges, m);
                assert_eq!(got[m][..], want[..], "columns {columns} element {m}");
            }
        }
    }
}

// =============================================================================================
// the NEON kernel: 128-lane planes, negacyclic slots, slot pairs
// =============================================================================================

/// The fold on NEON, tuned for Neoverse V2 (four 128-bit ALU pipes; `EOR3` only on one).
///
/// # The algebra
///
/// Per component, `v = sum_j c_j W_j` lives in `Z[Y] / (Y^162 - Y^81 + 1)`; since
/// `(Y^81 + 1)(Y^162 - Y^81 + 1) = Y^243 + 1`, the convolution is first taken in
/// `Z[Y] / (Y^243 + 1)` (243 *slots*: a term at position `p` adds bit `q` of the word at slot
/// `p + q`, and with the opposite sign at slot `p + q - 243` past the end) and then reduced:
/// output slot `s < 81` is `slot s - slot (s + 162)`, `81 <= s < 162` is `slot s + slot (s + 81)`.
///
/// # The computation
///
/// A *block* is 128 consecutive words of every column (32 ring elements); each column's 128
/// words are transposed into 128 bit planes of 128 lanes, one `q` register each, lane `l`
/// carrying word `l` of the block. Columns go in groups of [`GROUP`]; a group's planes sit in a
/// buffer with a zero guard plane after every column (and two in front), so a slot *pair*
/// `(2i, 2i + 1)` reads two adjacent planes per term — the one an offset names and the next —
/// and a term that reaches only one slot of the pair reads a guard for the other.
///
/// For every pair and every group the terms reaching the pair are a window of the group's
/// terms sorted by position (per sign), so the schedule is one sorted `u16` offset list per
/// sign and group plus per-pair `[lo, hi)` windows; the base pointer advances 32 bytes per pair.
/// The entries are summed by a 32-input carry-save tree per slot (full adders of three plain
/// ops: `t = a ^ b`, `s ^= t`, carry `BSL(t, s, a)`), the five low counter bits staying in
/// registers for the pair's pass and the tree's weight-32 carries counted in a three-bit
/// register counter that is added to the ten upper counter bits in memory every seven trees.
///
/// A negative term adds the complement of its plane, `!b = 1 - b`, and its pair's *bias*
/// takes one. The complement is never formed: a full adder of complemented inputs `!a`, `!b`
/// has the same sum, and the complement of its carry is `BSL(t, s', a)` (with `s' = s ^ t`),
/// three ops again, so a negative tree runs on the stored planes and passes complemented
/// carries up; its top carry is complemented once.
#[cfg(target_arch = "aarch64")]
pub mod neon {
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
    /// Most columns per group: the groups split the columns as evenly as they can.
    pub const GROUP: usize = 25;
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
    const LO: usize = 5;
    /// Counter bits above them, in memory: counts stay below `2^15`.
    const HI: usize = 10;
    /// Planes added to every offset so that it is non-negative (positions are below 162).
    const BIAS_PLANES: usize = 160;
    const _: () = assert!(16 * (GUARD + (GROUP - 1) * STRIDE + BIAS_PLANES + WORD_BITS) <= u16::MAX as usize);
    const _: () = assert!(16 * (BIAS_PLANES + PLANES) <= u16::MAX as usize);
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
        list: [Vec<u16>; 2],
        /// Per pair: positive `[lo, hi)`, negative `[lo, hi)`.
        win: Vec<[u16; 4]>,
        /// Per pair and sign, the last `len % SUB` entries of the window, then entries of the
        /// pad planes (zero for positive, all-ones for negative: neither adds anything), as
        /// offsets from that pair's base.
        tail: Vec<[[u16; SUB]; 2]>,
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
            let each = |f: &mut dyn FnMut(usize, usize, u16)| {
                for (jl, c) in cs.iter().enumerate() {
                    assert!(c.weight <= c.positions.len());
                    let col = (GUARD + jl * STRIDE + BIAS_PLANES) as i32;
                    for t in 0..c.weight {
                        let p = c.positions[t] as usize;
                        assert!(p < Q162, "a position is below 162");
                        let negative = ((c.signs >> t) & 1 == 1) ^ (p % 2 == 1);
                        let m = p as i32;
                        f(negative as usize, (m + KEY0) as usize, (16 * (col - m)) as u16);
                        // Slots `p + q >= 243` wrap to `p + q - 243` with the other sign: a
                        // term at position `p - 243`.
                        if p + WORD_BITS > SLOTS {
                            let m = m - SLOTS as i32;
                            f(!negative as usize, (m + KEY0) as usize, (16 * (col - m)) as u16);
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
            let mut list = [vec![0u16; start[0][KEYS] as usize], vec![0u16; start[1][KEYS] as usize]];
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
                        let pad = (16 * (BIAS_PLANES + pad) - 32 * i) as u16;
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

    /// A full adder of `s` and two planes: the new sum in `s`, the carry returned. With `NEG`
    /// the planes stand for their complements and the *complement* of the carry is returned.
    ///
    /// Three plain ops in an asm block: `t = a ^ b` (into `b`), the carry `BIT(a, s, t)` (into
    /// `a`, so no register copies) and `s ^= t`. Written as intrinsics, LLVM reassociates the
    /// chain of `s ^= t` into `EOR3`s, which Neoverse V2 issues on one pipe only.
    #[inline(always)]
    unsafe fn fa<const NEG: bool>(s: &mut V, mut a: V, b: V) -> V {
        if NEG {
            core::arch::asm!(
                "eor {b:v}.16b, {a:v}.16b, {b:v}.16b",
                "eor {s:v}.16b, {s:v}.16b, {b:v}.16b",
                "bit {a:v}.16b, {s:v}.16b, {b:v}.16b",
                a = inout(vreg) a, b = inout(vreg) b => _, s = inout(vreg) *s,
                options(pure, nomem, nostack, preserves_flags)
            );
        } else {
            core::arch::asm!(
                "eor {b:v}.16b, {a:v}.16b, {b:v}.16b",
                "bit {a:v}.16b, {s:v}.16b, {b:v}.16b",
                "eor {s:v}.16b, {s:v}.16b, {b:v}.16b",
                a = inout(vreg) a, b = inout(vreg) b => _, s = inout(vreg) *s,
                options(pure, nomem, nostack, preserves_flags)
            );
        }
        a
    }

    /// Bits of the register counter `h` above the `LO` bits (weights `2^5 .. 2^8`), added into
    /// the memory bits at the end of a pass, or within it once the trees have added
    /// `H_MAX - 1` carries.
    const HB: usize = 4;
    /// Carries of weight 32 that `h` holds.
    const H_MAX: usize = (1 << HB) - 1;

    /// `hi += h`: the four-bit counter `h` (weights `2^5 .. 2^8`) into the ten upper counter
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
        pos: *const u16,
        pos8: usize,
        neg: *const u16,
        neg8: usize,
        tail: &[[u16; SUB]; 2],
        lo: *mut V,
        hi: *mut V,
    ) {
        let bo = be.add(16);
        let z = vdupq_n_u8(0);
        let ones = vdupq_n_u8(0xFF);
        let mut s0: [V; LO] = core::array::from_fn(|b| *lo.add(b));
        let mut s1: [V; LO] = core::array::from_fn(|b| *lo.add(LO + b));
        let mut h0 = [z; HB];
        let mut h1 = [z; HB];
        // Carries in `h`: the trees' carries flush it at `H_MAX - 1`, which leaves room for
        // the one carry of the list's end.
        let mut hn = 0usize;
        let mut t: *const u16;
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
                let o0 = *t.add($off) as usize;
                let o1 = *t.add($off + 1) as usize;
                let a0 = vld1q_u8(be.add(o0));
                let a1 = vld1q_u8(bo.add(o0));
                let b0 = vld1q_u8(be.add(o1));
                let b1 = vld1q_u8(bo.add(o1));
                (fa::<$neg>(&mut s0[0], a0, b0), fa::<$neg>(&mut s1[0], a1, b1))
            }};
        }
        macro_rules! level {
            ($bit:literal, $lower:ident, $off:expr, $neg:literal) => {{
                let (x0, x1) = $lower!($off, $neg);
                let (y0, y1) = $lower!($off + (1 << $bit), $neg);
                (fa::<$neg>(&mut s0[$bit], x0, y0), fa::<$neg>(&mut s1[$bit], x1, y1))
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
        macro_rules! entries {
            ($list:expr, $n8:expr, $tail:expr, $neg:literal) => {{
                let none = if $neg { ones } else { z };
                let (mut p30, mut p31) = (none, none);
                let (mut p40, mut p41) = (none, none);
                t = $list;
                // Whole 32-entry trees, one branch each.
                for _ in 0..$n8 / 4 {
                    let (a0, a1) = eight!(0, $neg);
                    let (b0, b1) = eight!(SUB, $neg);
                    let d0 = fa::<$neg>(&mut s0[3], a0, b0);
                    let d1 = fa::<$neg>(&mut s1[3], a1, b1);
                    let (a0, a1) = eight!(2 * SUB, $neg);
                    let (b0, b1) = eight!(3 * SUB, $neg);
                    let f0 = fa::<$neg>(&mut s0[3], a0, b0);
                    let f1 = fa::<$neg>(&mut s1[3], a1, b1);
                    let e0 = fa::<$neg>(&mut s0[4], d0, f0);
                    let e1 = fa::<$neg>(&mut s1[4], d1, f1);
                    add_h!(fix!(e0, $neg), fix!(e1, $neg));
                    t = t.add(4 * SUB);
                    hn += 1;
                    if hn >= H_MAX - 1 {
                        flush(hi, &h0);
                        flush(hi.add(HI), &h1);
                        h0 = [z; HB];
                        h1 = [z; HB];
                        hn = 0;
                    }
                }
                // Up to three more subtrees of eight.
                let left = $n8 % 4;
                if left != 0 {
                    (p30, p31) = eight!(0, $neg);
                    t = t.add(SUB);
                    if left != 1 {
                        let (c0, c1) = eight!(0, $neg);
                        t = t.add(SUB);
                        p40 = fa::<$neg>(&mut s0[3], p30, c0);
                        p41 = fa::<$neg>(&mut s1[3], p31, c1);
                        (p30, p31) = (none, none);
                        if left != 2 {
                            (p30, p31) = eight!(0, $neg);
                        }
                    }
                }
                // The tail, and whatever waits at bits 3 and 4, into `h`.
                t = $tail.as_ptr();
                let (c0, c1) = eight!(0, $neg);
                let d0 = fa::<$neg>(&mut s0[3], p30, c0);
                let d1 = fa::<$neg>(&mut s1[3], p31, c1);
                let e0 = fa::<$neg>(&mut s0[4], p40, d0);
                let e1 = fa::<$neg>(&mut s1[4], p41, d1);
                add_h!(fix!(e0, $neg), fix!(e1, $neg));
                hn += 1;
            }};
        }
        entries!(pos, pos8, tail[0], false);
        entries!(neg, neg8, tail[1], true);
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
        pool.run(workers, |_| {
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
}
