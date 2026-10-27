//! Commitment modulo 9721 * 2^22 with a single witness NTT.
//!
//! `t = sum_j A_j s_j` is wanted modulo `Q = 9721 * 2^22`. The witness transform modulo 9721 feeds
//! nine accumulation streams that differ only in their public key:
//!
//! - stream 0: `A mod 9721`, accumulated over the whole column (the ordinary residue);
//! - streams 1..=8: the balanced base-8 digits `L_d` of `A mod 2^22` (`L_0..L_6` in `[-4, 4)`,
//!   `L_7` the remaining bit), so that `A ≡ sum_d 8^d L_d (mod 2^22)`.
//!
//! A digit stream's sum `X = sum_{j in chunk} L_{d,j} s_j` over a chunk of [`CHUNK_PAIRS`] pair
//! batches is an integer whose mean `mu = (1/2) sum_j L_{d,j} U` (`U` the all-ones polynomial on
//! the 512 witness coefficients) is public. Its deviation from `mu` is a sum of independent bounded
//! terms of the uniformly random table bits: Hoeffding bounds it by `Z sigma` except with
//! probability `2 exp(-Z^2 / 2)`, where `sigma^2 = (1/4) sum_j sum_i M(L_{d,j})[m][i]^2` is
//! computed from the public key ([`SmsKeys::build`] records the smallest margin `Z` over every
//! chunk, digit and coefficient; at this shape it is above 12). Within that margin the residue of
//! `X` modulo 9721 determines `X`: `X = mu + centered(X - mu mod 9721)`. The chunk's residues come
//! out of one inverse transform per digit for two chunks at once, and `t mod 2^22 = sum_d 8^d X_d`
//! accumulates in wrapping 32-bit lanes.
//!
//! Only public parameters are precomputed. Every invocation consumes its complete fresh table.
use super::*;
use crate::simd::ntt::gen_small;

pub const P: u16 = 9721;
pub const TAIL_BITS: u32 = 22;
pub const MODULUS: u128 = (P as u128) << TAIL_BITS;
/// Accumulation streams: the base residue and eight digits.
pub const STREAMS: usize = 9;
pub const DIGITS: usize = STREAMS - 1;
/// Bit position of digit `d`.
pub const fn digit_shift(d: usize) -> u32 {
    3 * d as u32
}
/// Pair batches (two ring elements of each of 16 columns) per reconstruction chunk: 128 ring
/// elements of a column.
pub const CHUNK_PAIRS: usize = 128;
/// Digits reconstructed per chunk; the last digit, a single bit with a tenth of the second moment,
/// is reconstructed per half column (both halves in one inverse transform).
pub const CHUNKED: usize = DIGITS - 1;
/// Pair batches transformed in lockstep per accumulator read-modify-write.
pub const LOCK: usize = 8;
const TAIL_MASK: u32 = (1u32 << TAIL_BITS) - 1;

/// Centered `2^16 mod q`, the weight [`reduce_vec_c`] gives the high half.
const fn r16c(q: u16) -> i32 {
    (65536 % q as i32) - q as i32
}
const fn acc_after_reduce_c(q: u16) -> i64 {
    32768 * (1 + r16c(q).unsigned_abs() as i64)
}
/// One dword-product step: two products of a lazily reduced transform lane and a centered key.
const fn pair_per_batch(q: u16) -> i64 {
    2 * w_bound(q) * a_bound(q)
}
/// Pair batches between two fold-backs: the largest power of two `P` with
/// `acc_after_reduce_c + P * pair_per_batch <= i32::MAX`.
const fn pair_period(q: u16) -> usize {
    let mut p = 1usize;
    while acc_after_reduce_c(q) + 2 * (p as i64) * pair_per_batch(q) <= i32::MAX as i64 {
        p *= 2;
    }
    p
}
/// [`red16`]'s bias: the largest multiple of q below 2^31.
const fn bias(q: u16) -> i64 {
    ((1i64 << 31) / q as i64) * q as i64
}
pub const PP: usize = pair_period(P);
const _: () = {
    assert!(acc_after_reduce_c(P) + (PP as i64) * pair_per_batch(P) <= bias(P));
    assert!(CHUNK_PAIRS % PP == 0 && (PP % LOCK == 0 || LOCK % PP == 0) && CHUNK_PAIRS % LOCK == 0 && LOCK <= 2 * PP);
    assert!(2 * (P as i32) <= gen_small::TwI::<P>::IN_BOUND);
};

const fn qinv16(q: u16) -> u32 {
    let q = q as u32;
    let mut x = q;
    let mut i = 0;
    while i < 4 {
        x = x.wrapping_mul(2u32.wrapping_sub(q.wrapping_mul(x))) & 0xffff;
        i += 1;
    }
    x
}
const fn red16_w(q: u16) -> (i32, i32) {
    let wl = (1u32 << 16) % q as u32;
    let wh = ((1u64 << 32) % q as u64) as u32;
    let qi = qinv16(q);
    (
        (wl | (wh << 16)) as i32,
        (((wl * qi) & 0xffff) | (((wh * qi) & 0xffff) << 16)) as i32,
    )
}
/// A representative of `x mod q` with `|r| < 2q`, lane-wise, for `|x| <= bias(q)`.
#[inline(always)]
unsafe fn red16<const Q: u16>(x: __m512i) -> __m512i {
    let (w, wp) = red16_w(Q);
    let y = _mm512_add_epi32(x, _mm512_set1_epi32(bias(Q) as i32));
    let m = _mm512_mullo_epi16(y, _mm512_set1_epi32(wp));
    let r = _mm512_sub_epi16(
        _mm512_mulhi_epu16(y, _mm512_set1_epi32(w)),
        _mm512_mulhi_epu16(m, _mm512_set1_epi16(Q as i16)),
    );
    _mm512_madd_epi16(r, _mm512_set1_epi16(1))
}
/// Canonical representative in `[0, Q)` via [`red16`].
#[inline(always)]
unsafe fn canon_red16<const Q: u16>(x: __m512i) -> __m512i {
    let r = red16::<Q>(x);
    let q = _mm512_set1_epi32(Q as i32);
    let q2 = _mm512_set1_epi32(2 * Q as i32);
    let r = _mm512_mask_add_epi32(r, _mm512_cmp_epi32_mask(r, _mm512_setzero_si512(), _MM_CMPINT_LT), r, q2);
    _mm512_mask_sub_epi32(r, _mm512_cmp_epi32_mask(r, q, _MM_CMPINT_NLT), r, q)
}
fn zeroed_slice<T>(n: usize) -> Box<[T]> {
    unsafe {
        let l = std::alloc::Layout::array::<T>(n).unwrap();
        let p = std::alloc::alloc_zeroed(l) as *mut T;
        if p.is_null() {
            std::alloc::handle_alloc_error(l)
        }
        Box::from_raw(core::ptr::slice_from_raw_parts_mut(p, n))
    }
}

// =============================================================================================
// public parameters
// =============================================================================================

/// Words of the pair key per lockstep step: one dword per pair batch per slot and stream.
pub const LOCK_WORDS: usize = N * STREAMS * LOCK;

/// The public data the commitment reads: the keys of all streams in the lockstep layout, the
/// digit chunks' centered mean residues, and the total of the means.
pub struct SmsKeys<'a> {
    /// `pairs[(l * N + j) * STREAMS * LOCK + LOCK s + h]`: for lockstep step `l` (ring elements
    /// `2 LOCK l .. 2 LOCK (l + 1)` of every column), slot `j` and stream `s`, the dword of the
    /// stream's NTT-domain keys of ring elements `2 LOCK l + 2h` (low half) and `+ 1` (high half),
    /// centered.
    pub pairs: &'a [u32],
    /// `mur[(d * chunks + c) * N + m]`: the chunk mean `mu'` of digit `d < CHUNKED`, chunk `c`,
    /// coefficient `m`, as a centered residue modulo 9721; then the half-column means of the last
    /// digit at `mur[(CHUNKED * chunks + h) * N + m]`, `h` the half.
    pub mur: &'a [i16],
    /// `mu[m] = sum_{d, c} 8^d mu'_{d,c}[m]`, wrapping.
    pub mu: &'a [u32],
    /// Ring elements per column.
    pub elements: usize,
}

/// Owned [`SmsKeys`] together with the statistical margin they were certified at.
pub struct SmsKeyData {
    pub pairs: Vec<u32>,
    pub mur: Vec<i16>,
    pub mu: Vec<u32>,
    pub elements: usize,
    /// The smallest `((P - 1) / 2) / sigma` over every chunk, digit and coefficient.
    pub margin: f64,
    /// The same per digit, and where each digit's minimum is (chunk, coefficient, |mean|).
    pub margins: [(f64, usize, usize, i64); DIGITS],
}

impl SmsKeyData {
    pub fn keys(&self) -> SmsKeys<'_> {
        SmsKeys { pairs: &self.pairs, mur: &self.mur, mu: &self.mu, elements: self.elements }
    }
}

/// The balanced digits of `x mod 2^22`: `x ≡ sum_d 8^d L_d`, `L_0..L_6 in [-4, 4)`, `L_7 in {0, 1}`.
pub fn digits(x: u128) -> [i8; DIGITS] {
    let mut r = (x & TAIL_MASK as u128) as i64;
    let mut out = [0i8; DIGITS];
    for d in 0..DIGITS - 1 {
        let v = ((r + 4) & 7) - 4;
        out[d] = v as i8;
        r = (r - v) >> 3;
    }
    out[DIGITS - 1] = (r & 1) as i8;
    out
}

/// `sum_i M(L)[m][i]^k` over the 512 witness coefficients `i` for `k = 1` and `k = 2`, where
/// `M(L)[m][i]` is coefficient `m` of `X^i L mod X^648 - X^324 + 1`. For `m >= 324` it is
/// `L[m - i] + L[m + 324 - i]`, for `m < 324` it is `L[m - i] - L[m + 648 - i] - L[m + 972 - i]`
/// (`L` zero outside `0..648`): each a function of `x = m - i` alone, so both sums are windows of
/// a prefix sum.
fn moment_rows(l: &[i8; N], s1: &mut [i64; N], s2: &mut [i64; N]) {
    let at = |x: i64| -> i64 { if (0..N as i64).contains(&x) { l[x as usize] as i64 } else { 0 } };
    // x runs over -511 ..= 647; index x + 511.
    const W: usize = 512 + N - 1;
    let mut fh = [0i64; W + 1];
    let mut fh2 = [0i64; W + 1];
    let mut fl = [0i64; W + 1];
    let mut fl2 = [0i64; W + 1];
    for k in 0..W {
        let x = k as i64 - 511;
        let h = at(x) + at(x + 324);
        let g = at(x) - at(x + 648) - at(x + 972);
        fh[k + 1] = fh[k] + h;
        fh2[k + 1] = fh2[k] + h * h;
        fl[k + 1] = fl[k] + g;
        fl2[k + 1] = fl2[k] + g * g;
    }
    for m in 0..N {
        // x from m - 511 to m: indices m .. m + 512
        let (a, b) = (m, m + 512);
        if m >= 324 {
            s1[m] += fh[b] - fh[a];
            s2[m] += fh2[b] - fh2[a];
        } else {
            s1[m] += fl[b] - fl[a];
            s2[m] += fl2[b] - fl2[a];
        }
    }
}

impl SmsKeyData {
    /// The keys of the matrix `a` (`elements * 648` coefficients modulo [`MODULUS`]).
    pub fn build(a: &[u128], elements: usize) -> SmsKeyData {
        use crate::key::CommitmentKey;
        use crate::ring::Modulus;
        assert_eq!(a.len(), elements * N);
        assert!(elements % (2 * CHUNK_PAIRS) == 0, "whole chunks of ring elements");
        let base = Modulus::ALL.into_iter().find(|m| m.prime() == P).unwrap();
        let len_f162 = 4 * elements;
        let dig: Vec<[i8; DIGITS]> = a.iter().map(|&x| digits(x)).collect();
        let mut keys = Vec::with_capacity(STREAMS);
        keys.push(CommitmentKey::from_coefficients(len_f162, a, MODULUS, base, &[]));
        for d in 0..DIGITS {
            let c: Vec<u128> = dig.iter().map(|v| (v[d] as i64).rem_euclid(P as i64) as u128).collect();
            keys.push(CommitmentKey::from_coefficients(len_f162, &c, P as u128, base, &[]));
        }
        let steps = elements / (2 * LOCK);
        let mut pairs = vec![0u32; steps * LOCK_WORDS];
        for l in 0..steps {
            let j0 = 2 * LOCK * l;
            let (b, lane) = (j0 / 32, j0 % 32);
            for (s, key) in keys.iter().enumerate() {
                let batch = &key.row(0)[b];
                for j in 0..N {
                    for h in 0..LOCK {
                        let lo = batch.v[j][lane + 2 * h] as u16 as u32;
                        let hi = batch.v[j][lane + 2 * h + 1] as u16 as u32;
                        pairs[(l * N + j) * STREAMS * LOCK + LOCK * s + h] = lo | (hi << 16);
                    }
                }
            }
        }
        // Chunk moments: ring elements 2 CHUNK_PAIRS c .. of every column.
        let chunks = elements / (2 * CHUNK_PAIRS);
        let mut mur = vec![0i16; (CHUNKED * chunks + 2) * N];
        let mut mu = vec![0u32; N];
        let mut margin = f64::INFINITY;
        let mut margins = [(f64::INFINITY, 0usize, 0usize, 0i64); DIGITS];
        let half = ((P - 1) / 2) as f64;
        for d in 0..DIGITS {
            let (per, nc, at) = if d < CHUNKED { (2 * CHUNK_PAIRS, chunks, d * chunks) } else { (elements / 2, 2, CHUNKED * chunks) };
            for c in 0..nc {
                let mut s1 = [0i64; N];
                let mut s2 = [0i64; N];
                for j in c * per..(c + 1) * per {
                    let l: [i8; N] = core::array::from_fn(|m| dig[j * N + m][d]);
                    moment_rows(&l, &mut s1, &mut s2);
                }
                for m in 0..N {
                    // mean s1 / 2, variance s2 / 4
                    let mean = s1[m].div_euclid(2);
                    let sigma = (s2[m] as f64 / 4.0).sqrt();
                    if sigma > 0.0 {
                        let z = (half - 0.5) / sigma;
                        margin = margin.min(z);
                        if z < margins[d].0 {
                            margins[d] = (z, c, m, mean.abs());
                        }
                    }
                    let r = mean.rem_euclid(P as i64);
                    mur[(at + c) * N + m] = if r > (P as i64 - 1) / 2 { (r - P as i64) as i16 } else { r as i16 };
                    mu[m] = mu[m].wrapping_add((mean as u32).wrapping_shl(digit_shift(d)));
                }
            }
        }
        SmsKeyData { pairs, mur, mu, elements, margin, margins }
    }
}

// =============================================================================================
// the kernel's sink
// =============================================================================================

/// One column group's nine stream accumulators: `v[j][s]` is slot `j`, stream `s`, 16 columns.
#[repr(C, align(64))]
pub struct SAcc {
    pub v: [[[i32; 16]; STREAMS]; N],
}

#[repr(C, align(64))]
struct Blk27([i16; 27 * 32]);

/// Fold-back constants of [`reduce_vec_c`] as memory operands: the `vpmaddwd` weights, the sign
/// test mask and the correction.
#[repr(C, align(64))]
struct Fold([i32; 48]);
static FOLD: Fold = Fold({
    let mut f = [0i32; 48];
    let mut i = 0;
    while i < 16 {
        f[i] = 1 | ((r16c(P) as u16 as i32) << 16);
        f[16 + i] = 0x8000;
        f[32 + i] = r16c(P);
        i += 1;
    }
    f
});

/// One slot's nine streams, round by round: the nine accumulators are loaded, take the first
/// lockstep product, then the second, ..., and are stored (folded back first in `reduce`), so at
/// most nine dependent `vpdpwssd` wait at a time and every round's nine are independent.
macro_rules! sms_slot {
    (plain) => { concat!(
            "vmovdqa64 zmm20, zmmword ptr [{a} + 0]\n",
            "vmovdqa64 zmm21, zmmword ptr [{a} + 64]\n",
            "vmovdqa64 zmm22, zmmword ptr [{a} + 128]\n",
            "vmovdqa64 zmm23, zmmword ptr [{a} + 192]\n",
            "vmovdqa64 zmm24, zmmword ptr [{a} + 256]\n",
            "vmovdqa64 zmm25, zmmword ptr [{a} + 320]\n",
            "vmovdqa64 zmm26, zmmword ptr [{a} + 384]\n",
            "vmovdqa64 zmm27, zmmword ptr [{a} + 448]\n",
            "vmovdqa64 zmm28, zmmword ptr [{a} + 512]\n",
            "vpdpwssd zmm20, {x0}, dword ptr [{k} + 0]{{1to16}}\n",
            "vpdpwssd zmm21, {x0}, dword ptr [{k} + 32]{{1to16}}\n",
            "vpdpwssd zmm22, {x0}, dword ptr [{k} + 64]{{1to16}}\n",
            "vpdpwssd zmm23, {x0}, dword ptr [{k} + 96]{{1to16}}\n",
            "vpdpwssd zmm24, {x0}, dword ptr [{k} + 128]{{1to16}}\n",
            "vpdpwssd zmm25, {x0}, dword ptr [{k} + 160]{{1to16}}\n",
            "vpdpwssd zmm26, {x0}, dword ptr [{k} + 192]{{1to16}}\n",
            "vpdpwssd zmm27, {x0}, dword ptr [{k} + 224]{{1to16}}\n",
            "vpdpwssd zmm28, {x0}, dword ptr [{k} + 256]{{1to16}}\n",
            "vpdpwssd zmm20, {x1}, dword ptr [{k} + 4]{{1to16}}\n",
            "vpdpwssd zmm21, {x1}, dword ptr [{k} + 36]{{1to16}}\n",
            "vpdpwssd zmm22, {x1}, dword ptr [{k} + 68]{{1to16}}\n",
            "vpdpwssd zmm23, {x1}, dword ptr [{k} + 100]{{1to16}}\n",
            "vpdpwssd zmm24, {x1}, dword ptr [{k} + 132]{{1to16}}\n",
            "vpdpwssd zmm25, {x1}, dword ptr [{k} + 164]{{1to16}}\n",
            "vpdpwssd zmm26, {x1}, dword ptr [{k} + 196]{{1to16}}\n",
            "vpdpwssd zmm27, {x1}, dword ptr [{k} + 228]{{1to16}}\n",
            "vpdpwssd zmm28, {x1}, dword ptr [{k} + 260]{{1to16}}\n",
            "vpdpwssd zmm20, {x2}, dword ptr [{k} + 8]{{1to16}}\n",
            "vpdpwssd zmm21, {x2}, dword ptr [{k} + 40]{{1to16}}\n",
            "vpdpwssd zmm22, {x2}, dword ptr [{k} + 72]{{1to16}}\n",
            "vpdpwssd zmm23, {x2}, dword ptr [{k} + 104]{{1to16}}\n",
            "vpdpwssd zmm24, {x2}, dword ptr [{k} + 136]{{1to16}}\n",
            "vpdpwssd zmm25, {x2}, dword ptr [{k} + 168]{{1to16}}\n",
            "vpdpwssd zmm26, {x2}, dword ptr [{k} + 200]{{1to16}}\n",
            "vpdpwssd zmm27, {x2}, dword ptr [{k} + 232]{{1to16}}\n",
            "vpdpwssd zmm28, {x2}, dword ptr [{k} + 264]{{1to16}}\n",
            "vpdpwssd zmm20, {x3}, dword ptr [{k} + 12]{{1to16}}\n",
            "vpdpwssd zmm21, {x3}, dword ptr [{k} + 44]{{1to16}}\n",
            "vpdpwssd zmm22, {x3}, dword ptr [{k} + 76]{{1to16}}\n",
            "vpdpwssd zmm23, {x3}, dword ptr [{k} + 108]{{1to16}}\n",
            "vpdpwssd zmm24, {x3}, dword ptr [{k} + 140]{{1to16}}\n",
            "vpdpwssd zmm25, {x3}, dword ptr [{k} + 172]{{1to16}}\n",
            "vpdpwssd zmm26, {x3}, dword ptr [{k} + 204]{{1to16}}\n",
            "vpdpwssd zmm27, {x3}, dword ptr [{k} + 236]{{1to16}}\n",
            "vpdpwssd zmm28, {x3}, dword ptr [{k} + 268]{{1to16}}\n",
            "vpdpwssd zmm20, {x4}, dword ptr [{k} + 16]{{1to16}}\n",
            "vpdpwssd zmm21, {x4}, dword ptr [{k} + 48]{{1to16}}\n",
            "vpdpwssd zmm22, {x4}, dword ptr [{k} + 80]{{1to16}}\n",
            "vpdpwssd zmm23, {x4}, dword ptr [{k} + 112]{{1to16}}\n",
            "vpdpwssd zmm24, {x4}, dword ptr [{k} + 144]{{1to16}}\n",
            "vpdpwssd zmm25, {x4}, dword ptr [{k} + 176]{{1to16}}\n",
            "vpdpwssd zmm26, {x4}, dword ptr [{k} + 208]{{1to16}}\n",
            "vpdpwssd zmm27, {x4}, dword ptr [{k} + 240]{{1to16}}\n",
            "vpdpwssd zmm28, {x4}, dword ptr [{k} + 272]{{1to16}}\n",
            "vpdpwssd zmm20, {x5}, dword ptr [{k} + 20]{{1to16}}\n",
            "vpdpwssd zmm21, {x5}, dword ptr [{k} + 52]{{1to16}}\n",
            "vpdpwssd zmm22, {x5}, dword ptr [{k} + 84]{{1to16}}\n",
            "vpdpwssd zmm23, {x5}, dword ptr [{k} + 116]{{1to16}}\n",
            "vpdpwssd zmm24, {x5}, dword ptr [{k} + 148]{{1to16}}\n",
            "vpdpwssd zmm25, {x5}, dword ptr [{k} + 180]{{1to16}}\n",
            "vpdpwssd zmm26, {x5}, dword ptr [{k} + 212]{{1to16}}\n",
            "vpdpwssd zmm27, {x5}, dword ptr [{k} + 244]{{1to16}}\n",
            "vpdpwssd zmm28, {x5}, dword ptr [{k} + 276]{{1to16}}\n",
            "vpdpwssd zmm20, {x6}, dword ptr [{k} + 24]{{1to16}}\n",
            "vpdpwssd zmm21, {x6}, dword ptr [{k} + 56]{{1to16}}\n",
            "vpdpwssd zmm22, {x6}, dword ptr [{k} + 88]{{1to16}}\n",
            "vpdpwssd zmm23, {x6}, dword ptr [{k} + 120]{{1to16}}\n",
            "vpdpwssd zmm24, {x6}, dword ptr [{k} + 152]{{1to16}}\n",
            "vpdpwssd zmm25, {x6}, dword ptr [{k} + 184]{{1to16}}\n",
            "vpdpwssd zmm26, {x6}, dword ptr [{k} + 216]{{1to16}}\n",
            "vpdpwssd zmm27, {x6}, dword ptr [{k} + 248]{{1to16}}\n",
            "vpdpwssd zmm28, {x6}, dword ptr [{k} + 280]{{1to16}}\n",
            "vpdpwssd zmm20, {x7}, dword ptr [{k} + 28]{{1to16}}\n",
            "vpdpwssd zmm21, {x7}, dword ptr [{k} + 60]{{1to16}}\n",
            "vpdpwssd zmm22, {x7}, dword ptr [{k} + 92]{{1to16}}\n",
            "vpdpwssd zmm23, {x7}, dword ptr [{k} + 124]{{1to16}}\n",
            "vpdpwssd zmm24, {x7}, dword ptr [{k} + 156]{{1to16}}\n",
            "vpdpwssd zmm25, {x7}, dword ptr [{k} + 188]{{1to16}}\n",
            "vpdpwssd zmm26, {x7}, dword ptr [{k} + 220]{{1to16}}\n",
            "vpdpwssd zmm27, {x7}, dword ptr [{k} + 252]{{1to16}}\n",
            "vpdpwssd zmm28, {x7}, dword ptr [{k} + 284]{{1to16}}\n",
            "vmovdqa64 zmmword ptr [{a} + 0], zmm20\n",
            "vmovdqa64 zmmword ptr [{a} + 64], zmm21\n",
            "vmovdqa64 zmmword ptr [{a} + 128], zmm22\n",
            "vmovdqa64 zmmword ptr [{a} + 192], zmm23\n",
            "vmovdqa64 zmmword ptr [{a} + 256], zmm24\n",
            "vmovdqa64 zmmword ptr [{a} + 320], zmm25\n",
            "vmovdqa64 zmmword ptr [{a} + 384], zmm26\n",
            "vmovdqa64 zmmword ptr [{a} + 448], zmm27\n",
            "vmovdqa64 zmmword ptr [{a} + 512], zmm28\n",
        )
    };
    (reduce) => { concat!(
            "vmovdqa64 zmm20, zmmword ptr [{a} + 0]\n",
            "vmovdqa64 zmm21, zmmword ptr [{a} + 64]\n",
            "vmovdqa64 zmm22, zmmword ptr [{a} + 128]\n",
            "vmovdqa64 zmm23, zmmword ptr [{a} + 192]\n",
            "vmovdqa64 zmm24, zmmword ptr [{a} + 256]\n",
            "vmovdqa64 zmm25, zmmword ptr [{a} + 320]\n",
            "vmovdqa64 zmm26, zmmword ptr [{a} + 384]\n",
            "vmovdqa64 zmm27, zmmword ptr [{a} + 448]\n",
            "vmovdqa64 zmm28, zmmword ptr [{a} + 512]\n",
            "vpdpwssd zmm20, {x0}, dword ptr [{k} + 0]{{1to16}}\n",
            "vpdpwssd zmm21, {x0}, dword ptr [{k} + 32]{{1to16}}\n",
            "vpdpwssd zmm22, {x0}, dword ptr [{k} + 64]{{1to16}}\n",
            "vpdpwssd zmm23, {x0}, dword ptr [{k} + 96]{{1to16}}\n",
            "vpdpwssd zmm24, {x0}, dword ptr [{k} + 128]{{1to16}}\n",
            "vpdpwssd zmm25, {x0}, dword ptr [{k} + 160]{{1to16}}\n",
            "vpdpwssd zmm26, {x0}, dword ptr [{k} + 192]{{1to16}}\n",
            "vpdpwssd zmm27, {x0}, dword ptr [{k} + 224]{{1to16}}\n",
            "vpdpwssd zmm28, {x0}, dword ptr [{k} + 256]{{1to16}}\n",
            "vpdpwssd zmm20, {x1}, dword ptr [{k} + 4]{{1to16}}\n",
            "vpdpwssd zmm21, {x1}, dword ptr [{k} + 36]{{1to16}}\n",
            "vpdpwssd zmm22, {x1}, dword ptr [{k} + 68]{{1to16}}\n",
            "vpdpwssd zmm23, {x1}, dword ptr [{k} + 100]{{1to16}}\n",
            "vpdpwssd zmm24, {x1}, dword ptr [{k} + 132]{{1to16}}\n",
            "vpdpwssd zmm25, {x1}, dword ptr [{k} + 164]{{1to16}}\n",
            "vpdpwssd zmm26, {x1}, dword ptr [{k} + 196]{{1to16}}\n",
            "vpdpwssd zmm27, {x1}, dword ptr [{k} + 228]{{1to16}}\n",
            "vpdpwssd zmm28, {x1}, dword ptr [{k} + 260]{{1to16}}\n",
            "vpdpwssd zmm20, {x2}, dword ptr [{k} + 8]{{1to16}}\n",
            "vpdpwssd zmm21, {x2}, dword ptr [{k} + 40]{{1to16}}\n",
            "vpdpwssd zmm22, {x2}, dword ptr [{k} + 72]{{1to16}}\n",
            "vpdpwssd zmm23, {x2}, dword ptr [{k} + 104]{{1to16}}\n",
            "vpdpwssd zmm24, {x2}, dword ptr [{k} + 136]{{1to16}}\n",
            "vpdpwssd zmm25, {x2}, dword ptr [{k} + 168]{{1to16}}\n",
            "vpdpwssd zmm26, {x2}, dword ptr [{k} + 200]{{1to16}}\n",
            "vpdpwssd zmm27, {x2}, dword ptr [{k} + 232]{{1to16}}\n",
            "vpdpwssd zmm28, {x2}, dword ptr [{k} + 264]{{1to16}}\n",
            "vpdpwssd zmm20, {x3}, dword ptr [{k} + 12]{{1to16}}\n",
            "vpdpwssd zmm21, {x3}, dword ptr [{k} + 44]{{1to16}}\n",
            "vpdpwssd zmm22, {x3}, dword ptr [{k} + 76]{{1to16}}\n",
            "vpdpwssd zmm23, {x3}, dword ptr [{k} + 108]{{1to16}}\n",
            "vpdpwssd zmm24, {x3}, dword ptr [{k} + 140]{{1to16}}\n",
            "vpdpwssd zmm25, {x3}, dword ptr [{k} + 172]{{1to16}}\n",
            "vpdpwssd zmm26, {x3}, dword ptr [{k} + 204]{{1to16}}\n",
            "vpdpwssd zmm27, {x3}, dword ptr [{k} + 236]{{1to16}}\n",
            "vpdpwssd zmm28, {x3}, dword ptr [{k} + 268]{{1to16}}\n",
            "vpdpwssd zmm20, {x4}, dword ptr [{k} + 16]{{1to16}}\n",
            "vpdpwssd zmm21, {x4}, dword ptr [{k} + 48]{{1to16}}\n",
            "vpdpwssd zmm22, {x4}, dword ptr [{k} + 80]{{1to16}}\n",
            "vpdpwssd zmm23, {x4}, dword ptr [{k} + 112]{{1to16}}\n",
            "vpdpwssd zmm24, {x4}, dword ptr [{k} + 144]{{1to16}}\n",
            "vpdpwssd zmm25, {x4}, dword ptr [{k} + 176]{{1to16}}\n",
            "vpdpwssd zmm26, {x4}, dword ptr [{k} + 208]{{1to16}}\n",
            "vpdpwssd zmm27, {x4}, dword ptr [{k} + 240]{{1to16}}\n",
            "vpdpwssd zmm28, {x4}, dword ptr [{k} + 272]{{1to16}}\n",
            "vpdpwssd zmm20, {x5}, dword ptr [{k} + 20]{{1to16}}\n",
            "vpdpwssd zmm21, {x5}, dword ptr [{k} + 52]{{1to16}}\n",
            "vpdpwssd zmm22, {x5}, dword ptr [{k} + 84]{{1to16}}\n",
            "vpdpwssd zmm23, {x5}, dword ptr [{k} + 116]{{1to16}}\n",
            "vpdpwssd zmm24, {x5}, dword ptr [{k} + 148]{{1to16}}\n",
            "vpdpwssd zmm25, {x5}, dword ptr [{k} + 180]{{1to16}}\n",
            "vpdpwssd zmm26, {x5}, dword ptr [{k} + 212]{{1to16}}\n",
            "vpdpwssd zmm27, {x5}, dword ptr [{k} + 244]{{1to16}}\n",
            "vpdpwssd zmm28, {x5}, dword ptr [{k} + 276]{{1to16}}\n",
            "vpdpwssd zmm20, {x6}, dword ptr [{k} + 24]{{1to16}}\n",
            "vpdpwssd zmm21, {x6}, dword ptr [{k} + 56]{{1to16}}\n",
            "vpdpwssd zmm22, {x6}, dword ptr [{k} + 88]{{1to16}}\n",
            "vpdpwssd zmm23, {x6}, dword ptr [{k} + 120]{{1to16}}\n",
            "vpdpwssd zmm24, {x6}, dword ptr [{k} + 152]{{1to16}}\n",
            "vpdpwssd zmm25, {x6}, dword ptr [{k} + 184]{{1to16}}\n",
            "vpdpwssd zmm26, {x6}, dword ptr [{k} + 216]{{1to16}}\n",
            "vpdpwssd zmm27, {x6}, dword ptr [{k} + 248]{{1to16}}\n",
            "vpdpwssd zmm28, {x6}, dword ptr [{k} + 280]{{1to16}}\n",
            "vpdpwssd zmm20, {x7}, dword ptr [{k} + 28]{{1to16}}\n",
            "vpdpwssd zmm21, {x7}, dword ptr [{k} + 60]{{1to16}}\n",
            "vpdpwssd zmm22, {x7}, dword ptr [{k} + 92]{{1to16}}\n",
            "vpdpwssd zmm23, {x7}, dword ptr [{k} + 124]{{1to16}}\n",
            "vpdpwssd zmm24, {x7}, dword ptr [{k} + 156]{{1to16}}\n",
            "vpdpwssd zmm25, {x7}, dword ptr [{k} + 188]{{1to16}}\n",
            "vpdpwssd zmm26, {x7}, dword ptr [{k} + 220]{{1to16}}\n",
            "vpdpwssd zmm27, {x7}, dword ptr [{k} + 252]{{1to16}}\n",
            "vpdpwssd zmm28, {x7}, dword ptr [{k} + 284]{{1to16}}\n",
            "vpmaddwd zmm29, zmm20, zmmword ptr [{f}]\n",
            "vptestmd k1, zmm20, zmmword ptr [{f} + 64]\n",
            "vpaddd zmm29 {{k1}}, zmm29, zmmword ptr [{f} + 128]\n",
            "vmovdqa64 zmmword ptr [{a} + 0], zmm29\n",
            "vpmaddwd zmm29, zmm21, zmmword ptr [{f}]\n",
            "vptestmd k1, zmm21, zmmword ptr [{f} + 64]\n",
            "vpaddd zmm29 {{k1}}, zmm29, zmmword ptr [{f} + 128]\n",
            "vmovdqa64 zmmword ptr [{a} + 64], zmm29\n",
            "vpmaddwd zmm29, zmm22, zmmword ptr [{f}]\n",
            "vptestmd k1, zmm22, zmmword ptr [{f} + 64]\n",
            "vpaddd zmm29 {{k1}}, zmm29, zmmword ptr [{f} + 128]\n",
            "vmovdqa64 zmmword ptr [{a} + 128], zmm29\n",
            "vpmaddwd zmm29, zmm23, zmmword ptr [{f}]\n",
            "vptestmd k1, zmm23, zmmword ptr [{f} + 64]\n",
            "vpaddd zmm29 {{k1}}, zmm29, zmmword ptr [{f} + 128]\n",
            "vmovdqa64 zmmword ptr [{a} + 192], zmm29\n",
            "vpmaddwd zmm29, zmm24, zmmword ptr [{f}]\n",
            "vptestmd k1, zmm24, zmmword ptr [{f} + 64]\n",
            "vpaddd zmm29 {{k1}}, zmm29, zmmword ptr [{f} + 128]\n",
            "vmovdqa64 zmmword ptr [{a} + 256], zmm29\n",
            "vpmaddwd zmm29, zmm25, zmmword ptr [{f}]\n",
            "vptestmd k1, zmm25, zmmword ptr [{f} + 64]\n",
            "vpaddd zmm29 {{k1}}, zmm29, zmmword ptr [{f} + 128]\n",
            "vmovdqa64 zmmword ptr [{a} + 320], zmm29\n",
            "vpmaddwd zmm29, zmm26, zmmword ptr [{f}]\n",
            "vptestmd k1, zmm26, zmmword ptr [{f} + 64]\n",
            "vpaddd zmm29 {{k1}}, zmm29, zmmword ptr [{f} + 128]\n",
            "vmovdqa64 zmmword ptr [{a} + 384], zmm29\n",
            "vpmaddwd zmm29, zmm27, zmmword ptr [{f}]\n",
            "vptestmd k1, zmm27, zmmword ptr [{f} + 64]\n",
            "vpaddd zmm29 {{k1}}, zmm29, zmmword ptr [{f} + 128]\n",
            "vmovdqa64 zmmword ptr [{a} + 448], zmm29\n",
            "vpmaddwd zmm29, zmm28, zmmword ptr [{f}]\n",
            "vptestmd k1, zmm28, zmmword ptr [{f} + 64]\n",
            "vpaddd zmm29 {{k1}}, zmm29, zmmword ptr [{f} + 128]\n",
            "vmovdqa64 zmmword ptr [{a} + 512], zmm29\n",
        )
    };
}
const _: () = assert!(STREAMS == 9 && LOCK == 8);

/// The 4-batch lockstep sink: the batches' 27-slot blocks are staged in L1 and every slot's nine
/// accumulators take all four products in one read-modify-write, folded back when `RED`.
struct Sink<const RED: bool> {
    k: *const u32,
    acc: *mut i32,
}
impl<const RED: bool> vb::SliceSink<LOCK> for Sink<RED> {
    #[inline(always)]
    unsafe fn rows(&mut self, blk: usize, w: [*const i16; LOCK], r0: usize, r1: usize) {
        sink_rows::<RED>(self.k, self.acc, blk, w, r0, r1);
    }
}

/// [`Sink::rows`], one copy for every call site: the transform's hot loop and this body together
/// then fit the half of the decoded-uop cache a hyperthread gets.
#[inline(never)]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
unsafe fn sink_rows<const RED: bool>(kp: *const u32, accp: *mut i32, blk: usize, w: [*const i16; LOCK], r0: usize, r1: usize) {
    let s = Sink::<RED> { k: kp, acc: accp };
    s.rows_inline(blk, w, r0, r1);
}

impl<const RED: bool> Sink<RED> {
    #[inline(always)]
    unsafe fn rows_inline(&self, blk: usize, w: [*const i16; LOCK], r0: usize, r1: usize) {
        for i in r0..r1 {
            let j = 27 * blk + i;
            let x: [__m512i; LOCK] = core::array::from_fn(|h| _mm512_load_si512((w[h] as *const __m512i).add(i)));
            let k = self.k.add(j * STREAMS * LOCK);
            let a = self.acc.add(j * STREAMS * 16);
            if RED {
                core::arch::asm!(sms_slot!(reduce), a = in(reg) a, k = in(reg) k, f = in(reg) FOLD.0.as_ptr(),
                    x0 = in(zmm_reg) x[0], x1 = in(zmm_reg) x[1], x2 = in(zmm_reg) x[2], x3 = in(zmm_reg) x[3], x4 = in(zmm_reg) x[4], x5 = in(zmm_reg) x[5], x6 = in(zmm_reg) x[6], x7 = in(zmm_reg) x[7],
                    out("zmm20") _, out("zmm21") _, out("zmm22") _, out("zmm23") _, out("zmm24") _, out("zmm25") _,
                    out("zmm26") _, out("zmm27") _, out("zmm28") _, out("zmm29") _, out("k1") _, options(nostack, preserves_flags));
            } else {
                core::arch::asm!(concat!(sms_slot!(plain), "/* {f} */"), a = in(reg) a, k = in(reg) k, f = in(reg) FOLD.0.as_ptr(),
                    x0 = in(zmm_reg) x[0], x1 = in(zmm_reg) x[1], x2 = in(zmm_reg) x[2], x3 = in(zmm_reg) x[3], x4 = in(zmm_reg) x[4], x5 = in(zmm_reg) x[5], x6 = in(zmm_reg) x[6], x7 = in(zmm_reg) x[7],
                    out("zmm20") _, out("zmm21") _, out("zmm22") _, out("zmm23") _, out("zmm24") _, out("zmm25") _,
                    out("zmm26") _, out("zmm27") _, out("zmm28") _, out("zmm29") _, out("k1") _, options(nostack, preserves_flags));
            }
        }
    }
}

const PAIR_PACK_PERM: [i64; 8] = [0, 2, 4, 6, 1, 3, 5, 7];

/// Stream `s` of the accumulators modulo 9721 as `i16` representatives (`|r| < 2q`) into lanes
/// `16 half .. 16 half + 16` of `out`, the stream cleared for the next chunk.
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn finish_stream(a: &mut SAcc, s: usize, out: &mut Batch32, half: usize) {
    let p = (a.v.as_mut_ptr() as *mut __m512i).add(s);
    let zero = _mm512_setzero_si512();
    let perm = _mm512_loadu_si512(PAIR_PACK_PERM.as_ptr() as *const __m512i);
    let l0 = 16 * half;
    for k in 0..N / 2 {
        let j = 2 * k;
        let x0 = _mm512_load_si512(p.add(STREAMS * j));
        let x1 = _mm512_load_si512(p.add(STREAMS * (j + 1)));
        _mm512_store_si512(p.add(STREAMS * j), zero);
        _mm512_store_si512(p.add(STREAMS * (j + 1)), zero);
        let packed = _mm512_packs_epi32(red16::<P>(x0), red16::<P>(x1));
        let u = _mm512_permutexvar_epi64(perm, packed);
        _mm256_storeu_si256(out.v[j].as_mut_ptr().add(l0) as *mut __m256i, _mm512_castsi512_si256(u));
        _mm256_storeu_si256(out.v[j + 1].as_mut_ptr().add(l0) as *mut __m256i, _mm512_extracti64x4_epi64::<1>(u));
    }
}

/// A column-group lane total: `t[j][0..16]` takes the even chunks, `t[j][16..32]` the odd ones.
#[repr(C, align(64))]
struct Total {
    t: [[i32; 32]; N],
}

/// The digit `d` residues of two chunks (`c0` in lanes 0..16, `c0 + 1` in lanes 16..32 when
/// `both`), inverse transformed and centered, reconstructed around their means and added to the
/// total with weight `8^d`.
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
unsafe fn reconstruct(b: &Batch32, d: usize, keys: &SmsKeys, c0: usize, both: bool, total: &mut Total) {
    let chunks = keys.elements / (2 * CHUNK_PAIRS);
    let at = if d < CHUNKED { d * chunks + c0 } else { CHUNKED * chunks + c0 };
    let mlo = keys.mur.as_ptr().add(at * N);
    let mhi = if both { keys.mur.as_ptr().add((at + 1) * N) } else { mlo };
    let q = _mm512_set1_epi16(P as i16);
    let hq = _mm512_set1_epi16(((P - 1) / 2) as i16);
    let nhq = _mm512_set1_epi16(-(((P - 1) / 2) as i16));
    let sh = _mm_cvtsi32_si128(digit_shift(d) as i32);
    for m in 0..N {
        let x = _mm512_load_si512(b.v[m].as_ptr() as *const __m512i);
        let mu = _mm512_mask_blend_epi16(0xFFFF_0000, _mm512_set1_epi16(*mlo.add(m)), _mm512_set1_epi16(*mhi.add(m)));
        let mut y = _mm512_sub_epi16(x, mu);
        y = _mm512_mask_sub_epi16(y, _mm512_cmpgt_epi16_mask(y, hq), y, q);
        y = _mm512_mask_add_epi16(y, _mm512_cmplt_epi16_mask(y, nhq), y, q);
        let lo = _mm512_sll_epi32(_mm512_cvtepi16_epi32(_mm512_castsi512_si256(y)), sh);
        let hi = _mm512_sll_epi32(_mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(y)), sh);
        let t = total.t[m].as_mut_ptr() as *mut __m512i;
        _mm512_store_si512(t, _mm512_add_epi32(_mm512_load_si512(t), lo));
        if both {
            _mm512_store_si512(t.add(1), _mm512_add_epi32(_mm512_load_si512(t.add(1)), hi));
        }
    }
}

/// Per-worker scratch, allocated once and reused by every group the worker takes.
struct SmsWorker {
    acc: Box<[SAcc]>,
    pend: Box<[Batch32]>,
    total: Box<[Total]>,
    idx: Box<[BinaryIndex32]>,
    w: Box<[Blk27]>,
}
impl SmsWorker {
    fn new() -> Self {
        SmsWorker {
            acc: zeroed_slice(1),
            pend: zeroed_slice(DIGITS),
            total: zeroed_slice(1),
            idx: zeroed_slice(LOCK),
            w: zeroed_slice(2 * LOCK),
        }
    }
}

/// 16 x 16 u32 transpose: row `s` holds columns `0..16` of slot `j0 + s`; output `c` holds slots
/// `j0 .. j0 + 16` of column `c`.
#[inline(always)]
unsafe fn transpose16x16_u32(r: &[__m512i; 16]) -> [__m512i; 16] {
    let mut u = [[_mm512_setzero_si512(); 4]; 4];
    for q in 0..4 {
        let t0 = _mm512_unpacklo_epi32(r[4 * q], r[4 * q + 1]);
        let t1 = _mm512_unpackhi_epi32(r[4 * q], r[4 * q + 1]);
        let t2 = _mm512_unpacklo_epi32(r[4 * q + 2], r[4 * q + 3]);
        let t3 = _mm512_unpackhi_epi32(r[4 * q + 2], r[4 * q + 3]);
        u[q] = [
            _mm512_unpacklo_epi64(t0, t2),
            _mm512_unpackhi_epi64(t0, t2),
            _mm512_unpacklo_epi64(t1, t3),
            _mm512_unpackhi_epi64(t1, t3),
        ];
    }
    let mut out = [_mm512_setzero_si512(); 16];
    for m in 0..4 {
        let t0 = _mm512_shuffle_i64x2::<0x88>(u[0][m], u[1][m]);
        let t1 = _mm512_shuffle_i64x2::<0xDD>(u[0][m], u[1][m]);
        let t2 = _mm512_shuffle_i64x2::<0x88>(u[2][m], u[3][m]);
        let t3 = _mm512_shuffle_i64x2::<0xDD>(u[2][m], u[3][m]);
        let v = [
            _mm512_shuffle_i64x2::<0x88>(t0, t2),
            _mm512_shuffle_i64x2::<0x88>(t1, t3),
            _mm512_shuffle_i64x2::<0xDD>(t0, t2),
            _mm512_shuffle_i64x2::<0xDD>(t1, t3),
        ];
        for l in 0..4 {
            out[4 * l + m] = v[l];
        }
    }
    out
}

/// The group's two rows per column: the base stream canonical in `[0, 9721)` (NTT domain) and the
/// tail `t mod 2^22` (coefficients); accumulators and total cleared for the next group.
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
unsafe fn write_rows(acc: &mut SAcc, total: &mut Total, keys: &SmsKeys, gc: usize, dst: *mut [u32; N]) {
    let zero = _mm512_setzero_si512();
    let mask = _mm512_set1_epi32(TAIL_MASK as i32);
    let pa = acc.v.as_mut_ptr() as *mut __m512i;
    let pt = total.t.as_mut_ptr() as *mut __m512i;
    let mut j0 = 0;
    while j0 < N {
        let n = 16.min(N - j0);
        let mut r0 = [zero; 16];
        let mut rt = [zero; 16];
        for s in 0..n {
            let j = j0 + s;
            let v0 = _mm512_load_si512(pa.add(STREAMS * j));
            _mm512_store_si512(pa.add(STREAMS * j), zero);
            r0[s] = canon_red16::<P>(v0);
            let a = _mm512_load_si512(pt.add(2 * j));
            let b = _mm512_load_si512(pt.add(2 * j + 1));
            _mm512_store_si512(pt.add(2 * j), zero);
            _mm512_store_si512(pt.add(2 * j + 1), zero);
            let t = _mm512_add_epi32(_mm512_add_epi32(a, b), _mm512_set1_epi32(keys.mu[j] as i32));
            rt[s] = _mm512_and_si512(t, mask);
        }
        let c0 = transpose16x16_u32(&r0);
        let ct = transpose16x16_u32(&rt);
        for c in 0..gc {
            let m = if n == 16 { 0xFFFF } else { (1u16 << n) - 1 };
            _mm512_mask_storeu_epi32((dst.add(2 * c) as *mut i32).add(j0), m, c0[c]);
            _mm512_mask_storeu_epi32((dst.add(2 * c + 1) as *mut i32).add(j0), m, ct[c]);
        }
        j0 += 16;
    }
}

/// Commits a group of up to 16 columns across all `np` pair batches and writes its `2 g` rows to
/// `dst`. `sync(b_done)` is called at every [`PP`] boundary so hyperthread siblings can share the
/// key stream in L2.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vnni,gfni")]
unsafe fn group<F: FnMut(usize)>(cols: &[&[B128]], keys: &SmsKeys, np: usize, ws: &mut SmsWorker, dst: *mut [u32; N], mut sync: F) {
    let g = cols.len();
    assert!(g >= 1 && g <= 16 && np % CHUNK_PAIRS == 0 && (np / 2) % LOCK == 0 && 2 * np == keys.elements);
    static ZERO8: [B128; 8] = [B128::ZERO; 8];
    let kbase = keys.pairs.as_ptr();
    let mut half = 0usize;
    let acc = ws.acc[0].v.as_mut_ptr() as *mut i32;
    let w = ws.w.as_mut_ptr() as *mut i16;
    for b in (0..np).step_by(LOCK) {
        for h in 0..LOCK {
            let cols16: [*const B128; 16] = core::array::from_fn(|l| {
                if l < g { cols[l].as_ptr().add(8 * (b + h)) } else { ZERO8.as_ptr() }
            });
            crate::simd::transpose_f162::slice_b128_cols16_into(&cols16, &mut ws.idx[h]);
        }
        let ins: [&BinaryIndex32; LOCK] = core::array::from_fn(|h| &ws.idx[h]);
        let k = kbase.add((b / LOCK) * LOCK_WORDS);
        let mode = bench_mode();
        if mode == 4 { continue; }
        if mode == 1 {
            struct Nop;
            impl vb::SliceSink<LOCK> for Nop { unsafe fn rows(&mut self, _: usize, w: [*const i16; LOCK], _: usize, _: usize) { core::hint::black_box(w); } }
            vb::ntt_bin_batch32_pipelined::<P, LOCK, _>(ins, w, &mut Nop);
        } else if mode == 2 {
            let mut sk = Sink::<false> { k, acc };
            for kk in 0..24 { vb::SliceSink::<LOCK>::rows(&mut sk, kk, core::array::from_fn(|h| w.add(32 * 27 * h) as *const i16), 0, 27); }
        } else if (b + LOCK) % PP == 0 || LOCK > PP {
            vb::ntt_bin_batch32_pipelined::<P, LOCK, _>(ins, w, &mut Sink::<true> { k, acc });
            sync(b + LOCK);
        } else {
            vb::ntt_bin_batch32_pipelined::<P, LOCK, _>(ins, w, &mut Sink::<false> { k, acc });
        }
        if (b + LOCK) % (np / 2) == 0 && mode != 3 {
            let d = DIGITS - 1;
            finish_stream(&mut ws.acc[0], 1 + d, &mut ws.pend[d], (b + LOCK) / (np / 2) - 1);
        }
        if (b + LOCK) % CHUNK_PAIRS == 0 && mode != 3 {
            let chunk = (b + LOCK) / CHUNK_PAIRS - 1;
            for d in 0..CHUNKED {
                finish_stream(&mut ws.acc[0], 1 + d, &mut ws.pend[d], half);
            }
            if half == 1 || b + LOCK == np {
                let both = half == 1;
                for d in 0..CHUNKED {
                    let pd = &mut ws.pend[d];
                    pd.representation = Representation::Ntt;
                    gen_small::intt_gen_batch32::<P>(pd);
                    reconstruct(pd, d, keys, chunk - half, both, &mut ws.total[0]);
                }
                half = 0;
            } else {
                half = 1;
            }
        }
    }
    if bench_mode() != 3 {
        let d = DIGITS - 1;
        let pd = &mut ws.pend[d];
        pd.representation = Representation::Ntt;
        gen_small::intt_gen_batch32::<P>(pd);
        reconstruct(pd, d, keys, 0, true, &mut ws.total[0]);
    }
    write_rows(&mut ws.acc[0], &mut ws.total[0], keys, g, dst);
}

/// Benchmark-only phase isolation (`SMS_MODE`): 0 everything, 1 no accumulation, 2 accumulation
/// only, 3 no chunk reconstruction, 4 slicing only. Results are meaningless unless 0.
#[inline(always)]
fn bench_mode() -> u32 {
    #[cfg(test)]
    {
        static M: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
        *M.get_or_init(|| std::env::var("SMS_MODE").ok().and_then(|v| v.parse().ok()).unwrap_or(0))
    }
    #[cfg(not(test))]
    {
        0
    }
}

/// A batch counter on a cache line of its own.
#[repr(align(128))]
struct Progress(std::sync::atomic::AtomicUsize);
const SIBLING_WAIT: std::time::Duration = std::time::Duration::from_micros(500);
/// Maximum lead (in pair batches) between two hyperthread siblings on one physical core.
const PAIR_LEAD: usize = 2 * PP;

/// Hyperthread siblings `2k`, `2k + 1` claim 32-column super-groups (16 columns each) and advance
/// through the key within [`PAIR_LEAD`] pair batches of each other, so both read every key line from
/// their core's L2.
fn commit_paired(keys: &SmsKeys, columns: &[&[B128]], np: usize, workers: usize, placement: Option<&crate::par::CorePlacement>, dst: usize) {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    let g = 16usize;
    let sg = 2 * g;
    let pairs = workers / 2;
    let super_groups = columns.len().div_ceil(sg);
    let rounds = super_groups + 1;
    let next = AtomicUsize::new(0);
    let slots: Vec<std::sync::OnceLock<usize>> = (0..pairs * rounds).map(|_| std::sync::OnceLock::new()).collect();
    let done: Vec<Progress> = (0..2 * pairs).map(|_| Progress(AtomicUsize::new(0))).collect();
    let coupled: Vec<AtomicBool> = (0..pairs).map(|_| AtomicBool::new(true)).collect();
    crate::par::pool().run(2 * pairs, |worker| {
        let _guard = placement.and_then(|p| p.pin(worker));
        let (pair, s) = (worker / 2, worker % 2);
        let mine = &done[worker].0;
        let other = &done[worker ^ 1].0;
        let coupled = &coupled[pair];
        let mut ws: Option<SmsWorker> = None;
        for r in 0..rounds {
            let gi = *slots[pair * rounds + r].get_or_init(|| next.fetch_add(1, Ordering::Relaxed));
            if gi >= super_groups {
                break;
            }
            let c0 = gi * sg;
            let gc = sg.min(columns.len() - c0);
            let (lo, n) = if s == 0 { (c0, gc.min(g)) } else { (c0 + g, gc.saturating_sub(g)) };
            let base = r * np;
            if n == 0 {
                mine.store(base + np, Ordering::Release);
                continue;
            }
            let ws = ws.get_or_insert_with(SmsWorker::new);
            unsafe {
                group(&columns[lo..lo + n], keys, np, ws, (dst as *mut [u32; N]).add(2 * lo), |b_done| {
                    let d = base + b_done;
                    mine.store(d, Ordering::Release);
                    if !coupled.load(Ordering::Relaxed) {
                        return;
                    }
                    if other.load(Ordering::Acquire) + PAIR_LEAD >= d {
                        return;
                    }
                    let start = std::time::Instant::now();
                    let mut spins = 0u32;
                    while other.load(Ordering::Acquire) + PAIR_LEAD < d {
                        core::hint::spin_loop();
                        spins += 1;
                        if spins % 64 == 0 && start.elapsed() > SIBLING_WAIT {
                            coupled.store(false, Ordering::Relaxed);
                            return;
                        }
                    }
                })
            };
        }
        mine.store(usize::MAX / 2, Ordering::Release);
    });
}

/// The commitment's rows: `rows[2c]` is column `c` modulo 9721 in the NTT domain (canonical),
/// `rows[2c + 1]` its coefficients modulo `2^22`.
pub fn commit_columns_rows(keys: &SmsKeys, columns: &[&[B128]], workers: usize) -> Vec<[u32; N]> {
    assert!(!columns.is_empty(), "at least one witness column");
    let len = columns[0].len();
    for c in columns {
        assert_eq!(c.len(), len);
    }
    assert_eq!(len % 8, 0);
    let np = len / 8;
    assert_eq!(keys.elements, 2 * np);
    assert_eq!(keys.pairs.len(), (np / LOCK) * LOCK_WORDS);
    let g = 16usize;
    let groups = columns.len().div_ceil(g);
    let mut rows: Vec<[u32; N]> = Vec::with_capacity(columns.len() * 2);
    let dst = rows.as_mut_ptr() as usize;
    let paired = if workers >= 2 && workers % 2 == 0 && groups > 1 { crate::par::CorePlacement::for_sibling_pairs(workers) } else { None };
    if let Some(placement) = paired {
        commit_paired(keys, columns, np, workers, Some(&placement), dst);
    } else {
        let workers = workers.clamp(1, groups);
        let next = std::sync::atomic::AtomicUsize::new(0);
        let placement = crate::par::CorePlacement::for_logical(workers).or_else(|| crate::par::CorePlacement::for_workers(workers));
        crate::par::pool().run(workers, |worker| {
            let _guard = placement.as_ref().and_then(|p| p.pin(worker));
            let mut ws: Option<SmsWorker> = None;
            loop {
                let gi = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if gi >= groups {
                    break;
                }
                let c0 = gi * g;
                let gc = g.min(columns.len() - c0);
                let ws = ws.get_or_insert_with(SmsWorker::new);
                unsafe { group(&columns[c0..c0 + gc], keys, np, ws, (dst as *mut [u32; N]).add(2 * c0), |_| {}) };
            }
        });
    }
    unsafe { rows.set_len(columns.len() * 2) };
    rows
}

/// Column `c`'s coefficients modulo [`MODULUS`] from its two rows (reference recombination).
pub fn column_coefficients(base: &[u32; N], tail: &[u32; N]) -> Vec<u64> {
    let mut out = vec![0i16; N];
    crate::bd::coefficients_from_rows(P, &[base], &mut out);
    let inv = inv_tail();
    (0..N)
        .map(|i| {
            let r = (out[i] as i64).rem_euclid(P as i64) as u64;
            let t = tail[i] as u64;
            // v = t + 2^22 k with v ≡ r (mod P)
            let k = ((r + P as u64 - t % P as u64) % P as u64) * inv % P as u64;
            t + (k << TAIL_BITS)
        })
        .collect()
}
/// `2^-22 mod 9721`.
pub const fn inv_tail() -> u64 {
    let t = (1u64 << TAIL_BITS) % P as u64;
    let mut x = 1u64;
    while (x * t) % P as u64 != 1 {
        x += 1;
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    fn xorshift(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    /// `t = sum_j A_j s_j mod Q` straight from the coefficients.
    fn reference(a: &[u128], col: &[B128]) -> Vec<u64> {
        let q = MODULUS;
        let elements = col.len() / 4;
        let mut acc = vec![0u128; 2 * N];
        for j in 0..elements {
            let aj = &a[j * N..(j + 1) * N];
            for c in 0..512 {
                let (bit, k) = (c / 4, c % 4);
                if (col[4 * j + k].0 >> bit) & 1 == 1 {
                    for m in 0..N {
                        acc[c + m] = (acc[c + m] + aj[m]) % q;
                    }
                }
            }
        }
        for k in (N..2 * N).rev() {
            let v = acc[k];
            acc[k - 324] = (acc[k - 324] + v) % q;
            acc[k - 648] = (acc[k - 648] + q - v) % q;
        }
        acc[..N].iter().map(|&x| x as u64).collect()
    }

    #[test]
    fn digits_reconstruct() {
        let mut s = 0x1234u64;
        for _ in 0..10000 {
            let x = (xorshift(&mut s) as u128) % MODULUS;
            let d = digits(x);
            let mut v = 0i64;
            for k in (0..DIGITS).rev() {
                v = v * 8 + d[k] as i64;
            }
            // L_7 has weight 2^21 = 8^7: the loop gives sum 8^d L_d
            assert_eq!((v.rem_euclid(1 << TAIL_BITS)) as u128, x % (1 << TAIL_BITS));
            for k in 0..DIGITS - 1 {
                assert!((-4..4).contains(&d[k]));
            }
            assert!((0..2).contains(&d[DIGITS - 1]));
        }
    }

    #[test]
    fn moments_match_direct() {
        let mut s = 0x77u64;
        let l: [i8; N] = core::array::from_fn(|_| ((xorshift(&mut s) % 8) as i8) - 4);
        let mut s1 = [0i64; N];
        let mut s2 = [0i64; N];
        moment_rows(&l, &mut s1, &mut s2);
        let mut d1 = [0i64; N];
        let mut d2 = [0i64; N];
        for i in 0..512 {
            let mut k = vec![0i64; 2 * N];
            for m in 0..N {
                k[m + i] = l[m] as i64;
            }
            for j in (N..2 * N).rev() {
                let v = k[j];
                k[j - 324] += v;
                k[j - 648] -= v;
            }
            for m in 0..N {
                d1[m] += k[m];
                d2[m] += k[m] * k[m];
            }
        }
        assert_eq!(s1, d1);
        assert_eq!(s2, d2);
    }

    #[test]
    fn commitment_matches_reference() {
        let elements = 512usize; // 4 chunks
        let ncols = 37usize;
        let mut s = 0x9e3779b97f4a7c15u64;
        let a: Vec<u128> = (0..elements * N).map(|_| ((xorshift(&mut s) as u128) << 64 | xorshift(&mut s) as u128) % MODULUS).collect();
        let data = SmsKeyData::build(&a, elements);
        eprintln!("margin {}", data.margin);
        assert!(data.margin > 8.0);
        let len = 4 * elements;
        let words: Vec<B128> = (0..len * ncols).map(|_| B128(((xorshift(&mut s) as u128) << 64) | xorshift(&mut s) as u128)).collect();
        let cols: Vec<&[B128]> = words.chunks_exact(len).collect();
        let keys = data.keys();
        for workers in [1usize, 3, 4] {
            let rows = commit_columns_rows(&keys, &cols, workers);
            for c in [0usize, 1, 15, 16, 31, 36] {
                let got = column_coefficients(&rows[2 * c], &rows[2 * c + 1]);
                let want = reference(&a, cols[c]);
                assert!(got == want, "column {c} with {workers} workers differs");
            }
        }
    }
}

#[cfg(test)]
mod bench {
    use super::*;
    /// Full-shape timing against the hybrid commitment (`cargo test --release -- --ignored sms_bench --nocapture`).
    #[test]
    #[ignore]
    fn sms_bench() {
        let elements = 4096usize;
        let ncols = 1024usize;
        let mut s = 0x9e3779b97f4a7c15u64;
        let mut next = || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; s };
        let a: Vec<u128> = (0..elements * N).map(|_| ((next() as u128) << 64 | next() as u128) % MODULUS).collect();
        let t = std::time::Instant::now();
        let data = SmsKeyData::build(&a, elements);
        eprintln!("keys {:?} margin {:.2}", t.elapsed(), data.margin);
        let len = 4 * elements;
        let words: Vec<B128> = (0..len * ncols).map(|_| B128(((next() as u128) << 64) | next() as u128)).collect();
        let cols: Vec<&[B128]> = words.chunks_exact(len).collect();
        let keys = data.keys();
        let workers = std::env::var("SMS_WORKERS").ok().and_then(|v| v.parse().ok()).unwrap_or(crate::par::threads());
        let _ = crate::par::pool();
        let reps: usize = std::env::var("SMS_REPS").ok().and_then(|v| v.parse().ok()).unwrap_or(7);
        let mut ts = Vec::new();
        let mut chk = 0u32;
        for _ in 0..reps {
            let t = std::time::Instant::now();
            let rows = commit_columns_rows(&keys, &cols, workers);
            ts.push(t.elapsed().as_secs_f64() * 1e3);
            chk ^= rows[1][0];
        }
        ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
        eprintln!("sms commit min {:.1} med {:.1} ms ({chk})", ts[0], ts[ts.len() / 2]);
        if std::env::var_os("SMS_ONLY").is_some() {return;}
        let base = crate::key::CommitmentKey::random(len, 11, crate::ring::Modulus::Q3889_FS_S, &[crate::ring::Modulus::Q9721_FS_S]);
        let tail = crate::key::CommitmentKey::random(len, 12, crate::ring::Modulus::Q3889_FS_S, &[crate::ring::Modulus::Q9721_FS_S]);
        let pk = super::super::hybrid::pair_keys(&base, &tail, workers);
        for _ in 0..3 {
            let t = std::time::Instant::now();
            let rows = super::super::hybrid::commit_columns_rows_pairs(&pk, &cols, workers);
            eprintln!("hybrid commit {:?} ({})", t.elapsed(), rows[2][0]);
        }
    }
}

#[cfg(test)]
mod bench_parts {
    use super::*;
    #[test]
    #[ignore]
    fn sms_parts_bench() {
        let mut b = zeroed_slice::<Batch32>(1);
        let mut s = 7u64;
        for j in 0..N { for l in 0..32 { s ^= s << 13; s ^= s >> 7; s ^= s << 17; b[0].v[j][l] = ((s % 19441) as i32 - 9720) as i16; } }
        let n = 20000;
        let t = std::time::Instant::now();
        for _ in 0..n {
            b[0].representation = Representation::Ntt;
            unsafe { gen_small::intt_gen_batch32::<P>(&mut b[0]) };
            for j in 0..N { b[0].v[j][0] = b[0].v[j][0].wrapping_mul(3) % 9721; }
        }
        let dt = t.elapsed().as_secs_f64() / n as f64;
        eprintln!("intt_gen_batch32::<9721>: {:.2} us = {:.0} cycles at 3.5 GHz", dt * 1e6, dt * 3.5e9);
        let mut acc = zeroed_slice::<SAcc>(1);
        let t = std::time::Instant::now();
        for _ in 0..n {
            unsafe { finish_stream(&mut acc[0], 3, &mut b[0], 1) };
        }
        let dt = t.elapsed().as_secs_f64() / n as f64;
        eprintln!("finish_stream: {:.2} us", dt * 1e6);
    }
}
