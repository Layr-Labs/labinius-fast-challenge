//! The commitment modulo `Q = 9721 * M`, `M = 55 * 2^16`, with one witness transform per element
//! (the x86 track's "SMS" commitment, after mpjunior92's submission 85c24ce6).
//!
//! `t mod 9721` comes from the transform exactly as in [`super::commit`] ("base" row).
//!
//! `t mod M`: [`lattice::digits_pair`] writes every coefficient pair `(A_j, A_{j+324})` of an
//! element, modulo `M`, as six small digit pairs `(L_{d,j}, L_{d,j+324})`, `|digit| <= 32`, with
//! `A = sum_d W_d L_d (mod M)` for the fixed ring constants `W_d = u_d + v_d Y`, `Y = X^324`
//! ([`lattice::WEIGHTS`]). So `t = sum_d W_d sum_m L_{d,m} s_m (mod M)`. The same witness transform
//! is multiplied into six more accumulators against the transformed digit rows modulo 9721; per
//! chunk `c` of [`CHUNK`] elements the integer sums `S_{d,c} = sum_{m in c} L_{d,m} s_m` come back
//! from their residues modulo 9721 as `mean_{d,c} + centered((S - mean_{d,c}) mod 9721)`.
//!
//! That last step is STATISTICAL: `mean_{d,c}` is the expectation of `S_{d,c}` for uniform random
//! witness bits, and the recovery is exact whenever `|S - mean| <= 4860`. [`matrix`] computes the
//! smallest margin `z = (4860 - 0.5) / sigma` over every chunk, digit and coefficient and refuses
//! a matrix below [`MIN_MARGIN`] standard deviations (as the x86 build does). Everything else is
//! exact.
use super::*;
use crate::ntt::{centered, for_blocks};

#[path = "sms_lattice.rs"]
#[allow(dead_code)]
mod lattice;

/// The transform prime, by index into [`PRIMES`].
const PB: usize = 1;
const P1: u32 = PRIMES[PB];
/// The digit-encoded factor of the modulus.
pub const TAIL: u32 = lattice::M as u32;
/// The commitment modulus.
pub const MODULUS: u128 = P1 as u128 * TAIL as u128;
const _: () = assert!(MODULUS == 35_039_150_080);
const _: () = assert!(MODULUS >= 34_816_965_049, "the module-BKZ floor");
/// Digit rows per element.
pub const DIGITS: usize = 6;
/// Rows per element: `A mod 9721`, then the six digit rows, all transformed modulo 9721.
pub const ROWS: usize = 1 + DIGITS;
/// Elements per statistically recombined digit chunk.
pub const CHUNK: usize = 160;
const _: () = assert!(CHUNK % LOCK == 0);
/// Nodes of the encoder's closest-vector search: 1 is Babai's nearest plane alone, about 2000
/// times cheaper than the x86 build's 512 for a slightly larger digit spread (rms 3.88 against
/// 3.55). [`matrix`] computes the smallest margin and the Hoeffding union bound over every chunk,
/// digit and coefficient of a column from the public matrix and refuses a weaker one; `build.rs`
/// prints both for the [`CHUNK`] in use.
const SEARCH: usize = 1;
/// The smallest admissible reconstruction margin, in standard deviations.
pub const MIN_MARGIN: f64 = 7.5;
/// The largest admissible failure bound of one column (`sum 2 exp(-z^2 / 2)`, Hoeffding, over
/// every chunk, digit and coefficient): 1e-9, i.e. below 1e-6 for a proof's 1024 columns.
pub const MAX_COLUMN_FAILURE: f64 = 1e-9;
/// Unit terms of one reduced product coefficient: a witness coefficient `i < 512` times a
/// coefficient `x < 648` folds onto coefficient `i + x` and, for `i + x >= 648`, two more.
const TERMS: i64 = 836;
// every chunk digit sum and its mean fit in i32 with room: |S| <= CHUNK * TERMS * 32
const _: () = assert!((CHUNK as i64) * TERMS * (lattice::MAX_DIGIT_ABS as i64) * 4 < i32::MAX as i64);

/// The SMS matrix: in the transform domain, slot `j` of row `r` of element `m` is
/// `data[(((m / 8) N + j) ROWS + r) 8 + m % 8]`, centered residues, so that one vector holds the
/// eight elements of a tile; per chunk and digit the centered residue
/// of the mean modulo 9721; and the weighted sum of all means modulo `M`.
pub struct SmsMatrix {
    pub elements: usize,
    pub data: std::borrow::Cow<'static, [i16]>,
    /// `mean_residues[(c DIGITS + d) N + coefficient]`.
    pub mean_residues: std::borrow::Cow<'static, [i16]>,
    /// `sum_{d, c} W_d mean_{d,c} mod M`, by coefficient.
    pub mu: std::borrow::Cow<'static, [u32]>,
    /// The smallest reconstruction margin in standard deviations.
    pub margin: f64,
    /// Hoeffding union bound on one column's reconstruction failing, for uniform random bits.
    pub column_failure: f64,
}

/// `len` little-endian `T` at `image`, if it is exactly that long and aligned.
fn view<T>(image: &'static [u8], len: usize) -> Option<&'static [T]> {
    if image.len() != len * core::mem::size_of::<T>()
        || image.as_ptr() as usize % core::mem::align_of::<T>() != 0
        || cfg!(target_endian = "big")
    {
        return None;
    }
    // SAFETY: in bounds, aligned, and every bit pattern is a valid i16 / u32.
    Some(unsafe { core::slice::from_raw_parts(image.as_ptr() as *const T, len) })
}

impl SmsMatrix {
    /// The matrix from build-time images of [`matrix`]`(elements)`: `data`, `mean_residues`
    /// (i16) and `mu` (u32), little-endian, and `meta` = margin and column failure bound (f64).
    pub fn from_images(data: &'static [u8], means: &'static [u8], mu: &'static [u8], meta: &'static [u8], elements: usize) -> Option<SmsMatrix> {
        let chunks = elements.div_ceil(CHUNK);
        let data = view::<i16>(data, ROWS * elements.div_ceil(LOCK) * LOCK * N)?;
        let mean_residues = view::<i16>(means, chunks * DIGITS * N)?;
        let mu = view::<u32>(mu, N)?;
        if meta.len() != 16 {
            return None;
        }
        let margin = f64::from_le_bytes(meta[..8].try_into().ok()?);
        let column_failure = f64::from_le_bytes(meta[8..].try_into().ok()?);
        if !(margin >= MIN_MARGIN && column_failure <= MAX_COLUMN_FAILURE) {
            return None;
        }
        use std::borrow::Cow::Borrowed;
        Some(SmsMatrix { elements, data: Borrowed(data), mean_residues: Borrowed(mean_residues), mu: Borrowed(mu), margin, column_failure })
    }

    /// The four images [`SmsMatrix::from_images`] reads.
    pub fn images(&self) -> [Vec<u8>; 4] {
        let mut meta = self.margin.to_le_bytes().to_vec();
        meta.extend_from_slice(&self.column_failure.to_le_bytes());
        [
            self.data.iter().flat_map(|v| v.to_le_bytes()).collect(),
            self.mean_residues.iter().flat_map(|v| v.to_le_bytes()).collect(),
            self.mu.iter().flat_map(|v| v.to_le_bytes()).collect(),
            meta,
        ]
    }

    #[inline]
    fn mean_residues(&self, c: usize, d: usize) -> &[i16] {
        &self.mean_residues[(c * DIGITS + d) * N..][..N]
    }
}

/// The digit rows of one element (coefficients modulo [`MODULUS`]): `out[d N + j]`.
pub fn element_digits(coefficients: &[u128], out: &mut [i16]) {
    debug_assert!(coefficients.len() == N && out.len() == DIGITS * N);
    for j in 0..N / 2 {
        let (lo, hi) = lattice::digits_pair_limit(coefficients[j], coefficients[j + N / 2], SEARCH);
        for d in 0..DIGITS {
            out[d * N + j] = lo[d] as i16;
            out[d * N + j + N / 2] = hi[d] as i16;
        }
    }
}

/// `sum_i M(L)[m][i]^k` over the 512 witness coefficients `i` for `k = 1, 2`, added into `s1`,
/// `s2`, where `M(L)[m][i]` is coefficient `m` of `X^i L mod X^648 - X^324 + 1`: for `m >= 324`
/// `L[m - i] + L[m + 324 - i]`, for `m < 324` `L[m - i] - L[m + 648 - i] - L[m + 972 - i]` (`L`
/// zero outside `0..648`), a function of `x = m - i`, so both sums are prefix-sum windows.
fn moment_rows(l: &[i16], s1: &mut [i64], s2: &mut [i64]) {
    let at = |x: i64| -> i64 { if (0..N as i64).contains(&x) { l[x as usize] as i64 } else { 0 } };
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
        let (a, b) = (m, m + 512);
        if m >= N / 2 {
            s1[m] += fh[b] - fh[a];
            s2[m] += fh2[b] - fh2[a];
        } else {
            s1[m] += fl[b] - fl[a];
            s2[m] += fl2[b] - fl2[a];
        }
    }
}

/// `acc += W_d e` in `R`, `acc` and `e` by coefficient.
fn add_weighted(acc: &mut [i64], d: usize, e: &[i64]) {
    let (u, v) = lattice::WEIGHTS[d];
    let (u, v) = (u as i64, v as i64);
    for m in 0..N / 2 {
        acc[m] += u * e[m];
        acc[m + N / 2] += v * e[m];
    }
    for m in N / 2..N {
        acc[m - N / 2] -= v * e[m];
        acc[m] += (u + v) * e[m];
    }
}

/// The SMS matrix over `elements` elements, built on `workers` threads.
pub fn matrix(elements: usize, workers: usize) -> SmsMatrix {
    let p = P1;
    // coefficient rows: A mod p centered, then the six digit rows
    let mut raw = vec![0i16; ROWS * N * elements];
    for_blocks(&mut raw, ROWS * N, 1, workers, |first, block| {
        let mut c = vec![0u128; N];
        for (e, rows) in block.chunks_exact_mut(ROWS * N).enumerate() {
            crate::rq::ajtai_element(MODULUS, elements, first + e, &mut c);
            for (o, &v) in rows[..N].iter_mut().zip(&c) {
                *o = centered((v % p as u128) as i64, p);
            }
            element_digits(&c, &mut rows[N..]);
        }
    });
    // chunk moments
    let chunks = elements.div_ceil(CHUNK);
    let mut moments = vec![0i64; chunks * DIGITS * 2 * N];
    for_blocks(&mut moments, 2 * N, 1, workers, |first, block| {
        for (i, out) in block.chunks_exact_mut(2 * N).enumerate() {
            let (c, d) = ((first + i) / DIGITS, (first + i) % DIGITS);
            let (s1, s2) = out.split_at_mut(N);
            for m in c * CHUNK..((c + 1) * CHUNK).min(elements) {
                moment_rows(&raw[(ROWS * m + 1 + d) * N..][..N], s1, s2);
            }
        }
    });
    let half = ((p - 1) / 2) as f64;
    let mut margin = f64::INFINITY;
    let mut column_failure = 0f64;
    let mut mean_residues = vec![0i16; chunks * DIGITS * N];
    let mut mu = vec![0i64; N];
    for c in 0..chunks {
        for d in 0..DIGITS {
            let (s1, s2) = moments[(c * DIGITS + d) * 2 * N..][..2 * N].split_at(N);
            let mut mean = vec![0i64; N];
            for k in 0..N {
                // mean s1 / 2, variance s2 / 4
                mean[k] = s1[k].div_euclid(2);
                let sigma = (s2[k] as f64 / 4.0).sqrt();
                if sigma > 0.0 {
                    let z = (half - 0.5) / sigma;
                    margin = margin.min(z);
                    column_failure += 2.0 * (-z * z / 2.0).exp();
                }
                mean_residues[(c * DIGITS + d) * N + k] = centered(mean[k], p);
            }
            add_weighted(&mut mu, d, &mean);
        }
    }
    assert!(margin >= MIN_MARGIN, "SMS reconstruction margin {margin} below {MIN_MARGIN}");
    assert!(column_failure <= MAX_COLUMN_FAILURE, "SMS column failure bound {column_failure:e} above {MAX_COLUMN_FAILURE:e}");
    let mu = mu.into_iter().map(|v| v.rem_euclid(TAIL as i64) as u32).collect();
    // the transformed rows, eight elements per vector
    let mut data = vec![0i16; elements.div_ceil(LOCK) * N * ROWS * LOCK];
    let raw = &raw;
    for_blocks(&mut data, N * ROWS * LOCK, 1, workers, |first, block| {
        let mut lanes = vec![[0i16; 8]; N];
        let lanes: &mut [[i16; 8]; N] = (&mut lanes[..]).try_into().unwrap();
        for (g, batch) in block.chunks_exact_mut(N * ROWS * LOCK).enumerate() {
            let base = (first + g) * LOCK;
            let here = (elements - base).min(LOCK);
            for r in 0..ROWS {
                for (j, slot) in lanes.iter_mut().enumerate() {
                    for (e, s) in slot.iter_mut().enumerate() {
                        *s = if e < here { raw[(ROWS * (base + e) + r) * N + j] } else { 0 };
                    }
                }
                super::transform_general(PB, lanes);
                for (j, l) in lanes.iter().enumerate() {
                    for e in 0..here {
                        batch[(j * ROWS + r) * LOCK + e] = centered(l[e] as i64, p);
                    }
                }
            }
        }
    });
    use std::borrow::Cow::Owned;
    SmsMatrix { elements, data: Owned(data), mean_residues: Owned(mean_residues), mu: Owned(mu), margin, column_failure }
}

/// Elements transformed per pass of the slot-wise products over the accumulators.
const LOCK: usize = 8;

/// The accumulators of slot `s`, row `r`: `acc[idx..idx + 4]`, two columns per `i64` pair.
#[inline(always)]
const fn idx(s: usize, r: usize) -> usize {
    (s * ROWS + r) * 4
}

/// Four elements' lanes of one slot regrouped by column: `out[q]` holds columns `2 q` and
/// `2 q + 1`, the four elements of each in order.
#[inline(always)]
unsafe fn pack(a: int16x8_t, b: int16x8_t, c: int16x8_t, d: int16x8_t) -> [int16x8_t; 4] {
    let ab0 = vreinterpretq_s32_s16(vzip1q_s16(a, b));
    let ab1 = vreinterpretq_s32_s16(vzip2q_s16(a, b));
    let cd0 = vreinterpretq_s32_s16(vzip1q_s16(c, d));
    let cd1 = vreinterpretq_s32_s16(vzip2q_s16(c, d));
    [
        vreinterpretq_s16_s32(vzip1q_s32(ab0, cd0)),
        vreinterpretq_s16_s32(vzip2q_s32(ab0, cd0)),
        vreinterpretq_s16_s32(vzip1q_s32(ab1, cd1)),
        vreinterpretq_s16_s32(vzip2q_s32(ab1, cd1)),
    ]
}

/// `acc[q] += ` the four-term dot product per `i64` lane of `a[q]` with the low half of `key`, and
/// of `b[q]` with the high half: SVE's halfword `sdot`, eight products an instruction.
#[cfg(target_feature = "sve")]
#[inline(always)]
unsafe fn dot(acc: *mut int64x2_t, a: &[int16x8_t; 4], b: &[int16x8_t; 4], key: int16x8_t) {
    let k0 = vreinterpretq_s16_s64(vdupq_laneq_s64::<0>(vreinterpretq_s64_s16(key)));
    let k1 = vreinterpretq_s16_s64(vdupq_laneq_s64::<1>(vreinterpretq_s64_s16(key)));
    let (mut s0, mut s1, mut s2, mut s3) = (*acc, *acc.add(1), *acc.add(2), *acc.add(3));
    core::arch::asm!(
        "sdot z0.d, z4.h, z12.h", "sdot z1.d, z5.h, z12.h",
        "sdot z2.d, z6.h, z12.h", "sdot z3.d, z7.h, z12.h",
        "sdot z0.d, z8.h, z13.h", "sdot z1.d, z9.h, z13.h",
        "sdot z2.d, z10.h, z13.h", "sdot z3.d, z11.h, z13.h",
        inout("v0") s0, inout("v1") s1, inout("v2") s2, inout("v3") s3,
        in("v4") a[0], in("v5") a[1], in("v6") a[2], in("v7") a[3],
        in("v8") b[0], in("v9") b[1], in("v10") b[2], in("v11") b[3],
        in("v12") k0, in("v13") k1,
        options(pure, nomem, nostack, preserves_flags)
    );
    (*acc, *acc.add(1), *acc.add(2), *acc.add(3)) = (s0, s1, s2, s3);
}

/// The same sums without SVE, for development on other AArch64 hosts.
#[cfg(not(target_feature = "sve"))]
#[inline(always)]
unsafe fn dot(acc: *mut int64x2_t, a: &[int16x8_t; 4], b: &[int16x8_t; 4], key: int16x8_t) {
    let (k0, k1) = (vget_low_s16(key), vget_high_s16(key));
    for q in 0..4 {
        let pair = |v: int16x8_t, k: int16x4_t| {
            let (lo, hi) = (vmull_s16(vget_low_s16(v), k), vmull_s16(vget_high_s16(v), k));
            vcombine_s64(vdup_n_s64(vaddlvq_s32(lo)), vdup_n_s64(vaddlvq_s32(hi)))
        };
        *acc.add(q) = vaddq_s64(vaddq_s64(*acc.add(q), pair(a[q], k0)), pair(b[q], k1));
    }
}

/// `acc(s, r) += row_r(e)[s] s_e[s]` for the eight transforms of `-s_e` in `xs`, every slot and
/// row, with `keys` the tile's matrix vectors: each accumulator is read and written once.
#[inline(never)]
unsafe fn tile(xs: &[[int16x8_t; N]; LOCK], keys: *const i16, acc: &mut [int64x2_t]) {
    debug_assert!(acc.len() == ROWS * 4 * N);
    let accp = acc.as_mut_ptr();
    for s in 0..N {
        let a = pack(xs[0][s], xs[1][s], xs[2][s], xs[3][s]);
        let b = pack(xs[4][s], xs[5][s], xs[6][s], xs[7][s]);
        for r in 0..ROWS {
            let key = vnegq_s16(vld1q_s16(keys.add((s * ROWS + r) * LOCK)));
            dot(accp.add(idx(s, r)), &a, &b, key);
        }
    }
}

/// The centered residue of a sum below `2^41` in magnitude. Binary64 holds such a sum exactly and
/// its quotient by 9721 to far better than half a unit, so the rounded quotient is off by at most
/// one half step: `|result| <= 4861`.
#[inline(always)]
unsafe fn narrow_sum(v: int64x2_t) -> int32x2_t {
    const _: () = assert!(P1 == 9721);
    let q = vcvtnq_s64_f64(vmulq_n_f64(vcvtq_f64_s64(v), 1.0 / P1 as f64));
    vmovn_s64(vsubq_s64(v, vmull_n_s32(vmovn_s64(q), P1 as i32)))
}

/// Row `r`'s sums as residues, one `i16` vector per slot, lane = column.
#[inline(always)]
unsafe fn narrow(acc: &[int64x2_t], r: usize, out: &mut [int16x8_t; N]) {
    for (s, o) in out.iter_mut().enumerate() {
        let at = idx(s, r);
        let lo = vcombine_s32(narrow_sum(acc[at]), narrow_sum(acc[at + 1]));
        let hi = vcombine_s32(narrow_sum(acc[at + 2]), narrow_sum(acc[at + 3]));
        *o = vcombine_s16(vmovn_s32(lo), vmovn_s32(hi));
    }
}

/// `x` in `[-(p-1)/2, (p-1)/2]` for `|x| <= p + p/2`.
#[inline(always)]
unsafe fn canonical(x: int16x8_t, p: i16) -> int16x8_t {
    let h = vdupq_n_s16((p - 1) / 2);
    let pv = vdupq_n_s16(p);
    let over = vcgtq_s16(x, h);
    let under = vcltq_s16(x, vnegq_s16(h));
    let x = vsubq_s16(x, vandq_s16(vreinterpretq_s16_u16(over), pv));
    vaddq_s16(x, vandq_s16(vreinterpretq_s16_u16(under), pv))
}

/// The batch's pass: each chunk's digit sums inverted to canonical coefficients, handed to
/// `chunk(c, d, coefficients)`; on return `y` holds the base row's coefficients.
#[inline(never)]
unsafe fn sums(
    words: &[(*const u8, usize); LANES],
    a: &SmsMatrix,
    m: &mut [int16x8_t; BITS],
    xs: &mut [[int16x8_t; N]; LOCK],
    acc: &mut [int64x2_t],
    y: &mut [int16x8_t; N],
    mut chunk: impl FnMut(usize, usize, &mut [int16x8_t; N]),
) {
    let kern = &kernels()[PB];
    acc.fill(vdupq_n_s64(0));
    let mut e = 0usize;
    while e < a.elements {
        let end = ((e / CHUNK + 1) * CHUNK).min(a.elements);
        while e < end {
            // A short tile only occurs at the end of the column, where the missing elements' key
            // lanes are zero, so the stale transforms of those lanes contribute nothing.
            for i in 0..LOCK.min(end - e) {
                let at: [*const u8; LANES] = core::array::from_fn(|c| words[c].0.add((e + i) * words[c].1));
                masks(&at, m);
                let x = &mut xs[i];
                pass_a(m, x, kern);
                pass_b::<PB>(x, kern);
                pass_c::<PB>(x, kern);
                pass_d::<PB, false>(x, kern, &[], &mut []);
            }
            tile(xs, a.data.as_ptr().add(e / LOCK * N * ROWS * LOCK), acc);
            e += LOCK;
        }
        e = end;
        for d in 0..DIGITS {
            narrow(acc, 1 + d, y);
            super::inverse::<PB>(y);
            chunk((end - 1) / CHUNK, d, y);
        }
        for s in 0..N {
            for d in 0..DIGITS {
                let at = idx(s, 1 + d);
                acc[at..at + 4].fill(vdupq_n_s64(0));
            }
        }
    }
    narrow(acc, 0, y);
    super::inverse::<PB>(y);
}

/// `r mod 9721` and `t mod M` into `[0, Q)`.
#[inline]
fn crt(r: u32, t: u32) -> u128 {
    const INV: u64 = {
        // 9721^-1 mod M by the extended Euclidean algorithm
        let (mut a, mut b) = (P1 as i64, TAIL as i64);
        let (mut x0, mut x1) = (1i64, 0i64);
        while b != 0 {
            let q = a / b;
            (a, b) = (b, a - q * b);
            (x0, x1) = (x1, x0 - q * x1);
        }
        assert!(a == 1);
        x0.rem_euclid(TAIL as i64) as u64
    };
    const _: () = assert!(P1 as u64 * INV % TAIL as u64 == 1);
    let k = (t as u64 + TAIL as u64 - r as u64 % TAIL as u64) % TAIL as u64 * INV % TAIL as u64;
    r as u128 + P1 as u128 * k as u128
}

/// The commitment of every column of `values` (columns of `4 a.elements` words) modulo
/// [`MODULUS`].
pub fn commit(values: &[B128], a: &SmsMatrix, workers: usize) -> Vec<u128> {
    let len = 4 * a.elements;
    assert_eq!(values.len() % len, 0);
    let columns = values.len() / len;
    let mut out = vec![0u128; columns * N];
    for_blocks(&mut out, N, LANES, workers, |first, block| unsafe {
        let mut m = Box::new([vdupq_n_s16(0); BITS]);
        let mut xs = Box::new([[vdupq_n_s16(0); N]; LOCK]);
        let mut y = Box::new([vdupq_n_s16(0); N]);
        let mut acc = vec![vdupq_n_s64(0); ROWS * 4 * N];
        // per digit, the chunk deviations S - mean summed over the chunks
        let mut dev = vec![vdupq_n_s32(0); DIGITS * 2 * N];
        let p = P1 as i16;
        let count = block.len() / N;
        for (b, target) in block.chunks_mut(LANES * N).enumerate() {
            let col0 = first + LANES * b;
            let here = (count - LANES * b).min(LANES);
            let words: [(*const u8, usize); LANES] = core::array::from_fn(|c| {
                if c < here {
                    (values[(col0 + c) * len..].as_ptr() as *const u8, 64)
                } else {
                    (ZERO_WORDS.as_ptr() as *const u8, 0)
                }
            });
            dev.fill(vdupq_n_s32(0));
            sums(&words, a, &mut m, &mut xs, &mut acc, &mut y, |c, d, r| {
                let mr = a.mean_residues(c, d);
                let dv = &mut dev[d * 2 * N..][..2 * N];
                for s in 0..N {
                    let v = canonical(vsubq_s16(r[s], vdupq_n_s16(mr[s])), p);
                    dv[2 * s] = vaddw_s16(dv[2 * s], vget_low_s16(v));
                    dv[2 * s + 1] = vaddw_high_s16(dv[2 * s + 1], v);
                }
            });
            let base: &[[i16; LANES]; N] = &*(y.as_ptr() as *const [[i16; LANES]; N]);
            let devs: &[[i32; 4]] = core::slice::from_raw_parts(dev.as_ptr() as *const [i32; 4], dev.len());
            for (c, column) in target.chunks_exact_mut(N).enumerate() {
                let mut tail = [0i64; N];
                for (t, &v) in tail.iter_mut().zip(a.mu.iter()) {
                    *t = v as i64;
                }
                let mut e = [0i64; N];
                for d in 0..DIGITS {
                    for (k, v) in e.iter_mut().enumerate() {
                        *v = devs[d * 2 * N + 2 * k + c / 4][c % 4] as i64;
                    }
                    add_weighted(&mut tail, d, &e);
                }
                for (k, o) in column.iter_mut().enumerate() {
                    let r = (base[k][c] as i32).rem_euclid(P1 as i32) as u32;
                    *o = crt(r, tail[k].rem_euclid(TAIL as i64) as u32);
                }
            }
        }
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `sum_d W_d L_d = A (mod M)` on random matrix elements, coefficient by coefficient.
    #[test]
    fn digits_reconstruct_the_element() {
        let mut c = vec![0u128; N];
        let mut digits = vec![0i16; DIGITS * N];
        for index in [0usize, 1, 977, 4095] {
            crate::rq::ajtai_element(MODULUS, 4096, index, &mut c);
            element_digits(&c, &mut digits);
            assert!(digits.iter().all(|v| v.abs() <= 32));
            let mut acc = vec![0i64; N];
            for d in 0..DIGITS {
                let e: Vec<i64> = digits[d * N..][..N].iter().map(|&v| v as i64).collect();
                add_weighted(&mut acc, d, &e);
            }
            for k in 0..N {
                assert_eq!(acc[k].rem_euclid(TAIL as i64) as u128, c[k] % TAIL as u128, "element {index} coefficient {k}");
            }
        }
    }

    /// The moments against the ring product of `L` by every unit `X^i`, `i < 512`.
    #[test]
    fn moments_match_the_ring_map() {
        let mut state = 3u64;
        let l: Vec<i16> = (0..N).map(|_| (crate::ntt::tests::xorshift(&mut state) % 65) as i16 - 32).collect();
        let (mut s1, mut s2) = (vec![0i64; N], vec![0i64; N]);
        moment_rows(&l, &mut s1, &mut s2);
        let (mut w1, mut w2) = (vec![0i64; N], vec![0i64; N]);
        for i in 0..512 {
            // X^i L reduced with X^648 = X^324 - 1
            let mut raw = vec![0i64; 2 * N];
            for (x, &v) in l.iter().enumerate() {
                raw[i + x] += v as i64;
            }
            for k in (N..2 * N).rev() {
                let v = raw[k];
                raw[k] = 0;
                raw[k - N / 2] += v;
                raw[k - N] -= v;
            }
            for m in 0..N {
                w1[m] += raw[m];
                w2[m] += raw[m] * raw[m];
            }
        }
        assert_eq!(s1, w1);
        assert_eq!(s2, w2);
    }
}

#[cfg(test)]
mod timing {
    use super::*;

    /// Phase costs of [`matrix`] on one thread: `cargo test --release sms_phase_timing --
    /// --ignored --nocapture`.
    #[test]
    #[ignore]
    fn sms_phase_timing() {
        let n = 256;
        let mut c = vec![0u128; N];
        let mut digits = vec![0i16; DIGITS * N];
        let t = std::time::Instant::now();
        for i in 0..n {
            crate::rq::ajtai_element(MODULUS, 4096, i, &mut c);
        }
        let ajtai = t.elapsed();
        let t = std::time::Instant::now();
        for i in 0..n {
            crate::rq::ajtai_element(MODULUS, 4096, i, &mut c);
            element_digits(&c, &mut digits);
        }
        let both = t.elapsed();
        let t = std::time::Instant::now();
        let (mut s1, mut s2) = (vec![0i64; N], vec![0i64; N]);
        for _ in 0..n * DIGITS {
            moment_rows(&digits[..N], &mut s1, &mut s2);
        }
        let mom = t.elapsed();
        let t = std::time::Instant::now();
        let mut lanes = [[0i16; 8]; N];
        for _ in 0..n * ROWS / 8 {
            super::super::transform_general(PB, &mut lanes);
        }
        let tr = t.elapsed();
        // search limit against digit size and the margin at several chunk sizes
        let elements = 4096;
        let all: Vec<Vec<u128>> = (0..elements).map(|i| { let mut c = vec![0u128; N]; crate::rq::ajtai_element(MODULUS, elements, i, &mut c); c }).collect();
        for limit in [1usize, 4, 16, 64, 512] {
            let t = std::time::Instant::now();
            let mut raw = vec![0i16; elements * DIGITS * N];
            let mut maxd = 0;
            std::thread::scope(|sc| {
                for (w, block) in raw.chunks_mut(elements / 8 * DIGITS * N).enumerate() {
                    let all = &all;
                    sc.spawn(move || {
                        for (e, out) in block.chunks_exact_mut(DIGITS * N).enumerate() {
                            let c = &all[w * elements / 8 + e];
                            for j in 0..N / 2 {
                                let (lo, hi) = lattice::digits_pair_limit(c[j], c[j + N / 2], limit);
                                for d in 0..DIGITS {
                                    out[d * N + j] = lo[d] as i16;
                                    out[d * N + j + N / 2] = hi[d] as i16;
                                }
                            }
                        }
                    });
                }
            });
            let el = t.elapsed();
            for v in &raw { maxd = maxd.max(v.abs()); }
            let rms = (raw.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / raw.len() as f64).sqrt();
            let mut out = format!("limit {limit}: {el:?} (8 threads) max {maxd} rms {rms:.2}");
            for chunk in [256usize, 128, 64] {
                let mut margin = f64::INFINITY;
                for c in 0..elements / chunk {
                    for d in 0..DIGITS {
                        let (mut s1, mut s2) = (vec![0i64; N], vec![0i64; N]);
                        for m in c * chunk..(c + 1) * chunk {
                            moment_rows(&raw[(m * DIGITS + d) * N..][..N], &mut s1, &mut s2);
                        }
                        for k in 0..N {
                            let sigma = (s2[k] as f64 / 4.0).sqrt();
                            if sigma > 0.0 { margin = margin.min((4860.0 - 0.5) / sigma); }
                        }
                    }
                }
                out += &format!(" z{chunk}={margin:.2}");
            }
            println!("{out}");
        }
        println!("per element: ajtai {:?} digits {:?} moments {:?} transforms {:?}", ajtai / n as u32, (both - ajtai) / n as u32, mom / n as u32, tr / n as u32);
    }
}
