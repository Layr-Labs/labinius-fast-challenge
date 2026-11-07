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
pub const CHUNK: usize = 256;
const _: () = assert!(CHUNK % LOCK == 0);
/// Nodes of the encoder's closest-vector search, the x86 build's setting: it runs at build time
/// only, and the smaller digit spread (rms 3.55 against Babai's 3.88) allows longer chunks.
/// [`matrix`] computes the smallest margin and the Hoeffding union bound over every chunk,
/// digit and coefficient of a column from the public matrix and refuses a weaker one; `build.rs`
/// prints both for the [`CHUNK`] in use.
const SEARCH: usize = 512;
/// The smallest admissible reconstruction margin, in standard deviations.
pub const MIN_MARGIN: f64 = 7.5;
/// The largest admissible failure bound of one column (`sum 2 exp(-z^2 / 2)`, Hoeffding, over
/// every chunk, digit and coefficient): 1e-7, i.e. below 1e-4 for a proof's 1024 columns. A failed
/// recombination gives a wrong commitment, which the verifier rejects.
pub const MAX_COLUMN_FAILURE: f64 = 1e-7;
/// Unit terms of one reduced product coefficient: a witness coefficient `i < 512` times a
/// coefficient `x < 648` folds onto coefficient `i + x` and, for `i + x >= 648`, two more.
const TERMS: i64 = 836;
// every chunk digit sum and its mean fit in i32 with room: |S| <= CHUNK * TERMS * 32
const _: () = assert!((CHUNK as i64) * TERMS * (lattice::MAX_DIGIT_ABS as i64) * 4 < i32::MAX as i64);

/// The SMS matrix: in the transform domain, slot `j` of row `r` of element `m` is
/// `data[(((m / 8) N + j) ROWS + r) 8 + m % 8]`, centered residues, so that one vector holds the
/// eight elements of a tile, each residue negated because the forward kernel transforms `-s` and
/// multiplied by [`KEY_SCALE`] `= 324^-1` so that the inverses need no `1 / r`; per chunk and digit
/// the centered residue
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

/// The key image copied out of the binary's read-only 4 KiB pages into anonymous memory, which
/// the allocator (`hugealloc`, 4 MiB and up) maps 2 MiB-aligned with `MADV_HUGEPAGE`: about 19
/// TLB entries for the 37 MB stream instead of 9,000, prefetch across page boundaries, and no
/// dependence on the page cache. Started at process start by [`prefetch_key`] (the copy overlaps
/// the table read); [`SmsMatrix::from_images`] waits for it and uses it when it matches the image.
/// The source image (its address) and the copy.
static KEY: std::sync::OnceLock<(usize, Vec<u8>)> = std::sync::OnceLock::new();
static KEY_JOB: std::sync::Mutex<Option<std::thread::JoinHandle<()>>> = std::sync::Mutex::new(None);

/// Starts copying `image` (the SMS key data image) into [`KEY`] on four threads. With
/// `LABINIUS_KEY_IN_PLACE` set the key is read in place instead.
pub fn prefetch_key(image: &'static [u8]) {
    if image.is_empty() || std::env::var_os("LABINIUS_KEY_IN_PLACE").is_some() {
        return;
    }
    let job = std::thread::spawn(move || {
        let len = image.len();
        let mut v: Vec<u8> = Vec::with_capacity(len);
        let dst = v.as_mut_ptr() as usize;
        // pieces of whole 2 MiB pages, so that each thread faults in its own huge pages
        let piece = len.div_ceil(4).next_multiple_of(2 << 20);
        std::thread::scope(|s| {
            for start in (0..len).step_by(piece) {
                let n = piece.min(len - start);
                s.spawn(move || {
                    // SAFETY: disjoint ranges of the capacity and of the image.
                    unsafe { core::ptr::copy_nonoverlapping(image.as_ptr().add(start), (dst as *mut u8).add(start), n) }
                });
            }
        });
        // SAFETY: every byte below `len` was written above.
        unsafe { v.set_len(len) };
        let _ = KEY.set((image.as_ptr() as usize, v));
    });
    *KEY_JOB.lock().unwrap() = Some(job);
}

/// The copy of [`prefetch_key`] of `image`, once it is done (waiting for it), if one was started.
fn key_copy(image: &'static [u8]) -> Option<&'static [u8]> {
    if let Some(job) = KEY_JOB.lock().unwrap().take() {
        let _ = job.join();
    }
    KEY.get().filter(|(src, v)| *src == image.as_ptr() as usize && v.len() == image.len()).map(|(_, v)| &v[..])
}

impl SmsMatrix {
    /// The matrix from build-time images of [`matrix`]`(elements)`: `data`, `mean_residues`
    /// (i16) and `mu` (u32), little-endian, and `meta` = margin and column failure bound (f64).
    pub fn from_images(data: &'static [u8], means: &'static [u8], mu: &'static [u8], meta: &'static [u8], elements: usize) -> Option<SmsMatrix> {
        let chunks = elements.div_ceil(CHUNK);
        let data = key_copy(data).unwrap_or(data);
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
                        batch[(j * ROWS + r) * LOCK + e] = -centered(l[e] as i64 * KEY_SCALE, p);
                    }
                }
            }
        }
    });
    use std::borrow::Cow::Owned;
    SmsMatrix { elements, data: Owned(data), mean_residues: Owned(mean_residues), mu: Owned(mu), margin, column_failure }
}

/// `324^-1 mod p` (`324 = 2^2 3^4`, [`super::FOLD`], the product of the radices), carried by every
/// transformed key row: the slot sums then carry it too, and [`super::inverse_folded`] (no `1 / r`)
/// inverts them to the same coefficients as [`super::inverse`] of the unscaled sums. The
/// certificate's moments, the means and `mu` are of the coefficient rows and do not change; only
/// the key image does.
const KEY_SCALE: i64 = crate::ntt::pow_mod(super::FOLD as u64, P1 as u64 - 2, P1 as u64) as i64;
const _: () = assert!(KEY_SCALE * super::FOLD as i64 % P1 as i64 == 1 && KEY_SCALE == 9691);

/// Elements per vector of the key image, and of a tile of the slot-wise products.
const LOCK: usize = 8;
/// Elements transformed per pass over the accumulators: two tiles, so that each accumulator is
/// read and written once per two tiles.
const PASS: usize = 2 * LOCK;
const _: () = assert!(CHUNK % PASS == 0);

/// The accumulators of slot `s`, row `r`: `acc[idx..idx + 4]`, two columns per `i64` pair.
#[inline(always)]
const fn idx(s: usize, r: usize) -> usize {
    (s * ROWS + r) * 4
}

/// `acc[q] += ` the four-term dot product per `i64` lane of `a[q]` with the low half of `key`, and
/// of `b[q]` with the high half: SVE's indexed halfword `sdot`, eight products an instruction.
/// With SVE the sums go through [`slot_sink`]; this is the reference for its operand order.
#[cfg(target_feature = "sve")]
#[allow(dead_code)]
#[inline(always)]
unsafe fn dot(acc: *mut int64x2_t, a: &[int16x8_t; 4], b: &[int16x8_t; 4], key: int16x8_t) {
    let (mut s0, mut s1, mut s2, mut s3) = (*acc, *acc.add(1), *acc.add(2), *acc.add(3));
    core::arch::asm!(
        "sdot z0.d, z4.h, z12.h[0]", "sdot z1.d, z5.h, z12.h[0]",
        "sdot z2.d, z6.h, z12.h[0]", "sdot z3.d, z7.h, z12.h[0]",
        "sdot z0.d, z8.h, z12.h[1]", "sdot z1.d, z9.h, z12.h[1]",
        "sdot z2.d, z10.h, z12.h[1]", "sdot z3.d, z11.h, z12.h[1]",
        inout("v0") s0, inout("v1") s1, inout("v2") s2, inout("v3") s3,
        in("v4") a[0], in("v5") a[1], in("v6") a[2], in("v7") a[3],
        in("v8") b[0], in("v9") b[1], in("v10") b[2], in("v11") b[3],
        in("v12") key,
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

/// Key groups (tiles of [`LOCK`] elements) per sweep of [`sink`] over the accumulators.
const SINK: usize = 4;
/// Elements transformed per sweep: up to four tiles, each accumulator read and written once.
const SWEEP: usize = SINK * LOCK;
const _: () = assert!(CHUNK % SWEEP == 0 && SWEEP == 2 * PASS);

/// [`slot_sink`]'s body; `$rows13` and `$rows46` start rows 1-3 (v20-v31) and rows 4-6
/// (v16-v27): loaded from `acc`, or zeroed for a chunk's first sweep ([`slot_sink_fresh`]).
#[cfg(target_feature = "sve")]
macro_rules! slot_sink_body {
    ($acc:expr, $z:expr, $k:expr, $groups:expr, [$($rows13:literal),*], [$($rows46:literal),*]) => {
        core::arch::asm!(
        "ldp q16, q17, [{a}, #0]",
        "ldp q18, q19, [{a}, #32]",
        $($rows13,)*
        "mov {zp}, {z}",
        "mov {kp}, {k}",
        "mov {n}, {g}",
        "2:",
        "ldp q0, q1, [{kp}]",
        "ldp q2, q3, [{kp}, #32]",
        "ldr q4, [{zp}]",
        "ldr q5, [{zp}, #{xs1}]",
        "ldr q6, [{zp}, #{xs2}]",
        "ldr q7, [{zp}, #{xs3}]",
        "add {zp}, {zp}, {xs4}",
        "sdot z16.d, z4.h, z0.h[0]", "sdot z17.d, z5.h, z0.h[0]", "sdot z18.d, z6.h, z0.h[0]", "sdot z19.d, z7.h, z0.h[0]",
        "sdot z20.d, z4.h, z1.h[0]", "sdot z21.d, z5.h, z1.h[0]", "sdot z22.d, z6.h, z1.h[0]", "sdot z23.d, z7.h, z1.h[0]",
        "sdot z24.d, z4.h, z2.h[0]", "sdot z25.d, z5.h, z2.h[0]", "sdot z26.d, z6.h, z2.h[0]", "sdot z27.d, z7.h, z2.h[0]",
        "sdot z28.d, z4.h, z3.h[0]", "sdot z29.d, z5.h, z3.h[0]", "sdot z30.d, z6.h, z3.h[0]", "sdot z31.d, z7.h, z3.h[0]",
        "ldr q4, [{zp}]",
        "ldr q5, [{zp}, #{xs1}]",
        "ldr q6, [{zp}, #{xs2}]",
        "ldr q7, [{zp}, #{xs3}]",
        "add {zp}, {zp}, {xs4}",
        "add {kp}, {kp}, {ks}",
        "sdot z16.d, z4.h, z0.h[1]", "sdot z17.d, z5.h, z0.h[1]", "sdot z18.d, z6.h, z0.h[1]", "sdot z19.d, z7.h, z0.h[1]",
        "sdot z20.d, z4.h, z1.h[1]", "sdot z21.d, z5.h, z1.h[1]", "sdot z22.d, z6.h, z1.h[1]", "sdot z23.d, z7.h, z1.h[1]",
        "sdot z24.d, z4.h, z2.h[1]", "sdot z25.d, z5.h, z2.h[1]", "sdot z26.d, z6.h, z2.h[1]", "sdot z27.d, z7.h, z2.h[1]",
        "sdot z28.d, z4.h, z3.h[1]", "sdot z29.d, z5.h, z3.h[1]", "sdot z30.d, z6.h, z3.h[1]", "sdot z31.d, z7.h, z3.h[1]",
        "subs {n}, {n}, #1",
        "b.ne 2b",
        "stp q16, q17, [{a}, #0]",
        "stp q18, q19, [{a}, #32]",
        "stp q20, q21, [{a}, #64]",
        "stp q22, q23, [{a}, #96]",
        "stp q24, q25, [{a}, #128]",
        "stp q26, q27, [{a}, #160]",
        "stp q28, q29, [{a}, #192]",
        "stp q30, q31, [{a}, #224]",
        $($rows46,)*
        "mov {zp}, {z}",
        "mov {kp}, {k}",
        "mov {n}, {g}",
        "3:",
        "ldp q0, q1, [{kp}, #64]",
        "ldr q2, [{kp}, #96]",
        "ldr q4, [{zp}]",
        "ldr q5, [{zp}, #{xs1}]",
        "ldr q6, [{zp}, #{xs2}]",
        "ldr q7, [{zp}, #{xs3}]",
        "add {zp}, {zp}, {xs4}",
        "sdot z16.d, z4.h, z0.h[0]", "sdot z17.d, z5.h, z0.h[0]", "sdot z18.d, z6.h, z0.h[0]", "sdot z19.d, z7.h, z0.h[0]",
        "sdot z20.d, z4.h, z1.h[0]", "sdot z21.d, z5.h, z1.h[0]", "sdot z22.d, z6.h, z1.h[0]", "sdot z23.d, z7.h, z1.h[0]",
        "sdot z24.d, z4.h, z2.h[0]", "sdot z25.d, z5.h, z2.h[0]", "sdot z26.d, z6.h, z2.h[0]", "sdot z27.d, z7.h, z2.h[0]",
        "ldr q4, [{zp}]",
        "ldr q5, [{zp}, #{xs1}]",
        "ldr q6, [{zp}, #{xs2}]",
        "ldr q7, [{zp}, #{xs3}]",
        "add {zp}, {zp}, {xs4}",
        "add {kp}, {kp}, {ks}",
        "sdot z16.d, z4.h, z0.h[1]", "sdot z17.d, z5.h, z0.h[1]", "sdot z18.d, z6.h, z0.h[1]", "sdot z19.d, z7.h, z0.h[1]",
        "sdot z20.d, z4.h, z1.h[1]", "sdot z21.d, z5.h, z1.h[1]", "sdot z22.d, z6.h, z1.h[1]", "sdot z23.d, z7.h, z1.h[1]",
        "sdot z24.d, z4.h, z2.h[1]", "sdot z25.d, z5.h, z2.h[1]", "sdot z26.d, z6.h, z2.h[1]", "sdot z27.d, z7.h, z2.h[1]",
        "subs {n}, {n}, #1",
        "b.ne 3b",
        "stp q16, q17, [{a}, #256]",
        "stp q18, q19, [{a}, #288]",
        "stp q20, q21, [{a}, #320]",
        "stp q22, q23, [{a}, #352]",
        "stp q24, q25, [{a}, #384]",
        "stp q26, q27, [{a}, #416]",
            a = in(reg) $acc,
            z = in(reg) $z,
            k = in(reg) $k,
            g = in(reg) $groups,
            xs4 = in(reg) 4 * XS,
            ks = in(reg) KS,
            xs1 = const XS,
            xs2 = const 2 * XS,
            xs3 = const 3 * XS,
            zp = out(reg) _,
            kp = out(reg) _,
            n = out(reg) _,
            out("v0") _, out("v1") _, out("v2") _, out("v3") _,
            out("v4") _, out("v5") _, out("v6") _, out("v7") _,
            out("v16") _, out("v17") _, out("v18") _, out("v19") _,
            out("v20") _, out("v21") _, out("v22") _, out("v23") _,
            out("v24") _, out("v25") _, out("v26") _, out("v27") _,
            out("v28") _, out("v29") _, out("v30") _, out("v31") _,
            options(nostack)
        )
    };
}

/// The register-resident sink for one slot (from 4381e86b's `sve::slot_sink`, after arm-big's
/// `slot_chunk`): `acc(s, r) +=` the slot's products summed over `groups` key tiles (eight
/// elements each) of transforms starting at `z` (transform stride `XS`, the [`pass_words`] lane
/// order), keys starting at `k` (tile stride `KS`). Rows 0-3, then rows 4-6, keep their sums in
/// v16-v31 across all the tiles: one load and one store per sum per sweep. The key halves are
/// `sdot`'s indexed operand exactly as in [`dot`]: transforms `0..4` of a tile with key elements
/// `0..4` (`[0]`), transforms `4..8` with `4..8` (`[1]`). `groups >= 1`.
#[cfg(target_feature = "sve")]
#[inline(never)]
unsafe fn slot_sink(z: *const int16x8_t, k: *const i16, acc: *mut int64x2_t, groups: usize) {
    const XS: usize = N * 16;
    const KS: usize = N * ROWS * LOCK * 2;
    debug_assert!((1..=SINK).contains(&groups));
    slot_sink_body!(acc, z, k, groups, ["ldp q20, q21, [{a}, #64]", "ldp q22, q23, [{a}, #96]", "ldp q24, q25, [{a}, #128]", "ldp q26, q27, [{a}, #160]", "ldp q28, q29, [{a}, #192]", "ldp q30, q31, [{a}, #224]"], ["ldp q16, q17, [{a}, #256]", "ldp q18, q19, [{a}, #288]", "ldp q20, q21, [{a}, #320]", "ldp q22, q23, [{a}, #352]", "ldp q24, q25, [{a}, #384]", "ldp q26, q27, [{a}, #416]"]);
}

/// [`slot_sink`] for a chunk's first sweep: the digit rows' sums (rows 1-6) start from zero instead
/// of being loaded, and are stored as usual, so the chunk end needs no zeroing pass; the base
/// row (row 0) accumulates as in [`slot_sink`].
#[cfg(target_feature = "sve")]
#[inline(never)]
unsafe fn slot_sink_fresh(z: *const int16x8_t, k: *const i16, acc: *mut int64x2_t, groups: usize) {
    const XS: usize = N * 16;
    const KS: usize = N * ROWS * LOCK * 2;
    debug_assert!((1..=SINK).contains(&groups));
    slot_sink_body!(acc, z, k, groups, ["movi v20.16b, #0", "movi v21.16b, #0", "movi v22.16b, #0", "movi v23.16b, #0", "movi v24.16b, #0", "movi v25.16b, #0", "movi v26.16b, #0", "movi v27.16b, #0", "movi v28.16b, #0", "movi v29.16b, #0", "movi v30.16b, #0", "movi v31.16b, #0"], ["movi v16.16b, #0", "movi v17.16b, #0", "movi v18.16b, #0", "movi v19.16b, #0", "movi v20.16b, #0", "movi v21.16b, #0", "movi v22.16b, #0", "movi v23.16b, #0", "movi v24.16b, #0", "movi v25.16b, #0", "movi v26.16b, #0", "movi v27.16b, #0"]);
}

/// `acc +=` the products of the first `groups` tiles of `xs` (transforms `8 g..8 g + 8` of tile
/// `g`) with the key tiles at `keys` (`N ROWS LOCK` entries apart), every slot and row; the sums of
/// [`dot`] per tile, in another order. A tile past the column's end has zero keys. With `FRESH` (a
/// chunk's first sweep) the digit rows' sums are overwritten instead of added to, so the chunk end
/// needs no zeroing pass; the base row always accumulates.
#[inline(never)]
unsafe fn sink<const FRESH: bool>(xs: &[[int16x8_t; N]; SWEEP], keys: *const i16, acc: &mut [int64x2_t], groups: usize) {
    debug_assert!(acc.len() == ROWS * 4 * N && (1..=SINK).contains(&groups));
    let accp = acc.as_mut_ptr();
    #[cfg(target_feature = "sve")]
    {
        let z = xs.as_ptr() as *const int16x8_t;
        for s in 0..N {
            if FRESH {
                slot_sink_fresh(z.add(s), keys.add(s * ROWS * LOCK), accp.add(idx(s, 0)), groups);
            } else {
                slot_sink(z.add(s), keys.add(s * ROWS * LOCK), accp.add(idx(s, 0)), groups);
            }
        }
    }
    #[cfg(not(target_feature = "sve"))]
    for g in 0..groups {
        let tile = keys.add(g * N * ROWS * LOCK);
        for s in 0..N {
            let y: [int16x8_t; LOCK] = core::array::from_fn(|i| *xs[LOCK * g + i].get_unchecked(s));
            let (a, b): (&[int16x8_t; 4], &[int16x8_t; 4]) = (&*(y.as_ptr() as *const _), &*(y.as_ptr().add(4) as *const _));
            for r in 0..ROWS {
                if FRESH && g == 0 && r > 0 {
                    for q in 0..4 {
                        *accp.add(idx(s, r) + q) = vdupq_n_s64(0);
                    }
                }
                dot(accp.add(idx(s, r)), a, b, vld1q_s16(tile.add((s * ROWS + r) * LOCK)));
            }
        }
    }
}

/// The eight witness pointers of transform `t` of a pass starting at element `e`, in dot-product
/// lane order: lane `4 l + k` of transform `4 h + q` is column `2 q + l` of element `e + 4 h + k`.
/// Each lane is an independent transform, so a 64-bit lane then holds one column's four elements
/// and `sdot` reads it directly, with no transpose. Elements past the column read as zero.
#[inline(always)]
unsafe fn pass_words(words: &[(*const u8, usize); LANES], elements: usize, e: usize, t: usize) -> [*const u8; LANES] {
    let (h, q) = (t / 4, t % 4);
    core::array::from_fn(|lane| {
        let (l, k) = (lane / 4, lane % 4);
        let element = e + 4 * h + k;
        let (w, stride) = words[2 * q + l];
        if element < elements { w.add(element * stride) } else { ZERO_WORDS.as_ptr() as *const u8 }
    })
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

/// The largest product of a key slot (a centered residue) and a transform slot.
const PRODUCT: u64 = (P1 as u64 - 1) / 2 * crate::ntt::lut_transform_bound(P1, SCHEDULES[PB]) as u64;
/// `2^16 mod p`, centered.
const C16: i32 = centered(1 << 16, P1) as i32;
/// `round(2^31 / p)`.
const V31: i32 = ((1u64 << 32) / P1 as u64).div_ceil(2) as i32;
// a chunk's digit sums `v` (|v| <= CHUNK PRODUCT): `(v >> 16) C16 + (v mod 2^16)` fits i32
const _: () = assert!(((CHUNK as u64 * PRODUCT >> 16) + 1) * C16.unsigned_abs() as u64 + 65535 < i32::MAX as u64);

/// [`narrow`] for a chunk's digit rows, in integers: `v = h 2^16 + l`, `l = v mod 2^16`, so
/// `v = h (2^16 mod p) + l (mod p)` within `i32`, then a 32-bit Barrett reduction to
/// `|r| <= 0.65 p` (the inverse takes `|x| <= p`). After the k6 chunk end (dadon line).
#[inline(always)]
unsafe fn narrow_digits(acc: &[int64x2_t], r: usize, out: &mut [int16x8_t; N]) {
    for (s, o) in out.iter_mut().enumerate() {
        *o = narrow_digit(acc, s, r);
    }
}

/// Slot `s` of [`narrow_digits`].
#[inline(always)]
unsafe fn narrow_digit(acc: &[int64x2_t], s: usize, r: usize) -> int16x8_t {
    let c = vdupq_n_s32(C16);
    let low = vdupq_n_s32(0xFFFF);
    let half = |a: int64x2_t, b: int64x2_t| -> int32x4_t {
        let h = vshrn_high_n_s64::<16>(vshrn_n_s64::<16>(a), b);
        let l = vandq_s32(vmovn_high_s64(vmovn_s64(a), b), low);
        let t = vmlaq_s32(l, h, c);
        vmlsq_n_s32(t, vqrdmulhq_n_s32(t, V31), P1 as i32)
    };
    let at = idx(s, r);
    let lo = half(*acc.get_unchecked(at), *acc.get_unchecked(at + 1));
    let hi = half(*acc.get_unchecked(at + 2), *acc.get_unchecked(at + 3));
    vmovn_high_s32(vmovn_s32(lo), hi)
}

/// One digit row's chunk end in one go: the sums narrowed in [`super::inverse_lazy_io`]'s first
/// pass, and the deviations `S - mean` (centered) summed into `dev` (the row's `2 N` lanes) in its
/// last pass, instead of three passes over the slots.
#[inline(always)]
unsafe fn chunk_row(acc: &[int64x2_t], r: usize, mr: &[i16], dev: &mut [int32x4_t], y: &mut [int16x8_t; N]) {
    let p = P1 as i16;
    let dv = dev.as_mut_ptr();
    super::inverse_lazy_io(y, |s| narrow_digit(acc, s, r), |s, x| {
        let v = canonical(vsubq_s16(x, vld1q_dup_s16(mr.as_ptr().add(s))), p);
        *dv.add(2 * s) = vaddw_s16(*dv.add(2 * s), vget_low_s16(v));
        *dv.add(2 * s + 1) = vaddw_high_s16(*dv.add(2 * s + 1), v);
    });
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
    nib: &mut [uint8x16_t; N / 4],
    xs: &mut [[int16x8_t; N]; SWEEP],
    acc: &mut [int64x2_t],
    y: &mut [int16x8_t; N],
    dev: &mut [int32x4_t],
) {
    let kern = &kernels()[PB];
    acc.fill(vdupq_n_s64(0));
    let mut e = 0usize;
    while e < a.elements {
        let end = ((e / CHUNK + 1) * CHUNK).min(a.elements);
        while e < end {
            // Up to four tiles per sweep, never across a chunk end (chunks are whole tiles). A short
            // tile only occurs at the end of the column, where the missing elements read zero
            // words and their key lanes are zero, so those lanes contribute nothing.
            let l = SWEEP.min(end - e);
            let groups = l.div_ceil(LOCK);
            for i in 0..groups * LOCK {
                let at = pass_words(words, a.elements, e, i);
                indices(&at, nib);
                let x = &mut xs[i];
                pass_lut::<PB>(nib, x, kern);
                pass_c::<PB>(x, kern);
                pass_d::<PB, false>(x, kern, &[], &mut []);
            }
            let keys = a.data.as_ptr().add(e / LOCK * N * ROWS * LOCK);
            // the chunk's first sweep starts the digit rows' sums afresh
            if e % CHUNK == 0 {
                sink::<true>(xs, keys, acc, groups);
            } else {
                sink::<false>(xs, keys, acc, groups);
            }
            e += l;
        }
        let c = (end - 1) / CHUNK;
        for d in 0..DIGITS {
            chunk_row(acc, 1 + d, a.mean_residues(c, d), &mut dev[d * 2 * N..][..2 * N], y);
        }
    }
    narrow(acc, 0, y);
    super::inverse_folded::<PB>(y);
}

#[inline]
fn crt(r: u32, t: i64) -> u128 {
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
    const _: () = assert!(P1 < TAIL && MODULUS < 1 << 63);
    let k = (t - r as i64).rem_euclid(TAIL as i64) as u64 * INV % TAIL as u64;
    (r as u64 + P1 as u64 * k) as u128
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
        let mut nib = Box::new([vdupq_n_u8(0); N / 4]);
        let mut xs = Box::new([[vdupq_n_s16(0); N]; SWEEP]);
        let mut y = Box::new([vdupq_n_s16(0); N]);
        let mut acc = vec![vdupq_n_s64(0); ROWS * 4 * N];
        // per digit, the chunk deviations S - mean summed over the chunks
        let mut dev = vec![vdupq_n_s32(0); DIGITS * 2 * N];
        // the eight columns' tails, `i64` per lane (41 KiB, off the stack)
        let mut tail = Box::new([[0i64; LANES]; N]);
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
            sums(&words, a, &mut m, &mut nib, &mut xs, &mut acc, &mut y, &mut dev);
            // The base row's centered residues (`|x| <= (p - 1) / 2`, [`super::inverse`]) to
            // `[0, p)` lane-wise, eight columns per vector: `+ p` where negative, as `rem_euclid`.
            let pv = vdupq_n_s16(p);
            let mut base = [[0u16; LANES]; N];
            for (k, b) in base.iter_mut().enumerate() {
                let v = y[k];
                let v = vaddq_s16(v, vandq_s16(vreinterpretq_s16_u16(vcltzq_s16(v)), pv));
                vst1q_u16(b.as_mut_ptr(), vreinterpretq_u16_s16(v));
            }
            // `tail = mu + sum_d W_d dev_d` ([`add_weighted`]'s map) for the eight columns at
            // once (from 4381e86b / de004ec5): one pass per digit over the slots, not per column.
            let devs: &[[i32; 4]] = core::slice::from_raw_parts(dev.as_ptr() as *const [i32; 4], dev.len());
            let tail = &mut *tail;
            for (t, &mu) in tail.iter_mut().zip(a.mu.iter()) {
                *t = [mu as i64; LANES];
            }
            for d in 0..DIGITS {
                let (u, v) = lattice::WEIGHTS[d];
                let (u, v) = (u as i64, v as i64);
                let e = |k: usize, c: usize| devs[d * 2 * N + 2 * k + c / 4][c % 4] as i64;
                for k in 0..N / 2 {
                    for c in 0..LANES {
                        let x = e(k, c);
                        tail[k][c] += u * x;
                        tail[k + N / 2][c] += v * x;
                    }
                }
                for k in N / 2..N {
                    for c in 0..LANES {
                        let x = e(k, c);
                        tail[k - N / 2][c] -= v * x;
                        tail[k][c] += (u + v) * x;
                    }
                }
            }
            // a partial batch has fewer than eight columns in `target`: only those lanes are written
            for (c, column) in target.chunks_exact_mut(N).enumerate() {
                for (k, o) in column.iter_mut().enumerate() {
                    *o = crt(base[k][c] as u32, tail[k][c]);
                }
            }
        }
    });
    out
}

#[cfg(test)]
mod chunk_bench {
    use super::*;

    /// One digit row's chunk end, the record's three passes against [`chunk_row`]: `cargo test
    /// --release chunk_end_timing -- --ignored --nocapture` (on the board's core).
    #[test]
    #[ignore]
    fn chunk_end_timing() {
        let mut st = 7u64;
        let bound = 1i64 << 34;
        let acc: Vec<int64x2_t> = (0..ROWS * 4 * N).map(|_| unsafe { vcombine_s64(vdup_n_s64((crate::ntt::tests::xorshift(&mut st) % (2 * bound as u64)) as i64 - bound), vdup_n_s64((crate::ntt::tests::xorshift(&mut st) % (2 * bound as u64)) as i64 - bound)) }).collect();
        let mr: Vec<i16> = (0..N).map(|i| (i as i16 * 37) % 4860).collect();
        let mut dev = vec![unsafe { vdupq_n_s32(0) }; 2 * N];
        let mut dev2 = dev.clone();
        let mut y = Box::new([unsafe { vdupq_n_s16(0) }; N]);
        // same deviations both ways
        unsafe {
            narrow_digits(&acc, 1, &mut y);
            super::super::inverse_folded::<PB>(&mut y);
            for s in 0..N {
                let v = canonical(vsubq_s16(y[s], vdupq_n_s16(mr[s])), P1 as i16);
                dev[2 * s] = vaddw_s16(dev[2 * s], vget_low_s16(v));
                dev[2 * s + 1] = vaddw_high_s16(dev[2 * s + 1], v);
            }
            chunk_row(&acc, 1, &mr, &mut dev2, &mut y);
        }
        let a: &[[i32; 4]] = unsafe { core::slice::from_raw_parts(dev.as_ptr() as *const [i32; 4], dev.len()) };
        let b: &[[i32; 4]] = unsafe { core::slice::from_raw_parts(dev2.as_ptr() as *const [i32; 4], dev2.len()) };
        assert_eq!(a, b, "the fused chunk end gives the same deviations");
        let reps = 20000;
        for _ in 0..3 {
            let t = std::time::Instant::now();
            for _ in 0..reps {
                unsafe {
                    narrow_digits(&acc, 1, &mut y);
                    super::super::inverse_folded::<PB>(&mut y);
                    for s in 0..N {
                        let v = canonical(vsubq_s16(y[s], vdupq_n_s16(mr[s])), P1 as i16);
                        dev[2 * s] = vaddw_s16(dev[2 * s], vget_low_s16(v));
                        dev[2 * s + 1] = vaddw_high_s16(dev[2 * s + 1], v);
                    }
                }
                core::hint::black_box(&mut dev);
            }
            let old = t.elapsed().as_nanos() as f64 / reps as f64;
            let t = std::time::Instant::now();
            for _ in 0..reps {
                unsafe { chunk_row(&acc, 1, &mr, &mut dev2, &mut y) };
                core::hint::black_box(&mut dev2);
            }
            let new = t.elapsed().as_nanos() as f64 / reps as f64;
            let t = std::time::Instant::now();
            for _ in 0..reps {
                unsafe { super::super::inverse_folded::<PB>(&mut y) };
                core::hint::black_box(&mut *y);
            }
            let f = t.elapsed().as_nanos() as f64 / reps as f64;
            let t = std::time::Instant::now();
            for _ in 0..reps {
                unsafe { super::super::inverse_lazy(&mut y); for v in y.iter_mut() { *v = vandq_s16(*v, vdupq_n_s16(0x0fff)); } };
                core::hint::black_box(&mut *y);
            }
            let l = t.elapsed().as_nanos() as f64 / reps as f64;
            println!("chunk-end row: record {old:.0} ns, fused {new:.0} ns ({:+.1}%); inverse_folded {f:.0} ns, inverse_lazy {l:.0} ns", 100.0 * (new / old - 1.0));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// [`narrow_digits`] is congruent to the sum and within `p`, on the extremes of a chunk's sums
    /// (`+-CHUNK PRODUCT`), small and boundary values, and random values in range.
    #[test]
    fn digit_narrowing_is_exact() {
        let bound = (CHUNK as u64 * PRODUCT) as i64;
        let mut state = 5u64;
        let edge = [bound, -bound, bound - 1, -bound + 1, 0, 1, -1, 65535, -65536, 65536, 4860, -4861];
        let mut want = vec![0i64; ROWS * 8 * N];
        for (i, w) in want.iter_mut().enumerate() {
            *w = if i < edge.len() { edge[i] } else { (crate::ntt::tests::xorshift(&mut state) % (2 * bound as u64 + 1)) as i64 - bound };
        }
        let mut acc = vec![unsafe { vdupq_n_s64(0) }; ROWS * 4 * N];
        unsafe {
            core::ptr::copy_nonoverlapping(want.as_ptr() as *const int64x2_t, acc.as_mut_ptr(), acc.len());
            for r in 0..ROWS {
                let mut out = [vdupq_n_s16(0); N];
                narrow_digits(&acc, r, &mut out);
                for s in 0..N {
                    let o: [i16; 8] = core::mem::transmute(out[s]);
                    for c in 0..8 {
                        let v = want[2 * idx(s, r) + c];
                        assert_eq!((o[c] as i64 - v).rem_euclid(P1 as i64), 0, "{v}");
                        assert!((o[c] as i64).abs() <= P1 as i64);
                    }
                }
            }
        }
    }

    /// A fresh sweep ([`sink`]`::<true>`) over accumulators holding garbage in the digit rows equals
    /// an accumulating sweep over the same accumulators with those rows zeroed, bit for bit, and the
    /// base row accumulates in both: one to four tiles, random keys and transforms.
    #[test]
    fn fresh_sweep_overwrites_the_digit_rows() {
        use core::arch::aarch64::*;
        let mut state = 23u64;
        let mut next = move || crate::ntt::tests::xorshift(&mut state);
        let mut xs = Box::new([[unsafe { vdupq_n_s16(0) }; N]; SWEEP]);
        for groups in 1..=SINK {
            let keys: Vec<i16> = (0..groups * N * ROWS * LOCK).map(|_| ((next() % 9721) as i64 - 4860) as i16).collect();
            for x in xs.iter_mut().flatten() {
                let v: [i16; 8] = core::array::from_fn(|_| next() as i16);
                *x = unsafe { vld1q_s16(v.as_ptr()) };
            }
            let garbage: Vec<i64> = (0..ROWS * 8 * N).map(|_| (next() >> 24) as i64 - (1 << 39)).collect();
            let mut fresh = vec![unsafe { vdupq_n_s64(0) }; ROWS * 4 * N];
            unsafe { core::ptr::copy_nonoverlapping(garbage.as_ptr() as *const int64x2_t, fresh.as_mut_ptr(), fresh.len()) };
            let mut zeroed = fresh.clone();
            for s in 0..N {
                for r in 1..ROWS {
                    zeroed[idx(s, r)..idx(s, r) + 4].fill(unsafe { vdupq_n_s64(0) });
                }
            }
            unsafe {
                sink::<true>(&xs, keys.as_ptr(), &mut fresh, groups);
                sink::<false>(&xs, keys.as_ptr(), &mut zeroed, groups);
            }
            let words = |v: &[int64x2_t]| -> Vec<[i64; 2]> { v.iter().map(|&x| unsafe { core::mem::transmute(x) }).collect() };
            assert!(words(&fresh) == words(&zeroed), "{groups} tiles");
        }
    }

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

    /// The folded path (key times [`KEY_SCALE`], [`super::super::inverse_folded`]) against the
    /// unscaled one ([`super::super::inverse`], the path before the fold), bit for bit, on every
    /// row of one chunk: random key vectors (centered residues, and the extremes `+-(p - 1) / 2`),
    /// random transforms over all of `i16`, [`CHUNK`]` / `[`SWEEP`] full sweeps through [`sink`] and
    /// a last one of one to three tiles (on SVE hosts, [`slot_sink`]'s `sdot`), then [`narrow`] and
    /// the inverse.
    #[test]
    fn folded_key_matches_unscaled_path() {
        use core::arch::aarch64::*;
        let h = (P1 as i64 - 1) / 2;
        let mut state = 17u64;
        let mut next = move || crate::ntt::tests::xorshift(&mut state);
        let lanes = |v: &[int16x8_t; N]| -> Vec<[i16; 8]> { v.iter().map(|&x| unsafe { core::mem::transmute(x) }).collect() };
        let mut xs = Box::new([[unsafe { vdupq_n_s16(0) }; N]; SWEEP]);
        for case in 0..3 {
            let mut old_acc = vec![unsafe { vdupq_n_s64(0) }; ROWS * 4 * N];
            let mut new_acc = old_acc.clone();
            for sweep in 0..=CHUNK / SWEEP {
                let groups = if sweep < CHUNK / SWEEP { SINK } else { 1 + case };
                let old: Vec<i16> = (0..groups * N * ROWS * LOCK)
                    .map(|_| match case {
                        1 => if next() & 1 == 0 { h as i16 } else { -h as i16 },
                        _ => ((next() % (2 * h as u64 + 1)) as i64 - h) as i16,
                    })
                    .collect();
                let new: Vec<i16> = old.iter().map(|&k| centered(k as i64 * KEY_SCALE, P1)).collect();
                for x in xs.iter_mut().flatten() {
                    let v: [i16; 8] = core::array::from_fn(|_| match case {
                        2 => if next() & 1 == 0 { i16::MAX } else { i16::MIN },
                        _ => next() as i16,
                    });
                    *x = unsafe { vld1q_s16(v.as_ptr()) };
                }
                unsafe {
                    sink::<false>(&xs, old.as_ptr(), &mut old_acc, groups);
                    sink::<false>(&xs, new.as_ptr(), &mut new_acc, groups);
                }
            }
            let mut y = Box::new([unsafe { vdupq_n_s16(0) }; N]);
            for r in 0..ROWS {
                unsafe {
                    narrow(&old_acc, r, &mut y);
                    super::super::inverse::<PB>(&mut y);
                }
                let want = lanes(&y);
                unsafe {
                    narrow(&new_acc, r, &mut y);
                    super::super::inverse_folded::<PB>(&mut y);
                }
                assert!(lanes(&y) == want, "case {case} row {r}");
                assert!(want.iter().flatten().all(|&v| v as i64 >= -h && v as i64 <= h));
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
