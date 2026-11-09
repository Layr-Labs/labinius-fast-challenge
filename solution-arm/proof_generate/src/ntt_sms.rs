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
pub const CHUNK: usize = 384;
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

/// The SMS matrix: in the transform domain, slot `j` of row `r` of element `m` is the centered
/// residue split into the byte limbs of [`key_limbs`], at the 16-byte unit [`key_at`]`(m / 8, j, r)`
/// of `data` (low byte at `m % 8`, high byte at `8 + m % 8`), so that one `SMMLA` operand holds the
/// eight elements of an octet, each residue negated because the forward kernel transforms `-s` and
/// multiplied by [`KEY_SCALE`] `= 324^-1` so that the inverses need no `1 / r`; per chunk and digit
/// the centered residue
/// of the mean modulo 9721; and the weighted sum of all means modulo `M`.
pub struct SmsMatrix {
    pub elements: usize,
    pub data: std::borrow::Cow<'static, [i8]>,
    /// `mean_residues[(c DIGITS + d) N + coefficient]`.
    pub mean_residues: std::borrow::Cow<'static, [i16]>,
    /// `sum_{d, c} W_d mean_{d,c} mod M`, by coefficient.
    pub mu: std::borrow::Cow<'static, [u32]>,
    /// The digit rows in the coefficient domain, `digits[(m DIGITS + d) N + coefficient]`
    /// (`|digit| <= 32`), read only by [`repair`].
    pub digits: std::borrow::Cow<'static, [i8]>,
    /// The exact integer mean `floor(S1 / 2)` of each chunk's digit sums,
    /// `mean_int[(c DIGITS + d) N + coefficient]`.
    pub mean_int: std::borrow::Cow<'static, [i32]>,
    /// Per chunk and digit, `P - B` (0 if `B >= P`): a centered residue at least this large in
    /// magnitude is ambiguous and recomputed exactly ([`repair`]); below it, it is the only
    /// value in `[-B, B]` with its residue.
    pub threshold: std::borrow::Cow<'static, [i16]>,
    /// The smallest recovery margin `(B - 1/2) / sigma` in standard deviations.
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
        let piece = len.div_ceil(8).next_multiple_of(2 << 20);
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
    pub fn from_images(data: &'static [u8], means: &'static [u8], mu: &'static [u8], meta: &'static [u8], digits: &'static [u8], mean_int: &'static [u8], threshold: &'static [u8], elements: usize) -> Option<SmsMatrix> {
        let chunks = elements.div_ceil(CHUNK);
        let data = key_copy(data).unwrap_or(data);
        let data = view::<i8>(data, key_len(elements))?;
        let mean_residues = view::<i16>(means, chunks * DIGITS * N)?;
        let mu = view::<u32>(mu, N)?;
        let digits = view::<i8>(digits, elements * DIGITS * N)?;
        let mean_int = view::<i32>(mean_int, chunks * DIGITS * N)?;
        let threshold = view::<i16>(threshold, chunks * DIGITS)?;
        if meta.len() != 16 {
            return None;
        }
        let margin = f64::from_le_bytes(meta[..8].try_into().ok()?);
        let column_failure = f64::from_le_bytes(meta[8..].try_into().ok()?);
        if !(margin >= MIN_MARGIN && column_failure <= MAX_COLUMN_FAILURE) {
            return None;
        }
        use std::borrow::Cow::Borrowed;
        Some(SmsMatrix { elements, data: Borrowed(data), mean_residues: Borrowed(mean_residues), mu: Borrowed(mu), digits: Borrowed(digits), mean_int: Borrowed(mean_int), threshold: Borrowed(threshold), margin, column_failure })
    }

    /// The seven images [`SmsMatrix::from_images`] reads.
    pub fn images(&self) -> [Vec<u8>; 7] {
        let mut meta = self.margin.to_le_bytes().to_vec();
        meta.extend_from_slice(&self.column_failure.to_le_bytes());
        [
            self.data.iter().map(|&v| v as u8).collect(),
            self.mean_residues.iter().flat_map(|v| v.to_le_bytes()).collect(),
            self.mu.iter().flat_map(|v| v.to_le_bytes()).collect(),
            meta,
            self.digits.iter().map(|&v| v as u8).collect(),
            self.mean_int.iter().flat_map(|v| v.to_le_bytes()).collect(),
            self.threshold.iter().flat_map(|v| v.to_le_bytes()).collect(),
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
    // Per chunk and digit, the recovery bound `B`: the largest `ceil(Z sigma + 1/2)` over the
    // coefficients, `Z` = [`MIN_MARGIN`]. A deviation `S - mean` within `[-B, B]` is recovered
    // exactly: from its residue when that is below `P - B` in magnitude, else by [`repair`].
    let mut margin = f64::INFINITY;
    let mut column_failure = 0f64;
    let mut mean_residues = vec![0i16; chunks * DIGITS * N];
    let mut mean_int = vec![0i32; chunks * DIGITS * N];
    let mut threshold = vec![0i16; chunks * DIGITS];
    let mut mu = vec![0i64; N];
    for c in 0..chunks {
        for d in 0..DIGITS {
            let (s1, s2) = moments[(c * DIGITS + d) * 2 * N..][..2 * N].split_at(N);
            let mut mean = vec![0i64; N];
            let mut bound = 0i64;
            for k in 0..N {
                // mean s1 / 2, variance s2 / 4
                mean[k] = s1[k].div_euclid(2);
                let sigma = (s2[k] as f64 / 4.0).sqrt();
                bound = bound.max((MIN_MARGIN * sigma + 0.5).ceil() as i64 + 1);
                mean_residues[(c * DIGITS + d) * N + k] = centered(mean[k], p);
                mean_int[(c * DIGITS + d) * N + k] = i32::try_from(mean[k]).expect("a chunk mean fits i32");
            }
            for k in 0..N {
                let sigma = (s2[k] as f64 / 4.0).sqrt();
                if sigma > 0.0 {
                    let z = (bound as f64 - 0.5) / sigma;
                    margin = margin.min(z);
                    column_failure += 2.0 * (-z * z / 2.0).exp();
                }
            }
            threshold[c * DIGITS + d] = (p as i64 - bound).max(0) as i16;
            add_weighted(&mut mu, d, &mean);
        }
    }
    assert!(margin >= MIN_MARGIN, "SMS recovery margin {margin} below {MIN_MARGIN}");
    assert!(column_failure <= MAX_COLUMN_FAILURE, "SMS column failure bound {column_failure:e} above {MAX_COLUMN_FAILURE:e}");
    let digits: Vec<i8> = (0..elements * DIGITS * N)
        .map(|i| {
            let (m, rest) = (i / (DIGITS * N), i % (DIGITS * N));
            i8::try_from(raw[(ROWS * m + 1) * N + rest]).expect("a digit fits i8")
        })
        .collect();
    let mu = mu.into_iter().map(|v| v.rem_euclid(TAIL as i64) as u32).collect();
    // the transformed rows, eight elements per vector, then as SMMLA limbs per sweep
    let mut tiles = vec![0i16; elements.div_ceil(LOCK) * N * ROWS * LOCK];
    let raw = &raw;
    for_blocks(&mut tiles, N * ROWS * LOCK, 1, workers, |first, block| {
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
    let mut data = vec![0i8; key_len(elements)];
    let tiles = &tiles;
    for_blocks(&mut data, N * OCT * ROWS * 16, 1, workers, |first, block| {
        for (w, sweep) in block.chunks_exact_mut(N * OCT * ROWS * 16).enumerate() {
            let g0 = (first + w) * OCT;
            for o in 0..OCT.min(elements.div_ceil(LOCK) - g0) {
                let g = g0 + o;
                for j in 0..N {
                    for r in 0..ROWS {
                        let k: &[i16; LOCK] = tiles[((g * N + j) * ROWS + r) * LOCK..][..LOCK].try_into().unwrap();
                        let at = (key_at(g, j, r) - key_at(g0, 0, 0)) * 16;
                        sweep[at..at + 16].copy_from_slice(&key_limbs(k));
                    }
                }
            }
        }
    });
    use std::borrow::Cow::Owned;
    SmsMatrix { elements, data: Owned(data), mean_residues: Owned(mean_residues), mu: Owned(mu), digits: Owned(digits), mean_int: Owned(mean_int), threshold: Owned(threshold), margin, column_failure }
}

/// `324^-1 mod p` (`324 = 2^2 3^4`, [`super::FOLD`], the product of the radices), carried by every
/// transformed key row: the slot sums then carry it too, and [`super::inverse_folded`] (no `1 / r`)
/// inverts them to the same coefficients as [`super::inverse`] of the unscaled sums. The
/// certificate's moments, the means and `mu` are of the coefficient rows and do not change; only
/// the key image does.
const KEY_SCALE: i64 = crate::ntt::pow_mod(super::FOLD as u64, P1 as u64 - 2, P1 as u64) as i64;
const _: () = assert!(KEY_SCALE * super::FOLD as i64 % P1 as i64 == 1 && KEY_SCALE == 9691);

/// Elements per lane vector of a transform and per key octet: eight elements of one column.
const LOCK: usize = 8;
/// Element octets per sweep of [`sink`] over the accumulators.
const OCT: usize = 16;
/// Elements transformed per sweep: each accumulator is read and written once per sweep.
const SWEEP: usize = OCT * LOCK;
const _: () = assert!(CHUNK % SWEEP == 0);

/// The accumulators of slot `s`, row `r`, column `c`: `acc[idx(s, r) + c]`, the four `i32` limb
/// sums `(lo lo, lo hi, hi lo, hi hi)` of `SMMLA` (transform limbs times key limbs).
#[inline(always)]
const fn idx(s: usize, r: usize) -> usize {
    (s * ROWS + r) * LANES
}

/// One octet's eight key entries (centered residues) as an `SMMLA` operand: the low bytes, then
/// the rounded high bytes, `k = lo + 256 hi` with `|hi| <= 19`.
fn key_limbs(k: &[i16; LOCK]) -> [i8; 16] {
    let mut out = [0i8; 16];
    for (e, &v) in k.iter().enumerate() {
        let hi = (v as i32 + 128) >> 8;
        out[e] = (v as i32 - 256 * hi) as i8;
        out[LOCK + e] = hi as i8;
    }
    out
}

/// Offset in `i8x16` units of key octet `g` (elements `8 g..8 g + 8`), slot `s`, row `r`: sweeps
/// of [`OCT`] octets, slot-major within a sweep, so that one slot's keys of a sweep are one
/// contiguous run.
#[inline(always)]
const fn key_at(g: usize, s: usize, r: usize) -> usize {
    (((g / OCT) * N + s) * OCT + g % OCT) * ROWS + r
}

/// Bytes of the key image of `elements` elements: whole sweeps, zero past the last element.
pub const fn key_len(elements: usize) -> usize {
    elements.div_ceil(SWEEP) * N * OCT * ROWS * 16
}


/// The transform buffer of a sweep: slot `s`, column `c`, octet `o` at `s XSTRIDE + c OCT + o`, the
/// transform's byte limbs ([`super::pass_d_limbs`]): the octets of one column run on, so that four
/// consecutive transforms of a column fill each other's cache lines.
const XS_LEN: usize = N * XSTRIDE;
/// Vectors between slots of the transform buffer: one 64-byte line more than the slot's data, so
/// that the 648 lines a transform writes do not alias to a few L1 sets (a 2 KiB stride maps them
/// to 8 of the 256).
const XSTRIDE: usize = OCT * LANES + 4;
// the generated level 6 (`gen/emit.py`, `pass_d_limbs`) writes slots 2112 bytes apart
const _: () = assert!(XSTRIDE * 16 == 2112);

/// `acc +=` every slot's products of the sweep's first `groups` octets of transforms `xs` with
/// the sweep's keys `keys` (its first octet's [`key_at`]): `SMMLA`, all seven rows of three, three
/// and two columns at a time, the sums register-resident over the octets, so that each transform
/// vector is read once. With `FRESH` (a chunk's first sweep) the digit rows' sums are overwritten
/// instead of added to, so the chunk end needs no zeroing pass; the base row always accumulates.
/// An octet past the column's end has zero keys.
#[inline(never)]
unsafe fn sink<const FRESH: bool>(xs: *const int8x16_t, keys: *const int8x16_t, acc: &mut [int32x4_t], groups: usize) {
    debug_assert!(acc.len() == ROWS * LANES * N && (1..=OCT).contains(&groups));
    const _: () = assert!(ROWS == 7 && LANES == 8);
    for s in 0..N {
        let base = acc.as_mut_ptr().add(idx(s, 0));
        let xb = xs.add(s * XSTRIDE);
        let k0 = keys.add(s * OCT * ROWS);
        for c0 in [0usize, 3] {
            let (a, x, k) = (base.add(c0), xb.add(c0 * OCT), k0);
            if FRESH {
                core::arch::asm!(
        "ldr q0, [{a}, #0]",
        "ldr q1, [{a}, #16]",
        "ldr q2, [{a}, #32]",
        "movi v3.16b, #0",
        "movi v4.16b, #0",
        "movi v5.16b, #0",
        "movi v6.16b, #0",
        "movi v7.16b, #0",
        "movi v8.16b, #0",
        "movi v9.16b, #0",
        "movi v10.16b, #0",
        "movi v11.16b, #0",
        "movi v12.16b, #0",
        "movi v13.16b, #0",
        "movi v14.16b, #0",
        "movi v15.16b, #0",
        "movi v16.16b, #0",
        "movi v17.16b, #0",
        "movi v18.16b, #0",
        "movi v19.16b, #0",
        "movi v20.16b, #0",
        "2:",
        "ldr q22, [{x}, #{c1}]",
        "ldr q23, [{x}, #{c2}]",
        "ldr q21, [{x}], #16",
        "ld1 {{v24.16b, v25.16b, v26.16b, v27.16b}}, [{k}], #64",
        "ld1 {{v28.16b, v29.16b, v30.16b}}, [{k}], #48",
        "smmla v0.4s, v21.16b, v24.16b",
        "smmla v1.4s, v22.16b, v24.16b",
        "smmla v2.4s, v23.16b, v24.16b",
        "smmla v3.4s, v21.16b, v25.16b",
        "smmla v4.4s, v22.16b, v25.16b",
        "smmla v5.4s, v23.16b, v25.16b",
        "smmla v6.4s, v21.16b, v26.16b",
        "smmla v7.4s, v22.16b, v26.16b",
        "smmla v8.4s, v23.16b, v26.16b",
        "smmla v9.4s, v21.16b, v27.16b",
        "smmla v10.4s, v22.16b, v27.16b",
        "smmla v11.4s, v23.16b, v27.16b",
        "smmla v12.4s, v21.16b, v28.16b",
        "smmla v13.4s, v22.16b, v28.16b",
        "smmla v14.4s, v23.16b, v28.16b",
        "smmla v15.4s, v21.16b, v29.16b",
        "smmla v16.4s, v22.16b, v29.16b",
        "smmla v17.4s, v23.16b, v29.16b",
        "smmla v18.4s, v21.16b, v30.16b",
        "smmla v19.4s, v22.16b, v30.16b",
        "smmla v20.4s, v23.16b, v30.16b",
        "subs {n}, {n}, #1",
        "b.ne 2b",
        "str q0, [{a}, #0]",
        "str q1, [{a}, #16]",
        "str q2, [{a}, #32]",
        "str q3, [{a}, #128]",
        "str q4, [{a}, #144]",
        "str q5, [{a}, #160]",
        "str q6, [{a}, #256]",
        "str q7, [{a}, #272]",
        "str q8, [{a}, #288]",
        "str q9, [{a}, #384]",
        "str q10, [{a}, #400]",
        "str q11, [{a}, #416]",
        "str q12, [{a}, #512]",
        "str q13, [{a}, #528]",
        "str q14, [{a}, #544]",
        "str q15, [{a}, #640]",
        "str q16, [{a}, #656]",
        "str q17, [{a}, #672]",
        "str q18, [{a}, #768]",
        "str q19, [{a}, #784]",
        "str q20, [{a}, #800]",
            a = in(reg) a,
            x = inout(reg) x => _,
            k = inout(reg) k => _,
            n = inout(reg) groups => _,
            c1 = const OCT * 16,
            c2 = const 2 * OCT * 16,
            out("v0") _, out("v1") _, out("v2") _, out("v3") _, out("v4") _, out("v5") _, out("v6") _, out("v7") _, out("v8") _, out("v9") _, out("v10") _, out("v11") _, out("v12") _, out("v13") _, out("v14") _, out("v15") _, out("v16") _, out("v17") _, out("v18") _, out("v19") _, out("v20") _, out("v21") _, out("v22") _, out("v23") _, out("v24") _, out("v25") _, out("v26") _, out("v27") _, out("v28") _, out("v29") _, out("v30") _,
            options(nostack)
        );
            } else {
                core::arch::asm!(
        "ldr q0, [{a}, #0]",
        "ldr q1, [{a}, #16]",
        "ldr q2, [{a}, #32]",
        "ldr q3, [{a}, #128]",
        "ldr q4, [{a}, #144]",
        "ldr q5, [{a}, #160]",
        "ldr q6, [{a}, #256]",
        "ldr q7, [{a}, #272]",
        "ldr q8, [{a}, #288]",
        "ldr q9, [{a}, #384]",
        "ldr q10, [{a}, #400]",
        "ldr q11, [{a}, #416]",
        "ldr q12, [{a}, #512]",
        "ldr q13, [{a}, #528]",
        "ldr q14, [{a}, #544]",
        "ldr q15, [{a}, #640]",
        "ldr q16, [{a}, #656]",
        "ldr q17, [{a}, #672]",
        "ldr q18, [{a}, #768]",
        "ldr q19, [{a}, #784]",
        "ldr q20, [{a}, #800]",
        "2:",
        "ldr q22, [{x}, #{c1}]",
        "ldr q23, [{x}, #{c2}]",
        "ldr q21, [{x}], #16",
        "ld1 {{v24.16b, v25.16b, v26.16b, v27.16b}}, [{k}], #64",
        "ld1 {{v28.16b, v29.16b, v30.16b}}, [{k}], #48",
        "smmla v0.4s, v21.16b, v24.16b",
        "smmla v1.4s, v22.16b, v24.16b",
        "smmla v2.4s, v23.16b, v24.16b",
        "smmla v3.4s, v21.16b, v25.16b",
        "smmla v4.4s, v22.16b, v25.16b",
        "smmla v5.4s, v23.16b, v25.16b",
        "smmla v6.4s, v21.16b, v26.16b",
        "smmla v7.4s, v22.16b, v26.16b",
        "smmla v8.4s, v23.16b, v26.16b",
        "smmla v9.4s, v21.16b, v27.16b",
        "smmla v10.4s, v22.16b, v27.16b",
        "smmla v11.4s, v23.16b, v27.16b",
        "smmla v12.4s, v21.16b, v28.16b",
        "smmla v13.4s, v22.16b, v28.16b",
        "smmla v14.4s, v23.16b, v28.16b",
        "smmla v15.4s, v21.16b, v29.16b",
        "smmla v16.4s, v22.16b, v29.16b",
        "smmla v17.4s, v23.16b, v29.16b",
        "smmla v18.4s, v21.16b, v30.16b",
        "smmla v19.4s, v22.16b, v30.16b",
        "smmla v20.4s, v23.16b, v30.16b",
        "subs {n}, {n}, #1",
        "b.ne 2b",
        "str q0, [{a}, #0]",
        "str q1, [{a}, #16]",
        "str q2, [{a}, #32]",
        "str q3, [{a}, #128]",
        "str q4, [{a}, #144]",
        "str q5, [{a}, #160]",
        "str q6, [{a}, #256]",
        "str q7, [{a}, #272]",
        "str q8, [{a}, #288]",
        "str q9, [{a}, #384]",
        "str q10, [{a}, #400]",
        "str q11, [{a}, #416]",
        "str q12, [{a}, #512]",
        "str q13, [{a}, #528]",
        "str q14, [{a}, #544]",
        "str q15, [{a}, #640]",
        "str q16, [{a}, #656]",
        "str q17, [{a}, #672]",
        "str q18, [{a}, #768]",
        "str q19, [{a}, #784]",
        "str q20, [{a}, #800]",
            a = in(reg) a,
            x = inout(reg) x => _,
            k = inout(reg) k => _,
            n = inout(reg) groups => _,
            c1 = const OCT * 16,
            c2 = const 2 * OCT * 16,
            out("v0") _, out("v1") _, out("v2") _, out("v3") _, out("v4") _, out("v5") _, out("v6") _, out("v7") _, out("v8") _, out("v9") _, out("v10") _, out("v11") _, out("v12") _, out("v13") _, out("v14") _, out("v15") _, out("v16") _, out("v17") _, out("v18") _, out("v19") _, out("v20") _, out("v21") _, out("v22") _, out("v23") _, out("v24") _, out("v25") _, out("v26") _, out("v27") _, out("v28") _, out("v29") _, out("v30") _,
            options(nostack)
        );
            }
        }
        let (a, x, k) = (base.add(6), xb.add(6 * OCT), k0);
        if FRESH {
            core::arch::asm!(
        "ldr q0, [{a}, #0]",
        "ldr q1, [{a}, #16]",
        "movi v2.16b, #0",
        "movi v3.16b, #0",
        "movi v4.16b, #0",
        "movi v5.16b, #0",
        "movi v6.16b, #0",
        "movi v7.16b, #0",
        "movi v8.16b, #0",
        "movi v9.16b, #0",
        "movi v10.16b, #0",
        "movi v11.16b, #0",
        "movi v12.16b, #0",
        "movi v13.16b, #0",
        "2:",
        "ldr q22, [{x}, #{c1}]",
        "ldr q21, [{x}], #16",
        "ld1 {{v24.16b, v25.16b, v26.16b, v27.16b}}, [{k}], #64",
        "ld1 {{v28.16b, v29.16b, v30.16b}}, [{k}], #48",
        "smmla v0.4s, v21.16b, v24.16b",
        "smmla v1.4s, v22.16b, v24.16b",
        "smmla v2.4s, v21.16b, v25.16b",
        "smmla v3.4s, v22.16b, v25.16b",
        "smmla v4.4s, v21.16b, v26.16b",
        "smmla v5.4s, v22.16b, v26.16b",
        "smmla v6.4s, v21.16b, v27.16b",
        "smmla v7.4s, v22.16b, v27.16b",
        "smmla v8.4s, v21.16b, v28.16b",
        "smmla v9.4s, v22.16b, v28.16b",
        "smmla v10.4s, v21.16b, v29.16b",
        "smmla v11.4s, v22.16b, v29.16b",
        "smmla v12.4s, v21.16b, v30.16b",
        "smmla v13.4s, v22.16b, v30.16b",
        "subs {n}, {n}, #1",
        "b.ne 2b",
        "str q0, [{a}, #0]",
        "str q1, [{a}, #16]",
        "str q2, [{a}, #128]",
        "str q3, [{a}, #144]",
        "str q4, [{a}, #256]",
        "str q5, [{a}, #272]",
        "str q6, [{a}, #384]",
        "str q7, [{a}, #400]",
        "str q8, [{a}, #512]",
        "str q9, [{a}, #528]",
        "str q10, [{a}, #640]",
        "str q11, [{a}, #656]",
        "str q12, [{a}, #768]",
        "str q13, [{a}, #784]",
            a = in(reg) a,
            x = inout(reg) x => _,
            k = inout(reg) k => _,
            n = inout(reg) groups => _,
            c1 = const OCT * 16,
            out("v0") _, out("v1") _, out("v2") _, out("v3") _, out("v4") _, out("v5") _, out("v6") _, out("v7") _, out("v8") _, out("v9") _, out("v10") _, out("v11") _, out("v12") _, out("v13") _, out("v21") _, out("v22") _, out("v24") _, out("v25") _, out("v26") _, out("v27") _, out("v28") _, out("v29") _, out("v30") _,
            options(nostack)
        );
        } else {
            core::arch::asm!(
        "ldr q0, [{a}, #0]",
        "ldr q1, [{a}, #16]",
        "ldr q2, [{a}, #128]",
        "ldr q3, [{a}, #144]",
        "ldr q4, [{a}, #256]",
        "ldr q5, [{a}, #272]",
        "ldr q6, [{a}, #384]",
        "ldr q7, [{a}, #400]",
        "ldr q8, [{a}, #512]",
        "ldr q9, [{a}, #528]",
        "ldr q10, [{a}, #640]",
        "ldr q11, [{a}, #656]",
        "ldr q12, [{a}, #768]",
        "ldr q13, [{a}, #784]",
        "2:",
        "ldr q22, [{x}, #{c1}]",
        "ldr q21, [{x}], #16",
        "ld1 {{v24.16b, v25.16b, v26.16b, v27.16b}}, [{k}], #64",
        "ld1 {{v28.16b, v29.16b, v30.16b}}, [{k}], #48",
        "smmla v0.4s, v21.16b, v24.16b",
        "smmla v1.4s, v22.16b, v24.16b",
        "smmla v2.4s, v21.16b, v25.16b",
        "smmla v3.4s, v22.16b, v25.16b",
        "smmla v4.4s, v21.16b, v26.16b",
        "smmla v5.4s, v22.16b, v26.16b",
        "smmla v6.4s, v21.16b, v27.16b",
        "smmla v7.4s, v22.16b, v27.16b",
        "smmla v8.4s, v21.16b, v28.16b",
        "smmla v9.4s, v22.16b, v28.16b",
        "smmla v10.4s, v21.16b, v29.16b",
        "smmla v11.4s, v22.16b, v29.16b",
        "smmla v12.4s, v21.16b, v30.16b",
        "smmla v13.4s, v22.16b, v30.16b",
        "subs {n}, {n}, #1",
        "b.ne 2b",
        "str q0, [{a}, #0]",
        "str q1, [{a}, #16]",
        "str q2, [{a}, #128]",
        "str q3, [{a}, #144]",
        "str q4, [{a}, #256]",
        "str q5, [{a}, #272]",
        "str q6, [{a}, #384]",
        "str q7, [{a}, #400]",
        "str q8, [{a}, #512]",
        "str q9, [{a}, #528]",
        "str q10, [{a}, #640]",
        "str q11, [{a}, #656]",
        "str q12, [{a}, #768]",
        "str q13, [{a}, #784]",
            a = in(reg) a,
            x = inout(reg) x => _,
            k = inout(reg) k => _,
            n = inout(reg) groups => _,
            c1 = const OCT * 16,
            out("v0") _, out("v1") _, out("v2") _, out("v3") _, out("v4") _, out("v5") _, out("v6") _, out("v7") _, out("v8") _, out("v9") _, out("v10") _, out("v11") _, out("v12") _, out("v13") _, out("v21") _, out("v22") _, out("v24") _, out("v25") _, out("v26") _, out("v27") _, out("v28") _, out("v29") _, out("v30") _,
            options(nostack)
        );
        }
    }
}

/// The eight witness pointers of the transform of octet `o` (elements `e + 8 o..e + 8 o + 8`) of
/// column `c`: lane `k` is element `e + 8 o + k`. Elements past the column read as zero.
#[inline(always)]
unsafe fn octet_words(words: &[(*const u8, usize); LANES], elements: usize, e: usize, c: usize) -> [*const u8; LANES] {
    let (w, stride) = words[c];
    core::array::from_fn(|k| {
        let element = e + k;
        if element < elements { w.add(element * stride) } else { ZERO_WORDS.as_ptr() as *const u8 }
    })
}

/// `2^16 mod p`, centered.
const C16: i32 = centered(1 << 16, P1) as i32;
/// `round(2^31 / p)`.
const V31: i32 = ((1u64 << 32) / P1 as u64).div_ceil(2) as i32;
/// The transform's outputs, `|v| <= ` this: the limbs are exact below 32640.
const XBOUND: u32 = crate::ntt::lut_transform_bound(P1, SCHEDULES[PB]);
const _: () = assert!(XBOUND < 32640);
/// Limb bounds: a low byte, the transform's and the key's high bytes.
const LO: i64 = 128;
const XHI: i64 = (XBOUND as i64 + 128) >> 8;
const KHI: i64 = ((P1 as i64 - 1) / 2 + 128) >> 8;
// a chunk's limb sums: `lo lo + 256 hi lo` and `lo hi + 256 hi hi` fit i32
const _: () = assert!((CHUNK as i64) * (LO * LO + 256 * XHI * LO) < i32::MAX as i64);
const _: () = assert!((CHUNK as i64) * (LO * KHI + 256 * XHI * KHI) < i32::MAX as i64);

/// `a - round(a / p) p` with the quotient from `sqrdmulh` by `round(2^31 / p)`, off from `a / p`
/// by at most `1/2 + |a| 0.85 / 2^31`: `|result| <= 0.51 p` for `|a| <= 2^25`, `1.14 p` for any
/// `i32` lane within `1.62e9`.
#[inline(always)]
unsafe fn reduce32(a: int32x4_t) -> int32x4_t {
    vmlsq_n_s32(a, vqrdmulhq_n_s32(a, V31), P1 as i32)
}

/// One slot's eight columns of a chunk's digit row `r` as residues, `|r| <= 0.51 p`: per column
/// `S = (lo lo + 256 hi lo) + 256 (lo hi + 256 hi hi)`, each part within `i32` over a chunk.
#[inline(always)]
unsafe fn narrow_digit(acc: &[int32x4_t], s: usize, r: usize) -> int16x8_t {
    let at = idx(s, r);
    let quad = |c: usize| -> int32x4_t {
        let v = |i: usize| vreinterpretq_s64_s32(*acc.get_unchecked(at + c + i));
        // (lo lo, lo hi) and (hi lo, hi hi) of two columns, then A and B of each
        let t = |i: usize| -> int32x4_t {
            let (l, h) = (vzip1q_s64(v(i), v(i + 1)), vzip2q_s64(v(i), v(i + 1)));
            vaddq_s32(vreinterpretq_s32_s64(l), vshlq_n_s32::<8>(vreinterpretq_s32_s64(h)))
        };
        let (t01, t23) = (t(0), t(2));
        let (a, b) = (vuzp1q_s32(t01, t23), vuzp2q_s32(t01, t23));
        reduce32(vaddq_s32(reduce32(a), vshlq_n_s32::<8>(reduce32(b))))
    };
    vmovn_high_s32(vmovn_s32(quad(0)), quad(4))
}

/// The base row's eight columns of slot `s` as residues over a whole column: each limb lane
/// reduced first (lanes reach `4096 * 128 * 128`), then `l0 + 256 (l1 + l2) + C16 l3`.
#[inline(always)]
unsafe fn narrow_base(acc: &[int32x4_t], s: usize) -> int16x8_t {
    let w = vld1q_s32([1, 256, 256, C16].as_ptr());
    let at = idx(s, 0);
    let m = |c: usize| vmulq_s32(reduce32(*acc.get_unchecked(at + c)), w);
    let quad = |c: usize| reduce32(vpaddq_s32(vpaddq_s32(m(c), m(c + 1)), vpaddq_s32(m(c + 2), m(c + 3))));
    vmovn_high_s32(vmovn_s32(quad(0)), quad(4))
}

/// Row 0's sums as residues, one `i16` vector per slot, lane = column.
#[inline(always)]
unsafe fn narrow(acc: &[int32x4_t], out: &mut [int16x8_t; N]) {
    for (s, o) in out.iter_mut().enumerate() {
        *o = narrow_base(acc, s);
    }
}
/// One digit row's chunk end in one go: the sums narrowed in [`super::inverse_lazy_io`]'s first
/// pass, and the deviations `S - mean` (centered) summed into `dev` (the row's `2 N` lanes) in its
/// last pass, instead of three passes over the slots.
#[inline(always)]
#[cfg_attr(not(test), allow(dead_code))]
unsafe fn chunk_row(acc: &[int32x4_t], r: usize, mr: &[i16], dev: &mut [int32x4_t], y: &mut [int16x8_t; N]) {
    let p = P1 as i16;
    let dv = dev.as_mut_ptr();
    super::inverse_lazy_io(y, |s| narrow_digit(acc, s, r), |s, x| {
        let v = canonical(vsubq_s16(x, vld1q_dup_s16(mr.as_ptr().add(s))), p);
        *dv.add(2 * s) = vaddw_s16(*dv.add(2 * s), vget_low_s16(v));
        *dv.add(2 * s + 1) = vaddw_high_s16(*dv.add(2 * s + 1), v);
    });
}

#[inline(always)]
unsafe fn chunk_row2(acc: &[int32x4_t], r: usize, mr: [&[i16]; 2], dev: [*mut int32x4_t; 2], y: [&mut [int16x8_t; N]; 2], ctx: &Repair) {
    let p = P1 as i16;
    let thr = [vdupq_n_s16(ctx.a.threshold[ctx.c * DIGITS + r - 1]), vdupq_n_s16(ctx.a.threshold[ctx.c * DIGITS + r])];
    super::inverse_lazy_io2(y, |h, s| narrow_digit(acc, s, r + h), |h, s, x| {
        let mut v = canonical(vsubq_s16(x, vld1q_dup_s16(mr[h].as_ptr().add(s))), p);
        let dv = dev[h];
        let flag = vcgeq_s16(vabsq_s16(v), thr[h]);
        if vmaxvq_u16(flag) != 0 {
            v = repair_lanes(ctx, r - 1 + h, s, v, flag, dv);
        }
        *dv.add(2 * s) = vaddw_s16(*dv.add(2 * s), vget_low_s16(v));
        *dv.add(2 * s + 1) = vaddw_high_s16(*dv.add(2 * s + 1), v);
    });
}


/// What [`repair`] needs at a chunk end: the batch's column words, the matrix and the chunk.
struct Repair<'a> {
    words: &'a [(*const u8, usize); LANES],
    a: &'a SmsMatrix,
    c: usize,
}

/// The flagged lanes of slot `s`'s deviations `v` (digit `d`) recomputed exactly and added to
/// their `dev` lanes; returned with those lanes zeroed, so the caller's ordinary sum skips them.
#[cold]
#[inline(never)]
unsafe fn repair_lanes(ctx: &Repair, d: usize, s: usize, v: int16x8_t, flag: uint16x8_t, dv: *mut int32x4_t) -> int16x8_t {
    let mut lanes = [0i16; 8];
    vst1q_s16(lanes.as_mut_ptr(), v);
    let mut flags = [0u16; 8];
    vst1q_u16(flags.as_mut_ptr(), flag);
    let mut sums = [[0i32; 4]; 2];
    vst1q_s32(sums[0].as_mut_ptr(), *dv.add(2 * s));
    vst1q_s32(sums[1].as_mut_ptr(), *dv.add(2 * s + 1));
    for lane in 0..8 {
        if flags[lane] != 0 {
            let exact = repair(ctx.words[lane], ctx.a, ctx.c, d, s);
            debug_assert_eq!((exact - lanes[lane] as i64).rem_euclid(P1 as i64), 0, "the exact deviation has the residue's class");
            sums[lane / 4][lane % 4] += i32::try_from(exact).expect("a chunk deviation fits i32");
            lanes[lane] = 0;
        }
    }
    *dv.add(2 * s) = vld1q_s32(sums[0].as_ptr());
    *dv.add(2 * s + 1) = vld1q_s32(sums[1].as_ptr());
    vld1q_s16(lanes.as_ptr())
}

/// Front padding of [`repair`]'s reversed digit row: at least `N` and a multiple of 16.
const REPAIR_PRE: usize = 656;

/// `sum` of the signed byte products of `a` and `b` in four `i32` lanes (`SDOT`).
#[inline(always)]
unsafe fn sdot(mut acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
    core::arch::asm!("sdot {acc:v}.4s, {a:v}.16b, {b:v}.16b", acc = inout(vreg) acc, a = in(vreg) a, b = in(vreg) b,
        options(pure, nomem, nostack, preserves_flags));
    acc
}

/// The exact deviation `S - mean` of coefficient `k` of digit `d`'s chunk-`c` sum
/// `S = sum_m (L_{d,m} s_m)[k]` over the chunk's elements `m`, for the column whose element `m`
/// has its four witness words at `col.0 + m col.1` (coefficient `4 q + j` is bit `q` of word `j`).
/// Coefficient `k` of `X^i L` is `L[k - i] - L[k + 648 - i] - L[k + 972 - i]` for `k < 324` and
/// `L[k - i] + L[k + 324 - i]` for `k >= 324` (`L` zero outside `0..648`), so per element the
/// weights of the 512 witness coefficients are windows of the reversed digit row.
#[cold]
#[inline(never)]
pub unsafe fn repair(col: (*const u8, usize), a: &SmsMatrix, c: usize, d: usize, k: usize) -> i64 {
    const PRE: usize = REPAIR_PRE;
    const _: () = assert!(PRE >= N && PRE % 16 == 0);
    // buf[PRE + j] = L[N - 1 - j]; zero elsewhere, so L[x] = buf[PRE + N - 1 - x] for any x the
    // windows reach (x in -511 .. 1158)
    let mut buf = [0i8; PRE + N + 512 + 16];
    let mut mask = [0i8; 512];
    let bits: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
    let bits = vld1q_u8(bits.as_ptr());
    let start = c * CHUNK;
    let end = ((c + 1) * CHUNK).min(a.elements);
    let (o1, o2, o3) = if k < N / 2 {
        (PRE + N - 1 - k, PRE - 1 - k, PRE - (N / 2 + 1) - k)
    } else {
        (PRE + N - 1 - k, PRE + N / 2 - 1 - k, 0)
    };
    let mut acc = vdupq_n_s32(0);
    for m in start..end {
        let w = col.0.add(m * col.1);
        let wv: [uint8x16_t; 4] = core::array::from_fn(|j| vld1q_u8(w.add(16 * j)));
        // mask[4 q + j] = -1 where bit q of word j is set
        for ch in 0..8u8 {
            let idx = vcombine_u8(vdup_n_u8(2 * ch), vdup_n_u8(2 * ch + 1));
            let v = uint8x16x4_t(
                vtstq_u8(vqtbl1q_u8(wv[0], idx), bits),
                vtstq_u8(vqtbl1q_u8(wv[1], idx), bits),
                vtstq_u8(vqtbl1q_u8(wv[2], idx), bits),
                vtstq_u8(vqtbl1q_u8(wv[3], idx), bits),
            );
            vst4q_u8(mask.as_mut_ptr().add(64 * ch as usize) as *mut u8, v);
        }
        let l = a.digits.as_ptr().add((m * DIGITS + d) * N);
        for t in 0..N / 8 {
            let v = vrev64_s8(vld1_s8(l.add(8 * t)));
            vst1_s8(buf.as_mut_ptr().add(PRE + N - 8 - 8 * t), v);
        }
        let b = buf.as_ptr();
        for v in 0..32 {
            let i = 16 * v;
            let w1 = vld1q_s8(b.add(o1 + i));
            let wt = if k < N / 2 {
                vsubq_s8(vsubq_s8(w1, vld1q_s8(b.add(o2 + i))), vld1q_s8(b.add(o3 + i)))
            } else {
                vaddq_s8(w1, vld1q_s8(b.add(o2 + i)))
            };
            acc = sdot(acc, wt, vld1q_s8(mask.as_ptr().add(i)));
        }
    }
    // the masks are -1: acc holds -S
    -(vaddlvq_s32(acc)) - a.mean_int[(c * DIGITS + d) * N + k] as i64
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

/// The batch's pass: each chunk's digit sums inverted to canonical coefficients and summed as
/// deviations into `dev`; on return `y` holds the base row's coefficients. `x` is the transform's
/// working buffer, `xs` a sweep's transforms as `SMMLA` limbs ([`XS_LEN`]), `y2` a second inverse
/// buffer.
#[inline(never)]
#[allow(clippy::too_many_arguments)]
unsafe fn sums(
    words: &[(*const u8, usize); LANES],
    a: &SmsMatrix,
    nib: &mut [uint8x16_t; N / 4],
    x: &mut [int16x8_t; N],
    xs: *mut int8x16_t,
    acc: &mut [int32x4_t],
    y: &mut [int16x8_t; N],
    y2: &mut [int16x8_t; N],
    dev: &mut [int32x4_t],
) {
    let kern = &kernels()[PB];
    for s in 0..N {
        acc[idx(s, 0)..][..LANES].fill(vdupq_n_s32(0));
    }
    let mut e = 0usize;
    while e < a.elements {
        let end = ((e / CHUNK + 1) * CHUNK).min(a.elements);
        while e < end {
            // Up to OCT octets per sweep, never across a chunk end (chunks are whole sweeps). A
            // short octet only occurs at the end of the column, where the missing elements read
            // zero words and their key lanes are zero, so those lanes contribute nothing.
            let l = SWEEP.min(end - e);
            let groups = l.div_ceil(LOCK);
            for c in 0..LANES {
                for o in 0..groups {
                    let at = octet_words(words, a.elements, e + LOCK * o, c);
                    indices(&at, nib);
                    pass_lut_asm(nib, x, kern);
                    pass_c_asm(x, kern);
                    pass_d_limbs_asm(x, kern, xs.add(c * OCT + o));
                }
            }
            let keys = (a.data.as_ptr() as *const int8x16_t).add(key_at(e / LOCK, 0, 0));
            // the chunk's first sweep starts the digit rows' sums afresh
            if e % CHUNK == 0 {
                sink::<true>(xs, keys, acc, groups);
            } else {
                sink::<false>(xs, keys, acc, groups);
            }
            e += l;
        }
        let c = (end - 1) / CHUNK;
        const _: () = assert!(DIGITS % 2 == 0);
        let dv = dev.as_mut_ptr();
        for d in (0..DIGITS).step_by(2) {
            let mr = [a.mean_residues(c, d), a.mean_residues(c, d + 1)];
            let rows = [dv.add(d * 2 * N), dv.add((d + 1) * 2 * N)];
            chunk_row2(acc, 1 + d, mr, rows, [&mut *y, &mut *y2], &Repair { words, a, c });
        }
    }
    narrow(acc, y);
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

#[repr(C)]
struct Scratch {
    xs: [int8x16_t; XS_LEN],
    acc: [int32x4_t; ROWS * LANES * N],
    x: [int16x8_t; N],
    y2: [int16x8_t; N],
    dev: [int32x4_t; DIGITS * 2 * N],
    nib: [uint8x16_t; N / 4],
    y: [int16x8_t; N],
    tail: [[i64; LANES]; N],
}

const ARENA: usize = 4 << 20;
const _: () = assert!(core::mem::size_of::<Scratch>() <= ARENA, "the scratch fits the arena");

struct Arena(*mut Scratch);

impl Arena {
    fn layout() -> std::alloc::Layout {
        std::alloc::Layout::from_size_align(ARENA.max(core::mem::size_of::<Scratch>()), core::mem::align_of::<Scratch>())
            .expect("the scratch layout")
    }

    fn new() -> Self {
        let layout = Self::layout();
        let p = unsafe { std::alloc::alloc_zeroed(layout) } as *mut Scratch;
        if p.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        Arena(p)
    }
}

impl Drop for Arena {
    fn drop(&mut self) {
        unsafe { std::alloc::dealloc(self.0 as *mut u8, Self::layout()) }
    }
}

/// The commitment of every column of `values` (columns of `4 a.elements` words) modulo
/// [`MODULUS`].
pub fn commit(values: &[B128], a: &SmsMatrix, workers: usize) -> Vec<u128> {
    let len = 4 * a.elements;
    assert_eq!(values.len() % len, 0);
    let columns = values.len() / len;
    let mut out = vec![0u128; columns * N];
    for_blocks(&mut out, N, LANES, workers, |first, block| unsafe {
        let arena = Arena::new();
        let Scratch { xs, acc, x, y2, dev, nib, y, tail } = &mut *arena.0;
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
            sums(&words, a, nib, x, xs.as_mut_ptr(), acc, y, y2, dev);
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
            let tail = &mut *tail;
            for k in 0..N / 2 {
                let h = k + N / 2;
                let mut lo = [vdupq_n_s64(a.mu[k] as i64); 4];
                let mut hi = [vdupq_n_s64(a.mu[h] as i64); 4];
                for d in 0..DIGITS {
                    let (u, v) = lattice::WEIGHTS[d];
                    let w = u + v;
                    for half in 0..2 {
                        let x = dev[d * 2 * N + 2 * k + half];
                        let y = dev[d * 2 * N + 2 * h + half];
                        let (xl, yl) = (vget_low_s32(x), vget_low_s32(y));
                        let (i, j) = (2 * half, 2 * half + 1);
                        lo[i] = vmlal_n_s32(lo[i], xl, u);
                        lo[j] = vmlal_high_n_s32(lo[j], x, u);
                        lo[i] = vmlsl_n_s32(lo[i], yl, v);
                        lo[j] = vmlsl_high_n_s32(lo[j], y, v);
                        hi[i] = vmlal_n_s32(hi[i], xl, v);
                        hi[j] = vmlal_high_n_s32(hi[j], x, v);
                        hi[i] = vmlal_n_s32(hi[i], yl, w);
                        hi[j] = vmlal_high_n_s32(hi[j], y, w);
                    }
                }
                for q in 0..4 {
                    vst1q_s64(tail[k].as_mut_ptr().add(2 * q), lo[q]);
                    vst1q_s64(tail[h].as_mut_ptr().add(2 * q), hi[q]);
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
mod tests {
    use super::*;

    /// [`narrow_digit`] (a chunk's limb sums) and [`narrow_base`] (a column's) are congruent to
    /// `l0 + 256 (l1 + l2) + 65536 l3` and within `0.51 p` (the inverses take `p`), at the extremes of their bounds and
    /// on random values in range.
    #[test]
    fn limb_narrowing_is_exact() {
        let mut state = 5u64;
        let mut next = move || crate::ntt::tests::xorshift(&mut state);
        for base in [false, true] {
            let n = if base { 4096 } else { CHUNK as i64 };
            let bound = [n * LO * LO, n * LO * KHI, n * XHI * LO, n * XHI * KHI];
            let mut acc = vec![unsafe { vdupq_n_s32(0) }; ROWS * LANES * N];
            let mut want = vec![[0i32; 4]; ROWS * LANES * N];
            for (i, w) in want.iter_mut().enumerate() {
                for (l, v) in w.iter_mut().enumerate() {
                    let b = bound[l];
                    *v = match i % 5 {
                        0 => b as i32,
                        1 => -b as i32,
                        _ => ((next() % (2 * b as u64 + 1)) as i64 - b) as i32,
                    };
                }
                acc[i] = unsafe { vld1q_s32(w.as_ptr()) };
            }
            for s in 0..N {
                for r in if base { 0..1 } else { 1..ROWS } {
                    let o: [i16; 8] = unsafe { core::mem::transmute(if base { narrow_base(&acc, s) } else { narrow_digit(&acc, s, r) }) };
                    for c in 0..LANES {
                        let l = want[idx(s, r) + c].map(|v| v as i64);
                        let v = l[0] + 256 * (l[1] + l[2]) + 65536 * l[3];
                        assert_eq!((o[c] as i64 - v).rem_euclid(P1 as i64), 0, "{l:?}");
                        assert!((o[c] as i64).abs() * 100 <= 51 * P1 as i64);
                    }
                }
            }
        }
    }

    /// [`sink`] against the limb sums computed lane by lane, every octet count, fresh and
    /// accumulating, random limbs over all of `i8` and garbage in the accumulators.
    #[test]
    fn sink_matches_the_limb_sums() {
        let mut state = 23u64;
        let mut next = move || crate::ntt::tests::xorshift(&mut state);
        let mut xs = vec![unsafe { vdupq_n_s8(0) }; XS_LEN];
        let mut keys = vec![unsafe { vdupq_n_s8(0) }; N * OCT * ROWS];
        let bytes = |v: &int8x16_t| -> [i8; 16] { unsafe { core::mem::transmute(*v) } };
        for groups in 1..=OCT {
            for v in xs.iter_mut().chain(keys.iter_mut()) {
                let b: [i8; 16] = core::array::from_fn(|_| next() as i8);
                *v = unsafe { vld1q_s8(b.as_ptr()) };
            }
            for fresh in [false, true] {
                let start: Vec<[i32; 4]> = (0..ROWS * LANES * N).map(|_| core::array::from_fn(|_| (next() >> 40) as i32)).collect();
                let mut acc: Vec<int32x4_t> = start.iter().map(|w| unsafe { vld1q_s32(w.as_ptr()) }).collect();
                unsafe {
                    if fresh {
                        sink::<true>(xs.as_ptr(), keys.as_ptr(), &mut acc, groups);
                    } else {
                        sink::<false>(xs.as_ptr(), keys.as_ptr(), &mut acc, groups);
                    }
                }
                for s in 0..N {
                    for r in 0..ROWS {
                        for c in 0..LANES {
                            let mut want = if fresh && r > 0 { [0i32; 4] } else { start[idx(s, r) + c] };
                            for o in 0..groups {
                                let x = bytes(&xs[s * XSTRIDE + c * OCT + o]);
                                let k = bytes(&keys[(s * OCT + o) * ROWS + r]);
                                for i in 0..2 {
                                    for j in 0..2 {
                                        let d: i32 = (0..8).map(|q| x[8 * i + q] as i32 * k[8 * j + q] as i32).sum();
                                        want[2 * i + j] = want[2 * i + j].wrapping_add(d);
                                    }
                                }
                            }
                            let got: [i32; 4] = unsafe { core::mem::transmute(acc[idx(s, r) + c]) };
                            assert_eq!(got, want, "groups {groups} fresh {fresh} slot {s} row {r} column {c}");
                        }
                    }
                }
            }
        }
    }

    /// The limbs reassemble their value: [`key_limbs`] over every centered residue, and
    /// [`super::super::pass_d_limbs`] against [`super::super::pass_d`] on random transforms.
    #[test]
    fn limbs_reassemble() {
        let h = (P1 as i16 - 1) / 2;
        for v in -h..=h {
            let b = key_limbs(&[v; LOCK]);
            assert!(b[LOCK].unsigned_abs() as i64 <= KHI);
            assert_eq!(b[0] as i32 + 256 * b[LOCK] as i32, v as i32);
        }
        let mut state = 41u64;
        let kern = &kernels()[PB];
        for _ in 0..20 {
            let mut x = [unsafe { vdupq_n_s16(0) }; N];
            for v in x.iter_mut() {
                let l: [i16; 8] = core::array::from_fn(|_| ((crate::ntt::tests::xorshift(&mut state) % (2 * h as u64 + 1)) as i64 - h as i64) as i16);
                *v = unsafe { vld1q_s16(l.as_ptr()) };
            }
            let mut out = vec![unsafe { vdupq_n_s8(0) }; 2 * N];
            let mut y = x;
            unsafe {
                super::super::pass_d_limbs::<PB>(&x, kern, out.as_mut_ptr(), 2);
                super::super::pass_d::<PB, false>(&mut y, kern, &[], &mut []);
            }
            for s in 0..N {
                let b: [i8; 16] = unsafe { core::mem::transmute(out[2 * s]) };
                let w: [i16; 8] = unsafe { core::mem::transmute(y[s]) };
                for e in 0..8 {
                    assert_eq!(b[e] as i32 + 256 * b[8 + e] as i32, w[e] as i32, "slot {s} lane {e}");
                }
            }
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

#[cfg(test)]
mod store_timing {
    use super::*;

    /// One sweep's 128 transforms with their outputs stored three ways: in place (the SDOT
    /// layout), as limbs into the sweep buffer (the SMMLA layout), and as limbs contiguously.
    /// `cargo test --release store_timing -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn store_timing() {
        let kern = &kernels()[PB];
        let mut state = 9u64;
        let words: Vec<u128> = (0..4 * 4096 * 8).map(|_| (crate::ntt::tests::xorshift(&mut state) as u128) << 64 | crate::ntt::tests::xorshift(&mut state) as u128).collect();
        let cols: [(*const u8, usize); LANES] = core::array::from_fn(|c| (words[c * 4 * 4096..].as_ptr() as *const u8, 64));
        let mut nib = Box::new([unsafe { vdupq_n_u8(0) }; N / 4]);
        let mut x = Box::new([unsafe { vdupq_n_s16(0) }; N]);
        let mut old = vec![[unsafe { vdupq_n_s16(0) }; N]; SWEEP];
        let mut xs = vec![unsafe { vdupq_n_s8(0) }; XS_LEN];
        let reps = 200;
        for round in 0..3 {
            let t = std::time::Instant::now();
            for rep in 0..reps {
                let e = (rep % 32) * SWEEP;
                for i in 0..SWEEP {
                    unsafe {
                        let at = octet_words(&cols, 4096, e + LOCK * (i % OCT), i / OCT);
                        indices(&at, &mut nib);
                        let y = &mut old[i];
                        pass_lut::<PB>(&nib, y, kern);
                        pass_c::<4>(y, kern);
                        pass_d::<PB, false>(y, kern, &[], &mut []);
                    }
                }
            }
            let a = t.elapsed().as_nanos() as f64 / (reps * SWEEP) as f64;
            let t = std::time::Instant::now();
            for rep in 0..reps {
                let e = (rep % 32) * SWEEP;
                for c in 0..LANES {
                    for o in 0..OCT {
                        unsafe {
                            let at = octet_words(&cols, 4096, e + LOCK * o, c);
                            indices(&at, &mut nib);
                            pass_lut::<PB>(&nib, &mut x, kern);
                            pass_c::<4>(&mut x, kern);
                            super::super::pass_d_limbs::<PB>(&x, kern, xs.as_mut_ptr().add(c * OCT + o), XSTRIDE);
                        }
                    }
                }
            }
            let b = t.elapsed().as_nanos() as f64 / (reps * SWEEP) as f64;
            let t = std::time::Instant::now();
            for rep in 0..reps {
                let e = (rep % 32) * SWEEP;
                for c in 0..LANES {
                    for o in 0..OCT {
                        unsafe {
                            let at = octet_words(&cols, 4096, e + LOCK * o, c);
                            indices(&at, &mut nib);
                            pass_lut::<PB>(&nib, &mut x, kern);
                            pass_c::<4>(&mut x, kern);
                            super::super::pass_d_limbs::<PB>(&x, kern, xs.as_mut_ptr().add((c * OCT + o) * N), 1);
                        }
                    }
                }
            }
            let d = t.elapsed().as_nanos() as f64 / (reps * SWEEP) as f64;
            println!("round {round}: in place {a:.0} ns, limbs to sweep buffer {b:.0} ns, limbs contiguous {d:.0} ns per transform");
        }
    }
}

#[cfg(test)]
mod asm_tests {
    use super::*;

    #[test]
    fn pass_d_limbs_asm_matches() {
        let kern = &kernels()[PB];
        let mut st = 29u64;
        for _ in 0..30 {
            let mut x: Box<[int16x8_t; N]> = unsafe { Box::new(core::mem::zeroed()) };
            let raw: Vec<i16> = (0..N * 8).map(|_| crate::ntt::tests::xorshift(&mut st) as i16).collect();
            for (i, v) in x.iter_mut().enumerate() { *v = unsafe { vld1q_s16(raw.as_ptr().add(8 * i)) }; }
            let mut a = vec![unsafe { vdupq_n_s8(0) }; N * XSTRIDE];
            let mut b = vec![unsafe { vdupq_n_s8(1) }; N * XSTRIDE];
            unsafe {
                super::super::pass_d_limbs::<PB>(&x, kern, a.as_mut_ptr().add(5), XSTRIDE);
                pass_d_limbs_asm(&x, kern, b.as_mut_ptr().add(5));
            }
            for s in 0..N {
                let (u, w): ([u8; 16], [u8; 16]) = unsafe { (core::mem::transmute(a[5 + s * XSTRIDE]), core::mem::transmute(b[5 + s * XSTRIDE])) };
                assert_eq!(u, w, "slot {s}");
            }
        }
    }

    #[test]
    fn pass_lut_asm_matches() {
        let kern = &kernels()[PB];
        let mut st = 17u64;
        for case in 0..50 {
            let raw: Vec<u8> = (0..N * 4).map(|_| crate::ntt::tests::xorshift(&mut st) as u8).collect();
            let mut idx = Box::new([unsafe { vdupq_n_u8(0) }; N / 4]);
            // half the cases with in-range table indices only, half with arbitrary bytes
            for (i, v) in idx.iter_mut().enumerate() {
                *v = unsafe { vld1q_u8(raw.as_ptr().add(16 * i)) };
                if case % 2 == 0 { *v = unsafe { vandq_u8(*v, vdupq_n_u8(31)) }; }
            }
            let mut a: Box<[int16x8_t; N]> = unsafe { Box::new(core::mem::zeroed()) };
            let mut b = a.clone();
            unsafe { pass_lut::<PB>(&idx, &mut a, kern); pass_lut_asm(&idx, &mut b, kern); }
            let x: &[[i16; 8]] = unsafe { core::slice::from_raw_parts(a.as_ptr() as *const [i16; 8], N) };
            let y: &[[i16; 8]] = unsafe { core::slice::from_raw_parts(b.as_ptr() as *const [i16; 8], N) };
            assert_eq!(x, y);
        }
    }

    #[test]
    fn pass_c_asm_matches() {
        let kern = &kernels()[PB];
        let mut st = 13u64;
        for _ in 0..50 {
            let mut a: Box<[int16x8_t; N]> = unsafe { Box::new(core::mem::zeroed()) };
            let raw: Vec<i16> = (0..N * 8).map(|_| crate::ntt::tests::xorshift(&mut st) as i16).collect();
            for (i, v) in a.iter_mut().enumerate() { *v = unsafe { vld1q_s16(raw.as_ptr().add(8 * i)) }; }
            let mut b = a.clone();
            unsafe { pass_c::<4>(&mut a, kern); pass_c_asm(&mut b, kern); }
            let x: &[[i16; 8]] = unsafe { core::slice::from_raw_parts(a.as_ptr() as *const [i16; 8], N) };
            let y: &[[i16; 8]] = unsafe { core::slice::from_raw_parts(b.as_ptr() as *const [i16; 8], N) };
            assert_eq!(x, y);
        }
    }

    #[test]
    fn pass_d_asm_matches() {
        let kern = &kernels()[PB];
        let mut st = 11u64;
        for _ in 0..50 {
            // inputs within pass_c's output bound for 9721 (values up to the lut transform bound)
            let bound = crate::ntt::lut_transform_bound(P1, crate::ntt::SCHEDULES[PB]) as u64;
            let mut a: Box<[int16x8_t; N]> = unsafe { Box::new(core::mem::zeroed()) };
            let raw: Vec<i16> = (0..N * 8).map(|_| ((crate::ntt::tests::xorshift(&mut st) % (2 * bound + 1)) as i64 - bound as i64) as i16).collect();
            for (i, v) in a.iter_mut().enumerate() { *v = unsafe { vld1q_s16(raw.as_ptr().add(8 * i)) }; }
            let mut b = a.clone();
            unsafe { pass_d::<PB, false>(&mut a, kern, &[], &mut []); pass_d_asm(&mut b, kern); }
            let x: &[[i16; 8]] = unsafe { core::slice::from_raw_parts(a.as_ptr() as *const [i16; 8], N) };
            let y: &[[i16; 8]] = unsafe { core::slice::from_raw_parts(b.as_ptr() as *const [i16; 8], N) };
            assert_eq!(x, y);
        }
    }
}

#[cfg(test)]
mod sink_timing {
    use super::*;

    /// One sweep of [`sink`] per call, keys streamed from a full 37 MB image or repeated from one
    /// sweep's 1.2 MB, on one thread and on eight at once. `cargo test --release sink_timing --
    /// --ignored --nocapture`.
    #[test]
    #[ignore]
    fn sink_timing() {
        let sweeps = 32;
        let keys: Vec<i8> = (0..sweeps * N * OCT * ROWS * 16).map(|i| (i * 7 % 251) as i8).collect();
        for threads in [1usize, 8] {
            for streamed in [true, false] {
                let keys = &keys;
                let per = std::thread::scope(|sc| {
                    let hs: Vec<_> = (0..threads).map(|_| sc.spawn(move || {
                        let xs = vec![unsafe { vdupq_n_s8(3) }; XS_LEN];
                        let mut acc = vec![unsafe { vdupq_n_s32(0) }; ROWS * LANES * N];
                        let t = std::time::Instant::now();
                        let reps = 96;
                        for rep in 0..reps {
                            let w = if streamed { rep % sweeps } else { 0 };
                            let k = unsafe { (keys.as_ptr() as *const int8x16_t).add(w * N * OCT * ROWS) };
                            unsafe { sink::<false>(xs.as_ptr(), k, &mut acc, OCT) };
                        }
                        t.elapsed().as_nanos() as f64 / reps as f64 / 1000.0
                    })).collect();
                    hs.into_iter().map(|h| h.join().unwrap()).sum::<f64>() / threads as f64
                });
                println!("threads {threads} keys {}: {per:.1} us per sweep", if streamed { "streamed 37 MB" } else { "one sweep, cached" });
            }
        }
    }
}
