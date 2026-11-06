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
use crate::ntt::{centered, for_blocks, front_transform_bound, FRONT_SCHEDULES};

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
pub const CHUNK: usize = 144;
/// Nodes of the encoder's closest-vector search: 1 is Babai's nearest plane alone, about 2000
/// times cheaper than the x86 build's 512 for a slightly larger digit spread (rms 3.88 against
/// 3.55). With [`CHUNK`] 64 the smallest margin is 12.16 standard deviations, and the Hoeffding
/// union bound over every chunk, digit and coefficient of a column is 9.8e-30 (1.0e-26 per proof of
/// 1024 columns); [`matrix`] computes both from the public matrix and refuses a weaker one.
const SEARCH: usize = 512;
/// The smallest admissible reconstruction margin, in standard deviations.
pub const MIN_MARGIN: f64 = 7.5;
/// The largest admissible failure bound of one column (`sum 2 exp(-z^2 / 2)`, Hoeffding, over
/// every chunk, digit and coefficient): 1e-16, i.e. below 1e-13 for a proof's 1024 columns.
pub const MAX_COLUMN_FAILURE: f64 = 1e-16;
/// Unit terms of one reduced product coefficient: a witness coefficient `i < 512` times a
/// coefficient `x < 648` folds onto coefficient `i + x` and, for `i + x >= 648`, two more.
const TERMS: i64 = 836;
// every chunk digit sum and its mean fit in i32 with room: |S| <= CHUNK * TERMS * 32
const _: () = assert!((CHUNK as i64) * TERMS * (lattice::MAX_DIGIT_ABS as i64) * 4 < i32::MAX as i64);

/// The SMS matrix: the transformed rows grouped by eight elements, `data[((g N + j) ROWS + r) 8 + e]`
/// = slot `j` of row `r` of element `8 g + e` (zero past the last element), centered residues; per chunk and digit the centered residue
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
        // the anonymous copy of this image, if one was made
        let data = key_copy(data).unwrap_or(data);
        let data = view::<i16>(data, GROUP * N * ROWS * elements.div_ceil(GROUP))?;
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

    /// The keys of group `g`: `[slot][row][element]`.
    #[inline]
    fn group(&self, g: usize) -> *const i16 {
        self.data[g * GROUP * N * ROWS..].as_ptr()
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
    let groups = elements.div_ceil(GROUP);
    let mut data = vec![0i16; GROUP * N * ROWS * groups];
    let raw = &raw;
    for_blocks(&mut data, GROUP * N * ROWS, 1, workers, |first, block| {
        let mut lanes = vec![[0i16; 8]; N];
        let lanes: &mut [[i16; 8]; N] = (&mut lanes[..]).try_into().unwrap();
        for (i, keys) in block.chunks_exact_mut(GROUP * N * ROWS).enumerate() {
            let base = GROUP * (first + i);
            let here = (elements - base).min(GROUP);
            for r in 0..ROWS {
                for (j, slot) in lanes.iter_mut().enumerate() {
                    for (e, s) in slot.iter_mut().enumerate() {
                        *s = if e < here { raw[(ROWS * (base + e) + r) * N + j] } else { 0 };
                    }
                }
                super::transform_general(PB, lanes);
                for (j, l) in lanes.iter().enumerate() {
                    for e in 0..GROUP {
                        keys[(j * ROWS + r) * GROUP + e] = if e < here { centered(l[e] as i64, p) } else { 0 };
                    }
                }
            }
        }
    });
    use std::borrow::Cow::Owned;
    SmsMatrix { elements, data: Owned(data), mean_residues: Owned(mean_residues), mu: Owned(mu), margin, column_failure }
}

/// The schedule of the binary transforms: the table front end of prime [`PB`]
/// ([`super::pass_ab`], `FRONT_SCHEDULES[PB]`).
const FRONT: usize = 4 + PB;
/// The largest product of a key slot (a centered residue) and a transform slot.
const PRODUCT: u64 = (P1 as u64 - 1) / 2 * front_transform_bound(P1, FRONT_SCHEDULES[PB]) as u64;
/// Elements whose products the SVE `i64` sums take below the `2^41` that [`sve::narrow`]'s
/// residues need (the main-stream sum runs over every element).
pub const MAX_ELEMENTS: usize = (((1u64 << 41) - 1) / PRODUCT) as usize;
const _: () = assert!(MAX_ELEMENTS >= 4096);

/// Elements per key group, transformed together and multiplied in one pass over the accumulators.
pub const GROUP: usize = 8;
const _: () = assert!(CHUNK % GROUP == 0);

/// The accumulator pair of slot `s`, row `r`: `acc[idx..idx + 2]`, columns 0..4 then 4..8.
#[inline(always)]
const fn idx(s: usize, r: usize) -> usize {
    (s * ROWS + r) * 2
}

/// `acc(s, r) -= key(e, s, r) y_e[s]` for the group's eight transformed elements (the keys of a
/// missing element are zero) over every slot and row: each accumulator read and written once.
#[inline(always)]
unsafe fn mac(xs: &[[int16x8_t; N]; GROUP], keys: *const i16, acc: &mut [int32x4_t]) {
    debug_assert!(acc.len() == ROWS * 2 * N);
    let accp = acc.as_mut_ptr();
    for j in 0..N {
        let y: [int16x8_t; GROUP] = core::array::from_fn(|e| *xs[e].get_unchecked(j));
        for r in 0..ROWS {
            let k = vld1q_s16(keys.add((j * ROWS + r) * GROUP));
            let p = accp.add(idx(j, r));
            let (mut lo, mut hi) = (*p, *p.add(1));
            macro_rules! el {
                ($e:literal) => {
                    lo = vmlsl_laneq_s16::<$e>(lo, vget_low_s16(y[$e]), k);
                    hi = vmlsl_high_laneq_s16::<$e>(hi, y[$e], k);
                };
            }
            el!(0); el!(1); el!(2); el!(3); el!(4); el!(5); el!(6); el!(7);
            *p = lo;
            *p.add(1) = hi;
        }
    }
}

/// SVE (vector length 128) halfword dot products: the MAC with `i64` accumulators, four elements
/// per product lane, after the record's `sms_dot.rs` (dadon, submissions d5e6c5a3 / b0a8dc26).
pub mod sve {
    use super::*;

    /// SVE present with 128-bit vectors (the `z` registers are then exactly the `v` registers).
    pub fn available() -> bool {
        use std::sync::OnceLock;
        static OK: OnceLock<bool> = OnceLock::new();
        *OK.get_or_init(|| {
            if std::env::var_os("LABINIUS_NO_SVE").is_some() || !std::arch::is_aarch64_feature_detected!("sve") {
                return false;
            }
            let bytes: usize;
            // SAFETY: `cntb` only reads the vector length; SVE is present.
            unsafe { core::arch::asm!(".arch_extension sve", "cntb {x}", x = out(reg) bytes, options(pure, nomem, nostack)) };
            bytes == 16
        })
    }

    /// Four elements' vectors (lane = column) as four vectors of two columns, each 64-bit lane
    /// the four elements of one column.
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

    /// `acc[0..4] += sum_e (-key_e) y_e` for elements 0..4 (`a`, `k0`) and 4..8 (`b`, `k1`).
    #[inline]
    #[target_feature(enable = "sve")]
    unsafe fn dot(acc: *mut int64x2_t, a: &[int16x8_t; 4], b: &[int16x8_t; 4], key: int16x8_t) {
        let key = vnegq_s16(key);
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
        *acc = s0;
        *acc.add(1) = s1;
        *acc.add(2) = s2;
        *acc.add(3) = s3;
    }

    /// [`super::mac`] into `i64` accumulators `acc[4 (j ROWS + r)..][..4]`: columns (0, 1), (2, 3),
    /// (4, 5), (6, 7). No reduction is ever needed: a full main-stream sum is below 2^41.
    #[target_feature(enable = "sve")]
    pub(super) unsafe fn mac(xs: &[[int16x8_t; N]; GROUP], keys: *const i16, acc: &mut [int64x2_t]) {
        debug_assert!(acc.len() == ROWS * 4 * N);
        let accp = acc.as_mut_ptr();
        for j in 0..N {
            let a = pack(xs[0][j], xs[1][j], xs[2][j], xs[3][j]);
            let b = pack(xs[4][j], xs[5][j], xs[6][j], xs[7][j]);
            for r in 0..ROWS {
                dot(accp.add(4 * (j * ROWS + r)), &a, &b, vld1q_s16(keys.add((j * ROWS + r) * GROUP)));
            }
        }
    }

    /// [`mac`] for transforms of `+s` in the dot-product lane order: `xs[4 h + q]`, lane `4 l + k`,
    /// is column `2 q + l` of the group's element `4 h + k`. A 64-bit lane then holds one column's
    /// four elements, and the halves of the key vector (elements 0..4, 4..8) are the indexed
    /// operands of `sdot`: no packing, no duplicated or negated keys.
    #[target_feature(enable = "sve")]
    /// With `FRESH` (a chunk's first tile) the digit rows' sums start from zero instead of being
    /// loaded: the chunk end needs no zeroing pass.
    pub(super) unsafe fn mac_lanes<const FRESH: bool>(xs: &[[int16x8_t; N]; GROUP], keys: *const i16, acc: &mut [int64x2_t]) {
        debug_assert!(acc.len() == ROWS * 4 * N);
        let accp = acc.as_mut_ptr();
        for j in 0..N {
            let y: [int16x8_t; GROUP] = core::array::from_fn(|t| *xs[t].get_unchecked(j));
            for r in 0..ROWS {
                let k = vld1q_s16(keys.add((j * ROWS + r) * GROUP));
                let p = accp.add(4 * (j * ROWS + r));
                let (mut s0, mut s1, mut s2, mut s3) = if FRESH && r > 0 {
                    (vdupq_n_s64(0), vdupq_n_s64(0), vdupq_n_s64(0), vdupq_n_s64(0))
                } else {
                    (*p, *p.add(1), *p.add(2), *p.add(3))
                };
                core::arch::asm!(
                    "sdot z0.d, z4.h, z12.h[0]", "sdot z1.d, z5.h, z12.h[0]",
                    "sdot z2.d, z6.h, z12.h[0]", "sdot z3.d, z7.h, z12.h[0]",
                    "sdot z0.d, z8.h, z12.h[1]", "sdot z1.d, z9.h, z12.h[1]",
                    "sdot z2.d, z10.h, z12.h[1]", "sdot z3.d, z11.h, z12.h[1]",
                    inout("v0") s0, inout("v1") s1, inout("v2") s2, inout("v3") s3,
                    in("v4") y[0], in("v5") y[1], in("v6") y[2], in("v7") y[3],
                    in("v8") y[4], in("v9") y[5], in("v10") y[6], in("v11") y[7],
                    in("v12") k,
                    options(pure, nomem, nostack, preserves_flags)
                );
                *p = s0;
                *p.add(1) = s1;
                *p.add(2) = s2;
                *p.add(3) = s3;
            }
        }
    }

    /// [`mac_lanes`] for two key groups (`xs[..8]` with `keys0`, `xs[8..]` with `keys1`): each
    /// sum loaded and stored once for sixteen elements.
    #[target_feature(enable = "sve")]
    pub(super) unsafe fn mac_lanes2<const FRESH: bool>(xs: &[[int16x8_t; N]; 2 * GROUP], keys0: *const i16, keys1: *const i16, acc: &mut [int64x2_t]) {
        debug_assert!(acc.len() == ROWS * 4 * N);
        let accp = acc.as_mut_ptr();
        for j in 0..N {
            let y: [int16x8_t; 2 * GROUP] = core::array::from_fn(|t| *xs[t].get_unchecked(j));
            for r in 0..ROWS {
                let k0 = vld1q_s16(keys0.add((j * ROWS + r) * GROUP));
                let k1 = vld1q_s16(keys1.add((j * ROWS + r) * GROUP));
                let p = accp.add(4 * (j * ROWS + r));
                let (mut s0, mut s1, mut s2, mut s3) = if FRESH && r > 0 {
                    (vdupq_n_s64(0), vdupq_n_s64(0), vdupq_n_s64(0), vdupq_n_s64(0))
                } else {
                    (*p, *p.add(1), *p.add(2), *p.add(3))
                };
                core::arch::asm!(
                    "sdot z0.d, z4.h, z12.h[0]", "sdot z1.d, z5.h, z12.h[0]",
                    "sdot z2.d, z6.h, z12.h[0]", "sdot z3.d, z7.h, z12.h[0]",
                    "sdot z0.d, z8.h, z12.h[1]", "sdot z1.d, z9.h, z12.h[1]",
                    "sdot z2.d, z10.h, z12.h[1]", "sdot z3.d, z11.h, z12.h[1]",
                    "sdot z0.d, z14.h, z13.h[0]", "sdot z1.d, z15.h, z13.h[0]",
                    "sdot z2.d, z16.h, z13.h[0]", "sdot z3.d, z17.h, z13.h[0]",
                    "sdot z0.d, z18.h, z13.h[1]", "sdot z1.d, z19.h, z13.h[1]",
                    "sdot z2.d, z20.h, z13.h[1]", "sdot z3.d, z21.h, z13.h[1]",
                    inout("v0") s0, inout("v1") s1, inout("v2") s2, inout("v3") s3,
                    in("v4") y[0], in("v5") y[1], in("v6") y[2], in("v7") y[3],
                    in("v8") y[4], in("v9") y[5], in("v10") y[6], in("v11") y[7],
                    in("v14") y[8], in("v15") y[9], in("v16") y[10], in("v17") y[11],
                    in("v18") y[12], in("v19") y[13], in("v20") y[14], in("v21") y[15],
                    in("v12") k0, in("v13") k1,
                    options(pure, nomem, nostack, preserves_flags)
                );
                *p = s0;
                *p.add(1) = s1;
                *p.add(2) = s2;
                *p.add(3) = s3;
            }
        }
    }

    /// `v mod 9721` in `[-4860, 4860]` for `|v| < 2^41`: the double-precision quotient is off by
    /// less than `1 / (2 p)`, so its rounding is the nearest integer quotient.
    #[inline(always)]
    unsafe fn residue(v: int64x2_t) -> int32x2_t {
        let q = vcvtnq_s64_f64(vmulq_n_f64(vcvtq_f64_s64(v), 1.0 / 9721.0));
        vmovn_s64(vsubq_s64(v, vmull_n_s32(vmovn_s64(q), 9721)))
    }

    /// Row `r`'s sums as canonical residues, one `i16` vector per slot, lane = column.
    #[inline(always)]
    pub(super) unsafe fn narrow(acc: &[int64x2_t], r: usize, out: &mut [int16x8_t; N]) {
        for (j, o) in out.iter_mut().enumerate() {
            let at = 4 * (j * ROWS + r);
            let lo = vcombine_s32(residue(acc[at]), residue(acc[at + 1]));
            let hi = vcombine_s32(residue(acc[at + 2]), residue(acc[at + 3]));
            *o = vcombine_s16(vmovn_s32(lo), vmovn_s32(hi));
        }
    }

    /// `2^16 mod p`, centered.
    const C16: i32 = centered(1 << 16, P1) as i32;
    /// `round(2^31 / p)`.
    const V31: i32 = ((1u64 << 32) / P1 as u64).div_ceil(2) as i32;
    // a chunk's digit sums `v` (|v| <= CHUNK PRODUCT): `(v >> 16) C16 + (v mod 2^16)` fits i32
    const _: () = assert!(((CHUNK as u64 * PRODUCT >> 16) + 1) * C16.unsigned_abs() as u64 + 65535 < i32::MAX as u64);

    /// [`narrow`] for a chunk's digit rows in integers: `v = h 2^16 + l`, `l = v mod 2^16`, so
    /// `v = h (2^16 mod p) + l (mod p)` within `i32`, then a 32-bit Barrett reduction to
    /// `|r| <= 0.65 p` (the inverse takes `|x| <= p`).
    #[inline(always)]
    pub(super) unsafe fn narrow_digits(acc: &[int64x2_t], r: usize, out: &mut [int16x8_t; N]) {
        let c = vdupq_n_s32(C16);
        let low = vdupq_n_s32(0xFFFF);
        let half = |a: int64x2_t, b: int64x2_t| -> int32x4_t {
            let h = vshrn_high_n_s64::<16>(vshrn_n_s64::<16>(a), b);
            let l = vandq_s32(vmovn_high_s64(vmovn_s64(a), b), low);
            let t = vmlaq_s32(l, h, c);
            vmlsq_n_s32(t, vqrdmulhq_n_s32(t, V31), P1 as i32)
        };
        for (j, o) in out.iter_mut().enumerate() {
            let at = 4 * (j * ROWS + r);
            let lo = half(*acc.get_unchecked(at), *acc.get_unchecked(at + 1));
            let hi = half(*acc.get_unchecked(at + 2), *acc.get_unchecked(at + 3));
            *o = vmovn_high_s32(vmovn_s32(lo), hi);
        }
    }

    /// Row `r`'s sums as reduced `i32` pairs in the layout of the NEON accumulators.
    pub(super) unsafe fn to_i32(acc: &[int64x2_t], r: usize, out: &mut [int32x4_t]) {
        for j in 0..N {
            let at = 4 * (j * ROWS + r);
            out[idx(j, r)] = vcombine_s32(residue(acc[at]), residue(acc[at + 1]));
            out[idx(j, r) + 1] = vcombine_s32(residue(acc[at + 2]), residue(acc[at + 3]));
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        #[test]
        fn digit_narrowing_is_exact() {
            if !available() {
                return;
            }
            let bound = (CHUNK as u64 * PRODUCT) as i64;
            let mut state = 5u64;
            let mut acc = vec![unsafe { vdupq_n_s64(0) }; ROWS * 4 * N];
            let edge = [bound, -bound, bound - 1, -bound + 1, 0, 1, -1, 65535, -65536, 4860, -4861];
            let mut want = vec![0i64; ROWS * 8 * N];
            for (i, w) in want.iter_mut().enumerate() {
                *w = if i < edge.len() { edge[i] } else { (crate::ntt::tests::xorshift(&mut state) % (2 * bound as u64 + 1)) as i64 - bound };
            }
            unsafe {
                core::ptr::copy_nonoverlapping(want.as_ptr() as *const int64x2_t, acc.as_mut_ptr(), acc.len());
                for r in 0..ROWS {
                    let mut out = [vdupq_n_s16(0); N];
                    narrow_digits(&acc, r, &mut out);
                    for j in 0..N {
                        let o: [i16; 8] = core::mem::transmute(out[j]);
                        for c in 0..8 {
                            let v = want[(4 * (j * ROWS + r)) * 2 + c];
                            assert_eq!((o[c] as i64 - v).rem_euclid(P1 as i64), 0, "{v}");
                            assert!((o[c] as i64).abs() <= P1 as i64);
                        }
                    }
                }
            }
        }

        #[test]
        fn residue_is_exact() {
            if !available() {
                eprintln!("no SVE here; skipped");
                return;
            }
            for v in [-(1i64 << 41) + 1, -9722, -4861, -4860, -1, 0, 1, 4860, 4861, 9722, (1 << 41) - 1, 123456789012] {
                let r: [i32; 2] = unsafe { core::mem::transmute(residue(vdupq_n_s64(v))) };
                assert_eq!((r[0] as i64 - v).rem_euclid(9721), 0);
                assert!((r[0] as i64).abs() <= 4860);
            }
        }
    }
}

/// Row `r`'s reduced `i32` slot sums (`|v| <= p`) as one `i16` vector per slot, lane = column.
#[inline(always)]
unsafe fn narrow(acc: &[int32x4_t], r: usize, out: &mut [int16x8_t; N]) {
    for (s, o) in out.iter_mut().enumerate() {
        *o = vcombine_s16(vmovn_s32(acc[idx(s, r)]), vmovn_s32(acc[idx(s, r) + 1]));
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

/// The batch's pass: the base slot sums (row 0 of `acc`, reduced, `|v| <= p`), and each chunk's
/// digit sums inverted to canonical coefficients, handed to `chunk(c, d, coefficients)`. With
/// `acc64` (SVE available) the products go into its `i64` lanes instead.
#[inline(never)]
unsafe fn sums(
    words: &[(*const u8, usize); LANES],
    a: &SmsMatrix,
    ix: &mut [uint8x16_t; 162],
    xs: &mut [[int16x8_t; N]; 2 * GROUP],
    acc: &mut [int32x4_t],
    mut acc64: Option<&mut [int64x2_t]>,
    y: &mut [int16x8_t; N],
    chunk: impl FnMut(usize, usize, &mut [int16x8_t; N]),
) {
    acc.fill(vdupq_n_s32(0));
    if let Some(w) = acc64.as_deref_mut() {
        w.fill(vdupq_n_s64(0));
    }
    let mut since = 0usize;
    sums_range(words, a, 0, a.elements.div_ceil(GROUP), ix, xs, acc, acc64.as_deref_mut(), y, &mut since, chunk);
    finish_sums(acc, acc64.as_deref());
}

/// The base slot sums as reduced `i32` (`|v| <= p`) in `acc`, row 0.
#[inline(never)]
unsafe fn finish_sums(acc: &mut [int32x4_t], acc64: Option<&[int64x2_t]>) {
    match acc64 {
        Some(w) => sve::to_i32(w, 0, acc),
        None => reduce_sums(acc, P1 as i32, kernels()[PB].v32),
    }
}

/// [`sums`] over the key groups `g0..g1` (`g1` a chunk end or inside a chunk), from and into the
/// running sums: the chunk ends inside the range are handed to `chunk`. `since` is the NEON path's
/// count of elements summed since the last reduction.
#[inline(never)]
#[allow(clippy::too_many_arguments)]
unsafe fn sums_range(
    words: &[(*const u8, usize); LANES],
    a: &SmsMatrix,
    g0: usize,
    g1: usize,
    ix: &mut [uint8x16_t; 162],
    xs: &mut [[int16x8_t; N]; 2 * GROUP],
    acc: &mut [int32x4_t],
    mut acc64: Option<&mut [int64x2_t]>,
    y: &mut [int16x8_t; N],
    since: &mut usize,
    mut chunk: impl FnMut(usize, usize, &mut [int16x8_t; N]),
) {
    let kern = &kernels()[PB];
    let front = &fronts()[PB];
    let front_plus = &fronts_plus()[PB];
    let s: Schedule = schedule(FRONT);
    const { assert!(GROUP <= FRONT_SCHEDULES[PB].accumulate) };
    let p = P1 as i32;
    let mut g = g0;
    while g < g1 {
        let e0 = GROUP * g;
        // SVE: two key groups per pass over the sums when the chunk does not end after the first
        let two = acc64.is_some() && g + 1 < g1 && (e0 + GROUP) % CHUNK != 0;
        // SVE: the chunk's first tile overwrites the digit rows' sums
        let fresh = e0 % CHUNK == 0;
        let here = (a.elements - e0).min(if two { 2 * GROUP } else { GROUP });
        match acc64.as_deref_mut() {
            Some(w) => {
                // the dot-product lane order ([`sve::mac_lanes`]): transform `4 h + q`, lane
                // `4 l + k` is column `2 q + l` of element `e0 + 4 h + k`; missing elements read
                // zeros (their keys are zero as well)
                for (t, x) in xs.iter_mut().enumerate().take(if two { 2 * GROUP } else { GROUP }) {
                    let at: [*const u8; LANES] = core::array::from_fn(|lane| {
                        let (c, e) = (2 * (t % 4) + lane / 4, e0 + 4 * (t / 4) + lane % 4);
                        if e < a.elements { words[c].0.add(e * words[c].1) } else { ZERO_WORDS.as_ptr() as *const u8 }
                    });
                    indices(&at, ix);
                    pass_ab4::<FRONT>(ix, x, kern, front_plus);
                    pass_56::<FRONT>(x, kern);
                }
                if two {
                    if fresh {
                        sve::mac_lanes2::<true>(xs, a.group(g), a.group(g + 1), w);
                    } else {
                        sve::mac_lanes2::<false>(xs, a.group(g), a.group(g + 1), w);
                    }
                } else {
                    if fresh {
                        sve::mac_lanes::<true>((&xs[..GROUP]).try_into().unwrap(), a.group(g), w);
                    } else {
                        sve::mac_lanes::<false>((&xs[..GROUP]).try_into().unwrap(), a.group(g), w);
                    }
                }
            }
            None => {
                for i in 0..here {
                    let at: [*const u8; LANES] = core::array::from_fn(|c| words[c].0.add((e0 + i) * words[c].1));
                    indices(&at, ix);
                    let x = &mut xs[i];
                    pass_ab4::<FRONT>(ix, x, kern, front);
                    pass_56::<FRONT>(x, kern);
                }
                // a missing element's keys are zero; its stale transform contributes nothing
                mac((&xs[..GROUP]).try_into().unwrap(), a.group(g), acc);
            }
        }
        g += if two { 2 } else { 1 };
        let e = e0 + here;
        *since += GROUP;
        let end = e % CHUNK == 0 || e == a.elements;
        if end {
            if acc64.is_none() {
                reduce_sums(acc, p, kern.v32);
            }
            *since = 0;
            for d in 0..DIGITS {
                match acc64.as_deref() {
                    Some(w) => sve::narrow_digits(w, 1 + d, y),
                    None => narrow(acc, 1 + d, y),
                }
                super::inverse::<PB>(y);
                chunk((e - 1) / CHUNK, d, y);
            }
            // SVE: the next chunk's first tile starts the digit rows' sums afresh
            match acc64.as_deref_mut() {
                Some(_) => {}
                None => {
                    for sl in 0..N {
                        for d in 0..DIGITS {
                            acc[idx(sl, 1 + d)] = vdupq_n_s32(0);
                            acc[idx(sl, 1 + d) + 1] = vdupq_n_s32(0);
                        }
                    }
                }
            }
        } else if acc64.is_none() && *since + GROUP > s.accumulate {
            reduce_sums(acc, p, kern.v32);
            *since = 0;
        }
    }
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

/// Batches (of eight columns) per pass over the key on the SVE path ([`commit`]): each key block
/// of [`BLOCK`] groups is read once for this many batches, whose sums (290 KB of `i64` and 124 KB
/// of chunk deviations each) stay in L2 beside the block.
const PASS: usize = 3;
/// Key groups per block of a pass (32 elements, 290 KB of key); blocks end at chunk ends.
const BLOCK: usize = 4;

/// The commitment of every column of `values` (columns of `4 a.elements` words) modulo
/// [`MODULUS`].
pub fn commit(values: &[B128], a: &SmsMatrix, workers: usize) -> Vec<u128> {
    let len = 4 * a.elements;
    assert_eq!(values.len() % len, 0);
    assert!(a.elements <= MAX_ELEMENTS, "the i64 slot sums need |sum| < 2^41");
    let columns = values.len() / len;
    let mut out = vec![0u128; columns * N];
    let sve = sve::available();
    for_blocks(&mut out, N, LANES, workers, |first, block| unsafe {
        let mut ix = Box::new([vdupq_n_u8(0); 162]);
        let mut xs = Box::new([[vdupq_n_s16(0); N]; 2 * GROUP]);
        let mut y = Box::new([vdupq_n_s16(0); N]);
        let mut acc = vec![vdupq_n_s32(0); ROWS * 2 * N];
        let p = P1 as i16;
        let count = block.len() / N;
        let batch_words = |b: usize| -> [(*const u8, usize); LANES] {
            let col0 = first + LANES * b;
            let here = (count - LANES * b).min(LANES);
            core::array::from_fn(|c| {
                if c < here {
                    (values[(col0 + c) * len..].as_ptr() as *const u8, 64)
                } else {
                    (ZERO_WORDS.as_ptr() as *const u8, 0)
                }
            })
        };
        // per digit, the chunk deviations S - mean summed over the chunks, into `dev`
        let deviations = |dev: &mut [int32x4_t], c: usize, d: usize, r: &mut [int16x8_t; N]| {
            let mr = a.mean_residues(c, d);
            let dv = &mut dev[d * 2 * N..][..2 * N];
            for s in 0..N {
                let v = canonical(vsubq_s16(r[s], vdupq_n_s16(mr[s])), p);
                dv[2 * s] = vaddw_s16(dv[2 * s], vget_low_s16(v));
                dv[2 * s + 1] = vaddw_high_s16(dv[2 * s + 1], v);
            }
        };
        let batches: Vec<&mut [u128]> = block.chunks_mut(LANES * N).collect();
        if sve {
            // key-block outer, batch inner: PASS batches share each block's key reads
            let groups = a.elements.div_ceil(GROUP);
            let mut accs: Vec<Vec<int64x2_t>> = (0..PASS.min(batches.len())).map(|_| vec![vdupq_n_s64(0); ROWS * 4 * N]).collect();
            let mut devs: Vec<Vec<int32x4_t>> = (0..accs.len()).map(|_| vec![vdupq_n_s32(0); DIGITS * 2 * N]).collect();
            let mut batches = batches;
            for (pass, targets) in batches.chunks_mut(PASS).enumerate() {
                let words: Vec<[(*const u8, usize); LANES]> = (0..targets.len()).map(|i| batch_words(PASS * pass + i)).collect();
                for i in 0..targets.len() {
                    accs[i].fill(vdupq_n_s64(0));
                    devs[i].fill(vdupq_n_s32(0));
                }
                let mut g0 = 0;
                while g0 < groups {
                    // the block ends at BLOCK groups or at the chunk's end
                    let chunk_end = ((GROUP * g0 / CHUNK + 1) * CHUNK).div_ceil(GROUP).min(groups);
                    let g1 = (g0 + BLOCK).min(chunk_end);
                    for i in 0..targets.len() {
                        let dev = &mut devs[i];
                        let mut since = 0;
                        sums_range(&words[i], a, g0, g1, &mut ix, &mut xs, &mut acc, Some(&mut accs[i]), &mut y, &mut since, |c, d, r| deviations(dev, c, d, r));
                    }
                    g0 = g1;
                }
                for (i, target) in targets.iter_mut().enumerate() {
                    finish_sums(&mut acc, Some(&accs[i]));
                    columns_out(a, &acc, &devs[i], &mut y, target);
                }
            }
        } else {
            let mut dev = vec![vdupq_n_s32(0); DIGITS * 2 * N];
            for (b, target) in batches.into_iter().enumerate() {
                let words = batch_words(b);
                dev.fill(vdupq_n_s32(0));
                sums(&words, a, &mut ix, &mut xs, &mut acc, None, &mut y, |c, d, r| deviations(&mut dev, c, d, r));
                columns_out(a, &acc, &dev, &mut y, target);
            }
        }
    });
    out
}

/// A batch's columns from its reduced base slot sums (`acc` row 0) and chunk deviations: the base
/// row of all eight columns through one vector inverse transform (lane = column), after the
/// e-man07 record (submission 3edf439b), instead of a scalar inverse per column.
unsafe fn columns_out(a: &SmsMatrix, acc: &[int32x4_t], dev: &[int32x4_t], y: &mut [int16x8_t; N], target: &mut [u128]) {
    narrow(acc, 0, y);
    super::inverse::<PB>(y);
    // The centered residues (`|x| <= (p - 1) / 2`) to `[0, p)` lane-wise, eight columns per
    // vector: `+ p` where negative, the value `rem_euclid` gives.
    let pv = vdupq_n_s16(P1 as i16);
    let mut base = [[0u16; LANES]; N];
    for (k, b) in base.iter_mut().enumerate() {
        let v = y[k];
        let v = vaddq_s16(v, vandq_s16(vreinterpretq_s16_u16(vcltzq_s16(v)), pv));
        vst1q_u16(b.as_mut_ptr(), vreinterpretq_u16_s16(v));
    }
    // `tail = mu + sum_d W_d dev_d` ([`add_weighted`]'s map) for the eight columns at once, in
    // `i64` lanes: one pass per digit over the slots instead of one per column.
    let devs: &[[i32; 4]] = core::slice::from_raw_parts(dev.as_ptr() as *const [i32; 4], dev.len());
    let mut tail = [[0i64; LANES]; N];
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
            *o = crt(base[k][c] as u32, tail[k][c].rem_euclid(TAIL as i64) as u32);
        }
    }
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
