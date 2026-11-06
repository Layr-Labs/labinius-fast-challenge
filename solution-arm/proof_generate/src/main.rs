//! The ARM track's prover: the CLI, transcript and proof format of the x86 track's `labinius-fast`,
//! computed without AVX-512.
//!
//! The default commitment modulus is `Q = 3889 * 9721 * 17497` ([`ntt::MODULUS`]). Every factor is
//! `1 mod 1944`, so the commitment `C_j = A W_j` is computed per prime through number-theoretic
//! transforms of `R_648` and recombined by the Chinese remainder theorem ([`ntt`]). Any other
//! modulus the checker accepts takes the earlier paths: below [`FAST_MODULUS`] a binary witness
//! ring element makes `A_m W_{j,m}` a sum of shifted copies of `A_m`, eight coefficients per lookup
//! of a 256-entry table of partial sums (the "four Russians" trick); above it the schoolbook
//! product. The field switch, row evaluation, and fold follow the checker's reference prover.
mod hugealloc;
mod ntt;

#[global_allocator]
static ALLOCATOR: hugealloc::Allocator = hugealloc::Allocator;
#[allow(dead_code)]
mod rq;

use labinius::challenge::{sample_short_challenge, ShortChallenge, DEFAULT_BOUND, DEFAULT_WEIGHT};
use labinius::eval::eq_table;
use labinius::fields::{
    crossfield as cf,
    scalar::{B128, F162},
};
use labinius::ring::Representation;
use labinius::scheme::FoldingSource;
use labinius::{wire, FoldedWitness, RingElement648, RowEvaluation, Transcript};
use serde::Serialize;

const LOG_VARS: usize = 31;
const LOG_PACKED: usize = LOG_VARS - cf::LOG_PACK;
const COLUMN_LOG: u32 = 10;
const COLUMNS: usize = 1 << COLUMN_LOG;
const ROW_LOG: usize = LOG_PACKED - COLUMN_LOG as usize;
const COLUMN_LEN: usize = 1 << ROW_LOG;
const RING_ELEMENTS: usize = COLUMN_LEN / 4;
const LIMIT: u64 = 64 * 1024 * 1024;
const MIN_MODULUS: u128 = 34_816_965_049;
/// The default commitment modulus: the product of the three transform primes.
/// The hybrid commitment matrix in the transform domain, built by build.rs (empty when the build
/// host could not; the prover then builds it at run time).
#[repr(C, align(64))]
struct Aligned64<T: ?Sized>(T);
const HYBRID_MATRIX_LEN: usize = include_bytes!(concat!(env!("OUT_DIR"), "/hybrid-matrix.bin")).len();
static HYBRID_MATRIX: Aligned64<[u8; HYBRID_MATRIX_LEN]> =
    Aligned64(*include_bytes!(concat!(env!("OUT_DIR"), "/hybrid-matrix.bin")));
macro_rules! image {
    ($name:ident, $file:literal) => {
        static $name: &Aligned64<[u8]> = &Aligned64(*include_bytes!(concat!(env!("OUT_DIR"), "/", $file)));
    };
}
// The SMS matrix images, built by build.rs (empty when the build host could not; the prover then
// builds the matrix at run time).
image!(SMS_DATA, "sms-data.bin");
image!(SMS_MEANS, "sms-means.bin");
image!(SMS_MU, "sms-mu.bin");
image!(SMS_META, "sms-meta.bin");
const MODULUS: u128 = ntt::neon::sms::MODULUS;

/// Coefficients of a witness ring element per table lookup in [`commit`].
const GROUP: usize = 8;
/// Coefficients a binary witness ring element can have set: F162 lifts B128, so bit `p < 128`.
const WITNESS_COEFFICIENTS: usize = 4 * 128;
const GROUPS: usize = WITNESS_COEFFICIENTS / GROUP;
/// One table row: `A_m` shifted by up to `GROUP - 1`, padded to an even length.
const ROW: usize = (GROUP - 1 + rq::N + 1) & !1;
/// The unreduced product of a witness column with `A`, padded like [`ROW`].
const RAW: usize = (GROUPS - 1) * GROUP + ROW;
/// Largest modulus [`commit`]'s `u64` accumulator holds: a raw coefficient sums at most
/// `WITNESS_COEFFICIENTS * RING_ELEMENTS = 2^21` residues below `Q`.
const FAST_MODULUS: u128 = 1 << 43;
/// Raw coefficients [`commit`] sums in registers per pass over a column's groups.
const BLOCK: usize = 32;
/// A table row stored between a block of zeros on either side, so that every block-sized
/// window a group can reach reads in bounds, zeros outside the row.
const PADDED: usize = BLOCK + ROW + BLOCK;
/// Output blocks per column, covering the raw product.
const BLOCKS: usize = RAW.div_ceil(BLOCK);
/// How many blocks ahead [`add_block`] prefetches each group's table window; 0 disables it.
const PREFETCH_BLOCKS: usize = 1;

#[derive(Serialize)]
struct Proof {
    version: u32,
    modulus: u128,
    commitment: Vec<u8>,
    partials: Vec<u128>,
    rounds: Vec<[[u64; 3]; 2]>,
    opened: [u64; 3],
    row: Vec<u8>,
    folded: Vec<u8>,
}

#[cfg(test)]
fn codec() -> impl bincode::Options {
    use bincode::Options;
    bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .with_limit(LIMIT)
        .reject_trailing_bytes()
}

fn seed() -> [u8; 32] {
    let s = std::env::var("LABINIUS_SEED").expect("LABINIUS_SEED is required");
    assert_eq!(s.len(), 64);
    core::array::from_fn(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).expect("invalid seed"))
}

fn parse_modulus(text: &str) -> Result<u128, String> {
    let q: u128 = text
        .parse()
        .map_err(|_| format!("not a decimal integer: {text}"))?;
    if q < MIN_MODULUS {
        return Err("commitment modulus is below the security bound".into());
    }
    if q >= rq::MAX_MODULUS {
        return Err("commitment modulus is above what the checker's arithmetic covers".into());
    }
    Ok(q)
}

fn threads() -> usize {
    labinius::par::threads()
}

/// The read-only file mapping avoids a 256 MiB copy. Other platforms retain the owned path.
enum Table {
    #[cfg(all(target_os = "linux", target_endian = "little"))]
    Mapped { ptr: *mut u8, len: usize },
    Owned(Vec<B128>),
}

#[cfg(all(target_os = "linux", target_endian = "little"))]
mod table_sys {
    extern "C" {
        pub fn mmap(addr: *mut u8, len: usize, prot: i32, flags: i32, fd: i32, off: i64) -> *mut u8;
        pub fn munmap(addr: *mut u8, len: usize) -> i32;
        pub fn madvise(addr: *mut u8, len: usize, advice: i32) -> i32;
    }
}

impl std::ops::Deref for Table {
    type Target = [B128];

    fn deref(&self) -> &[B128] {
        match self {
            #[cfg(all(target_os = "linux", target_endian = "little"))]
            Self::Mapped { ptr, len } => {
                // SAFETY: this mapping is read-only, alive until Drop, page aligned, and contains
                // exactly len little-endian words. B128 is transparent over u128.
                unsafe { std::slice::from_raw_parts(*ptr as *const B128, *len) }
            }
            Self::Owned(words) => words,
        }
    }
}

impl Drop for Table {
    fn drop(&mut self) {
        #[cfg(all(target_os = "linux", target_endian = "little"))]
        if let Self::Mapped { ptr, len } = self {
            unsafe { table_sys::munmap(*ptr, 16 * *len) };
        }
    }
}

fn read_table(path: &str) -> Table {
    load_table(path, 1 << LOG_PACKED)
}

fn load_table(path: &str, n: usize) -> Table {
    use std::os::unix::fs::FileExt;
    let file = std::fs::File::open(path).expect("read witness table");
    let len = file.metadata().expect("read witness table").len();
    assert_eq!(len, 16 * n as u64, "witness table has the wrong length");
    #[cfg(all(target_os = "linux", target_endian = "little"))]
    if n != 0 {
        use std::os::fd::AsRawFd;
        // PROT_READ | MAP_PRIVATE: the trusted input is never writable.
        let ptr = unsafe { table_sys::mmap(std::ptr::null_mut(), 16 * n, 1, 2, file.as_raw_fd(), 0) };
        if ptr as isize != -1 {
            let base = ptr as usize;
            // The range boundaries are 2 MiB aligned, also valid on 4/16/64 KiB kernels.
            labinius::par::map_blocks(n, threads(), (2 << 20) / 16, |_, s, e| {
                if s < e {
                    let addr = (base + 16 * s) as *mut u8;
                    // Linux 5.14+: install the page-cache pages in parallel. Older kernels
                    // simply fault on access. MADV_HUGEPAGE is a best-effort hint for the file.
                    unsafe {
                        table_sys::madvise(addr, 16 * (e - s), 14);
                        table_sys::madvise(addr, 16 * (e - s), 22);
                    }
                }
            });
            return Table::Mapped { ptr, len: n };
        }
    }
    let mut words = vec![0u128; n];
    let per = n.div_ceil(threads()).max(1);
    std::thread::scope(|scope| {
        for (i, chunk) in words.chunks_mut(per).enumerate() {
            let file = &file;
            scope.spawn(move || {
                // SAFETY: the bytes of initialized `u128`s; every pattern is a valid `u128`.
                let bytes = unsafe {
                    std::slice::from_raw_parts_mut(chunk.as_mut_ptr() as *mut u8, 16 * chunk.len())
                };
                file.read_exact_at(bytes, (16 * per * i) as u64)
                    .expect("read witness table");
            });
        }
    });
    #[cfg(target_endian = "big")]
    for w in words.iter_mut() {
        *w = u128::from_le(*w);
    }
    let mut words = std::mem::ManuallyDrop::new(words);
    // SAFETY: `B128` is `repr(transparent)` over `u128`: same size, alignment and validity.
    Table::Owned(unsafe {
        Vec::from_raw_parts(words.as_mut_ptr() as *mut B128, words.len(), words.capacity())
    })
}

fn transcript(seed: &[u8; 32], modulus: u128, commitment: &[u8]) -> Transcript {
    let mut t = Transcript::new(b"labinius-fast/v1");
    t.absorb_bytes(seed);
    t.absorb_u64(LOG_VARS as u64);
    t.absorb_u64(COLUMN_LOG as u64);
    t.absorb_bytes(&modulus.to_le_bytes());
    t.absorb_bytes(commitment);
    t
}

/// The table of group `pattern`s for ring element `element` of `A`: row `s` is the sum of
/// `A_m X^r` over the set bits `r` of `s`. Each row is a previous row plus one shifted copy.
/// Row `s` occupies `table[s * PADDED..][..PADDED]` with its `ROW` coefficients from `BLOCK`.
/// The table starts zeroed and only the coefficients are written, so row 0 and the padding
/// stay zero; each row is written in one pass from the previous one.
fn build_table(table: &mut [u64], element: &[u64]) {
    for s in 1usize..1 << GROUP {
        let r = s.ilog2() as usize;
        let (done, rest) = table.split_at_mut(s * PADDED);
        let src = &done[(s & !(1 << r)) * PADDED + BLOCK..][..ROW];
        let dst = &mut rest[BLOCK..BLOCK + ROW];
        dst[..r].copy_from_slice(&src[..r]);
        for ((d, &x), &a) in dst[r..r + rq::N].iter_mut().zip(&src[r..r + rq::N]).zip(element) {
            *d = x + a;
        }
        dst[r + rq::N..].copy_from_slice(&src[r + rq::N..]);
    }
}

/// The groups whose shifted rows reach output block `b`: group `g` adds row coefficient `i - 8g`
/// to output coefficient `i`, nonzero for `0 <= i - 8g < ROW`.
const fn group_range(b: usize) -> (usize, usize) {
    let first = b * BLOCK;
    let lo = if first + 1 > ROW { (first + 1 - ROW).div_ceil(GROUP) } else { 0 };
    let hi = (first + BLOCK - 1) / GROUP;
    (lo, if hi < GROUPS - 1 { hi } else { GROUPS - 1 })
}

/// `acc += sum_g table[start_g + first..][..BLOCK]` for output block `b` of one column, with
/// `start_g` the padded row of group `g`'s pattern moved back by its shift `8g`; the block's
/// sums stay in registers across the groups.
#[inline(always)]
fn add_block(acc: &mut [u64], table: &[u64], starts: &[usize; GROUPS], b: usize) {
    let (lo, hi) = group_range(b);
    let first = b * BLOCK;
    let out: &mut [u64; BLOCK] = (&mut acc[first..first + BLOCK]).try_into().unwrap();
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use core::arch::aarch64::*;
        let mut sum: [uint64x2_t; BLOCK / 2] =
            core::array::from_fn(|i| vld1q_u64(out.as_ptr().add(2 * i)));
        for &start in &starts[lo..=hi] {
            let at = start.wrapping_add(first);
            if PREFETCH_BLOCKS > 0 {
                // The same row's window `PREFETCH_BLOCKS` blocks on; a hint, never a fault.
                let ahead = table.as_ptr().wrapping_add(at + PREFETCH_BLOCKS * BLOCK);
                for line in 0..BLOCK / 8 {
                    core::arch::asm!(
                        "prfm pldl1keep, [{p}]",
                        p = in(reg) ahead.wrapping_add(8 * line),
                        options(nostack, preserves_flags, readonly)
                    );
                }
            }
            let row: &[u64; BLOCK] = table[at..at + BLOCK].try_into().unwrap();
            for (i, s) in sum.iter_mut().enumerate() {
                *s = vaddq_u64(*s, vld1q_u64(row.as_ptr().add(2 * i)));
            }
        }
        for (i, s) in sum.iter().enumerate() {
            vst1q_u64(out.as_mut_ptr().add(2 * i), *s);
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let mut sum = *out;
        for &start in &starts[lo..=hi] {
            let at = start.wrapping_add(first);
            let row: &[u64; BLOCK] = table[at..at + BLOCK].try_into().unwrap();
            for (s, &x) in sum.iter_mut().zip(row) {
                *s += x;
            }
        }
        *out = sum;
    }
}

/// Group `g` of ring element `m` of a column as a table index. Coefficient `4 p + k` is bit `p`
/// of word `k`, so the group's eight coefficients are bits `2g` and `2g + 1` of the four words.
#[inline(always)]
fn pattern(words: &[u128; 4], g: usize) -> usize {
    let mut s = 0;
    for (k, &w) in words.iter().enumerate() {
        let pair = (w >> (2 * g)) as usize & 3;
        s |= (pair & 1) << k | (pair >> 1) << (4 + k);
    }
    s
}

/// A raw product reduced modulo `q` and folded into `R_Q` with `X^648 = X^324 - 1`.
fn reduce(raw: &[u64], q: u64) -> Vec<u128> {
    let mut r = vec![0u64; 2 * rq::N - 1];
    for (slot, &x) in r.iter_mut().zip(raw) {
        *slot = x % q;
    }
    for k in (rq::N..2 * rq::N - 1).rev() {
        let v = r[k];
        let up = &mut r[k - rq::SPLIT];
        *up += v;
        if *up >= q {
            *up -= q;
        }
        let down = &mut r[k - rq::N];
        *down += q - v;
        if *down >= q {
            *down -= q;
        }
    }
    r[..rq::N].iter().map(|&x| x as u128).collect()
}

/// Every column's commitment, flattened. Columns go out over the cores; each worker walks the
/// ring elements once, building that element's table and applying it to all of its columns.
/// Per column the raw product is summed block by block: each `BLOCK` of output coefficients
/// takes the slices of every group's table row that land on it while the block's sums stay in
/// registers, so the accumulator is read and written once per block instead of once per group.
///
/// The default modulus [`ntt::MODULUS`] goes through the transforms of [`ntt`] instead.
fn commit(values: &[B128], modulus: u128) -> Vec<u128> {
    #[cfg(target_arch = "aarch64")]
    if modulus == ntt::neon::sms::MODULUS {
        use ntt::neon::sms;
        let matrix = sms::SmsMatrix::from_images(&SMS_DATA.0, &SMS_MEANS.0, &SMS_MU.0, &SMS_META.0, RING_ELEMENTS)
            .unwrap_or_else(|| sms::matrix(RING_ELEMENTS, threads()));
        return sms::commit(values, &matrix, threads());
    }
    #[cfg(target_arch = "aarch64")]
    if modulus == ntt::neon::hybrid::MODULUS {
        use ntt::neon::hybrid;
        let matrix = hybrid::HybridMatrix::from_image(&HYBRID_MATRIX.0, RING_ELEMENTS)
            .unwrap_or_else(|| hybrid::matrix(RING_ELEMENTS, threads()));
        return hybrid::commit(values, &matrix, threads());
    }
    if modulus == ntt::MODULUS {
        return ntt::commit(values, &ntt::matrix(RING_ELEMENTS, threads()), threads());
    }
    let matrix = rq::ajtai_matrix(modulus, RING_ELEMENTS);
    if modulus >= FAST_MODULUS {
        return commit_reference(values, &matrix, modulus);
    }
    let matrix: Vec<u64> = matrix.iter().map(|&a| a as u64).collect();
    let q = modulus as u64;
    let per = COLUMNS.div_ceil(threads());
    let mut out = vec![0u128; COLUMNS * rq::N];
    std::thread::scope(|scope| {
        for (block, first) in out.chunks_mut(per * rq::N).zip((0..).step_by(per)) {
            let matrix = &matrix;
            scope.spawn(move || {
                let count = block.len() / rq::N;
                let mut table = vec![0u64; PADDED << GROUP];
                let width = BLOCKS * BLOCK;
                let mut raw = vec![0u64; count * width];
                for (m, element) in matrix.chunks_exact(rq::N).enumerate() {
                    build_table(&mut table, element);
                    for (c, acc) in raw.chunks_exact_mut(width).enumerate() {
                        let base = (first + c) * COLUMN_LEN + 4 * m;
                        let words: [u128; 4] = core::array::from_fn(|k| values[base + k].0);
                        let starts: [usize; GROUPS] = core::array::from_fn(|g| {
                            (pattern(&words, g) * PADDED + BLOCK).wrapping_sub(GROUP * g)
                        });
                        for b in 0..BLOCKS {
                            add_block(acc, &table, &starts, b);
                        }
                    }
                }
                for (target, acc) in block.chunks_exact_mut(rq::N).zip(raw.chunks_exact(width)) {
                    target.copy_from_slice(&reduce(&acc[..RAW], q));
                }
            });
        }
    });
    out
}

/// The checker's schoolbook commitment, for a modulus too wide for [`commit`]'s `u64` lanes.
fn commit_reference(values: &[B128], matrix: &[u128], modulus: u128) -> Vec<u128> {
    let column_len = 4 * (matrix.len() / rq::N);
    let columns = values.len() / column_len;
    let per = columns.div_ceil(threads());
    let mut out = vec![0u128; columns * rq::N];
    std::thread::scope(|scope| {
        for (block, columns) in out
            .chunks_mut(per * rq::N)
            .zip(values.chunks(per * column_len))
        {
            scope.spawn(move || {
                for (target, column) in block
                    .chunks_mut(rq::N)
                    .zip(columns.chunks_exact(column_len))
                {
                    let witness: Vec<F162> = column.iter().map(|&x| F162::from_b128(x)).collect();
                    target.copy_from_slice(&rq::commit_binary_column(matrix, &witness, modulus));
                }
            });
        }
    });
    out
}

fn dot_lifted(eq: &[F162], column: &[B128]) -> F162 {
    cf::dot_lifted(eq, column)
}

/// `u_j = sum_i eq(p0, i) W[i + COLUMN_LEN j]`, one field element per column.
fn row_evaluate(values: &[B128], p0: &[F162]) -> Vec<F162> {
    let eq = eq_table(p0);
    let per = COLUMNS.div_ceil(threads());
    let mut out = vec![F162::ZERO; COLUMNS];
    std::thread::scope(|scope| {
        for (block, columns) in out.chunks_mut(per).zip(values.chunks(per * COLUMN_LEN)) {
            let eq = &eq;
            scope.spawn(move || {
                for (slot, column) in block.iter_mut().zip(columns.chunks_exact(COLUMN_LEN)) {
                    *slot = dot_lifted(eq, column);
                }
            });
        }
    });
    out
}

/// The sumcheck coordinates split into the row point `p0` and the column point `p1`.
fn split_point(coords: &[F162]) -> (Vec<F162>, Vec<F162>) {
    assert_eq!(
        coords.len(),
        LOG_PACKED,
        "one coordinate per packed variable"
    );
    let (high, low) = coords.split_at(COLUMN_LOG as usize);
    (
        low.iter().rev().copied().collect(),
        high.iter().rev().copied().collect(),
    )
}

fn folding_challenges(t: &mut Transcript, row: &RowEvaluation) -> Vec<ShortChallenge> {
    row.absorb(t);
    labinius::challenge::sample_short_challenges(t, COLUMNS, DEFAULT_WEIGHT, DEFAULT_BOUND)
}

/// `v_m = sum_j c_j W_{j,m}` over the integers. A challenge sits in `R_648` as `c(-X^4)`, so each
/// of its terms adds or subtracts the column's binary element shifted by a multiple of four.
fn fold_element(values: &[B128], challenges: &[ShortChallenge], m: usize) -> RingElement648 {
    let mut raw = [0i32; 2 * rq::N];
    let mut bits = [0i32; rq::N];
    for (j, challenge) in challenges.iter().enumerate() {
        let packed = &values[j * COLUMN_LEN + 4 * m..][..4];
        for (k, x) in packed.iter().enumerate() {
            for p in 0..128 {
                bits[4 * p + k] = ((x.0 >> p) & 1) as i32;
            }
        }
        for t in 0..challenge.weight {
            let position = challenge.positions[t] as usize;
            let negative = ((challenge.signs >> t) & 1 == 1) ^ (position % 2 == 1);
            let window = &mut raw[4 * position..][..WITNESS_COEFFICIENTS];
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
    for k in (rq::N..2 * rq::N - 1).rev() {
        let value = raw[k];
        raw[k - rq::SPLIT] += value;
        raw[k - rq::N] -= value;
    }
    let mut element = RingElement648::zero(Representation::Coefficients);
    for (slot, &value) in element.v.iter_mut().zip(&raw[..rq::N]) {
        *slot = i16::try_from(value).expect("a fold coefficient does not fit the wire codec");
    }
    element
}

/// [`fold_element`] in narrow lanes: a raw coefficient gains at most one `+-1` per challenge
/// term, so `sum_j weight_j <= i16::MAX` bounds every `i16` sum, and the latest columns are
/// summed in `i8` lanes, sixteen to a vector, until their terms could reach `i8::MAX`. A column's
/// bits become `0 / -1` byte masks in coefficient order; the ring reduction runs in `i32`.
fn fold_element_narrow(values: &[B128], challenges: &[ShortChallenge], m: usize) -> RingElement648 {
    let mut raw = [0i16; 2 * rq::N];
    let mut near = [0i8; 2 * rq::N];
    let mut pending = 0usize;
    let mut mask = [0i8; WITNESS_COEFFICIENTS];
    for (j, challenge) in challenges.iter().enumerate() {
        if pending + challenge.weight > i8::MAX as usize {
            flush(&mut raw, &mut near);
            pending = 0;
        }
        pending += challenge.weight;
        expand_mask(&values[j * COLUMN_LEN + 4 * m..][..4], &mut mask);
        for t in 0..challenge.weight {
            let position = challenge.positions[t] as usize;
            let negative = ((challenge.signs >> t) & 1 == 1) ^ (position % 2 == 1);
            let window = &mut near[4 * position..][..WITNESS_COEFFICIENTS];
            if negative {
                for (slot, &b) in window.iter_mut().zip(&mask) {
                    *slot = slot.wrapping_add(b);
                }
            } else {
                for (slot, &b) in window.iter_mut().zip(&mask) {
                    *slot = slot.wrapping_sub(b);
                }
            }
        }
    }
    flush(&mut raw, &mut near);
    let mut wide = [0i32; 2 * rq::N];
    for (w, &r) in wide.iter_mut().zip(&raw) {
        *w = r as i32;
    }
    for k in (rq::N..2 * rq::N - 1).rev() {
        let value = wide[k];
        wide[k - rq::SPLIT] += value;
        wide[k - rq::N] -= value;
    }
    let mut element = RingElement648::zero(Representation::Coefficients);
    for (slot, &value) in element.v.iter_mut().zip(&wide[..rq::N]) {
        *slot = i16::try_from(value).expect("a fold coefficient does not fit the wire codec");
    }
    element
}

/// `raw += near`, then `near = 0`.
#[inline(always)]
fn flush(raw: &mut [i16; 2 * rq::N], near: &mut [i8; 2 * rq::N]) {
    for (r, n) in raw.iter_mut().zip(near.iter_mut()) {
        *r = r.wrapping_add(*n as i16);
        *n = 0;
    }
}

/// `mask[4p + k] = -1` where bit `p` of `words[k]` is set, else `0`: witness coefficient order.
#[inline(always)]
fn expand_mask(words: &[B128], mask: &mut [i8; WITNESS_COEFFICIENTS]) {
    assert_eq!(words.len(), 4);
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use core::arch::aarch64::*;
        let bits: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
        let bits = vld1q_u8(bits.as_ptr());
        let w: [uint8x16_t; 4] = core::array::from_fn(|k| vld1q_u8(words.as_ptr().add(k) as *const u8));
        // Chunk c holds bits 16c..16c+16 of each word: bytes 2c and 2c+1, each spread over
        // eight lanes and tested bit by bit, then interleaved word by word.
        for c in 0..8u8 {
            let idx = vcombine_u8(vdup_n_u8(2 * c), vdup_n_u8(2 * c + 1));
            let v = uint8x16x4_t(
                vtstq_u8(vqtbl1q_u8(w[0], idx), bits),
                vtstq_u8(vqtbl1q_u8(w[1], idx), bits),
                vtstq_u8(vqtbl1q_u8(w[2], idx), bits),
                vtstq_u8(vqtbl1q_u8(w[3], idx), bits),
            );
            vst4q_u8(mask.as_mut_ptr().add(64 * c as usize) as *mut u8, v);
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    for (k, w) in words.iter().enumerate() {
        for p in 0..128 {
            mask[4 * p + k] = -(((w.0 >> p) & 1) as i8);
        }
    }
}

fn fold(values: &[B128], challenges: &[ShortChallenge]) -> Vec<RingElement648> {
    // The NEON bit-sliced fold's 16-plane counters hold sums below 2^15.
    #[cfg(target_arch = "aarch64")]
    if challenges.iter().map(|c| c.weight).sum::<usize>() < 1 << 15 {
        return fold_neon(values, challenges);
    }
    // The scheduled bit-sliced fold's 15-bit counters hold the total weight plus its padding.
    if let Some(out) = fold_scheduled(values, challenges) {
        return out;
    }
    // The bit-sliced fold's 16-plane counters hold sums below 2^15.
    if challenges.iter().map(|c| c.weight).sum::<usize>() < 1 << 15 {
        return fold_bitsliced(values, challenges);
    }
    fold_bytes(values, challenges)
}

/// The NEON bit-sliced fold (`fold_bits::neon`): 128-lane blocks, negacyclic slot pairs,
/// plain-op full adders, the elements written in place.
#[cfg(target_arch = "aarch64")]
fn fold_neon(values: &[B128], challenges: &[ShortChallenge]) -> Vec<RingElement648> {
    use labinius::fold_bits as fb;
    const _: () = assert!(COLUMN_LEN % fb::neon::LANES == 0 && fb::N == rq::N);
    let cs: Vec<fb::Challenge> = challenges
        .iter()
        .map(|c| fb::Challenge {
            positions: c.positions,
            signs: c.signs,
            weight: c.weight,
        })
        .collect();
    // SAFETY: `B128` is `repr(transparent)` over `u128`.
    let words = unsafe { core::slice::from_raw_parts(values.as_ptr() as *const u128, values.len()) };
    fb::neon::fold(words, COLUMN_LEN, &cs, threads())
}

/// The x86 track's scheduled bit-sliced fold (`fold_bits2`) on NEON, written straight into the
/// ring elements; `None` when its counters could overflow.
fn fold_scheduled(values: &[B128], challenges: &[ShortChallenge]) -> Option<Vec<RingElement648>> {
    use labinius::fold_bits as fb;
    use labinius::fold_bits2 as f2;
    let cs: Vec<fb::Challenge> = challenges
        .iter()
        .map(|c| fb::Challenge {
            positions: c.positions,
            signs: c.signs,
            weight: c.weight,
        })
        .collect();
    if !f2::fits(&cs) || !f2::shape_ok(COLUMN_LEN) || f2::N != rq::N {
        return None;
    }
    assert_eq!(values.len(), COLUMNS * COLUMN_LEN);
    let mut out: Vec<RingElement648> = Vec::with_capacity(RING_ELEMENTS);
    let base = out.as_mut_ptr();
    // SAFETY: the kernel writes every coefficient of the `RING_ELEMENTS` elements, at
    // `v + e * stride` (the elements' `v` fields are `stride` `i16` apart), and the
    // representation is written here; `values` holds `COLUMNS` columns of `COLUMN_LEN` words.
    unsafe {
        for e in 0..RING_ELEMENTS {
            core::ptr::addr_of_mut!((*base.add(e)).representation).write(Representation::Coefficients);
        }
        const _: () = assert!(core::mem::size_of::<RingElement648>() % 2 == 0);
        let stride = core::mem::size_of::<RingElement648>() / 2;
        f2::fold_into(
            values.as_ptr() as *const u128,
            COLUMNS,
            COLUMN_LEN,
            &cs,
            threads(),
            core::ptr::addr_of_mut!((*base).v) as *mut i16,
            stride,
        );
        out.set_len(RING_ELEMENTS);
    }
    Some(out)
}

/// The fold of the x86 track's bit-sliced kernel (`fold_bits`), on NEON: blocks of 128 ring
/// elements, each column's 512 words transposed into bit planes and counted per raw row with
/// carry-save adders; blocks are handed out to the workers dynamically.
fn fold_bitsliced(values: &[B128], challenges: &[ShortChallenge]) -> Vec<RingElement648> {
    use labinius::fold_bits as fb;
    use std::sync::atomic::{AtomicUsize, Ordering};
    const _: () = assert!(COLUMN_LEN % fb::LANES == 0 && fb::N == rq::N);
    let cs: Vec<fb::Challenge> = challenges
        .iter()
        .map(|c| fb::Challenge {
            positions: c.positions,
            signs: c.signs,
            weight: c.weight,
        })
        .collect();
    let reach = fb::Reach::of(&cs);
    let blocks = RING_ELEMENTS / fb::BLOCK_ELEMENTS;
    let mut out: Vec<[i16; rq::N]> = vec![[0i16; rq::N]; RING_ELEMENTS];
    let out_addr = out.as_mut_ptr() as usize;
    let words = values.as_ptr() as usize;
    let next = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for _ in 0..threads().min(blocks) {
            let (cs, reach, next) = (&cs, &reach, &next);
            scope.spawn(move || loop {
                let block = next.fetch_add(1, Ordering::Relaxed);
                if block >= blocks {
                    break;
                }
                // SAFETY: blocks are disjoint runs of `BLOCK_ELEMENTS` elements of `out`, and
                // `values` holds `COLUMNS` columns of `COLUMN_LEN` words.
                unsafe {
                    let chunk = core::slice::from_raw_parts_mut(
                        (out_addr as *mut [i16; rq::N]).add(block * fb::BLOCK_ELEMENTS),
                        fb::BLOCK_ELEMENTS,
                    );
                    fb::fold_block::<fb::V, fb::Portable>(
                        words as *const u128,
                        COLUMNS,
                        COLUMN_LEN,
                        cs,
                        reach,
                        block,
                        &fb::Portable,
                        chunk,
                    );
                }
            });
        }
    });
    out.into_iter()
        .map(|v| {
            let mut e = RingElement648::zero(Representation::Coefficients);
            e.v.copy_from_slice(&v);
            e
        })
        .collect()
}

fn fold_bytes(values: &[B128], challenges: &[ShortChallenge]) -> Vec<RingElement648> {
    let narrow = challenges.iter().map(|c| c.weight).sum::<usize>() <= i16::MAX as usize;
    let per = RING_ELEMENTS.div_ceil(threads());
    let mut out: Vec<RingElement648> = (0..RING_ELEMENTS)
        .map(|_| RingElement648::zero(Representation::Coefficients))
        .collect();
    std::thread::scope(|scope| {
        for (block, base) in out.chunks_mut(per).zip((0..).step_by(per)) {
            scope.spawn(move || {
                for (i, slot) in block.iter_mut().enumerate() {
                    *slot = if narrow {
                        fold_element_narrow(values, challenges, base + i)
                    } else {
                        fold_element(values, challenges, base + i)
                    };
                }
            });
        }
    });
    out
}

fn pack_folded(folded: &FoldedWitness) -> Vec<u8> {
    let elements = folded.elements();
    let mut out = vec![0u8; 4 * rq::N * elements.len()];
    for (chunk, e) in out.chunks_exact_mut(4 * rq::N).zip(elements) {
        for (slot, &c) in chunk.chunks_exact_mut(4).zip(&e.v) {
            slot.copy_from_slice(&(c as i32).to_le_bytes());
        }
    }
    out
}

impl Proof {
    /// The proof's bincode encoding under [`codec`]'s options (fixed-width little-endian
    /// integers, `u64` sequence lengths, arrays without one), written field by field with
    /// whole-slice copies; serde would hand bincode the byte vectors one `u8` at a time.
    fn encode(&self) -> Vec<u8> {
        let size = 4
            + 16
            + (8 + self.commitment.len())
            + (8 + 16 * self.partials.len())
            + (8 + 48 * self.rounds.len())
            + 24
            + (8 + self.row.len())
            + (8 + self.folded.len());
        assert!(size as u64 <= LIMIT, "serialize proof: the proof exceeds the size limit");
        let mut out = Vec::with_capacity(size);
        let len = |out: &mut Vec<u8>, n: usize| out.extend_from_slice(&(n as u64).to_le_bytes());
        out.extend_from_slice(&self.version.to_le_bytes());
        out.extend_from_slice(&self.modulus.to_le_bytes());
        len(&mut out, self.commitment.len());
        out.extend_from_slice(&self.commitment);
        len(&mut out, self.partials.len());
        for p in &self.partials {
            out.extend_from_slice(&p.to_le_bytes());
        }
        len(&mut out, self.rounds.len());
        for x in self.rounds.iter().flatten().flatten() {
            out.extend_from_slice(&x.to_le_bytes());
        }
        for x in &self.opened {
            out.extend_from_slice(&x.to_le_bytes());
        }
        len(&mut out, self.row.len());
        out.extend_from_slice(&self.row);
        len(&mut out, self.folded.len());
        out.extend_from_slice(&self.folded);
        assert_eq!(out.len(), size);
        out
    }
}

fn switch_proof(values: &[B128], lo: &[B128], hi: &[B128], t: &mut Transcript) -> (Vec<B128>, Vec<[[u64; 3]; 2]>, Vec<F162>, F162) {
    #[cfg(target_arch = "aarch64")]
    if let Some(prepared) = labinius::fields::lean::Prepared::new(values, hi) {
        let partials = prepared.partials();
        let value = partials.iter().zip(cf::eq_expand_b128(lo))
            .fold(B128::ZERO, |a, (&v, e)| a + v * e);
        t.absorb_b128(&[value]);
        t.absorb_b128(&partials);
        let batch = cf::eq_expand_f162(&t.sample_f162(b"switch-batch", 7));
        let mut switch = labinius::fields::lean::Switch::new(values, prepared, &batch);
        let mut rounds = Vec::with_capacity(LOG_PACKED);
        let mut coords = Vec::with_capacity(LOG_PACKED);
        for _ in 0..LOG_PACKED {
            let msg = switch.msg();
            t.absorb_f162(&msg);
            let r = t.sample_f162(b"switch-round", 1)[0];
            rounds.push(msg.map(|v| v.0));
            coords.push(r);
            switch.fold(r);
        }
        return (partials, rounds, coords, switch.final_eval());
    }
    let mut ws = cf::Workspace::new(LOG_PACKED, 1);
    let partials = ws.partial_evals(0, values, hi);
    let value = partials
        .iter()
        .zip(cf::eq_expand_b128(lo))
        .fold(B128::ZERO, |a, (&v, e)| a + v * e);
    t.absorb_b128(&[value]);
    t.absorb_b128(&partials);
    let batch = cf::eq_expand_f162(&t.sample_f162(b"switch-batch", 7));
    let mut switch = cf::SwitchProver::from_workspace(ws, values, &[F162::ONE], &batch);
    let mut rounds = Vec::with_capacity(LOG_PACKED);
    let mut coords = Vec::with_capacity(LOG_PACKED);
    for _ in 0..LOG_PACKED {
        let msg = switch.msg();
        t.absorb_f162(&msg);
        let r = t.sample_f162(b"switch-round", 1)[0];
        rounds.push(msg.map(|v| v.0));
        coords.push(r);
        switch.fold(r);
    }
    let opened = switch.final_eval();
    (partials, rounds, coords, opened)
}

fn prove(seed: &[u8; 32], modulus: u128, values: &[B128], timer: &mut Timer) -> Proof {
    let commitment = rq::pack_ring_elements(&commit(values, modulus), modulus);
    timer.lap("commit");
    let mut t = transcript(seed, modulus, &commitment);
    let lo = t.sample_b128(b"r-lo", 7);
    let hi = t.sample_b128(b"r-hi", LOG_PACKED);
    let (partials, rounds, coords, opened) = switch_proof(values, &lo, &hi, &mut t);
    t.absorb_f162(&[opened]);
    timer.lap("switch");
    let (p0, _) = split_point(&coords);
    let row = RowEvaluation::of(row_evaluate(values, &p0));
    timer.lap("row");
    let challenges = folding_challenges(&mut t, &row);
    let folded = FoldedWitness::of(fold(values, &challenges));
    timer.lap("fold");
    Proof {
        version: 1,
        modulus,
        commitment,
        partials: partials.iter().map(|v| v.0).collect(),
        rounds,
        opened: opened.0,
        row: wire::pack_row_evaluation(&row),
        folded: pack_folded(&folded),
    }
}

/// Phase times on stderr, diagnostic only.
struct Timer(std::time::Instant, Vec<String>);

impl Timer {
    fn lap(&mut self, phase: &str) {
        self.1
            .push(format!("{phase}_ms={}", self.0.elapsed().as_millis()));
        self.0 = std::time::Instant::now();
    }
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert!(
        args.len() == 5 || args.len() == 7,
        "usage: labinius-fast --prove FILE --table FILE [--modulus Q]"
    );
    assert_eq!(args[1], "--prove", "unknown command");
    assert_eq!(args[3], "--table", "witness table path is required");
    let seed = seed();
    let modulus = if args.len() == 7 {
        assert_eq!(args[5], "--modulus", "unknown prover option");
        parse_modulus(&args[6]).expect("invalid commitment modulus")
    } else {
        MODULUS
    };
    #[cfg(target_arch = "aarch64")] if modulus == ntt::neon::sms::MODULUS { ntt::neon::sms::prefetch_key(&SMS_DATA.0) }
    let mut timer = Timer(std::time::Instant::now(), Vec::new());
    let values = read_table(&args[4]);
    timer.lap("read");
    let proof = prove(&seed, modulus, &values, &mut timer);
    let bytes = proof.encode();
    std::fs::write(&args[2], bytes).expect("write proof");
    timer.lap("write");
    eprintln!("labinius-arm {}", timer.1.join(" "));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_load_matches_little_endian_words() {
        let path = std::env::temp_dir().join(format!("labinius-tail-table-{}", std::process::id()));
        for n in [0, 1, 17, 4097] {
            let words: Vec<_> = (0..n)
                .map(|i| B128((i as u128 * 0x0102030405060708) ^ u128::MAX))
                .collect();
            let bytes: Vec<_> = words.iter().flat_map(|w| w.0.to_le_bytes()).collect();
            std::fs::write(&path, &bytes).unwrap();
            let table = load_table(path.to_str().unwrap(), n);
            assert_eq!(&*table, &words);
            assert_eq!(std::fs::read(&path).unwrap(), bytes);
        }
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    #[should_panic(expected = "witness table has the wrong length")]
    fn table_load_rejects_wrong_length() {
        let path = std::env::temp_dir().join(format!("labinius-tail-short-{}", std::process::id()));
        std::fs::write(&path, [0u8; 15]).unwrap();
        // Remove the temporary file before resuming the expected assertion.
        let result = std::panic::catch_unwind(|| load_table(path.to_str().unwrap(), 1));
        std::fs::remove_file(path).unwrap();
        match result {
            Ok(_) => panic!("wrong length was accepted"),
            Err(e) => std::panic::resume_unwind(e),
        }
    }

    fn column(seed: u64) -> Vec<B128> {
        let mut state = seed;
        (0..COLUMN_LEN)
            .map(|_| {
                let mut word = 0u128;
                for _ in 0..2 {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    word = word << 64 | state as u128;
                }
                B128(word)
            })
            .collect()
    }

    /// The direct encoding is byte for byte bincode's under the proof codec.
    /// The SMS commitment agrees with the schoolbook commitment at full size on random columns
    /// (its digit recombination is statistical: random witnesses), one and three workers.
    #[test]
    fn sms_commit_matches_reference() {
        // the embedded images are the runtime matrix, byte for byte
        {
            use ntt::neon::sms;
            let built = sms::matrix(RING_ELEMENTS, 8);
            let embedded = sms::SmsMatrix::from_images(&SMS_DATA.0, &SMS_MEANS.0, &SMS_MU.0, &SMS_META.0, RING_ELEMENTS)
                .expect("the build embeds the SMS images");
            assert!(built.data == embedded.data && built.mean_residues == embedded.mean_residues && built.mu == embedded.mu);
            assert_eq!(built.margin, embedded.margin);
        }
        use ntt::neon::sms;
        let q = sms::MODULUS;
        let values: Vec<B128> = (0..9).flat_map(|s| column(100 + s)).collect();
        let matrix = rq::ajtai_matrix(q, RING_ELEMENTS);
        let want = commit_reference(&values, &matrix, q);
        let a = sms::matrix(RING_ELEMENTS, 8);
        eprintln!("sms margin {}", a.margin);
        for workers in [1, 3] {
            assert!(sms::commit(&values, &a, workers) == want, "workers {workers}");
        }
        // a short last chunk and a short last lockstep group, partial batches
        let elements = 301;
        let values: Vec<B128> = (0..11).flat_map(|s| column(200 + s)).take(11 * 4 * elements).collect();
        let matrix = rq::ajtai_matrix(q, elements);
        let want = commit_reference(&values, &matrix, q);
        let a = sms::matrix(elements, 3);
        assert!(sms::commit(&values, &a, 3) == want, "{elements} elements");
    }
    #[test]
    fn encoding_matches_bincode() {
        use bincode::Options;
        let mut x = 0x0123_4567_89ab_cdefu64;
        let mut next = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        for n in [0usize, 1, 3, 300] {
            let proof = Proof {
                version: next() as u32,
                modulus: (next() as u128) << 64 | next() as u128,
                commitment: (0..n).map(|_| next() as u8).collect(),
                partials: (0..n).map(|_| (next() as u128) << 64 | next() as u128).collect(),
                rounds: (0..n)
                    .map(|_| [[next(), next(), next()], [next(), next(), next()]])
                    .collect(),
                opened: [next(), next(), next()],
                row: (0..2 * n).map(|_| next() as u8).collect(),
                folded: (0..3 * n + 1).map(|_| next() as u8).collect(),
            };
            assert_eq!(proof.encode(), codec().serialize(&proof).unwrap(), "n {n}");
        }
    }

    /// The narrow fold agrees with the `i32` fold, on dense random columns and on all-ones
    /// columns (every coefficient takes every term).
    #[test]
    fn narrow_fold_matches_wide() {
        let mut t = Transcript::new(b"labinius-fast/test");
        let challenges: Vec<ShortChallenge> = (0..COLUMNS)
            .map(|_| sample_short_challenge(&mut t, DEFAULT_WEIGHT, DEFAULT_BOUND).0)
            .collect();
        let mut values = vec![B128::ZERO; COLUMNS * COLUMN_LEN];
        let elements = [0usize, 1, 17, 2048, RING_ELEMENTS - 1];
        for (e, &m) in elements.iter().enumerate() {
            for j in 0..COLUMNS {
                let col = column((e * COLUMNS + j) as u64 + 11);
                for k in 0..4 {
                    values[j * COLUMN_LEN + 4 * m + k] =
                        if m == 2048 { B128(u128::MAX) } else { col[k] };
                }
            }
        }
        for m in elements {
            assert_eq!(
                fold_element_narrow(&values, &challenges, m).v,
                fold_element(&values, &challenges, m).v,
                "element {m}"
            );
        }
    }

    /// The table lookup agrees with the checker's coefficient-by-coefficient commitment, at the
    /// earlier default modulus, which the four-Russians path still serves.
    #[test]
    fn grouped_commit_matches_reference() {
        let modulus = 40_772_829_184;
        assert!(modulus < FAST_MODULUS && modulus != ntt::MODULUS);
        let values: Vec<B128> = (0..COLUMNS as u64)
            .flat_map(|j| {
                if j < 2 {
                    column(j + 1)
                } else {
                    vec![B128::ZERO; COLUMN_LEN]
                }
            })
            .collect();
        let fast = commit(&values, modulus);
        let matrix = rq::ajtai_matrix(modulus, RING_ELEMENTS);
        for j in 0..2 {
            let witness: Vec<F162> = values[j * COLUMN_LEN..][..COLUMN_LEN]
                .iter()
                .map(|&x| F162::from_b128(x))
                .collect();
            assert_eq!(
                fast[j * rq::N..][..rq::N],
                rq::commit_binary_column(&matrix, &witness, modulus)[..],
                "column {j}"
            );
        }
        assert!(fast[2 * rq::N..].iter().all(|&x| x == 0));
    }

    /// Columns of `elements` ring elements for the transform tests: dense random columns and the
    /// edge cases (all zero, all ones, single bits at coefficients 0 and 511 of the first and the
    /// last element), `count` columns in all, so the last batch of eight is partial.
    fn edge_columns(elements: usize, count: usize) -> Vec<B128> {
        let len = 4 * elements;
        let mut out = Vec::with_capacity(count * len);
        for j in 0..count {
            let mut col = vec![B128::ZERO; len];
            match j % 8 {
                1 => col.fill(B128(u128::MAX)),
                2 => {}
                // coefficient 0 of the first element: bit 0 of word 0
                3 => col[0] = B128(1),
                // coefficient 511 = 4 * 127 + 3 of the last element: bit 127 of word 3
                4 => col[len - 1] = B128(1 << 127),
                5 => {
                    col[3] = B128(1 << 127);
                    col[len - 4] = B128(1);
                }
                _ => {
                    let dense = column(j as u64 * 7 + 3);
                    for (c, &d) in col.iter_mut().zip(dense.iter().cycle()) {
                        *c = d;
                    }
                }
            }
            out.extend(col);
        }
        out
    }

    /// The transform commitment agrees with the checker's schoolbook commitment at full size, on
    /// one, three and eight workers, through the matrix transformed on as many.
    #[test]
    fn ntt_commit_matches_reference() {
        let values = edge_columns(RING_ELEMENTS, 11);
        let matrix = rq::ajtai_matrix(ntt::MODULUS, RING_ELEMENTS);
        let want = commit_reference(&values, &matrix, ntt::MODULUS);
        assert!(want.iter().all(|&x| x < ntt::MODULUS));
        assert!(want[2 * rq::N..3 * rq::N].iter().all(|&x| x == 0));
        for workers in [1, 3, 8] {
            let transformed = ntt::transform_matrix(&matrix, workers);
            assert_eq!(ntt::commit(&values, &transformed, workers), want, "workers {workers}");
            if workers == 3 {
                assert_eq!(ntt::scalar_commit(&values, &transformed, workers), want);
            }
        }
    }

    /// The matrix built element by element and transformed eight at a time is the scalar
    /// transform of `rq::ajtai_matrix`, for element counts around the vector width and on one,
    /// three and eight workers.
    #[test]
    fn fused_matrix_matches_scalar_transform() {
        for elements in [1usize, 7, 8, 9, 23, 64] {
            let want = ntt::transform_matrix(&rq::ajtai_matrix(ntt::MODULUS, elements), 3);
            for workers in [1, 3, 8] {
                let got = ntt::matrix(elements, workers);
                assert!(got.data == want.data, "elements {elements} workers {workers}");
            }
        }
        let full = ntt::matrix(RING_ELEMENTS, 8);
        let want = ntt::transform_matrix(&rq::ajtai_matrix(ntt::MODULUS, RING_ELEMENTS), 8);
        assert!(full.data == want.data);
    }

    /// The same on short columns, many cases, column counts around the batch width.
    #[test]
    fn ntt_commit_matches_reference_short_columns() {
        for (elements, count) in [(1usize, 1usize), (3, 8), (5, 9), (7, 17), (2, 23)] {
            let values = edge_columns(elements, count);
            let matrix = rq::ajtai_matrix(ntt::MODULUS, elements);
            let want = commit_reference(&values, &matrix, ntt::MODULUS);
            for workers in [1, 3, 8] {
                let transformed = ntt::transform_matrix(&matrix, workers);
                assert_eq!(
                    ntt::commit(&values, &transformed, workers),
                    want,
                    "elements {elements} columns {count} workers {workers}"
                );
            }
        }
    }
}
