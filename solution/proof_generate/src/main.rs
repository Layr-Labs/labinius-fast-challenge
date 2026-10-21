//! Packed-binary PCS opening, including the GHASH-to-F162 field switch.
//!
//! The default commitment modulus is 3889 * 9721 * 1024. Two witness NTTs feed
//! ordinary residues and an exact auxiliary computation of the 1024 residue.
//! Public matrix images and reconstruction bounds are computed at build time
//! from the fixed matrix seed and modulus. Other supported prime products use
//! the generic RNS implementation.
//!
//! The timed process reads the fresh table in place, commits without retaining
//! witness transforms, runs the eighth-table field switch, computes the exact
//! integer fold, and writes the checker-compatible proof. Transcript challenges
//! are derived using the chosen modulus and the resulting commitment.
use bincode::Options;
use labinius::challenge::{
    sample_short_challenges_parallel, ShortChallenge, DEFAULT_BOUND, DEFAULT_WEIGHT,
};
use labinius::fields::{
    crossfield as cf,
    scalar::{B128, F162},
};
#[cfg(test)]
use labinius::key::CommitmentKey;
use labinius::params::N;
use labinius::ring::Representation;
use labinius::scheme::*;
use labinius::{wire, Transcript};
use serde::{Deserialize, Serialize};

mod keygen;
mod modulus_choice;
mod parameters;
/// Byte-for-byte the checker's `rq.rs`, so that the two sides of the format can be diffed. A
/// release build of the prover only uses the matrix expansion and the packer; the rest is what
/// the tests check the proof against.
#[allow(dead_code)]
mod rq;
#[cfg(test)]
use parameters::RING_ELEMENTS;
use parameters::{COLUMNS, COLUMN_LEN, COLUMN_LOG, LOG_PACKED, LOG_VARS, MODULUS};

/// The commitment key of the default modulus in the column-pair layout
/// (`hybrid::pair_keys`), precomputed by `build.rs` and embedded as little-endian `u32` words
/// (empty when the build host could not run the kernels). 64-byte aligned so it is read in place.
#[repr(C, align(64))]
struct Aligned64<T>(T);
const EMBEDDED_KEY_LEN: usize = include_bytes!(concat!(env!("OUT_DIR"), "/key.bin")).len();
static EMBEDDED_KEY: Aligned64<[u8; EMBEDDED_KEY_LEN]> =
    Aligned64(*include_bytes!(concat!(env!("OUT_DIR"), "/key.bin")));

const LIMIT: u64 = 64 * 1024 * 1024;

#[derive(Serialize, Deserialize)]
struct Proof {
    version: u32,
    modulus: u128,
    #[serde(serialize_with = "serialize_blob")]
    commitment: Vec<u8>,
    partials: Vec<u128>,
    rounds: Vec<[[u64; 3]; 2]>,
    opened: [u64; 3],
    #[serde(serialize_with = "serialize_blob")]
    row: Vec<u8>,
    #[serde(serialize_with = "serialize_blob")]
    folded: Vec<u8>,
}
/// A `Vec<u8>` as one byte string: under this codec's fixed-width encoding that is the same
/// `u64` length and the same bytes a sequence of `u8` gives, written as one block instead of
/// one byte at a time. The checker deserialises it as the `Vec<u8>` it always was.
fn serialize_blob<S: serde::Serializer>(bytes: &[u8], serializer: S) -> Result<S::Ok, S::Error> {
    serializer.serialize_bytes(bytes)
}
fn codec() -> impl Options {
    bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .with_limit(LIMIT)
        .reject_trailing_bytes()
}
use keygen::params;
fn seed() -> [u8; 32] {
    let s = std::env::var("LABINIUS_SEED").expect("LABINIUS_SEED is required");
    assert_eq!(s.len(), 64);
    core::array::from_fn(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).expect("invalid seed"))
}

/// Threads the parallel phases use: what the sandbox lets the process see.
fn threads() -> usize {
    labinius::par::threads()
}

// =============================================================================================
// the witness table, read in place
// =============================================================================================

#[cfg(target_os = "linux")]
mod sys {
    use core::ffi::c_void;
    extern "C" {
        pub fn mmap(
            addr: *mut c_void,
            len: usize,
            prot: i32,
            flags: i32,
            fd: i32,
            offset: i64,
        ) -> *mut c_void;
        pub fn munmap(addr: *mut c_void, len: usize) -> i32;
        pub fn madvise(addr: *mut c_void, len: usize, advice: i32) -> i32;
    }
    pub const PROT_READ: i32 = 1;
    pub const MAP_PRIVATE: i32 = 2;
    /// Linux 5.14+: fault the range in now, read-only, without blocking a later writer.
    pub const MADV_POPULATE_READ: i32 = 22;
}

/// The harness witness table: `(1 << LOG_VARS) / 8` BLAKE3 XOF bytes, each 16 little-endian
/// bytes one GHASH word. The timed prover does not recompute it, and does not copy it either: on
/// Linux the file is mapped read-only and the mapping is read as the `B128` words it is (`B128`
/// is `repr(transparent)` over its little-endian `u128`, and the host is little-endian). The
/// pages are faulted in by [`Table::populate`] on a thread of its own while the key is set up.
/// Elsewhere, or if the mapping fails, the file is read into a buffer.
enum Table {
    #[cfg(target_os = "linux")]
    Mapped {
        ptr: *const B128,
        bytes: usize,
    },
    Owned(Vec<B128>),
}

impl Table {
    const WORDS: usize = 1 << LOG_PACKED;
    const BYTES: usize = (1 << LOG_VARS) / 8;

    fn load(path: &str) -> Table {
        assert!(cfg!(target_endian = "little"), "the table is little-endian");
        assert_eq!(core::mem::size_of::<B128>(), 16, "B128 is 16 bytes");
        let file = std::fs::File::open(path).expect("open witness table");
        let len = file.metadata().expect("stat witness table").len();
        assert_eq!(len, Self::BYTES as u64, "witness table has the wrong length");
        #[cfg(target_os = "linux")]
        {
            use std::os::fd::AsRawFd;
            let ptr = unsafe {
                sys::mmap(
                    core::ptr::null_mut(),
                    Self::BYTES,
                    sys::PROT_READ,
                    sys::MAP_PRIVATE,
                    file.as_raw_fd(),
                    0,
                )
            };
            if ptr as isize != -1 && !ptr.is_null() && (ptr as usize) % 16 == 0 {
                return Table::Mapped {
                    ptr: ptr as *const B128,
                    bytes: Self::BYTES,
                };
            }
        }
        let mut words = vec![B128::ZERO; Self::WORDS];
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(words.as_mut_ptr() as *mut u8, Self::BYTES)
        };
        let mut file = file;
        std::io::Read::read_exact(&mut file, bytes).expect("witness table has the wrong length");
        let mut extra = [0u8; 1];
        let extra_n = std::io::Read::read(&mut file, &mut extra).expect("read witness table");
        assert_eq!(extra_n, 0, "witness table has the wrong length");
        Table::Owned(words)
    }

    fn words(&self) -> &[B128] {
        match self {
            #[cfg(target_os = "linux")]
            Table::Mapped { ptr, .. } => unsafe { std::slice::from_raw_parts(*ptr, Self::WORDS) },
            Table::Owned(v) => v,
        }
    }

}

/// Fault the words' pages in, page-cache pages into page tables: `MADV_POPULATE_READ` where the
/// kernel has it (older kernels refuse it, harmlessly), one read per page otherwise (the
/// fault-around maps a run of pages per fault). A read-only touch, so it runs on another thread
/// while the key is built.
fn populate_words(words: &[B128], threads: usize) {
    let runs = labinius::par::ranges(words.len(), threads, 4096 / 16);
    labinius::par::pool().run(runs.len(), |i| populate_part(&words[runs[i].clone()]));
}

fn populate_part(words: &[B128]) {
    let (addr, bytes) = (words.as_ptr() as usize, words.len() * 16);
    #[cfg(target_os = "linux")]
    {
        let done =
            unsafe { sys::madvise(addr as *mut core::ffi::c_void, bytes, sys::MADV_POPULATE_READ) };
        if done == 0 {
            return;
        }
    }
    let mut off = 0;
    let mut sink = 0u8;
    while off < bytes {
        sink ^= unsafe { core::ptr::read_volatile((addr as *const u8).add(off)) };
        off += 4096;
    }
    std::hint::black_box(sink);
}

impl Drop for Table {
    fn drop(&mut self) {
        #[cfg(target_os = "linux")]
        if let Table::Mapped { ptr, bytes } = self {
            unsafe { sys::munmap(*ptr as *mut core::ffi::c_void, *bytes) };
        }
    }
}

#[cfg(test)]
fn trace(seed: &[u8; 32]) -> Vec<B128> {
    let mut h = blake3::Hasher::new_keyed(seed);
    h.update(b"labinius-fast/binary-table/v1");
    let mut b = vec![0; (1 << LOG_VARS) / 8];
    h.finalize_xof().fill(&mut b);
    b.chunks_exact(16)
        .map(|v| B128(u128::from_le_bytes(v.try_into().unwrap())))
        .collect()
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
fn point(t: &mut Transcript) -> (Vec<B128>, Vec<B128>) {
    (
        t.sample_b128(b"r-lo", 7),
        t.sample_b128(b"r-hi", LOG_PACKED),
    )
}
fn claim(partials: &[B128], lo: &[B128]) -> B128 {
    partials
        .iter()
        .zip(cf::eq_expand_b128(lo))
        .fold(B128::ZERO, |a, (&v, e)| a + v * e)
}
/// The checker's folded-witness encoding, which this must reproduce byte for byte: one
/// little-endian `i32` per coefficient, element by element. Written into a sized buffer, in
/// parallel over elements.
const FOLDED_BYTES: usize = 4;
fn pack_folded(folded: &FoldedWitness) -> Vec<u8> {
    let elements = folded.elements();
    assert!(
        elements
            .iter()
            .all(|e| e.representation == Representation::Coefficients),
        "a fold is in coefficient form"
    );
    let mut out = vec![0u8; FOLDED_BYTES * N * elements.len()];
    let runs = labinius::par::ranges(elements.len(), threads(), 1);
    let out_addr = out.as_mut_ptr() as usize;
    labinius::par::pool().run(runs.len(), |i| {
        let run = runs[i].clone();
        // Run `i` owns the bytes of elements `run`.
        let mine = unsafe {
            std::slice::from_raw_parts_mut(
                (out_addr as *mut u8).add(FOLDED_BYTES * N * run.start),
                FOLDED_BYTES * N * run.len(),
            )
        };
        for (slot, &c) in mine
            .chunks_exact_mut(FOLDED_BYTES)
            .zip(elements[run].iter().flat_map(|e| e.v.iter()))
        {
            slot.copy_from_slice(&(c as i32).to_le_bytes());
        }
    });
    out
}

/// `rq::pack_ring_elements`, byte for byte, into a sized buffer in parallel: `coefficient_bytes`
/// little-endian bytes per coefficient, no header.
fn pack_coefficients(coefficients: &[u128], modulus: u128) -> Vec<u8> {
    let width = rq::coefficient_bytes(modulus);
    let mut out = vec![0u8; width * coefficients.len()];
    let runs = labinius::par::ranges(coefficients.len(), threads(), 1);
    let out_addr = out.as_mut_ptr() as usize;
    labinius::par::pool().run(runs.len(), |i| {
        let run = runs[i].clone();
        // Run `i` owns the bytes of coefficients `run`.
        let mine = unsafe {
            std::slice::from_raw_parts_mut((out_addr as *mut u8).add(width * run.start), width * run.len())
        };
        for (slot, &x) in mine.chunks_exact_mut(width).zip(&coefficients[run]) {
            debug_assert!(x < modulus, "a coefficient is not reduced");
            slot.copy_from_slice(&x.to_le_bytes()[..width]);
        }
    });
    out
}
#[cfg(test)]
fn unpack_folded(bytes: &[u8], elements: usize) -> Result<FoldedWitness, String> {
    if bytes.len() != FOLDED_BYTES * N * elements {
        return Err("folded witness has the wrong length".into());
    }
    let mut words = bytes.chunks_exact(FOLDED_BYTES);
    let mut fold = Vec::with_capacity(elements);
    for _ in 0..elements {
        let mut e = labinius::RingElement648::zero(Representation::Coefficients);
        for slot in e.v.iter_mut() {
            let word = i32::from_le_bytes(words.next().unwrap().try_into().unwrap());
            *slot = i16::try_from(word).map_err(|_| "coefficient out of range".to_string())?;
        }
        fold.push(e);
    }
    Ok(FoldedWitness::of(fold))
}
/// The `COLUMNS` folding challenges, derived the way the checker derives them: the row
/// evaluation absorbed, then one weight-28 signed subring element per column. The samples are
/// independent derivations of the transcript, so they are drawn on every core; the result is
/// the sequence `sample_short_challenge` would produce.
fn folding_challenges(t: &mut Transcript, row: &RowEvaluation) -> Vec<ShortChallenge> {
    row.absorb(t);
    sample_short_challenges_parallel(t, COLUMNS, DEFAULT_WEIGHT, DEFAULT_BOUND, threads())
}

/// Garner's lift of one coefficient from its residues modulo `primes` (each already in
/// `[0, q)`) to the integer modulo their product. The reductions are Barrett's with the
/// library's magic `floor(2^43 / q)`: exact for operands below `2^31`, which every product of
/// two residues of primes below `2^15` is, and no division.
struct Garner {
    primes: Vec<u64>,
    magic: Vec<u64>,
    inverses: Vec<Vec<u64>>,
}

impl Garner {
    fn of(primes: &[u16]) -> Garner {
        let limbs = primes.len();
        Garner {
            primes: primes.iter().map(|&q| q as u64).collect(),
            magic: primes.iter().map(|&q| (1u64 << 43) / q as u64).collect(),
            inverses: (0..limbs)
                .map(|k| {
                    (0..k)
                        .map(|m| {
                            let (q, p) = (primes[k] as u64, primes[m] as u64);
                            (1..q).find(|x| x * p % q == 1).expect("the primes are distinct")
                        })
                        .collect()
                })
                .collect(),
        }
    }

    /// `x mod q_k` for `x < 2^31`.
    #[inline(always)]
    fn reduce(&self, k: usize, x: u64) -> u64 {
        let q = self.primes[k];
        let r = x - ((x * self.magic[k]) >> 43) * q;
        if r >= q {
            r - q
        } else {
            r
        }
    }

    #[inline]
    fn lift(&self, residues: &[u64], mixed: &mut [u64]) -> u128 {
        let limbs = self.primes.len();
        for k in 0..limbs {
            let q = self.primes[k];
            let mut x = residues[k];
            for m in 0..k {
                // `mixed[m] < q_m < 2^15`, so `x + q - reduce(mixed[m]) < 2^16` and the
                // product with an inverse below `q` stays below `2^31`.
                x = self.reduce(k, (x + q - self.reduce(k, mixed[m])) * self.inverses[k][m]);
            }
            mixed[k] = x;
        }
        let mut value = 0u128;
        for k in (0..limbs).rev() {
            value = value * self.primes[k] as u128 + mixed[k] as u128;
        }
        value
    }
}

/// The commitment's columns as coefficients modulo `modulus`, which is what the proof carries,
/// from the kernel's raw rows: `rows[c * limbs + k]` is column `c`'s 648 NTT-domain rows
/// modulo limb `k`. Each limb's rows go through that limb's inverse transform
/// ([`labinius::bd::coefficients_from_rows`]) 32 columns at a time, and Garner's algorithm
/// lifts the residues to the single integer modulo their product; the lift is exact because
/// `modulus` is that product. The column blocks are independent and run on `workers` threads.
fn commitment_coefficients_rows(
    rows: &[[u32; N]],
    primes: &[u16],
    modulus: u128,
    columns: usize,
    workers: usize,
) -> Vec<u128> {
    let limbs = primes.len();
    let hybrid=modulus==modulus_choice::HYBRID_MODULUS;
    let stride=if hybrid {3}else{limbs};
    assert_eq!(rows.len(), columns * stride);
    let garner = Garner::of(primes);
    let mut out = vec![0u128; columns * N];
    let blocks = columns.div_ceil(32);
    let runs = labinius::par::ranges(blocks, workers, 1);
    let out_addr = out.as_mut_ptr() as usize;
    // One run per logical CPU, placed explicitly for this dispatch and released at its end:
    // this is the first dispatch after the commitment's per-core placement, and the runner
    // spread the scheduler produced from there measured the phase at twice its cost.
    let placement = labinius::par::CorePlacement::for_logical(runs.len());
    labinius::par::pool().run(runs.len(), |r| {
        let _affinity = placement.as_ref().and_then(|p| p.pin(r));
        let mut residues: Vec<Vec<i16>> = (0..limbs).map(|_| vec![0i16; 32 * N]).collect();
        let mut mixed = vec![0u64; limbs];
        let mut at = vec![0u64; limbs];
        for blk in runs[r].clone() {
            let c0 = blk * 32;
            let cols = (columns - c0).min(32);
            for k in 0..limbs {
                let refs: Vec<&[u32; N]> = (0..cols).map(|c| &rows[(c0 + c) * stride + k]).collect();
                labinius::bd::coefficients_from_rows(primes[k], &refs, &mut residues[k][..cols * N]);
            }
            // Block `blk` owns coefficients `c0 * N .. (c0 + cols) * N` of `out`.
            let dst = unsafe { (out_addr as *mut u128).add(c0 * N) };
            for i in 0..cols * N {
                for k in 0..limbs {
                    at[k] = (residues[k][i] as i32).rem_euclid(primes[k] as i32) as u64;
                }
                let mut value = garner.lift(&at, &mut mixed);
                if hybrid {
                    let tail=rows[(c0+i/N)*3+2][i%N] as u64;
                    let digit=tail.wrapping_sub(value as u64).wrapping_mul(153)&1023;
                    value+=(3889u128*9721)*(digit as u128);
                }
                debug_assert!(value < modulus);
                unsafe { dst.add(i).write(value) };
            }
        }
    });
    out
}

/// The baseline's recombination through the library's commitment matrix, kept as the reference
/// the tests check [`commitment_coefficients_rows`] against.
#[cfg(test)]
fn commitment_coefficients(commitment: &Commitment, primes: &[u16], modulus: u128) -> Vec<u128> {
    let residues = labinius::bd::column_coefficients(commitment.matrix(), primes);
    let garner = Garner::of(primes);
    let limbs = primes.len();
    let mut mixed = vec![0u64; limbs];
    let mut at = vec![0u64; limbs];
    let mut out = Vec::with_capacity(residues[0].len());
    for i in 0..residues[0].len() {
        for k in 0..limbs {
            at[k] = (residues[k][i] as i32).rem_euclid(primes[k] as i32) as u64;
        }
        let value = garner.lift(&at, &mut mixed);
        debug_assert!(value < modulus);
        out.push(value);
    }
    out
}

/// The switch's rounds: the messages, the coordinates and the opened value, driven through
/// either prover.
trait SwitchRounds {
    fn msg(&mut self) -> [F162; 2];
    fn fold(&mut self, r: F162);
    fn final_eval(&self) -> F162;
}
impl SwitchRounds for cf::LeanSwitch<'_> {
    fn msg(&mut self) -> [F162; 2] {
        cf::LeanSwitch::msg(self)
    }
    fn fold(&mut self, r: F162) {
        cf::LeanSwitch::fold(self, r)
    }
    fn final_eval(&self) -> F162 {
        cf::LeanSwitch::final_eval(self)
    }
}
impl SwitchRounds for cf::SwitchProver {
    fn msg(&mut self) -> [F162; 2] {
        cf::SwitchProver::msg(self)
    }
    fn fold(&mut self, r: F162) {
        cf::SwitchProver::fold(self, r)
    }
    fn final_eval(&self) -> F162 {
        cf::SwitchProver::final_eval(self)
    }
}
fn switch_rounds(
    sw: &mut impl SwitchRounds,
    t: &mut Transcript,
) -> (Vec<[[u64; 3]; 2]>, Vec<F162>, F162) {
    let mut rounds = Vec::with_capacity(LOG_PACKED);
    let mut coords = Vec::with_capacity(LOG_PACKED);
    for _ in 0..LOG_PACKED {
        let msg = sw.msg();
        t.absorb_f162(&msg);
        let r = t.sample_f162(b"switch-round", 1)[0];
        rounds.push(msg.map(|v| v.0));
        coords.push(r);
        sw.fold(r);
    }
    let opened = sw.final_eval();
    t.absorb_f162(&[opened]);
    (rounds, coords, opened)
}

/// Batch positions of the fold recomputed through the transform path and compared with the
/// bit-sliced fold: 32 ring elements each, out of 4096. Every challenge and every column enters
/// each element, so one position exercises the whole accumulation; a mismatch means a defect
/// in the bit-sliced path on this machine, and the transform path then produces the fold.
const FOLD_CHECK_POSITIONS: core::ops::Range<usize> = 0..1;

/// `v = sum_j c_j W_j`: the bit-sliced fold from the words, spot-checked against the transform
/// path on [`FOLD_CHECK_POSITIONS`], with the transform path as the fallback.
fn fold_checked(
    values: &[B128],
    challenges: &[ShortChallenge],
    base: u16,
    workers: usize,
) -> Vec<labinius::RingElement648> {
    let bpc = COLUMN_LEN / 128;
    let bits = labinius::fold::fold_witness_bits(values, COLUMNS, COLUMN_LEN, challenges);
    let check = labinius::fold::fold_witness_positions(
        values,
        COLUMNS,
        challenges,
        bpc,
        base,
        FOLD_CHECK_POSITIONS,
    );
    let first = 32 * FOLD_CHECK_POSITIONS.start;
    let agree = check
        .iter()
        .zip(&bits[first..first + check.len()])
        .all(|(a, b)| a.v == b.v);
    if agree {
        return bits;
    }
    eprintln!("labinius-fold-check mismatch: the bit-sliced fold disagrees with the transform path; folding through the transform");
    labinius::fold::fold_witness_streaming(values, COLUMNS, challenges, bpc, base, workers)
}

/// The generic (non-hybrid) commitment: one task per physical core, temporarily placed by the
/// key's worker entry point; eight columns share each pass over the public key.
const COMMIT_GROUP: usize = 8;
const COMMIT_WORKERS: usize = 4;

type ProverStorage = (Table, Vec<[u32; N]>, FoldedWitness);

/// Keep large read-only/intermediate mappings alive through the output write.
/// The existing process exit retires them with the address space in one operation.
fn prove_with_storage(seed: &[u8; 32], modulus: u128, table_path: &str) -> (Proof, ProverStorage) {
    let load_clock = std::time::Instant::now();
    let table = Table::load(table_path);
    let values = table.words();
    let load_time = load_clock.elapsed();
    let workers = threads();
    let clock = std::time::Instant::now();
    // The pool's workers are started here, once; every parallel phase runs on them.
    let _ = labinius::par::pool();
    // The switch's buffers: huge pages requested, faulted in by their first writers on every
    // core. (A thread touching them during the commitment ran on a hyperthread sibling of a
    // commitment worker and cost the commitment more than it saved the expansion.)
    // The first eighth of the expansion is all the switch reads.
    let mut eq = cf::HugeBuf::<B128>::new(1 << (LOG_PACKED - 3));
    let planes = cf::LeanSwitch::planes(1 << LOG_PACKED);
    let primes = modulus_choice::limbs(modulus).expect("invalid commitment modulus");
    let p = params(&primes);
    // The table's pages come in on every core; the key is the embedded one when the modulus
    // is the default and the build host could compute it, and is built otherwise.
    populate_words(values, workers);
    let matrix_time = clock.elapsed();
    let hybrid = modulus == modulus_choice::HYBRID_MODULUS;
    // The default modulus commits through the embedded column-pair image, read in place; any
    // other modulus (or a build host that could not compute the image) builds its key here.
    let embedded = (hybrid && modulus == MODULUS)
        .then(|| keygen::pair_words(&EMBEDDED_KEY.0))
        .flatten();
    let (key, owned_pairs) = match embedded {
        Some(_) => (None, None),
        None if hybrid => {
            let key = keygen::build_key(modulus);
            let tail = keygen::build_tail_key(modulus);
            let pairs = labinius::simd::commit::hybrid::pair_keys(&key, &tail, workers);
            (None, Some(pairs))
        }
        None => (Some(keygen::build_key(modulus)), None),
    };
    let setup_time = clock.elapsed();
    let columns: Vec<&[B128]> = values.chunks_exact(COLUMN_LEN).collect();
    assert_eq!(columns.len(), COLUMNS, "the table is COLUMNS columns");
    let rows = match (embedded, &owned_pairs, &key) {
        (Some(pairs), _, _) => labinius::simd::commit::hybrid::commit_columns_rows_pairs(pairs, &columns, workers),
        (None, Some(pairs), _) => labinius::simd::commit::hybrid::commit_columns_rows_pairs(pairs, &columns, workers),
        (None, None, Some(key)) => key.commit_columns_rows(&columns, workers.min(COMMIT_WORKERS), COMMIT_GROUP),
        _ => unreachable!("a key or a pair image"),
    };
    let commit_time = clock.elapsed();
    let commitment = pack_coefficients(
        &commitment_coefficients_rows(&rows, &primes, modulus, COLUMNS, workers),
        modulus,
    );
    let recombine_time = clock.elapsed();
    let mut t = transcript(seed, modulus, &commitment);
    let (lo, hi) = point(&mut t);
    // A coordinate equal to one (probability 2^-128 per coordinate) has no half ratio: the
    // baseline's prover over its own full workspace takes over for the whole switch.
    let degenerate = hi.iter().any(|&r| r == B128::ONE);
    let eighths = if degenerate {
        None
    } else {
        Some(unsafe { cf::eq_expand_and_partials_eighths(&hi, values, eq.as_mut_ptr(), workers) })
    };
    let partials = match &eighths {
        Some(sets) => cf::partials_from_eighths(sets, &hi),
        None => {
            let mut ws = cf::Workspace::new(LOG_PACKED, 1);
            ws.partial_evals(0, values, &hi)
        }
    };
    let value = claim(&partials, &lo);
    t.absorb_b128(&[value]);
    t.absorb_b128(&partials);
    let partials_time = clock.elapsed();
    let batch = cf::eq_expand_f162(&t.sample_f162(b"switch-batch", 7));
    let lean = eighths.and_then(|sets| {
        // Only the nondegenerate path initialized this prefix.
        let eq_words: &[B128] = unsafe { eq.as_slice() };
        cf::LeanSwitch::with_eighths(values, eq_words, &hi, &batch, workers, planes, sets)
    });
    let (rounds, coords, opened) = match lean {
        Some(mut sw) => switch_rounds(&mut sw, &mut t),
        None => {
            // The baseline's prover, whose workspace holds its own full expansion.
            let mut ws = cf::Workspace::new(LOG_PACKED, 1);
            let again = ws.partial_evals(0, values, &hi);
            assert_eq!(again, partials);
            let mut sw = cf::SwitchProver::from_workspace(ws, values, &[F162::ONE], &batch);
            switch_rounds(&mut sw, &mut t)
        }
    };
    let switch_time = clock.elapsed();
    // The expansion is done with; its pages go back while the compute-bound phases run.
    std::thread::spawn(move || drop(eq));
    let point = EvaluationPoint::msb_first(&p, &coords);
    let row = RowEvaluation::of(labinius::eval::row_evaluate_words(values, point.p0(), workers));
    let row_time = clock.elapsed();
    let challenges = folding_challenges(&mut t, &row);
    let challenge_time = clock.elapsed();
    let folded = FoldedWitness::of(labinius::fold::fold_witness_bitsliced(
        values, COLUMNS, &challenges, COLUMN_LEN / 128, primes[0], workers,
    ));
    let fold_time = clock.elapsed();
    let proof = Proof {
        version: 1,
        modulus,
        commitment,
        partials: partials.iter().map(|v| v.0).collect(),
        rounds,
        opened: opened.0,
        row: wire::pack_row_evaluation(&row),
        folded: pack_folded(&folded),
    };
    let encode_time = clock.elapsed();
    eprintln!("labinius-profile load_us={} matrix_us={} setup_us={} commit_us={} recombine_us={} partials_us={} switch_us={} row_us={} challenges_us={} fold_us={} encode_us={} total_us={} threads={} fold_gfni={}",
        load_time.as_micros(), matrix_time.as_micros(), (setup_time-matrix_time).as_micros(),
        (commit_time-setup_time).as_micros(), (recombine_time-commit_time).as_micros(),
        (partials_time-recombine_time).as_micros(), (switch_time-partials_time).as_micros(),
        (row_time-switch_time).as_micros(), (challenge_time-row_time).as_micros(),
        (fold_time-challenge_time).as_micros(), (encode_time-fold_time).as_micros(),
        encode_time.as_micros(), workers,
        labinius::fold::FOLD_USED_GFNI.load(std::sync::atomic::Ordering::Relaxed));
    (proof, (table, rows, folded))
}

#[cfg(test)]
fn prove(seed: &[u8; 32], modulus: u128, table_path: &str) -> Proof {
    prove_with_storage(seed, modulus, table_path).0
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
        modulus_choice::parse(&args[6]).expect("invalid commitment modulus")
    } else {
        MODULUS
    };
    let output_clock = std::time::Instant::now();
    let (proof, _storage) = prove_with_storage(&seed, modulus, &args[4]);
    let prove_return = output_clock.elapsed();
    {
        use std::io::Write;
        let file = std::fs::File::create(&args[2]).expect("create proof file");
        let mut out = std::io::BufWriter::with_capacity(1 << 20, file);
        codec().serialize_into(&mut out, &proof).expect("write proof");
        out.flush().expect("flush proof");
    }
    eprintln!("labinius-output prove_return_us={} write_us={}",
        prove_return.as_micros(), (output_clock.elapsed()-prove_return).as_micros());
    // The proof is on disk; nothing the destructors would do is observable.
    std::process::exit(0);
}

#[cfg(test)]
mod tests {
    use super::*;
    use labinius::eval::eq_table;
    use labinius::ring::Representation;

    fn checked_f162(x: [u64; 3]) -> F162 {
        assert_eq!(x[2] >> 34, 0, "noncanonical F162 element");
        F162(x)
    }
    fn dot_f162(a: &[F162], b: &[F162]) -> F162 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b)
            .fold(F162::ZERO, |acc, (&x, &y)| acc + x * y)
    }
    /// `v mod 2` as field elements, in the witness's index order.
    fn components_mod_2(v: &[labinius::RingElement648]) -> Vec<F162> {
        let mut out = vec![F162::ZERO; 4 * v.len()];
        for (m, e) in v.iter().enumerate() {
            for p in 0..rq::N / 4 {
                for k in 0..4 {
                    if e.v[4 * p + k] & 1 != 0 {
                        out[4 * m + k].0[p >> 6] |= 1u64 << (p & 63);
                    }
                }
            }
        }
        out
    }
    /// The checker's acceptance condition, reimplemented here so that the round trip tests the
    /// proof this prover writes rather than the objects it holds. The ring arithmetic is the
    /// same [`rq`] the checker runs.
    fn verify(seed: &[u8; 32], proof: &Proof) {
        assert_eq!(proof.version, 1);
        assert_eq!(proof.partials.len(), 128);
        assert_eq!(proof.rounds.len(), LOG_PACKED);
        let modulus = proof.modulus;
        let values = trace(seed);
        let mut t = transcript(seed, modulus, &proof.commitment);
        let (lo, hi) = point(&mut t);
        let mut ws = cf::Workspace::new(LOG_PACKED, 1);
        let trusted = ws.partial_evals(0, &values, &hi);
        let value = claim(&trusted, &lo);
        let partials: Vec<_> = proof.partials.iter().copied().map(B128).collect();
        t.absorb_b128(&[value]);
        t.absorb_b128(&partials);
        let batch = cf::eq_expand_f162(&t.sample_f162(b"switch-batch", 7));
        let mut sw = cf::SwitchVerifier::start(&partials, value, &lo, &batch).unwrap();
        let mut coords = Vec::new();
        for raw in &proof.rounds {
            let msg = raw.map(checked_f162);
            t.absorb_f162(&msg);
            let r = t.sample_f162(b"switch-round", 1)[0];
            sw.round(msg, r);
            coords.push(r);
        }
        let opened = checked_f162(proof.opened);
        sw.finish(&hi, &coords, &batch, opened).unwrap();
        t.absorb_f162(&[opened]);
        let (high, low) = coords.split_at(COLUMN_LOG as usize);
        let p0: Vec<F162> = low.iter().rev().copied().collect();
        let p1: Vec<F162> = high.iter().rev().copied().collect();
        let row = wire::unpack_row_evaluation(&proof.row, COLUMNS).unwrap();
        let folded = unpack_folded(&proof.folded, RING_ELEMENTS).unwrap();
        let challenges = {
            row.absorb(&mut t);
            (0..COLUMNS)
                .map(|_| {
                    labinius::challenge::sample_short_challenge(&mut t, DEFAULT_WEIGHT, DEFAULT_BOUND).0
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(dot_f162(&eq_table(&p1), row.values()), opened);
        let v = folded.elements();
        assert_eq!(v.len(), RING_ELEMENTS);
        let mut normsq = 0u64;
        let mut worst = 0u128;
        for e in v {
            assert_eq!(e.representation, Representation::Coefficients);
            for &c in &e.v {
                let magnitude = u64::from((c as i32).unsigned_abs());
                normsq += magnitude * magnitude;
                worst = worst.max(magnitude as u128);
            }
        }
        assert!(normsq <= 293_534_171_136);
        assert!(worst <= (modulus - 1) / 2);
        let columns =
            rq::unpack_ring_elements(&proof.commitment, COLUMNS * rq::N, modulus).unwrap();
        let matrix = rq::ajtai_matrix(modulus, RING_ELEMENTS);
        let coefficients: Vec<i16> = v.iter().flat_map(|e| e.v).collect();
        assert_eq!(
            rq::a_times_v(&matrix, &coefficients, modulus),
            rq::challenge_combination(&columns, &challenges, modulus),
            "the fold does not open the folded commitment"
        );
        let parities: Vec<F162> = challenges.iter().map(|c| c.to_f162()).collect();
        assert_eq!(
            dot_f162(&eq_table(&p0), &components_mod_2(v)),
            dot_f162(&parities, row.values())
        );
    }

    struct TableFile(std::path::PathBuf);
    impl Drop for TableFile {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }
    impl TableFile {
        fn path(&self) -> &str {
            self.0.to_str().unwrap()
        }
    }
    fn table_file(seed: &[u8; 32]) -> TableFile {
        let path = std::env::temp_dir().join(format!(
            "labinius-fast-table-{}-{}.bin",
            std::process::id(),
            seed.iter().map(|byte| format!("{byte:02x}")).collect::<String>()
        ));
        let mut bytes = Vec::new();
        for value in trace(seed) {
            bytes.extend_from_slice(&value.0.to_le_bytes());
        }
        std::fs::write(&path, bytes).unwrap();
        TableFile(path)
    }

    #[test]
    fn loaded_table_matches_trace() {
        let seed = [11u8; 32];
        let table = table_file(&seed);
        let loaded = Table::load(table.path());
        assert_eq!(loaded.words(), &trace(&seed)[..]);
    }

    #[test]
    fn baseline_prove_reads_witness_file() {
        let path = std::env::temp_dir().join(format!(
            "labinius-fast-short-{}.bin",
            std::process::id()
        ));
        std::fs::write(&path, [1u8; 32]).unwrap();
        let result = std::panic::catch_unwind(|| {
            prove(&[1; 32], MODULUS, path.to_str().unwrap());
        });
        let _ = std::fs::remove_file(&path);
        assert!(result.is_err(), "the prover accepted a truncated witness table");
    }

    #[test]
    fn round_trip_and_reject_wrong_instance() {
        let seed = [42; 32];
        let table = table_file(&seed);
        let proof = prove(&seed, MODULUS, table.path());
        let bytes = codec().serialize(&proof).unwrap();
        let decoded: Proof = codec().deserialize(&bytes).unwrap();
        verify(&seed, &decoded);
        assert!(std::panic::catch_unwind(|| verify(&[43; 32], &decoded)).is_err());
    }

    #[test]
    fn solution_selected_modulus_round_trip() {
        let modulus = 9721u128 * 2917 * 4861;
        assert_ne!(modulus, MODULUS);
        let seed = [44; 32];
        let table = table_file(&seed);
        let proof = prove(&seed, modulus, table.path());
        let bytes = codec().serialize(&proof).unwrap();
        let decoded: Proof = codec().deserialize(&bytes).unwrap();
        assert_eq!(decoded.modulus, modulus);
        verify(&[44; 32], &decoded);
    }

    #[test]
    fn packers_match_the_checker_format() {
        let mut x = 0x2545F4914F6CDD1Du128;
        let mut next = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let coefficients: Vec<u128> = (0..3 * rq::N + 5).map(|_| next() % MODULUS).collect();
        assert_eq!(
            pack_coefficients(&coefficients, MODULUS),
            rq::pack_ring_elements(&coefficients, MODULUS)
        );
        let elements: Vec<labinius::RingElement648> = (0..19)
            .map(|_| {
                let mut e = labinius::RingElement648::zero(Representation::Coefficients);
                for slot in e.v.iter_mut() {
                    *slot = (next() % 2001) as i16 - 1000;
                }
                e
            })
            .collect();
        let folded = FoldedWitness::of(elements);
        let mut want = Vec::new();
        for e in folded.elements() {
            for &c in &e.v {
                want.extend_from_slice(&(c as i32).to_le_bytes());
            }
        }
        assert_eq!(pack_folded(&folded), want);
    }

    #[test]
    fn garner_lift_is_the_chinese_remainder() {
        let primes = modulus_choice::limbs(MODULUS).unwrap();
        let product:u128=primes.iter().map(|&q|q as u128).product();
        let garner = Garner::of(&primes);
        let mut x = 0x1234_5678u128;
        let mut mixed = vec![0u64; primes.len()];
        for _ in 0..2000 {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            let value = x % product;
            let residues: Vec<u64> = primes.iter().map(|&q| (value % q as u128) as u64).collect();
            assert_eq!(garner.lift(&residues, &mut mixed), value);
        }
        let residues: Vec<u64> = primes.iter().map(|&q| q as u64 - 1).collect();
        assert_eq!(garner.lift(&residues, &mut mixed), product - 1);
    }

    #[test]
    fn bulk_byte_fields_serialize_as_byte_sequences() {
        #[derive(Serialize)]
        struct Plain<'a> {
            version: u32,
            modulus: u128,
            commitment: &'a Vec<u8>,
            partials: &'a Vec<u128>,
            rounds: &'a Vec<[[u64; 3]; 2]>,
            opened: [u64; 3],
            row: &'a Vec<u8>,
            folded: &'a Vec<u8>,
        }
        let proof = Proof {
            version: 1,
            modulus: MODULUS,
            commitment: (0..=255).collect(),
            partials: vec![0, u128::MAX, 17],
            rounds: vec![[[0, 1, 2], [u64::MAX, 4, 5]]],
            opened: [7, 8, 9],
            row: Vec::new(),
            folded: vec![3; 1025],
        };
        let plain = Plain {
            version: proof.version,
            modulus: proof.modulus,
            commitment: &proof.commitment,
            partials: &proof.partials,
            rounds: &proof.rounds,
            opened: proof.opened,
            row: &proof.row,
            folded: &proof.folded,
        };
        let bytes = codec().serialize(&proof).unwrap();
        assert_eq!(bytes, codec().serialize(&plain).unwrap());
        let decoded: Proof = codec().deserialize(&bytes).unwrap();
        assert_eq!(decoded.commitment, proof.commitment);
        assert_eq!(decoded.row, proof.row);
        assert_eq!(decoded.folded, proof.folded);
    }

    /// The embedded column-pair image is the one the prover would build from the key and the
    /// 1024-lift key, word for word.
    #[test]
    fn embedded_pairs_match_the_built_keys() {
        if EMBEDDED_KEY.0.is_empty() {
            // Built on a host without the kernels: the prover builds the pairs at run time.
            return;
        }
        let key = keygen::build_key(MODULUS);
        let tail = keygen::build_tail_key(MODULUS);
        let built = labinius::simd::commit::hybrid::pair_keys(&key, &tail, threads());
        let read = keygen::pair_words(&EMBEDDED_KEY.0).expect("a valid image");
        assert_eq!(read, &built[..], "the embedded image differs");
    }

    #[test]
    fn modulus_changes_transcript() {
        let mut a = transcript(&[0; 32], MODULUS, &[]);
        let mut b = transcript(&[0; 32], 9721u128 * 2917 * 4861, &[]);
        let mut x = [0; 32];
        let mut y = [0; 32];
        a.fill(b"test", &mut x);
        b.fill(b"test", &mut y);
        assert_ne!(x, y);
    }

    /// The limb path's commitment, recombined, is the schoolbook `A W_j` in `R_Q`.
    ///
    /// This is the one place where the derived limb form of `A` and the inverse of the
    /// commitment's per-limb NTT decomposition are checked against the definition in `R_Q`
    /// directly, with no transform anywhere in the reference. Two columns are enough: each is a
    /// 4096-term inner product of ring elements. Both the baseline's recombination through the
    /// commitment matrix and the direct one from the kernel's rows are checked, against each
    /// other and against the definition.
    #[test]
    fn commitment_is_the_schoolbook_product() {
        let modulus = 110_277_094_573; // Retained generic RNS path.
        let primes = modulus_choice::limbs(modulus).unwrap();
        let p = params(&primes);
        let matrix = rq::ajtai_matrix(modulus, RING_ELEMENTS);
        let key = CommitmentKey::from_coefficients(
            COLUMN_LEN,
            &matrix,
            modulus,
            p.base,
            &p.extra_moduli,
        );
        let pp = PublicParameters::with_key(p.clone(), rq::MATRIX_SEED, key);
        let values = trace(&[7; 32]);
        let w = Witness::lifted(&p, &values).unwrap();
        let (commitment, _) = Prover::cold(&pp).commit(&w);
        let coefficients = commitment_coefficients(&commitment, &primes, modulus);
        let columns: Vec<&[B128]> = values.chunks_exact(COLUMN_LEN).collect();
        let rows = pp.key().commit_columns_rows(&columns, 3, 4);
        let direct = commitment_coefficients_rows(&rows, &primes, modulus, COLUMNS, 3);
        assert_eq!(coefficients, direct, "the two recombinations differ");
        for j in [0usize, COLUMNS - 1] {
            let column = &w.elements()[j * COLUMN_LEN..(j + 1) * COLUMN_LEN];
            assert_eq!(
                coefficients[j * rq::N..(j + 1) * rq::N],
                rq::commit_binary_column(&matrix, column, modulus)[..],
                "column {j} does not match the schoolbook product"
            );
        }
    }

    /// The streaming fold from the words is the fold from the kept transform, and the row
    /// evaluation from the words is the one from the lifted witness.
    #[test]
    fn streaming_fold_and_row_match_the_library() {
        let modulus = MODULUS;
        let primes = modulus_choice::limbs(modulus).unwrap();
        let p = params(&primes);
        let matrix = rq::ajtai_matrix(modulus, RING_ELEMENTS);
        let key = CommitmentKey::from_coefficients(
            COLUMN_LEN,
            &matrix,
            modulus,
            p.base,
            &p.extra_moduli,
        );
        let pp = PublicParameters::with_key(p.clone(), rq::MATRIX_SEED, key);
        let values = trace(&[8; 32]);
        let w = Witness::lifted(&p, &values).unwrap();
        let mut prover = Prover::cold(&pp);
        let (_, opening) = prover.commit(&w);
        let mut t = Transcript::new(b"labinius-fast/test-fold");
        let coords = t.sample_f162(b"coords", LOG_PACKED);
        let point = EvaluationPoint::msb_first(&p, &coords);
        let row = w.row_evaluate(&point);
        let direct = labinius::eval::row_evaluate_words(&values, point.p0(), 3);
        assert_eq!(row.values(), &direct[..]);
        let challenges = folding_challenges(&mut t, &row);
        let folded = prover.fold(opening, &FoldingChallenges::of(challenges.clone()));
        let streamed = labinius::fold::fold_witness_streaming(
            &values,
            COLUMNS,
            &challenges,
            COLUMN_LEN / 128,
            primes[0],
            3,
        );
        assert_eq!(folded.elements(), &streamed[..]);
        let bits = labinius::fold::fold_witness_bits(&values, COLUMNS, COLUMN_LEN, &challenges);
        assert_eq!(folded.elements(), &bits[..], "the bit-sliced fold differs");
        let positions = labinius::fold::fold_witness_positions(
            &values,
            COLUMNS,
            &challenges,
            COLUMN_LEN / 128,
            primes[0],
            3..5,
        );
        assert_eq!(&folded.elements()[96..160], &positions[..]);
        assert_eq!(&fold_checked(&values, &challenges, primes[0], 3)[..], folded.elements());
    }
    #[test]
    fn hybrid_commitment_matches_schoolbook_under_new_modulus() {
        let q=modulus_choice::HYBRID_MODULUS;let ring=128usize;let columns=4;
        let primes=modulus_choice::limbs(q).unwrap();let p=params(&primes);
        let matrix=rq::ajtai_matrix(q,ring);
        let key=CommitmentKey::from_coefficients(ring*4,&matrix,q,p.base,&p.extra_moduli);
        let tail=keygen::build_tail_key_from_matrix(&matrix);
        let mut t=Transcript::new(b"hybrid-commitment-regression");
        let values=t.sample_b128(b"binary",ring*4*columns);
        let cols:Vec<&[B128]>=values.chunks_exact(ring*4).collect();
        let rows=labinius::simd::commit::hybrid::commit_columns_rows(&key,&tail,&cols,3,4);
        let coefficients=commitment_coefficients_rows(&rows,&primes,q,columns,3);
        for c in 0..columns {
            let lifted:Vec<F162>=cols[c].iter().map(|&v|F162::from_b128(v)).collect();
            assert_eq!(&coefficients[c*N..(c+1)*N],&rq::commit_binary_column(&matrix,&lifted,q)[..]);
        }
    }
    /// The column-pair kernel produces the grouped kernel's rows exactly, across a full and a
    /// partial certified chunk and a full and a partial 16-column group, and its coefficients
    /// are the schoolbook commitment.
    #[test]
    fn pair_commitment_matches_grouped_kernel_and_schoolbook() {
        let q=modulus_choice::HYBRID_MODULUS;let ring=288usize;let columns=17;
        let primes=modulus_choice::limbs(q).unwrap();let p=params(&primes);
        let matrix=rq::ajtai_matrix(q,ring);
        let key=CommitmentKey::from_coefficients(ring*4,&matrix,q,p.base,&p.extra_moduli);
        let tail=keygen::build_tail_key_from_matrix(&matrix);
        let mut t=Transcript::new(b"pair-commitment-regression");
        let values=t.sample_b128(b"binary",ring*4*columns);
        let cols:Vec<&[B128]>=values.chunks_exact(ring*4).collect();
        let grouped=labinius::simd::commit::hybrid::commit_columns_rows(&key,&tail,&cols,3,4);
        let pairs=labinius::simd::commit::hybrid::pair_keys(&key,&tail,3);
        let rows=labinius::simd::commit::hybrid::commit_columns_rows_pairs(&pairs,&cols,3);
        assert_eq!(rows,grouped);
        let coefficients=commitment_coefficients_rows(&rows,&primes,q,columns,3);
        for c in [0,16] {
            let lifted:Vec<F162>=cols[c].iter().map(|&v|F162::from_b128(v)).collect();
            assert_eq!(&coefficients[c*N..(c+1)*N],&rq::commit_binary_column(&matrix,&lifted,q)[..]);
        }
    }

}
