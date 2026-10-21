//! One packed-binary PCS instance, including the GHASH -> F162 switch.
//!
//! The commitment modulus the checker accepts is any `u128` of the right size. This prover
//! reaches the subset that factors into the library's primes: it commits modulo each limb with
//! the AVX-512 kernels and then recombines the residues into the coefficients modulo their
//! product that the proof carries. `A` is defined by its coefficients in `R_Q`
//! ([`rq::ajtai_matrix`]), so the limb form of the key is derived rather than drawn, which costs
//! one reduction pass and one forward transform per limb.
use bincode::Options;
use labinius::challenge::{sample_short_challenge, ShortChallenge, DEFAULT_BOUND, DEFAULT_WEIGHT};
use labinius::fields::{
    crossfield as cf,
    scalar::{B128, F162},
};
use labinius::key::CommitmentKey;
use labinius::params::N;
use labinius::ring::{Modulus, Representation};
use labinius::scheme::*;
use labinius::{wire, Transcript};
use serde::{Deserialize, Serialize};

mod modulus_choice;
mod parameters;
/// Byte-for-byte the checker's `rq.rs`, so that the two sides of the format can be diffed. A
/// release build of the prover only uses the matrix expansion and the packer; the rest is what
/// the tests check the proof against.
#[allow(dead_code)]
mod rq;
use parameters::{COLUMNS, COLUMN_LEN, COLUMN_LOG, LOG_PACKED, LOG_VARS, MODULUS, RING_ELEMENTS};

const LIMIT: u64 = 64 * 1024 * 1024;

#[derive(Serialize, Deserialize)]
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
fn codec() -> impl Options {
    bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .with_limit(LIMIT)
        .reject_trailing_bytes()
}
fn params(primes: &[u16]) -> Params {
    let ms: Vec<_> = primes
        .iter()
        .map(|q| Modulus::ALL.into_iter().find(|m| m.prime() == *q).unwrap())
        .collect();
    Params::with_base(
        LOG_PACKED as u32,
        COLUMN_LOG,
        ms[0],
        ms[1..].to_vec(),
        Opening::Clear,
    )
    .unwrap()
}
fn seed() -> [u8; 32] {
    let s = std::env::var("LABINIUS_SEED").expect("LABINIUS_SEED is required");
    assert_eq!(s.len(), 64);
    core::array::from_fn(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).expect("invalid seed"))
}
/// The harness witness table: `(1 << LOG_VARS) / 8` little-endian BLAKE3 XOF bytes, each 16
/// bytes one GHASH word, mapped read-only instead of copied.
struct Table {
    ptr: *const u8,
    len: usize,
}

impl Table {
    fn words(&self) -> &[B128] {
        unsafe { core::slice::from_raw_parts(self.ptr as *const B128, self.len / 16) }
    }
}

fn load_table(path: &str) -> Table {
    use std::os::unix::io::AsRawFd;
    extern "C" {
        fn mmap(
            addr: *mut core::ffi::c_void,
            len: usize,
            prot: i32,
            flags: i32,
            fd: i32,
            offset: i64,
        ) -> *mut core::ffi::c_void;
    }
    assert!(
        cfg!(target_endian = "little"),
        "the table is little-endian"
    );
    assert_eq!(core::mem::size_of::<B128>(), 16, "a GHASH word is 16 bytes");
    let file = std::fs::File::open(path).expect("open witness table");
    let len = file.metadata().expect("stat witness table").len() as usize;
    assert_eq!(
        len,
        (1 << LOG_VARS) / 8,
        "witness table has the wrong length"
    );
    let ptr = unsafe { mmap(core::ptr::null_mut(), len, 1, 2, file.as_raw_fd(), 0) };
    assert!(ptr as isize != -1, "witness table could not be mapped");
    Table {
        ptr: ptr as *const u8,
        len,
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
/// The checker's folded-witness encoding, which this must reproduce byte for byte.
const FOLDED_BYTES: usize = 4;
fn pack_folded(folded: &FoldedWitness) -> Vec<u8> {
    let elements = folded.elements();
    assert!(
        elements
            .iter()
            .all(|e| e.representation == Representation::Coefficients),
        "a fold is in coefficient form"
    );
    let mut out = Vec::with_capacity(FOLDED_BYTES * N * elements.len());
    for e in elements {
        for &c in &e.v {
            out.extend_from_slice(&(c as i32).to_le_bytes());
        }
    }
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
/// evaluation absorbed, then one weight-28 signed subring element per column.
fn folding_challenges(t: &mut Transcript, row: &RowEvaluation) -> Vec<ShortChallenge> {
    row.absorb(t);
    (0..COLUMNS)
        .map(|_| sample_short_challenge(t, DEFAULT_WEIGHT, DEFAULT_BOUND).0)
        .collect()
}
/// The commitment's columns as coefficients modulo `modulus`, which is what the proof carries.
///
/// The library's commitment is the four `R_162` components of each column in each limb's own NTT
/// domain. `bd::column_coefficients` inverts that per limb — the recombination of the four
/// components, then the limb's inverse transform — and Garner's algorithm lifts the residues to
/// the single integer modulo their product. The lift is exact because `modulus` is that product,
/// which is also why only such a modulus is reachable from here.
fn commitment_coefficients(residues: &[Vec<i16>], primes: &[u16], modulus: u128) -> Vec<u128> {
    let limbs = primes.len();
    let inverses: Vec<Vec<u64>> = (0..limbs)
        .map(|k| {
            (0..k)
                .map(|m| {
                    let (q, p) = (primes[k] as u64, primes[m] as u64);
                    (1..q).find(|x| x * p % q == 1).expect("the primes are distinct")
                })
                .collect()
        })
        .collect();
    let mut mixed = vec![0u64; limbs];
    let count = residues[0].len();
    let mut out = vec![0u128; count];
    let threads = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(1)
        .clamp(1, 8)
        .min(count.max(1));
    let per = count.div_ceil(threads);
    if threads > 1 && per > 0 {
        std::thread::scope(|scope| {
            for (t, chunk) in out.chunks_mut(per).enumerate() {
                let residues = &residues;
                let inverses = &inverses;
                let primes = primes;
                scope.spawn(move || {
                    let mut mixed = vec![0u64; limbs];
                    let start = t * per;
                    for (j, slot) in chunk.iter_mut().enumerate() {
                        let i = start + j;
                        for k in 0..limbs {
                            let q = primes[k] as u64;
                            let mut x =
                                (residues[k][i] as i32).rem_euclid(primes[k] as i32) as u64;
                            for m in 0..k {
                                x = (x + q - mixed[m] % q) % q * inverses[k][m] % q;
                            }
                            mixed[k] = x;
                        }
                        let mut value = 0u128;
                        for k in (0..limbs).rev() {
                            value = value * primes[k] as u128 + mixed[k] as u128;
                        }
                        debug_assert!(value < modulus);
                        *slot = value;
                    }
                });
            }
        });
    } else {
        for (i, slot) in out.iter_mut().enumerate() {
            for k in 0..limbs {
                let q = primes[k] as u64;
                let mut x = (residues[k][i] as i32).rem_euclid(primes[k] as i32) as u64;
                for m in 0..k {
                    x = (x + q - mixed[m] % q) % q * inverses[k][m] % q;
                }
                mixed[k] = x;
            }
            let mut value = 0u128;
            for k in (0..limbs).rev() {
                value = value * primes[k] as u128 + mixed[k] as u128;
            }
            debug_assert!(value < modulus);
            *slot = value;
        }
    }
    out
}
enum Sw<'a> {
    Lean(cf::LeanSwitch<'a>),
    Base(cf::SwitchProver),
}

impl Sw<'_> {
    fn msg(&mut self) -> [F162; 2] {
        match self {
            Sw::Lean(s) => s.msg(),
            Sw::Base(s) => s.msg(),
        }
    }
    fn fold(&mut self, r: F162) {
        match self {
            Sw::Lean(s) => s.fold(r),
            Sw::Base(s) => s.fold(r),
        }
    }
    fn final_eval(&self) -> F162 {
        match self {
            Sw::Lean(s) => s.final_eval(),
            Sw::Base(s) => s.final_eval(),
        }
    }
}

fn prove(seed: &[u8; 32], modulus: u128, table_path: &str) -> Proof {
    let load_start = std::time::Instant::now();
    let table = load_table(table_path);
    let values = table.words();
    eprintln!(
        "labinius-load load_us={}",
        load_start.elapsed().as_micros()
    );
    let clock = std::time::Instant::now();
    let primes = modulus_choice::limbs(modulus).expect("invalid commitment modulus");
    let p = params(&primes);
    let matrix = rq::ajtai_matrix(modulus, RING_ELEMENTS);
    let matrix_time = clock.elapsed();
    let key = CommitmentKey::from_coefficients(
        COLUMN_LEN,
        &matrix,
        modulus,
        p.base,
        &p.extra_moduli,
    );
    let pp = PublicParameters::with_key(p.clone(), rq::MATRIX_SEED, key);
    let setup_time = clock.elapsed();
    let mut prover = Prover::cold_without_transform(&pp);
    let witness_time = clock.elapsed();
    let (_commitment, opening) = prover.commit_words(values);
    let commit_time = clock.elapsed();
    let residues = opening.coefficients(&primes);
    let commitment = rq::pack_ring_elements(
        &commitment_coefficients(&residues, &primes, modulus),
        modulus,
    );
    let recombine_time = clock.elapsed();
    let mut t = transcript(seed, modulus, &commitment);
    let (lo, hi) = point(&mut t);
    let mut ws = cf::Workspace::new_lean(LOG_PACKED);
    let partials = ws.partial_evals(0, values, &hi);
    let value = claim(&partials, &lo);
    t.absorb_b128(&[value]);
    t.absorb_b128(&partials);
    let partials_time = clock.elapsed();
    let batch = cf::eq_expand_f162(&t.sample_f162(b"switch-batch", 7));
    let eq = ws.into_eq();
    let mut sw = if hi.iter().all(|&h| h.0 != 1) {
        Sw::Lean(cf::LeanSwitch::new(values, eq, &batch, &hi).expect("the point is invertible"))
    } else {
        Sw::Base(cf::SwitchProver::from_workspace(
            cf::Workspace::with_eq(eq),
            values,
            &[F162::ONE],
            &batch,
        ))
    };
    let switch_build_time = clock.elapsed();
    let mut rounds = Vec::new();
    let mut coords = Vec::new();
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
    let switch_time = clock.elapsed();
    let point = EvaluationPoint::msb_first(&p, &coords);
    let row = RowEvaluation::of(labinius::eval::row_evaluate_words(values, point.p0()));
    let row_time = clock.elapsed();
    let challenges = folding_challenges(&mut t, &row);
    let challenges_time = clock.elapsed();
    let folded = prover.fold_from_witness_words(opening, &FoldingChallenges::of(challenges), values);
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
    eprintln!("labinius-profile matrix_us={} setup_us={} witness_us={} commit_us={} recombine_us={} partials_us={} switch_build_us={} switch_rounds_us={} row_us={} challenges_us={} fold_us={} encode_us={} total_us={}",
        matrix_time.as_micros(), (setup_time-matrix_time).as_micros(),
        (witness_time-setup_time).as_micros(), (commit_time-witness_time).as_micros(),
        (recombine_time-commit_time).as_micros(), (partials_time-recombine_time).as_micros(),
        (switch_build_time-partials_time).as_micros(), (switch_time-switch_build_time).as_micros(),
        (row_time-switch_time).as_micros(), (challenges_time-row_time).as_micros(),
        (fold_time-challenges_time).as_micros(), (encode_time-fold_time).as_micros(),
        encode_time.as_micros());
    proof
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
    let proof = prove(&seed, modulus, &args[4]);
    std::fs::write(&args[2], codec().serialize(&proof).unwrap()).unwrap();
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
        let challenges = folding_challenges(&mut t, &row);
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
        let loaded = load_table(table.path());
        assert_eq!(loaded.words(), trace(&seed).as_slice());
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
    /// 4096-term inner product of ring elements.
    #[test]
    fn commitment_is_the_schoolbook_product() {
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
        let values = trace(&[7; 32]);
        let w = Witness::lifted(&p, &values).unwrap();
        let (_commitment, opening) = Prover::cold(&pp).commit(&w);
        let coefficients = commitment_coefficients(&opening.coefficients(&primes), &primes, modulus);
        for j in [0usize, COLUMNS - 1] {
            let column = &w.elements()[j * COLUMN_LEN..(j + 1) * COLUMN_LEN];
            assert_eq!(
                coefficients[j * rq::N..(j + 1) * rq::N],
                rq::commit_binary_column(&matrix, column, modulus)[..],
                "column {j} does not match the schoolbook product"
            );
        }
    }
}
