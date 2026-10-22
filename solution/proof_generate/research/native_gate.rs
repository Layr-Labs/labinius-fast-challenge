//! Build-time opportunity screen on the scored CPU. This is not a proof benchmark.
//! All three backends run serially on the same fresh input, with their actual public
//! key image mapped and populated inside the measured interval. A pass only permits
//! the candidate to be tested by the unchanged five-proof official benchmark.
use crate::{keygen, parameters, r9, reference_keygen};
use labinius::fields::scalar::B128;
use labinius::simd::commit::hybrid;
use labinius_reference126 as reference;
use std::ffi::c_void;
use std::io::Read;
use std::os::fd::AsRawFd;
use std::path::Path;
use std::time::{Duration, Instant};

const Q: u128 = parameters::MODULUS;
const KEY126_BYTES: usize = 21_233_664;
const MIN_RELATIVE_GAIN: f64 = 0.01;
const MIN_WINS: usize = 4;
const TRIALS: usize = 5;

unsafe extern "C" {
    fn mmap(addr: *mut c_void, len: usize, prot: i32, flags: i32, fd: i32, offset: i64) -> *mut c_void;
    fn munmap(addr: *mut c_void, len: usize) -> i32;
    fn madvise(addr: *mut c_void, len: usize, advice: i32) -> i32;
}
struct Image { ptr: *mut c_void, len: usize }
impl Image {
    fn open(path: &Path) -> Self {
        let file = std::fs::File::open(path).expect("open fixed public image");
        let len = file.metadata().unwrap().len() as usize;
        assert!(len > 0);
        let ptr = unsafe { mmap(std::ptr::null_mut(), len, 1, 2, file.as_raw_fd(), 0) };
        assert_ne!(ptr as usize, usize::MAX, "map fixed public image");
        Self { ptr, len }
    }
    fn bytes(&self) -> &[u8] { unsafe { std::slice::from_raw_parts(self.ptr as *const u8, self.len) } }
}
impl Drop for Image {
    fn drop(&mut self) { assert_eq!(unsafe { munmap(self.ptr, self.len) }, 0); }
}
fn populate_part(part: &[u8]) {
    // Same Linux population policy as the actual prover, with read-only fallback.
    if unsafe { madvise(part.as_ptr() as *mut c_void, part.len(), 22) } != 0 {
        let mut sink = 0u8;
        for offset in (0..part.len()).step_by(4096) {
            sink ^= unsafe { std::ptr::read_volatile(part.as_ptr().add(offset)) };
        }
        std::hint::black_box(sink);
    }
}
fn populate(bytes: &[u8]) {
    let ranges = labinius::par::ranges(bytes.len(), 8, 4096);
    labinius::par::pool().run(ranges.len(), |i| populate_part(&bytes[ranges[i].clone()]));
}
fn populate_reference(bytes: &[u8]) {
    let ranges = reference::par::ranges(bytes.len(), 8, 4096);
    reference::par::pool().run(ranges.len(), |i| populate_part(&bytes[ranges[i].clone()]));
}
fn median(v: &[f64]) -> f64 {
    let mut values = v.to_vec();
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}
fn quiet() {
    // Both independent library pools spin for 2ms before parking. Apply the same
    // quiet interval before every backend, outside the changed-cost interval.
    std::thread::sleep(Duration::from_millis(3));
}
fn run_current(words: &[B128], path: &Path) -> (Vec<[u32; 648]>, f64) {
    quiet();
    let started = Instant::now();
    let image = Image::open(path);
    populate(image.bytes());
    assert_eq!(image.len, 2 * keygen::BATCHES * std::mem::size_of::<hybrid::KeyBatch>());
    assert_eq!(image.ptr as usize % std::mem::align_of::<hybrid::KeyBatch>(), 0);
    // This is exactly the byte image emitted by current arranged_image. KeyBatch
    // contains only i16 arrays, has no invalid bit patterns, and remains borrowed
    // from the live mapping until the synchronous commit returns. No static cast.
    let keys = unsafe { std::slice::from_raw_parts(image.ptr as *const hybrid::KeyBatch, 2 * keygen::BATCHES) };
    let columns: Vec<_> = words.chunks_exact(parameters::COLUMN_LEN).collect();
    let rows = hybrid::commit_columns_rows_arranged(keys, &columns, 8, 32);
    drop(columns);
    drop(image);
    (rows, started.elapsed().as_secs_f64() * 1000.0)
}
fn run_reference(words: &[B128], path: &Path) -> (Vec<[u32; 648]>, f64) {
    quiet();
    let started = Instant::now();
    let image = Image::open(path);
    populate_reference(image.bytes());
    assert_eq!(image.len, KEY126_BYTES);
    assert_eq!(image.ptr as usize % std::mem::align_of::<u32>(), 0);
    let keys = unsafe { std::slice::from_raw_parts(image.ptr as *const u32, image.len / 4) };
    // Both types are repr(transparent) u128, Copy, alignment16, every bit pattern
    // valid. Read-only reborrow of the same input, never a key/batch type cast.
    assert_eq!(std::mem::size_of::<B128>(), std::mem::size_of::<reference::fields::scalar::B128>());
    assert_eq!(std::mem::align_of::<B128>(), std::mem::align_of::<reference::fields::scalar::B128>());
    let values = unsafe { std::slice::from_raw_parts(words.as_ptr() as *const reference::fields::scalar::B128, words.len()) };
    let columns: Vec<_> = values.chunks_exact(parameters::COLUMN_LEN).collect();
    let rows = reference::simd::commit::hybrid::commit_columns_rows_pairs(keys, &columns, 8);
    drop(columns);
    drop(image);
    (rows, started.elapsed().as_secs_f64() * 1000.0)
}
fn run_candidate(words: &[B128], path: &Path) -> (Vec<[u32; 648]>, f64) {
    quiet();
    let started = Instant::now();
    let image = Image::open(path);
    populate(image.bytes());
    let rows = r9::commit(words, image.bytes());
    drop(image);
    (rows, started.elapsed().as_secs_f64() * 1000.0)
}

pub fn evaluate(dir: &Path) -> bool {
    assert_eq!(Q, 38_712_288_256);
    assert_eq!(std::env::var("NUM_JOBS").as_deref(), Ok("1"), "native timing must run with build.jobs=1, without competing Cargo compilations");
    assert!(cfg!(target_endian = "little"));
    assert!(std::fs::read_to_string("/proc/cpuinfo").unwrap().contains("8488C"), "native opportunity gate requires8488C");
    assert_eq!(labinius::par::threads(), 8);
    assert_eq!(labinius::par::pool().width(), 8);
    assert_eq!(reference::par::pool().width(), 8);
    assert!(labinius::par::CorePlacement::for_sibling_pairs(8).is_some(), "requires four complete sibling pairs");
    let key = reference_keygen::build_key(Q);
    let tail = reference_keygen::build_tail_key(Q);
    let pairkeys = reference::simd::commit::hybrid::pair_keys(&key, &tail, 8);
    assert_eq!(pairkeys.len() * 4, KEY126_BYTES);
    let bytes = unsafe { std::slice::from_raw_parts(pairkeys.as_ptr() as *const u8, KEY126_BYTES) };
    std::fs::write(dir.join("reference126-key.bin"), bytes).unwrap();
    drop(pairkeys);
    drop(tail);
    drop(key);
    let mut times = [Vec::new(), Vec::new(), Vec::new()];
    let mut log = String::from("opportunity screen only; accepted_proofs=0; whole_prover_speedup_unverified=true\n");
    for trial in 0..TRIALS {
        let mut seed = [0u8; 32];
        std::fs::File::open("/dev/urandom").unwrap().read_exact(&mut seed).unwrap();
        let mut words = vec![B128::ZERO; parameters::COLUMNS * parameters::COLUMN_LEN];
        let input_bytes = unsafe { std::slice::from_raw_parts_mut(words.as_mut_ptr() as *mut u8, words.len() * 16) };
        let mut hash = blake3::Hasher::new_derive_key("labinius-r9/native-opportunity/fresh/v2");
        hash.update(&seed);
        hash.finalize_xof().fill(input_bytes);
        let input_hash = blake3::hash(input_bytes);
        populate(input_bytes);
        let mut outputs: [Option<Vec<[u32; 648]>>; 3] = [None, None, None];
        for offset in 0..3 {
            let backend = (trial + offset) % 3;
            let (rows, ms) = match backend {
                0 => run_current(&words, &dir.join("hybrid-key.bin")),
                1 => run_reference(&words, &dir.join("reference126-key.bin")),
                2 => run_candidate(&words, &dir.join("r9-key.bin")),
                _ => unreachable!(),
            };
            assert_eq!(rows.len(), 3 * parameters::COLUMNS);
            times[backend].push(ms);
            outputs[backend] = Some(rows);
        }
        let expected = outputs[0].as_ref().unwrap();
        for backend in 1..3 {
            let actual = outputs[backend].as_ref().unwrap();
            for row in 0..expected.len() {
                for coef in 0..648 {
                    assert_eq!(actual[row][coef], expected[row][coef], "backend{backend} trial{trial} row{row} coefficient{coef}");
                }
            }
        }
        let line = format!("trial={trial} input_hash={input_hash} order_start={} current8af_ms={:.6} reference126_ms={:.6} candidate_ms={:.6} all3072x648_equal=true\n", trial % 3, times[0][trial], times[1][trial], times[2][trial]);
        println!("cargo:warning={}", line.trim());
        log.push_str(&line);
    }
    let medians: [f64; 3] = std::array::from_fn(|i| median(&times[i]));
    let gains: [f64; 2] = std::array::from_fn(|i| 1.0 - medians[2] / medians[i]);
    let wins: [usize; 2] = std::array::from_fn(|i| (0..TRIALS).filter(|&trial| times[2][trial] < times[i][trial]).count());
    let admitted = (0..2).all(|i| gains[i] >= MIN_RELATIVE_GAIN && wins[i] >= MIN_WINS);
    let line = format!("current8af_median_ms={:.6} reference126_median_ms={:.6} candidate_median_ms={:.6} current8af_gain_percent={:.4} reference126_gain_percent={:.4} current8af_wins={} reference126_wins={} required_median_gain_percent={:.3} required_wins={MIN_WINS}/{TRIALS} admitted={admitted} accepted_proofs=0 whole_prover_speedup_unverified=true\n", medians[0], medians[1], medians[2], 100.0 * gains[0], 100.0 * gains[1], wins[0], wins[1], 100.0 * MIN_RELATIVE_GAIN);
    println!("cargo:warning={}", line.trim());
    log.push_str(&line);
    log.push_str("interval=mapping+key_population+permission_if_called+allocation+dispatch+complete_footer+workspace_drop+unmap; ordinary_full_inverse=unchanged_proof_encode; input=anonymous_prefaulted256MiB; quiet_before_each_backend_ms=3; output_vec_drop=outside_all_intervals; cargo_jobs=1; gate_uses_one_process_so_final_fresh_process_proof_test_required\n");
    std::fs::write(dir.join("r9-native-gate.txt"), log).unwrap();
    admitted
}
