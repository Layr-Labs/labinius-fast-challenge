use labinius::fields::{crossfield as cf, scalar::{B128, F162}};
use labinius::Transcript;
use std::io::Write;
use std::time::Instant;

#[path = "../../proof_generate/src/parameters.rs"]
mod parameters;
use parameters::{COLUMNS, COLUMN_LEN, COLUMN_LOG, LOG_PACKED, LOG_VARS, MODULUS};
const FIXED_SEED: [u8; 32] = [0x51; 32];

// Exact copies of the production allocator, Table, and prefault functions.
include!("production_memory.rs");

#[repr(C, align(64))]
struct Aligned64<T>(T);
const KEY_LEN: usize = include_bytes!(env!("LABINIUS_DIAGNOSTIC_PAIR_KEY")).len();
static KEY: Aligned64<[u8; KEY_LEN]> =
    Aligned64(*include_bytes!(env!("LABINIUS_DIAGNOSTIC_PAIR_KEY")));

fn write_fixed_input(path: &str) {
    let mut file = std::fs::File::create(path).expect("create fixed diagnostic input");
    let mut block = vec![0u8; 1 << 20];
    let mut state = 0x6b65_726e_656c_6469u64;
    for _ in 0..Table::BYTES / block.len() {
        for word in block.chunks_exact_mut(8) {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            word.copy_from_slice(&state.to_le_bytes());
        }
        file.write_all(&block).expect("write fixed diagnostic input");
    }
}

static PLANE_PREP_FAILED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

fn prepare_planes(condition: usize, words: usize) -> [cf::HugeBuf<u64>; 3] {
    const CHUNK: usize = 1 << 21;
    let sets = [32, 64][condition];
    let planes = cf::LeanSwitch::planes(words);
    let bound = (words / sets * 8).min(planes[0].len() * 8);
    let mut statuses = Vec::new();
    for plane in &planes {
        let (addr, _) = plane.region();
        for offset in (0..bound).step_by(CHUNK) {
            let status = unsafe {
                sys::madvise((addr + offset) as *mut _, CHUNK.min(bound - offset), sys::MADV_POPULATE_WRITE)
            };
            let errno = if status == 0 { 0 } else {
                PLANE_PREP_FAILED.store(true, std::sync::atomic::Ordering::Relaxed);
                std::io::Error::last_os_error().raw_os_error().unwrap_or(0)
            };
            statuses.push((status, errno));
        }
    }
    println!("native-diag phase=plane-prepare condition={condition} sets={sets} plane_bytes={} prefix_bytes_per_plane={bound} total_prefix_bytes={} chunk_bytes={CHUNK} return_errno={statuses:?}", planes[0].len() * 8, 3 * bound);
    planes
}

#[derive(PartialEq)]
struct PhaseOutput {
    lo: Vec<B128>,
    hi: Vec<B128>,
    partials: Vec<B128>,
    value: B128,
    batch: Vec<F162>,
    messages: Vec<[F162; 2]>,
    challenges: Vec<F162>,
    opened: F162,
    following_challenge: F162,
}

// Thin orchestration of the actual production APIs; no prepared point or partial data.
#[inline(never)]
fn phase<const G: usize>(
    values: &[B128],
    commitment: &[u8],
    workers: usize,
    planes: [cf::HugeBuf<u64>; 3],
) -> PhaseOutput {
    let mut t = Transcript::new(b"labinius-fast/v1");
    t.absorb_bytes(&FIXED_SEED);
    t.absorb_u64(LOG_VARS as u64);
    t.absorb_u64(COLUMN_LOG as u64);
    t.absorb_bytes(&MODULUS.to_le_bytes());
    t.absorb_bytes_parallel(commitment, workers);
    let lo = t.sample_b128(b"r-lo", 7);
    let hi = t.sample_b128(b"r-hi", LOG_PACKED);
    assert!(hi.iter().all(|&r| r != B128::ONE), "fixed phase point is nondegenerate");
    let (sets, eq) = cf::eq_expand_and_partials_factored_blocks::<G>(&hi, values, workers);
    let partials = cf::partials_from_sets(&sets, &hi);
    let value = partials.iter().zip(cf::eq_expand_b128(&lo))
        .fold(B128::ZERO, |a, (&v, e)| a + v * e);
    t.absorb_b128(&[value]);
    t.absorb_b128(&partials);
    let batch = cf::eq_expand_f162(&t.sample_f162(b"switch-batch", 7));
    let mut sw = cf::LeanSwitch::with_sets_factored(
        values, eq, &hi, &batch, workers, planes, sets.into(),
    ).expect("fixed phase uses the production LeanSwitch path");
    let mut messages = Vec::with_capacity(LOG_PACKED);
    let mut challenges = Vec::with_capacity(LOG_PACKED);
    for _ in 0..LOG_PACKED {
        let msg = if G == 32 { sw.msg_baseline() } else { sw.msg() };
        t.absorb_f162(&msg);
        let r = t.sample_f162(b"switch-round", 1)[0];
        messages.push(msg);
        challenges.push(r);
        sw.fold(r);
    }
    let opened = sw.final_eval();
    t.absorb_f162(&[opened]);
    let following_challenge = t.sample_f162(b"following-round", 1)[0];
    drop(sw);
    PhaseOutput { lo, hi, partials, value, batch, messages, challenges, opened, following_challenge }
}

pub fn run() {
    assert!(is_x86_feature_detected!("avx512f")
        && is_x86_feature_detected!("avx512bw")
        && is_x86_feature_detected!("avx512vl")
        && is_x86_feature_detected!("avx512vbmi")
        && is_x86_feature_detected!("avx512vbmi2")
        && is_x86_feature_detected!("avx512vnni")
        && is_x86_feature_detected!("gfni"), "native diagnostic needs the scored CPU features");
    let clock = Instant::now();
    let workers = labinius::par::threads();
    assert_eq!(workers, 8, "native diagnostic needs eight available CPUs");
    extern "C" { fn sched_getscheduler(pid: i32) -> i32; }
    let policy = unsafe { sched_getscheduler(0) };
    assert_eq!(policy, 0, "native diagnostic needs normal scheduling");
    let paired = labinius::par::CorePlacement::for_sibling_pairs(workers).is_some();
    println!("native-diag config workers={workers} sibling_pairs={paired} words={} columns={COLUMNS} column_words={COLUMN_LEN} scheduling_policy={policy} allocator=production-copy profile=release opt=3 lto=fat codegen_units=1", Table::WORDS);
    println!("native-diag config rustflags={:?} encoded_rustflags={:?}", option_env!("LABINIUS_DIAGNOSTIC_RUSTFLAGS"), option_env!("LABINIUS_DIAGNOSTIC_ENCODED_RUSTFLAGS"));

    let path = std::env::args().nth(1).expect("fixed diagnostic input path");
    write_fixed_input(&path);
    let table = Table::load(&path);
    assert!(matches!(&table, Table::Mapped { .. }), "native diagnostic input must be mapped");
    std::fs::remove_file(&path).expect("remove fixed diagnostic input name");
    let values = table.words();
    let _ = labinius::par::pool();
    populate_words(values, &KEY.0, workers);
    let key_words = 4 * (4096 / 2) * 648;
    assert_eq!(KEY_LEN, 4 * key_words, "exact production pair key image length");
    assert_eq!(KEY.0.as_ptr() as usize % 64, 0, "production key alignment");
    assert!(cfg!(target_endian = "little"));
    assert_eq!(values.len(), COLUMNS * COLUMN_LEN);
    let coefficient_bytes = (128 - (MODULUS - 1).leading_zeros()).div_ceil(8) as usize;
    let commitment = vec![0u8; COLUMNS * 648 * coefficient_bytes];
    println!("native-diag fixture=partials-switch-v1 seed_byte=81 commitment=zero-canonical commitment_bytes={} point_setup=inside_timer rounds={LOG_PACKED} baseline_sets=32 candidate_sets=64 output_guard=all-public-phase-values", commitment.len());
    println!("native-diag timing=whole-partials-switch planes=fresh-before-clock switch_drop=inside-clock caller_rows_drop=absent plane_prefix_prefault=synchronous-before-clock output_file_tail=absent no_cost_subtraction=true");
    println!("native-diag phase=setup elapsed_ns={} input_address_mod4096={} key_address_mod64={} key_bytes={KEY_LEN}", clock.elapsed().as_nanos(), values.as_ptr() as usize % 4096, KEY.0.as_ptr() as usize % 64);

    let mut prepare = |condition| prepare_planes(condition, values.len());
    let mut baseline = |planes| phase::<32>(values, &commitment, workers, planes);
    let mut candidate = |planes| phase::<64>(values, &commitment, workers, planes);
    crate::compare::run("partials-switch", &mut prepare, &mut baseline, &mut candidate);
    let prep_ok = !PLANE_PREP_FAILED.load(std::sync::atomic::Ordering::Relaxed);
    println!("native-diag phase=calibration plane_prefix_prefault_ok={prep_ok} time_subtraction=false");
    assert!(prep_ok, "plane prefix preparation failed; calibration and retention are blocked");
    println!("native-diag phase=complete elapsed_ns={}", clock.elapsed().as_nanos());
}
