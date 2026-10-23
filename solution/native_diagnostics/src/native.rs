use labinius::fields::scalar::B128;
use labinius::Transcript;
use std::io::Write;
use std::time::Instant;

const LOG_PACKED: usize = 24;
const LOG_VARS: usize = 31;
const COLUMNS: usize = 1024;
const COLUMN_LEN: usize = 1 << 14;

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
    let mut transcript = Transcript::new(b"labinius-fast/row-eqblock-diagnostic/v1");
    let p0 = transcript.sample_f162(b"row-point", 14);
    assert_eq!(1usize << p0.len(), COLUMN_LEN);
    println!("native-diag fixture=row-eqblock-v1 point_sampler=production row_variables={} output_fields={COLUMNS} prior_switch_workspace=absent timing_scope=whole-public-row", p0.len());
    println!("native-diag phase=setup elapsed_ns={} input_address_mod4096={} key_address_mod64={} key_bytes={KEY_LEN}", clock.elapsed().as_nanos(), values.as_ptr() as usize % 4096, KEY.0.as_ptr() as usize % 64);

    let mut baseline = || labinius::eval::row_evaluate_words_baseline(values, &p0, workers);
    let mut candidate = || labinius::eval::row_evaluate_words(values, &p0, workers);
    crate::compare::run("row", &mut baseline, &mut candidate);
    println!("native-diag phase=complete elapsed_ns={}", clock.elapsed().as_nanos());
}
