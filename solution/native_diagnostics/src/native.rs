use labinius::fields::{crossfield as cf, scalar::B128};
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
    let planes = cf::LeanSwitch::planes(1 << LOG_PACKED);
    populate_words(values, &KEY.0, workers);
    let key_words = 4 * (4096 / 2) * 648;
    assert_eq!(KEY_LEN, 4 * key_words, "exact production pair key image length");
    assert_eq!(KEY.0.as_ptr() as usize % 64, 0, "production key alignment");
    assert!(cfg!(target_endian = "little"));
    let keys = unsafe { std::slice::from_raw_parts(KEY.0.as_ptr().cast::<u32>(), key_words) };
    let columns: Vec<&[B128]> = values.chunks_exact(COLUMN_LEN).collect();
    assert_eq!(columns.len(), COLUMNS);
    println!("native-diag phase=setup elapsed_ns={} input_address_mod4096={} key_address_mod64={} key_bytes={KEY_LEN}", clock.elapsed().as_nanos(), values.as_ptr() as usize % 4096, KEY.0.as_ptr() as usize % 64);

    crate::pair_checks::run(keys, &columns);
    let mut baseline = || labinius::simd::commit::hybrid::commit_columns_rows_pairs(keys, &columns, workers);
    let mut omega = || crate::commit_adapter::commit_columns_rows_pairs::<1, true>(keys, &columns, workers);
    let mut lut6 = || crate::commit_adapter::commit_columns_rows_pairs::<2, true>(keys, &columns, workers);
    let mut combined = || crate::commit_adapter::commit_columns_rows_pairs::<3, true>(keys, &columns, workers);
    crate::compare::run("commit", &mut [
        ("baseline", &mut baseline), ("omega-only", &mut omega),
        ("lut6-only", &mut lut6), ("combined", &mut combined),
    ]);
    std::hint::black_box(&planes);
    println!("native-diag phase=complete elapsed_ns={}", clock.elapsed().as_nanos());
}
