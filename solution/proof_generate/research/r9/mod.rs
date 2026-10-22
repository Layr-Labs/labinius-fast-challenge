//! R9 coupled256 subgroup commitment runtime candidate, not a proven speedup.
//! Shared by the bounded native comparison and conditional prover integration.
//! Uses exact fresh1024-column input; publickey image only is precomputed.
use labinius::fields::scalar::B128;
use std::alloc::{alloc, dealloc, handle_alloc_error, Layout};
use std::ptr::NonNull;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Barrier,
};
use std::time::{Duration, Instant};
mod footer;
pub mod keys;
#[path = "producer/producer.rs"]
mod producer;
const COLS: usize = 1024;
const WORDS_PER_COL: usize = 16384;
const COHORTS: usize = COLS / 16;
const C_BYTES: usize = 147456;
const KEY_SLAB: usize = 24576;
const KEY_IMAGE_BYTES: usize = 75497472;
struct Aligned {
    ptr: NonNull<u8>,
    layout: Layout,
}
unsafe impl Send for Aligned {}
unsafe impl Sync for Aligned {}
impl Aligned {
    fn new(bytes: usize) -> Self {
        let layout = Layout::from_size_align(bytes, 64).unwrap();
        let ptr =
            NonNull::new(unsafe { alloc(layout) }).unwrap_or_else(|| handle_alloc_error(layout));
        Self { ptr, layout }
    }
    fn ptr(&self) -> *mut u8 {
        self.ptr.as_ptr()
    }
}
impl Drop for Aligned {
    fn drop(&mut self) {
        unsafe { dealloc(self.ptr(), self.layout) }
    }
}

#[repr(align(128))]
struct Slot {
    state: AtomicUsize,
    data: Aligned,
}
struct Pair {
    prefix: Aligned,
    slots: [Slot; 2],
}
impl Pair {
    fn new() -> Self {
        Self {
            prefix: Aligned::new(producer::WORD_BYTES),
            slots: std::array::from_fn(|i| Slot {
                state: AtomicUsize::new(i),
                data: Aligned::new(producer::SLOT_BYTES),
            }),
        }
    }
}

core::arch::global_asm!(
    r#"
.text
.globl r9_subgroup_accumulate_27
r9_subgroup_accumulate_27:
    mov r10d, 64
    mov r11d, 256
    tileloadd tmm0, [rdx + r10]
    tileloadd tmm1, [rdx + r10 + 1024]
    tileloadd tmm2, [rdx + r10 + 2048]
    tileloadd tmm3, [rdx + r10 + 3072]
    tileloadd tmm4, [rdx + r10 + 4096]
    tileloadd tmm5, [rdx + r10 + 5120]
    mov r8d, 27
2:
    tileloadd tmm6, [rdi + r10]
    tileloadd tmm7, [rsi + r11]
    tdpbssd tmm0, tmm6, tmm7
    tileloadd tmm7, [rsi + r11 + 4096]
    tdpbssd tmm1, tmm6, tmm7
    tileloadd tmm6, [rdi + r10 + 1024]
    tileloadd tmm7, [rsi + r11 + 8192]
    tdpbssd tmm2, tmm6, tmm7
    tileloadd tmm7, [rsi + r11 + 12288]
    tdpbssd tmm3, tmm6, tmm7
    tileloadd tmm6, [rdi + r10 + 2048]
    tileloadd tmm7, [rsi + r11 + 16384]
    tdpbssd tmm4, tmm6, tmm7
    tileloadd tmm7, [rsi + r11 + 20480]
    tdpbssd tmm5, tmm6, tmm7
    add rdi, 3072
    sub rsi, 4
    dec r8d
    jne 2b
    tilestored [rdx + r10], tmm0
    tilestored [rdx + r10 + 1024], tmm1
    tilestored [rdx + r10 + 2048], tmm2
    tilestored [rdx + r10 + 3072], tmm3
    tilestored [rdx + r10 + 4096], tmm4
    tilestored [rdx + r10 + 5120], tmm5
    ret
"#
);

extern "C" {
    fn r9_subgroup_accumulate_27(a: *const i8, b: *const i8, c: *mut i32);
    fn syscall(number: i64, ...) -> i64;
}

#[repr(C, align(64))]
struct TileConfig([u8; 64]);
unsafe fn configure_tiles() {
    let mut cfg = TileConfig([0; 64]);
    cfg.0[0] = 1;
    for t in 0..8 {
        cfg.0[16 + 2 * t] = 64;
        cfg.0[48 + t] = 16;
    }
    core::arch::asm!("ldtilecfg [{cfg}]", cfg = in(reg) cfg.0.as_ptr(), options(nostack, readonly));
}
fn enable_amx() -> Result<(), String> {
    let cpuid = unsafe { core::arch::x86_64::__cpuid_count(7, 0) };
    if cpuid.edx & (1 << 24) == 0 || cpuid.edx & (1 << 25) == 0 {
        return Err("AMX-TILE/AMX-INT8 CPUID support is missing".into());
    }
    // Linux x86-64 arch_prctl(ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA).
    if unsafe { syscall(158, 0x1023i64, 18i64) } != 0 {
        return Err(format!(
            "AMX permission request failed: {}",
            std::io::Error::last_os_error()
        ));
    }
    Ok(())
}
fn wait_state(slot: &Slot, wanted: usize) {
    let start = Instant::now();
    let mut spins = 0usize;
    while slot.state.load(Ordering::Acquire) != wanted {
        std::hint::spin_loop();
        spins += 1;
        if spins & ((1 << 20) - 1) == 0 && start.elapsed() > Duration::from_secs(30) {
            panic!("two-slot pipeline did not make progress for30s");
        }
    }
}

/// Compute fullysplit ordinaryNTT rows and exact auxiliary coefficientmod1024.
///
/// Called only from the top-level fullprover or serial native comparison. This
/// requires an idle width8 pool and four complete SMT sibling pairs. The pool's
/// try_run_concurrent rejects nested/busy/singlethread fallback WITHOUT executing
/// any closure, because serial execution of a barrier task would deadlock.
/// All instance allocations, prefix/pack, AMX, footer, copy and teardown are real
/// work here. Public image/table prefault policy belongs to the caller's timer.
/// This function does not validate a proof or make a performance claim.
pub fn commit(words: &[B128], image: &[u8]) -> Vec<[u32; 648]> {
    assert!(
        cfg!(all(target_os = "linux", target_arch = "x86_64")),
        "R9 requires nativeLinuxx86_64"
    );
    assert_eq!(
        words.len(),
        COLS * WORDS_PER_COL,
        "complete fresh witness table required"
    );
    assert_eq!(
        image.len(),
        KEY_IMAGE_BYTES,
        "exact coupledE key image required"
    );
    assert_eq!(
        (image.as_ptr() as usize) & 63,
        0,
        "public key image must be64B aligned"
    );
    assert!(
        std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512bw")
            && std::is_x86_feature_detected!("avx512vl")
            && std::is_x86_feature_detected!("avx512vbmi")
            && std::is_x86_feature_detected!("avx512vbmi2")
            && std::is_x86_feature_detected!("avx512vnni")
            && std::is_x86_feature_detected!("gfni"),
        "R9 CPU feature requirements not met"
    );
    // Permission is per process and cleared across exec. Build-time permission
    // does not enable the runtime. First tiledata use/faults remain actual work.
    enable_amx().unwrap_or_else(|e| panic!("R9 AMX permission: {e}"));
    producer::prepare();
    let pool = labinius::par::pool();
    assert_eq!(
        pool.width(),
        8,
        "R9 requires exactly8 simultaneous pool runners"
    );
    let placement = labinius::par::CorePlacement::for_sibling_pairs(8)
        .expect("R9 requires four complete physical SMT sibling pairs");
    let pairs: Vec<Pair> = (0..4).map(|_| Pair::new()).collect();
    // Standard Vec allocation has the correct allocator layout for its type.
    // Never transfer an align64 Aligned buffer via Vec::from_raw_parts.
    let mut output: Vec<[u32; 648]> = Vec::with_capacity(COLS * 3);
    let out = output.as_mut_ptr() as usize;
    let input = words.as_ptr() as usize;
    let key = image.as_ptr() as usize;
    let ready = Barrier::new(8);
    let dispatched = pool.try_run_concurrent(8, |job| {
        // Never unwind through a pool dispatch while other tasks still borrow
        // its job/buffers. Build scripts can use unwinding even in release.
        let ran = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _affinity = placement.pin(job).expect("R9 pool affinity pin failed");
            let pair_id = job / 2;
            let p = &pairs[pair_id];
            ready.wait();
            let mut seq = 0usize;
            if job % 2 == 0 {
                for cohort in (pair_id..COHORTS).step_by(4) {
                    for quarter in 0..4 {
                        for br in 0..64 {
                            unsafe {
                                producer::fill_prefix(
                                    input as *const B128,
                                    16 * cohort,
                                    quarter,
                                    br,
                                    p.prefix.ptr() as *mut i16,
                                );
                            }
                            for subgroup in 0..3 {
                                let slot = &p.slots[seq % 2];
                                wait_state(slot, seq);
                                unsafe {
                                    producer::pack_subgroup(
                                        p.prefix.ptr() as *const i16,
                                        subgroup,
                                        slot.data.ptr() as *mut i8,
                                    );
                                }
                                slot.state.store(seq + 1, Ordering::Release);
                                seq += 1;
                            }
                        }
                    }
                }
            } else {
                let c = Aligned::new(C_BYTES);
                unsafe {
                    configure_tiles();
                }
                let _tiles = ActiveTiles;
                for cohort in (pair_id..COHORTS).step_by(4) {
                    let mut foot = footer::Footer::new();
                    for quarter in 0..4 {
                        unsafe {
                            std::ptr::write_bytes(c.ptr(), 0, C_BYTES);
                        }
                        for br in 0..64 {
                            for subgroup in 0..3 {
                                let slot = &p.slots[seq % 2];
                                wait_state(slot, seq + 1);
                                for q in 0..2 {
                                    for local_f in 0..2 {
                                        for n in 0..2 {
                                            let factor = 2 * subgroup + local_f;
                                            unsafe {
                                                let a = slot
                                                    .data
                                                    .ptr()
                                                    .add(q * 165888 + local_f * 82944);
                                                let b = (key as *const u8).add(
                                                    (((br * 4 + quarter) * 2 + q) * 6 + factor)
                                                        * KEY_SLAB
                                                        + 108
                                                        + n * 64,
                                                );
                                                let cp =
                                                    c.ptr().add(((q * 6 + factor) * 2 + n) * 6144);
                                                r9_subgroup_accumulate_27(
                                                    a as *const i8,
                                                    b as *const i8,
                                                    cp as *mut i32,
                                                );
                                            }
                                        }
                                    }
                                }
                                // Only release after all reads of this W slot end.
                                slot.state.store(seq + 2, Ordering::Release);
                                seq += 1;
                            }
                            if (br + 1) % 4 == 0 {
                                unsafe {
                                    foot.flush_aux(quarter, br / 4, c.ptr() as *mut i32);
                                }
                            }
                        }
                        unsafe {
                            foot.finish_quarter(quarter, c.ptr() as *mut i32);
                        }
                    }
                    let rows = unsafe { foot.finish() };
                    assert_eq!(rows.len(), 48);
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            rows.as_ptr(),
                            (out as *mut [u32; 648]).add(48 * cohort),
                            48,
                        );
                    }
                }
                // Tile release before restoring this pool runner's prior affinity.
                drop(_tiles);
                drop(c);
            }
        }));
        if ran.is_err() {
            eprintln!("R9 worker failed; aborting without unwinding an active concurrent dispatch");
            std::process::abort();
        }
    });
    assert!(
        dispatched,
        "R9 requires idle exclusive pool dispatch; serial fallback prohibited"
    );
    // try_run_concurrent returns only after all eight tasks completed. The four
    // consumers partition64cohorts and initialized every returned row exactlyonce.
    unsafe {
        output.set_len(COLS * 3);
    }
    drop(pairs);
    output
}

struct ActiveTiles;
impl Drop for ActiveTiles {
    fn drop(&mut self) {
        unsafe {
            core::arch::asm!("tilerelease", options(nostack));
        }
    }
}
