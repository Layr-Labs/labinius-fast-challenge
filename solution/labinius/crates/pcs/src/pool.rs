//! Eight worker threads, started once. `thread::scope` was creating a fresh set on every
//! sumcheck round, and that startup was larger than the round.
use std::sync::Once;

pub fn install() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(8)
            .build_global();
    });
}

pub const THREADS: usize = 8;
