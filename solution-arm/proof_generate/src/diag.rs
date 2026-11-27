//! Per-thread commit timers (`cntvct_el0`), printed to stderr when a commit thread finishes: the
//! time of the sweeps' transforms, of the sink and of the chunk ends. They do not touch the proof.
#![allow(dead_code)]
use super::*;
use std::cell::Cell;

#[inline(always)]
pub fn tick() -> u64 {
    let t: u64;
    unsafe { core::arch::asm!("mrs {}, cntvct_el0", out(reg) t, options(nomem, nostack, preserves_flags)) };
    t
}

fn freq() -> f64 {
    let f: u64;
    unsafe { core::arch::asm!("mrs {}, cntfrq_el0", out(reg) f, options(nomem, nostack, preserves_flags)) };
    f as f64
}

thread_local! { static ACC: Cell<[u64; 3]> = const { Cell::new([0; 3]) }; }

#[inline(always)]
pub fn add(i: usize, d: u64) {
    ACC.with(|a| { let mut v = a.get(); v[i] += d; a.set(v); });
}

pub fn report(w: usize) {
    let v = ACC.with(|a| a.get());
    let ms = freq() / 1000.0;
    eprintln!("diag thread={w} transforms_ms={:.2} sink_ms={:.2} chunk_ms={:.2}", v[0] as f64 / ms, v[1] as f64 / ms, v[2] as f64 / ms);
}
