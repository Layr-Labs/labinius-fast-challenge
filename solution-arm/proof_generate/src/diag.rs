//! DIAGNOSTIC instrumentation, printed to stderr; it does not touch the proof. Per-thread time of
//! the commit's transforms, sink and chunk ends, and micro-benchmarks of the record's SVE `sdot`
//! slot sink against the byte matrix-multiply one, run once before the commit.
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

/// Nanoseconds per instruction of 24 independent chains of `$ins` (operands v0, v1 / z0, z1).
macro_rules! chains {
    ($n:expr, $($ins:literal),*) => {{
        let t = tick();
        unsafe {
            core::arch::asm!(
                "movi v0.16b, #1", "movi v1.16b, #2",
                "2:",
                $($ins,)*
                "subs {n}, {n}, #1",
                "b.ne 2b",
                n = inout(reg) $n => _,
                out("v0") _, out("v1") _,
                out("v8") _, out("v9") _, out("v10") _, out("v11") _, out("v12") _, out("v13") _, out("v14") _, out("v15") _,
                out("v16") _, out("v17") _, out("v18") _, out("v19") _, out("v20") _, out("v21") _, out("v22") _, out("v23") _,
                out("v24") _, out("v25") _, out("v26") _, out("v27") _, out("v28") _, out("v29") _, out("v30") _, out("v31") _,
                options(nostack)
            );
        }
        (tick() - t) as f64 * 1e9 / freq() / ($n as f64 * 24.0)
    }};
}

#[cfg(target_feature = "sve")]
macro_rules! old_sink_body {
    ($acc:expr, $z:expr, $k:expr, $groups:expr, [$($rows13:literal),*], [$($rows46:literal),*]) => {
        core::arch::asm!(
        "ldp q16, q17, [{a}, #0]",
        "ldp q18, q19, [{a}, #32]",
        $($rows13,)*
        "mov {zp}, {z}",
        "mov {kp}, {k}",
        "mov {n}, {g}",
        "2:",
        "ldp q0, q1, [{kp}]",
        "ldp q2, q3, [{kp}, #32]",
        "ldr q4, [{zp}]",
        "ldr q5, [{zp}, #{xs1}]",
        "ldr q6, [{zp}, #{xs2}]",
        "ldr q7, [{zp}, #{xs3}]",
        "add {zp}, {zp}, {xs4}",
        "sdot z16.d, z4.h, z0.h[0]", "sdot z17.d, z5.h, z0.h[0]", "sdot z18.d, z6.h, z0.h[0]", "sdot z19.d, z7.h, z0.h[0]",
        "sdot z20.d, z4.h, z1.h[0]", "sdot z21.d, z5.h, z1.h[0]", "sdot z22.d, z6.h, z1.h[0]", "sdot z23.d, z7.h, z1.h[0]",
        "sdot z24.d, z4.h, z2.h[0]", "sdot z25.d, z5.h, z2.h[0]", "sdot z26.d, z6.h, z2.h[0]", "sdot z27.d, z7.h, z2.h[0]",
        "sdot z28.d, z4.h, z3.h[0]", "sdot z29.d, z5.h, z3.h[0]", "sdot z30.d, z6.h, z3.h[0]", "sdot z31.d, z7.h, z3.h[0]",
        "ldr q4, [{zp}]",
        "ldr q5, [{zp}, #{xs1}]",
        "ldr q6, [{zp}, #{xs2}]",
        "ldr q7, [{zp}, #{xs3}]",
        "add {zp}, {zp}, {xs4}",
        "add {kp}, {kp}, {ks}",
        "sdot z16.d, z4.h, z0.h[1]", "sdot z17.d, z5.h, z0.h[1]", "sdot z18.d, z6.h, z0.h[1]", "sdot z19.d, z7.h, z0.h[1]",
        "sdot z20.d, z4.h, z1.h[1]", "sdot z21.d, z5.h, z1.h[1]", "sdot z22.d, z6.h, z1.h[1]", "sdot z23.d, z7.h, z1.h[1]",
        "sdot z24.d, z4.h, z2.h[1]", "sdot z25.d, z5.h, z2.h[1]", "sdot z26.d, z6.h, z2.h[1]", "sdot z27.d, z7.h, z2.h[1]",
        "sdot z28.d, z4.h, z3.h[1]", "sdot z29.d, z5.h, z3.h[1]", "sdot z30.d, z6.h, z3.h[1]", "sdot z31.d, z7.h, z3.h[1]",
        "subs {n}, {n}, #1",
        "b.ne 2b",
        "stp q16, q17, [{a}, #0]",
        "stp q18, q19, [{a}, #32]",
        "stp q20, q21, [{a}, #64]",
        "stp q22, q23, [{a}, #96]",
        "stp q24, q25, [{a}, #128]",
        "stp q26, q27, [{a}, #160]",
        "stp q28, q29, [{a}, #192]",
        "stp q30, q31, [{a}, #224]",
        $($rows46,)*
        "mov {zp}, {z}",
        "mov {kp}, {k}",
        "mov {n}, {g}",
        "3:",
        "ldp q0, q1, [{kp}, #64]",
        "ldr q2, [{kp}, #96]",
        "ldr q4, [{zp}]",
        "ldr q5, [{zp}, #{xs1}]",
        "ldr q6, [{zp}, #{xs2}]",
        "ldr q7, [{zp}, #{xs3}]",
        "add {zp}, {zp}, {xs4}",
        "sdot z16.d, z4.h, z0.h[0]", "sdot z17.d, z5.h, z0.h[0]", "sdot z18.d, z6.h, z0.h[0]", "sdot z19.d, z7.h, z0.h[0]",
        "sdot z20.d, z4.h, z1.h[0]", "sdot z21.d, z5.h, z1.h[0]", "sdot z22.d, z6.h, z1.h[0]", "sdot z23.d, z7.h, z1.h[0]",
        "sdot z24.d, z4.h, z2.h[0]", "sdot z25.d, z5.h, z2.h[0]", "sdot z26.d, z6.h, z2.h[0]", "sdot z27.d, z7.h, z2.h[0]",
        "ldr q4, [{zp}]",
        "ldr q5, [{zp}, #{xs1}]",
        "ldr q6, [{zp}, #{xs2}]",
        "ldr q7, [{zp}, #{xs3}]",
        "add {zp}, {zp}, {xs4}",
        "add {kp}, {kp}, {ks}",
        "sdot z16.d, z4.h, z0.h[1]", "sdot z17.d, z5.h, z0.h[1]", "sdot z18.d, z6.h, z0.h[1]", "sdot z19.d, z7.h, z0.h[1]",
        "sdot z20.d, z4.h, z1.h[1]", "sdot z21.d, z5.h, z1.h[1]", "sdot z22.d, z6.h, z1.h[1]", "sdot z23.d, z7.h, z1.h[1]",
        "sdot z24.d, z4.h, z2.h[1]", "sdot z25.d, z5.h, z2.h[1]", "sdot z26.d, z6.h, z2.h[1]", "sdot z27.d, z7.h, z2.h[1]",
        "subs {n}, {n}, #1",
        "b.ne 3b",
        "stp q16, q17, [{a}, #256]",
        "stp q18, q19, [{a}, #288]",
        "stp q20, q21, [{a}, #320]",
        "stp q22, q23, [{a}, #352]",
        "stp q24, q25, [{a}, #384]",
        "stp q26, q27, [{a}, #416]",
            a = in(reg) $acc,
            z = in(reg) $z,
            k = in(reg) $k,
            g = in(reg) $groups,
            xs4 = in(reg) 4 * XS,
            ks = in(reg) KS,
            xs1 = const XS,
            xs2 = const 2 * XS,
            xs3 = const 3 * XS,
            zp = out(reg) _,
            kp = out(reg) _,
            n = out(reg) _,
            out("v0") _, out("v1") _, out("v2") _, out("v3") _,
            out("v4") _, out("v5") _, out("v6") _, out("v7") _,
            out("v16") _, out("v17") _, out("v18") _, out("v19") _,
            out("v20") _, out("v21") _, out("v22") _, out("v23") _,
            out("v24") _, out("v25") _, out("v26") _, out("v27") _,
            out("v28") _, out("v29") _, out("v30") _, out("v31") _,
            options(nostack)
        )
    };
}


/// The record's SVE `sdot` slot sink (submission 09c9d888), verbatim, for timing only.
#[cfg(target_feature = "sve")]
#[inline(never)]
unsafe fn old_slot_sink(z: *const int16x8_t, k: *const i16, acc: *mut int64x2_t, groups: usize) {
    const XS: usize = N * 16;
    const KS: usize = N * ROWS * LOCK * 2;
    old_sink_body!(acc, z, k, groups, ["ldp q20, q21, [{a}, #64]", "ldp q22, q23, [{a}, #96]", "ldp q24, q25, [{a}, #128]", "ldp q26, q27, [{a}, #160]", "ldp q28, q29, [{a}, #192]", "ldp q30, q31, [{a}, #224]"], ["ldp q16, q17, [{a}, #256]", "ldp q18, q19, [{a}, #288]", "ldp q20, q21, [{a}, #320]", "ldp q22, q23, [{a}, #352]", "ldp q24, q25, [{a}, #384]", "ldp q26, q27, [{a}, #416]"]);
}

#[inline(never)]
unsafe fn new_slot_sink(z: *const int16x8_t, k: *const i8, acc: *mut int32x4_t, groups: usize) {
    #[cfg(target_feature = "i8mm")]
    slot_sink::<false>(z, k, acc, groups);
    #[cfg(not(target_feature = "i8mm"))]
    slot_sink_scalar::<false>(z, k, acc, groups);
}

/// One thread's suite: ns per (slot, tile) of each sink with its data in L1 (8 slots), over all
/// slots with one batch's accumulators, and over all slots alternating two batches'
/// accumulators with the sweep buffer rewritten between sinks (the paired commit's footprint).
fn suite() -> String {
    let mut out = String::new();
    let n = 40_000usize;
    let usmmla = chains!(n, "usmmla v8.4s, v0.16b, v1.16b", "usmmla v9.4s, v0.16b, v1.16b", "usmmla v10.4s, v0.16b, v1.16b", "usmmla v11.4s, v0.16b, v1.16b",
        "usmmla v12.4s, v0.16b, v1.16b", "usmmla v13.4s, v0.16b, v1.16b", "usmmla v14.4s, v0.16b, v1.16b", "usmmla v15.4s, v0.16b, v1.16b",
        "usmmla v16.4s, v0.16b, v1.16b", "usmmla v17.4s, v0.16b, v1.16b", "usmmla v18.4s, v0.16b, v1.16b", "usmmla v19.4s, v0.16b, v1.16b",
        "usmmla v20.4s, v0.16b, v1.16b", "usmmla v21.4s, v0.16b, v1.16b", "usmmla v22.4s, v0.16b, v1.16b", "usmmla v23.4s, v0.16b, v1.16b",
        "usmmla v24.4s, v0.16b, v1.16b", "usmmla v25.4s, v0.16b, v1.16b", "usmmla v26.4s, v0.16b, v1.16b", "usmmla v27.4s, v0.16b, v1.16b",
        "usmmla v28.4s, v0.16b, v1.16b", "usmmla v29.4s, v0.16b, v1.16b", "usmmla v30.4s, v0.16b, v1.16b", "usmmla v31.4s, v0.16b, v1.16b");
    let mul = chains!(n, "mul v8.8h, v0.8h, v1.8h", "mul v9.8h, v0.8h, v1.8h", "mul v10.8h, v0.8h, v1.8h", "mul v11.8h, v0.8h, v1.8h",
        "mul v12.8h, v0.8h, v1.8h", "mul v13.8h, v0.8h, v1.8h", "mul v14.8h, v0.8h, v1.8h", "mul v15.8h, v0.8h, v1.8h",
        "mul v16.8h, v0.8h, v1.8h", "mul v17.8h, v0.8h, v1.8h", "mul v18.8h, v0.8h, v1.8h", "mul v19.8h, v0.8h, v1.8h",
        "mul v20.8h, v0.8h, v1.8h", "mul v21.8h, v0.8h, v1.8h", "mul v22.8h, v0.8h, v1.8h", "mul v23.8h, v0.8h, v1.8h",
        "mul v24.8h, v0.8h, v1.8h", "mul v25.8h, v0.8h, v1.8h", "mul v26.8h, v0.8h, v1.8h", "mul v27.8h, v0.8h, v1.8h",
        "mul v28.8h, v0.8h, v1.8h", "mul v29.8h, v0.8h, v1.8h", "mul v30.8h, v0.8h, v1.8h", "mul v31.8h, v0.8h, v1.8h");
    let add = chains!(n, "add v8.8h, v8.8h, v1.8h", "add v9.8h, v9.8h, v1.8h", "add v10.8h, v10.8h, v1.8h", "add v11.8h, v11.8h, v1.8h",
        "add v12.8h, v12.8h, v1.8h", "add v13.8h, v13.8h, v1.8h", "add v14.8h, v14.8h, v1.8h", "add v15.8h, v15.8h, v1.8h",
        "add v16.8h, v16.8h, v1.8h", "add v17.8h, v17.8h, v1.8h", "add v18.8h, v18.8h, v1.8h", "add v19.8h, v19.8h, v1.8h",
        "add v20.8h, v20.8h, v1.8h", "add v21.8h, v21.8h, v1.8h", "add v22.8h, v22.8h, v1.8h", "add v23.8h, v23.8h, v1.8h",
        "add v24.8h, v24.8h, v1.8h", "add v25.8h, v25.8h, v1.8h", "add v26.8h, v26.8h, v1.8h", "add v27.8h, v27.8h, v1.8h",
        "add v28.8h, v28.8h, v1.8h", "add v29.8h, v29.8h, v1.8h", "add v30.8h, v30.8h, v1.8h", "add v31.8h, v31.8h, v1.8h");
    out += &format!("usmmla_ns={usmmla:.4} mul8h_ns={mul:.4} add8h_ns={add:.4}");
    #[cfg(target_feature = "sve")]
    {
        let sdot = chains!(n, "sdot z8.d, z0.h, z1.h[0]", "sdot z9.d, z0.h, z1.h[0]", "sdot z10.d, z0.h, z1.h[0]", "sdot z11.d, z0.h, z1.h[0]",
            "sdot z12.d, z0.h, z1.h[0]", "sdot z13.d, z0.h, z1.h[0]", "sdot z14.d, z0.h, z1.h[0]", "sdot z15.d, z0.h, z1.h[0]",
            "sdot z16.d, z0.h, z1.h[0]", "sdot z17.d, z0.h, z1.h[0]", "sdot z18.d, z0.h, z1.h[0]", "sdot z19.d, z0.h, z1.h[0]",
            "sdot z20.d, z0.h, z1.h[0]", "sdot z21.d, z0.h, z1.h[0]", "sdot z22.d, z0.h, z1.h[0]", "sdot z23.d, z0.h, z1.h[0]",
            "sdot z24.d, z0.h, z1.h[0]", "sdot z25.d, z0.h, z1.h[0]", "sdot z26.d, z0.h, z1.h[0]", "sdot z27.d, z0.h, z1.h[0]",
            "sdot z28.d, z0.h, z1.h[0]", "sdot z29.d, z0.h, z1.h[0]", "sdot z30.d, z0.h, z1.h[0]", "sdot z31.d, z0.h, z1.h[0]");
        out += &format!(" sdot_hd_ns={sdot:.4}");
    }
    let ns = |t: u64, units: usize| t as f64 * 1e9 / freq() / units as f64;
    unsafe {
        let mut xs = vec![vdupq_n_s16(3); SWEEP * N];
        let xw = xs.as_mut_ptr();
        let z = xw as *const int16x8_t;
        let keys = vec![5i8; SINK * N * KT];
        let mut mm = [vec![vdupq_n_s32(0); MM * N], vec![vdupq_n_s32(0); MM * N]];
        let reps = 200;
        let t = tick();
        for _ in 0..reps { for s in 0..8 { new_slot_sink(z.add(s), keys.as_ptr().add(s * KT), mm[0].as_mut_ptr().add(mm_at(s, 0, 0, 0)), SINK); } }
        out += &format!(" new_l1_ns={:.2}", ns(tick() - t, reps * 8 * SINK));
        let calls = 12;
        let t = tick();
        for _ in 0..calls { for s in 0..N { new_slot_sink(z.add(s), keys.as_ptr().add(s * KT), mm[0].as_mut_ptr().add(mm_at(s, 0, 0, 0)), SINK); } }
        out += &format!(" new_single_ns={:.2}", ns(tick() - t, calls * N * SINK));
        let mut spent = 0;
        for c in 0..calls {
            for i in 0..SWEEP * N { *xw.add(i) = vdupq_n_s16(c as i16); }
            let t = tick();
            for s in 0..N { new_slot_sink(z.add(s), keys.as_ptr().add(s * KT), mm[c % 2].as_mut_ptr().add(mm_at(s, 0, 0, 0)), SINK); }
            spent += tick() - t;
        }
        out += &format!(" new_paired_ns={:.2}", ns(spent, calls * N * SINK));
        let mut spent = 0;
        for c in 0..calls {
            for i in 0..SWEEP * N / 2 { *xw.add(i) = vdupq_n_s16(c as i16); }
            let t = tick();
            for s in 0..N { new_slot_sink(z.add(s), keys.as_ptr().add(s * KT), mm[c % 2].as_mut_ptr().add(mm_at(s, 0, 0, 0)), SINK / 2); }
            spent += tick() - t;
        }
        out += &format!(" new_paired4_ns={:.2}", ns(spent, calls * N * SINK / 2));
        #[cfg(target_feature = "sve")]
        {
            let okeys = vec![5i16; SINK * N * ROWS * LOCK];
            let mut acc = [vec![vdupq_n_s64(0); ROWS * 4 * N], vec![vdupq_n_s64(0); ROWS * 4 * N]];
            let t = tick();
            for _ in 0..reps { for s in 0..8 { old_slot_sink(z.add(s), okeys.as_ptr().add(s * ROWS * LOCK), acc[0].as_mut_ptr().add(s * ROWS * 4), SINK); } }
            out += &format!(" old_l1_ns={:.2}", ns(tick() - t, reps * 8 * SINK));
            let t = tick();
            for _ in 0..calls { for s in 0..N { old_slot_sink(z.add(s), okeys.as_ptr().add(s * ROWS * LOCK), acc[0].as_mut_ptr().add(s * ROWS * 4), SINK); } }
            out += &format!(" old_single_ns={:.2}", ns(tick() - t, calls * N * SINK));
            let mut spent = 0;
            for c in 0..calls {
                for i in 0..SWEEP * N { *xw.add(i) = vdupq_n_s16(c as i16); }
                let t = tick();
                for s in 0..N { old_slot_sink(z.add(s), okeys.as_ptr().add(s * ROWS * LOCK), acc[c % 2].as_mut_ptr().add(s * ROWS * 4), SINK); }
                spent += tick() - t;
            }
            out += &format!(" old_paired_ns={:.2}", ns(spent, calls * N * SINK));
            let mut spent = 0;
            for c in 0..calls {
                for i in 0..SWEEP * N / 2 { *xw.add(i) = vdupq_n_s16(c as i16); }
                let t = tick();
                for s in 0..N { old_slot_sink(z.add(s), okeys.as_ptr().add(s * ROWS * LOCK), acc[c % 2].as_mut_ptr().add(s * ROWS * 4), SINK / 2); }
                spent += tick() - t;
            }
            out += &format!(" old_paired4_ns={:.2}", ns(spent, calls * N * SINK / 2));
            core::hint::black_box(&acc);
        }
        core::hint::black_box(&mm);
        core::hint::black_box(&mut xs);
    }
    out
}

/// The suite on one thread, then on `threads` threads at once (each its own buffers).
pub fn bench(threads: usize) {
    eprintln!("diag cntfrq={} one {}", freq(), suite());
    let all: Vec<String> = std::thread::scope(|s| {
        let h: Vec<_> = (0..threads).map(|w| s.spawn(move || { pin_worker(w); suite() })).collect();
        h.into_iter().map(|h| h.join().unwrap()).collect()
    });
    for (w, r) in all.iter().enumerate() {
        eprintln!("diag all{threads} w={w} {r}");
    }
}
