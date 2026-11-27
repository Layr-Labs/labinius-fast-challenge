//! Isolated commitment candidates; the parent SMS implementation is the unchanged leader.
use super::*;

const SLOT_BLOCK: usize = 27;

#[cfg(target_feature = "sve")]
macro_rules! slot_sink_body {
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
            xs4 = in(reg) 4 * sweep_stride::<V>(),
            ks = in(reg) KS,
            xs1 = const sweep_stride::<V>(),
            xs2 = const 2 * sweep_stride::<V>(),
            xs3 = const 3 * sweep_stride::<V>(),
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



#[inline(always)]
const fn sweep_slot<const V: u8>(slot: usize, transform: usize) -> usize {
    if V == 5 || V == 10 {
        (slot / SLOT_BLOCK * SWEEP + transform) * SLOT_BLOCK + slot % SLOT_BLOCK
    } else {
        transform * N + slot
    }
}

#[inline(always)]
const fn sweep_stride<const V: u8>() -> usize {
    if V == 5 || V == 10 { SLOT_BLOCK * 16 } else { N * 16 }
}

#[cfg(target_feature = "sve")]
#[inline(never)]
unsafe fn slot_sink<const V: u8>(z: *const int16x8_t, k: *const i16, acc: *mut int64x2_t, groups: usize) {
    const KS: usize = N * ROWS * LOCK * 2;
    debug_assert!((1..=SINK).contains(&groups));
    slot_sink_body!(acc, z, k, groups, ["ldp q20, q21, [{a}, #64]", "ldp q22, q23, [{a}, #96]", "ldp q24, q25, [{a}, #128]", "ldp q26, q27, [{a}, #160]", "ldp q28, q29, [{a}, #192]", "ldp q30, q31, [{a}, #224]"], ["ldp q16, q17, [{a}, #256]", "ldp q18, q19, [{a}, #288]", "ldp q20, q21, [{a}, #320]", "ldp q22, q23, [{a}, #352]", "ldp q24, q25, [{a}, #384]", "ldp q26, q27, [{a}, #416]"]);
}

#[cfg(target_feature = "sve")]
#[inline(never)]
unsafe fn slot_sink_fresh<const V: u8>(z: *const int16x8_t, k: *const i16, acc: *mut int64x2_t, groups: usize) {
    const KS: usize = N * ROWS * LOCK * 2;
    debug_assert!((1..=SINK).contains(&groups));
    slot_sink_body!(acc, z, k, groups, ["movi v20.16b, #0", "movi v21.16b, #0", "movi v22.16b, #0", "movi v23.16b, #0", "movi v24.16b, #0", "movi v25.16b, #0", "movi v26.16b, #0", "movi v27.16b, #0", "movi v28.16b, #0", "movi v29.16b, #0", "movi v30.16b, #0", "movi v31.16b, #0"], ["movi v16.16b, #0", "movi v17.16b, #0", "movi v18.16b, #0", "movi v19.16b, #0", "movi v20.16b, #0", "movi v21.16b, #0", "movi v22.16b, #0", "movi v23.16b, #0", "movi v24.16b, #0", "movi v25.16b, #0", "movi v26.16b, #0", "movi v27.16b, #0"]);
}

#[inline(never)]
unsafe fn sink<const FRESH: bool, const V: u8>(xs: &[[int16x8_t; N]; SWEEP], keys: *const i16, acc: &mut [int64x2_t], groups: usize) {
    debug_assert!(acc.len() == ROWS * 4 * N && (1..=SINK).contains(&groups));
    let accp = acc.as_mut_ptr();
    #[cfg(target_feature = "sve")]
    {
        let z = xs.as_ptr() as *const int16x8_t;
        for s in 0..N {
            if FRESH {
                slot_sink_fresh::<V>(z.add(sweep_slot::<V>(s, 0)), keys.add(s * ROWS * LOCK), accp.add(idx(s, 0)), groups);
            } else {
                slot_sink::<V>(z.add(sweep_slot::<V>(s, 0)), keys.add(s * ROWS * LOCK), accp.add(idx(s, 0)), groups);
            }
        }
    }
    #[cfg(not(target_feature = "sve"))]
    for g in 0..groups {
        let tile = keys.add(g * N * ROWS * LOCK);
        for s in 0..N {
            let y: [int16x8_t; LOCK] = core::array::from_fn(|i| *(xs.as_ptr() as *const int16x8_t).add(sweep_slot::<V>(s, LOCK * g + i)));
            let (a, b): (&[int16x8_t; 4], &[int16x8_t; 4]) = (&*(y.as_ptr() as *const _), &*(y.as_ptr().add(4) as *const _));
            for r in 0..ROWS {
                if FRESH && g == 0 && r > 0 {
                    for q in 0..4 {
                        *accp.add(idx(s, r) + q) = vdupq_n_s64(0);
                    }
                }
                dot(accp.add(idx(s, r)), a, b, vld1q_s16(tile.add((s * ROWS + r) * LOCK)));
            }
        }
    }
}

#[inline(always)]
unsafe fn tail_sweep<const V: u8>(
    sc: &mut [int16x8_t; N],
    xs: &mut [[int16x8_t; N]; SWEEP],
    transform: usize,
    kern: &Kernel,
) {
    if V == 5 || V == 10 {
        let out = xs.as_mut_ptr() as *mut int16x8_t;
        pass_tail_calls_c_strided::<{ SLOT_BLOCK * SWEEP }>(sc, out.add(transform * SLOT_BLOCK), kern);
    } else {
        tail::<V>(sc, &mut xs[transform], kern);
    }
}


#[inline(always)]
unsafe fn tail<const V: u8>(sc: &mut [int16x8_t; N], x: &mut [int16x8_t; N], kern: &Kernel) {
    if V == 8 { tail_both(sc, x, kern); } else { super::tail::<4>(sc, x, kern); }
}

#[inline(always)]
unsafe fn transform<const V: u8>(nib: &[uint8x16_t; N / 4], sc: &mut [int16x8_t; N], xs: &mut [[int16x8_t; N]; SWEEP], i: usize, kern: &Kernel, paired: bool) {
    #[cfg(target_feature = "sve")]
    if V == 7 { super::super::pass_front_tail(nib, sc, &mut xs[i], kern); return; }
    #[cfg(target_feature = "sve")]
    if V == 9 || V == 10 { super::super::pass_lut_paired(nib, sc, kern); }
    else if paired { lut_pass2(nib, sc, kern); } else { lut_pass(nib, sc, kern); }
    #[cfg(not(target_feature = "sve"))]
    if paired { lut_pass2(nib, sc, kern); } else { lut_pass(nib, sc, kern); }
    tail_sweep::<V>(sc, xs, i, kern);
}

#[inline(always)]
unsafe fn inverse_pair<const V: u8>(x: [&mut [int16x8_t; N]; 2], input: impl FnMut(usize, usize) -> int16x8_t, output: impl FnMut(usize, usize, int16x8_t)) {
    if V == 6 || V == 10 { super::super::inverse_lazy_io2_chunk_candidate(x, input, output); }
    else { super::super::inverse_lazy_io2(x, input, output); }
}

#[inline(always)]
unsafe fn chunk_row2<const V: u8>(acc: &[int64x2_t], r: usize, mr: [&[i16]; 2], dev: [*mut int32x4_t; 2], y: [&mut [int16x8_t; N]; 2], ctx: &Repair) {
    let p = P1 as i16;
    let thr = [vdupq_n_s16(ctx.a.threshold[ctx.c * DIGITS + r - 1]), vdupq_n_s16(ctx.a.threshold[ctx.c * DIGITS + r])];
    inverse_pair::<V>(y, |h, s| narrow_digit(acc, s, r + h), |h, s, x| {
        let mut v = canonical(vsubq_s16(x, vld1q_dup_s16(mr[h].as_ptr().add(s))), p);
        let dv = dev[h];
        let flag = vcgeq_s16(vabsq_s16(v), thr[h]);
        if vmaxvq_u16(flag) != 0 {
            v = repair_lanes(ctx, r - 1 + h, s, v, flag, dv);
        }
        *dv.add(2 * s) = vaddw_s16(*dv.add(2 * s), vget_low_s16(v));
        *dv.add(2 * s + 1) = vaddw_high_s16(*dv.add(2 * s + 1), v);
    });
}

#[inline(always)]
unsafe fn chunk_row2_fast<const V: u8>(acc: &[int64x2_t], r: usize, mr: [&[i16]; 2], dev: [*mut int32x4_t; 2], dev16: [*mut int16x8_t; 2], y: [&mut [int16x8_t; N]; 2], ctx: &Repair) {
    let p = P1 as i16;
    let [y0, y1] = y;
    let mut top = [vdupq_n_u16(0); 2];
    inverse_pair::<V>([&mut *y0, &mut *y1], |h, s| narrow_digit(acc, s, r + h), |h, s, x| {
        let v = canonical(vsubq_s16(x, vld1q_dup_s16(mr[h].as_ptr().add(s))), p);
        top[h] = vmaxq_u16(top[h], vreinterpretq_u16_s16(vabsq_s16(v)));
        let sum = dev16[h].add(s);
        *sum = vaddq_s16(*sum, v);
    });
    let ys = [y0, y1];
    for (h, y) in ys.into_iter().enumerate() {
        // `|v| >= threshold` in some slot and lane (a threshold of 0 flags every lane, as there)
        if vmaxvq_u16(top[h]) as i32 >= ctx.a.threshold[ctx.c * DIGITS + r - 1 + h] as i32 {
            chunk_row_exact(acc, r + h, mr[h], dev[h], dev16[h], y, ctx);
        }
    }
}

#[inline(never)]
unsafe fn sums<const V: u8>(
    words: &[(*const u8, usize); LANES],
    a: &SmsMatrix,
    m: &mut [int16x8_t; BITS],
    nib: &mut [uint8x16_t; N / 4],
    xs: &mut [[int16x8_t; N]; SWEEP],
    sc: &mut [int16x8_t; N],
    acc: &mut [int64x2_t],
    y: &mut [int16x8_t; N],
    dev: &mut [int32x4_t],
) {
    let kern = &kernels()[PB];
    for s in 0..N {
        acc[idx(s, 0)..][..4].fill(vdupq_n_s64(0));
    }
    let mut e = 0usize;
    while e < a.elements {
        let end = ((e / CHUNK + 1) * CHUNK).min(a.elements);
        while e < end {
            // Up to eight tiles per sweep, never across a chunk end (chunks are whole tiles). A short
            // tile only occurs at the end of the column, where the missing elements read zero
            // words and their key lanes are zero, so those lanes contribute nothing.
            let l = SWEEP.min(end - e);
            let groups = l.div_ceil(LOCK);
            for i in 0..groups * LOCK {
                let at = pass_words(words, a.elements, e, i);
                indices_asm(&at, nib);
                transform::<V>(nib, sc, xs, i, kern, false);
            }
            let keys = a.data.as_ptr().add(e / LOCK * N * ROWS * LOCK);
            // the chunk's first sweep starts the digit rows' sums afresh
            if e % CHUNK == 0 {
                sink::<true, V>(xs, keys, acc, groups);
            } else {
                sink::<false, V>(xs, keys, acc, groups);
            }
            e += l;
        }
        let c = (end - 1) / CHUNK;
        const _: () = assert!(DIGITS % 2 == 0);
        let dv = dev.as_mut_ptr();
        for d in (0..DIGITS).step_by(2) {
            let mr = [a.mean_residues(c, d), a.mean_residues(c, d + 1)];
            let rows = [dv.add(d * 2 * N), dv.add((d + 1) * 2 * N)];
            chunk_row2::<V>(acc, 1 + d, mr, rows, [&mut *y, &mut xs[0]], &Repair { words, a, c });
        }
    }
    narrow(acc, 0, y);
    super::super::inverse_folded::<PB>(y);
}

#[inline(never)]
unsafe fn sums2<const V: u8>(
    words: [&[(*const u8, usize); LANES]; 2],
    a: &SmsMatrix,
    nib: &mut [uint8x16_t; N / 4],
    xs: &mut [[int16x8_t; N]; SWEEP],
    sc: &mut [int16x8_t; N],
    acc: [&mut [int64x2_t]; 2],
    y: [&mut [int16x8_t; N]; 2],
    dev: [&mut [int32x4_t]; 2],
    dev16: [&mut [int16x8_t]; 2],
) {
    let kern = &kernels()[PB];
    let mut dev16 = dev16;
    for sums in dev16.iter_mut() {
        sums.fill(vdupq_n_s16(0));
    }
    let [acc0, acc1] = acc;
    let [y0, y1] = y;
    let [dev0, dev1] = dev;
    let accs: [&mut [int64x2_t]; 2] = [acc0, acc1];
    let ys: [&mut [int16x8_t; N]; 2] = [y0, y1];
    let devs: [&mut [int32x4_t]; 2] = [dev0, dev1];
    let (mut accs, mut ys, mut devs) = (accs, ys, devs);
    for t in 0..2 {
        for s in 0..N {
            accs[t][idx(s, 0)..][..4].fill(vdupq_n_s64(0));
        }
    }
    let mut e = 0usize;
    while e < a.elements {
        let end = ((e / CHUNK + 1) * CHUNK).min(a.elements);
        while e < end {
            let l = SWEEP.min(end - e);
            let groups = l.div_ceil(LOCK);
            let keys = a.data.as_ptr().add(e / LOCK * N * ROWS * LOCK);
            for t in 0..2 {
                let count = groups * LOCK;
                for i in 0..count {
                    // The input words [`PF_T`] transforms on: later in this sweep, or the start of
                    // the sweep that follows it (the other batch's, over the same elements for
                    // the first batch and over the next ones for the second). A counter profile
                    // on Neoverse-V2 had 30% of the table's lines arriving late (L2 refills in
                    // the index packing, 41% of its cycles stalled).
                    if i + PF_L1 < count {
                        prefetch_words_l1(words[t], e, i + PF_L1);
                    }
                    let ahead = i + PF_T;
                    if ahead < count {
                        prefetch_words(words[t], e, ahead);
                    } else if t == 0 {
                        prefetch_words(words[1], e, ahead - count);
                    } else {
                        prefetch_words(words[0], e + l, ahead - count);
                    }
                    let at = pass_words(words[t], a.elements, e, i);
                    indices_asm(&at, nib);
                    transform::<V>(nib, sc, xs, i, kern, true);
                }
                if e % CHUNK == 0 {
                    sink::<true, V>(xs, keys, accs[t], groups);
                } else {
                    sink::<false, V>(xs, keys, accs[t], groups);
                }
            }
            e += l;
        }
        let c = (end - 1) / CHUNK;
        for t in 0..2 {
            let dv = devs[t].as_mut_ptr();
            let dv16 = dev16[t].as_mut_ptr();
            for d in (0..DIGITS).step_by(2) {
                let mr = [a.mean_residues(c, d), a.mean_residues(c, d + 1)];
                let rows = [dv.add(d * 2 * N), dv.add((d + 1) * 2 * N)];
                let rows16 = [dv16.add(d * N), dv16.add((d + 1) * N)];
                chunk_row2_fast::<V>(accs[t], 1 + d, mr, rows, rows16, [&mut *ys[t], &mut xs[0]], &Repair { words: words[t], a, c });
            }
            if (c + 1) % FLUSH == 0 || end == a.elements {
                flush16(devs[t], dev16[t]);
            }
        }
    }
    for t in 0..2 {
        narrow(accs[t], 0, ys[t]);
        super::super::inverse_folded::<PB>(ys[t]);
    }
    let _ = (&mut accs, &mut ys, &mut devs);
}

pub fn commit_candidate<const V: u8>(values: &[B128], a: &SmsMatrix, workers: usize, q: u128, out: &mut [u8], hook: bool) {
    let len = 4 * a.elements;
    assert_eq!(values.len() % len, 0);
    let columns = values.len() / len;
    // `MODULUS < 2^40`: five wire bytes per coefficient, a constant so the stores are fixed-size.
    const W: usize = 5;
    const _: () = assert!(MODULUS < 1 << (8 * W) && MODULUS >= 1 << (8 * (W - 1)));
    assert_eq!(crate::rq::coefficient_bytes(q), W);
    let width = W;
    assert_eq!(out.len(), columns * N * width);
    for_blocks(out, N * width, LANES, workers, |first, block| unsafe {
        pin_worker(first * workers / columns);
        let arena = Arena::new();
        let Scratch { xs, sc, acc, dev, acc2, dev2, dev16, y2, m, nib, y, tail } = &mut *arena.0;
        let p = P1 as i16;
        let count = block.len() / (N * width);
        let batch_words = |b: usize| -> [(*const u8, usize); LANES] {
            let col0 = first + LANES * b;
            let here = (count - LANES * b).min(LANES);
            core::array::from_fn(|c| {
                if c < here {
                    (values[(col0 + c) * len..].as_ptr() as *const u8, 64)
                } else {
                    (ZERO_WORDS.as_ptr() as *const u8, 0)
                }
            })
        };
        let batches = count.div_ceil(LANES);
        let mut targets: Vec<&mut [u8]> = block.chunks_mut(LANES * N * width).collect();
        for b in 0..batches {
            // pairs of batches share one key sweep; the second one's results wait in `*2`
            if b % 2 == 0 {
                dev.fill(vdupq_n_s32(0));
                if b + 1 < batches {
                    dev2.fill(vdupq_n_s32(0));
                    let (w0, w1) = (batch_words(b), batch_words(b + 1));
                    let [d16a, d16b] = dev16;
                    sums2::<V>([&w0, &w1], a, nib, xs, sc, [&mut acc[..], &mut acc2[..]], [&mut *y, &mut *y2], [&mut dev[..], &mut dev2[..]], [&mut d16a[..], &mut d16b[..]]);
                } else {
                    sums::<V>(&batch_words(b), a, m, nib, xs, sc, acc, y, dev);
                }
            } else {
                core::mem::swap(dev, dev2);
                core::mem::swap(y, y2);
            }
            let target = &mut *targets[b];
            // The base row's centered residues (`|x| <= (p - 1) / 2`, [`super::inverse`]) to
            // `[0, p)` lane-wise, eight columns per vector: `+ p` where negative, as `rem_euclid`.
            let pv = vdupq_n_s16(p);
            let mut base = [[0u16; LANES]; N];
            for (k, b) in base.iter_mut().enumerate() {
                let v = y[k];
                let v = vaddq_s16(v, vandq_s16(vreinterpretq_s16_u16(vcltzq_s16(v)), pv));
                vst1q_u16(b.as_mut_ptr(), vreinterpretq_u16_s16(v));
            }
            // `tail = mu + sum_d W_d dev_d` ([`add_weighted`]'s map) for the eight columns at
            let tail = &mut *tail;
            for k in 0..N / 2 {
                let h = k + N / 2;
                let mut lo = [vdupq_n_s64(a.mu[k] as i64); 4];
                let mut hi = [vdupq_n_s64(a.mu[h] as i64); 4];
                for d in 0..DIGITS {
                    let (u, v) = lattice::WEIGHTS[d];
                    let w = u + v;
                    for half in 0..2 {
                        let x = dev[d * 2 * N + 2 * k + half];
                        let y = dev[d * 2 * N + 2 * h + half];
                        let (xl, yl) = (vget_low_s32(x), vget_low_s32(y));
                        let (i, j) = (2 * half, 2 * half + 1);
                        lo[i] = vmlal_n_s32(lo[i], xl, u);
                        lo[j] = vmlal_high_n_s32(lo[j], x, u);
                        lo[i] = vmlsl_n_s32(lo[i], yl, v);
                        lo[j] = vmlsl_high_n_s32(lo[j], y, v);
                        hi[i] = vmlal_n_s32(hi[i], xl, v);
                        hi[j] = vmlal_high_n_s32(hi[j], x, v);
                        hi[i] = vmlal_n_s32(hi[i], yl, w);
                        hi[j] = vmlal_high_n_s32(hi[j], y, w);
                    }
                }
                for q in 0..4 {
                    vst1q_s64(tail[k].as_mut_ptr().add(2 * q), lo[q]);
                    vst1q_s64(tail[h].as_mut_ptr().add(2 * q), hi[q]);
                }
            }
            // a partial batch has fewer than eight columns in `target`: only those lanes are written
            for (c, column) in target.chunks_exact_mut(N * width).enumerate() {
                let (slots, _) = column.as_chunks_mut::<W>();
                for (k, slot) in slots.iter_mut().enumerate() {
                    let v = crt(base[k][c] as u32, tail[k][c]);
                    slot[..4].copy_from_slice(&(v as u32).to_le_bytes());
                    slot[4] = (v >> 32) as u8;
                }
            }
            if b % 2 == 1 && hook {
                if let Some(between) = BETWEEN.get() {
                    between(first * workers / columns, (b + 1) / 2);
                }
            }
        }
    });
}
