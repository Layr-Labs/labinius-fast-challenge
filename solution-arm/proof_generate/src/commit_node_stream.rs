//! The isolated 162-slot stream from the verified node-stream implementation.
use super::*;

const NODE_STREAM: u8 = 11;

/// [`slot_sink`]'s body; `$rows13` and `$rows46` start rows 1-3 (v20-v31) and rows 4-6
/// (v16-v27): loaded from `acc`, or zeroed for a chunk's first sweep ([`slot_sink_fresh`]).
#[cfg(target_feature = "sve")]
macro_rules! slot_sink_body {
    ($acc:expr, $z:expr, $k:expr, $groups:expr, $xs:expr, [$($rows13:literal),*], [$($rows46:literal),*]) => {
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
            xs4 = in(reg) 4 * $xs,
            ks = in(reg) KS,
            xs1 = const $xs,
            xs2 = const 2 * $xs,
            xs3 = const 3 * $xs,
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

/// The register-resident sink for one slot (from 4381e86b's `sve::slot_sink`, after arm-big's
/// `slot_chunk`): `acc(s, r) +=` the slot's products summed over `groups` key tiles (eight
/// elements each) of transforms starting at `z` (transform stride `XS`, the [`pass_words`] lane
/// order), keys starting at `k` (tile stride `KS`). Rows 0-3, then rows 4-6, keep their sums in
/// v16-v31 across all the tiles: one load and one store per sum per sweep. The key halves are
/// `sdot`'s indexed operand exactly as in [`dot`]: transforms `0..4` of a tile with key elements
/// `0..4` (`[0]`), transforms `4..8` with `4..8` (`[1]`). `groups >= 1`.
#[cfg(target_feature = "sve")]
#[inline(never)]
unsafe fn slot_sink<const WIDTH: usize>(z: *const int16x8_t, k: *const i16, acc: *mut int64x2_t, groups: usize) {
    const KS: usize = N * ROWS * LOCK * 2;
    debug_assert!((1..=SINK).contains(&groups));
    slot_sink_body!(acc, z, k, groups, WIDTH * 16, ["ldp q20, q21, [{a}, #64]", "ldp q22, q23, [{a}, #96]", "ldp q24, q25, [{a}, #128]", "ldp q26, q27, [{a}, #160]", "ldp q28, q29, [{a}, #192]", "ldp q30, q31, [{a}, #224]"], ["ldp q16, q17, [{a}, #256]", "ldp q18, q19, [{a}, #288]", "ldp q20, q21, [{a}, #320]", "ldp q22, q23, [{a}, #352]", "ldp q24, q25, [{a}, #384]", "ldp q26, q27, [{a}, #416]"]);
}

/// [`slot_sink`] for a chunk's first sweep: the digit rows' sums (rows 1-6) start from zero instead
/// of being loaded, and are stored as usual, so the chunk end needs no zeroing pass; the base
/// row (row 0) accumulates as in [`slot_sink`].
#[cfg(target_feature = "sve")]
#[inline(never)]
unsafe fn slot_sink_fresh<const WIDTH: usize>(z: *const int16x8_t, k: *const i16, acc: *mut int64x2_t, groups: usize) {
    const KS: usize = N * ROWS * LOCK * 2;
    debug_assert!((1..=SINK).contains(&groups));
    slot_sink_body!(acc, z, k, groups, WIDTH * 16, ["movi v20.16b, #0", "movi v21.16b, #0", "movi v22.16b, #0", "movi v23.16b, #0", "movi v24.16b, #0", "movi v25.16b, #0", "movi v26.16b, #0", "movi v27.16b, #0", "movi v28.16b, #0", "movi v29.16b, #0", "movi v30.16b, #0", "movi v31.16b, #0"], ["movi v16.16b, #0", "movi v17.16b, #0", "movi v18.16b, #0", "movi v19.16b, #0", "movi v20.16b, #0", "movi v21.16b, #0", "movi v22.16b, #0", "movi v23.16b, #0", "movi v24.16b, #0", "movi v25.16b, #0", "movi v26.16b, #0", "movi v27.16b, #0"]);
}

/// `acc +=` the products of the first `groups` tiles of `xs` (transforms `8 g..8 g + 8` of tile
/// `g`) with the key tiles at `keys` (`N ROWS LOCK` entries apart), every slot and row; the sums of
/// [`dot`] per tile, in another order. A tile past the column's end has zero keys. With `FRESH` (a
/// chunk's first sweep) the digit rows' sums are overwritten instead of added to, so the chunk end
/// needs no zeroing pass; the base row always accumulates.
#[inline(never)]
unsafe fn sink<const FRESH: bool, const WIDTH: usize>(
    xs: &[[int16x8_t; WIDTH]; SWEEP], keys: *const i16, acc: &mut [int64x2_t], groups: usize, first: usize,
) {
    debug_assert!(acc.len() == ROWS * 4 * N && (1..=SINK).contains(&groups));
    debug_assert!(first + WIDTH <= N);
    let keys = keys.add(first * ROWS * LOCK);
    let accp = acc.as_mut_ptr().add(idx(first, 0));
    #[cfg(target_feature = "sve")]
    {
        let z = xs.as_ptr() as *const int16x8_t;
        for s in 0..WIDTH {
            if FRESH {
                slot_sink_fresh::<WIDTH>(z.add(s), keys.add(s * ROWS * LOCK), accp.add(idx(s, 0)), groups);
            } else {
                slot_sink::<WIDTH>(z.add(s), keys.add(s * ROWS * LOCK), accp.add(idx(s, 0)), groups);
            }
        }
    }
    #[cfg(not(target_feature = "sve"))]
    for g in 0..groups {
        let tile = keys.add(g * N * ROWS * LOCK);
        for s in 0..WIDTH {
            let y: [int16x8_t; LOCK] = core::array::from_fn(|i| *xs[LOCK * g + i].get_unchecked(s));
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

/// Pack the input once, then reuse one node of output storage for all four
/// producer/consumer passes over a sweep. A partial tile still has zero key lanes.
#[inline(always)]
unsafe fn stream_sweep<const WIDTH: usize, const BANK: usize>(
    xs: &mut [[int16x8_t; WIDTH]; SWEEP], bank: &mut [PackedIndices; BANK],
    sc: &mut [int16x8_t; N], keys: *const i16, acc: &mut [int64x2_t],
    groups: usize, fresh: bool, kern: &Kernel, mut words_at: impl FnMut(usize) -> [*const u8; LANES],
) {
    debug_assert!(WIDTH == GROUPS && BANK == SWEEP);
    let count = groups * LOCK;
    for (i, nib) in bank[..count].iter_mut().enumerate() {
        indices_packed_asm(&words_at(i), nib);
    }
    let scratch = sc.first_chunk_mut::<GROUPS>().unwrap();
    for node in 0..N / GROUPS {
        let out = xs.as_flattened_mut().as_chunks_mut::<GROUPS>().0;
        pass_front_nodes(&bank[..count], scratch, &mut out[..count], kern, node);
        if fresh {
            sink::<true, WIDTH>(xs, keys, acc, groups, node * GROUPS);
        } else {
            sink::<false, WIDTH>(xs, keys, acc, groups, node * GROUPS);
        }
    }
}

#[inline(never)]
unsafe fn sums<const V: u8, const WIDTH: usize, const BANK: usize>(
    words: &[(*const u8, usize); LANES],
    a: &SmsMatrix,
    m: &mut [int16x8_t; BITS],
    nib: &mut [uint8x16_t; N / 4],
    xs: &mut [[int16x8_t; WIDTH]; SWEEP],
    bank: &mut [PackedIndices; BANK],
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
            let keys = a.data.as_ptr().add(e / LOCK * N * ROWS * LOCK);
            let fresh = e % CHUNK == 0;
            let words_at = |i| pass_words(words, a.elements, e, i);
            if V == NODE_STREAM {
                stream_sweep(xs, bank, sc, keys, acc, groups, fresh, kern, words_at);
            } else {
                for (i, x) in xs[..groups * LOCK].iter_mut().enumerate() {
                    indices_asm(&words_at(i), nib);
                    lut_pass(nib, sc, kern);
                    tail::<V>(sc, x.first_chunk_mut().unwrap(), kern);
                }
                if fresh {
                    sink::<true, WIDTH>(xs, keys, acc, groups, 0);
                } else {
                    sink::<false, WIDTH>(xs, keys, acc, groups, 0);
                }
            }
            e += l;
        }
        let c = (end - 1) / CHUNK;
        const _: () = assert!(DIGITS % 2 == 0);
        let dv = dev.as_mut_ptr();
        for d in (0..DIGITS).step_by(2) {
            let mr = [a.mean_residues(c, d), a.mean_residues(c, d + 1)];
            let rows = [dv.add(d * 2 * N), dv.add((d + 1) * 2 * N)];
            chunk_row2(acc, 1 + d, mr, rows, [&mut *y, sc], &Repair { words, a, c });
        }
    }
    narrow(acc, 0, y);
    super::inverse_folded::<PB>(y);
}

/// [`sums`] for two batches against one sweep of the key (after IvanLudvig's submission
/// ce686fcc): per sweep, batch A's transforms into `xs` and their sink, then batch B's into the
/// same `xs` and their sink against the same key lines, which are then still cached.
#[inline(never)]
unsafe fn sums2<const V: u8, const WIDTH: usize, const BANK: usize>(
    words: [&[(*const u8, usize); LANES]; 2],
    a: &SmsMatrix,
    nib: &mut [uint8x16_t; N / 4],
    xs: &mut [[int16x8_t; WIDTH]; SWEEP],
    bank: &mut [PackedIndices; BANK],
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
                let words_at = |i| {
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
                    pass_words(words[t], a.elements, e, i)
                };
                let fresh = e % CHUNK == 0;
                if V == NODE_STREAM {
                    stream_sweep(xs, bank, sc, keys, accs[t], groups, fresh, kern, words_at);
                } else {
                    for (i, x) in xs[..count].iter_mut().enumerate() {
                        indices_asm(&words_at(i), nib);
                        lut_pass2(nib, sc, kern);
                        tail::<V>(sc, x.first_chunk_mut().unwrap(), kern);
                    }
                    if fresh {
                        sink::<true, WIDTH>(xs, keys, accs[t], groups, 0);
                    } else {
                        sink::<false, WIDTH>(xs, keys, accs[t], groups, 0);
                    }
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
                chunk_row2_fast(accs[t], 1 + d, mr, rows, rows16, [&mut *ys[t], sc], &Repair { words: words[t], a, c });
            }
            if (c + 1) % FLUSH == 0 || end == a.elements {
                flush16(devs[t], dev16[t]);
            }
        }
    }
    for t in 0..2 {
        narrow(accs[t], 0, ys[t]);
        super::inverse_folded::<PB>(ys[t]);
    }
    let _ = (&mut accs, &mut ys, &mut devs);
}

#[repr(C)]
struct Scratch<const WIDTH: usize, const BANK: usize> {
    xs: [[int16x8_t; WIDTH]; SWEEP],
    bank: [PackedIndices; BANK],
    /// The transform scratch: the table front and levels 4-5 run here (L1-resident, reused for every
    /// transform) and level 6 writes the sweep buffer out of place ([`pass_d_out`]).
    sc: [int16x8_t; N],
    acc: [int64x2_t; ROWS * 4 * N],
    dev: [int32x4_t; DIGITS * 2 * N],
    acc2: [int64x2_t; ROWS * 4 * N],
    dev2: [int32x4_t; DIGITS * 2 * N],
    /// The 16-bit deviation sums of the two batches between [`flush16`]s.
    dev16: [[int16x8_t; DIGITS * N]; 2],
    y2: [int16x8_t; N],
    m: [int16x8_t; BITS],
    nib: [uint8x16_t; N / 4],
    y: [int16x8_t; N],
    tail: [[i64; LANES]; N],
}

const ARENA: usize = 4 << 20;
const _: () = assert!(core::mem::size_of::<Scratch<N, 0>>() <= ARENA && core::mem::size_of::<Scratch<GROUPS, SWEEP>>() <= ARENA, "the scratch fits the arena");

struct Arena<const WIDTH: usize, const BANK: usize>(*mut Scratch<WIDTH, BANK>);

impl<const WIDTH: usize, const BANK: usize> Arena<WIDTH, BANK> {
    fn layout() -> std::alloc::Layout {
        std::alloc::Layout::from_size_align(ARENA.max(core::mem::size_of::<Scratch<WIDTH, BANK>>()), core::mem::align_of::<Scratch<WIDTH, BANK>>())
            .expect("the scratch layout")
    }

    fn new() -> Self {
        let layout = Self::layout();
        let p = unsafe { std::alloc::alloc_zeroed(layout) } as *mut Scratch<WIDTH, BANK>;
        if p.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        Arena(p)
    }
}

impl<const WIDTH: usize, const BANK: usize> Drop for Arena<WIDTH, BANK> {
    fn drop(&mut self) {
        unsafe { std::alloc::dealloc(self.0 as *mut u8, Self::layout()) }
    }
}

pub fn commit_node_stream(values: &[B128], a: &SmsMatrix, workers: usize, q: u128, out: &mut [u8], hook: bool) {
    commit_inner::<NODE_STREAM, GROUPS, SWEEP>(values, a, workers, q, out, hook);
}

fn commit_inner<const V: u8, const WIDTH: usize, const BANK: usize>(
    values: &[B128], a: &SmsMatrix, workers: usize, q: u128, out: &mut [u8], hook: bool,
) {
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
        let arena = Arena::<WIDTH, BANK>::new();
        let Scratch { xs, bank, sc, acc, dev, acc2, dev2, dev16, y2, m, nib, y, tail } = &mut *arena.0;
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
                    sums2::<V, WIDTH, BANK>([&w0, &w1], a, nib, xs, bank, sc, [&mut acc[..], &mut acc2[..]], [&mut *y, &mut *y2], [&mut dev[..], &mut dev2[..]], [&mut d16a[..], &mut d16b[..]]);
                } else {
                    sums::<V, WIDTH, BANK>(&batch_words(b), a, m, nib, xs, bank, sc, acc, y, dev);
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

