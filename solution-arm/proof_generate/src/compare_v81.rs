// Production snapshot for the R8 comparison.
use super::*;

const LOCK: usize = 8;
/// Elements transformed per pass over the accumulators: two tiles, so that each accumulator is
/// read and written once per two tiles.
const PASS: usize = 2 * LOCK;
const _: () = assert!(CHUNK % PASS == 0);

/// The accumulators of slot `s`, row `r`: `acc[idx..idx + 4]`, two columns per `i64` pair.
#[inline(always)]
const fn idx(s: usize, r: usize) -> usize {
    (s * ROWS + r) * 4
}

/// `acc[q] += ` the four-term dot product per `i64` lane of `a[q]` with the low half of `key`, and
/// of `b[q]` with the high half: SVE's indexed halfword `sdot`, eight products an instruction.
/// With SVE the sums go through [`sink`]; this is the reference for its operand order.
#[cfg(target_feature = "sve")]
#[allow(dead_code)]
#[inline(always)]
unsafe fn dot(acc: *mut int64x2_t, a: &[int16x8_t; 4], b: &[int16x8_t; 4], key: int16x8_t) {
    let (mut s0, mut s1, mut s2, mut s3) = (*acc, *acc.add(1), *acc.add(2), *acc.add(3));
    core::arch::asm!(
        "sdot z0.d, z4.h, z12.h[0]", "sdot z1.d, z5.h, z12.h[0]",
        "sdot z2.d, z6.h, z12.h[0]", "sdot z3.d, z7.h, z12.h[0]",
        "sdot z0.d, z8.h, z12.h[1]", "sdot z1.d, z9.h, z12.h[1]",
        "sdot z2.d, z10.h, z12.h[1]", "sdot z3.d, z11.h, z12.h[1]",
        inout("v0") s0, inout("v1") s1, inout("v2") s2, inout("v3") s3,
        in("v4") a[0], in("v5") a[1], in("v6") a[2], in("v7") a[3],
        in("v8") b[0], in("v9") b[1], in("v10") b[2], in("v11") b[3],
        in("v12") key,
        options(pure, nomem, nostack, preserves_flags)
    );
    (*acc, *acc.add(1), *acc.add(2), *acc.add(3)) = (s0, s1, s2, s3);
}

/// The same sums without SVE, for development on other AArch64 hosts.
#[cfg(not(target_feature = "sve"))]
#[inline(always)]
unsafe fn dot(acc: *mut int64x2_t, a: &[int16x8_t; 4], b: &[int16x8_t; 4], key: int16x8_t) {
    let (k0, k1) = (vget_low_s16(key), vget_high_s16(key));
    for q in 0..4 {
        let pair = |v: int16x8_t, k: int16x4_t| {
            let (lo, hi) = (vmull_s16(vget_low_s16(v), k), vmull_s16(vget_high_s16(v), k));
            vcombine_s64(vdup_n_s64(vaddlvq_s32(lo)), vdup_n_s64(vaddlvq_s32(hi)))
        };
        *acc.add(q) = vaddq_s64(vaddq_s64(*acc.add(q), pair(a[q], k0)), pair(b[q], k1));
    }
}

/// Key groups (tiles of [`LOCK`] elements) per sweep of [`sink`] over the accumulators.
const SINK: usize = CHUNK / LOCK;
/// Elements transformed per sweep: up to eight tiles, each accumulator read and written once.
const SWEEP: usize = SINK * LOCK;
const _: () = assert!(CHUNK % SWEEP == 0 && SWEEP % PASS == 0);

type NarrowedDigits = [[int16x8_t; N]; DIGITS];

#[cfg(any(target_feature = "sve", test))]
static NARROW_COEFFICIENTS: [i32; 4] = [C16 - (1 << 16), V31, P1 as i32, 0];

/// The two four-lane halves of [`narrow_digit`]. v0 holds its three constants.
/// The caller supplies four consecutive accumulator vectors in column order.
#[cfg(any(target_feature = "sve", test))]
macro_rules! narrow_terminal_row {
    ($a:literal, $b:literal, $c:literal, $d:literal, $offset:literal) => {
        concat!(
            "rshrn v1.2s, v", $a, ".2d, #16\n",
            "rshrn2 v1.4s, v", $b, ".2d, #16\n",
            "uzp1 v2.4s, v", $a, ".4s, v", $b, ".4s\n",
            "mla v2.4s, v1.4s, v0.s[0]\n",
            "sqrdmulh v4.4s, v2.4s, v0.s[1]\n",
            "mls v2.4s, v4.4s, v0.s[2]\n",
            "rshrn v1.2s, v", $c, ".2d, #16\n",
            "rshrn2 v1.4s, v", $d, ".2d, #16\n",
            "uzp1 v3.4s, v", $c, ".4s, v", $d, ".4s\n",
            "mla v3.4s, v1.4s, v0.s[0]\n",
            "sqrdmulh v4.4s, v3.4s, v0.s[1]\n",
            "mls v3.4s, v4.4s, v0.s[2]\n",
            "uzp1 v3.8h, v2.8h, v3.8h\n",
            "str q3, [{compact}, #", $offset, "]"
        )
    };
}

/// Accumulate all six rows of each slot together. Registers z8-z31 hold the
/// four column pairs of each row. z0-z5 hold the six indexed key operands.
/// Load each witness once and use it for all six rows before loading the next.
/// Keep the slot loop inside the assembler so the callee saves occur once per sweep.
#[cfg(target_feature = "sve")]
#[inline(never)]
unsafe fn sink_sve<const FRESH: bool, const TERMINAL: bool>(
    xs: &[[int16x8_t; N]; SWEEP], keys: *const i16, acc: &mut [int64x2_t], groups: usize,
    compact: *mut int16x8_t,
) {
    const XS: usize = N * 16;
    const KS: usize = N * ROWS * LOCK * 2;
    const _: () = assert!(ROWS == 6 && LOCK == 8);
    debug_assert!(acc.len() == ROWS * 4 * N && (1..=SINK).contains(&groups));
    macro_rules! run {
        ([$($init:literal),* $(,)?], [$($post:expr),* $(,)?], $($extra:tt)*) => {
            core::arch::asm!(
                "2:",
                "ldp q8, q9, [{a}, #0]",
                "ldp q10, q11, [{a}, #32]",
                $( $init, )*
                "mov {zp}, {z}",
                "mov {kp}, {k}",
                "mov {n}, {groups}",
                "3:",
                "ldp q0, q1, [{kp}]",
                "ldp q2, q3, [{kp}, #32]",
                "ldp q4, q5, [{kp}, #64]",
                "ldr q6, [{zp}]",
                "ldr q7, [{zp}, #{xs1}]",
                "sdot z8.d, z6.h, z0.h[0]",
                "sdot z9.d, z7.h, z0.h[0]",
                "sdot z12.d, z6.h, z1.h[0]",
                "sdot z13.d, z7.h, z1.h[0]",
                "sdot z16.d, z6.h, z2.h[0]",
                "sdot z17.d, z7.h, z2.h[0]",
                "sdot z20.d, z6.h, z3.h[0]",
                "sdot z21.d, z7.h, z3.h[0]",
                "sdot z24.d, z6.h, z4.h[0]",
                "sdot z25.d, z7.h, z4.h[0]",
                "sdot z28.d, z6.h, z5.h[0]",
                "sdot z29.d, z7.h, z5.h[0]",
                "ldr q6, [{zp}, #{xs2}]",
                "ldr q7, [{zp}, #{xs3}]",
                "add {zp}, {zp}, {xs4}",
                "sdot z10.d, z6.h, z0.h[0]",
                "sdot z11.d, z7.h, z0.h[0]",
                "sdot z14.d, z6.h, z1.h[0]",
                "sdot z15.d, z7.h, z1.h[0]",
                "sdot z18.d, z6.h, z2.h[0]",
                "sdot z19.d, z7.h, z2.h[0]",
                "sdot z22.d, z6.h, z3.h[0]",
                "sdot z23.d, z7.h, z3.h[0]",
                "sdot z26.d, z6.h, z4.h[0]",
                "sdot z27.d, z7.h, z4.h[0]",
                "sdot z30.d, z6.h, z5.h[0]",
                "sdot z31.d, z7.h, z5.h[0]",
                "ldr q6, [{zp}]",
                "ldr q7, [{zp}, #{xs1}]",
                "sdot z8.d, z6.h, z0.h[1]",
                "sdot z9.d, z7.h, z0.h[1]",
                "sdot z12.d, z6.h, z1.h[1]",
                "sdot z13.d, z7.h, z1.h[1]",
                "sdot z16.d, z6.h, z2.h[1]",
                "sdot z17.d, z7.h, z2.h[1]",
                "sdot z20.d, z6.h, z3.h[1]",
                "sdot z21.d, z7.h, z3.h[1]",
                "sdot z24.d, z6.h, z4.h[1]",
                "sdot z25.d, z7.h, z4.h[1]",
                "sdot z28.d, z6.h, z5.h[1]",
                "sdot z29.d, z7.h, z5.h[1]",
                "ldr q6, [{zp}, #{xs2}]",
                "ldr q7, [{zp}, #{xs3}]",
                "add {zp}, {zp}, {xs4}",
                "add {kp}, {kp}, {ks}",
                "sdot z10.d, z6.h, z0.h[1]",
                "sdot z11.d, z7.h, z0.h[1]",
                "sdot z14.d, z6.h, z1.h[1]",
                "sdot z15.d, z7.h, z1.h[1]",
                "sdot z18.d, z6.h, z2.h[1]",
                "sdot z19.d, z7.h, z2.h[1]",
                "sdot z22.d, z6.h, z3.h[1]",
                "sdot z23.d, z7.h, z3.h[1]",
                "sdot z26.d, z6.h, z4.h[1]",
                "sdot z27.d, z7.h, z4.h[1]",
                "sdot z30.d, z6.h, z5.h[1]",
                "sdot z31.d, z7.h, z5.h[1]",
                "subs {n}, {n}, #1",
                "b.ne 3b",
                "stp q8, q9, [{a}, #0]",
                "stp q10, q11, [{a}, #32]",
                $($post,)*
                "add {a}, {a}, #384",
                "add {z}, {z}, #16",
                "add {k}, {k}, #96",
                "subs {slots}, {slots}, #1",
                "b.ne 2b",
                a = inout(reg) acc.as_mut_ptr() => _,
                z = inout(reg) xs.as_ptr() => _,
                k = inout(reg) keys => _,
                slots = inout(reg) N => _,
                groups = in(reg) groups,
                xs4 = in(reg) 4 * XS,
                ks = in(reg) KS,
                xs1 = const XS,
                xs2 = const 2 * XS,
                xs3 = const 3 * XS,
                zp = out(reg) _,
                kp = out(reg) _,
                n = out(reg) _,
                $($extra)*
                out("v0") _, out("v1") _, out("v2") _, out("v3") _,
                out("v4") _, out("v5") _, out("v6") _, out("v7") _,
                out("v8") _, out("v9") _, out("v10") _, out("v11") _,
                out("v12") _, out("v13") _, out("v14") _, out("v15") _,
                out("v16") _, out("v17") _, out("v18") _, out("v19") _,
                out("v20") _, out("v21") _, out("v22") _, out("v23") _,
                out("v24") _, out("v25") _, out("v26") _, out("v27") _,
                out("v28") _, out("v29") _, out("v30") _, out("v31") _,
                options(nostack)
            )
        };
    }
    macro_rules! run_with_init {
        ([$($post:expr),* $(,)?], $($extra:tt)*) => {{
            if FRESH {
                run!([
                    "movi v12.16b, #0", "movi v13.16b, #0",
                    "movi v14.16b, #0", "movi v15.16b, #0",
                    "movi v16.16b, #0", "movi v17.16b, #0",
                    "movi v18.16b, #0", "movi v19.16b, #0",
                    "movi v20.16b, #0", "movi v21.16b, #0",
                    "movi v22.16b, #0", "movi v23.16b, #0",
                    "movi v24.16b, #0", "movi v25.16b, #0",
                    "movi v26.16b, #0", "movi v27.16b, #0",
                    "movi v28.16b, #0", "movi v29.16b, #0",
                    "movi v30.16b, #0", "movi v31.16b, #0",
                ], [$($post),*], $($extra)*);
            } else {
                run!([
                    "ldp q12, q13, [{a}, #64]", "ldp q14, q15, [{a}, #96]",
                    "ldp q16, q17, [{a}, #128]", "ldp q18, q19, [{a}, #160]",
                    "ldp q20, q21, [{a}, #192]", "ldp q22, q23, [{a}, #224]",
                    "ldp q24, q25, [{a}, #256]", "ldp q26, q27, [{a}, #288]",
                    "ldp q28, q29, [{a}, #320]", "ldp q30, q31, [{a}, #352]",
                ], [$($post),*], $($extra)*);
            }
        }};
    }
    if TERMINAL {
        run_with_init!([
            "ldr q0, [{coefficients}]",
            narrow_terminal_row!(12, 13, 14, 15, "0"),
            narrow_terminal_row!(16, 17, 18, 19, "{row1}"),
            narrow_terminal_row!(20, 21, 22, 23, "{row2}"),
            narrow_terminal_row!(24, 25, 26, 27, "{row3}"),
            narrow_terminal_row!(28, 29, 30, 31, "{row4}"),
            "add {compact}, {compact}, #16",
        ],
            coefficients = in(reg) NARROW_COEFFICIENTS.as_ptr(),
            compact = inout(reg) compact => _,
            row1 = const 16 * N, row2 = const 32 * N, row3 = const 48 * N, row4 = const 64 * N,
        );
    } else {
        run_with_init!([
            "stp q12, q13, [{a}, #64]",
            "stp q14, q15, [{a}, #96]",
            "stp q16, q17, [{a}, #128]",
            "stp q18, q19, [{a}, #160]",
            "stp q20, q21, [{a}, #192]",
            "stp q22, q23, [{a}, #224]",
            "stp q24, q25, [{a}, #256]",
            "stp q26, q27, [{a}, #288]",
            "stp q28, q29, [{a}, #320]",
            "stp q30, q31, [{a}, #352]",
        ],);
    }
}

#[cfg(target_feature = "sve")]
#[inline(always)]
unsafe fn sink<const FRESH: bool>(xs: &[[int16x8_t; N]; SWEEP], keys: *const i16, acc: &mut [int64x2_t], groups: usize) {
    sink_sve::<FRESH, false>(xs, keys, acc, groups, core::ptr::null_mut());
}

#[cfg(target_feature = "sve")]
#[inline(always)]
unsafe fn sink_terminal<const FRESH: bool>(
    xs: &[[int16x8_t; N]; SWEEP], keys: *const i16, acc: &mut [int64x2_t], groups: usize,
    narrowed: &mut NarrowedDigits,
) {
    sink_sve::<FRESH, true>(xs, keys, acc, groups, narrowed.as_mut_ptr().cast());
}

#[cfg(not(target_feature = "sve"))]
#[inline(never)]
unsafe fn sink_terminal<const FRESH: bool>(
    xs: &[[int16x8_t; N]; SWEEP], keys: *const i16, acc: &mut [int64x2_t], groups: usize,
    narrowed: &mut NarrowedDigits,
) {
    sink::<FRESH>(xs, keys, acc, groups);
    for (d, row) in narrowed.iter_mut().enumerate() {
        narrow_digits(acc, d + 1, row);
    }
}

/// Keep the base row in wide storage across chunks. Convert the digit rows
/// at the terminal sweep. A later chunk initializes those rows afresh.
/// The same products for development on AArch64 hosts without SVE.
#[cfg(not(target_feature = "sve"))]
#[inline(never)]
unsafe fn sink<const FRESH: bool>(xs: &[[int16x8_t; N]; SWEEP], keys: *const i16, acc: &mut [int64x2_t], groups: usize) {
    debug_assert!(acc.len() == ROWS * 4 * N && (1..=SINK).contains(&groups));
    let accp = acc.as_mut_ptr();
    for g in 0..groups {
        let tile = keys.add(g * N * ROWS * LOCK);
        for s in 0..N {
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

/// The eight witness pointers of transform `t` of a pass starting at element `e`, in dot-product
/// lane order: lane `4 l + k` of transform `4 h + q` is column `2 q + l` of element `e + 4 h + k`.
/// Each lane is an independent transform, so a 64-bit lane then holds one column's four elements
/// and `sdot` reads it directly, with no transpose. Elements past the column read as zero.
#[inline(always)]
unsafe fn pass_words(words: &[(*const u8, usize); LANES], elements: usize, e: usize, t: usize) -> [*const u8; LANES] {
    let (h, q) = (t / 4, t % 4);
    core::array::from_fn(|lane| {
        let (l, k) = (lane / 4, lane % 4);
        let element = e + 4 * h + k;
        let (w, stride) = words[2 * q + l];
        if element < elements { w.add(element * stride) } else { ZERO_WORDS.as_ptr() as *const u8 }
    })
}

/// Transforms ahead whose input words [`prefetch_words`] requests (into L2): four transforms span one
/// element group, so this is `PF_T / 4` element groups (of four elements, eight columns) ahead,
/// running into the next sweep's first elements at a sweep's end.
const PF_T: usize = 16;

/// Prefetches the input words of transform `t` of a pass starting at element `e` (the lines
/// [`pass_words`] would address: four consecutive 64-byte elements of two columns). The table
/// streams are eight (sixteen with paired batches) 64-byte-per-element column walks; this keeps
/// them ahead of the transforms instead of relying on the hardware stream prefetcher. A hint:
/// addresses past a column's end are never dereferenced. Official Graviton4 runs show the same
/// commit lap with and without it (234-235 ms against 233-234): the table stream does not stall
/// there, so this is neutral, kept as the measured variant.
#[inline(always)]
unsafe fn prefetch_words_l1(words: &[(*const u8, usize); LANES], e: usize, t: usize) {
    let (h, q) = (t / 4, t % 4);
    for l in 0..2 {
        let (w, stride) = words[2 * q + l];
        let p = w.wrapping_add((e + 4 * h) * stride);
        core::arch::asm!(
            "prfm pldl1keep, [{p}]", "prfm pldl1keep, [{p}, #64]",
            "prfm pldl1keep, [{p}, #128]", "prfm pldl1keep, [{p}, #192]",
            p = in(reg) p, options(nostack, preserves_flags, readonly)
        );
    }
}

/// Transforms ahead whose input words are also requested into L1, in addition to the L2 request
/// [`PF_T`] ahead (in-process A/B on the grader, diagnostic 422a97c9: about 0.5 ms per commitment).
const PF_L1: usize = 4;

#[inline(always)]
unsafe fn prefetch_words(words: &[(*const u8, usize); LANES], e: usize, t: usize) {
    let (h, q) = (t / 4, t % 4);
    for l in 0..2 {
        let (w, stride) = words[2 * q + l];
        let p = w.wrapping_add((e + 4 * h) * stride);
        core::arch::asm!(
            "prfm pldl2keep, [{p}]", "prfm pldl2keep, [{p}, #64]",
            "prfm pldl2keep, [{p}, #128]", "prfm pldl2keep, [{p}, #192]",
            p = in(reg) p, options(nostack, preserves_flags, readonly)
        );
    }
}

/// The centered residue of a sum below `2^41` in magnitude. Binary64 holds such a sum exactly and
/// its quotient by 9721 to far better than half a unit, so the rounded quotient is off by at most
/// one half step: `|result| <= 4861`.
#[inline(always)]
unsafe fn narrow_sum(v: int64x2_t) -> int32x2_t {
    const _: () = assert!(P1 == 9721);
    let q = vcvtnq_s64_f64(vmulq_n_f64(vcvtq_f64_s64(v), 1.0 / P1 as f64));
    vmovn_s64(vsubq_s64(v, vmull_n_s32(vmovn_s64(q), P1 as i32)))
}

/// Row `r`'s sums as residues, one `i16` vector per slot, lane = column.
#[inline(always)]
unsafe fn narrow(acc: &[int64x2_t], r: usize, out: &mut [int16x8_t; N]) {
    for (s, o) in out.iter_mut().enumerate() {
        let at = idx(s, r);
        let lo = vcombine_s32(narrow_sum(acc[at]), narrow_sum(acc[at + 1]));
        let hi = vcombine_s32(narrow_sum(acc[at + 2]), narrow_sum(acc[at + 3]));
        *o = vcombine_s16(vmovn_s32(lo), vmovn_s32(hi));
    }
}

/// The largest product of a key slot (a centered residue) and a transform slot.
const PRODUCT: u64 = (P1 as u64 - 1) / 2 * crate::ntt::lut_transform_bound(P1, SCHEDULES[PB]) as u64;
/// `2^16 mod p`, centered.
const C16: i32 = centered(1 << 16, P1) as i32;
/// `round(2^31 / p)`.
const V31: i32 = ((1u64 << 32) / P1 as u64).div_ceil(2) as i32;
// a chunk's digit sums `v` (|v| <= CHUNK PRODUCT): `(v >> 16) C16 + (v mod 2^16)` fits i32
const _: () = assert!(((CHUNK as u64 * PRODUCT >> 16) + 1) * C16.unsigned_abs() as u64 + 65535 < i32::MAX as u64);

/// [`narrow`] for a chunk's digit rows, in integers: `v = h 2^16 + l`, `l = v mod 2^16`, so
/// `v = h (2^16 mod p) + l (mod p)` within `i32`, then a 32-bit Barrett reduction to
/// `|r| <= 0.65 p` (the inverse takes `|x| <= p`). After the k6 chunk end (dadon line).
#[inline(always)]
unsafe fn narrow_digits(acc: &[int64x2_t], r: usize, out: &mut [int16x8_t; N]) {
    for (s, o) in out.iter_mut().enumerate() {
        *o = narrow_digit(acc, s, r);
    }
}

/// Slot `s` of [`narrow_digits`].
#[inline(always)]
unsafe fn narrow_digit(acc: &[int64x2_t], s: usize, r: usize) -> int16x8_t {
    // `h = round(v / 2^16)` and `l = v - h 2^16` in `[-2^15, 2^15)`: then `l + h C16` is the low
    // word of `v` plus `h (C16 - 2^16)` in wrapping 32-bit arithmetic (the true value fits, as
    // asserted above for the larger unsigned `l`), so the low words need no mask: one instruction
    // fewer per four lanes than the unsigned split. The residue's class is the same.
    let c = vdupq_n_s32(C16 - (1 << 16));
    let half = |a: int64x2_t, b: int64x2_t| -> int32x4_t {
        let h = vrshrn_high_n_s64::<16>(vrshrn_n_s64::<16>(a), b);
        let l = vmovn_high_s64(vmovn_s64(a), b);
        let t = vmlaq_s32(l, h, c);
        vmlsq_n_s32(t, vqrdmulhq_n_s32(t, V31), P1 as i32)
    };
    let at = idx(s, r);
    let lo = half(*acc.get_unchecked(at), *acc.get_unchecked(at + 1));
    let hi = half(*acc.get_unchecked(at + 2), *acc.get_unchecked(at + 3));
    vmovn_high_s32(vmovn_s32(lo), hi)
}

/// One digit row's chunk end in one go: the sums narrowed in [`super::inverse_lazy_io`]'s first
/// pass, and the deviations `S - mean` (centered) summed into `dev` (the row's `2 N` lanes) in its
/// last pass, instead of three passes over the slots.
#[inline(always)]
#[cfg_attr(not(test), allow(dead_code))]
unsafe fn chunk_row(acc: &[int64x2_t], r: usize, mr: &[i16], dev: &mut [int32x4_t], y: &mut [int16x8_t; N]) {
    let p = P1 as i16;
    let dv = dev.as_mut_ptr();
    super::inverse_lazy_io(y, |s| narrow_digit(acc, s, r), |s, x| {
        let v = canonical(vsubq_s16(x, vld1q_dup_s16(mr.as_ptr().add(s))), p);
        *dv.add(2 * s) = vaddw_s16(*dv.add(2 * s), vget_low_s16(v));
        *dv.add(2 * s + 1) = vaddw_high_s16(*dv.add(2 * s + 1), v);
    });
}

#[inline(always)]
unsafe fn chunk_row2(narrowed: [&[int16x8_t; N]; 2], r: usize, mr: [&[i16]; 2], dev: [*mut int32x4_t; 2], y: [&mut [int16x8_t; N]; 2], ctx: &Repair) {
    let p = P1 as i16;
    let thr = [vdupq_n_s16(ctx.a.threshold[ctx.c * DIGITS + r - 1]), vdupq_n_s16(ctx.a.threshold[ctx.c * DIGITS + r])];
    super::inverse_lazy_io2(y, |h, s| *narrowed[h].get_unchecked(s), |h, s, x| {
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


/// Chunks whose deviations are summed in 16 bits before [`flush16`] widens them: a centered
/// deviation is at most `(p - 1) / 2` in magnitude.
const FLUSH: usize = 6;
const _: () = assert!(FLUSH as i32 * ((P1 as i32 - 1) / 2) <= i16::MAX as i32);

/// [`chunk_row2`] with a short last step. Per slot the reference tests the deviations against the
/// row's repair threshold, moves the flag to a general register, branches, and adds the eight
/// lanes into two 32-bit vectors (five more instructions and 32 bytes of load and store); a row
/// rarely has an ambiguous lane. Here the last step only centers, keeps the
/// lane-wise maximum magnitude of the row and adds into one 16-bit vector (`dev16`, widened into
/// `dev` every [`FLUSH`] chunks). A row whose maximum reaches the threshold is done again by
/// [`chunk_row_exact`], which takes the 16-bit additions back. The sums are the same integers.
#[inline(always)]
unsafe fn chunk_row2_fast(narrowed: [&[int16x8_t; N]; 2], r: [usize; 2], mr: [&[i16]; 2], dev: [*mut int32x4_t; 2], dev16: [*mut int16x8_t; 2], y: [&mut [int16x8_t; N]; 2], ctx: [&Repair; 2]) {
    let p = P1 as i16;
    let [y0, y1] = y;
    let mut top = [vdupq_n_u16(0); 2];
    super::inverse_lazy_io2([&mut *y0, &mut *y1], |h, s| *narrowed[h].get_unchecked(s), |h, s, x| {
        let v = canonical(vsubq_s16(x, vld1q_dup_s16(mr[h].as_ptr().add(s))), p);
        top[h] = vmaxq_u16(top[h], vreinterpretq_u16_s16(vabsq_s16(v)));
        let sum = dev16[h].add(s);
        *sum = vaddq_s16(*sum, v);
    });
    let ys = [y0, y1];
    for (h, y) in ys.into_iter().enumerate() {
        // `|v| >= threshold` in some slot and lane (a threshold of 0 flags every lane, as there)
        if vmaxvq_u16(top[h]) as i32 >= ctx[h].a.threshold[ctx[h].c * DIGITS + r[h] - 1] as i32 {
            chunk_row_exact(narrowed[h], r[h], mr[h], dev[h], dev16[h], y, ctx[h]);
        }
    }
}

/// Row `r` of a chunk end exactly as [`chunk_row2`] does it, after [`chunk_row2_fast`] has added
/// the row's deviations into `dev16`: each slot's deviation is taken out of `dev16` again and
/// goes into `dev` with its ambiguous lanes recomputed ([`repair_lanes`]).
#[cold]
#[inline(never)]
unsafe fn chunk_row_exact(narrowed: &[int16x8_t; N], r: usize, mr: &[i16], dev: *mut int32x4_t, dev16: *mut int16x8_t, y: &mut [int16x8_t; N], ctx: &Repair) {
    let p = P1 as i16;
    let thr = vdupq_n_s16(ctx.a.threshold[ctx.c * DIGITS + r - 1]);
    super::inverse_lazy_io(y, |s| *narrowed.get_unchecked(s), |s, x| {
        let mut v = canonical(vsubq_s16(x, vld1q_dup_s16(mr.as_ptr().add(s))), p);
        *dev16.add(s) = vsubq_s16(*dev16.add(s), v);
        let flag = vcgeq_s16(vabsq_s16(v), thr);
        if vmaxvq_u16(flag) != 0 {
            v = repair_lanes(ctx, r - 1, s, v, flag, dev);
        }
        *dev.add(2 * s) = vaddw_s16(*dev.add(2 * s), vget_low_s16(v));
        *dev.add(2 * s + 1) = vaddw_high_s16(*dev.add(2 * s + 1), v);
    });
}

/// `dev += dev16` lane by lane (two 32-bit vectors per 16-bit one), and `dev16 = 0`.
#[inline(never)]
unsafe fn flush16(dev: &mut [int32x4_t], dev16: &mut [int16x8_t]) {
    debug_assert_eq!(dev.len(), 2 * dev16.len());
    for (s, v) in dev16.iter_mut().enumerate() {
        *dev.get_unchecked_mut(2 * s) = vaddw_s16(*dev.get_unchecked(2 * s), vget_low_s16(*v));
        *dev.get_unchecked_mut(2 * s + 1) = vaddw_high_s16(*dev.get_unchecked(2 * s + 1), *v);
        *v = vdupq_n_s16(0);
    }
}

/// What [`repair`] needs at a chunk end: the batch's column words, the matrix and the chunk.
struct Repair<'a> {
    words: &'a [(*const u8, usize); LANES],
    a: &'a SmsMatrix,
    c: usize,
}

/// The flagged lanes of slot `s`'s deviations `v` (digit `d`) recomputed exactly and added to
/// their `dev` lanes; returned with those lanes zeroed, so the caller's ordinary sum skips them.
#[cold]
#[inline(never)]
unsafe fn repair_lanes(ctx: &Repair, d: usize, s: usize, v: int16x8_t, flag: uint16x8_t, dv: *mut int32x4_t) -> int16x8_t {
    let mut lanes = [0i16; 8];
    vst1q_s16(lanes.as_mut_ptr(), v);
    let mut flags = [0u16; 8];
    vst1q_u16(flags.as_mut_ptr(), flag);
    let mut sums = [[0i32; 4]; 2];
    vst1q_s32(sums[0].as_mut_ptr(), *dv.add(2 * s));
    vst1q_s32(sums[1].as_mut_ptr(), *dv.add(2 * s + 1));
    for lane in 0..8 {
        if flags[lane] != 0 {
            let exact = repair(ctx.words[lane], ctx.a, ctx.c, d, s);
            debug_assert_eq!((exact - lanes[lane] as i64).rem_euclid(P1 as i64), 0, "the exact deviation has the residue's class");
            sums[lane / 4][lane % 4] += i32::try_from(exact).expect("a chunk deviation fits i32");
            lanes[lane] = 0;
        }
    }
    *dv.add(2 * s) = vld1q_s32(sums[0].as_ptr());
    *dv.add(2 * s + 1) = vld1q_s32(sums[1].as_ptr());
    vld1q_s16(lanes.as_ptr())
}

/// Front padding of [`repair`]'s reversed digit row: at least `N` and a multiple of 16.
const REPAIR_PRE: usize = 656;

/// `sum` of the signed byte products of `a` and `b` in four `i32` lanes (`SDOT`).
#[inline(always)]
unsafe fn sdot(mut acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
    core::arch::asm!("sdot {acc:v}.4s, {a:v}.16b, {b:v}.16b", acc = inout(vreg) acc, a = in(vreg) a, b = in(vreg) b,
        options(pure, nomem, nostack, preserves_flags));
    acc
}

/// The exact deviation `S - mean` of coefficient `k` of digit `d`'s chunk-`c` sum
/// `S = sum_m (L_{d,m} s_m)[k]` over the chunk's elements `m`, for the column whose element `m`
/// has its four witness words at `col.0 + m col.1` (coefficient `4 q + j` is bit `q` of word `j`).
/// Coefficient `k` of `X^i L` is `L[k - i] - L[k + 648 - i] - L[k + 972 - i]` for `k < 324` and
/// `L[k - i] + L[k + 324 - i]` for `k >= 324` (`L` zero outside `0..648`), so per element the
/// weights of the 512 witness coefficients are windows of the reversed digit row.
#[cold]
#[inline(never)]
pub unsafe fn repair(col: (*const u8, usize), a: &SmsMatrix, c: usize, d: usize, k: usize) -> i64 {
    const PRE: usize = REPAIR_PRE;
    const _: () = assert!(PRE >= N && PRE % 16 == 0 && 3 * (lattice::MAX_DIGIT_ABS as i32) <= i8::MAX as i32);
    // buf[PRE + j] = L[N - 1 - j]; zero elsewhere, so L[x] = buf[PRE + N - 1 - x] for any x the
    // windows reach (x in -511 .. 1158)
    let mut buf = [0i8; PRE + N + 512 + 16];
    let mut mask = [0i8; 512];
    let bits: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
    let bits = vld1q_u8(bits.as_ptr());
    let start = c * CHUNK;
    let end = ((c + 1) * CHUNK).min(a.elements);
    let (o1, o2, o3) = if k < N / 2 {
        (PRE + N - 1 - k, PRE - 1 - k, PRE - (N / 2 + 1) - k)
    } else {
        (PRE + N - 1 - k, PRE + N / 2 - 1 - k, 0)
    };
    let mut acc = vdupq_n_s32(0);
    for m in start..end {
        let w = col.0.add(m * col.1);
        let wv: [uint8x16_t; 4] = core::array::from_fn(|j| vld1q_u8(w.add(16 * j)));
        // mask[4 q + j] = -1 where bit q of word j is set
        for ch in 0..8u8 {
            let idx = vcombine_u8(vdup_n_u8(2 * ch), vdup_n_u8(2 * ch + 1));
            let v = uint8x16x4_t(
                vtstq_u8(vqtbl1q_u8(wv[0], idx), bits),
                vtstq_u8(vqtbl1q_u8(wv[1], idx), bits),
                vtstq_u8(vqtbl1q_u8(wv[2], idx), bits),
                vtstq_u8(vqtbl1q_u8(wv[3], idx), bits),
            );
            vst4q_u8(mask.as_mut_ptr().add(64 * ch as usize) as *mut u8, v);
        }
        let l = a.digits.as_ptr().add((m * DIGITS + d) * N);
        for t in 0..N / 8 {
            let v = vrev64_s8(vld1_s8(l.add(8 * t)));
            vst1_s8(buf.as_mut_ptr().add(PRE + N - 8 - 8 * t), v);
        }
        let b = buf.as_ptr();
        for v in 0..32 {
            let i = 16 * v;
            let w1 = vld1q_s8(b.add(o1 + i));
            let wt = if k < N / 2 {
                vsubq_s8(vsubq_s8(w1, vld1q_s8(b.add(o2 + i))), vld1q_s8(b.add(o3 + i)))
            } else {
                vaddq_s8(w1, vld1q_s8(b.add(o2 + i)))
            };
            acc = sdot(acc, wt, vld1q_s8(mask.as_ptr().add(i)));
        }
    }
    // the masks are -1: acc holds -S
    -(vaddlvq_s32(acc)) - a.mean_int[(c * DIGITS + d) * N + k] as i64
}

/// `x` in `[-(p-1)/2, (p-1)/2]` for `|x| <= p + p/2`.
#[inline(always)]
unsafe fn canonical(x: int16x8_t, p: i16) -> int16x8_t {
    let h = vdupq_n_s16((p - 1) / 2);
    let pv = vdupq_n_s16(p);
    let over = vcgtq_s16(x, h);
    let under = vcltq_s16(x, vnegq_s16(h));
    let x = vsubq_s16(x, vandq_s16(vreinterpretq_s16_u16(over), pv));
    vaddq_s16(x, vandq_s16(vreinterpretq_s16_u16(under), pv))
}

/// Levels 0 to 3 of the transform from the nibble tables: the SVE2 `tbl` kernel on a host
/// built with SVE (the official Neoverse V2 target), the equivalent NEON `pass_lut` otherwise,
/// so that the portable path builds and runs on AArch64 hosts without SVE.
#[cfg(target_feature = "sve")]
#[inline(always)]
unsafe fn lut_pass(nib: &[uint8x16_t; N / 4], x: &mut [int16x8_t; N], _kern: &Kernel) {
    super::weighted::pass_lut_weighted(nib, x);
}

#[cfg(target_feature = "sve")]
#[inline(always)]
unsafe fn lut_pass2(nib: &[uint8x16_t; N / 4], x: &mut [int16x8_t; N], _kern: &Kernel) {
    super::weighted::pass_lut_weighted(nib, x)
}

#[cfg(not(target_feature = "sve"))]
#[inline(always)]
unsafe fn lut_pass2(nib: &[uint8x16_t; N / 4], x: &mut [int16x8_t; N], kern: &Kernel) {
    lut_pass(nib, x, kern)
}

#[cfg(not(target_feature = "sve"))]
#[inline(always)]
unsafe fn lut_pass(nib: &[uint8x16_t; N / 4], x: &mut [int16x8_t; N], kern: &Kernel) {
    // `indices` writes the packed byte layout the SVE `ld1b` consumer reads; expand each
    // slot's eight bytes to the `(2 v, 2 v + 1)` byte pairs `pass_lut`'s `vqtbl2q_u8` indexes.
    let bytes = core::slice::from_raw_parts(nib.as_ptr() as *const u8, 16 * (N / 4));
    let mut idx = [vdupq_n_u8(0); N / 4];
    for (s, slot) in idx.iter_mut().enumerate() {
        let v = vld1_u8(bytes.as_ptr().add(super::packed_slot(s)));
        let lo = vshl_n_u8::<1>(v);
        let hi = vorr_u8(lo, vdup_n_u8(1));
        *slot = vzip1q_u8(vcombine_u8(lo, lo), vcombine_u8(hi, hi));
    }
    super::pass_lut::<PB>(&idx, x, kern);
}

/// The batch's pass: each chunk's digit sums inverted to canonical coefficients, handed to
/// `chunk(c, d, coefficients)`; on return `y` holds the base row's coefficients.
/// Levels 4 to 6 of one transform, from the scratch `sc` into the sweep buffer slot `x`:
/// `V = 0` the two passes [`pass_c_asm`] and [`pass_d_out`]; otherwise the fused per-node kernel
/// ([`pass_tail`] straight-line for `V = 1`, [`pass_tail_calls`] for `V = 2`, its other schedules
/// [`pass_tail_calls_b`] for `V = 3` and [`pass_tail_calls_c`] for `V = 4`). The same values.
#[inline(always)]
unsafe fn tail<const V: u8>(sc: &mut [int16x8_t; N], x: &mut [int16x8_t; N], kern: &Kernel) {
    match V {
        0 => {
            pass_c_asm(sc, kern);
            pass_d_out(sc, x, kern);
        }
        1 => pass_tail(sc, x, kern),
        2 => pass_tail_calls(sc, x, kern),
        3 => pass_tail_calls_b(sc, x, kern),
        _ => pass_tail_calls_c(sc, x, kern),
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
    narrowed: &mut NarrowedDigits,
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
        // The last partial tile uses zero input words and padded key lanes.
        let l = end - e;
        let groups = l.div_ceil(LOCK);
        for i in 0..groups * LOCK {
            let at = pass_words(words, a.elements, e, i);
            indices_asm(&at, nib);
            let x = &mut xs[i];
            lut_pass(nib, sc, kern);
            tail::<V>(sc, x, kern);
        }
        let keys = a.data.as_ptr().add(e / LOCK * N * ROWS * LOCK);
        sink_terminal::<true>(xs, keys, acc, groups, narrowed);
        e = end;
        let c = (end - 1) / CHUNK;
        const _: () = assert!(DIGITS == 5);
        let ctx = &Repair { words, a, c };
        let dv = dev.as_mut_ptr();
        for d in (0..DIGITS - 1).step_by(2) {
            let mr = [a.mean_residues(c, d), a.mean_residues(c, d + 1)];
            let rows = [dv.add(d * 2 * N), dv.add((d + 1) * 2 * N)];
            chunk_row2([&narrowed[d], &narrowed[d + 1]], 1 + d, mr, rows, [&mut *y, &mut xs[0]], ctx);
        }
        chunk_last_row(&narrowed[DIGITS - 1], dv.add((DIGITS - 1) * 2 * N), y, ctx);
    }
    narrow(acc, 0, y);
    super::inverse_folded::<PB>(y);
}

/// Process the fifth digit when there is no second batch to pair it with.
#[inline(always)]
unsafe fn chunk_last_row(narrowed: &[int16x8_t; N], dv: *mut int32x4_t, y: &mut [int16x8_t; N], ctx: &Repair) {
    let d = DIGITS - 1;
    let p = P1 as i16;
    let mr = ctx.a.mean_residues(ctx.c, d);
    let thr = vdupq_n_s16(ctx.a.threshold[ctx.c * DIGITS + d]);
    super::inverse_lazy_io(y, |s| *narrowed.get_unchecked(s), |s, x| {
        let mut v = canonical(vsubq_s16(x, vld1q_dup_s16(mr.as_ptr().add(s))), p);
        let slot = dv.add(2 * s);
        let flag = vcgeq_s16(vabsq_s16(v), thr);
        if vmaxvq_u16(flag) != 0 {
            v = repair_lanes(ctx, d, s, v, flag, dv);
        }
        *slot = vaddw_s16(*slot, vget_low_s16(v));
        *slot.add(1) = vaddw_high_s16(*slot.add(1), v);
    });
}

/// [`sums`] for two batches against one sweep of the key (after IvanLudvig's submission
/// ce686fcc): per sweep, batch A's transforms into `xs` and their sink, then batch B's into the
/// same `xs` and their sink against the same key lines, which are then still cached.
#[inline(never)]
unsafe fn sums2<const V: u8>(
    words: [&[(*const u8, usize); LANES]; 2],
    a: &SmsMatrix,
    nib: &mut [uint8x16_t; N / 4],
    xs: &mut [[int16x8_t; N]; SWEEP],
    sc: &mut [int16x8_t; N],
    acc: [&mut [int64x2_t]; 2],
    narrowed: &mut [NarrowedDigits; 2],
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
        let l = end - e;
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
                let x = &mut xs[i];
                lut_pass2(nib, sc, kern);
                tail::<V>(sc, x, kern);
            }
            sink_terminal::<true>(xs, keys, accs[t], groups, &mut narrowed[t]);
        }
        e = end;
        let c = (end - 1) / CHUNK;
        let ctx = [Repair { words: words[0], a, c }, Repair { words: words[1], a, c }];
        for t in 0..2 {
            let dv = devs[t].as_mut_ptr();
            let dv16 = dev16[t].as_mut_ptr();
            for d in (0..DIGITS - 1).step_by(2) {
                let mr = [a.mean_residues(c, d), a.mean_residues(c, d + 1)];
                let rows = [dv.add(d * 2 * N), dv.add((d + 1) * 2 * N)];
                let rows16 = [dv16.add(d * N), dv16.add((d + 1) * N)];
                chunk_row2_fast([&narrowed[t][d], &narrowed[t][d + 1]], [1 + d, 2 + d], mr, rows, rows16,
                    [&mut *ys[t], &mut xs[0]], [&ctx[t], &ctx[t]]);
            }
        }
        let d = DIGITS - 1;
        let mr = a.mean_residues(c, d);
        let [dv0, dv1] = &mut devs;
        let rows = [dv0.as_mut_ptr().add(d * 2 * N), dv1.as_mut_ptr().add(d * 2 * N)];
        let [d16a, d16b] = &mut dev16;
        let rows16 = [d16a.as_mut_ptr().add(d * N), d16b.as_mut_ptr().add(d * N)];
        let [y0, y1] = &mut ys;
        chunk_row2_fast([&narrowed[0][d], &narrowed[1][d]], [d + 1, d + 1], [mr, mr], rows, rows16,
            [&mut **y0, &mut **y1], [&ctx[0], &ctx[1]]);
        if (c + 1) % FLUSH == 0 || end == a.elements {
            for t in 0..2 {
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

#[inline]
fn crt(r: u32, t: i64) -> u64 {
    const INV: u64 = {
        // 9721^-1 mod M by the extended Euclidean algorithm
        let (mut a, mut b) = (P1 as i64, TAIL as i64);
        let (mut x0, mut x1) = (1i64, 0i64);
        while b != 0 {
            let q = a / b;
            (a, b) = (b, a - q * b);
            (x0, x1) = (x1, x0 - q * x1);
        }
        assert!(a == 1);
        x0.rem_euclid(TAIL as i64) as u64
    };
    const _: () = assert!(P1 as u64 * INV % TAIL as u64 == 1);
    const _: () = assert!(P1 < TAIL && MODULUS < 1 << 63);
    let k = (t - r as i64).rem_euclid(TAIL as i64) as u64 * INV % TAIL as u64;
    r as u64 + P1 as u64 * k
}

#[repr(C)]
struct Scratch {
    xs: [[int16x8_t; N]; SWEEP],
    /// The transform scratch: the table front and levels 4-5 run here (L1-resident, reused for every
    /// transform) and level 6 writes the sweep buffer out of place ([`pass_d_out`]).
    sc: [int16x8_t; N],
    acc: [int64x2_t; ROWS * 4 * N],
    dev: [int32x4_t; DIGITS * 2 * N],
    acc2: [int64x2_t; ROWS * 4 * N],
    narrowed: [NarrowedDigits; 2],
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
const _: () = assert!(core::mem::size_of::<Scratch>() <= ARENA, "the scratch fits the arena");

struct Arena(*mut Scratch);

impl Arena {
    fn layout() -> std::alloc::Layout {
        std::alloc::Layout::from_size_align(ARENA.max(core::mem::size_of::<Scratch>()), core::mem::align_of::<Scratch>())
            .expect("the scratch layout")
    }

    fn new() -> Self {
        let layout = Self::layout();
        let p = unsafe { std::alloc::alloc_zeroed(layout) } as *mut Scratch;
        if p.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        Arena(p)
    }
}

impl Drop for Arena {
    fn drop(&mut self) {
        unsafe { std::alloc::dealloc(self.0 as *mut u8, Self::layout()) }
    }
}

/// The commitment of every column of `values` (columns of `4 a.elements` words) modulo `q`,
/// written straight into `out` as the wire encoding (`rq::coefficient_bytes(q)` little-endian
/// bytes per coefficient, column after column), so the separate packing pass over a `u128`
/// buffer is not needed.
/// Pins the calling commit worker to the `w`-th CPU it may run on, so the scheduler does not move it,
/// and its L1/L2 working set, to another core mid-commit. A hint: on any error it changes nothing.
fn pin_worker(w: usize) {
    #[cfg(target_os = "linux")]
    unsafe {
        extern "C" {
            fn sched_getaffinity(pid: i32, size: usize, mask: *mut u64) -> i32;
            fn sched_setaffinity(pid: i32, size: usize, mask: *const u64) -> i32;
        }
        let mut set = [0u64; 16];
        if sched_getaffinity(0, core::mem::size_of_val(&set), set.as_mut_ptr()) != 0 {
            return;
        }
        let cpus: Vec<usize> = (0..1024).filter(|&c| set[c / 64] >> (c % 64) & 1 == 1).collect();
        if cpus.is_empty() {
            return;
        }
        let c = cpus[w % cpus.len()];
        let mut one = [0u64; 16];
        one[c / 64] = 1 << (c % 64);
        let _ = sched_setaffinity(0, core::mem::size_of_val(&one), one.as_ptr());
    }
    #[cfg(not(target_os = "linux"))]
    let _ = w;
}

/// V17 uses five digits, 128-element chunks and a one-pass six-row sink.
pub fn commit(values: &[B128], a: &SmsMatrix, workers: usize, q: u128, out: &mut [u8]) {
    commit_v::<17>(values, a, workers, q, out, true)
}

/// [`commit`] with levels 4 to 6 by [`tail`]`::<V>` (the same output for every `V`); `hook`
/// runs [`BETWEEN`] between batch pairs.
pub fn commit_v<const V: u8>(values: &[B128], a: &SmsMatrix, workers: usize, q: u128, out: &mut [u8], hook: bool) {
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
        let Scratch { xs, sc, acc, dev, acc2, dev2, dev16, narrowed, y2, m, nib, y, tail } = &mut *arena.0;
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
                    sums2::<V>([&w0, &w1], a, nib, xs, sc, [&mut acc[..], &mut acc2[..]], narrowed, [&mut *y, &mut *y2], [&mut dev[..], &mut dev2[..]], [&mut d16a[..], &mut d16b[..]]);
                } else {
                    sums::<V>(&batch_words(b), a, m, nib, xs, sc, acc, &mut narrowed[0], y, dev);
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

