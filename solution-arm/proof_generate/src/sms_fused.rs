//! Register-resident last radix-3 stage and SMS multiply-accumulate.
//! Keeps all three outputs of eight transforms in vector registers. No completed
//! transform is materialized. Arithmetic, key layout and statistical bounds are unchanged.
use super::*;

#[inline(never)]
pub(super) unsafe fn tile(
    xs: &[[int16x8_t; N]; PASS],
    first: usize,
    keys: *const i16,
    acc: &mut [int64x2_t],
    kern: &Kernel,
) {
    const {
        assert!(PB == 1 && SCHEDULES[PB].before[6] == FIRST13 && !SCHEDULES[PB].reduce_d);
    }
    debug_assert!(first + LOCK <= PASS);
    debug_assert_eq!(acc.len(), N * ROWS * 4);
    let k = vld1q_s16(kern.k.as_ptr());
    for node in 0..N / 3 {
        // Each d row covers two consecutive nodes. Duplicate the selected half
        // so the same four by-element operands serve either node.
        let row = &kern.d[node / 2];
        let off = 4 * (node % 2);
        let four = vld1_s16(row.as_ptr().add(off));
        let tw = vcombine_s16(four, four);
        let ap = acc.as_mut_ptr().add(idx(3 * node, 0));
        let kp = keys.add(3 * node * ROWS * LOCK);
        core::arch::asm!(
            "ldp q8, q9, [{p0}]",
            "ldr q10, [{p0}, #32]",
            "ldp q11, q12, [{p1}]",
            "ldr q13, [{p1}, #32]",
            "ldp q14, q15, [{p2}]",
            "ldr q16, [{p2}, #32]",
            "ldp q17, q18, [{p3}]",
            "ldr q19, [{p3}, #32]",
            "ldp q20, q21, [{p4}]",
            "ldr q22, [{p4}, #32]",
            "ldp q23, q24, [{p5}]",
            "ldr q25, [{p5}, #32]",
            "ldp q26, q27, [{p6}]",
            "ldr q28, [{p6}, #32]",
            "ldp q29, q30, [{p7}]",
            "ldr q31, [{p7}, #32]",
            "srshr v0.8h, v8.8h, #13",
            "mls v8.8h, v0.8h, v6.h[0]",
            "mul v0.8h, v9.8h, v7.h[0]",
            "sqrdmulh v1.8h, v9.8h, v7.h[1]",
            "mls v0.8h, v1.8h, v6.h[0]",
            "mul v1.8h, v10.8h, v7.h[2]",
            "sqrdmulh v2.8h, v10.8h, v7.h[3]",
            "mls v1.8h, v2.8h, v6.h[0]",
            "sub v2.8h, v0.8h, v1.8h",
            "mul v3.8h, v2.8h, v6.h[2]",
            "sqrdmulh v2.8h, v2.8h, v6.h[3]",
            "mls v3.8h, v2.8h, v6.h[0]",
            "add v9.8h, v0.8h, v1.8h",
            "add v9.8h, v8.8h, v9.8h",
            "sub v10.8h, v8.8h, v1.8h",
            "add v10.8h, v10.8h, v3.8h",
            "sub v8.8h, v8.8h, v0.8h",
            "sub v8.8h, v8.8h, v3.8h",
            "srshr v0.8h, v11.8h, #13",
            "mls v11.8h, v0.8h, v6.h[0]",
            "mul v0.8h, v12.8h, v7.h[0]",
            "sqrdmulh v1.8h, v12.8h, v7.h[1]",
            "mls v0.8h, v1.8h, v6.h[0]",
            "mul v1.8h, v13.8h, v7.h[2]",
            "sqrdmulh v2.8h, v13.8h, v7.h[3]",
            "mls v1.8h, v2.8h, v6.h[0]",
            "sub v2.8h, v0.8h, v1.8h",
            "mul v3.8h, v2.8h, v6.h[2]",
            "sqrdmulh v2.8h, v2.8h, v6.h[3]",
            "mls v3.8h, v2.8h, v6.h[0]",
            "add v12.8h, v0.8h, v1.8h",
            "add v12.8h, v11.8h, v12.8h",
            "sub v13.8h, v11.8h, v1.8h",
            "add v13.8h, v13.8h, v3.8h",
            "sub v11.8h, v11.8h, v0.8h",
            "sub v11.8h, v11.8h, v3.8h",
            "srshr v0.8h, v14.8h, #13",
            "mls v14.8h, v0.8h, v6.h[0]",
            "mul v0.8h, v15.8h, v7.h[0]",
            "sqrdmulh v1.8h, v15.8h, v7.h[1]",
            "mls v0.8h, v1.8h, v6.h[0]",
            "mul v1.8h, v16.8h, v7.h[2]",
            "sqrdmulh v2.8h, v16.8h, v7.h[3]",
            "mls v1.8h, v2.8h, v6.h[0]",
            "sub v2.8h, v0.8h, v1.8h",
            "mul v3.8h, v2.8h, v6.h[2]",
            "sqrdmulh v2.8h, v2.8h, v6.h[3]",
            "mls v3.8h, v2.8h, v6.h[0]",
            "add v15.8h, v0.8h, v1.8h",
            "add v15.8h, v14.8h, v15.8h",
            "sub v16.8h, v14.8h, v1.8h",
            "add v16.8h, v16.8h, v3.8h",
            "sub v14.8h, v14.8h, v0.8h",
            "sub v14.8h, v14.8h, v3.8h",
            "srshr v0.8h, v17.8h, #13",
            "mls v17.8h, v0.8h, v6.h[0]",
            "mul v0.8h, v18.8h, v7.h[0]",
            "sqrdmulh v1.8h, v18.8h, v7.h[1]",
            "mls v0.8h, v1.8h, v6.h[0]",
            "mul v1.8h, v19.8h, v7.h[2]",
            "sqrdmulh v2.8h, v19.8h, v7.h[3]",
            "mls v1.8h, v2.8h, v6.h[0]",
            "sub v2.8h, v0.8h, v1.8h",
            "mul v3.8h, v2.8h, v6.h[2]",
            "sqrdmulh v2.8h, v2.8h, v6.h[3]",
            "mls v3.8h, v2.8h, v6.h[0]",
            "add v18.8h, v0.8h, v1.8h",
            "add v18.8h, v17.8h, v18.8h",
            "sub v19.8h, v17.8h, v1.8h",
            "add v19.8h, v19.8h, v3.8h",
            "sub v17.8h, v17.8h, v0.8h",
            "sub v17.8h, v17.8h, v3.8h",
            "srshr v0.8h, v20.8h, #13",
            "mls v20.8h, v0.8h, v6.h[0]",
            "mul v0.8h, v21.8h, v7.h[0]",
            "sqrdmulh v1.8h, v21.8h, v7.h[1]",
            "mls v0.8h, v1.8h, v6.h[0]",
            "mul v1.8h, v22.8h, v7.h[2]",
            "sqrdmulh v2.8h, v22.8h, v7.h[3]",
            "mls v1.8h, v2.8h, v6.h[0]",
            "sub v2.8h, v0.8h, v1.8h",
            "mul v3.8h, v2.8h, v6.h[2]",
            "sqrdmulh v2.8h, v2.8h, v6.h[3]",
            "mls v3.8h, v2.8h, v6.h[0]",
            "add v21.8h, v0.8h, v1.8h",
            "add v21.8h, v20.8h, v21.8h",
            "sub v22.8h, v20.8h, v1.8h",
            "add v22.8h, v22.8h, v3.8h",
            "sub v20.8h, v20.8h, v0.8h",
            "sub v20.8h, v20.8h, v3.8h",
            "srshr v0.8h, v23.8h, #13",
            "mls v23.8h, v0.8h, v6.h[0]",
            "mul v0.8h, v24.8h, v7.h[0]",
            "sqrdmulh v1.8h, v24.8h, v7.h[1]",
            "mls v0.8h, v1.8h, v6.h[0]",
            "mul v1.8h, v25.8h, v7.h[2]",
            "sqrdmulh v2.8h, v25.8h, v7.h[3]",
            "mls v1.8h, v2.8h, v6.h[0]",
            "sub v2.8h, v0.8h, v1.8h",
            "mul v3.8h, v2.8h, v6.h[2]",
            "sqrdmulh v2.8h, v2.8h, v6.h[3]",
            "mls v3.8h, v2.8h, v6.h[0]",
            "add v24.8h, v0.8h, v1.8h",
            "add v24.8h, v23.8h, v24.8h",
            "sub v25.8h, v23.8h, v1.8h",
            "add v25.8h, v25.8h, v3.8h",
            "sub v23.8h, v23.8h, v0.8h",
            "sub v23.8h, v23.8h, v3.8h",
            "srshr v0.8h, v26.8h, #13",
            "mls v26.8h, v0.8h, v6.h[0]",
            "mul v0.8h, v27.8h, v7.h[0]",
            "sqrdmulh v1.8h, v27.8h, v7.h[1]",
            "mls v0.8h, v1.8h, v6.h[0]",
            "mul v1.8h, v28.8h, v7.h[2]",
            "sqrdmulh v2.8h, v28.8h, v7.h[3]",
            "mls v1.8h, v2.8h, v6.h[0]",
            "sub v2.8h, v0.8h, v1.8h",
            "mul v3.8h, v2.8h, v6.h[2]",
            "sqrdmulh v2.8h, v2.8h, v6.h[3]",
            "mls v3.8h, v2.8h, v6.h[0]",
            "add v27.8h, v0.8h, v1.8h",
            "add v27.8h, v26.8h, v27.8h",
            "sub v28.8h, v26.8h, v1.8h",
            "add v28.8h, v28.8h, v3.8h",
            "sub v26.8h, v26.8h, v0.8h",
            "sub v26.8h, v26.8h, v3.8h",
            "srshr v0.8h, v29.8h, #13",
            "mls v29.8h, v0.8h, v6.h[0]",
            "mul v0.8h, v30.8h, v7.h[0]",
            "sqrdmulh v1.8h, v30.8h, v7.h[1]",
            "mls v0.8h, v1.8h, v6.h[0]",
            "mul v1.8h, v31.8h, v7.h[2]",
            "sqrdmulh v2.8h, v31.8h, v7.h[3]",
            "mls v1.8h, v2.8h, v6.h[0]",
            "sub v2.8h, v0.8h, v1.8h",
            "mul v3.8h, v2.8h, v6.h[2]",
            "sqrdmulh v2.8h, v2.8h, v6.h[3]",
            "mls v3.8h, v2.8h, v6.h[0]",
            "add v30.8h, v0.8h, v1.8h",
            "add v30.8h, v29.8h, v30.8h",
            "sub v31.8h, v29.8h, v1.8h",
            "add v31.8h, v31.8h, v3.8h",
            "sub v29.8h, v29.8h, v0.8h",
            "sub v29.8h, v29.8h, v3.8h",
            "mov {rows}, #{row_count}",
            "2:",
            "ldp q0, q1, [{ap}]",
            "ldp q2, q3, [{ap}, #32]",
            "ldr q4, [{kp}], #16",
            "sdot z0.d, z9.h, z4.h[0]",
            "sdot z1.d, z12.h, z4.h[0]",
            "sdot z2.d, z15.h, z4.h[0]",
            "sdot z3.d, z18.h, z4.h[0]",
            "sdot z0.d, z21.h, z4.h[1]",
            "sdot z1.d, z24.h, z4.h[1]",
            "sdot z2.d, z27.h, z4.h[1]",
            "sdot z3.d, z30.h, z4.h[1]",
            "stp q0, q1, [{ap}]",
            "stp q2, q3, [{ap}, #32]",
            "add {ap}, {ap}, #64",
            "subs {rows}, {rows}, #1",
            "b.ne 2b",
            "mov {rows}, #{row_count}",
            "2:",
            "ldp q0, q1, [{ap}]",
            "ldp q2, q3, [{ap}, #32]",
            "ldr q4, [{kp}], #16",
            "sdot z0.d, z10.h, z4.h[0]",
            "sdot z1.d, z13.h, z4.h[0]",
            "sdot z2.d, z16.h, z4.h[0]",
            "sdot z3.d, z19.h, z4.h[0]",
            "sdot z0.d, z22.h, z4.h[1]",
            "sdot z1.d, z25.h, z4.h[1]",
            "sdot z2.d, z28.h, z4.h[1]",
            "sdot z3.d, z31.h, z4.h[1]",
            "stp q0, q1, [{ap}]",
            "stp q2, q3, [{ap}, #32]",
            "add {ap}, {ap}, #64",
            "subs {rows}, {rows}, #1",
            "b.ne 2b",
            "mov {rows}, #{row_count}",
            "2:",
            "ldp q0, q1, [{ap}]",
            "ldp q2, q3, [{ap}, #32]",
            "ldr q4, [{kp}], #16",
            "sdot z0.d, z8.h, z4.h[0]",
            "sdot z1.d, z11.h, z4.h[0]",
            "sdot z2.d, z14.h, z4.h[0]",
            "sdot z3.d, z17.h, z4.h[0]",
            "sdot z0.d, z20.h, z4.h[1]",
            "sdot z1.d, z23.h, z4.h[1]",
            "sdot z2.d, z26.h, z4.h[1]",
            "sdot z3.d, z29.h, z4.h[1]",
            "stp q0, q1, [{ap}]",
            "stp q2, q3, [{ap}, #32]",
            "add {ap}, {ap}, #64",
            "subs {rows}, {rows}, #1",
            "b.ne 2b",
            p0 = in(reg) xs[first + 0].as_ptr().add(3 * node),
            p1 = in(reg) xs[first + 1].as_ptr().add(3 * node),
            p2 = in(reg) xs[first + 2].as_ptr().add(3 * node),
            p3 = in(reg) xs[first + 3].as_ptr().add(3 * node),
            p4 = in(reg) xs[first + 4].as_ptr().add(3 * node),
            p5 = in(reg) xs[first + 5].as_ptr().add(3 * node),
            p6 = in(reg) xs[first + 6].as_ptr().add(3 * node),
            p7 = in(reg) xs[first + 7].as_ptr().add(3 * node),
            ap = inout(reg) ap => _, kp = inout(reg) kp => _,
            rows = out(reg) _, row_count = const ROWS,
            in("v6") k, in("v7") tw,
            out("v0") _, out("v1") _, out("v2") _, out("v3") _,
            out("v4") _, out("v5") _,
            out("v8") _, out("v9") _, out("v10") _, out("v11") _,
            out("v12") _, out("v13") _, out("v14") _, out("v15") _,
            out("v16") _, out("v17") _, out("v18") _, out("v19") _,
            out("v20") _, out("v21") _, out("v22") _, out("v23") _,
            out("v24") _, out("v25") _, out("v26") _, out("v27") _,
            out("v28") _, out("v29") _, out("v30") _, out("v31") _,
            options(nostack)
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn state(x: &mut u64) -> u64 {
        *x ^= *x << 13;
        *x ^= *x >> 7;
        *x ^= *x << 17;
        *x
    }

    #[test]
    fn fused_last_stage_matches_separate_stage_and_scalar_dot() {
        unsafe {
            let kern = &kernels()[PB];
            let mut seed = 0x0ab73fd9215u64;
            for case in 0..5 {
                let words: Vec<u128> = (0..LANES * PASS * 4)
                    .map(|_| {
                        if case == 0 {
                            0
                        } else if case == 1 {
                            u128::MAX
                        } else {
                            (state(&mut seed) as u128) << 64 | state(&mut seed) as u128
                        }
                    })
                    .collect();
                let pointers: [(*const u8, usize); LANES] = core::array::from_fn(|c| {
                    if case == 4 && c >= 3 {
                        (ZERO_WORDS.as_ptr() as *const u8, 0)
                    } else {
                        (words.as_ptr().add(c * PASS * 4) as *const u8, 64)
                    }
                });
                let mut xs = Box::new([[vdupq_n_s16(0); N]; PASS]);
                let mut nib = Box::new([vdupq_n_u8(0); N / 4]);
                for t in 0..PASS {
                    let at = pass_words(&pointers, PASS, 0, t);
                    indices(&at, &mut nib);
                    pass_lut::<PB>(&nib, &mut xs[t], kern);
                    pass_c::<PB>(&mut xs[t], kern);
                }
                let mut finished = xs.clone();
                for x in finished.iter_mut() {
                    pass_d::<PB, false>(x, kern, &[], &mut []);
                }
                let keys: Vec<i16> = (0..2 * N * ROWS * LOCK)
                    .map(|i| {
                        if case == 2 {
                            if i % 2 == 0 {
                                4860
                            } else {
                                -4860
                            }
                        } else {
                            (state(&mut seed) % P1 as u64) as i16 - 4860
                        }
                    })
                    .collect();
                let mut wanted = vec![vdupq_n_s64(0); N * ROWS * 4];
                // Independent scalar definition of the exact lane-order dot product.
                let finished = &*(finished.as_ptr() as *const [[[i16; LANES]; N]; PASS]);
                let expected = core::slice::from_raw_parts_mut(
                    wanted.as_mut_ptr() as *mut [i64; 2],
                    wanted.len(),
                );
                for s in 0..N {
                    for r in 0..ROWS {
                        for col in 0..LANES {
                            let mut sum = 0i64;
                            for e in 0..PASS {
                                let t = 4 * (e / 4) + col / 2;
                                let lane = 4 * (col % 2) + e % 4;
                                let key = keys[((e / LOCK * N + s) * ROWS + r) * LOCK + e % LOCK];
                                sum += i64::from(finished[t][s][lane]) * i64::from(key);
                            }
                            expected[idx(s, r) + col / 2][col % 2] = sum;
                        }
                    }
                }
                let mut got = vec![vdupq_n_s64(0); N * ROWS * 4];
                tile(&xs, 0, keys.as_ptr(), &mut got, kern);
                tile(
                    &xs,
                    LOCK,
                    keys.as_ptr().add(N * ROWS * LOCK),
                    &mut got,
                    kern,
                );
                let got = core::slice::from_raw_parts(got.as_ptr() as *const [i64; 2], got.len());
                assert_eq!(got, expected, "case {case}");
                // Test accumulation into nonzero state as well as the zero initial state.
                tile(&xs, 0, keys.as_ptr(), &mut wanted, kern);
                tile(
                    &xs,
                    LOCK,
                    keys.as_ptr().add(N * ROWS * LOCK),
                    &mut wanted,
                    kern,
                );
                let twice =
                    core::slice::from_raw_parts(wanted.as_ptr() as *const [i64; 2], wanted.len());
                for (a, b) in twice.iter().zip(got) {
                    assert_eq!(*a, [2 * b[0], 2 * b[1]]);
                }
            }
        }
    }
    #[test]
    fn fused_commit_short_and_chunk_boundary() {
        let mut seed = 0x4313dd0761u64;
        for (elements, columns) in [(9usize, 3usize), (225, 3)] {
            let values: Vec<B128> = (0..4 * elements * columns)
                .map(|_| B128((state(&mut seed) as u128) << 64 | state(&mut seed) as u128))
                .collect();
            let public = crate::rq::ajtai_matrix(MODULUS, elements);
            let want = crate::commit_reference(&values, &public, MODULUS);
            let a = matrix(elements, 3);
            for workers in [1, 3] {
                assert_eq!(
                    commit(&values, &a, workers),
                    want,
                    "elements {elements}, columns {columns}, workers {workers}"
                );
            }
        }
    }
}
