// Production inverse snapshot.
use super::*;

/// [`InverseKernel`]'s twiddles without the `1 / r` factors, for [`inverse_lazy`]: per radix-3
/// node, levels 6 down to 3 (216, 72, 24, 8 nodes), `[tau^-1, tau^-2, eps^-1]` as pairs; per
/// radix-2 node, levels 2 and 1 (4, 2 nodes), `[tau^-1]`; and the trinomial unsplit's
/// `[inv_split, zeta^324, inv_split * zeta^324]`.
pub struct LazyInverse {
    r3: Vec<[i16; 8]>,
    r2: Vec<[i16; 8]>,
    u: [i16; 8],
}

impl LazyInverse {
    fn new(index: usize) -> LazyInverse {
        use super::super::{ORDER, RADICES};
        let plan = &primes()[index];
        let p = plan.p;
        let (mut r3, mut r2) = (Vec::new(), Vec::new());
        for level in (0..RADICES.len()).rev() {
            let r = RADICES[level];
            for &e in &plan.exps[level] {
                let tau = e as usize / r;
                let mut row = [0i16; 8];
                for i in 1..r {
                    row[2 * (i - 1)..2 * i].copy_from_slice(&twiddle(p, plan.root(ORDER - tau * i % ORDER)));
                }
                if r == 3 {
                    row[4..6].copy_from_slice(&twiddle(p, plan.root(ORDER - ORDER / 3)));
                    r3.push(row);
                } else {
                    r2.push(row);
                }
            }
        }
        assert!(r3.len() == 216 + 72 + 24 + 8 && r2.len() == 4 + 2);
        let mut u = [0i16; 8];
        u[..2].copy_from_slice(&twiddle(p, plan.inv_split as u64));
        let w = plan.root(super::super::HALF);
        u[2..4].copy_from_slice(&twiddle(p, w));
        u[4..6].copy_from_slice(&twiddle(p, plan.inv_split as u64 * w % p as u64));
        LazyInverse { r3, r2, u }
    }
}

pub fn lazy_inverses() -> &'static [LazyInverse; 3] {
    static K: OnceLock<[LazyInverse; 3]> = OnceLock::new();
    K.get_or_init(|| [LazyInverse::new(0), LazyInverse::new(1), LazyInverse::new(2)])
}

/// The factor [`inverse_lazy`]'s output carries: `3^4 2^2`, the radices it does not divide by.
pub const LAZY_SCALE: u64 = 324;

/// `a - round(a / 2^13) p`: `|result| <= 0.18664 |a| + p / 2` for `p = 9721`.
#[inline(always)]
unsafe fn r13(a: int16x8_t, k: int16x8_t) -> int16x8_t {
    vmlsq_laneq_s16::<0>(a, vrshrq_n_s16::<13>(a), k)
}

/// The radix-3 step of [`inverse_lazy`] on inputs `|y| <= B`, `t` a node's twiddles:
/// `a = y0 + y1 + y2` (cheaply reduced), `b = tau^-1 (y0 + eps^-1 y1 + eps^-2 y2)`,
/// `c = tau^-2 (y0 + eps^-2 y1 + eps^-1 y2)`, `eps^-2 = -1 - eps^-1`: three times the inverse
/// butterfly, with three Barrett products instead of four.
#[inline(always)]
unsafe fn ib3(y0: int16x8_t, y1: int16x8_t, y2: int16x8_t, t: int16x8_t, k: int16x8_t, raw_sum: bool) -> (int16x8_t, int16x8_t, int16x8_t) {
    let sum = vaddq_s16(vaddq_s16(y0, y1), y2);
    let a = if raw_sum { sum } else { r13(sum, k) };
    let e = mulc::<4, 5>(vsubq_s16(y1, y2), t, k);
    let b = mulc::<0, 1>(vaddq_s16(vsubq_s16(y0, y2), e), t, k);
    let c = mulc::<2, 3>(vsubq_s16(vsubq_s16(y0, y1), e), t, k);
    (a, b, c)
}

/// The radix-2 step: `a = y0 + y1` (cheaply reduced), `b = tau^-1 (y0 - y1)`: twice the inverse.
#[inline(always)]
unsafe fn ib2(y0: int16x8_t, y1: int16x8_t, t: int16x8_t, k: int16x8_t) -> (int16x8_t, int16x8_t) {
    (r13(vaddq_s16(y0, y1), k), mulc::<0, 1>(vsubq_s16(y0, y1), t, k))
}

/// The digit route lets the caller subtract its centered mean before the final
/// reduction. At this step the two inputs have magnitude at most 8196. For
/// `d = u - z`, the mapped signed bounds are `|a| <= 13054`, `|b| <= 25048`.
/// After subtracting any centered mean they still fit i16.
#[inline(always)]
unsafe fn lazy_unsplit<const DIGIT: bool>(
    u: int16x8_t, z: int16x8_t, uv: int16x8_t, k: int16x8_t,
) -> (int16x8_t, int16x8_t) {
    let d = vsubq_s16(u, z);
    if DIGIT {
        // w has order six and w^5 = 1 - w. For D = w - w^5,
        // (2w - D)/D = 1/D, so 2t - d is the second coefficient.
        let t = mulc::<4, 5>(d, uv, k);
        (vsubq_s16(u, t), vsubq_s16(vaddq_s16(t, t), d))
    } else {
        let b = mulc::<0, 1>(d, uv, k);
        let a = r13(vsubq_s16(u, mulc::<2, 3>(b, uv, k)), k);
        (a, b)
    }
}


/// Inverse of the last pair of radix-2 levels and the unsplit, per digit group.
/// The leaf roots are those of X^8 - X^4 + 1. Pair opposite roots first.
/// Eight input reductions keep the even combinations and the odd Hadamard
/// sums in range. The even branch uses three products, the odd branch four.
/// The digit accumulator and the first four inverse levels guarantee
/// |input| <= 10,212. All results fit i16 after subtracting a centered mean.
/// Use the normal full-word canonical reduction in the output callback.
#[inline(always)]
unsafe fn digit_inverse_eight(
    [v0, v1, v2, v3, v4, v5, v6, v7]: [int16x8_t; 8],
    k: int16x8_t,
) -> [int16x8_t; 8] {
    let t0 = vld1q_s16([1283, 4325, -2362, -7962, -2203, -7426, 2653, 8943].as_ptr());
    let t1 = vld1q_s16([4476, 15088, -4347, -14653, -2512, -8468, 4861, 0].as_ptr());
    let v8 = vaddq_s16(v0, v1);
    let v9 = r13(v8, k);
    let v10 = vaddq_s16(v2, v3);
    let v11 = r13(v10, k);
    let v12 = vaddq_s16(v4, v5);
    let v13 = r13(v12, k);
    let v14 = vaddq_s16(v6, v7);
    let v15 = r13(v14, k);
    let v16 = vsubq_s16(v0, v1);
    let v17 = r13(v16, k);
    let v18 = vsubq_s16(v2, v3);
    let v19 = r13(v18, k);
    let v20 = vsubq_s16(v4, v5);
    let v21 = r13(v20, k);
    let v22 = vsubq_s16(v6, v7);
    let v23 = r13(v22, k);
    let v24 = vaddq_s16(v9, v11);
    let v25 = vaddq_s16(v13, v15);
    let v28 = vsubq_s16(v24, v25);
    let v29 = mulc::<0, 1>(v28, t0, k);
    let v30 = vaddq_s16(v24, v25);
    let odd = vandq_s16(veorq_s16(v30, v29), vdupq_n_s16(1));
    let mut half = vhsubq_s16(v30, v29);
    core::arch::asm!(
        "mla {half:v}.8h, {odd:v}.8h, {t:v}.h[6]",
        half = inout(vreg) half, odd = in(vreg) odd, t = in(vreg_low16) t1,
        options(pure, nomem, nostack, preserves_flags)
    );
    let v32 = half;
    let v33 = vsubq_s16(v9, v11);
    let v35 = vsubq_s16(v13, v15);
    let v37 = vsubq_s16(v33, v35);
    let v38 = mulc::<2, 3>(v37, t0, k);
    let v39 = vaddq_s16(v38, v38);
    let v40 = vaddq_s16(v33, v35);
    let v41 = mulc::<4, 5>(v40, t0, k);
    let v42 = vsubq_s16(v41, v38);
    let v43 = vaddq_s16(v17, v19);
    let v45 = vaddq_s16(v21, v23);
    let v47 = vsubq_s16(v17, v19);
    let v49 = vsubq_s16(v21, v23);
    let v51 = vaddq_s16(v43, v45);
    let v52 = vaddq_s16(v47, v49);
    let v53 = vsubq_s16(v43, v45);
    let v54 = vsubq_s16(v47, v49);
    let v55 = mulc::<6, 7>(v51, t0, k);
    let v56 = mulc::<0, 1>(v52, t1, k);
    let v57 = mulc::<2, 3>(v53, t1, k);
    let v58 = mulc::<4, 5>(v54, t1, k);
    let v59 = vaddq_s16(v56, v57);
    let v60 = vaddq_s16(v55, v58);
    let v61 = vaddq_s16(v59, v60);
    let v62 = vsubq_s16(v59, v60);
    let v63 = vsubq_s16(v56, v57);
    let v64 = vsubq_s16(v55, v58);
    let v65 = vaddq_s16(v63, v64);
    let v66 = vsubq_s16(v55, v56);
    let v67 = vaddq_s16(v66, v66);
    [v32, v61, v39, v62, v29, v65, v42, v67]
}

/// The inverse transform times [`LAZY_SCALE`] for the prime 9721, on eight lanes: inputs
/// `|x| <= p`, outputs congruent to `324` times [`inverse`]'s with `|x| <= p` (not centered).
/// Four passes: levels 6 and 5 per level-4 node, level 4, level 3, then levels 2, 1 and the
/// trinomial unsplit per group of eight slots `j + 81 b`, each slot loaded and stored once per
/// pass. Bounds (inputs `<= 9721`, `r13`'s `0.18664 |a| + 4860.5`, [`super::product_bound`]):
/// the radix-3 sums reach 29163, 30912, 31890, 32439 and the outputs 10304, 10630, 10813,
/// 10915; the radix-2 sums 21830 and 17870, outputs 8935 and 8196; the unsplit outputs 7499
/// and 7292. The test `lazy_inverse_matches` checks the congruence and the output bound.
#[inline(never)]
pub unsafe fn inverse_lazy(x: &mut [int16x8_t; N]) {
    let at = x.as_mut_ptr();
    inverse_lazy_io(x, |i| *at.add(i), |i, v| *at.add(i) = v);
}

/// [`inverse_lazy`] with its first pass reading slot `i` from `input(i)` instead of `x`, and its
/// last pass handing slot `i` to `output(i, value)` instead of storing it: `x` is the scratch of
/// the middle passes. `input` must return values `|x| <= p`; `output` sees every slot once.
#[inline(always)]
pub unsafe fn inverse_lazy_io(x: &mut [int16x8_t; N], input: impl FnMut(usize) -> int16x8_t, output: impl FnMut(usize, int16x8_t)) {
    inverse_lazy_io_mode::<false>(x, input, output);
}

/// With DIGIT, the terminal accumulation bounds each input by 5459.
/// The first butterfly of a nine-slot group can keep its raw sum.
/// The callback subtracts a centered mean and centers the result.
#[inline(always)]
pub(in super::super) unsafe fn inverse_lazy_io_mode<const DIGIT: bool>(
    x: &mut [int16x8_t; N], mut input: impl FnMut(usize) -> int16x8_t,
    mut output: impl FnMut(usize, int16x8_t),
) {
    const { assert!(PRIMES[1] == 9721) };
    let k = vld1q_s16(kernels()[1].k.as_ptr());
    let inv = &lazy_inverses()[1];
    let (r3, r2) = (&inv.r3[..], &inv.r2[..]);
    // levels 6 and 5, per level-4 node m (slots 9 m..9 m + 9)
    for m in 0..72 {
        let base = 9 * m;
        let mut v: [int16x8_t; 9] = core::array::from_fn(|i| input(base + i));
        for g in 0..3 {
            let t = vld1q_s16(r3.get_unchecked(3 * m + g).as_ptr());
            (v[3 * g], v[3 * g + 1], v[3 * g + 2]) = ib3(v[3 * g], v[3 * g + 1], v[3 * g + 2], t, k, DIGIT && g == 0);
        }
        let t = vld1q_s16(r3.get_unchecked(216 + m).as_ptr());
        for j in 0..3 {
            (v[j], v[j + 3], v[j + 6]) = ib3(v[j], v[j + 3], v[j + 6], t, k, false);
        }
        for (i, &y) in v.iter().enumerate() {
            store(x, base + i, y);
        }
    }
    // level 4, per level-3 node n (27 slots): (j, j + 9, j + 18)
    for n in 0..24 {
        let t = vld1q_s16(r3.get_unchecked(288 + n).as_ptr());
        let base = 27 * n;
        for j0 in (0..9).step_by(3) {
            let v: [[int16x8_t; 3]; 3] = core::array::from_fn(|q| core::array::from_fn(|i| load(x, base + j0 + q + 9 * i)));
            let y: [(int16x8_t, int16x8_t, int16x8_t); 3] = core::array::from_fn(|q| ib3(v[q][0], v[q][1], v[q][2], t, k, false));
            for (q, &(a, b, c)) in y.iter().enumerate() {
                store(x, base + j0 + q, a);
                store(x, base + j0 + q + 9, b);
                store(x, base + j0 + q + 18, c);
            }
        }
    }
    // level 3, per level-2 node n (81 slots): (j, j + 27, j + 54)
    for n in 0..8 {
        let t = vld1q_s16(r3.get_unchecked(312 + n).as_ptr());
        let base = 81 * n;
        for j0 in (0..27).step_by(3) {
            let v: [[int16x8_t; 3]; 3] = core::array::from_fn(|q| core::array::from_fn(|i| load(x, base + j0 + q + 27 * i)));
            let y: [(int16x8_t, int16x8_t, int16x8_t); 3] = core::array::from_fn(|q| ib3(v[q][0], v[q][1], v[q][2], t, k, false));
            for (q, &(a, b, c)) in y.iter().enumerate() {
                store(x, base + j0 + q, a);
                store(x, base + j0 + q + 27, b);
                store(x, base + j0 + q + 54, c);
            }
        }
    }
    if DIGIT {
        for j in 0..81 {
            let input = core::array::from_fn(|i| load(x, j + 81 * i));
            for (i, y) in digit_inverse_eight(input, k).into_iter().enumerate() {
                output(j + 81 * i, y);
            }
        }
        return;
    }
    // levels 2 and 1 and the unsplit, per j < 81: slots j + 81 i, i < 8
    let t2: [int16x8_t; 4] = core::array::from_fn(|n| vld1q_s16(r2.get_unchecked(n).as_ptr()));
    let t1: [int16x8_t; 2] = core::array::from_fn(|n| vld1q_s16(r2.get_unchecked(4 + n).as_ptr()));
    let uv = vld1q_s16(inv.u.as_ptr());
    for j in 0..81 {
        let mut v: [int16x8_t; 8] = core::array::from_fn(|i| load(x, j + 81 * i));
        // level 2: node n pairs slots 162 n + j and 162 n + j + 81
        for n in 0..4 {
            (v[2 * n], v[2 * n + 1]) = ib2(v[2 * n], v[2 * n + 1], t2[n], k);
        }
        // level 1: node n pairs 324 n + i and 324 n + i + 162, i = j, j + 81
        for n in 0..2 {
            for h in 0..2 {
                (v[4 * n + h], v[4 * n + h + 2]) = ib2(v[4 * n + h], v[4 * n + h + 2], t1[n], k);
            }
        }
        // the unsplit: b = (u - v) inv_split, a = u - zeta^324 b, pairs i and i + 324
        for i in 0..4 {
            let (u, w) = (v[i], v[i + 4]);
            (v[i], v[i + 4]) = lazy_unsplit::<DIGIT>(u, w, uv, k);
        }
        for (i, &y) in v.iter().enumerate() {
            output(j + 81 * i, y);
        }
    }
}

#[inline(always)]
pub unsafe fn inverse_lazy_io2(
    x: [&mut [int16x8_t; N]; 2],
    input: impl FnMut(usize, usize) -> int16x8_t,
    output: impl FnMut(usize, usize, int16x8_t),
) {
    inverse_lazy_io2_mode::<false>(x, input, output);
}

/// Paired form of [`inverse_lazy_io_mode`].
#[inline(always)]
pub(in super::super) unsafe fn inverse_lazy_io2_mode<const DIGIT: bool>(
    x: [&mut [int16x8_t; N]; 2],
    mut input: impl FnMut(usize, usize) -> int16x8_t,
    mut output: impl FnMut(usize, usize, int16x8_t),
) {
    const { assert!(PRIMES[1] == 9721) };
    let k = vld1q_s16(kernels()[1].k.as_ptr());
    let inv = &lazy_inverses()[1];
    let (r3, r2) = (&inv.r3[..], &inv.r2[..]);
    let [x0, x1] = x;
    let xs: [*mut int16x8_t; 2] = [x0.as_mut_ptr(), x1.as_mut_ptr()];
    // levels 6 and 5, per level-4 node m (slots 9 m..9 m + 9)
    for m in 0..72 {
        let base = 9 * m;
        let mut v: [[int16x8_t; 9]; 2] = core::array::from_fn(|h| core::array::from_fn(|i| input(h, base + i)));
        for g in 0..3 {
            let t = vld1q_s16(r3.get_unchecked(3 * m + g).as_ptr());
            for w in v.iter_mut() {
                (w[3 * g], w[3 * g + 1], w[3 * g + 2]) = ib3(w[3 * g], w[3 * g + 1], w[3 * g + 2], t, k, DIGIT && g == 0);
            }
        }
        let t = vld1q_s16(r3.get_unchecked(216 + m).as_ptr());
        for j in 0..3 {
            for w in v.iter_mut() {
                (w[j], w[j + 3], w[j + 6]) = ib3(w[j], w[j + 3], w[j + 6], t, k, false);
            }
        }
        for (h, w) in v.iter().enumerate() {
            for (i, &y) in w.iter().enumerate() {
                *xs[h].add(base + i) = y;
            }
        }
    }
    // level 4, per level-3 node n (27 slots): (j, j + 9, j + 18)
    for n in 0..24 {
        let t = vld1q_s16(r3.get_unchecked(288 + n).as_ptr());
        let base = 27 * n;
        for j0 in (0..9).step_by(3) {
            for h in 0..2 {
                let p = xs[h];
                let v: [[int16x8_t; 3]; 3] = core::array::from_fn(|q| core::array::from_fn(|i| *p.add(base + j0 + q + 9 * i)));
                let y: [(int16x8_t, int16x8_t, int16x8_t); 3] = core::array::from_fn(|q| ib3(v[q][0], v[q][1], v[q][2], t, k, false));
                for (q, &(a, b, c)) in y.iter().enumerate() {
                    *p.add(base + j0 + q) = a;
                    *p.add(base + j0 + q + 9) = b;
                    *p.add(base + j0 + q + 18) = c;
                }
            }
        }
    }
    // level 3, per level-2 node n (81 slots): (j, j + 27, j + 54)
    for n in 0..8 {
        let t = vld1q_s16(r3.get_unchecked(312 + n).as_ptr());
        let base = 81 * n;
        for j0 in (0..27).step_by(3) {
            for h in 0..2 {
                let p = xs[h];
                let v: [[int16x8_t; 3]; 3] = core::array::from_fn(|q| core::array::from_fn(|i| *p.add(base + j0 + q + 27 * i)));
                let y: [(int16x8_t, int16x8_t, int16x8_t); 3] = core::array::from_fn(|q| ib3(v[q][0], v[q][1], v[q][2], t, k, false));
                for (q, &(a, b, c)) in y.iter().enumerate() {
                    *p.add(base + j0 + q) = a;
                    *p.add(base + j0 + q + 27) = b;
                    *p.add(base + j0 + q + 54) = c;
                }
            }
        }
    }
    if DIGIT {
        for j in 0..81 {
            let rows: [[int16x8_t; 8]; 2] =
                core::array::from_fn(|h| core::array::from_fn(|i| *xs[h].add(j + 81 * i)));
            let rows = [digit_inverse_eight(rows[0], k), digit_inverse_eight(rows[1], k)];
            for (h, row) in rows.into_iter().enumerate() {
                for (i, y) in row.into_iter().enumerate() {
                    output(h, j + 81 * i, y);
                }
            }
        }
        return;
    }
    // levels 2 and 1 and the unsplit, per j < 81: slots j + 81 i, i < 8
    let t2: [int16x8_t; 4] = core::array::from_fn(|n| vld1q_s16(r2.get_unchecked(n).as_ptr()));
    let t1: [int16x8_t; 2] = core::array::from_fn(|n| vld1q_s16(r2.get_unchecked(4 + n).as_ptr()));
    let uv = vld1q_s16(inv.u.as_ptr());
    for j in 0..81 {
        let mut v: [[int16x8_t; 8]; 2] = core::array::from_fn(|h| core::array::from_fn(|i| *xs[h].add(j + 81 * i)));
        for w in v.iter_mut() {
            // level 2: node n pairs slots 162 n + j and 162 n + j + 81
            for n in 0..4 {
                (w[2 * n], w[2 * n + 1]) = ib2(w[2 * n], w[2 * n + 1], t2[n], k);
            }
        }
        for w in v.iter_mut() {
            // level 1: node n pairs 324 n + i and 324 n + i + 162, i = j, j + 81
            for n in 0..2 {
                for h in 0..2 {
                    (w[4 * n + h], w[4 * n + h + 2]) = ib2(w[4 * n + h], w[4 * n + h + 2], t1[n], k);
                }
            }
        }
        for w in v.iter_mut() {
            // the unsplit: b = (u - v) inv_split, a = u - zeta^324 b, pairs i and i + 324
            for i in 0..4 {
                let (u, z) = (w[i], w[i + 4]);
                (w[i], w[i + 4]) = lazy_unsplit::<DIGIT>(u, z, uv, k);
            }
        }
        for (h, w) in v.iter().enumerate() {
            for (i, &y) in w.iter().enumerate() {
                output(h, j + 81 * i, y);
            }
        }
    }
}

