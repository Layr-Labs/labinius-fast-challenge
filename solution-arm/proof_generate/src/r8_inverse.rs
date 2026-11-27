use super::*;

impl LazyInverse {
    fn new_r8(index: usize) -> LazyInverse {
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

fn lazy_inverses() -> &'static [LazyInverse; 3] {
    static K: OnceLock<[LazyInverse; 3]> = OnceLock::new();
    K.get_or_init(|| [LazyInverse::new_r8(0), LazyInverse::new_r8(1), LazyInverse::new_r8(2)])
}

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
            (v[3 * g], v[3 * g + 1], v[3 * g + 2]) = ib3(v[3 * g], v[3 * g + 1], v[3 * g + 2], t, k);
        }
        let t = vld1q_s16(r3.get_unchecked(216 + m).as_ptr());
        for j in 0..3 {
            (v[j], v[j + 3], v[j + 6]) = ib3(v[j], v[j + 3], v[j + 6], t, k);
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
            let y: [(int16x8_t, int16x8_t, int16x8_t); 3] = core::array::from_fn(|q| ib3(v[q][0], v[q][1], v[q][2], t, k));
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
            let y: [(int16x8_t, int16x8_t, int16x8_t); 3] = core::array::from_fn(|q| ib3(v[q][0], v[q][1], v[q][2], t, k));
            for (q, &(a, b, c)) in y.iter().enumerate() {
                store(x, base + j0 + q, a);
                store(x, base + j0 + q + 27, b);
                store(x, base + j0 + q + 54, c);
            }
        }
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
                (w[3 * g], w[3 * g + 1], w[3 * g + 2]) = ib3(w[3 * g], w[3 * g + 1], w[3 * g + 2], t, k);
            }
        }
        let t = vld1q_s16(r3.get_unchecked(216 + m).as_ptr());
        for j in 0..3 {
            for w in v.iter_mut() {
                (w[j], w[j + 3], w[j + 6]) = ib3(w[j], w[j + 3], w[j + 6], t, k);
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
                let y: [(int16x8_t, int16x8_t, int16x8_t); 3] = core::array::from_fn(|q| ib3(v[q][0], v[q][1], v[q][2], t, k));
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
                let y: [(int16x8_t, int16x8_t, int16x8_t); 3] = core::array::from_fn(|q| ib3(v[q][0], v[q][1], v[q][2], t, k));
                for (q, &(a, b, c)) in y.iter().enumerate() {
                    *p.add(base + j0 + q) = a;
                    *p.add(base + j0 + q + 27) = b;
                    *p.add(base + j0 + q + 54) = c;
                }
            }
        }
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

