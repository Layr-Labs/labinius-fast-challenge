use super::{super::*, LUT, QUARTER_SCALE};

const P: i32 = 9721;
const OFFSETS: [usize; 6] = [0, 81, 27, 108, 54, 135];

fn butterfly(
    t: &[[i16; 8]; 11],
    [i0, i1, i2, i3, i4, i5]: [usize; 6],
    check: impl Fn(i32) -> i32,
) -> [i32; 6] {
    let add = |a, b| check(a + b);
    let sub = |a, b| check(a - b);
    let x0 = t[i0 / 8][i0 % 8] as i32;
    let y0 = t[2][i1] as i32;
    let a = add(x0, y0);
    let b = sub(x0, y0);
    let p0 = add(t[4][i3] as i32, t[5][i4] as i32);
    let p1 = add(t[8][i3] as i32, t[9][i4] as i32);
    let q0 = add(t[3][i2] as i32, t[6][i5] as i32);
    let q1 = add(t[7][i2] as i32, t[10][i5] as i32);
    let s0 = add(p0, q0);
    let s1 = add(p1, q1);
    let d0 = sub(p0, q0);
    let d1 = sub(p1, q1);
    [
        add(a, s0),
        add(a, s1),
        sub(sub(a, s0), s1),
        sub(sub(b, d0), d1),
        add(b, d0),
        add(b, d1),
    ]
}

fn checked(value: i32) -> i32 {
    assert!(
        (i16::MIN as i32..=i16::MAX as i32).contains(&value),
        "{value}"
    );
    value
}

type Range = (i32, i32);
const EMPTY: Range = (i32::MAX, i32::MIN);

fn include((lo, hi): Range, value: i32) -> Range {
    (lo.min(value), hi.max(value))
}

fn add((a, b): Range, (c, d): Range) -> Range {
    (checked(a + c), checked(b + d))
}

fn sub((a, b): Range, (c, d): Range) -> Range {
    (checked(a - d), checked(b - c))
}

// The full product and the correction wrap in the machine instructions. The product's
// signed remainder is representable, so those wraps cancel in the low 16 bits.
fn mul((lo, hi): Range, t: &[i16]) -> Range {
    (lo..=hi)
        .map(|a| {
            let quotient = checked((a * t[1] as i32 + 16384) >> 15);
            checked(a * t[0] as i32 - quotient * P)
        })
        .fold(EMPTY, include)
}

fn red13((lo, hi): Range) -> Range {
    (lo..=hi)
        .map(|a| checked(a - ((a + 4096) >> 13) * P))
        .fold(EMPTY, include)
}

fn radix3(a: Range, b: Range, c: Range, t: &[i16], k: &Kernel) -> [Range; 3] {
    let b = mul(b, t);
    let c = mul(c, &t[2..]);
    let e = mul(sub(b, c), &k.k[2..]);
    [add(a, add(b, c)), add(sub(a, c), e), sub(sub(a, b), e)]
}

#[test]
fn weighted_lut_signed_intervals() {
    let kern = &kernels()[1];
    let eps = kern.k[2] as i32;
    let mut x = [EMPTY; N];
    let mut seen = [false; GROUPS];
    for (n, table) in LUT.iter().enumerate() {
        let scale = QUARTER_SCALE[n];
        for t in 0..6 {
            for pat in 0..if t == 0 { 16 } else { 8 } {
                assert_eq!(
                    (table[(t + usize::from(t != 0)) + pat / 8][pat % 8] as i32
                        - scale * kern.lut[n][t][pat] as i32)
                        .rem_euclid(P),
                    0
                );
            }
            if t >= 2 {
                let root = if t < 4 {
                    eps
                } else {
                    (eps * eps).rem_euclid(P)
                };
                for pat in 0..8 {
                    assert_eq!(
                        (table[t + 5][pat] as i32 - scale * (root * kern.lut[n][t][pat] as i32).rem_euclid(P))
                            .rem_euclid(P),
                        0
                    );
                }
            }
        }
        // The six indices are independent. Each is a disjoint group of witness bits.
        // X0 reaches bit 486 plus j, so X0 has 16 entries until j = 26.
        let mut range = [[EMPTY; 6]; 2];
        for pattern in 0..16 * 8usize.pow(5) {
            let indices = core::array::from_fn(|t| {
                if t == 0 {
                    pattern & 15
                } else {
                    (pattern >> (4 + 3 * (t - 1))) & 7
                }
            });
            let out = butterfly(table, indices, checked);
            let n0 = kern.lut[n][0][indices[0]] as i32;
            let n1 = kern.lut[n][1][indices[1]] as i32;
            let x1 = kern.lut[n][2][indices[2]] as i32;
            let y1 = kern.lut[n][3][indices[3]] as i32;
            let x2 = kern.lut[n][4][indices[4]] as i32;
            let y2 = kern.lut[n][5][indices[5]] as i32;
            let a0 = n0 + n1;
            let b0 = n0 - n1;
            let a1 = x1 + y1;
            let a2 = x2 + y2;
            let b1 = x1 - y1;
            let b2 = x2 - y2;
            let even = eps * (a1 - a2);
            let odd = eps * (b1 + b2);
            let want = [
                a0 + a1 + a2,
                a0 - a2 + even,
                a0 - a1 - even,
                b0 + b1 + odd,
                b0 - b1 + b2,
                b0 - b2 - odd,
            ];
            for i in 0..6 {
                assert_eq!((out[i] - scale * want[i].rem_euclid(P)).rem_euclid(P), 0);
                range[0][i] = include(range[0][i], out[i]);
                if indices[0] < 8 {
                    range[1][i] = include(range[1][i], out[i]);
                }
            }
        }
        for j in 0..27 {
            if n == 0 {
                for (t, offset) in OFFSETS.into_iter().enumerate() {
                    let slot = j + offset;
                    assert!(!core::mem::replace(&mut seen[slot], true));
                    assert_eq!(slot + 486 < BITS, t == 0 && j < 26);
                    assert!(slot + 324 < BITS);
                }
            }
            for (i, bounds) in range[usize::from(j == 26)].into_iter().enumerate() {
                x[162 * n + j + 27 * i] = bounds;
            }
        }
    }
    assert!(seen.into_iter().all(|found| found));

    // Production tail levels 4 to 6, with its original sparse reductions.
    for n in 0..24 {
        for j in 0..9 {
            let at = 27 * n + j;
            let a = if n == 10 {
                red13(x[at])
            } else {
                x[at]
            };
            for (i, bounds) in radix3(a, x[at + 9], x[at + 18], &kern.c[n], kern)
                .into_iter()
                .enumerate()
            {
                x[at + 9 * i] = bounds;
            }
        }
    }
    for n in 0..72 {
        for j in 0..3 {
            let at = 9 * n + j;
            let a = if n / 3 == 10 { x[at] } else { red13(x[at]) };
            let t = &kern.c[n / 3][4 + 4 * (n % 3)..];
            for (i, bounds) in radix3(a, x[at + 3], x[at + 6], t, kern)
                .into_iter()
                .enumerate()
            {
                x[at + 3 * i] = bounds;
            }
        }
    }
    let bound = super::super::super::lut_transform_bound(P as u32, SCHEDULES[1]) as i32;
    let mut final_bound = 0;
    for n in 0..216 {
        let t = &kern.d[n / 2][4 * (n % 2)..];
        for (lo, hi) in radix3(red13(x[3 * n]), x[3 * n + 1], x[3 * n + 2], t, kern) {
            final_bound = final_bound.max(lo.abs()).max(hi.abs());
        }
    }
    assert!(final_bound <= bound, "{final_bound} > {bound}");
    println!("Weighted tail bound {final_bound}; contract {bound}");
}

#[test]
fn weighted_lut_matches_scalar() {
    let plan = &primes()[1];
    let kern = &kernels()[1];
    let mut state = 17u64;
    for case in 0..BITS + 34 {
        let words: [[u128; 4]; LANES] = core::array::from_fn(|lane| {
            core::array::from_fn(|w| {
                if case < BITS {
                    let bit = (case + lane * 37) % BITS;
                    if w == bit % 4 {
                        1u128 << (bit / 4)
                    } else {
                        0
                    }
                } else if case == BITS {
                    u128::MAX
                } else if case == BITS + 1 {
                    0
                } else {
                    use super::super::super::tests::xorshift;
                    (xorshift(&mut state) as u128) << 64 | xorshift(&mut state) as u128
                }
            })
        });
        let ptrs = words.each_ref().map(|w| w.as_ptr().cast::<u8>());
        unsafe {
            let mut indices = [vdupq_n_u8(0); GROUPS];
            super::super::indices(&ptrs, &mut indices);
            let bytes: &[u8; 16 * GROUPS] = core::mem::transmute(&indices);
            let mut expected = [[0i16; LANES]; N];
            for (n, table) in LUT.iter().enumerate() {
                for j in 0..27 {
                    for lane in 0..LANES {
                        let select = OFFSETS.map(|o| bytes[packed_slot(j + o) + lane] as usize);
                        for (k, value) in butterfly(table, select, checked).into_iter().enumerate()
                        {
                            expected[162 * n + j + 27 * k][lane] = value as i16;
                        }
                    }
                }
            }
            let x = {
                let mut x = [vdupq_n_s16(i16::MIN); N];
                #[cfg(target_feature = "sve")]
                super::pass_lut_weighted(&indices, &mut x);
                #[cfg(not(target_feature = "sve"))]
                super::pass_lut_weighted_neon(&indices, &mut x);
                let got: &[[i16; LANES]; N] = core::mem::transmute(&x);
                assert_eq!(*got, expected);
                x
            };
            let mut y = [vdupq_n_s16(0); N];
            pass_tail_calls_c(&x, &mut y, kern);
            let got: &[[i16; LANES]; N] = core::mem::transmute(&y);
            for (lane, w) in words.iter().enumerate() {
                let mut want = [0u32; N];
                super::super::super::element_bits(&w.map(B128), &mut want);
                plan.forward(&mut want);
                for (slot, a) in got.iter().enumerate() {
                    assert_eq!(
                        (a[lane] as i32 + QUARTER_SCALE[slot / 162] * want[slot] as i32).rem_euclid(P),
                        0,
                        "case {case}, lane {lane}, slot {slot}"
                    );
                }
            }
        }
    }
}
