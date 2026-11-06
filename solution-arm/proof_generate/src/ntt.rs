//! The commitment through number-theoretic transforms modulo the three primes of
//! [`MODULUS`] `= 3889 * 9721 * 17497`.
//!
//! Every prime is `1 mod 1944`, so modulo each of them `X^648 - X^324 + 1 = Phi_1944` splits into
//! the 648 linear factors `X - zeta^e`, `e` coprime to 6, for a primitive 1944th root of unity
//! `zeta`. The transform reduces a polynomial modulo those factors along a tree:
//!
//! - level 0 splits the trinomial as `(X^324 - zeta^324)(X^324 - zeta^1620)`, the two primitive
//!   6th roots of unity `w` and `w^5` (`w + w^5 = 1`, `w * w^5 = 1`);
//! - levels 1 and 2 split every binomial `X^n - zeta^e` by its square roots,
//!   `(X^(n/2) - zeta^(e/2))(X^(n/2) + zeta^(e/2))`;
//! - levels 3 to 6 split it by its cube roots, `X^(n/3) - zeta^(e/3 + 648 k)`, `k < 3`.
//!
//! A node of length `n` and exponent `e` holds a polynomial modulo `X^n - zeta^e` in place; its
//! `r` children are stored one after the other, child `k` in the `k`-th `n / r` coefficients, so
//! slot `i` of the result is the evaluation at `zeta^leaf(i)` (see [`Prime::leaves`]). A ring
//! product is the slot-wise product, so a column's commitment `t = sum_m A_m s_m` is summed slot by
//! slot, transformed back per prime and recombined by the Chinese remainder theorem into `[0, Q)`.
//!
//! [`commit`] runs the hot part, the forward transform of every binary ring element of the
//! witness and the slot-wise products, in [`neon`] on AArch64 and in [`scalar_commit`] elsewhere.
//! [`matrix`] expands and transforms the matrix eight elements per vector on AArch64; the inverse
//! transform and the recombination are scalar. Every step is exact: the vector code keeps every
//! value inside the `i16` and `i32` bounds that the `const` checks below verify.
use labinius::fields::scalar::B128;
use std::sync::OnceLock;

/// Degree of the ring.
pub const N: usize = 648;
/// Half the degree: the trinomial's middle exponent.
const HALF: usize = N / 2;
/// The order of `zeta`.
const ORDER: usize = 1944;
/// Radix of the binomial levels 1 to 6.
const RADICES: [usize; 6] = [2, 2, 3, 3, 3, 3];
/// The three primes.
pub const PRIMES: [u32; 3] = [3889, 9721, 17497];
/// The commitment modulus the transforms cover, the product of [`PRIMES`].
pub const MODULUS: u128 = PRIMES[0] as u128 * PRIMES[1] as u128 * PRIMES[2] as u128;
const _: () = assert!(MODULUS == 661_473_542_593);
const _: () = {
    let mut i = 0;
    while i < 3 {
        assert!((PRIMES[i] - 1) % ORDER as u32 == 0, "every prime is 1 mod 1944");
        assert!(is_prime(PRIMES[i]));
        assert!(PRIMES[i] < 1 << 15, "residues are i16 lanes");
        i += 1;
    }
};

const fn is_prime(p: u32) -> bool {
    let mut d = 2;
    while d * d <= p {
        if p % d == 0 {
            return false;
        }
        d += 1;
    }
    p > 1
}

const fn pow_mod(base: u64, mut e: u64, p: u64) -> u64 {
    let mut b = base % p;
    let mut r = 1 % p;
    while e > 0 {
        if e & 1 == 1 {
            r = r * b % p;
        }
        b = b * b % p;
        e >>= 1;
    }
    r
}

/// The first `g^((p - 1) / 1944)`, `g = 2, 3, ...`, of order exactly 1944: its 972nd and its
/// 648th powers are not 1, and 1944 = 2^3 3^5 has no other maximal proper divisors.
const fn primitive_root(p: u32) -> u32 {
    let p = p as u64;
    let mut g = 2;
    loop {
        let z = pow_mod(g, (p - 1) / ORDER as u64, p);
        if pow_mod(z, 972, p) != 1 && pow_mod(z, 648, p) != 1 {
            return z as u32;
        }
        g += 1;
    }
}

/// `x` centered: the representative of `x mod p` in `[-(p - 1) / 2, (p - 1) / 2]`.
pub const fn centered(x: i64, p: u32) -> i16 {
    let p = p as i64;
    let r = x.rem_euclid(p);
    (if r > p / 2 { r - p } else { r }) as i16
}

/// One prime's transform: the powers of its root and the exponent of every node.
pub struct Prime {
    pub p: u32,
    /// `zeta^i` for `i < 1944`, in `[0, p)`.
    pub pow: Vec<u32>,
    /// `exps[l]`: the exponents `e` of the nodes `X^n - zeta^e` after level `l`, in storage
    /// order; `exps[0]` = `[324, 1620]` and `exps[6]` holds the 648 leaves.
    pub exps: Vec<Vec<u32>>,
    inv_radix: [u32; 4],
    /// `1 / (w - w^5)`, which undoes the trinomial split.
    inv_split: u32,
}

impl Prime {
    fn new(p: u32) -> Prime {
        let zeta = primitive_root(p) as u64;
        let pow: Vec<u32> = (0..ORDER as u64).map(|i| pow_mod(zeta, i, p as u64) as u32).collect();
        let mut exps = vec![vec![HALF as u32, (5 * HALF) as u32]];
        for &r in &RADICES {
            let next: Vec<u32> = exps
                .last()
                .unwrap()
                .iter()
                .flat_map(|&e| {
                    assert_eq!(e as usize % r, 0, "every split has its roots");
                    (0..r).map(move |k| (e as usize / r + k * ORDER / r) as u32)
                })
                .collect();
            exps.push(next);
        }
        let inv = |x: u64| pow_mod(x, p as u64 - 2, p as u64) as u32;
        let w = pow[HALF] as u64;
        let w5 = pow[5 * HALF] as u64;
        Prime {
            p,
            inv_radix: [0, 1, inv(2), inv(3)],
            inv_split: inv((w + p as u64 - w5) % p as u64),
            pow,
            exps,
        }
    }

    /// `zeta^e` for any exponent.
    #[inline]
    pub fn root(&self, e: usize) -> u64 {
        self.pow[e % ORDER] as u64
    }

    /// The exponents of the 648 slots: slot `i` of a transform is the evaluation at
    /// `zeta^leaves()[i]`.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn leaves(&self) -> &[u32] {
        &self.exps[RADICES.len()]
    }

    /// The transform of `x` (residues in `[0, p)`) in place.
    pub fn forward(&self, x: &mut [u32; N]) {
        // the modulus as a constant, so that `%` is a multiplication
        match self.p {
            3889 => self.forward_mod(x, 3889),
            9721 => self.forward_mod(x, 9721),
            17497 => self.forward_mod(x, 17497),
            p => self.forward_mod(x, p as u64),
        }
    }

    #[inline(always)]
    fn forward_mod(&self, x: &mut [u32; N], p: u64) {
        debug_assert_eq!(p, self.p as u64);
        let (w, w5) = (self.root(HALF), self.root(5 * HALF));
        for j in 0..HALF {
            let (a, b) = (x[j] as u64, x[j + HALF] as u64);
            x[j] = ((a + w * b) % p) as u32;
            x[j + HALF] = ((a + w5 * b) % p) as u32;
        }
        let mut len = HALF;
        for (level, &r) in RADICES.iter().enumerate() {
            let sub = len / r;
            for (node, &e) in self.exps[level].iter().enumerate() {
                let base = node * len;
                let tau = e as usize / r;
                for j in 0..sub {
                    let mut a = [0u64; 3];
                    for (i, slot) in a.iter_mut().enumerate().take(r) {
                        *slot = x[base + j + i * sub] as u64 * self.root(tau * i) % p;
                    }
                    for k in 0..r {
                        let mut acc = 0u64;
                        for (i, &v) in a.iter().enumerate().take(r) {
                            acc += v * self.root(k * i * (ORDER / r));
                        }
                        x[base + j + k * sub] = (acc % p) as u32;
                    }
                }
            }
            len = sub;
        }
    }

    /// The inverse of [`Prime::forward`], residues in `[0, p)` in and out.
    pub fn inverse(&self, x: &mut [u32; N]) {
        match self.p {
            3889 => self.inverse_mod(x, 3889),
            9721 => self.inverse_mod(x, 9721),
            17497 => self.inverse_mod(x, 17497),
            p => self.inverse_mod(x, p as u64),
        }
    }

    #[inline(always)]
    fn inverse_mod(&self, x: &mut [u32; N], p: u64) {
        debug_assert_eq!(p, self.p as u64);
        let mut len = 1;
        for level in (0..RADICES.len()).rev() {
            let r = RADICES[level];
            let parent = len * r;
            for (node, &e) in self.exps[level].iter().enumerate() {
                let base = node * parent;
                let tau = e as usize / r;
                for j in 0..len {
                    let mut y = [0u64; 3];
                    for (k, slot) in y.iter_mut().enumerate().take(r) {
                        *slot = x[base + j + k * len] as u64;
                    }
                    // a_i = r^-1 tau^-i sum_k eps^-(k i) y_k
                    for i in 0..r {
                        let mut acc = 0u64;
                        for (k, &v) in y.iter().enumerate().take(r) {
                            acc += v * self.root(ORDER - k * i * (ORDER / r) % ORDER);
                        }
                        let scale = self.root(ORDER - tau * i % ORDER) * self.inv_radix[r] as u64 % p;
                        x[base + j + i * len] = (acc % p * scale % p) as u32;
                    }
                }
            }
            len = parent;
        }
        let w = self.root(HALF);
        for j in 0..HALF {
            let (u, v) = (x[j] as u64, x[j + HALF] as u64);
            let b = (u + p - v) * self.inv_split as u64 % p;
            x[j] = ((u + p * p - w * b) % p) as u32;
            x[j + HALF] = b as u32;
        }
    }
}

/// The three primes' transforms, built once.
pub fn primes() -> &'static [Prime; 3] {
    static PLANS: OnceLock<[Prime; 3]> = OnceLock::new();
    PLANS.get_or_init(|| PRIMES.map(Prime::new))
}

/// Garner's recombination of residues `r_i mod p_i` into the residue in `[0, MODULUS)`.
#[inline]
pub fn crt(r: [u32; 3]) -> u128 {
    const P0: u64 = PRIMES[0] as u64;
    const P1: u64 = PRIMES[1] as u64;
    const P2: u64 = PRIMES[2] as u64;
    const INV_P0_MOD_P1: u64 = pow_mod(P0, P1 - 2, P1);
    const INV_P0P1_MOD_P2: u64 = pow_mod(P0 * P1 % P2, P2 - 2, P2);
    let (r0, r1, r2) = (r[0] as u64, r[1] as u64, r[2] as u64);
    debug_assert!(r0 < P0 && r1 < P1 && r2 < P2);
    let x1 = (r1 + P1 - r0 % P1) % P1 * INV_P0_MOD_P1 % P1;
    let low = r0 + P0 * x1;
    let x2 = (r2 + P2 - low % P2) % P2 * INV_P0P1_MOD_P2 % P2;
    // x0 + p0 x1 + p0 p1 x2 <= (p0 - 1) + p0 (p1 - 1) + p0 p1 (p2 - 1) = Q - 1
    (low + P0 * P1 * x2) as u128
}

/// Where the vector transform ([`neon`]) reduces, per prime. Lanes are `i16`; a level's
/// multiplications are Barrett products by constants and its additions are lazy, so values
/// grow level by level until a reduction brings them back to [`reduced_bound`].
///
/// Levels: 0 the trinomial split, 1 the first radix-2 split (both on the binary input, as
/// selections of constants), 2 the second radix-2 split, 3 to 6 the radix-3 splits.
#[derive(Clone, Copy)]
pub struct Schedule {
    /// `before[l]`: which inputs of level `l` to reduce (used for `l = 2..=6`): [`NONE`],
    /// [`ALL`], or [`FIRST`], the input that is not multiplied by a twiddle.
    pub before: [u8; 7],
    /// Reduce `d = tau b - tau^2 c` inside the radix-3 butterfly before it is multiplied by the
    /// cube root of unity.
    pub reduce_d: bool,
    /// Reduce the multiplied inputs (under [`ALL`]) and `d` with [`cheap_bound`]'s one-multiply
    /// reduction instead of the Barrett reduction; the first inputs stay Barrett-reduced.
    pub cheap: bool,
    /// Elements summed into the `i32` slot accumulators between two reductions of them, within
    /// [`accumulator_bound`].
    pub accumulate: usize,
}

pub const NONE: u8 = 0;
pub const ALL: u8 = 1;
pub const FIRST: u8 = 2;
/// As [`FIRST`], with the one-multiply reduction of [`round13_bound`].
pub const FIRST13: u8 = 3;

pub const SCHEDULES: [Schedule; 3] = [
    Schedule { before: [NONE; 7], reduce_d: false, cheap: false, accumulate: 35 },
    Schedule {
        before: [NONE, NONE, NONE, FIRST13, FIRST13, FIRST13, FIRST13],
        reduce_d: false,
        cheap: false,
        accumulate: 15,
    },
    Schedule {
        before: [NONE, NONE, FIRST, ALL, ALL, ALL, ALL],
        reduce_d: true,
        cheap: true,
        accumulate: 7,
    },
];

/// The schedule of the vector transform of a matrix element (any residues, see
/// [`general_transform_bound`]): every level's inputs reduced, as for the largest prime.
pub const GENERAL: Schedule = Schedule {
    before: [ALL; 7],
    reduce_d: true,
    cheap: false,
    accumulate: 0,
};

/// The schedule of the binary transforms (`0..3`, by prime) or of [`GENERAL`] (`3`).
pub const fn schedule(index: usize) -> Schedule {
    if index < 3 {
        SCHEDULES[index]
    } else {
        GENERAL
    }
}

/// The bound on every value of the vector transform of centered residues (`|x| <= (p - 1) / 2`):
/// levels 0 and 1 are Barrett products and sums with every value reduced in between, then
/// levels 2 to 6 follow [`GENERAL`].
pub const fn general_transform_bound(p: u32) -> u32 {
    const MAX: u32 = i16::MAX as u32;
    let h = (p - 1) / 2;
    assert!(h + product_bound(p, h) <= MAX, "level 0: a + w b");
    let r = reduced_bound(p);
    assert!(r + product_bound(p, r) <= MAX, "level 1: a +- tau b");
    transform_bound(p, GENERAL)
}

/// Largest `|a w - round(a w' / 2^15) p|` for `|a| <= b`, `w' = round(w 2^15 / p)`, any `w`:
/// `p (|a| / 2^16 + 1 / 2)` (`|w / p - w' / 2^15| <= 2^-16` and the rounding of the high
/// product is at most `1 / 2`), rounded up.
pub const fn product_bound(p: u32, b: u32) -> u32 {
    ((p as u64 * (b as u64 + 32768)).div_ceil(65536)) as u32
}

/// Largest `|a - round(floor(a v / 2^15) / 2^11) p|` over `|a| <= 2^15`, `v = round(2^26 / p)`:
/// the quotient is off from `a / p` by at most `1/2 + 2^-11 + 2^-12`, so the remainder is at most
/// `p / 2 + 3 p / 4096`, rounded up. The exhaustive test confirms it per prime.
pub const fn reduced_bound(p: u32) -> u32 {
    p.div_ceil(2) + (3 * p).div_ceil(4096)
}

/// Largest `|a - round(a / 2^14) p|` for `|a| <= b`: `|a| |1 - p / 2^14| + p / 2`, rounded up.
/// Close to [`reduced_bound`] only for a prime close to `2^14`.
pub const fn cheap_bound(p: u32, b: u32) -> u32 {
    let off = (p as i64 - (1 << 14)).unsigned_abs();
    ((b as u64 * off + (p as u64) * (1 << 13)).div_ceil(1 << 14)) as u32
}

/// Largest `|a - round(a / 2^13) p|` for `|a| <= b`: `|a| |1 - p / 2^13| + p / 2`, rounded up.
pub const fn round13_bound(p: u32, b: u32) -> u32 {
    let off = (p as i64 - (1 << 13)).unsigned_abs();
    ((b as u64 * off + (p as u64) * (1 << 12)).div_ceil(1 << 13)) as u32
}

/// The bounds on a level's first input `a` and on its multiplied inputs, from the bound `b` on
/// the previous level's outputs and the level's reduction mode.
const fn level_inputs(p: u32, b: u32, mode: u8, cheap: bool) -> (u32, u32) {
    match mode {
        ALL if cheap => (reduced_bound(p), cheap_bound(p, b)),
        ALL => (reduced_bound(p), reduced_bound(p)),
        FIRST => (reduced_bound(p), b),
        FIRST13 => (round13_bound(p, b), b),
        _ => (b, b),
    }
}

/// The bound on every value the schedule produces, checked level by level against `i16`;
/// returns the bound on the transform's output before its final reduction.
pub const fn transform_bound(p: u32, s: Schedule) -> u32 {
    const MAX: u32 = i16::MAX as u32;
    let h = (p - 1) / 2;
    // levels 0 and 1: m_j - (m_{j+324} & c) -+ ((m_{j+162} & c') + (m_{j+486} & c''))
    let b = 1 + 3 * h;
    assert!(b <= MAX);
    // level 2, radix 2: a +- tau b
    let (a, x) = level_inputs(p, b, s.before[2], s.cheap);
    let b = a + product_bound(p, x);
    assert!(b <= MAX);
    radix3_levels(p, s, b, 3)
}

/// One radix-3 level's output bound from its first input's bound `a` and its multiplied inputs'
/// bound `m` (`b' = tau b`, `c' = tau^2 c`, already multiplied): `y0 = a + (b' + c')`,
/// `y1 = (a - c') + e`, `y2 = (a - b') - e`, `e = eps (b' - c')`, `d = b' - c'` reduced first
/// when the schedule says so. Checked against `i16`.
const fn radix3_output(p: u32, s: Schedule, a: u32, m: u32) -> u32 {
    const MAX: u32 = i16::MAX as u32;
    assert!(2 * m <= MAX, "b' + c' and b' - c' fit");
    let e = if s.reduce_d && s.cheap {
        product_bound(p, cheap_bound(p, 2 * m))
    } else if s.reduce_d {
        product_bound(p, reduced_bound(p))
    } else {
        product_bound(p, 2 * m)
    };
    let y0 = a + 2 * m;
    let y12 = a + m + e;
    let b = if y0 > y12 { y0 } else { y12 };
    assert!(b <= MAX, "a level's outputs fit i16");
    b
}

/// The radix-3 levels `first..=6` of the schedule from the bound `b` on their first level's
/// inputs as the previous level leaves them; returns the bound on the last level's outputs.
const fn radix3_levels(p: u32, s: Schedule, mut b: u32, first: usize) -> u32 {
    let mut l = first;
    while l <= 6 {
        let (a, x) = level_inputs(p, b, s.before[l], s.cheap);
        b = radix3_output(p, s, a, product_bound(p, x));
        l += 1;
    }
    b
}

/// The bound on every value of the table-fed binary transform (`neon::pass_lut`): levels 0 to 2
/// and the level-3 twiddles are lookups in tables of centered residues (`|entry| <= h`,
/// `h = (p - 1) / 2`), so every level-3 input, `a` as well as the multiplied `b' = tau b` and
/// `c' = tau^2 c`, is the sum or difference of two entries, `|v| <= 2 h`, and level 3 multiplies
/// only by `eps` (`s.before[3]` is not used: nothing is reduced before it). Levels 4 to 6 follow
/// `s` as in [`transform_bound`]. Returns the bound on the transform's output.
pub const fn lut_transform_bound(p: u32, s: Schedule) -> u32 {
    radix3_levels(p, s, lut_level3_bound(p, s), 4)
}

/// The bound on the outputs of the table-fed level 3 (see [`lut_transform_bound`]).
pub const fn lut_level3_bound(p: u32, s: Schedule) -> u32 {
    const MAX: u32 = i16::MAX as u32;
    let h = (p - 1) / 2;
    let t = 2 * h;
    assert!(t <= MAX, "a level-3 input fits i16");
    radix3_output(p, s, t, t)
}

/// The accumulator bound: after a reduction a slot sum is at most `p` in magnitude, and each
/// element adds a product of a centered matrix slot (`<= (p - 1) / 2`) and a transform slot as
/// the last level leaves it (`<= transform_bound(p, s)`, not reduced again).
pub const fn accumulator_bound(p: u32, s: Schedule) -> u64 {
    p as u64 + s.accumulate as u64 * ((p as u64 - 1) / 2) * transform_bound(p, s) as u64
}

/// [`accumulator_bound`] for the slots of the table-fed transform (`<= lut_transform_bound`).
pub const fn lut_accumulator_bound(p: u32, s: Schedule) -> u64 {
    p as u64 + s.accumulate as u64 * ((p as u64 - 1) / 2) * lut_transform_bound(p, s) as u64
}

const _: () = {
    let mut i = 0;
    while i < 3 {
        let p = PRIMES[i];
        transform_bound(p, SCHEDULES[i]);
        general_transform_bound(p);
        assert!(accumulator_bound(p, SCHEDULES[i]) <= i32::MAX as u64);
        assert!(reduced_bound(p) < (1 << 14));
        i += 1;
    }
};

/// Stride of a transformed matrix row: the 648 slots and zeros up to a multiple of eight, so that
/// an eight-slot vector load at any multiple of six stays inside the row.
pub const ROW: usize = 656;
const _: () = assert!(ROW >= 6 * (N / 6 - 1) + 8 && ROW % 8 == 0);

/// The matrix in the transform domain: element `m`'s transform modulo prime `i` is
/// `data[(3 m + i) ROW..][..N]`, centered residues (`|a| <= (p - 1) / 2`).
pub struct Matrix {
    pub elements: usize,
    pub data: Vec<i16>,
}

impl Matrix {
    #[inline]
    pub fn row(&self, m: usize, prime: usize) -> &[i16] {
        &self.data[(3 * m + prime) * ROW..][..ROW]
    }

    #[inline]
    #[cfg_attr(target_arch = "aarch64", allow(dead_code))]
    pub fn slots(&self, m: usize, prime: usize) -> &[i16] {
        &self.row(m, prime)[..N]
    }
}

/// Runs `f(first, block)` on `workers` scoped threads over contiguous blocks of `out`, whose
/// items come in groups of `unit` (`first` counts groups), block edges on multiples of `align`
/// groups.
pub fn for_blocks<T: Send>(
    out: &mut [T],
    unit: usize,
    align: usize,
    workers: usize,
    f: impl Fn(usize, &mut [T]) + Sync,
) {
    let groups = out.len() / unit;
    let workers = workers.clamp(1, groups.div_ceil(align).max(1));
    let edges: Vec<(usize, usize)> = (0..workers)
        .map(|t| {
            let (s, e) = labinius::par::block(groups, workers, t, align);
            (s * unit, e * unit)
        })
        .collect();
    let blocks = labinius::par::split_mut(out, &edges);
    let f = &f;
    std::thread::scope(|scope| {
        for (block, &(s, _)) in blocks.into_iter().zip(&edges) {
            scope.spawn(move || f(s / unit, block));
        }
    });
}

/// The transform of every element of `matrix` (`rq::ajtai_matrix` layout, residues below
/// [`MODULUS`]) modulo each prime. The reference for [`matrix`].
#[cfg_attr(not(test), allow(dead_code))]
pub fn transform_matrix(matrix: &[u128], workers: usize) -> Matrix {
    assert_eq!(matrix.len() % N, 0);
    let elements = matrix.len() / N;
    let plans = primes();
    let mut data = vec![0i16; 3 * elements * ROW];
    for_blocks(&mut data, 3 * ROW, 1, workers, |first, block| {
        for (i, element) in block.chunks_exact_mut(3 * ROW).enumerate() {
            let a = &matrix[(first + i) * N..][..N];
            for (plan, out) in plans.iter().zip(element.chunks_exact_mut(ROW)) {
                let mut x = [0u32; N];
                for (slot, &c) in x.iter_mut().zip(a) {
                    *slot = (c % plan.p as u128) as u32;
                }
                plan.forward(&mut x);
                for (o, &v) in out.iter_mut().zip(&x) {
                    *o = centered(v as i64, plan.p);
                }
            }
        }
    });
    Matrix { elements, data }
}

/// The transformed matrix for [`MODULUS`] over `elements` elements: each element is expanded
/// on its own (`rq::ajtai_element`, the same coefficients as `rq::ajtai_matrix`) and
/// transformed, eight elements per vector on AArch64, with no full-width matrix in memory.
pub fn matrix(elements: usize, workers: usize) -> Matrix {
    let plans = primes();
    let mut data = vec![0i16; 3 * elements * ROW];
    for_blocks(&mut data, 3 * ROW, 8, workers, |first, block| {
        let mut coefficients = vec![[0u128; N]; 8];
        let mut lanes = vec![[[0i16; 8]; N]; 3];
        for (g, chunk) in block.chunks_mut(8 * 3 * ROW).enumerate() {
            let base = first + 8 * g;
            let here = chunk.len() / (3 * ROW);
            for (e, c) in coefficients.iter_mut().enumerate().take(here) {
                crate::rq::ajtai_element(MODULUS, elements, base + e, c);
            }
            for (i, plan) in plans.iter().enumerate() {
                for (e, c) in coefficients.iter().enumerate().take(here) {
                    for (slot, &v) in lanes[i].iter_mut().zip(c) {
                        slot[e] = centered((v as u64 % plan.p as u64) as i64, plan.p);
                    }
                }
                transform_lanes(i, &mut lanes[i], here);
                for (e, row) in chunk.chunks_exact_mut(3 * ROW).enumerate() {
                    for (o, l) in row[i * ROW..][..N].iter_mut().zip(&lanes[i]) {
                        *o = centered(l[e] as i64, plan.p);
                    }
                }
            }
        }
    });
    Matrix { elements, data }
}

/// The transform of the first `count` lanes of `x` (centered residues) modulo prime `i`, into
/// values congruent to the transform.
fn transform_lanes(i: usize, x: &mut [[i16; 8]; N], count: usize) {
    #[cfg(target_arch = "aarch64")]
    {
        let _ = count;
        neon::transform_general(i, x);
    }
    #[cfg(not(target_arch = "aarch64"))]
    transform_lanes_scalar(i, x, count);
}

/// [`transform_lanes`] lane by lane in scalar arithmetic, residues in `[0, p)` out.
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
fn transform_lanes_scalar(i: usize, x: &mut [[i16; 8]; N], count: usize) {
    let plan = &primes()[i];
    for e in 0..count {
        let mut v: [u32; N] =
            core::array::from_fn(|j| (x[j][e] as i64).rem_euclid(plan.p as i64) as u32);
        plan.forward(&mut v);
        for (slot, &y) in x.iter_mut().zip(&v) {
            slot[e] = y as i16;
        }
    }
}

/// Coefficient `4 q + k` of a binary witness element is bit `q` of word `k`.
#[inline]
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
pub fn element_bits(words: &[B128], out: &mut [u32; N]) {
    out.fill(0);
    for (k, w) in words.iter().enumerate() {
        for q in 0..128 {
            out[4 * q + k] = ((w.0 >> q) & 1) as u32;
        }
    }
}

/// The transform-domain sums of one column, `acc[i][slot] = sum_m A_m(slot) s_m(slot) mod p_i`,
/// back to coefficients and recombined.
pub fn finish_column(acc: &[[u32; N]; 3], out: &mut [u128]) {
    let mut residues = [[0u32; N]; 3];
    for ((plan, r), a) in primes().iter().zip(residues.iter_mut()).zip(acc) {
        *r = *a;
        plan.inverse(r);
    }
    for (t, o) in out.iter_mut().enumerate() {
        *o = crt([residues[0][t], residues[1][t], residues[2][t]]);
    }
}

/// The commitment in plain integer arithmetic: every column's elements transformed one by one,
/// the slot products summed in `u64` (at most `elements` products below `p^2 < 2^29`), reduced
/// once. Exact for any column; the reference for [`commit`] and the path off AArch64.
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
pub fn scalar_commit(values: &[B128], a: &Matrix, workers: usize) -> Vec<u128> {
    let len = 4 * a.elements;
    assert_eq!(values.len() % len, 0);
    assert!((a.elements as u64) << 29 < u64::MAX >> 1);
    let columns = values.len() / len;
    let plans = primes();
    let mut out = vec![0u128; columns * N];
    for_blocks(&mut out, N, 1, workers, |first, block| {
        for (c, target) in block.chunks_exact_mut(N).enumerate() {
            let column = &values[(first + c) * len..][..len];
            let mut acc = [[0u64; N]; 3];
            let mut bits = [0u32; N];
            for m in 0..a.elements {
                element_bits(&column[4 * m..4 * m + 4], &mut bits);
                for (i, plan) in plans.iter().enumerate() {
                    let mut x = bits;
                    plan.forward(&mut x);
                    let p = plan.p as i64;
                    for ((s, &v), &w) in acc[i].iter_mut().zip(&x).zip(a.slots(m, i)) {
                        *s += v as u64 * (w as i64).rem_euclid(p) as u64;
                    }
                }
            }
            let mut reduced = [[0u32; N]; 3];
            for (i, plan) in plans.iter().enumerate() {
                for (r, &s) in reduced[i].iter_mut().zip(&acc[i]) {
                    *r = (s % plan.p as u64) as u32;
                }
            }
            finish_column(&reduced, target);
        }
    });
    out
}

#[cfg(target_arch = "aarch64")]
#[path = "ntt_neon.rs"]
pub mod neon;

/// The commitment of every column of `values` under the transformed matrix `a`.
pub fn commit(values: &[B128], a: &Matrix, workers: usize) -> Vec<u128> {
    #[cfg(target_arch = "aarch64")]
    {
        neon::commit(values, a, workers)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        scalar_commit(values, a, workers)
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;

    pub fn xorshift(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    fn random_residues(seed: u64, p: u32) -> [u32; N] {
        let mut s = seed | 1;
        core::array::from_fn(|_| (xorshift(&mut s) % p as u64) as u32)
    }

    /// `zeta` has order exactly 1944 and the slots are the 648 primitive 1944th roots, once each.
    #[test]
    fn roots_and_leaves() {
        for plan in primes() {
            let p = plan.p as u64;
            assert_eq!(plan.root(1) * plan.root(ORDER - 1) % p, 1);
            assert_eq!(pow_mod(plan.root(1), ORDER as u64, p), 1);
            for d in [972u64, 648] {
                assert_ne!(pow_mod(plan.root(1), d, p), 1);
            }
            let mut seen = vec![false; ORDER];
            for &e in plan.leaves() {
                assert!(e % 2 == 1 && e % 3 != 0, "leaf {e} is not primitive");
                assert!(!seen[e as usize]);
                seen[e as usize] = true;
            }
            assert_eq!(plan.leaves().len(), N);
            // (X^324 - w)(X^324 - w^5) = X^648 - X^324 + 1
            assert_eq!((plan.root(HALF) + plan.root(5 * HALF)) % p, 1);
            assert_eq!(plan.root(HALF) * plan.root(5 * HALF) % p, 1);
        }
    }

    /// Slot `i` of the transform is the evaluation at `zeta^leaf(i)`.
    #[test]
    fn forward_is_evaluation() {
        for (n, plan) in primes().iter().enumerate() {
            let p = plan.p as u64;
            let x = random_residues(17 + n as u64, plan.p);
            let mut y = x;
            plan.forward(&mut y);
            for (i, &e) in plan.leaves().iter().enumerate() {
                let mut want = 0u64;
                for (j, &c) in x.iter().enumerate() {
                    want = (want + c as u64 * plan.root(e as usize * j)) % p;
                }
                assert_eq!(y[i] as u64, want, "prime {} slot {i}", plan.p);
            }
        }
    }

    /// The inverse undoes the transform, on random and on edge inputs.
    #[test]
    fn round_trip() {
        for (n, plan) in primes().iter().enumerate() {
            let mut inputs = vec![[0u32; N], [plan.p - 1; N], [1u32; N]];
            for t in [0usize, 1, 323, 324, 511, 647] {
                let mut x = [0u32; N];
                x[t] = 1;
                inputs.push(x);
            }
            for s in 0..4 {
                inputs.push(random_residues(100 * n as u64 + s, plan.p));
            }
            for x in inputs {
                let mut y = x;
                plan.forward(&mut y);
                assert!(y.iter().all(|&v| v < plan.p));
                plan.inverse(&mut y);
                assert_eq!(y, x, "prime {}", plan.p);
            }
        }
    }

    /// The slot-wise product is the product in `Z_p[X]/(X^648 - X^324 + 1)`, against the library's
    /// schoolbook `mul_mod_phi`.
    #[test]
    fn product_matches_schoolbook() {
        for (n, plan) in primes().iter().enumerate() {
            for s in 0..3u64 {
                let a = random_residues(1000 + 10 * n as u64 + s, plan.p);
                let mut b = random_residues(2000 + 10 * n as u64 + s, plan.p);
                if s == 2 {
                    b = core::array::from_fn(|i| (i < 512 && i % 3 != 1) as u32);
                }
                let want = labinius::scalar::mul_mod_phi(&a, &b, plan.p as u16);
                let (mut x, mut y) = (a, b);
                plan.forward(&mut x);
                plan.forward(&mut y);
                let mut z: [u32; N] =
                    core::array::from_fn(|i| (x[i] as u64 * y[i] as u64 % plan.p as u64) as u32);
                plan.inverse(&mut z);
                assert_eq!(z, want, "prime {} case {s}", plan.p);
            }
        }
    }

    /// The scalar lane transform (the matrix path off AArch64) agrees with [`transform_lanes`]
    /// modulo each prime, on random residues and on the extremes.
    #[test]
    fn lane_transforms_agree() {
        let mut s = 5u64;
        for (i, &p) in PRIMES.iter().enumerate() {
            let h = (p as i64 - 1) / 2;
            for case in 0..3 {
                let x: [[i16; 8]; N] = core::array::from_fn(|_| {
                    core::array::from_fn(|l| match (case, l % 2) {
                        (0, 0) => h as i16,
                        (0, _) => -h as i16,
                        _ => ((xorshift(&mut s) % (2 * h + 1) as u64) as i64 - h) as i16,
                    })
                });
                let (mut a, mut b) = (x, x);
                transform_lanes(i, &mut a, 8);
                transform_lanes_scalar(i, &mut b, 8);
                for (u, v) in a.iter().flatten().zip(b.iter().flatten()) {
                    assert_eq!((*u as i64 - *v as i64).rem_euclid(p as i64), 0, "prime {p} case {case} u {u} v {v}");
                }
            }
        }
    }

    /// Garner's recombination is the residue in `[0, Q)`.
    #[test]
    fn crt_is_exact() {
        let mut s = 99u64;
        let mut cases: Vec<u128> = vec![0, 1, MODULUS - 1, MODULUS - 2, 3888, 3889, 37_804_969];
        for _ in 0..10_000 {
            cases.push(((xorshift(&mut s) as u128) << 64 | xorshift(&mut s) as u128) % MODULUS);
        }
        for x in cases {
            let r = PRIMES.map(|p| (x % p as u128) as u32);
            assert_eq!(crt(r), x);
        }
    }
}
