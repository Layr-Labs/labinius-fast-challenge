//! Public coefficient encoding in the kernel of five fixed Phi_6 weights.
//! The rank-10 kernel `{L : sum_d W_d L_d = 0 (mod M)}` (index `M^2`, since `W_0 = 1`) has the
//! LLL-reduced basis below, derived deterministically (weights `W_1..W_4` from a fixed SplitMix64
//! seed, exact-integer LLL with delta 0.99 under the encoder's metric, the smallest Babai bound
//! of eight fixed seeds). Floating point guides a bounded closest-vector search; every returned
//! vector is an exact integer translate by kernel rows. The full public key is checked against
//! the unchanged reconstruction-margin floor when it is built.
/// Digits per coefficient pair's encoding.
pub(super) const ND: usize = 5;
/// Coordinates of an encoding: a pair `(x, y)` per digit.
const DIM: usize = 2 * ND;
/// Every encoding's digits are within this: the search returns a residual no costlier than its
/// Babai start, whose `2 cost` is at most `sum(norms) / 4` (each Gram-Schmidt coefficient within
/// one half); a pair `(a, b)` with `s^2 2 (a^2 + ab + b^2) <= T` has `|a|, |b| <= sqrt(T / (1.5
/// s^2))`, here `sqrt(1841.26 / 1.5) = 35.04` (unit scales). [`digits_pair_limit`] asserts it.
pub(super) const MAX_DIGIT_ABS: i8 = 35;
pub(super) const M: i64 = 55 * (1 << 16);
struct Basis {
    scales: [i64; ND],
    rows: [[i64; DIM]; DIM],
    dual: [[f64; 2]; DIM],
    mu: [[f64; DIM]; DIM],
    norms: [f64; DIM],
}
const BASIS: Basis = Basis {
    scales: [1, 1, 1, 1, 1],
    rows: [
        [-3, 3, 9, -4, -3, 17, -1, -7, -9, 1],
        [-3, 0, 4, 5, -17, 14, 7, -8, -1, -8],
        [-3, 14, -4, 4, 2, -10, -11, 11, -1, -10],
        [11, 3, 0, 4, -8, -2, 0, 11, -11, 1],
        [-7, -3, 9, -18, -13, 7, 11, -9, 1, -3],
        [3, -7, 23, -18, 7, -3, 1, 3, -5, 7],
        [13, -15, -1, -9, -1, 3, 9, 5, 8, -14],
        [-2, -13, -10, 1, 2, 1, 14, -9, -6, -8],
        [2, -6, 5, 1, 2, 11, -21, 26, 2, -1],
        [27, -17, 7, -4, 6, -5, -6, -4, 1, -2],
    ],
    dual: [
        [-0.003355704697986577, 0.003355704697986577],
        [-0.006711409395973154, -0.006711409395973154],
        [0.007656451501757218, 0.028336992154044557],
        [0.032678355204221264, 0.008682726100353413],
        [-0.010061354519766101, 9.063340462399995e-05],
        [-0.006828414219342735, -0.01347556162943747],
        [0.011732766043105108, -0.011047716891177369],
        [-0.022552133216973233, -0.023008832651591725],
        [-0.025480235706676135, -0.022290316495028408],
        [0.012733598188920454, -0.019113436612215907],
    ],
    mu: [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [
            -0.29977628635346754,
            -0.03131991051454139,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            -0.17337807606263983,
            0.2841163310961969,
            0.5,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.26174496644295303,
            0.48322147651006714,
            -0.41331548222600684,
            -0.17400891899075616,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.2684563758389262,
            -0.4966442953020134,
            -0.3371644303560705,
            0.3263110227306288,
            0.5,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            -0.24496644295302014,
            0.2348993288590604,
            -0.28351758695031554,
            0.2774337720636733,
            0.40124800077769657,
            0.011224732404465079,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.053691275167785234,
            0.3624161073825503,
            -0.34983412252291274,
            -0.14480070091847885,
            0.2090425496921971,
            -0.3956356345754641,
            0.5,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.203579418344519,
            -0.42953020134228187,
            0.2195144761323476,
            0.19011429302723934,
            -0.36650313870493834,
            0.4635778490654239,
            0.4405937975046687,
            -0.4453650357670548,
            0.0,
            0.0,
        ],
        [
            -0.017897091722595078,
            -0.32662192393736017,
            -0.1745380533761235,
            0.463182091139091,
            -0.26318381666890217,
            0.41293266916681476,
            0.44567932442237457,
            -0.21791127962114135,
            -0.5,
            0.0,
        ],
    ],
    norms: [
        894.0,
        670.5,
        909.0022371364653,
        681.751677852349,
        724.2605211604532,
        543.1953908703399,
        859.1396473735315,
        644.3547355301486,
        822.1937548268344,
        616.6453161201258,
    ],
};
pub(super) const WEIGHTS: [(i32, i32); ND] = [
    (1, 0),
    (-1750057, 1555894),
    (-1552521, 1078029),
    (991329, 581069),
    (226869, -534657),
];
fn norm(a: i64, b: i64) -> i64 {
    a * a + a * b + b * b
}
fn cost(v: &[i64; DIM], t: &Basis) -> i64 {
    (0..ND)
        .map(|d| t.scales[d] * t.scales[d] * norm(v[2 * d], v[2 * d + 1]))
        .sum()
}
struct Search<'a> {
    b: &'a Basis,
    target: [i64; DIM],
    projection: [f64; DIM],
    k: [i64; DIM],
    best: [i64; DIM],
    bound: f64,
    nodes: usize,
    limit: usize,
}
impl Search<'_> {
    fn residual(&self) -> [i64; DIM] {
        let n = DIM;
        let mut r = self.target;
        for i in 0..n {
            for j in 0..n {
                r[j] -= self.k[i] * self.b.rows[i][j];
            }
        }
        r
    }
    fn visit(&mut self, level: isize, partial: f64) {
        if self.nodes >= self.limit {
            return;
        }
        self.nodes += 1;
        if level < 0 {
            let r = self.residual();
            let d = (2 * cost(&r, self.b)) as f64;
            if d < self.bound {
                self.best = r;
                self.bound = d;
            }
            return;
        }
        let i = level as usize;
        let n = DIM;
        let center = self.projection[i]
            - (i + 1..n)
                .map(|j| self.k[j] as f64 * self.b.mu[j][i])
                .sum::<f64>();
        let remain = self.bound - partial + 0.25;
        if remain < 0. {
            return;
        }
        let radius = (remain / self.b.norms[i]).sqrt();
        let lo = (center - radius).ceil() as i64;
        let hi = (center + radius).floor() as i64;
        if lo > hi {
            return;
        }
        let z = center.round() as i64;
        let dir = if center - z as f64 >= 0. { 1 } else { -1 };
        let mut visited = 0;
        let count = hi - lo + 1;
        for step in 0..2 * count + 2 {
            let off = if step == 0 {
                0
            } else {
                ((step + 1) / 2) * if step % 2 == 1 { dir } else { -dir }
            };
            let k = z + off;
            if k < lo || k > hi {
                continue;
            }
            visited += 1;
            let delta = center - k as f64;
            let d = partial + delta * delta * self.b.norms[i];
            if d <= self.bound + 0.25 {
                self.k[i] = k;
                self.visit(level - 1, d);
            }
            if visited == count || self.nodes >= self.limit {
                break;
            }
        }
    }
}
fn encode(a: i64, b: i64, t: &Basis, limit: usize) -> ([i64; DIM], usize) {
    let n = DIM;
    let mut target = [0i64; DIM];
    target[0] = a;
    target[1] = b;
    let projection = std::array::from_fn(|i| a as f64 * t.dual[i][0] + b as f64 * t.dual[i][1]);
    let mut s = Search {
        b: t,
        target,
        projection,
        k: [0; DIM],
        best: target,
        bound: 0.,
        nodes: 0,
        limit,
    };
    for i in (0..n).rev() {
        let center = s.projection[i] - (i + 1..n).map(|j| s.k[j] as f64 * t.mu[j][i]).sum::<f64>();
        s.k[i] = center.round() as i64;
    }
    s.best = s.residual();
    s.bound = (2 * cost(&s.best, t)) as f64;
    s.visit(n as isize - 1, 0.);
    (s.best, s.nodes)
}

pub(super) fn digits_pair(x: u128, y: u128) -> ([i8; ND], [i8; ND]) {
    digits_pair_limit(x, y, 512)
}

/// [`digits_pair`] with a bound on the search's nodes.
pub(super) fn digits_pair_limit(x: u128, y: u128, limit: usize) -> ([i8; ND], [i8; ND]) {
    let center = |v: u128| {
        let r = (v % M as u128) as i64;
        if r >= M / 2 {
            r - M
        } else {
            r
        }
    };
    let (r, _) = encode(center(x), center(y), &BASIS, limit);
    assert!(r.iter().all(|v| v.abs() <= MAX_DIGIT_ABS as i64));
    (
        std::array::from_fn(|d| r[2 * d] as i8),
        std::array::from_fn(|d| r[2 * d + 1] as i8),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basis_is_the_full_modular_kernel() {
        for row in BASIS.rows {
            let (mut x, mut y) = (0i64, 0i64);
            for d in 0..ND {
                let (u, v) = WEIGHTS[d];
                x += u as i64 * row[2 * d] - v as i64 * row[2 * d + 1];
                y += v as i64 * row[2 * d] + (u + v) as i64 * row[2 * d + 1];
            }
            assert_eq!((x.rem_euclid(M), y.rem_euclid(M)), (0, 0));
        }
        let mut a = BASIS.rows.map(|row| row.map(i128::from));
        let (mut previous, mut sign) = (1i128, 1i128);
        for k in 0..DIM - 1 {
            let pivot = (k..DIM).find(|&i| a[i][k] != 0).unwrap();
            if pivot != k {
                a.swap(k, pivot);
                sign = -sign;
            }
            let p = a[k][k];
            for i in k + 1..DIM {
                for j in k + 1..DIM {
                    let numerator = a[i][j] * p - a[i][k] * a[k][j];
                    assert_eq!(numerator % previous, 0);
                    a[i][j] = numerator / previous;
                }
            }
            for i in k + 1..DIM {
                a[i][k] = 0;
            }
            previous = p;
        }
        assert_eq!((sign * a[DIM - 1][DIM - 1]).abs(), M as i128 * M as i128);
    }

    #[test]
    fn gram_schmidt_tables_match_the_integer_basis() {
        let dot = |a: &[f64; DIM], b: &[f64; DIM]| {
            (0..ND)
                .map(|d| {
                    let w = (BASIS.scales[d] * BASIS.scales[d]) as f64;
                    w * (2. * a[2 * d] * b[2 * d]
                        + a[2 * d] * b[2 * d + 1]
                        + a[2 * d + 1] * b[2 * d]
                        + 2. * a[2 * d + 1] * b[2 * d + 1])
                })
                .sum::<f64>()
        };
        let mut gs = [[0.; DIM]; DIM];
        for i in 0..DIM {
            let mut v = BASIS.rows[i].map(|x| x as f64);
            for j in 0..i {
                for k in 0..DIM {
                    v[k] -= BASIS.mu[i][j] * gs[j][k];
                }
            }
            gs[i] = v;
            assert!(BASIS.norms[i] > 0. && BASIS.norms[i].is_finite());
            assert!((dot(&v, &v) - BASIS.norms[i]).abs() < 1e-8 * BASIS.norms[i]);
            for j in 0..i {
                assert!(dot(&v, &gs[j]).abs() < 1e-8 * (BASIS.norms[i] * BASIS.norms[j]).sqrt());
            }
            let scale = (BASIS.scales[0] * BASIS.scales[0]) as f64;
            let expected = [
                scale * (2. * v[0] + v[1]) / BASIS.norms[i],
                scale * (v[0] + 2. * v[1]) / BASIS.norms[i],
            ];
            for k in 0..2 {
                assert!((expected[k] - BASIS.dual[i][k]).abs() < 1e-12);
            }
        }
    }
}
