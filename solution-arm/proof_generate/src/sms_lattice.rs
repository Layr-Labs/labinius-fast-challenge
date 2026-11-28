//! Public coefficient encoding in the kernel of five fixed Phi_6 weights.
//! Every encoded digit vector is an integer translate by kernel rows.
//! Floating point guides the bounded search. Matrix construction checks the
//! statistical reconstruction margin for the complete public key.
pub(super) const DIGITS: usize = 5;
const DIM: usize = 2 * DIGITS;
pub(super) const MAX_DIGIT_ABS: i8 = 34;
pub(super) const M: i64 = 55 * (1 << 16);
struct Basis {
    nd: usize,
    scales: [i64; DIGITS],
    rows: [[i64; DIM]; DIM],
    dual: [[f64; 2]; DIM],
    mu: [[f64; DIM]; DIM],
    norms: [f64; DIM],
}
// Weights from SplitMix64 seed 0xa718. Exact kernel LLL uses delta 0.99.
// The nearest-plane metric bound is sum(norms) / 4; a digit's squared
// magnitude is at most 2/3 of that bound.
const BASIS: Basis = Basis {
    nd: DIGITS,
    scales: [1; DIGITS],
    rows: [
        [-13, 10, -4, -4, 13, 0, 3, -1, 7, 1],
        [-10, -3, 4, -8, 0, 13, 1, 2, -1, 8],
        [-4, 14, 5, -10, 7, 0, -16, 11, -2, 0],
        [-14, 10, 10, -5, 0, 7, -11, -5, 0, -2],
        [-7, 12, -10, -7, -11, 5, 1, -2, -6, 1],
        [-12, 5, 7, -17, -5, -6, 2, -1, -1, -5],
        [-2, 7, 7, -9, -5, 6, 21, -10, 14, -4],
        [-5, -2, 2, 7, -1, -5, -11, 21, -10, 14],
        [-8, -1, -10, 11, -8, 7, -11, 3, 17, -2],
        [-1, 9, 11, -1, 7, 1, 3, 8, -2, -15],
    ],
    dual: [
        [-0.01904761904761905, 0.008333333333333333],
        [-0.023809523809523808, -0.030952380952380953],
        [0.011311962223962765, 0.02193083784590938],
        [-0.021699808978570663, 0.00046205773467743297],
        [-0.004643759962453151, 0.013046159018536174],
        [-0.02049071866635033, -0.014889119295628316],
        [0.021245271999487815, 0.023111936041339796],
        [-0.016652400055461185, 0.012919071971757224],
        [-0.004203935102982954, -0.01026639071377841],
        [-0.010885897549715908, -0.001239013671875],
    ],
    mu: [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.3630952380952381, -0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.29404761904761906, 0.28809523809523807, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.12023809523809524, -0.05952380952380952, 0.21146184924521177, -0.06981016188925265, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.10476190476190476, 0.09047619047619047, 0.1580885460395454, 0.17655676830058545, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.38333333333333336, 0.047619047619047616, -0.4714538477576113, -0.06487403291879618, 0.19447459646476228, 0.28848848100979085, 0.0, 0.0, 0.0, 0.0],
        [-0.22738095238095238, 0.3595238095238095, 0.2843824485679028, -0.4390168312982132, -0.3136036589897243, 0.050230355959866836, -0.5, 0.0, 0.0, 0.0],
        [0.21071428571428572, 0.1595238095238095, -0.19429246000664913, 0.3772363312615867, -0.02102638121841539, -0.42151032809910866, -0.1563290384879947, 0.3358545594744801, 0.0, 0.0],
        [0.014285714285714285, -0.2904761904761905, 0.38007347844951456, 0.0056742943758557925, -0.30561955546512376, 0.2317815452679697, 0.33005543884985744, -0.011598241249245364, -0.5, 0.0],
    ],
    norms: [840.0, 630.0, 845.0809523809523, 633.8107142857143, 850.746223241504, 638.059667431128, 847.3571066989413, 635.517830024206, 812.4396556900224, 609.3297417675168],
};
pub(super) const WEIGHTS: [(i32, i32); DIGITS] = [
    (1, 0),
    (916496, -714991),
    (1593471, -1286804),
    (-105278, 1288145),
    (-146902, -284761),
];
fn norm(a: i64, b: i64) -> i64 {
    a * a + a * b + b * b
}
fn cost(v: &[i64; DIM], t: &Basis) -> i64 {
    (0..t.nd)
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
        let n = 2 * self.b.nd;
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
        let n = 2 * self.b.nd;
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
    let n = 2 * t.nd;
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

pub(super) fn digits_pair(x: u128, y: u128) -> ([i8; DIGITS], [i8; DIGITS]) {
    digits_pair_limit(x, y, 512)
}

/// [`digits_pair`] with a bound on the search's nodes.
pub(super) fn digits_pair_limit(x: u128, y: u128, limit: usize) -> ([i8; DIGITS], [i8; DIGITS]) {
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
            for d in 0..DIGITS {
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
            (0..DIGITS)
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
