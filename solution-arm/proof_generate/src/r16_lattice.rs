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
// Isotropic weights from SplitMix64 seed 0x1b2fb.
// The integer kernel determinant and the rounded nearest-plane bound were checked
// from exact Gram-Schmidt data. The CVP search accepts only lower integer costs.
const BASIS: Basis = Basis {
    nd: DIGITS,
    scales: [1; DIGITS],
    rows: [
        [-12, 6, -1, -6, -9, 13, 4, 8, 7, 0],
        [6, -12, 7, -1, -4, -9, -12, 4, -7, 7],
        [-5, 4, 17, -8, 3, 5, -14, 6, 4, 1],
        [-1, 5, 9, -17, 8, -3, -8, 14, 5, -4],
        [1, 6, 2, -7, 19, -15, -1, -11, 15, -2],
        [9, -4, -2, -1, 10, -2, 14, -1, -19, 10],
        [-16, 10, -1, 5, -10, -9, 5, -2, -9, 0],
        [10, 6, 5, -4, -9, 19, -2, -3, 0, 9],
        [13, -16, 4, -12, 5, 4, 7, -12, -8, -7],
        [10, -8, -5, 20, 2, -2, 10, -2, 19, -13],
    ],
    dual: [
        [-0.020224719101123594, 0.0],
        [-0.01348314606741573, -0.02696629213483146],
        [-0.0014353520690273858, 0.011017958211681809],
        [0.015647512328260667, 0.00925910823315772],
        [-0.007931339273258672, -0.0030373643913799934],
        [-0.0012377403269991237, -0.008550209436758233],
        [-0.020569703697850657, 0.015702683202287234],
        [0.034650046734950084, 0.0378947270653257],
        [-0.008346973765980113, -0.015502513538707387],
        [0.015105368874289773, -0.0007942893288352273],
    ],
    mu: [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [-0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.09775280898876404, 0.3393258426966292, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.30337078651685395, 0.07191011235955057, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [-0.3741573033707865, -0.4966292134831461, 0.4633686190716969, 0.42314613950099766, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [-0.2348314606741573, -0.39325842696629215, -0.4143247049101002, 0.32505831117780426, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.04719101123595506, 0.2876404494382023, -0.4575837152783441, -0.2797196713913672, -0.1141395565104909, -0.255973125423443, 0.0, 0.0, 0.0, 0.0],
        [0.19213483146067414, -0.19101123595505617, 0.4385816111826975, -0.3177238795826605, -0.1349100658123368, 0.24212611922221242, -0.5, 0.0, 0.0, 0.0],
        [-0.4, -0.23370786516853934, -0.10072746252589343, 0.24492869453641933, -0.297258837438121, 0.3356020263462161, -0.426384359977698, -0.20015791825194543, 0.0, 0.0],
        [-0.16741573033707866, -0.451685393258427, -0.24693764373909213, -0.4242933337682885, 0.1459775212822261, -0.4512128444099984, -0.43692625870011004, -0.4735366808963292, -0.5, 0.0],
    ],
    norms: [890.0, 667.5, 826.638202247191, 619.9786516853933, 852.2763270210464, 639.2072452657848, 844.2124855352208, 633.1593641514156, 785.4120201847735, 589.0590151385802],
};
pub(super) const WEIGHTS: [(i32, i32); DIGITS] = [
    (1, 0),
    (-1754843, 1122278),
    (-1306481, 579304),
    (1075091, -376923),
    (-287032, 1645821),
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
