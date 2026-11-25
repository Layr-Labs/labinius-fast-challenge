//! Public coefficient encoding in the kernel of six fixed Phi_6 weights.
//! The integer basis is unimodularly derived from the two modular constraints.
//! Floating point guides a bounded closest-vector search; every returned vector
//! is an exact integer translate by kernel rows. The full public key is checked
//! against the unchanged reconstruction-margin floor when it is built.
pub(super) const MAX_DIGIT_ABS: i8 = 32;
pub(super) const M: i64 = 55 * (1 << 16);
struct Basis {
    nd: usize,
    scales: [i64; 6],
    rows: [[i64; 12]; 12],
    dual: [[f64; 2]; 12],
    mu: [[f64; 12]; 12],
    norms: [f64; 12],
}
const BASIS: Basis = Basis {
    nd: 6,
    scales: [973, 991, 1004, 1018, 992, 1022],
    rows: [
        [-10, 2, 7, -3, 1, 7, 3, -5, -1, -4, -5, 0],
        [-2, 2, -1, -3, 4, 8, 1, -3, -1, 6, -9, 0],
        [7, -2, 4, -6, 4, 8, 5, -5, -4, -2, -3, 1],
        [5, -7, -2, -4, 12, -4, 0, -5, -6, 4, -2, 3],
        [-5, 3, 8, -6, -1, 7, -2, 7, 5, -9, 6, 2],
        [-7, -2, -5, -5, -2, 1, 8, -9, 0, 1, -8, 0],
        [4, 4, -7, 9, -1, 3, -2, 6, 2, -7, 13, -3],
        [-4, 3, -5, 7, 0, 2, 10, -3, 6, -4, -5, -7],
        [-2, 0, 3, -4, -8, 12, 3, -2, -6, 5, 0, -9],
        [-11, 9, -3, -5, 6, 1, 1, 4, -1, 3, 2, -7],
        [14, -10, 1, -1, 5, -4, -9, 6, -5, 1, -6, 0],
        [-10, 6, -2, 8, -2, 1, -2, 5, -7, 0, -7, 3],
    ],
    dual: [
        [-0.03553391160132725, -0.011844637200442418],
        [0.016727106859266642, 0.01208746514496687],
        [0.047629659851985626, 0.009697166774606946],
        [-0.021247772809373146, -0.03690915141392352],
        [-0.013817520734074351, -0.0004022205762515205],
        [-0.01888994097183405, -0.019410470811116864],
        [-0.009717743057761352, 0.009884909919734905],
        [0.0027511556041574735, -0.011776976661163849],
        [-0.015727412278186446, -0.03636463362381201],
        [0.0038961522468997836, 0.06906010418515818],
        [0.004763932661576723, -0.009947898171164735],
        [-0.016439819335937512, -0.012983842329545478],
    ],
    mu: [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [
            0.4916670310529466,
            0.0,
            0.0,
            0.0,
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
            0.40773358010260324,
            0.42443735988686393,
            0.0,
            0.0,
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
            0.03941300008055074,
            0.44671623902321184,
            0.48031125870318336,
            0.0,
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
            0.35382590893420734,
            -0.4768317917630476,
            0.2768830250431176,
            -0.3728714054722716,
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
            0.4524454468907362,
            0.38480270162773283,
            -0.2989358135941185,
            0.25749115470643247,
            -0.4629252933658902,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            -0.49399220961942664,
            -0.46374147031704693,
            0.3637878818719226,
            -0.43084408762778764,
            0.4410753458353047,
            -0.3311323997726116,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.323477308973021,
            0.25545116814999574,
            -0.286198526709673,
            -0.507051560908696,
            -0.5034919827468918,
            0.25843907703734453,
            0.4843897110063898,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.4737355957686485,
            0.36139506027235097,
            0.09302310571491759,
            -0.5092531245655242,
            -0.2761562194097677,
            0.15384080703611228,
            -0.004108599670918131,
            -0.36110092319232084,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.28271823365382437,
            0.4442014588550154,
            -0.33654095279111423,
            -0.012183988322030617,
            0.467012524933354,
            0.2635788173162268,
            0.3878175786366579,
            0.2734157117979434,
            0.4905913433510327,
            0.0,
            0.0,
            0.0,
        ],
        [
            -0.32651833611570147,
            0.29841621300814886,
            0.45972275971286675,
            0.39051430372384766,
            -0.18512094761115255,
            -0.4826236577413338,
            -0.4843217394169693,
            0.1227112199508347,
            0.18094411232261368,
            -0.29139694073628075,
            0.0,
            0.0,
        ],
        [
            0.3753108079431163,
            -0.16367455794196087,
            -0.47720593331504363,
            -0.15199432545059388,
            -0.21226515494853318,
            -0.32973372599013756,
            0.25921139438990876,
            -0.1104941479365234,
            -0.16398417032056456,
            0.3068839178938644,
            0.4999999999999998,
            0.0,
        ],
    ],
    norms: [
        479573490.0,
        387700031.697853,
        326613019.4374291,
        322721679.70716137,
        316218465.82915276,
        293257816.72539204,
        264542906.09424448,
        208799689.96045503,
        200558606.32155377,
        171851994.11055285,
        217392133.17748696,
        163044099.88311526,
    ],
};
pub(super) const WEIGHTS: [(i32, i32); 6] = [
    (1, 0),
    (9479, -8074),
    (13138, 2774),
    (7458, -11604),
    (-16795, 1143510),
    (1639102, 1040483),
];
fn norm(a: i64, b: i64) -> i64 {
    a * a + a * b + b * b
}
fn cost(v: &[i64; 12], t: &Basis) -> i64 {
    (0..t.nd)
        .map(|d| t.scales[d] * t.scales[d] * norm(v[2 * d], v[2 * d + 1]))
        .sum()
}
struct Search<'a> {
    b: &'a Basis,
    target: [i64; 12],
    projection: [f64; 12],
    k: [i64; 12],
    best: [i64; 12],
    bound: f64,
    nodes: usize,
    limit: usize,
}
impl Search<'_> {
    fn residual(&self) -> [i64; 12] {
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
fn encode(a: i64, b: i64, t: &Basis, limit: usize) -> ([i64; 12], usize) {
    let n = 2 * t.nd;
    let mut target = [0i64; 12];
    target[0] = a;
    target[1] = b;
    let projection = std::array::from_fn(|i| a as f64 * t.dual[i][0] + b as f64 * t.dual[i][1]);
    let mut s = Search {
        b: t,
        target,
        projection,
        k: [0; 12],
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

pub(super) fn digits_pair(x: u128, y: u128) -> ([i8; 6], [i8; 6]) {
    digits_pair_limit(x, y, 512)
}

/// [`digits_pair`] with a bound on the search's nodes.
pub(super) fn digits_pair_limit(x: u128, y: u128, limit: usize) -> ([i8; 6], [i8; 6]) {
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
            for d in 0..6 {
                let (u, v) = WEIGHTS[d];
                x += u as i64 * row[2 * d] - v as i64 * row[2 * d + 1];
                y += v as i64 * row[2 * d] + (u + v) as i64 * row[2 * d + 1];
            }
            assert_eq!((x.rem_euclid(M), y.rem_euclid(M)), (0, 0));
        }
        let mut a = BASIS.rows.map(|row| row.map(i128::from));
        let (mut previous, mut sign) = (1i128, 1i128);
        for k in 0..11 {
            let pivot = (k..12).find(|&i| a[i][k] != 0).unwrap();
            if pivot != k {
                a.swap(k, pivot);
                sign = -sign;
            }
            let p = a[k][k];
            for i in k + 1..12 {
                for j in k + 1..12 {
                    let numerator = a[i][j] * p - a[i][k] * a[k][j];
                    assert_eq!(numerator % previous, 0);
                    a[i][j] = numerator / previous;
                }
            }
            for i in k + 1..12 {
                a[i][k] = 0;
            }
            previous = p;
        }
        assert_eq!((sign * a[11][11]).abs(), M as i128 * M as i128);
    }

    #[test]
    fn gram_schmidt_tables_match_the_integer_basis() {
        let dot = |a: &[f64; 12], b: &[f64; 12]| {
            (0..6)
                .map(|d| {
                    let w = (BASIS.scales[d] * BASIS.scales[d]) as f64;
                    w * (2. * a[2 * d] * b[2 * d]
                        + a[2 * d] * b[2 * d + 1]
                        + a[2 * d + 1] * b[2 * d]
                        + 2. * a[2 * d + 1] * b[2 * d + 1])
                })
                .sum::<f64>()
        };
        let mut gs = [[0.; 12]; 12];
        for i in 0..12 {
            let mut v = BASIS.rows[i].map(|x| x as f64);
            for j in 0..i {
                for k in 0..12 {
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
