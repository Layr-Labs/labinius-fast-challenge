//! Public coefficient encoding in the kernel of six fixed Phi_6 weights.
//! The integer basis is unimodularly derived from the two modular constraints.
//! Floating point guides a bounded closest-vector search; every returned vector
//! is an exact integer translate by kernel rows. The full public key is checked
//! against the unchanged reconstruction-margin floor when it is built.
pub(super) const MAX_DIGIT_ABS: i8 = 32;
pub(super) const M: i64 = 3604480;
struct Basis {
    nd: usize,
    scales: [i64; 6],
    rows: [[i64; 12]; 12],
    dual: [[f64; 2]; 12],
    mu: [[f64; 12]; 12],
    norms: [f64; 12],
}
const BASIS: Basis = Basis {nd:5,scales:[1;6],rows:[[5,-13,-12,9,13,-11,-3,0,9,-5,0,0],[-13,8,9,3,-11,-2,0,3,-5,-4,0,0],[-8,5,-10,-6,-2,-15,-7,1,-9,5,0,0],[5,3,-6,16,-15,17,1,6,5,4,0,0],[3,1,-2,9,-13,4,10,-1,14,-17,0,0],[3,-9,-19,12,0,0,-10,-2,-12,7,0,0],[-8,5,0,4,-9,15,-16,18,1,-7,0,0],[-15,-4,10,-8,-1,10,10,1,4,3,0,0],[-14,18,-4,6,-6,18,12,-4,12,0,0,0],[-3,8,4,0,6,9,2,16,-6,-1,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0]],dual:[[-0.0032397408207343412,-0.02267818574514039],[-0.028077753779697623,-0.01079913606911447],[-0.00438555373475181,0.00474554788484687],[0.009251099669630369,0.009011103569566995],[0.017515707566625925,0.011202866925122226],[0.0032600175224123495,-0.01588569880541974],[-0.008737112391573007,-0.0002786106597914975],[-0.033649147240300256,-0.028263900832459082],[-0.014495569884054147,0.026781273456838953],[0.006076882102272729,0.00812045010653409],[0.0,0.0],[0.0,0.0]],mu:[[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[-0.5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.05939524838012959,0.2958963282937365,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[-0.25161987041036715,-0.08855291576673865,-0.5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[-0.03779697624190065,0.468682505399568,-0.4134251568412013,0.39921476276010515,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.47516198704103674,-0.06479481641468683,0.4073015063505217,0.3869674617787462,-0.4999999999999999,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[-0.2861771058315335,0.30885529157667385,-0.21592524121483023,0.36339159488658307,-0.3124688222846418,0.13425653373493435,0.0,0.0,0.0,0.0,0.0,0.0],[-0.20302375809935205,0.09935205183585313,-0.36569218250203434,-0.3099962125615459,-0.25374361476049995,-0.434256537402092,0.10005545871892103,0.0,0.0,0.0,0.0,0.0],[-0.42764578833693306,-0.24190064794816415,-0.4503489318298577,0.4793097112171928,0.19882059567131893,-0.4708718834615461,0.06973694438267404,0.49818929907150183,0.0,0.0,0.0,0.0],[-0.3747300215982721,-0.13174946004319654,-0.3805063167723524,-0.034229443771538665,-0.2569268114435216,-0.24534055541717464,0.5000000000000001,-0.0180969729356282,0.09103397846624021,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]],norms:[926.0,694.5,1151.9265658747302,863.9449244060474,613.5442474059798,460.15818555448493,763.9208655997869,831.9778481396593,629.4196740439264,567.452058667694,0.0,0.0]};
pub(super) const WEIGHTS: [(i32,i32);5]=[(1,0),(3491785,1481300),(2493332,497181),(1933295,1033395),(718137,3337492)];
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

pub(super) fn digits_pair(x: u128, y: u128) -> ([i8; 5], [i8; 5]) {
    digits_pair_limit(x, y, 512)
}

/// [`digits_pair`] with a bound on the search's nodes.
pub(super) fn digits_pair_limit(x: u128, y: u128, limit: usize) -> ([i8; 5], [i8; 5]) {
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
        for row in BASIS.rows.iter().take(2*BASIS.nd) {
            let (mut x, mut y) = (0i64, 0i64);
            for d in 0..BASIS.nd {
                let (u, v) = WEIGHTS[d];
                x += u as i64 * row[2 * d] - v as i64 * row[2 * d + 1];
                y += v as i64 * row[2 * d] + (u + v) as i64 * row[2 * d + 1];
            }
            assert_eq!((x.rem_euclid(M), y.rem_euclid(M)), (0, 0));
        }
        let mut a = BASIS.rows.map(|row| row.map(i128::from));
        let (mut previous, mut sign) = (1i128, 1i128);
        let n=2*BASIS.nd;
        for k in 0..n-1 {
            let pivot = (k..n).find(|&i| a[i][k] != 0).unwrap();
            if pivot != k {
                a.swap(k, pivot);
                sign = -sign;
            }
            let p = a[k][k];
            for i in k + 1..n {
                for j in k + 1..n {
                    let numerator = a[i][j] * p - a[i][k] * a[k][j];
                    assert_eq!(numerator % previous, 0);
                    a[i][j] = numerator / previous;
                }
            }
            for i in k + 1..n {
                a[i][k] = 0;
            }
            previous = p;
        }
        assert_eq!((sign * a[n-1][n-1]).abs(), M as i128 * M as i128);
    }

    #[test]
    fn gram_schmidt_tables_match_the_integer_basis() {
        let dot = |a: &[f64; 12], b: &[f64; 12]| {
            (0..BASIS.nd)
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
        for i in 0..2*BASIS.nd {
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
