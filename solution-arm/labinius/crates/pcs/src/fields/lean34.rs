//! The field switch without the full-length tables.
//!
//! [`SwitchProver`](super::crossfield::SwitchProver) expands `eq(r_hi, .)` over all `n = 2^l`
//! packed words and materialises `a[j] = psi(eq[j])` and `p[j] = lift(pi0[j])` as two full-length
//! word-sliced polynomials before its first round. This computes the same partials, the same
//! messages and the same final value from far less:
//!
//! - **Blocks.** Split the index as `j = u B + i` with `B = 2^(l - K)`, `u < 2^K`. The serial
//!   doubling puts the first coordinate in the top index bit, so `eq[u B + i] = top[u] rest[i]`
//!   with `top = eq(r_hi[..K])` and `rest = eq(r_hi[K..])`. Only `rest` (`B` entries) is expanded.
//! - **Partials.** One pass over the words gives the block sets
//!   `S_u[k] = sum_i bit_k(pi0[u B + i]) rest[i]`, and `v[k] = sum_u top[u] S_u[k]` (exact field
//!   identities, so the 128 partials are the full table's).
//! - **The first `K` messages.** With `psi` GF(2)-linear and `lift(x) = sum_k bit_k(x) X^k`,
//!   `M[u1][u2] = sum_i a[u1 B + i] p[u2 B + i] = sum_k psi(top[u1] S_u2[k]) X^k`. Round `t < K`
//!   folds block coordinates only, so its two message sums are combinations of `M` with the
//!   weights `eq(rho[..t])` of the challenges so far: no pass over the table.
//! - **Round `K`.** `a_K[i] = sum_u w[u] psi(top[u] rest[i])` is the linear map
//!   `psi_K = sum_u w[u] (psi o top[u])` applied to `rest`, and `p_K[i] = sum_u w[u] lift(pi0[u B + i])`
//!   with `w = eq(rho[..K])`: one pass over the words. The remaining `l - K` rounds run on these
//!   `B`-entry polynomials with the existing round kernels.
//!
//! No ratio or inverse appears, so there is no degenerate point.
use super::crossfield::{eq_expand_b128, eq_expand_f162, psi, psi_table, LOG_PACK};
use super::scalar::{B128, F162};
use super::sumcheck::{self, Poly};
use crate::par;

const PACK: usize = 1 << LOG_PACK;

/// Block coordinates handled by the partial sets: `2^LEAN_K` blocks.
pub const LEAN_K: usize = 4;

/// Rows of `rest` per worker task of the partial-set pass.
const SET_CHUNK: usize = 1 << 12;

pub struct LeanSwitch<'a> {
    pi0: &'a [B128],
    l: usize,
    k: usize,
    top: Vec<B128>,
    rest: Vec<B128>,
    sets: Vec<Vec<B128>>,
    m: Vec<F162>,
    batch: Vec<F162>,
    rho: Vec<F162>,
    a: Poly,
    p: Poly,
    half: usize,
}

/// The `F162` monomial `X^k`, `k < 128`: the lift of the `B128` with only bit `k` set.
#[inline]
fn monomial(k: usize) -> F162 {
    F162::from_b128(B128(1u128 << k))
}

/// `sets[u]` for every block over `rest`, split over the workers by rows of `rest`.
fn partial_sets(pi0: &[B128], rest: &[B128], blocks: usize) -> Vec<Vec<B128>> {
    let b = rest.len();
    let tasks = b.div_ceil(SET_CHUNK);
    let parts = par::workers(tasks, 1);
    let per_worker = par::map_blocks(tasks, parts, 1, |_, t0, t1| {
        let mut acc = vec![vec![B128::ZERO; PACK]; blocks];
        for t in t0..t1 {
            let (s, e) = (t * SET_CHUNK, ((t + 1) * SET_CHUNK).min(b));
            let part = block_sets(pi0, b, blocks, s, &rest[s..e]);
            for (a, p) in acc.iter_mut().zip(part) {
                for (x, y) in a.iter_mut().zip(p) {
                    *x = *x + y;
                }
            }
        }
        acc
    });
    let mut sets = vec![vec![B128::ZERO; PACK]; blocks];
    for w in per_worker {
        for (a, p) in sets.iter_mut().zip(w) {
            for (x, y) in a.iter_mut().zip(p) {
                *x = *x + y;
            }
        }
    }
    sets
}

#[cfg(target_arch = "aarch64")]
fn block_sets(pi0: &[B128], stride: usize, blocks: usize, j0: usize, eq: &[B128]) -> Vec<Vec<B128>> {
    super::neon34::partial_sets_rows(pi0, stride, blocks, j0, eq)
}

#[cfg(not(target_arch = "aarch64"))]
fn block_sets(pi0: &[B128], stride: usize, blocks: usize, j0: usize, eq: &[B128]) -> Vec<Vec<B128>> {
    (0..blocks)
        .map(|u| {
            let rows = &pi0[u * stride + j0..][..eq.len()];
            let mut v = vec![B128::ZERO; PACK];
            for (&p, &e) in rows.iter().zip(eq) {
                for (k, vk) in v.iter_mut().enumerate() {
                    if p.bit(k) {
                        *vk = *vk + e;
                    }
                }
            }
            v
        })
        .collect()
}

/// `p_K[i] = sum_u w[u] lift(pi0[u B + i])` into `out`, split over the workers.
fn lift_combine(pi0: &[B128], b: usize, w: &[F162], out: &mut Poly) {
    let parts = par::workers(b, 1 << 14);
    let edges: Vec<(usize, usize)> = (0..parts).map(|t| par::block(b, parts, t, 8)).collect();
    let [w0, w1, w2] = &mut out.w;
    let (c0, c1, c2) = (
        par::split_mut(&mut w0[..b], &edges),
        par::split_mut(&mut w1[..b], &edges),
        par::split_mut(&mut w2[..b], &edges),
    );
    std::thread::scope(|scope| {
        for (((x0, x1), x2), &(s, _)) in c0.into_iter().zip(c1).zip(c2).zip(&edges) {
            scope.spawn(move || lift_combine_rows(pi0, b, w, s, [x0, x1, x2]));
        }
    });
}

fn lift_combine_rows(pi0: &[B128], stride: usize, w: &[F162], j0: usize, out: [&mut [u64]; 3]) {
    #[cfg(target_arch = "aarch64")]
    if super::neon34::pmull() {
        // SAFETY: the CPU has PMULL.
        return unsafe { super::neon34::lift_combine_rows(pi0, stride, w, j0, out) };
    }
    let [o0, o1, o2] = out;
    for j in 0..o0.len() {
        let y = w
            .iter()
            .enumerate()
            .fold(F162::ZERO, |acc, (u, &wu)| acc + wu * F162::from_b128(pi0[u * stride + j0 + j]));
        o0[j] = y.0[0];
        o1[j] = y.0[1];
        o2[j] = y.0[2];
    }
}

/// `a[i] = images(rest[i])` for the linear map given by its 128 basis images, split over the
/// workers.
fn apply_map(rest: &[B128], images: &[F162], out: &mut Poly) {
    let b = rest.len();
    let parts = par::workers(b, 1 << 14);
    let edges: Vec<(usize, usize)> = (0..parts).map(|t| par::block(b, parts, t, 8)).collect();
    let [w0, w1, w2] = &mut out.w;
    let (c0, c1, c2) = (
        par::split_mut(&mut w0[..b], &edges),
        par::split_mut(&mut w1[..b], &edges),
        par::split_mut(&mut w2[..b], &edges),
    );
    #[cfg(target_arch = "aarch64")]
    let tabs = [super::neon34::ByteTables::new(images)];
    #[cfg(not(target_arch = "aarch64"))]
    let tabs = psi_table(images);
    let tabs = &tabs;
    std::thread::scope(|scope| {
        for (((x0, x1), x2), &(s, e)) in c0.into_iter().zip(c1).zip(c2).zip(&edges) {
            let rows = &rest[s..e];
            scope.spawn(move || {
                #[cfg(target_arch = "aarch64")]
                super::neon34::poly_from_eq_rows(&[rows], tabs, [x0, x1, x2]);
                #[cfg(not(target_arch = "aarch64"))]
                for (i, &x) in rows.iter().enumerate() {
                    let y = psi(tabs, x);
                    x0[i] = y.0[0];
                    x1[i] = y.0[1];
                    x2[i] = y.0[2];
                }
            });
        }
    });
}

impl<'a> LeanSwitch<'a> {
    /// The partials of `pi0` at `r_hi`, and the switch state that keeps what the rounds need.
    pub fn partials(pi0: &'a [B128], r_hi: &[B128]) -> (Self, Vec<B128>) {
        Self::partials_with(pi0, r_hi, LEAN_K)
    }

    /// [`Self::partials`] with `k` block coordinates (at most `r_hi.len()`).
    pub fn partials_with(pi0: &'a [B128], r_hi: &[B128], k: usize) -> (Self, Vec<B128>) {
        let l = r_hi.len();
        assert_eq!(pi0.len(), 1 << l);
        let k = k.min(l);
        let blocks = 1usize << k;
        let top = eq_expand_b128(&r_hi[..k]);
        let rest = eq_expand_b128(&r_hi[k..]);
        let sets = partial_sets(pi0, &rest, blocks);
        let mut v = vec![B128::ZERO; PACK];
        for (t, s) in top.iter().zip(&sets) {
            for (vk, &sk) in v.iter_mut().zip(s) {
                *vk = *vk + *t * sk;
            }
        }
        let b = rest.len();
        let lean = Self {
            pi0,
            l,
            k,
            top,
            rest,
            sets,
            m: Vec::new(),
            batch: Vec::new(),
            rho: Vec::with_capacity(l),
            a: Poly::zero(b),
            p: Poly::zero(b),
            half: b,
        };
        (lean, v)
    }

    /// Fix the batching vector of the claim; must precede the first [`Self::msg`].
    pub fn start(&mut self, batch: &[F162]) {
        assert_eq!(batch.len(), PACK);
        self.batch = batch.to_vec();
        let blocks = 1usize << self.k;
        let tab = psi_table(batch);
        let x: Vec<F162> = (0..PACK).map(monomial).collect();
        let (top, sets, tab, x) = (&self.top, &self.sets, &tab, &x);
        let rows = par::map_blocks(blocks, par::workers(blocks, 1), 1, |_, u0, u1| {
            let mut out = Vec::with_capacity((u1 - u0) * blocks);
            for u1 in u0..u1 {
                for s in sets.iter() {
                    let mut acc = F162::ZERO;
                    for (k, &sk) in s.iter().enumerate() {
                        acc += psi(tab, top[u1] * sk) * x[k];
                    }
                    out.push(acc);
                }
            }
            out
        });
        self.m = rows.into_iter().flatten().collect();
        if self.k == 0 {
            self.materialize();
        }
    }

    #[inline]
    fn cross(&self, u1: usize, u2: usize) -> F162 {
        self.m[(u1 << self.k) + u2]
    }

    /// The next round's message `[sum a0 p0, sum (a0 + a1)(p0 + p1)]`.
    pub fn msg(&mut self) -> [F162; 2] {
        let t = self.rho.len();
        assert!(t < self.l, "every round has been run");
        assert!(!self.batch.is_empty(), "start() fixes the batch first");
        if t >= self.k {
            self.half /= 2;
            return sumcheck::msg(&self.a, &self.p, self.half);
        }
        // Round t < K: entries are (s B + i), s < 2^(K - t), each a w-combination of blocks
        // (c 2^(K - t) + s), c < 2^t.
        let w = eq_expand_f162(&self.rho);
        let span = 1usize << (self.k - t);
        let h = span / 2;
        let comb = |s1: usize, s2: usize| {
            let mut acc = F162::ZERO;
            for (c1, &w1) in w.iter().enumerate() {
                let mut row = F162::ZERO;
                for (c2, &w2) in w.iter().enumerate() {
                    row += w2 * self.cross(c1 * span + s1, c2 * span + s2);
                }
                acc += w1 * row;
            }
            acc
        };
        let mut e0 = F162::ZERO;
        let mut einf = F162::ZERO;
        for s in 0..h {
            e0 += comb(s, s);
            einf += comb(s, s) + comb(s, s + h) + comb(s + h, s) + comb(s + h, s + h);
        }
        [e0, einf]
    }

    /// Bind the current round's coordinate to `r`.
    pub fn fold(&mut self, r: F162) {
        let t = self.rho.len();
        assert!(t < self.l, "every round has been run");
        self.rho.push(r);
        if t >= self.k {
            sumcheck::fold(&mut self.a, &mut self.p, self.half, r);
            return;
        }
        if t + 1 == self.k {
            self.materialize();
        }
    }

    /// `a_K` and `p_K` from the block weights `eq(rho[..K])`.
    fn materialize(&mut self) {
        let w = eq_expand_f162(&self.rho[..self.k]);
        let tab = psi_table(&self.batch);
        let images: Vec<F162> = (0..PACK)
            .map(|k| {
                let e = B128(1u128 << k);
                w.iter()
                    .zip(&self.top)
                    .fold(F162::ZERO, |acc, (&wu, &tu)| acc + wu * psi(&tab, tu * e))
            })
            .collect();
        apply_map(&self.rest, &images, &mut self.a);
        let b = self.rest.len();
        lift_combine(self.pi0, b, &w, &mut self.p);
    }

    /// The opened value once every round is folded.
    pub fn final_eval(&self) -> F162 {
        assert_eq!(self.rho.len(), self.l, "rounds remain");
        self.p.get(0)
    }
}

#[cfg(test)]
mod tests {
    use super::super::crossfield::{SwitchProver, Workspace};
    use super::*;

    const M34: u64 = (1 << 34) - 1;

    struct Rng(u128);
    impl Rng {
        fn next(&mut self) -> u128 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x.wrapping_mul(0x2545F4914F6CDD1D_9E3779B97F4A7C15) ^ (x >> 64)
        }
        fn f162(&mut self) -> F162 {
            F162([self.next() as u64, self.next() as u64, (self.next() as u64) & M34])
        }
    }

    fn check(l: usize, k: usize, seed: u128) {
        let mut rng = Rng(seed);
        let pi0: Vec<B128> = (0..1usize << l).map(|_| B128(rng.next())).collect();
        let r_hi: Vec<B128> = (0..l).map(|_| B128(rng.next())).collect();
        let batch: Vec<F162> = (0..PACK).map(|_| rng.f162()).collect();
        let rs: Vec<F162> = (0..l).map(|_| rng.f162()).collect();

        let mut ws = Workspace::new(l, 1);
        let want_v = ws.partial_evals(0, &pi0, &r_hi);
        let mut sp = SwitchProver::from_workspace(ws, &pi0, &[F162::ONE], &batch);
        let mut want = Vec::new();
        for &r in &rs {
            want.push(sp.msg());
            sp.fold(r);
        }
        let want_final = sp.final_eval();

        let (mut lean, v) = LeanSwitch::partials_with(&pi0, &r_hi, k);
        assert_eq!(v, want_v, "partials l={l} k={k}");
        lean.start(&batch);
        for (t, &r) in rs.iter().enumerate() {
            assert_eq!(lean.msg(), want[t], "message {t} l={l} k={k}");
            lean.fold(r);
        }
        assert_eq!(lean.final_eval(), want_final, "final l={l} k={k}");
    }

    #[test]
    fn lean_switch_matches_switch_prover() {
        for (l, k) in [(10, 0), (10, 1), (10, 4), (11, 3), (13, 4), (16, 4), (17, 4), (17, 5), (18, 4)] {
            check(l, k, 0x1234_5678_9ABC_DEF0 + (l * 31 + k) as u128);
        }
    }
}
