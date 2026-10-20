use super::bitmat::{transpose_tile, untranspose_planes, GF_IDENT};
use super::scalar::{B128, F162};
use crate::simd::transpose_f162::transpose8x8_q;
use std::arch::x86_64::*;

pub const LOG_PACK: usize = 7;
pub const PACK: usize = 1 << LOG_PACK;

#[derive(Clone)]
struct Tensor {
    pub c: Vec<F162>,
}

impl Tensor {
    fn from_vertical(a: F162) -> Self {
        let mut c = vec![F162::ZERO; 128];
        c[0] = a;
        Self { c }
    }

    fn scale_vertical(&self, a: F162) -> Self {
        Self {
            c: self.c.iter().map(|&x| x * a).collect(),
        }
    }

    fn shift_x(&self) -> Self {
        let mut c = vec![F162::ZERO; 128];
        c[1..].copy_from_slice(&self.c[..127]);
        let top = self.c[127];
        for k in [0usize, 1, 2, 7] {
            c[k] += top;
        }
        Self { c }
    }

    fn scale_horizontal(&self, h: B128) -> Self {
        let mut acc = Self {
            c: vec![F162::ZERO; 128],
        };
        for k in (0..128).rev() {
            acc = acc.shift_x();
            if h.bit(k) {
                for i in 0..128 {
                    acc.c[i] += self.c[i];
                }
            }
        }
        acc
    }

    fn add(&self, o: &Self) -> Self {
        Self {
            c: self.c.iter().zip(&o.c).map(|(&a, &b)| a + b).collect(),
        }
    }

    fn fold_vertical(&self, batch: &[F162]) -> F162 {
        self.c
            .iter()
            .zip(batch)
            .fold(F162::ZERO, |a, (&x, &y)| a + x * y)
    }
}

pub fn eq_expand_f162(r: &[F162]) -> Vec<F162> {
    let mut t = vec![F162::ONE];
    for &ri in r {
        let mut n = Vec::with_capacity(t.len() * 2);
        for &x in &t {
            let hi = x * ri;
            n.push(x + hi);
            n.push(hi);
        }
        t = n;
    }
    t
}

pub fn eq_expand_b128(r: &[B128]) -> Vec<B128> {
    let mut t = vec![B128::ZERO; 1 << r.len()];
    eq_expand_b128_into(r, &mut t);
    t
}

/// The expansion in place, doubling from the back so no entry is read after it is written.
pub fn eq_expand_b128_into(r: &[B128], t: &mut [B128]) {
    assert_eq!(t.len(), 1 << r.len());
    t[0] = B128::ONE;
    for (round, &ri) in r.iter().enumerate() {
        for i in (0..1 << round).rev() {
            let hi = t[i] * ri;
            t[2 * i + 1] = hi;
            t[2 * i] = t[i] + hi;
        }
    }
}

pub fn transparent_coeff(r_hi: &[B128], r_pp: &[F162], batch: &[F162]) -> F162 {
    assert_eq!(r_hi.len(), r_pp.len());
    let mut t = Tensor::from_vertical(F162::ONE);
    for (&v, &h) in r_pp.iter().zip(r_hi) {
        let vs = t.scale_vertical(v);
        let hs = t.scale_horizontal(h);
        t = t.add(&vs).add(&hs);
    }
    t.fold_vertical(batch)
}

pub fn psi_table(batch: &[F162]) -> Vec<Vec<F162>> {
    (0..16)
        .map(|chunk| {
            let mut t = vec![F162::ZERO; 256];
            for m in 1..256usize {
                let b = m.trailing_zeros() as usize;
                t[m] = t[m ^ (1 << b)] + batch[chunk * 8 + b];
            }
            t
        })
        .collect()
}

#[inline]
pub fn psi(tab: &[Vec<F162>], x: B128) -> F162 {
    let mut acc = F162::ZERO;
    for (c, t) in tab.iter().enumerate() {
        acc += t[((x.0 >> (8 * c)) & 0xff) as usize];
    }
    acc
}

#[inline(always)]
unsafe fn affine(x: __m512i, a: __m512i) -> __m512i {
    _mm512_gf2p8affine_epi64_epi8::<0>(x, a)
}

#[inline(always)]
unsafe fn xor3(a: __m512i, b: __m512i, c: __m512i) -> __m512i {
    _mm512_ternarylogic_epi64::<0x96>(a, b, c)
}

const TILE: usize = 64;

/// `v[k] = sum_j bit_k(pi0[j]) eq_hi[j]`, a `128 x n` by `n x 128` bit product: per tile the
/// bit-transposed bytes of `pi0` are the matrices, those of `eq_hi` the vectors, and
/// `acc[h][kb].qword[i].byte[t].bit[k']` sums `V[8kb + 7 - k'][8(8h + i) + t]`.
fn partial_evals(pi0: &[B128], eq_hi: &[B128]) -> Vec<B128> {
    assert_eq!(pi0.len(), eq_hi.len());
    let n = pi0.len();
    let tiles = n / TILE;
    let mut v = partial_evals_tiles(pi0, eq_hi, 0, tiles);
    for (&p, &e) in pi0[TILE * tiles..].iter().zip(&eq_hi[TILE * tiles..]) {
        for (k, vk) in v.iter_mut().enumerate() {
            if p.bit(k) {
                *vk = *vk + e;
            }
        }
    }
    v
}

/// The tiles `t0 .. t1` of [`partial_evals`] — the 128 partials over the words
/// `64 t0 .. 64 t1` alone. The accumulation is GF(2)-linear, so the partials of a split range
/// are the XOR of the partials of its parts.
fn partial_evals_tiles(pi0: &[B128], eq_hi: &[B128], t0: usize, t1: usize) -> Vec<B128> {
    debug_assert!(TILE * t1 <= pi0.len() && pi0.len() == eq_hi.len());
    let mut v = vec![B128::ZERO; PACK];
    unsafe {
        let ident = _mm512_set1_epi64(GF_IDENT);
        let mut acc = [[_mm512_setzero_si512(); 16]; 2];
        let mut xe = [_mm512_setzero_si512(); 16];
        let mut xp = [_mm512_setzero_si512(); 16];
        let mut ap = [[0u64; 8]; 16];
        for t in t0..t1 {
            transpose_tile(eq_hi.as_ptr().add(TILE * t) as *const u8, &mut xe);
            transpose_tile(pi0.as_ptr().add(TILE * t) as *const u8, &mut xp);
            for k in 0..16 {
                _mm512_storeu_si512(ap[k].as_mut_ptr() as *mut __m512i, affine(ident, xp[k]));
            }
            let te: [__m512i; 16] = core::array::from_fn(|b| affine(ident, xe[b]));
            for (h, acc) in acc.iter_mut().enumerate() {
                let tep = transpose8x8_q(core::array::from_fn(|i| te[8 * h + i]));
                for (k, a) in acc.iter_mut().enumerate() {
                    for j in (0..8).step_by(2) {
                        *a = xor3(
                            *a,
                            affine(tep[j], _mm512_set1_epi64(ap[k][j] as i64)),
                            affine(tep[j + 1], _mm512_set1_epi64(ap[k][j + 1] as i64)),
                        );
                    }
                }
            }
        }
        for (h, acc) in acc.iter().enumerate() {
            for (k, a) in acc.iter().enumerate() {
                let q: [u64; 8] = core::mem::transmute(*a);
                for (i, qi) in q.iter().enumerate() {
                    for t in 0..8 {
                        let byte = (qi >> (8 * t)) & 0xff;
                        for kp in 0..8 {
                            if (byte >> kp) & 1 == 1 {
                                v[8 * k + 7 - kp].0 ^= 1u128 << (8 * (8 * h + i) + t);
                            }
                        }
                    }
                }
            }
        }
    }
    v
}

/// [`partial_evals`] with the tiles split over `threads` threads and the per-thread partials
/// XORed together; the sub-tile tail is taken once, serially.
pub fn partial_evals_parallel(pi0: &[B128], eq_hi: &[B128], threads: usize) -> Vec<B128> {
    assert_eq!(pi0.len(), eq_hi.len());
    let n = pi0.len();
    let tiles = n / TILE;
    let runs = crate::par::ranges(tiles, threads, 1);
    let parts: Vec<Vec<B128>> = std::thread::scope(|scope| {
        let handles: Vec<_> = runs
            .into_iter()
            .map(|run| scope.spawn(move || partial_evals_tiles(pi0, eq_hi, run.start, run.end)))
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let mut v = vec![B128::ZERO; PACK];
    for part in parts {
        for (vk, &x) in v.iter_mut().zip(&part) {
            *vk = *vk + x;
        }
    }
    for (&p, &e) in pi0[TILE * tiles..].iter().zip(&eq_hi[TILE * tiles..]) {
        for (k, vk) in v.iter_mut().enumerate() {
            if p.bit(k) {
                *vk = *vk + e;
            }
        }
    }
    v
}

/// The eight `B128` at `x` as their low words and their high words.
#[inline(always)]
unsafe fn split_words(x: *const B128) -> (__m512i, __m512i) {
    let a = _mm512_loadu_si512(x as *const __m512i);
    let b = _mm512_loadu_si512(x.add(4) as *const __m512i);
    let lo = _mm512_setr_epi64(0, 2, 4, 6, 8, 10, 12, 14);
    let hi = _mm512_setr_epi64(1, 3, 5, 7, 9, 11, 13, 15);
    (
        _mm512_permutex2var_epi64(a, lo, b),
        _mm512_permutex2var_epi64(a, hi, b),
    )
}

/// `pi0` in the sumcheck's layout: the bits of a `B128` are the low coefficients of an `F162`.
fn poly_from_b128(pi0: &[B128], out: &mut super::sumcheck::Poly) {
    let n = pi0.len();
    assert_eq!(out.n, n);
    let w = &mut out.w;
    w[2].fill(0);
    let blocks = n / 8;
    unsafe {
        for b in 0..blocks {
            let (lo, hi) = split_words(pi0.as_ptr().add(8 * b));
            _mm512_storeu_si512(w[0].as_mut_ptr().add(8 * b) as *mut __m512i, lo);
            _mm512_storeu_si512(w[1].as_mut_ptr().add(8 * b) as *mut __m512i, hi);
        }
    }
    for j in 8 * blocks..n {
        w[0][j] = pi0[j].0 as u64;
        w[1][j] = (pi0[j].0 >> 64) as u64;
    }
}

/// `m[cb][kb].byte[7 - c].bit[k] = batch[8kb + k].bit(8cb + c)`: the matrix taking byte `kb` of
/// a word to byte `cb` of its `F162` image.
fn eq_matrices(batch: &[F162]) -> [[u64; 16]; 24] {
    let mut m = [[0u64; 16]; 24];
    for (cb, row) in m.iter_mut().enumerate() {
        for (kb, q) in row.iter_mut().enumerate() {
            for c in 0..8 {
                let mut byte = 0u64;
                for k in 0..8 {
                    byte |= ((batch[8 * kb + k].0[cb / 8] >> (8 * (cb % 8) + c)) & 1) << k;
                }
                *q |= byte << (8 * (7 - c));
            }
        }
    }
    m
}

/// `a[j] = sum_i sum_k bit_k(eq[i][j]) batch[i][k]`, straight into the sumcheck's layout: per
/// tile the byte planes of `eq` through [`eq_matrices`], and the 24 output planes back to qwords.
fn poly_from_eq(eqs: &[&[B128]], batches: &[Vec<F162>], out: &mut super::sumcheck::Poly) {
    let n = eqs[0].len();
    assert_eq!(out.n, n);
    let w = &mut out.w;
    let tiles = n / TILE;
    let ms: Vec<[[u64; 16]; 24]> = batches.iter().map(|b| eq_matrices(b)).collect();
    unsafe {
        let mut xe = [_mm512_setzero_si512(); 16];
        for t in 0..tiles {
            let mut acc = [_mm512_setzero_si512(); 24];
            for (eq, m) in eqs.iter().zip(&ms) {
                transpose_tile(eq.as_ptr().add(TILE * t) as *const u8, &mut xe);
                for (a, row) in acc.iter_mut().zip(m) {
                    for k in (0..16).step_by(2) {
                        *a = xor3(
                            *a,
                            affine(xe[k], _mm512_set1_epi64(row[k] as i64)),
                            affine(xe[k + 1], _mm512_set1_epi64(row[k + 1] as i64)),
                        );
                    }
                }
            }
            for (l, wl) in w.iter_mut().enumerate() {
                let out = untranspose_planes(core::array::from_fn(|c| acc[8 * l + c]));
                for (g, o) in out.iter().enumerate() {
                    _mm512_storeu_si512(wl.as_mut_ptr().add(TILE * t + 8 * g) as *mut __m512i, *o);
                }
            }
        }
    }
    if TILE * tiles < n {
        let tabs: Vec<_> = batches.iter().map(|b| psi_table(b)).collect();
        for j in TILE * tiles..n {
            let x = eqs
                .iter()
                .zip(&tabs)
                .fold(F162::ZERO, |acc, (e, t)| acc + psi(t, e[j]));
            for l in 0..3 {
                w[l][j] = x.0[l];
            }
        }
    }
}

pub struct SwitchProof {
    pub v: Vec<B128>,
    pub rounds: Vec<[F162; 2]>,
    pub final_eval: F162,
}

pub struct Transcript {
    pub r_prime: Vec<F162>,
    pub r_pp: Vec<F162>,
}

pub fn prove(pi0: &[B128], r_lo: &[B128], r_hi: &[B128], challenges: &Transcript) -> SwitchProof {
    assert_eq!(r_lo.len(), LOG_PACK);
    let l = r_hi.len();
    assert_eq!(pi0.len(), 1 << l);

    let eq_hi = eq_expand_b128(r_hi);
    let v = partial_evals(pi0, &eq_hi);

    let batch = eq_expand_f162(&challenges.r_prime);
    let mut ap = super::sumcheck::Poly::zero(pi0.len());
    let mut pp = super::sumcheck::Poly::zero(pi0.len());
    poly_from_eq(&[&eq_hi], &[batch], &mut ap);
    poly_from_b128(pi0, &mut pp);
    let mut rounds = Vec::with_capacity(l);
    let mut half = 1usize << l;
    for round in 0..l {
        half /= 2;
        rounds.push(super::sumcheck::round(
            &mut ap,
            &mut pp,
            half,
            challenges.r_pp[round],
        ));
    }
    let final_eval = pp.get(0);

    SwitchProof {
        v,
        rounds,
        final_eval,
    }
}

pub fn verify(
    proof: &SwitchProof,
    claim: B128,
    r_lo: &[B128],
    r_hi: &[B128],
    challenges: &Transcript,
) -> Result<F162, &'static str> {
    let eq_lo = eq_expand_b128(r_lo);
    let recomputed = proof
        .v
        .iter()
        .zip(&eq_lo)
        .fold(B128::ZERO, |acc, (&vi, &e)| acc + vi * e);
    if recomputed != claim {
        return Err("partial evaluation mismatch");
    }

    let mut u = vec![0u128; PACK];
    for (i, &vi) in proof.v.iter().enumerate() {
        for k in 0..PACK {
            if vi.bit(k) {
                u[k] |= 1u128 << i;
            }
        }
    }
    let batch = eq_expand_f162(&challenges.r_prime);
    let mut s = F162::ZERO;
    for k in 0..PACK {
        s += F162::from_b128(B128(u[k])) * batch[k];
    }

    for (round, msg) in proof.rounds.iter().enumerate() {
        let [e0, einf] = *msg;
        let e1 = s + e0;
        let r = challenges.r_pp[round];
        s = e0 + r * (e0 + e1 + einf) + r * r * einf;
    }

    let d = transparent_coeff(r_hi, &challenges.r_pp, &batch);
    if s != d * proof.final_eval {
        return Err("sumcheck final check failed");
    }
    Ok(proof.final_eval)
}

/// The switch's buffers, sized once and reused: fresh pages cost more than the kernels do.
pub struct Workspace {
    eq: Vec<Vec<B128>>,
    a: super::sumcheck::Poly,
    p: super::sumcheck::Poly,
}

impl Workspace {
    pub fn new(l: usize, claims: usize) -> Self {
        Self {
            eq: (0..claims).map(|_| vec![B128::ZERO; 1 << l]).collect(),
            a: super::sumcheck::Poly::zero(1 << l),
            p: super::sumcheck::Poly::zero(1 << l),
        }
    }

    pub fn fits(&self, l: usize, claims: usize) -> bool {
        self.eq.len() == claims && self.a.n == 1 << l
    }

    /// The partial evaluations of claim `i` at `r_hi`; its `eq` stays for the sumcheck.
    pub fn partial_evals(&mut self, i: usize, pi0: &[B128], r_hi: &[B128]) -> Vec<B128> {
        eq_expand_b128_into(r_hi, &mut self.eq[i]);
        partial_evals(pi0, &self.eq[i])
    }
}

pub struct SwitchProver {
    ws: Workspace,
    half: usize,
}

impl SwitchProver {
    pub fn partial_evals_and_eq(pi0: &[B128], r_hi: &[B128]) -> (Vec<B128>, Vec<B128>) {
        let eq_hi = eq_expand_b128(r_hi);
        let v = partial_evals(pi0, &eq_hi);
        (v, eq_hi)
    }

    pub fn new(pi0: &[B128], eq_hi: &[B128], batch: &[F162]) -> Self {
        Self::batched(pi0, &[eq_hi], &[F162::ONE], batch)
    }

    pub fn batched(pi0: &[B128], eq_his: &[&[B128]], gammas: &[F162], batch: &[F162]) -> Self {
        let mut ws = Workspace::new(pi0.len().trailing_zeros() as usize, eq_his.len());
        for (dst, src) in ws.eq.iter_mut().zip(eq_his) {
            dst.copy_from_slice(src);
        }
        Self::from_workspace(ws, pi0, gammas, batch)
    }

    /// The sumcheck over the `eq` the workspace holds from [`Workspace::partial_evals`].
    pub fn from_workspace(
        mut ws: Workspace,
        pi0: &[B128],
        gammas: &[F162],
        batch: &[F162],
    ) -> Self {
        let batches: Vec<Vec<F162>> = gammas
            .iter()
            .map(|&g| batch.iter().map(|&x| g * x).collect())
            .collect();
        let views: Vec<&[B128]> = ws.eq.iter().map(Vec::as_slice).collect();
        poly_from_eq(&views, &batches, &mut ws.a);
        poly_from_b128(pi0, &mut ws.p);
        Self {
            ws,
            half: pi0.len(),
        }
    }

    pub fn into_workspace(self) -> Workspace {
        self.ws
    }

    pub fn msg(&mut self) -> [F162; 2] {
        self.half /= 2;
        super::sumcheck::msg(&self.ws.a, &self.ws.p, self.half)
    }

    pub fn fold(&mut self, r: F162) {
        super::sumcheck::fold(&mut self.ws.a, &mut self.ws.p, self.half, r);
    }

    pub fn final_eval(&self) -> F162 {
        self.ws.p.get(0)
    }
}

pub fn slice_sum(v: &[B128], batch: &[F162]) -> F162 {
    let mut u = vec![0u128; PACK];
    for (i, &vi) in v.iter().enumerate() {
        for k in 0..PACK {
            if vi.bit(k) {
                u[k] |= 1u128 << i;
            }
        }
    }
    (0..PACK).fold(F162::ZERO, |s, k| {
        s + F162::from_b128(B128(u[k])) * batch[k]
    })
}

pub struct SwitchVerifier {
    pub s: F162,
    round: usize,
}

impl SwitchVerifier {
    pub fn start(
        v: &[B128],
        claim: B128,
        r_lo: &[B128],
        batch: &[F162],
    ) -> Result<Self, &'static str> {
        let eq_lo = eq_expand_b128(r_lo);
        let recomputed = v
            .iter()
            .zip(&eq_lo)
            .fold(B128::ZERO, |acc, (&vi, &e)| acc + vi * e);
        if recomputed != claim {
            return Err("partial evaluation mismatch");
        }
        Ok(Self::from_sum(slice_sum(v, batch)))
    }

    pub const fn from_sum(s: F162) -> Self {
        Self { s, round: 0 }
    }

    pub fn round(&mut self, msg: [F162; 2], r: F162) {
        let [e0, einf] = msg;
        let e1 = self.s + e0;
        self.s = e0 + r * (e0 + e1 + einf) + r * r * einf;
        self.round += 1;
    }

    pub fn finish(
        &self,
        r_hi: &[B128],
        r_pp: &[F162],
        batch: &[F162],
        opened: F162,
    ) -> Result<(), &'static str> {
        let d = transparent_coeff(r_hi, r_pp, batch);
        if d == F162::ZERO {
            return Err("the switch left the opened value unbound");
        }
        if self.s != d * opened {
            return Err("sumcheck final check failed");
        }
        Ok(())
    }
}

pub fn eval_pi1(pi0: &[B128], r_pp: &[F162]) -> F162 {
    let eq = eq_expand_f162(r_pp);
    pi0.iter()
        .zip(&eq)
        .fold(F162::ZERO, |a, (&p, &e)| a + F162::from_b128(p) * e)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference(pi0: &[B128], eq_hi: &[B128]) -> Vec<B128> {
        let mut v = vec![B128::ZERO; PACK];
        for (&p, &e) in pi0.iter().zip(eq_hi) {
            for (k, vk) in v.iter_mut().enumerate() {
                if p.bit(k) {
                    *vk = *vk + e;
                }
            }
        }
        v
    }

    #[test]
    fn poly_from_eq_matches_psi() {
        let mut x = 0x2545F4914F6CDD1Du128;
        let mut next = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let batches: Vec<Vec<F162>> = (0..2)
            .map(|_| {
                (0..PACK)
                    .map(|_| F162([next() as u64, next() as u64, (next() as u64) & M34_TEST]))
                    .collect()
            })
            .collect();
        let tabs: Vec<_> = batches.iter().map(|b| psi_table(b)).collect();
        for n in [1usize, 8, 13, 64, 200, 1024] {
            let eqs: Vec<Vec<B128>> = (0..2)
                .map(|_| (0..n).map(|_| B128(next())).collect())
                .collect();
            let views: Vec<&[B128]> = eqs.iter().map(|e| e.as_slice()).collect();
            let mut poly = super::super::sumcheck::Poly::zero(n);
            poly_from_eq(&views, &batches, &mut poly);
            for j in 0..n {
                let want = views
                    .iter()
                    .zip(&tabs)
                    .fold(F162::ZERO, |acc, (e, t)| acc + psi(t, e[j]));
                assert_eq!(poly.get(j), want);
            }
            let p: Vec<B128> = (0..n).map(|_| B128(next())).collect();
            let mut poly = super::super::sumcheck::Poly::zero(n);
            poly_from_b128(&p, &mut poly);
            for j in 0..n {
                assert_eq!(poly.get(j), F162::from_b128(p[j]));
            }
        }
    }

    const M34_TEST: u64 = (1 << 34) - 1;

    #[test]
    fn workspace_reuse_is_clean() {
        let mut x = 0x0123_4567_89AB_CDEFu128;
        let mut next = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let l = 8;
        let pi0: Vec<B128> = (0..1 << l).map(|_| B128(next())).collect();
        let r_hi: Vec<B128> = (0..l).map(|_| B128(next())).collect();
        let batch: Vec<F162> = (0..PACK)
            .map(|_| F162([next() as u64, next() as u64, (next() as u64) & M34_TEST]))
            .collect();
        let r: Vec<F162> = (0..l)
            .map(|_| F162([next() as u64, next() as u64, (next() as u64) & M34_TEST]))
            .collect();
        let run = |ws: Workspace| {
            let mut ws = ws;
            let v = ws.partial_evals(0, &pi0, &r_hi);
            let mut p = SwitchProver::from_workspace(ws, &pi0, &[F162::ONE], &batch);
            let mut msgs = Vec::new();
            for &ri in &r {
                msgs.push(p.msg());
                p.fold(ri);
            }
            (v, msgs, p.final_eval(), p.into_workspace())
        };
        let (v1, m1, f1, ws) = run(Workspace::new(l, 1));
        let (v2, m2, f2, _) = run(ws);
        assert_eq!((v1, m1, f1), (v2, m2, f2));
    }

    #[test]
    fn partial_evals_match_the_bitwise_sum() {
        let mut x = 0x9E3779B97F4A7C15u128;
        let mut next = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            B128(x.wrapping_mul(0x2545F4914F6CDD1Du128) ^ (x >> 64))
        };
        for n in [1usize, 7, 8, 64, 1000, 4096] {
            let pi0: Vec<B128> = (0..n).map(|_| next()).collect();
            let eq: Vec<B128> = (0..n).map(|_| next()).collect();
            assert_eq!(partial_evals(&pi0, &eq), reference(&pi0, &eq));
        }
    }
}

// =============================================================================================
// buffers the big tables live in
// =============================================================================================

/// `len` elements of `T` on 2 MB-aligned storage with transparent huge pages requested where the
/// platform has them (a hint; the arithmetic is the same on 4 KB pages). Uninitialised: the
/// owner writes every element before reading any, through [`as_mut_ptr`](Self::as_mut_ptr).
pub struct HugeBuf<T> {
    ptr: *mut T,
    len: usize,
    layout: std::alloc::Layout,
}

unsafe impl<T: Send> Send for HugeBuf<T> {}
unsafe impl<T: Sync> Sync for HugeBuf<T> {}

#[cfg(target_os = "linux")]
extern "C" {
    fn madvise(addr: *mut core::ffi::c_void, len: usize, advice: i32) -> i32;
}

impl<T> HugeBuf<T> {
    pub fn new(len: usize) -> Self {
        let bytes = (len * core::mem::size_of::<T>()).max(1);
        let align = (1usize << 21).max(core::mem::align_of::<T>());
        let layout = std::alloc::Layout::from_size_align(bytes, align).expect("a valid layout");
        let ptr = unsafe { std::alloc::alloc(layout) } as *mut T;
        assert!(!ptr.is_null(), "out of memory allocating {bytes} bytes");
        #[cfg(target_os = "linux")]
        unsafe {
            // MADV_HUGEPAGE = 14; the return value is advisory.
            madvise(ptr as *mut core::ffi::c_void, bytes, 14);
        }
        HugeBuf { ptr, len, layout }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }

    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }

    /// The elements as a slice.
    ///
    /// # Safety
    /// Every element must have been written.
    pub unsafe fn as_slice(&self) -> &[T] {
        core::slice::from_raw_parts(self.ptr, self.len)
    }

    /// The buffer's address and size in bytes, for [`touch_pages`] on another thread.
    pub fn region(&self) -> (usize, usize) {
        (self.ptr as usize, self.len * core::mem::size_of::<T>())
    }
}

/// Fault in every page of `[addr, addr + bytes)` by writing one zero byte per 4 KB (a huge
/// page faults on its first touch and the rest of its touches hit), so that the first real
/// writer finds the pages mapped. Meant for a background thread while a compute-bound phase
/// runs.
///
/// # Safety
/// The region must be writable, owned by the caller, and not read or written by anyone else
/// until this returns.
pub unsafe fn touch_pages(addr: usize, bytes: usize) {
    let p = addr as *mut u8;
    let mut off = 0;
    while off < bytes {
        core::ptr::write_volatile(p.add(off), 0);
        off += 4096;
    }
    if bytes > 0 {
        core::ptr::write_volatile(p.add(bytes - 1), 0);
    }
}

impl<T> Drop for HugeBuf<T> {
    fn drop(&mut self) {
        unsafe { std::alloc::dealloc(self.ptr as *mut u8, self.layout) };
    }
}

// =============================================================================================
// the expansion on several threads
// =============================================================================================

/// [`eq_expand_b128_into`] on up to `threads` threads, into the `2^r.len()` words at `out`.
///
/// The serial doubling appends the newest coordinate in the lowest bit of the index, so entry
/// `j` of the table is `high[j >> h] * low[j & (2^h - 1)]` with `high` the expansion of the
/// first `n - h` coordinates and `low` that of the last `h`: the two small tables are built
/// serially and the product is cut over the threads along `j >> h`, each writing its own
/// contiguous run. Below 16 coordinates the serial expansion is used as it is.
///
/// # Safety
/// `out` must be writable for `2^r.len()` words.
pub unsafe fn eq_expand_b128_parallel(r: &[B128], out: *mut B128, threads: usize) {
    let n = r.len();
    if n < 16 || threads <= 1 {
        let t = eq_expand_b128(r);
        core::ptr::copy_nonoverlapping(t.as_ptr(), out, t.len());
        return;
    }
    let h = n / 2;
    let high = eq_expand_b128(&r[..n - h]);
    let low = eq_expand_b128(&r[n - h..]);
    let runs = crate::par::ranges(high.len(), threads, 1);
    let out = out as usize;
    std::thread::scope(|scope| {
        for run in runs {
            let (high, low) = (&high, &low);
            scope.spawn(move || {
                let out = out as *mut B128;
                for jh in run {
                    let base = unsafe { out.add(jh << h) };
                    unsafe { scale_low(high[jh], low, base) };
                }
            });
        }
    });
}

/// `dst[j] = hv * low[j]` for every entry, four words at a time. The product is the scalar
/// `B128` multiply (same polynomial, same Karatsuba, same two reduction rounds).
///
/// # Safety
/// `dst` is writable for `low.len()` words. AVX-512F and VPCLMULQDQ.
#[target_feature(enable = "avx512f,avx512bw,vpclmulqdq")]
unsafe fn scale_low(hv: B128, low: &[B128], dst: *mut B128) {
    let n = low.len();
    let hb = _mm512_broadcast_i32x4(_mm_loadu_si128(&hv as *const B128 as *const __m128i));
    let mut j = 0;
    while j + 4 <= n {
        let lv = _mm512_loadu_si512(low.as_ptr().add(j) as *const __m512i);
        _mm512_storeu_si512(dst.add(j) as *mut __m512i, ghash_mul4(hb, lv));
        j += 4;
    }
    for jl in j..n {
        dst.add(jl).write(hv * low[jl]);
    }
}

/// Four GHASH products, one per 128-bit lane. Matches [`B128`]'s `Mul`.
#[target_feature(enable = "avx512f,avx512bw,vpclmulqdq")]
unsafe fn ghash_mul4(a: __m512i, b: __m512i) -> __m512i {
    let swap = _mm512_shuffle_epi32::<0x4E>(a);
    let a01 = _mm512_xor_si512(a, swap);
    let swap = _mm512_shuffle_epi32::<0x4E>(b);
    let b01 = _mm512_xor_si512(b, swap);
    let lo = _mm512_clmulepi64_epi128::<0x00>(a, b);
    let hi = _mm512_clmulepi64_epi128::<0x11>(a, b);
    let mid = _mm512_xor_si512(
        _mm512_clmulepi64_epi128::<0x00>(a01, b01),
        _mm512_xor_si512(lo, hi),
    );
    let p_lo = _mm512_xor_si512(lo, _mm512_bslli_epi128::<8>(mid));
    let p_hi = _mm512_xor_si512(hi, _mm512_bsrli_epi128::<8>(mid));
    let poly = _mm512_set1_epi64(0x87);
    let mut acc = p_lo;
    let mut h = p_hi;
    for _ in 0..2 {
        let h_hi = _mm512_bsrli_epi128::<8>(h);
        let c_lo = _mm512_clmulepi64_epi128::<0x00>(h, poly);
        let c_hi = _mm512_clmulepi64_epi128::<0x00>(h_hi, poly);
        acc = _mm512_xor_si512(acc, _mm512_xor_si512(c_lo, _mm512_bslli_epi128::<8>(c_hi)));
        h = _mm512_bsrli_epi128::<8>(c_hi);
    }
    acc
}

// =============================================================================================
// the lean switch prover
// =============================================================================================
//
// [`SwitchProver`] materialises the two sumcheck polynomials in full — `a[j] = sum_k bit_k(eq[j])
// batch[k]` and `p[j] = lift(pi0[j])`, 24 bytes each, 2^l of each — and halves both round by
// round. Neither is needed in that form.
//
// `a` is a GF(2)-linear image of the `eq` word: `a_0[j] = L_0(eq[j])` with `L_0` the map
// `poly_from_eq` applies. The fold `a_{k+1}[j] = a_k[j] + r_k (a_k[j] + a_k[j + half])` keeps
// it one: bit `l - 1 - k` of the index is coordinate `k`, so for `j < half` the two halves
// differ by the constant `c_k = r_hi[k] / (1 + r_hi[k])`, `eq[j + half] = c_k eq[j]`, and
//
//     a_{k+1}[j] = L_k(eq[j]) + r_k L_k((1 + c_k) eq[j]) = L_{k+1}(eq[j]),
//     L_{k+1}(x) = L_k(x) + r_k L_k(d_k x),     d_k = 1 + c_k = (1 + r_hi[k])^-1,
//
// which is again linear in `x`. So the prover keeps `L_k` as the images of the 128 basis words
// and applies it to the *original* `eq` table, which it never rewrites, through the same GFNI
// tiles `poly_from_eq` uses; the message's `a_k[j] + a_k[j + half]` is the second map
// `L'_k = L_k o (x -> d_k x)` on the same tile. The identity needs `1 + r_hi[k] != 0`; a point
// with a coordinate equal to one is refused by [`LeanSwitch::new`] and the caller falls back.
//
// `p` has no such structure past round 0, but rounds 0 and 1 read it straight from the words
// (`p_1` is one fold of two words, formed on the fly and never stored), and every later
// round's fold is fused with the next round's message: one pass per round that reads `p_k`,
// writes `p_{k+1}` in place over the low half and accumulates the message of round `k + 1`
// from the elements it just wrote. The words are read three times, `eq` is read once per round
// at the round's length, and `p` lives in `2^(l-2)` elements written once each.
//
// The messages are the values [`SwitchProver`] sends: the same products, reassociated over the
// threads (XOR), so the transcript and the proof are unchanged.

use super::sumcheck::Poly;

/// `x^-1` in `B128` (`x^(2^128 - 2)`); `x` must be nonzero.
fn b128_inv(x: B128) -> B128 {
    let mut r = B128::ONE;
    let mut b = x;
    let mut e: u128 = u128::MAX - 1;
    while e != 0 {
        if e & 1 == 1 {
            r = r * b;
        }
        b = b * b;
        e >>= 1;
    }
    r
}

/// `L(x) = sum_k bit_k(x) images[k]`.
fn apply_images(images: &[F162], x: B128) -> F162 {
    let mut acc = F162::ZERO;
    for (k, &im) in images.iter().enumerate() {
        if x.bit(k) {
            acc += im;
        }
    }
    acc
}

/// `L o (x -> d x)` as images: `L'(e_i) = L(d e_i)`.
fn compose_scale(images: &[F162], d: B128) -> Vec<F162> {
    (0..PACK)
        .map(|i| apply_images(images, B128(1u128 << i) * d))
        .collect()
}

/// Where a pass reads `p_k` from: the words themselves in round 0, the words folded once by
/// the first challenge in round 1 (`p_1[j] = p_0[j] + r_0 (p_0[j] + p_0[j + half_0])`, never
/// stored), and the planes from round 2 on.
#[derive(Clone, Copy)]
enum Src {
    Words(*const B128),
    WordsFolded {
        words: *const B128,
        half: usize,
        r: F162,
    },
    Planes([*const u64; 3]),
}

#[inline(always)]
unsafe fn load_words(w: *const B128) -> [__m512i; 3] {
    let (lo, hi) = split_words(w);
    [lo, hi, _mm512_setzero_si512()]
}

#[inline(always)]
unsafe fn load_p(src: Src, j: usize) -> [__m512i; 3] {
    match src {
        Src::Words(w) => load_words(w.add(j)),
        Src::WordsFolded { words, half, r } => {
            fold8(bcast3(r), load_words(words.add(j)), load_words(words.add(j + half)))
        }
        Src::Planes(p) => [
            _mm512_loadu_si512(p[0].add(j) as *const __m512i),
            _mm512_loadu_si512(p[1].add(j) as *const __m512i),
            _mm512_loadu_si512(p[2].add(j) as *const __m512i),
        ],
    }
}

#[inline(always)]
unsafe fn store_p(dst: [*mut u64; 3], j: usize, v: [__m512i; 3]) {
    for k in 0..3 {
        _mm512_storeu_si512(dst[k].add(j) as *mut __m512i, v[k]);
    }
}

unsafe fn get_p(src: Src, j: usize) -> F162 {
    match src {
        Src::Words(w) => F162::from_b128(*w.add(j)),
        Src::WordsFolded { words, half, r } => {
            let (p0, p1) = (F162::from_b128(*words.add(j)), F162::from_b128(*words.add(j + half)));
            p0 + r * (p0 + p1)
        }
        Src::Planes(p) => F162([*p[0].add(j), *p[1].add(j), *p[2].add(j)]),
    }
}

unsafe fn set_p(dst: [*mut u64; 3], j: usize, x: F162) {
    for k in 0..3 {
        *dst[k].add(j) = x.0[k];
    }
}

#[inline(always)]
unsafe fn xor3p(a: [__m512i; 3], b: [__m512i; 3]) -> [__m512i; 3] {
    [
        _mm512_xor_si512(a[0], b[0]),
        _mm512_xor_si512(a[1], b[1]),
        _mm512_xor_si512(a[2], b[2]),
    ]
}

#[inline(always)]
unsafe fn bcast3(x: F162) -> [__m512i; 3] {
    [
        _mm512_set1_epi64(x.0[0] as i64),
        _mm512_set1_epi64(x.0[1] as i64),
        _mm512_set1_epi64(x.0[2] as i64),
    ]
}

/// `p0 + r (p0 + p1)` on eight elements.
#[inline(always)]
unsafe fn fold8(rb: [__m512i; 3], p0: [__m512i; 3], p1: [__m512i; 3]) -> [__m512i; 3] {
    xor3p(p0, super::f162::mul_soa8(rb, xor3p(p0, p1)))
}

/// The eight lanes of a finished accumulator summed into one element.
unsafe fn horiz8(acc: [__m512i; 12]) -> F162 {
    let r = super::f162::reduce_soa8(acc);
    let mut w = [[0u64; 8]; 3];
    for k in 0..3 {
        _mm512_storeu_si512(w[k].as_mut_ptr() as *mut __m512i, r[k]);
    }
    let mut out = F162::ZERO;
    for i in 0..8 {
        out += F162([w[0][i], w[1][i], w[2][i]]);
    }
    out
}

/// Both maps applied to the 64 words at `eq`, sharing one transpose of the tile. `out[l][g]` is
/// limb `l` of words `8 g .. 8 g + 8`.
#[inline(always)]
unsafe fn map_tile2(
    eq: *const B128,
    m0: &[[u64; 16]; 24],
    m1: &[[u64; 16]; 24],
    o0: &mut [[__m512i; 8]; 3],
    o1: &mut [[__m512i; 8]; 3],
) {
    let mut xe = [_mm512_setzero_si512(); 16];
    transpose_tile(eq as *const u8, &mut xe);
    apply_map(&xe, m0, o0);
    apply_map(&xe, m1, o1);
}

#[inline(always)]
unsafe fn apply_map(xe: &[__m512i; 16], m: &[[u64; 16]; 24], out: &mut [[__m512i; 8]; 3]) {
    let mut acc = [_mm512_setzero_si512(); 24];
    for (a, row) in acc.iter_mut().zip(m) {
        for k in (0..16).step_by(2) {
            *a = xor3(
                *a,
                affine(xe[k], _mm512_set1_epi64(row[k] as i64)),
                affine(xe[k + 1], _mm512_set1_epi64(row[k + 1] as i64)),
            );
        }
    }
    for (l, o) in out.iter_mut().enumerate() {
        *o = untranspose_planes(core::array::from_fn(|c| acc[8 * l + c]));
    }
}

/// The message of a round over `j` in `range`, `half` the round's half: `e0 = sum L(eq[j])
/// p[j]`, `einf = sum L'(eq[j]) (p[j] + p[j + half])`.
unsafe fn message_range(
    eq: *const B128,
    src: Src,
    half: usize,
    ml: &[[u64; 16]; 24],
    mlp: &[[u64; 16]; 24],
    range: core::ops::Range<usize>,
) -> [F162; 2] {
    let mut acc0 = [_mm512_setzero_si512(); 12];
    let mut acci = [_mm512_setzero_si512(); 12];
    let mut av = [[_mm512_setzero_si512(); 8]; 3];
    let mut apv = [[_mm512_setzero_si512(); 8]; 3];
    for j0 in range.step_by(TILE) {
        map_tile2(eq.add(j0), ml, mlp, &mut av, &mut apv);
        for g in 0..8 {
            let j = j0 + 8 * g;
            let p0 = load_p(src, j);
            let p1 = load_p(src, j + half);
            super::f162::mac_soa8(&mut acc0, [av[0][g], av[1][g], av[2][g]], p0);
            super::f162::mac_soa8(&mut acci, [apv[0][g], apv[1][g], apv[2][g]], xor3p(p0, p1));
        }
    }
    [horiz8(acc0), horiz8(acci)]
}

/// One fused step over `j` in `range` (a sub-range of `0 .. hh`): the fold of round `k`,
/// `p_{k+1}[i] = p_k[i] + r (p_k[i] + p_k[i + half])` for `i = j` and `i = j + hh`, written to
/// `dst`, and the message of round `k + 1` (half `hh`) over the elements just written.
unsafe fn fold_message_range(
    eq: *const B128,
    src: Src,
    dst: [*mut u64; 3],
    half: usize,
    hh: usize,
    r: F162,
    ml: &[[u64; 16]; 24],
    mlp: &[[u64; 16]; 24],
    range: core::ops::Range<usize>,
) -> [F162; 2] {
    let rb = bcast3(r);
    let mut acc0 = [_mm512_setzero_si512(); 12];
    let mut acci = [_mm512_setzero_si512(); 12];
    let mut av = [[_mm512_setzero_si512(); 8]; 3];
    let mut apv = [[_mm512_setzero_si512(); 8]; 3];
    for j0 in range.step_by(TILE) {
        map_tile2(eq.add(j0), ml, mlp, &mut av, &mut apv);
        for g in 0..8 {
            let j = j0 + 8 * g;
            let q0 = fold8(rb, load_p(src, j), load_p(src, j + half));
            store_p(dst, j, q0);
            let q1 = fold8(rb, load_p(src, j + hh), load_p(src, j + hh + half));
            store_p(dst, j + hh, q1);
            super::f162::mac_soa8(&mut acc0, [av[0][g], av[1][g], av[2][g]], q0);
            super::f162::mac_soa8(&mut acci, [apv[0][g], apv[1][g], apv[2][g]], xor3p(q0, q1));
        }
    }
    [horiz8(acc0), horiz8(acci)]
}

/// The switch prover that keeps `a` as a linear map and `p` in one half-size buffer (see the
/// section comment). Drives the same transcript as [`SwitchProver`]: `msg`, `fold`, `msg`, ...
/// `final_eval`.
pub struct LeanSwitch<'a> {
    pi0: &'a [B128],
    eq: &'a [B128],
    r_hi: &'a [B128],
    /// `L_k`, as the images of the 128 basis words.
    l: Vec<F162>,
    /// `L'_k = L_k o (x -> d_k x)`.
    lp: Vec<F162>,
    /// `p_k` for `k >= 2`, word-sliced; [`LeanSwitch::plane_len`] elements per plane.
    p: [HugeBuf<u64>; 3],
    /// Round 0's challenge, which defines the virtual `p_1`.
    r0: F162,
    /// The current round's half, once `msg` has been called.
    half: usize,
    round: usize,
    pending: Option<[F162; 2]>,
    threads: usize,
}

impl<'a> LeanSwitch<'a> {
    /// Elements each of the three planes must hold for `n` words: `p_2` is `n / 4` elements,
    /// and the final value needs one.
    pub fn plane_len(n: usize) -> usize {
        (n / 4).max(1)
    }

    /// `None` when some coordinate of `r_hi` is one, which the caller treats with
    /// [`SwitchProver`]. `planes` are the three buffers `p` lives in, at least
    /// [`plane_len`](Self::plane_len) elements each; they may be prefaulted ahead of time.
    pub fn new(
        pi0: &'a [B128],
        eq: &'a [B128],
        r_hi: &'a [B128],
        batch: &[F162],
        threads: usize,
        planes: [HugeBuf<u64>; 3],
    ) -> Option<Self> {
        let n = pi0.len();
        assert!(n >= 2 && n.is_power_of_two() && n == 1 << r_hi.len(), "2^l words");
        assert_eq!(eq.len(), n, "one eq entry per word");
        assert_eq!(batch.len(), PACK, "one batch weight per bit");
        for plane in &planes {
            assert!(plane.len() >= Self::plane_len(n), "a plane is too short");
        }
        if r_hi.iter().any(|&r| r == B128::ONE) {
            return None;
        }
        let l = batch.to_vec();
        let lp = compose_scale(&l, b128_inv(B128::ONE + r_hi[0]));
        Some(LeanSwitch {
            pi0,
            eq,
            r_hi,
            l,
            lp,
            p: planes,
            r0: F162::ZERO,
            half: n,
            round: 0,
            pending: None,
            threads: threads.max(1),
        })
    }

    /// The buffers for a fresh switch over `n` words.
    pub fn planes(n: usize) -> [HugeBuf<u64>; 3] {
        let len = Self::plane_len(n);
        [HugeBuf::new(len), HugeBuf::new(len), HugeBuf::new(len)]
    }

    fn src(&self) -> Src {
        match self.round {
            0 => Src::Words(self.pi0.as_ptr()),
            1 => Src::WordsFolded {
                words: self.pi0.as_ptr(),
                half: self.pi0.len() / 2,
                r: self.r0,
            },
            _ => Src::Planes([self.p[0].as_ptr(), self.p[1].as_ptr(), self.p[2].as_ptr()]),
        }
    }

    /// The message over `src` at `half`: `e0 = sum_{j < half} L(eq[j]) p[j]`,
    /// `einf = sum L'(eq[j]) (p[j] + p[j + half])`, with the maps the struct holds.
    fn message(&self, src: Src, half: usize) -> [F162; 2] {
        if half < TILE {
            let mut e0 = F162::ZERO;
            let mut einf = F162::ZERO;
            for j in 0..half {
                let (a, ap) = (apply_images(&self.l, self.eq[j]), apply_images(&self.lp, self.eq[j]));
                let (p0, p1) = unsafe { (get_p(src, j), get_p(src, j + half)) };
                e0 += a * p0;
                einf += ap * (p0 + p1);
            }
            return [e0, einf];
        }
        let ml = eq_matrices(&self.l);
        let mlp = eq_matrices(&self.lp);
        let runs = crate::par::ranges(half, self.threads, TILE);
        let eq = self.eq.as_ptr() as usize;
        let parts: Vec<[F162; 2]> = std::thread::scope(|scope| {
            let handles: Vec<_> = runs
                .into_iter()
                .map(|run| {
                    let (ml, mlp) = (&ml, &mlp);
                    scope.spawn(move || unsafe {
                        message_range(eq as *const B128, src, half, ml, mlp, run)
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let mut m = [F162::ZERO; 2];
        for part in parts {
            m[0] += part[0];
            m[1] += part[1];
        }
        m
    }

    fn dst(&mut self) -> [*mut u64; 3] {
        [
            self.p[0].as_mut_ptr(),
            self.p[1].as_mut_ptr(),
            self.p[2].as_mut_ptr(),
        ]
    }

    /// The round's two message values.
    pub fn msg(&mut self) -> [F162; 2] {
        self.half /= 2;
        if let Some(m) = self.pending.take() {
            return m;
        }
        debug_assert_eq!(self.round, 0);
        self.message(self.src(), self.half)
    }

    /// Bind the round's challenge: fold `p`, advance the map, and — when a round follows —
    /// compute its message in the same pass. Round 0's fold is not written: round 1 reads the
    /// words folded by `r` on the fly.
    pub fn fold(&mut self, r: F162) {
        let half = self.half;
        let hh = half / 2;
        let last = self.round + 1 == self.r_hi.len();
        let src = self.src();
        let dst = self.dst();
        // L_{k+1} = L_k + r L'_k, then L'_{k+1} from the next coordinate.
        for i in 0..PACK {
            self.l[i] = self.l[i] + r * self.lp[i];
        }
        if !last {
            self.lp = compose_scale(&self.l, b128_inv(B128::ONE + self.r_hi[self.round + 1]));
        }
        if last {
            debug_assert_eq!(half, 1);
            let (p0, p1) = unsafe { (get_p(src, 0), get_p(src, 1)) };
            unsafe { set_p(dst, 0, p0 + r * (p0 + p1)) };
            self.round += 1;
            return;
        }
        if self.round == 0 {
            self.r0 = r;
            self.round = 1;
            let m = self.message(self.src(), hh);
            self.pending = Some(m);
            return;
        }
        if hh < TILE {
            let mut e0 = F162::ZERO;
            let mut einf = F162::ZERO;
            for j in 0..hh {
                let (p0, p1) = unsafe { (get_p(src, j), get_p(src, j + half)) };
                let q0 = p0 + r * (p0 + p1);
                let (p2, p3) = unsafe { (get_p(src, j + hh), get_p(src, j + hh + half)) };
                let q1 = p2 + r * (p2 + p3);
                unsafe {
                    set_p(dst, j, q0);
                    set_p(dst, j + hh, q1);
                }
                let (a, ap) = (apply_images(&self.l, self.eq[j]), apply_images(&self.lp, self.eq[j]));
                e0 += a * q0;
                einf += ap * (q0 + q1);
            }
            self.pending = Some([e0, einf]);
            self.round += 1;
            return;
        }
        let ml = eq_matrices(&self.l);
        let mlp = eq_matrices(&self.lp);
        let runs = crate::par::ranges(hh, self.threads, TILE);
        let eq = self.eq.as_ptr() as usize;
        let dst = dst.map(|p| p as usize);
        let parts: Vec<[F162; 2]> = std::thread::scope(|scope| {
            let handles: Vec<_> = runs
                .into_iter()
                .map(|run| {
                    let (ml, mlp) = (&ml, &mlp);
                    scope.spawn(move || unsafe {
                        fold_message_range(
                            eq as *const B128,
                            src,
                            dst.map(|p| p as *mut u64),
                            half,
                            hh,
                            r,
                            ml,
                            mlp,
                            run,
                        )
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let mut m = [F162::ZERO; 2];
        for part in parts {
            m[0] += part[0];
            m[1] += part[1];
        }
        self.pending = Some(m);
        self.round += 1;
    }

    /// `p` at the end of the sumcheck: the value the commitment opens to. Every last fold
    /// writes it to the planes' first element.
    pub fn final_eval(&self) -> F162 {
        assert_eq!(self.round, self.r_hi.len(), "the sumcheck has not finished");
        F162([
            unsafe { *self.p[0].as_ptr() },
            unsafe { *self.p[1].as_ptr() },
            unsafe { *self.p[2].as_ptr() },
        ])
    }
}

// `Src` holds raw pointers into buffers that outlive every pass and are only written through
// `dst` on disjoint ranges.
unsafe impl Send for Src {}
unsafe impl Sync for Src {}

/// [`Poly`] is referenced for the fallback path's type; keep the import live.
#[allow(dead_code)]
fn _poly_type_marker(_: &Poly) {}

#[cfg(test)]
mod lean_tests {
    use super::*;

    fn next(x: &mut u128) -> u128 {
        *x ^= *x << 13;
        *x ^= *x >> 7;
        *x ^= *x << 17;
        x.wrapping_mul(0x2545F4914F6CDD1Du128) ^ (*x >> 64)
    }

    /// The lean prover sends the messages `SwitchProver` sends and ends on the same value, at
    /// lengths that exercise the tiled, the scalar and the mixed paths, single- and
    /// multi-threaded.
    #[test]
    fn lean_switch_matches_switch_prover() {
        let mut x = 0x0123_4567_89AB_CDEF_1122_3344_5566_7788u128;
        for l in [1usize, 3, 6, 7, 8, 10] {
            for threads in [1usize, 3] {
                let n = 1 << l;
                let pi0: Vec<B128> = (0..n).map(|_| B128(next(&mut x))).collect();
                let r_hi: Vec<B128> = (0..l).map(|_| B128(next(&mut x))).collect();
                let batch: Vec<F162> = (0..PACK)
                    .map(|_| F162([next(&mut x) as u64, next(&mut x) as u64, (next(&mut x) as u64) & ((1 << 34) - 1)]))
                    .collect();
                let r: Vec<F162> = (0..l)
                    .map(|_| F162([next(&mut x) as u64, next(&mut x) as u64, (next(&mut x) as u64) & ((1 << 34) - 1)]))
                    .collect();
                let eq = eq_expand_b128(&r_hi);
                let mut eq_par = vec![B128::ZERO; n];
                unsafe { eq_expand_b128_parallel(&r_hi, eq_par.as_mut_ptr(), threads) };
                assert_eq!(eq, eq_par, "l {l}");
                assert_eq!(partial_evals(&pi0, &eq), partial_evals_parallel(&pi0, &eq, threads));

                let mut reference = SwitchProver::new(&pi0, &eq, &batch);
                let mut lean =
                    LeanSwitch::new(&pi0, &eq, &r_hi, &batch, threads, LeanSwitch::planes(n)).unwrap();
                for (round, &ri) in r.iter().enumerate() {
                    let want = reference.msg();
                    let got = lean.msg();
                    assert_eq!(want, got, "l {l} round {round}");
                    reference.fold(ri);
                    lean.fold(ri);
                }
                assert_eq!(reference.final_eval(), lean.final_eval(), "l {l}");
            }
        }
    }

    #[test]
    fn inverse_and_scaled_maps() {
        let mut x = 0x9E3779B97F4A7C15u128;
        for _ in 0..8 {
            let v = B128(next(&mut x) | 1);
            assert_eq!(v * b128_inv(v), B128::ONE);
        }
        let images: Vec<F162> = (0..PACK)
            .map(|_| F162([next(&mut x) as u64, next(&mut x) as u64, (next(&mut x) as u64) & ((1 << 34) - 1)]))
            .collect();
        let d = B128(next(&mut x));
        let scaled = compose_scale(&images, d);
        for _ in 0..8 {
            let w = B128(next(&mut x));
            assert_eq!(apply_images(&scaled, w), apply_images(&images, w * d));
        }
        let r = B128(next(&mut x));
        assert_eq!((B128::ONE + r) * b128_inv(B128::ONE + r), B128::ONE);
    }
}
