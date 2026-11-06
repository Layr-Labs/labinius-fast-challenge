use super::crossfield::{eq_expand_b128, eq_expand_f162};
use super::neon::{self, ByteTables};
use super::scalar::{B128, F162};
use crate::par;

const PACK: usize = 128;
const PREFIX: usize = 6;

fn inverse(x: B128) -> B128 {
    assert_ne!(x, B128::ZERO);
    let (mut a, mut b, mut e) = (B128::ONE, x, u128::MAX - 1);
    while e != 0 {
        if e & 1 != 0 {
            a = a * b;
        }
        b = b * b;
        e >>= 1;
    }
    a
}

fn scales(ratios: &[B128]) -> Vec<B128> {
    (0..1usize << ratios.len())
        .map(|q| {
            ratios.iter().enumerate().fold(B128::ONE, |a, (i, &r)| {
                if q & (1 << (ratios.len() - 1 - i)) != 0 { a * r } else { a }
            })
        })
        .collect()
}

fn compose(images: &[F162], d: B128) -> Vec<F162> {
    let map = ByteTables::new(images);
    (0..PACK).map(|i| map.apply(B128(1u128 << i) * d)).collect()
}

/// A first equality block in factored form, in high/low index order.
/// Workers expand only the tile they consume; no full equality table is stored.
struct EqFactors {
    high: Vec<B128>,
    low: Vec<B128>,
}

impl EqFactors {
    fn new(r: &[B128], scale: B128) -> Self {
        let split = r.len().div_ceil(2);
        let high = eq_expand_b128(&r[..split]).into_iter().map(|x| x * scale).collect();
        let low = eq_expand_b128(&r[split..]);
        Self { high, low }
    }

    fn len(&self) -> usize { self.high.len() * self.low.len() }

    fn fill(&self, start: usize, out: &mut [B128]) {
        let mask = self.low.len() - 1;
        let shift = self.low.len().trailing_zeros();
        for (i, e) in out.iter_mut().enumerate() {
            let j = start + i;
            *e = self.high[j >> shift] * self.low[j & mask];
        }
    }
}

/// The first equality block and the bit partials of each witness block.
/// All blocks use the first block's weights; ratios restore the prefix coordinates.
pub struct Prepared {
    eq: EqFactors,
    sets: Vec<Vec<B128>>,
    inverses: Vec<B128>,
    ratios: Vec<B128>,
    prefix: usize,
}

impl Prepared {
    pub fn new(words: &[B128], r: &[B128]) -> Option<Self> {
        Self::with_prefix(words, r, PREFIX)
    }

    fn with_prefix(words: &[B128], r: &[B128], requested: usize) -> Option<Self> {
        assert_eq!(words.len(), 1usize << r.len());
        if !neon::pmull() || r.len() < 3 || r.contains(&B128::ONE) {
            return None;
        }
        let prefix = requested.min(r.len() - 3);
        let inverses: Vec<_> = r.iter().map(|&x| inverse(B128::ONE + x)).collect();
        let ratios = r.iter().zip(&inverses).map(|(&x, &d)| x * d).collect();
        let scale = r[..prefix].iter().fold(B128::ONE, |a, &x| a * (B128::ONE + x));
        let eq = EqFactors::new(&r[prefix..], scale);
        let parts = par::workers(words.len(), 1 << 16).min(eq.len() / 8);
        let parts = par::map_blocks(eq.len(), parts, 64.min(eq.len()), |_, start, end| {
            neon::partial_evals_sets_tiled(words, eq.len(), start, end, |base, tile| eq.fill(base, tile))
        });
        let mut sets = vec![vec![B128::ZERO; PACK]; 1 << prefix];
        for part in parts {
            for (dst, src) in sets.iter_mut().zip(part) {
                for (d, s) in dst.iter_mut().zip(src) {
                    *d = *d + s;
                }
            }
        }
        Some(Self { eq, sets, inverses, ratios, prefix })
    }

    pub fn partials(&self) -> Vec<B128> {
        let mut out = vec![B128::ZERO; PACK];
        for (set, scale) in self.sets.iter().zip(scales(&self.ratios[..self.prefix])) {
            for (o, &s) in out.iter_mut().zip(set) {
                *o = *o + s * scale;
            }
        }
        out
    }
}

/// The equality polynomial stays a linear map on the original equality weights.
/// Prefix messages use partial sets, so only the folded witness is materialized.
pub struct Switch<'a> {
    words: &'a [B128],
    prepared: Prepared,
    l: Vec<F162>,
    lp: Vec<F162>,
    prefix_r: Vec<F162>,
    planes: [Vec<u64>; 3],
    half: usize,
    round: usize,
    pending: Option<[F162; 2]>,
}

impl<'a> Switch<'a> {
    pub fn new(words: &'a [B128], prepared: Prepared, batch: &[F162]) -> Self {
        let lp = compose(batch, prepared.inverses[0]);
        let prefix_r = vec![F162::ZERO; prepared.prefix];
        let mut out = Self {
            words,
            prepared,
            l: batch.to_vec(),
            lp,
            prefix_r,
            planes: core::array::from_fn(|_| Vec::new()),
            half: words.len(),
            round: 0,
            pending: None,
        };
        if out.prepared.prefix == 0 { out.materialize(); }
        out
    }

    fn prefix_message(&self) -> [F162; 2] {
        let k = self.round;
        let prefix = self.prepared.prefix;
        let suffix = prefix - k - 1;
        let scale = scales(&self.prepared.ratios[k + 1..prefix]);
        let weights = eq_expand_f162(&self.prefix_r[..k]);
        let l = ByteTables::new(&self.l);
        let lp = ByteTables::new(&self.lp);
        // Independent prefix / bit tiles keep the six-round prefix work on all workers.
        let tasks = weights.len() * PACK;
        let sums = par::map_blocks(tasks, par::workers(tasks, 16), 16, |_, start, end| {
            let mut msg = [F162::ZERO; 2];
            let mut task = start;
            while task < end {
                let z = task / PACK;
                let first = task % PACK;
                let count = (PACK - first).min(end - task);
                let g0 = z << (prefix - k);
                let g1 = g0 | (1 << suffix);
                let mut a = [F162::ZERO; PACK];
                let mut d = a;
                let mut basis = [B128::ZERO; PACK];
                for j in 0..count {
                    let bit = first + j;
                    let mut s0 = self.prepared.sets[g0][bit];
                    let mut s1 = self.prepared.sets[g1][bit];
                    for (q, &s) in scale.iter().enumerate().skip(1) {
                        s0 = s0 + self.prepared.sets[g0 | q][bit] * s;
                        s1 = s1 + self.prepared.sets[g1 | q][bit] * s;
                    }
                    a[j] = l.apply(s0);
                    d[j] = lp.apply(s0 + s1);
                    basis[j] = B128(1u128 << bit);
                }
                msg[0] += weights[z] * neon::dot_lifted(&a[..count], &basis[..count]);
                msg[1] += weights[z] * neon::dot_lifted(&d[..count], &basis[..count]);
                task += count;
            }
            msg
        });
        sums.into_iter().fold([F162::ZERO; 2], |a, b| [a[0] + b[0], a[1] + b[1]])
    }

    pub fn msg(&mut self) -> [F162; 2] {
        self.half /= 2;
        if let Some(m) = self.pending.take() { return m; }
        if self.round < self.prepared.prefix {
            return self.prefix_message();
        }
        self.message(self.half)
    }

    fn message(&self, half: usize) -> [F162; 2] {
        let l = ByteTables::new(&self.l);
        let lp = ByteTables::new(&self.lp);
        let p = &self.planes;
        let parts = par::workers(half, 1 << 10);
        let sums = par::map_blocks(half, parts, 64.min(half), |_, start, end| unsafe {
            let mut out = [F162::ZERO; 2];
            let mut eq = [B128::ZERO; 64];
            for base in (start..end).step_by(64) {
                let count = (end - base).min(64);
                self.prepared.eq.fill(base, &mut eq[..count]);
                let m = neon::linear_message(
                    &eq[..count],
                    core::array::from_fn(|k| &p[k][base..base + count]),
                    core::array::from_fn(|k| &p[k][half + base..half + base + count]),
                    &l, &lp,
                );
                out[0] += m[0];
                out[1] += m[1];
            }
            out
        });
        sums.into_iter().fold([F162::ZERO; 2], |a, b| [a[0] + b[0], a[1] + b[1]])
    }

    pub fn fold(&mut self, r: F162) {
        for (l, &lp) in self.l.iter_mut().zip(&self.lp) {
            *l += r * lp;
        }
        if self.round + 1 < self.prepared.inverses.len() {
            self.lp = compose(&self.l, self.prepared.inverses[self.round + 1]);
        }
        if self.round < self.prepared.prefix {
            self.prefix_r[self.round] = r;
            if self.round + 1 == self.prepared.prefix {
                self.materialize();
            }
        } else if self.half == 1 {
            // The final fold has no following message.
            let [p0, p1, p2] = &mut self.planes;
            let (l0, h0) = p0.split_at_mut(1);
            let (l1, h1) = p1.split_at_mut(1);
            let (l2, h2) = p2.split_at_mut(1);
            unsafe { neon::fold_share([l0, l1, l2], [&h0[..1], &h1[..1], &h2[..1]], r); }
        } else {
            self.pending = Some(self.fold_message(r));
        }
        self.round += 1;
    }

    fn fold_message(&mut self, r: F162) -> [F162; 2] {
        let hh = self.half / 2;
        let l = ByteTables::new(&self.l);
        let lp = ByteTables::new(&self.lp);
        let eq = &self.prepared.eq;
        let parts = par::workers(hh, 1 << 10);
        let edges: Vec<_> = (0..parts).map(|t| par::block(hh, parts, t, 64.min(hh))).collect();
        let [p0, p1, p2] = &mut self.planes;
        fn split(p: &mut [u64], hh: usize) -> (&mut [u64], &mut [u64], &[u64], &[u64]) {
            let (lo, hi) = p[..4 * hh].split_at_mut(2 * hh);
            let (q0, q1) = lo.split_at_mut(hh);
            let (h0, h1) = hi.split_at(hh);
            (q0, q1, h0, h1)
        }
        let (a0, a1, ah0, ah1) = split(p0, hh);
        let (b0, b1, bh0, bh1) = split(p1, hh);
        let (c0, c1, ch0, ch1) = split(p2, hh);
        let work = |q0: [&mut [u64]; 3], q1: [&mut [u64]; 3], start: usize, end: usize| unsafe {
            let [a0, b0, c0] = q0;
            let [a1, b1, c1] = q1;
            let mut out = [F162::ZERO; 2];
            let mut tile = [B128::ZERO; 64];
            for base in (start..end).step_by(64) {
                let count = (end - base).min(64);
                eq.fill(base, &mut tile[..count]);
                let offset = base - start;
                let m = neon::fold_linear_message(
                    &tile[..count],
                    // Reborrow each disjoint output quarter for this tile.
                    [ &mut a0[offset..offset + count], &mut b0[offset..offset + count], &mut c0[offset..offset + count] ],
                    [ &mut a1[offset..offset + count], &mut b1[offset..offset + count], &mut c1[offset..offset + count] ],
                    [&ah0[base..base + count], &bh0[base..base + count], &ch0[base..base + count]],
                    [&ah1[base..base + count], &bh1[base..base + count], &ch1[base..base + count]],
                    r, &l, &lp,
                );
                out[0] += m[0]; out[1] += m[1];
            }
            out
        };
        if parts == 1 {
            return work([a0, b0, c0], [a1, b1, c1], 0, hh);
        }
        let chunks = par::split_mut(a0, &edges).into_iter().zip(par::split_mut(a1, &edges))
            .zip(par::split_mut(b0, &edges)).zip(par::split_mut(b1, &edges))
            .zip(par::split_mut(c0, &edges)).zip(par::split_mut(c1, &edges));
        let sums = std::thread::scope(|scope| {
            let handles: Vec<_> = chunks.zip(&edges).map(|((((((a0, a1), b0), b1), c0), c1), &(start, end))| {
                let work = &work;
                scope.spawn(move || work([a0, b0, c0], [a1, b1, c1], start, end))
            }).collect();
            handles.into_iter().map(|h| h.join().expect("switch worker panicked")).collect::<Vec<_>>()
        });
        sums.into_iter().fold([F162::ZERO; 2], |a, b| [a[0] + b[0], a[1] + b[1]])
    }

    fn materialize(&mut self) {
        let n = self.words.len() >> self.prepared.prefix;
        self.planes = core::array::from_fn(|_| vec![0; n]);
        let weights = eq_expand_f162(&self.prefix_r);
        let words = self.words;
        let parts = par::workers(n, 1 << 12);
        let edges: Vec<_> = (0..parts).map(|t| par::block(n, parts, t, 2)).collect();
        let [p0, p1, p2] = &mut self.planes;
        let chunks = par::split_mut(p0, &edges).into_iter()
            .zip(par::split_mut(p1, &edges)).zip(par::split_mut(p2, &edges));
        let weights = &weights;
        std::thread::scope(|scope| {
            for (((a, b), c), &(start, _)) in chunks.zip(&edges) {
                scope.spawn(move || unsafe { neon::materialize_prefix(words, weights, start, [a, b, c]) });
            }
        });
    }

    pub fn final_eval(&self) -> F162 {
        assert_eq!(self.round, self.prepared.inverses.len());
        F162(core::array::from_fn(|k| self.planes[k][0]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::crossfield::SwitchProver;

    #[test]
    fn prefix_switch_matches_materialized_transcript() {
        let mut state = 0x9e3779b97f4a7c15u128;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for (log, prefix) in [(3, 0), (7, 1), (7, 4), (8, 5), (10, 4), (10, 6), (14, 6), (18, 4), (19, 6)] {
            let words: Vec<_> = (0..1 << log).map(|_| B128(next())).collect();
            let r: Vec<_> = (0..log).map(|_| B128(next())).collect();
            let batch = eq_expand_f162(&(0..7).map(|_| F162::from_b128(B128(next()))).collect::<Vec<_>>());
            let prepared = Prepared::with_prefix(&words, &r, prefix).unwrap();
            let (v, eq) = SwitchProver::partial_evals_and_eq(&words, &r);
            assert_eq!(prepared.partials(), v, "partials log={log}");
            let mut lean = Switch::new(&words, prepared, &batch);
            let mut full = SwitchProver::new(&words, &eq, &batch);
            for round in 0..log {
                assert_eq!(lean.msg(), full.msg(), "log={log} round={round}");
                let x = next();
                let challenge = F162([x as u64, (x >> 64) as u64, next() as u64 & ((1 << 34) - 1)]);
                lean.fold(challenge);
                full.fold(challenge);
            }
            assert_eq!(lean.final_eval(), full.final_eval());
        }
    }

    #[test]
    fn zero_coordinates_match_the_materialized_switch() {
        let log = 10;
        let words: Vec<_> = (0..1 << log).map(|j| B128((j as u128).wrapping_mul(0x3812_8901_0102_0304_0506_879A_FEDC_7641))).collect();
        let batch = eq_expand_f162(&vec![F162([123, 45, 6]); 7]);
        for r in [vec![B128::ZERO; log], (0..log).map(|k| if k % 2 == 0 { B128::ZERO } else { B128(789) }).collect()] {
            let prepared = Prepared::new(&words, &r).unwrap();
            let (v, eq) = SwitchProver::partial_evals_and_eq(&words, &r);
            assert_eq!(prepared.partials(), v);
            let mut lean = Switch::new(&words, prepared, &batch);
            let mut full = SwitchProver::new(&words, &eq, &batch);
            for k in 0..log {
                assert_eq!(lean.msg(), full.msg(), "round={k}");
                let challenge = F162([k as u64 + 789, 123456789, 1 << 33]);
                lean.fold(challenge); full.fold(challenge);
            }
            assert_eq!(lean.final_eval(), full.final_eval());
        }
    }

    #[test]
    fn exceptional_coordinates_use_reference_path() {
        use super::super::crossfield::{eval_pi1, SwitchVerifier};
        let log = 10;
        let words: Vec<_> = (0..1 << log).map(|j| B128((j as u128).wrapping_mul(0xFEDC_1234_5678_9ABC_0A1B_2C3D_4E5F_6789))).collect();
        let mut r = vec![B128(3); log];
        let lo = vec![B128(456); 7];
        let batch = eq_expand_f162(&vec![F162([123, 45, 6]); 7]);
        let challenges: Vec<_> = (0..log).map(|k| F162([k as u64 + 99, 456, 1 << 31])).collect();
        for k in 0..r.len() {
            r[k] = B128::ONE;
            assert!(Prepared::new(&words, &r).is_none());
            let (partials, eq) = SwitchProver::partial_evals_and_eq(&words, &r);
            let claim = partials.iter().zip(eq_expand_b128(&lo)).fold(B128::ZERO, |a, (&v, e)| a + v * e);
            let mut verifier = SwitchVerifier::start(&partials, claim, &lo, &batch).unwrap();
            let mut full = SwitchProver::new(&words, &eq, &batch);
            for &challenge in &challenges {
                verifier.round(full.msg(), challenge);
                full.fold(challenge);
            }
            assert_eq!(full.final_eval(), eval_pi1(&words, &challenges));
            verifier.finish(&r, &challenges, &batch, full.final_eval()).unwrap();
            r[k] = B128(3);
        }
    }
}
