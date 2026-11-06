use super::crossfield::eq_expand_f162;
use super::neon::{self, ByteTables};
use super::scalar::{B128, F162};
use crate::par;

const PACK: usize = 128;
const PREFIX: usize = 4;

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

/// The first equality block and the bit partials of each witness block.
/// All blocks use the first block's weights; ratios restore the prefix coordinates.
pub struct Prepared {
    eq: Vec<B128>,
    sets: Vec<Vec<B128>>,
    inverses: Vec<B128>,
    ratios: Vec<B128>,
}

impl Prepared {
    pub fn new(words: &[B128], r: &[B128]) -> Option<Self> {
        assert_eq!(words.len(), 1usize << r.len());
        if !neon::pmull() || r.len() < PREFIX + 3 || r.contains(&B128::ONE) {
            return None;
        }
        let inverses: Vec<_> = r.iter().map(|&x| inverse(B128::ONE + x)).collect();
        let ratios = r.iter().zip(&inverses).map(|(&x, &d)| x * d).collect();
        let scale = r[..PREFIX].iter().fold(B128::ONE, |a, &x| a * (B128::ONE + x));
        let eq = super::crossfield::eq_expand_b128_scaled(&r[PREFIX..], scale);
        let parts = par::workers(eq.len(), 1 << 13);
        let parts = par::map_blocks(eq.len(), parts, 64, |_, start, end| {
            neon::partial_evals_sets(words, &eq, start, end)
        });
        let mut sets = vec![vec![B128::ZERO; PACK]; 1 << PREFIX];
        for part in parts {
            for (dst, src) in sets.iter_mut().zip(part) {
                for (d, s) in dst.iter_mut().zip(src) {
                    *d = *d + s;
                }
            }
        }
        Some(Self { eq, sets, inverses, ratios })
    }

    pub fn partials(&self) -> Vec<B128> {
        let mut out = vec![B128::ZERO; PACK];
        for (set, scale) in self.sets.iter().zip(scales(&self.ratios[..PREFIX])) {
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
    prefix_r: [F162; PREFIX],
    planes: [Vec<u64>; 3],
    half: usize,
    round: usize,
}

impl<'a> Switch<'a> {
    pub fn new(words: &'a [B128], prepared: Prepared, batch: &[F162]) -> Self {
        let lp = compose(batch, prepared.inverses[0]);
        Self {
            words,
            prepared,
            l: batch.to_vec(),
            lp,
            prefix_r: [F162::ZERO; PREFIX],
            planes: core::array::from_fn(|_| Vec::new()),
            half: words.len(),
            round: 0,
        }
    }

    fn prefix_message(&self) -> [F162; 2] {
        let k = self.round;
        let suffix = PREFIX - k - 1;
        let scale = scales(&self.prepared.ratios[k + 1..PREFIX]);
        let weights = eq_expand_f162(&self.prefix_r[..k]);
        let l = ByteTables::new(&self.l);
        let lp = ByteTables::new(&self.lp);
        let mut msg = [F162::ZERO; 2];
        for (z, weight) in weights.into_iter().enumerate() {
            let g0 = z << (PREFIX - k);
            let g1 = g0 | (1 << suffix);
            let mut m = [F162::ZERO; 2];
            for bit in 0..PACK {
                let mut s0 = B128::ZERO;
                let mut s1 = B128::ZERO;
                for (q, &s) in scale.iter().enumerate() {
                    s0 = s0 + self.prepared.sets[g0 | q][bit] * s;
                    s1 = s1 + self.prepared.sets[g1 | q][bit] * s;
                }
                let basis = F162::from_b128(B128(1u128 << bit));
                m[0] += l.apply(s0) * basis;
                m[1] += lp.apply(s0 + s1) * basis;
            }
            msg[0] += weight * m[0];
            msg[1] += weight * m[1];
        }
        msg
    }

    pub fn msg(&mut self) -> [F162; 2] {
        self.half /= 2;
        if self.round < PREFIX {
            return self.prefix_message();
        }
        let l = ByteTables::new(&self.l);
        let lp = ByteTables::new(&self.lp);
        let p = &self.planes;
        let half = self.half;
        let parts = par::workers(half, 1 << 12);
        let sums = par::map_blocks(half, parts, 2, |_, start, end| unsafe {
            neon::linear_message(
                &self.prepared.eq[start..end],
                core::array::from_fn(|k| &p[k][start..end]),
                core::array::from_fn(|k| &p[k][half + start..half + end]),
                &l,
                &lp,
            )
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
        if self.round < PREFIX {
            self.prefix_r[self.round] = r;
            if self.round + 1 == PREFIX {
                self.materialize();
            }
        } else {
            let half = self.half;
            let parts = par::workers(half, 1 << 13);
            let edges: Vec<_> = (0..parts).map(|t| par::block(half, parts, t, 2)).collect();
            let [p0, p1, p2] = &mut self.planes;
            let (l0, h0) = p0.split_at_mut(half);
            let (l1, h1) = p1.split_at_mut(half);
            let (l2, h2) = p2.split_at_mut(half);
            let chunks = par::split_mut(l0, &edges).into_iter()
                .zip(par::split_mut(l1, &edges)).zip(par::split_mut(l2, &edges));
            std::thread::scope(|scope| {
                for (((a, b), c), &(start, end)) in chunks.zip(&edges) {
                    let hi = [&h0[start..end], &h1[start..end], &h2[start..end]];
                    scope.spawn(move || unsafe { neon::fold_share([a, b, c], hi, r) });
                }
            });
        }
        self.round += 1;
    }

    fn materialize(&mut self) {
        let n = self.words.len() >> PREFIX;
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
        for log in [7, 8, 10, 14] {
            let words: Vec<_> = (0..1 << log).map(|_| B128(next())).collect();
            let r: Vec<_> = (0..log).map(|_| B128(next())).collect();
            let batch = eq_expand_f162(&(0..7).map(|_| F162::from_b128(B128(next()))).collect::<Vec<_>>());
            let prepared = Prepared::new(&words, &r).unwrap();
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
    fn exceptional_coordinates_use_reference_path() {
        let words = vec![B128::ZERO; 1 << 7];
        let mut r = vec![B128(3); 7];
        for k in 0..r.len() {
            r[k] = B128::ONE;
            assert!(Prepared::new(&words, &r).is_none());
            r[k] = B128(3);
        }
    }
}
