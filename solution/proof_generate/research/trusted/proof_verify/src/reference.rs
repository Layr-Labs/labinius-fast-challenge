//! A reference prover, generic in the commitment modulus.
//!
//! The scored prover is `solution/proof_generate`, which commits through the library's limb
//! kernels and so only reaches a modulus that is a product of their primes. This one computes
//! the commitment `C_j = A W_j` coefficient by coefficient in `R_Q`, which costs about `7e11`
//! `u128` additions — tens of seconds across the machine's cores — and reaches any modulus the
//! checker accepts, a power of two included. It is what `--prove` runs, and what demonstrates
//! that the checker's gate is the only thing standing between a contestant and a modulus of
//! their own choosing.
use crate::rq;
use labinius::challenge::ShortChallenge;
use labinius::eval::eq_table;
use labinius::fields::{
    crossfield as cf,
    scalar::{B128, F162},
};
use labinius::ring::Representation;
use labinius::{wire, FoldedWitness, RingElement648, RowEvaluation};

/// Workers the three parallel passes split their independent work over.
fn threads() -> usize {
    std::thread::available_parallelism().map_or(1, |n| n.get())
}

/// The trace lifted into `F162` bit for bit, which is the witness the commitment reads.
fn witness(values: &[B128]) -> Vec<F162> {
    values.iter().map(|&x| F162::from_b128(x)).collect()
}

/// Every column's commitment, flattened the way [`rq::challenge_combination`] reads it. The
/// columns are independent, so they go out over the cores.
fn commit(matrix: &[u128], witness: &[F162], modulus: u128) -> Vec<u128> {
    let per = crate::COLUMNS.div_ceil(threads());
    let mut out = vec![0u128; crate::COLUMNS * rq::N];
    std::thread::scope(|scope| {
        let blocks = out.chunks_mut(per * rq::N);
        let sources = witness.chunks(per * crate::COLUMN_LEN);
        for (block, columns) in blocks.zip(sources) {
            scope.spawn(move || {
                let targets = block.chunks_mut(rq::N);
                for (target, column) in targets.zip(columns.chunks_exact(crate::COLUMN_LEN)) {
                    target.copy_from_slice(&rq::commit_binary_column(matrix, column, modulus));
                }
            });
        }
    });
    out
}

/// `u_j = sum_i eq(p0, i) W[i + COLUMN_LEN j]`, one field element per column: the prover's
/// message. One `2^ROW_LOG`-term inner product per column.
fn row_evaluate(witness: &[F162], p0: &[F162]) -> Vec<F162> {
    let eq = eq_table(p0);
    let per = crate::COLUMNS.div_ceil(threads());
    let mut out = vec![F162::ZERO; crate::COLUMNS];
    std::thread::scope(|scope| {
        let blocks = out.chunks_mut(per);
        let sources = witness.chunks(per * crate::COLUMN_LEN);
        for (block, columns) in blocks.zip(sources) {
            let eq = &eq;
            scope.spawn(move || {
                let columns = columns.chunks_exact(crate::COLUMN_LEN);
                for (slot, column) in block.iter_mut().zip(columns) {
                    *slot = crate::dot_f162(eq, column);
                }
            });
        }
    });
    out
}

/// `v = sum_j c_j W_j` over the integers: one ring element of the fold.
///
/// A challenge sits in `R_648` as `c(-X^4)` (see [`rq::challenge_combination`]), so every term is
/// one shifted add of a column's binary ring element. A coefficient of `v` is a sum of
/// `28 * COLUMNS` signed 0/1 terms, a few hundred in magnitude, which is why the fold travels as
/// `i16` and why the integer identity the checker tests is the one the prover computed.
fn fold_element(witness: &[F162], challenges: &[ShortChallenge], m: usize) -> RingElement648 {
    let mut raw = vec![0i32; 2 * rq::N - 1];
    let mut bits = [0i32; rq::N];
    for (j, challenge) in challenges.iter().enumerate() {
        let packed = &witness[j * crate::COLUMN_LEN + 4 * m..][..4];
        for (k, x) in packed.iter().enumerate() {
            for p in 0..rq::N / 4 {
                bits[4 * p + k] = ((x.0[p >> 6] >> (p & 63)) & 1) as i32;
            }
        }
        for t in 0..challenge.weight {
            let position = challenge.positions[t] as usize;
            let negative =
                ((challenge.signs >> t) & 1 == 1) ^ (position % 2 == 1);
            let window = &mut raw[4 * position..4 * position + rq::N];
            if negative {
                for (slot, &b) in window.iter_mut().zip(&bits) {
                    *slot -= b;
                }
            } else {
                for (slot, &b) in window.iter_mut().zip(&bits) {
                    *slot += b;
                }
            }
        }
    }
    for k in (rq::N..2 * rq::N - 1).rev() {
        let value = raw[k];
        raw[k] = 0;
        raw[k - rq::SPLIT] += value;
        raw[k - rq::N] -= value;
    }
    let mut element = RingElement648::zero(Representation::Coefficients);
    for (slot, &value) in element.v.iter_mut().zip(&raw[..rq::N]) {
        *slot = i16::try_from(value).expect("a fold coefficient does not fit the wire codec");
    }
    element
}

/// The whole fold. The ring elements are independent, so they go out over the cores.
fn fold(witness: &[F162], challenges: &[ShortChallenge]) -> Vec<RingElement648> {
    let per = crate::RING_ELEMENTS.div_ceil(threads());
    let mut out: Vec<RingElement648> = (0..crate::RING_ELEMENTS)
        .map(|_| RingElement648::zero(Representation::Coefficients))
        .collect();
    std::thread::scope(|scope| {
        for (block, base) in out.chunks_mut(per).zip((0..).step_by(per)) {
            scope.spawn(move || {
                for (i, slot) in block.iter_mut().enumerate() {
                    *slot = fold_element(witness, challenges, base + i);
                }
            });
        }
    });
    out
}

/// A proof of `trace(seed)` under `modulus`. The transcript messages here are what
/// `crate::verify` expects, in the same order.
pub fn prove(seed: &[u8; 32], modulus: u128) -> crate::Proof {
    let values = crate::trace(seed);
    let witness = witness(&values);
    let matrix = rq::ajtai_matrix(modulus, crate::RING_ELEMENTS);
    let commitment = rq::pack_ring_elements(&commit(&matrix, &witness, modulus), modulus);
    let mut t = crate::transcript(seed, modulus, &commitment);
    let (lo, hi) = crate::point(&mut t);
    let mut ws = cf::Workspace::new(crate::LOG_PACKED, 1);
    let partials = ws.partial_evals(0, &values, &hi);
    let value = crate::claim(&partials, &lo);
    t.absorb_b128(&[value]);
    t.absorb_b128(&partials);
    let batch = cf::eq_expand_f162(&t.sample_f162(b"switch-batch", 7));
    let mut switch = cf::SwitchProver::from_workspace(ws, &values, &[F162::ONE], &batch);
    let mut rounds = Vec::new();
    let mut coords = Vec::new();
    for _ in 0..crate::LOG_PACKED {
        let msg = switch.msg();
        t.absorb_f162(&msg);
        let r = t.sample_f162(b"switch-round", 1)[0];
        rounds.push(msg.map(|v| v.0));
        coords.push(r);
        switch.fold(r);
    }
    let opened = switch.final_eval();
    t.absorb_f162(&[opened]);
    let (p0, _) = crate::split_point(&coords);
    let row = RowEvaluation::of(row_evaluate(&witness, &p0));
    let challenges = crate::folding_challenges(&mut t, &row);
    let folded = FoldedWitness::of(fold(&witness, &challenges));
    crate::Proof {
        version: 1,
        modulus,
        commitment,
        partials: partials.iter().map(|v| v.0).collect(),
        rounds,
        opened: opened.0,
        row: wire::pack_row_evaluation(&row),
        folded: crate::pack_folded(&folded),
    }
}
