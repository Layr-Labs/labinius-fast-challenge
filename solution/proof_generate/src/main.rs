//! One packed-binary PCS instance, including the GHASH -> F162 switch.
use bincode::Options;
use labinius::fields::{
    crossfield as cf,
    scalar::{B128, F162},
};
use labinius::ring::Modulus;
use labinius::scheme::*;
use labinius::{wire, Transcript};
use serde::{Deserialize, Serialize};

mod modulus_choice;
mod parameters;
use parameters::{COLUMN_LOG, LOG_PACKED, LOG_VARS, PRIMES};

const LIMIT: u64 = 64 * 1024 * 1024;

#[derive(Serialize, Deserialize)]
struct Proof {
    version: u32,
    primes: Vec<u16>,
    commitment: Vec<u8>,
    partials: Vec<u128>,
    rounds: Vec<[[u64; 3]; 2]>,
    opened: [u64; 3],
    row: Vec<u8>,
    folded: Vec<u8>,
}
fn codec() -> impl Options {
    bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .with_limit(LIMIT)
        .reject_trailing_bytes()
}
fn params(primes: &[u16]) -> Params {
    modulus_choice::validate(primes).expect("invalid commitment modulus");
    let ms: Vec<_> = primes
        .iter()
        .map(|q| Modulus::ALL.into_iter().find(|m| m.prime() == *q).unwrap())
        .collect();
    Params::with_base(
        LOG_PACKED as u32,
        COLUMN_LOG,
        ms[0],
        ms[1..].to_vec(),
        Opening::Clear,
    )
    .unwrap()
}
fn seed() -> [u8; 32] {
    let s = std::env::var("LABINIUS_SEED").expect("LABINIUS_SEED is required");
    assert_eq!(s.len(), 64);
    core::array::from_fn(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).expect("invalid seed"))
}
fn trace(seed: &[u8; 32]) -> Vec<B128> {
    let mut h = blake3::Hasher::new_keyed(seed);
    h.update(b"labinius-fast/binary-table/v1");
    let mut b = vec![0; (1 << LOG_VARS) / 8];
    h.finalize_xof().fill(&mut b);
    b.chunks_exact(16)
        .map(|v| B128(u128::from_le_bytes(v.try_into().unwrap())))
        .collect()
}
fn transcript(seed: &[u8; 32], primes: &[u16], commitment: &[u8]) -> Transcript {
    let mut t = Transcript::new(b"labinius-fast/v1");
    t.absorb_bytes(seed);
    t.absorb_u64(LOG_VARS as u64);
    t.absorb_u64(COLUMN_LOG as u64);
    t.absorb_u64(primes.len() as u64);
    for q in primes {
        t.absorb_u64(*q as u64);
    }
    t.absorb_bytes(commitment);
    t
}
fn point(t: &mut Transcript) -> (Vec<B128>, Vec<B128>) {
    (
        t.sample_b128(b"r-lo", 7),
        t.sample_b128(b"r-hi", LOG_PACKED),
    )
}
fn claim(partials: &[B128], lo: &[B128]) -> B128 {
    partials
        .iter()
        .zip(cf::eq_expand_b128(lo))
        .fold(B128::ZERO, |a, (&v, e)| a + v * e)
}
#[cfg(test)]
fn checked_f162(x: [u64; 3]) -> F162 {
    assert_eq!(x[2] >> 34, 0, "noncanonical F162 element");
    F162(x)
}
fn prove(seed: &[u8; 32], primes: &[u16]) -> Proof {
    let clock = std::time::Instant::now();
    let p = params(primes);
    let pp = PublicParameters::from_seed(p.clone(), [0x5a; 32]);
    let setup_time = clock.elapsed();
    let mut prover = Prover::cold(&pp);
    let verifier = Verifier::new(&pp);
    let values = trace(seed);
    let w = Witness::lifted(&p, &values).unwrap();
    let witness_time = clock.elapsed();
    let (commitment, opening) = prover.commit(&w);
    let commitment = commitment.to_bytes();
    let commit_time = clock.elapsed();
    let mut t = transcript(seed, primes, &commitment);
    let (lo, hi) = point(&mut t);
    let mut ws = cf::Workspace::new(LOG_PACKED, 1);
    let partials = ws.partial_evals(0, &values, &hi);
    let value = claim(&partials, &lo);
    t.absorb_b128(&[value]);
    t.absorb_b128(&partials);
    let partials_time = clock.elapsed();
    let batch = cf::eq_expand_f162(&t.sample_f162(b"switch-batch", 7));
    let mut sw = cf::SwitchProver::from_workspace(ws, &values, &[F162::ONE], &batch);
    let mut rounds = Vec::new();
    let mut coords = Vec::new();
    for _ in 0..LOG_PACKED {
        let msg = sw.msg();
        t.absorb_f162(&msg);
        let r = t.sample_f162(b"switch-round", 1)[0];
        rounds.push(msg.map(|v| v.0));
        coords.push(r);
        sw.fold(r);
    }
    let opened = sw.final_eval();
    t.absorb_f162(&[opened]);
    let switch_time = clock.elapsed();
    let point = EvaluationPoint::msb_first(&p, &coords);
    let row = w.row_evaluate(&point);
    let challenges = verifier.derive_folding_challenges(&mut t, &row);
    let folded = prover.fold(opening, &challenges);
    let fold_time = clock.elapsed();
    let proof = Proof {
        version: 1,
        primes: primes.to_vec(),
        commitment,
        partials: partials.iter().map(|v| v.0).collect(),
        rounds,
        opened: opened.0,
        row: wire::pack_row_evaluation(&row),
        folded: wire::encode(&folded, p.base.prime()),
    };
    let encode_time = clock.elapsed();
    eprintln!("labinius-profile setup_us={} witness_us={} commit_us={} partials_us={} switch_us={} fold_us={} encode_us={} total_us={}",
        setup_time.as_micros(), (witness_time-setup_time).as_micros(),
        (commit_time-witness_time).as_micros(), (partials_time-commit_time).as_micros(),
        (switch_time-partials_time).as_micros(), (fold_time-switch_time).as_micros(),
        (encode_time-fold_time).as_micros(), encode_time.as_micros());
    proof
}
#[cfg(test)]
fn verify(seed: &[u8; 32], proof: &Proof) {
    assert_eq!(proof.version, 1);
    assert_eq!(proof.partials.len(), 128);
    assert_eq!(proof.rounds.len(), LOG_PACKED);
    let p = params(&proof.primes);
    let pp = PublicParameters::from_seed(p.clone(), [0x5a; 32]);
    let verifier = Verifier::new(&pp);
    // The challenge supplies a fresh table. Bind the commitment to that table,
    // rather than accepting an opening of an arbitrary prover-selected table.
    let values = trace(seed);
    let w = Witness::lifted(&p, &values).unwrap();
    let expected = Prover::new(&pp).commit(&w).0;
    assert_eq!(
        proof.commitment,
        expected.to_bytes(),
        "commitment does not match instance"
    );
    let commitment = Commitment::from_bytes(&p, &proof.commitment).unwrap();
    let mut t = transcript(seed, &proof.primes, &proof.commitment);
    let (lo, hi) = point(&mut t);
    let mut ws = cf::Workspace::new(LOG_PACKED, 1);
    let trusted_partials = ws.partial_evals(0, &values, &hi);
    let value = claim(&trusted_partials, &lo);
    let partials: Vec<_> = proof.partials.iter().copied().map(B128).collect();
    t.absorb_b128(&[value]);
    t.absorb_b128(&partials);
    let batch = cf::eq_expand_f162(&t.sample_f162(b"switch-batch", 7));
    let mut sw = cf::SwitchVerifier::start(&partials, value, &lo, &batch).unwrap();
    let mut coords = Vec::new();
    for raw in &proof.rounds {
        let msg = raw.map(checked_f162);
        t.absorb_f162(&msg);
        let r = t.sample_f162(b"switch-round", 1)[0];
        sw.round(msg, r);
        coords.push(r);
    }
    let opened = checked_f162(proof.opened);
    sw.finish(&hi, &coords, &batch, opened).unwrap();
    t.absorb_f162(&[opened]);
    let point = EvaluationPoint::msb_first(&p, &coords);
    let row = wire::unpack_row_evaluation(&proof.row, p.columns()).unwrap();
    let folded = wire::decode(&proof.folded).unwrap();
    let challenges = verifier.derive_folding_challenges(&mut t, &row);
    verifier.verify_evaluation(&point, &opened, &row).unwrap();
    let folded_commitment = verifier.fold_commitment(&commitment, &challenges);
    let folded_row_value = verifier.fold_row_evaluation(&row, &challenges);
    verifier
        .verify_opening(
            &commitment,
            &challenges,
            &point,
            OpeningMessage::Clear {
                folded_commitment: &folded_commitment,
                folded_witness: &folded,
                folded_row_value: &folded_row_value,
            },
        )
        .unwrap();
}
fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert!(
        args.len() == 3 || args.len() == 5,
        "usage: labinius-fast --prove FILE [--primes q1,q2,...]"
    );
    let seed = seed();
    match args[1].as_str() {
        "--prove" => {
            let primes = if args.len() == 5 {
                assert_eq!(args[3], "--primes", "unknown prover option");
                modulus_choice::parse(&args[4]).expect("invalid commitment modulus")
            } else {
                PRIMES.to_vec()
            };
            let proof = prove(&seed, &primes);
            std::fs::write(&args[2], codec().serialize(&proof).unwrap()).unwrap();
        }
        _ => panic!("unknown command"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_and_reject_wrong_instance() {
        let seed = [42; 32];
        let proof = prove(&seed, PRIMES);
        let bytes = codec().serialize(&proof).unwrap();
        let decoded: Proof = codec().deserialize(&bytes).unwrap();
        verify(&seed, &decoded);
        assert!(std::panic::catch_unwind(|| verify(&[43; 32], &decoded)).is_err());
    }

    #[test]
    fn solution_selected_modulus_round_trip() {
        let seed = [44; 32];
        let primes = [9721, 2917, 4861];
        assert_ne!(primes.as_slice(), PRIMES);
        let proof = prove(&seed, &primes);
        let bytes = codec().serialize(&proof).unwrap();
        let decoded: Proof = codec().deserialize(&bytes).unwrap();
        assert_eq!(decoded.primes, primes);
        verify(&seed, &decoded);
    }

    #[test]
    fn modulus_changes_transcript() {
        let mut a = transcript(&[0; 32], PRIMES, &[]);
        let mut b = transcript(&[0; 32], &[9721, 2917, 4861], &[]);
        let mut x = [0; 32];
        let mut y = [0; 32];
        a.fill(b"test", &mut x);
        b.fill(b"test", &mut y);
        assert_ne!(x, y);
    }
}
