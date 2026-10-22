//! The commitment key for a modulus, and its byte image: what the build script precomputes for
//! the default modulus and the prover embeds, and what the prover falls back to building at
//! run time for any other modulus. Shared by `build.rs` and `main.rs` through `#[path]`.
use crate::modulus_choice;
use crate::parameters::{COLUMN_LEN, COLUMN_LOG, LOG_PACKED, RING_ELEMENTS};
use crate::rq;
use labinius::key::CommitmentKey;
use labinius::ring::{Batch32, Modulus, Representation};
use labinius::scheme::{Opening, Params};

/// Bytes one batch of the key takes in memory and in the image: the 41472 bytes of its 648
/// rows, the representation tag, 63 bytes of padding.
pub const BATCH_BYTES: usize = core::mem::size_of::<Batch32>();
const _: () = assert!(BATCH_BYTES == 41536 && core::mem::align_of::<Batch32>() == 64);

/// Batches per limb of the key: one per 128 elements of a column.
pub const BATCHES: usize = COLUMN_LEN / 128;

pub fn params(primes: &[u16]) -> Params {
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

/// The key for `modulus`, built from the matrix `A` in `R_Q`: the expansion and one reduction
/// pass and forward transform per limb.
pub fn build_key(modulus: u128) -> CommitmentKey {
    let primes = modulus_choice::limbs(modulus).expect("invalid commitment modulus");
    let p = params(&primes);
    let matrix = rq::ajtai_matrix(modulus, RING_ELEMENTS);
    CommitmentKey::from_coefficients(COLUMN_LEN, &matrix, modulus, p.base, &p.extra_moduli)
}

/// The byte image of a key: limb by limb, batch by batch, each batch as its rows, its tag
/// (`Representation::Ntt`) and zero padding — the in-memory layout of a `Batch32`, so the image
/// placed at a 64-byte aligned address is a `[Batch32]` per limb.
#[allow(dead_code)]
pub fn key_image(key: &CommitmentKey) -> Vec<u8> {
    let mut out = Vec::with_capacity(key.limbs() * BATCHES * BATCH_BYTES);
    for k in 0..key.limbs() {
        let rows = key.row(k);
        assert_eq!(rows.len(), BATCHES);
        for b in rows {
            assert_eq!(b.representation, Representation::Ntt);
            for row in &b.v {
                for &x in row {
                    out.extend_from_slice(&x.to_le_bytes());
                }
            }
            out.push(Representation::Ntt as u8);
            out.extend_from_slice(&[0u8; BATCH_BYTES - 41472 - 1]);
        }
    }
    assert_eq!(out.len(), key.limbs() * BATCHES * BATCH_BYTES);
    out
}

/// The key `image` describes, if it is an image of a key for `modulus` at this shape: one
/// `[Batch32]` per limb, read in place. `image` must be 64-byte aligned and outlive the key.
pub fn key_from_image(modulus: u128, image: &'static [u8]) -> Option<CommitmentKey> {
    // The build script embeds only the default-modulus matrix. Another modulus
    // with the same number of residue limbs needs its own runtime-built matrix.
    if modulus != crate::parameters::MODULUS { return None; }
    let primes = modulus_choice::limbs(modulus).ok()?;
    if image.len() != primes.len() * BATCHES * BATCH_BYTES || image.as_ptr() as usize % 64 != 0 {
        return None;
    }
    assert!(cfg!(target_endian = "little"), "the image is little-endian");
    let p = params(&primes);
    let rows: Vec<&'static [Batch32]> = (0..primes.len())
        .map(|k| unsafe {
            std::slice::from_raw_parts(
                image.as_ptr().add(k * BATCHES * BATCH_BYTES) as *const Batch32,
                BATCHES,
            )
        })
        .collect();
    Some(CommitmentKey::from_static_rows(
        COLUMN_LEN,
        p.base,
        &p.extra_moduli,
        &rows,
    ))
}

/// Choose congruent lifts of a coefficient pair separated by X^324. Reduction by
/// X^648 - X^324 + 1 uses a, b, and a+b, so minimizing their combined absolute
/// values can lower the exact-convolution bound relative to independent centering.
/// Keep the centered pair on ties; this depends only on public key coefficients.
fn lift_tail_pair(a:i16,b:i16)->(i16,i16) {
    let (a,b)=(a as i32,b as i32);
    let norm=|a:i32,b:i32| a.abs()+b.abs()+(a+b).abs();
    let mut best=(a,b);let mut bound=norm(a,b);
    for u in -1..=1 {for v in -1..=1 {
        let (x,y)=(a+1024*u,b+1024*v);
        let next=norm(x,y);
        if next<bound {best=(x,y);bound=next;}
    }}
    (best.0 as i16,best.1 as i16)
}

/// A cyclotomic-coupled coefficient lift of A mod 1024, certified for every B128 input.
pub fn build_tail_key_from_matrix(matrix:&[u128])->CommitmentKey {
    use labinius::simd::commit::hybrid;
    assert_eq!(matrix.len()%648,0);
    let raw:Vec<[i16;648]>=matrix.chunks_exact(648).map(|a|{
        let mut out=core::array::from_fn(|j|{
            let v=(a[j]&1023) as i16;if v>=512 {v-1024}else{v}
        });
        for j in 0..324 {
            (out[j],out[j+324])=lift_tail_pair(out[j],out[j+324]);
        }
        out
    }).collect();
    let _bound=hybrid::certify(&raw,hybrid::CHUNK_BATCHES);
    let product=3889u128*9721;
    let lifted:Vec<u128>=raw.iter().flatten().map(|&v|(v as i64).rem_euclid(product as i64) as u128).collect();
    let p=params(&[3889,9721]);
    CommitmentKey::from_coefficients(raw.len()*4,&lifted,product,p.base,&p.extra_moduli)
}
pub fn build_tail_key(modulus:u128)->CommitmentKey {
    assert_eq!(modulus,modulus_choice::HYBRID_MODULUS);
    let matrix=rq::ajtai_matrix(modulus,RING_ELEMENTS);
    build_tail_key_from_matrix(&matrix)
}

/// The embedded column-pair image as the flat `u32` buffer `hybrid::pair_keys` builds, if it
/// has the size of one at this shape. `image` must be 64-byte aligned.
#[allow(dead_code)]
pub fn pair_words(image: &'static [u8]) -> Option<&'static [u32]> {
    let words = 4 * (crate::parameters::RING_ELEMENTS / 2) * 648;
    if image.len() != 4 * words || image.as_ptr() as usize % 64 != 0 {
        return None;
    }
    assert!(cfg!(target_endian = "little"), "the image is little-endian");
    Some(unsafe { std::slice::from_raw_parts(image.as_ptr() as *const u32, words) })
}

#[cfg(test)]
mod tail_lift_tests {
    use super::lift_tail_pair;
    #[test]
    fn all_tail_pairs_preserve_residues_and_minimize_cyclotomic_norm() {
        let norm=|a:i32,b:i32| a.abs()+b.abs()+(a+b).abs();
        for a in -512i16..512 {for b in -512i16..512 {
            let (x,y)=lift_tail_pair(a,b);
            assert_eq!((x-a).rem_euclid(1024),0);
            assert_eq!((y-b).rem_euclid(1024),0);
            assert!(x.abs()<=682 && y.abs()<=682);
            let n=norm(x as i32,y as i32);
            assert!(n<=norm(a as i32,b as i32));
            for u in -2..=2 {for v in -2..=2 {
                assert!(n<=norm(a as i32+1024*u,b as i32+1024*v));
            }}
            if n==norm(a as i32,b as i32) {assert_eq!((x,y),(a,b));}
        }}
    }
}
