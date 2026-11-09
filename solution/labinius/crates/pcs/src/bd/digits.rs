use crate::limb::dispatch_limb;
use crate::params::{inv_mod, ParamsQ, N, QUAD_CLASS_SLOT, QUAD_POW3_CLASS};
use crate::recursion::limbs::recombination;
use crate::ring::{
    Batch32, PowerOfThreeRingElement, PowerOfThreeRingElementWithLimbs, Representation,
    VerticallyAlignedMatrix, N162, SLOT_648,
};
use crate::simd::bd as kernel;
use crate::simd::ntt::bin_large as vl;
use crate::simd::ntt::gen_small::intt_gen_batch32;
use crate::simd::ntt::gen_large as vgl;
use crate::simd::ntt::gen_quad::intt_quad_gen_batch32;
use super::*;

struct Inv<const Q: u16>;

impl<const Q: u16> Inv<Q> {
    const LARGE: bool = vl::is_large(Q);
}

fn drain(batch: &Batch32, first: usize, cols: usize, out: &mut [i16]) {
    for p in 0..cols {
        let at = (first + p) * N;
        for i in 0..N {
            out[at + i] = batch.v[i][p];
        }
    }
}

fn columns_split<const Q: u16>(
    matrix: &VerticallyAlignedMatrix<PowerOfThreeRingElementWithLimbs>,
    limb: usize,
    out: &mut [i16],
) {
    let q = Q as i32;
    let half = (q - 1) / 2;
    let e = recombination::<Q>();
    let r = matrix.cols();
    let mut batch = Batch32::zero(Representation::Ntt);
    for first in (0..r).step_by(32) {
        let cols = (r - first).min(32);
        batch.v.iter_mut().for_each(|row| *row = [0i16; 32]);
        batch.representation = Representation::Ntt;
        for p in 0..cols {
            let c: [&PowerOfThreeRingElement; 4] =
                core::array::from_fn(|k| &matrix.get(k, first + p).limbs[limb]);
            for s in 0..N162 {
                let y: [i32; 4] = core::array::from_fn(|k| c[k].v[s] as i32);
                for t in 0..4 {
                    let g = &e[(s * 4 + t) * 4..(s * 4 + t) * 4 + 4];
                    let acc = (0..4)
                        .map(|k| g[k] as i32 * y[k])
                        .sum::<i32>()
                        .rem_euclid(q);
                    batch.v[SLOT_648[t][s] as usize][p] = if acc > half {
                        (acc - q) as i16
                    } else {
                        acc as i16
                    };
                }
            }
        }
        unsafe {
            if Inv::<Q>::LARGE {
                vgl::intt_gen_batch32::<Q>(&mut batch);
            } else {
                intt_gen_batch32::<Q>(&mut batch);
            }
        }
        drain(&batch, first, cols, out);
    }
}

fn columns_quad<const Q: u16>(
    matrix: &VerticallyAlignedMatrix<PowerOfThreeRingElementWithLimbs>,
    limb: usize,
    out: &mut [i16],
) {
    let q = Q as i32;
    let half = (q - 1) / 2;
    let pv: [i32; N162] =
        core::array::from_fn(|s| ParamsQ::<Q>::psi_pow(QUAD_POW3_CLASS[s] as u32) as i32);
    let r = matrix.cols();
    let mut batch = Batch32::zero(Representation::Ntt);
    for first in (0..r).step_by(32) {
        let cols = (r - first).min(32);
        batch.v.iter_mut().for_each(|row| *row = [0i16; 32]);
        batch.representation = Representation::Ntt;
        for p in 0..cols {
            let c: [&PowerOfThreeRingElement; 4] =
                core::array::from_fn(|k| &matrix.get(k, first + p).limbs[limb]);
            for s in 0..N162 {
                let (jp, jm) = (
                    QUAD_CLASS_SLOT[0][s] as usize,
                    QUAD_CLASS_SLOT[1][s] as usize,
                );
                for k in 0..2 {
                    let y0 = (c[k].v[s] as i32).rem_euclid(q);
                    let y2 = (pv[s] as i64 * (c[k + 2].v[s] as i32).rem_euclid(q) as i64 % q as i64)
                        as i32;
                    let plus = (y0 + y2) % q;
                    let minus = (y0 + q - y2) % q;
                    batch.v[2 * jp + k][p] = if plus > half {
                        (plus - q) as i16
                    } else {
                        plus as i16
                    };
                    batch.v[2 * jm + k][p] = if minus > half {
                        (minus - q) as i16
                    } else {
                        minus as i16
                    };
                }
            }
        }
        unsafe { intt_quad_gen_batch32::<Q>(&mut batch) };
        drain(&batch, first, cols, out);
    }
}

pub fn column_coefficients(
    matrix: &VerticallyAlignedMatrix<PowerOfThreeRingElementWithLimbs>,
    primes: &[u16],
) -> Vec<Vec<i16>> {
    let r = matrix.cols();
    primes
        .iter()
        .enumerate()
        .map(|(limb, &q)| {
            let mut out = vec![0i16; r * N];
            dispatch_limb!(
                q,
                split |Q| columns_split::<Q>(matrix, limb, &mut out),
                quad |Q| columns_quad::<Q>(matrix, limb, &mut out),
            );
            out
        })
        .collect()
}

pub fn drop_bits(
    matrix: &VerticallyAlignedMatrix<PowerOfThreeRingElementWithLimbs>,
    primes: &[u16],
    dropped_bits: u32,
) -> Dropped {
    let residues = column_coefficients(matrix, primes);
    let columns = matrix.cols();
    let count = columns * N;
    let limbs = primes.len();
    let inverses: Vec<Vec<u32>> = (0..limbs)
        .map(|k| {
            (0..k)
                .map(|m| inv_mod(primes[m] as u64, primes[k] as u64) as u32)
                .collect()
        })
        .collect();
    let mut top = vec![0u16; count];
    let mut digits: Vec<Vec<u16>> = (1..limbs).map(|_| vec![0u16; count]).collect();
    let round = 1u32 << (dropped_bits - 1);
    let mut mixed = vec![0u32; limbs];
    for i in 0..count {
        let base = (residues[0][i] as i32).rem_euclid(primes[0] as i32) as u32;
        top[i] = ((base + round) >> dropped_bits) as u16;
        mixed[0] = base;
        for k in 1..limbs {
            let q = primes[k] as u64;
            let mut x = (residues[k][i] as i32).rem_euclid(primes[k] as i32) as u64;
            for m in 0..k {
                x = (x + q - mixed[m] as u64 % q) % q * inverses[k][m] as u64 % q;
            }
            mixed[k] = x as u32;
            digits[k - 1][i] = x as u16;
        }
    }
    Dropped {
        primes: primes.to_vec(),
        columns,
        dropped_bits,
        top,
        digits,
    }
}

pub fn limb_residues(dropped: &Dropped, limb: usize, out: &mut [i16]) {
    let digits: Vec<&[u16]> = dropped.digits.iter().map(|d| d.as_slice()).collect();
    let q = dropped.primes[limb];
    unsafe {
        dispatch_limb!(q, |Q| kernel::residues::<Q>(
            &dropped.top,
            &digits,
            dropped.dropped_bits,
            &dropped.primes,
            out,
        ))
    }
}

pub struct Garner {
    primes: Vec<u64>,
    magic: Vec<u64>,
    inverses: Vec<Vec<u64>>,
    modulus: u64,
}

impl Garner {
    pub fn of(primes: &[u16]) -> Garner {
        let primes: Vec<u64> = primes.iter().map(|&q| q as u64).collect();
        Garner {
            magic: primes.iter().map(|&q| (1u64 << 43) / q).collect(),
            inverses: (0..primes.len())
                .map(|k| (0..k).map(|m| inv_mod(primes[m], primes[k])).collect())
                .collect(),
            modulus: primes.iter().product(),
            primes,
        }
    }

    #[inline(always)]
    fn reduce(&self, k: usize, x: u64) -> u64 {
        let q = self.primes[k];
        let r = x - (x * self.magic[k] >> 43) * q;
        if r >= q {
            r - q
        } else {
            r
        }
    }

    pub(super) fn normsq(&self, centred: &[[i16; N]]) -> u128 {
        let limbs = self.primes.len();
        let mut mixed = vec![0u64; limbs];
        let mut normsq = 0u128;
        for i in 0..N {
            for k in 0..limbs {
                let q = self.primes[k];
                let x = centred[k][i] as i64;
                let mut x = if x < 0 {
                    (x + q as i64) as u64
                } else {
                    x as u64
                };
                for m in 0..k {
                    x = self.reduce(k, (x + q - self.reduce(k, mixed[m])) * self.inverses[k][m]);
                }
                mixed[k] = x;
            }
            let mut value = 0u64;
            for k in (0..limbs).rev() {
                value = value * self.primes[k] + mixed[k];
            }
            let magnitude = if value > self.modulus / 2 {
                self.modulus - value
            } else {
                value
            } as u128;
            normsq += magnitude * magnitude;
        }
        normsq
    }
}

// =============================================================================================
// coefficients straight from the kernel's rows
// =============================================================================================

/// The inverse transform of one limb on a batch of centered NTT-domain rows, whichever tree the
/// limb runs: the step `residual` takes to read a commitment back as coefficients.
fn inverse_rows(q: u16, batch: &mut Batch32) {
    batch.representation = Representation::Ntt;
    unsafe {
        if q == 3889 {
            crate::simd::ntt::gen_small::intt_hybrid_batch32_3889(batch);
            return;
        }
        dispatch_limb!(
            q,
            small |Q| intt_gen_batch32::<Q>(batch),
            large |Q| vgl::intt_gen_batch32::<Q>(batch),
            quad |Q| intt_quad_gen_batch32::<Q>(batch),
        )
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn tr8_inlane_i16(x: &[core::arch::x86_64::__m512i; 8]) -> [core::arch::x86_64::__m512i; 8] {
    use core::arch::x86_64::*;
    let a0 = _mm512_unpacklo_epi16(x[0], x[1]);
    let a1 = _mm512_unpackhi_epi16(x[0], x[1]);
    let a2 = _mm512_unpacklo_epi16(x[2], x[3]);
    let a3 = _mm512_unpackhi_epi16(x[2], x[3]);
    let a4 = _mm512_unpacklo_epi16(x[4], x[5]);
    let a5 = _mm512_unpackhi_epi16(x[4], x[5]);
    let a6 = _mm512_unpacklo_epi16(x[6], x[7]);
    let a7 = _mm512_unpackhi_epi16(x[6], x[7]);
    let b0 = _mm512_unpacklo_epi32(a0, a2);
    let b1 = _mm512_unpackhi_epi32(a0, a2);
    let b2 = _mm512_unpacklo_epi32(a1, a3);
    let b3 = _mm512_unpackhi_epi32(a1, a3);
    let b4 = _mm512_unpacklo_epi32(a4, a6);
    let b5 = _mm512_unpackhi_epi32(a4, a6);
    let b6 = _mm512_unpacklo_epi32(a5, a7);
    let b7 = _mm512_unpackhi_epi32(a5, a7);
    [
        _mm512_unpacklo_epi64(b0, b4),
        _mm512_unpackhi_epi64(b0, b4),
        _mm512_unpacklo_epi64(b1, b5),
        _mm512_unpackhi_epi64(b1, b5),
        _mm512_unpacklo_epi64(b2, b6),
        _mm512_unpackhi_epi64(b2, b6),
        _mm512_unpacklo_epi64(b3, b7),
        _mm512_unpackhi_epi64(b3, b7),
    ]
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
unsafe fn coefficients_from_rows_32(q: u16, rows: &[&[u32; N]], out: &mut [i16]) {
    use core::arch::x86_64::*;
    const PERM: [i64; 8] = [0, 2, 4, 6, 1, 3, 5, 7];
    let half = _mm512_set1_epi32(((q - 1) / 2) as i32);
    let qv = _mm512_set1_epi32(q as i32);
    let perm = _mm512_loadu_si512(PERM.as_ptr() as *const __m512i);
    let mut batch = core::mem::MaybeUninit::<Batch32>::uninit();
    let bp = batch.as_mut_ptr();
    (*bp).representation = Representation::Ntt;
    let bv = (*bp).v.as_mut_ptr();
    for b in 0..(N / 8) {
        let u0 = 8 * b;
        let mut c = [_mm512_setzero_si512(); 8];
        for w in 0..8 {
            let v0 = _mm256_loadu_si256(rows[w].as_ptr().add(u0) as *const __m256i);
            let v1 = _mm256_loadu_si256(rows[8 + w].as_ptr().add(u0) as *const __m256i);
            let v2 = _mm256_loadu_si256(rows[16 + w].as_ptr().add(u0) as *const __m256i);
            let v3 = _mm256_loadu_si256(rows[24 + w].as_ptr().add(u0) as *const __m256i);
            let lo = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(v0), v1);
            let hi = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(v2), v3);
            let lo = _mm512_mask_sub_epi32(lo, _mm512_cmpgt_epi32_mask(lo, half), lo, qv);
            let hi = _mm512_mask_sub_epi32(hi, _mm512_cmpgt_epi32_mask(hi, half), hi, qv);
            c[w] = _mm512_permutexvar_epi64(perm, _mm512_packs_epi32(lo, hi));
        }
        let r = tr8_inlane_i16(&c);
        for k in 0..8 {
            _mm512_store_si512((*bv.add(u0 + k)).as_mut_ptr() as *mut __m512i, r[k]);
        }
    }
    let batch = batch.assume_init_mut();
    inverse_rows(q, batch);
    let op = out.as_mut_ptr();
    for b in 0..(N / 8) {
        let i0 = 8 * b;
        let r: [__m512i; 8] =
            core::array::from_fn(|k| _mm512_load_si512(batch.v[i0 + k].as_ptr() as *const __m512i));
        let c = tr8_inlane_i16(&r);
        for w in 0..8 {
            _mm_storeu_si128(op.add(w * N + i0) as *mut __m128i, _mm512_castsi512_si128(c[w]));
            _mm_storeu_si128(
                op.add((8 + w) * N + i0) as *mut __m128i,
                _mm512_extracti32x4_epi32::<1>(c[w]),
            );
            _mm_storeu_si128(
                op.add((16 + w) * N + i0) as *mut __m128i,
                _mm512_extracti32x4_epi32::<2>(c[w]),
            );
            _mm_storeu_si128(
                op.add((24 + w) * N + i0) as *mut __m128i,
                _mm512_extracti32x4_epi32::<3>(c[w]),
            );
        }
    }
}

/// The coefficients modulo `q` of up to 32 commitments given as the kernel's 648 rows in
/// `[0, q)` (the `[u32; N]` `simd::commit` finishes with, and `AuxData::commitment` holds):
/// centered into a batch, one inverse transform, then column `p`'s `N` coefficients at
/// `out[p * N..]` as `i16` representatives.
///
/// This is the direct form of [`column_coefficients`], which first splits the rows into the
/// four `R_162` components and then recombines them; `residual` reads `A v - sum c_j C_j` back
/// through exactly this inverse, so the two agree.
pub fn coefficients_from_rows(q: u16, rows: &[&[u32; N]], out: &mut [i16]) {
    assert!(!rows.is_empty() && rows.len() <= 32, "a batch holds 32 columns");
    assert_eq!(out.len(), rows.len() * N);
    #[cfg(target_arch = "x86_64")]
    if rows.len() == 32
        && std::arch::is_x86_feature_detected!("avx512f")
        && std::arch::is_x86_feature_detected!("avx512bw")
        && std::arch::is_x86_feature_detected!("avx512vl")
    {
        unsafe { coefficients_from_rows_32(q, rows, out) };
        return;
    }
    let half = ((q - 1) / 2) as u32;
    let mut batch = Batch32::zero(Representation::Ntt);
    for (p, column) in rows.iter().enumerate() {
        for u in 0..N {
            let x = column[u];
            debug_assert!(x < q as u32, "a row is not reduced");
            batch.v[u][p] = if x > half { (x as i32 - q as i32) as i16 } else { x as i16 };
        }
    }
    inverse_rows(q, &mut batch);
    for p in 0..rows.len() {
        for i in 0..N {
            out[p * N + i] = batch.v[i][p];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coefficients_from_rows_32_matches_scalar() {
        let mut s = 0x726f777333325f74u64;
        for q in [3889u16, 9721] {
            let owned: Vec<[u32; N]> = (0..32)
                .map(|_| {
                    core::array::from_fn(|_| {
                        s ^= s << 13;
                        s ^= s >> 7;
                        s ^= s << 17;
                        (s % q as u64) as u32
                    })
                })
                .collect();
            let refs: Vec<&[u32; N]> = owned.iter().collect();
            let mut got = vec![0i16; 32 * N];
            coefficients_from_rows(q, &refs, &mut got);
            let mut want = vec![0i16; 32 * N];
            coefficients_from_rows(q, &refs[..16], &mut want[..16 * N]);
            coefficients_from_rows(q, &refs[16..], &mut want[16 * N..]);
            assert_eq!(got, want, "q={q}");
        }
    }
}
