//! The commitment modulo `Q = 3889 * 9721 * 1024` with two witness transforms instead of three.
//!
//! The residues of `t = sum_m A_m s_m` modulo 3889 and 9721 come from the transforms exactly as
//! in [`super::commit`]. The residue modulo `T = 1024` is the sum over chunks of [`CHUNK`]
//! elements of the exact integer `S_c = sum_{m in c} a_m s_m`, where `a_m = A_m mod T` taken in
//! `[-T/2, T/2)`: after the ring reduction (`X^648 = X^324 - 1`, so `X^972 = -1`) every
//! coefficient of one product `a_m s_m` is a signed sum of at most [`TERMS`] `= 836` coefficients
//! of `a_m` (the witness has its 512 low coefficients only), each of magnitude at most `T/2`,
//! so `|S_c| <= CHUNK * 836 * 512 < 3889 * 9721 / 2`, and `S_c` is recovered
//! exactly from its two residues, which the same two witness transforms give against a second,
//! "tail", matrix row. No probabilistic step: the recombination is exact for every witness.
//!
//! Per element and prime: one transform of the binary witness (as before) and two slot-wise
//! products (main row and tail row). Per chunk and prime: one vector inverse transform of the
//! tail sums; the 3889 chunk residues wait in a buffer until the 9721 pass recombines them, in
//! `i16` arithmetic modulo `2^16` (only `S_c mod 1024` is needed).
use super::*;
use crate::ntt::{centered, for_blocks, primes, ROW};

/// The two transform primes, by index into [`PRIMES`].
const PA: usize = 0;
const PB: usize = 1;
const P0: u32 = PRIMES[PA];
const P1: u32 = PRIMES[PB];
/// The power-of-two tail of the modulus.
pub const TAIL: u32 = 1024;
/// The commitment modulus.
pub const MODULUS: u128 = P0 as u128 * P1 as u128 * TAIL as u128;
const _: () = assert!(MODULUS == 38_712_288_256);
const _: () = assert!(MODULUS >= 34_816_965_049, "the module-BKZ floor");
/// Elements per exactly recombined tail chunk.
pub const CHUNK: usize = 44;
/// The most coefficients of `a` (counted with multiplicity) that one coefficient of the reduced
/// ring product `a s` collects, over binary `s` supported on coefficients `0..512`; checked by
/// the test `terms_bound_is_exact`. More than `N`: a raw degree in `648..972` folds onto two
/// coefficients.
pub const TERMS: usize = 836;
const _: () = assert!(
    (CHUNK as u64) * (TERMS as u64) * (TAIL as u64 / 2) < (P0 as u64 * P1 as u64) / 2,
    "a chunk's tail sums are below half of p0 p1"
);

/// Rows per element: `A mod p0`, `a mod p0`, `A mod p1`, `a mod p1`, all transformed.
pub const ROWS: usize = 4;

/// The hybrid matrix in the transform domain: element `m`'s row `r` is
/// `data[(ROWS m + r) ROW..][..N]`, centered residues.
pub struct HybridMatrix {
    pub elements: usize,
    pub data: std::borrow::Cow<'static, [i16]>,
}

impl HybridMatrix {
    /// The matrix from a build-time image of [`matrix`]`(elements)` (little-endian `i16`), if the
    /// image is that size and 2-byte aligned.
    pub fn from_image(image: &'static [u8], elements: usize) -> Option<HybridMatrix> {
        let len = ROWS * elements * ROW;
        if image.len() != 2 * len || image.as_ptr() as usize % 2 != 0 || cfg!(target_endian = "big") {
            return None;
        }
        // SAFETY: `len` i16 at a 2-byte-aligned address inside the static image; every bit
        // pattern is a valid i16.
        let data = unsafe { core::slice::from_raw_parts(image.as_ptr() as *const i16, len) };
        Some(HybridMatrix { elements, data: std::borrow::Cow::Borrowed(data) })
    }

    #[inline]
    pub fn row(&self, m: usize, r: usize) -> &[i16] {
        &self.data[(ROWS * m + r) * ROW..][..ROW]
    }
}

/// `A mod T` in `[-T/2, T/2)`.
#[inline]
pub fn tail_centered(v: u128) -> i64 {
    let r = (v % TAIL as u128) as i64;
    if r >= TAIL as i64 / 2 {
        r - TAIL as i64
    } else {
        r
    }
}

/// The transformed hybrid matrix over `elements` elements, eight elements per vector.
pub fn matrix(elements: usize, workers: usize) -> HybridMatrix {
    let plans = primes();
    let mut data = vec![0i16; ROWS * elements * ROW];
    for_blocks(&mut data, ROWS * ROW, 8, workers, |first, block| {
        let mut coefficients = vec![[0u128; N]; 8];
        let mut lanes = vec![[[0i16; 8]; N]; ROWS];
        for (g, chunk) in block.chunks_mut(8 * ROWS * ROW).enumerate() {
            let base = first + 8 * g;
            let here = chunk.len() / (ROWS * ROW);
            for (e, c) in coefficients.iter_mut().enumerate().take(here) {
                crate::rq::ajtai_element(MODULUS, elements, base + e, c);
            }
            for r in 0..ROWS {
                let i = if r < 2 { PA } else { PB };
                let p = plans[i].p;
                for (e, c) in coefficients.iter().enumerate().take(here) {
                    for (slot, &v) in lanes[r].iter_mut().zip(c) {
                        slot[e] = if r % 2 == 0 {
                            centered((v % p as u128) as i64, p)
                        } else {
                            centered(tail_centered(v), p)
                        };
                    }
                }
                super::transform_general(i, &mut lanes[r]);
                for (e, row) in chunk.chunks_exact_mut(ROWS * ROW).enumerate().take(here) {
                    for (o, l) in row[r * ROW..][..N].iter_mut().zip(&lanes[r]) {
                        *o = centered(l[e] as i64, p);
                    }
                }
            }
        }
    });
    HybridMatrix { elements, data: std::borrow::Cow::Owned(data) }
}

/// Level 6 and the two slot-wise products: `acc -= main * y`, `tail -= tail_row * y`.
#[inline(always)]
unsafe fn pass_d2<const S: usize>(
    x: &mut [int16x8_t; N],
    kern: &Kernel,
    main_row: &[i16],
    tail_row: &[i16],
    acc: &mut [int32x4_t],
    tail: &mut [int32x4_t],
) {
    let s: Schedule = schedule(S);
    let k = vld1q_s16(kern.k.as_ptr());
    debug_assert!(main_row.len() == ROW && tail_row.len() == ROW);
    debug_assert!(acc.len() == 2 * N && tail.len() == 2 * N);
    #[inline(always)]
    unsafe fn compute(x: &[int16x8_t; N], s: Schedule, kern: &Kernel, k: int16x8_t, q: usize) -> [int16x8_t; 6] {
        let t = vld1q_s16(kern.d[q].as_ptr());
        let mut v: [int16x8_t; 6] = core::array::from_fn(|i| load(x, 6 * q + i));
        reduce_inputs(&mut v, s.before[6], s.cheap, &[0, 3], k);
        let (y0, y1, y2) = r3!(s, 0, 1, 2, 3, v[0], v[1], v[2], t, k);
        let (y3, y4, y5) = r3!(s, 4, 5, 6, 7, v[3], v[4], v[5], t, k);
        [y0, y1, y2, y3, y4, y5]
    }
    // three groups per step, computed before their products are summed (arm-big): 108 = 36 * 3
    for q in (0..N / 6).step_by(3) {
        let ys = [compute(x, s, kern, k, q), compute(x, s, kern, k, q + 1), compute(x, s, kern, k, q + 2)];
        for (d, y) in ys.iter().enumerate() {
            let base = 6 * (q + d);
            mac6(acc, base, y, vld1q_s16(main_row.as_ptr().add(base)));
            mac6(tail, base, y, vld1q_s16(tail_row.as_ptr().add(base)));
        }
    }
}

/// The reduced `i32` slot sums (`|v| <= p`) as one `i16` vector per slot, lane = column.
#[inline(always)]
unsafe fn narrow(acc: &[int32x4_t], out: &mut [int16x8_t; N]) {
    for (s, o) in out.iter_mut().enumerate() {
        *o = vcombine_s16(vmovn_s32(acc[2 * s]), vmovn_s32(acc[2 * s + 1]));
    }
}

/// The inverse transform of the tail chunk sums, to canonical centered residues.
#[inline(always)]
unsafe fn inverse_tail<const P: usize>(x: &mut [int16x8_t; N]) {
    super::inverse::<P>(x);
}

/// `x` in `[-(p-1)/2, (p-1)/2]` for `|x| <= p + p/2` (one conditional correction each way).
#[inline(always)]
unsafe fn canonical(x: int16x8_t, p: i16) -> int16x8_t {
    let h = vdupq_n_s16((p - 1) / 2);
    let pv = vdupq_n_s16(p);
    let over = vcgtq_s16(x, h);
    let under = vcltq_s16(x, vnegq_s16(h));
    let x = vsubq_s16(x, vandq_s16(vreinterpretq_s16_u16(over), pv));
    vaddq_s16(x, vandq_s16(vreinterpretq_s16_u16(under), pv))
}

/// Twiddle pair of `p0^-1 mod p1`, for [`mulc`].
fn inv_p0_pair() -> [i16; 8] {
    let inv = crate::ntt::pow_mod(P0 as u64 % P1 as u64, P1 as u64 - 2, P1 as u64);
    let t = super::twiddle(P1, inv);
    [t[0], t[1], 0, 0, 0, 0, 0, 0]
}

/// The exact chunk sums `S = r0 + p0 x1`, modulo `2^16`, added into `tail16`:
/// `x1 = (r1 - r0) / p0 mod p1`, canonical, so `|S| <= (p0 p1 - 1) / 2`.
#[inline(always)]
unsafe fn recombine(r0: &[int16x8_t; N], r1: &[int16x8_t; N], tail16: &mut [int16x8_t; N], inv: int16x8_t, k1: int16x8_t) {
    let p0 = vdupq_n_s16(P0 as i16);
    for s in 0..N {
        let d = vsubq_s16(r1[s], r0[s]);
        let x1 = canonical(reduce(mulc::<0, 1>(d, inv, k1), k1), P1 as i16);
        let v = vmlaq_s16(r0[s], x1, p0);
        tail16[s] = vaddq_s16(tail16[s], v);
    }
}

/// One prime's pass over a batch: the main slot sums in `acc` (reduced, `|v| <= p`), and each
/// chunk's tail sums inverted to canonical coefficients, handed to `chunk(c, coefficients)`.
#[inline(never)]
unsafe fn sums<const P: usize>(
    words: &[(*const u8, usize); LANES],
    a: &HybridMatrix,
    m: &mut [int16x8_t; BITS],
    x: &mut [int16x8_t; N],
    acc: &mut [int32x4_t],
    tail: &mut [int32x4_t],
    y: &mut [int16x8_t; N],
    mut chunk: impl FnMut(usize, &mut [int16x8_t; N]),
) {
    const { assert!(P < 2) };
    let kern = &kernels()[P];
    let s: Schedule = SCHEDULES[P];
    let p = PRIMES[P] as i32;
    let (rm, rt) = (2 * P, 2 * P + 1);
    acc.fill(vdupq_n_s32(0));
    tail.fill(vdupq_n_s32(0));
    let mut since = 0usize;
    for e in 0..a.elements {
        let at: [*const u8; LANES] = core::array::from_fn(|c| words[c].0.add(e * words[c].1));
        masks(&at, m);
        pass_a(m, x, kern);
        pass_b::<P>(x, kern);
        pass_c::<P>(x, kern);
        pass_d2::<P>(x, kern, a.row(e, rm), a.row(e, rt), acc, tail);
        if (e + 1) % s.accumulate == 0 {
            reduce_sums(acc, p, kern.v32);
        }
        since += 1;
        let end = (e + 1) % CHUNK == 0 || e + 1 == a.elements;
        if end || since == s.accumulate {
            reduce_sums(tail, p, kern.v32);
            since = 0;
        }
        if end {
            narrow(tail, y);
            inverse_tail::<P>(y);
            chunk(e / CHUNK, y);
            tail.fill(vdupq_n_s32(0));
        }
    }
    reduce_sums(acc, p, kern.v32);
}

/// `(r1 - r0) p0^-1 mod p1` recombination of `r0 mod p0`, `r1 mod p1` into `[0, p0 p1)`.
#[inline]
fn crt2(r0: u32, r1: u32) -> u64 {
    const INV: u64 = crate::ntt::pow_mod(P0 as u64, P1 as u64 - 2, P1 as u64);
    let (r0, r1) = (r0 as u64, r1 as u64);
    let x1 = (r1 + P1 as u64 - r0 % P1 as u64) % P1 as u64 * INV % P1 as u64;
    r0 + P0 as u64 * x1
}

/// `r mod p0 p1` and `t mod T` into `[0, Q)`.
#[inline]
pub fn crt_tail(r01: u64, t: u32) -> u128 {
    const PP: u64 = P0 as u64 * P1 as u64;
    // PP^-1 mod 1024 by Newton's iteration (PP is odd)
    const INV: u64 = {
        let mut x = PP;
        let mut i = 0;
        while i < 5 {
            x = x.wrapping_mul(2u64.wrapping_sub(PP.wrapping_mul(x)));
            i += 1;
        }
        x % TAIL as u64
    };
    const _: () = assert!(PP.wrapping_mul(INV) % TAIL as u64 == 1);
    let k = ((t as u64 + TAIL as u64 - r01 % TAIL as u64) % TAIL as u64) * INV % TAIL as u64;
    (r01 + PP * k) as u128
}

/// The commitment of every column of `values` (columns of `4 a.elements` words) modulo
/// [`MODULUS`].
pub fn commit(values: &[B128], a: &HybridMatrix, workers: usize) -> Vec<u128> {
    let len = 4 * a.elements;
    assert_eq!(values.len() % len, 0);
    let columns = values.len() / len;
    let chunks = a.elements.div_ceil(CHUNK);
    let mut out = vec![0u128; columns * N];
    for_blocks(&mut out, N, LANES, workers, |first, block| unsafe {
        let mut m = Box::new([vdupq_n_s16(0); BITS]);
        let mut x = Box::new([vdupq_n_s16(0); N]);
        let mut y = Box::new([vdupq_n_s16(0); N]);
        let mut acc = vec![vdupq_n_s32(0); 2 * 2 * N];
        let mut tail = vec![vdupq_n_s32(0); 2 * N];
        let mut held: Vec<[int16x8_t; N]> = vec![[vdupq_n_s16(0); N]; chunks];
        let mut tail16 = Box::new([vdupq_n_s16(0); N]);
        let inv = vld1q_s16(inv_p0_pair().as_ptr());
        let k1 = vld1q_s16(kernels()[PB].k.as_ptr());
        let count = block.len() / N;
        for (b, target) in block.chunks_mut(LANES * N).enumerate() {
            let col0 = first + LANES * b;
            let here = (count - LANES * b).min(LANES);
            let words: [(*const u8, usize); LANES] = core::array::from_fn(|c| {
                if c < here {
                    (values[(col0 + c) * len..].as_ptr() as *const u8, 64)
                } else {
                    (ZERO_WORDS.as_ptr() as *const u8, 0)
                }
            });
            let (a0, a1) = acc.split_at_mut(2 * N);
            sums::<PA>(&words, a, &mut m, &mut x, a0, &mut tail, &mut y, |c, r| {
                held[c] = *r;
            });
            tail16.fill(vdupq_n_s16(0));
            sums::<PB>(&words, a, &mut m, &mut x, a1, &mut tail, &mut y, |c, r| {
                recombine(&held[c], r, &mut tail16, inv, k1);
            });
            // the main sums back to canonical coefficients, eight columns per vector
            let (a0, a1) = acc.split_at(2 * N);
            narrow(a0, &mut y);
            inverse_tail::<PA>(&mut y);
            narrow(a1, &mut x);
            inverse_tail::<PB>(&mut x);
            let r0: &[[i16; 8]; N] = &*(y.as_ptr() as *const [[i16; 8]; N]);
            let r1: &[[i16; 8]; N] = &*(x.as_ptr() as *const [[i16; 8]; N]);
            let t16: &[[i16; 8]; N] = &*(tail16.as_ptr() as *const [[i16; 8]; N]);
            for (c, column) in target.chunks_exact_mut(N).enumerate() {
                for (t, o) in column.iter_mut().enumerate() {
                    let u0 = (r0[t][c] as i32).rem_euclid(P0 as i32) as u32;
                    let u1 = (r1[t][c] as i32).rem_euclid(P1 as i32) as u32;
                    let tl = (t16[t][c] as u16 as u32) % TAIL;
                    *o = crt_tail(crt2(u0, u1), tl);
                }
            }
        }
    });
    out
}

#[cfg(test)]
mod bound_tests {
    use super::*;

    /// [`TERMS`] is the exact worst case: reduce every monomial `X^(j + l)`, `j < 648`, `l < 512`,
    /// and count, per output coefficient, the absolute weight it receives.
    #[test]
    fn terms_bound_is_exact() {
        let mut weight = vec![0usize; N];
        for d in 0..N + 511 {
            let mut poly = vec![0i64; 2 * N];
            poly[d] = 1;
            for k in (N..2 * N).rev() {
                let c = poly[k];
                if c != 0 {
                    poly[k] = 0;
                    poly[k - 324] += c;
                    poly[k - N] -= c;
                }
            }
            let pairs = (0..N).filter(|&j| d >= j && d - j < 512).count();
            for (i, &c) in poly[..N].iter().enumerate() {
                weight[i] += pairs * c.unsigned_abs() as usize;
            }
        }
        assert_eq!(weight.iter().copied().max().unwrap(), TERMS);
    }
}
