//! The binary shadow of the scheme: everything the commitment and the fold leave to the field
//! `F = F162 = GF(2)[x]/(x^162 + x^81 + 1)`.
//!
//! # The algebra
//!
//! The witness is the `wdim x r` matrix `W` over `F` held as the flat `&[F162]` the commitment
//! takes: entry `(i, j)` is `witness[i + wdim * j]`, so a column is one chunk of the commitment.
//! `F` is exactly `R_162 mod 2` under the crate's plain lift — the coefficients of an `R_162`
//! element reduced mod 2 are the bits of an `F162`, and a binary challenge is its own reduction —
//! so the whole fold has a shadow over `F`.
//!
//! Both sides of that shadow are multilinear extensions in `nu = log2(wdim) + log2(r)` variables,
//! split as `p0` (the row index `i`, low variables) and `p1` (the column index `j`, high ones),
//! with `eq(p, b) = prod_k (p_k if b_k = 1 else 1 + p_k)`. Writing `B = eq(p0, .)` for the row
//! vector of the left variables, the statement, the prover's message and the two checks are
//!
//! ```text
//!     t   = sum_{i,j} eq(p1, j) eq(p0, i) W[i + wdim j]    the claimed evaluation
//!     u   = B W          (r entries of F)                  the row evaluation, sent by the prover
//!     t   = u^T eq(p1)                                     the verifier's claim check
//!     v   = W c          (over R_648, the fold)            c_j the challenges
//!     B v = u^T c        (over F)                          the binary check
//! ```
//!
//! The last line is the only new identity: `B (W c) = (B W) c` mod 2, linearity of the row
//! evaluation against the challenge vector. Its left side is read off the folded witness `v` the
//! fold already produced (component `k` of packed element `m` is the `F162` at index `4m + k`,
//! and a coefficient's parity is that element's bit), its right side is the `r`-term inner
//! product of `u` against the challenges reduced mod 2
//! ([`ShortChallenge::to_f162`](crate::challenge::ShortChallenge::to_f162)).
//!
//! # How it is computed
//!
//! Every step is one dot product over `F`. The kernels are [`fields::f162`](crate::fields::f162)' word-sliced AVX-512
//! ones — limb `k` of 8 consecutive elements in one `zmm`, `mac_soa8` accumulating the 12
//! unreduced `clmul` products of a block and a single `reduce_soa8` at the end of the whole
//! product, exactly as that crate's sumcheck round does. One operand (the `eq` table, or the
//! challenges) is word-sliced once up front; the other is a raw `&[F162]` run — the witness never
//! leaves its own layout, it is transposed 8 elements at a time by three `vpermi2q`/`vpermq`
//! pairs inside the loop ([`load_soa8`]).
use crate::challenge::ShortChallenge;
use crate::fields::f162 as bf;
use crate::fields::scalar::F162;
use crate::fields::sumcheck::Poly;
use crate::params::N;
use crate::ring::RingElement;
use core::arch::x86_64::*;

/// `eq(ps, b) = prod_k (ps_k if bit k of b is 1 else 1 + ps_k)`, all `2^ps.len()` of them, by
/// doubling: variable `k` is bit `k` of the index.
pub fn eq_table(ps: &[F162]) -> Vec<F162> {
    let mut t = vec![F162::ONE];
    for &p in ps {
        let n = t.len();
        t.resize(2 * n, F162::ZERO);
        for b in 0..n {
            // x (1 + p) = x + x p: one product per entry.
            let x = t[b];
            let y = x * p;
            t[b + n] = y;
            t[b] = x + y;
        }
    }
    t
}

// =============================================================================================
// the dot product over F
// =============================================================================================

/// The `vpermi2q` / `vpermq` index vectors of [`load_soa8`]: 24 consecutive `u64` (8 `F162`) to
/// limb 0, 1, 2 of those 8 elements. Limb `k` of element `p` is `u64` number `3p + k`, which for
/// `k = 0` is `0, 3, 6, 9, 12, 15` inside the first two vectors and `2, 5` inside the third, and
/// so on.
struct Idx {
    a: [__m512i; 3],
    b: [__m512i; 3],
    m: [__mmask8; 3],
}

#[inline(always)]
unsafe fn idx() -> Idx {
    Idx {
        a: [
            _mm512_setr_epi64(0, 3, 6, 9, 12, 15, 0, 0),
            _mm512_setr_epi64(1, 4, 7, 10, 13, 0, 0, 0),
            _mm512_setr_epi64(2, 5, 8, 11, 14, 0, 0, 0),
        ],
        b: [
            _mm512_setr_epi64(0, 0, 0, 0, 0, 0, 2, 5),
            _mm512_setr_epi64(0, 0, 0, 0, 0, 0, 3, 6),
            _mm512_setr_epi64(0, 0, 0, 0, 0, 1, 4, 7),
        ],
        m: [0b1100_0000, 0b1110_0000, 0b1110_0000],
    }
}

/// 8 consecutive `F162` (192 contiguous bytes) read as the word-sliced `[limb0, limb1, limb2]`
/// the [`fields::f162`](crate::fields::f162) kernels take.
///
/// # Safety
/// `p` addresses 24 readable `u64`.
#[inline(always)]
unsafe fn load_soa8(ix: &Idx, p: *const u64) -> [__m512i; 3] {
    let v0 = _mm512_loadu_si512(p as *const __m512i);
    let v1 = _mm512_loadu_si512(p.add(8) as *const __m512i);
    let v2 = _mm512_loadu_si512(p.add(16) as *const __m512i);
    core::array::from_fn(|k| {
        _mm512_mask_permutexvar_epi64(
            _mm512_permutex2var_epi64(v0, ix.a[k], v1),
            ix.m[k],
            ix.b[k],
            v2,
        )
    })
}

/// The 8 lanes of a finished accumulator summed into one element.
#[inline(always)]
unsafe fn horizontal(acc: [__m512i; 12]) -> F162 {
    let r = bf::reduce_soa8(acc);
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

/// `sum_k a_k b_k` over `F`, `a` word-sliced and `b` a raw run of `n` `F162`, with the reduction
/// deferred to the end: 12 unreduced `clmul` products per 8 elements, one [`bf::reduce_soa8`].
///
/// # Safety
/// `b` addresses `n` `F162` and `a` holds at least `n` elements.
unsafe fn dot(a: &Poly, b: *const F162, n: usize) -> F162 {
    debug_assert!(a.n >= n);
    let ix = idx();
    let (mut acc, blocks) = ([_mm512_setzero_si512(); 12], n / 8);
    for t in 0..blocks {
        let x = [
            _mm512_loadu_si512(a.w[0].as_ptr().add(8 * t) as *const __m512i),
            _mm512_loadu_si512(a.w[1].as_ptr().add(8 * t) as *const __m512i),
            _mm512_loadu_si512(a.w[2].as_ptr().add(8 * t) as *const __m512i),
        ];
        bf::mac_soa8(&mut acc, x, load_soa8(&ix, (b as *const u64).add(24 * t)));
    }
    let mut out = if blocks > 0 {
        horizontal(acc)
    } else {
        F162::ZERO
    };
    for k in 8 * blocks..n {
        out += a.get(k) * *b.add(k);
    }
    out
}

/// [`dot`] on two plain slices.
fn dot_slices(a: &[F162], b: &[F162]) -> F162 {
    assert_eq!(a.len(), b.len(), "the two operands have different lengths");
    unsafe { dot(&Poly::from_scalars(a), b.as_ptr(), b.len()) }
}

// =============================================================================================
// the four steps
// =============================================================================================

/// `u_j = sum_i eq(p0, i) W[i + wdim j]`, `wdim = 2^p0.len()`: one `wdim`-term dot product per
/// column, each a single deferred-reduction accumulation over a contiguous run of the witness.
pub(crate) fn row_evaluate(witness: &[F162], p0: &[F162]) -> Vec<F162> {
    let wdim = 1usize << p0.len();
    assert_eq!(
        witness.len() % wdim,
        0,
        "the witness is not a whole number of columns of {wdim}"
    );
    let eq = Poly::from_scalars(&eq_table(p0));
    (0..witness.len() / wdim)
        .map(|j| unsafe { dot(&eq, witness.as_ptr().add(j * wdim), wdim) })
        .collect()
}

/// `u^T eq(p1)`, the claim a row evaluation `u` implies.
pub(crate) fn claim(u: &[F162], p1: &[F162]) -> F162 {
    assert_eq!(
        u.len(),
        1 << p1.len(),
        "the row evaluation is not 2^{} elements",
        p1.len()
    );
    dot_slices(&eq_table(p1), u)
}

/// `u^T c`, the binary side of the fold: `sum_j u_j (c_j mod 2)`.
pub(crate) fn fold_binary(u: &[F162], challenges: &[ShortChallenge]) -> F162 {
    assert_eq!(u.len(), challenges.len(), "one challenge per column");
    let c: Vec<F162> = challenges.iter().map(|c| c.to_f162()).collect();
    dot_slices(&c, u)
}

/// The `4 * v.len()` field elements of `v mod 2`, in the witness's own index order: element
/// `4m + k` is component `k` of packed ring element `m`, and its bit `p` is the parity of
/// coefficient `4p + k` of that element.
pub(crate) fn components_mod_2(v: &[RingElement]) -> Vec<F162> {
    let mut out = vec![F162::ZERO; 4 * v.len()];
    for (m, e) in v.iter().enumerate() {
        unsafe { element_mod_2(&e.v, &mut out[4 * m..4 * m + 4]) };
    }
    out
}

#[target_feature(enable = "avx512f,avx512bw,bmi2")]
unsafe fn element_mod_2(e: &[i16; N], out: &mut [F162]) {
    let ones = _mm512_set1_epi16(1);
    let p = e.as_ptr();
    let mut w = [[0u64; 3]; 4];
    for t in 0..N / 64 {
        let lo = _mm512_test_epi16_mask(
            _mm512_loadu_si512(p.add(64 * t) as *const __m512i),
            ones,
        ) as u64;
        let hi = _mm512_test_epi16_mask(
            _mm512_loadu_si512(p.add(64 * t + 32) as *const __m512i),
            ones,
        ) as u64;
        let bits = lo | (hi << 32);
        for (k, wk) in w.iter_mut().enumerate() {
            wk[t >> 2] |= _pext_u64(bits, 0x1111_1111_1111_1111u64 << k) << (16 * (t & 3));
        }
    }
    let rest = N - 64 * (N / 64);
    let bits = _mm512_test_epi16_mask(
        _mm512_maskz_loadu_epi16((1u64 << rest) as u32 - 1, p.add(64 * (N / 64))),
        ones,
    ) as u64;
    for (k, wk) in w.iter_mut().enumerate() {
        let q = N / 64 * 16;
        wk[q >> 6] |= _pext_u64(bits, 0x1111_1111_1111_1111u64 << k) << (q & 63);
    }
    for (k, wk) in w.iter().enumerate() {
        out[k] = F162(*wk);
    }
}

/// The binary check `B v == u^T c` over `F`: `sum_i eq(p0, i) (v_i mod 2) == u_folded`.
pub(crate) fn binary_check(p0: &[F162], v: &[RingElement], u_folded: F162) -> bool {
    4 * v.len() == 1usize << p0.len() && dot_slices(&eq_table(p0), &components_mod_2(v)) == u_folded
}

// =============================================================================================
// the row evaluation straight from the words
// =============================================================================================

/// The eight `B128` at `x` as `[limb0, limb1]` of their lifts.
///
/// # Safety
/// `x` addresses 8 readable `B128`.
#[inline(always)]
unsafe fn load_words2(x: *const F162Word, lo: __m512i, hi: __m512i) -> [__m512i; 2] {
    let a = _mm512_loadu_si512(x as *const __m512i);
    let b = _mm512_loadu_si512(x.add(4) as *const __m512i);
    [
        _mm512_permutex2var_epi64(a, lo, b),
        _mm512_permutex2var_epi64(a, hi, b),
    ]
}

/// Accumulate the 10 nonzero `clmul` products of 8 `F162` (`x = [x0, x1, x2]`, `sa = x0 ^ x1`)
/// against 8 lifted `B128` words (`b = [b0, b1]`). The two `x2 * b2` products are zero and the
/// Karatsuba cross-term adjustment `acc[8..12] ^= acc[0..4]` is deferred to [`horizontal_word10`].
#[inline(always)]
unsafe fn mac_word10(acc: &mut [__m512i; 10], x: [__m512i; 3], sa: __m512i, b: [__m512i; 2]) {
    let sb = _mm512_xor_si512(b[0], b[1]);
    acc[0] = _mm512_xor_si512(acc[0], _mm512_clmulepi64_epi128::<0x00>(x[0], b[0]));
    acc[1] = _mm512_xor_si512(acc[1], _mm512_clmulepi64_epi128::<0x11>(x[0], b[0]));
    acc[2] = _mm512_xor_si512(acc[2], _mm512_clmulepi64_epi128::<0x00>(x[1], b[1]));
    acc[3] = _mm512_xor_si512(acc[3], _mm512_clmulepi64_epi128::<0x11>(x[1], b[1]));
    acc[4] = _mm512_xor_si512(acc[4], _mm512_clmulepi64_epi128::<0x00>(sa, sb));
    acc[5] = _mm512_xor_si512(acc[5], _mm512_clmulepi64_epi128::<0x11>(sa, sb));
    acc[6] = _mm512_xor_si512(acc[6], _mm512_clmulepi64_epi128::<0x00>(x[2], b[0]));
    acc[7] = _mm512_xor_si512(acc[7], _mm512_clmulepi64_epi128::<0x11>(x[2], b[0]));
    acc[8] = _mm512_xor_si512(acc[8], _mm512_clmulepi64_epi128::<0x00>(x[2], b[1]));
    acc[9] = _mm512_xor_si512(acc[9], _mm512_clmulepi64_epi128::<0x11>(x[2], b[1]));
}

#[inline(always)]
unsafe fn horizontal_word10(acc: [__m512i; 10]) -> F162 {
    let z = _mm512_setzero_si512();
    horizontal([
        acc[0],
        acc[1],
        acc[2],
        acc[3],
        z,
        z,
        acc[4],
        acc[5],
        _mm512_xor_si512(acc[6], acc[0]),
        _mm512_xor_si512(acc[7], acc[1]),
        _mm512_xor_si512(acc[8], acc[2]),
        _mm512_xor_si512(acc[9], acc[3]),
    ])
}

type F162Word = crate::fields::scalar::B128;

/// [`dot`] against a run of `n` `B128` words read as their lifts.
///
/// # Safety
/// `b` addresses `n` words and `a` holds at least `n` elements.
unsafe fn dot_words(a: &Poly, b: *const F162Word, n: usize) -> F162 {
    debug_assert!(a.n >= n);
    let lo = _mm512_setr_epi64(0, 2, 4, 6, 8, 10, 12, 14);
    let hi = _mm512_setr_epi64(1, 3, 5, 7, 9, 11, 13, 15);
    let (mut acc, blocks) = ([_mm512_setzero_si512(); 10], n / 8);
    for t in 0..blocks {
        // The words' lines 1 KB ahead, requested explicitly: with eight runners streaming the
        // table the hardware prefetcher alone leaves the clmul work waiting (6.6 -> 5.9 ms for
        // the phase; 4 KB and 16 KB ahead are no better). A prefetch past the end is harmless.
        let ahead = b.wrapping_add(8 * t + 64) as *const i8;
        core::arch::x86_64::_mm_prefetch::<{ core::arch::x86_64::_MM_HINT_T0 }>(ahead);
        core::arch::x86_64::_mm_prefetch::<{ core::arch::x86_64::_MM_HINT_T0 }>(ahead.wrapping_add(64));
        let x = [
            _mm512_loadu_si512(a.w[0].as_ptr().add(8 * t) as *const __m512i),
            _mm512_loadu_si512(a.w[1].as_ptr().add(8 * t) as *const __m512i),
            _mm512_loadu_si512(a.w[2].as_ptr().add(8 * t) as *const __m512i),
        ];
        let sa = _mm512_xor_si512(x[0], x[1]);
        mac_word10(&mut acc, x, sa, load_words2(b.add(8 * t), lo, hi));
    }
    let mut out = if blocks > 0 {
        horizontal_word10(acc)
    } else {
        F162::ZERO
    };
    for k in 8 * blocks..n {
        out += a.get(k) * F162::from_b128(*b.add(k));
    }
    out
}

/// [`row_evaluate`] of the witness `Witness::lifted` would build from `words`, read in place
/// and one column per task over `workers` threads: `u_j = sum_i eq(p0, i) lift(words[i + wdim j])`.
/// `eq(p0, .)` straight into the word-sliced layout [`dot_words`] reads: entry `i` is
/// `eq(p0[..3], i & 7) eq(p0[3..], i >> 3)`, eight entries per [`bf::mul_soa8`] against the
/// eight low ones, cut over the pool. The field is exact, so the entries are [`eq_table`]'s.
fn eq_poly(p0: &[F162], workers: usize) -> Poly {
    const LO: usize = 3;
    let n = 1usize << p0.len();
    if p0.len() < 2 * LO {
        return Poly::from_scalars(&eq_table(p0));
    }
    let lo = eq_table(&p0[..LO]);
    let hi = eq_table(&p0[LO..]);
    let mut w: [Vec<u64>; 3] = core::array::from_fn(|_| Vec::with_capacity(n));
    let addrs = w.each_mut().map(|v| v.as_mut_ptr() as usize);
    let runs = crate::par::ranges(hi.len(), workers, 8);
    crate::par::pool().run(runs.len(), |r| unsafe {
        let lo_v: [__m512i; 3] = core::array::from_fn(|k| {
            let limbs: [u64; 8] = core::array::from_fn(|i| lo[i].0[k]);
            _mm512_loadu_si512(limbs.as_ptr() as *const __m512i)
        });
        for ih in runs[r].clone() {
            let h = hi[ih].0;
            let v = bf::mul_soa8(lo_v, [
                _mm512_set1_epi64(h[0] as i64),
                _mm512_set1_epi64(h[1] as i64),
                _mm512_set1_epi64(h[2] as i64),
            ]);
            for (k, &a) in addrs.iter().enumerate() {
                _mm512_storeu_si512((a as *mut u64).add(8 * ih) as *mut __m512i, v[k]);
            }
        }
    });
    // Every entry was written above.
    for v in &mut w {
        unsafe { v.set_len(n) };
    }
    Poly { w, n }
}

#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct EqBlock([__m512i; 3]);

/// `eq(p0, .)` packed into contiguous 3-vector blocks `[a01_0, a01_1, a2_pair]` for [`dot_words_4cols`]:
/// in each block `ih`, `a01_0` holds `[a0, a1]` of elements `0..4`, `a01_1` holds `[a0, a1]` of
/// elements `4..8`, and `a2_pair` holds `[a2[0..4], a2[4..8]]` interleaved in even/odd qwords.
fn eq_packed(p0: &[F162], workers: usize) -> Vec<EqBlock> {
    const LO: usize = 3;
    let n = 1usize << p0.len();
    debug_assert!(p0.len() >= 2 * LO);
    let blocks = n / 8;
    let lo = eq_table(&p0[..LO]);
    let hi = eq_table(&p0[LO..]);
    let mut out: Vec<EqBlock> = Vec::with_capacity(blocks);
    let out_addr = out.as_mut_ptr() as usize;
    let runs = crate::par::ranges(hi.len(), workers, 8);
    crate::par::pool().run(runs.len(), |r| unsafe {
        const ORDER: [usize; 8] = [0, 4, 1, 5, 2, 6, 3, 7];
        let lo_v: [__m512i; 3] = core::array::from_fn(|k| {
            let limbs: [u64; 8] = core::array::from_fn(|i| lo[ORDER[i]].0[k]);
            _mm512_loadu_si512(limbs.as_ptr() as *const __m512i)
        });
        let dst = out_addr as *mut EqBlock;
        for ih in runs[r].clone() {
            let h = hi[ih].0;
            let v = bf::mul_soa8(lo_v, [
                _mm512_set1_epi64(h[0] as i64),
                _mm512_set1_epi64(h[1] as i64),
                _mm512_set1_epi64(h[2] as i64),
            ]);
            let a01_0 = _mm512_unpacklo_epi64(v[0], v[1]);
            let a01_1 = _mm512_unpackhi_epi64(v[0], v[1]);
            dst.add(ih).write(EqBlock([a01_0, a01_1, v[2]]));
        }
    });
    unsafe { out.set_len(blocks) };
    out
}

#[inline(always)]
unsafe fn fold_lanes128(v: __m512i) -> (u64, u64) {
    let a = _mm256_xor_si256(_mm512_castsi512_si256(v), _mm512_extracti64x4_epi64::<1>(v));
    let b = _mm_xor_si128(_mm256_castsi256_si128(a), _mm256_extracti128_si256::<1>(a));
    (_mm_cvtsi128_si64(b) as u64, _mm_extract_epi64::<1>(b) as u64)
}

#[inline(always)]
unsafe fn reduce_acc4(acc: [__m512i; 4]) -> F162 {
    let (s0_lo, s0_hi) = fold_lanes128(acc[0]);
    let (s1_lo, s1_hi) = fold_lanes128(acc[1]);
    let (s2_lo, s2_hi) = fold_lanes128(acc[2]);
    let (s3_lo, s3_hi) = fold_lanes128(acc[3]);

    let p0 = s0_lo;
    let p1 = s0_hi ^ s1_lo;
    let p2 = s1_hi ^ s2_lo;
    let p3 = s2_hi ^ s3_lo;
    let p4 = s3_hi;

    const M17: u64 = (1u64 << 17) - 1;
    const M34: u64 = (1u64 << 34) - 1;
    let h00 = (p2 >> 34) | (p3 << 30);
    let h01 = (p3 >> 34) & M17;
    let h10 = (p3 >> 51) | (p4 << 13);
    let h11 = p4 >> 51;

    let r0 = p0 ^ h00 ^ h10;
    let r1 = p1 ^ h01 ^ h11 ^ (h00 << 17);
    let r2 = (p2 & M34) ^ (h01 << 17) ^ (h00 >> 47);
    F162([r0, r1, r2])
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,vpclmulqdq")]
unsafe fn dot_words_4cols(eq: &[EqBlock], cols: [*const F162Word; 4]) -> [F162; 4] {
    let zero = _mm512_setzero_si512();
    let mut acc = [[zero; 4]; 4];
    for (t, e) in eq.iter().enumerate() {
        let a01_0 = e.0[0];
        let a01_1 = e.0[1];
        let a2 = e.0[2];
        for c in 0..4 {
            let b = cols[c].add(8 * t);
            let ahead = b.wrapping_add(64) as *const i8;
            core::arch::x86_64::_mm_prefetch::<{ core::arch::x86_64::_MM_HINT_T0 }>(ahead);
            core::arch::x86_64::_mm_prefetch::<{ core::arch::x86_64::_MM_HINT_T0 }>(ahead.wrapping_add(64));
            let z0 = _mm512_loadu_si512(b as *const __m512i);
            let z1 = _mm512_loadu_si512(b.add(4) as *const __m512i);
            acc[c][0] = _mm512_ternarylogic_epi64::<0x96>(
                acc[c][0],
                _mm512_clmulepi64_epi128::<0x00>(a01_0, z0),
                _mm512_clmulepi64_epi128::<0x00>(a01_1, z1),
            );
            acc[c][1] = _mm512_ternarylogic_epi64::<0x96>(
                acc[c][1],
                _mm512_clmulepi64_epi128::<0x01>(a01_0, z0),
                _mm512_clmulepi64_epi128::<0x10>(a01_0, z0),
            );
            acc[c][1] = _mm512_ternarylogic_epi64::<0x96>(
                acc[c][1],
                _mm512_clmulepi64_epi128::<0x01>(a01_1, z1),
                _mm512_clmulepi64_epi128::<0x10>(a01_1, z1),
            );
            acc[c][2] = _mm512_ternarylogic_epi64::<0x96>(
                acc[c][2],
                _mm512_clmulepi64_epi128::<0x11>(a01_0, z0),
                _mm512_clmulepi64_epi128::<0x00>(a2, z0),
            );
            acc[c][2] = _mm512_ternarylogic_epi64::<0x96>(
                acc[c][2],
                _mm512_clmulepi64_epi128::<0x11>(a01_1, z1),
                _mm512_clmulepi64_epi128::<0x01>(a2, z1),
            );
            acc[c][3] = _mm512_ternarylogic_epi64::<0x96>(
                acc[c][3],
                _mm512_clmulepi64_epi128::<0x10>(a2, z0),
                _mm512_clmulepi64_epi128::<0x11>(a2, z1),
            );
        }
    }
    core::array::from_fn(|c| reduce_acc4(acc[c]))
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,vpclmulqdq")]
unsafe fn dot_words_1col(eq: &[EqBlock], col: *const F162Word) -> F162 {
    let zero = _mm512_setzero_si512();
    let mut acc = [zero; 4];
    for (t, e) in eq.iter().enumerate() {
        let a01_0 = e.0[0];
        let a01_1 = e.0[1];
        let a2 = e.0[2];
        let b = col.add(8 * t);
        let ahead = b.wrapping_add(64) as *const i8;
        core::arch::x86_64::_mm_prefetch::<{ core::arch::x86_64::_MM_HINT_T0 }>(ahead);
        core::arch::x86_64::_mm_prefetch::<{ core::arch::x86_64::_MM_HINT_T0 }>(ahead.wrapping_add(64));
        let z0 = _mm512_loadu_si512(b as *const __m512i);
        let z1 = _mm512_loadu_si512(b.add(4) as *const __m512i);
        acc[0] = _mm512_ternarylogic_epi64::<0x96>(
            acc[0],
            _mm512_clmulepi64_epi128::<0x00>(a01_0, z0),
            _mm512_clmulepi64_epi128::<0x00>(a01_1, z1),
        );
        acc[1] = _mm512_ternarylogic_epi64::<0x96>(
            acc[1],
            _mm512_clmulepi64_epi128::<0x01>(a01_0, z0),
            _mm512_clmulepi64_epi128::<0x10>(a01_0, z0),
        );
        acc[1] = _mm512_ternarylogic_epi64::<0x96>(
            acc[1],
            _mm512_clmulepi64_epi128::<0x01>(a01_1, z1),
            _mm512_clmulepi64_epi128::<0x10>(a01_1, z1),
        );
        acc[2] = _mm512_ternarylogic_epi64::<0x96>(
            acc[2],
            _mm512_clmulepi64_epi128::<0x11>(a01_0, z0),
            _mm512_clmulepi64_epi128::<0x00>(a2, z0),
        );
        acc[2] = _mm512_ternarylogic_epi64::<0x96>(
            acc[2],
            _mm512_clmulepi64_epi128::<0x11>(a01_1, z1),
            _mm512_clmulepi64_epi128::<0x01>(a2, z1),
        );
        acc[3] = _mm512_ternarylogic_epi64::<0x96>(
            acc[3],
            _mm512_clmulepi64_epi128::<0x10>(a2, z0),
            _mm512_clmulepi64_epi128::<0x11>(a2, z1),
        );
    }
    reduce_acc4(acc)
}

pub fn row_evaluate_words(words: &[F162Word], p0: &[F162], workers: usize) -> Vec<F162> {
    let wdim = 1usize << p0.len();
    assert_eq!(
        words.len() % wdim,
        0,
        "the words are not a whole number of columns of {wdim}"
    );
    let columns = words.len() / wdim;
    if p0.len() < 6 {
        let eq = eq_poly(p0, workers);
        let mut out = vec![F162::ZERO; columns];
        for j in 0..columns {
            out[j] = unsafe { dot_words(&eq, words.as_ptr().add(j * wdim), wdim) };
        }
        return out;
    }
    let mut out = vec![F162::ZERO; columns];
    if columns == 0 {
        return out;
    }
    let eq = eq_packed(p0, workers);
    // Columns handed out in multiples of 4 from a shared counter: each 4-column group shares one
    // pass over `eq` in registers.
    let chunk = (columns.div_ceil(32 * workers.max(1)).max(4) + 3) & !3;
    let chunks = columns.div_ceil(chunk);
    let next = std::sync::atomic::AtomicUsize::new(0);
    let out_addr = out.as_mut_ptr() as usize;
    crate::par::pool().run(workers.clamp(1, chunks), |_| loop {
        let k = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if k >= chunks {
            break;
        }
        let start = k * chunk;
        let end = ((k + 1) * chunk).min(columns);
        let mut j = start;
        while j + 4 <= end {
            let cols = core::array::from_fn(|c| unsafe { words.as_ptr().add((j + c) * wdim) });
            let values = unsafe { dot_words_4cols(&eq, cols) };
            for (c, value) in values.into_iter().enumerate() {
                unsafe { (out_addr as *mut F162).add(j + c).write(value) };
            }
            j += 4;
        }
        while j < end {
            let value = unsafe { dot_words_1col(&eq, words.as_ptr().add(j * wdim)) };
            unsafe { (out_addr as *mut F162).add(j).write(value) };
            j += 1;
        }
    });
    out
}

#[cfg(test)]
mod word_tests {
    use super::*;

    #[test]
    fn word_rows_match_scalar_products() {
        let mut state = 0x123456789abcdefu64;
        let mut next = || {
            state ^= state << 13; state ^= state >> 7; state ^= state << 17; state
        };
        for log in [0usize, 1, 3, 6, 7, 10, 14] {
            let p: Vec<_> = (0..log).map(|_| F162([next(), next(), next() & ((1 << 34) - 1)])).collect();
            let words: Vec<_> = (0..(5 << log)).map(|_| crate::fields::scalar::B128((next() as u128) | ((next() as u128) << 64))).collect();
            let eq = eq_table(&p);
            let expected: Vec<_> = words.chunks_exact(1 << log).map(|column|
                eq.iter().zip(column).fold(F162::ZERO, |a, (&e, &w)| a + e * F162::from_b128(w))).collect();
            for workers in [1usize, 3, 8] {
                assert_eq!(row_evaluate_words(&words, &p, workers), expected, "log {log}, workers {workers}");
                let poly = eq_poly(&p, workers);
                assert_eq!((0..eq.len()).map(|i| poly.get(i)).collect::<Vec<_>>(), eq, "log {log}");
            }
        }
    }
}
