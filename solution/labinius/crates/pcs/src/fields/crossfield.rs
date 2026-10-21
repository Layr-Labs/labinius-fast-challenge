use super::bitmat::{transpose_tile, untranspose_planes, GF_IDENT};
use super::f162 as bf;
use super::scalar::{B128, F162};
use super::sumcheck::{bcast, horiz};
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
    let n = 1 << r.len();
    let mut t: Vec<B128> = Vec::with_capacity(n);
    unsafe { t.set_len(n) };
    eq_expand_b128_into(r, &mut t);
    t
}

fn worker_count(n: usize) -> usize {
    if n < 2 {
        return 1;
    }
    std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(1)
        .clamp(1, 8)
        .min(n)
}

/// The expansion in place, doubling from the back so no entry is read after it is written.
pub fn eq_expand_b128_into(r: &[B128], t: &mut [B128]) {
    assert_eq!(t.len(), 1 << r.len());
    if r.len() >= 16 {
        let h = r.len() / 2;
        let hi = eq_expand_b128(&r[..r.len() - h]);
        let lo = eq_expand_b128(&r[r.len() - h..]);
        let rows = lo.len();
        let cols = hi.len();
        let threads = worker_count(cols);
        let per = cols.div_ceil(threads);
        std::thread::scope(|scope| {
            for (k, chunk) in t.chunks_mut(rows * per).enumerate() {
                let (lo, hi) = (&lo, &hi);
                scope.spawn(move || {
                    for (n, row) in chunk.chunks_mut(rows).enumerate() {
                        let h = hi[k * per + n];
                        for (out, &l) in row.iter_mut().zip(lo.iter()) {
                            *out = l * h;
                        }
                    }
                });
            }
        });
        return;
    }
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
fn partial_evals_tiles(
    pi0: &[B128],
    eq_hi: &[B128],
    first: usize,
    last: usize,
    v: &mut [B128; PACK],
) {
    unsafe {
        let ident = _mm512_set1_epi64(GF_IDENT);
        let mut acc = [[_mm512_setzero_si512(); 16]; 2];
        let mut xe = [_mm512_setzero_si512(); 16];
        let mut xp = [_mm512_setzero_si512(); 16];
        let mut ap = [[0u64; 8]; 16];
        for t in first..last {
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
}

fn partial_evals(pi0: &[B128], eq_hi: &[B128]) -> Vec<B128> {
    assert_eq!(pi0.len(), eq_hi.len());
    let n = pi0.len();
    let tiles = n / TILE;
    let mut v = [B128::ZERO; PACK];
    let threads = worker_count(tiles);
    let per = tiles.div_ceil(threads);
    if threads > 1 && per > 0 {
        let mut parts = Vec::with_capacity(threads);
        std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(threads);
            for first in (0..tiles).step_by(per) {
                let last = (first + per).min(tiles);
                handles.push(scope.spawn(move || {
                    let mut part = [B128::ZERO; PACK];
                    partial_evals_tiles(pi0, eq_hi, first, last, &mut part);
                    part
                }));
            }
            for handle in handles {
                parts.push(handle.join().unwrap());
            }
        });
        for part in parts {
            for (a, b) in v.iter_mut().zip(part) {
                *a = *a + b;
            }
        }
    } else {
        partial_evals_tiles(pi0, eq_hi, 0, tiles, &mut v);
    }
    let mut out = v.to_vec();
    for (&p, &e) in pi0[TILE * tiles..].iter().zip(&eq_hi[TILE * tiles..]) {
        for (k, vk) in out.iter_mut().enumerate() {
            if p.bit(k) {
                *vk = *vk + e;
            }
        }
    }
    out
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

unsafe fn poly_from_b128_blocks(
    pi0: &[B128],
    first: usize,
    last: usize,
    w0: *mut u64,
    w1: *mut u64,
) {
    for b in first..last {
        let (lo, hi) = split_words(pi0.as_ptr().add(8 * b));
        _mm512_storeu_si512(w0.add(8 * b) as *mut __m512i, lo);
        _mm512_storeu_si512(w1.add(8 * b) as *mut __m512i, hi);
    }
}

/// `pi0` in the sumcheck's layout: the bits of a `B128` are the low coefficients of an `F162`.
fn poly_from_b128(pi0: &[B128], out: &mut super::sumcheck::Poly) {
    let n = pi0.len();
    assert_eq!(out.n, n);
    let w = &mut out.w;
    w[2].fill(0);
    let blocks = n / 8;
    let threads = worker_count(blocks);
    let per = blocks.div_ceil(threads);
    if threads > 1 && per > 0 {
        let w0 = w[0].as_mut_ptr() as usize;
        let w1 = w[1].as_mut_ptr() as usize;
        std::thread::scope(|scope| {
            for first in (0..blocks).step_by(per) {
                let last = (first + per).min(blocks);
                scope.spawn(move || unsafe {
                    poly_from_b128_blocks(pi0, first, last, w0 as *mut u64, w1 as *mut u64);
                });
            }
        });
    } else {
        unsafe { poly_from_b128_blocks(pi0, 0, blocks, w[0].as_mut_ptr(), w[1].as_mut_ptr()) };
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
unsafe fn poly_from_eq_tiles(
    eqs: &[&[B128]],
    ms: &[[[u64; 16]; 24]],
    first: usize,
    last: usize,
    w: [*mut u64; 3],
) {
    let mut xe = [_mm512_setzero_si512(); 16];
    for t in first..last {
        let mut acc = [_mm512_setzero_si512(); 24];
        for (eq, m) in eqs.iter().zip(ms) {
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
        for l in 0..3 {
            let out = untranspose_planes(core::array::from_fn(|c| acc[8 * l + c]));
            let wl = *w.get_unchecked(l);
            for (g, o) in out.iter().enumerate() {
                _mm512_storeu_si512(wl.add(TILE * t + 8 * g) as *mut __m512i, *o);
            }
        }
    }
}

fn poly_from_eq(eqs: &[&[B128]], batches: &[Vec<F162>], out: &mut super::sumcheck::Poly) {
    let n = eqs[0].len();
    assert_eq!(out.n, n);
    let w = &mut out.w;
    let tiles = n / TILE;
    let ms: Vec<[[u64; 16]; 24]> = batches.iter().map(|b| eq_matrices(b)).collect();
    let threads = worker_count(tiles);
    let per = tiles.div_ceil(threads);
    if threads > 1 && per > 0 {
        let ptrs = [
            w[0].as_mut_ptr() as usize,
            w[1].as_mut_ptr() as usize,
            w[2].as_mut_ptr() as usize,
        ];
        std::thread::scope(|scope| {
            for first in (0..tiles).step_by(per) {
                let last = (first + per).min(tiles);
                let ms = &ms;
                scope.spawn(move || unsafe {
                    poly_from_eq_tiles(
                        eqs,
                        ms,
                        first,
                        last,
                        [
                            ptrs[0] as *mut u64,
                            ptrs[1] as *mut u64,
                            ptrs[2] as *mut u64,
                        ],
                    );
                });
            }
        });
    } else {
        unsafe {
            poly_from_eq_tiles(
                eqs,
                &ms,
                0,
                tiles,
                [w[0].as_mut_ptr(), w[1].as_mut_ptr(), w[2].as_mut_ptr()],
            );
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

/// The 128 F162 images of the GF(2)-linear map the transparent coefficient is, applied to one
/// GHASH word: xor of the images its set bits select.
#[inline]
fn image_apply(images: &[F162], x: B128) -> F162 {
    let mut acc = F162::ZERO;
    let mut bits = x.0;
    while bits != 0 {
        let k = bits.trailing_zeros() as usize;
        acc += images[k];
        bits &= bits - 1;
    }
    acc
}

/// The images of `L o (x -> c x)` from the images of `L`.
fn image_compose_mul(images: &[F162], c: B128) -> Vec<F162> {
    (0..128)
        .map(|k| image_apply(images, c * B128(1u128 << k)))
        .collect()
}

/// `x^(2^128 - 2)` in the GHASH field, by square and multiply.
fn b128_inv(x: B128) -> B128 {
    let mut result = B128::ONE;
    let mut base = x;
    let mut e = u128::MAX - 1;
    while e != 0 {
        if e & 1 == 1 {
            result = result * base;
        }
        base = base * base;
        e >>= 1;
    }
    result
}

/// 8 consecutive GHASH words as the word-sliced `[low qwords, high qwords, 0]` an F162 run takes.
#[inline(always)]
unsafe fn words_soa8(p: *const B128) -> [__m512i; 3] {
    let a = _mm512_loadu_si512(p as *const __m512i);
    let b = _mm512_loadu_si512(p.add(4) as *const __m512i);
    let lo = _mm512_setr_epi64(0, 2, 4, 6, 8, 10, 12, 14);
    let hi = _mm512_setr_epi64(1, 3, 5, 7, 9, 11, 13, 15);
    [
        _mm512_permutex2var_epi64(a, lo, b),
        _mm512_permutex2var_epi64(a, hi, b),
        _mm512_setzero_si512(),
    ]
}

/// Both linear maps over one 64-word equality tile: the current map (for the `p[j]` half of the
/// message) and its composition with `x -> (1+c) x` (for the `p[j] + p[j+half]` half).
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,gfni")]
unsafe fn tile_maps(
    xe: &[__m512i; 16],
    ms_l: &[[u64; 16]; 24],
    ms_m: &[[u64; 16]; 24],
) -> ([[__m512i; 8]; 3], [[__m512i; 8]; 3]) {
    let mut al = [_mm512_setzero_si512(); 24];
    let mut am = [_mm512_setzero_si512(); 24];
    for (k, (rl, rm)) in ms_l.iter().zip(ms_m).enumerate() {
        for j in (0..16).step_by(2) {
            let l0 = _mm512_set1_epi64(rl[j] as i64);
            let l1 = _mm512_set1_epi64(rl[j + 1] as i64);
            let m0 = _mm512_set1_epi64(rm[j] as i64);
            let m1 = _mm512_set1_epi64(rm[j + 1] as i64);
            al[k] = xor3(al[k], affine(xe[j], l0), affine(xe[j + 1], l1));
            am[k] = xor3(am[k], affine(xe[j], m0), affine(xe[j + 1], m1));
        }
    }
    let pack = |a: &[__m512i; 24]| {
        let mut out = [[_mm512_setzero_si512(); 8]; 3];
        for l in 0..3 {
            out[l] = untranspose_planes(core::array::from_fn(|c| a[8 * l + c]));
        }
        out
    };
    (pack(&al), pack(&am))
}

#[inline(always)]
unsafe fn load_planes(p: [*const u64; 3], at: usize) -> [__m512i; 3] {
    [
        _mm512_loadu_si512(p[0].add(at) as *const __m512i),
        _mm512_loadu_si512(p[1].add(at) as *const __m512i),
        _mm512_loadu_si512(p[2].add(at) as *const __m512i),
    ]
}

/// Accumulate one tile of `a[j] * p[j]`, optionally against `p[j] + p[j+half]`.
#[inline(always)]
unsafe fn dot_planes_tile(
    acc: &mut [__m512i; 12],
    a: &[[__m512i; 8]; 3],
    p: [*const u64; 3],
    at: usize,
    half: Option<usize>,
) {
    for g in 0..8 {
        let mut b = load_planes(p, at + 8 * g);
        if let Some(h) = half {
            let q = load_planes(p, at + h + 8 * g);
            for l in 0..3 {
                b[l] = _mm512_xor_si512(b[l], q[l]);
            }
        }
        bf::mac_soa8(acc, [a[0][g], a[1][g], a[2][g]], b);
    }
}

#[inline(always)]
unsafe fn dot_words_tile(
    acc: &mut [__m512i; 12],
    a: &[[__m512i; 8]; 3],
    w: *const B128,
    at: usize,
    half: Option<usize>,
) {
    for g in 0..8 {
        let mut b = words_soa8(w.add(at + 8 * g));
        if let Some(h) = half {
            let q = words_soa8(w.add(at + h + 8 * g));
            for l in 0..3 {
                b[l] = _mm512_xor_si512(b[l], q[l]);
            }
        }
        bf::mac_soa8(acc, [a[0][g], a[1][g], a[2][g]], b);
    }
}

/// One round's message over the current `p` planes.
fn lean_msg_planes(
    eq: &[B128],
    p: [&[u64]; 3],
    images: &[F162],
    m_images: &[F162],
    n: usize,
    ms_l: &[[u64; 16]; 24],
    ms_m: &[[u64; 16]; 24],
) -> [F162; 2] {
    let half = n / 2;
    let tiles = half / TILE;
    let threads = worker_count(tiles);
    let per = tiles.div_ceil(threads);
    let eq_ptr = eq.as_ptr() as usize;
    let pp = [
        p[0].as_ptr() as usize,
        p[1].as_ptr() as usize,
        p[2].as_ptr() as usize,
    ];
    let mut parts = Vec::new();
    if threads > 1 && per > 0 {
        std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(threads);
            for t0 in (0..tiles).step_by(per) {
                let t1 = (t0 + per).min(tiles);
                handles.push(scope.spawn(move || unsafe {
                    let mut acc0 = [_mm512_setzero_si512(); 12];
                    let mut acci = [_mm512_setzero_si512(); 12];
                    let mut xe = [_mm512_setzero_si512(); 16];
                    for t in t0..t1 {
                        let at = TILE * t;
                        transpose_tile((eq_ptr as *const B128).add(at) as *const u8, &mut xe);
                        let (al, am) = tile_maps(&xe, ms_l, ms_m);
                        let planes = [pp[0] as *const u64, pp[1] as *const u64, pp[2] as *const u64];
                        dot_planes_tile(&mut acc0, &al, planes, at, None);
                        dot_planes_tile(&mut acci, &am, planes, at, Some(half));
                    }
                    [horiz(acc0), horiz(acci)]
                }));
            }
            for h in handles {
                parts.push(h.join().unwrap());
            }
        });
    } else {
        unsafe {
            let mut acc0 = [_mm512_setzero_si512(); 12];
            let mut acci = [_mm512_setzero_si512(); 12];
            let mut xe = [_mm512_setzero_si512(); 16];
            for t in 0..tiles {
                let at = TILE * t;
                transpose_tile((eq_ptr as *const B128).add(at) as *const u8, &mut xe);
                let (al, am) = tile_maps(&xe, ms_l, ms_m);
                let planes = [pp[0] as *const u64, pp[1] as *const u64, pp[2] as *const u64];
                dot_planes_tile(&mut acc0, &al, planes, at, None);
                dot_planes_tile(&mut acci, &am, planes, at, Some(half));
            }
            parts.push([horiz(acc0), horiz(acci)]);
        }
    }
    let mut e0 = F162::ZERO;
    let mut einf = F162::ZERO;
    for [a, b] in parts {
        e0 += a;
        einf += b;
    }
    for j in (TILE * tiles)..half {
        let pj = F162([p[0][j], p[1][j], p[2][j]]);
        let ph = F162([p[0][j + half], p[1][j + half], p[2][j + half]]);
        e0 += image_apply(images, eq[j]) * pj;
        einf += image_apply(m_images, eq[j]) * (pj + ph);
    }
    [e0, einf]
}

/// Round 0's message, with `p` read straight from the words.
fn lean_msg_words(
    eq: &[B128],
    words: &[B128],
    images: &[F162],
    m_images: &[F162],
    half: usize,
    ms_l: &[[u64; 16]; 24],
    ms_m: &[[u64; 16]; 24],
) -> [F162; 2] {
    let tiles = half / TILE;
    let threads = worker_count(tiles);
    let per = tiles.div_ceil(threads);
    let eq_ptr = eq.as_ptr() as usize;
    let w_ptr = words.as_ptr() as usize;
    let mut parts = Vec::new();
    if threads > 1 && per > 0 {
        std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(threads);
            for t0 in (0..tiles).step_by(per) {
                let t1 = (t0 + per).min(tiles);
                handles.push(scope.spawn(move || unsafe {
                    let mut acc0 = [_mm512_setzero_si512(); 12];
                    let mut acci = [_mm512_setzero_si512(); 12];
                    let mut xe = [_mm512_setzero_si512(); 16];
                    for t in t0..t1 {
                        let at = TILE * t;
                        transpose_tile((eq_ptr as *const B128).add(at) as *const u8, &mut xe);
                        let (al, am) = tile_maps(&xe, ms_l, ms_m);
                        let w = w_ptr as *const B128;
                        dot_words_tile(&mut acc0, &al, w, at, None);
                        dot_words_tile(&mut acci, &am, w, at, Some(half));
                    }
                    [horiz(acc0), horiz(acci)]
                }));
            }
            for h in handles {
                parts.push(h.join().unwrap());
            }
        });
    } else {
        unsafe {
            let mut acc0 = [_mm512_setzero_si512(); 12];
            let mut acci = [_mm512_setzero_si512(); 12];
            let mut xe = [_mm512_setzero_si512(); 16];
            for t in 0..tiles {
                let at = TILE * t;
                transpose_tile((eq_ptr as *const B128).add(at) as *const u8, &mut xe);
                let (al, am) = tile_maps(&xe, ms_l, ms_m);
                dot_words_tile(&mut acc0, &al, w_ptr as *const B128, at, None);
                dot_words_tile(&mut acci, &am, w_ptr as *const B128, at, Some(half));
            }
            parts.push([horiz(acc0), horiz(acci)]);
        }
    }
    let mut e0 = F162::ZERO;
    let mut einf = F162::ZERO;
    for [a, b] in parts {
        e0 += a;
        einf += b;
    }
    for j in (TILE * tiles)..half {
        let pj = F162::from_b128(words[j]);
        let ph = F162::from_b128(words[j + half]);
        e0 += image_apply(images, eq[j]) * pj;
        einf += image_apply(m_images, eq[j]) * (pj + ph);
    }
    [e0, einf]
}

#[inline(always)]
unsafe fn store_planes(p: [*mut u64; 3], at: usize, v: [__m512i; 3]) {
    for l in 0..3 {
        _mm512_storeu_si512(p[l].add(at) as *mut __m512i, v[l]);
    }
}

#[inline(always)]
unsafe fn xor_arrays(a: [__m512i; 3], b: [__m512i; 3]) -> [__m512i; 3] {
    core::array::from_fn(|l| _mm512_xor_si512(a[l], b[l]))
}

/// In-place fold of the `p` planes over the first half.
fn lean_fold_planes(p: &mut [Vec<u64>; 3], half: usize, r: F162) {
    if half < 8 {
        for j in 0..half {
            let lo = F162([p[0][j], p[1][j], p[2][j]]);
            let hi = F162([p[0][j + half], p[1][j + half], p[2][j + half]]);
            let v = lo + r * (lo + hi);
            p[0][j] = v.0[0];
            p[1][j] = v.0[1];
            p[2][j] = v.0[2];
        }
        return;
    }
    let rb = unsafe { bcast(r) };
    let blocks = half / 8;
    let threads = worker_count(blocks);
    let per = blocks.div_ceil(threads);
    let bases = [
        p[0].as_mut_ptr() as usize,
        p[1].as_mut_ptr() as usize,
        p[2].as_mut_ptr() as usize,
    ];
    if threads > 1 && per > 0 {
        std::thread::scope(|scope| {
            for b0 in (0..blocks).step_by(per) {
                let b1 = (b0 + per).min(blocks);
                scope.spawn(move || unsafe {
                    let planes = [
                        bases[0] as *mut u64,
                        bases[1] as *mut u64,
                        bases[2] as *mut u64,
                    ];
                    for b in b0..b1 {
                        let j = 8 * b;
                        let lo = load_planes([planes[0], planes[1], planes[2]], j);
                        let hi = load_planes([planes[0], planes[1], planes[2]], j + half);
                        let d = xor_arrays(lo, hi);
                        store_planes(planes, j, xor_arrays(lo, bf::mul_soa8(rb, d)));
                    }
                });
            }
        });
    } else {
        let planes = [p[0].as_mut_ptr(), p[1].as_mut_ptr(), p[2].as_mut_ptr()];
        unsafe {
            for b in 0..blocks {
                let j = 8 * b;
                let lo = load_planes([planes[0], planes[1], planes[2]], j);
                let hi = load_planes([planes[0], planes[1], planes[2]], j + half);
                let d = xor_arrays(lo, hi);
                store_planes(planes, j, xor_arrays(lo, bf::mul_soa8(rb, d)));
            }
        }
    }
    let _ = &bases;
}

/// Round 0's fold, reading `p` from the words and writing the half-size planes.
fn lean_fold_words(words: &[B128], p: &mut [Vec<u64>; 3], half: usize, r: F162) {
    if half < 8 {
        for j in 0..half {
            let lo = F162::from_b128(words[j]);
            let hi = F162::from_b128(words[j + half]);
            let v = lo + r * (lo + hi);
            p[0][j] = v.0[0];
            p[1][j] = v.0[1];
            p[2][j] = v.0[2];
        }
        return;
    }
    let rb = unsafe { bcast(r) };
    let blocks = half / 8;
    let threads = worker_count(blocks);
    let per = blocks.div_ceil(threads);
    let w_ptr = words.as_ptr() as usize;
    let bases = [
        p[0].as_mut_ptr() as usize,
        p[1].as_mut_ptr() as usize,
        p[2].as_mut_ptr() as usize,
    ];
    if threads > 1 && per > 0 {
        std::thread::scope(|scope| {
            for b0 in (0..blocks).step_by(per) {
                let b1 = (b0 + per).min(blocks);
                scope.spawn(move || unsafe {
                    let planes = [
                        bases[0] as *mut u64,
                        bases[1] as *mut u64,
                        bases[2] as *mut u64,
                    ];
                    let w = w_ptr as *const B128;
                    for b in b0..b1 {
                        let j = 8 * b;
                        let lo = words_soa8(w.add(j));
                        let hi = words_soa8(w.add(j + half));
                        let d = xor_arrays(lo, hi);
                        store_planes(planes, j, xor_arrays(lo, bf::mul_soa8(rb, d)));
                    }
                });
            }
        });
    } else {
        let planes = [p[0].as_mut_ptr(), p[1].as_mut_ptr(), p[2].as_mut_ptr()];
        unsafe {
            for b in 0..blocks {
                let j = 8 * b;
                let lo = words_soa8((w_ptr as *const B128).add(j));
                let hi = words_soa8((w_ptr as *const B128).add(j + half));
                let d = xor_arrays(lo, hi);
                store_planes(planes, j, xor_arrays(lo, bf::mul_soa8(rb, d)));
            }
        }
    }
    let _ = &bases;
}

/// The field switch with `a` kept as a GF(2)-linear image of the equality table and `p` folded
/// in place in half the space. It sends exactly the messages [`SwitchProver`] sends.
pub struct LeanSwitch<'a> {
    words: &'a [B128],
    eq: Vec<B128>,
    hi: Vec<B128>,
    images: Vec<F162>,
    lc: Vec<F162>,
    p: [Vec<u64>; 3],
    n: usize,
    round: usize,
}

impl<'a> LeanSwitch<'a> {
    /// `None` when the equality point has a coordinate equal to one, which has no inverse for
    /// the composed map; the caller then uses the materialised [`SwitchProver`].
    pub fn new(words: &'a [B128], eq: Vec<B128>, batch: &[F162], hi: &[B128]) -> Option<Self> {
        let l = hi.len();
        if l == 0 || words.len() != 1usize << l || eq.len() != 1usize << l {
            return None;
        }
        if hi.iter().any(|&h| h + B128::ONE == B128::ZERO) {
            return None;
        }
        let half = 1usize << (l - 1);
        Some(Self {
            words,
            eq,
            hi: hi.to_vec(),
            images: batch.to_vec(),
            lc: vec![F162::ZERO; 128],
            p: [vec![0u64; half], vec![0u64; half], vec![0u64; half]],
            n: 1usize << l,
            round: 0,
        })
    }

    pub fn msg(&mut self) -> [F162; 2] {
        let t = self.round;
        let half = self.n / 2;
        let rk = self.hi[t];
        let c = rk * b128_inv(rk + B128::ONE);
        self.lc = image_compose_mul(&self.images, c);
        let m_images: Vec<F162> = self
            .images
            .iter()
            .zip(&self.lc)
            .map(|(&a, &b)| a + b)
            .collect();
        let ms_l = eq_matrices(&self.images);
        let ms_m = eq_matrices(&m_images);
        if t == 0 {
            lean_msg_words(
                &self.eq,
                self.words,
                &self.images,
                &m_images,
                half,
                &ms_l,
                &ms_m,
            )
        } else {
            let p = [&self.p[0][..], &self.p[1][..], &self.p[2][..]];
            lean_msg_planes(
                &self.eq,
                p,
                &self.images,
                &m_images,
                self.n,
                &ms_l,
                &ms_m,
            )
        }
    }

    pub fn fold(&mut self, r: F162) {
        let half = self.n / 2;
        if self.round == 0 {
            lean_fold_words(self.words, &mut self.p, half, r);
        } else {
            lean_fold_planes(&mut self.p, half, r);
        }
        for i in 0..128 {
            let v = self.images[i] + r * (self.images[i] + self.lc[i]);
            self.images[i] = v;
        }
        self.n = half;
        self.round += 1;
    }

    pub fn final_eval(&self) -> F162 {
        F162([self.p[0][0], self.p[1][0], self.p[2][0]])
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
        let n = 1usize << l;
        Self {
            eq: (0..claims)
                .map(|_| {
                    let mut v: Vec<B128> = Vec::with_capacity(n);
                    unsafe { v.set_len(n) };
                    v
                })
                .collect(),
            a: super::sumcheck::Poly::zero(n),
            p: super::sumcheck::Poly::zero(n),
        }
    }

    /// The equality table only: the lean switch keeps `a` as a linear map and `p` in half the
    /// space, so neither polynomial is allocated.
    pub fn new_lean(l: usize) -> Self {
        let n = 1usize << l;
        let mut v: Vec<B128> = Vec::with_capacity(n);
        unsafe { v.set_len(n) };
        Self {
            eq: vec![v],
            a: super::sumcheck::Poly::zero(0),
            p: super::sumcheck::Poly::zero(0),
        }
    }

    /// A full workspace around an equality table the caller already built.
    pub fn with_eq(eq: Vec<B128>) -> Self {
        let n = eq.len();
        Self {
            eq: vec![eq],
            a: super::sumcheck::Poly::zero(n),
            p: super::sumcheck::Poly::zero(n),
        }
    }

    /// The first claim's equality table.
    pub fn into_eq(mut self) -> Vec<B128> {
        self.eq.pop().expect("one equality table")
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
