use super::f162;
use super::scalar::F162;
use crate::arch::*;
use crate::par;

/// Fewest entries of a round's half a worker takes; smaller rounds run on the calling thread.
const PAR_MIN: usize = 1 << 14;

#[derive(Clone)]
pub struct Poly {
    pub w: [Vec<u64>; 3],
    pub n: usize,
}

impl Poly {
    pub fn zero(n: usize) -> Self {
        Self {
            w: [vec![0u64; n], vec![0u64; n], vec![0u64; n]],
            n,
        }
    }

    pub fn from_scalars(v: &[F162]) -> Self {
        let n = v.len();
        let mut w = [vec![0u64; n], vec![0u64; n], vec![0u64; n]];
        for (i, x) in v.iter().enumerate() {
            for k in 0..3 {
                w[k][i] = x.0[k];
            }
        }
        Self { w, n }
    }

    pub fn get(&self, i: usize) -> F162 {
        F162([self.w[0][i], self.w[1][i], self.w[2][i]])
    }
}

/// Eight entries from `j` of the three limb slices.
#[inline(always)]
unsafe fn load(w: [&[u64]; 3], j: usize) -> [__m512i; 3] {
    [
        _mm512_loadu_si512(w[0].as_ptr().add(j) as *const __m512i),
        _mm512_loadu_si512(w[1].as_ptr().add(j) as *const __m512i),
        _mm512_loadu_si512(w[2].as_ptr().add(j) as *const __m512i),
    ]
}

#[inline(always)]
unsafe fn store(w: &mut [&mut [u64]; 3], j: usize, v: [__m512i; 3]) {
    for k in 0..3 {
        _mm512_storeu_si512(w[k].as_mut_ptr().add(j) as *mut __m512i, v[k]);
    }
}

#[inline(always)]
fn get(w: [&[u64]; 3], j: usize) -> F162 {
    F162([w[0][j], w[1][j], w[2][j]])
}

/// One worker's share of a round: entries `[s, e)` of the low half `lo` and of the high half
/// `hi`, both indexed from the start of the share.
pub(super) struct Share<'a> {
    pub lo: [&'a mut [u64]; 3],
    pub hi: [&'a [u64]; 3],
}

/// `w` around `half` cut into the shares of `edges`, low halves writable.
fn shares<'a>(w: &'a mut [Vec<u64>; 3], half: usize, edges: &[(usize, usize)]) -> Vec<Share<'a>> {
    let [w0, w1, w2] = w;
    let cut = |v: &'a mut Vec<u64>| {
        let (lo, hi) = v.split_at_mut(half);
        let hi: &'a [u64] = &hi[..half];
        let his: Vec<&'a [u64]> = edges.iter().map(|&(s, e)| &hi[s..e]).collect();
        (par::split_mut(lo, edges), his)
    };
    let (l0, h0) = cut(w0);
    let (l1, h1) = cut(w1);
    let (l2, h2) = cut(w2);
    l0.into_iter()
        .zip(l1)
        .zip(l2)
        .zip(h0.into_iter().zip(h1).zip(h2))
        .map(|(((a, b), c), ((x, y), z))| Share {
            lo: [a, b, c],
            hi: [x, y, z],
        })
        .collect()
}

fn edges(half: usize) -> Vec<(usize, usize)> {
    let parts = par::workers(half, PAR_MIN);
    (0..parts).map(|t| par::block(half, parts, t, 8)).collect()
}

#[inline(always)]
unsafe fn xor3(a: [__m512i; 3], b: [__m512i; 3]) -> [__m512i; 3] {
    [
        _mm512_xor_si512(a[0], b[0]),
        _mm512_xor_si512(a[1], b[1]),
        _mm512_xor_si512(a[2], b[2]),
    ]
}

#[inline(always)]
unsafe fn bcast(x: F162) -> [__m512i; 3] {
    [
        _mm512_set1_epi64(x.0[0] as i64),
        _mm512_set1_epi64(x.0[1] as i64),
        _mm512_set1_epi64(x.0[2] as i64),
    ]
}

unsafe fn horiz(acc: [__m512i; 12]) -> F162 {
    let r = f162::reduce_soa8(acc);
    let w: [[u64; 8]; 3] = [
        std::mem::transmute(r[0]),
        std::mem::transmute(r[1]),
        std::mem::transmute(r[2]),
    ];
    let mut out = F162::ZERO;
    for i in 0..8 {
        out += F162([w[0][i], w[1][i], w[2][i]]);
    }
    out
}

/// The round message `[sum a0 p0, sum (a0 + a1)(p0 + p1)]` over `j < half`. Workers sum
/// disjoint ranges; their sums add (XOR) to the same field elements in any grouping.
pub(super) fn msg(a: &Poly, p: &Poly, half: usize) -> [F162; 2] {
    let ((a0, a1), (p0, p1)) = (halves(a, half), halves(p, half));
    let edges = edges(half);
    let sums = par::map_blocks(half, edges.len(), 8, |_, s, e| {
        let (a0, a1, p0, p1) = (cut(a0, s, e), cut(a1, s, e), cut(p0, s, e), cut(p1, s, e));
        #[cfg(target_arch = "aarch64")]
        if super::neon::pmull() {
            // SAFETY: the CPU has PMULL.
            return unsafe { super::neon::msg_range(a0, a1, p0, p1) };
        }
        msg_range(a0, a1, p0, p1)
    });
    sums.into_iter()
        .fold([F162::ZERO; 2], |acc, m| [acc[0] + m[0], acc[1] + m[1]])
}

#[inline]
fn cut(w: [&[u64]; 3], s: usize, e: usize) -> [&[u64]; 3] {
    [&w[0][s..e], &w[1][s..e], &w[2][s..e]]
}

/// The low and high halves of `x` around `half`.
fn halves(x: &Poly, half: usize) -> ([&[u64]; 3], [&[u64]; 3]) {
    (
        [&x.w[0][..half], &x.w[1][..half], &x.w[2][..half]],
        [&x.w[0][half..2 * half], &x.w[1][half..2 * half], &x.w[2][half..2 * half]],
    )
}

/// [`msg`] over one range of the halves.
fn msg_range(a0s: [&[u64]; 3], a1s: [&[u64]; 3], p0s: [&[u64]; 3], p1s: [&[u64]; 3]) -> [F162; 2] {
    let m = a0s[0].len();
    let mut e0 = F162::ZERO;
    let mut einf = F162::ZERO;
    let blocks = m / 8;
    unsafe {
        let mut acc0 = [_mm512_setzero_si512(); 12];
        let mut acci = [_mm512_setzero_si512(); 12];
        for b in 0..blocks {
            let j = b * 8;
            let a0 = load(a0s, j);
            let a1 = load(a1s, j);
            let p0 = load(p0s, j);
            let p1 = load(p1s, j);
            f162::mac_soa8(&mut acc0, a0, p0);
            f162::mac_soa8(&mut acci, xor3(a0, a1), xor3(p0, p1));
        }
        if blocks > 0 {
            e0 = horiz(acc0);
            einf = horiz(acci);
        }
    }
    for j in blocks * 8..m {
        let (a0, a1) = (get(a0s, j), get(a1s, j));
        let (p0, p1) = (get(p0s, j), get(p1s, j));
        e0 += a0 * p0;
        einf += (a0 + a1) * (p0 + p1);
    }
    [e0, einf]
}

/// `x[j] <- x[j] + r (x[j] + x[j + half])` for `j < half`, both polynomials, on the workers.
pub(super) fn fold(a: &mut Poly, p: &mut Poly, half: usize, r: F162) {
    let edges = edges(half);
    let sa = shares(&mut a.w, half, &edges);
    let sp = shares(&mut p.w, half, &edges);
    if edges.len() <= 1 {
        for (x, y) in sa.into_iter().zip(sp) {
            fold_share(x, r);
            fold_share(y, r);
        }
        return;
    }
    std::thread::scope(|scope| {
        for (x, y) in sa.into_iter().zip(sp) {
            scope.spawn(move || {
                fold_share(x, r);
                fold_share(y, r);
            });
        }
    });
}

/// [`fold`] on one share of one polynomial.
fn fold_share(s: Share<'_>, r: F162) {
    #[cfg(target_arch = "aarch64")]
    if super::neon::pmull() {
        // SAFETY: the CPU has PMULL.
        return unsafe { super::neon::fold_share(s.lo, s.hi, r) };
    }
    fold_share_emulated(s, r)
}

/// [`fold_share`] through the AVX-512 kernel.
fn fold_share_emulated(s: Share<'_>, r: F162) {
    let Share { mut lo, hi } = s;
    let m = hi[0].len();
    let blocks = m / 8;
    unsafe {
        let rb = bcast(r);
        for b in 0..blocks {
            let j = b * 8;
            let x0 = load([&*lo[0], &*lo[1], &*lo[2]], j);
            let x1 = load(hi, j);
            store(&mut lo, j, xor3(x0, f162::mul_soa8(rb, xor3(x0, x1))));
        }
    }
    for j in blocks * 8..m {
        let x0 = F162([lo[0][j], lo[1][j], lo[2][j]]);
        let x1 = get(hi, j);
        let y = x0 + r * (x0 + x1);
        for k in 0..3 {
            lo[k][j] = y.0[k];
        }
    }
}

pub(super) fn round(a: &mut Poly, p: &mut Poly, half: usize, r: F162) -> [F162; 2] {
    let m = msg(a, p, half);
    fold(a, p, half, r);
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_poly(n: usize, seed: &mut u128) -> Poly {
        let v: Vec<F162> = (0..n)
            .map(|_| {
                let mut word = || {
                    *seed ^= *seed << 13;
                    *seed ^= *seed >> 7;
                    *seed ^= *seed << 17;
                    (seed.wrapping_mul(0x2545F4914F6CDD1D_9E3779B97F4A7C15) >> 64) as u64
                };
                F162([word(), word(), word() & ((1 << 34) - 1)])
            })
            .collect();
        Poly::from_scalars(&v)
    }

    /// Rounds over several workers (native kernels where the CPU has them) against the emulated
    /// kernels on one range and against the scalar field.
    #[test]
    fn rounds_match_the_emulated_kernels_and_the_scalar_field() {
        let mut seed = 0x1357_9BDF_2468_ACE0_FDB9_7531_0ECA_8642u128;
        for half in [1usize, 3, 8, 13, (1 << 15) + 3, 1 << 16] {
            let n = 2 * half;
            let (mut a, mut p) = (random_poly(n, &mut seed), random_poly(n, &mut seed));
            let r = random_poly(1, &mut seed).get(0);
            let got = msg(&a, &p, half);
            let ((a0, a1), (p0, p1)) = (halves(&a, half), halves(&p, half));
            assert_eq!(got, msg_range(a0, a1, p0, p1), "msg vs emulated, half {half}");
            if half <= 64 {
                let mut want = [F162::ZERO; 2];
                for j in 0..half {
                    let (x0, x1, y0, y1) = (a.get(j), a.get(j + half), p.get(j), p.get(j + half));
                    want[0] += x0 * y0;
                    want[1] += (x0 + x1) * (y0 + y1);
                }
                assert_eq!(got, want, "msg vs scalar, half {half}");
            }
            let (mut a_emu, mut p_emu) = (a.clone(), p.clone());
            fold(&mut a, &mut p, half, r);
            for x in [&mut a_emu, &mut p_emu] {
                let whole = [(0, half)];
                for s in shares(&mut x.w, half, &whole) {
                    fold_share_emulated(s, r);
                }
            }
            assert!(a.w == a_emu.w && p.w == p_emu.w, "fold vs emulated, half {half}");
        }
    }
}
