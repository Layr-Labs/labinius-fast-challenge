use super::f162;
use super::scalar::F162;
use std::arch::x86_64::*;

#[derive(Clone)]
pub struct Poly {
    pub w: [Vec<u64>; 3],
    pub n: usize,
}

impl Poly {
    pub fn zero(n: usize) -> Self {
        Self {
            w: [
                crate::huge::zeroed(n),
                crate::huge::zeroed(n),
                crate::huge::zeroed(n),
            ],
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

    fn set(&mut self, i: usize, x: F162) {
        for k in 0..3 {
            self.w[k][i] = x.0[k];
        }
    }
}

#[inline(always)]
unsafe fn load(p: &Poly, i: usize) -> [__m512i; 3] {
    [
        _mm512_loadu_si512(p.w[0].as_ptr().add(i) as *const __m512i),
        _mm512_loadu_si512(p.w[1].as_ptr().add(i) as *const __m512i),
        _mm512_loadu_si512(p.w[2].as_ptr().add(i) as *const __m512i),
    ]
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

pub(super) fn msg(a: &Poly, p: &Poly, half: usize) -> [F162; 2] {
    let blocks = half / 8;
    let threads = if blocks >= 2048 {
        crate::huge::workers(blocks / 512)
    } else {
        1
    };
    let mut e0 = F162::ZERO;
    let mut einf = F162::ZERO;
    if threads == 1 {
        if blocks > 0 {
            let part = msg_blocks_at(a, p, half, 0, blocks);
            e0 = part[0];
            einf = part[1];
        }
    } else {
        let mut parts = vec![[F162::ZERO; 2]; threads];
        let parts_ptr = parts.as_mut_ptr() as usize;
        std::thread::scope(|scope| {
            for t in 0..threads {
                let (a, p) = (a, p);
                scope.spawn(move || {
                    let b0 = t * blocks / threads;
                    let b1 = (t + 1) * blocks / threads;
                    let part = msg_blocks_at(a, p, half, b0, b1);
                    unsafe { (parts_ptr as *mut [F162; 2]).add(t).write(part) };
                });
            }
        });
        for part in parts {
            e0 += part[0];
            einf += part[1];
        }
    }
    for j in blocks * 8..half {
        let (a0, a1) = (a.get(j), a.get(j + half));
        let (p0, p1) = (p.get(j), p.get(j + half));
        e0 += a0 * p0;
        einf += (a0 + a1) * (p0 + p1);
    }
    [e0, einf]
}

fn msg_blocks_at(a: &Poly, p: &Poly, half: usize, b0: usize, b1: usize) -> [F162; 2] {
    if b0 == b1 {
        return [F162::ZERO, F162::ZERO];
    }
    unsafe {
        let mut acc0 = [_mm512_setzero_si512(); 12];
        let mut acci = [_mm512_setzero_si512(); 12];
        for b in b0..b1 {
            let j = b * 8;
            let a0 = load(a, j);
            let a1 = load(a, j + half);
            let p0 = load(p, j);
            let p1 = load(p, j + half);
            f162::mac_soa8(&mut acc0, a0, p0);
            f162::mac_soa8(&mut acci, xor3(a0, a1), xor3(p0, p1));
        }
        [horiz(acc0), horiz(acci)]
    }
}

pub(super) fn fold(a: &mut Poly, p: &mut Poly, half: usize, r: F162) {
    let blocks = half / 8;
    let threads = if blocks >= 2048 {
        crate::huge::workers(blocks / 512)
    } else {
        1
    };
    if threads == 1 {
        fold_blocks(a, p, half, r, 0, blocks);
    } else {
        // Addresses, not references: workers write disjoint index ranges
        // (`[b0*8, b1*8)` and that interval shifted by `half`).
        let aw = [
            a.w[0].as_mut_ptr() as usize,
            a.w[1].as_mut_ptr() as usize,
            a.w[2].as_mut_ptr() as usize,
        ];
        let pw = [
            p.w[0].as_mut_ptr() as usize,
            p.w[1].as_mut_ptr() as usize,
            p.w[2].as_mut_ptr() as usize,
        ];
        std::thread::scope(|scope| {
            for t in 0..threads {
                let (aw, pw) = (aw, pw);
                scope.spawn(move || {
                    let b0 = t * blocks / threads;
                    let b1 = (t + 1) * blocks / threads;
                    fold_blocks_raw(
                        [aw[0] as *mut u64, aw[1] as *mut u64, aw[2] as *mut u64],
                        [pw[0] as *mut u64, pw[1] as *mut u64, pw[2] as *mut u64],
                        half,
                        r,
                        b0,
                        b1,
                    );
                });
            }
        });
    }
    for j in blocks * 8..half {
        let (a0, a1) = (a.get(j), a.get(j + half));
        let (p0, p1) = (p.get(j), p.get(j + half));
        a.set(j, a0 + r * (a0 + a1));
        p.set(j, p0 + r * (p0 + p1));
    }
}

fn fold_blocks(a: &mut Poly, p: &mut Poly, half: usize, r: F162, b0: usize, b1: usize) {
    fold_blocks_raw(
        [a.w[0].as_mut_ptr(), a.w[1].as_mut_ptr(), a.w[2].as_mut_ptr()],
        [p.w[0].as_mut_ptr(), p.w[1].as_mut_ptr(), p.w[2].as_mut_ptr()],
        half,
        r,
        b0,
        b1,
    );
}

fn fold_blocks_raw(a: [*mut u64; 3], p: [*mut u64; 3], half: usize, r: F162, b0: usize, b1: usize) {
    unsafe {
        let rb = bcast(r);
        for b in b0..b1 {
            let j = b * 8;
            let a0 = load_raw(a, j);
            let a1 = load_raw(a, j + half);
            let p0 = load_raw(p, j);
            let p1 = load_raw(p, j + half);
            store_raw(a, j, xor3(a0, f162::mul_soa8(rb, xor3(a0, a1))));
            store_raw(p, j, xor3(p0, f162::mul_soa8(rb, xor3(p0, p1))));
        }
    }
}

#[inline(always)]
unsafe fn load_raw(w: [*mut u64; 3], i: usize) -> [__m512i; 3] {
    [
        _mm512_loadu_si512(w[0].add(i) as *const __m512i),
        _mm512_loadu_si512(w[1].add(i) as *const __m512i),
        _mm512_loadu_si512(w[2].add(i) as *const __m512i),
    ]
}

#[inline(always)]
unsafe fn store_raw(w: [*mut u64; 3], i: usize, v: [__m512i; 3]) {
    for k in 0..3 {
        _mm512_storeu_si512(w[k].add(i) as *mut __m512i, v[k]);
    }
}

pub(super) fn round(a: &mut Poly, p: &mut Poly, half: usize, r: F162) -> [F162; 2] {
    let m = msg(a, p, half);
    fold(a, p, half, r);
    m
}
