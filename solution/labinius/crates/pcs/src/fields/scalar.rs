use std::arch::x86_64::*;
use std::ops::{Add, AddAssign, Mul};

const GHASH_MOD: u128 = (1 << 7) | (1 << 2) | (1 << 1) | 1;

/// `repr(transparent)`: a `B128` is exactly the 16 little-endian bytes of its word, which is what
/// lets the witness table be read in place.
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct B128(pub u128);

/// `repr(C)`: three consecutive limbs, 24 bytes, which the bit-slicing kernels read raw.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct F162(pub [u64; 3]);

#[inline]
fn clmul(a: u64, b: u64) -> u128 {
    unsafe {
        let r =
            _mm_clmulepi64_si128::<0x00>(_mm_set_epi64x(0, a as i64), _mm_set_epi64x(0, b as i64));
        let lo = _mm_cvtsi128_si64(r) as u64;
        let hi = _mm_extract_epi64::<1>(r) as u64;
        (lo as u128) | ((hi as u128) << 64)
    }
}

impl B128 {
    pub const ZERO: Self = Self(0);
    pub const ONE: Self = Self(1);

    #[inline]
    pub fn bit(self, i: usize) -> bool {
        (self.0 >> i) & 1 == 1
    }
}

impl Add for B128 {
    type Output = Self;
    #[inline]
    fn add(self, o: Self) -> Self {
        Self(self.0 ^ o.0)
    }
}

impl Mul for B128 {
    type Output = Self;
    #[inline]
    fn mul(self, o: Self) -> Self {
        let (a0, a1) = (self.0 as u64, (self.0 >> 64) as u64);
        let (b0, b1) = (o.0 as u64, (o.0 >> 64) as u64);
        let lo = clmul(a0, b0);
        let hi = clmul(a1, b1);
        let mid = clmul(a0 ^ a1, b0 ^ b1) ^ lo ^ hi;
        let p_lo = lo ^ (mid << 64);
        let p_hi = hi ^ (mid >> 64);
        let mut acc = p_lo;
        let mut h = p_hi;
        for _ in 0..2 {
            let c = clmul(h as u64, GHASH_MOD as u64)
                ^ ((clmul((h >> 64) as u64, GHASH_MOD as u64)) << 64);
            let carry = clmul((h >> 64) as u64, GHASH_MOD as u64) >> 64;
            acc ^= c;
            h = carry;
        }
        Self(acc)
    }
}

const M34: u64 = (1 << 34) - 1;

impl F162 {
    pub const ZERO: Self = Self([0; 3]);
    pub const ONE: Self = Self([1, 0, 0]);

    #[inline]
    pub fn from_b128(x: B128) -> Self {
        Self([x.0 as u64, (x.0 >> 64) as u64, 0])
    }
}

impl Add for F162 {
    type Output = Self;
    #[inline]
    fn add(self, o: Self) -> Self {
        Self([self.0[0] ^ o.0[0], self.0[1] ^ o.0[1], self.0[2] ^ o.0[2]])
    }
}

impl AddAssign for F162 {
    #[inline]
    fn add_assign(&mut self, o: Self) {
        *self = *self + o;
    }
}

impl Mul for F162 {
    type Output = Self;
    #[inline]
    fn mul(self, o: Self) -> Self {
        let a = self.0;
        let b = o.0;
        // The same six-product Karatsuba decomposition as the word-sliced
        // kernel. Recombine the complete 324-bit polynomial before reducing it.
        let m0 = clmul(a[0], b[0]);
        let m1 = clmul(a[1], b[1]);
        let m2 = clmul(a[2], b[2]);
        let d1 = clmul(a[0] ^ a[1], b[0] ^ b[1]) ^ m0 ^ m1;
        let d2 = clmul(a[0] ^ a[2], b[0] ^ b[2]) ^ m0 ^ m2 ^ m1;
        let d3 = clmul(a[1] ^ a[2], b[1] ^ b[2]) ^ m1 ^ m2;
        let p = [
            m0 as u64,
            (m0 >> 64) as u64 ^ d1 as u64,
            (d1 >> 64) as u64 ^ d2 as u64,
            (d2 >> 64) as u64 ^ d3 as u64,
            (d3 >> 64) as u64 ^ m2 as u64,
            (m2 >> 64) as u64,
        ];
        let h00 = (p[2] >> 34) | (p[3] << 30);
        let h01 = (p[3] >> 34) & ((1 << 17) - 1);
        let h10 = (p[3] >> 51) | (p[4] << 13);
        let h11 = (p[4] >> 51) | (p[5] << 13);
        Self([
            p[0] ^ h00 ^ h10,
            p[1] ^ h01 ^ h11 ^ (h00 << 17),
            (p[2] & M34) ^ (h01 << 17) ^ (h00 >> 47),
        ])
    }
}
