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
        // The fixed-operand GHASH formula used by scale_b128_row, in one
        // 128-bit lane. Folding the high operand first leaves one final fold.
        unsafe {
            let a = core::mem::transmute::<u128, __m128i>(self.0);
            let b = core::mem::transmute::<u128, __m128i>(o.0);
            let poly = _mm_set_epi64x(0, GHASH_MOD as i64);
            let kh = _mm_clmulepi64_si128::<0x01>(b, poly);
            let k = _mm_xor_si128(kh, _mm_slli_si128::<8>(b));
            let y0 = _mm_xor_si128(
                _mm_clmulepi64_si128::<0x00>(a, b),
                _mm_clmulepi64_si128::<0x01>(a, k),
            );
            let y1 = _mm_xor_si128(
                _mm_clmulepi64_si128::<0x10>(a, b),
                _mm_clmulepi64_si128::<0x11>(a, k),
            );
            let lo = _mm_xor_si128(y0, _mm_slli_si128::<8>(y1));
            let hi = _mm_srli_si128::<8>(y1);
            let res = _mm_xor_si128(lo, _mm_clmulepi64_si128::<0x00>(hi, poly));
            Self(core::mem::transmute::<__m128i, u128>(res))
        }
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
        let mut p = [0u64; 6];
        for i in 0..3 {
            for j in 0..3 {
                let t = clmul(a[i], b[j]);
                p[i + j] ^= t as u64;
                p[i + j + 1] ^= (t >> 64) as u64;
            }
        }
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
