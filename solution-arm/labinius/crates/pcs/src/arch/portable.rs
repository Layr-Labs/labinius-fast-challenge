//! Portable emulation of the AVX-512 / GFNI / VPCLMULQDQ intrinsics the kernels use.
//!
//! Every function follows the lane semantics of the Intel Intrinsics Guide entry of the same name,
//! including wrap-around, saturation-free truncation and out-of-range shift counts, so a kernel
//! computes the same bits here as on the scored machine. Register-only operations are safe
//! functions, as in current `core::arch::x86_64`; operations through raw pointers are `unsafe`.
//! Masked loads read only the selected lanes, matching AVX-512 fault suppression.
#![allow(non_camel_case_types, clippy::missing_safety_doc, clippy::too_many_arguments)]

use core::mem::transmute_copy;

#[derive(Clone, Copy, Debug)]
#[repr(C, align(64))]
pub struct __m512i([u64; 8]);

#[derive(Clone, Copy, Debug)]
#[repr(C, align(32))]
pub struct __m256i([u64; 4]);

#[derive(Clone, Copy, Debug)]
#[repr(C, align(16))]
pub struct __m128i([u64; 2]);

#[derive(Clone, Copy, Debug)]
#[repr(C, align(64))]
pub struct __m512d([f64; 8]);

#[derive(Clone, Copy, Debug)]
#[repr(C, align(64))]
pub struct __m512([f32; 16]);

pub type __mmask8 = u8;
pub type __mmask16 = u16;
pub type __mmask32 = u32;
pub type __mmask64 = u64;

pub const _MM_HINT_T0: i32 = 3;
pub const _MM_HINT_T1: i32 = 2;
pub const _MM_HINT_T2: i32 = 1;
pub const _MM_HINT_NTA: i32 = 0;

pub const _MM_CMPINT_EQ: i32 = 0x00;
pub const _MM_CMPINT_LT: i32 = 0x01;
pub const _MM_CMPINT_LE: i32 = 0x02;
pub const _MM_CMPINT_FALSE: i32 = 0x03;
pub const _MM_CMPINT_NE: i32 = 0x04;
pub const _MM_CMPINT_NLT: i32 = 0x05;
pub const _MM_CMPINT_NLE: i32 = 0x06;
pub const _MM_CMPINT_TRUE: i32 = 0x07;

pub const _MM_FROUND_TO_NEAREST_INT: i32 = 0x00;
pub const _MM_FROUND_TO_NEG_INF: i32 = 0x01;
pub const _MM_FROUND_TO_POS_INF: i32 = 0x02;
pub const _MM_FROUND_TO_ZERO: i32 = 0x03;
pub const _MM_FROUND_CUR_DIRECTION: i32 = 0x04;
pub const _MM_FROUND_NO_EXC: i32 = 0x08;

pub const _CMP_EQ_OQ: i32 = 0x00;
pub const _CMP_LT_OS: i32 = 0x01;
pub const _CMP_LE_OS: i32 = 0x02;
pub const _CMP_UNORD_Q: i32 = 0x03;
pub const _CMP_NEQ_UQ: i32 = 0x04;
pub const _CMP_NLT_US: i32 = 0x05;
pub const _CMP_NLE_US: i32 = 0x06;
pub const _CMP_ORD_Q: i32 = 0x07;
pub const _CMP_EQ_UQ: i32 = 0x08;
pub const _CMP_NGE_US: i32 = 0x09;
pub const _CMP_NGT_US: i32 = 0x0a;
pub const _CMP_FALSE_OQ: i32 = 0x0b;
pub const _CMP_NEQ_OQ: i32 = 0x0c;
pub const _CMP_GE_OS: i32 = 0x0d;
pub const _CMP_GT_OS: i32 = 0x0e;
pub const _CMP_TRUE_UQ: i32 = 0x0f;
pub const _CMP_EQ_OS: i32 = 0x10;
pub const _CMP_LT_OQ: i32 = 0x11;
pub const _CMP_LE_OQ: i32 = 0x12;
pub const _CMP_UNORD_S: i32 = 0x13;
pub const _CMP_NEQ_US: i32 = 0x14;
pub const _CMP_NLT_UQ: i32 = 0x15;
pub const _CMP_NLE_UQ: i32 = 0x16;
pub const _CMP_ORD_S: i32 = 0x17;
pub const _CMP_EQ_US: i32 = 0x18;
pub const _CMP_NGE_UQ: i32 = 0x19;
pub const _CMP_NGT_UQ: i32 = 0x1a;
pub const _CMP_FALSE_OS: i32 = 0x1b;
pub const _CMP_NEQ_OS: i32 = 0x1c;
pub const _CMP_GE_OQ: i32 = 0x1d;
pub const _CMP_GT_OQ: i32 = 0x1e;
pub const _CMP_TRUE_US: i32 = 0x1f;

// ---------------------------------------------------------------------------------------------
// lane views
// ---------------------------------------------------------------------------------------------

macro_rules! views {
    ($vec:ident: $($get:ident / $put:ident: [$t:ty; $n:literal]),* $(,)?) => {
        $(
            #[allow(dead_code)]
            #[inline(always)]
            fn $get(a: $vec) -> [$t; $n] {
                // SAFETY: plain data of equal size; every bit pattern is valid for both.
                unsafe { transmute_copy(&a) }
            }
            #[allow(dead_code)]
            #[inline(always)]
            fn $put(x: [$t; $n]) -> $vec {
                // SAFETY: as above.
                unsafe { transmute_copy(&x) }
            }
        )*
    };
}

views!(__m128i:
    u8x16 / from_u8x16: [u8; 16],
);
views!(__m512i:
    i8x64 / from_i8x64: [i8; 64],
    u8x64 / from_u8x64: [u8; 64],
    i16x32 / from_i16x32: [i16; 32],
    u16x32 / from_u16x32: [u16; 32],
    i32x16 / from_i32x16: [i32; 16],
    u32x16 / from_u32x16: [u32; 16],
    i64x8 / from_i64x8: [i64; 8],
    u64x8 / from_u64x8: [u64; 8],
);
views!(__m256i:
    i16x16 / from_i16x16: [i16; 16],
    u16x16 / from_u16x16: [u16; 16],
    u8x32 / from_u8x32: [u8; 32],
    i32x8 / from_i32x8: [i32; 8],
    u64x4 / from_u64x4: [u64; 4],
);

#[inline(always)]
fn f64x8(a: __m512d) -> [f64; 8] {
    a.0
}
#[inline(always)]
fn from_f64x8(x: [f64; 8]) -> __m512d {
    __m512d(x)
}

macro_rules! map2 {
    ($get:ident, $put:ident, $a:expr, $b:expr, |$x:ident, $y:ident| $e:expr) => {{
        let a = $get($a);
        let b = $get($b);
        let mut r = a;
        for i in 0..r.len() {
            let ($x, $y) = (a[i], b[i]);
            r[i] = $e;
        }
        $put(r)
    }};
}

macro_rules! map1 {
    ($get:ident, $put:ident, $a:expr, |$x:ident| $e:expr) => {{
        let a = $get($a);
        let mut r = a;
        for i in 0..r.len() {
            let $x = a[i];
            r[i] = $e;
        }
        $put(r)
    }};
}

macro_rules! masked2 {
    ($get:ident, $put:ident, $src:expr, $k:expr, $a:expr, $b:expr, |$x:ident, $y:ident| $e:expr) => {{
        let s = $get($src);
        let a = $get($a);
        let b = $get($b);
        let mut r = s;
        for i in 0..r.len() {
            if ($k >> i) & 1 == 1 {
                let ($x, $y) = (a[i], b[i]);
                r[i] = $e;
            }
        }
        $put(r)
    }};
}

macro_rules! cmp_mask {
    ($get:ident, $m:ty, $a:expr, $b:expr, |$x:ident, $y:ident| $e:expr) => {{
        let a = $get($a);
        let b = $get($b);
        let mut k: $m = 0;
        for i in 0..a.len() {
            let ($x, $y) = (a[i], b[i]);
            if $e {
                k |= 1 << i;
            }
        }
        k
    }};
}

// ---------------------------------------------------------------------------------------------
// loads and stores
// ---------------------------------------------------------------------------------------------

#[inline(always)]
pub unsafe fn _mm512_load_si512(p: *const __m512i) -> __m512i {
    p.read_unaligned()
}
#[inline(always)]
pub unsafe fn _mm512_loadu_si512(p: *const __m512i) -> __m512i {
    p.read_unaligned()
}
#[inline(always)]
pub unsafe fn _mm512_loadu_epi32<T>(p: *const T) -> __m512i {
    (p as *const __m512i).read_unaligned()
}
#[inline(always)]
pub unsafe fn _mm512_store_si512(p: *mut __m512i, a: __m512i) {
    p.write_unaligned(a)
}
#[inline(always)]
pub unsafe fn _mm512_storeu_si512(p: *mut __m512i, a: __m512i) {
    p.write_unaligned(a)
}
#[inline(always)]
pub unsafe fn _mm512_storeu_epi32<T>(p: *mut T, a: __m512i) {
    (p as *mut __m512i).write_unaligned(a)
}
#[inline(always)]
pub unsafe fn _mm512_stream_si512<T>(p: *mut T, a: __m512i) {
    (p as *mut __m512i).write_unaligned(a)
}
#[inline(always)]
pub unsafe fn _mm512_loadu_pd<T>(p: *const T) -> __m512d {
    (p as *const __m512d).read_unaligned()
}
#[inline(always)]
pub unsafe fn _mm256_load_si256<T>(p: *const T) -> __m256i {
    (p as *const __m256i).read_unaligned()
}
#[inline(always)]
pub unsafe fn _mm256_loadu_si256<T>(p: *const T) -> __m256i {
    (p as *const __m256i).read_unaligned()
}
#[inline(always)]
pub unsafe fn _mm256_storeu_si256<T>(p: *mut T, a: __m256i) {
    (p as *mut __m256i).write_unaligned(a)
}

#[inline(always)]
pub unsafe fn _mm512_maskz_loadu_epi8<T>(k: __mmask64, p: *const T) -> __m512i {
    let p = p as *const i8;
    let mut r = [0i8; 64];
    for i in 0..64 {
        if (k >> i) & 1 == 1 {
            r[i] = p.add(i).read_unaligned();
        }
    }
    from_i8x64(r)
}
#[inline(always)]
pub unsafe fn _mm512_maskz_loadu_epi16<T>(k: __mmask32, p: *const T) -> __m512i {
    let p = p as *const i16;
    let mut r = [0i16; 32];
    for i in 0..32 {
        if (k >> i) & 1 == 1 {
            r[i] = p.add(i).read_unaligned();
        }
    }
    from_i16x32(r)
}
#[inline(always)]
pub unsafe fn _mm512_maskz_loadu_epi64<T>(k: __mmask8, p: *const T) -> __m512i {
    _mm512_mask_loadu_epi64(_mm512_setzero_si512(), k, p)
}
#[inline(always)]
pub unsafe fn _mm512_mask_loadu_epi64<T>(src: __m512i, k: __mmask8, p: *const T) -> __m512i {
    let p = p as *const i64;
    let mut r = i64x8(src);
    for i in 0..8 {
        if (k >> i) & 1 == 1 {
            r[i] = p.add(i).read_unaligned();
        }
    }
    from_i64x8(r)
}
#[inline(always)]
pub unsafe fn _mm256_maskz_loadu_epi16<T>(k: __mmask16, p: *const T) -> __m256i {
    let p = p as *const i16;
    let mut r = [0i16; 16];
    for i in 0..16 {
        if (k >> i) & 1 == 1 {
            r[i] = p.add(i).read_unaligned();
        }
    }
    from_i16x16(r)
}
#[inline(always)]
pub unsafe fn _mm512_mask_storeu_epi64<T>(p: *mut T, k: __mmask8, a: __m512i) {
    let p = p as *mut i64;
    let a = i64x8(a);
    for i in 0..8 {
        if (k >> i) & 1 == 1 {
            p.add(i).write_unaligned(a[i]);
        }
    }
}
#[inline(always)]
pub unsafe fn _mm512_mask_storeu_epi32<T>(p: *mut T, k: __mmask16, a: __m512i) {
    let p = p as *mut i32;
    let a = i32x16(a);
    for i in 0..16 {
        if (k >> i) & 1 == 1 {
            p.add(i).write_unaligned(a[i]);
        }
    }
}
#[inline(always)]
pub unsafe fn _mm256_mask_storeu_epi16<T>(p: *mut T, k: __mmask16, a: __m256i) {
    let p = p as *mut i16;
    let a = i16x16(a);
    for i in 0..16 {
        if (k >> i) & 1 == 1 {
            p.add(i).write_unaligned(a[i]);
        }
    }
}
/// `vpmovdw` to memory: each selected 32-bit lane truncated to 16 bits.
#[inline(always)]
pub unsafe fn _mm512_mask_cvtepi32_storeu_epi16<T>(p: *mut T, k: __mmask16, a: __m512i) {
    let p = p as *mut i16;
    let a = i32x16(a);
    for i in 0..16 {
        if (k >> i) & 1 == 1 {
            p.add(i).write_unaligned(a[i] as i16);
        }
    }
}

/// `offsets` are sign-extended 32-bit indices scaled by `SCALE` bytes.
#[inline(always)]
pub unsafe fn _mm512_i32gather_epi64<const SCALE: i32>(
    offsets: __m256i,
    base: *const impl Sized,
) -> __m512i {
    let base = base as *const u8;
    let o = i32x8(offsets);
    let mut r = [0i64; 8];
    for i in 0..8 {
        r[i] = (base.offset(o[i] as isize * SCALE as isize) as *const i64).read_unaligned();
    }
    from_i64x8(r)
}
#[inline(always)]
pub unsafe fn _mm512_i32gather_epi32<const SCALE: i32>(
    offsets: __m512i,
    base: *const impl Sized,
) -> __m512i {
    let base = base as *const u8;
    let o = i32x16(offsets);
    let mut r = [0i32; 16];
    for i in 0..16 {
        r[i] = (base.offset(o[i] as isize * SCALE as isize) as *const i32).read_unaligned();
    }
    from_i32x16(r)
}

/// A cache hint only: on AArch64 the matching `prfm` (T0 to L1, T1 to L2, T2 to L3, NTA as a
/// streaming L1 fill); elsewhere nothing.
#[inline(always)]
pub unsafe fn _mm_prefetch<const STRATEGY: i32>(p: *const i8) {
    #[cfg(target_arch = "aarch64")]
    match STRATEGY {
        _MM_HINT_T0 => core::arch::asm!("prfm pldl1keep, [{0}]", in(reg) p, options(nostack, readonly, preserves_flags)),
        _MM_HINT_T1 => core::arch::asm!("prfm pldl2keep, [{0}]", in(reg) p, options(nostack, readonly, preserves_flags)),
        _MM_HINT_T2 => core::arch::asm!("prfm pldl3keep, [{0}]", in(reg) p, options(nostack, readonly, preserves_flags)),
        _ => core::arch::asm!("prfm pldl1strm, [{0}]", in(reg) p, options(nostack, readonly, preserves_flags)),
    }
    #[cfg(not(target_arch = "aarch64"))]
    let _ = p;
}
#[inline(always)]
pub unsafe fn _mm_sfence() {
    core::sync::atomic::fence(core::sync::atomic::Ordering::SeqCst);
}

// ---------------------------------------------------------------------------------------------
// construction and casts
// ---------------------------------------------------------------------------------------------

#[inline(always)]
pub fn _mm512_setzero_si512() -> __m512i {
    __m512i([0; 8])
}
#[inline(always)]
pub fn _mm256_setzero_si256() -> __m256i {
    __m256i([0; 4])
}
#[inline(always)]
pub fn _mm512_set1_epi8(a: i8) -> __m512i {
    from_i8x64([a; 64])
}
#[inline(always)]
pub fn _mm512_set1_epi16(a: i16) -> __m512i {
    from_i16x32([a; 32])
}
#[inline(always)]
pub fn _mm512_set1_epi32(a: i32) -> __m512i {
    from_i32x16([a; 16])
}
#[inline(always)]
pub fn _mm512_set1_epi64(a: i64) -> __m512i {
    from_i64x8([a; 8])
}
#[inline(always)]
pub fn _mm512_setr_epi64(
    e0: i64,
    e1: i64,
    e2: i64,
    e3: i64,
    e4: i64,
    e5: i64,
    e6: i64,
    e7: i64,
) -> __m512i {
    from_i64x8([e0, e1, e2, e3, e4, e5, e6, e7])
}
#[inline(always)]
pub fn _mm512_setr_epi32(
    e0: i32,
    e1: i32,
    e2: i32,
    e3: i32,
    e4: i32,
    e5: i32,
    e6: i32,
    e7: i32,
    e8: i32,
    e9: i32,
    e10: i32,
    e11: i32,
    e12: i32,
    e13: i32,
    e14: i32,
    e15: i32,
) -> __m512i {
    from_i32x16([e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15])
}
#[inline(always)]
pub fn _mm512_setzero_pd() -> __m512d {
    __m512d([0.0; 8])
}
#[inline(always)]
pub fn _mm512_set1_pd(a: f64) -> __m512d {
    __m512d([a; 8])
}
#[inline(always)]
pub fn _mm_set_epi64x(e1: i64, e0: i64) -> __m128i {
    __m128i([e0 as u64, e1 as u64])
}
#[inline(always)]
pub fn _mm_cvtsi128_si64(a: __m128i) -> i64 {
    a.0[0] as i64
}
#[inline(always)]
pub fn _mm_extract_epi64<const IMM8: i32>(a: __m128i) -> i64 {
    a.0[(IMM8 & 1) as usize] as i64
}

#[inline(always)]
pub fn _mm512_castsi512_si256(a: __m512i) -> __m256i {
    __m256i([a.0[0], a.0[1], a.0[2], a.0[3]])
}
/// The upper half is undefined in the Intel guide; zero here.
#[inline(always)]
pub fn _mm512_castsi256_si512(a: __m256i) -> __m512i {
    __m512i([a.0[0], a.0[1], a.0[2], a.0[3], 0, 0, 0, 0])
}
#[inline(always)]
pub fn _mm512_castsi512_pd(a: __m512i) -> __m512d {
    from_f64x8(u64x8(a).map(f64::from_bits))
}
#[inline(always)]
pub fn _mm512_extracti64x4_epi64<const IMM1: i32>(a: __m512i) -> __m256i {
    let h = 4 * (IMM1 & 1) as usize;
    __m256i([a.0[h], a.0[h + 1], a.0[h + 2], a.0[h + 3]])
}
#[inline(always)]
pub fn _mm512_inserti64x4<const IMM8: i32>(a: __m512i, b: __m256i) -> __m512i {
    let h = 4 * (IMM8 & 1) as usize;
    let mut r = a;
    r.0[h..h + 4].copy_from_slice(&b.0);
    r
}

// ---------------------------------------------------------------------------------------------
// integer arithmetic
// ---------------------------------------------------------------------------------------------

#[inline(always)]
pub fn _mm512_add_epi16(a: __m512i, b: __m512i) -> __m512i {
    map2!(i16x32, from_i16x32, a, b, |x, y| x.wrapping_add(y))
}
#[inline(always)]
pub fn _mm512_sub_epi16(a: __m512i, b: __m512i) -> __m512i {
    map2!(i16x32, from_i16x32, a, b, |x, y| x.wrapping_sub(y))
}
#[inline(always)]
pub fn _mm512_add_epi32(a: __m512i, b: __m512i) -> __m512i {
    map2!(i32x16, from_i32x16, a, b, |x, y| x.wrapping_add(y))
}
#[inline(always)]
pub fn _mm512_sub_epi32(a: __m512i, b: __m512i) -> __m512i {
    map2!(i32x16, from_i32x16, a, b, |x, y| x.wrapping_sub(y))
}
#[inline(always)]
pub fn _mm512_add_epi64(a: __m512i, b: __m512i) -> __m512i {
    map2!(i64x8, from_i64x8, a, b, |x, y| x.wrapping_add(y))
}
#[inline(always)]
pub fn _mm512_mullo_epi16(a: __m512i, b: __m512i) -> __m512i {
    map2!(i16x32, from_i16x32, a, b, |x, y| x.wrapping_mul(y))
}
#[inline(always)]
pub fn _mm512_mulhi_epi16(a: __m512i, b: __m512i) -> __m512i {
    map2!(i16x32, from_i16x32, a, b, |x, y| ((x as i32 * y as i32) >> 16) as i16)
}
#[inline(always)]
pub fn _mm512_mulhi_epu16(a: __m512i, b: __m512i) -> __m512i {
    map2!(u16x32, from_u16x32, a, b, |x, y| ((x as u32 * y as u32) >> 16) as u16)
}
/// `vpmulhrsw`: `((a * b >> 14) + 1) >> 1`, truncated to 16 bits.
#[inline(always)]
pub fn _mm512_mulhrs_epi16(a: __m512i, b: __m512i) -> __m512i {
    map2!(i16x32, from_i16x32, a, b, |x, y| ((((x as i32 * y as i32) >> 14) + 1) >> 1) as i16)
}
#[inline(always)]
pub fn _mm512_mullo_epi32(a: __m512i, b: __m512i) -> __m512i {
    map2!(i32x16, from_i32x16, a, b, |x, y| x.wrapping_mul(y))
}
/// Low 32 bits of each 64-bit lane, unsigned, multiplied to 64 bits.
#[inline(always)]
pub fn _mm512_mul_epu32(a: __m512i, b: __m512i) -> __m512i {
    map2!(u64x8, from_u64x8, a, b, |x, y| (x & 0xffff_ffff) * (y & 0xffff_ffff))
}
#[inline(always)]
pub fn _mm512_madd_epi16(a: __m512i, b: __m512i) -> __m512i {
    // `smull` / `smull2` give the eight word products of each 128-bit chunk, `addp` sums the
    // adjacent pairs: dword `j` = `a[2j] b[2j] + a[2j+1] b[2j+1]`, wrapping, as `vpmaddwd`.
    #[cfg(target_arch = "aarch64")]
    // SAFETY: NEON is part of the aarch64 baseline; the views are 64-byte arrays.
    unsafe {
        use core::arch::aarch64::*;
        let (a, b) = (i16x32(a), i16x32(b));
        let mut r = [0i32; 16];
        for c in 0..4 {
            let x = vld1q_s16(a.as_ptr().add(8 * c));
            let y = vld1q_s16(b.as_ptr().add(8 * c));
            let lo = vmull_s16(vget_low_s16(x), vget_low_s16(y));
            let hi = vmull_high_s16(x, y);
            vst1q_s32(r.as_mut_ptr().add(4 * c), vpaddq_s32(lo, hi));
        }
        return from_i32x16(r);
    }
    #[allow(unreachable_code)]
    let (a, b) = (i16x32(a), i16x32(b));
    let mut r = [0i32; 16];
    for j in 0..16 {
        let lo = a[2 * j] as i32 * b[2 * j] as i32;
        let hi = a[2 * j + 1] as i32 * b[2 * j + 1] as i32;
        r[j] = lo.wrapping_add(hi);
    }
    from_i32x16(r)
}
#[inline(always)]
pub fn _mm512_dpwssd_epi32(src: __m512i, a: __m512i, b: __m512i) -> __m512i {
    _mm512_add_epi32(src, _mm512_madd_epi16(a, b))
}
#[inline(always)]
pub fn _mm512_abs_epi16(a: __m512i) -> __m512i {
    map1!(i16x32, from_i16x32, a, |x| x.wrapping_abs())
}
#[inline(always)]
pub fn _mm512_max_epi16(a: __m512i, b: __m512i) -> __m512i {
    map2!(i16x32, from_i16x32, a, b, |x, y| x.max(y))
}
#[inline(always)]
pub fn _mm512_max_epi32(a: __m512i, b: __m512i) -> __m512i {
    map2!(i32x16, from_i32x16, a, b, |x, y| x.max(y))
}
#[inline(always)]
pub fn _mm512_min_epu16(a: __m512i, b: __m512i) -> __m512i {
    map2!(u16x32, from_u16x32, a, b, |x, y| x.min(y))
}
#[inline(always)]
pub fn _mm512_min_epu32(a: __m512i, b: __m512i) -> __m512i {
    map2!(u32x16, from_u32x16, a, b, |x, y| x.min(y))
}
#[inline(always)]
pub fn _mm512_mask_add_epi16(src: __m512i, k: __mmask32, a: __m512i, b: __m512i) -> __m512i {
    masked2!(i16x32, from_i16x32, src, k, a, b, |x, y| x.wrapping_add(y))
}
#[inline(always)]
pub fn _mm512_mask_sub_epi16(src: __m512i, k: __mmask32, a: __m512i, b: __m512i) -> __m512i {
    masked2!(i16x32, from_i16x32, src, k, a, b, |x, y| x.wrapping_sub(y))
}
#[inline(always)]
pub fn _mm512_mask_add_epi32(src: __m512i, k: __mmask16, a: __m512i, b: __m512i) -> __m512i {
    masked2!(i32x16, from_i32x16, src, k, a, b, |x, y| x.wrapping_add(y))
}
#[inline(always)]
pub fn _mm512_mask_sub_epi32(src: __m512i, k: __mmask16, a: __m512i, b: __m512i) -> __m512i {
    masked2!(i32x16, from_i32x16, src, k, a, b, |x, y| x.wrapping_sub(y))
}
#[inline(always)]
pub fn _mm512_reduce_add_epi64(a: __m512i) -> i64 {
    i64x8(a).iter().fold(0i64, |s, &x| s.wrapping_add(x))
}
#[inline(always)]
pub fn _mm512_reduce_max_epi32(a: __m512i) -> i32 {
    i32x16(a).into_iter().max().unwrap()
}

// ---------------------------------------------------------------------------------------------
// bitwise logic and shifts
// ---------------------------------------------------------------------------------------------

#[inline(always)]
pub fn _mm512_and_si512(a: __m512i, b: __m512i) -> __m512i {
    map2!(u64x8, from_u64x8, a, b, |x, y| x & y)
}
#[inline(always)]
pub fn _mm512_or_si512(a: __m512i, b: __m512i) -> __m512i {
    map2!(u64x8, from_u64x8, a, b, |x, y| x | y)
}
#[inline(always)]
pub fn _mm512_xor_si512(a: __m512i, b: __m512i) -> __m512i {
    map2!(u64x8, from_u64x8, a, b, |x, y| x ^ y)
}
#[inline(always)]
pub fn _mm512_xor_pd(a: __m512d, b: __m512d) -> __m512d {
    let (a, b) = (f64x8(a), f64x8(b));
    let mut r = a;
    for i in 0..8 {
        r[i] = f64::from_bits(a[i].to_bits() ^ b[i].to_bits());
    }
    from_f64x8(r)
}

/// Bit `i` of the result is bit `(a_i << 2) | (b_i << 1) | c_i` of `IMM8`.
#[inline(always)]
fn ternary(imm: i32, a: u64, b: u64, c: u64) -> u64 {
    let mut r = 0;
    for m in 0..8 {
        if (imm >> m) & 1 == 1 {
            let x = if m & 4 != 0 { a } else { !a };
            let y = if m & 2 != 0 { b } else { !b };
            let z = if m & 1 != 0 { c } else { !c };
            r |= x & y & z;
        }
    }
    r
}
#[inline(always)]
pub fn _mm512_ternarylogic_epi64<const IMM8: i32>(a: __m512i, b: __m512i, c: __m512i) -> __m512i {
    let (a, b, c) = (u64x8(a), u64x8(b), u64x8(c));
    let mut r = [0u64; 8];
    for i in 0..8 {
        r[i] = ternary(IMM8, a[i], b[i], c[i]);
    }
    from_u64x8(r)
}
#[inline(always)]
pub fn _mm512_ternarylogic_epi32<const IMM8: i32>(a: __m512i, b: __m512i, c: __m512i) -> __m512i {
    _mm512_ternarylogic_epi64::<IMM8>(a, b, c)
}

#[inline(always)]
pub fn _mm512_srli_epi64<const IMM8: u32>(a: __m512i) -> __m512i {
    map1!(u64x8, from_u64x8, a, |x| if IMM8 >= 64 { 0 } else { x >> IMM8 })
}
#[inline(always)]
pub fn _mm512_slli_epi64<const IMM8: u32>(a: __m512i) -> __m512i {
    map1!(u64x8, from_u64x8, a, |x| if IMM8 >= 64 { 0 } else { x << IMM8 })
}
#[inline(always)]
pub fn _mm512_srli_epi32<const IMM8: u32>(a: __m512i) -> __m512i {
    map1!(u32x16, from_u32x16, a, |x| if IMM8 >= 32 { 0 } else { x >> IMM8 })
}
#[inline(always)]
pub fn _mm512_slli_epi32<const IMM8: u32>(a: __m512i) -> __m512i {
    map1!(u32x16, from_u32x16, a, |x| if IMM8 >= 32 { 0 } else { x << IMM8 })
}
#[inline(always)]
pub fn _mm512_mask_slli_epi32<const IMM8: u32>(src: __m512i, k: __mmask16, a: __m512i) -> __m512i {
    let s = u32x16(src);
    let a = u32x16(a);
    let mut r = s;
    for i in 0..16 {
        if (k >> i) & 1 == 1 {
            r[i] = if IMM8 >= 32 { 0 } else { a[i] << IMM8 };
        }
    }
    from_u32x16(r)
}
#[inline(always)]
pub fn _mm512_srlv_epi32(a: __m512i, count: __m512i) -> __m512i {
    map2!(u32x16, from_u32x16, a, count, |x, n| if n >= 32 { 0 } else { x >> n })
}
#[inline(always)]
pub fn _mm512_sllv_epi32(a: __m512i, count: __m512i) -> __m512i {
    map2!(u32x16, from_u32x16, a, count, |x, n| if n >= 32 { 0 } else { x << n })
}
/// The low 64 bits of `(b:a) >> IMM8`, `b` the high half.
#[inline(always)]
pub fn _mm512_shrdi_epi64<const IMM8: i32>(a: __m512i, b: __m512i) -> __m512i {
    let s = (IMM8 & 63) as u32;
    map2!(u64x8, from_u64x8, a, b, |x, y| if s == 0 { x } else { (x >> s) | (y << (64 - s)) })
}
/// The high 64 bits of `(a:b) << IMM8`, `a` the high half.
#[inline(always)]
pub fn _mm512_shldi_epi64<const IMM8: i32>(a: __m512i, b: __m512i) -> __m512i {
    let s = (IMM8 & 63) as u32;
    map2!(u64x8, from_u64x8, a, b, |x, y| if s == 0 { x } else { (x << s) | (y >> (64 - s)) })
}

// ---------------------------------------------------------------------------------------------
// comparisons into masks
// ---------------------------------------------------------------------------------------------

#[inline(always)]
pub fn _mm512_cmpgt_epi16_mask(a: __m512i, b: __m512i) -> __mmask32 {
    cmp_mask!(i16x32, __mmask32, a, b, |x, y| x > y)
}
#[inline(always)]
pub fn _mm512_cmplt_epi16_mask(a: __m512i, b: __m512i) -> __mmask32 {
    cmp_mask!(i16x32, __mmask32, a, b, |x, y| x < y)
}
#[inline(always)]
pub fn _mm512_cmpgt_epu16_mask(a: __m512i, b: __m512i) -> __mmask32 {
    cmp_mask!(u16x32, __mmask32, a, b, |x, y| x > y)
}
#[inline(always)]
pub fn _mm512_cmpgt_epi32_mask(a: __m512i, b: __m512i) -> __mmask16 {
    cmp_mask!(i32x16, __mmask16, a, b, |x, y| x > y)
}
#[inline(always)]
pub fn _mm512_cmpeq_epi32_mask(a: __m512i, b: __m512i) -> __mmask16 {
    cmp_mask!(i32x16, __mmask16, a, b, |x, y| x == y)
}
#[inline(always)]
pub fn _mm512_cmpgt_epu32_mask(a: __m512i, b: __m512i) -> __mmask16 {
    cmp_mask!(u32x16, __mmask16, a, b, |x, y| x > y)
}
#[inline(always)]
pub fn _mm512_cmplt_epu32_mask(a: __m512i, b: __m512i) -> __mmask16 {
    cmp_mask!(u32x16, __mmask16, a, b, |x, y| x < y)
}
#[inline(always)]
pub fn _mm512_mask_cmpge_epu32_mask(k: __mmask16, a: __m512i, b: __m512i) -> __mmask16 {
    k & cmp_mask!(u32x16, __mmask16, a, b, |x, y| x >= y)
}
#[inline(always)]
pub fn _mm512_test_epi16_mask(a: __m512i, b: __m512i) -> __mmask32 {
    cmp_mask!(u16x32, __mmask32, a, b, |x, y| x & y != 0)
}
#[inline(always)]
pub fn _mm512_test_epi32_mask(a: __m512i, b: __m512i) -> __mmask16 {
    cmp_mask!(u32x16, __mmask16, a, b, |x, y| x & y != 0)
}
#[inline(always)]
pub fn _mm512_movepi32_mask(a: __m512i) -> __mmask16 {
    let a = i32x16(a);
    let mut k = 0;
    for i in 0..16 {
        if a[i] < 0 {
            k |= 1 << i;
        }
    }
    k
}

// ---------------------------------------------------------------------------------------------
// permutes, shuffles, unpacks, width conversions
// ---------------------------------------------------------------------------------------------

/// 128-bit lanes: `a[imm[1:0]], a[imm[3:2]], b[imm[5:4]], b[imm[7:6]]`.
#[inline(always)]
pub fn _mm512_shuffle_i64x2<const MASK: i32>(a: __m512i, b: __m512i) -> __m512i {
    let sel = |src: &[u64; 8], i: i32| {
        let l = 2 * ((MASK >> (2 * i)) & 3) as usize;
        [src[l], src[l + 1]]
    };
    let [a0, a1] = sel(&a.0, 0);
    let [a2, a3] = sel(&a.0, 1);
    let [b0, b1] = sel(&b.0, 2);
    let [b2, b3] = sel(&b.0, 3);
    __m512i([a0, a1, a2, a3, b0, b1, b2, b3])
}

/// `vpermb`: byte `i` of the result is byte `idx[i] & 63` of `a`.
#[inline(always)]
pub fn _mm512_permutexvar_epi8(idx: __m512i, a: __m512i) -> __m512i {
    #[cfg(target_arch = "aarch64")]
    {
        use core::arch::aarch64::*;
        let (ix, t) = (u8x64(idx), u8x64(a));
        let mut r = [0u8; 64];
        // SAFETY: NEON is part of the aarch64 baseline; all pointers address 64-byte arrays.
        unsafe {
            let table = vld1q_u8_x4(t.as_ptr());
            let six = vdupq_n_u8(63);
            for c in 0..4 {
                let i = vandq_u8(vld1q_u8(ix.as_ptr().add(16 * c)), six);
                vst1q_u8(r.as_mut_ptr().add(16 * c), vqtbl4q_u8(table, i));
            }
        }
        from_u8x64(r)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let (ix, t) = (u8x64(idx), u8x64(a));
        from_u8x64(core::array::from_fn(|i| t[(ix[i] & 63) as usize]))
    }
}
#[inline(always)]
pub fn _mm512_permutexvar_epi32(idx: __m512i, a: __m512i) -> __m512i {
    let (ix, t) = (u32x16(idx), u32x16(a));
    from_u32x16(core::array::from_fn(|i| t[(ix[i] & 15) as usize]))
}
#[inline(always)]
pub fn _mm512_mask_permutexvar_epi64(src: __m512i, k: __mmask8, idx: __m512i, a: __m512i) -> __m512i {
    let (s, ix, t) = (u64x8(src), u64x8(idx), u64x8(a));
    from_u64x8(core::array::from_fn(|i| if (k >> i) & 1 == 1 { t[(ix[i] & 7) as usize] } else { s[i] }))
}
#[inline(always)]
pub fn _mm512_permutex2var_epi64(a: __m512i, idx: __m512i, b: __m512i) -> __m512i {
    let (a, ix, b) = (u64x8(a), u64x8(idx), u64x8(b));
    from_u64x8(core::array::from_fn(|i| {
        let j = (ix[i] & 7) as usize;
        if ix[i] & 8 != 0 { b[j] } else { a[j] }
    }))
}
#[inline(always)]
pub fn _mm512_permutex2var_epi32(a: __m512i, idx: __m512i, b: __m512i) -> __m512i {
    let (a, ix, b) = (u32x16(a), u32x16(idx), u32x16(b));
    from_u32x16(core::array::from_fn(|i| {
        let j = (ix[i] & 15) as usize;
        if ix[i] & 16 != 0 { b[j] } else { a[j] }
    }))
}
#[inline(always)]
pub fn _mm512_permutex2var_epi16(a: __m512i, idx: __m512i, b: __m512i) -> __m512i {
    let (a, ix, b) = (u16x32(a), u16x32(idx), u16x32(b));
    from_u16x32(core::array::from_fn(|i| {
        let j = (ix[i] & 31) as usize;
        if ix[i] & 32 != 0 { b[j] } else { a[j] }
    }))
}
#[inline(always)]
pub fn _mm512_permutex2var_epi8(a: __m512i, idx: __m512i, b: __m512i) -> __m512i {
    // `tbl` over `a` for indices below 64 (zero above), then `tbx` over `b` with the index
    // lowered by 64, which leaves the lanes whose lowered index wrapped past 63 untouched.
    #[cfg(target_arch = "aarch64")]
    // SAFETY: NEON is part of the aarch64 baseline; the views are 64-byte arrays.
    unsafe {
        use core::arch::aarch64::*;
        let (ta, ix, tb) = (u8x64(a), u8x64(idx), u8x64(b));
        let mut r = [0u8; 64];
        let (lo, hi) = (vld1q_u8_x4(ta.as_ptr()), vld1q_u8_x4(tb.as_ptr()));
        let (seven, sixty_four) = (vdupq_n_u8(127), vdupq_n_u8(64));
        for c in 0..4 {
            let i = vandq_u8(vld1q_u8(ix.as_ptr().add(16 * c)), seven);
            let x = vqtbl4q_u8(lo, i);
            vst1q_u8(r.as_mut_ptr().add(16 * c), vqtbx4q_u8(x, hi, vsubq_u8(i, sixty_four)));
        }
        return from_u8x64(r);
    }
    #[allow(unreachable_code)]
    let (a, ix, b) = (u8x64(a), u8x64(idx), u8x64(b));
    from_u8x64(core::array::from_fn(|i| {
        let j = (ix[i] & 63) as usize;
        if ix[i] & 64 != 0 { b[j] } else { a[j] }
    }))
}
/// `vpmultishiftqb`: byte `j` of each 64-bit lane of the result is the 8 bits of that lane of
/// `b` starting at bit `a.byte[j] & 63`, wrapping around the lane.
#[inline(always)]
pub fn _mm512_multishift_epi64_epi8(a: __m512i, b: __m512i) -> __m512i {
    // Control `s = 8k + r` reads bits `r..` of byte `k` and the rest from byte `(k + 1) & 7` of
    // the same qword: two `tbl` lookups give that byte pair as a word, a variable `ushl` by
    // `-r` aligns it, and `uzp1` keeps the low bytes.
    #[cfg(target_arch = "aarch64")]
    // SAFETY: NEON is part of the aarch64 baseline; the views are 64-byte arrays.
    unsafe {
        use core::arch::aarch64::*;
        let (ctrl, data) = (u8x64(a), u8x64(b));
        let mut r = [0u8; 64];
        let table = vld1q_u8_x4(data.as_ptr());
        let qbase: [u8; 16] = core::array::from_fn(|p| 8 * (p as u8 / 8));
        let (m63, seven, one) = (vdupq_n_u8(63), vdupq_n_u8(7), vdupq_n_u8(1));
        for c in 0..4 {
            let s = vandq_u8(vld1q_u8(ctrl.as_ptr().add(16 * c)), m63);
            let k = vshrq_n_u8::<3>(s);
            let base = vaddq_u8(vld1q_u8(qbase.as_ptr()), vdupq_n_u8(16 * c as u8));
            let lo = vqtbl4q_u8(table, vaddq_u8(base, k));
            let hi = vqtbl4q_u8(table, vaddq_u8(base, vandq_u8(vaddq_u8(k, one), seven)));
            // `-r` per byte, sign-extended to the word lanes it shifts.
            let neg = vnegq_s8(vreinterpretq_s8_u8(vandq_u8(s, seven)));
            let w0 = vshlq_u16(vreinterpretq_u16_u8(vzip1q_u8(lo, hi)), vmovl_s8(vget_low_s8(neg)));
            let w1 = vshlq_u16(vreinterpretq_u16_u8(vzip2q_u8(lo, hi)), vmovl_high_s8(neg));
            vst1q_u8(r.as_mut_ptr().add(16 * c), vuzp1q_u8(vreinterpretq_u8_u16(w0), vreinterpretq_u8_u16(w1)));
        }
        return from_u8x64(r);
    }
    #[allow(unreachable_code)]
    let (ctrl, data) = (u8x64(a), u64x8(b));
    from_u8x64(core::array::from_fn(|i| {
        data[i / 8].rotate_right((ctrl[i] & 63) as u32) as u8
    }))
}
/// Contiguous elements of `a` to the set positions of `k`, zero elsewhere.
#[inline(always)]
pub fn _mm512_maskz_expand_epi32(k: __mmask16, a: __m512i) -> __m512i {
    let a = u32x16(a);
    let mut r = [0u32; 16];
    let mut m = 0;
    for i in 0..16 {
        if (k >> i) & 1 == 1 {
            r[i] = a[m];
            m += 1;
        }
    }
    from_u32x16(r)
}

macro_rules! unpack {
    ($name:ident, $get:ident, $put:ident, $per:literal, $hi:literal) => {
        /// Interleave the low (or high) halves of each 128-bit lane, `a` first.
        #[inline(always)]
        pub fn $name(a: __m512i, b: __m512i) -> __m512i {
            let (a, b) = ($get(a), $get(b));
            let mut r = a;
            for lane in 0..4 {
                let base = lane * $per;
                let half = if $hi { $per / 2 } else { 0 };
                for j in 0..$per / 2 {
                    r[base + 2 * j] = a[base + half + j];
                    r[base + 2 * j + 1] = b[base + half + j];
                }
            }
            $put(r)
        }
    };
}
unpack!(_mm512_unpacklo_epi8, u8x64, from_u8x64, 16, false);
unpack!(_mm512_unpackhi_epi8, u8x64, from_u8x64, 16, true);
unpack!(_mm512_unpacklo_epi16, u16x32, from_u16x32, 8, false);
unpack!(_mm512_unpackhi_epi16, u16x32, from_u16x32, 8, true);
unpack!(_mm512_unpacklo_epi32, u32x16, from_u32x16, 4, false);
unpack!(_mm512_unpackhi_epi32, u32x16, from_u32x16, 4, true);
unpack!(_mm512_unpacklo_epi64, u64x8, from_u64x8, 2, false);
unpack!(_mm512_unpackhi_epi64, u64x8, from_u64x8, 2, true);

/// `vpmovdw`: truncate each 32-bit lane to 16 bits.
#[inline(always)]
pub fn _mm512_cvtepi32_epi16(a: __m512i) -> __m256i {
    let a = i32x16(a);
    from_i16x16(core::array::from_fn(|i| a[i] as i16))
}
#[inline(always)]
pub fn _mm512_cvtepi32_epi64(a: __m256i) -> __m512i {
    let a = i32x8(a);
    from_i64x8(core::array::from_fn(|i| a[i] as i64))
}
#[inline(always)]
pub fn _mm512_cvtepu16_epi32(a: __m256i) -> __m512i {
    let a = u16x16(a);
    from_i32x16(core::array::from_fn(|i| a[i] as i32))
}
#[inline(always)]
pub fn _mm512_cvtepi16_epi32(a: __m256i) -> __m512i {
    let a = i16x16(a);
    from_i32x16(core::array::from_fn(|i| a[i] as i32))
}

// ---------------------------------------------------------------------------------------------
// GF(2) arithmetic: GFNI affine transform and carry-less multiplication
// ---------------------------------------------------------------------------------------------

/// `vgf2p8affineqb`: bit `i` of each result byte is `parity(x & A.byte[7 - i]) ^ B[i]`, with `A`
/// the 64-bit lane of `a` holding that byte.
#[inline(always)]
pub fn _mm512_gf2p8affine_epi64_epi8<const B: i32>(x: __m512i, a: __m512i) -> __m512i {
    // Bit `i` of result byte `j` is the parity of `A.byte[7 - i] & x.byte[j]` in its qword:
    // per `i`, `tbl` broadcasts row `7 - i` across the qword, `cnt` gives the parities, and
    // `sli` inserts them at bit `i` above the bits already placed.
    #[cfg(target_arch = "aarch64")]
    // SAFETY: NEON is part of the aarch64 baseline; the views are 64-byte arrays.
    unsafe {
        use core::arch::aarch64::*;
        let (xb, ab) = (u8x64(x), u8x64(a));
        let mut r = [0u8; 64];
        let one = vdupq_n_u8(1);
        let bias = vdupq_n_u8(B as u8);
        for c in 0..4 {
            let xv = vld1q_u8(xb.as_ptr().add(16 * c));
            let av = vld1q_u8(ab.as_ptr().add(16 * c));
            let row = |i: u8| -> uint8x16_t {
                let sel: [u8; 16] = core::array::from_fn(|p| 8 * (p as u8 / 8) + 7 - i);
                let m = vandq_u8(vqtbl1q_u8(av, vld1q_u8(sel.as_ptr())), xv);
                vandq_u8(vcntq_u8(m), one)
            };
            let mut acc = row(0);
            acc = vsliq_n_u8::<1>(acc, row(1));
            acc = vsliq_n_u8::<2>(acc, row(2));
            acc = vsliq_n_u8::<3>(acc, row(3));
            acc = vsliq_n_u8::<4>(acc, row(4));
            acc = vsliq_n_u8::<5>(acc, row(5));
            acc = vsliq_n_u8::<6>(acc, row(6));
            acc = vsliq_n_u8::<7>(acc, row(7));
            vst1q_u8(r.as_mut_ptr().add(16 * c), veorq_u8(acc, bias));
        }
        return from_u8x64(r);
    }
    #[allow(unreachable_code)]
    let (xb, aq) = (u8x64(x), u64x8(a));
    let mut r = [0u8; 64];
    for q in 0..8 {
        let m = aq[q].to_le_bytes();
        for j in 0..8 {
            let v = xb[8 * q + j];
            let mut out = 0u8;
            for i in 0..8 {
                out |= (((m[7 - i] & v).count_ones() & 1) as u8) << i;
            }
            r[8 * q + j] = out ^ (B as u8);
        }
    }
    from_u8x64(r)
}

#[inline(always)]
fn clmul64(a: u64, b: u64) -> u128 {
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    {
        // SAFETY: `pmull` is available whenever the `aes` target feature is.
        unsafe { core::arch::aarch64::vmull_p64(a, b) }
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "aes")))]
    {
        let mut r = 0u128;
        for i in 0..64 {
            if (b >> i) & 1 == 1 {
                r ^= (a as u128) << i;
            }
        }
        r
    }
}
/// Per 128-bit lane: the carry-less product of qword `IMM8[0]` of `a` and qword `IMM8[4]` of `b`.
#[inline(always)]
pub fn _mm512_clmulepi64_epi128<const IMM8: i32>(a: __m512i, b: __m512i) -> __m512i {
    let (sa, sb) = ((IMM8 & 1) as usize, ((IMM8 >> 4) & 1) as usize);
    let mut r = [0u64; 8];
    for l in 0..4 {
        let p = clmul64(a.0[2 * l + sa], b.0[2 * l + sb]);
        r[2 * l] = p as u64;
        r[2 * l + 1] = (p >> 64) as u64;
    }
    __m512i(r)
}
#[inline(always)]
pub fn _mm_clmulepi64_si128<const IMM8: i32>(a: __m128i, b: __m128i) -> __m128i {
    let p = clmul64(a.0[(IMM8 & 1) as usize], b.0[((IMM8 >> 4) & 1) as usize]);
    __m128i([p as u64, (p >> 64) as u64])
}

/// BMI2 `pext`: the bits of `a` at the set positions of `mask`, packed toward bit 0.
#[inline(always)]
pub fn _pext_u64(a: u64, mask: u64) -> u64 {
    let (mut m, mut r, mut k) = (mask, 0u64, 0);
    while m != 0 {
        if a & m & m.wrapping_neg() != 0 {
            r |= 1 << k;
        }
        k += 1;
        m &= m - 1;
    }
    r
}

// ---------------------------------------------------------------------------------------------
// double precision
// ---------------------------------------------------------------------------------------------

#[inline(always)]
pub fn _mm512_add_pd(a: __m512d, b: __m512d) -> __m512d {
    map2!(f64x8, from_f64x8, a, b, |x, y| x + y)
}
#[inline(always)]
pub fn _mm512_mul_pd(a: __m512d, b: __m512d) -> __m512d {
    map2!(f64x8, from_f64x8, a, b, |x, y| x * y)
}
/// `maxpd`: `a > b ? a : b`, so `b` on NaN and on equal zeros.
#[inline(always)]
pub fn _mm512_max_pd(a: __m512d, b: __m512d) -> __m512d {
    map2!(f64x8, from_f64x8, a, b, |x, y| if x > y { x } else { y })
}
/// `-(a * b) + c`, fused.
#[inline(always)]
pub fn _mm512_fnmadd_pd(a: __m512d, b: __m512d, c: __m512d) -> __m512d {
    let (a, b, c) = (f64x8(a), f64x8(b), f64x8(c));
    from_f64x8(core::array::from_fn(|i| (-a[i]).mul_add(b[i], c[i])))
}
#[inline(always)]
pub fn _mm512_mask_add_pd(src: __m512d, k: __mmask8, a: __m512d, b: __m512d) -> __m512d {
    masked2!(f64x8, from_f64x8, src, k, a, b, |x, y| x + y)
}
#[inline(always)]
pub fn _mm512_mask_sub_pd(src: __m512d, k: __mmask8, a: __m512d, b: __m512d) -> __m512d {
    masked2!(f64x8, from_f64x8, src, k, a, b, |x, y| x - y)
}
#[inline(always)]
pub fn _mm512_reduce_max_pd(a: __m512d) -> f64 {
    let a = f64x8(a);
    let mut m = a[0];
    for &x in &a[1..] {
        if x > m {
            m = x;
        }
    }
    m
}
#[inline(always)]
pub fn _mm512_cvtepi32_pd(a: __m256i) -> __m512d {
    let a = i32x8(a);
    from_f64x8(core::array::from_fn(|i| a[i] as f64))
}
/// Truncating conversion; NaN and out-of-range lanes give the integer indefinite `i32::MIN`.
#[inline(always)]
pub fn _mm512_cvttpd_epi32(a: __m512d) -> __m256i {
    let a = f64x8(a);
    from_i32x8(core::array::from_fn(|i| {
        let t = a[i].trunc();
        if t.is_nan() || t < i32::MIN as f64 || t > i32::MAX as f64 {
            i32::MIN
        } else {
            t as i32
        }
    }))
}
/// `vrndscalepd`: round to `IMM8[7:4]` fraction bits with the mode in `IMM8[1:0]` (nearest even,
/// down, up, toward zero); `IMM8[2]` (use MXCSR) is taken as nearest even, the default MXCSR.
#[inline(always)]
pub fn _mm512_roundscale_pd<const IMM8: i32>(a: __m512d) -> __m512d {
    let scale = (1u64 << ((IMM8 >> 4) & 15)) as f64;
    let mode = if IMM8 & 4 != 0 { 0 } else { IMM8 & 3 };
    map1!(f64x8, from_f64x8, a, |x| {
        let y = x * scale;
        let r = match mode {
            0 => y.round_ties_even(),
            1 => y.floor(),
            2 => y.ceil(),
            _ => y.trunc(),
        };
        r / scale
    })
}
#[inline(always)]
pub fn _mm512_cmp_pd_mask<const IMM8: i32>(a: __m512d, b: __m512d) -> __mmask8 {
    cmp_mask!(f64x8, __mmask8, a, b, |x, y| {
        let unordered = x.is_nan() || y.is_nan();
        match IMM8 & 15 {
            0 => !unordered && x == y,
            1 => x < y,
            2 => x <= y,
            3 => unordered,
            4 => unordered || x != y,
            5 => !(x < y),
            6 => !(x <= y),
            7 => !unordered,
            8 => unordered || x == y,
            9 => !(x >= y),
            10 => !(x > y),
            11 => false,
            12 => !unordered && x != y,
            13 => x >= y,
            14 => x > y,
            _ => true,
        }
    })
}


// ---------------------------------------------------------------------------------------------
// further integer, mask and float operations
// ---------------------------------------------------------------------------------------------

/// The `_MM_CMPINT_*` predicate `imm` (low three bits) on an ordered pair.
#[inline(always)]
fn cmpint<T: PartialOrd>(x: T, y: T, imm: i32) -> bool {
    match imm & 7 {
        _MM_CMPINT_EQ => x == y,
        _MM_CMPINT_LT => x < y,
        _MM_CMPINT_LE => x <= y,
        _MM_CMPINT_FALSE => false,
        _MM_CMPINT_NE => x != y,
        _MM_CMPINT_NLT => x >= y,
        _MM_CMPINT_NLE => x > y,
        _ => true,
    }
}
#[inline(always)]
pub fn _mm512_cmp_epi32_mask(a: __m512i, b: __m512i, imm: i32) -> __mmask16 {
    cmp_mask!(i32x16, __mmask16, a, b, |x, y| cmpint(x, y, imm))
}
#[inline(always)]
pub fn _mm512_cmp_epu32_mask(a: __m512i, b: __m512i, imm: i32) -> __mmask16 {
    cmp_mask!(u32x16, __mmask16, a, b, |x, y| cmpint(x, y, imm))
}
#[inline(always)]
pub fn _mm512_cmp_epi16_mask(a: __m512i, b: __m512i, imm: i32) -> __mmask32 {
    cmp_mask!(i16x32, __mmask32, a, b, |x, y| cmpint(x, y, imm))
}
#[inline(always)]
pub fn _mm512_cmplt_epi32_mask(a: __m512i, b: __m512i) -> __mmask16 {
    cmp_mask!(i32x16, __mmask16, a, b, |x, y| x < y)
}
#[inline(always)]
pub fn _mm512_cmpneq_epi16_mask(a: __m512i, b: __m512i) -> __mmask32 {
    cmp_mask!(i16x32, __mmask32, a, b, |x, y| x != y)
}
#[inline(always)]
pub fn _mm512_adds_epi16(a: __m512i, b: __m512i) -> __m512i {
    map2!(i16x32, from_i16x32, a, b, |x, y| x.saturating_add(y))
}
#[inline(always)]
pub fn _mm512_mask_or_epi32(src: __m512i, k: __mmask16, a: __m512i, b: __m512i) -> __m512i {
    masked2!(u32x16, from_u32x16, src, k, a, b, |x, y| x | y)
}
#[inline(always)]
pub fn _mm512_srai_epi32<const IMM8: u32>(a: __m512i) -> __m512i {
    map1!(i32x16, from_i32x16, a, |x| x >> IMM8.min(31))
}
#[inline(always)]
pub fn _mm512_srli_epi16<const IMM8: u32>(a: __m512i) -> __m512i {
    map1!(u16x32, from_u16x32, a, |x| if IMM8 > 15 { 0 } else { x >> IMM8 })
}
/// `vpackssdw`: per 128-bit lane, four saturated words of `a` then four of `b`.
#[inline(always)]
pub fn _mm512_packs_epi32(a: __m512i, b: __m512i) -> __m512i {
    let (a, b) = (i32x16(a), i32x16(b));
    let sat = |x: i32| x.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
    let mut r = [0i16; 32];
    for l in 0..4 {
        for i in 0..4 {
            r[8 * l + i] = sat(a[4 * l + i]);
            r[8 * l + 4 + i] = sat(b[4 * l + i]);
        }
    }
    from_i16x32(r)
}
/// `vpshufd`: dword `i` of each 128-bit lane is dword `(MASK >> 2 i) & 3` of that lane.
#[inline(always)]
pub fn _mm512_shuffle_epi32<const MASK: i32>(a: __m512i) -> __m512i {
    let a = u32x16(a);
    let mut r = [0u32; 16];
    for l in 0..4 {
        for i in 0..4 {
            r[4 * l + i] = a[4 * l + ((MASK as usize >> (2 * i)) & 3)];
        }
    }
    from_u32x16(r)
}
#[inline(always)]
pub fn _mm512_permutexvar_epi64(idx: __m512i, a: __m512i) -> __m512i {
    let (idx, a) = (u64x8(idx), u64x8(a));
    from_u64x8(core::array::from_fn(|i| a[(idx[i] & 7) as usize]))
}
#[inline(always)]
pub fn _mm512_permutexvar_epi16(idx: __m512i, a: __m512i) -> __m512i {
    // Word index `k = idx & 31` becomes the byte pair `(2k, 2k + 1)` of a 64-byte `tbl`.
    #[cfg(target_arch = "aarch64")]
    // SAFETY: NEON is part of the aarch64 baseline; the views are 64-byte arrays.
    unsafe {
        use core::arch::aarch64::*;
        let (ix, ta) = (u16x32(idx), u8x64(a));
        let mut r = [0u8; 64];
        let table = vld1q_u8_x4(ta.as_ptr());
        let (mask, pair) = (vdupq_n_u16(31), vdupq_n_u16(0x0100));
        for c in 0..4 {
            let k = vandq_u16(vld1q_u16(ix.as_ptr().add(8 * c)), mask);
            let k2 = vshlq_n_u16::<1>(k);
            let bytes = vreinterpretq_u8_u16(vaddq_u16(vorrq_u16(k2, vshlq_n_u16::<8>(k2)), pair));
            vst1q_u8(r.as_mut_ptr().add(16 * c), vqtbl4q_u8(table, bytes));
        }
        return from_u8x64(r);
    }
    #[allow(unreachable_code)]
    let (idx, a) = (u16x32(idx), u16x32(a));
    from_u16x32(core::array::from_fn(|i| a[(idx[i] & 31) as usize]))
}
#[inline(always)]
pub fn _mm512_broadcast_i32x4(a: __m128i) -> __m512i {
    __m512i([a.0[0], a.0[1], a.0[0], a.0[1], a.0[0], a.0[1], a.0[0], a.0[1]])
}
/// `vpsrldq`: each 128-bit lane shifted right by `IMM8` bytes, zeros shifted in.
#[inline(always)]
pub fn _mm512_bsrli_epi128<const IMM8: i32>(a: __m512i) -> __m512i {
    let a = u8x64(a);
    let s = (IMM8 as u32 & 0xff) as usize;
    from_u8x64(core::array::from_fn(|i| {
        let (l, j) = (i / 16, i % 16);
        if s > 15 || j + s > 15 { 0 } else { a[16 * l + j + s] }
    }))
}
/// `vpslldq`: each 128-bit lane shifted left by `IMM8` bytes, zeros shifted in.
#[inline(always)]
pub fn _mm512_bslli_epi128<const IMM8: i32>(a: __m512i) -> __m512i {
    let a = u8x64(a);
    let s = (IMM8 as u32 & 0xff) as usize;
    from_u8x64(core::array::from_fn(|i| {
        let (l, j) = (i / 16, i % 16);
        if s > 15 || j < s { 0 } else { a[16 * l + j - s] }
    }))
}
#[inline(always)]
pub fn _mm_xor_si128(a: __m128i, b: __m128i) -> __m128i {
    __m128i([a.0[0] ^ b.0[0], a.0[1] ^ b.0[1]])
}
/// `pslldq`: the 128 bits shifted left by `IMM8` bytes.
#[inline(always)]
pub fn _mm_slli_si128<const IMM8: i32>(a: __m128i) -> __m128i {
    let a = u8x16(a);
    let s = (IMM8 as u32 & 0xff) as usize;
    from_u8x16(core::array::from_fn(|j| if s > 15 || j < s { 0 } else { a[j - s] }))
}
#[inline(always)]
pub fn _mm256_and_si256(a: __m256i, b: __m256i) -> __m256i {
    __m256i(core::array::from_fn(|i| a.0[i] & b.0[i]))
}
#[inline(always)]
pub fn _mm256_add_epi16(a: __m256i, b: __m256i) -> __m256i {
    let (a, b) = (i16x16(a), i16x16(b));
    from_i16x16(core::array::from_fn(|i| a[i].wrapping_add(b[i])))
}
#[inline(always)]
pub fn _mm256_set1_epi16(a: i16) -> __m256i {
    from_i16x16([a; 16])
}
#[inline(always)]
pub unsafe fn _mm512_maskz_loadu_epi32(k: __mmask16, p: *const i32) -> __m512i {
    let mut r = [0i32; 16];
    for i in 0..16 {
        if (k >> i) & 1 == 1 {
            r[i] = p.add(i).read_unaligned();
        }
    }
    from_i32x16(r)
}
#[inline(always)]
pub unsafe fn _mm512_mask_storeu_epi16(p: *mut i16, k: __mmask32, a: __m512i) {
    let a = i16x32(a);
    for i in 0..32 {
        if (k >> i) & 1 == 1 {
            p.add(i).write_unaligned(a[i]);
        }
    }
}
#[inline(always)]
pub fn _mm512_set1_ps(a: f32) -> __m512 {
    __m512([a; 16])
}
#[inline(always)]
pub fn _mm512_mul_ps(a: __m512, b: __m512) -> __m512 {
    __m512(core::array::from_fn(|i| a.0[i] * b.0[i]))
}
/// `vcvtdq2ps`: round to nearest even, as under the default MXCSR.
#[inline(always)]
pub fn _mm512_cvtepi32_ps(a: __m512i) -> __m512 {
    let a = i32x16(a);
    __m512(core::array::from_fn(|i| a[i] as f32))
}
/// `vcvtps2dq` with an explicit rounding: NaN and values outside `i32` give `0x80000000`.
#[inline(always)]
pub fn _mm512_cvt_roundps_epi32<const ROUNDING: i32>(a: __m512) -> __m512i {
    from_i32x16(core::array::from_fn(|i| {
        let x = a.0[i];
        let r = match ROUNDING & 7 {
            _MM_FROUND_TO_NEG_INF => x.floor(),
            _MM_FROUND_TO_POS_INF => x.ceil(),
            _MM_FROUND_TO_ZERO => x.trunc(),
            _ => x.round_ties_even(),
        };
        if r.is_nan() || r < -2147483648.0 || r >= 2147483648.0 { i32::MIN } else { r as i32 }
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clmul_matches_the_bit_loop() {
        let mut s = 0x9e37_79b9_7f4a_7c15u64;
        for _ in 0..1000 {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let a = s;
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let b = s;
            let mut want = 0u128;
            for i in 0..64 {
                if (b >> i) & 1 == 1 {
                    want ^= (a as u128) << i;
                }
            }
            assert_eq!(clmul64(a, b), want);
        }
    }

    #[test]
    fn permutexvar_epi8_wraps_the_index() {
        let t: [u8; 64] = core::array::from_fn(|i| (i as u8).wrapping_mul(37) ^ 0x5a);
        let ix: [u8; 64] = core::array::from_fn(|i| (i as u8).wrapping_mul(201).wrapping_add(7));
        let r = u8x64(_mm512_permutexvar_epi8(from_u8x64(ix), from_u8x64(t)));
        for i in 0..64 {
            assert_eq!(r[i], t[(ix[i] & 63) as usize]);
        }
    }

    #[test]
    fn shuffle_and_unpack_lanes() {
        let a = from_u64x8([0, 1, 2, 3, 4, 5, 6, 7]);
        let b = from_u64x8([10, 11, 12, 13, 14, 15, 16, 17]);
        assert_eq!(u64x8(_mm512_shuffle_i64x2::<0x88>(a, b)), [0, 1, 4, 5, 10, 11, 14, 15]);
        assert_eq!(u64x8(_mm512_shuffle_i64x2::<0xDD>(a, b)), [2, 3, 6, 7, 12, 13, 16, 17]);
        assert_eq!(u64x8(_mm512_unpacklo_epi64(a, b)), [0, 10, 2, 12, 4, 14, 6, 16]);
        assert_eq!(u64x8(_mm512_unpackhi_epi64(a, b)), [1, 11, 3, 13, 5, 15, 7, 17]);
        let a = from_u32x16(core::array::from_fn(|i| i as u32));
        let b = from_u32x16(core::array::from_fn(|i| 100 + i as u32));
        assert_eq!(
            u32x16(_mm512_unpacklo_epi32(a, b))[..8],
            [0, 100, 1, 101, 4, 104, 5, 105]
        );
    }

    #[test]
    fn ternarylogic_xor3_and_majority_or() {
        let (a, b, c) = (0xF0F0u64, 0xCCCCu64, 0xAAAAu64);
        assert_eq!(ternary(0x96, a, b, c), a ^ b ^ c);
        assert_eq!(ternary(0xEA, a, b, c), (a & b) | c);
    }

    #[test]
    fn funnel_shifts() {
        let a = from_u64x8([0x0123_4567_89ab_cdef; 8]);
        let b = from_u64x8([0xfedc_ba98_7654_3210; 8]);
        let x = 0x0123_4567_89ab_cdefu64;
        let y = 0xfedc_ba98_7654_3210u64;
        assert_eq!(u64x8(_mm512_shrdi_epi64::<34>(a, b))[0], (x >> 34) | (y << 30));
        assert_eq!(u64x8(_mm512_shldi_epi64::<17>(a, b))[0], (x << 17) | (y >> 47));
    }

    #[test]
    fn mulhrs_edge() {
        let a = _mm512_set1_epi16(i16::MIN);
        assert_eq!(i16x32(_mm512_mulhrs_epi16(a, a))[0], i16::MIN);
        let b = _mm512_set1_epi16(16384);
        assert_eq!(i16x32(_mm512_mulhrs_epi16(_mm512_set1_epi16(3), b))[0], 2);
    }

    #[test]
    fn gf2p8affine_identity_and_transpose() {
        // The identity matrix (byte 7 - i selects bit i) leaves every byte unchanged.
        let id = u64::from_le_bytes(core::array::from_fn(|k| 1u8 << (7 - k)));
        let x: [u8; 64] = core::array::from_fn(|i| (i as u8).wrapping_mul(29));
        let r = u8x64(_mm512_gf2p8affine_epi64_epi8::<0>(from_u8x64(x), from_u64x8([id; 8])));
        assert_eq!(r, x);
    }
}
