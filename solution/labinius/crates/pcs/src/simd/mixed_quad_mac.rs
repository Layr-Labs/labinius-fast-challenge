//! Direct four-product quadratic MAC with two output accumulators.
//!
//! Public forms are k0,k1,center(c_j*k1). The first form is reused by
//! both output channels. Key-image dimensions and caller API match C9.
use crate::params::barrett_v;
use crate::ring::{Batch32,Representation};
use crate::simd::ntt::{bin_quad,gen_quad};
use core::arch::x86_64::*;
const Q:u16=4861;
pub const LEAVES:usize=324;
pub const KEY_WORDS:usize=LEAVES*3*4;
pub const W_BOUND:i64=bin_quad::output_bound(Q) as i64;
const A_BOUND:i64=((Q-1)/2) as i64;
const R16C:i32=(65536%Q as i32)-Q as i32;
const AFTER_REDUCE:i64=32768*(1+R16C.unsigned_abs() as i64);
const BIAS:i64=((1i64<<31)/Q as i64)*Q as i64;
/// Each pair batch contributes two products per ring, two rings per column.
const PER_PAIR:i64=4*W_BOUND*A_BOUND;
const fn period()->usize {let mut p=1;while AFTER_REDUCE+2*p as i64*PER_PAIR<=BIAS {p*=2;}p}
pub const OUTPUT_PERIOD:usize=period();
const _:()={
 assert!(W_BOUND<=i16::MAX as i64);
 assert!(AFTER_REDUCE+OUTPUT_PERIOD as i64*PER_PAIR<=BIAS);
 assert!(BIAS<=i32::MAX as i64);
 assert!(OUTPUT_PERIOD==8 && OUTPUT_PERIOD%2==0 && 128%OUTPUT_PERIOD==0);
 assert!(2*(Q as i32)<i16::MAX as i32);
 assert!(A_BOUND<=gen_quad::in_bound(Q) as i64);
 assert!(barrett_v(Q)==7 && 1259*2+(Q as i32)<2*(Q as i32));
};
/// [leaf][constant/linear output][ordinary/auxiliary stream][16columns].
#[repr(C,align(128))]
pub struct QuadPairAcc {pub v:[[[[i32;16];2];2];LEAVES]}
const _:()=assert!(core::mem::size_of::<QuadPairAcc>()==LEAVES*2*2*64);
impl QuadPairAcc {
 pub fn zero()->Box<Self>{let mut p=Box::<Self>::new_uninit();unsafe{core::ptr::write_bytes(p.as_mut_ptr() as *mut u8,0,core::mem::size_of::<Self>());p.assume_init()}}
}

/// x = signed_low + 2^16*(signed_high + carry(bit15)); replace
/// 2^16 by the congruent centered residue R16C.
#[inline(always)]
unsafe fn reduce_c(x: __m512i) -> __m512i {
    let word = (1u32 | ((R16C as u16 as u32) << 16)) as i32;
    let y = _mm512_madd_epi16(x, _mm512_set1_epi32(word));
    let carry = _mm512_test_epi32_mask(x, _mm512_set1_epi32(0x8000));
    _mm512_mask_add_epi32(y, carry, y, _mm512_set1_epi32(R16C))
}

#[inline(always)]
unsafe fn dp<const OFF: usize>(mut a: __m512i, w: __m512i, key: *const u32) -> __m512i {
    core::arch::asm!(
        "vpdpwssd {a}, {w}, dword ptr [{key} + {off}]{{1to16}}",
        a = inout(zmm_reg) a,
        w = in(zmm_reg) w,
        key = in(reg) key,
        off = const OFF,
        options(pure, readonly, nostack, preserves_flags)
    );
    a
}


#[inline(never)]
#[target_feature(enable="avx512f,avx512bw,avx512vnni")]
unsafe fn accumulate<const RED:bool>(w0:&Batch32,w1:&Batch32,key:*const u32,acc:&mut QuadPairAcc){
 for j in 0..LEAVES {
  let a0=_mm512_load_si512(w0.v[2*j].as_ptr() as *const __m512i);
  let a1=_mm512_load_si512(w0.v[2*j+1].as_ptr() as *const __m512i);
  let b0=_mm512_load_si512(w1.v[2*j].as_ptr() as *const __m512i);
  let b1=_mm512_load_si512(w1.v[2*j+1].as_ptr() as *const __m512i);
  let p0=acc.v[j][0].as_mut_ptr() as *mut __m512i;
  let p1=acc.v[j][1].as_mut_ptr() as *mut __m512i;
  let mut y0a=_mm512_load_si512(p0);let mut y0t=_mm512_load_si512(p0.add(1));
  let mut y1a=_mm512_load_si512(p1);let mut y1t=_mm512_load_si512(p1.add(1));
  let k0=key.add(12*j);let k1=k0.add(4);let kc=k0.add(8);
  y0a=dp::<0>(y0a,a0,k0); y0t=dp::<4>(y0t,a0,k0);
  y0a=dp::<0>(y0a,a1,kc); y0t=dp::<4>(y0t,a1,kc);
  y1a=dp::<0>(y1a,a0,k1); y1t=dp::<4>(y1t,a0,k1);
  y1a=dp::<0>(y1a,a1,k0); y1t=dp::<4>(y1t,a1,k0);
  y0a=dp::<8>(y0a,b0,k0); y0t=dp::<12>(y0t,b0,k0);
  y0a=dp::<8>(y0a,b1,kc); y0t=dp::<12>(y0t,b1,kc);
  y1a=dp::<8>(y1a,b0,k1); y1t=dp::<12>(y1t,b0,k1);
  y1a=dp::<8>(y1a,b1,k0); y1t=dp::<12>(y1t,b1,k0);
  if RED {y0a=reduce_c(y0a);y0t=reduce_c(y0t);y1a=reduce_c(y1a);y1t=reduce_c(y1t);}
  _mm512_store_si512(p0,y0a);_mm512_store_si512(p0.add(1),y0t);
  _mm512_store_si512(p1,y1a);_mm512_store_si512(p1.add(1),y1t);
 }
}

/// Accumulate two consecutive pair batches using four products per leaf.
///
/// # Safety
/// AVX512F/BW/VNNI. Batches are4861quadraticNTT output with |lane|<=W_BOUND.
/// Key covers KEY_WORDS nonaliasing dwords, forms k0,k1,center(c_j*k1),
/// every signed16bit component bounded by2430. Start with zero accumulators,
/// advance b_done by two, and preserve every OUTPUT_PERIOD fold boundary.
#[target_feature(enable="avx512f,avx512bw,avx512vnni")]
pub unsafe fn accumulate_lock2(w0:&Batch32,w1:&Batch32,key:*const u32,acc:&mut QuadPairAcc,b_done:usize){
 debug_assert!(b_done>0&&b_done%2==0);
 debug_assert_eq!(w0.representation,Representation::Ntt);debug_assert_eq!(w1.representation,Representation::Ntt);
 if b_done%OUTPUT_PERIOD==0 {accumulate::<true>(w0,w1,key,acc);}else{accumulate::<false>(w0,w1,key,acc);}
}

const fn qinv16() -> u32 {
    let mut x = Q as u32;
    let mut i = 0;
    while i < 4 { x = x.wrapping_mul(2u32.wrapping_sub(Q as u32 * x)) & 0xffff; i += 1; }
    x
}

/// The existing hybrid red16 construction: for |x|<=BIAS, result is
/// congruent to x and has absolute value strictly below 2Q.
#[inline(always)]
unsafe fn red16(x: __m512i) -> __m512i {
    const WL: u32 = (1u32 << 16) % Q as u32;
    const WH: u32 = ((1u64 << 32) % Q as u64) as u32;
    const W: i32 = (WL | (WH << 16)) as i32;
    const WP: i32 = (((WL * qinv16()) & 0xffff) | (((WH * qinv16()) & 0xffff) << 16)) as i32;
    let y = _mm512_add_epi32(x, _mm512_set1_epi32(BIAS as i32));
    let m = _mm512_mullo_epi16(y, _mm512_set1_epi32(WP));
    let r = _mm512_sub_epi16(
        _mm512_mulhi_epu16(y, _mm512_set1_epi32(W)),
        _mm512_mulhi_epu16(m, _mm512_set1_epi16(Q as i16)),
    );
    _mm512_madd_epi16(r, _mm512_set1_epi16(1))
}

#[inline(always)]
unsafe fn center16(x: __m256i) -> __m256i {
    let q = _mm256_set1_epi16(Q as i16);
    let t = _mm256_mulhrs_epi16(x, _mm256_set1_epi16(barrett_v(Q)));
    let r = _mm256_sub_epi16(x, _mm256_mullo_epi16(t, q));
    let hi = _mm256_cmpgt_epi16_mask(r, _mm256_set1_epi16((Q / 2) as i16));
    let r = _mm256_mask_sub_epi16(r, hi, r, q);
    let lo = _mm256_cmplt_epi16_mask(r, _mm256_set1_epi16(-((Q / 2) as i16)));
    _mm256_mask_add_epi16(r, lo, r, q)
}


/// Reduce and clear one two-output stream, retaining the other stream and
/// the other16-lane output half. Both outputs are centered for gen_quad.
///
/// # Safety
/// AVX512F/BW/VL. All accumulator lanes obey the declared BIAS bound.
#[target_feature(enable="avx512f,avx512bw,avx512vl")]
pub unsafe fn finish_stream(acc:&mut QuadPairAcc,stream:usize,out:&mut Batch32,half:usize){
 assert!(stream<2&&half<2);let zero=_mm512_setzero_si512();
 for j in 0..LEAVES {
  let p0=acc.v[j][0][stream].as_mut_ptr() as *mut __m512i;
  let p1=acc.v[j][1][stream].as_mut_ptr() as *mut __m512i;
  let y0=center16(_mm512_cvtepi32_epi16(red16(_mm512_load_si512(p0))));
  let y1=center16(_mm512_cvtepi32_epi16(red16(_mm512_load_si512(p1))));
  _mm512_store_si512(p0,zero);_mm512_store_si512(p1,zero);
  _mm256_storeu_si256(out.v[2*j].as_mut_ptr().add(16*half) as *mut __m256i,y0);
  _mm256_storeu_si256(out.v[2*j+1].as_mut_ptr().add(16*half) as *mut __m256i,y1);
 }
 out.representation=Representation::Ntt;
}
