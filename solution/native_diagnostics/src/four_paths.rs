// Temporary four-path NTT check. No production caller.
#![allow(dead_code)]
pub use labinius::simd::transpose_f162::BinaryIndex32;
macro_rules! r3_twiddles {
    ($n:literal, $pair:path, $zetas:expr) => {{
        let mut t = [[0u32; 4]; $n];
        let mut i = 0;
        while i < $n {
            t[i] = $pair($zetas[i]);
            i += 1;
        }
        t
    }};
}
#[path="baseline.rs"] pub mod baseline;
#[path="omega.rs"] pub mod omega;
#[path="lut.rs"] pub mod lut;
#[path="both.rs"] pub mod both;

// PATH: 0=current, 1=omega register, 2=level6 LUT, 3=both.
#[inline(always)]
pub unsafe fn pair<const PATH:usize,const Q:u16>(ins:[&BinaryIndex32;2],key:*const u32,acc:*mut i32,red:bool) {
 match PATH {
  0=>baseline::ntt_bin_batch32_pair::<Q>(ins,key,acc,red),
  1=>omega::ntt_bin_batch32_pair::<Q>(ins,key,acc,red),
  2=>lut::ntt_bin_batch32_pair::<Q>(ins,key,acc,red),
  3=>both::ntt_bin_batch32_pair::<Q>(ins,key,acc,red),
  _=>unreachable!("four fixed paths"),
 }
}
