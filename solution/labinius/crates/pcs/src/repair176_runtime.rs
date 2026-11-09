//! Exact recovery of rare ambiguous single-transform commitment coefficients.
//!
//! Let B cover the centered integer coefficient. For |v| < P-B, no other
//! representative v+kP lies in [-B,B]. Residues outside that interval are
//! recomputed from the current input, then corrected modulo the tail modulus.
//! B is certified from public second moments at 7.75 sigma; this is the same
//! independent Bernoulli model used by the original statistical centering.
//! It is not a guarantee for arbitrary adversarial Boolean tables.
#![allow(dead_code)]
use std::arch::x86_64::*;
const N:usize=648;const D:usize=6;const P:i64=9721;const M:i64=55*65536;
#[derive(Clone,Copy)]struct Term{base:usize,mask:u64,perm:__m512i,negative:bool}
#[derive(Clone,Copy)]struct Segment{terms:[Term;2],n:usize}
unsafe fn plans(m:usize)->[Segment;8]{
 assert!(m<N);let empty=Term{base:0,mask:0,perm:_mm512_setzero_si512(),negative:false};
 core::array::from_fn(|seg|{let mut p=Segment{terms:[empty;2],n:0};let offsets:&[(i32,bool)]=if m<324{&[(0,false),(648,true),(972,true)]}else{&[(0,false),(324,false)]};
  for &(offset,negative)in offsets{let start=m as i32-64*seg as i32+offset;if start<0||start-63>=N as i32{continue;}
   let(base,mask)=if start>=63&&start<N as i32{((start-63)as usize,u64::MAX)}else if start<63{(0,(1u64<<(start+1))-1)}else{(N-64,!((1u64<<(start-(N as i32-1)))-1))};
   let index:[u8;64]=core::array::from_fn(|i|((start-base as i32-i as i32)&63)as u8);assert!(p.n<2);p.terms[p.n]=Term{base,mask,perm:core::mem::transmute(index),negative};p.n+=1;
  }assert!(p.n>=1&&p.n<=2);p})
}
#[inline]unsafe fn load_term(poly:*const i8,t:Term)->__m512i{let x=_mm512_loadu_si512(poly.add(t.base).cast());let x=_mm512_maskz_mov_epi8(t.mask,_mm512_permutexvar_epi8(t.perm,x));if t.negative{_mm512_sub_epi8(_mm512_setzero_si512(),x)}else{x}}
#[inline]unsafe fn word_mask(words:*const u64,e:usize,seg:usize)->u64{
 let q=seg/4;let shift=(seg%4)*16;let p=words.add(8*e+q);let a=p.read_unaligned()>>shift;let b=p.add(2).read_unaligned()>>shift;let c=p.add(4).read_unaligned()>>shift;let d=p.add(6).read_unaligned()>>shift;let mask=0x1111111111111111u64;
 _pdep_u64(a,mask)|(_pdep_u64(b,mask)<<1)|(_pdep_u64(c,mask)<<2)|(_pdep_u64(d,mask)<<3)
}
#[inline]unsafe fn segment<const TWO:bool>(bank:*const i8,words:*const u64,first:usize,last:usize,seg:usize,p:Segment)->__m512i{
 let mut acc=_mm512_setzero_si512();for e in first..last{let poly=bank.add(e*N);let mut coefficient=load_term(poly,p.terms[0]);if TWO{coefficient=_mm512_add_epi8(coefficient,load_term(poly,p.terms[1]));}let mask=word_mask(words,e,seg);let bits=_mm512_maskz_set1_epi8(mask,1);acc=_mm512_dpbusd_epi32(acc,bits,coefficient);}acc
}
#[inline(never)]#[target_feature(enable="avx512f,avx512bw,avx512vbmi,avx512vnni,bmi2")]
pub unsafe fn dot(raw:&[i8],elements:usize,d:usize,m:usize,first:usize,last:usize,words:*const u64)->i64{
 assert_eq!(raw.len(),D*elements*N);assert!(d<D&&m<N&&first<=last&&last<=elements);let plan=plans(m);let mut sum=_mm512_setzero_si512();let bank=raw.as_ptr().add(d*elements*N);
 for seg in 0..8{let p=plan[seg];let x=if p.n==1{segment::<false>(bank,words,first,last,seg,p)}else{segment::<true>(bank,words,first,last,seg,p)};sum=_mm512_add_epi32(sum,x);}
 let lanes:[i32;16]=core::mem::transmute(sum);lanes.iter().map(|&x|x as i64).sum()
}

#[derive(Clone,Copy)]pub struct Key<'a>{pub digits:&'a[i8],pub means:&'a[i32],pub limits:&'a[i16]}
pub struct Public{pub digits:Vec<i8>,pub means:Vec<i32>,pub limits:Vec<i16>,pub margin:f64}
impl Public{pub fn key(&self)->Key<'_>{Key{digits:&self.digits,means:&self.means,limits:&self.limits}}}
pub fn recovery_bound(s2:i64)->u64{assert!(s2>=0);let n=(s2 as u128*961+63)/64;let mut x=(n as f64).sqrt()as u128;while x*x<n{x+=1;}while x>0&&(x-1)*(x-1)>=n{x-=1;}x as u64+1}
#[cfg(test)]pub static FLAGS:std::sync::atomic::AtomicU64=std::sync::atomic::AtomicU64::new(0);
#[cfg(test)]pub static DELTAS:std::sync::atomic::AtomicU64=std::sync::atomic::AtomicU64::new(0);
#[cfg(test)]pub fn reset(){use std::sync::atomic::Ordering::Relaxed;FLAGS.store(0,Relaxed);DELTAS.store(0,Relaxed);}
#[cfg(test)]pub fn counts()->(u64,u64){use std::sync::atomic::Ordering::Relaxed;(FLAGS.load(Relaxed),DELTAS.load(Relaxed))}
