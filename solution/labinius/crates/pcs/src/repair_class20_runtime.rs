//! Compact public parity in four bit classes, interleaved by 64-bit word index.
//! This preserves the existing guarded parity window and its explicit failure budget.
use std::arch::x86_64::*;
const N:usize=648;const D:usize=6;const LENGTH:usize=1159;
pub const STRIDE:usize=20;pub const MIN_CUT:i16=1024;
fn at(poly:&[i8],j:i32)->i16{if(0..N as i32).contains(&j){poly[j as usize]as i16}else{0}}
fn expanded(poly:&[i8],j:usize)->i16{let t=323-j as i32;at(poly,t)-at(poly,t+648)-at(poly,t+972)}
pub struct ParityData{pub elements:usize,pub words:Vec<u64>}
#[derive(Clone,Copy)]pub struct ParityKey<'a>{pub elements:usize,pub words:&'a[u64]}
impl ParityData{
 pub fn build(raw:&[i8],elements:usize)->Self{assert!(elements>0);assert_eq!(raw.len(),D*elements*N);let mut words=vec![0u64;D*elements*STRIDE];for(pe,poly)in raw.chunks_exact(N).enumerate(){for j in 0..LENGTH{let bit=j/4;words[pe*STRIDE+4*(bit/64)+j%4]|=((expanded(poly,j)&1)as u64)<<(bit%64);}}Self{elements,words}}
 pub fn key(&self)->ParityKey<'_>{ParityKey{elements:self.elements,words:&self.words}}
}
impl ParityKey<'_>{
 #[inline(never)]#[target_feature(enable="avx512f,avx512vbmi2,popcnt")]
 pub unsafe fn odd(&self,d:usize,m:usize,first:usize,last:usize,input:*const u64)->u8{
  assert!(d<D&&m<N&&first<=last&&last<=self.elements);assert_eq!(self.words.len(),D*self.elements*STRIDE);
  let start=if m<324{323-m}else{971-m};let r=start%4;let q=start/4;let word=4*(q/64);let t=q%64;
  let mask=(((1u16<<r)-1)as u8)|((((1u16<<r)-1)as u8)<<4);
  let base=_mm512_setr_epi64(0,4,1,5,2,6,3,7);let permutation=_mm512_or_si512(_mm512_and_si512(_mm512_add_epi64(base,_mm512_set1_epi64(r as i64)),_mm512_set1_epi64(3)),_mm512_and_si512(base,_mm512_set1_epi64(4)));
  let shifts=_mm512_and_si512(_mm512_mask_add_epi64(_mm512_set1_epi64(t as i64),mask,_mm512_set1_epi64(t as i64),_mm512_set1_epi64(1)),_mm512_set1_epi64(63));
  let bank=self.words.as_ptr().add(d*self.elements*STRIDE+word);
  macro_rules! sweep{($carry:expr)=>{{let mut sum=_mm512_setzero_si512();for e in first..last{
   if e+8<last{let f=bank.add((e+8)*STRIDE);_mm_prefetch(f.cast(),_MM_HINT_T1);_mm_prefetch(f.add(8).cast(),_MM_HINT_T1);}
   let p=bank.add(e*STRIDE);let mut lo=_mm512_loadu_si512(p.cast());let mut hi=_mm512_loadu_si512(p.add(4).cast());
   if $carry{let third=_mm512_loadu_si512(p.add(8).cast());lo=_mm512_mask_blend_epi64(mask,lo,hi);hi=_mm512_mask_blend_epi64(mask,hi,third);}
   let coefficients=_mm512_permutexvar_epi64(permutation,_mm512_shrdv_epi64(lo,hi,shifts));let bits=_mm512_loadu_si512(input.add(8*e).cast());sum=_mm512_xor_si512(sum,_mm512_and_si512(bits,coefficients));
  }sum}};}
  let sum=if t==63&&r!=0{assert!(word+16<=STRIDE);sweep!(true)}else{assert!(word+12<=STRIDE);sweep!(false)};
  let lanes:[u64;8]=core::mem::transmute(sum);(lanes.into_iter().fold(0u64,|a,b|a^b).count_ones()&1)as u8
 }
}
#[cfg(test)]mod proof_tests{use super::*;
#[test]fn parity_window_skip_is_exact_inside_certified_interval(){let p=9721i64;for bound in[1i64,1024,4860,5860,6129,8697,9720]{let threshold=p-bound;for residue in -4860i64..=4860{for k in -3i64..=3{let actual=residue+k*p;if actual.abs()>bound{continue;}if residue.abs()<threshold{assert_eq!(actual,residue);}else{assert!((-1..=1).contains(&k));for mean in[-1i64,0,1]{let raw_parity=((actual+mean)&1)as u8;let parity_after_mean=raw_parity^((mean&1)as u8);if parity_after_mean==(residue&1)as u8{assert_eq!(actual,residue);}else{assert_ne!(k,0);}}}}}}}
#[test]fn parity_additional_even_wrap_budget(){
 // A false matching parity requires a nonzero even multiple of odd P.
 // With centered residue <=4860, |actual|>=2P-4860=14582.
 // E[actual] is zero or 1/2. A >=7.75-sigma certified B with cut>=1024
 // gives sigma² <=16*B²/961, B<=8697. Hoeffding exponent >=84.
 let p=9721u128;assert_eq!(MIN_CUT,1024);let b=p-MIN_CUT as u128;let twice_deviation=2*(2*p-p/2)-1;assert_eq!(twice_deviation,29163);let numerator=twice_deviation*twice_deviation*961;let denominator=128*b*b;assert!(numerator>=84*denominator);
 // At fixed shape, even allowing one reconstruction chunk per pair,
 // number of coefficients is D*648*(2^24 packed inputs)/(4*2) <2^34.
 let max_coefficients=D as u128*N as u128*(1u128<<21);assert!(max_coefficients<1u128<<34);
 // exp(.7) > 1+.7+.7²/2+.7³/6+.7⁴/24 =482921/240000 >2,
 // so ln2<.7 and2^35 exp(-84)<2^-85<2^-80.
 assert!(482921u128>2*240000);assert!(84*10>=(35+85)*7);
}

}

#[cfg(test)]mod layout_guards{use super::*;fn rng(s:&mut u64)->u64{*s^=*s<<13;*s^=*s>>7;*s^=*s<<17;*s}fn row(l:&[i8],m:usize,i:usize)->i64{let t=m as i32-i as i32;if m<324{(at(l,t)-at(l,t+648)-at(l,t+972))as i64}else{(at(l,t)+at(l,t+324))as i64}}
#[test]fn every_coefficient_and_input_bit_matches_integer_ring(){let elements=2;let mut state=0x20648512u64;let raw:Vec<i8>=(0..D*elements*N).map(|_|rng(&mut state)as i8).collect();let data=ParityData::build(&raw,elements);let key=data.key();let mut input=vec![0u64;8*elements];for d in 0..D{for e in 0..elements{let poly=&raw[(d*elements+e)*N..(d*elements+e+1)*N];for i in 0..512{let at=8*e+2*(i%4)+(i/4)/64;input[at]=1u64<<((i/4)%64);for m in 0..N{assert_eq!(unsafe{key.odd(d,m,e,e+1,input.as_ptr())},(row(poly,m,i)&1)as u8,"d={d} e={e} m={m} i={i}");}input[at]=0;}}}}
#[test]fn full_and_partial_chunks_match_scalar_integer_sum(){let elements=416;let mut state=0x20ac416u64;let raw:Vec<i8>=(0..D*elements*N).map(|_|rng(&mut state)as i8).collect();let data=ParityData::build(&raw,elements);let key=data.key();let mut input=vec![0u64;elements*8];for case in 0..96{match case%3{0=>input.fill(0),1=>input.fill(u64::MAX),_=>for v in &mut input{*v=rng(&mut state)}}let d=case%D;let m=if case<24{[0usize,1,2,3,68,69,70,71,323,324,460,461,462,463,464,465,566,567,568,644,645,646,647,81][case]}else{rng(&mut state)as usize%N};let(first,last)=match case%4{0=>(0,0),1=>(0,416),2=>(17,401),_=>(415,416)};let mut want=0i64;for e in first..last{let poly=&raw[(d*elements+e)*N..(d*elements+e+1)*N];for i in 0..512{let bit=(input[8*e+2*(i%4)+(i/4)/64]>>((i/4)%64))&1;want+=row(poly,m,i)*bit as i64;}}assert_eq!(unsafe{key.odd(d,m,first,last,input.as_ptr())},(want&1)as u8);}}}
