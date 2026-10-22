// Complete current LOCK2 commitment. RADIX_ORDER selects the isolated prefix. WAIT stays true.
#![allow(dead_code)]
use core::arch::x86_64::*;
use labinius::fields::scalar::B128;
use labinius::params::*;
use labinius::ring::{Batch32,Representation};
use labinius::simd::commit::{w_bound,a_bound};
use labinius::simd::transpose_f162::BinaryIndex32;
use labinius::simd::commit::hybrid::PAcc;
use labinius::simd::ntt::{bin_asm as vb,gen_small};
const P0:u16=3889;
const P1:u16=9721;
const CHUNK_PAIRS:usize=128;
const LOCK:usize=2;
const PP0:usize=pair_period(P0);
const PP1:usize=pair_period(P1);
const PAIR_LEAD:usize=2*PP0;
const PAIR_PACK_PERM:[i64;8]=[0,2,4,6,1,3,5,7];
#[repr(align(128))]
struct Progress(std::sync::atomic::AtomicUsize);
const SIBLING_WAIT:std::time::Duration=std::time::Duration::from_micros(500);
const _:()=assert!(PP0==16&&PP1==8&&CHUNK_PAIRS%PP0==0&&CHUNK_PAIRS%PP1==0);

const fn r16c(q:u16)->i32 {(65536%q as i32)-q as i32}

const fn acc_after_reduce_c(q:u16)->i64 {32768*(1+r16c(q).unsigned_abs() as i64)}

const fn bias(q:u16)->i64 {((1i64<<31)/q as i64)*q as i64}

const fn qinv16(q:u16)->u32 {
 let q=q as u32;let mut x=q;let mut i=0;
 while i<4 {x=x.wrapping_mul(2u32.wrapping_sub(q.wrapping_mul(x)))&0xffff;i+=1;}
 x
}

const fn red16_w(q:u16)->(i32,i32) {
 let wl=(1u32<<16)%q as u32;let wh=((1u64<<32)%q as u64) as u32;
 let qi=qinv16(q);
 ((wl|(wh<<16)) as i32,(((wl*qi)&0xffff)|(((wh*qi)&0xffff)<<16)) as i32)
}

#[inline(always)]
unsafe fn red16<const Q:u16>(x:__m512i)->__m512i {
 let (w,wp)=red16_w(Q);
 let y=_mm512_add_epi32(x,_mm512_set1_epi32(bias(Q) as i32));
 let m=_mm512_mullo_epi16(y,_mm512_set1_epi32(wp));
 let r=_mm512_sub_epi16(_mm512_mulhi_epu16(y,_mm512_set1_epi32(w)),_mm512_mulhi_epu16(m,_mm512_set1_epi16(Q as i16)));
 _mm512_madd_epi16(r,_mm512_set1_epi16(1))
}

#[target_feature(enable="avx512f,avx512bw")]
unsafe fn crt_accumulate(a:&Batch32,b:&Batch32,tail:&mut Batch32) {
 const INV:u16=inv_mod(P0 as u64,P1 as u64) as u16;
 const W:i16=Params::<P1>::to_mont(INV);
 const WP:i16=Params::<P1>::mont_pre(W);
 let qv=_mm512_set1_epi16(P1 as i16);
 let half=_mm512_set1_epi16((P1/2) as i16);
 let nhalf=_mm512_set1_epi16(-((P1/2) as i16));
 for j in 0..648 {
  let av=_mm512_load_si512(a.v[j].as_ptr() as *const __m512i);
  let bv=_mm512_load_si512(b.v[j].as_ptr() as *const __m512i);
  let d=_mm512_sub_epi16(bv,av);
  let m=_mm512_mullo_epi16(d,_mm512_set1_epi16(WP));
  let h=_mm512_mulhi_epi16(d,_mm512_set1_epi16(W));
  let mut u=_mm512_sub_epi16(h,_mm512_mulhi_epi16(m,qv));
  u=_mm512_mask_sub_epi16(u,_mm512_cmp_epi16_mask(u,half,_MM_CMPINT_NLE),u,qv);
  u=_mm512_mask_add_epi16(u,_mm512_cmp_epi16_mask(u,nhalf,_MM_CMPINT_LT),u,qv);
  let low=_mm512_add_epi16(av,_mm512_mullo_epi16(u,_mm512_set1_epi16(P0 as i16)));
  let old=_mm512_load_si512(tail.v[j].as_ptr() as *const __m512i);
  _mm512_store_si512(tail.v[j].as_mut_ptr() as *mut __m512i,_mm512_add_epi16(old,low));
 }
}

fn zeroed_slice<T>(n:usize)->Box<[T]> {
 unsafe {
  let l=std::alloc::Layout::array::<T>(n).unwrap();
  let p=std::alloc::alloc_zeroed(l) as *mut T;
  if p.is_null() {std::alloc::handle_alloc_error(l)}
  Box::from_raw(core::ptr::slice_from_raw_parts_mut(p,n))
 }
}

const fn pair_per_batch(q:u16)->i64 {2*w_bound(q)*a_bound(q)}

const fn pair_period(q:u16)->usize {
 let mut p=1usize;
 while acc_after_reduce_c(q)+2*(p as i64)*pair_per_batch(q)<=i32::MAX as i64 {p*=2;}
 p
}

#[target_feature(enable="avx512f,avx512bw")]
unsafe fn finish_pair_stream<const Q:u16,const STREAM:usize>(a:&mut PAcc,out:&mut Batch32,half:usize) {
 let p=(a.v.as_mut_ptr() as *mut __m512i).add(STREAM);
 let zero=_mm512_setzero_si512();
 let perm=_mm512_loadu_si512(PAIR_PACK_PERM.as_ptr() as *const __m512i);
 let l0=16*half;
 for k in 0..324 {
  let j=2*k;
  let x0=_mm512_load_si512(p.add(2*j));
  let x1=_mm512_load_si512(p.add(2*(j+1)));
  _mm512_store_si512(p.add(2*j),zero);
  _mm512_store_si512(p.add(2*(j+1)),zero);
  let packed=_mm512_packs_epi32(red16::<Q>(x0),red16::<Q>(x1));
  let u=_mm512_permutexvar_epi64(perm,packed);
  _mm256_storeu_si256(out.v[j].as_mut_ptr().add(l0) as *mut __m256i,_mm512_castsi512_si256(u));
  _mm256_storeu_si256(out.v[j+1].as_mut_ptr().add(l0) as *mut __m256i,_mm512_extracti64x4_epi64::<1>(u));
 }
}

#[inline(always)]
unsafe fn canon_red16<const Q:u16>(x:__m512i)->__m512i {
 let r=red16::<Q>(x);
 let q=_mm512_set1_epi32(Q as i32);
 let q2=_mm512_set1_epi32(2*Q as i32);
 let r=_mm512_mask_add_epi32(r,_mm512_cmp_epi32_mask(r,_mm512_setzero_si512(),_MM_CMPINT_LT),r,q2);
 _mm512_mask_sub_epi32(r,_mm512_cmp_epi32_mask(r,q,_MM_CMPINT_NLT),r,q)
}

#[target_feature(enable="avx512f,avx512bw")]
unsafe fn clear_upper(out:&mut Batch32) {
 let zero=_mm256_setzero_si256();
 for j in 0..648 {_mm256_storeu_si256(out.v[j].as_mut_ptr().add(16) as *mut __m256i,zero);}
}

#[inline(always)]
unsafe fn tr4(r0:__m512i,r1:__m512i,r2:__m512i,r3:__m512i)->[__m512i;4] {
 let t0=_mm512_shuffle_i64x2::<0x88>(r0,r1);let t1=_mm512_shuffle_i64x2::<0xDD>(r0,r1);
 let t2=_mm512_shuffle_i64x2::<0x88>(r2,r3);let t3=_mm512_shuffle_i64x2::<0xDD>(r2,r3);
 [_mm512_shuffle_i64x2::<0x88>(t0,t2),_mm512_shuffle_i64x2::<0x88>(t1,t3),
  _mm512_shuffle_i64x2::<0xDD>(t0,t2),_mm512_shuffle_i64x2::<0xDD>(t1,t3)]
}

#[inline(always)]
unsafe fn transpose16x16_u32(r:&[__m512i;16])->[__m512i;16] {
 let mut u=[[_mm512_setzero_si512();4];4];
 for q in 0..4 {
  let t0=_mm512_unpacklo_epi32(r[4*q],r[4*q+1]);
  let t1=_mm512_unpackhi_epi32(r[4*q],r[4*q+1]);
  let t2=_mm512_unpacklo_epi32(r[4*q+2],r[4*q+3]);
  let t3=_mm512_unpackhi_epi32(r[4*q+2],r[4*q+3]);
  u[q]=[
   _mm512_unpacklo_epi64(t0,t2),
   _mm512_unpackhi_epi64(t0,t2),
   _mm512_unpacklo_epi64(t1,t3),
   _mm512_unpackhi_epi64(t1,t3),
  ];
 }
 let mut out=[_mm512_setzero_si512();16];
 for m in 0..4 {
  let v=tr4(u[0][m],u[1][m],u[2][m],u[3][m]);
  for l in 0..4 {out[4*l+m]=v[l];}
 }
 out
}

#[inline(never)]
#[target_feature(enable="avx512f,avx512bw,avx512vl")]
unsafe fn write_pair_group_rows(
 acc0:&mut PAcc,
 acc1:&mut PAcc,
 total:&mut Batch32,
 gc:usize,
 dst:*mut [u32;N],
) {
 let mask1023=_mm256_set1_epi16(1023);
 let zero=_mm512_setzero_si512();
 let p0=acc0.v.as_mut_ptr() as *mut __m512i;
 let p1=acc1.v.as_mut_ptr() as *mut __m512i;
 let pt=total.v.as_mut_ptr() as *mut __m512i;
 for tile in 0..40 {
  let j0=16*tile;
  let mut r0=[zero;16];
  let mut r1=[zero;16];
  let mut rt=[zero;16];
  for s in 0..16 {
   let j=j0+s;
   let v0=_mm512_load_si512(p0.add(2*j));
   let v1=_mm512_load_si512(p1.add(2*j));
   let vt=_mm512_load_si512(pt.add(j));
   _mm512_store_si512(p0.add(2*j),zero);
   _mm512_store_si512(p1.add(2*j),zero);
   _mm512_store_si512(pt.add(j),zero);
   r0[s]=canon_red16::<P0>(v0);
   r1[s]=canon_red16::<P1>(v1);
   let st=_mm256_add_epi16(_mm512_castsi512_si256(vt),_mm512_extracti64x4_epi64::<1>(vt));
   rt[s]=_mm512_cvtepu16_epi32(_mm256_and_si256(st,mask1023));
  }
  let c0=transpose16x16_u32(&r0);
  let c1=transpose16x16_u32(&r1);
  let ct=transpose16x16_u32(&rt);
  for c in 0..gc {
   _mm512_storeu_si512((dst.add(3*c) as *mut u32).add(j0) as *mut __m512i,c0[c]);
   _mm512_storeu_si512((dst.add(3*c+1) as *mut u32).add(j0) as *mut __m512i,c1[c]);
   _mm512_storeu_si512((dst.add(3*c+2) as *mut u32).add(j0) as *mut __m512i,ct[c]);
  }
 }
 // Remaining 8 slots (j = 640..648).
 let j0=640;
 let mut r0=[zero;16];
 let mut r1=[zero;16];
 let mut rt=[zero;16];
 for s in 0..8 {
  let j=j0+s;
  let v0=_mm512_load_si512(p0.add(2*j));
  let v1=_mm512_load_si512(p1.add(2*j));
  let vt=_mm512_load_si512(pt.add(j));
  _mm512_store_si512(p0.add(2*j),zero);
  _mm512_store_si512(p1.add(2*j),zero);
  _mm512_store_si512(pt.add(j),zero);
  r0[s]=canon_red16::<P0>(v0);
  r1[s]=canon_red16::<P1>(v1);
  let st=_mm256_add_epi16(_mm512_castsi512_si256(vt),_mm512_extracti64x4_epi64::<1>(vt));
  rt[s]=_mm512_cvtepu16_epi32(_mm256_and_si256(st,mask1023));
 }
 let c0=transpose16x16_u32(&r0);
 let c1=transpose16x16_u32(&r1);
 let ct=transpose16x16_u32(&rt);
 for c in 0..gc {
  _mm256_storeu_si256((dst.add(3*c) as *mut u32).add(j0) as *mut __m256i,_mm512_castsi512_si256(c0[c]));
  _mm256_storeu_si256((dst.add(3*c+1) as *mut u32).add(j0) as *mut __m256i,_mm512_castsi512_si256(c1[c]));
  _mm256_storeu_si256((dst.add(3*c+2) as *mut u32).add(j0) as *mut __m256i,_mm512_castsi512_si256(ct[c]));
 }
}

struct PairWorker {
 acc:Box<[PAcc]>,
 partial:Box<[Batch32]>,
 total:Box<[Batch32]>,
 idx:Box<[BinaryIndex32]>,
}
impl PairWorker {
 fn new()->Self {
  PairWorker{
   acc:zeroed_slice(2),
   partial:zeroed_slice(2),
   total:zeroed_slice(1),
   idx:zeroed_slice(LOCK),
  }
 }
}


#[target_feature(enable="avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vnni,gfni")]
unsafe fn group_pairs_full<const RADIX_ORDER:bool,F:FnMut(usize)>(
 cols:&[&[B128]],
 keys:&[u32],
 np:usize,
 ws:&mut PairWorker,
 dst:*mut [u32;N],
 mut sync:F,
) {
 let g=cols.len();assert!(g>=1 && g<=16 && np%LOCK==0);
 // Lanes of columns past the group's end read a zero element.
 static ZERO8:[B128;8]=[B128::ZERO;8];
 let kbase=keys.as_ptr();
 let mut half=0usize;
 for b in (0..np).step_by(LOCK) {
  for h in 0..LOCK {
   let cols16:[*const B128;16]=core::array::from_fn(|l|{
    if l<g {cols[l].as_ptr().add(8*(b+h))} else {ZERO8.as_ptr()}
   });
   labinius::simd::transpose_f162::slice_b128_cols16_into(&cols16,&mut ws.idx[h]);
  }
  let (acc0,acc1)={
   let [x,y]=&mut ws.acc[..2] else {unreachable!()};
   (x.v.as_mut_ptr() as *mut i32,y.v.as_mut_ptr() as *mut i32)
  };
  let ins:[&BinaryIndex32;LOCK]=[&ws.idx[0],&ws.idx[1]];
  let kb=kbase.add(b*648*4);
  if RADIX_ORDER {
   crate::ntt_radix_order::ntt_bin_batch32_pair::<P0>(ins,kb,acc0,(b+LOCK)%PP0==0);
   crate::ntt_radix_order::ntt_bin_batch32_pair::<P1>(ins,kb.add(648*4),acc1,(b+LOCK)%PP1==0);
  } else {
   vb::ntt_bin_batch32_pair::<P0>(ins,kb,acc0,(b+LOCK)%PP0==0);
   vb::ntt_bin_batch32_pair::<P1>(ins,kb.add(648*4),acc1,(b+LOCK)%PP1==0);
  }
  if (b+LOCK)%PP0==0 || b+LOCK==np {
   sync(b+LOCK);
  }
  if (b+LOCK)%CHUNK_PAIRS==0 || b+LOCK==np {
   // Two finished chunks share one pair of inverse transforms: the first in lanes 0..16, the
   // second in lanes 16..32 (each chunk's CRT stays exact on its own lanes).
   let (p0,p1)=ws.partial.split_at_mut(1);
   finish_pair_stream::<P0,1>(&mut ws.acc[0],&mut p0[0],half);
   finish_pair_stream::<P1,1>(&mut ws.acc[1],&mut p1[0],half);
   if half==1 || b+LOCK==np {
    if half==0 {clear_upper(&mut p0[0]);clear_upper(&mut p1[0]);}
    p0[0].representation=Representation::Ntt;p1[0].representation=Representation::Ntt;
    // finish_pair_stream uses red16: every live lane has |r| < 2*P0; unused lanes are zero.
    gen_small::intt_hybrid_batch32_3889(&mut p0[0]);
    gen_small::intt_gen_batch32::<P1>(&mut p1[0]);
    crt_accumulate(&p0[0],&p1[0],&mut ws.total[0]);
    half=0;
   } else {half=1;}
  }
 }
 let [acc0,acc1]=&mut ws.acc[..2] else {unreachable!()};
 write_pair_group_rows(acc0,acc1,&mut ws.total[0],g,dst);
}

fn commit_paired_pairs<const RADIX_ORDER:bool,const WAIT:bool>(
 keys:&[u32],
 columns:&[&[B128]],
 np:usize,
 workers:usize,
 placement:Option<&labinius::par::CorePlacement>,
 dst:usize,
) {
 use std::sync::atomic::{AtomicBool,AtomicUsize,Ordering};
 let g=16usize;let sg=2*g;
 let pairs=workers/2;
 let super_groups=columns.len().div_ceil(sg);
 let rounds=super_groups+1;
 let next=AtomicUsize::new(0);
 let slots:Vec<std::sync::OnceLock<usize>>=(0..pairs*rounds).map(|_|std::sync::OnceLock::new()).collect();
 let done:Vec<Progress>=(0..2*pairs).map(|_|Progress(AtomicUsize::new(0))).collect();
 let coupled:Vec<AtomicBool>=(0..pairs).map(|_|AtomicBool::new(true)).collect();
 labinius::par::pool().run(2*pairs,|worker|{
  let _guard=placement.and_then(|p|p.pin(worker));
  let (pair,s)=(worker/2,worker%2);
  let mine=&done[worker].0;let other=&done[worker^1].0;let coupled=&coupled[pair];
  let mut ws:Option<PairWorker>=None;
  for r in 0..rounds {
   let gi=*slots[pair*rounds+r].get_or_init(||next.fetch_add(1,Ordering::Relaxed));
   if gi>=super_groups {break;}
   let c0=gi*sg;let gc=sg.min(columns.len()-c0);
   let (lo,n)=if s==0 {(c0,gc.min(g))} else {(c0+g,gc.saturating_sub(g))};
   let base=r*np;
   if n==0 {mine.store(base+np,Ordering::Release);continue;}
   let ws=ws.get_or_insert_with(PairWorker::new);
   unsafe{group_pairs_full::<RADIX_ORDER,_>(&columns[lo..lo+n],keys,np,ws,(dst as *mut [u32;N]).add(3*lo),|b_done|{
    if !WAIT {return;}
    let d=base+b_done;
    mine.store(d,Ordering::Release);
    if !coupled.load(Ordering::Relaxed) {return;}
    if other.load(Ordering::Acquire)+PAIR_LEAD>=d {return;}
    let start=std::time::Instant::now();
    let mut spins=0u32;
    while other.load(Ordering::Acquire)+PAIR_LEAD<d {
     core::hint::spin_loop();
     spins+=1;
     if spins%64==0 && start.elapsed()>SIBLING_WAIT {coupled.store(false,Ordering::Relaxed);return;}
    }
   })};
  }
  mine.store(usize::MAX/2,Ordering::Release);
 });
}

pub fn commit_columns_rows_pairs<const RADIX_ORDER:bool,const WAIT:bool>(keys:&[u32],columns:&[&[B128]],workers:usize)->Vec<[u32;N]> {
 assert!(!columns.is_empty(),"at least one witness column");
 let len=columns[0].len();for c in columns {assert_eq!(c.len(),len);}
 assert_eq!(len%8,0);let np=len/8;
 assert_eq!(keys.len(),4*np*648);
 let g=16usize;let groups=columns.len().div_ceil(g);
 // Uninitialised: the groups partition the columns and each writes all three rows of its own.
 let mut rows:Vec<[u32;N]>=Vec::with_capacity(columns.len()*3);let dst=rows.as_mut_ptr() as usize;
 let paired=if workers>=2 && workers%2==0 && groups>1 {labinius::par::CorePlacement::for_sibling_pairs(workers)} else {None};
 if let Some(placement)=paired {
  commit_paired_pairs::<RADIX_ORDER,WAIT>(keys,columns,np,workers,Some(&placement),dst);
 } else {
  let workers=workers.clamp(1,groups);
  let next=std::sync::atomic::AtomicUsize::new(0);
  let placement=labinius::par::CorePlacement::for_logical(workers).or_else(||labinius::par::CorePlacement::for_workers(workers));
  labinius::par::pool().run(workers,|worker|{
   let _guard=placement.as_ref().and_then(|p|p.pin(worker));
   let mut ws:Option<PairWorker>=None;
   loop {
    let gi=next.fetch_add(1,std::sync::atomic::Ordering::Relaxed);
    if gi>=groups {break;}
    let c0=gi*g;let gc=g.min(columns.len()-c0);
    let ws=ws.get_or_insert_with(PairWorker::new);
    unsafe{group_pairs_full::<RADIX_ORDER,_>(&columns[c0..c0+gc],keys,np,ws,(dst as *mut [u32;N]).add(3*c0),|_|{})};
   }
  });
 }
 unsafe{rows.set_len(columns.len()*3);}
 rows
}
