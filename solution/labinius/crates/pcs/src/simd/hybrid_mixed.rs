//! Mixed splitting/quadratic hybrid commitment, Q=9721*2917*1280.
//!
//! The fully splitting prime retains the LOCK2 column-pair kernel. The second
//! prime uses 324 quadratic leaves and two direct-product output sums per
//! key stream; it is never treated as 648 scalar pointwise products.
use core::arch::x86_64::*;
use crate::fields::scalar::B128;
use crate::key::CommitmentKey;
use crate::params::*;
use crate::ring::{Batch32,Representation};
use crate::simd::commit::hybrid::PAcc;
use crate::simd::ntt::{bin_asm as vb,gen_small,gen_quad};
use crate::simd::mixed_quad_mac::{self as qm,QuadPairAcc};
use crate::simd::transpose_f162::BinaryIndex32;

const P0:u16=9721;
const P1:u16=2917;
const PRODUCT:i64=P0 as i64*P1 as i64;
pub const MODULUS:u128=PRODUCT as u128*1280;
pub const CHUNK_BATCHES:usize=5;
const CHUNK_PAIRS:usize=CHUNK_BATCHES*16;
pub const SCALAR_KEY_WORDS:usize=648*4;
pub const LOCK_KEY_WORDS:usize=SCALAR_KEY_WORDS+324*3*4;
const PP0:usize=8;
const SYNC_PERIOD:usize=16;
const _:()=assert!(CHUNK_PAIRS%PP0==0 && CHUNK_PAIRS%qm::OUTPUT_PERIOD==0 && CHUNK_PAIRS%SYNC_PERIOD==0);
const PAIR_LEAD:usize=32;
const LOCK:usize=2;
const SIBLING_WAIT:std::time::Duration=std::time::Duration::from_micros(500);
const _:()=assert!(2*(P0 as i32)<=gen_small::TwI::<P0>::IN_BOUND);

fn zeroed_slice<T>(n:usize)->Box<[T]> {
 unsafe {
  let layout=std::alloc::Layout::array::<T>(n).unwrap();
  let p=std::alloc::alloc_zeroed(layout) as *mut T;
  if p.is_null(){std::alloc::handle_alloc_error(layout)}
  Box::from_raw(core::ptr::slice_from_raw_parts_mut(p,n))
 }
}

fn check_keys(base:&CommitmentKey,tail:&CommitmentKey) {
 assert_eq!(base.limbs(),2);assert_eq!(tail.limbs(),2);
 assert_eq!(base.prime(0),P0);assert_eq!(base.prime(1),P1);
 assert_eq!(tail.prime(0),P0);assert_eq!(tail.prime(1),P1);
 assert_eq!(base.row(0).len(),base.row(1).len());
 assert_eq!(base.row(0).len(),tail.row(0).len());
 assert_eq!(base.row(0).len(),tail.row(1).len());
}
fn pair_word(a:i16,b:i16)->u32 {(a as u16 as u32)|((b as u16 as u32)<<16)}
fn centered_scaled(c:u16,b:i16)->i16 {
 let r=(c as i32*b as i32).rem_euclid(P1 as i32);
 (if r>P1 as i32/2 {r-P1 as i32}else{r}) as i16
}

/// One LOCK2 image: scalar9721 rows then three2917 key forms per leaf.
pub fn pair_keys(base:&CommitmentKey,tail:&CommitmentKey,workers:usize)->Vec<u32> {
 check_keys(base,tail);
 let rows=[[base.row(0),tail.row(0)],[base.row(1),tail.row(1)]];
 let nb=rows[0][0].len();
 let mut out=vec![0u32;nb*8*LOCK_KEY_WORDS];
 let address=out.as_mut_ptr() as usize;
 let runs=crate::par::ranges(nb,workers.max(1),1);
 crate::par::pool().run(runs.len(),|runner|{
  for batch in runs[runner].clone() {
   for lock in 0..8 {
    let dst=unsafe{(address as *mut u32).add((8*batch+lock)*LOCK_KEY_WORDS)};
    for h in 0..2 {for stream in 0..2 {
     let lane=4*lock+2*h;
     let scalar=&rows[0][stream][batch];
     for j in 0..648 {
      unsafe{dst.add(4*j+2*h+stream).write(pair_word(scalar.v[j][lane],scalar.v[j][lane+1]));}
     }
     let quad=&rows[1][stream][batch];
     for leaf in 0..324 {
      let a=&quad.v[2*leaf];let b=&quad.v[2*leaf+1];
      let words=[pair_word(a[lane],a[lane+1]),pair_word(b[lane],b[lane+1]),
        pair_word(centered_scaled(ParamsQ::<P1>::LEAF_C[leaf],b[lane]),centered_scaled(ParamsQ::<P1>::LEAF_C[leaf],b[lane+1]))];
      for form in 0..3 {
       unsafe{dst.add(SCALAR_KEY_WORDS+4*(3*leaf+form)+2*h+stream).write(words[form]);}
      }
     }
    }}
   }
  }
 });
 out
}

const fn qinv16(q:u16)->u32 {
 let q=q as u32;let mut x=q;let mut i=0;
 while i<4 {x=x.wrapping_mul(2u32.wrapping_sub(q.wrapping_mul(x)))&0xffff;i+=1;}x
}
const fn bias(q:u16)->i64 {((1i64<<31)/q as i64)*q as i64}
const fn red16_w(q:u16)->(i32,i32) {
 let wl=(1u32<<16)%q as u32;let wh=((1u64<<32)%q as u64) as u32;let qi=qinv16(q);
 ((wl|(wh<<16)) as i32,(((wl*qi)&0xffff)|(((wh*qi)&0xffff)<<16)) as i32)
}
#[inline(always)]
unsafe fn red16<const Q:u16>(x:__m512i)->__m512i {
 let(w,wp)=red16_w(Q);let y=_mm512_add_epi32(x,_mm512_set1_epi32(bias(Q) as i32));
 let m=_mm512_mullo_epi16(y,_mm512_set1_epi32(wp));
 let r=_mm512_sub_epi16(_mm512_mulhi_epu16(y,_mm512_set1_epi32(w)),_mm512_mulhi_epu16(m,_mm512_set1_epi16(Q as i16)));
 _mm512_madd_epi16(r,_mm512_set1_epi16(1))
}
const PAIR_PACK_PERM:[i64;8]=[0,2,4,6,1,3,5,7];
#[target_feature(enable="avx512f,avx512bw")]
unsafe fn finish_split<const STREAM:usize>(a:&mut PAcc,out:&mut Batch32,half:usize) {
 let p=(a.v.as_mut_ptr() as *mut __m512i).add(STREAM);
 let zero=_mm512_setzero_si512();let perm=_mm512_loadu_si512(PAIR_PACK_PERM.as_ptr() as *const __m512i);
 for k in 0..324 {
  let j=2*k;let x0=_mm512_load_si512(p.add(2*j));let x1=_mm512_load_si512(p.add(2*(j+1)));
  _mm512_store_si512(p.add(2*j),zero);_mm512_store_si512(p.add(2*(j+1)),zero);
  let packed=_mm512_packs_epi32(red16::<P0>(x0),red16::<P0>(x1));let u=_mm512_permutexvar_epi64(perm,packed);
  _mm256_storeu_si256(out.v[j].as_mut_ptr().add(16*half) as *mut __m256i,_mm512_castsi512_si256(u));
  _mm256_storeu_si256(out.v[j+1].as_mut_ptr().add(16*half) as *mut __m256i,_mm512_extracti64x4_epi64::<1>(u));
 }
}
#[target_feature(enable="avx512f,avx512bw")]
unsafe fn clear_upper(out:&mut Batch32) {
 let zero=_mm256_setzero_si256();
 for j in 0..648 {_mm256_storeu_si256(out.v[j].as_mut_ptr().add(16) as *mut __m256i,zero);}
}

/// One exact integer auxiliary total per column, with the same data size as Batch32.
#[repr(C,align(64))]
struct TailAcc {v:[[i32;16];648]}
const _:()=assert!(core::mem::size_of::<TailAcc>()==648*64);

/// Exact centered CRT and exact i32 accumulation of the paired chunk halves.
/// certify_total bounds every partial sum; no i16 or i32 wrap is used.
#[target_feature(enable="avx512f,avx512bw")]
unsafe fn crt_accumulate(a:&Batch32,b:&Batch32,tail:&mut TailAcc) {
 const W:i16=-167;
 const WP:i16=((W as u16 as u32*qinv16(P1))&65535) as u16 as i16;
 let qv=_mm512_set1_epi16(P1 as i16);let half=_mm512_set1_epi16((P1/2) as i16);let nhalf=_mm512_set1_epi16(-((P1/2) as i16));
 for j in 0..648 {
  let av=_mm512_load_si512(a.v[j].as_ptr() as *const __m512i);let bv=_mm512_load_si512(b.v[j].as_ptr() as *const __m512i);
  let d=_mm512_sub_epi16(bv,av);
  let m=_mm512_mullo_epi16(d,_mm512_set1_epi16(WP));
  let mut u=_mm512_sub_epi16(_mm512_mulhi_epi16(d,_mm512_set1_epi16(W)),_mm512_mulhi_epi16(m,qv));
  u=_mm512_mask_sub_epi16(u,_mm512_cmp_epi16_mask(u,half,_MM_CMPINT_NLE),u,qv);
  u=_mm512_mask_add_epi16(u,_mm512_cmp_epi16_mask(u,nhalf,_MM_CMPINT_LT),u,qv);
  // The two chunk halves have the same16columns. Their sums fit i16:
  // |a0+a1|<=9720, |u0+u1|<=2916; the exact CRT pair fits |x0+x1|<=P-1.
  let asum=_mm256_add_epi16(_mm512_castsi512_si256(av),_mm512_extracti64x4_epi64::<1>(av));
  let usum=_mm256_add_epi16(_mm512_castsi512_si256(u),_mm512_extracti64x4_epi64::<1>(u));
  let x=_mm512_add_epi32(_mm512_cvtepi16_epi32(asum),_mm512_mullo_epi32(_mm512_cvtepi16_epi32(usum),_mm512_set1_epi32(P0 as i32)));
  let old=_mm512_load_si512(tail.v[j].as_ptr() as *const __m512i);
  _mm512_store_si512(tail.v[j].as_mut_ptr() as *mut __m512i,_mm512_add_epi32(old,x));
 }
}

struct Worker {
 split:Box<[PAcc]>,quad:Box<[QuadPairAcc]>,idx:Box<[BinaryIndex32]>,
 partial:Box<[Batch32]>,total:Box<[TailAcc]>,
}
impl Worker {fn new()->Self {Self{split:zeroed_slice(1),quad:zeroed_slice(1),idx:zeroed_slice(2),partial:zeroed_slice(2),total:zeroed_slice(1)}}}

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

/// Canonicalize the unchanged finish_split output: strict |x|<2*P0.
#[inline(always)]
unsafe fn canonical_finished_base(v:__m256i)->__m512i {
 let x=_mm512_cvtepi16_epi32(v);let q=_mm512_set1_epi32(P0 as i32);
 let x=_mm512_mask_add_epi32(x,_mm512_cmp_epi32_mask(x,_mm512_setzero_si512(),_MM_CMPINT_LT),x,_mm512_set1_epi32(2*P0 as i32));
 _mm512_mask_sub_epi32(x,_mm512_cmp_epi32_mask(x,q,_MM_CMPINT_NLT),x,q)
}

/// Canonicalize the unchanged quadratic finish output: |x|<=1458.
#[inline(always)]
unsafe fn canonical_finished_quad(v:__m256i)->__m512i {
 let x=_mm512_cvtepi16_epi32(v);
 _mm512_mask_add_epi32(x,_mm512_cmp_epi32_mask(x,_mm512_setzero_si512(),_MM_CMPINT_LT),x,_mm512_set1_epi32(P1 as i32))
}

/// Post-finish output only. Tail modulo1280 remains scalar before transposition.
#[inline(never)]
#[target_feature(enable="avx512f,avx512bw,avx512vl")]
unsafe fn write_mixed_group_rows(p0:&Batch32,p1:&Batch32,total:&TailAcc,gc:usize,dst:*mut [u32;N]) {
 let zero=_mm512_setzero_si512();
 for tile in 0..40 {
  let j0=16*tile;
  let mut r0=[zero;16];let mut r1=[zero;16];let mut rt=[zero;16];
  for s in 0..16 {
   let j=j0+s;
   r0[s]=canonical_finished_base(_mm256_load_si256(p0.v[j].as_ptr() as *const __m256i));
   r1[s]=canonical_finished_quad(_mm256_load_si256(p1.v[j].as_ptr() as *const __m256i));
   let mut tail=[0u32;16];
   for c in 0..gc {tail[c]=total.v[j][c].rem_euclid(1280) as u32;}
   rt[s]=_mm512_loadu_si512(tail.as_ptr() as *const __m512i);
  }
  let c0=transpose16x16_u32(&r0);let c1=transpose16x16_u32(&r1);let ct=transpose16x16_u32(&rt);
  for c in 0..gc {
   _mm512_storeu_si512((dst.add(3*c) as *mut u32).add(j0) as *mut __m512i,c0[c]);
   _mm512_storeu_si512((dst.add(3*c+1) as *mut u32).add(j0) as *mut __m512i,c1[c]);
   _mm512_storeu_si512((dst.add(3*c+2) as *mut u32).add(j0) as *mut __m512i,ct[c]);
  }
 }
 // The final8slots use half-width stores; no write extends past a648-coefficient row.
 let j0=640;let mut r0=[zero;16];let mut r1=[zero;16];let mut rt=[zero;16];
 for s in 0..8 {
  let j=j0+s;
  r0[s]=canonical_finished_base(_mm256_load_si256(p0.v[j].as_ptr() as *const __m256i));
  r1[s]=canonical_finished_quad(_mm256_load_si256(p1.v[j].as_ptr() as *const __m256i));
  let mut tail=[0u32;16];
  for c in 0..gc {tail[c]=total.v[j][c].rem_euclid(1280) as u32;}
  rt[s]=_mm512_loadu_si512(tail.as_ptr() as *const __m512i);
 }
 let c0=transpose16x16_u32(&r0);let c1=transpose16x16_u32(&r1);let ct=transpose16x16_u32(&rt);
 for c in 0..gc {
  _mm256_storeu_si256((dst.add(3*c) as *mut u32).add(j0) as *mut __m256i,_mm512_castsi512_si256(c0[c]));
  _mm256_storeu_si256((dst.add(3*c+1) as *mut u32).add(j0) as *mut __m256i,_mm512_castsi512_si256(c1[c]));
  _mm256_storeu_si256((dst.add(3*c+2) as *mut u32).add(j0) as *mut __m256i,_mm512_castsi512_si256(ct[c]));
 }
}

#[target_feature(enable="avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vnni,gfni")]
unsafe fn group<F:FnMut(usize)>(cols:&[&[B128]],keys:&[u32],np:usize,ws:&mut Worker,dst:*mut [u32;N],mut sync:F) {
 let gc=cols.len();assert!(gc>=1&&gc<=16&&np%LOCK==0);static ZERO8:[B128;8]=[B128::ZERO;8];
 let mut half=0usize;
 for b in (0..np).step_by(LOCK) {
  for h in 0..2 {
   let ptrs:[*const B128;16]=core::array::from_fn(|c|if c<gc {cols[c].as_ptr().add(8*(b+h))}else{ZERO8.as_ptr()});
   crate::simd::transpose_f162::slice_b128_cols16_into(&ptrs,&mut ws.idx[h]);
  }
  let key=keys.as_ptr().add((b/2)*LOCK_KEY_WORDS);
  vb::ntt_bin_batch32_pair::<P0>([&ws.idx[0],&ws.idx[1]],key,ws.split[0].v.as_mut_ptr() as *mut i32,(b+2)%PP0==0);
  qm::accumulate_lock2_pair_ntt([&ws.idx[0],&ws.idx[1]],key.add(SCALAR_KEY_WORDS),&mut ws.quad[0],b+2);
  if (b+2)%SYNC_PERIOD==0||b+2==np {sync(b+2);}
  if (b+2)%CHUNK_PAIRS==0||b+2==np {
   let(p0,p1)=ws.partial.split_at_mut(1);
   finish_split::<1>(&mut ws.split[0],&mut p0[0],half);
   qm::finish_stream(&mut ws.quad[0],1,&mut p1[0],half);
   if half==1||b+2==np {
    if half==0 {clear_upper(&mut p0[0]);clear_upper(&mut p1[0]);}
    p0[0].representation=Representation::Ntt;p1[0].representation=Representation::Ntt;
    gen_small::intt_gen_batch32::<P0>(&mut p0[0]);gen_quad::intt_quad_gen_batch32::<P1>(&mut p1[0]);
    crt_accumulate(&p0[0],&p1[0],&mut ws.total[0]);half=0;
   }else{half=1;}
  }
 }
 let(p0,p1)=ws.partial.split_at_mut(1);
 finish_split::<0>(&mut ws.split[0],&mut p0[0],0);
 qm::finish_stream(&mut ws.quad[0],0,&mut p1[0],0);
 write_mixed_group_rows(&p0[0],&p1[0],&ws.total[0],gc,dst);
 for row in &mut ws.total[0].v {row.fill(0);}
}

#[repr(align(128))]
struct Progress(std::sync::atomic::AtomicUsize);
fn paired(keys:&[u32],columns:&[&[B128]],np:usize,workers:usize,placement:&crate::par::CorePlacement,dst:usize) {
 use std::sync::atomic::{AtomicBool,AtomicUsize,Ordering};
 let pairs=workers/2;let super_groups=columns.len().div_ceil(32);let rounds=super_groups+1;
 let next=AtomicUsize::new(0);let slots:Vec<std::sync::OnceLock<usize>>=(0..pairs*rounds).map(|_|std::sync::OnceLock::new()).collect();
 let done:Vec<Progress>=(0..2*pairs).map(|_|Progress(AtomicUsize::new(0))).collect();let coupled:Vec<AtomicBool>=(0..pairs).map(|_|AtomicBool::new(true)).collect();
 crate::par::pool().run(2*pairs,|worker|{
  let _guard=placement.pin(worker);let(pair,s)=(worker/2,worker%2);
  let mine=&done[worker].0;let other=&done[worker^1].0;let coupled=&coupled[pair];let mut ws:Option<Worker>=None;
  for round in 0..rounds {
   let gi=*slots[pair*rounds+round].get_or_init(||next.fetch_add(1,Ordering::Relaxed));if gi>=super_groups{break}
   let c0=gi*32;let gc=32.min(columns.len()-c0);let(lo,n)=if s==0{(c0,gc.min(16))}else{(c0+16,gc.saturating_sub(16))};let base=round*np;
   if n==0{mine.store(base+np,Ordering::Release);continue}
   let ws=ws.get_or_insert_with(Worker::new);
   unsafe{group(&columns[lo..lo+n],keys,np,ws,(dst as *mut [u32;N]).add(3*lo),|b_done|{
    let d=base+b_done;mine.store(d,Ordering::Release);if !coupled.load(Ordering::Relaxed){return}
    if other.load(Ordering::Acquire)+PAIR_LEAD>=d{return}
    let start=std::time::Instant::now();let mut spins=0u32;
    while other.load(Ordering::Acquire)+PAIR_LEAD<d {
     core::hint::spin_loop();spins+=1;
     if spins%64==0&&start.elapsed()>SIBLING_WAIT{coupled.store(false,Ordering::Relaxed);return}
    }
   })};
  }
  mine.store(usize::MAX/2,Ordering::Release);
 });
}

/// Three rows per column: scalar NTT, quadratic NTT, canonical coefficient residue mod1280.
pub fn commit_columns_rows_pairs(keys:&[u32],columns:&[&[B128]],workers:usize)->Vec<[u32;N]> {
 assert!(!columns.is_empty());let len=columns[0].len();for c in columns{assert_eq!(c.len(),len)}
 assert_eq!(len%16,0);let np=len/8;assert_eq!(keys.len(),(np/2)*LOCK_KEY_WORDS);
 let groups=columns.len().div_ceil(16);
 // Groups write all three rows of each disjoint column before the pool returns.
 let mut out:Vec<[u32;N]>=Vec::with_capacity(columns.len()*3);let dst=out.as_mut_ptr() as usize;
 let placement=if workers>=2&&workers%2==0&&groups>1{crate::par::CorePlacement::for_sibling_pairs(workers)}else{None};
 if let Some(p)=placement {paired(keys,columns,np,workers,&p,dst)}else{
  let workers=workers.clamp(1,groups);let next=std::sync::atomic::AtomicUsize::new(0);
  let placement=crate::par::CorePlacement::for_logical(workers).or_else(||crate::par::CorePlacement::for_workers(workers));
  crate::par::pool().run(workers,|worker|{
   let _guard=placement.as_ref().and_then(|p|p.pin(worker));let mut ws=Worker::new();
   loop{let gi=next.fetch_add(1,std::sync::atomic::Ordering::Relaxed);if gi>=groups{break}
    let first=16*gi;let end=(first+16).min(columns.len());
    unsafe{group(&columns[first..end],keys,np,&mut ws,(dst as *mut [u32;N]).add(3*first),|_|{})};
   }
  });
 }
 // Every coefficient is initialized; neither worker path publishes the vector early.
 unsafe{out.set_len(columns.len()*3);}
 out
}

pub fn commit_columns_rows(base:&CommitmentKey,tail:&CommitmentKey,columns:&[&[B128]],workers:usize,_group_size:usize)->Vec<[u32;N]> {
 commit_columns_rows_pairs(&pair_keys(base,tail,workers),columns,workers)
}

fn coefficient_bound(raw:&[[i16;648]])->i64 {
 let mut worst=0i64;
 {let chunk=raw;
  let mut pos=[0i64;648];let mut neg=[0i64;648];
  for a in chunk {
   let mut ap=[0i64;649];let mut an=ap;
   let mut sp=[0i64;325];let mut sn=sp;
   for j in 0..648 {ap[j+1]=ap[j]+(a[j] as i64).max(0);an[j+1]=an[j]+(-(a[j] as i64)).max(0);}
   for j in 0..324 {let v=a[j] as i64+a[j+324] as i64;sp[j+1]=sp[j]+v.max(0);sn[j+1]=sn[j]+(-v).max(0);}
   let range=|p:&[i64],n:&[i64],lo:usize,hi:usize|->(i64,i64) {
    if lo>hi {(0,0)}else{(p[hi+1]-p[lo],n[hi+1]-n[lo])}
   };
   for j in 0..648 {
    let mut parts=Vec::with_capacity(3);
    if j<324 {
     parts.push((range(&ap,&an,0,j),false));
     if j<=186 {
      parts.push((range(&sp,&sn,j+137,323),true));
      parts.push((range(&ap,&an,324,647),true));
     }else{parts.push((range(&ap,&an,j+137,647),true));}
    }else{
     parts.push((range(&sp,&sn,j.saturating_sub(511),323),false));
     parts.push((range(&ap,&an,324,j),false));
     if j<511 {parts.push((range(&ap,&an,j-187,323),false));}
    }
    for ((p,n),negative) in parts {if negative {pos[j]+=n;neg[j]+=p;}else{pos[j]+=p;neg[j]+=n;}}
   }
  }
  for j in 0..648 {worst=worst.max(pos[j]).max(neg[j]);}
 }
 worst
}

/// Exact centered CRT capacity of each fixed auxiliary chunk, for every binary witness.
pub fn certify(raw:&[[i16;648]],chunk_batches:usize)->i64 {
 assert!(chunk_batches>0);
 let worst=raw.chunks(chunk_batches*32).map(coefficient_bound).max().unwrap_or(0);
 assert!(2*worst<PRODUCT,"CRT capacity exceeded: {worst} vs {}",PRODUCT/2);
 worst
}

/// Positive/negative bounds of the entire key also bound every partial sum.
pub fn certify_total(raw:&[[i16;648]])->i64 {
 let worst=coefficient_bound(raw);
 assert!(worst<=i32::MAX as i64,"exact auxiliary total exceeds i32: {worst}");
 worst
}

#[cfg(test)]
mod exact_tail_tests {
 use super::*;
 fn centered(x:i64,q:i64)->i16 {let r=x.rem_euclid(q);(if r>q/2 {r-q}else{r}) as i16}
 #[test]
 fn paired_chunk_crt_adds_full_signed_values_in_column_order() {
  assert!(std::arch::is_x86_feature_detected!("avx512bw"));
  let edges=[-PRODUCT/2,-13_825_292,-65537,-65536,-32768,-1,0,1,32768,65536,65537,13_825_292,PRODUCT/2];
  for odd in [false,true] {
   let mut a=Box::new(Batch32::zero(Representation::Coefficients));let mut b=a.clone();
   let mut tail=zeroed_slice::<TailAcc>(1);let mut expected=vec![[0i32;16];648];
   for j in 0..648 {for c in 0..16 {
    let x=edges[(j+c)%edges.len()];let y=if odd {0}else{edges[(3*j+5*c+1)%edges.len()]};
    for (half,value) in [x,y].into_iter().enumerate(){a.v[j][16*half+c]=centered(value,P0 as i64);b.v[j][16*half+c]=centered(value,P1 as i64);}
    let old=[-300_000_000,0,300_000_000][(j+c)%3];tail[0].v[j][c]=old;
    expected[j][c]=(old as i64+x+y) as i32;
   }}
   unsafe{crt_accumulate(&a,&b,&mut tail[0]);}
   assert_eq!(tail[0].v.as_slice(),expected.as_slice());
  }
 }
}



#[cfg(test)]
mod vector_row_tests {
 use super::*;
 #[test]
 fn every_i16_pattern_in_the_declared_finish_domains_is_canonical() {
  assert!(std::arch::is_x86_feature_detected!("avx512bw"));
  for start in (-32768i32..32768).step_by(16) {
   let input:[i16;16]=core::array::from_fn(|i|(start+i as i32) as i16);
   let mut a=[0i32;16];let mut b=a;
   unsafe{let v=_mm256_loadu_si256(input.as_ptr() as *const __m256i);_mm512_storeu_si512(a.as_mut_ptr() as *mut __m512i,canonical_finished_base(v));_mm512_storeu_si512(b.as_mut_ptr() as *mut __m512i,canonical_finished_quad(v));}
   for i in 0..16 {let x=input[i] as i32;
    if x.abs()<2*P0 as i32 {assert_eq!(a[i],x.rem_euclid(P0 as i32));}
    if x.abs()<=P1 as i32/2 {assert_eq!(b[i],x.rem_euclid(P1 as i32));}
   }
  }
 }
 #[test]
 fn vector_rows_match_scalar_all_groups_and_guarded_final8() {
  assert!(std::arch::is_x86_feature_detected!("avx512bw"));
  let mut a=Box::new(Batch32::zero(Representation::Ntt));let mut b=a.clone();let mut t=zeroed_slice::<TailAcc>(1);
  let tail_edges=[i32::MIN,-348_779_492,-65536,-1,0,1,65536,348_623_070,i32::MAX];
  for j in 0..648 {for c in 0..16 {
   a.v[j][c]=(((97*j+213*c)%38883) as i32-19441) as i16;
   b.v[j][c]=((31*j+17*c)%2917) as i16-1458;
   a.v[j][16+c]=i16::MIN;b.v[j][16+c]=i16::MAX;
   t[0].v[j][c]=tail_edges[(j+3*c)%tail_edges.len()];
  }}
  for gc in 1..=16 {
   let mut out=vec![[0xdead_beefu32;648];3*gc+2];
   unsafe{write_mixed_group_rows(&a,&b,&t[0],gc,out.as_mut_ptr().add(1));}
   assert!(out[0].iter().all(|&x|x==0xdead_beef));assert!(out[3*gc+1].iter().all(|&x|x==0xdead_beef));
   for c in 0..gc {for j in 0..648 {
    assert_eq!(out[1+3*c][j],(a.v[j][c] as i64).rem_euclid(P0 as i64) as u32);
    assert_eq!(out[2+3*c][j],(b.v[j][c] as i64).rem_euclid(P1 as i64) as u32);
    assert_eq!(out[3+3*c][j],(t[0].v[j][c] as i64).rem_euclid(1280) as u32);
   }}
  }
 }
}
