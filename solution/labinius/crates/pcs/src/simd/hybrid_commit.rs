//! Commitment modulo 3889 * 9721 * 2^10 using only two witness NTTs.
//!
//! Both transforms feed two public-key streams: the ordinary residue and the centered
//! coefficient lift of A mod 1024. The latter is reconstructed exactly in certified
//! chunks, inverse transformed across columns, and accumulated mod 1024. Only public
//! parameters are precomputed. Every invocation consumes its complete fresh table.
use super::*;
use crate::simd::ntt::gen_small;
const P0:u16=3889;
const P1:u16=9721;
const PRODUCT:i64=P0 as i64*P1 as i64;
pub const MODULUS:u128=PRODUCT as u128*1024;
pub const CHUNK_BATCHES:usize=7;
/// Accumulator vectors per 27-slot block: each packs eight slots of two lanes (the last slot
/// five times), so a column-prime's accumulator is 6 KB.
const QUADS:usize=4;
#[repr(C,align(64))]
struct Acc4 {v:[[i32;16];24*QUADS]}
impl Acc4 {
 fn zero()->Box<Self> {
  unsafe {
   let mut b=Box::<Self>::new_uninit();
   core::ptr::write_bytes(b.as_mut_ptr() as *mut u8,0,core::mem::size_of::<Self>());
   b.assume_init()
  }
 }
}
/// A lane sums sixteen products per batch, four times [`acc_per_batch`]'s four, so the
/// fold-back comes every 2 batches for 3889 and every batch for 9721.
const fn period4(q:u16)->usize {period_for(q,4*acc_per_batch(q))}
const _:()={
 let qs=[P0,P1];let mut i=0;
 while i<2 {
  assert!(acc_after_reduce(qs[i])+(period4(qs[i]) as i64)*4*acc_per_batch(qs[i])<=i32::MAX as i64);
  assert!(2*acc_after_reduce(qs[i])<=i32::MAX as i64);
  i+=1;
 }
 assert!(period4(P0)==2 && period4(P1)==1);
};
/// Four 16-lane product vectors folded to four lanes each, in lane groups `[p0, p2, p1, p3]`.
#[inline(always)]
unsafe fn fold4(p0:__m512i,p1:__m512i,p2:__m512i,p3:__m512i)->__m512i {
 let u=pack2(p0,p1);let v=pack2(p2,p3);
 let y=_mm512_permutex2var_epi64(u,_mm512_set_epi64(13,12,7,6,9,8,3,2),v);
 _mm512_add_epi32(_mm512_mask_blend_epi64(0xCC,u,v),y)
}
/// Eight product vectors folded to two lanes each: 64-bit word `h` of lane group `k` holds
/// `p[4h + [0, 2, 1, 3][k]]`.
#[inline(always)]
unsafe fn fold8(p:[__m512i;8])->__m512i {
 let r0=fold4(p[0],p[1],p[2],p[3]);let r1=fold4(p[4],p[5],p[6],p[7]);
 _mm512_add_epi32(_mm512_mask_blend_epi64(0xAA,r0,r1),_mm512_alignr_epi8::<8>(r1,r0))
}
#[target_feature(enable="avx512f,avx512bw")]
unsafe fn reduce4<const Q:u16>(acc:*mut i32) {
 for j in 0..24*QUADS {
  let s=_mm512_load_si512(acc.add(16*j) as *const __m512i);
  _mm512_store_si512(acc.add(16*j) as *mut __m512i,reduce_vec::<Q>(s));
 }
}
#[inline(always)]
unsafe fn m(k:*const i16,w:__m512i,j:usize)->__m512i {_mm512_madd_epi16(w,_mm512_load_si512(k.add(32*j) as *const __m512i))}
#[inline(always)]
unsafe fn rd<const Q:u16>(p:*const i32)->__m512i {reduce_vec::<Q>(_mm512_load_si512(p as *const __m512i))}
const EVEN:[i32;16]=[0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30];
const ODD:[i32;16]=[1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31];
/// Lane `8v + i` is the sum of lanes `2i, 2i + 1` of `[a, b][v]`.
#[inline(always)]
unsafe fn hsum2(a:__m512i,b:__m512i)->__m512i {
 let e=_mm512_loadu_si512(EVEN.as_ptr() as *const __m512i);let o=_mm512_loadu_si512(ODD.as_ptr() as *const __m512i);
 _mm512_add_epi32(_mm512_permutex2var_epi32(a,e,b),_mm512_permutex2var_epi32(a,o,b))
}
/// Slot `s` of a 16-slot run sits in lane `8 (s / 8) + 2 k + (s % 8) / 4`, `k` its lane group
/// in [`fold8`] order.
const SLOT_LANE:[i32;16]={
 let mut x=[0i32;16];let mut s=0;
 while s<16 {x[s]=(8*(s/8)+2*[0,2,1,3][s%4]+(s%8)/4) as i32;s+=1;}
 x
};
#[target_feature(enable="avx512f,avx512bw")]
unsafe fn finish4<const Q:u16>(acc:&Acc4)->[u32;N] {
 let idx=_mm512_loadu_si512(SLOT_LANE.as_ptr() as *const __m512i);
 let mut y=[0u32;N];
 let p=acc.v.as_ptr() as *const i32;
 for bl in 0..24 {
  let q=p.add(16*QUADS*bl);
  let lo=mod_q::<Q>(hsum2(rd::<Q>(q),rd::<Q>(q.add(16))));
  let hi=mod_q::<Q>(hsum2(rd::<Q>(q.add(32)),rd::<Q>(q.add(48))));
  let o=y.as_mut_ptr().add(27*bl) as *mut i32;
  _mm512_storeu_si512(o as *mut __m512i,_mm512_permutexvar_epi32(idx,lo));
  _mm512_mask_storeu_epi32(o.add(16),0x07ff,_mm512_permutexvar_epi32(idx,hi));
 }
 y
}
struct TwinSink {buf:*mut i16,a:*const i16,t:*const i16,base:*mut i32,tail:*mut i32}
impl BlockSink for TwinSink {
 #[inline(always)]unsafe fn dst(&mut self,_:usize)->*mut i16 {self.buf}
 #[target_feature(enable="avx512f,avx512bw")]
 unsafe fn block(&mut self,blk:usize,w:*const i16) {
  let a=self.a.add(32*27*blk);let t=self.t.add(32*27*blk);
  for p in 0..QUADS {
   let j0=8*p;
   let j1=j0+1;
   let j2=j0+2;
   let j3=(j0+3).min(26);
   let j4=(j0+4).min(26);
   let j5=(j0+5).min(26);
   let j6=(j0+6).min(26);
   let j7=(j0+7).min(26);
   let w0=_mm512_load_si512(w.add(32*j0) as *const __m512i);
   let w1=_mm512_load_si512(w.add(32*j1) as *const __m512i);
   let w2=_mm512_load_si512(w.add(32*j2) as *const __m512i);
   let w3=_mm512_load_si512(w.add(32*j3) as *const __m512i);
   let w4=_mm512_load_si512(w.add(32*j4) as *const __m512i);
   let w5=_mm512_load_si512(w.add(32*j5) as *const __m512i);
   let w6=_mm512_load_si512(w.add(32*j6) as *const __m512i);
   let w7=_mm512_load_si512(w.add(32*j7) as *const __m512i);
   let bp=fold8([m(a,w0,j0),m(a,w1,j1),m(a,w2,j2),m(a,w3,j3),m(a,w4,j4),m(a,w5,j5),m(a,w6,j6),m(a,w7,j7)]);
   let tp=fold8([m(t,w0,j0),m(t,w1,j1),m(t,w2,j2),m(t,w3,j3),m(t,w4,j4),m(t,w5,j5),m(t,w6,j6),m(t,w7,j7)]);
   let ba=self.base.add(16*(QUADS*blk+p)) as *mut __m512i;
   let ta=self.tail.add(16*(QUADS*blk+p)) as *mut __m512i;
   _mm512_store_si512(ba,_mm512_add_epi32(_mm512_load_si512(ba),bp));
   _mm512_store_si512(ta,_mm512_add_epi32(_mm512_load_si512(ta),tp));
  }
 }
}
#[inline(always)]
unsafe fn canonical(x:__m512i,q:i16)->__m512i {
 _mm512_mask_add_epi16(x,_mm512_cmp_epi16_mask(x,_mm512_setzero_si512(),_MM_CMPINT_LT),x,_mm512_set1_epi16(q))
}
#[target_feature(enable="avx512f,avx512bw")]
unsafe fn crt_accumulate(a:&Batch32,b:&Batch32,tail:&mut Batch32) {
 const INV:u16=inv_mod(P0 as u64,P1 as u64) as u16;
 const W:i16=Params::<P1>::to_mont(INV);
 const WP:i16=Params::<P1>::mont_pre(W);
 let qv=_mm512_set1_epi16(P1 as i16);
 for j in 0..648 {
  let av=canonical(_mm512_load_si512(a.v[j].as_ptr() as *const __m512i),P0 as i16);
  let bv=canonical(_mm512_load_si512(b.v[j].as_ptr() as *const __m512i),P1 as i16);
  let d=_mm512_sub_epi16(bv,av);
  let m=_mm512_mullo_epi16(d,_mm512_set1_epi16(WP));
  let h=_mm512_mulhi_epi16(d,_mm512_set1_epi16(W));
  let u=canonical(_mm512_sub_epi16(h,_mm512_mulhi_epi16(m,qv)),P1 as i16);
  let gt=_mm512_cmp_epi16_mask(u,_mm512_set1_epi16((P1/2) as i16),_MM_CMPINT_NLE);
  let eq=_mm512_cmp_epi16_mask(u,_mm512_set1_epi16((P1/2) as i16),_MM_CMPINT_EQ);
  let agt=_mm512_cmp_epi16_mask(av,_mm512_set1_epi16((P0/2) as i16),_MM_CMPINT_NLE);
  let mut low=_mm512_add_epi16(av,_mm512_mullo_epi16(u,_mm512_set1_epi16(P0 as i16)));
  low=_mm512_mask_sub_epi16(low,gt|(eq&agt),low,_mm512_set1_epi16((PRODUCT&65535) as u16 as i16));
  let old=_mm512_load_si512(tail.v[j].as_ptr() as *const __m512i);
  _mm512_store_si512(tail.v[j].as_mut_ptr() as *mut __m512i,_mm512_add_epi16(old,low));
 }
}
struct Out {base:Vec<[u32;N]>,tail:Vec<[u16;N]>}
#[target_feature(enable="avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vnni,gfni")]
unsafe fn group(cols:&[&[B128]],basekeys:&[&[Batch32]],tailkeys:&[&[Batch32]],nb:usize,chunk:usize)->Out {
 let g=cols.len();assert!(g<=32);
 let mut accs:Vec<Box<Acc4>>=(0..g*2).map(|_|Acc4::zero()).collect();
 let mut tails:Vec<Box<Acc4>>=(0..g*2).map(|_|Acc4::zero()).collect();
 let mut partial0=Batch32::zero(Representation::Ntt);let mut partial1=Batch32::zero(Representation::Ntt);
 let mut total=Batch32::zero(Representation::Coefficients);
 let mut idx=BinaryIndex32::zero();let mut buf=Blk27([0;27*32]);let bp=buf.0.as_mut_ptr();
 for b in 0..nb {
  for (c,col) in cols.iter().enumerate() {
   col.slice_batch(b,&mut idx);
   let p0=accs[2*c].v.as_mut_ptr() as *mut i32;let t0=tails[2*c].v.as_mut_ptr() as *mut i32;
   let p1=accs[2*c+1].v.as_mut_ptr() as *mut i32;let t1=tails[2*c+1].v.as_mut_ptr() as *mut i32;
   vb::ntt_bin_batch32_sink::<P0,_>(&idx,&mut TwinSink{buf:bp,a:basekeys[0][b].v.as_ptr() as *const i16,
    t:tailkeys[0][b].v.as_ptr() as *const i16,base:p0,tail:t0});
   vb::ntt_bin_batch32_sink::<P1,_>(&idx,&mut TwinSink{buf:bp,a:basekeys[1][b].v.as_ptr() as *const i16,
    t:tailkeys[1][b].v.as_ptr() as *const i16,base:p1,tail:t1});
   if (b+1)%period4(P0)==0 {reduce4::<P0>(p0);reduce4::<P0>(t0);}
   if (b+1)%period4(P1)==0 {reduce4::<P1>(p1);reduce4::<P1>(t1);}
  }
  if (b+1)%chunk==0 || b+1==nb {
   for c in 0..g {
    let a=finish4::<P0>(&tails[2*c]);let v=finish4::<P1>(&tails[2*c+1]);
    for j in 0..648 {partial0.v[j][c]=a[j] as i16;partial1.v[j][c]=v[j] as i16;}
    tails[2*c].v.fill([0;16]);tails[2*c+1].v.fill([0;16]);
   }
   partial0.representation=Representation::Ntt;partial1.representation=Representation::Ntt;
   gen_small::intt_gen_batch32::<P0>(&mut partial0);gen_small::intt_gen_batch32::<P1>(&mut partial1);
   crt_accumulate(&partial0,&partial1,&mut total);
  }
 }
 Out{
  base:(0..g).flat_map(|c|[finish4::<P0>(&accs[2*c]),finish4::<P1>(&accs[2*c+1])]).collect(),
  tail:(0..g).map(|c|core::array::from_fn(|j|(total.v[j][c] as u16)&1023)).collect(),
 }
}
pub fn certify(raw:&[[i16;648]],chunk_batches:usize)->i64 {
 let mut worst=0i64;
 for chunk in raw.chunks(chunk_batches*32) {
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
 assert!(2*worst<PRODUCT,"CRT capacity exceeded: {worst} vs {}",PRODUCT/2);
 worst
}

/// Commit B128 columns under two ordinary limbs and an auxiliary 1024 residue.
/// The tail key is the NTT of the centered coefficient lift of A mod 1024.
/// The public-key certificate establishes exact reconstruction for CHUNK_BATCHES.
pub fn commit_columns_rows(
 base:&crate::key::CommitmentKey, tail:&crate::key::CommitmentKey,
 columns:&[&[B128]],workers:usize,group_size:usize,
)->Vec<[u32;N]> {
 assert_eq!(base.limbs(),2);assert_eq!(tail.limbs(),2);
 assert_eq!((base.prime(0),base.prime(1)),(P0,P1));
 assert_eq!((tail.prime(0),tail.prime(1)),(P0,P1));
 assert_eq!(base.len_f162(),tail.len_f162());
 for c in columns {assert_eq!(c.len(),base.len_f162());}
 let nb=base.row(0).len();let bk=[base.row(0),base.row(1)];let tk=[tail.row(0),tail.row(1)];
 assert!(!columns.is_empty(),"at least one witness column");
 let g=group_size.clamp(1,32).min(columns.len());
 let groups=columns.len().div_ceil(g);let workers=workers.clamp(1,groups);
 let mut rows=vec![[0u32;N];columns.len()*3];let dst=rows.as_mut_ptr() as usize;
 let next=std::sync::atomic::AtomicUsize::new(0);
 let placement=crate::par::CorePlacement::for_workers(workers);
 crate::par::pool().run(workers,|worker|{
  let _guard=placement.as_ref().and_then(|p|p.pin(worker));
  loop {
   let gi=next.fetch_add(1,std::sync::atomic::Ordering::Relaxed);
   if gi>=groups {break;}
   let c0=gi*g;let gc=g.min(columns.len()-c0);
   let out=unsafe{group(&columns[c0..c0+gc],&bk,&tk,nb,CHUNK_BATCHES)};
   for c in 0..gc {unsafe{
    let p=(dst as *mut [u32;N]).add((c0+c)*3);
    p.write(out.base[2*c]);p.add(1).write(out.base[2*c+1]);
    p.add(2).write(core::array::from_fn(|j|out.tail[c][j] as u32));
   }}
  }
 });
 rows
}
#[cfg(test)]
mod tests {
 use super::*;
 #[test]
 fn centered_crt_boundaries() {
  unsafe{
   let mut a=Batch32::zero(Representation::Coefficients);let mut b=Batch32::zero(Representation::Coefficients);
   let mut out=Batch32::zero(Representation::Coefficients);
   let vals=[-PRODUCT/2,-PRODUCT/2+1,-9721,-3889,-1,0,1,3889,9721,PRODUCT/2-1,PRODUCT/2];
   for j in 0..648 {for p in 0..32 {
    let v=vals[(j*32+p)%vals.len()];
    a.v[j][p]=center(v.rem_euclid(P0 as i64) as u64,P0 as u64);
    b.v[j][p]=center(v.rem_euclid(P1 as i64) as u64,P1 as u64);
   }}
   crt_accumulate(&a,&b,&mut out);
   for j in 0..648 {for p in 0..32 {
    assert_eq!((out.v[j][p] as u16)&1023,vals[(j*32+p)%vals.len()].rem_euclid(1024) as u16);
   }}
  }
 }
 #[test]
 fn public_key_certificate_matches_independent_monomials() {
  let mut x=0x38476591u64;
  let raw:Vec<[i16;648]>=(0..4).map(|_|core::array::from_fn(|_|{x^=x<<13;x^=x>>7;x^=x<<17;(x%1024) as i16-512})).collect();
  let mut pos=[0i64;648];let mut neg=[0i64;648];
  for a in &raw {for bit in 0..512 {
   let mut k=[0i64;1295];
   for j in 0..648 {k[j+bit]=a[j] as i64;}
   for j in (648..1295).rev() {let v=k[j];k[j-324]+=v;k[j-648]-=v;}
   for j in 0..648 {pos[j]+=k[j].max(0);neg[j]+=(-k[j]).max(0);}
  }}
  let expected=*pos.iter().chain(neg.iter()).max().unwrap();
  assert_eq!(certify(&raw,CHUNK_BATCHES),expected);
 }
}
