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
/// The paired key layout the sink streams. Within each 27-slot block, slots `2p` and `2p + 1`
/// (`p < 13`) are stored as two rows: ring elements 0..16 of both slots, then elements 16..32 of
/// both; slot 26 keeps its row. [`TwinSink`] forms the same two halves of the transform block
/// with two `vshufi64x2` per pair (shared by the two key streams of a prime) and feeds them to
/// `vpdpwssd`, so accumulator lane `k` receives the products of elements `2k, 2k+1, 2k+16,
/// 2k+17` of slot `2p` (lanes 0..8) or `2p + 1` (lanes 8..16): the same lane sums [`pack2`] of
/// two `vpmaddwd` gave, in the same integer, without the three shuffle-and-add uops per pair and
/// without the separate accumulator add.
pub fn pair_batch(src:&Batch32)->Batch32 {
 let mut out=src.clone();
 for blk in 0..24 {
  for p in 0..13 {
   let (j0,j1)=(27*blk+2*p,27*blk+2*p+1);
   out.v[j0][..16].copy_from_slice(&src.v[j0][..16]);out.v[j0][16..].copy_from_slice(&src.v[j1][..16]);
   out.v[j1][..16].copy_from_slice(&src.v[j0][16..]);out.v[j1][16..].copy_from_slice(&src.v[j1][16..]);
  }
 }
 out
}
/// The owned, paired copy of a key for [`commit_columns_rows`], which takes paired keys only.
pub fn pair_key(key:&crate::key::CommitmentKey)->crate::key::CommitmentKey {key.map_rows(pair_batch)}
struct TwinSink {buf:*mut i16,a:*const i16,t:*const i16,base:*mut i32,tail:*mut i32}
impl BlockSink for TwinSink {
 #[inline(always)]unsafe fn dst(&mut self,_:usize)->*mut i16 {self.buf}
 #[target_feature(enable="avx512f,avx512bw,avx512vnni")]
 unsafe fn block(&mut self,blk:usize,w:*const i16) {
  let a=self.a.add(32*27*blk);let t=self.t.add(32*27*blk);
  let base=self.base.add(16*ACC_PER_BLK*blk);let tail=self.tail.add(16*ACC_PER_BLK*blk);
  for p in 0..13 {
   let w0=_mm512_load_si512(w.add(64*p) as *const __m512i);
   let w1=_mm512_load_si512(w.add(64*p+32) as *const __m512i);
   let lo=_mm512_shuffle_i64x2::<0x44>(w0,w1);
   let hi=_mm512_shuffle_i64x2::<0xEE>(w0,w1);
   let a0=_mm512_load_si512(a.add(64*p) as *const __m512i);
   let a1=_mm512_load_si512(a.add(64*p+32) as *const __m512i);
   let t0=_mm512_load_si512(t.add(64*p) as *const __m512i);
   let t1=_mm512_load_si512(t.add(64*p+32) as *const __m512i);
   let ba=base.add(16*p) as *mut __m512i;let ta=tail.add(16*p) as *mut __m512i;
   let mut sb=_mm512_load_si512(ba);let mut st=_mm512_load_si512(ta);
   sb=_mm512_dpwssd_epi32(sb,lo,a0);sb=_mm512_dpwssd_epi32(sb,hi,a1);
   st=_mm512_dpwssd_epi32(st,lo,t0);st=_mm512_dpwssd_epi32(st,hi,t1);
   _mm512_store_si512(ba,sb);_mm512_store_si512(ta,st);
  }
  // Slot 26 stands alone in vector 13, with its sum in both lane groups: the row and its
  // 256-bit halves swapped, against the block's row 26 and its halves swapped.
  let w26=_mm512_load_si512(w.add(32*26) as *const __m512i);
  let w26s=_mm512_shuffle_i64x2::<0x4E>(w26,w26);
  let a26=_mm512_load_si512(a.add(32*26) as *const __m512i);
  let t26=_mm512_load_si512(t.add(32*26) as *const __m512i);
  let ba=base.add(16*13) as *mut __m512i;let ta=tail.add(16*13) as *mut __m512i;
  let mut sb=_mm512_load_si512(ba);let mut st=_mm512_load_si512(ta);
  sb=_mm512_dpwssd_epi32(sb,w26,a26);sb=_mm512_dpwssd_epi32(sb,w26s,_mm512_shuffle_i64x2::<0x4E>(a26,a26));
  st=_mm512_dpwssd_epi32(st,w26,t26);st=_mm512_dpwssd_epi32(st,w26s,_mm512_shuffle_i64x2::<0x4E>(t26,t26));
  _mm512_store_si512(ba,sb);_mm512_store_si512(ta,st);
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
 let mut accs:Vec<Box<Acc>>=(0..g*2).map(|_|Acc::zero()).collect();
 let mut tails:Vec<Box<Acc>>=(0..g*2).map(|_|Acc::zero()).collect();
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
   if (b+1)%red_period(P0)==0 {reduce_acc::<P0>(p0);reduce_acc::<P0>(t0);}
   if (b+1)%red_period(P1)==0 {reduce_acc::<P1>(p1);reduce_acc::<P1>(t1);}
  }
  if (b+1)%chunk==0 || b+1==nb {
   for c in 0..g {
    let a=finish::<P0>(&tails[2*c]);let v=finish::<P1>(&tails[2*c+1]);
    for j in 0..648 {partial0.v[j][c]=a[j] as i16;partial1.v[j][c]=v[j] as i16;}
    tails[2*c].v.fill([0;16]);tails[2*c+1].v.fill([0;16]);
   }
   partial0.representation=Representation::Ntt;partial1.representation=Representation::Ntt;
   gen_small::intt_gen_batch32::<P0>(&mut partial0);gen_small::intt_gen_batch32::<P1>(&mut partial1);
   crt_accumulate(&partial0,&partial1,&mut total);
  }
 }
 Out{
  base:(0..g).flat_map(|c|[finish::<P0>(&accs[2*c]),finish::<P1>(&accs[2*c+1])]).collect(),
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
/// The tail key is the NTT of the centered coefficient lift of A mod 1024. Both keys must be
/// in the paired layout of [`pair_key`].
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
 fn paired_sink_matches_pack2() {
  let mut x=0x9e3779b97f4a7c15u64;
  let mut next=||{x^=x<<13;x^=x>>7;x^=x<<17;x};
  let mut key=Batch32::zero(Representation::Ntt);
  for j in 0..N {for p in 0..32 {key.v[j][p]=(next()%3889) as i16-1944;}}
  let paired=pair_batch(&key);
  let mut w=Blk27([0;27*32]);
  for v in w.0.iter_mut() {*v=(next()%3889) as i16-1944;}
  let mut acc_ref=Acc::zero();let mut acc_new=Acc::zero();let mut dummy=Acc::zero();
  unsafe {
   let apf=key.v.as_ptr() as *const i8;
   super::super::mac27::<false>(w.0.as_ptr(),key.v.as_ptr() as *const i16,apf,acc_ref.v.as_mut_ptr() as *mut i32);
   let mut sink=TwinSink{buf:w.0.as_mut_ptr(),a:paired.v.as_ptr() as *const i16,t:paired.v.as_ptr() as *const i16,
    base:acc_new.v.as_mut_ptr() as *mut i32,tail:dummy.v.as_mut_ptr() as *mut i32};
   sink.block(0,w.0.as_ptr());
  }
  assert_eq!(&acc_ref.v[..ACC_PER_BLK],&acc_new.v[..ACC_PER_BLK]);
  assert_eq!(&acc_new.v[..ACC_PER_BLK],&dummy.v[..ACC_PER_BLK]);
 }
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
