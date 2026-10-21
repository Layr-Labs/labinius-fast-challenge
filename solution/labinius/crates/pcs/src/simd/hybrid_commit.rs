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
struct TwinSink {buf:*mut i16,a:*const i16,t:*const i16,base:*mut i32,tail:*mut i32}
impl BlockSink for TwinSink {
 #[inline(always)]unsafe fn dst(&mut self,_:usize)->*mut i16 {self.buf}
 #[target_feature(enable="avx512f,avx512bw")]
 unsafe fn block(&mut self,blk:usize,w:*const i16) {
  for p in 0..14 {
   let j0=2*p;let j1=(j0+1).min(26);
   let w0=_mm512_load_si512(w.add(32*j0) as *const __m512i);
   let w1=_mm512_load_si512(w.add(32*j1) as *const __m512i);
   let a0=_mm512_load_si512(self.a.add(32*(27*blk+j0)) as *const __m512i);
   let a1=_mm512_load_si512(self.a.add(32*(27*blk+j1)) as *const __m512i);
   let t0=_mm512_load_si512(self.t.add(32*(27*blk+j0)) as *const __m512i);
   let t1=_mm512_load_si512(self.t.add(32*(27*blk+j1)) as *const __m512i);
   let ba=self.base.add(16*(ACC_PER_BLK*blk+p)) as *mut __m512i;
   let ta=self.tail.add(16*(ACC_PER_BLK*blk+p)) as *mut __m512i;
   let bp=pack2(_mm512_madd_epi16(w0,a0),_mm512_madd_epi16(w1,a1));
   let tp=pack2(_mm512_madd_epi16(w0,t0),_mm512_madd_epi16(w1,t1));
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

// =============================================================================================
// column-pair layout: lanes are 16 columns x 2 consecutive ring elements
// =============================================================================================
//
// Lane 2l+e of a transform batch is ring element 2b+e of column l of a 16-column group, so the
// two lanes `vpmaddwd` pairs belong to the same column and multiply the two key elements
// (A_{2b}, A_{2b+1}), which are the same for every column: one `vpdpwssd` with a broadcast dword
// per slot and key accumulates 16 columns at once, with no lane folding at all.

/// The keys of both primes and both streams in the pair layout, one flat buffer of four
/// streams: stream `s = 2*prime + (0 for A, 1 for the 1024 lift)` holds, for pair batch `b` and
/// slot `j`, the dword `(A_{2b}[j], A_{2b+1}[j])` at `s * np * 648 + 648 b + j`, where `np` is
/// the number of pair batches of a column.
pub fn pair_keys(base:&crate::key::CommitmentKey,tail:&crate::key::CommitmentKey,workers:usize)->Vec<u32> {
 let rows=[base.row(0),tail.row(0),base.row(1),tail.row(1)];
 let nb=rows[0].len();let np=16*nb;
 let mut k:Vec<u32>=Vec::with_capacity(4*np*648);
 unsafe{k.set_len(4*np*648);}
 let addr=k.as_mut_ptr() as usize;
 let runs=crate::par::ranges(4*nb,workers.max(1),1);
 crate::par::pool().run(runs.len(),|r|{
  for t in runs[r].clone() {
   let (s,b)=(t/nb,t%nb);let src=&rows[s][b];
   let dst=(addr as *mut u32).wrapping_add(s*np*648);
   for h in 0..16 {
    for j in 0..648 {
     let lo=src.v[j][2*h] as u16 as u32;let hi=src.v[j][2*h+1] as u16 as u32;
     unsafe{dst.add((16*b+h)*648+j).write(lo|(hi<<16));}
    }
   }
  }
 });
 k
}

/// One dword-product step bound: two products of a lazily reduced transform lane and a centered
/// key lane.
const fn pair_per_batch(q:u16)->i64 {2*w_bound(q)*a_bound(q)}
/// Pair batches between two fold-backs: the largest power of two P with
/// `acc_after_reduce + P * pair_per_batch <= i32::MAX`.
const fn pair_period(q:u16)->usize {
 let mut p=1usize;
 while acc_after_reduce(q)+2*(p as i64)*pair_per_batch(q)<=i32::MAX as i64 {p*=2;}
 p
}
const PP0:usize=pair_period(P0);const PP1:usize=pair_period(P1);
const _:()=assert!(acc_after_reduce(P0)+(PP0 as i64)*pair_per_batch(P0)<=i32::MAX as i64);
const _:()=assert!(acc_after_reduce(P1)+(PP1 as i64)*pair_per_batch(P1)<=i32::MAX as i64);
/// Pair batches per certified chunk (the chunk is `CHUNK_BATCHES * 32` ring elements).
const CHUNK_PAIRS:usize=CHUNK_BATCHES*16;
const _:()=assert!(CHUNK_PAIRS%PP0==0 && CHUNK_PAIRS%PP1==0);

#[repr(C,align(64))]
pub struct PAcc { pub v:[[i32;16];648] }
impl PAcc {
 fn zero()->Box<PAcc> {
  unsafe{let mut b=Box::<PAcc>::new_uninit();core::ptr::write_bytes(b.as_mut_ptr() as *mut u8,0,core::mem::size_of::<PAcc>());b.assume_init()}
 }
}

struct PairSink {buf:*mut i16,ka:*const u32,kt:*const u32,aa:*mut i32,at:*mut i32}
impl BlockSink for PairSink {
 #[inline(always)]unsafe fn dst(&mut self,_:usize)->*mut i16 {self.buf}
 #[target_feature(enable="avx512f,avx512bw,avx512vnni")]
 unsafe fn block(&mut self,blk:usize,w:*const i16) {
  let s0=27*blk;
  for j in 0..27 {
   let wv=_mm512_load_si512(w.add(32*j) as *const __m512i);
   let pa=self.aa.add(16*(s0+j)) as *mut __m512i;
   let pt=self.at.add(16*(s0+j)) as *mut __m512i;
   let ka=_mm512_set1_epi32(*self.ka.add(s0+j) as i32);
   let kt=_mm512_set1_epi32(*self.kt.add(s0+j) as i32);
   _mm512_store_si512(pa,_mm512_dpwssd_epi32(_mm512_load_si512(pa),wv,ka));
   _mm512_store_si512(pt,_mm512_dpwssd_epi32(_mm512_load_si512(pt),wv,kt));
  }
 }
}
#[target_feature(enable="avx512f,avx512bw")]
unsafe fn reduce_pacc<const Q:u16>(a:&mut PAcc) {
 let p=a.v.as_mut_ptr() as *mut __m512i;
 for j in 0..648 {_mm512_store_si512(p.add(j),reduce_vec::<Q>(_mm512_load_si512(p.add(j))));}
}
/// Slot sums in [0, q) as i16 into lanes 0..16 of `out` (lanes 16..32 are left as they are).
#[target_feature(enable="avx512f,avx512bw")]
unsafe fn pacc_to_batch<const Q:u16>(a:&PAcc,out:&mut Batch32) {
 let p=a.v.as_ptr() as *const __m512i;
 for j in 0..648 {
  let r=mod_q::<Q>(_mm512_load_si512(p.add(j)));
  _mm256_store_si256(out.v[j].as_mut_ptr() as *mut __m256i,_mm512_cvtepi32_epi16(r));
 }
}
struct PairOut {base:Vec<[[u32;N];2]>,tail:Vec<[u16;N]>}
#[target_feature(enable="avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vnni,gfni")]
unsafe fn group_pairs(cols:&[&[B128]],keys:&[u32],np:usize)->PairOut {
 let g=cols.len();assert!(g<=16);
 let mut acc:[Box<PAcc>;4]=core::array::from_fn(|_|PAcc::zero());
 let mut partial0=Batch32::zero(Representation::Ntt);let mut partial1=Batch32::zero(Representation::Ntt);
 let mut total=Batch32::zero(Representation::Coefficients);
 let mut idx=BinaryIndex32::zero();let mut buf=Blk27([0;27*32]);let bp=buf.0.as_mut_ptr();
 #[repr(C,align(64))] struct Staged([F162;128]);
 let mut staged=Staged([F162::ZERO;128]);
 let ka:[*const u32;4]=core::array::from_fn(|s|keys.as_ptr().add(s*np*648));
 for b in 0..np {
  for l in 0..g {
   let src=cols[l].as_ptr().add(8*b);
   for m in 0..8 {staged.0[8*l+m]=F162::from_b128(*src.add(m));}
  }
  crate::simd::transpose_f162::slice_f162_into(&staged.0,&mut idx);
  let (a0,t0,a1,t1)={let [x,y,z,w]=&mut acc;(x.v.as_mut_ptr() as *mut i32,y.v.as_mut_ptr() as *mut i32,z.v.as_mut_ptr() as *mut i32,w.v.as_mut_ptr() as *mut i32)};
  vb::ntt_bin_batch32_sink::<P0,_>(&idx,&mut PairSink{buf:bp,ka:ka[0].add(648*b),kt:ka[1].add(648*b),aa:a0,at:t0});
  vb::ntt_bin_batch32_sink::<P1,_>(&idx,&mut PairSink{buf:bp,ka:ka[2].add(648*b),kt:ka[3].add(648*b),aa:a1,at:t1});
  if (b+1)%PP0==0 {reduce_pacc::<P0>(&mut acc[0]);reduce_pacc::<P0>(&mut acc[1]);}
  if (b+1)%PP1==0 {reduce_pacc::<P1>(&mut acc[2]);reduce_pacc::<P1>(&mut acc[3]);}
  if (b+1)%CHUNK_PAIRS==0 || b+1==np {
   pacc_to_batch::<P0>(&acc[1],&mut partial0);pacc_to_batch::<P1>(&acc[3],&mut partial1);
   acc[1].v.fill([0;16]);acc[3].v.fill([0;16]);
   partial0.representation=Representation::Ntt;partial1.representation=Representation::Ntt;
   gen_small::intt_gen_batch32::<P0>(&mut partial0);gen_small::intt_gen_batch32::<P1>(&mut partial1);
   crt_accumulate(&partial0,&partial1,&mut total);
  }
 }
 let mut base=vec![[[0u32;N];2];g];
 let mut tmp=Batch32::zero(Representation::Ntt);
 for (k,a) in [(0usize,&acc[0]),(1,&acc[2])] {
  if k==0 {pacc_to_batch::<P0>(a,&mut tmp);} else {pacc_to_batch::<P1>(a,&mut tmp);}
  for j in 0..N {for l in 0..g {base[l][k][j]=tmp.v[j][l] as u32;}}
 }
 PairOut{base,tail:(0..g).map(|l|core::array::from_fn(|j|(total.v[j][l] as u16)&1023)).collect()}
}

/// [`commit_columns_rows`] in the column-pair layout: groups of 16 columns, one pass over the
/// pair keys per group.
pub fn commit_columns_rows_pairs(keys:&[u32],columns:&[&[B128]],workers:usize)->Vec<[u32;N]> {
 assert!(!columns.is_empty(),"at least one witness column");
 let len=columns[0].len();for c in columns {assert_eq!(c.len(),len);}
 assert_eq!(len%8,0);let np=len/8;
 assert_eq!(keys.len(),4*np*648);
 let g=16;let groups=columns.len().div_ceil(g);let workers=workers.clamp(1,groups);
 let mut rows=vec![[0u32;N];columns.len()*3];let dst=rows.as_mut_ptr() as usize;
 let next=std::sync::atomic::AtomicUsize::new(0);
 let placement=crate::par::CorePlacement::for_workers(workers);
 crate::par::pool().run(workers,|worker|{
  let _guard=placement.as_ref().and_then(|p|p.pin(worker));
  loop {
   let gi=next.fetch_add(1,std::sync::atomic::Ordering::Relaxed);
   if gi>=groups {break;}
   let c0=gi*g;let gc=g.min(columns.len()-c0);
   let out=unsafe{group_pairs(&columns[c0..c0+gc],keys,np)};
   for c in 0..gc {unsafe{
    let p=(dst as *mut [u32;N]).add((c0+c)*3);
    p.write(out.base[c][0]);p.add(1).write(out.base[c][1]);
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
