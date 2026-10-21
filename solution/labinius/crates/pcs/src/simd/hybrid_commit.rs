//! Commitment modulo 3889 * 9721 * 2^10 using only two witness NTTs.
//!
//! Both transforms feed two public-key streams: the ordinary residue and the centered
//! coefficient lift of A mod 1024. The latter is reconstructed exactly in certified
//! chunks, inverse transformed across columns, and accumulated mod 1024. Only public
//! parameters are precomputed. Every invocation consumes its complete fresh table.
//!
//! # The sink: transpose the transform, not the products
//!
//! A row of the transform is 32 i16 lanes, one per ring element of the batch, and the commitment
//! wants each row's lane sum against two keys. Folding each stream's products down to two lanes
//! per row costs a shuffle tree per stream (30 uops per 8 rows per stream). Instead every group of
//! 8 rows is transposed once at qword granularity (24 shuffles, shared by both streams): vector
//! `q` of the transposed group holds qword `q` (i16 lanes `4q .. 4q + 4`) of row `s` in its qword
//! slot `s`. Against keys stored in the same arrangement ([`KeyBatch`]), eight chained `vpdpwssd`
//! per stream leave every row's whole batch in the two i32 lanes of its slot of one accumulator
//! vector: 40 uops per 8 rows for both streams, the accumulate fused, and the same two-lanes-per-row
//! accumulator footprint as the folded form (6 KB per column-prime-stream), which matters more
//! than uops here: the L1 is shared with the transform's working set and its hyperthread sibling.
//! Rows 24..26 of a 27-row block are one group of four transposed at 128-bit granularity (four
//! lanes per row, a zero-key pad slot).
use super::*;
use crate::simd::ntt::gen_small;
const P0:u16=3889;
const P1:u16=9721;
const PRODUCT:i64=P0 as i64*P1 as i64;
pub const MODULUS:u128=PRODUCT as u128*1024;
pub const CHUNK_BATCHES:usize=7;

/// Arranged key vectors per 27-row block: three 8-row groups of eight qword vectors and one
/// 4-row group of four chunk vectors, each for A and then T.
const KBLK:usize=56;
/// Accumulator vectors per block: base and tail of the four groups.
const ABLK:usize=8;
/// One batch of one prime's two keys, arranged for the sink. Block `blk`, 8-row group `g < 3`:
/// vector `KBLK blk + 16 g + q` holds in its qword slot `s` the i16 lanes `4q .. 4q + 4` of row
/// `27 blk + 8 g + s` of A, vector `.. + 8 + q` the same of T. The 4-row group: vector
/// `KBLK blk + 48 + c` holds in its 128-bit slot `s` the lanes `8c .. 8c + 8` of row
/// `27 blk + 24 + s` of A (zero for `s = 3`), vector `.. + 4 + c` the same of T. 84 KB.
#[repr(C,align(64))]
pub struct KeyBatch {v:[[i16;32];24*KBLK]}
/// One column-prime's accumulators: vector `ABLK blk + 2 g` is the base stream of group `g` of
/// block `blk` (8-row groups: lanes `2s, 2s + 1` sum row `8 g + s`; the 4-row group: lanes
/// `4s .. 4s + 4` sum row `24 + s`), the next one its tail stream. 12 KB.
#[repr(C,align(64))]
struct GAcc {v:[[i32;16];24*ABLK]}

/// What one batch adds to one lane of an 8-row group's accumulator: eight qwords of two
/// products `W * K`, `|W| <= w_bound(q)` (the kernel's output bound) and `|K| <= a_bound(q)`
/// (both keys are stored centered). The 4-row group's lanes gain half of it.
const fn per_batch(q:u16)->i64 {16*w_bound(q)*a_bound(q)}
/// Batches between two fold-backs of both streams: the largest power of two P with
/// `acc_after_reduce + P * per_batch <= i32::MAX` (2 for 3889, 1 for 9721, as for the folded
/// accumulator, whose lanes also gain sixteen products per batch). The tail stream restarts from
/// zero at every chunk, which only shortens its runs.
const fn period(q:u16)->usize {period_for(q,per_batch(q))}
const _:()={
 let qs=[P0,P1];let mut i=0;
 while i<2 {
  let q=qs[i];
  assert!(acc_after_reduce(q)+(period(q) as i64)*per_batch(q)<=i32::MAX as i64);
  // the finish folds every lane back (to at most acc_after_reduce) and sums at most four
  assert!(4*acc_after_reduce(q)<=i32::MAX as i64);
  i+=1;
 }
 assert!(period(P0)==2 && period(P1)==1);
};

/// `acc + x . [k + OFF]` pairwise: one `vpdpwssd` with a memory operand. Written as `asm!`
/// because LLVM splits `vpdpwssd` into `vpmaddwd` + `vpaddd` (twice the uops) whenever the
/// accumulator is on a dependency chain, which here it always is; the displacement is a `const`
/// operand so that the key address needs no `lea` of its own.
#[inline(always)]
unsafe fn dp<const OFF:usize>(acc:__m512i,x:__m512i,k:*const i16)->__m512i {
 let mut r=acc;
 core::arch::asm!("vpdpwssd {r}, {x}, zmmword ptr [{k} + {o}]",r=inout(zmm_reg) r,x=in(zmm_reg) x,k=in(reg) k,o=const OFF,
  options(pure,readonly,nostack,preserves_flags));
 r
}
/// The base and tail accumulations of a transposed group: vector `q` of `x` against arranged
/// key vectors `q` (base) and `T + q` (tail) of the group at `k`.
macro_rules! dp_group {
 ($sb:ident,$st:ident,$x:ident,$k:ident,$t:literal; $($q:literal),*) => {
  $( $sb=dp::<{64*$q}>($sb,$x[$q],$k); $st=dp::<{64*($t+$q)}>($st,$x[$q],$k); )*
 };
}
/// 8 x 8 qword transpose: qword slot `s` of output `q` is qword `q` of `r[s]` (24 shuffles).
#[inline(always)]
unsafe fn tr8(r:&[__m512i;8])->[__m512i;8] {
 // u: [r0.q(2i), r1.q(2i)] / [r0.q(2i+1), r1.q(2i+1)] in 128-bit lane i, and so on
 let u0=_mm512_unpacklo_epi64(r[0],r[1]);let u1=_mm512_unpackhi_epi64(r[0],r[1]);
 let u2=_mm512_unpacklo_epi64(r[2],r[3]);let u3=_mm512_unpackhi_epi64(r[2],r[3]);
 let u4=_mm512_unpacklo_epi64(r[4],r[5]);let u5=_mm512_unpackhi_epi64(r[4],r[5]);
 let u6=_mm512_unpacklo_epi64(r[6],r[7]);let u7=_mm512_unpackhi_epi64(r[6],r[7]);
 let w0=_mm512_shuffle_i64x2::<0x88>(u0,u2);let w1=_mm512_shuffle_i64x2::<0xDD>(u0,u2);
 let w2=_mm512_shuffle_i64x2::<0x88>(u4,u6);let w3=_mm512_shuffle_i64x2::<0xDD>(u4,u6);
 let w4=_mm512_shuffle_i64x2::<0x88>(u1,u3);let w5=_mm512_shuffle_i64x2::<0xDD>(u1,u3);
 let w6=_mm512_shuffle_i64x2::<0x88>(u5,u7);let w7=_mm512_shuffle_i64x2::<0xDD>(u5,u7);
 [_mm512_shuffle_i64x2::<0x88>(w0,w2),_mm512_shuffle_i64x2::<0x88>(w4,w6),
  _mm512_shuffle_i64x2::<0x88>(w1,w3),_mm512_shuffle_i64x2::<0x88>(w5,w7),
  _mm512_shuffle_i64x2::<0xDD>(w0,w2),_mm512_shuffle_i64x2::<0xDD>(w4,w6),
  _mm512_shuffle_i64x2::<0xDD>(w1,w3),_mm512_shuffle_i64x2::<0xDD>(w5,w7)]
}
/// 4 x 4 transpose of 128-bit chunks: slot `s` of output `c` is chunk `c` of `r[s]` (8 shuffles).
#[inline(always)]
unsafe fn tr4(r0:__m512i,r1:__m512i,r2:__m512i,r3:__m512i)->[__m512i;4] {
 let t0=_mm512_shuffle_i64x2::<0x88>(r0,r1);let t1=_mm512_shuffle_i64x2::<0xDD>(r0,r1);
 let t2=_mm512_shuffle_i64x2::<0x88>(r2,r3);let t3=_mm512_shuffle_i64x2::<0xDD>(r2,r3);
 [_mm512_shuffle_i64x2::<0x88>(t0,t2),_mm512_shuffle_i64x2::<0x88>(t1,t3),
  _mm512_shuffle_i64x2::<0xDD>(t0,t2),_mm512_shuffle_i64x2::<0xDD>(t1,t3)]
}

/// The multiply-accumulate sink. With `RED` (the batch that ends a fold-back period) every
/// accumulator vector is folded back ([`reduce_vec`]) on its way back to memory: the same values
/// a separate pass over the column's accumulators after the batch would leave, without that
/// pass's 192 loads and stores. (Software prefetching of the keys, of the next batch into L2 or
/// of the next block into L1, only measured slower: the hardware streamers already follow the
/// sequential key reads, and the extra requests compete for the fill buffers.)
struct TSink<const Q:u16,const RED:bool> {buf:*mut i16,k:*const i16,acc:*mut i32}
impl<const Q:u16,const RED:bool> TSink<Q,RED> {
 #[inline(always)]
 unsafe fn put(p:*mut __m512i,v:__m512i) {
  _mm512_store_si512(p,if RED {reduce_vec::<Q>(v)} else {v});
 }
}
impl<const Q:u16,const RED:bool> BlockSink for TSink<Q,RED> {
 #[inline(always)]unsafe fn dst(&mut self,_:usize)->*mut i16 {self.buf}
 #[target_feature(enable="avx512f,avx512bw")]
 unsafe fn block(&mut self,blk:usize,w:*const i16) {
  let k=self.k.add(32*KBLK*blk);let acc=self.acc.add(16*ABLK*blk);
  for g in 0..3 {
   let mut r=[_mm512_setzero_si512();8];
   for s in 0..8 {r[s]=_mm512_load_si512(w.add(32*(8*g+s)) as *const __m512i);}
   let x=tr8(&r);
   let kg=k.add(32*16*g);
   let pb=acc.add(32*g) as *mut __m512i;let pt=acc.add(32*g+16) as *mut __m512i;
   let mut sb=_mm512_load_si512(pb);let mut st=_mm512_load_si512(pt);
   dp_group!(sb,st,x,kg,8; 0,1,2,3,4,5,6,7);
   Self::put(pb,sb);Self::put(pt,st);
  }
  let r24=_mm512_load_si512(w.add(32*24) as *const __m512i);
  let r25=_mm512_load_si512(w.add(32*25) as *const __m512i);
  let r26=_mm512_load_si512(w.add(32*26) as *const __m512i);
  let x=tr4(r24,r25,r26,r26);
  let kg=k.add(32*48);
  let pb=acc.add(32*3) as *mut __m512i;let pt=acc.add(32*3+16) as *mut __m512i;
  let mut sb=_mm512_load_si512(pb);let mut st=_mm512_load_si512(pt);
  dp_group!(sb,st,x,kg,4; 0,1,2,3);
  Self::put(pb,sb);Self::put(pt,st);
 }
}
/// One column's batch `b` modulo `Q` (`keys[2b]` for 3889, `keys[2b + 1]` for 9721), folded back
/// when it ends a period.
#[inline(always)]
unsafe fn sink_batch<const Q:u16>(idx:&BinaryIndex32,bp:*mut i16,keys:&[KeyBatch],b:usize,acc:*mut i32) {
 let k=keys[2*b+if Q==P0 {0} else {1}].v.as_ptr() as *const i16;
 if (b+1)%period(Q)==0 {
  vb::ntt_bin_batch32_sink::<Q,_>(idx,&mut TSink::<Q,true>{buf:bp,k,acc});
 } else {
  vb::ntt_bin_batch32_sink::<Q,_>(idx,&mut TSink::<Q,false>{buf:bp,k,acc});
 }
}
const EVEN:[i32;16]=[0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30];
const ODD:[i32;16]=[1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31];
/// Lane `8v + i` is the sum of lanes `2i, 2i + 1` of `[a, b][v]`.
#[inline(always)]
unsafe fn hsum2(a:__m512i,b:__m512i)->__m512i {
 let e=_mm512_loadu_si512(EVEN.as_ptr() as *const __m512i);let o=_mm512_loadu_si512(ODD.as_ptr() as *const __m512i);
 _mm512_add_epi32(_mm512_permutex2var_epi32(a,e,b),_mm512_permutex2var_epi32(a,o,b))
}
/// The two-lane partial sums of the 4-row group's slots packed into lanes `2s, 2s + 1`.
const PAIRS:[i32;16]=[0,1,4,5,8,9,12,13,0,0,0,0,0,0,0,0];
/// One stream's (`OFF` 0: base, 16: tail) sums of block `blk`, reduced to [0, q): rows
/// `27 blk .. + 16` and, in lanes 0..11, `27 blk + 16 .. + 27`. With `ZERO` the stream's
/// accumulators are cleared as they are read. Every lane is folded back first, so no partial sum
/// exceeds `4 acc_after_reduce` (asserted above).
#[inline(always)]
unsafe fn block_sums<const Q:u16,const OFF:usize,const ZERO:bool>(acc:*mut i32,blk:usize)->(__m512i,__m512i) {
 let mut v=[_mm512_setzero_si512();4];
 for g in 0..4 {
  let p=acc.add(16*(ABLK*blk+2*g)+OFF) as *mut __m512i;
  v[g]=reduce_vec::<Q>(_mm512_load_si512(p));
  if ZERO {_mm512_store_si512(p,_mm512_setzero_si512());}
 }
 let lo=mod_q::<Q>(hsum2(v[0],v[1]));
 let v3=_mm512_add_epi32(v[3],_mm512_shuffle_epi32::<0x4E>(v[3]));
 let v3=_mm512_permutexvar_epi32(_mm512_loadu_si512(PAIRS.as_ptr() as *const __m512i),v3);
 let hi=mod_q::<Q>(hsum2(v[2],v3));
 (lo,hi)
}
#[target_feature(enable="avx512f,avx512bw")]
unsafe fn finish_base<const Q:u16>(acc:*mut i32)->[u32;N] {
 let mut y=[0u32;N];
 for blk in 0..24 {
  let (lo,hi)=block_sums::<Q,0,false>(acc,blk);
  let o=y.as_mut_ptr().add(27*blk) as *mut i32;
  _mm512_storeu_si512(o as *mut __m512i,lo);
  _mm512_mask_storeu_epi32(o.add(16),0x07ff,hi);
 }
 y
}
/// A column's tail chunk, summed into lane `lane` of `pend` (in [0, q), within the inverse
/// transform's input bound) and cleared for the next chunk.
#[target_feature(enable="avx512f,avx512bw")]
unsafe fn finish_tail<const Q:u16>(acc:*mut i32,pend:&mut Batch32,lane:usize) {
 for blk in 0..24 {
  let (lo,hi)=block_sums::<Q,16,true>(acc,blk);
  let mut v=[0i32;32];
  _mm512_storeu_si512(v.as_mut_ptr() as *mut __m512i,lo);
  _mm512_storeu_si512(v.as_mut_ptr().add(16) as *mut __m512i,hi);
  for r in 0..27 {pend.v[27*blk+r][lane]=v[r] as i16;}
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
/// `n` all-zero `T` on the heap (every type it is used for is valid as zeroes).
fn zeroed_slice<T>(n:usize)->Box<[T]> {
 unsafe {
  let l=std::alloc::Layout::array::<T>(n).unwrap();
  let p=std::alloc::alloc_zeroed(l) as *mut T;
  if p.is_null() {std::alloc::handle_alloc_error(l)}
  Box::from_raw(core::ptr::slice_from_raw_parts_mut(p,n))
 }
}
/// A worker's scratch, allocated once and reused by every group it takes: the accumulators of up
/// to `gmax` columns (`acc[2c + k]`, column `c`, prime `k`), the two batches the finished tail
/// chunks are packed into for the inverse transform, the lane-wise CRT total, the index rows and
/// the kernel's block buffer.
struct Worker {acc:Box<[GAcc]>,pend:Box<[Batch32]>,total:Box<[Batch32]>,idx:Box<[BinaryIndex32]>,buf:Box<[Blk27]>}
impl Worker {
 fn new(gmax:usize)->Self {
  Worker{acc:zeroed_slice(2*gmax),pend:zeroed_slice(2),total:zeroed_slice(1),idx:zeroed_slice(1),buf:zeroed_slice(1)}
 }
}
/// Zero lanes `from..32` of every row of `b`.
#[target_feature(enable="avx512f,avx512bw")]
unsafe fn clear_lanes(b:&mut Batch32,from:usize) {
 let m=!((1u64<<from)-1) as u32;
 for j in 0..N {_mm512_mask_storeu_epi16(b.v[j].as_mut_ptr(),m,_mm512_setzero_si512());}
}
/// One group of `g` columns: the batch is the outer loop, so each key batch is read from L3 once
/// per group and `g` times from L2. A finished tail chunk of column `c` goes to lane `s g + c`
/// of the pending pair, `s` the chunk's slot; the pair is inverse transformed and CRT-combined
/// into the lane-wise total when its `32 / g` slots are full and after the last chunk, and a
/// column's tail is the sum of its lanes of the total. Writes `out[3c .. 3c + 3]`.
#[target_feature(enable="avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vnni,gfni")]
unsafe fn group(cols:&[&[B128]],keys:&[KeyBatch],nb:usize,ws:&mut Worker,out:*mut [u32;N]) {
 let g=cols.len();assert!(g>=1 && 2*g<=ws.acc.len() && keys.len()==2*nb);
 let slots=32/g;
 for a in ws.acc[..2*g].iter_mut() {core::ptr::write_bytes(a as *mut GAcc,0,1);}
 core::ptr::write_bytes(&mut ws.total[0] as *mut Batch32,0,1);
 let bp=ws.buf[0].0.as_mut_ptr();
 let mut filled=0usize;
 for b in 0..nb {
  for c in 0..g {
   cols[c].slice_batch(b,&mut ws.idx[0]);
   let a0=ws.acc[2*c].v.as_mut_ptr() as *mut i32;let a1=ws.acc[2*c+1].v.as_mut_ptr() as *mut i32;
   sink_batch::<P0>(&ws.idx[0],bp,keys,b,a0);
   sink_batch::<P1>(&ws.idx[0],bp,keys,b,a1);
  }
  if (b+1)%CHUNK_BATCHES==0 || b+1==nb {
   let (p0,p1)=ws.pend.split_at_mut(1);
   for c in 0..g {
    finish_tail::<P0>(ws.acc[2*c].v.as_mut_ptr() as *mut i32,&mut p0[0],filled*g+c);
    finish_tail::<P1>(ws.acc[2*c+1].v.as_mut_ptr() as *mut i32,&mut p1[0],filled*g+c);
   }
   filled+=1;
   if filled==slots || b+1==nb {
    // lanes of unfilled slots still hold the previous flush's coefficients
    if filled*g<32 {clear_lanes(&mut p0[0],filled*g);clear_lanes(&mut p1[0],filled*g);}
    p0[0].representation=Representation::Ntt;p1[0].representation=Representation::Ntt;
    gen_small::intt_gen_batch32::<P0>(&mut p0[0]);gen_small::intt_gen_batch32::<P1>(&mut p1[0]);
    crt_accumulate(&p0[0],&p1[0],&mut ws.total[0]);
    filled=0;
   }
  }
 }
 let total=&ws.total[0];
 for c in 0..g {
  out.add(3*c).write(finish_base::<P0>(ws.acc[2*c].v.as_mut_ptr() as *mut i32));
  out.add(3*c+1).write(finish_base::<P1>(ws.acc[2*c+1].v.as_mut_ptr() as *mut i32));
  out.add(3*c+2).write(core::array::from_fn(|j|{
   let mut s=0u16;
   for k in 0..slots {s=s.wrapping_add(total.v[j][k*g+c] as u16);}
   (s&1023) as u32
  }));
 }
}
/// One batch of one prime's base and tail keys in the sink's arrangement ([`KeyBatch`]).
fn arrange(a:&Batch32,t:&Batch32,out:&mut KeyBatch) {
 for blk in 0..24 {
  for (si,key) in [(0usize,a),(1,t)] {
   for g in 0..3 {for q in 0..8 {for s in 0..8 {
    out.v[KBLK*blk+16*g+8*si+q][4*s..4*s+4].copy_from_slice(&key.v[27*blk+8*g+s][4*q..4*q+4]);
   }}}
   for c in 0..4 {for s in 0..4 {
    let dst=&mut out.v[KBLK*blk+48+4*si+c][8*s..8*s+8];
    if s<3 {dst.copy_from_slice(&key.v[27*blk+24+s][8*c..8*c+8]);} else {dst.fill(0);}
   }}
  }
 }
}
fn check_keys(base:&crate::key::CommitmentKey,tail:&crate::key::CommitmentKey) {
 assert_eq!(base.limbs(),2);assert_eq!(tail.limbs(),2);
 assert_eq!((base.prime(0),base.prime(1)),(P0,P1));
 assert_eq!((tail.prime(0),tail.prime(1)),(P0,P1));
 assert_eq!(base.len_f162(),tail.len_f162());
}
/// Bytes of one arranged key batch.
pub const KEY_BATCH_BYTES:usize=core::mem::size_of::<KeyBatch>();
const _:()=assert!(KEY_BATCH_BYTES==24*KBLK*64 && core::mem::align_of::<KeyBatch>()==64);
/// Both keys of every batch in the sink's arrangement as bytes, `keys[2b]` (modulo 3889) and
/// `keys[2b + 1]` (modulo 9721) in order: the public-parameter image a build script embeds so
/// that the prover reads the arranged keys in place ([`keys_from_image`]).
pub fn arranged_image(base:&crate::key::CommitmentKey,tail:&crate::key::CommitmentKey)->Vec<u8> {
 check_keys(base,tail);
 let nb=base.row(0).len();
 let mut out=Vec::with_capacity(2*nb*KEY_BATCH_BYTES);
 let mut kb:Box<[KeyBatch]>=zeroed_slice(1);
 for i in 0..2*nb {
  arrange(&base.row(i%2)[i/2],&tail.row(i%2)[i/2],&mut kb[0]);
  for row in kb[0].v.iter() {for &x in row {out.extend_from_slice(&x.to_le_bytes());}}
 }
 out
}
/// The arranged keys of `nb` batches an [`arranged_image`] holds, read in place (`image` must be
/// 64-byte aligned; `None` for any other length).
pub fn keys_from_image(image:&'static [u8],nb:usize)->Option<&'static [KeyBatch]> {
 if !cfg!(target_endian="little") || image.len()!=2*nb*KEY_BATCH_BYTES || image.as_ptr() as usize%64!=0 {return None;}
 Some(unsafe{core::slice::from_raw_parts(image.as_ptr() as *const KeyBatch,2*nb)})
}
/// Both keys of every batch in the sink's arrangement: `keys[2b]` modulo 3889, `keys[2b + 1]`
/// modulo 9721.
pub fn arrange_keys(base:&crate::key::CommitmentKey,tail:&crate::key::CommitmentKey,workers:usize)->Box<[KeyBatch]> {
 check_keys(base,tail);
 let nb=base.row(0).len();
 let mut keys:Box<[KeyBatch]>=zeroed_slice(2*nb);
 let kp=keys.as_mut_ptr() as usize;
 let tasks=workers.clamp(1,2*nb);
 crate::par::pool().run(tasks,|w|{
  let mut i=w;
  while i<2*nb {
   unsafe{arrange(&base.row(i%2)[i/2],&tail.row(i%2)[i/2],&mut *(kp as *mut KeyBatch).add(i));}
   i+=tasks;
  }
 });
 keys
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
/// Arranges the keys first; [`commit_columns_rows_arranged`] takes them arranged.
pub fn commit_columns_rows(
 base:&crate::key::CommitmentKey, tail:&crate::key::CommitmentKey,
 columns:&[&[B128]],workers:usize,group_size:usize,
)->Vec<[u32;N]> {
 check_keys(base,tail);
 for c in columns {assert_eq!(c.len(),base.len_f162());}
 commit_columns_rows_arranged(&arrange_keys(base,tail,workers),columns,workers,group_size)
}
/// [`commit_columns_rows`] against keys already in the sink's arrangement ([`arrange_keys`],
/// [`keys_from_image`]): `keys[2b]`, `keys[2b + 1]` for each batch of 128 words of a column.
pub fn commit_columns_rows_arranged(keys:&[KeyBatch],columns:&[&[B128]],workers:usize,group_size:usize)->Vec<[u32;N]> {
 assert!(!columns.is_empty(),"at least one witness column");
 let nb=keys.len()/2;
 assert!(nb>0 && keys.len()==2*nb,"two arranged key batches per batch");
 for c in columns {assert_eq!(c.len(),128*nb,"128 words per key batch");}
 let g=group_size.clamp(1,32).min(columns.len());
 let groups=columns.len().div_ceil(g);let workers=workers.clamp(1,groups);
 // Uninitialised: every group writes all 3 rows of each of its columns below, and the groups
 // cover the columns. (`vec![[0u32;N];n]` is not a zeroed allocation for N > 16: it memsets
 // and faults in the 8 MB on this thread, ~3 ms before any worker starts.)
 let mut rows:Vec<[u32;N]>=Vec::with_capacity(columns.len()*3);let dst=rows.as_mut_ptr() as usize;
 let next=std::sync::atomic::AtomicUsize::new(0);
 let placement=crate::par::CorePlacement::for_workers(workers);
 crate::par::pool().run(workers,|worker|{
  let _guard=placement.as_ref().and_then(|p|p.pin(worker));
  let mut ws:Option<Worker>=None;
  loop {
   let gi=next.fetch_add(1,std::sync::atomic::Ordering::Relaxed);
   if gi>=groups {break;}
   let c0=gi*g;let gc=g.min(columns.len()-c0);
   let ws=ws.get_or_insert_with(||Worker::new(g));
   unsafe{group(&columns[c0..c0+gc],&keys,nb,ws,(dst as *mut [u32;N]).add(3*c0))};
  }
 });
 // Every row was written: the groups partition the columns.
 unsafe{rows.set_len(columns.len()*3);}
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
 /// The transposed-group sink against the folded-lane implementation it replaced, bit for bit,
 /// on random centered keys and random columns: 128 batches (fold-backs at both periods,
 /// 19 tail chunks, several full and partial lane flushes), every group size from 1 to 32
 /// that the prover or the flush logic distinguishes, and a ragged last group.
 #[test]
 fn row_sink_matches_folded_reference() {
  use crate::fields::scalar::B128;
  use crate::key::CommitmentKey;
  use crate::ring::Modulus;
  let len=128*128;let ncols=37;
  let base=CommitmentKey::random(len,11,Modulus::Q3889_FS_S,&[Modulus::Q9721_FS_S]);
  let tail=CommitmentKey::random(len,12,Modulus::Q3889_FS_S,&[Modulus::Q9721_FS_S]);
  let mut x=0x9e3779b97f4a7c15u64;
  let words:Vec<B128>=(0..len*ncols).map(|_|{
   let mut w=[0u64;2];
   for v in w.iter_mut() {x^=x<<13;x^=x>>7;x^=x<<17;*v=x;}
   B128(((w[1] as u128)<<64)|w[0] as u128)
  }).collect();
  let cols:Vec<&[B128]>=words.chunks_exact(len).collect();
  let want=reference::commit_columns_rows(&base,&tail,&cols,3,16);
  for g in [1usize,2,3,4,5,6,8,16,32] {
   let got=commit_columns_rows(&base,&tail,&cols,3,g);
   assert!(got==want,"group size {g} differs from the reference");
  }
 }
}
/// The previous sink (products folded to two lanes per row, 6 KB per column-prime-stream), kept
/// as the bit-exact reference of [`tests::row_sink_matches_folded_reference`].
#[cfg(test)]
mod reference {
 use super::*;
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
 const fn period4(q:u16)->usize {period_for(q,4*acc_per_batch(q))}
 #[inline(always)]
 unsafe fn fold4(p0:__m512i,p1:__m512i,p2:__m512i,p3:__m512i)->__m512i {
  let u=pack2(p0,p1);let v=pack2(p2,p3);
  let y=_mm512_permutex2var_epi64(u,_mm512_set_epi64(13,12,7,6,9,8,3,2),v);
  _mm512_add_epi32(_mm512_mask_blend_epi64(0xCC,u,v),y)
 }
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
 #[inline(always)]
 unsafe fn hsum2(a:__m512i,b:__m512i)->__m512i {
  let e=_mm512_loadu_si512(EVEN.as_ptr() as *const __m512i);let o=_mm512_loadu_si512(ODD.as_ptr() as *const __m512i);
  _mm512_add_epi32(_mm512_permutex2var_epi32(a,e,b),_mm512_permutex2var_epi32(a,o,b))
 }
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
 pub fn commit_columns_rows(
  base:&crate::key::CommitmentKey, tail:&crate::key::CommitmentKey,
  columns:&[&[B128]],workers:usize,group_size:usize,
 )->Vec<[u32;N]> {
  let nb=base.row(0).len();let bk=[base.row(0),base.row(1)];let tk=[tail.row(0),tail.row(1)];
  let g=group_size.clamp(1,32).min(columns.len());
  let groups=columns.len().div_ceil(g);let workers=workers.clamp(1,groups);
  let mut rows=vec![[0u32;N];columns.len()*3];let dst=rows.as_mut_ptr() as usize;
  let next=std::sync::atomic::AtomicUsize::new(0);
  crate::par::pool().run(workers,|_|{
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
}
