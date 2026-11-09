// Same signed prefix values and tail operations as the packed-binary producer.
// The first pair uses two byte lookups, with a word shift selecting the second.
// The other pairs use joint64-entry word tables holding exact old sums/differences.
// Public key scaling, source bound31296 and all24 ordered callbacks are unchanged.
// The private typed producer requires Q9721 and the fixed512-bit SMS shape;
// the existing generic/full648-bit producers continue to serve their callers.
#[repr(C,align(64))]struct SmsJointPairLut([[[[[i16;64];2];2];4];3]);
const fn sms_joint_pair_lut()->SmsJointPairLut{let old=sms_band_lut();let mut out=[[[[[0i16;64];2];2];4];3];let mut band=0;
while band<3{let mut k=0;while k<4{let mut rr=0;while rr<2{let mut n=0;while n<64{
let x=(n&1)|((n>>1)&2)|((n>>2)&4);let y=((n>>1)&1)|((n>>2)&2)|((n>>3)&4);let tx=12*k+2*(rr+1);let ty=tx+1;
let vx=(old.0[band][tx][x]as u16|((old.0[band][tx][16+x]as u16)<<8))as i16;
let vy=(old.0[band][ty][y]as u16|((old.0[band][ty][16+y]as u16)<<8))as i16;
out[band][k][rr][0][n]=vx+vy;out[band][k][rr][1][n]=if rr==0{vy-vx}else{vx-vy};n+=1;}rr+=1;}k+=1;}band+=1;}SmsJointPairLut(out)}
static SMS_JOINT_PAIR_LUT:SmsJointPairLut=sms_joint_pair_lut();
#[repr(C,align(64))]struct SmsJointR0Lut([[[[u8;64];2];4];3]);
const fn sms_joint_r0_lut()->SmsJointR0Lut{let old=sms_band_lut();let mut out=[[[[0u8;64];2];4];3];let mut band=0;while band<3{let mut k=0;while k<4{let mut i=0;while i<64{
out[band][k][0][i]=old.0[band][12*k][(i&15)+16*((i>>4)&1)];
out[band][k][1][i]=old.0[band][12*k+1][((i>>1)&7)+16*(i&1)];i+=1;}k+=1;}band+=1;}SmsJointR0Lut(out)}
static SMS_JOINT_R0_LUT:SmsJointR0Lut=sms_joint_r0_lut();
#[inline(never)]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vnni,gfni")]
pub(crate) unsafe fn ntt_bin_batch32_sms_source<
    const Q: u16,
    const H: usize,
    const FUSED: bool,
    const SPARSE512:bool,
    S: SliceSink<H>,
    F: FnMut(usize, *mut crate::simd::transpose_f162::SmsBinaryIndex32),
>(
    idx: *mut crate::simd::transpose_f162::SmsBinaryIndex32,
    wbuf: *mut i16,
    sink: &mut S,
    fill: &mut F,
) {
    assert!(Q==9721 && SPARSE512,"private joint producer is only the fixed packed-binary path");
    if !FUSED {
        for h in 0..H {
            fill(h, idx.add(h));
        }
    }
    let t = tables::<Q>();
    let c = C {
        q: bc(&t.qd),
        om: bc(&t.om[0]),
        omp: bc(&t.om[1]),
    };
    let mut bv = bc(&t.bv);
    // Prefix blocks use the callback buffer's block-major layout. A leaf loads
    // all 27 input vectors before storing any output, so its pointers may alias.
    let buf = |kk: usize| -> [*mut i16; H] { core::array::from_fn(|h| wbuf.add(32 * 27 * (H * (kk % 6) + h))) };
    let ro = |w: [*mut i16; H]| -> [*const i16; H] { core::array::from_fn(|h| w[h] as *const i16) };

    for k in 0..4 {
        if FUSED&&k==0{for h in 0..core::hint::black_box(H){fill(h,idx.add(h));}}
        for band in 0..core::hint::black_box(3usize){
            let lut=SMS_JOINT_R0_LUT.0.get_unchecked(band).get_unchecked(k).as_ptr()as *const u8;
            let l=|r:usize,ab:usize|_mm512_load_si512(lut.add(64*(r*2+ab))as *const __m512i);
            let(l000,l001)=(l(0,0),l(0,1));
            let pair_band = SMS_JOINT_PAIR_LUT.0.get_unchecked(band).get_unchecked(k);
            let pair=|r:usize,s:usize,h:usize|_mm512_load_si512(pair_band[r][s].as_ptr().add(32*h).cast());
            let (a10,a11,b10,b11)=(pair(0,0,0),pair(0,0,1),pair(0,1,0),pair(0,1,1));
            let (a20,a21,b20,b21)=(pair(1,0,0),pair(1,0,1),pair(1,1,0),pair(1,1,1));

            if band == 0 {
                for h in 0..core::hint::black_box(H) {
                    let ip = core::ptr::addr_of!((*idx.add(h)).rows).cast::<u8>();
                    let bp = wbuf.add(32 * 27 * h);
                    for i in 0..9 {
                        let nx=ldb(ip,i);let ny=_mm512_srli_epi16::<4>(nx);let n1=ldb(ip,i+27);let n2=ldb(ip,i+54);
                        let x=_mm512_permutexvar_epi8(nx,l000);let y=_mm512_permutexvar_epi8(ny,l001);
                        let a0=_mm512_add_epi16(x,y);let b0=_mm512_sub_epi16(x,y);
                        let a1=_mm512_permutex2var_epi16(a10,n1,a11);let b1=_mm512_permutex2var_epi16(b10,n1,b11);
                        let a2=_mm512_permutex2var_epi16(a20,n2,a21);let b2=_mm512_permutex2var_epi16(b20,n2,b21);

                        let (u0, u1, u2) = r3_folded(&c, a0, a1, a2);
                        let (v0, v1, v2) = r3_folded(&c, b0, b1, b2);
                        st(bp, i, u0);
                        st(bp, i + 27 * H, u1);
                        st(bp, i + 54 * H, u2);
                        st(bp, i + 81 * H, v2);
                        st(bp, i + 108 * H, v0);
                        st(bp, i + 135 * H, v1);
                    }
                }
            } else {
                core::arch::asm!("/* {0} */", inout(zmm_reg) bv, options(pure, nomem, nostack, preserves_flags));
                for h in 0..core::hint::black_box(H) {
                    let ip = core::ptr::addr_of!((*idx.add(h)).rows).cast::<u8>();
                    let bp = wbuf.add(32 * 27 * h);
                    for i in 9*band..9*band+9 {
                        let nx=ldb(ip,i);let ny=_mm512_srli_epi16::<4>(nx);let n1=ldb(ip,i+27);let n2=ldb(ip,i+54);
                        let x=_mm512_permutexvar_epi8(nx,l000);let y=_mm512_permutexvar_epi8(ny,l001);
                        let a0=_mm512_add_epi16(x,y);let b0=_mm512_sub_epi16(x,y);
                        let a1=_mm512_permutex2var_epi16(a10,n1,a11);let b1=_mm512_permutex2var_epi16(b10,n1,b11);
                        let a2=_mm512_permutex2var_epi16(a20,n2,a21);let b2=_mm512_permutex2var_epi16(b20,n2,b21);

                        let (mut u0, u1, u2) = r3_folded(&c, a0, a1, a2);
                        let (mut v0, v1, v2) = r3_folded(&c, b0, b1, b2);
                        u0=_mm512_sub_epi16(u0,_mm512_mullo_epi16(_mm512_mulhrs_epi16(u0,bv),c.q));
                        v0=_mm512_sub_epi16(v0,_mm512_mullo_epi16(_mm512_mulhrs_epi16(v0,bv),c.q));
                        st(bp, i, u0);
                        st(bp, i + 27 * H, u1);
                        st(bp, i + 54 * H, u2);
                        st(bp, i + 81 * H, v2);
                        st(bp, i + 108 * H, v0);
                        st(bp, i + 135 * H, v1);
                    }
                }
            }
        }
        let cv = t.cv.as_ptr() as *const i16;
        let tail_blocks = core::hint::black_box(6usize);
        for j in 0..tail_blocks {
            let kk = 6 * k + j;
            let (t4, t5, t6) = if Q == 9721 {
                (SMS_BAND_T4.0.as_ptr().add(4 * kk).cast::<u32>(),
                SMS_TWIDDLES.t5.as_ptr().add(12 * kk).cast::<u32>(),
                SMS_TWIDDLES.t6.as_ptr().add(36 * kk).cast::<u32>())
            } else { (
                t.tw4.as_ptr().add(kk).cast::<u32>(),
                t.tw5.as_ptr().add(3 * kk).cast::<u32>(),
                t.tw6.as_ptr().add(9 * kk).cast::<u32>(),
                ) };
            for h in 0..core::hint::black_box(H) {
                let bpj = wbuf.add(32 * 27 * (H * j + h));
                if Q==9721 {if SPARSE512 {if j==0{tail27_sms_band0_leaf(bpj,bpj,t4,t5,t6,cv);}else if j==4{tail27_sms_band4_leaf(bpj,bpj,t4,t5,t6,cv);}else{tail27_scaled_leaf(bpj,bpj,t4,t5,t6,cv);}}else{tail27_relocated_leaf(bpj,bpj,t4,t5,t6,cv);}} else {tail27_out_of_line::<Q>(bpj, bpj, t4, t5, t6, cv);}
            }
        }
        for j in 0..6 {
            let kk = 6 * k + j;
            sink.rows(kk, ro(buf(kk)), 0, 27);
        }
    }
}




#[cfg(test)]mod sms_joint_source_guards{use super::*;use crate::fields::scalar::B128;use crate::simd::transpose_f162::{SmsBinaryIndex32,slice_b128_cols16_into,slice_b128_cols16_sms_into};
#[repr(C,align(64))]struct Out([i16;648*32]);
struct Collect<const H:usize>{output:Vec<Out>,next:usize}
impl<const H:usize>SliceSink<H>for Collect<H>{unsafe fn rows(&mut self,blk:usize,w:[*const i16;H],r0:usize,r1:usize){assert_eq!(blk,self.next);self.next+=1;assert_eq!((r0,r1),(0,27));for h in 0..H{for i in r0..r1{core::ptr::copy_nonoverlapping(w[h].add(32*i),self.output[h].0.as_mut_ptr().add((27*blk+i)*32),32);}}}}
unsafe fn check<const H:usize,const FUSED:bool>(){
 let poison=_mm512_set1_epi32(0x1759acf3);let mut state=0x531789fdd8712ac3u128;
 for fixture in 0..20{
  let mut old:Vec<BinaryIndex32>=(0..H).map(|_|BinaryIndex32{rows:[[0;64];162]}).collect();let mut joint:Vec<SmsBinaryIndex32>=(0..H).map(|_|SmsBinaryIndex32{rows:[[0;32];81]}).collect();
  for h in 0..H{let mut columns=[[B128(0);8];16];for lane in 0..32{let col=lane/2;let e=lane%2;if fixture<16{let bit=fixture*32+lane;columns[col][4*e+bit%4].0=1u128<<(bit/4);}else{for word in 0..4{state^=state<<13;state^=state>>7;state^=state<<17;columns[col][4*e+word].0=match fixture{16=>0,17=>u128::MAX,18=>if (h+lane+word)&1==0{u128::MAX}else{0},_=>state};}}}
   let ptrs=core::array::from_fn(|i|columns[i].as_ptr());slice_b128_cols16_into(&ptrs,&mut old[h]);slice_b128_cols16_sms_into(&ptrs,&mut joint[h]);
  }
  let mut ib:Vec<BinaryIndex32>=(0..H).map(|_|BinaryIndex32{rows:[[0x75;64];162]}).collect();let mut ij:Vec<SmsBinaryIndex32>=(0..H).map(|_|SmsBinaryIndex32{rows:[[0x7531;32];81]}).collect();
  let mut wb=vec![poison;6*H*27+2];let mut wj=vec![poison;6*H*27+2];let mut expected=Collect::<H>{output:(0..H).map(|_|Out([0;648*32])).collect(),next:0};let mut got=Collect::<H>{output:(0..H).map(|_|Out([0;648*32])).collect(),next:0};
  let mut bnext=0;let mut jnext=0;
  ntt_bin_batch32_batched_source::<9721,H,FUSED,true,_,_>(ib.as_mut_ptr(),wb.as_mut_ptr().add(1).cast(),&mut expected,&mut |h,out|{assert_eq!(h,bnext);bnext+=1;core::ptr::copy_nonoverlapping(&old[h],out,1);});
  ntt_bin_batch32_sms_source::<9721,H,FUSED,true,_,_>(ij.as_mut_ptr(),wj.as_mut_ptr().add(1).cast(),&mut got,&mut |h,out|{assert_eq!(h,jnext);jnext+=1;core::ptr::copy_nonoverlapping(&joint[h],out,1);});
  assert_eq!((bnext,jnext,expected.next,got.next),(H,H,24,24));for h in 0..H{assert_eq!(got.output[h].0,expected.output[h].0,"fixture={fixture} H={H} fused={FUSED} h={h}");}
  for w in[&wb,&wj]{assert_eq!(_mm512_cmpeq_epi32_mask(w[0],poison),u16::MAX);assert_eq!(_mm512_cmpeq_epi32_mask(w[w.len()-1],poison),u16::MAX);}
 }
}
#[test]fn sms_joint_source_all512basis_exact_original_words(){unsafe{check::<1,true>();check::<2,true>();check::<3,true>();check::<4,true>();check::<8,true>();check::<16,true>();check::<1,false>();check::<2,false>();check::<3,false>();check::<4,false>();check::<8,false>();check::<16,false>();}}
}
