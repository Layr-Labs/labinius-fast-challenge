// A batched binary NTT with an input factory. Each input's first prefix can
// consume its newly converted indices while they are still warm in L1.
// The prefix shares the callback scratch; tail arithmetic and all 24 callbacks
// are unchanged. Each leaf reads its entire input before its first store.
// Keeping this producer out of line prevents duplication in the SMS worker.
//
// Safety: idx addresses H aligned BinaryIndex32 slots, and wbuf addresses
// 6 * H * 27 * 32 aligned i16 slots. These regions must not overlap. The
// factory must initialize the entire supplied input slot before returning.
// It is called once for each h, in increasing order. FUSED selects whether
// the first prefix follows each fill or all fills precede the transform.
// Both orders perform all input conversion and all transform operations.

#[inline(never)]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vnni,gfni")]
pub(crate) unsafe fn ntt_bin_batch32_batched_source<
    const Q: u16,
    const H: usize,
    const FUSED: bool,
    S: SliceSink<H>,
    F: FnMut(usize, *mut BinaryIndex32),
>(
    idx: *mut BinaryIndex32,
    wbuf: *mut i16,
    sink: &mut S,
    fill: &mut F,
) {
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
    let ips: [*const u8; H] = core::array::from_fn(|h| {
        core::ptr::addr_of!((*idx.add(h)).rows).cast::<u8>()
    });
    // Prefix blocks use the callback buffer's block-major layout. A leaf loads
    // all 27 input vectors before storing any output, so its pointers may alias.
    let bps: [*mut i16; H] = core::array::from_fn(|h| wbuf.add(32 * 27 * h));
    let buf = |kk: usize| -> [*mut i16; H] { core::array::from_fn(|h| wbuf.add(32 * 27 * (H * (kk % 6) + h))) };
    let ro = |w: [*mut i16; H]| -> [*const i16; H] { core::array::from_fn(|h| w[h] as *const i16) };

    for k in 0..4 {
        if FUSED && k == 0 {
            for h in 0..core::hint::black_box(H) {
                fill(h, idx.add(h));
                let lut = t.lut.as_ptr().add(12 * k) as *const u8;
                let l = |s2: usize, r: usize, ab: usize| -> __m512i {
                    _mm512_load_si512(lut.add(64 * (((s2 * 3) + r) * 2 + ab)) as *const __m512i)
                };
                let (l000, l001) = (l(0, 0, 0), l(0, 0, 1));
                let (l010, l011) = (l(0, 1, 0), l(0, 1, 1));
                let (l020, l021) = (l(0, 2, 0), l(0, 2, 1));
                // Opaque trip counts keep one copy of each loop body: unrolled, the eight inlined tails
                // and prefixes outgrow the instruction cache and the decoded-uop cache.

                let (ip, bp) = (ips[h], bps[h]);
                for i in 0..27 {
                    let n0 = ldb(ip, i);
                    let n0h = ldb(ip, i + 81);
                    let n1 = ldb(ip, i + 27);
                    let n1h = ldb(ip, i + 108);
                    let n2 = ldb(ip, i + 54);
                    let n2h = ldb(ip, i + 135);

                    let x = _mm512_permutexvar_epi8(n0, l000);
                    let y = _mm512_permutexvar_epi8(n0h, l001);
                    let a0 = _mm512_add_epi16(x, y);
                    let b0 = _mm512_sub_epi16(x, y);

                    let x = _mm512_permutexvar_epi8(n1, l010);
                    let y = _mm512_permutexvar_epi8(n1h, l011);
                    let a1 = _mm512_add_epi16(x, y);
                    let b1 = _mm512_sub_epi16(y, x);

                    let x = _mm512_permutexvar_epi8(n2, l020);
                    let y = _mm512_permutexvar_epi8(n2h, l021);
                    let a2 = _mm512_add_epi16(x, y);
                    let b2 = _mm512_sub_epi16(x, y);

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
        }else{
            let lut = t.lut.as_ptr().add(12 * k) as *const u8;
            let l = |s2: usize, r: usize, ab: usize| -> __m512i {
                _mm512_load_si512(lut.add(64 * (((s2 * 3) + r) * 2 + ab)) as *const __m512i)
            };
            let (l000, l001) = (l(0, 0, 0), l(0, 0, 1));
            let (l010, l011) = (l(0, 1, 0), l(0, 1, 1));
            let (l020, l021) = (l(0, 2, 0), l(0, 2, 1));
            // Opaque trip counts keep one copy of each loop body: unrolled, the eight inlined tails
            // and prefixes outgrow the instruction cache and the decoded-uop cache.
            let hs = core::hint::black_box(H);
            for h in 0..hs {
                let (ip, bp) = (ips[h], bps[h]);
                for i in 0..27 {
                    let n0 = ldb(ip, i);
                    let n0h = ldb(ip, i + 81);
                    let n1 = ldb(ip, i + 27);
                    let n1h = ldb(ip, i + 108);
                    let n2 = ldb(ip, i + 54);
                    let n2h = ldb(ip, i + 135);

                    let x = _mm512_permutexvar_epi8(n0, l000);
                    let y = _mm512_permutexvar_epi8(n0h, l001);
                    let a0 = _mm512_add_epi16(x, y);
                    let b0 = _mm512_sub_epi16(x, y);

                    let x = _mm512_permutexvar_epi8(n1, l010);
                    let y = _mm512_permutexvar_epi8(n1h, l011);
                    let a1 = _mm512_add_epi16(x, y);
                    let b1 = _mm512_sub_epi16(y, x);

                    let x = _mm512_permutexvar_epi8(n2, l020);
                    let y = _mm512_permutexvar_epi8(n2h, l021);
                    let a2 = _mm512_add_epi16(x, y);
                    let b2 = _mm512_sub_epi16(x, y);

                    let (u0, u1, u2) = r3_folded(&c, a0, a1, a2);
                    let (v0, v1, v2) = r3_folded(&c, b0, b1, b2);
                    st(bp, i, u0);
                    st(bp, i + 27 * H, u1);
                    st(bp, i + 54 * H, u2);
                    st(bp, i + 81 * H, v2);
                    st(bp, i + 108 * H, v0);
                    st(bp, i + 135 * H, v1);
                }
            }        }

        let cv = t.cv.as_ptr() as *const i16;
        let tail_blocks = core::hint::black_box(6usize);
        for j in 0..tail_blocks {
            let kk = 6 * k + j;
            let os = buf(kk);
            let (t4, t5, t6) = if Q == 9721 {
                (SMS_TWIDDLES.t4.as_ptr().add(4 * kk).cast::<u32>(),
                SMS_TWIDDLES.t5.as_ptr().add(12 * kk).cast::<u32>(),
                SMS_TWIDDLES.t6.as_ptr().add(36 * kk).cast::<u32>())
            } else { (
                t.tw4.as_ptr().add(kk).cast::<u32>(),
                t.tw5.as_ptr().add(3 * kk).cast::<u32>(),
                t.tw6.as_ptr().add(9 * kk).cast::<u32>(),
                ) };
            for h in 0..core::hint::black_box(H) {
                let bpj = os[h];
                tail27_out_of_line::<Q>(bpj, os[h], t4, t5, t6, cv);
            }
        }
        for j in 0..6 {
            let kk = 6 * k + j;
            sink.rows(kk, ro(buf(kk)), 0, 27);
        }
    }
}


#[cfg(test)]mod sourced_batched_tests{use super::*;#[repr(C,align(64))]struct Out([i16;648*32]);struct Collect<const H:usize>(Vec<Out>);impl<const H:usize>SliceSink<H>for Collect<H>{unsafe fn rows(&mut self,blk:usize,w:[*const i16;H],r0:usize,r1:usize){for h in 0..H{for i in r0..r1{core::ptr::copy_nonoverlapping(w[h].add(32*i),self.0[h].0.as_mut_ptr().add((27*blk+i)*32),32);}}}}
unsafe fn collect<const Q:u16,const H:usize,const MODE:usize>(ins:&[BinaryIndex32])->Vec<Out>{let mut out=Collect::<H>((0..H).map(|_|Out([0;648*32])).collect());let poison=_mm512_set1_epi32(0x5a196ab3);let mut w=vec![poison;6*H*27+2];let mut idx:Vec<BinaryIndex32>=(0..H).map(|_|BinaryIndex32::zero()).collect();let mut count=[0usize;H];let mut fill=|h:usize,dst:*mut BinaryIndex32|{count[h]+=1;core::ptr::copy_nonoverlapping(&ins[h],dst,1);};if MODE==0{let refs=core::array::from_fn(|h|&ins[h]);ntt_bin_batch32_batched::<Q,H,_>(refs,w.as_mut_ptr().add(1).cast(),&mut out);}else if MODE==1{ntt_bin_batch32_batched_source::<Q,H,false,_,_>(idx.as_mut_ptr(),w.as_mut_ptr().add(1).cast(),&mut out,&mut fill);}else{ntt_bin_batch32_batched_source::<Q,H,true,_,_>(idx.as_mut_ptr(),w.as_mut_ptr().add(1).cast(),&mut out,&mut fill);}if MODE!=0{assert!(count.iter().all(|&v|v==1));}assert_eq!(_mm512_cmpeq_epi32_mask(w[0],poison),65535);assert_eq!(_mm512_cmpeq_epi32_mask(w[w.len()-1],poison),65535);out.0}
unsafe fn check<const Q:u16,const H:usize>(){let mut state=0xc21843ea937u64;for fixture in 0..10{let mut inputs:Vec<BinaryIndex32>=(0..H).map(|_|BinaryIndex32::zero()).collect();for(h,input)in inputs.iter_mut().enumerate(){for i in 0..162{for lane in 0..32{let mut n=0;for bit in 0..4{let coef=i+162*bit;state^=state<<13;state^=state>>7;state^=state<<17;let set=match fixture{0=>false,1=>true,2=>(coef+lane+h)%2==0,3=>coef==0,4=>coef==511,5=>coef==(lane+13*h)%512,_=>state&1!=0};if coef<512&&set{n|=1<<bit;}}input.rows[i][2*lane]=n;input.rows[i][2*lane+1]=n+16;}}}let a=collect::<Q,H,0>(&inputs);for(mode,b)in[(1,collect::<Q,H,1>(&inputs)),(2,collect::<Q,H,2>(&inputs))]{for h in 0..H{assert_eq!(a[h].0,b[h].0,"q={Q} h={H} fixture={fixture} batch={h} mode={mode}");}}}}
#[test]fn sourced_batched_matches_original_with_guards(){unsafe{check::<3889,1>();check::<3889,2>();check::<3889,4>();check::<3889,8>();check::<3889,16>();check::<9721,1>();check::<9721,2>();check::<9721,4>();check::<9721,8>();check::<9721,16>();}}
}
