// Fuse plane gathering with index expansion; keep the original four-bit encoding.
const fn affine_upper()->[u8;64] {let mut a=affine_both();let mut i=0;while i<64 {a[i]+=32;i+=1;}a}
static AFFINE_UPPER:A64<[u8;64]>=A64(affine_upper());
#[inline(always)]
unsafe fn uv_b128_to_rows_fused(wk:*mut Work,out:&mut BinaryIndex32) {
    let u=(*wk).u.as_mut_ptr();let v=(*wk).v.as_mut_ptr();let zero=_mm512_setzero_si512();
    _mm512_store_si512(u.add(128).cast(),zero);_mm512_store_si512(v.add(128).cast(),zero);
    let ident=_mm512_set1_epi64(GF_IDENT);
    let am=_mm512_load_si512(AFFINE_BOTH.0.as_ptr().cast());let ah=_mm512_load_si512(AFFINE_UPPER.0.as_ptr().cast());
    let da=_mm512_load_si512(DUP_A.0.as_ptr().cast());let db=_mm512_load_si512(DUP_B.0.as_ptr().cast());
    let p16=_mm512_load_si512(PLUS16.0.as_ptr().cast());let mask=_mm512_set1_epi8(15);
    let op=out.rows.as_mut_ptr().cast::<__m512i>();
    for pair in 0..2 {
        let (p0,p1,p2,p3,total)=if pair==0 {(u,v.add(40),u.add(81),v.add(121),41)}else{(v,u.add(41),v.add(81),u.add(122),40)};
        for t in (0..total).step_by(8) {
            let a=_mm512_loadu_si512(p0.add(t).cast());let b=_mm512_loadu_si512(p1.add(t).cast());
            let c=_mm512_loadu_si512(p2.add(t).cast());let d=if t==0 {_mm512_loadu_si512(p3.cast())}else{zero};
            let l01=_mm512_unpacklo_epi64(a,b);let h01=_mm512_unpackhi_epi64(a,b);
            let l23=_mm512_unpacklo_epi64(c,d);let h23=_mm512_unpackhi_epi64(c,d);
            let s0=_mm512_shuffle_i64x2::<0x44>(l01,l23);let s1=_mm512_shuffle_i64x2::<0xee>(l01,l23);
            let s2=_mm512_shuffle_i64x2::<0x44>(h01,h23);let s3=_mm512_shuffle_i64x2::<0xee>(h01,h23);
            let r0=_mm512_shuffle_i64x2::<0x88>(s0,s2);let r1=_mm512_shuffle_i64x2::<0xdd>(s0,s2);
            let r2=_mm512_shuffle_i64x2::<0x88>(s1,s3);let r3=_mm512_shuffle_i64x2::<0xdd>(s1,s3);
            macro_rules! emit {($r:expr,$off:expr,$perm:expr)=>{if t+$off<total {
                let n=_mm512_gf2p8affine_epi64_epi8::<0>(ident,_mm512_permutexvar_epi8($perm,$r));
                let a=_mm512_permutexvar_epi8(da,n);let b=_mm512_permutexvar_epi8(db,n);
                let row=4*(t+$off)+2*pair;
                _mm512_store_si512(op.add(row),_mm512_ternarylogic_epi32::<0xea>(a,mask,p16));
                _mm512_store_si512(op.add(row+1),_mm512_ternarylogic_epi32::<0xea>(b,mask,p16));
            }}}
            emit!(r0,0,am);emit!(r0,1,ah);emit!(r1,2,am);emit!(r1,3,ah);
            emit!(r2,4,am);emit!(r2,5,ah);emit!(r3,6,am);emit!(r3,7,ah);
        }
    }
}
