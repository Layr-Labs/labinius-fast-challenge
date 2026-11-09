// Packed-binary SMS index only: 32 ring lanes, each with512 Boolean coefficients.
// Rows0..27 encode x bits0..3, byte selector bit4 and y bits5..7; both
// byte selectors share one word. Rows27..81 contain six-bit pair indices.
// All81 rows are written before a producer reads them. The general F162 and
// full648-bit constructors retain their original representation and paths.
#[repr(C,align(64))]pub(crate)struct SmsBinaryIndex32{pub(crate)rows:[[u16;32];81]}
const fn sms_joint_dup(high:usize)->[u8;64]{let mut a=[0;64];let mut i=0;while i<64{a[i]=(high+i/2)as u8;i+=1;}a}
static SMS_JOINT_DUP0:A64<[u8;64]>=A64(sms_joint_dup(0));static SMS_JOINT_DUP1:A64<[u8;64]>=A64(sms_joint_dup(32));
const fn sms_joint_perm512(odd:usize)->[u8;64]{let mut p=[0;64];let mut half=0;while half<2{let mut g=0;while g<4{let mut j=0;while j<8{p[half*32+g*8+j]=(half*32+4*((7-j)^odd)+g)as u8;j+=1;}g+=1;}half+=1;}p}
static SMS_JOINT_PERM_EVEN:A64<[u8;64]>=A64(sms_joint_perm512(0));
static SMS_JOINT_PERM_ODD:A64<[u8;64]>=A64(sms_joint_perm512(1));
#[repr(C, align(64))]
struct SmsWork {
    u: [u64; 128],
    v: [u64; 128],
}

#[inline(never)]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,gfni")]
pub(crate) unsafe fn slice_b128_cols16_sms_into(
    cols: &[*const crate::fields::scalar::B128; 16],
    out: &mut SmsBinaryIndex32,
) {
    let mut work: MaybeUninit<SmsWork> = MaybeUninit::uninit();
    let wk = work.as_mut_ptr();
    let u = core::ptr::addr_of_mut!((*wk).u).cast::<u64>();
    let v = core::ptr::addr_of_mut!((*wk).v).cast::<u64>();
    let mut g: MaybeUninit<[[__m512i; 16]; 2]> = MaybeUninit::uninit();
    let gp = g.as_mut_ptr() as *mut [__m512i; 16];
    for h in 0..2 {
        let mut r: [__m512i; 16] = core::array::from_fn(|row| {
            let lane = 16 * h + ROW_LANE[row];
            _mm512_loadu_si512(cols[lane >> 1].add(4 * (lane & 1)) as *const __m512i)
        });
        macro_rules! stage {
            ($lo:ident, $hi:ident) => {{
                let n: [__m512i; 16] = core::array::from_fn(|j| {
                    let (x, y) = (r[j / 2], r[j / 2 + 8]);
                    if j % 2 == 0 { $lo(x, y) } else { $hi(x, y) }
                });
                r = n;
            }};
        }
        stage!(_mm512_unpacklo_epi8, _mm512_unpackhi_epi8);
        stage!(_mm512_unpacklo_epi16, _mm512_unpackhi_epi16);
        stage!(_mm512_unpacklo_epi32, _mm512_unpackhi_epi32);
        stage!(_mm512_unpacklo_epi64, _mm512_unpackhi_epi64);
        gp.add(h).write(r);
    }
    let g = g.assume_init_ref();
    let ident = _mm512_set1_epi64(GF_IDENT);
    let qt = _mm512_load_si512(BYTE_TR.0.as_ptr() as *const __m512i);
    let iu = _mm512_load_si512(LANE_IU.0.as_ptr() as *const __m512i);
    let iv = _mm512_load_si512(LANE_IV.0.as_ptr() as *const __m512i);
    for b in 0..16 {
        let tu = _mm512_permutex2var_epi64(g[0][b], iu, g[1][b]);
        let tv = _mm512_permutex2var_epi64(g[0][b], iv, g[1][b]);
        let mu = _mm512_permutexvar_epi8(qt, _mm512_gf2p8affine_epi64_epi8::<0>(ident, tu));
        let mv = _mm512_permutexvar_epi8(qt, _mm512_gf2p8affine_epi64_epi8::<0>(ident, tv));
        _mm512_store_si512(u.add(8 * b) as *mut __m512i, mu);
        _mm512_store_si512(v.add(8 * b) as *mut __m512i, mv);
    }
    uv_b128_to_sms_rows(u, v, out);
}

#[inline(always)]
unsafe fn uv_sms_step<const T: usize, const BASE: usize>(
    u: *const u64,
    v: *const u64,
    out: &mut SmsBinaryIndex32,
    zero: __m512i,
    ident: __m512i,
    perm: __m512i,
    dup0: __m512i,
    dup1: __m512i,
    tag_or: __m512i,
    matrix: __m512i,
) {
    macro_rules! load {
        ($j:literal) => {{
            if $j == 6 && BASE != 0 {
                zero
            } else {
                let k = (T + $j) % 4;
                let p = (if k < 2 { u } else { v }).add(BASE + 20 * $j + (T + $j) / 4).cast::<i64>();
                if $j == 6 {
                    let mask: __mmask8 = if T < 2 { 0x7f } else { 0x3f };
                    _mm512_maskz_loadu_epi64(mask, p)
                } else {
                    _mm512_loadu_si512(p.cast())
                }
            }
        }};
    }
    let a0 = load!(0);
    let a1 = load!(1);
    let a2 = load!(2);
    let a3 = load!(3);
    let a4 = load!(4);
    let a5 = load!(5);
    let a6 = load!(6);
    let (a, b, c, d) = if T % 2 == 0 {
        (
            _mm512_mask_blend_epi32(0xaaaa, a0, a1),
            _mm512_mask_blend_epi32(0xaaaa, a2, a3),
            _mm512_mask_blend_epi32(0xaaaa, a4, a5),
            _mm512_mask_blend_epi32(0xaaaa, a6, zero),
        )
    } else {
        (
            _mm512_mask_blend_epi32(0xaaaa, a1, a0),
            _mm512_mask_blend_epi32(0xaaaa, a3, a2),
            _mm512_mask_blend_epi32(0xaaaa, a5, a4),
            _mm512_mask_blend_epi32(0xaaaa, zero, a6),
        )
    };
    let l01 = _mm512_unpacklo_epi64(a, b);
    let h01 = _mm512_unpackhi_epi64(a, b);
    let l23 = _mm512_unpacklo_epi64(c, d);
    let h23 = _mm512_unpackhi_epi64(c, d);
    let s0 = _mm512_shuffle_i64x2::<0x44>(l01, l23);
    let s1 = _mm512_shuffle_i64x2::<0xee>(l01, l23);
    let s2 = _mm512_shuffle_i64x2::<0x44>(h01, h23);
    let s3 = _mm512_shuffle_i64x2::<0xee>(h01, h23);
    let r0 = _mm512_shuffle_i64x2::<0x88>(s0, s2);
    let r1 = _mm512_shuffle_i64x2::<0xdd>(s0, s2);
    let r2 = _mm512_shuffle_i64x2::<0x88>(s1, s3);
    let r3 = _mm512_shuffle_i64x2::<0xdd>(s1, s3);
    macro_rules! emit {
        ($r:expr, $h:literal) => {{
            let i = 4 * (BASE + $h) + T;
            if i < 81 {
                let n = _mm512_gf2p8affine_epi64_epi8::<0>(ident, _mm512_permutexvar_epi8(perm, $r));
                macro_rules! store_one {
                    ($row:expr, $high:literal) => {{
                        let row = $row;
                        if row < 81 {
                            if row < 27 {
                                let dup = if $high { dup1 } else { dup0 };
                                let tagged = _mm512_or_si512(_mm512_permutexvar_epi8(dup, n), tag_or);
                                _mm512_store_si512(
                                    out.rows[row].as_mut_ptr().cast(),
                                    _mm512_gf2p8affine_epi64_epi8::<0>(tagged, matrix),
                                );
                            } else {
                                let bytes = if $high {
                                    _mm512_extracti64x4_epi64::<1>(n)
                                } else {
                                    _mm512_castsi512_si256(n)
                                };
                                _mm512_store_si512(out.rows[row].as_mut_ptr().cast(), _mm512_cvtepu8_epi16(bytes));
                            }
                        }
                    }};
                }
                store_one!(i, false);
                store_one!(i + 4, true);
            }
        }};
    }
    emit!(r0, 0);
    emit!(r1, 2);
    emit!(r2, 4);
    emit!(r3, 6);
}

#[inline(always)]
unsafe fn uv_b128_to_sms_rows(u: *const u64, v: *const u64, out: &mut SmsBinaryIndex32) {
    let zero = _mm512_setzero_si512();
    let ident = _mm512_set1_epi64(GF_IDENT);
    let perm_even = _mm512_load_si512(SMS_JOINT_PERM_EVEN.0.as_ptr().cast());
    let perm_odd = _mm512_load_si512(SMS_JOINT_PERM_ODD.0.as_ptr().cast());
    let dup0 = _mm512_load_si512(SMS_JOINT_DUP0.0.as_ptr().cast());
    let dup1 = _mm512_load_si512(SMS_JOINT_DUP1.0.as_ptr().cast());
    let tag_or = _mm512_set1_epi16(0x8000u16 as i16);
    let matrix = _mm512_set1_epi64(0x0104104080020820u64 as i64);
    uv_sms_step::<0, 0>(u, v, out, zero, ident, perm_even, dup0, dup1, tag_or, matrix);
    uv_sms_step::<0, 8>(u, v, out, zero, ident, perm_even, dup0, dup1, tag_or, matrix);
    uv_sms_step::<0, 16>(u, v, out, zero, ident, perm_even, dup0, dup1, tag_or, matrix);
    uv_sms_step::<1, 0>(u, v, out, zero, ident, perm_odd, dup0, dup1, tag_or, matrix);
    uv_sms_step::<1, 8>(u, v, out, zero, ident, perm_odd, dup0, dup1, tag_or, matrix);
    uv_sms_step::<1, 16>(u, v, out, zero, ident, perm_odd, dup0, dup1, tag_or, matrix);
    uv_sms_step::<2, 0>(u, v, out, zero, ident, perm_even, dup0, dup1, tag_or, matrix);
    uv_sms_step::<2, 8>(u, v, out, zero, ident, perm_even, dup0, dup1, tag_or, matrix);
    uv_sms_step::<2, 16>(u, v, out, zero, ident, perm_even, dup0, dup1, tag_or, matrix);
    uv_sms_step::<3, 0>(u, v, out, zero, ident, perm_odd, dup0, dup1, tag_or, matrix);
    uv_sms_step::<3, 8>(u, v, out, zero, ident, perm_odd, dup0, dup1, tag_or, matrix);
    uv_sms_step::<3, 16>(u, v, out, zero, ident, perm_odd, dup0, dup1, tag_or, matrix);
}
#[cfg(test)]mod sms_joint_constructor_guards{use super::*;
 #[repr(C,align(64))]struct Guard{pre:[u64;8],value:SmsBinaryIndex32,post:[u64;8]}
 #[test]fn sms_joint_constructor_matches_every_source_bit_and_tags(){unsafe{
 let mut state=0x875a1793fe12c43du128;
 for fixture in 0..576{
 let mut columns=[[crate::fields::scalar::B128(0);8];16];
 for col in 0..16{for e in 0..2{
  if fixture<512{columns[col][4*e+fixture%4].0=1u128<<(fixture/4);}
  else{for word in 0..4{state^=state<<13;state^=state>>7;state^=state<<17;columns[col][4*e+word].0=match fixture{512=>0,513=>u128::MAX,514=>0xaaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaau128,515=>0x5555_5555_5555_5555_5555_5555_5555_5555u128,_=>state};}}
 }}
 let ptrs=core::array::from_fn(|i|columns[i].as_ptr());let mut old=BinaryIndex32{rows:[[0;64];162]};let mut got=Guard{pre:[0x75319acf1dfe4321;8],value:SmsBinaryIndex32{rows:[[0x3a55;32];81]},post:[0x75319acf1dfe4321;8]};
 slice_b128_cols16_into(&ptrs,&mut old);slice_b128_cols16_sms_into(&ptrs,&mut got.value);
 for i in 0..81{for lane in 0..32{let x=old.rows[i][2*lane]&15;let y=old.rows[i+81][2*lane]&15;let mut n=(x&8)<<3;for b in 0..3{n|=((x>>b)&1)<<(2*b);n|=((y>>b)&1)<<(2*b+1);}if i<27{let low=(x as u16)|((y as u16)<<5);assert_eq!(got.value.rows[i][lane],low|((low|16)<<8),"fixture={fixture} row={i} lane={lane}");}else{assert_eq!(got.value.rows[i][lane],n as u16,"fixture={fixture} row={i} lane={lane}");}}}

 assert_eq!(got.pre,[0x75319acf1dfe4321;8]);assert_eq!(got.post,[0x75319acf1dfe4321;8]);
 }
 }}
}
