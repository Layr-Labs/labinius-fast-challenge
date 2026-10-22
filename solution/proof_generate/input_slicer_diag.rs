// PR133 B128 column-16 input slicer, isolated for build-time comparison.
// Source: 3edbc49878a1cd893ad7c3d668b78c6736f76917, same file as PR130.
// Full source SHA-256: 7117fc167de2a1e259528150378184d5a29bff888c477d84d5a9c3411fc66618.
#![allow(dead_code)]
use core::arch::x86_64::*;
use core::mem::MaybeUninit;
use labinius::simd::transpose_f162::BinaryIndex32;
#[repr(C, align(64))]
struct A64<T>(T);
const GF_IDENT: i64 = 0x8040_2010_0804_0201u64 as i64;
static BYTE_TR: A64<[u8; 64]> = A64(byte_tr());
static AFFINE_BOTH: A64<[u8; 64]> = A64(affine_both());
static DUP_A: A64<[u8; 64]> = A64(dup_idx(0));
static DUP_B: A64<[u8; 64]> = A64(dup_idx(32));
static PLUS16: A64<[u8; 64]> = A64(plus16());
const ORDER: [usize; 16] = [0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15];
const ROW_LANE: [usize; 16] = row_lane();
static LANE_IU: A64<[i64; 8]> = A64([0, 1, 8, 9, 2, 3, 10, 11]);
static LANE_IV: A64<[i64; 8]> = A64([4, 5, 12, 13, 6, 7, 14, 15]);
const fn affine_both() -> [u8; 64] {
    let mut t = [0u8; 64];
    let mut q = 0;
    while q < 8 {
        let mut b = 0;
        while b < 4 {
            t[8 * q + 7 - b] = (8 * b + 4 * (q >> 2) + (q & 3)) as u8;
            b += 1;
        }
        q += 1;
    }
    t
}

const fn dup_idx(off: u8) -> [u8; 64] {
    let mut t = [0u8; 64];
    let mut p = 0;
    while p < 32 {
        t[2 * p] = off + p as u8;
        t[2 * p + 1] = off + p as u8;
        p += 1;
    }
    t
}

const fn plus16() -> [u8; 64] {
    let mut t = [0u8; 64];
    let mut p = 0;
    while p < 32 {
        t[2 * p + 1] = 16;
        p += 1;
    }
    t
}

const fn byte_tr() -> [u8; 64] {
    let mut t = [0u8; 64];
    let mut b = 0;
    while b < 8 {
        let mut q = 0;
        while q < 8 {
            t[8 * b + q] = (8 * q + b) as u8;
            q += 1;
        }
        b += 1;
    }
    t
}

#[repr(C, align(64))]
struct B128Work {
    u: [u64; 192],
    v: [u64; 192],
}

#[inline(always)]
unsafe fn uv_b128_to_rows(wk: *mut B128Work, out: &mut BinaryIndex32) {
    let u = (*wk).u.as_mut_ptr();
    let v = (*wk).v.as_mut_ptr();
    _mm512_store_si512(u.add(128) as *mut __m512i, _mm512_setzero_si512());
    _mm512_store_si512(v.add(128) as *mut __m512i, _mm512_setzero_si512());
    let op = out.rows.as_mut_ptr() as *mut __m512i;
    // Fixed groups avoid dynamic bounds checks and keep the gather outputs in registers.
    macro_rules! emit8 {
        ($r:ident, $t:expr, $side:literal) => {{
            index_pair($r[0], op.add(4 * ($t + 0) + $side));
            index_pair(_mm512_shuffle_i64x2::<0xEE>($r[0], $r[0]), op.add(4 * ($t + 1) + $side));
            index_pair($r[1], op.add(4 * ($t + 2) + $side));
            index_pair(_mm512_shuffle_i64x2::<0xEE>($r[1], $r[1]), op.add(4 * ($t + 3) + $side));
            index_pair($r[2], op.add(4 * ($t + 4) + $side));
            index_pair(_mm512_shuffle_i64x2::<0xEE>($r[2], $r[2]), op.add(4 * ($t + 5) + $side));
            index_pair($r[3], op.add(4 * ($t + 6) + $side));
            index_pair(_mm512_shuffle_i64x2::<0xEE>($r[3], $r[3]), op.add(4 * ($t + 7) + $side));
        }};
    }
    let r = gather8_regs(u.cast(), v.add(40).cast(), u.add(81).cast(), v.add(121).cast());
    emit8!(r, 0, 0);
    for i in 1..5 {
        let t = 8 * i;
        let r = gather8_z_regs(u.add(t).cast(), v.add(t+40).cast(), u.add(t+81).cast());
        emit8!(r, t, 0);
    }
    let r = gather8_z_regs(u.add(40).cast(), v.add(80).cast(), u.add(121).cast());
    index_pair(r[0], op.add(160));
    let r = gather8_regs(v.cast(), u.add(41).cast(), v.add(81).cast(), u.add(122).cast());
    emit8!(r, 0, 2);
    for i in 1..5 {
        let t = 8 * i;
        let r = gather8_z_regs(v.add(t).cast(), u.add(t+41).cast(), v.add(t+81).cast());
        emit8!(r, t, 2);
    }
}

const fn row_lane() -> [usize; 16] {
    let mut t = [0usize; 16];
    let mut pos = 0;
    while pos < 16 {
        t[ORDER[pos]] = if pos < 8 { 7 - pos } else { 23 - pos };
        pos += 1;
    }
    t
}

#[inline(never)]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,gfni")]
pub unsafe fn slice_b128_cols16_into(
    cols: &[*const labinius::fields::scalar::B128; 16],
    out: &mut BinaryIndex32,
) {
    let mut work: MaybeUninit<B128Work> = MaybeUninit::uninit();
    let wk = work.as_mut_ptr();
    let u = (*wk).u.as_mut_ptr();
    let v = (*wk).v.as_mut_ptr();
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
    uv_b128_to_rows(wk, out);
}

#[inline(always)]
unsafe fn gather8_regs(p0: *const i64, p1: *const i64, p2: *const i64, p3: *const i64) -> [__m512i;4] {
    let a = _mm512_loadu_si512(p0 as *const __m512i);
    let b = _mm512_loadu_si512(p1 as *const __m512i);
    let c = _mm512_loadu_si512(p2 as *const __m512i);
    let d = _mm512_loadu_si512(p3 as *const __m512i);
    let l01 = _mm512_unpacklo_epi64(a, b);
    let h01 = _mm512_unpackhi_epi64(a, b);
    let l23 = _mm512_unpacklo_epi64(c, d);
    let h23 = _mm512_unpackhi_epi64(c, d);
    let s0 = _mm512_shuffle_i64x2::<0x44>(l01, l23);
    let s1 = _mm512_shuffle_i64x2::<0xEE>(l01, l23);
    let s2 = _mm512_shuffle_i64x2::<0x44>(h01, h23);
    let s3 = _mm512_shuffle_i64x2::<0xEE>(h01, h23);
    [
        _mm512_shuffle_i64x2::<0x88>(s0, s2),
        _mm512_shuffle_i64x2::<0xDD>(s0, s2),
        _mm512_shuffle_i64x2::<0x88>(s1, s3),
        _mm512_shuffle_i64x2::<0xDD>(s1, s3),
    ]
}

#[inline(always)]
unsafe fn gather8_z_regs(p0: *const i64, p1: *const i64, p2: *const i64) -> [__m512i;4] {
    let a = _mm512_loadu_si512(p0 as *const __m512i);
    let b = _mm512_loadu_si512(p1 as *const __m512i);
    let l01 = _mm512_unpacklo_epi64(a, b);
    let h01 = _mm512_unpackhi_epi64(a, b);
    let l23 = _mm512_maskz_loadu_epi64(0x55, p2);
    let h23 = _mm512_maskz_loadu_epi64(0x55, p2.add(1));
    let s0 = _mm512_shuffle_i64x2::<0x44>(l01, l23);
    let s1 = _mm512_shuffle_i64x2::<0xEE>(l01, l23);
    let s2 = _mm512_shuffle_i64x2::<0x44>(h01, h23);
    let s3 = _mm512_shuffle_i64x2::<0xEE>(h01, h23);
    [
        _mm512_shuffle_i64x2::<0x88>(s0, s2),
        _mm512_shuffle_i64x2::<0xDD>(s0, s2),
        _mm512_shuffle_i64x2::<0x88>(s1, s3),
        _mm512_shuffle_i64x2::<0xDD>(s1, s3),
    ]
}

#[inline(always)]
unsafe fn index_pair(src: __m512i, op: *mut __m512i) {
 let ident = _mm512_set1_epi64(GF_IDENT);
 let am = _mm512_load_si512(AFFINE_BOTH.0.as_ptr() as *const __m512i);
 let n = _mm512_gf2p8affine_epi64_epi8::<0>(ident, _mm512_permutexvar_epi8(am, src));
 let a = _mm512_permutexvar_epi8(_mm512_load_si512(DUP_A.0.as_ptr() as *const __m512i), n);
 let b = _mm512_permutexvar_epi8(_mm512_load_si512(DUP_B.0.as_ptr() as *const __m512i), n);
 let p16 = _mm512_load_si512(PLUS16.0.as_ptr() as *const __m512i);
 let lo4 = _mm512_set1_epi8(15);
 _mm512_store_si512(op, _mm512_ternarylogic_epi32::<0xEA>(a,lo4,p16));
 _mm512_store_si512(op.add(1), _mm512_ternarylogic_epi32::<0xEA>(b,lo4,p16));
}
