//! Producer for the local R9 subgroup falsifier only.
//!
//! This is not a prover implementation or a performance result. The arithmetic
//! kernels below reuse `1035Z/r9-subgroup-a/split.rs`. The driver owns all aligned
//! word/W buffers, slot publication/release, timing, CPU feature checks and joins.
use core::arch::x86_64::*;
use labinius::fields::scalar::B128;
use labinius::params::{pow_mod, Params};
use labinius::simd::transpose_f162::{slice_b128_into, BinaryIndex32};
use std::sync::OnceLock;

pub const WORD_BYTES: usize = 663_552;
pub const SLOT_BYTES: usize = 331_776;
const COLUMN_WORDS: usize = 16_384;
const WORD_PRIME_I16: usize = 165_888;
const WORD_FACTOR_I16: usize = 27_648;
const WORD_COLUMN_I16: usize = 1_728;
const SLOT_PRIME_BYTES: usize = 165_888;

#[repr(C, align(64))]
struct Lookup([[u8; 64]; 96]);
struct LookupTables {
    p0: Lookup,
    p1: Lookup,
}
static TABLES: OnceLock<LookupTables> = OnceLock::new();

// Exact bin_asm::build_tables leading-LUT formula. All 96 entries are built,
// including s2=1, although the shared-prefix kernel reads only 6 at offset12*k.
fn lookup<const Q: u16>() -> Lookup {
    let q = Q as u64;
    let z6 = Params::<Q>::ZETA6 as u64;
    let kappa = [z6, (1 + q - z6) % q];
    let mut lut = [[0u8; 64]; 96];
    for k in 0..4 {
        let (s0, s1) = (k / 2, k % 2);
        let ka = kappa[s0];
        let z1 = Params::<Q>::ZETA_L1[s0] as u64;
        let zp = Params::<Q>::ZETA_L2[k] as u64;
        for s2 in 0..2 {
            let z3 = Params::<Q>::ZETA_L3[2 * k + s2] as u64;
            for r in 0..3 {
                let f = pow_mod(z3, r as u64, q);
                for ab in 0..2 {
                    let extra = if ab == 0 { 1 } else { zp };
                    for n in 0..16 {
                        let n0 = (n & 1) as u64;
                        let n1 = ((n >> 1) & 1) as u64;
                        let n2 = ((n >> 2) & 1) as u64;
                        let n3 = ((n >> 3) & 1) as u64;
                        let inner = z1 * ((n1 + ka * n3) % q) % q;
                        let t = if s1 == 0 { inner } else { (q - inner) % q };
                        let base = ((n0 + ka * n2) % q + t) % q;
                        let v = base * f % q * extra % q;
                        let centered = if v > q / 2 {
                            v as i64 - q as i64
                        } else {
                            v as i64
                        };
                        let e = centered as i16 as u16;
                        let ti = ((k * 2 + s2) * 3 + r) * 2 + ab;
                        lut[ti][n] = e as u8;
                        lut[ti][16 + n] = (e >> 8) as u8;
                    }
                }
            }
        }
    }
    Lookup(lut)
}

/// Initialize and write the fixed public lookup tables before the timed region.
/// Uses static storage and stack temporaries, with no dynamic heap allocation.
pub fn prepare() {
    TABLES.get_or_init(|| LookupTables {
        p0: lookup::<3889>(),
        p1: lookup::<9721>(),
    });
}

/// Fill one complete quarter prefix for a 16-column cohort and 64 rings.
///
/// `cohort_start` is the first column number (normally 0,16,...,1008), not a
/// cohort index. `br` is a 64-ring block number in 0..64. Output byte strides:
/// prime331776/factor55296/column3456/coefficient128/half32=64.
///
/// # Safety
/// AVX512F/BW/VL/VBMI/VNNI/GFNI must be available. `prepare()` must have run.
/// `words` addresses 1024*16384 initialized B128 values. `out` addresses
/// WORD_BYTES writable bytes, aligned to64, disjoint from input/live W buffers.
/// No other thread may read/write this word buffer until fill completes; it
/// cannot be overwritten again until the last subgroup pack has finished.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni,gfni")]
pub unsafe fn fill_prefix(
    words: *const B128,
    cohort_start: usize,
    quarter: usize,
    br: usize,
    out: *mut i16,
) {
    assert!(cohort_start <= 1008 && quarter < 4 && br < 64);
    assert_eq!((out as usize) & 63, 0, "word buffer must be64-byte aligned");
    let tables = TABLES
        .get()
        .expect("producer::prepare must run before timing");
    // A single initialized, aligned local scratch reused for every column/half
    // and both primes. Its zeroing/real slice calls belong inside the timer.
    let mut index = BinaryIndex32::zero();
    let l0 = tables.p0.0.as_ptr().add(12 * quarter) as *const __m512i;
    let l1 = tables.p1.0.as_ptr().add(12 * quarter) as *const __m512i;
    for column in 0..16 {
        for half in 0..2 {
            let offset = (cohort_start + column) * COLUMN_WORDS + br * 256 + half * 128;
            let input = &*(words.add(offset) as *const [B128; 128]);
            slice_b128_into(input, &mut index);
            let ip = index.rows.as_ptr() as *const u8;
            let dest = out.add(column * WORD_COLUMN_I16 + half * 32);
            prefix_3889(ip, l0, dest);
            prefix_9721(ip, l1, dest.add(WORD_PRIME_I16));
        }
    }
}

/// Pack one of the three two-factor subgroups into one complete W slot.
///
/// Output layout is [prime2][factor2][coefficient27][form3][column16][ring64],
/// byte strides prime165888/factor82944/coefficient3072/form1024/column64.
/// This directly matches A TILELOAD rowstride64, form1024, nextcoefficient3072.
///
/// # Safety
/// Same CPU requirements as fill_prefix. `word` addresses a completed,
/// 64-byte-aligned WORD_BYTES prefix and stays immutable throughout this call.
/// `out` addresses SLOT_BYTES writable bytes, aligned to64 and disjoint from
/// word/live consumer buffers. The driver must acquire ownership of the slot
/// before this call and publish it with release ordering only after return.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni,gfni")]
pub unsafe fn pack_subgroup(word: *const i16, subgroup: usize, out: *mut i8) {
    assert!(subgroup < 3);
    assert_eq!(
        (word as usize) & 63,
        0,
        "word buffer must be64-byte aligned"
    );
    assert_eq!((out as usize) & 63, 0, "W slot must be64-byte aligned");
    for column in 0..16 {
        for half in 0..2 {
            let input =
                word.add(subgroup * 2 * WORD_FACTOR_I16 + column * WORD_COLUMN_I16 + half * 32);
            let dest = (out as *mut u8).add(column * 64 + half * 32);
            pack2_3889(input, dest);
            pack2_9721(input.add(WORD_PRIME_I16), dest.add(SLOT_PRIME_BYTES));
        }
    }
}

// Frozen arithmetic kernels from the bounded split-emission gate. They are
// private, with no exported symbol collisions in the integrated local artifact.
#[inline(always)]
unsafe fn bc(p: *const u32) -> __m512i {
    let r: __m512i;
    core::arch::asm!(
        "vpbroadcastd {0}, dword ptr [{1}]",
        out(zmm_reg) r,
        in(reg) p,
        options(pure, readonly, nostack, preserves_flags)
    );
    r
}
#[inline(always)]
unsafe fn mulhi(a: __m512i, b: __m512i) -> __m512i {
    let r: __m512i;
    core::arch::asm!(
        "vpmulhw {0}, {1}, {2}",
        out(zmm_reg) r,
        in(zmm_reg) a,
        in(zmm_reg) b,
        options(pure, nomem, nostack, preserves_flags)
    );
    r
}
#[inline(always)]
unsafe fn mont(a: __m512i, w: __m512i, wp: __m512i, q: __m512i) -> __m512i {
    let m = _mm512_mullo_epi16(a, wp);
    let hi = mulhi(a, w);
    let t = mulhi(m, q);
    _mm512_sub_epi16(hi, t)
}
struct C {
    q: __m512i,
    om: __m512i,
    omp: __m512i,
}
#[inline(always)]
unsafe fn r3_folded(c: &C, a0: __m512i, t1: __m512i, t2: __m512i) -> (__m512i, __m512i, __m512i) {
    let u = mont(_mm512_sub_epi16(t1, t2), c.om, c.omp, c.q);
    (
        _mm512_add_epi16(a0, _mm512_add_epi16(t1, t2)),
        _mm512_add_epi16(_mm512_sub_epi16(a0, t2), u),
        _mm512_sub_epi16(_mm512_sub_epi16(a0, t1), u),
    )
}
#[inline(always)]
unsafe fn ldb(p: *const u8, j: usize) -> __m512i {
    _mm512_load_si512(p.add(64 * j) as *const __m512i)
}

#[inline(always)]
unsafe fn pack<const Q: u16>(x: __m512i, lo: *mut u8, hi: *mut u8, sm: *mut u8) {
    let q = _mm512_set1_epi16(Q as i16);
    let h = _mm512_set1_epi16((Q / 2) as i16);
    let mh = _mm512_set1_epi16(-((Q / 2) as i16));
    let recip = _mm512_set1_epi16(((32768u32 + Q as u32 / 2) / Q as u32) as i16);
    let t = _mm512_mulhrs_epi16(x, recip);
    let r = _mm512_sub_epi16(x, _mm512_mullo_epi16(t, q));
    let r = _mm512_mask_sub_epi16(r, _mm512_cmpgt_epi16_mask(r, h), r, q);
    let r = _mm512_mask_add_epi16(r, _mm512_cmplt_epi16_mask(r, mh), r, q);
    let vh = _mm512_srai_epi16::<7>(_mm512_add_epi16(r, _mm512_set1_epi16(64)));
    let vl = _mm512_sub_epi16(r, _mm512_slli_epi16::<7>(vh));
    let vs = _mm512_add_epi16(vl, vh);
    _mm256_storeu_si256(lo as *mut __m256i, _mm512_cvtsepi16_epi8(vl));
    _mm256_storeu_si256(hi as *mut __m256i, _mm512_cvtsepi16_epi8(vh));
    _mm256_storeu_si256(sm as *mut __m256i, _mm512_cvtsepi16_epi8(vs));
}
#[inline(never)]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni,gfni")]
unsafe fn prefix_3889(ip: *const u8, lut: *const __m512i, out: *mut i16) {
    let c = C {
        q: _mm512_set1_epi16(3889),
        om: _mm512_set1_epi16(-1702),
        omp: _mm512_set1_epi16(31866),
    };
    let l000 = _mm512_load_si512(lut.add(0));
    let l001 = _mm512_load_si512(lut.add(1));
    let l010 = _mm512_load_si512(lut.add(2));
    let l011 = _mm512_load_si512(lut.add(3));
    let l020 = _mm512_load_si512(lut.add(4));
    let l021 = _mm512_load_si512(lut.add(5));
    {
        let i: usize = 0;
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
        _mm512_store_si512(out.add(0 * 27648 + i * 64) as *mut __m512i, u0);
        _mm512_store_si512(out.add(1 * 27648 + i * 64) as *mut __m512i, u1);
        _mm512_store_si512(out.add(2 * 27648 + i * 64) as *mut __m512i, u2);
        _mm512_store_si512(out.add(3 * 27648 + i * 64) as *mut __m512i, v2);
        _mm512_store_si512(out.add(4 * 27648 + i * 64) as *mut __m512i, v0);
        _mm512_store_si512(out.add(5 * 27648 + i * 64) as *mut __m512i, v1);
    }
    {
        let i: usize = 1;
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
        _mm512_store_si512(out.add(0 * 27648 + i * 64) as *mut __m512i, u0);
        _mm512_store_si512(out.add(1 * 27648 + i * 64) as *mut __m512i, u1);
        _mm512_store_si512(out.add(2 * 27648 + i * 64) as *mut __m512i, u2);
        _mm512_store_si512(out.add(3 * 27648 + i * 64) as *mut __m512i, v2);
        _mm512_store_si512(out.add(4 * 27648 + i * 64) as *mut __m512i, v0);
        _mm512_store_si512(out.add(5 * 27648 + i * 64) as *mut __m512i, v1);
    }
    {
        let i: usize = 2;
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
        _mm512_store_si512(out.add(0 * 27648 + i * 64) as *mut __m512i, u0);
        _mm512_store_si512(out.add(1 * 27648 + i * 64) as *mut __m512i, u1);
        _mm512_store_si512(out.add(2 * 27648 + i * 64) as *mut __m512i, u2);
        _mm512_store_si512(out.add(3 * 27648 + i * 64) as *mut __m512i, v2);
        _mm512_store_si512(out.add(4 * 27648 + i * 64) as *mut __m512i, v0);
        _mm512_store_si512(out.add(5 * 27648 + i * 64) as *mut __m512i, v1);
    }
    {
        let i: usize = 3;
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
        _mm512_store_si512(out.add(0 * 27648 + i * 64) as *mut __m512i, u0);
        _mm512_store_si512(out.add(1 * 27648 + i * 64) as *mut __m512i, u1);
        _mm512_store_si512(out.add(2 * 27648 + i * 64) as *mut __m512i, u2);
        _mm512_store_si512(out.add(3 * 27648 + i * 64) as *mut __m512i, v2);
        _mm512_store_si512(out.add(4 * 27648 + i * 64) as *mut __m512i, v0);
        _mm512_store_si512(out.add(5 * 27648 + i * 64) as *mut __m512i, v1);
    }
    {
        let i: usize = 4;
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
        _mm512_store_si512(out.add(0 * 27648 + i * 64) as *mut __m512i, u0);
        _mm512_store_si512(out.add(1 * 27648 + i * 64) as *mut __m512i, u1);
        _mm512_store_si512(out.add(2 * 27648 + i * 64) as *mut __m512i, u2);
        _mm512_store_si512(out.add(3 * 27648 + i * 64) as *mut __m512i, v2);
        _mm512_store_si512(out.add(4 * 27648 + i * 64) as *mut __m512i, v0);
        _mm512_store_si512(out.add(5 * 27648 + i * 64) as *mut __m512i, v1);
    }
    {
        let i: usize = 5;
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
        _mm512_store_si512(out.add(0 * 27648 + i * 64) as *mut __m512i, u0);
        _mm512_store_si512(out.add(1 * 27648 + i * 64) as *mut __m512i, u1);
        _mm512_store_si512(out.add(2 * 27648 + i * 64) as *mut __m512i, u2);
        _mm512_store_si512(out.add(3 * 27648 + i * 64) as *mut __m512i, v2);
        _mm512_store_si512(out.add(4 * 27648 + i * 64) as *mut __m512i, v0);
        _mm512_store_si512(out.add(5 * 27648 + i * 64) as *mut __m512i, v1);
    }
    {
        let i: usize = 6;
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
        _mm512_store_si512(out.add(0 * 27648 + i * 64) as *mut __m512i, u0);
        _mm512_store_si512(out.add(1 * 27648 + i * 64) as *mut __m512i, u1);
        _mm512_store_si512(out.add(2 * 27648 + i * 64) as *mut __m512i, u2);
        _mm512_store_si512(out.add(3 * 27648 + i * 64) as *mut __m512i, v2);
        _mm512_store_si512(out.add(4 * 27648 + i * 64) as *mut __m512i, v0);
        _mm512_store_si512(out.add(5 * 27648 + i * 64) as *mut __m512i, v1);
    }
    {
        let i: usize = 7;
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
        _mm512_store_si512(out.add(0 * 27648 + i * 64) as *mut __m512i, u0);
        _mm512_store_si512(out.add(1 * 27648 + i * 64) as *mut __m512i, u1);
        _mm512_store_si512(out.add(2 * 27648 + i * 64) as *mut __m512i, u2);
        _mm512_store_si512(out.add(3 * 27648 + i * 64) as *mut __m512i, v2);
        _mm512_store_si512(out.add(4 * 27648 + i * 64) as *mut __m512i, v0);
        _mm512_store_si512(out.add(5 * 27648 + i * 64) as *mut __m512i, v1);
    }
    {
        let i: usize = 8;
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
        _mm512_store_si512(out.add(0 * 27648 + i * 64) as *mut __m512i, u0);
        _mm512_store_si512(out.add(1 * 27648 + i * 64) as *mut __m512i, u1);
        _mm512_store_si512(out.add(2 * 27648 + i * 64) as *mut __m512i, u2);
        _mm512_store_si512(out.add(3 * 27648 + i * 64) as *mut __m512i, v2);
        _mm512_store_si512(out.add(4 * 27648 + i * 64) as *mut __m512i, v0);
        _mm512_store_si512(out.add(5 * 27648 + i * 64) as *mut __m512i, v1);
    }
    {
        let i: usize = 9;
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
        _mm512_store_si512(out.add(0 * 27648 + i * 64) as *mut __m512i, u0);
        _mm512_store_si512(out.add(1 * 27648 + i * 64) as *mut __m512i, u1);
        _mm512_store_si512(out.add(2 * 27648 + i * 64) as *mut __m512i, u2);
        _mm512_store_si512(out.add(3 * 27648 + i * 64) as *mut __m512i, v2);
        _mm512_store_si512(out.add(4 * 27648 + i * 64) as *mut __m512i, v0);
        _mm512_store_si512(out.add(5 * 27648 + i * 64) as *mut __m512i, v1);
    }
    {
        let i: usize = 10;
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
        _mm512_store_si512(out.add(0 * 27648 + i * 64) as *mut __m512i, u0);
        _mm512_store_si512(out.add(1 * 27648 + i * 64) as *mut __m512i, u1);
        _mm512_store_si512(out.add(2 * 27648 + i * 64) as *mut __m512i, u2);
        _mm512_store_si512(out.add(3 * 27648 + i * 64) as *mut __m512i, v2);
        _mm512_store_si512(out.add(4 * 27648 + i * 64) as *mut __m512i, v0);
        _mm512_store_si512(out.add(5 * 27648 + i * 64) as *mut __m512i, v1);
    }
    {
        let i: usize = 11;
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
        _mm512_store_si512(out.add(0 * 27648 + i * 64) as *mut __m512i, u0);
        _mm512_store_si512(out.add(1 * 27648 + i * 64) as *mut __m512i, u1);
        _mm512_store_si512(out.add(2 * 27648 + i * 64) as *mut __m512i, u2);
        _mm512_store_si512(out.add(3 * 27648 + i * 64) as *mut __m512i, v2);
        _mm512_store_si512(out.add(4 * 27648 + i * 64) as *mut __m512i, v0);
        _mm512_store_si512(out.add(5 * 27648 + i * 64) as *mut __m512i, v1);
    }
    {
        let i: usize = 12;
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
        _mm512_store_si512(out.add(0 * 27648 + i * 64) as *mut __m512i, u0);
        _mm512_store_si512(out.add(1 * 27648 + i * 64) as *mut __m512i, u1);
        _mm512_store_si512(out.add(2 * 27648 + i * 64) as *mut __m512i, u2);
        _mm512_store_si512(out.add(3 * 27648 + i * 64) as *mut __m512i, v2);
        _mm512_store_si512(out.add(4 * 27648 + i * 64) as *mut __m512i, v0);
        _mm512_store_si512(out.add(5 * 27648 + i * 64) as *mut __m512i, v1);
    }
    {
        let i: usize = 13;
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
        _mm512_store_si512(out.add(0 * 27648 + i * 64) as *mut __m512i, u0);
        _mm512_store_si512(out.add(1 * 27648 + i * 64) as *mut __m512i, u1);
        _mm512_store_si512(out.add(2 * 27648 + i * 64) as *mut __m512i, u2);
        _mm512_store_si512(out.add(3 * 27648 + i * 64) as *mut __m512i, v2);
        _mm512_store_si512(out.add(4 * 27648 + i * 64) as *mut __m512i, v0);
        _mm512_store_si512(out.add(5 * 27648 + i * 64) as *mut __m512i, v1);
    }
    {
        let i: usize = 14;
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
        _mm512_store_si512(out.add(0 * 27648 + i * 64) as *mut __m512i, u0);
        _mm512_store_si512(out.add(1 * 27648 + i * 64) as *mut __m512i, u1);
        _mm512_store_si512(out.add(2 * 27648 + i * 64) as *mut __m512i, u2);
        _mm512_store_si512(out.add(3 * 27648 + i * 64) as *mut __m512i, v2);
        _mm512_store_si512(out.add(4 * 27648 + i * 64) as *mut __m512i, v0);
        _mm512_store_si512(out.add(5 * 27648 + i * 64) as *mut __m512i, v1);
    }
    {
        let i: usize = 15;
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
        _mm512_store_si512(out.add(0 * 27648 + i * 64) as *mut __m512i, u0);
        _mm512_store_si512(out.add(1 * 27648 + i * 64) as *mut __m512i, u1);
        _mm512_store_si512(out.add(2 * 27648 + i * 64) as *mut __m512i, u2);
        _mm512_store_si512(out.add(3 * 27648 + i * 64) as *mut __m512i, v2);
        _mm512_store_si512(out.add(4 * 27648 + i * 64) as *mut __m512i, v0);
        _mm512_store_si512(out.add(5 * 27648 + i * 64) as *mut __m512i, v1);
    }
    {
        let i: usize = 16;
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
        _mm512_store_si512(out.add(0 * 27648 + i * 64) as *mut __m512i, u0);
        _mm512_store_si512(out.add(1 * 27648 + i * 64) as *mut __m512i, u1);
        _mm512_store_si512(out.add(2 * 27648 + i * 64) as *mut __m512i, u2);
        _mm512_store_si512(out.add(3 * 27648 + i * 64) as *mut __m512i, v2);
        _mm512_store_si512(out.add(4 * 27648 + i * 64) as *mut __m512i, v0);
        _mm512_store_si512(out.add(5 * 27648 + i * 64) as *mut __m512i, v1);
    }
    {
        let i: usize = 17;
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
        _mm512_store_si512(out.add(0 * 27648 + i * 64) as *mut __m512i, u0);
        _mm512_store_si512(out.add(1 * 27648 + i * 64) as *mut __m512i, u1);
        _mm512_store_si512(out.add(2 * 27648 + i * 64) as *mut __m512i, u2);
        _mm512_store_si512(out.add(3 * 27648 + i * 64) as *mut __m512i, v2);
        _mm512_store_si512(out.add(4 * 27648 + i * 64) as *mut __m512i, v0);
        _mm512_store_si512(out.add(5 * 27648 + i * 64) as *mut __m512i, v1);
    }
    {
        let i: usize = 18;
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
        _mm512_store_si512(out.add(0 * 27648 + i * 64) as *mut __m512i, u0);
        _mm512_store_si512(out.add(1 * 27648 + i * 64) as *mut __m512i, u1);
        _mm512_store_si512(out.add(2 * 27648 + i * 64) as *mut __m512i, u2);
        _mm512_store_si512(out.add(3 * 27648 + i * 64) as *mut __m512i, v2);
        _mm512_store_si512(out.add(4 * 27648 + i * 64) as *mut __m512i, v0);
        _mm512_store_si512(out.add(5 * 27648 + i * 64) as *mut __m512i, v1);
    }
    {
        let i: usize = 19;
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
        _mm512_store_si512(out.add(0 * 27648 + i * 64) as *mut __m512i, u0);
        _mm512_store_si512(out.add(1 * 27648 + i * 64) as *mut __m512i, u1);
        _mm512_store_si512(out.add(2 * 27648 + i * 64) as *mut __m512i, u2);
        _mm512_store_si512(out.add(3 * 27648 + i * 64) as *mut __m512i, v2);
        _mm512_store_si512(out.add(4 * 27648 + i * 64) as *mut __m512i, v0);
        _mm512_store_si512(out.add(5 * 27648 + i * 64) as *mut __m512i, v1);
    }
    {
        let i: usize = 20;
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
        _mm512_store_si512(out.add(0 * 27648 + i * 64) as *mut __m512i, u0);
        _mm512_store_si512(out.add(1 * 27648 + i * 64) as *mut __m512i, u1);
        _mm512_store_si512(out.add(2 * 27648 + i * 64) as *mut __m512i, u2);
        _mm512_store_si512(out.add(3 * 27648 + i * 64) as *mut __m512i, v2);
        _mm512_store_si512(out.add(4 * 27648 + i * 64) as *mut __m512i, v0);
        _mm512_store_si512(out.add(5 * 27648 + i * 64) as *mut __m512i, v1);
    }
    {
        let i: usize = 21;
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
        _mm512_store_si512(out.add(0 * 27648 + i * 64) as *mut __m512i, u0);
        _mm512_store_si512(out.add(1 * 27648 + i * 64) as *mut __m512i, u1);
        _mm512_store_si512(out.add(2 * 27648 + i * 64) as *mut __m512i, u2);
        _mm512_store_si512(out.add(3 * 27648 + i * 64) as *mut __m512i, v2);
        _mm512_store_si512(out.add(4 * 27648 + i * 64) as *mut __m512i, v0);
        _mm512_store_si512(out.add(5 * 27648 + i * 64) as *mut __m512i, v1);
    }
    {
        let i: usize = 22;
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
        _mm512_store_si512(out.add(0 * 27648 + i * 64) as *mut __m512i, u0);
        _mm512_store_si512(out.add(1 * 27648 + i * 64) as *mut __m512i, u1);
        _mm512_store_si512(out.add(2 * 27648 + i * 64) as *mut __m512i, u2);
        _mm512_store_si512(out.add(3 * 27648 + i * 64) as *mut __m512i, v2);
        _mm512_store_si512(out.add(4 * 27648 + i * 64) as *mut __m512i, v0);
        _mm512_store_si512(out.add(5 * 27648 + i * 64) as *mut __m512i, v1);
    }
    {
        let i: usize = 23;
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
        _mm512_store_si512(out.add(0 * 27648 + i * 64) as *mut __m512i, u0);
        _mm512_store_si512(out.add(1 * 27648 + i * 64) as *mut __m512i, u1);
        _mm512_store_si512(out.add(2 * 27648 + i * 64) as *mut __m512i, u2);
        _mm512_store_si512(out.add(3 * 27648 + i * 64) as *mut __m512i, v2);
        _mm512_store_si512(out.add(4 * 27648 + i * 64) as *mut __m512i, v0);
        _mm512_store_si512(out.add(5 * 27648 + i * 64) as *mut __m512i, v1);
    }
    {
        let i: usize = 24;
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
        _mm512_store_si512(out.add(0 * 27648 + i * 64) as *mut __m512i, u0);
        _mm512_store_si512(out.add(1 * 27648 + i * 64) as *mut __m512i, u1);
        _mm512_store_si512(out.add(2 * 27648 + i * 64) as *mut __m512i, u2);
        _mm512_store_si512(out.add(3 * 27648 + i * 64) as *mut __m512i, v2);
        _mm512_store_si512(out.add(4 * 27648 + i * 64) as *mut __m512i, v0);
        _mm512_store_si512(out.add(5 * 27648 + i * 64) as *mut __m512i, v1);
    }
    {
        let i: usize = 25;
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
        _mm512_store_si512(out.add(0 * 27648 + i * 64) as *mut __m512i, u0);
        _mm512_store_si512(out.add(1 * 27648 + i * 64) as *mut __m512i, u1);
        _mm512_store_si512(out.add(2 * 27648 + i * 64) as *mut __m512i, u2);
        _mm512_store_si512(out.add(3 * 27648 + i * 64) as *mut __m512i, v2);
        _mm512_store_si512(out.add(4 * 27648 + i * 64) as *mut __m512i, v0);
        _mm512_store_si512(out.add(5 * 27648 + i * 64) as *mut __m512i, v1);
    }
    {
        let i: usize = 26;
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
        _mm512_store_si512(out.add(0 * 27648 + i * 64) as *mut __m512i, u0);
        _mm512_store_si512(out.add(1 * 27648 + i * 64) as *mut __m512i, u1);
        _mm512_store_si512(out.add(2 * 27648 + i * 64) as *mut __m512i, u2);
        _mm512_store_si512(out.add(3 * 27648 + i * 64) as *mut __m512i, v2);
        _mm512_store_si512(out.add(4 * 27648 + i * 64) as *mut __m512i, v0);
        _mm512_store_si512(out.add(5 * 27648 + i * 64) as *mut __m512i, v1);
    }
}
#[inline(never)]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni,gfni")]
unsafe fn pack2_3889(input: *const i16, out: *mut u8) {
    {
        let x = _mm512_load_si512(input.add(0) as *const __m512i);
        pack::<3889>(x, out.add(0), out.add(1024), out.add(2048));
    }
    {
        let x = _mm512_load_si512(input.add(64) as *const __m512i);
        pack::<3889>(x, out.add(3072), out.add(4096), out.add(5120));
    }
    {
        let x = _mm512_load_si512(input.add(128) as *const __m512i);
        pack::<3889>(x, out.add(6144), out.add(7168), out.add(8192));
    }
    {
        let x = _mm512_load_si512(input.add(192) as *const __m512i);
        pack::<3889>(x, out.add(9216), out.add(10240), out.add(11264));
    }
    {
        let x = _mm512_load_si512(input.add(256) as *const __m512i);
        pack::<3889>(x, out.add(12288), out.add(13312), out.add(14336));
    }
    {
        let x = _mm512_load_si512(input.add(320) as *const __m512i);
        pack::<3889>(x, out.add(15360), out.add(16384), out.add(17408));
    }
    {
        let x = _mm512_load_si512(input.add(384) as *const __m512i);
        pack::<3889>(x, out.add(18432), out.add(19456), out.add(20480));
    }
    {
        let x = _mm512_load_si512(input.add(448) as *const __m512i);
        pack::<3889>(x, out.add(21504), out.add(22528), out.add(23552));
    }
    {
        let x = _mm512_load_si512(input.add(512) as *const __m512i);
        pack::<3889>(x, out.add(24576), out.add(25600), out.add(26624));
    }
    {
        let x = _mm512_load_si512(input.add(576) as *const __m512i);
        pack::<3889>(x, out.add(27648), out.add(28672), out.add(29696));
    }
    {
        let x = _mm512_load_si512(input.add(640) as *const __m512i);
        pack::<3889>(x, out.add(30720), out.add(31744), out.add(32768));
    }
    {
        let x = _mm512_load_si512(input.add(704) as *const __m512i);
        pack::<3889>(x, out.add(33792), out.add(34816), out.add(35840));
    }
    {
        let x = _mm512_load_si512(input.add(768) as *const __m512i);
        pack::<3889>(x, out.add(36864), out.add(37888), out.add(38912));
    }
    {
        let x = _mm512_load_si512(input.add(832) as *const __m512i);
        pack::<3889>(x, out.add(39936), out.add(40960), out.add(41984));
    }
    {
        let x = _mm512_load_si512(input.add(896) as *const __m512i);
        pack::<3889>(x, out.add(43008), out.add(44032), out.add(45056));
    }
    {
        let x = _mm512_load_si512(input.add(960) as *const __m512i);
        pack::<3889>(x, out.add(46080), out.add(47104), out.add(48128));
    }
    {
        let x = _mm512_load_si512(input.add(1024) as *const __m512i);
        pack::<3889>(x, out.add(49152), out.add(50176), out.add(51200));
    }
    {
        let x = _mm512_load_si512(input.add(1088) as *const __m512i);
        pack::<3889>(x, out.add(52224), out.add(53248), out.add(54272));
    }
    {
        let x = _mm512_load_si512(input.add(1152) as *const __m512i);
        pack::<3889>(x, out.add(55296), out.add(56320), out.add(57344));
    }
    {
        let x = _mm512_load_si512(input.add(1216) as *const __m512i);
        pack::<3889>(x, out.add(58368), out.add(59392), out.add(60416));
    }
    {
        let x = _mm512_load_si512(input.add(1280) as *const __m512i);
        pack::<3889>(x, out.add(61440), out.add(62464), out.add(63488));
    }
    {
        let x = _mm512_load_si512(input.add(1344) as *const __m512i);
        pack::<3889>(x, out.add(64512), out.add(65536), out.add(66560));
    }
    {
        let x = _mm512_load_si512(input.add(1408) as *const __m512i);
        pack::<3889>(x, out.add(67584), out.add(68608), out.add(69632));
    }
    {
        let x = _mm512_load_si512(input.add(1472) as *const __m512i);
        pack::<3889>(x, out.add(70656), out.add(71680), out.add(72704));
    }
    {
        let x = _mm512_load_si512(input.add(1536) as *const __m512i);
        pack::<3889>(x, out.add(73728), out.add(74752), out.add(75776));
    }
    {
        let x = _mm512_load_si512(input.add(1600) as *const __m512i);
        pack::<3889>(x, out.add(76800), out.add(77824), out.add(78848));
    }
    {
        let x = _mm512_load_si512(input.add(1664) as *const __m512i);
        pack::<3889>(x, out.add(79872), out.add(80896), out.add(81920));
    }
    {
        let x = _mm512_load_si512(input.add(27648) as *const __m512i);
        pack::<3889>(x, out.add(82944), out.add(83968), out.add(84992));
    }
    {
        let x = _mm512_load_si512(input.add(27712) as *const __m512i);
        pack::<3889>(x, out.add(86016), out.add(87040), out.add(88064));
    }
    {
        let x = _mm512_load_si512(input.add(27776) as *const __m512i);
        pack::<3889>(x, out.add(89088), out.add(90112), out.add(91136));
    }
    {
        let x = _mm512_load_si512(input.add(27840) as *const __m512i);
        pack::<3889>(x, out.add(92160), out.add(93184), out.add(94208));
    }
    {
        let x = _mm512_load_si512(input.add(27904) as *const __m512i);
        pack::<3889>(x, out.add(95232), out.add(96256), out.add(97280));
    }
    {
        let x = _mm512_load_si512(input.add(27968) as *const __m512i);
        pack::<3889>(x, out.add(98304), out.add(99328), out.add(100352));
    }
    {
        let x = _mm512_load_si512(input.add(28032) as *const __m512i);
        pack::<3889>(x, out.add(101376), out.add(102400), out.add(103424));
    }
    {
        let x = _mm512_load_si512(input.add(28096) as *const __m512i);
        pack::<3889>(x, out.add(104448), out.add(105472), out.add(106496));
    }
    {
        let x = _mm512_load_si512(input.add(28160) as *const __m512i);
        pack::<3889>(x, out.add(107520), out.add(108544), out.add(109568));
    }
    {
        let x = _mm512_load_si512(input.add(28224) as *const __m512i);
        pack::<3889>(x, out.add(110592), out.add(111616), out.add(112640));
    }
    {
        let x = _mm512_load_si512(input.add(28288) as *const __m512i);
        pack::<3889>(x, out.add(113664), out.add(114688), out.add(115712));
    }
    {
        let x = _mm512_load_si512(input.add(28352) as *const __m512i);
        pack::<3889>(x, out.add(116736), out.add(117760), out.add(118784));
    }
    {
        let x = _mm512_load_si512(input.add(28416) as *const __m512i);
        pack::<3889>(x, out.add(119808), out.add(120832), out.add(121856));
    }
    {
        let x = _mm512_load_si512(input.add(28480) as *const __m512i);
        pack::<3889>(x, out.add(122880), out.add(123904), out.add(124928));
    }
    {
        let x = _mm512_load_si512(input.add(28544) as *const __m512i);
        pack::<3889>(x, out.add(125952), out.add(126976), out.add(128000));
    }
    {
        let x = _mm512_load_si512(input.add(28608) as *const __m512i);
        pack::<3889>(x, out.add(129024), out.add(130048), out.add(131072));
    }
    {
        let x = _mm512_load_si512(input.add(28672) as *const __m512i);
        pack::<3889>(x, out.add(132096), out.add(133120), out.add(134144));
    }
    {
        let x = _mm512_load_si512(input.add(28736) as *const __m512i);
        pack::<3889>(x, out.add(135168), out.add(136192), out.add(137216));
    }
    {
        let x = _mm512_load_si512(input.add(28800) as *const __m512i);
        pack::<3889>(x, out.add(138240), out.add(139264), out.add(140288));
    }
    {
        let x = _mm512_load_si512(input.add(28864) as *const __m512i);
        pack::<3889>(x, out.add(141312), out.add(142336), out.add(143360));
    }
    {
        let x = _mm512_load_si512(input.add(28928) as *const __m512i);
        pack::<3889>(x, out.add(144384), out.add(145408), out.add(146432));
    }
    {
        let x = _mm512_load_si512(input.add(28992) as *const __m512i);
        pack::<3889>(x, out.add(147456), out.add(148480), out.add(149504));
    }
    {
        let x = _mm512_load_si512(input.add(29056) as *const __m512i);
        pack::<3889>(x, out.add(150528), out.add(151552), out.add(152576));
    }
    {
        let x = _mm512_load_si512(input.add(29120) as *const __m512i);
        pack::<3889>(x, out.add(153600), out.add(154624), out.add(155648));
    }
    {
        let x = _mm512_load_si512(input.add(29184) as *const __m512i);
        pack::<3889>(x, out.add(156672), out.add(157696), out.add(158720));
    }
    {
        let x = _mm512_load_si512(input.add(29248) as *const __m512i);
        pack::<3889>(x, out.add(159744), out.add(160768), out.add(161792));
    }
    {
        let x = _mm512_load_si512(input.add(29312) as *const __m512i);
        pack::<3889>(x, out.add(162816), out.add(163840), out.add(164864));
    }
}
#[inline(never)]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni,gfni")]
unsafe fn prefix_9721(ip: *const u8, lut: *const __m512i, out: *mut i16) {
    let c = C {
        q: _mm512_set1_epi16(9721),
        om: _mm512_set1_epi16(2338),
        omp: _mm512_set1_epi16(12978),
    };
    let l000 = _mm512_load_si512(lut.add(0));
    let l001 = _mm512_load_si512(lut.add(1));
    let l010 = _mm512_load_si512(lut.add(2));
    let l011 = _mm512_load_si512(lut.add(3));
    let l020 = _mm512_load_si512(lut.add(4));
    let l021 = _mm512_load_si512(lut.add(5));
    {
        let i: usize = 0;
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
        _mm512_store_si512(out.add(0 * 27648 + i * 64) as *mut __m512i, u0);
        _mm512_store_si512(out.add(1 * 27648 + i * 64) as *mut __m512i, u1);
        _mm512_store_si512(out.add(2 * 27648 + i * 64) as *mut __m512i, u2);
        _mm512_store_si512(out.add(3 * 27648 + i * 64) as *mut __m512i, v2);
        _mm512_store_si512(out.add(4 * 27648 + i * 64) as *mut __m512i, v0);
        _mm512_store_si512(out.add(5 * 27648 + i * 64) as *mut __m512i, v1);
    }
    {
        let i: usize = 1;
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
        _mm512_store_si512(out.add(0 * 27648 + i * 64) as *mut __m512i, u0);
        _mm512_store_si512(out.add(1 * 27648 + i * 64) as *mut __m512i, u1);
        _mm512_store_si512(out.add(2 * 27648 + i * 64) as *mut __m512i, u2);
        _mm512_store_si512(out.add(3 * 27648 + i * 64) as *mut __m512i, v2);
        _mm512_store_si512(out.add(4 * 27648 + i * 64) as *mut __m512i, v0);
        _mm512_store_si512(out.add(5 * 27648 + i * 64) as *mut __m512i, v1);
    }
    {
        let i: usize = 2;
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
        _mm512_store_si512(out.add(0 * 27648 + i * 64) as *mut __m512i, u0);
        _mm512_store_si512(out.add(1 * 27648 + i * 64) as *mut __m512i, u1);
        _mm512_store_si512(out.add(2 * 27648 + i * 64) as *mut __m512i, u2);
        _mm512_store_si512(out.add(3 * 27648 + i * 64) as *mut __m512i, v2);
        _mm512_store_si512(out.add(4 * 27648 + i * 64) as *mut __m512i, v0);
        _mm512_store_si512(out.add(5 * 27648 + i * 64) as *mut __m512i, v1);
    }
    {
        let i: usize = 3;
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
        _mm512_store_si512(out.add(0 * 27648 + i * 64) as *mut __m512i, u0);
        _mm512_store_si512(out.add(1 * 27648 + i * 64) as *mut __m512i, u1);
        _mm512_store_si512(out.add(2 * 27648 + i * 64) as *mut __m512i, u2);
        _mm512_store_si512(out.add(3 * 27648 + i * 64) as *mut __m512i, v2);
        _mm512_store_si512(out.add(4 * 27648 + i * 64) as *mut __m512i, v0);
        _mm512_store_si512(out.add(5 * 27648 + i * 64) as *mut __m512i, v1);
    }
    {
        let i: usize = 4;
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
        _mm512_store_si512(out.add(0 * 27648 + i * 64) as *mut __m512i, u0);
        _mm512_store_si512(out.add(1 * 27648 + i * 64) as *mut __m512i, u1);
        _mm512_store_si512(out.add(2 * 27648 + i * 64) as *mut __m512i, u2);
        _mm512_store_si512(out.add(3 * 27648 + i * 64) as *mut __m512i, v2);
        _mm512_store_si512(out.add(4 * 27648 + i * 64) as *mut __m512i, v0);
        _mm512_store_si512(out.add(5 * 27648 + i * 64) as *mut __m512i, v1);
    }
    {
        let i: usize = 5;
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
        _mm512_store_si512(out.add(0 * 27648 + i * 64) as *mut __m512i, u0);
        _mm512_store_si512(out.add(1 * 27648 + i * 64) as *mut __m512i, u1);
        _mm512_store_si512(out.add(2 * 27648 + i * 64) as *mut __m512i, u2);
        _mm512_store_si512(out.add(3 * 27648 + i * 64) as *mut __m512i, v2);
        _mm512_store_si512(out.add(4 * 27648 + i * 64) as *mut __m512i, v0);
        _mm512_store_si512(out.add(5 * 27648 + i * 64) as *mut __m512i, v1);
    }
    {
        let i: usize = 6;
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
        _mm512_store_si512(out.add(0 * 27648 + i * 64) as *mut __m512i, u0);
        _mm512_store_si512(out.add(1 * 27648 + i * 64) as *mut __m512i, u1);
        _mm512_store_si512(out.add(2 * 27648 + i * 64) as *mut __m512i, u2);
        _mm512_store_si512(out.add(3 * 27648 + i * 64) as *mut __m512i, v2);
        _mm512_store_si512(out.add(4 * 27648 + i * 64) as *mut __m512i, v0);
        _mm512_store_si512(out.add(5 * 27648 + i * 64) as *mut __m512i, v1);
    }
    {
        let i: usize = 7;
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
        _mm512_store_si512(out.add(0 * 27648 + i * 64) as *mut __m512i, u0);
        _mm512_store_si512(out.add(1 * 27648 + i * 64) as *mut __m512i, u1);
        _mm512_store_si512(out.add(2 * 27648 + i * 64) as *mut __m512i, u2);
        _mm512_store_si512(out.add(3 * 27648 + i * 64) as *mut __m512i, v2);
        _mm512_store_si512(out.add(4 * 27648 + i * 64) as *mut __m512i, v0);
        _mm512_store_si512(out.add(5 * 27648 + i * 64) as *mut __m512i, v1);
    }
    {
        let i: usize = 8;
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
        _mm512_store_si512(out.add(0 * 27648 + i * 64) as *mut __m512i, u0);
        _mm512_store_si512(out.add(1 * 27648 + i * 64) as *mut __m512i, u1);
        _mm512_store_si512(out.add(2 * 27648 + i * 64) as *mut __m512i, u2);
        _mm512_store_si512(out.add(3 * 27648 + i * 64) as *mut __m512i, v2);
        _mm512_store_si512(out.add(4 * 27648 + i * 64) as *mut __m512i, v0);
        _mm512_store_si512(out.add(5 * 27648 + i * 64) as *mut __m512i, v1);
    }
    {
        let i: usize = 9;
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
        _mm512_store_si512(out.add(0 * 27648 + i * 64) as *mut __m512i, u0);
        _mm512_store_si512(out.add(1 * 27648 + i * 64) as *mut __m512i, u1);
        _mm512_store_si512(out.add(2 * 27648 + i * 64) as *mut __m512i, u2);
        _mm512_store_si512(out.add(3 * 27648 + i * 64) as *mut __m512i, v2);
        _mm512_store_si512(out.add(4 * 27648 + i * 64) as *mut __m512i, v0);
        _mm512_store_si512(out.add(5 * 27648 + i * 64) as *mut __m512i, v1);
    }
    {
        let i: usize = 10;
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
        _mm512_store_si512(out.add(0 * 27648 + i * 64) as *mut __m512i, u0);
        _mm512_store_si512(out.add(1 * 27648 + i * 64) as *mut __m512i, u1);
        _mm512_store_si512(out.add(2 * 27648 + i * 64) as *mut __m512i, u2);
        _mm512_store_si512(out.add(3 * 27648 + i * 64) as *mut __m512i, v2);
        _mm512_store_si512(out.add(4 * 27648 + i * 64) as *mut __m512i, v0);
        _mm512_store_si512(out.add(5 * 27648 + i * 64) as *mut __m512i, v1);
    }
    {
        let i: usize = 11;
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
        _mm512_store_si512(out.add(0 * 27648 + i * 64) as *mut __m512i, u0);
        _mm512_store_si512(out.add(1 * 27648 + i * 64) as *mut __m512i, u1);
        _mm512_store_si512(out.add(2 * 27648 + i * 64) as *mut __m512i, u2);
        _mm512_store_si512(out.add(3 * 27648 + i * 64) as *mut __m512i, v2);
        _mm512_store_si512(out.add(4 * 27648 + i * 64) as *mut __m512i, v0);
        _mm512_store_si512(out.add(5 * 27648 + i * 64) as *mut __m512i, v1);
    }
    {
        let i: usize = 12;
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
        _mm512_store_si512(out.add(0 * 27648 + i * 64) as *mut __m512i, u0);
        _mm512_store_si512(out.add(1 * 27648 + i * 64) as *mut __m512i, u1);
        _mm512_store_si512(out.add(2 * 27648 + i * 64) as *mut __m512i, u2);
        _mm512_store_si512(out.add(3 * 27648 + i * 64) as *mut __m512i, v2);
        _mm512_store_si512(out.add(4 * 27648 + i * 64) as *mut __m512i, v0);
        _mm512_store_si512(out.add(5 * 27648 + i * 64) as *mut __m512i, v1);
    }
    {
        let i: usize = 13;
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
        _mm512_store_si512(out.add(0 * 27648 + i * 64) as *mut __m512i, u0);
        _mm512_store_si512(out.add(1 * 27648 + i * 64) as *mut __m512i, u1);
        _mm512_store_si512(out.add(2 * 27648 + i * 64) as *mut __m512i, u2);
        _mm512_store_si512(out.add(3 * 27648 + i * 64) as *mut __m512i, v2);
        _mm512_store_si512(out.add(4 * 27648 + i * 64) as *mut __m512i, v0);
        _mm512_store_si512(out.add(5 * 27648 + i * 64) as *mut __m512i, v1);
    }
    {
        let i: usize = 14;
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
        _mm512_store_si512(out.add(0 * 27648 + i * 64) as *mut __m512i, u0);
        _mm512_store_si512(out.add(1 * 27648 + i * 64) as *mut __m512i, u1);
        _mm512_store_si512(out.add(2 * 27648 + i * 64) as *mut __m512i, u2);
        _mm512_store_si512(out.add(3 * 27648 + i * 64) as *mut __m512i, v2);
        _mm512_store_si512(out.add(4 * 27648 + i * 64) as *mut __m512i, v0);
        _mm512_store_si512(out.add(5 * 27648 + i * 64) as *mut __m512i, v1);
    }
    {
        let i: usize = 15;
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
        _mm512_store_si512(out.add(0 * 27648 + i * 64) as *mut __m512i, u0);
        _mm512_store_si512(out.add(1 * 27648 + i * 64) as *mut __m512i, u1);
        _mm512_store_si512(out.add(2 * 27648 + i * 64) as *mut __m512i, u2);
        _mm512_store_si512(out.add(3 * 27648 + i * 64) as *mut __m512i, v2);
        _mm512_store_si512(out.add(4 * 27648 + i * 64) as *mut __m512i, v0);
        _mm512_store_si512(out.add(5 * 27648 + i * 64) as *mut __m512i, v1);
    }
    {
        let i: usize = 16;
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
        _mm512_store_si512(out.add(0 * 27648 + i * 64) as *mut __m512i, u0);
        _mm512_store_si512(out.add(1 * 27648 + i * 64) as *mut __m512i, u1);
        _mm512_store_si512(out.add(2 * 27648 + i * 64) as *mut __m512i, u2);
        _mm512_store_si512(out.add(3 * 27648 + i * 64) as *mut __m512i, v2);
        _mm512_store_si512(out.add(4 * 27648 + i * 64) as *mut __m512i, v0);
        _mm512_store_si512(out.add(5 * 27648 + i * 64) as *mut __m512i, v1);
    }
    {
        let i: usize = 17;
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
        _mm512_store_si512(out.add(0 * 27648 + i * 64) as *mut __m512i, u0);
        _mm512_store_si512(out.add(1 * 27648 + i * 64) as *mut __m512i, u1);
        _mm512_store_si512(out.add(2 * 27648 + i * 64) as *mut __m512i, u2);
        _mm512_store_si512(out.add(3 * 27648 + i * 64) as *mut __m512i, v2);
        _mm512_store_si512(out.add(4 * 27648 + i * 64) as *mut __m512i, v0);
        _mm512_store_si512(out.add(5 * 27648 + i * 64) as *mut __m512i, v1);
    }
    {
        let i: usize = 18;
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
        _mm512_store_si512(out.add(0 * 27648 + i * 64) as *mut __m512i, u0);
        _mm512_store_si512(out.add(1 * 27648 + i * 64) as *mut __m512i, u1);
        _mm512_store_si512(out.add(2 * 27648 + i * 64) as *mut __m512i, u2);
        _mm512_store_si512(out.add(3 * 27648 + i * 64) as *mut __m512i, v2);
        _mm512_store_si512(out.add(4 * 27648 + i * 64) as *mut __m512i, v0);
        _mm512_store_si512(out.add(5 * 27648 + i * 64) as *mut __m512i, v1);
    }
    {
        let i: usize = 19;
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
        _mm512_store_si512(out.add(0 * 27648 + i * 64) as *mut __m512i, u0);
        _mm512_store_si512(out.add(1 * 27648 + i * 64) as *mut __m512i, u1);
        _mm512_store_si512(out.add(2 * 27648 + i * 64) as *mut __m512i, u2);
        _mm512_store_si512(out.add(3 * 27648 + i * 64) as *mut __m512i, v2);
        _mm512_store_si512(out.add(4 * 27648 + i * 64) as *mut __m512i, v0);
        _mm512_store_si512(out.add(5 * 27648 + i * 64) as *mut __m512i, v1);
    }
    {
        let i: usize = 20;
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
        _mm512_store_si512(out.add(0 * 27648 + i * 64) as *mut __m512i, u0);
        _mm512_store_si512(out.add(1 * 27648 + i * 64) as *mut __m512i, u1);
        _mm512_store_si512(out.add(2 * 27648 + i * 64) as *mut __m512i, u2);
        _mm512_store_si512(out.add(3 * 27648 + i * 64) as *mut __m512i, v2);
        _mm512_store_si512(out.add(4 * 27648 + i * 64) as *mut __m512i, v0);
        _mm512_store_si512(out.add(5 * 27648 + i * 64) as *mut __m512i, v1);
    }
    {
        let i: usize = 21;
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
        _mm512_store_si512(out.add(0 * 27648 + i * 64) as *mut __m512i, u0);
        _mm512_store_si512(out.add(1 * 27648 + i * 64) as *mut __m512i, u1);
        _mm512_store_si512(out.add(2 * 27648 + i * 64) as *mut __m512i, u2);
        _mm512_store_si512(out.add(3 * 27648 + i * 64) as *mut __m512i, v2);
        _mm512_store_si512(out.add(4 * 27648 + i * 64) as *mut __m512i, v0);
        _mm512_store_si512(out.add(5 * 27648 + i * 64) as *mut __m512i, v1);
    }
    {
        let i: usize = 22;
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
        _mm512_store_si512(out.add(0 * 27648 + i * 64) as *mut __m512i, u0);
        _mm512_store_si512(out.add(1 * 27648 + i * 64) as *mut __m512i, u1);
        _mm512_store_si512(out.add(2 * 27648 + i * 64) as *mut __m512i, u2);
        _mm512_store_si512(out.add(3 * 27648 + i * 64) as *mut __m512i, v2);
        _mm512_store_si512(out.add(4 * 27648 + i * 64) as *mut __m512i, v0);
        _mm512_store_si512(out.add(5 * 27648 + i * 64) as *mut __m512i, v1);
    }
    {
        let i: usize = 23;
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
        _mm512_store_si512(out.add(0 * 27648 + i * 64) as *mut __m512i, u0);
        _mm512_store_si512(out.add(1 * 27648 + i * 64) as *mut __m512i, u1);
        _mm512_store_si512(out.add(2 * 27648 + i * 64) as *mut __m512i, u2);
        _mm512_store_si512(out.add(3 * 27648 + i * 64) as *mut __m512i, v2);
        _mm512_store_si512(out.add(4 * 27648 + i * 64) as *mut __m512i, v0);
        _mm512_store_si512(out.add(5 * 27648 + i * 64) as *mut __m512i, v1);
    }
    {
        let i: usize = 24;
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
        _mm512_store_si512(out.add(0 * 27648 + i * 64) as *mut __m512i, u0);
        _mm512_store_si512(out.add(1 * 27648 + i * 64) as *mut __m512i, u1);
        _mm512_store_si512(out.add(2 * 27648 + i * 64) as *mut __m512i, u2);
        _mm512_store_si512(out.add(3 * 27648 + i * 64) as *mut __m512i, v2);
        _mm512_store_si512(out.add(4 * 27648 + i * 64) as *mut __m512i, v0);
        _mm512_store_si512(out.add(5 * 27648 + i * 64) as *mut __m512i, v1);
    }
    {
        let i: usize = 25;
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
        _mm512_store_si512(out.add(0 * 27648 + i * 64) as *mut __m512i, u0);
        _mm512_store_si512(out.add(1 * 27648 + i * 64) as *mut __m512i, u1);
        _mm512_store_si512(out.add(2 * 27648 + i * 64) as *mut __m512i, u2);
        _mm512_store_si512(out.add(3 * 27648 + i * 64) as *mut __m512i, v2);
        _mm512_store_si512(out.add(4 * 27648 + i * 64) as *mut __m512i, v0);
        _mm512_store_si512(out.add(5 * 27648 + i * 64) as *mut __m512i, v1);
    }
    {
        let i: usize = 26;
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
        _mm512_store_si512(out.add(0 * 27648 + i * 64) as *mut __m512i, u0);
        _mm512_store_si512(out.add(1 * 27648 + i * 64) as *mut __m512i, u1);
        _mm512_store_si512(out.add(2 * 27648 + i * 64) as *mut __m512i, u2);
        _mm512_store_si512(out.add(3 * 27648 + i * 64) as *mut __m512i, v2);
        _mm512_store_si512(out.add(4 * 27648 + i * 64) as *mut __m512i, v0);
        _mm512_store_si512(out.add(5 * 27648 + i * 64) as *mut __m512i, v1);
    }
}
#[inline(never)]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni,gfni")]
unsafe fn pack2_9721(input: *const i16, out: *mut u8) {
    {
        let x = _mm512_load_si512(input.add(0) as *const __m512i);
        pack::<9721>(x, out.add(0), out.add(1024), out.add(2048));
    }
    {
        let x = _mm512_load_si512(input.add(64) as *const __m512i);
        pack::<9721>(x, out.add(3072), out.add(4096), out.add(5120));
    }
    {
        let x = _mm512_load_si512(input.add(128) as *const __m512i);
        pack::<9721>(x, out.add(6144), out.add(7168), out.add(8192));
    }
    {
        let x = _mm512_load_si512(input.add(192) as *const __m512i);
        pack::<9721>(x, out.add(9216), out.add(10240), out.add(11264));
    }
    {
        let x = _mm512_load_si512(input.add(256) as *const __m512i);
        pack::<9721>(x, out.add(12288), out.add(13312), out.add(14336));
    }
    {
        let x = _mm512_load_si512(input.add(320) as *const __m512i);
        pack::<9721>(x, out.add(15360), out.add(16384), out.add(17408));
    }
    {
        let x = _mm512_load_si512(input.add(384) as *const __m512i);
        pack::<9721>(x, out.add(18432), out.add(19456), out.add(20480));
    }
    {
        let x = _mm512_load_si512(input.add(448) as *const __m512i);
        pack::<9721>(x, out.add(21504), out.add(22528), out.add(23552));
    }
    {
        let x = _mm512_load_si512(input.add(512) as *const __m512i);
        pack::<9721>(x, out.add(24576), out.add(25600), out.add(26624));
    }
    {
        let x = _mm512_load_si512(input.add(576) as *const __m512i);
        pack::<9721>(x, out.add(27648), out.add(28672), out.add(29696));
    }
    {
        let x = _mm512_load_si512(input.add(640) as *const __m512i);
        pack::<9721>(x, out.add(30720), out.add(31744), out.add(32768));
    }
    {
        let x = _mm512_load_si512(input.add(704) as *const __m512i);
        pack::<9721>(x, out.add(33792), out.add(34816), out.add(35840));
    }
    {
        let x = _mm512_load_si512(input.add(768) as *const __m512i);
        pack::<9721>(x, out.add(36864), out.add(37888), out.add(38912));
    }
    {
        let x = _mm512_load_si512(input.add(832) as *const __m512i);
        pack::<9721>(x, out.add(39936), out.add(40960), out.add(41984));
    }
    {
        let x = _mm512_load_si512(input.add(896) as *const __m512i);
        pack::<9721>(x, out.add(43008), out.add(44032), out.add(45056));
    }
    {
        let x = _mm512_load_si512(input.add(960) as *const __m512i);
        pack::<9721>(x, out.add(46080), out.add(47104), out.add(48128));
    }
    {
        let x = _mm512_load_si512(input.add(1024) as *const __m512i);
        pack::<9721>(x, out.add(49152), out.add(50176), out.add(51200));
    }
    {
        let x = _mm512_load_si512(input.add(1088) as *const __m512i);
        pack::<9721>(x, out.add(52224), out.add(53248), out.add(54272));
    }
    {
        let x = _mm512_load_si512(input.add(1152) as *const __m512i);
        pack::<9721>(x, out.add(55296), out.add(56320), out.add(57344));
    }
    {
        let x = _mm512_load_si512(input.add(1216) as *const __m512i);
        pack::<9721>(x, out.add(58368), out.add(59392), out.add(60416));
    }
    {
        let x = _mm512_load_si512(input.add(1280) as *const __m512i);
        pack::<9721>(x, out.add(61440), out.add(62464), out.add(63488));
    }
    {
        let x = _mm512_load_si512(input.add(1344) as *const __m512i);
        pack::<9721>(x, out.add(64512), out.add(65536), out.add(66560));
    }
    {
        let x = _mm512_load_si512(input.add(1408) as *const __m512i);
        pack::<9721>(x, out.add(67584), out.add(68608), out.add(69632));
    }
    {
        let x = _mm512_load_si512(input.add(1472) as *const __m512i);
        pack::<9721>(x, out.add(70656), out.add(71680), out.add(72704));
    }
    {
        let x = _mm512_load_si512(input.add(1536) as *const __m512i);
        pack::<9721>(x, out.add(73728), out.add(74752), out.add(75776));
    }
    {
        let x = _mm512_load_si512(input.add(1600) as *const __m512i);
        pack::<9721>(x, out.add(76800), out.add(77824), out.add(78848));
    }
    {
        let x = _mm512_load_si512(input.add(1664) as *const __m512i);
        pack::<9721>(x, out.add(79872), out.add(80896), out.add(81920));
    }
    {
        let x = _mm512_load_si512(input.add(27648) as *const __m512i);
        pack::<9721>(x, out.add(82944), out.add(83968), out.add(84992));
    }
    {
        let x = _mm512_load_si512(input.add(27712) as *const __m512i);
        pack::<9721>(x, out.add(86016), out.add(87040), out.add(88064));
    }
    {
        let x = _mm512_load_si512(input.add(27776) as *const __m512i);
        pack::<9721>(x, out.add(89088), out.add(90112), out.add(91136));
    }
    {
        let x = _mm512_load_si512(input.add(27840) as *const __m512i);
        pack::<9721>(x, out.add(92160), out.add(93184), out.add(94208));
    }
    {
        let x = _mm512_load_si512(input.add(27904) as *const __m512i);
        pack::<9721>(x, out.add(95232), out.add(96256), out.add(97280));
    }
    {
        let x = _mm512_load_si512(input.add(27968) as *const __m512i);
        pack::<9721>(x, out.add(98304), out.add(99328), out.add(100352));
    }
    {
        let x = _mm512_load_si512(input.add(28032) as *const __m512i);
        pack::<9721>(x, out.add(101376), out.add(102400), out.add(103424));
    }
    {
        let x = _mm512_load_si512(input.add(28096) as *const __m512i);
        pack::<9721>(x, out.add(104448), out.add(105472), out.add(106496));
    }
    {
        let x = _mm512_load_si512(input.add(28160) as *const __m512i);
        pack::<9721>(x, out.add(107520), out.add(108544), out.add(109568));
    }
    {
        let x = _mm512_load_si512(input.add(28224) as *const __m512i);
        pack::<9721>(x, out.add(110592), out.add(111616), out.add(112640));
    }
    {
        let x = _mm512_load_si512(input.add(28288) as *const __m512i);
        pack::<9721>(x, out.add(113664), out.add(114688), out.add(115712));
    }
    {
        let x = _mm512_load_si512(input.add(28352) as *const __m512i);
        pack::<9721>(x, out.add(116736), out.add(117760), out.add(118784));
    }
    {
        let x = _mm512_load_si512(input.add(28416) as *const __m512i);
        pack::<9721>(x, out.add(119808), out.add(120832), out.add(121856));
    }
    {
        let x = _mm512_load_si512(input.add(28480) as *const __m512i);
        pack::<9721>(x, out.add(122880), out.add(123904), out.add(124928));
    }
    {
        let x = _mm512_load_si512(input.add(28544) as *const __m512i);
        pack::<9721>(x, out.add(125952), out.add(126976), out.add(128000));
    }
    {
        let x = _mm512_load_si512(input.add(28608) as *const __m512i);
        pack::<9721>(x, out.add(129024), out.add(130048), out.add(131072));
    }
    {
        let x = _mm512_load_si512(input.add(28672) as *const __m512i);
        pack::<9721>(x, out.add(132096), out.add(133120), out.add(134144));
    }
    {
        let x = _mm512_load_si512(input.add(28736) as *const __m512i);
        pack::<9721>(x, out.add(135168), out.add(136192), out.add(137216));
    }
    {
        let x = _mm512_load_si512(input.add(28800) as *const __m512i);
        pack::<9721>(x, out.add(138240), out.add(139264), out.add(140288));
    }
    {
        let x = _mm512_load_si512(input.add(28864) as *const __m512i);
        pack::<9721>(x, out.add(141312), out.add(142336), out.add(143360));
    }
    {
        let x = _mm512_load_si512(input.add(28928) as *const __m512i);
        pack::<9721>(x, out.add(144384), out.add(145408), out.add(146432));
    }
    {
        let x = _mm512_load_si512(input.add(28992) as *const __m512i);
        pack::<9721>(x, out.add(147456), out.add(148480), out.add(149504));
    }
    {
        let x = _mm512_load_si512(input.add(29056) as *const __m512i);
        pack::<9721>(x, out.add(150528), out.add(151552), out.add(152576));
    }
    {
        let x = _mm512_load_si512(input.add(29120) as *const __m512i);
        pack::<9721>(x, out.add(153600), out.add(154624), out.add(155648));
    }
    {
        let x = _mm512_load_si512(input.add(29184) as *const __m512i);
        pack::<9721>(x, out.add(156672), out.add(157696), out.add(158720));
    }
    {
        let x = _mm512_load_si512(input.add(29248) as *const __m512i);
        pack::<9721>(x, out.add(159744), out.add(160768), out.add(161792));
    }
    {
        let x = _mm512_load_si512(input.add(29312) as *const __m512i);
        pack::<9721>(x, out.add(162816), out.add(163840), out.add(164864));
    }
}
