// Isolated LOCK4 comparison. Production entry points are unchanged.
#![allow(dead_code)]
use core::arch::x86_64::*;
use labinius::fields::scalar::B128;
use labinius::params::*;
use labinius::ring::{Batch32, Representation};
use labinius::simd::commit::{w_bound,a_bound};
use labinius::simd::transpose_f162::BinaryIndex32;
use labinius::simd::ntt::gen_small;
const P0:u16=3889; const P1:u16=9721;
const CHUNK_PAIRS:usize=128;
const PP0:usize=pair_period(P0); const PP1:usize=pair_period(P1);
const PAIR_LEAD:usize=2*PP0;
const _:()=assert!(PP0==16 && PP1==8 && CHUNK_PAIRS%PP0==0 && CHUNK_PAIRS%PP1==0);
const PAIR_PACK_PERM:[i64;8]=[0,2,4,6,1,3,5,7];
#[repr(align(128))]
struct Progress(std::sync::atomic::AtomicUsize);
const SIBLING_WAIT:std::time::Duration=std::time::Duration::from_micros(500);

#[path="input_slicer_diag.rs"] mod input_slicer;

#[inline(always)]
unsafe fn transform<const Q:u16,const LOCK:usize>(idx:&[BinaryIndex32],key:*const u32,acc:*mut i32,red:bool) {
 if LOCK==2 {
  labinius::simd::ntt::bin_asm::ntt_bin_batch32_pair::<Q>([&idx[0],&idx[1]],key,acc,red);
 } else {
  vb::ntt_bin_batch32_lock4::<Q>([&idx[0],&idx[1],&idx[2],&idx[3]],key,acc,red);
 }
}

mod vb {
use labinius::params::*;
macro_rules! r3_twiddles {
    ($n:literal, $pair:path, $zetas:expr) => {{
        let mut t = [[0u32; 4]; $n];
        let mut i = 0;
        while i < $n {
            t[i] = $pair($zetas[i]);
            i += 1;
        }
        t
    }};
}
pub use labinius::simd::transpose_f162::BinaryIndex32;

use core::arch::x86_64::*;

// ---------------------------------------------------------------------------------------------
// constant tables
// ---------------------------------------------------------------------------------------------

const fn dup(x: i16) -> u32 {
    (x as u16 as u32) | ((x as u16 as u32) << 16)
}

#[repr(C, align(64))]
pub struct Tables {
    /// `lut[((k * 2 + s2) * 3 + r) * 2 + ab]`: the 16 centered i16 values of
    /// `base_k(n) * zeta''_{2k+s2}^r * (ab == 1 ? zeta'_k : 1)`, **byte-split** so that a single
    /// `vpermb` (1 uop, port 5 - unlike `vpermw`, which is 2 uops and costs a port-0 slot on this
    /// core) does the lookup: byte n is the low half of entry n, byte 16+n the high half.
    lut: [[u8; 64]; 96],
    /// Full 512-bit constants used as memory operands / register constants by the asm tail:
    /// `[q, om, omp, bv, ms, corr, and, or, wcorr]` at byte offsets 0, 64, ... - `bv` is the
    /// `vpmulhrsw` Barrett constant of level 6, `wcorr` the 32-entry i16 `-k q` table of the
    /// level-4 lookup Barrett (`vpsrlw` + `vpermw`, see `barrett_lut_i16`); `ms`, `corr`, `and`
    /// and `or` belong to the older byte-split `vpmultishiftqb`/`vpermb` form of that lookup and
    /// are no longer read by the kernel.
    cv: [[i16; 32]; 9],
    /// `[w, w', w2, w2']` (Montgomery twiddle and companion for zeta and zeta^2), each i16
    /// duplicated into a u32 so `vpbroadcastd` is a pure load.
    tw4: [[u32; 4]; 24],
    tw5: [[u32; 4]; 72],
    tw6: [[u32; 4]; 216],
    /// omega and its companion.
    om: [u32; 2],
    /// q and the vpmulhrsw Barrett constant.
    qd: u32,
    bv: u32,
}

const fn mont_pair<const Q: u16>(x: u16) -> (u32, u32) {
    let w = Params::<Q>::to_mont(x);
    (dup(w), dup(Params::<Q>::mont_pre(w)))
}

const fn r3_pair<const Q: u16>(z: u16) -> [u32; 4] {
    let z2 = (z as u64 * z as u64 % Q as u64) as u16;
    let (a, b) = mont_pair::<Q>(z);
    let (c, d) = mont_pair::<Q>(z2);
    [a, b, c, d]
}

const fn build_tables<const Q: u16>() -> Tables {
    let q = Q as u64;
    let z6 = Params::<Q>::ZETA6 as u64;
    let kappa = [z6, (1 + q - z6) % q];

    let mut lut = [[0u8; 64]; 96];
    let mut k = 0;
    while k < 4 {
        let s0 = k / 2;
        let s1 = k % 2;
        let ka = kappa[s0];
        let z1 = Params::<Q>::ZETA_L1[s0] as u64;
        let zp = Params::<Q>::ZETA_L2[k] as u64;
        let mut s2 = 0;
        while s2 < 2 {
            let z3 = Params::<Q>::ZETA_L3[2 * k + s2] as u64;
            let mut r = 0;
            while r < 3 {
                let f = pow_mod(z3, r as u64, q);
                let mut ab = 0;
                while ab < 2 {
                    let extra = if ab == 0 { 1 } else { zp };
                    let mut n = 0;
                    while n < 16 {
                        let n0 = (n & 1) as u64;
                        let n1 = ((n >> 1) & 1) as u64;
                        let n2 = ((n >> 2) & 1) as u64;
                        let n3 = ((n >> 3) & 1) as u64;
                        let inner = z1 * ((n1 + ka * n3) % q) % q;
                        let t = if s1 == 0 { inner } else { (q - inner) % q };
                        let base = ((n0 + ka * n2) % q + t) % q;
                        let v = base * f % q * extra % q;
                        // the table holds the centred value itself; nothing is scaled into Montgomery form
                        let e = center(v, q) as u16;
                        let ti = ((k * 2 + s2) * 3 + r) * 2 + ab;
                        lut[ti][n] = e as u8;
                        lut[ti][16 + n] = (e >> 8) as u8;
                        n += 1;
                    }
                    ab += 1;
                }
                r += 1;
            }
            s2 += 1;
        }
        k += 1;
    }

    let tw4 = r3_twiddles!(24, r3_pair::<Q>, Params::<Q>::ZETA_L4);
    let tw5 = r3_twiddles!(72, r3_pair::<Q>, Params::<Q>::ZETA_L5);
    let tw6 = r3_twiddles!(216, r3_pair::<Q>, Params::<Q>::ZETA_L6);
    let (oa, ob) = mont_pair::<Q>(Params::<Q>::OMEGA);
    let mut cv = [[0i16; 32]; 9];
    let mut i = 0;
    while i < 32 {
        // the level-4 lookup Barrett's `vpermw` table: word s is -k(s) q for the window
        // s = a >> 11 (logical) = (a >> 11) & 31 (arithmetic) of the lane a.
        cv[8][i] = barrett_lut_corr(i, Q);
        cv[0][i] = Q as i16;
        cv[1][i] = Params::<Q>::to_mont(Params::<Q>::OMEGA);
        cv[2][i] = Params::<Q>::mont_pre(Params::<Q>::to_mont(Params::<Q>::OMEGA));
        cv[3][i] = barrett_v(Q);
        // vpmultishiftqb control: both bytes of word j of a qword take bits 11..18 of that
        // word, i.e. the 5-bit quotient window (a >> 11) & 31 plus 3 ignored/masked bits.
        cv[4][i] = ((16 * (i % 4) + 11) * 257) as i16;
        // the byte-split correction table: byte u (u < 32) is the low half of -k(u) q,
        // byte 32 + u its high half; the `vpermb` index is u for the low byte of every lane
        // and 32 + u for the high byte.
        let (u0, u1) = (2 * i, 2 * i + 1);
        let (b0, b1) = (lut_byte(u0, Q), lut_byte(u1, Q));
        cv[5][i] = (b0 as u16 | ((b1 as u16) << 8)) as i16;
        cv[6][i] = 0x1f1f;
        cv[7][i] = 0x2000;
        i += 1;
    }
    Tables {
        lut,
        cv,
        tw4,
        tw5,
        tw6,
        om: [oa, ob],
        qd: dup(Q as i16),
        bv: dup(barrett_v(Q)),
    }
}

static T3889: Tables = build_tables::<3889>();
static T9721: Tables = build_tables::<9721>();

#[inline(always)]
fn tables<const Q: u16>() -> &'static Tables {
    if Q == 3889 {
        &T3889
    } else {
        &T9721
    }
}

/// Quotient estimate of the lookup Barrett: the multiple of q subtracted from a value whose
/// 5-bit window `(a >> 11) & 31` is `s`. `a` is then known to lie in an interval of width 2^11
/// centred at `2048 * sgn + 1023.5`, and `k` is the nearest multiple of q to that centre, so
/// `|a - k q| <= q/2 + 2^10` for every i16 `a`.
const fn lut_k(s: usize, q: u16) -> i64 {
    let sgn = if s < 16 { s as i64 } else { s as i64 - 32 };
    let num = 4096 * sgn + 2047;
    let den = 2 * q as i64;
    if num >= 0 {
        (num + den / 2) / den
    } else {
        -((-num + den / 2) / den)
    }
}

/// `-k(s) * q`, the i16 the byte-split table adds back for window `s`.
pub const fn barrett_lut_corr(s: usize, q: u16) -> i16 {
    (-lut_k(s, q) * q as i64) as i16
}

/// Byte `u` of the 64-byte `vpermb` table: the low halves of `-k(s) q` at u = s < 32, the high
/// halves at u = 32 + s.
const fn lut_byte(u: usize, q: u16) -> u8 {
    if u < 32 {
        barrett_lut_corr(u, q) as u16 as u8
    } else {
        (barrett_lut_corr(u - 32, q) as u16 >> 8) as u8
    }
}

/// The kernel's level-4 reduction, lane-wise: `vpsrlw 11` (the 5-bit window `(a >> 11) & 31`,
/// as a logical shift of the lane), `vpermw` on the 32-entry i16 table of `-k q`, `vpaddw` -
/// **3 uops, not one multiply slot**, against the two-multiply Barrett's 2 multiplies
/// + 1 flexible. Exhaustively over all i16, `max |r| = 5625 = 0.579 q` for q = 9721, where
/// [`params::barrett_i16`] only reaches 0.809 q - which is what makes the level-5 reduction
/// droppable.
#[inline]
pub fn barrett_lut_i16(a: i16, q: u16) -> i16 {
    a.wrapping_add(barrett_lut_corr(((a >> 11) & 31) as usize, q))
}

/// Which of levels 4, 5 and 6 reduce their un-twiddled `a0` input with the lookup Barrett
/// ([`barrett_lut_i16`]) and which with the two-multiply one ([`params::barrett_i16`]),
/// for q = 9721; q = 3889 reduces nothing.
pub const LUT_BARRETT_LEVELS: [bool; 3] = [true, false, false];
pub const MUL_BARRETT_LEVELS: [bool; 3] = [false, false, true];

/// Does the 7.5 q growth of the un-Barretted schedule still fit in i16?
pub const fn needs_barrett(q: u16) -> bool {
    15 * (q as u32) >= 2 * 32768
}

/// Declared output bound: max |lane| of `ntt_bin_batch32`, as a multiple of q (numerator / 1000).
pub const fn output_bound_milli_q(q: u16) -> u32 {
    if needs_barrett(q) {
        2294
    } else {
        7500
    }
}

// ---------------------------------------------------------------------------------------------
// arithmetic helpers
// ---------------------------------------------------------------------------------------------

/// `vpbroadcastd zmm, m32` - one pure load uop. Written as `asm!` because LLVM otherwise
/// "recognises" the duplicated-u32 splat and rebuilds it with vpmovsxwd/vpmovdw/vinserti64x4.
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

/// `vpmulhw`. stdarch's `_mm512_mulhi_epi16` is written as sext -> mul -> shr -> trunc; LLVM
/// folds the multiply but leaves a `vpmovsxwd`/`vpmovdw`/`vinserti64x4` round trip on operands it
/// cannot see through (broadcast constants), and rematerialises it inside the hot loops.
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

/// 3-uop signed Montgomery twiddle multiply: a * x mod q in (-q, q).
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

/// Radix-3 butterfly whose twiddles are already folded into the inputs (level 3).
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
unsafe fn st(p: *mut i16, j: usize, v: __m512i) {
    _mm512_store_si512(p.add(32 * j) as *mut __m512i, v);
}

/// Levels 4, 5 and 6 of one 27-block, entirely in registers (deep schedule).
macro_rules! mont {
    ($tw:literal, $off:literal, $a1:literal, $a2:literal, $t:literal) => {
        concat!(
            "vpbroadcastd zmm", $t, ", dword ptr [{", $tw, "}", $off, "+4]\n",
            "vpmullw zmm", $t, ", zmm", $a1, ", zmm", $t, "\n",
            "vpbroadcastd zmm30, dword ptr [{", $tw, "}", $off, "]\n",
            "vpmulhw zmm", $a1, ", zmm", $a1, ", zmm30\n",
            "vpmulhw zmm", $t, ", zmm", $t, ", zmm27\n",
            "vpsubw zmm", $a1, ", zmm", $a1, ", zmm", $t, "\n",
            "vpbroadcastd zmm", $t, ", dword ptr [{", $tw, "}", $off, "+12]\n",
            "vpmullw zmm", $t, ", zmm", $a2, ", zmm", $t, "\n",
            "vpbroadcastd zmm30, dword ptr [{", $tw, "}", $off, "+8]\n",
            "vpmulhw zmm", $a2, ", zmm", $a2, ", zmm30\n",
            "vpmulhw zmm", $t, ", zmm", $t, ", zmm27\n",
            "vpsubw zmm", $a2, ", zmm", $a2, ", zmm", $t, "\n",
            "vpsubw zmm", $t, ", zmm", $a1, ", zmm", $a2, "\n",
            "vpmullw zmm30, zmm", $t, ", [{c}+128]\n",
            "vpmulhw zmm", $t, ", zmm", $t, ", [{c}+64]\n",
            "vpmulhw zmm30, zmm30, zmm27\n",
            "vpsubw zmm", $t, ", zmm", $t, ", zmm30\n",
        )
    };
}

macro_rules! bfly {
    ($a0:literal, $a1:literal, $a2:literal, $t:literal) => {
        concat!(
            "vpaddw zmm30, zmm", $a1, ", zmm", $a2, "\n",
            "vpsubw zmm", $a2, ", zmm", $a0, ", zmm", $a2, "\n",
            "vpaddw zmm", $a2, ", zmm", $a2, ", zmm", $t, "\n",
            "vpsubw zmm", $a1, ", zmm", $a0, ", zmm", $a1, "\n",
            "vpsubw zmm", $a1, ", zmm", $a1, ", zmm", $t, "\n",
            "vpaddw zmm", $a0, ", zmm", $a0, ", zmm30\n",
        )
    };
    ($a0:literal, $a1:literal, $a2:literal, $t:literal, $o0:literal, $o1:literal, $o2:literal) => {
        concat!(
            "vpaddw zmm30, zmm", $a1, ", zmm", $a2, "\n",
            "vpsubw zmm", $a2, ", zmm", $a0, ", zmm", $a2, "\n",
            "vpaddw zmm", $a2, ", zmm", $a2, ", zmm", $t, "\n",
            "vmovdqa64 [{o}+", $o1, "], zmm", $a2, "\n",
            "vpsubw zmm", $a1, ", zmm", $a0, ", zmm", $a1, "\n",
            "vpsubw zmm", $a1, ", zmm", $a1, ", zmm", $t, "\n",
            "vmovdqa64 [{o}+", $o2, "], zmm", $a1, "\n",
            "vpaddw zmm", $a0, ", zmm", $a0, ", zmm30\n",
            "vmovdqa64 [{o}+", $o0, "], zmm", $a0, "\n",
        )
    };
}

macro_rules! level4 {
    (plain $a0:literal, $a1:literal, $a2:literal,
     $o0:literal, $o1:literal, $o2:literal, $t:literal) => {
        concat!(
            "vmovdqa64 zmm", $a0, ", [{i}+", $o0, "]\n",
            "vmovdqa64 zmm", $a1, ", [{i}+", $o1, "]\n",
            "vmovdqa64 zmm", $a2, ", [{i}+", $o2, "]\n",
            mont!("t4", "", $a1, $a2, $t),
        )
    };
    (lut $a0:literal, $a1:literal, $a2:literal,
     $o0:literal, $o1:literal, $o2:literal, $t:literal) => {
        concat!(
            level4!(plain $a0, $a1, $a2, $o0, $o1, $o2, $t),
            // logical a >> 11 is the window (a >> 11) & 31 (bits 11..15 of the lane), and
            // vpermw only reads the low 5 bits of each word index anyway.
            "vpsrlw zmm30, zmm", $a0, ", 11\n",
            "vpermw zmm30, zmm30, zmmword ptr [{c}+512]\n",
            "vpaddw zmm", $a0, ", zmm", $a0, ", zmm30\n",
        )
    };
}

macro_rules! level6 {
    (plain $off:literal, $a0:literal, $a1:literal, $a2:literal, $t:literal) => {
        mont!("t6", $off, $a1, $a2, $t)
    };
    (barrett $off:literal, $a0:literal, $a1:literal, $a2:literal, $t:literal) => {
        concat!(
            "vpbroadcastd zmm", $t, ", dword ptr [{t6}", $off, "+4]\n",
            "vpmullw zmm", $t, ", zmm", $a1, ", zmm", $t, "\n",
            "vpmulhrsw zmm30, zmm", $a0, ", [{c}+192]\n",
            "vpmullw zmm30, zmm30, zmm27\n",
            "vpsubw zmm", $a0, ", zmm", $a0, ", zmm30\n",
            "vpbroadcastd zmm30, dword ptr [{t6}", $off, "]\n",
            "vpmulhw zmm", $a1, ", zmm", $a1, ", zmm30\n",
            "vpmulhw zmm", $t, ", zmm", $t, ", zmm27\n",
            "vpsubw zmm", $a1, ", zmm", $a1, ", zmm", $t, "\n",
            "vpbroadcastd zmm", $t, ", dword ptr [{t6}", $off, "+12]\n",
            "vpmullw zmm", $t, ", zmm", $a2, ", zmm", $t, "\n",
            "vpbroadcastd zmm30, dword ptr [{t6}", $off, "+8]\n",
            "vpmulhw zmm", $a2, ", zmm", $a2, ", zmm30\n",
            "vpmulhw zmm", $t, ", zmm", $t, ", zmm27\n",
            "vpsubw zmm", $a2, ", zmm", $a2, ", zmm", $t, "\n",
            "vpsubw zmm", $t, ", zmm", $a1, ", zmm", $a2, "\n",
            "vpmullw zmm30, zmm", $t, ", [{c}+128]\n",
            "vpmulhw zmm", $t, ", zmm", $t, ", [{c}+64]\n",
            "vpmulhw zmm30, zmm30, zmm27\n",
            "vpsubw zmm", $t, ", zmm", $t, ", zmm30\n",
        )
    };
}

macro_rules! tail27 {
    ($m4:ident, $m6:ident) => {
        tail27!($m4, $m6, bfly)
    };
    ($m4:ident, $m6:ident, $finish:ident) => {
        concat!(
            level4!($m4 0, 9, 18, 0, 576, 1152, 28),
            level4!($m4 3, 12, 21, 192, 768, 1344, 29),
            bfly!(0, 9, 18, 28),
            level4!($m4 6, 15, 24, 384, 960, 1536, 28),
            bfly!(3, 12, 21, 29),
            level4!($m4 1, 10, 19, 64, 640, 1216, 29),
            bfly!(6, 15, 24, 28),
            level4!($m4 4, 13, 22, 256, 832, 1408, 28),
            bfly!(1, 10, 19, 29),
            level4!($m4 7, 16, 25, 448, 1024, 1600, 29),
            bfly!(4, 13, 22, 28),
            level4!($m4 2, 11, 20, 128, 704, 1280, 28),
            bfly!(7, 16, 25, 29),
            level4!($m4 5, 14, 23, 320, 896, 1472, 29),
            bfly!(2, 11, 20, 28),
            level4!($m4 8, 17, 26, 512, 1088, 1664, 28),
            bfly!(5, 14, 23, 29),
            bfly!(8, 17, 26, 28),
            mont!("t5", "", 3, 6, 28),
            mont!("t5", "+16", 21, 24, 29),
            bfly!(0, 3, 6, 28),
            mont!("t5", "+32", 12, 15, 28),
            bfly!(18, 21, 24, 29),
            mont!("t5", "", 4, 7, 29),
            bfly!(9, 12, 15, 28),
            mont!("t5", "+16", 22, 25, 28),
            bfly!(1, 4, 7, 29),
            mont!("t5", "+32", 13, 16, 29),
            bfly!(19, 22, 25, 28),
            mont!("t5", "", 5, 8, 28),
            bfly!(10, 13, 16, 29),
            mont!("t5", "+16", 23, 26, 29),
            bfly!(2, 5, 8, 28),
            mont!("t5", "+32", 14, 17, 28),
            bfly!(20, 23, 26, 29),
            bfly!(11, 14, 17, 28),
            level6!($m6 "", 0, 1, 2, 28),
            level6!($m6 "+16", 6, 7, 8, 29),
            $finish!(0, 1, 2, 28, 0, 64, 128),
            level6!($m6 "+32", 3, 4, 5, 28),
            $finish!(6, 7, 8, 29, 192, 256, 320),
            level6!($m6 "+48", 18, 19, 20, 29),
            $finish!(3, 4, 5, 28, 384, 448, 512),
            level6!($m6 "+64", 24, 25, 26, 28),
            $finish!(18, 19, 20, 29, 576, 640, 704),
            level6!($m6 "+80", 21, 22, 23, 29),
            $finish!(24, 25, 26, 28, 768, 832, 896),
            level6!($m6 "+96", 9, 10, 11, 28),
            $finish!(21, 22, 23, 29, 960, 1024, 1088),
            level6!($m6 "+112", 15, 16, 17, 29),
            $finish!(9, 10, 11, 28, 1152, 1216, 1280),
            level6!($m6 "+128", 12, 13, 14, 28),
            $finish!(15, 16, 17, 29, 1344, 1408, 1472),
            $finish!(12, 13, 14, 28, 1536, 1600, 1664),
        )
    };
}

#[inline(always)]
unsafe fn tail27_p(
    bp: *const i16,
    op: *mut i16,
    t4: *const u32,
    t5: *const u32,
    t6: *const u32,
    cv: *const i16,
) {
    core::arch::asm!(
        concat!("vmovdqa64 zmm27, [{c}]\n", tail27!(plain, plain)),
        i = in(reg) bp,
        o = in(reg) op,
        t4 = in(reg) t4,
        t5 = in(reg) t5,
        t6 = in(reg) t6,
        c = in(reg) cv,
        out("zmm0") _,
        out("zmm1") _,
        out("zmm2") _,
        out("zmm3") _,
        out("zmm4") _,
        out("zmm5") _,
        out("zmm6") _,
        out("zmm7") _,
        out("zmm8") _,
        out("zmm9") _,
        out("zmm10") _,
        out("zmm11") _,
        out("zmm12") _,
        out("zmm13") _,
        out("zmm14") _,
        out("zmm15") _,
        out("zmm16") _,
        out("zmm17") _,
        out("zmm18") _,
        out("zmm19") _,
        out("zmm20") _,
        out("zmm21") _,
        out("zmm22") _,
        out("zmm23") _,
        out("zmm24") _,
        out("zmm25") _,
        out("zmm26") _,
        out("zmm27") _,
        out("zmm28") _,
        out("zmm29") _,
        out("zmm30") _,
        options(nostack)
    );
}

#[inline(always)]
unsafe fn tail27_b(
    bp: *const i16,
    op: *mut i16,
    t4: *const u32,
    t5: *const u32,
    t6: *const u32,
    cv: *const i16,
) {
    core::arch::asm!(
        concat!(
            "vmovdqa64 zmm27, [{c}]\n",
            tail27!(lut, barrett),
        ),
        i = in(reg) bp,
        o = in(reg) op,
        t4 = in(reg) t4,
        t5 = in(reg) t5,
        t6 = in(reg) t6,
        c = in(reg) cv,
        out("zmm0") _,
        out("zmm1") _,
        out("zmm2") _,
        out("zmm3") _,
        out("zmm4") _,
        out("zmm5") _,
        out("zmm6") _,
        out("zmm7") _,
        out("zmm8") _,
        out("zmm9") _,
        out("zmm10") _,
        out("zmm11") _,
        out("zmm12") _,
        out("zmm13") _,
        out("zmm14") _,
        out("zmm15") _,
        out("zmm16") _,
        out("zmm17") _,
        out("zmm18") _,
        out("zmm19") _,
        out("zmm20") _,
        out("zmm21") _,
        out("zmm22") _,
        out("zmm23") _,
        out("zmm24") _,
        out("zmm25") _,
        out("zmm26") _,
        out("zmm27") _,
        out("zmm28") _,
        out("zmm29") _,
        out("zmm30") _,
        options(nostack)
    );
}

#[repr(C, align(64))]
struct Blk([i16; 162 * 32]);


macro_rules! bfly_registers {
    ($a0:literal, $a1:literal, $a2:literal, $t:literal,
     $o0:literal, $o1:literal, $o2:literal) => {
        bfly!($a0, $a1, $a2, $t)
    };
}
macro_rules! hybrid_store {
    (plain, $reg:literal, $off:literal) => {
        concat!("vmovdqa64 [{acc}+", $off, "], zmm", $reg, "\n")
    };
    (reduce, $reg:literal, $off:literal) => {
        concat!(
            "vpmaddwd zmm31, zmm", $reg, ", [{fold}]\n",
            "vptestmd k1, zmm", $reg, ", [{fold}+64]\n",
            "vpaddd zmm31{{k1}}, zmm31, [{fold}+128]\n",
            "vmovdqa64 [{acc}+", $off, "], zmm31\n",
        )
    };
}
#[repr(C,align(64))]
struct HybridFold([[i32;16];3]);

const fn hybrid_fold<const Q: u16>() -> HybridFold {
    let r = (65536 % Q as i32) - Q as i32;
    HybridFold([
        [1 | ((r as u16 as i32) << 16); 16],
        [0x8000; 16],
        [r; 16],
    ])
}
#[repr(C,align(64))]
struct Blk27([i16;27*32]);

macro_rules! lock4_slot {
    ($mode:ident, $reg:literal, $off:literal, $k:literal, $a:literal, $t:literal) => {
        concat!(
            "vmovdqa64 zmm29, [{acc}+", $a, "]\n",
            "vmovdqa64 zmm30, [{acc}+", $t, "]\n",
            "vmovdqa64 zmm28, [{w0}+", $off, "]\n",
            "vpdpwssd zmm29, zmm28, dword ptr [{key}+", $k, "]{{1to16}}\n",
            "vpdpwssd zmm30, zmm28, dword ptr [{key}+", $k, "+4]{{1to16}}\n",
            "vmovdqa64 zmm28, [{w1}+", $off, "]\n",
            "vpdpwssd zmm29, zmm28, dword ptr [{key}+", $k, "+8]{{1to16}}\n",
            "vpdpwssd zmm30, zmm28, dword ptr [{key}+", $k, "+12]{{1to16}}\n",
            "vmovdqa64 zmm28, [{w2}+", $off, "]\n",
            "vpdpwssd zmm29, zmm28, dword ptr [{key}+", $k, "+20736]{{1to16}}\n",
            "vpdpwssd zmm30, zmm28, dword ptr [{key}+", $k, "+20740]{{1to16}}\n",
            "vpdpwssd zmm29, zmm", $reg, ", dword ptr [{key}+", $k, "+20744]{{1to16}}\n",
            "vpdpwssd zmm30, zmm", $reg, ", dword ptr [{key}+", $k, "+20748]{{1to16}}\n",
            hybrid_store!($mode, 29, $a),
            hybrid_store!($mode, 30, $t),
        )
    };
}
macro_rules! lock4_register_sink {
 ($mode:ident) => { concat!(
  lock4_slot!($mode, 0, 0, 0, 0, 64),
  lock4_slot!($mode, 2, 64, 16, 128, 192),
  lock4_slot!($mode, 1, 128, 32, 256, 320),
  lock4_slot!($mode, 6, 192, 48, 384, 448),
  lock4_slot!($mode, 8, 256, 64, 512, 576),
  lock4_slot!($mode, 7, 320, 80, 640, 704),
  lock4_slot!($mode, 3, 384, 96, 768, 832),
  lock4_slot!($mode, 5, 448, 112, 896, 960),
  lock4_slot!($mode, 4, 512, 128, 1024, 1088),
  lock4_slot!($mode, 18, 576, 144, 1152, 1216),
  lock4_slot!($mode, 20, 640, 160, 1280, 1344),
  lock4_slot!($mode, 19, 704, 176, 1408, 1472),
  lock4_slot!($mode, 24, 768, 192, 1536, 1600),
  lock4_slot!($mode, 26, 832, 208, 1664, 1728),
  lock4_slot!($mode, 25, 896, 224, 1792, 1856),
  lock4_slot!($mode, 21, 960, 240, 1920, 1984),
  lock4_slot!($mode, 23, 1024, 256, 2048, 2112),
  lock4_slot!($mode, 22, 1088, 272, 2176, 2240),
  lock4_slot!($mode, 9, 1152, 288, 2304, 2368),
  lock4_slot!($mode, 11, 1216, 304, 2432, 2496),
  lock4_slot!($mode, 10, 1280, 320, 2560, 2624),
  lock4_slot!($mode, 15, 1344, 336, 2688, 2752),
  lock4_slot!($mode, 17, 1408, 352, 2816, 2880),
  lock4_slot!($mode, 16, 1472, 368, 2944, 3008),
  lock4_slot!($mode, 12, 1536, 384, 3072, 3136),
  lock4_slot!($mode, 14, 1600, 400, 3200, 3264),
  lock4_slot!($mode, 13, 1664, 416, 3328, 3392),
 ) };
}
#[inline(always)]
unsafe fn tail27_lock4<const Q: u16>(
    bp: *const i16, w0: *const i16, w1: *const i16, w2: *const i16, t4: *const u32, t5: *const u32, t6: *const u32,
    cv: *const i16, key: *const u32, acc: *mut i32, red: u32,
) {
    macro_rules! run {
        ($m4:ident, $m6:ident) => {
            core::arch::asm!(
                concat!(
                    "vmovdqa64 zmm27, [{c}]\n",
                    tail27!($m4, $m6, bfly_registers),
                    "test {red:e}, {red:e}\n",
                    "jnz 2f\n",
                    lock4_register_sink!(plain),
                    "jmp 3f\n",
                    "2:\n",
                    lock4_register_sink!(reduce),
                    "3:\n",
                    "/* {fold} */\n",
                ),
                i = in(reg) bp, w0 = in(reg) w0, w1 = in(reg) w1, w2 = in(reg) w2, t4 = in(reg) t4, t5 = in(reg) t5, t6 = in(reg) t6,
                c = in(reg) cv, key = in(reg) key, acc = in(reg) acc,
                fold = in(reg) &const { hybrid_fold::<Q>() },
                red = in(reg) red,
                out("zmm0") _,
                out("zmm1") _,
                out("zmm2") _,
                out("zmm3") _,
                out("zmm4") _,
                out("zmm5") _,
                out("zmm6") _,
                out("zmm7") _,
                out("zmm8") _,
                out("zmm9") _,
                out("zmm10") _,
                out("zmm11") _,
                out("zmm12") _,
                out("zmm13") _,
                out("zmm14") _,
                out("zmm15") _,
                out("zmm16") _,
                out("zmm17") _,
                out("zmm18") _,
                out("zmm19") _,
                out("zmm20") _,
                out("zmm21") _,
                out("zmm22") _,
                out("zmm23") _,
                out("zmm24") _,
                out("zmm25") _,
                out("zmm26") _,
                out("zmm27") _,
                out("zmm28") _,
                out("zmm29") _,
                out("zmm30") _,
                out("zmm31") _,
                out("k1") _,
                options(nostack)
            )
        };
    }
    if needs_barrett(Q) {
        run!(lut, barrett);
    } else {
        run!(plain, plain);
    }
}
#[inline(never)]
#[target_feature(enable="avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vnni,gfni")]
pub unsafe fn ntt_bin_batch32_lock4<const Q: u16>(
    ins: [&BinaryIndex32; 4],
    key: *const u32,
    acc: *mut i32,
    red: bool,
) {
    assert!(Q == 3889 || Q == 9721, "the hybrid commitment's primes");
    let t = tables::<Q>();
    let c = C {
        q: bc(&t.qd),
        om: bc(&t.om[0]),
        omp: bc(&t.om[1]),
    };
    let bar = needs_barrett(Q);
    let ips: [*const u8; 4] = core::array::from_fn(|h| ins[h].rows.as_ptr() as *const u8);
    let mut blks: [core::mem::MaybeUninit<Blk>; 8] =
        [const { core::mem::MaybeUninit::uninit() }; 8];
    let bps: [*mut i16; 8] = core::array::from_fn(|h| blks[h].as_mut_ptr() as *mut i16);
    let mut staged: [core::mem::MaybeUninit<Blk27>; 3] =
        [const { core::mem::MaybeUninit::uninit() }; 3];
    let wp: [*mut i16; 3] = core::array::from_fn(|h| staged[h].as_mut_ptr() as *mut i16);
    let red_u32 = red as u32;

    // Process fixed adjacent quarter pairs: both prefixes share each six-row index load.
    // Each quarter retains its own two LOCK=2 buffers until its original six tails finish.
    for kp in 0..2 {
        let load_luts = |k: usize| {
            let lut = t.lut.as_ptr().add(12 * k) as *const u8;
            let l = |r: usize, ab: usize| -> __m512i {
                _mm512_load_si512(lut.add(64 * (r * 2 + ab)) as *const __m512i)
            };
            (l(0, 0), l(0, 1), l(1, 0), l(1, 1), l(2, 0), l(2, 1))
        };
        let ls0 = load_luts(2 * kp);
        let ls1 = load_luts(2 * kp + 1);
        for h in 0..core::hint::black_box(4usize) {
            let ip = *ips.as_ptr().add(h);
            for i in 0..27 {
                let n0 = ldb(ip, i);
                let n0h = ldb(ip, i + 81);
                let n1 = ldb(ip, i + 27);
                let n1h = ldb(ip, i + 108);
                let n2 = ldb(ip, i + 54);
                let n2h = ldb(ip, i + 135);

                macro_rules! prefix {
                    ($bp:expr, $ls:expr) => {{
                        let bp = $bp;
                        let (l000, l001, l010, l011, l020, l021) = $ls;
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
                        st(bp, i + 27, u1);
                        st(bp, i + 54, u2);
                        st(bp, i + 81, v2);
                        st(bp, i + 108, v0);
                        st(bp, i + 135, v1);
                    }};
                }
                prefix!(*bps.as_ptr().add(h), ls0);
                prefix!(*bps.as_ptr().add(4+h), ls1);
            }
        }
        for qpair in 0..core::hint::black_box(2usize) {
            let k = 2 * kp + qpair;
            let cv = t.cv.as_ptr() as *const i16;
            let tail_blocks = core::hint::black_box(6usize);
            for j in 0..tail_blocks {
                let kk = 6 * k + j;
                let (t4, t5, t6) = (
                    t.tw4.as_ptr().add(kk).cast::<u32>(),
                    t.tw5.as_ptr().add(3 * kk).cast::<u32>(),
                    t.tw6.as_ptr().add(9 * kk).cast::<u32>(),
                );
                for h in 0..core::hint::black_box(3usize) {
                    let bp = (*bps.as_ptr().add(4*qpair+h)).add(32 * 27 * j);
                    let dst = *wp.as_ptr().add(h);
                    if bar { tail27_b(bp, dst, t4, t5, t6, cv); }
                    else { tail27_p(bp, dst, t4, t5, t6, cv); }
                }
                tail27_lock4::<Q>(
                    (*bps.as_ptr().add(4*qpair+3)).add(32 * 27 * j),
                    wp[0], wp[1], wp[2], t4, t5, t6, cv,
                    key.add(4 * 27 * kk), acc.add(32 * 27 * kk), red_u32,
                );
            }
        }
    }
}

}
const fn r16c(q:u16)->i32 {(65536%q as i32)-q as i32}

const fn acc_after_reduce_c(q:u16)->i64 {32768*(1+r16c(q).unsigned_abs() as i64)}

const fn bias(q:u16)->i64 {((1i64<<31)/q as i64)*q as i64}

const fn qinv16(q:u16)->u32 {
 let q=q as u32;let mut x=q;let mut i=0;
 while i<4 {x=x.wrapping_mul(2u32.wrapping_sub(q.wrapping_mul(x)))&0xffff;i+=1;}
 x
}

const fn red16_w(q:u16)->(i32,i32) {
 let wl=(1u32<<16)%q as u32;let wh=((1u64<<32)%q as u64) as u32;
 let qi=qinv16(q);
 ((wl|(wh<<16)) as i32,(((wl*qi)&0xffff)|(((wh*qi)&0xffff)<<16)) as i32)
}

#[inline(always)]
unsafe fn red16<const Q:u16>(x:__m512i)->__m512i {
 let (w,wp)=red16_w(Q);
 let y=_mm512_add_epi32(x,_mm512_set1_epi32(bias(Q) as i32));
 let m=_mm512_mullo_epi16(y,_mm512_set1_epi32(wp));
 let r=_mm512_sub_epi16(_mm512_mulhi_epu16(y,_mm512_set1_epi32(w)),_mm512_mulhi_epu16(m,_mm512_set1_epi16(Q as i16)));
 _mm512_madd_epi16(r,_mm512_set1_epi16(1))
}

#[target_feature(enable="avx512f,avx512bw")]
unsafe fn crt_accumulate(a:&Batch32,b:&Batch32,tail:&mut Batch32) {
 const INV:u16=inv_mod(P0 as u64,P1 as u64) as u16;
 const W:i16=Params::<P1>::to_mont(INV);
 const WP:i16=Params::<P1>::mont_pre(W);
 let qv=_mm512_set1_epi16(P1 as i16);
 let half=_mm512_set1_epi16((P1/2) as i16);
 let nhalf=_mm512_set1_epi16(-((P1/2) as i16));
 for j in 0..648 {
  let av=_mm512_load_si512(a.v[j].as_ptr() as *const __m512i);
  let bv=_mm512_load_si512(b.v[j].as_ptr() as *const __m512i);
  let d=_mm512_sub_epi16(bv,av);
  let m=_mm512_mullo_epi16(d,_mm512_set1_epi16(WP));
  let h=_mm512_mulhi_epi16(d,_mm512_set1_epi16(W));
  let mut u=_mm512_sub_epi16(h,_mm512_mulhi_epi16(m,qv));
  u=_mm512_mask_sub_epi16(u,_mm512_cmp_epi16_mask(u,half,_MM_CMPINT_NLE),u,qv);
  u=_mm512_mask_add_epi16(u,_mm512_cmp_epi16_mask(u,nhalf,_MM_CMPINT_LT),u,qv);
  let low=_mm512_add_epi16(av,_mm512_mullo_epi16(u,_mm512_set1_epi16(P0 as i16)));
  let old=_mm512_load_si512(tail.v[j].as_ptr() as *const __m512i);
  _mm512_store_si512(tail.v[j].as_mut_ptr() as *mut __m512i,_mm512_add_epi16(old,low));
 }
}

fn zeroed_slice<T>(n:usize)->Box<[T]> {
 unsafe {
  let l=std::alloc::Layout::array::<T>(n).unwrap();
  let p=std::alloc::alloc_zeroed(l) as *mut T;
  if p.is_null() {std::alloc::handle_alloc_error(l)}
  Box::from_raw(core::ptr::slice_from_raw_parts_mut(p,n))
 }
}

const fn pair_per_batch(q:u16)->i64 {2*w_bound(q)*a_bound(q)}

const fn pair_period(q:u16)->usize {
 let mut p=1usize;
 while acc_after_reduce_c(q)+2*(p as i64)*pair_per_batch(q)<=i32::MAX as i64 {p*=2;}
 p
}

#[repr(C,align(128))]
pub struct PAcc { pub v:[[[i32;16];2];648] }

#[target_feature(enable="avx512f,avx512bw")]
unsafe fn finish_pair_stream<const Q:u16,const STREAM:usize>(a:&mut PAcc,out:&mut Batch32,half:usize) {
 let p=(a.v.as_mut_ptr() as *mut __m512i).add(STREAM);
 let zero=_mm512_setzero_si512();
 let perm=_mm512_loadu_si512(PAIR_PACK_PERM.as_ptr() as *const __m512i);
 let l0=16*half;
 for k in 0..324 {
  let j=2*k;
  let x0=_mm512_load_si512(p.add(2*j));
  let x1=_mm512_load_si512(p.add(2*(j+1)));
  _mm512_store_si512(p.add(2*j),zero);
  _mm512_store_si512(p.add(2*(j+1)),zero);
  let packed=_mm512_packs_epi32(red16::<Q>(x0),red16::<Q>(x1));
  let u=_mm512_permutexvar_epi64(perm,packed);
  _mm256_storeu_si256(out.v[j].as_mut_ptr().add(l0) as *mut __m256i,_mm512_castsi512_si256(u));
  _mm256_storeu_si256(out.v[j+1].as_mut_ptr().add(l0) as *mut __m256i,_mm512_extracti64x4_epi64::<1>(u));
 }
}

#[inline(always)]
unsafe fn canon_red16<const Q:u16>(x:__m512i)->__m512i {
 let r=red16::<Q>(x);
 let q=_mm512_set1_epi32(Q as i32);
 let q2=_mm512_set1_epi32(2*Q as i32);
 let r=_mm512_mask_add_epi32(r,_mm512_cmp_epi32_mask(r,_mm512_setzero_si512(),_MM_CMPINT_LT),r,q2);
 _mm512_mask_sub_epi32(r,_mm512_cmp_epi32_mask(r,q,_MM_CMPINT_NLT),r,q)
}

#[target_feature(enable="avx512f,avx512bw")]
unsafe fn clear_upper(out:&mut Batch32) {
 let zero=_mm256_setzero_si256();
 for j in 0..648 {_mm256_storeu_si256(out.v[j].as_mut_ptr().add(16) as *mut __m256i,zero);}
}

#[inline(always)]
unsafe fn tr4(r0:__m512i,r1:__m512i,r2:__m512i,r3:__m512i)->[__m512i;4] {
 let t0=_mm512_shuffle_i64x2::<0x88>(r0,r1);let t1=_mm512_shuffle_i64x2::<0xDD>(r0,r1);
 let t2=_mm512_shuffle_i64x2::<0x88>(r2,r3);let t3=_mm512_shuffle_i64x2::<0xDD>(r2,r3);
 [_mm512_shuffle_i64x2::<0x88>(t0,t2),_mm512_shuffle_i64x2::<0x88>(t1,t3),
  _mm512_shuffle_i64x2::<0xDD>(t0,t2),_mm512_shuffle_i64x2::<0xDD>(t1,t3)]
}

#[inline(always)]
unsafe fn transpose16x16_u32(r:&[__m512i;16])->[__m512i;16] {
 let mut u=[[_mm512_setzero_si512();4];4];
 for q in 0..4 {
  let t0=_mm512_unpacklo_epi32(r[4*q],r[4*q+1]);
  let t1=_mm512_unpackhi_epi32(r[4*q],r[4*q+1]);
  let t2=_mm512_unpacklo_epi32(r[4*q+2],r[4*q+3]);
  let t3=_mm512_unpackhi_epi32(r[4*q+2],r[4*q+3]);
  u[q]=[
   _mm512_unpacklo_epi64(t0,t2),
   _mm512_unpackhi_epi64(t0,t2),
   _mm512_unpacklo_epi64(t1,t3),
   _mm512_unpackhi_epi64(t1,t3),
  ];
 }
 let mut out=[_mm512_setzero_si512();16];
 for m in 0..4 {
  let v=tr4(u[0][m],u[1][m],u[2][m],u[3][m]);
  for l in 0..4 {out[4*l+m]=v[l];}
 }
 out
}

#[inline(never)]
#[target_feature(enable="avx512f,avx512bw,avx512vl")]
unsafe fn write_pair_group_rows(
 acc0:&mut PAcc,
 acc1:&mut PAcc,
 total:&mut Batch32,
 gc:usize,
 dst:*mut [u32;N],
) {
 let mask1023=_mm256_set1_epi16(1023);
 let zero=_mm512_setzero_si512();
 let p0=acc0.v.as_mut_ptr() as *mut __m512i;
 let p1=acc1.v.as_mut_ptr() as *mut __m512i;
 let pt=total.v.as_mut_ptr() as *mut __m512i;
 for tile in 0..40 {
  let j0=16*tile;
  let mut r0=[zero;16];
  let mut r1=[zero;16];
  let mut rt=[zero;16];
  for s in 0..16 {
   let j=j0+s;
   let v0=_mm512_load_si512(p0.add(2*j));
   let v1=_mm512_load_si512(p1.add(2*j));
   let vt=_mm512_load_si512(pt.add(j));
   _mm512_store_si512(p0.add(2*j),zero);
   _mm512_store_si512(p1.add(2*j),zero);
   _mm512_store_si512(pt.add(j),zero);
   r0[s]=canon_red16::<P0>(v0);
   r1[s]=canon_red16::<P1>(v1);
   let st=_mm256_add_epi16(_mm512_castsi512_si256(vt),_mm512_extracti64x4_epi64::<1>(vt));
   rt[s]=_mm512_cvtepu16_epi32(_mm256_and_si256(st,mask1023));
  }
  let c0=transpose16x16_u32(&r0);
  let c1=transpose16x16_u32(&r1);
  let ct=transpose16x16_u32(&rt);
  for c in 0..gc {
   _mm512_storeu_si512((dst.add(3*c) as *mut u32).add(j0) as *mut __m512i,c0[c]);
   _mm512_storeu_si512((dst.add(3*c+1) as *mut u32).add(j0) as *mut __m512i,c1[c]);
   _mm512_storeu_si512((dst.add(3*c+2) as *mut u32).add(j0) as *mut __m512i,ct[c]);
  }
 }
 // Remaining 8 slots (j = 640..648).
 let j0=640;
 let mut r0=[zero;16];
 let mut r1=[zero;16];
 let mut rt=[zero;16];
 for s in 0..8 {
  let j=j0+s;
  let v0=_mm512_load_si512(p0.add(2*j));
  let v1=_mm512_load_si512(p1.add(2*j));
  let vt=_mm512_load_si512(pt.add(j));
  _mm512_store_si512(p0.add(2*j),zero);
  _mm512_store_si512(p1.add(2*j),zero);
  _mm512_store_si512(pt.add(j),zero);
  r0[s]=canon_red16::<P0>(v0);
  r1[s]=canon_red16::<P1>(v1);
  let st=_mm256_add_epi16(_mm512_castsi512_si256(vt),_mm512_extracti64x4_epi64::<1>(vt));
  rt[s]=_mm512_cvtepu16_epi32(_mm256_and_si256(st,mask1023));
 }
 let c0=transpose16x16_u32(&r0);
 let c1=transpose16x16_u32(&r1);
 let ct=transpose16x16_u32(&rt);
 for c in 0..gc {
  _mm256_storeu_si256((dst.add(3*c) as *mut u32).add(j0) as *mut __m256i,_mm512_castsi512_si256(c0[c]));
  _mm256_storeu_si256((dst.add(3*c+1) as *mut u32).add(j0) as *mut __m256i,_mm512_castsi512_si256(c1[c]));
  _mm256_storeu_si256((dst.add(3*c+2) as *mut u32).add(j0) as *mut __m256i,_mm512_castsi512_si256(ct[c]));
 }
}

struct DiagWorker {
 acc:Box<[PAcc]>,
 partial:Box<[Batch32]>,
 total:Box<[Batch32]>,
 idx:Box<[BinaryIndex32]>,
}

impl DiagWorker {
 fn new<const LOCK:usize>()->Self {
  DiagWorker{
   acc:zeroed_slice(2),
   partial:zeroed_slice(2),
   total:zeroed_slice(1),
   idx:zeroed_slice(LOCK),
  }
 }
}

#[target_feature(enable="avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vnni,gfni")]
unsafe fn group_pairs_diag<const LOCK:usize,const REG_SLICER:bool,F:FnMut(usize)>(
 cols:&[&[B128]],
 keys:&[u32],
 np:usize,
 ws:&mut DiagWorker,
 dst:*mut [u32;N],
 mut sync:F,
) {
 let g=cols.len();assert!(g>=1 && g<=16 && np%LOCK==0);
 // Lanes of columns past the group's end read a zero element.
 static ZERO8:[B128;8]=[B128::ZERO;8];
 let kbase=keys.as_ptr();
 let mut half=0usize;
 for b in (0..np).step_by(LOCK) {
  for h in 0..LOCK {
   let cols16:[*const B128;16]=core::array::from_fn(|l|{
    if l<g {cols[l].as_ptr().add(8*(b+h))} else {ZERO8.as_ptr()}
   });
   if REG_SLICER {input_slicer::slice_b128_cols16_into(&cols16,&mut ws.idx[h]);}
   else {labinius::simd::transpose_f162::slice_b128_cols16_into(&cols16,&mut ws.idx[h]);}
  }
  let (acc0,acc1)={
   let [x,y]=&mut ws.acc[..2] else {unreachable!()};
   (x.v.as_mut_ptr() as *mut i32,y.v.as_mut_ptr() as *mut i32)
  };
  let kb=kbase.add(b*648*4);
  transform::<P0,LOCK>(&ws.idx,kb,acc0,(b+LOCK)%PP0==0);
  transform::<P1,LOCK>(&ws.idx,kb.add(648*4),acc1,(b+LOCK)%PP1==0);
  if (b+LOCK)%PP0==0 || b+LOCK==np {
   sync(b+LOCK);
  }
  if (b+LOCK)%CHUNK_PAIRS==0 || b+LOCK==np {
   // Two finished chunks share one pair of inverse transforms: the first in lanes 0..16, the
   // second in lanes 16..32 (each chunk's CRT stays exact on its own lanes).
   let (p0,p1)=ws.partial.split_at_mut(1);
   finish_pair_stream::<P0,1>(&mut ws.acc[0],&mut p0[0],half);
   finish_pair_stream::<P1,1>(&mut ws.acc[1],&mut p1[0],half);
   if half==1 || b+LOCK==np {
    if half==0 {clear_upper(&mut p0[0]);clear_upper(&mut p1[0]);}
    p0[0].representation=Representation::Ntt;p1[0].representation=Representation::Ntt;
    // finish_pair_stream uses red16: every live lane has |r| < 2*P0; unused lanes are zero.
    gen_small::intt_hybrid_batch32_3889(&mut p0[0]);
    gen_small::intt_gen_batch32::<P1>(&mut p1[0]);
    crt_accumulate(&p0[0],&p1[0],&mut ws.total[0]);
    half=0;
   } else {half=1;}
  }
 }
 let [acc0,acc1]=&mut ws.acc[..2] else {unreachable!()};
 write_pair_group_rows(acc0,acc1,&mut ws.total[0],g,dst);
}

fn commit_paired_pairs_diag<const LOCK:usize,const REG_SLICER:bool>(
 keys:&[u32],
 columns:&[&[B128]],
 np:usize,
 workers:usize,
 placement:Option<&labinius::par::CorePlacement>,
 dst:usize,
) {
 use std::sync::atomic::{AtomicBool,AtomicUsize,Ordering};
 let g=16usize;let sg=2*g;
 let pairs=workers/2;
 let super_groups=columns.len().div_ceil(sg);
 let rounds=super_groups+1;
 let next=AtomicUsize::new(0);
 let slots:Vec<std::sync::OnceLock<usize>>=(0..pairs*rounds).map(|_|std::sync::OnceLock::new()).collect();
 let done:Vec<Progress>=(0..2*pairs).map(|_|Progress(AtomicUsize::new(0))).collect();
 let coupled:Vec<AtomicBool>=(0..pairs).map(|_|AtomicBool::new(true)).collect();
 labinius::par::pool().run(2*pairs,|worker|{
  let _guard=placement.and_then(|p|p.pin(worker));
  let (pair,s)=(worker/2,worker%2);
  let mine=&done[worker].0;let other=&done[worker^1].0;let coupled=&coupled[pair];
  let mut ws:Option<DiagWorker>=None;
  for r in 0..rounds {
   let gi=*slots[pair*rounds+r].get_or_init(||next.fetch_add(1,Ordering::Relaxed));
   if gi>=super_groups {break;}
   let c0=gi*sg;let gc=sg.min(columns.len()-c0);
   let (lo,n)=if s==0 {(c0,gc.min(g))} else {(c0+g,gc.saturating_sub(g))};
   let base=r*np;
   if n==0 {mine.store(base+np,Ordering::Release);continue;}
   let ws=ws.get_or_insert_with(DiagWorker::new::<LOCK>);
   unsafe{group_pairs_diag::<LOCK,REG_SLICER,_>(&columns[lo..lo+n],keys,np,ws,(dst as *mut [u32;N]).add(3*lo),|b_done|{
    let d=base+b_done;
    mine.store(d,Ordering::Release);
    if !coupled.load(Ordering::Relaxed) {return;}
    if other.load(Ordering::Acquire)+PAIR_LEAD>=d {return;}
    let start=std::time::Instant::now();
    let mut spins=0u32;
    while other.load(Ordering::Acquire)+PAIR_LEAD<d {
     core::hint::spin_loop();
     spins+=1;
     if spins%64==0 && start.elapsed()>SIBLING_WAIT {coupled.store(false,Ordering::Relaxed);return;}
    }
   })};
  }
  mine.store(usize::MAX/2,Ordering::Release);
 });
}

pub fn commit_columns_rows_pairs<const LOCK:usize,const REG_SLICER:bool>(keys:&[u32],columns:&[&[B128]],workers:usize)->Vec<[u32;N]> {
 assert!(LOCK==2 || LOCK==4,"LOCK must be two or four");
 assert!(!columns.is_empty(),"at least one witness column");
 let len=columns[0].len();for c in columns {assert_eq!(c.len(),len);}
 assert_eq!(len%8,0);let np=len/8;assert_eq!(np%LOCK,0,"complete LOCK batches");
 assert_eq!(keys.len(),4*np*648);
 let g=16usize;let groups=columns.len().div_ceil(g);
 // Uninitialised: the groups partition the columns and each writes all three rows of its own.
 let mut rows:Vec<[u32;N]>=Vec::with_capacity(columns.len()*3);let dst=rows.as_mut_ptr() as usize;
 let paired=if workers>=2 && workers%2==0 && groups>1 {labinius::par::CorePlacement::for_sibling_pairs(workers)} else {None};
 if let Some(placement)=paired {
  commit_paired_pairs_diag::<LOCK,REG_SLICER>(keys,columns,np,workers,Some(&placement),dst);
 } else {
  let workers=workers.clamp(1,groups);
  let next=std::sync::atomic::AtomicUsize::new(0);
  let placement=labinius::par::CorePlacement::for_logical(workers).or_else(||labinius::par::CorePlacement::for_workers(workers));
  labinius::par::pool().run(workers,|worker|{
   let _guard=placement.as_ref().and_then(|p|p.pin(worker));
   let mut ws:Option<DiagWorker>=None;
   loop {
    let gi=next.fetch_add(1,std::sync::atomic::Ordering::Relaxed);
    if gi>=groups {break;}
    let c0=gi*g;let gc=g.min(columns.len()-c0);
    let ws=ws.get_or_insert_with(DiagWorker::new::<LOCK>);
    unsafe{group_pairs_diag::<LOCK,REG_SLICER,_>(&columns[c0..c0+gc],keys,np,ws,(dst as *mut [u32;N]).add(3*c0),|_|{})};
   }
  });
 }
 unsafe{rows.set_len(columns.len()*3);}
 rows
}
