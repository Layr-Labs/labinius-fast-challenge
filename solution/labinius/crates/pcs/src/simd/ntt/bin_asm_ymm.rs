//! Eight-column binary NTT in the current full-width index/key/PAcc layout.
//! The arithmetic and coefficient schedule are the PR147 binary pair path.
//! Only low16 i16 lanes and low8 i32 accumulator lanes are read or updated.
//! Full-width slicing, zeroing, inverse transforms and CRT remain in the caller.
use crate::params::*;
use crate::simd::ntt::r3_twiddles;
pub use crate::simd::transpose_f162::BinaryIndex32;
use core::arch::x86_64::*;

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

/// The existing binary schedule needs reductions for q9721 only.
const fn needs_barrett(q: u16) -> bool { 15 * q as u32 >= 2 * 32768 }

/// Keep the current duplicated-dword broadcast instruction.
#[inline(always)]
unsafe fn bc(p: *const u32) -> __m256i {
    let r: __m256i;
    core::arch::asm!(
        "vpbroadcastd {0:y}, dword ptr [{1}]",
        out(zmm_reg) r,
        in(reg) p,
        options(pure, readonly, nostack, preserves_flags)
    );
    r
}

/// Keep the current signed high-word multiply instruction.
#[inline(always)]
unsafe fn mulhi(a: __m256i, b: __m256i) -> __m256i {
    let r: __m256i;
    core::arch::asm!(
        "vpmulhw {0:y}, {1:y}, {2:y}",
        out(zmm_reg) r,
        in(zmm_reg) a,
        in(zmm_reg) b,
        options(pure, nomem, nostack, preserves_flags)
    );
    r
}

/// 3-uop signed Montgomery twiddle multiply: a * x mod q in (-q, q).
#[inline(always)]
unsafe fn mont(a: __m256i, w: __m256i, wp: __m256i, q: __m256i) -> __m256i {
    let m = _mm256_mullo_epi16(a, wp);
    let hi = mulhi(a, w);
    let t = mulhi(m, q);
    _mm256_sub_epi16(hi, t)
}

struct C {
    q: __m256i,
    om: __m256i,
    omp: __m256i,
}

/// Radix-3 butterfly whose twiddles are already folded into the inputs (level 3).
#[inline(always)]
unsafe fn r3_folded(c: &C, a0: __m256i, t1: __m256i, t2: __m256i) -> (__m256i, __m256i, __m256i) {
    let u = mont(_mm256_sub_epi16(t1, t2), c.om, c.omp, c.q);
    (
        _mm256_add_epi16(a0, _mm256_add_epi16(t1, t2)),
        _mm256_add_epi16(_mm256_sub_epi16(a0, t2), u),
        _mm256_sub_epi16(_mm256_sub_epi16(a0, t1), u),
    )
}

#[inline(always)]
unsafe fn ldb(p: *const u8, j: usize) -> __m256i {
    _mm256_load_si256(p.add(64 * j) as *const __m256i)
}
#[inline(always)]
unsafe fn st(p: *mut i16, j: usize, v: __m256i) {
    _mm256_store_si256(p.add(32 * j) as *mut __m256i, v);
}

/// Levels 4, 5 and 6 of one 27-block, entirely in registers (deep schedule).
macro_rules! mont {
    ($tw:literal, $off:literal, $a1:literal, $a2:literal, $t:literal) => {
        concat!(
            "vpbroadcastd ymm", $t, ", dword ptr [{", $tw, "}", $off, "+4]\n",
            "vpmullw ymm", $t, ", ymm", $a1, ", ymm", $t, "\n",
            "vpbroadcastd ymm30, dword ptr [{", $tw, "}", $off, "]\n",
            "vpmulhw ymm", $a1, ", ymm", $a1, ", ymm30\n",
            "vpmulhw ymm", $t, ", ymm", $t, ", ymm27\n",
            "vpsubw ymm", $a1, ", ymm", $a1, ", ymm", $t, "\n",
            "vpbroadcastd ymm", $t, ", dword ptr [{", $tw, "}", $off, "+12]\n",
            "vpmullw ymm", $t, ", ymm", $a2, ", ymm", $t, "\n",
            "vpbroadcastd ymm30, dword ptr [{", $tw, "}", $off, "+8]\n",
            "vpmulhw ymm", $a2, ", ymm", $a2, ", ymm30\n",
            "vpmulhw ymm", $t, ", ymm", $t, ", ymm27\n",
            "vpsubw ymm", $a2, ", ymm", $a2, ", ymm", $t, "\n",
            "vpsubw ymm", $t, ", ymm", $a1, ", ymm", $a2, "\n",
            "vpmullw ymm30, ymm", $t, ", [{c}+128]\n",
            "vpmulhw ymm", $t, ", ymm", $t, ", [{c}+64]\n",
            "vpmulhw ymm30, ymm30, ymm27\n",
            "vpsubw ymm", $t, ", ymm", $t, ", ymm30\n",
        )
    };
}

macro_rules! bfly {
    ($a0:literal, $a1:literal, $a2:literal, $t:literal) => {
        concat!(
            "vpaddw ymm30, ymm", $a1, ", ymm", $a2, "\n",
            "vpsubw ymm", $a2, ", ymm", $a0, ", ymm", $a2, "\n",
            "vpaddw ymm", $a2, ", ymm", $a2, ", ymm", $t, "\n",
            "vpsubw ymm", $a1, ", ymm", $a0, ", ymm", $a1, "\n",
            "vpsubw ymm", $a1, ", ymm", $a1, ", ymm", $t, "\n",
            "vpaddw ymm", $a0, ", ymm", $a0, ", ymm30\n",
        )
    };
    ($a0:literal, $a1:literal, $a2:literal, $t:literal, $o0:literal, $o1:literal, $o2:literal) => {
        concat!(
            "vpaddw ymm30, ymm", $a1, ", ymm", $a2, "\n",
            "vpsubw ymm", $a2, ", ymm", $a0, ", ymm", $a2, "\n",
            "vpaddw ymm", $a2, ", ymm", $a2, ", ymm", $t, "\n",
            "vmovdqa64 [{o}+", $o1, "], ymm", $a2, "\n",
            "vpsubw ymm", $a1, ", ymm", $a0, ", ymm", $a1, "\n",
            "vpsubw ymm", $a1, ", ymm", $a1, ", ymm", $t, "\n",
            "vmovdqa64 [{o}+", $o2, "], ymm", $a1, "\n",
            "vpaddw ymm", $a0, ", ymm", $a0, ", ymm30\n",
            "vmovdqa64 [{o}+", $o0, "], ymm", $a0, "\n",
        )
    };
}

macro_rules! level4 {
    (plain $a0:literal, $a1:literal, $a2:literal,
     $o0:literal, $o1:literal, $o2:literal, $t:literal) => {
        concat!(
            "vmovdqa64 ymm", $a0, ", [{i}+", $o0, "]\n",
            "vmovdqa64 ymm", $a1, ", [{i}+", $o1, "]\n",
            "vmovdqa64 ymm", $a2, ", [{i}+", $o2, "]\n",
            mont!("t4", "", $a1, $a2, $t),
        )
    };
    (lut $a0:literal, $a1:literal, $a2:literal,
     $o0:literal, $o1:literal, $o2:literal, $t:literal) => {
        concat!(
            level4!(plain $a0, $a1, $a2, $o0, $o1, $o2, $t),
            // logical a >> 11 is the window (a >> 11) & 31 (bits 11..15 of the lane), and
            // vpermw only reads the low 5 bits of each word index anyway.
            "vpsrlw ymm30, ymm", $a0, ", 11\n",
            "vpermi2w ymm30, ymm31, ymmword ptr [{c}+544]\n",
            "vpaddw ymm", $a0, ", ymm", $a0, ", ymm30\n",
        )
    };
}

macro_rules! level6 {
    (plain $off:literal, $a0:literal, $a1:literal, $a2:literal, $t:literal) => {
        mont!("t6", $off, $a1, $a2, $t)
    };
    (barrett $off:literal, $a0:literal, $a1:literal, $a2:literal, $t:literal) => {
        concat!(
            "vpbroadcastd ymm", $t, ", dword ptr [{t6}", $off, "+4]\n",
            "vpmullw ymm", $t, ", ymm", $a1, ", ymm", $t, "\n",
            "vpmulhrsw ymm30, ymm", $a0, ", [{c}+192]\n",
            "vpmullw ymm30, ymm30, ymm27\n",
            "vpsubw ymm", $a0, ", ymm", $a0, ", ymm30\n",
            "vpbroadcastd ymm30, dword ptr [{t6}", $off, "]\n",
            "vpmulhw ymm", $a1, ", ymm", $a1, ", ymm30\n",
            "vpmulhw ymm", $t, ", ymm", $t, ", ymm27\n",
            "vpsubw ymm", $a1, ", ymm", $a1, ", ymm", $t, "\n",
            "vpbroadcastd ymm", $t, ", dword ptr [{t6}", $off, "+12]\n",
            "vpmullw ymm", $t, ", ymm", $a2, ", ymm", $t, "\n",
            "vpbroadcastd ymm30, dword ptr [{t6}", $off, "+8]\n",
            "vpmulhw ymm", $a2, ", ymm", $a2, ", ymm30\n",
            "vpmulhw ymm", $t, ", ymm", $t, ", ymm27\n",
            "vpsubw ymm", $a2, ", ymm", $a2, ", ymm", $t, "\n",
            "vpsubw ymm", $t, ", ymm", $a1, ", ymm", $a2, "\n",
            "vpmullw ymm30, ymm", $t, ", [{c}+128]\n",
            "vpmulhw ymm", $t, ", ymm", $t, ", [{c}+64]\n",
            "vpmulhw ymm30, ymm30, ymm27\n",
            "vpsubw ymm", $t, ", ymm", $t, ", ymm30\n",
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
        concat!("vmovdqa64 ymm27, [{c}]\n", tail27!(plain, plain)),
        i = in(reg) bp,
        o = in(reg) op,
        t4 = in(reg) t4,
        t5 = in(reg) t5,
        t6 = in(reg) t6,
        c = in(reg) cv,
        out("ymm0") _,
        out("ymm1") _,
        out("ymm2") _,
        out("ymm3") _,
        out("ymm4") _,
        out("ymm5") _,
        out("ymm6") _,
        out("ymm7") _,
        out("ymm8") _,
        out("ymm9") _,
        out("ymm10") _,
        out("ymm11") _,
        out("ymm12") _,
        out("ymm13") _,
        out("ymm14") _,
        out("ymm15") _,
        out("ymm16") _,
        out("ymm17") _,
        out("ymm18") _,
        out("ymm19") _,
        out("ymm20") _,
        out("ymm21") _,
        out("ymm22") _,
        out("ymm23") _,
        out("ymm24") _,
        out("ymm25") _,
        out("ymm26") _,
        out("ymm27") _,
        out("ymm28") _,
        out("ymm29") _,
        out("ymm30") _,
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
            "vmovdqa64 ymm27, [{c}]\n",
            "vmovdqa64 ymm31, [{c}+512]\n",
            tail27!(lut, barrett),
        ),
        i = in(reg) bp,
        o = in(reg) op,
        t4 = in(reg) t4,
        t5 = in(reg) t5,
        t6 = in(reg) t6,
        c = in(reg) cv,
        out("ymm0") _,
        out("ymm1") _,
        out("ymm2") _,
        out("ymm3") _,
        out("ymm4") _,
        out("ymm5") _,
        out("ymm6") _,
        out("ymm7") _,
        out("ymm8") _,
        out("ymm9") _,
        out("ymm10") _,
        out("ymm11") _,
        out("ymm12") _,
        out("ymm13") _,
        out("ymm14") _,
        out("ymm15") _,
        out("ymm16") _,
        out("ymm17") _,
        out("ymm18") _,
        out("ymm19") _,
        out("ymm20") _,
        out("ymm21") _,
        out("ymm22") _,
        out("ymm23") _,
        out("ymm24") _,
        out("ymm25") _,
        out("ymm26") _,
        out("ymm27") _,
        out("ymm28") _,
        out("ymm29") _,
        out("ymm30") _,
        out("ymm31") _,
        options(nostack)
    );
}

#[repr(C, align(64))]
struct Blk([i16; 162 * 32]);

/// The final butterfly without its row-major scratch stores. The values and register
/// assignments are exactly those of `bfly!`; the arranged sink consumes them immediately.
macro_rules! bfly_registers {
    ($a0:literal, $a1:literal, $a2:literal, $t:literal,
     $o0:literal, $o1:literal, $o2:literal) => {
        bfly!($a0, $a1, $a2, $t)
    };
}

macro_rules! hybrid_store {
    (plain, $reg:literal, $off:literal) => {
        concat!("vmovdqa64 [{acc}+", $off, "], ymm", $reg, "\n")
    };
    (reduce, $reg:literal, $off:literal) => {
        concat!(
            "vpmaddwd ymm31, ymm", $reg, ", [{fold}]\n",
            "vptestmd k1, ymm", $reg, ", [{fold}+64]\n",
            "vpaddd ymm31{{k1}}, ymm31, [{fold}+128]\n",
            "vmovdqa64 [{acc}+", $off, "], ymm31\n",
        )
    };
}

#[repr(C, align(64))]
struct HybridFold([[i32; 16]; 3]);

const fn hybrid_fold<const Q: u16>() -> HybridFold {
    let r = (65536 % Q as i32) - Q as i32;
    HybridFold([
        [1 | ((r as u16 as i32) << 16); 16],
        [0x8000; 16],
        [r; 16],
    ])
}

macro_rules! pair_slot {
    ($mode:ident, $reg:literal, $w_off:literal, $k0:literal, $k1:literal, $k2:literal, $k3:literal, $a0:literal, $a1:literal) => {
        concat!(
            "vmovdqa64 ymm29, [{acc}+", $a0, "]\n",
            "vmovdqa64 ymm30, [{acc}+", $a1, "]\n",
            "vmovdqa64 ymm28, [{w0}+", $w_off, "]\n",
            "vpdpwssd ymm29, ymm28, dword ptr [{key}+", $k0, "]{{1to8}}\n",
            "vpdpwssd ymm30, ymm28, dword ptr [{key}+", $k1, "]{{1to8}}\n",
            "vpdpwssd ymm29, ymm", $reg, ", dword ptr [{key}+", $k2, "]{{1to8}}\n",
            "vpdpwssd ymm30, ymm", $reg, ", dword ptr [{key}+", $k3, "]{{1to8}}\n",
            hybrid_store!($mode, 29, $a0),
            hybrid_store!($mode, 30, $a1),
        )
    };
}

/// Register-resident LOCK=2 column-pair sink: consumes the second batch's 27 transformed vectors
/// directly from `ymm0..ymm26` while reading the first batch's 27 vectors from L1D-resident `w0`,
/// accumulating both against broadcast dword keys into `PAcc` (`[{acc} + 128*s]`, `[{acc} + 128*s + 64]`)
/// with zero `tr8`/`tr4` shuffles.
macro_rules! pair_register_sink {
    ($mode:ident) => {
        concat!(
            pair_slot!($mode, 0,    0,   0,   4,   8,  12,    0,   64),
            pair_slot!($mode, 2,   64,  16,  20,  24,  28,  128,  192),
            pair_slot!($mode, 1,  128,  32,  36,  40,  44,  256,  320),
            pair_slot!($mode, 6,  192,  48,  52,  56,  60,  384,  448),
            pair_slot!($mode, 8,  256,  64,  68,  72,  76,  512,  576),
            pair_slot!($mode, 7,  320,  80,  84,  88,  92,  640,  704),
            pair_slot!($mode, 3,  384,  96, 100, 104, 108,  768,  832),
            pair_slot!($mode, 5,  448, 112, 116, 120, 124,  896,  960),
            pair_slot!($mode, 4,  512, 128, 132, 136, 140, 1024, 1088),
            pair_slot!($mode, 18, 576, 144, 148, 152, 156, 1152, 1216),
            pair_slot!($mode, 20, 640, 160, 164, 168, 172, 1280, 1344),
            pair_slot!($mode, 19, 704, 176, 180, 184, 188, 1408, 1472),
            pair_slot!($mode, 24, 768, 192, 196, 200, 204, 1536, 1600),
            pair_slot!($mode, 26, 832, 208, 212, 216, 220, 1664, 1728),
            pair_slot!($mode, 25, 896, 224, 228, 232, 236, 1792, 1856),
            pair_slot!($mode, 21, 960, 240, 244, 248, 252, 1920, 1984),
            pair_slot!($mode, 23,1024, 256, 260, 264, 268, 2048, 2112),
            pair_slot!($mode, 22,1088, 272, 276, 280, 284, 2176, 2240),
            pair_slot!($mode, 9, 1152, 288, 292, 296, 300, 2304, 2368),
            pair_slot!($mode, 11,1216, 304, 308, 312, 316, 2432, 2496),
            pair_slot!($mode, 10,1280, 320, 324, 328, 332, 2560, 2624),
            pair_slot!($mode, 15,1344, 336, 340, 344, 348, 2688, 2752),
            pair_slot!($mode, 17,1408, 352, 356, 360, 364, 2816, 2880),
            pair_slot!($mode, 16,1472, 368, 372, 376, 380, 2944, 3008),
            pair_slot!($mode, 12,1536, 384, 388, 392, 396, 3072, 3136),
            pair_slot!($mode, 14,1600, 400, 404, 408, 412, 3200, 3264),
            pair_slot!($mode, 13,1664, 416, 420, 424, 428, 3328, 3392),
        )
    };
}

#[inline(always)]
unsafe fn tail27_pair<const Q: u16>(
    bp: *const i16, w0: *const i16, t4: *const u32, t5: *const u32, t6: *const u32,
    cv: *const i16, key: *const u32, acc: *mut i32, red: u32,
) {
    macro_rules! run {
        ($m4:ident, $m6:ident, $lookup:literal) => {
            core::arch::asm!(
                concat!(
                    "vmovdqa64 ymm27, [{c}]\n",
                    $lookup,
                    tail27!($m4, $m6, bfly_registers),
                    "test {red:e}, {red:e}\n",
                    "jnz 2f\n",
                    pair_register_sink!(plain),
                    "jmp 3f\n",
                    "2:\n",
                    pair_register_sink!(reduce),
                    "3:\n",
                    "/* {fold} */\n",
                ),
                i = in(reg) bp, w0 = in(reg) w0, t4 = in(reg) t4, t5 = in(reg) t5, t6 = in(reg) t6,
                c = in(reg) cv, key = in(reg) key, acc = in(reg) acc,
                fold = in(reg) &const { hybrid_fold::<Q>() },
                red = in(reg) red,
                out("ymm0") _,
                out("ymm1") _,
                out("ymm2") _,
                out("ymm3") _,
                out("ymm4") _,
                out("ymm5") _,
                out("ymm6") _,
                out("ymm7") _,
                out("ymm8") _,
                out("ymm9") _,
                out("ymm10") _,
                out("ymm11") _,
                out("ymm12") _,
                out("ymm13") _,
                out("ymm14") _,
                out("ymm15") _,
                out("ymm16") _,
                out("ymm17") _,
                out("ymm18") _,
                out("ymm19") _,
                out("ymm20") _,
                out("ymm21") _,
                out("ymm22") _,
                out("ymm23") _,
                out("ymm24") _,
                out("ymm25") _,
                out("ymm26") _,
                out("ymm27") _,
                out("ymm28") _,
                out("ymm29") _,
                out("ymm30") _,
                out("ymm31") _,
                out("k1") _,
                options(nostack)
            )
        };
    }
    if needs_barrett(Q) {
        run!(lut, barrett, "vmovdqa64 ymm31, [{c}+512]\n");
    } else {
        run!(plain, plain, "");
    }
}

#[repr(C, align(64))]
struct Blk27([i16; 27 * 32]);

/// Accumulate two pair batches for eight columns, preserving high8 PAcc dwords.
///
/// # Safety
/// AVX-512F/BW/VL/VBMI/VBMI2/VNNI and GFNI are required. Inputs retain the
/// BinaryIndex32 layout. Key and accumulator extents and alignments are the
/// current wide pair API. The caller keeps the current accumulator periods.
#[inline(never)]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vnni,gfni")]
pub unsafe fn ntt_bin_batch16_pair<const Q: u16>(
    ins: [&BinaryIndex32; 2],
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
    let ips: [*const u8; 2] = [
        ins[0].rows.as_ptr() as *const u8,
        ins[1].rows.as_ptr() as *const u8,
    ];
    let mut blks: [core::mem::MaybeUninit<Blk>; 4] =
        [const { core::mem::MaybeUninit::uninit() }; 4];
    let bps: [*mut i16; 4] = [
        blks[0].as_mut_ptr() as *mut i16,
        blks[1].as_mut_ptr() as *mut i16,
        blks[2].as_mut_ptr() as *mut i16,
        blks[3].as_mut_ptr() as *mut i16,
    ];
    let mut w0: core::mem::MaybeUninit<Blk27> = core::mem::MaybeUninit::uninit();
    let w0p = w0.as_mut_ptr() as *mut i16;
    let red_u32 = red as u32;

    // Process fixed adjacent quarter pairs: both prefixes share each six-row index load.
    // Each quarter retains its own two LOCK=2 buffers until its original six tails finish.
    for kp in 0..2 {
        let load_luts = |k: usize| {
            let lut = t.lut.as_ptr().add(12 * k) as *const u8;
            let l = |r: usize, ab: usize| -> __m256i {
                _mm256_load_si256(lut.add(64 * (r * 2 + ab)) as *const __m256i)
            };
            (l(0, 0), l(0, 1), l(1, 0), l(1, 1), l(2, 0), l(2, 1))
        };
        let ls0 = load_luts(2 * kp);
        let ls1 = load_luts(2 * kp + 1);
        for h in 0..2 {
            let ip = ips[h];
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
                        let x = _mm256_permutexvar_epi8(n0, l000);
                        let y = _mm256_permutexvar_epi8(n0h, l001);
                        let a0 = _mm256_add_epi16(x, y);
                        let b0 = _mm256_sub_epi16(x, y);

                        let x = _mm256_permutexvar_epi8(n1, l010);
                        let y = _mm256_permutexvar_epi8(n1h, l011);
                        let a1 = _mm256_add_epi16(x, y);
                        let b1 = _mm256_sub_epi16(y, x);

                        let x = _mm256_permutexvar_epi8(n2, l020);
                        let y = _mm256_permutexvar_epi8(n2h, l021);
                        let a2 = _mm256_add_epi16(x, y);
                        let b2 = _mm256_sub_epi16(x, y);

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
                prefix!(bps[h], ls0);
                prefix!(bps[2 + h], ls1);
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
                let bp0 = bps[2 * qpair].add(32 * 27 * j);
                if bar {
                    tail27_b(bp0, w0p, t4, t5, t6, cv);
                } else {
                    tail27_p(bp0, w0p, t4, t5, t6, cv);
                }
                let bp1 = bps[2 * qpair + 1].add(32 * 27 * j);
                tail27_pair::<Q>(
                    bp1,
                    w0p,
                    t4,
                    t5,
                    t6,
                    cv,
                    key.add(4 * 27 * kk),
                    acc.add(32 * 27 * kk),
                    red_u32,
                );
            }
        }
    }
}
