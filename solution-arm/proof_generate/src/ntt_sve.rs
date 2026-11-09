//! SVE2 paired transform for VL = 256 hosts (Neoverse V2): [`pass_c_sve`] and [`pass_d_sve`]
//! process two eight-column batches per vector, every operation element-wise identical to the
//! NEON kernels ([`super::pass_c`], [`super::pass_d`]) they replace. Only the instruction width
//! differs, so the transforms' lazy-reduction bounds carry over unchanged.
//!
//! The paired transform domain lives in a tile-major buffer of 32-byte coefficient rows
//! (`xsP[t][s]` = coefficient `s` of transform tile `t`, low half batch A, high half batch B):
//! [`pass_lut2`] is the NEON lookup front end storing each batch into its half; the sink's asm
//! walks the halves through a runtime tile stride. Hosts without SVE2 at sixteen `i16` lanes
//! take the unpaired NEON path in [`super::sms`].
use super::*;

/// The vector length in bytes, from `RDVL` (stable rustc has no SVE intrinsics; the kernels
/// below are raw asm over full 256-bit z registers, the sink's approach).
pub fn vector_length() -> usize {
    let vl: i64;
    // SAFETY: RDVL is a register read.
    unsafe { core::arch::asm!("rdvl {0}, #8", out(reg) vl, options(nomem, nostack, preserves_flags)) };
    vl as usize
}

/// Whether this host can run the paired kernels: an SVE vector of exactly 16 `i16` lanes.
pub fn paired_supported() -> bool {
    vector_length() == 256
}

/// The SVE register pool of one prime: `p` and `round(2^26 / p)` as splats, `eps` and `eps'`,
/// then per level-3 node the twelve twiddle splats of levels 4-5, and per level-6 group the
/// eight twiddle splats, each a full 32-byte row the kernels `ldr` at slot-scaled immediates.
pub struct SvePool {
    pub consts: [[i16; 16]; 5],
    pub c: Vec<[[i16; 16]; 16]>,
    pub d: Vec<[[i16; 16]; 8]>,
}

fn splat(x: i16) -> [i16; 16] {
    [x; 16]
}

impl SvePool {
    fn new(index: usize) -> SvePool {
        let k = &kernels()[index];
        SvePool {
            // [p, the 2^13-1 residue mask, eps, eps', the FIRST13 rounding bit 2^12]
            consts: [splat(k.k[0]), splat((1 << 13) - 1), splat(k.k[2]), splat(k.k[3]), splat(1 << 12)],
            c: k.c
                .iter()
                .map(|row| {
                    [
                        splat(row[0]), splat(row[1]), splat(row[2]), splat(row[3]),
                        splat(row[4]), splat(row[5]), splat(row[6]), splat(row[7]),
                        splat(row[8]), splat(row[9]), splat(row[10]), splat(row[11]),
                        splat(row[12]), splat(row[13]), splat(row[14]), splat(row[15]),
                    ]
                })
                .collect(),
            d: k.d
                .iter()
                .map(|row| {
                    [
                        splat(row[0]), splat(row[1]), splat(row[2]), splat(row[3]),
                        splat(row[4]), splat(row[5]), splat(row[6]), splat(row[7]),
                    ]
                })
                .collect(),
        }
    }
}

static POOLS: std::sync::OnceLock<Vec<SvePool>> = std::sync::OnceLock::new();

fn pool(index: usize) -> &'static SvePool {
    &POOLS.get_or_init(|| (0..3).map(SvePool::new).collect())[index]
}

/// The 24 level-3 nodes' kinds of prime 9721's schedule ([`super::lut_node_kind`]), baked as
/// literals for the asm string selection; a test asserts the two agree.
const KINDS: [(bool, [bool; 3]); 24] = [
    (true, [false, true, false]),
    (false, [false, false, false]),
    (true, [false, false, false]),
    (true, [false, false, true]),
    (true, [false, false, false]),
    (true, [true, true, true]),
    (true, [false, true, true]),
    (true, [false, true, true]),
    (true, [false, true, false]),
    (true, [false, true, false]),
    (true, [true, true, true]),
    (true, [false, true, true]),
    (true, [true, true, false]),
    (true, [false, false, false]),
    (true, [false, false, true]),
    (false, [false, false, false]),
    (true, [false, false, true]),
    (true, [false, false, false]),
    (true, [false, false, true]),
    (true, [false, false, true]),
    (true, [false, true, false]),
    (true, [false, false, true]),
    (true, [false, true, false]),
    (true, [false, true, true]),
];

/// Literal string selection (a runtime `if` cannot feed `concat!`).
macro_rules! str_if {
    (true, $t:expr, $f:expr) => {
        $t
    };
    (false, $t:expr, $f:expr) => {
        $f
    };
}

/// One in-place radix-3 butterfly over full 256-bit registers, [`super::radix3`] with
/// `reduce_d = false` (prime 9721's schedule): `zA,zB,zC` (literals `1..=9`) hold `a,b,c` in,
/// `y0,y1,y2` out. z0 = the `p` splat (`mls`'s indexed operand), z17/z18 = `eps`/`eps'` splats;
/// z10..z15 scratch; `.h` throughout. Twiddles `z20..z31` are per-twiddle splat vectors.
macro_rules! r3s {
    ($a:literal, $b:literal, $c:literal, $w:expr, $t:expr, $w1:expr, $t1:expr) => {
        concat!(
        "mul      z10.h, z", $b, ".h, ", $w, "\n",
        "sqrdmulh z11.h, z", $b, ".h, ", $t, "\n",
        "mls      z10.h, z11.h, z0.h[0]\n",                // b1 = b w - PERP p
        "mul      z12.h, z", $c, ".h, ", $w1, "\n",
        "sqrdmulh z11.h, z", $c, ".h, ", $t1, "\n",
        "mls      z12.h, z11.h, z0.h[0]\n",                // c1
        "add      z13.h, z10.h, z12.h\n",                  // s
        "sub      z14.h, z10.h, z12.h\n",                  // d (unreduced)
        "mul      z15.h, z14.h, z17.h\n",
        "sqrdmulh z11.h, z14.h, z18.h\n",
        "mls      z15.h, z11.h, z0.h[0]\n",                // e = eps d
        "sub      z11.h, z", $a, ".h, z12.h\n",
        "add      z", $b, ".h, z11.h, z15.h\n",            // y1 = (a - c1) + e
        "sub      z11.h, z", $a, ".h, z10.h\n",
        "sub      z", $c, ".h, z11.h, z15.h\n",            // y2 = (a - b1) - e
        "add      z", $a, ".h, z", $a, ".h, z13.h\n",      // y0 = a + s
        )
    };
}

/// `zX -= round(zX / 2^13) p` in place ([`super::reduce_round13`]), exact for every `i16`
/// input: `VRSHR #13`'s exact (non-wrapping) add is decomposed into the floor quotient plus the
/// rounding carry of the positive residue, whose own add (`<= 8191 + 4096`) cannot wrap; `mls`
/// by the `p` splat's lane. The lanes are the NEON kernel's bit for bit. (The FIRST13 sites'
/// inputs are bounded by the transform's interval arithmetic at 29160 for prime 9721 - ABOVE
/// the naive `x + 4096` wrapping-add threshold of 28671, so the decomposition is required.)
/// z16 = the `2^13 - 1` residue mask splat, z19 = the `2^12` rounding splat.
macro_rules! r13 {
    ($x:literal) => {
        concat!(
        "asr z10.h, z", $x, ".h, #13\n",
        "and z11.h, z", $x, ".h, z16.h\n",
        "add z11.h, z11.h, z19.h\n",
        "asr z11.h, z11.h, #13\n",
        "add z10.h, z10.h, z11.h\n",
        "mls z", $x, ".h, z10.h, z0.h[0]\n",
        )
    };
}

/// `pass_c_sve` for prime 9721's table-front schedule (`S == 4` of the NEON code): levels 4 and
/// 5 fused per slot column `j` of each of the 24 level-3 nodes. Per column: nine paired rows in
/// (slots `j + 3 i`, the `v[i]` of the NEON code), both levels' butterflies in registers, the
/// level-5 outputs back to the same slots. `xp` is the tile's paired rows, 32 bytes per
/// coefficient; `tw` the prime's splats.
///
/// Register map: z0 `p`, z1..z9 the nine values `v[0..9]`, z10..z15 scratch, z16 the `2^12`
/// rounding splat, z17/z18 `eps`/`eps'`, z20..z31 the node's twelve twiddle splats.
/// `$r4` reduces the level-4 first inputs `v[0..3]`; `$skip_g` leaves level-5 group `g`'s first
/// input `v[3g]` unreduced ([`super::lut_node_kind`], baked in [`KINDS`]).
#[cfg(target_feature = "sve2")]
unsafe fn pass_c_sve_s4(xp: *mut u16, kern: &Kernel, tw: &SvePool) {
    let _ = kern;
    macro_rules! block {
        ($n:literal, $j:literal, $r4:tt, $s0:tt, $s1:tt, $s2:tt) => {
            let xpn = xp.add(16 * 27 * $n + 16 * $j);
            let tp = tw.c[$n].as_ptr() as *const u8;
            let cp = tw.consts.as_ptr() as *const u8;
            // SAFETY: the block reads and writes only the tile's paired rows (xpn), the
            // read-only pools (tp, cp) and z registers; every value stays in registers between
            // its load and store. Slot immediates are scaled by VL bytes (`mul vl`).
            core::arch::asm!(
                concat!(
                "ldr z0, [{cp}]\n",
                "ldr z16, [{cp}, #1, mul vl]\n",
                "ldr z17, [{cp}, #2, mul vl]\n",
                "ldr z18, [{cp}, #3, mul vl]\n",
                "ldr z19, [{cp}, #4, mul vl]\n",
                "ldr z20, [{tp}]\n",
                "ldr z21, [{tp}, #1, mul vl]\n",
                "ldr z22, [{tp}, #2, mul vl]\n",
                "ldr z23, [{tp}, #3, mul vl]\n",
                "ldr z24, [{tp}, #4, mul vl]\n",
                "ldr z25, [{tp}, #5, mul vl]\n",
                "ldr z26, [{tp}, #6, mul vl]\n",
                "ldr z27, [{tp}, #7, mul vl]\n",
                "ldr z28, [{tp}, #8, mul vl]\n",
                "ldr z29, [{tp}, #9, mul vl]\n",
                "ldr z30, [{tp}, #10, mul vl]\n",
                "ldr z31, [{tp}, #11, mul vl]\n",
                // v[0] = slot j, v[3] = j + 9, v[6] = j + 18
                "ldr z1, [{xpn}]\n",
                "ldr z4, [{xpn}, #9, mul vl]\n",
                "ldr z7, [{xpn}, #18, mul vl]\n",
                str_if!($r4, r13!(1), ""),
                r3s!(1, 4, 7, "z20.h", "z21.h", "z22.h", "z23.h"),
                "ldr z2, [{xpn}, #3, mul vl]\n",
                "ldr z5, [{xpn}, #12, mul vl]\n",
                "ldr z8, [{xpn}, #21, mul vl]\n",
                str_if!($r4, r13!(2), ""),
                r3s!(2, 5, 8, "z20.h", "z21.h", "z22.h", "z23.h"),
                "ldr z3, [{xpn}, #6, mul vl]\n",
                "ldr z6, [{xpn}, #15, mul vl]\n",
                "ldr z9, [{xpn}, #24, mul vl]\n",
                str_if!($r4, r13!(3), ""),
                r3s!(3, 6, 9, "z20.h", "z21.h", "z22.h", "z23.h"),
                // the level-5 groups' first inputs v[0], v[3], v[6], unless skipped
                str_if!($s0, "", r13!(1)),
                str_if!($s1, "", r13!(4)),
                str_if!($s2, "", r13!(7)),
                "str z1, [{xpn}]\n",
                "str z2, [{xpn}, #3, mul vl]\n",
                "str z3, [{xpn}, #6, mul vl]\n",
                "str z4, [{xpn}, #9, mul vl]\n",
                "str z5, [{xpn}, #12, mul vl]\n",
                "str z6, [{xpn}, #15, mul vl]\n",
                "str z7, [{xpn}, #18, mul vl]\n",
                "str z8, [{xpn}, #21, mul vl]\n",
                "str z9, [{xpn}, #24, mul vl]\n",
                ),
                xpn = in(reg) xpn,
                tp = in(reg) tp,
                cp = in(reg) cp,
                out("v0") _, out("v1") _, out("v2") _, out("v3") _, out("v4") _, out("v5") _,
                out("v6") _, out("v7") _, out("v8") _, out("v9") _, out("v10") _, out("v11") _,
                out("v12") _, out("v13") _, out("v14") _, out("v15") _, out("v16") _, out("v17") _,
                out("v18") _, out("v19") _, out("v20") _, out("v21") _, out("v22") _, out("v23") _,
                out("v24") _, out("v25") _, out("v26") _, out("v27") _, out("v28") _, out("v29") _,
                out("v30") _, out("v31") _,
                options(nostack)
            );
            // Level 5 in a second block, the level-4 outputs through memory: three groups of
            // butterflies over the reloaded rows, outputs back to the same slots.
            // SAFETY: as the first block; only the tile's paired rows and the pools.
            core::arch::asm!(
                concat!(
                "ldr z0, [{cp}]\n",
                "ldr z17, [{cp}, #2, mul vl]\n",
                "ldr z18, [{cp}, #3, mul vl]\n",
                "ldr z24, [{tp}, #4, mul vl]\n",
                "ldr z25, [{tp}, #5, mul vl]\n",
                "ldr z26, [{tp}, #6, mul vl]\n",
                "ldr z27, [{tp}, #7, mul vl]\n",
                "ldr z28, [{tp}, #8, mul vl]\n",
                "ldr z29, [{tp}, #9, mul vl]\n",
                "ldr z30, [{tp}, #10, mul vl]\n",
                "ldr z31, [{tp}, #11, mul vl]\n",
                "ldr z1, [{xpn}]\n",
                "ldr z2, [{xpn}, #3, mul vl]\n",
                "ldr z3, [{xpn}, #6, mul vl]\n",
                "ldr z4, [{xpn}, #9, mul vl]\n",
                "ldr z5, [{xpn}, #12, mul vl]\n",
                "ldr z6, [{xpn}, #15, mul vl]\n",
                "ldr z7, [{xpn}, #18, mul vl]\n",
                "ldr z8, [{xpn}, #21, mul vl]\n",
                "ldr z9, [{xpn}, #24, mul vl]\n",
                r3s!(1, 2, 3, "z24.h", "z25.h", "z26.h", "z27.h"),
                r3s!(4, 5, 6, "z28.h", "z29.h", "z30.h", "z31.h"),
                // group 2's twiddles: t1 lanes 4..7 = pool rows 12..16
                "ldr z28, [{tp}, #12, mul vl]\n",
                "ldr z29, [{tp}, #13, mul vl]\n",
                "ldr z30, [{tp}, #14, mul vl]\n",
                "ldr z31, [{tp}, #15, mul vl]\n",
                r3s!(7, 8, 9, "z28.h", "z29.h", "z30.h", "z31.h"),
                "str z1, [{xpn}]\n",
                "str z2, [{xpn}, #3, mul vl]\n",
                "str z3, [{xpn}, #6, mul vl]\n",
                "str z4, [{xpn}, #9, mul vl]\n",
                "str z5, [{xpn}, #12, mul vl]\n",
                "str z6, [{xpn}, #15, mul vl]\n",
                "str z7, [{xpn}, #18, mul vl]\n",
                "str z8, [{xpn}, #21, mul vl]\n",
                "str z9, [{xpn}, #24, mul vl]\n",
                ),
                xpn = in(reg) xpn,
                tp = in(reg) tp,
                cp = in(reg) cp,
                out("v0") _, out("v1") _, out("v2") _, out("v3") _, out("v4") _, out("v5") _,
                out("v6") _, out("v7") _, out("v8") _, out("v9") _, out("v10") _, out("v11") _,
                out("v12") _, out("v13") _, out("v14") _, out("v15") _, out("v16") _, out("v17") _,
                out("v18") _, out("v19") _, out("v20") _, out("v21") _, out("v22") _, out("v23") _,
                out("v24") _, out("v25") _, out("v26") _, out("v27") _, out("v28") _, out("v29") _,
                out("v30") _, out("v31") _,
                options(nostack)
            );
        };
    }
    macro_rules! node3 {
        ($n:literal, $r4:tt, $s0:tt, $s1:tt, $s2:tt) => {
            block!($n, 0, $r4, $s0, $s1, $s2);
            block!($n, 1, $r4, $s0, $s1, $s2);
            block!($n, 2, $r4, $s0, $s1, $s2);
        };
    }
    node3!(0, true, false, true, false);
    node3!(1, false, false, false, false);
    node3!(2, true, false, false, false);
    node3!(3, true, false, false, true);
    node3!(4, true, false, false, false);
    node3!(5, true, true, true, true);
    node3!(6, true, false, true, true);
    node3!(7, true, false, true, true);
    node3!(8, true, false, true, false);
    node3!(9, true, false, true, false);
    node3!(10, true, true, true, true);
    node3!(11, true, false, true, true);
    node3!(12, true, true, true, false);
    node3!(13, true, false, false, false);
    node3!(14, true, false, false, true);
    node3!(15, false, false, false, false);
    node3!(16, true, false, false, true);
    node3!(17, true, false, false, false);
    node3!(18, true, false, false, true);
    node3!(19, true, false, false, true);
    node3!(20, true, false, true, false);
    node3!(21, true, false, false, true);
    node3!(22, true, false, true, false);
    node3!(23, true, false, true, true);
}

/// Levels 4 and 5 over the paired rows of one tile ([`super::pass_c`]); SMS uses prime 9721's
/// table-front schedule. `xp` points at the tile's `N` paired 32-byte rows.
#[cfg(target_feature = "sve2")]
#[inline(never)]
pub unsafe fn pass_c_sve<const S: usize>(xp: *mut u16, kern: &Kernel) {
    match S {
        4 => pass_c_sve_s4(xp, kern, pool(1)),
        _ => unreachable!("the SMS path only uses the table-front schedule"),
    }
}

/// Level 6 over the paired rows of one tile ([`super::pass_d`], store mode) for prime 9721's
/// schedule: the two butterflies of each of the 36 slot groups, with the schedule's two
/// `FIRST13` input reductions (v[0] and v[3]). `xp` is the tile's paired rows.
#[cfg(target_feature = "sve2")]
#[inline(never)]
pub unsafe fn pass_d_sve<const S: usize>(xp: *mut u16, kern: &Kernel) {
    let _ = (kern, S);
    let tw = pool(1);
    macro_rules! group {
        ($q:literal) => {
            let xq = xp.add(16 * 6 * $q);
            let tp = tw.d[$q].as_ptr() as *const u8;
            let cp = tw.consts.as_ptr() as *const u8;
            // SAFETY: as pass_c_sve's block; only the group's six paired rows and the pools.
            core::arch::asm!(
                concat!(
                "ldr z0, [{cp}]\n",
                "ldr z16, [{cp}, #1, mul vl]\n",
                "ldr z17, [{cp}, #2, mul vl]\n",
                "ldr z18, [{cp}, #3, mul vl]\n",
                "ldr z19, [{cp}, #4, mul vl]\n",
                "ldr z20, [{tp}]\n",
                "ldr z21, [{tp}, #1, mul vl]\n",
                "ldr z22, [{tp}, #2, mul vl]\n",
                "ldr z23, [{tp}, #3, mul vl]\n",
                "ldr z24, [{tp}, #4, mul vl]\n",
                "ldr z25, [{tp}, #5, mul vl]\n",
                "ldr z26, [{tp}, #6, mul vl]\n",
                "ldr z27, [{tp}, #7, mul vl]\n",
                "ldr z1, [{xq}]\n",
                "ldr z4, [{xq}, #3, mul vl]\n",
                r13!(1),
                r13!(4),
                "ldr z2, [{xq}, #1, mul vl]\n",
                "ldr z3, [{xq}, #2, mul vl]\n",
                "ldr z5, [{xq}, #4, mul vl]\n",
                "ldr z6, [{xq}, #5, mul vl]\n",
                r3s!(1, 2, 3, "z20.h", "z21.h", "z22.h", "z23.h"),
                r3s!(4, 5, 6, "z24.h", "z25.h", "z26.h", "z27.h"),
                "str z1, [{xq}]\n",
                "str z2, [{xq}, #1, mul vl]\n",
                "str z3, [{xq}, #2, mul vl]\n",
                "str z4, [{xq}, #3, mul vl]\n",
                "str z5, [{xq}, #4, mul vl]\n",
                "str z6, [{xq}, #5, mul vl]\n",
                ),
                xq = in(reg) xq as *const u8,
                tp = in(reg) tp,
                cp = in(reg) cp,
                out("v0") _, out("v1") _, out("v2") _, out("v3") _, out("v4") _, out("v5") _,
                out("v6") _, out("v7") _, out("v8") _, out("v9") _, out("v10") _, out("v11") _,
                out("v12") _, out("v13") _, out("v14") _, out("v15") _, out("v16") _, out("v17") _,
                out("v18") _, out("v19") _, out("v20") _, out("v21") _, out("v22") _, out("v23") _,
                out("v24") _, out("v25") _, out("v26") _, out("v27") _, out("v28") _, out("v29") _,
                out("v30") _, out("v31") _,
                options(nostack)
            );
        };
    }
    group!(0); group!(1); group!(2);
    group!(3); group!(4); group!(5);
    group!(6); group!(7); group!(8);
    group!(9); group!(10); group!(11);
    group!(12); group!(13); group!(14);
    group!(15); group!(16); group!(17);
    group!(18); group!(19); group!(20);
    group!(21); group!(22); group!(23);
    group!(24); group!(25); group!(26);
    group!(27); group!(28); group!(29);
    group!(30); group!(31); group!(32);
    group!(33); group!(34); group!(35);
    group!(36); group!(37); group!(38);
    group!(39); group!(40); group!(41);
    group!(42); group!(43); group!(44);
    group!(45); group!(46); group!(47);
    group!(48); group!(49); group!(50);
    group!(51); group!(52); group!(53);
    group!(54); group!(55); group!(56);
    group!(57); group!(58); group!(59);
    group!(60); group!(61); group!(62);
    group!(63); group!(64); group!(65);
    group!(66); group!(67); group!(68);
    group!(69); group!(70); group!(71);
    group!(72); group!(73); group!(74);
    group!(75); group!(76); group!(77);
    group!(78); group!(79); group!(80);
    group!(81); group!(82); group!(83);
    group!(84); group!(85); group!(86);
    group!(87); group!(88); group!(89);
    group!(90); group!(91); group!(92);
    group!(93); group!(94); group!(95);
    group!(96); group!(97); group!(98);
    group!(99); group!(100); group!(101);
    group!(102); group!(103); group!(104);
    group!(105); group!(106); group!(107);
}

/// [`super::pass_lut`] (levels 0 and 1 from the nibble tables) for one batch of eight columns,
/// storing into the paired rows: coefficient `at` of this batch lands in half `half` of
/// `xsp[2 at + half]`, `xsp` viewed as `i16x8` pairs.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub unsafe fn pass_lut2<const S: usize>(
    idx: &[uint8x16_t; GROUPS],
    xsp: *mut int16x8_t,
    half: usize,
    kern: &Kernel,
) {
    let s: Schedule = schedule(S);
    const { assert!(S < 3 && crate::ntt::lut_transform_bound(PRIMES[S], SCHEDULES[S]) <= i16::MAX as u32) };
    let k = vld1q_s16(kern.k.as_ptr());
    let ix = |i: usize| *idx.get_unchecked(i);
    macro_rules! store2 {
        ($at:expr, $v:expr) => {
            *xsp.add(2 * $at + half) = $v;
        };
    }
    for n in 0..4 {
        let l = &kern.lut[n];
        let t: [uint8x16x2_t; 6] = core::array::from_fn(|i| table(&l[i]));
        let base = 162 * n;
        let mut step = |j: usize| {
            let (x0, y0) = (look(t[0], ix(j)), look(t[1], ix(j + 81)));
            let (x1, y1) = (look(t[2], ix(j + 27)), look(t[3], ix(j + 108)));
            let (x2, y2) = (look(t[4], ix(j + 54)), look(t[5], ix(j + 135)));
            let (a0, b0) = (vaddq_s16(x0, y0), vsubq_s16(x0, y0));
            let (a1, b1) = (vaddq_s16(x1, y1), vsubq_s16(x1, y1));
            let (a2, b2) = (vaddq_s16(x2, y2), vsubq_s16(x2, y2));
            let (u, v) = match (s.reduce_d, s.cheap) {
                (false, _) => (radix3_twiddled::<0>(a0, a1, a2, k), radix3_sixth::<0>(b0, b1, b2, k)),
                (true, false) => (radix3_twiddled::<1>(a0, a1, a2, k), radix3_sixth::<1>(b0, b1, b2, k)),
                (true, true) => (radix3_twiddled::<2>(a0, a1, a2, k), radix3_sixth::<2>(b0, b1, b2, k)),
            };
            let at = base + j;
            store2!(at, u.0);
            store2!(at + 27, u.1);
            store2!(at + 54, u.2);
            store2!(at + 81, v.0);
            store2!(at + 108, v.1);
            store2!(at + 135, v.2);
        };
        // nine `j` per block, so that the scheduler interleaves their lookups and butterflies
        for j in (0..27).step_by(9) {
            step(j);
            step(j + 1);
            step(j + 2);
            step(j + 3);
            step(j + 4);
            step(j + 5);
            step(j + 6);
            step(j + 7);
            step(j + 8);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The baked [`KINDS`] match [`lut_node_kind`]'s const computation.
    #[test]
    fn kinds_match_neon() {
        for n in 0..24 {
            let (r4, skip) = lut_node_kind(n);
            let bits = [skip & 1 != 0, skip & 2 != 0, skip & 4 != 0];
            assert_eq!(KINDS[n], (r4, bits), "node {n}");
        }
    }
}

#[cfg(all(test, target_feature = "sve2"))]
mod full_probe {
    use super::*;

    /// The complete node-0 column-0 block (loads, three interleaved level-4 butterflies, the
    /// post reductions, the three level-5 butterflies, in-place stores) on a 27-row buffer,
    /// every intermediate dumped and checked against the NEON kernels' own primitives.
    #[test]
    fn full_block_node0_matches() {
        if !paired_supported() {
            return;
        }
        let kern = &kernels()[1];
        let tw = pool(1);
        unsafe {
            let mut st = 4242u64;
            let mut next = move || {
                st ^= st << 13;
                st ^= st >> 7;
                st ^= st << 17;
                st
            };
            let mut rows = Box::new([[0i16; 16]; 27]);
            for (s, row) in rows.iter_mut().enumerate() {
                for l in 0..8 {
                    let x = (((s as i32 * 131 + (next() % 29100) as i32) % 29100) - 14550) as i16;
                    row[l] = x;
                    row[l + 8] = x;
                }
            }
            let mut post = [[0i16; 16]; 9]; // z16, z1 loaded, z1 after r13, z7-z9 after post, then z1-z3 after post
            let mut fin = [[0i16; 16]; 8];  // z7, z8, z9 after level 5, then group-0 intermediates
            let pristine: Box<[[i16; 16]; 27]> = rows.clone();
            let xpn = rows.as_mut_ptr() as *mut u8;
            let tp = tw.c[0].as_ptr() as *const u8;
            let cp = tw.consts.as_ptr() as *const u8;
            core::arch::asm!(
                concat!(
                    "ldr z0, [{cp}]\n",
                    "ldr z16, [{cp}, #1, mul vl]\n",
                    "ldr z17, [{cp}, #2, mul vl]\n",
                    "ldr z18, [{cp}, #3, mul vl]\n",
                    "ldr z19, [{cp}, #4, mul vl]\n",
                    "ldr z20, [{tp}]\n",
                    "ldr z21, [{tp}, #1, mul vl]\n",
                    "ldr z22, [{tp}, #2, mul vl]\n",
                    "ldr z23, [{tp}, #3, mul vl]\n",
                    "ldr z1, [{xpn}]\n",
                    "ldr z4, [{xpn}, #9, mul vl]\n",
                    "ldr z7, [{xpn}, #18, mul vl]\n",
                    "str z16, [{pp}]\n",
                    "str z1, [{pp}, #1, mul vl]\n",
                    r13!(1),
                    "str z1, [{pp}, #2, mul vl]\n",
                    r3s!(1, 4, 7, "z20.h", "z21.h", "z22.h", "z23.h"),
                    "ldr z2, [{xpn}, #3, mul vl]\n",
                    "ldr z5, [{xpn}, #12, mul vl]\n",
                    "ldr z8, [{xpn}, #21, mul vl]\n",
                    r13!(2),
                    r3s!(2, 5, 8, "z20.h", "z21.h", "z22.h", "z23.h"),
                    "ldr z3, [{xpn}, #6, mul vl]\n",
                    "ldr z6, [{xpn}, #15, mul vl]\n",
                    "ldr z9, [{xpn}, #24, mul vl]\n",
                    r13!(3),
                    r3s!(3, 6, 9, "z20.h", "z21.h", "z22.h", "z23.h"),
                    // node 0 skips: v[3] only, so reduce v[0] (z1), v[6] (z7); v[3] (z4) skipped
                    r13!(1),
                    r13!(7),
                    "str z7, [{pp}, #3, mul vl]\n",
                    "str z8, [{pp}, #4, mul vl]\n",
                    "str z9, [{pp}, #5, mul vl]\n",
                    "str z1, [{pp}, #6, mul vl]\n",
                    "str z2, [{pp}, #7, mul vl]\n",
                    "str z3, [{pp}, #8, mul vl]\n",
                    "ldr z24, [{tp}, #4, mul vl]\n",
                    "ldr z25, [{tp}, #5, mul vl]\n",
                    "ldr z26, [{tp}, #6, mul vl]\n",
                    "ldr z27, [{tp}, #7, mul vl]\n",
                    "ldr z28, [{tp}, #8, mul vl]\n",
                    "ldr z29, [{tp}, #9, mul vl]\n",
                    "ldr z30, [{tp}, #10, mul vl]\n",
                    "ldr z31, [{tp}, #11, mul vl]\n",
                    "mul      z10.h, z2.h, z24.h\n",
                    "sqrdmulh z11.h, z2.h, z25.h\n",
                    "mls      z10.h, z11.h, z0.h[0]\n",
                    "str z10, [{ff}, #3, mul vl]\n",
                    "str z11, [{ff}, #4, mul vl]\n",
                    "mul      z12.h, z3.h, z26.h\n",
                    "sqrdmulh z11.h, z3.h, z27.h\n",
                    "mls      z12.h, z11.h, z0.h[0]\n",
                    "str z12, [{ff}, #5, mul vl]\n",
                    "add      z13.h, z10.h, z12.h\n",
                    "sub      z14.h, z10.h, z12.h\n",
                    "mul      z15.h, z14.h, z17.h\n",
                    "sqrdmulh z11.h, z14.h, z18.h\n",
                    "mls      z15.h, z11.h, z0.h[0]\n",
                    "str z15, [{ff}, #6, mul vl]\n",
                    "sub      z11.h, z1.h, z12.h\n",
                    "add      z2.h, z11.h, z15.h\n",
                    "sub      z11.h, z1.h, z10.h\n",
                    "sub      z3.h, z11.h, z15.h\n",
                    "add      z1.h, z1.h, z13.h\n",
                    r3s!(4, 5, 6, "z28.h", "z29.h", "z30.h", "z31.h"),
                    "ldr z28, [{tp}, #12, mul vl]\n",
                    "ldr z29, [{tp}, #13, mul vl]\n",
                    "ldr z30, [{tp}, #14, mul vl]\n",
                    "ldr z31, [{tp}, #15, mul vl]\n",
                    "str z7, [{pp}, #6, mul vl]\n",
                    "str z8, [{pp}, #7, mul vl]\n",
                    "str z9, [{pp}, #8, mul vl]\n",
                    "str z28, [{ff}, #3, mul vl]\n",
                    "str z29, [{ff}, #4, mul vl]\n",
                    r3s!(7, 8, 9, "z28.h", "z29.h", "z30.h", "z31.h"),
                    "str z1, [{xpn}]\n",
                    "str z2, [{xpn}, #3, mul vl]\n",
                    "str z3, [{xpn}, #6, mul vl]\n",
                    "str z4, [{xpn}, #9, mul vl]\n",
                    "str z5, [{xpn}, #12, mul vl]\n",
                    "str z6, [{xpn}, #15, mul vl]\n",
                    "str z7, [{xpn}, #18, mul vl]\n",
                    "str z8, [{xpn}, #21, mul vl]\n",
                    "str z9, [{xpn}, #24, mul vl]\n",
                ),
                xpn = in(reg) xpn,
                tp = in(reg) tp,
                cp = in(reg) cp,
                pp = in(reg) post.as_mut_ptr() as *mut u8,
                ff = in(reg) fin.as_mut_ptr() as *mut u8,
                out("v0") _, out("v1") _, out("v2") _, out("v3") _, out("v4") _, out("v5") _,
                out("v6") _, out("v7") _, out("v8") _, out("v9") _, out("v10") _, out("v11") _,
                out("v12") _, out("v13") _, out("v14") _, out("v15") _, out("v16") _, out("v17") _,
                out("v18") _, out("v19") _, out("v20") _, out("v21") _, out("v22") _, out("v23") _,
                out("v24") _, out("v25") _, out("v26") _, out("v27") _, out("v28") _, out("v29") _,
                out("v30") _, out("v31") _,
                options(nostack)
            );
            // NEON reference chain
            let ld = |s: usize| vld1q_s16(pristine[s].as_ptr());
            let t0 = vld1q_s16(kern.c[0].as_ptr());
            let t1 = vld1q_s16(kern.c[0].as_ptr().add(8));
            let k = vld1q_s16(kern.k.as_ptr());
            let wa = radix3::<0, 1, 2, 3, 0>(reduce_round13(ld(0), k), ld(9), ld(18), t0, k);
            let wb = radix3::<0, 1, 2, 3, 0>(reduce_round13(ld(3), k), ld(12), ld(21), t0, k);
            let wc = radix3::<0, 1, 2, 3, 0>(reduce_round13(ld(6), k), ld(15), ld(24), t0, k);
            let arr = |v: int16x8_t| -> [i16; 8] { core::mem::transmute(v) };
            // after the butterflies + node-0 post (v[3] skipped): v6 = reduce(y2 of A), v7 = y2 of B, v8 = y2 of C
            let v6 = reduce_round13(wa.2, k);
            let v7 = wb.2;
            let v8 = wc.2;
            let posts: [int16x8_t; 3] = [v6, v7, v8];
            let _ = posts;
            // level 5: group 2 = radix3(v6, v7, v8; t1)
            let w2 = radix3::<0, 1, 2, 3, 0>(v6, v7, v8, t1, k);
            for (i, want) in [&w2.0, &w2.1, &w2.2].into_iter().enumerate() {
                let wa2 = arr(*want);
                for l in 0..8 {
                    if fin[i][l] != wa2[l] || fin[i][8 + l] != wa2[l] {
                        eprintln!("final z{} lane {l}: got {} / {} want {}", 7 + i, fin[i][l], fin[i][8 + l], wa2[l]);
                    }
                }
            }
            let want_z1 = arr(reduce_round13(ld(0), k));
            eprintln!("z16 (mask?): {:?}", &post[0][..4]);
            eprintln!("z1 loaded:  {:?}", &post[1][..8]);
            eprintln!("pristine[0]:    {:?}", &pristine[0][..8]);
            eprintln!("z1 after r13: got {:?} want {:?}", &post[2][..8], want_z1);
            eprintln!("z7 before g2: {:?} (want {:?})", &post[6][..8], arr(v6));
            eprintln!("z8 before g2: {:?} (want {:?})", &post[7][..8], arr(v7));
            eprintln!("z9 before g2: {:?} (want {:?})", &post[8][..8], arr(v8));
            eprintln!("z28 (t1w0): {:?}", &fin[3][..2]);
            eprintln!("z29 (t1t0): {:?}", &fin[4][..2]);
            eprintln!("v0p got {:?} want {:?}", &post[6][..8], arr(reduce_round13(wa.0, k)));

            let wpost = [arr(v6), arr(v7), arr(v8)];
            for i in 0..3 {
                eprintln!("post z{}: got {:?} want {:?}", 7 + i, &post[3 + i][..8], wpost[i]);
            }
            // the final stored rows 0..26 vs NEON's pass_c on the same 27 slots
            let mut x1: [int16x8_t; crate::ntt::N] = std::mem::zeroed();
            for (s, v) in x1.iter_mut().enumerate().take(27) {
                let mut a8 = [0i16; 8];
                a8.copy_from_slice(&pristine[s][..8]);
                *v = vld1q_s16(a8.as_ptr());
            }
            pass_c::<4>(&mut x1, kern);
            eprintln!("g0 b1: {:?}  qrd: {:?}", &fin[3][..8], &fin[4][..8]);
            eprintln!("g0 c1: {:?}  e:   {:?}", &fin[5][..8], &fin[6][..8]);
            let t0h = vld1q_s16(kern.c[0].as_ptr().add(4));
            // the level-5 group-0 inputs: v0 = y0 of A (post-reduced), v1 = y0 of B, v2 = y0 of C
            let v0p = reduce_round13(wa.0, k);
            let want_b1 = arr(mulc::<4, 5>(wb.0, t0h, k));
            let want_c1 = arr(mulc::<6, 7>(wc.0, t0h, k));
            eprintln!("want b1: {:?}\nwant c1: {:?}", want_b1, want_c1);
            let want_g0 = radix3::<4, 5, 6, 7, 0>(v0p, wb.0, wc.0, t0h, k);
            eprintln!("want g0 y0: {:?}\nwant g0 y1: {:?}\nwant g0 y2: {:?}",
                arr(want_g0.0), arr(want_g0.1), arr(want_g0.2));
            let t1v = vld1q_s16(kern.c[0].as_ptr().add(8));
            let wg2 = radix3::<4, 5, 6, 7, 0>(v6, v7, v8, t1v, k);
            eprintln!("hand g2 y0: {:?} (t1 = {:?})", arr(wg2.0), &kern.c[0][8..12]);
            eprintln!("x1 slot18:  {:?}", arr(x1[18]));
            eprintln!("got slot18: {:?}", &rows[18][..8]);
            for s in [0usize, 3, 6, 9, 12, 15, 18, 21, 24] {
                let want: [i16; 8] = core::mem::transmute(x1[s]);
                for l in 0..8 {
                    assert_eq!(rows[s][l], want[l], "final slot {s} lane {l}");
                    assert_eq!(rows[s][8 + l], want[l], "final slot {s} half1 lane {l}");
                }
            }
        }
    }
}

#[cfg(all(test, target_feature = "sve2"))]
mod lvl4_probe {
    use super::*;

    /// The level-4 sequence (three interleaved butterflies with input reductions) against the
    /// crate's own `radix3`, all nine outputs, at the LUT output range.
    #[test]
    fn lvl4_sequence_matches_high() {
        if !paired_supported() {
            return;
        }
        let kern = &kernels()[1];
        let tw = pool(1);
        unsafe {
            let mut st = 4242u64;
            let mut next = move || {
                st ^= st << 13;
                st ^= st >> 7;
                st ^= st << 17;
                st
            };
            let mut rows = [[0i16; 16]; 9];
            let mut v8 = [[0i16; 8]; 9];
            for r in 0..9 {
                for l in 0..8 {
                    let x = (((next() as i32 % 29100) as i32 % 29100) - 14550) as i16;
                    rows[r][l] = x;
                    rows[r][l + 8] = x;
                    v8[r][l] = x;
                }
            }
            let mut out = [[0i16; 16]; 9];
            core::arch::asm!(
                concat!(
                    "ldr z0, [{cp}]\n",
                    "ldr z16, [{cp}, #1, mul vl]\n",
                    "ldr z17, [{cp}, #2, mul vl]\n",
                    "ldr z18, [{cp}, #3, mul vl]\n",
                    "ldr z19, [{cp}, #4, mul vl]\n",
                    "ldr z20, [{tp}]\n",
                    "ldr z21, [{tp}, #1, mul vl]\n",
                    "ldr z22, [{tp}, #2, mul vl]\n",
                    "ldr z23, [{tp}, #3, mul vl]\n",
                    "ldr z1, [{pa}]\n",
                    "ldr z4, [{pa}, #3, mul vl]\n",
                    "ldr z7, [{pa}, #6, mul vl]\n",
                    r13!(1),
                    r3s!(1, 4, 7, "z20.h", "z21.h", "z22.h", "z23.h"),
                    "ldr z2, [{pa}, #1, mul vl]\n",
                    "ldr z5, [{pa}, #4, mul vl]\n",
                    "ldr z8, [{pa}, #7, mul vl]\n",
                    r13!(2),
                    r3s!(2, 5, 8, "z20.h", "z21.h", "z22.h", "z23.h"),
                    "ldr z3, [{pa}, #2, mul vl]\n",
                    "ldr z6, [{pa}, #5, mul vl]\n",
                    "ldr z9, [{pa}, #8, mul vl]\n",
                    r13!(3),
                    r3s!(3, 6, 9, "z20.h", "z21.h", "z22.h", "z23.h"),
                    "str z1, [{po}]\n",
                    "str z2, [{po}, #1, mul vl]\n",
                    "str z3, [{po}, #2, mul vl]\n",
                    "str z4, [{po}, #3, mul vl]\n",
                    "str z5, [{po}, #4, mul vl]\n",
                    "str z6, [{po}, #5, mul vl]\n",
                    "str z7, [{po}, #6, mul vl]\n",
                    "str z8, [{po}, #7, mul vl]\n",
                    "str z9, [{po}, #8, mul vl]\n",
                ),
                pa = in(reg) rows.as_ptr(),
                po = in(reg) out.as_mut_ptr() as *mut u8,
                tp = in(reg) tw.c[0].as_ptr() as *const u8,
                cp = in(reg) tw.consts.as_ptr() as *const u8,
                out("v0") _, out("v1") _, out("v2") _, out("v3") _, out("v4") _, out("v5") _,
                out("v6") _, out("v7") _, out("v8") _, out("v9") _, out("v10") _, out("v11") _,
                out("v12") _, out("v13") _, out("v14") _, out("v15") _, out("v16") _, out("v17") _,
                out("v18") _, out("v19") _, out("v20") _, out("v21") _, out("v22") _, out("v23") _,
                out("v24") _, out("v25") _, out("v26") _, out("v27") _, out("v28") _, out("v29") _,
                out("v30") _, out("v31") _,
                options(nostack)
            );
            let t = vld1q_s16(kern.c[0].as_ptr());
            let k = vld1q_s16(kern.k.as_ptr());
            let ld = |i: usize| vld1q_s16(v8[i].as_ptr());
            let butterflies: [(usize, usize, usize); 3] = [(0, 3, 6), (1, 4, 7), (2, 5, 8)];
            for (bi, &(ia, ib, ic)) in butterflies.iter().enumerate() {
                let a = reduce_round13(ld(ia), k);
                let w = radix3::<0, 1, 2, 3, 0>(a, ld(ib), ld(ic), t, k);
                for (yi, wv) in [&w.0, &w.1, &w.2].into_iter().enumerate() {
                    let reg = [ia, ib, ic][yi];
                    let wa: [i16; 8] = core::mem::transmute(*wv);
                    for l in 0..8 {
                        assert_eq!(out[reg][l], wa[l], "butterfly {bi} y{yi} half0 lane {l}");
                        assert_eq!(out[reg][8 + l], wa[l], "butterfly {bi} y{yi} half1 lane {l}");
                    }
                }
            }
        }
    }
}

#[cfg(all(test, target_feature = "sve2"))]
mod lvl5_probe {
    use super::*;

    /// Level-5 group 0 alone: r3s on (v0', v1', v2') with the tp#4..7 twiddles, against the
    /// crate's `radix3::<4, 5, 6, 7, 0>` on the full `t0` vector, at the LUT range.
    #[test]
    fn lvl5_group0_matches() {
        if !paired_supported() {
            return;
        }
        let kern = &kernels()[1];
        let tw = pool(1);
        unsafe {
            let mut st = 777u64;
            let mut next = move || {
                st ^= st << 13;
                st ^= st >> 7;
                st ^= st << 17;
                st
            };
            // inputs in the level-4 output range (<= 28676)
            let mut rows = [[0i16; 16]; 9];
            let mut v8 = [[0i16; 8]; 9];
            for r in 0..9 {
                for l in 0..8 {
                    let x = (((next() as i32 % 28600) - 14300) as i16).max(-14300);
                    rows[r][l] = x;
                    rows[r][l + 8] = x;
                    v8[r][l] = x;
                }
            }
            let mut out = [[0i16; 16]; 9];
            core::arch::asm!(
                concat!(
                    "ldr z0, [{cp}]\n",
                    "ldr z17, [{cp}, #2, mul vl]\n",
                    "ldr z18, [{cp}, #3, mul vl]\n",
                    "ldr z24, [{tp}, #4, mul vl]\n",
                    "ldr z25, [{tp}, #5, mul vl]\n",
                    "ldr z26, [{tp}, #6, mul vl]\n",
                    "ldr z27, [{tp}, #7, mul vl]\n",
                    "ldr z28, [{tp}, #8, mul vl]\n",
                    "ldr z29, [{tp}, #9, mul vl]\n",
                    "ldr z30, [{tp}, #10, mul vl]\n",
                    "ldr z31, [{tp}, #11, mul vl]\n",
                    "ldr z1, [{pa}]\n",
                    "ldr z2, [{pa}, #1, mul vl]\n",
                    "ldr z3, [{pa}, #2, mul vl]\n",
                    "ldr z4, [{pa}, #3, mul vl]\n",
                    "ldr z5, [{pa}, #4, mul vl]\n",
                    "ldr z6, [{pa}, #5, mul vl]\n",
                    "ldr z7, [{pa}, #6, mul vl]\n",
                    "ldr z8, [{pa}, #7, mul vl]\n",
                    "ldr z9, [{pa}, #8, mul vl]\n",
                    r3s!(1, 2, 3, "z24.h", "z25.h", "z26.h", "z27.h"),
                    r3s!(4, 5, 6, "z28.h", "z29.h", "z30.h", "z31.h"),
                    "ldr z28, [{tp}, #12, mul vl]\n",
                    "ldr z29, [{tp}, #13, mul vl]\n",
                    "ldr z30, [{tp}, #14, mul vl]\n",
                    "ldr z31, [{tp}, #15, mul vl]\n",
                    r3s!(7, 8, 9, "z28.h", "z29.h", "z30.h", "z31.h"),
                    "str z1, [{po}]\n",
                    "str z2, [{po}, #1, mul vl]\n",
                    "str z3, [{po}, #2, mul vl]\n",
                    "str z4, [{po}, #3, mul vl]\n",
                    "str z5, [{po}, #4, mul vl]\n",
                    "str z6, [{po}, #5, mul vl]\n",
                    "str z7, [{po}, #6, mul vl]\n",
                    "str z8, [{po}, #7, mul vl]\n",
                    "str z9, [{po}, #8, mul vl]\n",
                ),
                pa = in(reg) rows.as_ptr(),
                po = in(reg) out.as_mut_ptr() as *mut u8,
                tp = in(reg) tw.c[0].as_ptr() as *const u8,
                cp = in(reg) tw.consts.as_ptr() as *const u8,
                out("v0") _, out("v1") _, out("v2") _, out("v3") _, out("v4") _, out("v5") _,
                out("v6") _, out("v7") _, out("v8") _, out("v9") _, out("v10") _, out("v11") _,
                out("v12") _, out("v13") _, out("v14") _, out("v15") _, out("v16") _, out("v17") _,
                out("v18") _, out("v19") _, out("v20") _, out("v21") _, out("v22") _, out("v23") _,
                out("v24") _, out("v25") _, out("v26") _, out("v27") _, out("v28") _, out("v29") _,
                out("v30") _, out("v31") _,
                options(nostack)
            );
            let t0 = vld1q_s16(kern.c[0].as_ptr());
            let t1 = vld1q_s16(kern.c[0].as_ptr().add(8));
            let k = vld1q_s16(kern.k.as_ptr());
            let ld = |i: usize| vld1q_s16(v8[i].as_ptr());
            for g in 0..3 {
                let (a, b, c) = (ld(3 * g), ld(3 * g + 1), ld(3 * g + 2));
                let w = if g == 0 {
                    radix3::<4, 5, 6, 7, 0>(a, b, c, t0, k)
                } else if g == 1 {
                    radix3::<0, 1, 2, 3, 0>(a, b, c, t1, k)
                } else {
                    radix3::<4, 5, 6, 7, 0>(a, b, c, t1, k)
                };
                for (yi, wv) in [&w.0, &w.1, &w.2].into_iter().enumerate() {
                    let wa: [i16; 8] = core::mem::transmute(*wv);
                    let reg = 3 * g + yi;
                    for l in 0..8 {
                        assert_eq!(out[reg][l], wa[l], "g{g} y{yi} half0 lane {l}");
                        assert_eq!(out[reg][8 + l], wa[l], "g{g} y{yi} half1 lane {l}");
                    }
                }
            }
        }
    }
}

#[cfg(all(test, target_feature = "sve2"))]
mod passd_probe {
    use super::*;

    /// pass_d's group 0 alone, every input and output dumped, against the NEON kernels'
    /// primitives on the same data.
    #[test]
    fn passd_group0_matches() {
        if !paired_supported() {
            return;
        }
        let kern = &kernels()[1];
        let tw = pool(1);
        unsafe {
            let mut st = 31337u64;
            let mut next = move || {
                st ^= st << 13;
                st ^= st >> 7;
                st ^= st << 17;
                st
            };
            // six rows in the level-5 output range
            let mut rows = [[0i16; 16]; 6];
            let mut v8 = [[0i16; 8]; 6];
            for r in 0..6 {
                for l in 0..8 {
                    let x = (((next() as i32 % 28400) - 14200) as i16).max(-14200);
                    rows[r][l] = x;
                    rows[r][l + 8] = x;
                    v8[r][l] = x;
                }
            }
            let mut out = [[0i16; 16]; 6];
            let mut dbg1 = [[0i16; 16]; 4];
            core::arch::asm!(
                concat!(
                    "ldr z0, [{cp}]\n",
                    "ldr z16, [{cp}, #1, mul vl]\n",
                    "ldr z17, [{cp}, #2, mul vl]\n",
                    "ldr z18, [{cp}, #3, mul vl]\n",
                    "ldr z19, [{cp}, #4, mul vl]\n",
                    "ldr z20, [{tp}]\n",
                    "ldr z21, [{tp}, #1, mul vl]\n",
                    "ldr z22, [{tp}, #2, mul vl]\n",
                    "ldr z23, [{tp}, #3, mul vl]\n",
                    "ldr z24, [{tp}, #4, mul vl]\n",
                    "ldr z25, [{tp}, #5, mul vl]\n",
                    "ldr z26, [{tp}, #6, mul vl]\n",
                    "ldr z27, [{tp}, #7, mul vl]\n",
                    "ldr z1, [{xq}]\n",
                    "ldr z4, [{xq}, #3, mul vl]\n",
                    r13!(1),
                    r13!(4),
                    "str z1, [{db}]\n",
                    "str z20, [{db}, #1, mul vl]\n",
                    "str z21, [{db}, #2, mul vl]\n",
                    "ldr z2, [{xq}, #1, mul vl]\n",
                    "ldr z3, [{xq}, #2, mul vl]\n",
                    "ldr z5, [{xq}, #4, mul vl]\n",
                    "str z2, [{db}, #1, mul vl]\n",
                    "mul      z10.h, z2.h, z20.h\n",
                    "sqrdmulh z11.h, z2.h, z21.h\n",
                    "mls      z10.h, z11.h, z0.h[0]\n",
                    "str z10, [{db}, #2, mul vl]\n",
                    "mul      z12.h, z3.h, z22.h\n",
                    "sqrdmulh z11.h, z3.h, z23.h\n",
                    "mls      z12.h, z11.h, z0.h[0]\n",
                    "str z12, [{db}, #3, mul vl]\n",
                    "add      z13.h, z10.h, z12.h\n",
                    "sub      z14.h, z10.h, z12.h\n",
                    "mul      z15.h, z14.h, z17.h\n",
                    "sqrdmulh z11.h, z14.h, z18.h\n",
                    "mls      z15.h, z11.h, z0.h[0]\n",
                    "sub      z11.h, z1.h, z12.h\n",
                    "add      z2.h, z11.h, z15.h\n",
                    "sub      z11.h, z1.h, z10.h\n",
                    "sub      z3.h, z11.h, z15.h\n",
                    "add      z1.h, z1.h, z13.h\n",
                    "ldr z6, [{xq}, #5, mul vl]\n",
                    r3s!(4, 5, 6, "z24.h", "z25.h", "z26.h", "z27.h"),
                    "str z1, [{o}]\n",
                    "str z2, [{o}, #1, mul vl]\n",
                    "str z3, [{o}, #2, mul vl]\n",
                    "str z4, [{o}, #3, mul vl]\n",
                    "str z5, [{o}, #4, mul vl]\n",
                    "str z6, [{o}, #5, mul vl]\n",
                ),
                xq = in(reg) rows.as_ptr(),
                o = in(reg) out.as_mut_ptr() as *mut u8,
                db = in(reg) dbg1.as_mut_ptr() as *mut u8,
                tp = in(reg) tw.d[0].as_ptr() as *const u8,
                cp = in(reg) tw.consts.as_ptr() as *const u8,
                out("v0") _, out("v1") _, out("v2") _, out("v3") _, out("v4") _, out("v5") _,
                out("v6") _, out("v7") _, out("v8") _, out("v9") _, out("v10") _, out("v11") _,
                out("v12") _, out("v13") _, out("v14") _, out("v15") _, out("v16") _, out("v17") _,
                out("v18") _, out("v19") _, out("v20") _, out("v21") _, out("v22") _, out("v23") _,
                out("v24") _, out("v25") _, out("v26") _, out("v27") _, out("v28") _, out("v29") _,
                out("v30") _, out("v31") _,
                options(nostack)
            );
            let k = vld1q_s16(kern.k.as_ptr());
            let dv = vld1q_s16(kern.d[0].as_ptr());
            let ld = |i: usize| vld1q_s16(v8[i].as_ptr());
            let a = reduce_round13(ld(0), k);
            let w = radix3::<0, 1, 2, 3, 0>(a, ld(1), ld(2), dv, k);
            let arr = |v: int16x8_t| -> [i16; 8] { core::mem::transmute(v) };
            eprintln!("z1 reduced: got {:?} want {:?}", &dbg1[0][..8], arr(a));
            eprintln!("z20: {:?} (want {} {})", &dbg1[1][..2], kern.d[0][0], kern.d[0][0]);
            eprintln!("z21: {:?} (want {} {})", &dbg1[2][..2], kern.d[0][1], kern.d[0][1]);
            eprintln!("y0: got {:?} want {:?}", &out[0][..8], arr(w.0));
            assert_eq!(&out[0][..8], &arr(w.0)[..], "y0");
            assert_eq!(&out[1][..8], &arr(w.1)[..], "y1");
            assert_eq!(&out[2][..8], &arr(w.2)[..], "y2");
        }
    }
}

#[cfg(all(test, target_feature = "sve2"))]
mod sdot_probe {
    use super::*;

    /// The indexed halfword `sdot`'s exact pairing: known values in, acc out, both halves.
    #[test]
    fn sdot_semantics() {
        if !paired_supported() {
            return;
        }
        unsafe {
            // low half zeros; high half carries the values: isolates the high half
            let mut x = [0i16; 16];
            for l in 0..8 {
                x[l + 8] = 200 + l as i16;
            }
            let mut k = [0i16; 16];
            for j in 0..8 {
                k[j + 8] = (j + 9) as i16;
            }
            let mut out = [0i64; 16];
            core::arch::asm!(
                concat!(
                    "ldr z4, [{x}]\n",
                    "ldr z0, [{k}]\n",
                    "eor z16.d, z16.d, z16.d\n",
                    "eor z17.d, z17.d, z17.d\n",
                    "sdot z16.d, z4.h, z0.h[0]\n",
                    "sdot z17.d, z4.h, z0.h[1]\n",
                    "str z16, [{o}]\n",
                    "str z17, [{o}, #1, mul vl]\n",
                ),
                x = in(reg) x.as_ptr(),
                k = in(reg) k.as_ptr(),
                o = in(reg) out.as_mut_ptr(),
                out("v0") _, out("v4") _, out("v16") _, out("v17") _,
                options(nostack)
            );
            // z16 low = out[0..4], z17 low = out[4..8], z16 high = out[8..12], z17 high = out[12..16]
            eprintln!("sdot[0] low lanes:  {:?}", &out[0..4]);
            eprintln!("sdot[1] low lanes:  {:?}", &out[4..8]);
            eprintln!("sdot[0] high lanes: {:?}", &out[8..12]);
            eprintln!("sdot[1] high lanes: {:?}", &out[12..16]);
        }
    }
}

