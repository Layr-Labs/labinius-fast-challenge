// Expanded Montgomery twiddles for the single-transform commitment.
// Memory-source multiplies replace separate broadcast loads. The integer twiddles,
// butterfly schedule, reductions, and output representatives are unchanged.
macro_rules! mont_mem {
    ($tw:literal, $off:literal, $a1:literal, $a2:literal, $t:literal) => {
        concat!(
            "vpmullw zmm", $t, ", zmm", $a1, ", [{", $tw, "}+0", $off, "*16+64]\n",
            "vpmulhw zmm", $a1, ", zmm", $a1, ", [{", $tw, "}+0", $off, "*16+0]\n",
            "vpmulhw zmm", $t, ", zmm", $t, ", zmm27\n",
            "vpsubw zmm", $a1, ", zmm", $a1, ", zmm", $t, "\n",
            "vpmullw zmm", $t, ", zmm", $a2, ", [{", $tw, "}+0", $off, "*16+192]\n",
            "vpmulhw zmm", $a2, ", zmm", $a2, ", [{", $tw, "}+0", $off, "*16+128]\n",
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

macro_rules! level4_mem {
    (plain $a0:literal, $a1:literal, $a2:literal,
     $o0:literal, $o1:literal, $o2:literal, $t:literal) => {
        concat!(
            "vmovdqa64 zmm", $a0, ", [{i}+", $o0, "]\n",
            "vmovdqa64 zmm", $a1, ", [{i}+", $o1, "]\n",
            "vmovdqa64 zmm", $a2, ", [{i}+", $o2, "]\n",
            mont_mem!("t4", "", $a1, $a2, $t),
        )
    };
    (lut $a0:literal, $a1:literal, $a2:literal,
     $o0:literal, $o1:literal, $o2:literal, $t:literal) => {
        concat!(
            level4_mem!(plain $a0, $a1, $a2, $o0, $o1, $o2, $t),
            // logical a >> 11 is the window (a >> 11) & 31 (bits 11..15 of the lane), and
            // vpermw only reads the low 5 bits of each word index anyway.
            "vpsrlw zmm30, zmm", $a0, ", 11\n",
            "vpermw zmm30, zmm30, zmmword ptr [{c}+512]\n",
            "vpaddw zmm", $a0, ", zmm", $a0, ", zmm30\n",
        )
    };
}

macro_rules! level6_mem {
    (plain $off:literal, $a0:literal, $a1:literal, $a2:literal, $t:literal) => {
        mont_mem!("t6", $off, $a1, $a2, $t)
    };
    (barrett $off:literal, $a0:literal, $a1:literal, $a2:literal, $t:literal) => {
        concat!(
            "vpmullw zmm", $t, ", zmm", $a1, ", [{t6}+0", $off, "*16+64]\n",
            "vpmulhrsw zmm30, zmm", $a0, ", [{c}+192]\n",
            "vpmullw zmm30, zmm30, zmm27\n",
            "vpsubw zmm", $a0, ", zmm", $a0, ", zmm30\n",
            "vpmulhw zmm", $a1, ", zmm", $a1, ", [{t6}+0", $off, "*16+0]\n",
            "vpmulhw zmm", $t, ", zmm", $t, ", zmm27\n",
            "vpsubw zmm", $a1, ", zmm", $a1, ", zmm", $t, "\n",
            "vpmullw zmm", $t, ", zmm", $a2, ", [{t6}+0", $off, "*16+192]\n",
            "vpmulhw zmm", $a2, ", zmm", $a2, ", [{t6}+0", $off, "*16+128]\n",
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

macro_rules! tail27_mem {
    ($m4:ident, $m6:ident) => {
        tail27_mem!($m4, $m6, bfly)
    };
    ($m4:ident, $m6:ident, $finish:ident) => {
        concat!(
            level4_mem!($m4 0, 9, 18, 0, 576, 1152, 28),
            level4_mem!($m4 3, 12, 21, 192, 768, 1344, 29),
            bfly!(0, 9, 18, 28),
            level4_mem!($m4 6, 15, 24, 384, 960, 1536, 28),
            bfly!(3, 12, 21, 29),
            level4_mem!($m4 1, 10, 19, 64, 640, 1216, 29),
            bfly!(6, 15, 24, 28),
            level4_mem!($m4 4, 13, 22, 256, 832, 1408, 28),
            bfly!(1, 10, 19, 29),
            level4_mem!($m4 7, 16, 25, 448, 1024, 1600, 29),
            bfly!(4, 13, 22, 28),
            level4_mem!($m4 2, 11, 20, 128, 704, 1280, 28),
            bfly!(7, 16, 25, 29),
            level4_mem!($m4 5, 14, 23, 320, 896, 1472, 29),
            bfly!(2, 11, 20, 28),
            level4_mem!($m4 8, 17, 26, 512, 1088, 1664, 28),
            bfly!(5, 14, 23, 29),
            bfly!(8, 17, 26, 28),
            mont_mem!("t5", "", 3, 6, 28),
            mont_mem!("t5", "+16", 21, 24, 29),
            bfly!(0, 3, 6, 28),
            mont_mem!("t5", "+32", 12, 15, 28),
            bfly!(18, 21, 24, 29),
            mont_mem!("t5", "", 4, 7, 29),
            bfly!(9, 12, 15, 28),
            mont_mem!("t5", "+16", 22, 25, 28),
            bfly!(1, 4, 7, 29),
            mont_mem!("t5", "+32", 13, 16, 29),
            bfly!(19, 22, 25, 28),
            mont_mem!("t5", "", 5, 8, 28),
            bfly!(10, 13, 16, 29),
            mont_mem!("t5", "+16", 23, 26, 29),
            bfly!(2, 5, 8, 28),
            mont_mem!("t5", "+32", 14, 17, 28),
            bfly!(20, 23, 26, 29),
            bfly!(11, 14, 17, 28),
            level6_mem!($m6 "", 0, 1, 2, 28),
            level6_mem!($m6 "+16", 6, 7, 8, 29),
            $finish!(0, 1, 2, 28, 0, 64, 128),
            level6_mem!($m6 "+32", 3, 4, 5, 28),
            $finish!(6, 7, 8, 29, 192, 256, 320),
            level6_mem!($m6 "+48", 18, 19, 20, 29),
            $finish!(3, 4, 5, 28, 384, 448, 512),
            level6_mem!($m6 "+64", 24, 25, 26, 28),
            $finish!(18, 19, 20, 29, 576, 640, 704),
            level6_mem!($m6 "+80", 21, 22, 23, 29),
            $finish!(24, 25, 26, 28, 768, 832, 896),
            level6_mem!($m6 "+96", 9, 10, 11, 28),
            $finish!(21, 22, 23, 29, 960, 1024, 1088),
            level6_mem!($m6 "+112", 15, 16, 17, 29),
            $finish!(9, 10, 11, 28, 1152, 1216, 1280),
            level6_mem!($m6 "+128", 12, 13, 14, 28),
            $finish!(15, 16, 17, 29, 1344, 1408, 1472),
            $finish!(12, 13, 14, 28, 1536, 1600, 1664),
        )
    };
}

#[repr(C,align(64))] struct ExpandedTwiddles{t4:[[i16;32];96],t5:[[i16;32];288],t6:[[i16;32];864]}
const fn expanded<const Q:u16>()->ExpandedTwiddles{
 let old=build_tables::<Q>();let mut t=ExpandedTwiddles{t4:[[0;32];96],t5:[[0;32];288],t6:[[0;32];864]};
 let mut i=0;while i<96{t.t4[i]=[old.tw4[i/4][i%4]as i16;32];i+=1;}
 i=0;while i<288{t.t5[i]=[old.tw5[i/4][i%4]as i16;32];i+=1;}
 i=0;while i<864{t.t6[i]=[old.tw6[i/4][i%4]as i16;32];i+=1;}t
}
static SMS_TWIDDLES:ExpandedTwiddles=expanded::<9721>();
#[inline(never)]
#[target_feature(enable="avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vnni,gfni")]
unsafe fn tail27_mem_b(
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
            tail27_mem!(lut, barrett),
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
