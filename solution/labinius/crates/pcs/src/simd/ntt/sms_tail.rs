// Private SMS producer tails. The ordinary batched binary NTT is unchanged.
// Fixed512-bit inputs use three public prefix row-band scales per quarter.
// Reciprocal public-key factors preserve every product. Four leaves per quarter
// use the583-op tail; two use511/521-op unit-twiddle tails, with their input
// reductions performed in the prefix. Every signed-word intermediate is bounded,
// and the same31296 output bound is used by the public eight-pair certificate.
// The unscaled full648-bit factory retains the604-op tail (bound30376).
pub(crate) const SMS_SOURCE_BOUND:i64=31296;

macro_rules! tail27_relocated {
    ($m4:ident, $m6:ident) => {
        tail27_relocated!($m4, $m6, bfly)
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
            "vpsrlw zmm28, zmm0, 11\n",
            "vpermw zmm28, zmm28, zmmword ptr [{c}+512]\n",
            "vpaddw zmm0, zmm0, zmm28\n",
            mont_mem!("t5", "", 3, 6, 28),
            "vpsrlw zmm29, zmm18, 11\n",
            "vpermw zmm29, zmm29, zmmword ptr [{c}+512]\n",
            "vpaddw zmm18, zmm18, zmm29\n",
            mont_mem!("t5", "+16", 21, 24, 29),
            bfly!(0, 3, 6, 28),
            "vpsrlw zmm28, zmm9, 11\n",
            "vpermw zmm28, zmm28, zmmword ptr [{c}+512]\n",
            "vpaddw zmm9, zmm9, zmm28\n",
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

#[inline(never)]
#[target_feature(enable="avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vnni,gfni")]
unsafe fn tail27_relocated_leaf(
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
            tail27_relocated!(lut, plain),
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

pub(crate) const SMS_SCALE_ALPHA:[i32;4]=[9626,4010,1282,5655];
pub(crate) const SMS_SCALE_INVERSE:[i32;4]=[4707,6955,8758,9513];
macro_rules! tail27_scaled {
    ($m4:ident, $m6:ident) => {
        tail27_scaled!($m4, $m6, bfly)
    };
    ($m4:ident, $m6:ident, $finish:ident $(, $rot:literal)?) => {
        concat!(
            level4_mem!(plain 0, 9, 18, 0, 576, 1152, 28 $(, $rot)?),
            level4_mem!(plain 3, 12, 21, 192, 768, 1344, 29 $(, $rot)?),
            bfly!(0, 9, 18, 28),
            level4_mem!(plain 6, 15, 24, 384, 960, 1536, 28 $(, $rot)?),
            bfly!(3, 12, 21, 29),
            level4_mem!(lut 1, 10, 19, 64, 640, 1216, 29 $(, $rot)?),
            bfly!(6, 15, 24, 28),
            level4_mem!(plain 4, 13, 22, 256, 832, 1408, 28 $(, $rot)?),
            bfly!(1, 10, 19, 29),
            level4_mem!(plain 7, 16, 25, 448, 1024, 1600, 29 $(, $rot)?),
            bfly!(4, 13, 22, 28),
            level4_mem!(lut 2, 11, 20, 128, 704, 1280, 28 $(, $rot)?),
            bfly!(7, 16, 25, 29),
            level4_mem!(plain 5, 14, 23, 320, 896, 1472, 29 $(, $rot)?),
            bfly!(2, 11, 20, 28),
            level4_mem!(plain 8, 17, 26, 512, 1088, 1664, 28 $(, $rot)?),
            bfly!(5, 14, 23, 29),
            bfly!(8, 17, 26, 28),
            "vpsrlw zmm28, zmm0, 11\n",
            "vpermw zmm28, zmm28, zmmword ptr [{c}+512]\n",
            "vpaddw zmm0, zmm0, zmm28\n",
            mont_mem!("t5", "", 3, 6, 28 $(, $rot)?),
            "vpsrlw zmm29, zmm18, 11\n",
            "vpermw zmm29, zmm29, zmmword ptr [{c}+512]\n",
            "vpaddw zmm18, zmm18, zmm29\n",
            mont_mem!("t5", "+16", 21, 24, 29 $(, $rot)?),
            bfly!(0, 3, 6, 28),
            "vpsrlw zmm28, zmm9, 11\n",
            "vpermw zmm28, zmm28, zmmword ptr [{c}+512]\n",
            "vpaddw zmm9, zmm9, zmm28\n",
            mont_mem!("t5", "+32", 12, 15, 28 $(, $rot)?),
            bfly!(18, 21, 24, 29),
            mont_mem!("t5", "", 4, 7, 29 $(, $rot)?),
            bfly!(9, 12, 15, 28),
            mont_mem!("t5", "+16", 22, 25, 28 $(, $rot)?),
            bfly!(1, 4, 7, 29),
            mont_mem!("t5", "+32", 13, 16, 29 $(, $rot)?),
            bfly!(19, 22, 25, 28),
            mont_mem!("t5", "", 5, 8, 28 $(, $rot)?),
            bfly!(10, 13, 16, 29),
            mont_mem!("t5", "+16", 23, 26, 29 $(, $rot)?),
            bfly!(2, 5, 8, 28),
            mont_mem!("t5", "+32", 14, 17, 28 $(, $rot)?),
            bfly!(20, 23, 26, 29),
            bfly!(11, 14, 17, 28),
            level6_mem!($m6 "", 0, 1, 2, 28 $(, $rot)?),
            level6_mem!($m6 "+16", 6, 7, 8, 29 $(, $rot)?),
            $finish!(0, 1, 2, 28, 0, 64, 128),
            level6_mem!($m6 "+32", 3, 4, 5, 28 $(, $rot)?),
            $finish!(6, 7, 8, 29, 192, 256, 320),
            level6_mem!($m6 "+48", 18, 19, 20, 29 $(, $rot)?),
            $finish!(3, 4, 5, 28, 384, 448, 512),
            level6_mem!($m6 "+64", 24, 25, 26, 28 $(, $rot)?),
            $finish!(18, 19, 20, 29, 576, 640, 704),
            level6_mem!($m6 "+80", 21, 22, 23, 29 $(, $rot)?),
            $finish!(24, 25, 26, 28, 768, 832, 896),
            level6_mem!($m6 "+96", 9, 10, 11, 28 $(, $rot)?),
            $finish!(21, 22, 23, 29, 960, 1024, 1088),
            level6_mem!($m6 "+112", 15, 16, 17, 29 $(, $rot)?),
            $finish!(9, 10, 11, 28, 1152, 1216, 1280),
            level6_mem!($m6 "+128", 12, 13, 14, 28 $(, $rot)?),
            $finish!(15, 16, 17, 29, 1344, 1408, 1472),
            $finish!(12, 13, 14, 28, 1536, 1600, 1664),
        )
    };
}

#[inline(never)]
#[target_feature(enable="avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vnni,gfni")]
unsafe fn tail27_scaled_leaf(
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
            tail27_scaled!(lut, plain),
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

// Public diagonal preconditioning of the first remaining radix3 layer.
#[repr(C,align(64))]struct SmsBandT4([[i16;32];96]);
static SMS_BAND_T4:SmsBandT4=SmsBandT4([[-2511i16;32],[-7i16;32],[-2511i16;32],[-7i16;32],[-437i16;32],[-30877i16;32],[1066i16;32],[10234i16;32],[1066i16;32],[10234i16;32],[-4502i16;32],[-3014i16;32],[919i16;32],[-26097i16;32],[-437i16;32],[-30877i16;32],[-173i16;32],[12971i16;32],[2338i16;32],[12978i16;32],[4782i16;32],[31646i16;32],[-919i16;32],[26097i16;32],[-2511i16;32],[-7i16;32],[-2511i16;32],[-7i16;32],[-437i16;32],[-30877i16;32],[1066i16;32],[10234i16;32],[1066i16;32],[10234i16;32],[-4502i16;32],[-3014i16;32],[919i16;32],[-26097i16;32],[-437i16;32],[-30877i16;32],[-173i16;32],[12971i16;32],[2338i16;32],[12978i16;32],[4782i16;32],[31646i16;32],[-919i16;32],[26097i16;32],[-2511i16;32],[-7i16;32],[-2511i16;32],[-7i16;32],[-437i16;32],[-30877i16;32],[1066i16;32],[10234i16;32],[1066i16;32],[10234i16;32],[-4502i16;32],[-3014i16;32],[919i16;32],[-26097i16;32],[-437i16;32],[-30877i16;32],[-173i16;32],[12971i16;32],[2338i16;32],[12978i16;32],[4782i16;32],[31646i16;32],[-919i16;32],[26097i16;32],[-2511i16;32],[-7i16;32],[-2511i16;32],[-7i16;32],[-437i16;32],[-30877i16;32],[1066i16;32],[10234i16;32],[1066i16;32],[10234i16;32],[-4502i16;32],[-3014i16;32],[919i16;32],[-26097i16;32],[-437i16;32],[-30877i16;32],[-173i16;32],[12971i16;32],[2338i16;32],[12978i16;32],[4782i16;32],[31646i16;32],[-919i16;32],[26097i16;32]]);
#[repr(C,align(64))]struct SmsBandLut([[[u8;64];48];3]);
const fn sms_band_lut()->SmsBandLut{let original=build_tables::<9721>();let scales:[[i32;3];4]=[[9626, 7203, 7344], [4010, 1866, 175], [1282, 7826, 1186], [5655, 6885, 3406]];let mut output=[[[0u8;64];48];3];let mut b=0;while b<3{let mut table=0;while table<48{let mut n=0;while n<16{let old=(original.lut[table][n]as u16|((original.lut[table][16+n]as u16)<<8))as i16 as i32;let mut v=old*scales[table/12][b]%9721;if v<0{v+=9721;}if v>4860{v-=9721;}output[b][table][n]=v as u16 as u8;output[b][table][16+n]=((v as u16)>>8)as u8;n+=1;}table+=1;}b+=1;}SmsBandLut(output)}
static SMS_BAND_LUT:SmsBandLut=sms_band_lut();

macro_rules! tail27_sms_band0 {
    () => {
        concat!("vmovdqa64 zmm27, [{c}]\n", tail27_sms_band0!(body "[{c}+128]"))
    };
    // The leaf body; `$rot` is the source of the rotation's Montgomery low factor.
    (body $rot:literal) => {
        concat!(
"vmovdqa64 zmm0, [{i}+0]\n",
"vmovdqa64 zmm9, [{i}+576]\n",
"vmovdqa64 zmm18, [{i}+1152]\n",


"vpsubw zmm28, zmm9, zmm18\n",
"vpmullw zmm30, zmm28, ", $rot, "\n",
"vpmulhw zmm28, zmm28, [{c}+64]\n",
"vpmulhw zmm30, zmm30, zmm27\n",
"vpsubw zmm28, zmm28, zmm30\n",
bfly!(0,9,18,28),
"vmovdqa64 zmm1, [{i}+64]\n",
"vmovdqa64 zmm10, [{i}+640]\n",
"vmovdqa64 zmm19, [{i}+1216]\n",
"vpsrlw zmm30, zmm1, 11\n",
"vpermw zmm30, zmm30, zmmword ptr [{c}+512]\n",
"vpaddw zmm1, zmm1, zmm30\n",


"vpsubw zmm28, zmm10, zmm19\n",
"vpmullw zmm30, zmm28, ", $rot, "\n",
"vpmulhw zmm28, zmm28, [{c}+64]\n",
"vpmulhw zmm30, zmm30, zmm27\n",
"vpsubw zmm28, zmm28, zmm30\n",
bfly!(1,10,19,28),
"vmovdqa64 zmm2, [{i}+128]\n",
"vmovdqa64 zmm11, [{i}+704]\n",
"vmovdqa64 zmm20, [{i}+1280]\n",
"vpsrlw zmm30, zmm2, 11\n",
"vpermw zmm30, zmm30, zmmword ptr [{c}+512]\n",
"vpaddw zmm2, zmm2, zmm30\n",


"vpsubw zmm28, zmm11, zmm20\n",
"vpmullw zmm30, zmm28, ", $rot, "\n",
"vpmulhw zmm28, zmm28, [{c}+64]\n",
"vpmulhw zmm30, zmm30, zmm27\n",
"vpsubw zmm28, zmm28, zmm30\n",
bfly!(2,11,20,28),
"vmovdqa64 zmm3, [{i}+192]\n",
"vmovdqa64 zmm12, [{i}+768]\n",
"vmovdqa64 zmm21, [{i}+1344]\n",


"vpsubw zmm28, zmm12, zmm21\n",
"vpmullw zmm30, zmm28, ", $rot, "\n",
"vpmulhw zmm28, zmm28, [{c}+64]\n",
"vpmulhw zmm30, zmm30, zmm27\n",
"vpsubw zmm28, zmm28, zmm30\n",
bfly!(3,12,21,28),
"vmovdqa64 zmm4, [{i}+256]\n",
"vmovdqa64 zmm13, [{i}+832]\n",
"vmovdqa64 zmm22, [{i}+1408]\n",


"vpsubw zmm28, zmm13, zmm22\n",
"vpmullw zmm30, zmm28, ", $rot, "\n",
"vpmulhw zmm28, zmm28, [{c}+64]\n",
"vpmulhw zmm30, zmm30, zmm27\n",
"vpsubw zmm28, zmm28, zmm30\n",
bfly!(4,13,22,28),
"vmovdqa64 zmm5, [{i}+320]\n",
"vmovdqa64 zmm14, [{i}+896]\n",
"vmovdqa64 zmm23, [{i}+1472]\n",


"vpsubw zmm28, zmm14, zmm23\n",
"vpmullw zmm30, zmm28, ", $rot, "\n",
"vpmulhw zmm28, zmm28, [{c}+64]\n",
"vpmulhw zmm30, zmm30, zmm27\n",
"vpsubw zmm28, zmm28, zmm30\n",
bfly!(5,14,23,28),
"vmovdqa64 zmm6, [{i}+384]\n",
"vmovdqa64 zmm15, [{i}+960]\n",
"vmovdqa64 zmm24, [{i}+1536]\n",


"vpsubw zmm28, zmm15, zmm24\n",
"vpmullw zmm30, zmm28, ", $rot, "\n",
"vpmulhw zmm28, zmm28, [{c}+64]\n",
"vpmulhw zmm30, zmm30, zmm27\n",
"vpsubw zmm28, zmm28, zmm30\n",
bfly!(6,15,24,28),
"vmovdqa64 zmm7, [{i}+448]\n",
"vmovdqa64 zmm16, [{i}+1024]\n",
"vmovdqa64 zmm25, [{i}+1600]\n",


"vpsubw zmm28, zmm16, zmm25\n",
"vpmullw zmm30, zmm28, ", $rot, "\n",
"vpmulhw zmm28, zmm28, [{c}+64]\n",
"vpmulhw zmm30, zmm30, zmm27\n",
"vpsubw zmm28, zmm28, zmm30\n",
bfly!(7,16,25,28),
"vmovdqa64 zmm8, [{i}+512]\n",
"vmovdqa64 zmm17, [{i}+1088]\n",
"vmovdqa64 zmm26, [{i}+1664]\n",


"vpsubw zmm28, zmm17, zmm26\n",
"vpmullw zmm30, zmm28, ", $rot, "\n",
"vpmulhw zmm28, zmm28, [{c}+64]\n",
"vpmulhw zmm30, zmm30, zmm27\n",
"vpsubw zmm28, zmm28, zmm30\n",
bfly!(8,17,26,28),
            "vpsrlw zmm28, zmm0, 11\n",
            "vpermw zmm28, zmm28, zmmword ptr [{c}+512]\n",
            "vpaddw zmm0, zmm0, zmm28\n",
            mont_mem!("t5", "", 3, 6, 28, $rot),
            "vpsrlw zmm29, zmm18, 11\n",
            "vpermw zmm29, zmm29, zmmword ptr [{c}+512]\n",
            "vpaddw zmm18, zmm18, zmm29\n",
            mont_mem!("t5", "+16", 21, 24, 29, $rot),
            bfly!(0, 3, 6, 28),
            "vpsrlw zmm28, zmm9, 11\n",
            "vpermw zmm28, zmm28, zmmword ptr [{c}+512]\n",
            "vpaddw zmm9, zmm9, zmm28\n",
            mont_mem!("t5", "+32", 12, 15, 28, $rot),
            bfly!(18, 21, 24, 29),
            mont_mem!("t5", "", 4, 7, 29, $rot),
            bfly!(9, 12, 15, 28),
            mont_mem!("t5", "+16", 22, 25, 28, $rot),
            bfly!(1, 4, 7, 29),
            mont_mem!("t5", "+32", 13, 16, 29, $rot),
            bfly!(19, 22, 25, 28),
            mont_mem!("t5", "", 5, 8, 28, $rot),
            bfly!(10, 13, 16, 29),
            mont_mem!("t5", "+16", 23, 26, 29, $rot),
            bfly!(2, 5, 8, 28),
            mont_mem!("t5", "+32", 14, 17, 28, $rot),
            bfly!(20, 23, 26, 29),
            bfly!(11, 14, 17, 28),
            level6_mem!(plain "", 0, 1, 2, 28, $rot),
            level6_mem!(plain "+16", 6, 7, 8, 29, $rot),
            bfly!(0, 1, 2, 28, 0, 64, 128),
            level6_mem!(plain "+32", 3, 4, 5, 28, $rot),
            bfly!(6, 7, 8, 29, 192, 256, 320),
            level6_mem!(plain "+48", 18, 19, 20, 29, $rot),
            bfly!(3, 4, 5, 28, 384, 448, 512),
            level6_mem!(plain "+64", 24, 25, 26, 28, $rot),
            bfly!(18, 19, 20, 29, 576, 640, 704),
            level6_mem!(plain "+80", 21, 22, 23, 29, $rot),
            bfly!(24, 25, 26, 28, 768, 832, 896),
            level6_mem!(plain "+96", 9, 10, 11, 28, $rot),
            bfly!(21, 22, 23, 29, 960, 1024, 1088),
            level6_mem!(plain "+112", 15, 16, 17, 29, $rot),
            bfly!(9, 10, 11, 28, 1152, 1216, 1280),
            level6_mem!(plain "+128", 12, 13, 14, 28, $rot),
            bfly!(15, 16, 17, 29, 1344, 1408, 1472),
            bfly!(12, 13, 14, 28, 1536, 1600, 1664),
)
    };
}

#[inline(never)]
#[target_feature(enable="avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vnni,gfni")]
unsafe fn tail27_sms_band0_leaf(
    bp: *const i16,
    op: *mut i16,
    t4: *const u32,
    t5: *const u32,
    t6: *const u32,
    cv: *const i16,
) {
    core::arch::asm!(
        tail27_sms_band0!(),
        i = in(reg) bp,
        o = in(reg) op,
        
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
macro_rules! tail27_sms_band4 {
    () => {
        concat!(
            "vmovdqa64 zmm27, [{c}]\n",
            "vpxord zmm31, zmm31, zmm31\n",
            tail27_sms_band4!(body "", "zmm31", "[{c}+128]"),
        )
    };
    // The leaf body. Each first-level negation subtracts from the zero in `$z`, after `$zi`
    // (empty when `$z` already holds zero for the whole leaf); `$rot` is the source of the
    // rotation's Montgomery low factor.
    (body $zi:literal, $z:literal, $rot:literal) => {
        concat!(
"vmovdqa64 zmm0, [{i}+0]\n",
"vmovdqa64 zmm9, [{i}+576]\n",
"vmovdqa64 zmm18, [{i}+1152]\n",


$zi, "vpsubw zmm9, ", $z, ", zmm9\n",
"vpsubw zmm28, zmm9, zmm18\n",
"vpmullw zmm30, zmm28, ", $rot, "\n",
"vpmulhw zmm28, zmm28, [{c}+64]\n",
"vpmulhw zmm30, zmm30, zmm27\n",
"vpsubw zmm28, zmm28, zmm30\n",
bfly!(0,9,18,28),
"vmovdqa64 zmm1, [{i}+64]\n",
"vmovdqa64 zmm10, [{i}+640]\n",
"vmovdqa64 zmm19, [{i}+1216]\n",
"vpsrlw zmm30, zmm1, 11\n",
"vpermw zmm30, zmm30, zmmword ptr [{c}+512]\n",
"vpaddw zmm1, zmm1, zmm30\n",


$zi, "vpsubw zmm10, ", $z, ", zmm10\n",
"vpsubw zmm28, zmm10, zmm19\n",
"vpmullw zmm30, zmm28, ", $rot, "\n",
"vpmulhw zmm28, zmm28, [{c}+64]\n",
"vpmulhw zmm30, zmm30, zmm27\n",
"vpsubw zmm28, zmm28, zmm30\n",
bfly!(1,10,19,28),
"vmovdqa64 zmm2, [{i}+128]\n",
"vmovdqa64 zmm11, [{i}+704]\n",
"vmovdqa64 zmm20, [{i}+1280]\n",
"vpsrlw zmm30, zmm2, 11\n",
"vpermw zmm30, zmm30, zmmword ptr [{c}+512]\n",
"vpaddw zmm2, zmm2, zmm30\n",


$zi, "vpsubw zmm11, ", $z, ", zmm11\n",
"vpsubw zmm28, zmm11, zmm20\n",
"vpmullw zmm30, zmm28, ", $rot, "\n",
"vpmulhw zmm28, zmm28, [{c}+64]\n",
"vpmulhw zmm30, zmm30, zmm27\n",
"vpsubw zmm28, zmm28, zmm30\n",
bfly!(2,11,20,28),
"vmovdqa64 zmm3, [{i}+192]\n",
"vmovdqa64 zmm12, [{i}+768]\n",
"vmovdqa64 zmm21, [{i}+1344]\n",


$zi, "vpsubw zmm12, ", $z, ", zmm12\n",
"vpsubw zmm28, zmm12, zmm21\n",
"vpmullw zmm30, zmm28, ", $rot, "\n",
"vpmulhw zmm28, zmm28, [{c}+64]\n",
"vpmulhw zmm30, zmm30, zmm27\n",
"vpsubw zmm28, zmm28, zmm30\n",
bfly!(3,12,21,28),
"vmovdqa64 zmm4, [{i}+256]\n",
"vmovdqa64 zmm13, [{i}+832]\n",
"vmovdqa64 zmm22, [{i}+1408]\n",


$zi, "vpsubw zmm13, ", $z, ", zmm13\n",
"vpsubw zmm28, zmm13, zmm22\n",
"vpmullw zmm30, zmm28, ", $rot, "\n",
"vpmulhw zmm28, zmm28, [{c}+64]\n",
"vpmulhw zmm30, zmm30, zmm27\n",
"vpsubw zmm28, zmm28, zmm30\n",
bfly!(4,13,22,28),
"vmovdqa64 zmm5, [{i}+320]\n",
"vmovdqa64 zmm14, [{i}+896]\n",
"vmovdqa64 zmm23, [{i}+1472]\n",


$zi, "vpsubw zmm14, ", $z, ", zmm14\n",
"vpsubw zmm28, zmm14, zmm23\n",
"vpmullw zmm30, zmm28, ", $rot, "\n",
"vpmulhw zmm28, zmm28, [{c}+64]\n",
"vpmulhw zmm30, zmm30, zmm27\n",
"vpsubw zmm28, zmm28, zmm30\n",
bfly!(5,14,23,28),
"vmovdqa64 zmm6, [{i}+384]\n",
"vmovdqa64 zmm15, [{i}+960]\n",
"vmovdqa64 zmm24, [{i}+1536]\n",


$zi, "vpsubw zmm15, ", $z, ", zmm15\n",
"vpsubw zmm28, zmm15, zmm24\n",
"vpmullw zmm30, zmm28, ", $rot, "\n",
"vpmulhw zmm28, zmm28, [{c}+64]\n",
"vpmulhw zmm30, zmm30, zmm27\n",
"vpsubw zmm28, zmm28, zmm30\n",
bfly!(6,15,24,28),
"vmovdqa64 zmm7, [{i}+448]\n",
"vmovdqa64 zmm16, [{i}+1024]\n",
"vmovdqa64 zmm25, [{i}+1600]\n",


$zi, "vpsubw zmm16, ", $z, ", zmm16\n",
"vpsubw zmm28, zmm16, zmm25\n",
"vpmullw zmm30, zmm28, ", $rot, "\n",
"vpmulhw zmm28, zmm28, [{c}+64]\n",
"vpmulhw zmm30, zmm30, zmm27\n",
"vpsubw zmm28, zmm28, zmm30\n",
bfly!(7,16,25,28),
"vmovdqa64 zmm8, [{i}+512]\n",
"vmovdqa64 zmm17, [{i}+1088]\n",
"vmovdqa64 zmm26, [{i}+1664]\n",


$zi, "vpsubw zmm17, ", $z, ", zmm17\n",
"vpsubw zmm28, zmm17, zmm26\n",
"vpmullw zmm30, zmm28, ", $rot, "\n",
"vpmulhw zmm28, zmm28, [{c}+64]\n",
"vpmulhw zmm30, zmm30, zmm27\n",
"vpsubw zmm28, zmm28, zmm30\n",
bfly!(8,17,26,28),
            "vpsrlw zmm28, zmm9, 11\n",
            "vpermw zmm28, zmm28, zmmword ptr [{c}+512]\n",
            "vpaddw zmm9, zmm9, zmm28\n",
            mont_mem!("t5", "", 12, 15, 28, $rot),
            "vpsrlw zmm29, zmm0, 11\n",
            "vpermw zmm29, zmm29, zmmword ptr [{c}+512]\n",
            "vpaddw zmm0, zmm0, zmm29\n",
            mont_mem!("t5", "+16", 3, 6, 29, $rot),
            bfly!(9, 12, 15, 28),
            "vpsrlw zmm28, zmm18, 11\n",
            "vpermw zmm28, zmm28, zmmword ptr [{c}+512]\n",
            "vpaddw zmm18, zmm18, zmm28\n",
            mont_mem!("t5", "+32", 21, 24, 28, $rot),
            bfly!(0, 3, 6, 29),
            mont_mem!("t5", "", 13, 16, 29, $rot),
            bfly!(18, 21, 24, 28),
            mont_mem!("t5", "+16", 4, 7, 28, $rot),
            bfly!(10, 13, 16, 29),
            mont_mem!("t5", "+32", 22, 25, 29, $rot),
            bfly!(1, 4, 7, 28),
            mont_mem!("t5", "", 14, 17, 28, $rot),
            bfly!(19, 22, 25, 29),
            mont_mem!("t5", "+16", 5, 8, 29, $rot),
            bfly!(11, 14, 17, 28),
            mont_mem!("t5", "+32", 23, 26, 28, $rot),
            bfly!(2, 5, 8, 29),
            bfly!(20, 23, 26, 28),
            level6_mem!(plain "", 9, 10, 11, 28, $rot),
            level6_mem!(plain "+16", 15, 16, 17, 29, $rot),
            bfly!(9, 10, 11, 28, 0, 64, 128),
            level6_mem!(plain "+32", 12, 13, 14, 28, $rot),
            bfly!(15, 16, 17, 29, 192, 256, 320),
            level6_mem!(plain "+48", 0, 1, 2, 29, $rot),
            bfly!(12, 13, 14, 28, 384, 448, 512),
            level6_mem!(plain "+64", 6, 7, 8, 28, $rot),
            bfly!(0, 1, 2, 29, 576, 640, 704),
            level6_mem!(plain "+80", 3, 4, 5, 29, $rot),
            bfly!(6, 7, 8, 28, 768, 832, 896),
            level6_mem!(plain "+96", 18, 19, 20, 28, $rot),
            bfly!(3, 4, 5, 29, 960, 1024, 1088),
            level6_mem!(plain "+112", 24, 25, 26, 29, $rot),
            bfly!(18, 19, 20, 28, 1152, 1216, 1280),
            level6_mem!(plain "+128", 21, 22, 23, 28, $rot),
            bfly!(24, 25, 26, 29, 1344, 1408, 1472),
            bfly!(21, 22, 23, 28, 1536, 1600, 1664),
)
    };
}

#[inline(never)]
#[target_feature(enable="avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vnni,gfni")]
unsafe fn tail27_sms_band4_leaf(
    bp: *const i16,
    op: *mut i16,
    t4: *const u32,
    t5: *const u32,
    t6: *const u32,
    cv: *const i16,
) {
    core::arch::asm!(
        tail27_sms_band4!(),
        i = in(reg) bp,
        o = in(reg) op,
        
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
        out("zmm30") _,out("zmm31") _,
        options(nostack)
    );
}

// Runs of the hot leaves. One call transforms the `n` consecutive in-place 27-vector blocks of
// one tail block index, i.e. that block of every lockstep pair batch of the quarter. The modulus
// stays in zmm27 and the rotation's Montgomery low factor in zmm31 for the whole run: a block
// neither reloads the modulus nor reads the rotation factor from the constant block for each of
// its 27 rotations, and the H blocks cost one call instead of H. The block bodies are the leaf
// texts above with that operand in zmm31, so every instruction computes the same words. Band 4
// takes the zero of its first-level negations from a zeroing idiom on the butterfly temporary,
// which is dead at each of those points, instead of from zmm31.
macro_rules! tail27_scaled_held {
    () => {
        tail27_scaled!(lut, plain, bfly, "zmm31")
    };
}
macro_rules! tail27_sms_band0_held {
    () => {
        tail27_sms_band0!(body "zmm31")
    };
}
macro_rules! tail27_sms_band4_held {
    () => {
        tail27_sms_band4!(body "vpxord zmm28, zmm28, zmm28\n", "zmm28", "zmm31")
    };
}

#[inline(never)]
#[target_feature(enable="avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vnni,gfni")]
unsafe fn tail27_scaled_run(bp: *mut i16, n: usize, t4: *const u32, t5: *const u32, t6: *const u32, cv: *const i16) {
    debug_assert!(n > 0);
    core::arch::asm!(
        "vmovdqa64 zmm27, [{c}]",
        "vmovdqa64 zmm31, [{c}+128]",
        "2:",
        tail27_scaled_held!(),
        "add {i}, 1728",
        "add {o}, 1728",
        "dec {n}",
        "jnz 2b",
        i = inout(reg) bp => _,
        o = inout(reg) bp => _,
        n = inout(reg) n => _,
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
        out("zmm31") _,
        options(nostack)
    );
}

#[inline(never)]
#[target_feature(enable="avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vnni,gfni")]
unsafe fn tail27_sms_band0_run(bp: *mut i16, n: usize, t5: *const u32, t6: *const u32, cv: *const i16) {
    debug_assert!(n > 0);
    core::arch::asm!(
        "vmovdqa64 zmm27, [{c}]",
        "vmovdqa64 zmm31, [{c}+128]",
        "2:",
        tail27_sms_band0_held!(),
        "add {i}, 1728",
        "add {o}, 1728",
        "dec {n}",
        "jnz 2b",
        i = inout(reg) bp => _,
        o = inout(reg) bp => _,
        n = inout(reg) n => _,
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
        out("zmm31") _,
        options(nostack)
    );
}

#[inline(never)]
#[target_feature(enable="avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vnni,gfni")]
unsafe fn tail27_sms_band4_run(bp: *mut i16, n: usize, t5: *const u32, t6: *const u32, cv: *const i16) {
    debug_assert!(n > 0);
    core::arch::asm!(
        "vmovdqa64 zmm27, [{c}]",
        "vmovdqa64 zmm31, [{c}+128]",
        "2:",
        tail27_sms_band4_held!(),
        "add {i}, 1728",
        "add {o}, 1728",
        "dec {n}",
        "jnz 2b",
        i = inout(reg) bp => _,
        o = inout(reg) bp => _,
        n = inout(reg) n => _,
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
        out("zmm31") _,
        options(nostack)
    );
}

#[cfg(test)]
mod held_run_tests {
    use super::*;

    const BLOCK: usize = 27 * 32;
    #[repr(C, align(64))]
    struct Blocks([i16; 18 * BLOCK]);

    /// Each run equals its single-block leaf applied to every block, for all 24 twiddle sets, on
    /// arbitrary words (both are the same wrapping word operations), and writes only its blocks.
    #[test]
    fn held_runs_match_the_single_block_leaves() {
        let t = tables::<9721>();
        let cv = t.cv.as_ptr() as *const i16;
        let mut s = 0x2545_f491_4f6c_dd1du64;
        let mut rnd = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        for n in [1usize, 2, 7, 16] {
            for kk in 0..24 {
                let j = kk % 6;
                let mut input = Box::new(Blocks([0; 18 * BLOCK]));
                for (i, x) in input.0.iter_mut().enumerate() {
                    *x = match (i / BLOCK + i / 7 + kk) % 6 {
                        0 => 0,
                        1 => i16::MAX,
                        2 => i16::MIN,
                        3 => ((rnd() % 62_593) as i32 - 31_296) as i16,
                        _ => rnd() as i16,
                    };
                }
                let mut expected = Box::new(Blocks(input.0));
                let mut got = Box::new(Blocks(input.0));
                unsafe {
                    let t4 = SMS_BAND_T4.0.as_ptr().add(4 * kk).cast::<u32>();
                    let t5 = SMS_TWIDDLES.t5.as_ptr().add(12 * kk).cast::<u32>();
                    let t6 = SMS_TWIDDLES.t6.as_ptr().add(36 * kk).cast::<u32>();
                    for b in 1..=n {
                        let (p, o) = (input.0.as_ptr().add(BLOCK * b), expected.0.as_mut_ptr().add(BLOCK * b));
                        match j {
                            0 => tail27_sms_band0_leaf(p, o, t4, t5, t6, cv),
                            4 => tail27_sms_band4_leaf(p, o, t4, t5, t6, cv),
                            _ => tail27_scaled_leaf(p, o, t4, t5, t6, cv),
                        }
                    }
                    let g = got.0.as_mut_ptr().add(BLOCK);
                    match j {
                        0 => tail27_sms_band0_run(g, n, t5, t6, cv),
                        4 => tail27_sms_band4_run(g, n, t5, t6, cv),
                        _ => tail27_scaled_run(g, n, t4, t5, t6, cv),
                    }
                }
                assert!(got.0 == expected.0, "n={n} kk={kk}");
            }
        }
    }
}
