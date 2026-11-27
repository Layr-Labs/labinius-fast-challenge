.text
.p2align 4
.globl labinius_guard_burst
.hidden labinius_guard_burst
labinius_guard_burst:
.cfi_startproc
sub sp, sp, #160
.cfi_def_cfa_offset 160
stp d8, d9, [sp, #0]
stp d10, d11, [sp, #16]
stp d12, d13, [sp, #32]
stp d14, d15, [sp, #48]
stp x19, x20, [sp, #64]
stp x21, x22, [sp, #80]
stp x23, x24, [sp, #96]
stp x25, x26, [sp, #112]
stp x27, x28, [sp, #128]
stp x29, x30, [sp, #144]
.cfi_offset x19, -96
.cfi_offset x20, -88
.cfi_offset x21, -80
.cfi_offset x22, -72
.cfi_offset x23, -64
.cfi_offset x24, -56
.cfi_offset x25, -48
.cfi_offset x26, -40
.cfi_offset x27, -32
.cfi_offset x28, -24
.cfi_offset x29, -16
.cfi_offset x30, -8
.cfi_offset d8, -160
.cfi_offset d9, -152
.cfi_offset d10, -144
.cfi_offset d11, -136
.cfi_offset d12, -128
.cfi_offset d13, -120
.cfi_offset d14, -112
.cfi_offset d15, -104
mov x19,x0
mov x20,x1
mov x21,x2
ldp x22,x23,[x19,#128]
ldp x24,x26,[x19,#144]
ldr x27,[x19,#160]
mov x9,#10368
madd x24,x20,x9,x24
mrs x25,fpsr
msr fpsr,xzr
.Lburst_loop:
and x9,x20,#3
add x9,x19,x9,lsl #4
ldp x0,x4,[x9]
lsr x10,x20,#2
add x0,x0,x10,lsl #8
add x4,x4,x10,lsl #8
ldr w11,[x19,#184]
cbz w11,.Lno_hints
cmp x20,#124
b.hs .Lskip_l1
prfm pldl1keep, [x0, #256]
prfm pldl1keep, [x0, #320]
prfm pldl1keep, [x0, #384]
prfm pldl1keep, [x0, #448]
prfm pldl1keep, [x4, #256]
prfm pldl1keep, [x4, #320]
prfm pldl1keep, [x4, #384]
prfm pldl1keep, [x4, #448]
.Lskip_l1:
cmp x20,#112
b.hs .Lnext_hint
prfm pldl2keep, [x0, #1024]
prfm pldl2keep, [x0, #1088]
prfm pldl2keep, [x0, #1152]
prfm pldl2keep, [x0, #1216]
prfm pldl2keep, [x4, #1024]
prfm pldl2keep, [x4, #1088]
prfm pldl2keep, [x4, #1152]
prfm pldl2keep, [x4, #1216]
b .Lno_hints
.Lnext_hint:
ldp x11,x12,[x9,#64]
sub x10,x10,#28
add x11,x11,x10,lsl #8
add x12,x12,x10,lsl #8
prfm pldl2keep, [x11, #0]
prfm pldl2keep, [x11, #64]
prfm pldl2keep, [x11, #128]
prfm pldl2keep, [x11, #192]
prfm pldl2keep, [x12, #0]
prfm pldl2keep, [x12, #64]
prfm pldl2keep, [x12, #128]
prfm pldl2keep, [x12, #192]
.Lno_hints:
add x1,x0,#64
add x2,x0,#128
add x3,x0,#192
add x5,x4,#64
add x6,x4,#128
add x7,x4,#192
mov x8,x22
bl .Lsealed_indices
mov x0,x23
mov x1,x26
add x2,x22,#64
add x3,x22,#192
add x4,x22,#320
add x5,x22,#448
add x6,x22,#576
add x7,x22,#704
add x8,x22,#832
add x9,x22,#960
add x10,x22,#1088
add x11,x22,#1216
add x12,x22,#1344
add x13,x22,#1472
bl .Lsealed_front
mov x0,x23
mov x1,x24
ldp x2,x3,[x19,#168]
mov x4,x27
bl .Lsealed_guard
add x20,x20,#1
add x24,x24,#2,lsl #12
add x24,x24,#2176
cmp x20,x21
b.lo .Lburst_loop
mrs x0,fpsr
msr fpsr,x25
ubfx x0,x0,#27,#1
ldp d8, d9, [sp, #0]
ldp d10, d11, [sp, #16]
ldp d12, d13, [sp, #32]
ldp d14, d15, [sp, #48]
ldp x19, x20, [sp, #64]
ldp x21, x22, [sp, #80]
ldp x23, x24, [sp, #96]
ldp x25, x26, [sp, #112]
ldp x27, x28, [sp, #128]
ldp x29, x30, [sp, #144]
add sp,sp,#160
.cfi_def_cfa_offset 0
ret
.cfi_endproc
.Lsealed_indices:
movi v28.16b, #0x0f
movi v29.16b, #0x55
movi v30.16b, #0x33
movi v31.16b, #0
ldr q16, [x0, #0]
ldr q17, [x1, #0]
ldr q18, [x2, #0]
ldr q19, [x3, #0]
ldr q20, [x4, #0]
ldr q21, [x5, #0]
ldr q22, [x6, #0]
ldr q23, [x7, #0]
ldr q24, [x0, #32]
ldr q25, [x1, #32]
ldr q26, [x2, #32]
ldr q27, [x3, #32]
ldr q0, [x4, #32]
ldr q1, [x5, #32]
zip1 v2.16b, v16.16b, v17.16b
zip2 v16.16b, v16.16b, v17.16b
ldr q17, [x6, #32]
ldr q3, [x7, #32]
zip1 v4.16b, v18.16b, v19.16b
zip2 v18.16b, v18.16b, v19.16b
zip1 v19.16b, v20.16b, v21.16b
zip2 v20.16b, v20.16b, v21.16b
zip1 v21.16b, v22.16b, v23.16b
zip2 v22.16b, v22.16b, v23.16b
zip2 v23.8h, v2.8h, v4.8h
zip1 v5.8h, v16.8h, v18.8h
zip2 v6.16b, v24.16b, v25.16b
zip2 v16.8h, v16.8h, v18.8h
zip1 v18.16b, v24.16b, v25.16b
zip1 v24.8h, v2.8h, v4.8h
zip2 v25.16b, v26.16b, v27.16b
zip2 v7.8h, v19.8h, v21.8h
zip1 v8.8h, v20.8h, v22.8h
zip2 v20.8h, v20.8h, v22.8h
zip2 v22.16b, v0.16b, v1.16b
zip1 v26.16b, v26.16b, v27.16b
zip1 v27.16b, v0.16b, v1.16b
zip1 v19.8h, v19.8h, v21.8h
zip2 v21.16b, v17.16b, v3.16b
zip1 v9.8h, v6.8h, v25.8h
zip2 v25.8h, v6.8h, v25.8h
zip1 v10.4s, v23.4s, v7.4s
zip2 v23.4s, v23.4s, v7.4s
zip1 v11.4s, v5.4s, v8.4s
zip2 v12.4s, v5.4s, v8.4s
zip1 v13.4s, v16.4s, v20.4s
zip1 v14.8h, v22.8h, v21.8h
zip2 v22.8h, v22.8h, v21.8h
zip2 v21.4s, v16.4s, v20.4s
zip1 v16.16b, v17.16b, v3.16b
zip2 v20.8h, v18.8h, v26.8h
zip1 v17.8h, v18.8h, v26.8h
zip1 v18.4s, v24.4s, v19.4s
zip2 v26.4s, v24.4s, v19.4s
zip2 v24.4s, v9.4s, v14.4s
zip1 v19.4s, v25.4s, v22.4s
zip2 v25.4s, v25.4s, v22.4s
zip2 v22.8h, v27.8h, v16.8h
zip1 v27.8h, v27.8h, v16.8h
zip1 v16.4s, v9.4s, v14.4s
ext v15.16b, v12.16b, v13.16b, #8
ushr v2.16b, v12.16b, #1
zip1 v4.4s, v20.4s, v22.4s
zip2 v20.4s, v20.4s, v22.4s
ext v22.16b, v25.16b, v31.16b, #8
ushr v0.16b, v18.16b, #1
zip1 v1.4s, v17.4s, v27.4s
zip2 v17.4s, v17.4s, v27.4s
shl v27.16b, v15.16b, #7
ext v6.16b, v10.16b, v23.16b, #8
ext v7.16b, v4.16b, v20.16b, #8
ushr v22.16b, v22.16b, #1
ext v5.16b, v24.16b, v19.16b, #8
shl v8.16b, v23.16b, #7
orr v27.16b, v2.16b, v27.16b
ushr v3.16b, v6.16b, #1
ushr v9.16b, v24.16b, #1
ext v20.16b, v20.16b, v16.16b, #8
shl v14.16b, v7.16b, #1
shl v15.16b, v22.16b, #1
bif v0.16b, v7.16b, v29.16b
ext v16.16b, v16.16b, v24.16b, #8
ushr v24.16b, v27.16b, #1
orr v2.16b, v3.16b, v8.16b
ext v23.16b, v23.16b, v11.16b, #8
bit v14.16b, v18.16b, v29.16b
bit v15.16b, v27.16b, v29.16b
ushr v18.16b, v0.16b, #2
shl v27.16b, v5.16b, #7
bif v24.16b, v22.16b, v29.16b
ext v22.16b, v21.16b, v31.16b, #8
shl v6.16b, v2.16b, #1
ushr v23.16b, v23.16b, #1
shl v7.16b, v15.16b, #2
ushr v3.16b, v14.16b, #2
shl v8.16b, v24.16b, #2
bif v18.16b, v24.16b, v30.16b
bit v6.16b, v1.16b, v29.16b
bit v7.16b, v14.16b, v30.16b
bif v3.16b, v15.16b, v30.16b
orr v24.16b, v9.16b, v27.16b
ushr v27.16b, v22.16b, #2
bit v8.16b, v0.16b, v30.16b
and v22.16b, v18.16b, v28.16b
ushr v18.16b, v18.16b, #4
ushr v5.16b, v1.16b, #1
and v14.16b, v7.16b, v28.16b
ushr v15.16b, v7.16b, #4
and v9.16b, v3.16b, v28.16b
ushr v0.16b, v3.16b, #4
and v1.16b, v8.16b, v28.16b
ushr v7.16b, v8.16b, #4
str q22, [x8, #384]
str q18, [x8, #400]
bif v5.16b, v2.16b, v29.16b
ushr v22.16b, v6.16b, #2
str q14, [x8, #0]
str q15, [x8, #16]
shl v18.16b, v27.16b, #1
ushr v3.16b, v24.16b, #1
str q1, [x8, #128]
str q7, [x8, #144]
ushr v8.16b, v5.16b, #2
ext v2.16b, v13.16b, v21.16b, #8
ushr v14.16b, v13.16b, #1
str q9, [x8, #256]
str q0, [x8, #272]
bit v18.16b, v24.16b, v29.16b
bif v3.16b, v27.16b, v29.16b
shl v24.16b, v20.16b, #1
ushr v27.16b, v26.16b, #1
shl v15.16b, v2.16b, #7
ext v1.16b, v19.16b, v25.16b, #8
shl v7.16b, v11.16b, #7
ext v13.16b, v11.16b, v12.16b, #8
shl v9.16b, v18.16b, #2
shl v0.16b, v3.16b, #2
bif v22.16b, v18.16b, v30.16b
bif v8.16b, v3.16b, v30.16b
orr v18.16b, v14.16b, v15.16b
bit v24.16b, v26.16b, v29.16b
bif v27.16b, v20.16b, v29.16b
ushr v26.16b, v19.16b, #1
bit v9.16b, v6.16b, v30.16b
bit v0.16b, v5.16b, v30.16b
and v20.16b, v22.16b, v28.16b
ushr v19.16b, v22.16b, #4
and v22.16b, v8.16b, v28.16b
ushr v2.16b, v8.16b, #4
ushr v11.16b, v18.16b, #1
and v18.16b, v18.16b, v29.16b
and v3.16b, v9.16b, v28.16b
ushr v14.16b, v9.16b, #4
and v15.16b, v0.16b, v28.16b
ushr v6.16b, v0.16b, #4
str q20, [x8, #320]
str q19, [x8, #336]
str q22, [x8, #448]
str q2, [x8, #464]
and v20.16b, v11.16b, v29.16b
shl v19.16b, v18.16b, #2
ushr v22.16b, v24.16b, #2
orr v23.16b, v23.16b, v7.16b
str q3, [x8, #64]
str q14, [x8, #80]
ushr v5.16b, v27.16b, #2
shl v8.16b, v1.16b, #7
ext v9.16b, v21.16b, v31.16b, #8
ext v0.16b, v25.16b, v31.16b, #8
str q15, [x8, #192]
str q6, [x8, #208]
shl v2.16b, v20.16b, #2
bit v19.16b, v24.16b, v30.16b
bif v22.16b, v18.16b, v30.16b
shl v24.16b, v23.16b, #1
bif v5.16b, v20.16b, v30.16b
orr v18.16b, v26.16b, v8.16b
ushr v20.16b, v17.16b, #1
ushr v26.16b, v21.16b, #1
bit v2.16b, v27.16b, v30.16b
and v21.16b, v19.16b, v28.16b
ushr v27.16b, v19.16b, #4
and v19.16b, v22.16b, v28.16b
ushr v22.16b, v22.16b, #4
and v11.16b, v5.16b, v28.16b
ushr v7.16b, v5.16b, #4
bit v24.16b, v17.16b, v29.16b
and v17.16b, v2.16b, v28.16b
ushr v3.16b, v2.16b, #4
str q21, [x8, #512]
str q27, [x8, #528]
ushr v21.16b, v18.16b, #1
bif v20.16b, v23.16b, v29.16b
str q19, [x8, #768]
str q22, [x8, #784]
and v27.16b, v18.16b, v29.16b
ushr v23.16b, v24.16b, #2
shl v19.16b, v9.16b, #7
str q17, [x8, #640]
str q3, [x8, #656]
and v22.16b, v21.16b, v29.16b
ushr v18.16b, v20.16b, #2
shl v17.16b, v16.16b, #1
str q11, [x8, #896]
str q7, [x8, #912]
shl v21.16b, v27.16b, #2
bif v23.16b, v27.16b, v30.16b
orr v27.16b, v26.16b, v19.16b
ushr v26.16b, v10.16b, #1
shl v19.16b, v22.16b, #2
bif v18.16b, v22.16b, v30.16b
bit v17.16b, v10.16b, v29.16b
ushr v22.16b, v13.16b, #1
bit v21.16b, v24.16b, v30.16b
and v24.16b, v23.16b, v28.16b
ushr v23.16b, v23.16b, #4
ushr v14.16b, v27.16b, #1
bit v19.16b, v20.16b, v30.16b
and v20.16b, v18.16b, v28.16b
ushr v18.16b, v18.16b, #4
bif v26.16b, v16.16b, v29.16b
and v16.16b, v21.16b, v28.16b
ushr v21.16b, v21.16b, #4
str q24, [x8, #832]
str q23, [x8, #848]
and v24.16b, v27.16b, v29.16b
and v23.16b, v14.16b, v29.16b
and v27.16b, v19.16b, v28.16b
ushr v19.16b, v19.16b, #4
str q20, [x8, #960]
str q18, [x8, #976]
ushr v20.16b, v17.16b, #2
str q16, [x8, #576]
str q21, [x8, #592]
shl v18.16b, v24.16b, #2
shl v16.16b, v23.16b, #2
str q27, [x8, #704]
str q19, [x8, #720]
ushr v21.16b, v26.16b, #2
bif v20.16b, v24.16b, v30.16b
shl v27.16b, v12.16b, #7
bit v18.16b, v17.16b, v30.16b
bit v16.16b, v26.16b, v30.16b
ushr v19.16b, v25.16b, #1
shl v24.16b, v0.16b, #7
bif v21.16b, v23.16b, v30.16b
and v17.16b, v20.16b, v28.16b
ushr v26.16b, v20.16b, #4
orr v25.16b, v22.16b, v27.16b
and v23.16b, v18.16b, v28.16b
ushr v20.16b, v18.16b, #4
and v22.16b, v16.16b, v28.16b
ushr v27.16b, v16.16b, #4
and v18.16b, v21.16b, v28.16b
ushr v16.16b, v21.16b, #4
str q17, [x8, #1280]
str q26, [x8, #1296]
orr v21.16b, v19.16b, v24.16b
shl v17.16b, v25.16b, #1
str q23, [x8, #1024]
str q20, [x8, #1040]
ushr v26.16b, v4.16b, #1
str q22, [x8, #1152]
str q27, [x8, #1168]
ushr v19.16b, v21.16b, #1
bit v17.16b, v4.16b, v29.16b
and v24.16b, v21.16b, v29.16b
str q18, [x8, #1408]
str q16, [x8, #1424]
bif v26.16b, v25.16b, v29.16b
and v23.16b, v19.16b, v29.16b
shl v20.16b, v24.16b, #2
ushr v22.16b, v17.16b, #2
ushr v27.16b, v26.16b, #2
shl v21.16b, v23.16b, #2
bit v20.16b, v17.16b, v30.16b
bif v22.16b, v24.16b, v30.16b
bif v27.16b, v23.16b, v30.16b
bit v21.16b, v26.16b, v30.16b
and v18.16b, v20.16b, v28.16b
ushr v16.16b, v20.16b, #4
and v25.16b, v22.16b, v28.16b
ushr v19.16b, v22.16b, #4
and v17.16b, v27.16b, v28.16b
ushr v24.16b, v27.16b, #4
and v23.16b, v21.16b, v28.16b
ushr v26.16b, v21.16b, #4
str q18, [x8, #1088]
str q16, [x8, #1104]
str q25, [x8, #1344]
str q19, [x8, #1360]
str q23, [x8, #1216]
str q26, [x8, #1232]
str q17, [x8, #1472]
str q24, [x8, #1488]
ldr q16, [x0, #16]
ldr q17, [x1, #16]
ldr q18, [x2, #16]
ldr q19, [x3, #16]
ldr q20, [x4, #16]
ldr q21, [x5, #16]
ldr q22, [x6, #16]
ldr q23, [x7, #16]
ldr q24, [x0, #48]
ldr q25, [x1, #48]
ldr q26, [x2, #48]
ldr q27, [x3, #48]
ldr q0, [x4, #48]
ldr q1, [x5, #48]
zip1 v2.16b, v16.16b, v17.16b
zip2 v16.16b, v16.16b, v17.16b
ldr q17, [x6, #48]
ldr q3, [x7, #48]
zip1 v4.16b, v18.16b, v19.16b
zip2 v18.16b, v18.16b, v19.16b
zip1 v19.16b, v20.16b, v21.16b
zip2 v20.16b, v20.16b, v21.16b
zip1 v21.16b, v22.16b, v23.16b
zip2 v22.16b, v22.16b, v23.16b
zip2 v23.8h, v2.8h, v4.8h
zip1 v5.8h, v16.8h, v18.8h
zip2 v6.16b, v24.16b, v25.16b
zip2 v16.8h, v16.8h, v18.8h
zip1 v18.16b, v24.16b, v25.16b
zip1 v24.8h, v2.8h, v4.8h
zip2 v25.16b, v26.16b, v27.16b
zip2 v7.8h, v19.8h, v21.8h
zip1 v8.8h, v20.8h, v22.8h
zip2 v20.8h, v20.8h, v22.8h
zip2 v22.16b, v0.16b, v1.16b
zip1 v26.16b, v26.16b, v27.16b
zip1 v27.16b, v0.16b, v1.16b
zip1 v19.8h, v19.8h, v21.8h
zip2 v21.16b, v17.16b, v3.16b
zip1 v9.8h, v6.8h, v25.8h
zip2 v25.8h, v6.8h, v25.8h
zip1 v10.4s, v23.4s, v7.4s
zip2 v23.4s, v23.4s, v7.4s
zip1 v11.4s, v5.4s, v8.4s
zip2 v12.4s, v5.4s, v8.4s
zip1 v13.4s, v16.4s, v20.4s
zip1 v14.8h, v22.8h, v21.8h
zip2 v22.8h, v22.8h, v21.8h
zip2 v21.4s, v16.4s, v20.4s
zip1 v16.16b, v17.16b, v3.16b
zip2 v20.8h, v18.8h, v26.8h
zip1 v17.8h, v18.8h, v26.8h
zip1 v18.4s, v24.4s, v19.4s
zip2 v26.4s, v24.4s, v19.4s
zip2 v24.4s, v9.4s, v14.4s
zip1 v19.4s, v25.4s, v22.4s
zip2 v25.4s, v25.4s, v22.4s
zip2 v22.8h, v27.8h, v16.8h
zip1 v27.8h, v27.8h, v16.8h
zip1 v16.4s, v9.4s, v14.4s
ext v15.16b, v12.16b, v13.16b, #8
ushr v2.16b, v12.16b, #1
zip1 v4.4s, v20.4s, v22.4s
zip2 v20.4s, v20.4s, v22.4s
ext v22.16b, v25.16b, v31.16b, #8
ushr v0.16b, v18.16b, #1
zip1 v1.4s, v17.4s, v27.4s
zip2 v17.4s, v17.4s, v27.4s
shl v27.16b, v15.16b, #7
ext v6.16b, v10.16b, v23.16b, #8
ext v7.16b, v4.16b, v20.16b, #8
ushr v22.16b, v22.16b, #1
ext v5.16b, v24.16b, v19.16b, #8
shl v8.16b, v23.16b, #7
orr v27.16b, v2.16b, v27.16b
ushr v3.16b, v6.16b, #1
ushr v9.16b, v24.16b, #1
ext v20.16b, v20.16b, v16.16b, #8
shl v14.16b, v7.16b, #1
shl v15.16b, v22.16b, #1
bif v0.16b, v7.16b, v29.16b
ext v16.16b, v16.16b, v24.16b, #8
ushr v24.16b, v27.16b, #1
orr v2.16b, v3.16b, v8.16b
ext v23.16b, v23.16b, v11.16b, #8
bit v14.16b, v18.16b, v29.16b
bit v15.16b, v27.16b, v29.16b
ushr v18.16b, v0.16b, #2
shl v27.16b, v5.16b, #7
bif v24.16b, v22.16b, v29.16b
ext v22.16b, v21.16b, v31.16b, #8
shl v6.16b, v2.16b, #1
ushr v23.16b, v23.16b, #1
shl v7.16b, v15.16b, #2
ushr v3.16b, v14.16b, #2
shl v8.16b, v24.16b, #2
bif v18.16b, v24.16b, v30.16b
bit v6.16b, v1.16b, v29.16b
bit v7.16b, v14.16b, v30.16b
bif v3.16b, v15.16b, v30.16b
orr v24.16b, v9.16b, v27.16b
ushr v27.16b, v22.16b, #2
bit v8.16b, v0.16b, v30.16b
and v22.16b, v18.16b, v28.16b
ushr v18.16b, v18.16b, #4
ushr v5.16b, v1.16b, #1
and v14.16b, v7.16b, v28.16b
ushr v15.16b, v7.16b, #4
and v9.16b, v3.16b, v28.16b
ushr v0.16b, v3.16b, #4
and v1.16b, v8.16b, v28.16b
ushr v7.16b, v8.16b, #4
str q22, [x8, #416]
str q18, [x8, #432]
bif v5.16b, v2.16b, v29.16b
ushr v22.16b, v6.16b, #2
str q14, [x8, #32]
str q15, [x8, #48]
shl v18.16b, v27.16b, #1
ushr v3.16b, v24.16b, #1
str q1, [x8, #160]
str q7, [x8, #176]
ushr v8.16b, v5.16b, #2
ext v2.16b, v13.16b, v21.16b, #8
ushr v14.16b, v13.16b, #1
str q9, [x8, #288]
str q0, [x8, #304]
bit v18.16b, v24.16b, v29.16b
bif v3.16b, v27.16b, v29.16b
shl v24.16b, v20.16b, #1
ushr v27.16b, v26.16b, #1
shl v15.16b, v2.16b, #7
ext v1.16b, v19.16b, v25.16b, #8
shl v7.16b, v11.16b, #7
ext v13.16b, v11.16b, v12.16b, #8
shl v9.16b, v18.16b, #2
shl v0.16b, v3.16b, #2
bif v22.16b, v18.16b, v30.16b
bif v8.16b, v3.16b, v30.16b
orr v18.16b, v14.16b, v15.16b
bit v24.16b, v26.16b, v29.16b
bif v27.16b, v20.16b, v29.16b
ushr v26.16b, v19.16b, #1
bit v9.16b, v6.16b, v30.16b
bit v0.16b, v5.16b, v30.16b
and v20.16b, v22.16b, v28.16b
ushr v19.16b, v22.16b, #4
and v22.16b, v8.16b, v28.16b
ushr v2.16b, v8.16b, #4
ushr v11.16b, v18.16b, #1
and v18.16b, v18.16b, v29.16b
and v3.16b, v9.16b, v28.16b
ushr v14.16b, v9.16b, #4
and v15.16b, v0.16b, v28.16b
ushr v6.16b, v0.16b, #4
str q20, [x8, #352]
str q19, [x8, #368]
str q22, [x8, #480]
str q2, [x8, #496]
and v20.16b, v11.16b, v29.16b
shl v19.16b, v18.16b, #2
ushr v22.16b, v24.16b, #2
orr v23.16b, v23.16b, v7.16b
str q3, [x8, #96]
str q14, [x8, #112]
ushr v5.16b, v27.16b, #2
shl v8.16b, v1.16b, #7
ext v9.16b, v21.16b, v31.16b, #8
ext v0.16b, v25.16b, v31.16b, #8
str q15, [x8, #224]
str q6, [x8, #240]
shl v2.16b, v20.16b, #2
bit v19.16b, v24.16b, v30.16b
bif v22.16b, v18.16b, v30.16b
shl v24.16b, v23.16b, #1
bif v5.16b, v20.16b, v30.16b
orr v18.16b, v26.16b, v8.16b
ushr v20.16b, v17.16b, #1
ushr v26.16b, v21.16b, #1
bit v2.16b, v27.16b, v30.16b
and v21.16b, v19.16b, v28.16b
ushr v27.16b, v19.16b, #4
and v19.16b, v22.16b, v28.16b
ushr v22.16b, v22.16b, #4
and v11.16b, v5.16b, v28.16b
ushr v7.16b, v5.16b, #4
bit v24.16b, v17.16b, v29.16b
and v17.16b, v2.16b, v28.16b
ushr v3.16b, v2.16b, #4
str q21, [x8, #544]
str q27, [x8, #560]
ushr v21.16b, v18.16b, #1
bif v20.16b, v23.16b, v29.16b
str q19, [x8, #800]
str q22, [x8, #816]
and v27.16b, v18.16b, v29.16b
ushr v23.16b, v24.16b, #2
shl v19.16b, v9.16b, #7
str q17, [x8, #672]
str q3, [x8, #688]
and v22.16b, v21.16b, v29.16b
ushr v18.16b, v20.16b, #2
shl v17.16b, v16.16b, #1
str q11, [x8, #928]
str q7, [x8, #944]
shl v21.16b, v27.16b, #2
bif v23.16b, v27.16b, v30.16b
orr v27.16b, v26.16b, v19.16b
ushr v26.16b, v10.16b, #1
shl v19.16b, v22.16b, #2
bif v18.16b, v22.16b, v30.16b
bit v17.16b, v10.16b, v29.16b
ushr v22.16b, v13.16b, #1
bit v21.16b, v24.16b, v30.16b
and v24.16b, v23.16b, v28.16b
ushr v23.16b, v23.16b, #4
ushr v14.16b, v27.16b, #1
bit v19.16b, v20.16b, v30.16b
and v20.16b, v18.16b, v28.16b
ushr v18.16b, v18.16b, #4
bif v26.16b, v16.16b, v29.16b
and v16.16b, v21.16b, v28.16b
ushr v21.16b, v21.16b, #4
str q24, [x8, #864]
str q23, [x8, #880]
and v24.16b, v27.16b, v29.16b
and v23.16b, v14.16b, v29.16b
and v27.16b, v19.16b, v28.16b
ushr v19.16b, v19.16b, #4
str q20, [x8, #992]
str q18, [x8, #1008]
ushr v20.16b, v17.16b, #2
str q16, [x8, #608]
str q21, [x8, #624]
shl v18.16b, v24.16b, #2
shl v16.16b, v23.16b, #2
str q27, [x8, #736]
str q19, [x8, #752]
ushr v21.16b, v26.16b, #2
bif v20.16b, v24.16b, v30.16b
shl v27.16b, v12.16b, #7
bit v18.16b, v17.16b, v30.16b
bit v16.16b, v26.16b, v30.16b
ushr v19.16b, v25.16b, #1
shl v24.16b, v0.16b, #7
bif v21.16b, v23.16b, v30.16b
and v17.16b, v20.16b, v28.16b
ushr v26.16b, v20.16b, #4
orr v25.16b, v22.16b, v27.16b
and v23.16b, v18.16b, v28.16b
ushr v20.16b, v18.16b, #4
and v22.16b, v16.16b, v28.16b
ushr v27.16b, v16.16b, #4
and v18.16b, v21.16b, v28.16b
ushr v16.16b, v21.16b, #4
str q17, [x8, #1312]
str q26, [x8, #1328]
orr v21.16b, v19.16b, v24.16b
shl v17.16b, v25.16b, #1
str q23, [x8, #1056]
str q20, [x8, #1072]
ushr v26.16b, v4.16b, #1
str q22, [x8, #1184]
str q27, [x8, #1200]
ushr v19.16b, v21.16b, #1
bit v17.16b, v4.16b, v29.16b
and v24.16b, v21.16b, v29.16b
str q18, [x8, #1440]
str q16, [x8, #1456]
bif v26.16b, v25.16b, v29.16b
and v23.16b, v19.16b, v29.16b
shl v20.16b, v24.16b, #2
ushr v22.16b, v17.16b, #2
ushr v27.16b, v26.16b, #2
shl v21.16b, v23.16b, #2
bit v20.16b, v17.16b, v30.16b
bif v22.16b, v24.16b, v30.16b
bif v27.16b, v23.16b, v30.16b
bit v21.16b, v26.16b, v30.16b
and v18.16b, v20.16b, v28.16b
ushr v16.16b, v20.16b, #4
and v25.16b, v22.16b, v28.16b
ushr v19.16b, v22.16b, #4
and v17.16b, v27.16b, v28.16b
ushr v24.16b, v27.16b, #4
and v23.16b, v21.16b, v28.16b
ushr v26.16b, v21.16b, #4
str q18, [x8, #1120]
str q16, [x8, #1136]
str q25, [x8, #1376]
str q19, [x8, #1392]
str q23, [x8, #1248]
str q26, [x8, #1264]
str q17, [x8, #1504]
str q24, [x8, #1520]
ret
.Lsealed_front:
ptrue p7.h
mov x14, #4
3:
ldp q16, q17, [x1, #0]
ldp q18, q19, [x1, #32]
ldp q20, q21, [x1, #64]
ldp q22, q23, [x1, #96]
ldp q24, q25, [x1, #128]
ldr q26, [x1, #160]
ld1b {z27.h}, p7/z, [x4, #6, mul vl]
ld1b {z28.h}, p7/z, [x9, #-7, mul vl]
ld1b {z29.h}, p7/z, [x3, #3, mul vl]
ld1b {z30.h}, p7/z, [x11, #4, mul vl]
ld1b {z31.h}, p7/z, [x2, #-8, mul vl]
ld1b {z0.h}, p7/z, [x6, #-2, mul vl]
tbl z1.h, {z19.h}, z27.h
ld1b {z2.h}, p7/z, [x5, #-6, mul vl]
tbl z27.h, {z23.h}, z27.h
tbl z3.h, {z20.h}, z28.h
ld1b {z4.h}, p7/z, [x9, #-3, mul vl]
tbl z28.h, {z24.h}, z28.h
tbl z5.h, {z21.h}, z29.h
ld1b {z6.h}, p7/z, [x3, #7, mul vl]
tbl z29.h, {z25.h}, z29.h
tbl z7.h, {z22.h}, z30.h
ld1b {z8.h}, p7/z, [x12, #-8, mul vl]
tbl z30.h, {z26.h}, z30.h
tbl z31.h, {z16.h, z17.h}, z31.h
add v9.8h, v3.8h, v5.8h
ld1b {z10.h}, p7/z, [x2, #-4, mul vl]
add v28.8h, v28.8h, v29.8h
tbl z29.h, {z18.h}, z0.h
add v11.8h, v1.8h, v7.8h
ld1b {z12.h}, p7/z, [x6, #2, mul vl]
add v27.8h, v27.8h, v30.8h
tbl z30.h, {z19.h}, z2.h
ld1b {z13.h}, p7/z, [x5, #-2, mul vl]
tbl z14.h, {z23.h}, z2.h
add v15.8h, v31.8h, v29.8h
sub v31.8h, v31.8h, v29.8h
add v29.8h, v9.8h, v11.8h
sub v3.8h, v9.8h, v11.8h
ld1b {z5.h}, p7/z, [x9, #1, mul vl]
add v0.8h, v28.8h, v27.8h
sub v28.8h, v28.8h, v27.8h
tbl z27.h, {z20.h}, z4.h
tbl z1.h, {z21.h}, z6.h
ld1b {z7.h}, p7/z, [x4, #-5, mul vl]
sub v2.8h, v15.8h, v29.8h
sub v9.8h, v31.8h, v3.8h
tbl z11.h, {z22.h}, z8.h
add v29.8h, v15.8h, v29.8h
add v15.8h, v15.8h, v0.8h
add v3.8h, v31.8h, v3.8h
add v31.8h, v31.8h, v28.8h
tbl z10.h, {z16.h, z17.h}, z10.h
sub v2.8h, v2.8h, v0.8h
sub v28.8h, v9.8h, v28.8h
tbl z0.h, {z18.h}, z12.h
tbl z9.h, {z24.h}, z4.h
ld1b {z12.h}, p7/z, [x2, #0, mul vl]
ld1b {z4.h}, p7/z, [x12, #-4, mul vl]
tbl z6.h, {z25.h}, z6.h
tbl z8.h, {z26.h}, z8.h
add v27.8h, v27.8h, v1.8h
add v30.8h, v30.8h, v11.8h
ld1b {z1.h}, p7/z, [x6, #6, mul vl]
add v11.8h, v10.8h, v0.8h
sub v10.8h, v10.8h, v0.8h
tbl z0.h, {z19.h}, z13.h
tbl z13.h, {z23.h}, z13.h
str q29, [x0, #0]
add v29.8h, v9.8h, v6.8h
add v9.8h, v14.8h, v8.8h
add v6.8h, v27.8h, v30.8h
sub v27.8h, v27.8h, v30.8h
str q15, [x0, #432]
tbl z30.h, {z20.h}, z5.h
tbl z14.h, {z21.h}, z7.h
tbl z8.h, {z24.h}, z5.h
tbl z15.h, {z25.h}, z7.h
str q2, [x0, #864]
add v5.8h, v29.8h, v9.8h
sub v29.8h, v29.8h, v9.8h
sub v7.8h, v11.8h, v6.8h
sub v2.8h, v10.8h, v27.8h
str q28, [x0, #1296]
add v28.8h, v11.8h, v6.8h
add v27.8h, v10.8h, v27.8h
tbl z9.h, {z16.h, z17.h}, z12.h
add v30.8h, v30.8h, v14.8h
str q3, [x0, #1728]
tbl z6.h, {z22.h}, z4.h
add v12.8h, v11.8h, v5.8h
sub v14.8h, v7.8h, v5.8h
sub v3.8h, v2.8h, v29.8h
str q31, [x0, #2160]
add v31.8h, v10.8h, v29.8h
tbl z29.h, {z18.h}, z1.h
tbl z11.h, {z26.h}, z4.h
add v7.8h, v8.8h, v15.8h
ld1b {z5.h}, p7/z, [x5, #2, mul vl]
add v2.8h, v0.8h, v6.8h
ld1b {z10.h}, p7/z, [x9, #5, mul vl]
add v1.8h, v9.8h, v29.8h
sub v29.8h, v9.8h, v29.8h
add v4.8h, v13.8h, v11.8h
ld1b {z8.h}, p7/z, [x4, #-1, mul vl]
add v15.8h, v30.8h, v2.8h
sub v30.8h, v30.8h, v2.8h
ld1b {z0.h}, p7/z, [x12, #0, mul vl]
add v6.8h, v7.8h, v4.8h
sub v9.8h, v7.8h, v4.8h
ld1b {z13.h}, p7/z, [x2, #4, mul vl]
sub v11.8h, v1.8h, v15.8h
sub v2.8h, v29.8h, v30.8h
add v7.8h, v1.8h, v15.8h
add v30.8h, v29.8h, v30.8h
ld1b {z4.h}, p7/z, [x7, #-6, mul vl]
add v15.8h, v1.8h, v6.8h
add v29.8h, v29.8h, v9.8h
str q28, [x0, #16]
tbl z28.h, {z19.h}, z5.h
sub v1.8h, v11.8h, v6.8h
sub v11.8h, v2.8h, v9.8h
tbl z6.h, {z23.h}, z5.h
str q12, [x0, #448]
tbl z2.h, {z20.h}, z10.h
tbl z9.h, {z24.h}, z10.h
str q14, [x0, #880]
tbl z5.h, {z21.h}, z8.h
tbl z12.h, {z25.h}, z8.h
str q3, [x0, #1312]
tbl z10.h, {z22.h}, z0.h
tbl z14.h, {z26.h}, z0.h
str q27, [x0, #1744]
tbl z27.h, {z16.h, z17.h}, z13.h
add v8.8h, v2.8h, v5.8h
add v3.8h, v9.8h, v12.8h
str q31, [x0, #2176]
tbl z31.h, {z18.h}, z4.h
add v28.8h, v28.8h, v10.8h
add v0.8h, v6.8h, v14.8h
ld1b {z13.h}, p7/z, [x5, #6, mul vl]
ld1b {z2.h}, p7/z, [x6, #-5, mul vl]
add v5.8h, v27.8h, v31.8h
sub v27.8h, v27.8h, v31.8h
add v31.8h, v8.8h, v28.8h
sub v28.8h, v8.8h, v28.8h
ld1b {z9.h}, p7/z, [x4, #3, mul vl]
add v12.8h, v3.8h, v0.8h
sub v4.8h, v3.8h, v0.8h
ld1b {z10.h}, p7/z, [x12, #4, mul vl]
sub v6.8h, v5.8h, v31.8h
sub v14.8h, v27.8h, v28.8h
add v31.8h, v5.8h, v31.8h
add v28.8h, v27.8h, v28.8h
ld1b {z8.h}, p7/z, [x3, #-8, mul vl]
add v3.8h, v5.8h, v12.8h
add v27.8h, v27.8h, v4.8h
ld1b {z0.h}, p7/z, [x7, #-2, mul vl]
sub v5.8h, v6.8h, v12.8h
sub v6.8h, v14.8h, v4.8h
str q7, [x0, #32]
tbl z12.h, {z19.h}, z13.h
tbl z14.h, {z23.h}, z13.h
str q15, [x0, #464]
tbl z4.h, {z20.h}, z2.h
tbl z7.h, {z24.h}, z2.h
str q1, [x0, #896]
tbl z13.h, {z21.h}, z9.h
tbl z15.h, {z25.h}, z9.h
str q11, [x0, #1328]
tbl z2.h, {z22.h}, z10.h
tbl z1.h, {z26.h}, z10.h
str q30, [x0, #1760]
tbl z30.h, {z16.h, z17.h}, z8.h
add v9.8h, v4.8h, v13.8h
add v11.8h, v7.8h, v15.8h
str q29, [x0, #2192]
tbl z29.h, {z18.h}, z0.h
add v10.8h, v12.8h, v2.8h
add v8.8h, v14.8h, v1.8h
ld1b {z4.h}, p7/z, [x2, #-7, mul vl]
ld1b {z13.h}, p7/z, [x6, #-1, mul vl]
add v7.8h, v30.8h, v29.8h
sub v30.8h, v30.8h, v29.8h
add v29.8h, v9.8h, v10.8h
sub v15.8h, v9.8h, v10.8h
ld1b {z0.h}, p7/z, [x4, #7, mul vl]
add v12.8h, v11.8h, v8.8h
sub v2.8h, v11.8h, v8.8h
ld1b {z14.h}, p7/z, [x13, #-8, mul vl]
sub v1.8h, v7.8h, v29.8h
sub v9.8h, v30.8h, v15.8h
add v29.8h, v7.8h, v29.8h
add v10.8h, v30.8h, v15.8h
ld1b {z11.h}, p7/z, [x3, #-4, mul vl]
add v8.8h, v7.8h, v12.8h
add v30.8h, v30.8h, v2.8h
ld1b {z15.h}, p7/z, [x7, #2, mul vl]
sub v7.8h, v1.8h, v12.8h
sub v1.8h, v9.8h, v2.8h
str q31, [x0, #48]
tbl z31.h, {z19.h}, z4.h
tbl z12.h, {z23.h}, z4.h
str q3, [x0, #480]
tbl z9.h, {z20.h}, z13.h
tbl z2.h, {z24.h}, z13.h
str q5, [x0, #912]
tbl z4.h, {z21.h}, z0.h
tbl z3.h, {z25.h}, z0.h
str q6, [x0, #1344]
tbl z13.h, {z22.h}, z14.h
tbl z5.h, {z26.h}, z14.h
str q28, [x0, #1776]
tbl z28.h, {z16.h, z17.h}, z11.h
add v0.8h, v9.8h, v4.8h
add v6.8h, v2.8h, v3.8h
str q27, [x0, #2208]
tbl z27.h, {z18.h}, z15.h
add v31.8h, v31.8h, v13.8h
add v14.8h, v12.8h, v5.8h
ld1b {z11.h}, p7/z, [x2, #-3, mul vl]
ld1b {z9.h}, p7/z, [x6, #3, mul vl]
add v4.8h, v28.8h, v27.8h
sub v28.8h, v28.8h, v27.8h
add v27.8h, v0.8h, v31.8h
sub v31.8h, v0.8h, v31.8h
ld1b {z2.h}, p7/z, [x5, #-5, mul vl]
add v3.8h, v6.8h, v14.8h
sub v15.8h, v6.8h, v14.8h
ld1b {z13.h}, p7/z, [x13, #-4, mul vl]
sub v12.8h, v4.8h, v27.8h
sub v5.8h, v28.8h, v31.8h
add v27.8h, v4.8h, v27.8h
add v31.8h, v28.8h, v31.8h
ld1b {z0.h}, p7/z, [x3, #0, mul vl]
add v6.8h, v4.8h, v3.8h
add v28.8h, v28.8h, v15.8h
ld1b {z14.h}, p7/z, [x7, #6, mul vl]
sub v4.8h, v12.8h, v3.8h
sub v12.8h, v5.8h, v15.8h
str q29, [x0, #64]
tbl z29.h, {z19.h}, z11.h
tbl z3.h, {z23.h}, z11.h
str q8, [x0, #496]
tbl z5.h, {z20.h}, z9.h
tbl z15.h, {z24.h}, z9.h
str q7, [x0, #928]
tbl z11.h, {z21.h}, z2.h
tbl z8.h, {z25.h}, z2.h
str q1, [x0, #1360]
tbl z9.h, {z22.h}, z13.h
tbl z7.h, {z26.h}, z13.h
str q10, [x0, #1792]
tbl z2.h, {z16.h, z17.h}, z0.h
add v1.8h, v5.8h, v11.8h
add v13.8h, v15.8h, v8.8h
str q30, [x0, #2224]
tbl z30.h, {z18.h}, z14.h
add v29.8h, v29.8h, v9.8h
add v10.8h, v3.8h, v7.8h
ld1b {z0.h}, p7/z, [x2, #1, mul vl]
ld1b {z5.h}, p7/z, [x6, #7, mul vl]
add v11.8h, v2.8h, v30.8h
sub v30.8h, v2.8h, v30.8h
add v15.8h, v1.8h, v29.8h
sub v29.8h, v1.8h, v29.8h
ld1b {z8.h}, p7/z, [x5, #-1, mul vl]
add v14.8h, v13.8h, v10.8h
sub v9.8h, v13.8h, v10.8h
ld1b {z3.h}, p7/z, [x13, #0, mul vl]
sub v7.8h, v11.8h, v15.8h
sub v2.8h, v30.8h, v29.8h
add v1.8h, v11.8h, v15.8h
add v29.8h, v30.8h, v29.8h
ld1b {z13.h}, p7/z, [x3, #4, mul vl]
add v10.8h, v11.8h, v14.8h
add v30.8h, v30.8h, v9.8h
ld1b {z15.h}, p7/z, [x8, #-6, mul vl]
sub v11.8h, v7.8h, v14.8h
sub v7.8h, v2.8h, v9.8h
str q27, [x0, #80]
tbl z27.h, {z19.h}, z0.h
tbl z14.h, {z23.h}, z0.h
str q6, [x0, #512]
tbl z2.h, {z20.h}, z5.h
tbl z9.h, {z24.h}, z5.h
str q4, [x0, #944]
tbl z0.h, {z21.h}, z8.h
tbl z6.h, {z25.h}, z8.h
str q12, [x0, #1376]
tbl z5.h, {z22.h}, z3.h
tbl z4.h, {z26.h}, z3.h
str q31, [x0, #1808]
tbl z31.h, {z16.h, z17.h}, z13.h
add v8.8h, v2.8h, v0.8h
add v12.8h, v9.8h, v6.8h
str q28, [x0, #2240]
tbl z28.h, {z18.h}, z15.h
add v27.8h, v27.8h, v5.8h
add v3.8h, v14.8h, v4.8h
ld1b {z13.h}, p7/z, [x2, #5, mul vl]
ld1b {z2.h}, p7/z, [x7, #-5, mul vl]
add v0.8h, v31.8h, v28.8h
sub v31.8h, v31.8h, v28.8h
add v28.8h, v8.8h, v27.8h
sub v27.8h, v8.8h, v27.8h
ld1b {z9.h}, p7/z, [x5, #3, mul vl]
add v6.8h, v12.8h, v3.8h
sub v15.8h, v12.8h, v3.8h
ld1b {z5.h}, p7/z, [x13, #4, mul vl]
sub v14.8h, v0.8h, v28.8h
sub v4.8h, v31.8h, v27.8h
add v28.8h, v0.8h, v28.8h
add v27.8h, v31.8h, v27.8h
ld1b {z8.h}, p7/z, [x4, #-8, mul vl]
add v12.8h, v0.8h, v6.8h
add v31.8h, v31.8h, v15.8h
ld1b {z3.h}, p7/z, [x8, #-2, mul vl]
sub v0.8h, v14.8h, v6.8h
sub v14.8h, v4.8h, v15.8h
str q1, [x0, #96]
tbl z6.h, {z19.h}, z13.h
tbl z4.h, {z23.h}, z13.h
str q10, [x0, #528]
tbl z15.h, {z20.h}, z2.h
tbl z1.h, {z24.h}, z2.h
str q11, [x0, #960]
tbl z13.h, {z21.h}, z9.h
tbl z10.h, {z25.h}, z9.h
str q7, [x0, #1392]
tbl z2.h, {z22.h}, z5.h
tbl z11.h, {z26.h}, z5.h
str q29, [x0, #1824]
tbl z29.h, {z16.h, z17.h}, z8.h
add v9.8h, v15.8h, v13.8h
add v7.8h, v1.8h, v10.8h
str q30, [x0, #2256]
tbl z30.h, {z18.h}, z3.h
add v5.8h, v6.8h, v2.8h
add v8.8h, v4.8h, v11.8h
ld1b {z15.h}, p7/z, [x3, #-7, mul vl]
ld1b {z13.h}, p7/z, [x7, #-1, mul vl]
add v1.8h, v29.8h, v30.8h
sub v29.8h, v29.8h, v30.8h
add v30.8h, v9.8h, v5.8h
sub v10.8h, v9.8h, v5.8h
ld1b {z3.h}, p7/z, [x5, #7, mul vl]
add v6.8h, v7.8h, v8.8h
sub v2.8h, v7.8h, v8.8h
ld1b {z4.h}, p7/z, [x10, #-6, mul vl]
sub v11.8h, v1.8h, v30.8h
sub v9.8h, v29.8h, v10.8h
add v30.8h, v1.8h, v30.8h
add v5.8h, v29.8h, v10.8h
ld1b {z7.h}, p7/z, [x4, #-4, mul vl]
add v8.8h, v1.8h, v6.8h
add v29.8h, v29.8h, v2.8h
ld1b {z10.h}, p7/z, [x8, #2, mul vl]
sub v1.8h, v11.8h, v6.8h
sub v11.8h, v9.8h, v2.8h
str q28, [x0, #112]
tbl z28.h, {z19.h}, z15.h
tbl z6.h, {z23.h}, z15.h
str q12, [x0, #544]
tbl z9.h, {z20.h}, z13.h
tbl z2.h, {z24.h}, z13.h
str q0, [x0, #976]
tbl z15.h, {z21.h}, z3.h
tbl z12.h, {z25.h}, z3.h
str q14, [x0, #1408]
tbl z13.h, {z22.h}, z4.h
tbl z0.h, {z26.h}, z4.h
str q27, [x0, #1840]
tbl z27.h, {z16.h, z17.h}, z7.h
add v3.8h, v9.8h, v15.8h
add v14.8h, v2.8h, v12.8h
str q31, [x0, #2272]
tbl z31.h, {z18.h}, z10.h
add v28.8h, v28.8h, v13.8h
add v4.8h, v6.8h, v0.8h
ld1b {z7.h}, p7/z, [x3, #-3, mul vl]
ld1b {z9.h}, p7/z, [x7, #3, mul vl]
add v15.8h, v27.8h, v31.8h
sub v27.8h, v27.8h, v31.8h
add v31.8h, v3.8h, v28.8h
sub v28.8h, v3.8h, v28.8h
ld1b {z2.h}, p7/z, [x6, #-8, mul vl]
add v12.8h, v14.8h, v4.8h
sub v10.8h, v14.8h, v4.8h
ld1b {z13.h}, p7/z, [x10, #-2, mul vl]
sub v6.8h, v15.8h, v31.8h
sub v0.8h, v27.8h, v28.8h
add v31.8h, v15.8h, v31.8h
add v28.8h, v27.8h, v28.8h
ld1b {z3.h}, p7/z, [x4, #0, mul vl]
add v14.8h, v15.8h, v12.8h
add v27.8h, v27.8h, v10.8h
ld1b {z4.h}, p7/z, [x8, #6, mul vl]
sub v15.8h, v6.8h, v12.8h
sub v6.8h, v0.8h, v10.8h
str q30, [x0, #128]
tbl z30.h, {z19.h}, z7.h
tbl z12.h, {z23.h}, z7.h
str q8, [x0, #560]
tbl z0.h, {z20.h}, z9.h
tbl z10.h, {z24.h}, z9.h
str q1, [x0, #992]
tbl z7.h, {z21.h}, z2.h
tbl z8.h, {z25.h}, z2.h
str q11, [x0, #1424]
tbl z9.h, {z22.h}, z13.h
tbl z1.h, {z26.h}, z13.h
str q5, [x0, #1856]
tbl z2.h, {z16.h, z17.h}, z3.h
add v11.8h, v0.8h, v7.8h
add v13.8h, v10.8h, v8.8h
str q29, [x0, #2288]
tbl z29.h, {z18.h}, z4.h
add v30.8h, v30.8h, v9.8h
add v5.8h, v12.8h, v1.8h
ld1b {z3.h}, p7/z, [x3, #1, mul vl]
ld1b {z0.h}, p7/z, [x7, #7, mul vl]
add v7.8h, v2.8h, v29.8h
sub v29.8h, v2.8h, v29.8h
add v10.8h, v11.8h, v30.8h
sub v30.8h, v11.8h, v30.8h
ld1b {z8.h}, p7/z, [x6, #-4, mul vl]
add v4.8h, v13.8h, v5.8h
sub v9.8h, v13.8h, v5.8h
ld1b {z12.h}, p7/z, [x10, #2, mul vl]
sub v1.8h, v7.8h, v10.8h
sub v2.8h, v29.8h, v30.8h
add v11.8h, v7.8h, v10.8h
add v30.8h, v29.8h, v30.8h
ld1b {z13.h}, p7/z, [x4, #4, mul vl]
add v5.8h, v7.8h, v4.8h
add v29.8h, v29.8h, v9.8h
ld1b {z10.h}, p7/z, [x9, #-6, mul vl]
sub v7.8h, v1.8h, v4.8h
sub v1.8h, v2.8h, v9.8h
str q31, [x0, #144]
tbl z31.h, {z19.h}, z3.h
tbl z4.h, {z23.h}, z3.h
str q14, [x0, #576]
tbl z2.h, {z20.h}, z0.h
tbl z9.h, {z24.h}, z0.h
str q15, [x0, #1008]
tbl z3.h, {z21.h}, z8.h
tbl z14.h, {z25.h}, z8.h
str q6, [x0, #1440]
tbl z0.h, {z22.h}, z12.h
tbl z15.h, {z26.h}, z12.h
str q28, [x0, #1872]
tbl z28.h, {z16.h, z17.h}, z13.h
add v8.8h, v2.8h, v3.8h
add v6.8h, v9.8h, v14.8h
str q27, [x0, #2304]
tbl z27.h, {z18.h}, z10.h
add v31.8h, v31.8h, v0.8h
add v12.8h, v4.8h, v15.8h
ld1b {z13.h}, p7/z, [x3, #5, mul vl]
ld1b {z2.h}, p7/z, [x8, #-5, mul vl]
add v3.8h, v28.8h, v27.8h
sub v28.8h, v28.8h, v27.8h
add v27.8h, v8.8h, v31.8h
sub v31.8h, v8.8h, v31.8h
ld1b {z9.h}, p7/z, [x6, #0, mul vl]
add v14.8h, v6.8h, v12.8h
sub v10.8h, v6.8h, v12.8h
ld1b {z0.h}, p7/z, [x10, #6, mul vl]
sub v4.8h, v3.8h, v27.8h
sub v15.8h, v28.8h, v31.8h
add v27.8h, v3.8h, v27.8h
add v31.8h, v28.8h, v31.8h
ld1b {z8.h}, p7/z, [x5, #-8, mul vl]
add v6.8h, v3.8h, v14.8h
add v28.8h, v28.8h, v10.8h
ld1b {z12.h}, p7/z, [x9, #-2, mul vl]
sub v3.8h, v4.8h, v14.8h
sub v4.8h, v15.8h, v10.8h
str q11, [x0, #160]
tbl z14.h, {z19.h}, z13.h
tbl z15.h, {z23.h}, z13.h
str q5, [x0, #592]
tbl z10.h, {z20.h}, z2.h
tbl z11.h, {z24.h}, z2.h
str q7, [x0, #1024]
tbl z13.h, {z21.h}, z9.h
tbl z5.h, {z25.h}, z9.h
str q1, [x0, #1456]
tbl z2.h, {z22.h}, z0.h
tbl z7.h, {z26.h}, z0.h
str q30, [x0, #1888]
tbl z30.h, {z16.h, z17.h}, z8.h
add v9.8h, v10.8h, v13.8h
add v1.8h, v11.8h, v5.8h
str q29, [x0, #2320]
tbl z29.h, {z18.h}, z12.h
add v0.8h, v14.8h, v2.8h
add v8.8h, v15.8h, v7.8h
ld1b {z10.h}, p7/z, [x4, #-7, mul vl]
ld1b {z13.h}, p7/z, [x8, #-1, mul vl]
add v11.8h, v30.8h, v29.8h
sub v30.8h, v30.8h, v29.8h
add v29.8h, v9.8h, v0.8h
sub v5.8h, v9.8h, v0.8h
ld1b {z12.h}, p7/z, [x6, #4, mul vl]
add v14.8h, v1.8h, v8.8h
sub v2.8h, v1.8h, v8.8h
ld1b {z15.h}, p7/z, [x11, #-6, mul vl]
sub v7.8h, v11.8h, v29.8h
sub v9.8h, v30.8h, v5.8h
add v29.8h, v11.8h, v29.8h
add v0.8h, v30.8h, v5.8h
ld1b {z1.h}, p7/z, [x5, #-4, mul vl]
add v8.8h, v11.8h, v14.8h
add v30.8h, v30.8h, v2.8h
ld1b {z5.h}, p7/z, [x9, #2, mul vl]
sub v11.8h, v7.8h, v14.8h
sub v7.8h, v9.8h, v2.8h
str q27, [x0, #176]
tbl z27.h, {z19.h}, z10.h
tbl z14.h, {z23.h}, z10.h
str q6, [x0, #608]
tbl z9.h, {z20.h}, z13.h
tbl z2.h, {z24.h}, z13.h
str q3, [x0, #1040]
tbl z10.h, {z21.h}, z12.h
tbl z6.h, {z25.h}, z12.h
str q4, [x0, #1472]
tbl z13.h, {z22.h}, z15.h
tbl z3.h, {z26.h}, z15.h
str q31, [x0, #1904]
tbl z31.h, {z16.h, z17.h}, z1.h
add v12.8h, v9.8h, v10.8h
add v4.8h, v2.8h, v6.8h
str q28, [x0, #2336]
tbl z28.h, {z18.h}, z5.h
add v27.8h, v27.8h, v13.8h
add v15.8h, v14.8h, v3.8h
ld1b {z1.h}, p7/z, [x4, #-3, mul vl]
ld1b {z9.h}, p7/z, [x8, #3, mul vl]
add v10.8h, v31.8h, v28.8h
sub v31.8h, v31.8h, v28.8h
add v28.8h, v12.8h, v27.8h
sub v27.8h, v12.8h, v27.8h
ld1b {z2.h}, p7/z, [x7, #-8, mul vl]
add v6.8h, v4.8h, v15.8h
sub v5.8h, v4.8h, v15.8h
ld1b {z13.h}, p7/z, [x11, #-2, mul vl]
sub v14.8h, v10.8h, v28.8h
sub v3.8h, v31.8h, v27.8h
add v28.8h, v10.8h, v28.8h
add v27.8h, v31.8h, v27.8h
ld1b {z12.h}, p7/z, [x5, #0, mul vl]
add v4.8h, v10.8h, v6.8h
add v31.8h, v31.8h, v5.8h
ld1b {z15.h}, p7/z, [x9, #6, mul vl]
sub v10.8h, v14.8h, v6.8h
sub v14.8h, v3.8h, v5.8h
str q29, [x0, #192]
tbl z29.h, {z19.h}, z1.h
tbl z6.h, {z23.h}, z1.h
str q8, [x0, #624]
tbl z3.h, {z20.h}, z9.h
tbl z5.h, {z24.h}, z9.h
str q11, [x0, #1056]
tbl z1.h, {z21.h}, z2.h
tbl z8.h, {z25.h}, z2.h
str q7, [x0, #1488]
tbl z9.h, {z22.h}, z13.h
tbl z11.h, {z26.h}, z13.h
str q0, [x0, #1920]
tbl z2.h, {z16.h, z17.h}, z12.h
add v7.8h, v3.8h, v1.8h
add v13.8h, v5.8h, v8.8h
str q30, [x0, #2352]
tbl z30.h, {z18.h}, z15.h
add v29.8h, v29.8h, v9.8h
add v0.8h, v6.8h, v11.8h
ld1b {z12.h}, p7/z, [x4, #1, mul vl]
ld1b {z3.h}, p7/z, [x8, #7, mul vl]
add v1.8h, v2.8h, v30.8h
sub v30.8h, v2.8h, v30.8h
add v5.8h, v7.8h, v29.8h
sub v29.8h, v7.8h, v29.8h
ld1b {z8.h}, p7/z, [x7, #-4, mul vl]
add v15.8h, v13.8h, v0.8h
sub v9.8h, v13.8h, v0.8h
ld1b {z6.h}, p7/z, [x11, #2, mul vl]
sub v11.8h, v1.8h, v5.8h
sub v2.8h, v30.8h, v29.8h
add v7.8h, v1.8h, v5.8h
add v29.8h, v30.8h, v29.8h
ld1b {z13.h}, p7/z, [x5, #4, mul vl]
add v0.8h, v1.8h, v15.8h
add v30.8h, v30.8h, v9.8h
ld1b {z5.h}, p7/z, [x6, #-7, mul vl]
sub v1.8h, v11.8h, v15.8h
sub v11.8h, v2.8h, v9.8h
str q28, [x0, #208]
tbl z28.h, {z19.h}, z12.h
tbl z15.h, {z23.h}, z12.h
str q4, [x0, #640]
tbl z2.h, {z20.h}, z3.h
tbl z9.h, {z24.h}, z3.h
str q10, [x0, #1072]
tbl z12.h, {z21.h}, z8.h
tbl z4.h, {z25.h}, z8.h
str q14, [x0, #1504]
tbl z3.h, {z22.h}, z6.h
tbl z10.h, {z26.h}, z6.h
str q27, [x0, #1936]
tbl z27.h, {z16.h, z17.h}, z13.h
add v8.8h, v2.8h, v12.8h
add v14.8h, v9.8h, v4.8h
str q31, [x0, #2368]
tbl z31.h, {z18.h}, z5.h
add v28.8h, v28.8h, v3.8h
add v6.8h, v15.8h, v10.8h
ld1b {z13.h}, p7/z, [x4, #5, mul vl]
ld1b {z2.h}, p7/z, [x9, #-5, mul vl]
add v12.8h, v27.8h, v31.8h
sub v27.8h, v27.8h, v31.8h
add v31.8h, v8.8h, v28.8h
sub v28.8h, v8.8h, v28.8h
ld1b {z9.h}, p7/z, [x7, #0, mul vl]
add v4.8h, v14.8h, v6.8h
sub v5.8h, v14.8h, v6.8h
ld1b {z3.h}, p7/z, [x11, #6, mul vl]
sub v15.8h, v12.8h, v31.8h
sub v10.8h, v27.8h, v28.8h
add v31.8h, v12.8h, v31.8h
add v28.8h, v27.8h, v28.8h
ld1b {z8.h}, p7/z, [x2, #-6, mul vl]
add v14.8h, v12.8h, v4.8h
add v27.8h, v27.8h, v5.8h
ld1b {z6.h}, p7/z, [x6, #-3, mul vl]
sub v12.8h, v15.8h, v4.8h
sub v15.8h, v10.8h, v5.8h
str q7, [x0, #224]
tbl z4.h, {z19.h}, z13.h
tbl z10.h, {z23.h}, z13.h
str q0, [x0, #656]
tbl z5.h, {z20.h}, z2.h
tbl z7.h, {z24.h}, z2.h
str q1, [x0, #1088]
tbl z13.h, {z21.h}, z9.h
tbl z0.h, {z25.h}, z9.h
str q11, [x0, #1520]
tbl z2.h, {z22.h}, z3.h
tbl z1.h, {z26.h}, z3.h
str q29, [x0, #1952]
tbl z29.h, {z16.h, z17.h}, z8.h
add v9.8h, v5.8h, v13.8h
add v11.8h, v7.8h, v0.8h
str q30, [x0, #2384]
tbl z30.h, {z18.h}, z6.h
add v3.8h, v4.8h, v2.8h
add v8.8h, v10.8h, v1.8h
ld1b {z5.h}, p7/z, [x5, #-7, mul vl]
ld1b {z13.h}, p7/z, [x9, #-1, mul vl]
add v7.8h, v29.8h, v30.8h
sub v29.8h, v29.8h, v30.8h
add v30.8h, v9.8h, v3.8h
sub v0.8h, v9.8h, v3.8h
ld1b {z6.h}, p7/z, [x7, #4, mul vl]
add v4.8h, v11.8h, v8.8h
sub v2.8h, v11.8h, v8.8h
ld1b {z10.h}, p7/z, [x12, #-6, mul vl]
sub v1.8h, v7.8h, v30.8h
sub v9.8h, v29.8h, v0.8h
add v30.8h, v7.8h, v30.8h
add v3.8h, v29.8h, v0.8h
ld1b {z11.h}, p7/z, [x2, #-2, mul vl]
add v8.8h, v7.8h, v4.8h
add v29.8h, v29.8h, v2.8h
ld1b {z0.h}, p7/z, [x6, #1, mul vl]
sub v7.8h, v1.8h, v4.8h
sub v1.8h, v9.8h, v2.8h
str q31, [x0, #240]
tbl z31.h, {z19.h}, z5.h
tbl z4.h, {z23.h}, z5.h
str q14, [x0, #672]
tbl z9.h, {z20.h}, z13.h
tbl z2.h, {z24.h}, z13.h
str q12, [x0, #1104]
tbl z5.h, {z21.h}, z6.h
tbl z14.h, {z25.h}, z6.h
str q15, [x0, #1536]
tbl z13.h, {z22.h}, z10.h
tbl z12.h, {z26.h}, z10.h
str q28, [x0, #1968]
tbl z28.h, {z16.h, z17.h}, z11.h
add v6.8h, v9.8h, v5.8h
add v15.8h, v2.8h, v14.8h
str q27, [x0, #2400]
tbl z27.h, {z18.h}, z0.h
add v31.8h, v31.8h, v13.8h
add v10.8h, v4.8h, v12.8h
ld1b {z11.h}, p7/z, [x5, #-3, mul vl]
ld1b {z9.h}, p7/z, [x9, #3, mul vl]
add v5.8h, v28.8h, v27.8h
sub v28.8h, v28.8h, v27.8h
add v27.8h, v6.8h, v31.8h
sub v31.8h, v6.8h, v31.8h
ld1b {z2.h}, p7/z, [x8, #-8, mul vl]
add v14.8h, v15.8h, v10.8h
sub v0.8h, v15.8h, v10.8h
ld1b {z13.h}, p7/z, [x12, #-2, mul vl]
sub v4.8h, v5.8h, v27.8h
sub v12.8h, v28.8h, v31.8h
add v27.8h, v5.8h, v27.8h
add v31.8h, v28.8h, v31.8h
ld1b {z6.h}, p7/z, [x2, #2, mul vl]
add v15.8h, v5.8h, v14.8h
add v28.8h, v28.8h, v0.8h
ld1b {z10.h}, p7/z, [x6, #5, mul vl]
sub v5.8h, v4.8h, v14.8h
sub v4.8h, v12.8h, v0.8h
str q30, [x0, #256]
tbl z30.h, {z19.h}, z11.h
tbl z14.h, {z23.h}, z11.h
str q8, [x0, #688]
tbl z12.h, {z20.h}, z9.h
tbl z0.h, {z24.h}, z9.h
str q7, [x0, #1120]
tbl z11.h, {z21.h}, z2.h
tbl z8.h, {z25.h}, z2.h
str q1, [x0, #1552]
tbl z9.h, {z22.h}, z13.h
tbl z7.h, {z26.h}, z13.h
str q3, [x0, #1984]
tbl z2.h, {z16.h, z17.h}, z6.h
add v1.8h, v12.8h, v11.8h
add v13.8h, v0.8h, v8.8h
str q29, [x0, #2416]
tbl z29.h, {z18.h}, z10.h
add v30.8h, v30.8h, v9.8h
add v3.8h, v14.8h, v7.8h
ld1b {z6.h}, p7/z, [x5, #1, mul vl]
ld1b {z12.h}, p7/z, [x9, #7, mul vl]
add v11.8h, v2.8h, v29.8h
sub v29.8h, v2.8h, v29.8h
add v0.8h, v1.8h, v30.8h
sub v30.8h, v1.8h, v30.8h
ld1b {z8.h}, p7/z, [x8, #-4, mul vl]
add v10.8h, v13.8h, v3.8h
sub v9.8h, v13.8h, v3.8h
ld1b {z14.h}, p7/z, [x12, #2, mul vl]
sub v7.8h, v11.8h, v0.8h
sub v2.8h, v29.8h, v30.8h
add v1.8h, v11.8h, v0.8h
add v30.8h, v29.8h, v30.8h
ld1b {z13.h}, p7/z, [x2, #6, mul vl]
add v3.8h, v11.8h, v10.8h
add v29.8h, v29.8h, v9.8h
ld1b {z0.h}, p7/z, [x7, #-7, mul vl]
sub v11.8h, v7.8h, v10.8h
sub v7.8h, v2.8h, v9.8h
str q27, [x0, #272]
tbl z27.h, {z19.h}, z6.h
tbl z10.h, {z23.h}, z6.h
str q15, [x0, #704]
tbl z2.h, {z20.h}, z12.h
tbl z9.h, {z24.h}, z12.h
str q5, [x0, #1136]
tbl z6.h, {z21.h}, z8.h
tbl z15.h, {z25.h}, z8.h
str q4, [x0, #1568]
tbl z12.h, {z22.h}, z14.h
tbl z5.h, {z26.h}, z14.h
str q31, [x0, #2000]
tbl z31.h, {z16.h, z17.h}, z13.h
add v8.8h, v2.8h, v6.8h
add v4.8h, v9.8h, v15.8h
str q28, [x0, #2432]
tbl z28.h, {z18.h}, z0.h
add v27.8h, v27.8h, v12.8h
add v14.8h, v10.8h, v5.8h
ld1b {z13.h}, p7/z, [x5, #5, mul vl]
ld1b {z2.h}, p7/z, [x10, #-8, mul vl]
add v6.8h, v31.8h, v28.8h
sub v31.8h, v31.8h, v28.8h
add v28.8h, v8.8h, v27.8h
sub v27.8h, v8.8h, v27.8h
ld1b {z9.h}, p7/z, [x8, #0, mul vl]
add v15.8h, v4.8h, v14.8h
sub v0.8h, v4.8h, v14.8h
ld1b {z12.h}, p7/z, [x12, #6, mul vl]
sub v10.8h, v6.8h, v28.8h
sub v5.8h, v31.8h, v27.8h
add v28.8h, v6.8h, v28.8h
add v27.8h, v31.8h, v27.8h
ld1b {z8.h}, p7/z, [x3, #-6, mul vl]
add v4.8h, v6.8h, v15.8h
add v31.8h, v31.8h, v0.8h
ld1b {z14.h}, p7/z, [x7, #-3, mul vl]
sub v6.8h, v10.8h, v15.8h
sub v10.8h, v5.8h, v0.8h
str q1, [x0, #288]
tbl z15.h, {z19.h}, z13.h
tbl z5.h, {z23.h}, z13.h
str q3, [x0, #720]
tbl z0.h, {z20.h}, z2.h
tbl z1.h, {z24.h}, z2.h
str q11, [x0, #1152]
tbl z13.h, {z21.h}, z9.h
tbl z3.h, {z25.h}, z9.h
str q7, [x0, #1584]
tbl z2.h, {z22.h}, z12.h
tbl z11.h, {z26.h}, z12.h
str q30, [x0, #2016]
tbl z30.h, {z16.h, z17.h}, z8.h
add v9.8h, v0.8h, v13.8h
add v7.8h, v1.8h, v3.8h
str q29, [x0, #2448]
tbl z29.h, {z18.h}, z14.h
add v12.8h, v15.8h, v2.8h
add v8.8h, v5.8h, v11.8h
ld1b {z0.h}, p7/z, [x2, #-5, mul vl]
ld1b {z13.h}, p7/z, [x10, #-4, mul vl]
add v1.8h, v30.8h, v29.8h
sub v30.8h, v30.8h, v29.8h
add v29.8h, v9.8h, v12.8h
sub v3.8h, v9.8h, v12.8h
ld1b {z14.h}, p7/z, [x8, #4, mul vl]
add v15.8h, v7.8h, v8.8h
sub v2.8h, v7.8h, v8.8h
ld1b {z5.h}, p7/z, [x13, #-6, mul vl]
sub v11.8h, v1.8h, v29.8h
sub v9.8h, v30.8h, v3.8h
add v29.8h, v1.8h, v29.8h
add v12.8h, v30.8h, v3.8h
ld1b {z7.h}, p7/z, [x3, #-2, mul vl]
add v8.8h, v1.8h, v15.8h
add v30.8h, v30.8h, v2.8h
ld1b {z3.h}, p7/z, [x7, #1, mul vl]
sub v1.8h, v11.8h, v15.8h
sub v11.8h, v9.8h, v2.8h
str q28, [x0, #304]
tbl z28.h, {z19.h}, z0.h
tbl z15.h, {z23.h}, z0.h
str q4, [x0, #736]
tbl z9.h, {z20.h}, z13.h
tbl z2.h, {z24.h}, z13.h
str q6, [x0, #1168]
tbl z0.h, {z21.h}, z14.h
tbl z4.h, {z25.h}, z14.h
str q10, [x0, #1600]
tbl z13.h, {z22.h}, z5.h
tbl z6.h, {z26.h}, z5.h
str q27, [x0, #2032]
tbl z27.h, {z16.h, z17.h}, z7.h
add v14.8h, v9.8h, v0.8h
add v10.8h, v2.8h, v4.8h
str q31, [x0, #2464]
tbl z31.h, {z18.h}, z3.h
add v28.8h, v28.8h, v13.8h
add v5.8h, v15.8h, v6.8h
ld1b {z7.h}, p7/z, [x2, #-1, mul vl]
ld1b {z9.h}, p7/z, [x10, #0, mul vl]
add v0.8h, v27.8h, v31.8h
sub v27.8h, v27.8h, v31.8h
add v31.8h, v14.8h, v28.8h
sub v28.8h, v14.8h, v28.8h
ld1b {z2.h}, p7/z, [x9, #-8, mul vl]
add v4.8h, v10.8h, v5.8h
sub v3.8h, v10.8h, v5.8h
ld1b {z13.h}, p7/z, [x13, #-2, mul vl]
sub v15.8h, v0.8h, v31.8h
sub v6.8h, v27.8h, v28.8h
add v31.8h, v0.8h, v31.8h
add v28.8h, v27.8h, v28.8h
ld1b {z14.h}, p7/z, [x3, #2, mul vl]
add v10.8h, v0.8h, v4.8h
add v27.8h, v27.8h, v3.8h
ld1b {z5.h}, p7/z, [x7, #5, mul vl]
sub v0.8h, v15.8h, v4.8h
sub v15.8h, v6.8h, v3.8h
str q29, [x0, #320]
tbl z29.h, {z19.h}, z7.h
tbl z4.h, {z23.h}, z7.h
str q8, [x0, #752]
tbl z6.h, {z20.h}, z9.h
tbl z3.h, {z24.h}, z9.h
str q1, [x0, #1184]
tbl z7.h, {z21.h}, z2.h
tbl z8.h, {z25.h}, z2.h
str q11, [x0, #1616]
tbl z9.h, {z22.h}, z13.h
tbl z1.h, {z26.h}, z13.h
str q12, [x0, #2048]
tbl z2.h, {z16.h, z17.h}, z14.h
add v11.8h, v6.8h, v7.8h
add v13.8h, v3.8h, v8.8h
str q30, [x0, #2480]
tbl z30.h, {z18.h}, z5.h
add v29.8h, v29.8h, v9.8h
add v12.8h, v4.8h, v1.8h
ld1b {z14.h}, p7/z, [x2, #3, mul vl]
ld1b {z6.h}, p7/z, [x10, #4, mul vl]
add v7.8h, v2.8h, v30.8h
sub v30.8h, v2.8h, v30.8h
add v3.8h, v11.8h, v29.8h
sub v29.8h, v11.8h, v29.8h
ld1b {z8.h}, p7/z, [x9, #-4, mul vl]
add v5.8h, v13.8h, v12.8h
sub v9.8h, v13.8h, v12.8h
ld1b {z4.h}, p7/z, [x13, #2, mul vl]
sub v1.8h, v7.8h, v3.8h
sub v2.8h, v30.8h, v29.8h
add v11.8h, v7.8h, v3.8h
add v29.8h, v30.8h, v29.8h
ld1b {z13.h}, p7/z, [x3, #6, mul vl]
add v12.8h, v7.8h, v5.8h
add v30.8h, v30.8h, v9.8h
ld1b {z3.h}, p7/z, [x8, #-7, mul vl]
sub v7.8h, v1.8h, v5.8h
sub v1.8h, v2.8h, v9.8h
str q31, [x0, #336]
tbl z31.h, {z19.h}, z14.h
tbl z5.h, {z23.h}, z14.h
str q10, [x0, #768]
tbl z2.h, {z20.h}, z6.h
tbl z9.h, {z24.h}, z6.h
str q0, [x0, #1200]
tbl z14.h, {z21.h}, z8.h
tbl z10.h, {z25.h}, z8.h
str q15, [x0, #1632]
tbl z6.h, {z22.h}, z4.h
tbl z0.h, {z26.h}, z4.h
str q28, [x0, #2064]
tbl z28.h, {z16.h, z17.h}, z13.h
add v8.8h, v2.8h, v14.8h
add v15.8h, v9.8h, v10.8h
str q27, [x0, #2496]
tbl z27.h, {z18.h}, z3.h
add v31.8h, v31.8h, v6.8h
add v4.8h, v5.8h, v0.8h
ld1b {z13.h}, p7/z, [x2, #7, mul vl]
ld1b {z2.h}, p7/z, [x11, #-8, mul vl]
add v14.8h, v28.8h, v27.8h
sub v28.8h, v28.8h, v27.8h
add v27.8h, v8.8h, v31.8h
sub v31.8h, v8.8h, v31.8h
ld1b {z9.h}, p7/z, [x9, #0, mul vl]
add v10.8h, v15.8h, v4.8h
sub v3.8h, v15.8h, v4.8h
ld1b {z6.h}, p7/z, [x13, #6, mul vl]
sub v5.8h, v14.8h, v27.8h
sub v0.8h, v28.8h, v31.8h
add v27.8h, v14.8h, v27.8h
add v31.8h, v28.8h, v31.8h
ld1b {z8.h}, p7/z, [x4, #-6, mul vl]
add v15.8h, v14.8h, v10.8h
add v28.8h, v28.8h, v3.8h
ld1b {z4.h}, p7/z, [x8, #-3, mul vl]
sub v14.8h, v5.8h, v10.8h
sub v5.8h, v0.8h, v3.8h
str q11, [x0, #352]
tbl z10.h, {z19.h}, z13.h
tbl z0.h, {z23.h}, z13.h
str q12, [x0, #784]
tbl z3.h, {z20.h}, z2.h
tbl z11.h, {z24.h}, z2.h
str q7, [x0, #1216]
tbl z13.h, {z21.h}, z9.h
tbl z12.h, {z25.h}, z9.h
str q1, [x0, #1648]
tbl z2.h, {z22.h}, z6.h
tbl z7.h, {z26.h}, z6.h
str q29, [x0, #2080]
tbl z29.h, {z16.h, z17.h}, z8.h
add v9.8h, v3.8h, v13.8h
add v1.8h, v11.8h, v12.8h
str q30, [x0, #2512]
tbl z30.h, {z18.h}, z4.h
add v6.8h, v10.8h, v2.8h
add v8.8h, v0.8h, v7.8h
ld1b {z3.h}, p7/z, [x3, #-5, mul vl]
ld1b {z13.h}, p7/z, [x11, #-4, mul vl]
add v11.8h, v29.8h, v30.8h
sub v29.8h, v29.8h, v30.8h
add v30.8h, v9.8h, v6.8h
sub v12.8h, v9.8h, v6.8h
ld1b {z4.h}, p7/z, [x9, #4, mul vl]
add v10.8h, v1.8h, v8.8h
sub v2.8h, v1.8h, v8.8h
ld1b {z0.h}, p7/z, [x10, #-7, mul vl]
sub v7.8h, v11.8h, v30.8h
sub v9.8h, v29.8h, v12.8h
add v30.8h, v11.8h, v30.8h
add v6.8h, v29.8h, v12.8h
ld1b {z1.h}, p7/z, [x4, #-2, mul vl]
add v8.8h, v11.8h, v10.8h
add v29.8h, v29.8h, v2.8h
ld1b {z12.h}, p7/z, [x8, #1, mul vl]
sub v11.8h, v7.8h, v10.8h
sub v7.8h, v9.8h, v2.8h
str q27, [x0, #368]
tbl z27.h, {z19.h}, z3.h
tbl z10.h, {z23.h}, z3.h
str q15, [x0, #800]
tbl z9.h, {z20.h}, z13.h
tbl z2.h, {z24.h}, z13.h
str q14, [x0, #1232]
tbl z3.h, {z21.h}, z4.h
tbl z15.h, {z25.h}, z4.h
str q5, [x0, #1664]
tbl z13.h, {z22.h}, z0.h
tbl z14.h, {z26.h}, z0.h
str q31, [x0, #2096]
tbl z31.h, {z16.h, z17.h}, z1.h
add v4.8h, v9.8h, v3.8h
add v5.8h, v2.8h, v15.8h
str q28, [x0, #2528]
tbl z28.h, {z18.h}, z12.h
add v27.8h, v27.8h, v13.8h
add v0.8h, v10.8h, v14.8h
ld1b {z1.h}, p7/z, [x3, #-1, mul vl]
ld1b {z9.h}, p7/z, [x11, #0, mul vl]
add v3.8h, v31.8h, v28.8h
sub v31.8h, v31.8h, v28.8h
add v28.8h, v4.8h, v27.8h
sub v27.8h, v4.8h, v27.8h
ld1b {z2.h}, p7/z, [x6, #-6, mul vl]
add v15.8h, v5.8h, v0.8h
sub v12.8h, v5.8h, v0.8h
ld1b {z13.h}, p7/z, [x10, #-3, mul vl]
sub v10.8h, v3.8h, v28.8h
sub v14.8h, v31.8h, v27.8h
add v28.8h, v3.8h, v28.8h
add v27.8h, v31.8h, v27.8h
ld1b {z4.h}, p7/z, [x4, #2, mul vl]
add v5.8h, v3.8h, v15.8h
add v31.8h, v31.8h, v12.8h
ld1b {z0.h}, p7/z, [x8, #5, mul vl]
sub v3.8h, v10.8h, v15.8h
sub v10.8h, v14.8h, v12.8h
str q30, [x0, #384]
tbl z30.h, {z19.h}, z1.h
tbl z15.h, {z23.h}, z1.h
str q8, [x0, #816]
tbl z14.h, {z20.h}, z9.h
tbl z12.h, {z24.h}, z9.h
str q11, [x0, #1248]
tbl z1.h, {z21.h}, z2.h
tbl z8.h, {z25.h}, z2.h
str q7, [x0, #1680]
tbl z9.h, {z22.h}, z13.h
tbl z11.h, {z26.h}, z13.h
str q6, [x0, #2112]
tbl z2.h, {z16.h}, z4.h
add v7.8h, v14.8h, v1.8h
add v13.8h, v12.8h, v8.8h
str q29, [x0, #2544]
tbl z29.h, {z18.h}, z0.h
add v30.8h, v30.8h, v9.8h
add v6.8h, v15.8h, v11.8h
str q28, [x0, #400]
str q5, [x0, #832]
add v28.8h, v2.8h, v29.8h
sub v29.8h, v2.8h, v29.8h
add v4.8h, v7.8h, v30.8h
sub v30.8h, v7.8h, v30.8h
str q3, [x0, #1264]
add v14.8h, v13.8h, v6.8h
sub v1.8h, v13.8h, v6.8h
str q10, [x0, #1696]
sub v12.8h, v28.8h, v4.8h
sub v8.8h, v29.8h, v30.8h
add v0.8h, v28.8h, v4.8h
add v30.8h, v29.8h, v30.8h
str q27, [x0, #2128]
add v27.8h, v28.8h, v14.8h
add v28.8h, v29.8h, v1.8h
str q31, [x0, #2560]
sub v29.8h, v12.8h, v14.8h
sub v31.8h, v8.8h, v1.8h
str q0, [x0, #416]
str q27, [x0, #848]
str q29, [x0, #1280]
str q31, [x0, #1712]
str q30, [x0, #2144]
str q28, [x0, #2576]
add x0, x0, #2592
add x1, x1, #176
subs x14, x14, #1
b.ne 3b
ret
.Lsealed_guard:
mov x17,x30
ldr q15, [x4, #0]
bl 20f
add x0, x0, #432
add x1, x1, #432
add x2, x2, #32
add x3, x3, #80
ldr q15, [x4, #16]
bl 20f
add x0, x0, #432
add x1, x1, #432
add x2, x2, #32
add x3, x3, #80
ldr q15, [x4, #32]
bl 20f
add x0, x0, #432
add x1, x1, #432
add x2, x2, #32
add x3, x3, #80
ldr q15, [x4, #48]
bl 20f
add x0, x0, #432
add x1, x1, #432
add x2, x2, #32
add x3, x3, #80
ldr q15, [x4, #64]
bl 20f
add x0, x0, #432
add x1, x1, #432
add x2, x2, #32
add x3, x3, #80
ldr q15, [x4, #80]
bl 20f
add x0, x0, #432
add x1, x1, #432
add x2, x2, #32
add x3, x3, #80
ldr q15, [x4, #96]
bl 20f
add x0, x0, #432
add x1, x1, #432
add x2, x2, #32
add x3, x3, #80
ldr q15, [x4, #112]
bl 20f
add x0, x0, #432
add x1, x1, #432
add x2, x2, #32
add x3, x3, #80
ldr q15, [x4, #128]
bl 20f
add x0, x0, #432
add x1, x1, #432
add x2, x2, #32
add x3, x3, #80
ldr q15, [x4, #144]
bl 20f
add x0, x0, #432
add x1, x1, #432
add x2, x2, #32
add x3, x3, #80
ldr q15, [x4, #160]
bl 21f
add x0, x0, #432
add x1, x1, #432
add x2, x2, #32
add x3, x3, #80
ldr q15, [x4, #176]
bl 20f
add x0, x0, #432
add x1, x1, #432
add x2, x2, #32
add x3, x3, #80
ldr q15, [x4, #192]
bl 20f
add x0, x0, #432
add x1, x1, #432
add x2, x2, #32
add x3, x3, #80
ldr q15, [x4, #208]
bl 20f
add x0, x0, #432
add x1, x1, #432
add x2, x2, #32
add x3, x3, #80
ldr q15, [x4, #224]
bl 20f
add x0, x0, #432
add x1, x1, #432
add x2, x2, #32
add x3, x3, #80
ldr q15, [x4, #240]
bl 20f
add x0, x0, #432
add x1, x1, #432
add x2, x2, #32
add x3, x3, #80
ldr q15, [x4, #256]
bl 20f
add x0, x0, #432
add x1, x1, #432
add x2, x2, #32
add x3, x3, #80
ldr q15, [x4, #272]
bl 20f
add x0, x0, #432
add x1, x1, #432
add x2, x2, #32
add x3, x3, #80
ldr q15, [x4, #288]
bl 20f
add x0, x0, #432
add x1, x1, #432
add x2, x2, #32
add x3, x3, #80
ldr q15, [x4, #304]
bl 20f
add x0, x0, #432
add x1, x1, #432
add x2, x2, #32
add x3, x3, #80
ldr q15, [x4, #320]
bl 20f
add x0, x0, #432
add x1, x1, #432
add x2, x2, #32
add x3, x3, #80
ldr q15, [x4, #336]
bl 20f
add x0, x0, #432
add x1, x1, #432
add x2, x2, #32
add x3, x3, #80
ldr q15, [x4, #352]
bl 20f
add x0, x0, #432
add x1, x1, #432
add x2, x2, #32
add x3, x3, #80
ldr q15, [x4, #368]
bl 20f
add x0, x0, #432
add x1, x1, #432
add x2, x2, #32
add x3, x3, #80
b 9f
20:
ldr q16, [x0, #144]
ldr q17, [x0, #288]
ldr q18, [x0, #160]
ldr q19, [x0, #304]
ldr q20, [x0, #0]
ldr q21, [x0, #16]
mul v22.8h, v16.8h, v15.h[4]
sqrdmulh v23.8h, v16.8h, v15.h[5]
ldr q24, [x0, #176]
mul v25.8h, v17.8h, v15.h[6]
sqrdmulh v26.8h, v17.8h, v15.h[7]
ldr q27, [x0, #320]
mul v28.8h, v18.8h, v15.h[4]
sqrdmulh v29.8h, v18.8h, v15.h[5]
ldr q30, [x0, #32]
mul v31.8h, v19.8h, v15.h[6]
sqrdmulh v16.8h, v19.8h, v15.h[7]
ldr q17, [x0, #192]
mls v22.8h, v23.8h, v15.h[0]
ldr q18, [x0, #336]
mls v25.8h, v26.8h, v15.h[0]
ldr q19, [x0, #208]
mls v28.8h, v29.8h, v15.h[0]
mul v23.8h, v24.8h, v15.h[4]
ldr q26, [x0, #352]
mls v31.8h, v16.8h, v15.h[0]
sqrdmulh v29.8h, v24.8h, v15.h[5]
ldr q16, [x0, #48]
mul v24.8h, v27.8h, v15.h[6]
sqrdmulh v27.8h, v27.8h, v15.h[7]
sub v0.8h, v20.8h, v22.8h
ldr q1, [x0, #64]
mul v2.8h, v17.8h, v15.h[4]
sqrdmulh v17.8h, v17.8h, v15.h[5]
sub v3.8h, v22.8h, v25.8h
add v22.8h, v22.8h, v25.8h
sub v25.8h, v20.8h, v25.8h
ldr q4, [x0, #224]
mul v5.8h, v18.8h, v15.h[6]
sqrdmulh v18.8h, v18.8h, v15.h[7]
sub v6.8h, v21.8h, v28.8h
ldr q7, [x0, #368]
mul v8.8h, v3.8h, v15.h[2]
sqrdmulh v9.8h, v3.8h, v15.h[3]
add v20.8h, v20.8h, v22.8h
sub v22.8h, v28.8h, v31.8h
ldr q10, [x0, #80]
mls v23.8h, v29.8h, v15.h[0]
mls v24.8h, v27.8h, v15.h[0]
add v29.8h, v28.8h, v31.8h
sub v27.8h, v21.8h, v31.8h
ldr q28, [x0, #240]
mul v31.8h, v22.8h, v15.h[2]
sqrdmulh v22.8h, v22.8h, v15.h[3]
srshr v11.8h, v20.8h, #13
ldr q12, [x0, #384]
mls v2.8h, v17.8h, v15.h[0]
mls v5.8h, v18.8h, v15.h[0]
add v17.8h, v21.8h, v29.8h
ldr q18, [x0, #256]
mls v8.8h, v9.8h, v15.h[0]
mul v21.8h, v19.8h, v15.h[4]
ldr q29, [x0, #400]
sqrdmulh v19.8h, v19.8h, v15.h[5]
mul v13.8h, v26.8h, v15.h[6]
sub v14.8h, v23.8h, v24.8h
add v3.8h, v23.8h, v24.8h
sub v24.8h, v30.8h, v24.8h
sub v23.8h, v30.8h, v23.8h
mls v31.8h, v22.8h, v15.h[0]
sqrdmulh v22.8h, v26.8h, v15.h[7]
ldr q26, [x0, #96]
mls v20.8h, v11.8h, v15.h[0]
add v30.8h, v30.8h, v3.8h
sub v9.8h, v2.8h, v5.8h
srshr v11.8h, v17.8h, #13
mul v3.8h, v14.8h, v15.h[2]
sqrdmulh v14.8h, v14.8h, v15.h[3]
add v25.8h, v25.8h, v8.8h
sub v0.8h, v0.8h, v8.8h
mls v21.8h, v19.8h, v15.h[0]
mls v17.8h, v11.8h, v15.h[0]
add v19.8h, v2.8h, v5.8h
sub v8.8h, v16.8h, v5.8h
sub v11.8h, v16.8h, v2.8h
ldr q5, [x2, #16]
mul v2.8h, v9.8h, v15.h[2]
sqrdmulh v9.8h, v9.8h, v15.h[3]
add v27.8h, v27.8h, v31.8h
sub v31.8h, v6.8h, v31.8h
mls v13.8h, v22.8h, v15.h[0]
mul v22.8h, v4.8h, v15.h[4]
add v16.8h, v16.8h, v19.8h
ldr q19, [x0, #112]
mls v3.8h, v14.8h, v15.h[0]
sqrdmulh v6.8h, v4.8h, v15.h[5]
ldr q14, [x0, #272]
mls v2.8h, v9.8h, v15.h[0]
sub v4.8h, v1.8h, v21.8h
mls v22.8h, v6.8h, v15.h[0]
mul v9.8h, v28.8h, v15.h[4]
add v24.8h, v24.8h, v3.8h
sub v23.8h, v23.8h, v3.8h
sub v6.8h, v21.8h, v13.8h
add v21.8h, v21.8h, v13.8h
sqrdmulh v28.8h, v28.8h, v15.h[5]
sub v3.8h, v1.8h, v13.8h
mul v13.8h, v6.8h, v15.h[2]
sqrdmulh v6.8h, v6.8h, v15.h[3]
add v8.8h, v8.8h, v2.8h
sub v11.8h, v11.8h, v2.8h
mul v2.8h, v7.8h, v15.h[6]
sqrdmulh v7.8h, v7.8h, v15.h[7]
add v21.8h, v1.8h, v21.8h
mls v9.8h, v28.8h, v15.h[0]
mul v28.8h, v18.8h, v15.h[4]
mls v13.8h, v6.8h, v15.h[0]
sqrdmulh v18.8h, v18.8h, v15.h[5]
sub v1.8h, v10.8h, v22.8h
mls v2.8h, v7.8h, v15.h[0]
mul v6.8h, v12.8h, v15.h[6]
sqrdmulh v7.8h, v12.8h, v15.h[7]
mul v12.8h, v29.8h, v15.h[6]
sqrdmulh v29.8h, v29.8h, v15.h[7]
add v3.8h, v3.8h, v13.8h
sub v4.8h, v4.8h, v13.8h
mls v28.8h, v18.8h, v15.h[0]
sub v18.8h, v22.8h, v2.8h
add v22.8h, v22.8h, v2.8h
sub v13.8h, v10.8h, v2.8h
mls v6.8h, v7.8h, v15.h[0]
sub v2.8h, v26.8h, v9.8h
mul v7.8h, v18.8h, v15.h[2]
sqrdmulh v18.8h, v18.8h, v15.h[3]
add v22.8h, v10.8h, v22.8h
mls v12.8h, v29.8h, v15.h[0]
ldr q29, [x0, #416]
mls v7.8h, v18.8h, v15.h[0]
sub v18.8h, v9.8h, v6.8h
add v10.8h, v9.8h, v6.8h
sub v9.8h, v26.8h, v6.8h
add v26.8h, v26.8h, v10.8h
sub v6.8h, v28.8h, v12.8h
mul v10.8h, v18.8h, v15.h[2]
sqrdmulh v18.8h, v18.8h, v15.h[3]
add v13.8h, v13.8h, v7.8h
sub v1.8h, v1.8h, v7.8h
mls v10.8h, v18.8h, v15.h[0]
mul v18.8h, v6.8h, v15.h[2]
sqrdmulh v7.8h, v6.8h, v15.h[3]
add v6.8h, v9.8h, v10.8h
sub v9.8h, v2.8h, v10.8h
add v2.8h, v28.8h, v12.8h
sub v10.8h, v19.8h, v12.8h
mls v18.8h, v7.8h, v15.h[0]
mul v12.8h, v14.8h, v15.h[4]
sub v28.8h, v19.8h, v28.8h
sqrdmulh v7.8h, v14.8h, v15.h[5]
mul v14.8h, v29.8h, v15.h[6]
add v19.8h, v19.8h, v2.8h
sqrdmulh v29.8h, v29.8h, v15.h[7]
add v2.8h, v10.8h, v18.8h
sub v28.8h, v28.8h, v18.8h
ldr q18, [x0, #128]
mls v12.8h, v7.8h, v15.h[0]
ldr q10, [x2, #0]
mls v14.8h, v29.8h, v15.h[0]
srshr v29.8h, v30.8h, #13
mls v30.8h, v29.8h, v15.h[0]
mul v29.8h, v8.8h, v5.h[0]
sqrdmulh v7.8h, v8.8h, v5.h[1]
sub v8.8h, v12.8h, v14.8h
mls v29.8h, v7.8h, v15.h[0]
mul v7.8h, v8.8h, v15.h[2]
sqrdmulh v8.8h, v8.8h, v15.h[3]
mls v7.8h, v8.8h, v15.h[0]
add v8.8h, v12.8h, v14.8h
sub v14.8h, v18.8h, v14.8h
sub v12.8h, v18.8h, v12.8h
add v18.8h, v18.8h, v8.8h
mul v8.8h, v16.8h, v10.h[4]
sqrdmulh v16.8h, v16.8h, v10.h[5]
add v14.8h, v14.8h, v7.8h
sub v12.8h, v12.8h, v7.8h
mul v7.8h, v26.8h, v10.h[6]
sqrdmulh v26.8h, v26.8h, v10.h[7]
mls v8.8h, v16.8h, v15.h[0]
mls v7.8h, v26.8h, v15.h[0]
mul v16.8h, v21.8h, v10.h[4]
sqrdmulh v26.8h, v21.8h, v10.h[5]
sub v21.8h, v20.8h, v8.8h
mls v16.8h, v26.8h, v15.h[0]
add v26.8h, v8.8h, v7.8h
sub v8.8h, v8.8h, v7.8h
sub v7.8h, v20.8h, v7.8h
add v20.8h, v20.8h, v26.8h
mul v26.8h, v8.8h, v15.h[2]
sqrdmulh v8.8h, v8.8h, v15.h[3]
mls v26.8h, v8.8h, v15.h[0]
add v8.8h, v7.8h, v26.8h
sub v21.8h, v21.8h, v26.8h
sub v26.8h, v17.8h, v16.8h
mul v7.8h, v19.8h, v10.h[6]
sqrdmulh v19.8h, v19.8h, v10.h[7]
mls v7.8h, v19.8h, v15.h[0]
sub v19.8h, v16.8h, v7.8h
add v16.8h, v16.8h, v7.8h
sub v7.8h, v17.8h, v7.8h
add v17.8h, v17.8h, v16.8h
mul v16.8h, v19.8h, v15.h[2]
sqrdmulh v19.8h, v19.8h, v15.h[3]
mls v16.8h, v19.8h, v15.h[0]
mul v19.8h, v22.8h, v10.h[4]
sqrdmulh v22.8h, v22.8h, v10.h[5]
add v7.8h, v7.8h, v16.8h
sub v26.8h, v26.8h, v16.8h
mul v16.8h, v18.8h, v10.h[6]
sqrdmulh v18.8h, v18.8h, v10.h[7]
mls v19.8h, v22.8h, v15.h[0]
mul v22.8h, v6.8h, v5.h[2]
mls v16.8h, v18.8h, v15.h[0]
sqrdmulh v18.8h, v6.8h, v5.h[3]
srshr v10.8h, v25.8h, #13
mls v25.8h, v10.8h, v15.h[0]
srshr v6.8h, v27.8h, #13
sub v10.8h, v30.8h, v19.8h
mls v22.8h, v18.8h, v15.h[0]
mls v27.8h, v6.8h, v15.h[0]
srshr v18.8h, v24.8h, #13
sub v6.8h, v19.8h, v16.8h
add v19.8h, v19.8h, v16.8h
sub v16.8h, v30.8h, v16.8h
mls v24.8h, v18.8h, v15.h[0]
mul v18.8h, v6.8h, v15.h[2]
sqrdmulh v6.8h, v6.8h, v15.h[3]
add v30.8h, v30.8h, v19.8h
mls v18.8h, v6.8h, v15.h[0]
sub v19.8h, v29.8h, v22.8h
mul v6.8h, v19.8h, v15.h[2]
sqrdmulh v19.8h, v19.8h, v15.h[3]
add v16.8h, v16.8h, v18.8h
sub v18.8h, v10.8h, v18.8h
add v10.8h, v29.8h, v22.8h
sub v22.8h, v25.8h, v22.8h
sub v29.8h, v25.8h, v29.8h
add v25.8h, v25.8h, v10.8h
mls v6.8h, v19.8h, v15.h[0]
mul v19.8h, v13.8h, v5.h[0]
sqrdmulh v10.8h, v13.8h, v5.h[1]
mul v13.8h, v14.8h, v5.h[2]
sqrdmulh v14.8h, v14.8h, v5.h[3]
add v22.8h, v22.8h, v6.8h
sub v29.8h, v29.8h, v6.8h
mul v6.8h, v3.8h, v5.h[0]
sqrdmulh v3.8h, v3.8h, v5.h[1]
mls v19.8h, v10.8h, v15.h[0]
mul v10.8h, v2.8h, v5.h[2]
sqrdmulh v2.8h, v2.8h, v5.h[3]
mls v13.8h, v14.8h, v15.h[0]
mls v6.8h, v3.8h, v15.h[0]
mul v14.8h, v11.8h, v5.h[4]
sqrdmulh v3.8h, v11.8h, v5.h[5]
mls v10.8h, v2.8h, v15.h[0]
sub v11.8h, v24.8h, v19.8h
sub v2.8h, v27.8h, v6.8h
mls v14.8h, v3.8h, v15.h[0]
sub v3.8h, v6.8h, v10.8h
add v6.8h, v6.8h, v10.8h
sub v10.8h, v27.8h, v10.8h
add v27.8h, v27.8h, v6.8h
mul v6.8h, v3.8h, v15.h[2]
sqrdmulh v3.8h, v3.8h, v15.h[3]
mls v6.8h, v3.8h, v15.h[0]
add v3.8h, v10.8h, v6.8h
sub v10.8h, v2.8h, v6.8h
sub v2.8h, v19.8h, v13.8h
add v19.8h, v19.8h, v13.8h
sub v6.8h, v24.8h, v13.8h
mul v13.8h, v2.8h, v15.h[2]
sqrdmulh v2.8h, v2.8h, v15.h[3]
add v24.8h, v24.8h, v19.8h
mls v13.8h, v2.8h, v15.h[0]
mul v19.8h, v9.8h, v5.h[6]
sqrdmulh v2.8h, v9.8h, v5.h[7]
add v9.8h, v6.8h, v13.8h
sub v6.8h, v11.8h, v13.8h
srshr v11.8h, v0.8h, #13
mls v19.8h, v2.8h, v15.h[0]
mul v13.8h, v4.8h, v5.h[4]
sqrdmulh v2.8h, v4.8h, v5.h[5]
mls v0.8h, v11.8h, v15.h[0]
srshr v4.8h, v31.8h, #13
mls v31.8h, v4.8h, v15.h[0]
mul v11.8h, v1.8h, v5.h[4]
sub v4.8h, v14.8h, v19.8h
mls v13.8h, v2.8h, v15.h[0]
add v2.8h, v14.8h, v19.8h
sub v19.8h, v0.8h, v19.8h
sub v14.8h, v0.8h, v14.8h
sqrdmulh v1.8h, v1.8h, v5.h[5]
add v0.8h, v0.8h, v2.8h
mul v2.8h, v4.8h, v15.h[2]
sqrdmulh v4.8h, v4.8h, v15.h[3]
mls v11.8h, v1.8h, v15.h[0]
mls v2.8h, v4.8h, v15.h[0]
sub v1.8h, v31.8h, v13.8h
mul v4.8h, v12.8h, v5.h[6]
sqrdmulh v12.8h, v12.8h, v5.h[7]
add v19.8h, v19.8h, v2.8h
sub v14.8h, v14.8h, v2.8h
mul v2.8h, v28.8h, v5.h[6]
sqrdmulh v28.8h, v28.8h, v5.h[7]
mls v4.8h, v12.8h, v15.h[0]
srshr v5.8h, v23.8h, #13
mls v2.8h, v28.8h, v15.h[0]
ldr q12, [x3, #0]
mls v23.8h, v5.8h, v15.h[0]
sub v28.8h, v13.8h, v2.8h
add v5.8h, v13.8h, v2.8h
sub v13.8h, v31.8h, v2.8h
mul v2.8h, v28.8h, v15.h[2]
sqrdmulh v28.8h, v28.8h, v15.h[3]
add v31.8h, v31.8h, v5.8h
mls v2.8h, v28.8h, v15.h[0]
sub v28.8h, v11.8h, v4.8h
mul v5.8h, v28.8h, v15.h[2]
sqrdmulh v28.8h, v28.8h, v15.h[3]
add v13.8h, v13.8h, v2.8h
sub v1.8h, v1.8h, v2.8h
add v2.8h, v11.8h, v4.8h
sub v4.8h, v23.8h, v4.8h
sub v11.8h, v23.8h, v11.8h
add v23.8h, v23.8h, v2.8h
mls v5.8h, v28.8h, v15.h[0]
mul v28.8h, v17.8h, v12.h[0]
sqrdmulh v17.8h, v17.8h, v12.h[1]
mul v2.8h, v30.8h, v12.h[2]
sqrdmulh v30.8h, v30.8h, v12.h[3]
add v4.8h, v4.8h, v5.8h
sub v11.8h, v11.8h, v5.8h
mls v28.8h, v17.8h, v15.h[0]
mul v17.8h, v7.8h, v12.h[4]
sqrdmulh v5.8h, v7.8h, v12.h[5]
mls v2.8h, v30.8h, v15.h[0]
mul v30.8h, v16.8h, v12.h[6]
sqrdmulh v16.8h, v16.8h, v12.h[7]
sqsub v7.8h, v20.8h, v28.8h
mls v17.8h, v5.8h, v15.h[0]
ldr q12, [x3, #16]
sub v5.8h, v28.8h, v2.8h
add v28.8h, v28.8h, v2.8h
sqsub v2.8h, v20.8h, v2.8h
mls v30.8h, v16.8h, v15.h[0]
mul v16.8h, v5.8h, v15.h[2]
sqrdmulh v5.8h, v5.8h, v15.h[3]
sqadd v20.8h, v20.8h, v28.8h
sqsub v28.8h, v8.8h, v17.8h
str q20, [x1, #0]
mls v16.8h, v5.8h, v15.h[0]
sub v20.8h, v17.8h, v30.8h
add v17.8h, v17.8h, v30.8h
sqsub v30.8h, v8.8h, v30.8h
mul v5.8h, v26.8h, v12.h[0]
sqrdmulh v26.8h, v26.8h, v12.h[1]
sqadd v17.8h, v8.8h, v17.8h
mul v8.8h, v20.8h, v15.h[2]
sqrdmulh v20.8h, v20.8h, v15.h[3]
sqadd v2.8h, v2.8h, v16.8h
sqsub v16.8h, v7.8h, v16.8h
str q17, [x1, #48]
mul v17.8h, v18.8h, v12.h[2]
sqrdmulh v18.8h, v18.8h, v12.h[3]
mls v5.8h, v26.8h, v15.h[0]
mul v26.8h, v27.8h, v12.h[4]
str q2, [x1, #16]
sqrdmulh v27.8h, v27.8h, v12.h[5]
mul v7.8h, v24.8h, v12.h[6]
str q16, [x1, #32]
mls v8.8h, v20.8h, v15.h[0]
ldr q2, [x3, #32]
mls v17.8h, v18.8h, v15.h[0]
sqrdmulh v16.8h, v24.8h, v12.h[7]
ldr q12, [x3, #48]
mls v26.8h, v27.8h, v15.h[0]
sqadd v20.8h, v30.8h, v8.8h
sqsub v18.8h, v28.8h, v8.8h
sqsub v24.8h, v21.8h, v5.8h
ldr q8, [x3, #64]
mls v7.8h, v16.8h, v15.h[0]
mul v27.8h, v3.8h, v2.h[0]
sqrdmulh v30.8h, v3.8h, v2.h[1]
sub v28.8h, v5.8h, v17.8h
add v16.8h, v5.8h, v17.8h
sqsub v17.8h, v21.8h, v17.8h
str q20, [x1, #64]
sqsub v20.8h, v25.8h, v26.8h
str q18, [x1, #80]
mul v18.8h, v28.8h, v15.h[2]
sqrdmulh v28.8h, v28.8h, v15.h[3]
sqadd v21.8h, v21.8h, v16.8h
mul v16.8h, v9.8h, v2.h[2]
sqrdmulh v3.8h, v9.8h, v2.h[3]
sub v5.8h, v26.8h, v7.8h
add v26.8h, v26.8h, v7.8h
sqsub v9.8h, v25.8h, v7.8h
mls v27.8h, v30.8h, v15.h[0]
mul v30.8h, v10.8h, v2.h[4]
str q21, [x1, #96]
mul v21.8h, v5.8h, v15.h[2]
sqrdmulh v7.8h, v5.8h, v15.h[3]
sqadd v25.8h, v25.8h, v26.8h
mls v18.8h, v28.8h, v15.h[0]
sqrdmulh v26.8h, v10.8h, v2.h[5]
mls v16.8h, v3.8h, v15.h[0]
mul v28.8h, v6.8h, v2.h[6]
str q25, [x1, #144]
sqrdmulh v25.8h, v6.8h, v2.h[7]
mul v5.8h, v31.8h, v12.h[0]
sqsub v10.8h, v22.8h, v27.8h
mls v21.8h, v7.8h, v15.h[0]
sqrdmulh v31.8h, v31.8h, v12.h[1]
mls v30.8h, v26.8h, v15.h[0]
mul v26.8h, v23.8h, v12.h[2]
sqadd v17.8h, v17.8h, v18.8h
sqsub v24.8h, v24.8h, v18.8h
sqrdmulh v18.8h, v23.8h, v12.h[3]
mul v23.8h, v13.8h, v12.h[4]
sub v3.8h, v27.8h, v16.8h
add v27.8h, v27.8h, v16.8h
sqsub v16.8h, v22.8h, v16.8h
mls v28.8h, v25.8h, v15.h[0]
sqrdmulh v25.8h, v13.8h, v12.h[5]
str q17, [x1, #112]
mul v17.8h, v3.8h, v15.h[2]
sqrdmulh v6.8h, v3.8h, v15.h[3]
sqadd v2.8h, v9.8h, v21.8h
sqsub v20.8h, v20.8h, v21.8h
str q24, [x1, #128]
mls v5.8h, v31.8h, v15.h[0]
mul v21.8h, v4.8h, v12.h[6]
sqadd v24.8h, v22.8h, v27.8h
sqsub v31.8h, v29.8h, v30.8h
mls v26.8h, v18.8h, v15.h[0]
sqrdmulh v22.8h, v4.8h, v12.h[7]
str q2, [x1, #160]
mls v23.8h, v25.8h, v15.h[0]
mul v27.8h, v1.8h, v8.h[0]
sub v18.8h, v30.8h, v28.8h
add v25.8h, v30.8h, v28.8h
sqsub v30.8h, v29.8h, v28.8h
str q20, [x1, #176]
mls v17.8h, v6.8h, v15.h[0]
sqrdmulh v28.8h, v1.8h, v8.h[1]
str q24, [x1, #192]
mul v20.8h, v18.8h, v15.h[2]
sqrdmulh v24.8h, v18.8h, v15.h[3]
sqadd v18.8h, v29.8h, v25.8h
sqsub v29.8h, v0.8h, v5.8h
mls v21.8h, v22.8h, v15.h[0]
mul v25.8h, v11.8h, v8.h[2]
sub v22.8h, v5.8h, v26.8h
add v7.8h, v5.8h, v26.8h
sqsub v26.8h, v0.8h, v26.8h
sqrdmulh v13.8h, v11.8h, v8.h[3]
sqsub v3.8h, v19.8h, v23.8h
str q18, [x1, #240]
mul v18.8h, v22.8h, v15.h[2]
sqrdmulh v22.8h, v22.8h, v15.h[3]
sqadd v16.8h, v16.8h, v17.8h
sqsub v17.8h, v10.8h, v17.8h
mls v20.8h, v24.8h, v15.h[0]
mls v27.8h, v28.8h, v15.h[0]
sqadd v24.8h, v0.8h, v7.8h
sub v28.8h, v23.8h, v21.8h
add v23.8h, v23.8h, v21.8h
sqsub v21.8h, v19.8h, v21.8h
str q16, [x1, #208]
mls v25.8h, v13.8h, v15.h[0]
str q17, [x1, #224]
mls v18.8h, v22.8h, v15.h[0]
mul v16.8h, v28.8h, v15.h[2]
sqadd v17.8h, v19.8h, v23.8h
str q24, [x1, #288]
sqrdmulh v22.8h, v28.8h, v15.h[3]
sqadd v19.8h, v30.8h, v20.8h
sqsub v23.8h, v31.8h, v20.8h
sqsub v24.8h, v14.8h, v27.8h
str q17, [x1, #336]
sub v28.8h, v27.8h, v25.8h
add v30.8h, v27.8h, v25.8h
sqsub v31.8h, v14.8h, v25.8h
str q19, [x1, #256]
sqadd v20.8h, v26.8h, v18.8h
sqsub v17.8h, v29.8h, v18.8h
str q23, [x1, #272]
mls v16.8h, v22.8h, v15.h[0]
mul v27.8h, v28.8h, v15.h[2]
sqadd v25.8h, v14.8h, v30.8h
sqrdmulh v19.8h, v28.8h, v15.h[3]
str q20, [x1, #304]
str q17, [x1, #320]
str q25, [x1, #384]
sqadd v26.8h, v21.8h, v16.8h
sqsub v29.8h, v3.8h, v16.8h
mls v27.8h, v19.8h, v15.h[0]
str q26, [x1, #352]
str q29, [x1, #368]
sqadd v18.8h, v31.8h, v27.8h
sqsub v23.8h, v24.8h, v27.8h
str q18, [x1, #400]
str q23, [x1, #416]
ret
21:
ldr q16, [x0, #144]
ldr q17, [x0, #288]
ldr q18, [x0, #0]
ldr q19, [x0, #160]
ldr q20, [x0, #304]
ldr q21, [x0, #16]
mul v22.8h, v16.8h, v15.h[4]
sqrdmulh v23.8h, v16.8h, v15.h[5]
ldr q24, [x0, #176]
mul v25.8h, v17.8h, v15.h[6]
sqrdmulh v26.8h, v17.8h, v15.h[7]
ldr q27, [x0, #320]
srshr v28.8h, v18.8h, #13
ldr q29, [x0, #32]
mul v30.8h, v19.8h, v15.h[4]
sqrdmulh v31.8h, v19.8h, v15.h[5]
ldr q16, [x0, #192]
mls v22.8h, v23.8h, v15.h[0]
mul v17.8h, v20.8h, v15.h[6]
ldr q19, [x0, #336]
mls v25.8h, v26.8h, v15.h[0]
sqrdmulh v23.8h, v20.8h, v15.h[7]
srshr v26.8h, v21.8h, #13
ldr q20, [x0, #48]
mls v18.8h, v28.8h, v15.h[0]
mul v28.8h, v24.8h, v15.h[4]
ldr q0, [x0, #208]
mls v30.8h, v31.8h, v15.h[0]
mls v21.8h, v26.8h, v15.h[0]
ldr q31, [x0, #352]
sqrdmulh v26.8h, v24.8h, v15.h[5]
mul v24.8h, v27.8h, v15.h[6]
srshr v1.8h, v29.8h, #13
ldr q2, [x0, #64]
mls v17.8h, v23.8h, v15.h[0]
sqrdmulh v23.8h, v27.8h, v15.h[7]
sub v27.8h, v22.8h, v25.8h
add v3.8h, v22.8h, v25.8h
ldr q4, [x0, #224]
mls v29.8h, v1.8h, v15.h[0]
mul v5.8h, v16.8h, v15.h[4]
sub v25.8h, v18.8h, v25.8h
sub v22.8h, v18.8h, v22.8h
ldr q6, [x0, #368]
mul v7.8h, v27.8h, v15.h[2]
sqrdmulh v27.8h, v27.8h, v15.h[3]
add v18.8h, v18.8h, v3.8h
sub v8.8h, v21.8h, v30.8h
ldr q9, [x0, #80]
mls v28.8h, v26.8h, v15.h[0]
sqrdmulh v26.8h, v16.8h, v15.h[5]
srshr v16.8h, v20.8h, #13
ldr q10, [x0, #240]
mls v24.8h, v23.8h, v15.h[0]
mul v23.8h, v19.8h, v15.h[6]
sub v11.8h, v30.8h, v17.8h
add v30.8h, v30.8h, v17.8h
sub v17.8h, v21.8h, v17.8h
ldr q12, [x0, #384]
sqrdmulh v19.8h, v19.8h, v15.h[7]
mul v13.8h, v0.8h, v15.h[4]
srshr v14.8h, v2.8h, #13
ldr q1, [x0, #96]
mls v7.8h, v27.8h, v15.h[0]
mul v27.8h, v11.8h, v15.h[2]
add v21.8h, v21.8h, v30.8h
ldr q30, [x0, #256]
sqrdmulh v3.8h, v11.8h, v15.h[3]
mls v5.8h, v26.8h, v15.h[0]
ldr q26, [x0, #400]
sqrdmulh v11.8h, v0.8h, v15.h[5]
mls v20.8h, v16.8h, v15.h[0]
srshr v16.8h, v9.8h, #13
mls v23.8h, v19.8h, v15.h[0]
mul v19.8h, v31.8h, v15.h[6]
sqrdmulh v31.8h, v31.8h, v15.h[7]
mls v2.8h, v14.8h, v15.h[0]
add v25.8h, v25.8h, v7.8h
sub v22.8h, v22.8h, v7.8h
sub v0.8h, v28.8h, v24.8h
ldr q14, [x0, #112]
mls v27.8h, v3.8h, v15.h[0]
mls v9.8h, v16.8h, v15.h[0]
add v16.8h, v28.8h, v24.8h
sub v24.8h, v29.8h, v24.8h
sub v28.8h, v29.8h, v28.8h
ldr q7, [x0, #272]
mls v13.8h, v11.8h, v15.h[0]
sub v3.8h, v20.8h, v5.8h
add v29.8h, v29.8h, v16.8h
sub v16.8h, v5.8h, v23.8h
add v11.8h, v5.8h, v23.8h
sub v23.8h, v20.8h, v23.8h
mls v19.8h, v31.8h, v15.h[0]
mul v31.8h, v4.8h, v15.h[4]
mul v5.8h, v0.8h, v15.h[2]
sqrdmulh v0.8h, v0.8h, v15.h[3]
add v17.8h, v17.8h, v27.8h
sub v27.8h, v8.8h, v27.8h
sqrdmulh v8.8h, v4.8h, v15.h[5]
add v20.8h, v20.8h, v11.8h
sub v4.8h, v2.8h, v13.8h
mls v5.8h, v0.8h, v15.h[0]
mul v11.8h, v16.8h, v15.h[2]
sqrdmulh v16.8h, v16.8h, v15.h[3]
mls v31.8h, v8.8h, v15.h[0]
srshr v0.8h, v1.8h, #13
mls v1.8h, v0.8h, v15.h[0]
srshr v8.8h, v14.8h, #13
add v24.8h, v24.8h, v5.8h
sub v28.8h, v28.8h, v5.8h
sub v0.8h, v13.8h, v19.8h
add v5.8h, v13.8h, v19.8h
mls v11.8h, v16.8h, v15.h[0]
mul v16.8h, v6.8h, v15.h[6]
sub v19.8h, v2.8h, v19.8h
sqrdmulh v13.8h, v6.8h, v15.h[7]
mls v14.8h, v8.8h, v15.h[0]
add v6.8h, v2.8h, v5.8h
sub v8.8h, v9.8h, v31.8h
ldr q2, [x0, #416]
mul v5.8h, v0.8h, v15.h[2]
sqrdmulh v0.8h, v0.8h, v15.h[3]
add v23.8h, v23.8h, v11.8h
sub v3.8h, v3.8h, v11.8h
mls v16.8h, v13.8h, v15.h[0]
mul v11.8h, v10.8h, v15.h[4]
sqrdmulh v13.8h, v10.8h, v15.h[5]
mls v5.8h, v0.8h, v15.h[0]
mul v10.8h, v12.8h, v15.h[6]
sqrdmulh v0.8h, v12.8h, v15.h[7]
mls v11.8h, v13.8h, v15.h[0]
mul v12.8h, v30.8h, v15.h[4]
sqrdmulh v30.8h, v30.8h, v15.h[5]
add v19.8h, v19.8h, v5.8h
sub v13.8h, v4.8h, v5.8h
sub v4.8h, v31.8h, v16.8h
add v31.8h, v31.8h, v16.8h
mls v10.8h, v0.8h, v15.h[0]
mul v5.8h, v26.8h, v15.h[6]
sub v16.8h, v9.8h, v16.8h
mul v0.8h, v4.8h, v15.h[2]
sqrdmulh v4.8h, v4.8h, v15.h[3]
add v31.8h, v9.8h, v31.8h
sqrdmulh v26.8h, v26.8h, v15.h[7]
mls v12.8h, v30.8h, v15.h[0]
sub v30.8h, v1.8h, v11.8h
mls v0.8h, v4.8h, v15.h[0]
sub v9.8h, v11.8h, v10.8h
add v4.8h, v11.8h, v10.8h
sub v11.8h, v1.8h, v10.8h
mls v5.8h, v26.8h, v15.h[0]
sub v26.8h, v14.8h, v12.8h
mul v10.8h, v7.8h, v15.h[4]
sqrdmulh v7.8h, v7.8h, v15.h[5]
add v1.8h, v1.8h, v4.8h
mul v4.8h, v9.8h, v15.h[2]
sqrdmulh v9.8h, v9.8h, v15.h[3]
add v16.8h, v16.8h, v0.8h
sub v8.8h, v8.8h, v0.8h
mls v10.8h, v7.8h, v15.h[0]
ldr q0, [x2, #0]
mls v4.8h, v9.8h, v15.h[0]
sub v7.8h, v12.8h, v5.8h
add v9.8h, v12.8h, v5.8h
sub v12.8h, v14.8h, v5.8h
mul v5.8h, v2.8h, v15.h[6]
sqrdmulh v2.8h, v2.8h, v15.h[7]
add v14.8h, v14.8h, v9.8h
mul v9.8h, v7.8h, v15.h[2]
sqrdmulh v7.8h, v7.8h, v15.h[3]
add v11.8h, v11.8h, v4.8h
sub v30.8h, v30.8h, v4.8h
mls v5.8h, v2.8h, v15.h[0]
mul v4.8h, v20.8h, v0.h[4]
sqrdmulh v20.8h, v20.8h, v0.h[5]
mls v9.8h, v7.8h, v15.h[0]
ldr q2, [x0, #128]
mls v4.8h, v20.8h, v15.h[0]
mul v20.8h, v6.8h, v0.h[4]
sqrdmulh v7.8h, v6.8h, v0.h[5]
add v6.8h, v12.8h, v9.8h
sub v26.8h, v26.8h, v9.8h
sub v12.8h, v10.8h, v5.8h
add v9.8h, v10.8h, v5.8h
mls v20.8h, v7.8h, v15.h[0]
srshr v7.8h, v2.8h, #13
mls v2.8h, v7.8h, v15.h[0]
mul v7.8h, v12.8h, v15.h[2]
sqrdmulh v12.8h, v12.8h, v15.h[3]
sub v5.8h, v2.8h, v5.8h
sub v10.8h, v2.8h, v10.8h
add v2.8h, v2.8h, v9.8h
mls v7.8h, v12.8h, v15.h[0]
mul v9.8h, v1.8h, v0.h[6]
sqrdmulh v12.8h, v1.8h, v0.h[7]
mul v1.8h, v14.8h, v0.h[6]
sqrdmulh v14.8h, v14.8h, v0.h[7]
add v5.8h, v5.8h, v7.8h
sub v10.8h, v10.8h, v7.8h
mls v9.8h, v12.8h, v15.h[0]
sub v7.8h, v18.8h, v4.8h
mls v1.8h, v14.8h, v15.h[0]
sub v12.8h, v21.8h, v20.8h
sub v14.8h, v4.8h, v9.8h
add v4.8h, v4.8h, v9.8h
sub v9.8h, v18.8h, v9.8h
add v18.8h, v18.8h, v4.8h
mul v4.8h, v14.8h, v15.h[2]
sqrdmulh v14.8h, v14.8h, v15.h[3]
mls v4.8h, v14.8h, v15.h[0]
add v14.8h, v9.8h, v4.8h
sub v9.8h, v7.8h, v4.8h
sub v7.8h, v20.8h, v1.8h
add v20.8h, v20.8h, v1.8h
sub v4.8h, v21.8h, v1.8h
mul v1.8h, v7.8h, v15.h[2]
sqrdmulh v7.8h, v7.8h, v15.h[3]
add v21.8h, v21.8h, v20.8h
mls v1.8h, v7.8h, v15.h[0]
mul v20.8h, v31.8h, v0.h[4]
sqrdmulh v31.8h, v31.8h, v0.h[5]
mul v7.8h, v2.8h, v0.h[6]
sqrdmulh v2.8h, v2.8h, v0.h[7]
add v0.8h, v4.8h, v1.8h
sub v4.8h, v12.8h, v1.8h
ldr q12, [x2, #16]
mls v20.8h, v31.8h, v15.h[0]
ldr q1, [x3, #0]
mls v7.8h, v2.8h, v15.h[0]
srshr v31.8h, v18.8h, #13
sub v2.8h, v29.8h, v20.8h
mls v18.8h, v31.8h, v15.h[0]
sub v31.8h, v20.8h, v7.8h
add v20.8h, v20.8h, v7.8h
sub v7.8h, v29.8h, v7.8h
add v29.8h, v29.8h, v20.8h
mul v20.8h, v31.8h, v15.h[2]
sqrdmulh v31.8h, v31.8h, v15.h[3]
mls v20.8h, v31.8h, v15.h[0]
mul v31.8h, v23.8h, v12.h[0]
sqrdmulh v23.8h, v23.8h, v12.h[1]
add v7.8h, v7.8h, v20.8h
sub v20.8h, v2.8h, v20.8h
mul v2.8h, v11.8h, v12.h[2]
sqrdmulh v11.8h, v11.8h, v12.h[3]
mls v31.8h, v23.8h, v15.h[0]
mls v2.8h, v11.8h, v15.h[0]
mul v23.8h, v19.8h, v12.h[0]
sqrdmulh v19.8h, v19.8h, v12.h[1]
sub v11.8h, v25.8h, v31.8h
mls v23.8h, v19.8h, v15.h[0]
add v19.8h, v31.8h, v2.8h
sub v31.8h, v31.8h, v2.8h
sub v2.8h, v25.8h, v2.8h
add v25.8h, v25.8h, v19.8h
mul v19.8h, v31.8h, v15.h[2]
sqrdmulh v31.8h, v31.8h, v15.h[3]
mls v19.8h, v31.8h, v15.h[0]
add v31.8h, v2.8h, v19.8h
sub v19.8h, v11.8h, v19.8h
sub v2.8h, v17.8h, v23.8h
mul v11.8h, v6.8h, v12.h[2]
sqrdmulh v6.8h, v6.8h, v12.h[3]
mls v11.8h, v6.8h, v15.h[0]
sub v6.8h, v23.8h, v11.8h
add v23.8h, v23.8h, v11.8h
sub v11.8h, v17.8h, v11.8h
add v17.8h, v17.8h, v23.8h
mul v23.8h, v6.8h, v15.h[2]
sqrdmulh v6.8h, v6.8h, v15.h[3]
mls v23.8h, v6.8h, v15.h[0]
mul v6.8h, v16.8h, v12.h[0]
sqrdmulh v16.8h, v16.8h, v12.h[1]
add v11.8h, v11.8h, v23.8h
sub v23.8h, v2.8h, v23.8h
mul v2.8h, v5.8h, v12.h[2]
sqrdmulh v5.8h, v5.8h, v12.h[3]
mls v6.8h, v16.8h, v15.h[0]
mls v2.8h, v5.8h, v15.h[0]
mul v16.8h, v3.8h, v12.h[4]
sqrdmulh v5.8h, v3.8h, v12.h[5]
sub v3.8h, v24.8h, v6.8h
mls v16.8h, v5.8h, v15.h[0]
sub v5.8h, v6.8h, v2.8h
add v6.8h, v6.8h, v2.8h
sub v2.8h, v24.8h, v2.8h
add v24.8h, v24.8h, v6.8h
mul v6.8h, v5.8h, v15.h[2]
sqrdmulh v5.8h, v5.8h, v15.h[3]
mls v6.8h, v5.8h, v15.h[0]
mul v5.8h, v30.8h, v12.h[6]
sqrdmulh v30.8h, v30.8h, v12.h[7]
add v2.8h, v2.8h, v6.8h
sub v3.8h, v3.8h, v6.8h
sub v6.8h, v22.8h, v16.8h
mls v5.8h, v30.8h, v15.h[0]
sub v30.8h, v16.8h, v5.8h
add v16.8h, v16.8h, v5.8h
sub v5.8h, v22.8h, v5.8h
add v22.8h, v22.8h, v16.8h
mul v16.8h, v30.8h, v15.h[2]
sqrdmulh v30.8h, v30.8h, v15.h[3]
mls v16.8h, v30.8h, v15.h[0]
mul v30.8h, v8.8h, v12.h[4]
sqrdmulh v8.8h, v8.8h, v12.h[5]
add v5.8h, v5.8h, v16.8h
sub v16.8h, v6.8h, v16.8h
mul v6.8h, v13.8h, v12.h[4]
sqrdmulh v13.8h, v13.8h, v12.h[5]
mls v30.8h, v8.8h, v15.h[0]
mul v8.8h, v26.8h, v12.h[6]
sqrdmulh v26.8h, v26.8h, v12.h[7]
mls v6.8h, v13.8h, v15.h[0]
mls v8.8h, v26.8h, v15.h[0]
mul v26.8h, v10.8h, v12.h[6]
sqrdmulh v13.8h, v10.8h, v12.h[7]
sub v10.8h, v27.8h, v6.8h
sub v12.8h, v6.8h, v8.8h
add v6.8h, v6.8h, v8.8h
sub v8.8h, v27.8h, v8.8h
mls v26.8h, v13.8h, v15.h[0]
mul v13.8h, v12.8h, v15.h[2]
sqrdmulh v12.8h, v12.8h, v15.h[3]
add v27.8h, v27.8h, v6.8h
mls v13.8h, v12.8h, v15.h[0]
sub v6.8h, v30.8h, v26.8h
mul v12.8h, v6.8h, v15.h[2]
sqrdmulh v6.8h, v6.8h, v15.h[3]
add v8.8h, v8.8h, v13.8h
sub v10.8h, v10.8h, v13.8h
add v13.8h, v30.8h, v26.8h
sub v26.8h, v28.8h, v26.8h
sub v30.8h, v28.8h, v30.8h
add v28.8h, v28.8h, v13.8h
mls v12.8h, v6.8h, v15.h[0]
mul v13.8h, v21.8h, v1.h[0]
sqrdmulh v21.8h, v21.8h, v1.h[1]
mul v6.8h, v29.8h, v1.h[2]
sqrdmulh v29.8h, v29.8h, v1.h[3]
add v26.8h, v26.8h, v12.8h
sub v30.8h, v30.8h, v12.8h
mls v13.8h, v21.8h, v15.h[0]
mul v21.8h, v0.8h, v1.h[4]
sqrdmulh v12.8h, v0.8h, v1.h[5]
mls v6.8h, v29.8h, v15.h[0]
mul v29.8h, v7.8h, v1.h[6]
sqrdmulh v0.8h, v7.8h, v1.h[7]
sub v7.8h, v18.8h, v13.8h
mls v21.8h, v12.8h, v15.h[0]
srshr v1.8h, v14.8h, #13
mls v14.8h, v1.8h, v15.h[0]
sub v12.8h, v13.8h, v6.8h
add v1.8h, v13.8h, v6.8h
sub v13.8h, v18.8h, v6.8h
mls v29.8h, v0.8h, v15.h[0]
ldr q6, [x3, #16]
mul v0.8h, v12.8h, v15.h[2]
sqrdmulh v12.8h, v12.8h, v15.h[3]
add v18.8h, v18.8h, v1.8h
sub v1.8h, v14.8h, v21.8h
str q18, [x1, #0]
mls v0.8h, v12.8h, v15.h[0]
sub v18.8h, v21.8h, v29.8h
add v21.8h, v21.8h, v29.8h
sub v29.8h, v14.8h, v29.8h
mul v12.8h, v4.8h, v6.h[0]
sqrdmulh v4.8h, v4.8h, v6.h[1]
add v21.8h, v14.8h, v21.8h
mul v14.8h, v18.8h, v15.h[2]
sqrdmulh v18.8h, v18.8h, v15.h[3]
add v13.8h, v13.8h, v0.8h
sub v7.8h, v7.8h, v0.8h
str q21, [x1, #48]
mul v21.8h, v20.8h, v6.h[2]
sqrdmulh v20.8h, v20.8h, v6.h[3]
mls v12.8h, v4.8h, v15.h[0]
mul v0.8h, v17.8h, v6.h[4]
srshr v4.8h, v9.8h, #13
str q13, [x1, #16]
sqrdmulh v17.8h, v17.8h, v6.h[5]
mul v13.8h, v24.8h, v6.h[6]
str q7, [x1, #32]
mls v14.8h, v18.8h, v15.h[0]
mls v9.8h, v4.8h, v15.h[0]
srshr v18.8h, v25.8h, #13
ldr q7, [x3, #32]
mls v21.8h, v20.8h, v15.h[0]
sqrdmulh v20.8h, v24.8h, v6.h[7]
srshr v24.8h, v31.8h, #13
srshr v4.8h, v19.8h, #13
mls v25.8h, v18.8h, v15.h[0]
ldr q6, [x3, #48]
mls v0.8h, v17.8h, v15.h[0]
mls v31.8h, v24.8h, v15.h[0]
srshr v18.8h, v22.8h, #13
srshr v17.8h, v5.8h, #13
mls v19.8h, v4.8h, v15.h[0]
add v24.8h, v29.8h, v14.8h
sub v29.8h, v1.8h, v14.8h
sub v4.8h, v9.8h, v12.8h
ldr q1, [x3, #64]
mls v13.8h, v20.8h, v15.h[0]
mls v22.8h, v18.8h, v15.h[0]
srshr v20.8h, v16.8h, #13
mul v18.8h, v11.8h, v7.h[0]
sqrdmulh v14.8h, v11.8h, v7.h[1]
sub v11.8h, v12.8h, v21.8h
add v12.8h, v12.8h, v21.8h
sub v21.8h, v9.8h, v21.8h
str q24, [x1, #64]
mls v5.8h, v17.8h, v15.h[0]
mls v16.8h, v20.8h, v15.h[0]
sub v24.8h, v25.8h, v0.8h
str q29, [x1, #80]
mul v17.8h, v11.8h, v15.h[2]
sqrdmulh v20.8h, v11.8h, v15.h[3]
add v29.8h, v9.8h, v12.8h
mul v11.8h, v2.8h, v7.h[2]
sqrdmulh v9.8h, v2.8h, v7.h[3]
sub v12.8h, v0.8h, v13.8h
add v2.8h, v0.8h, v13.8h
sub v0.8h, v25.8h, v13.8h
mls v18.8h, v14.8h, v15.h[0]
mul v13.8h, v23.8h, v7.h[4]
str q29, [x1, #96]
mul v29.8h, v12.8h, v15.h[2]
sqrdmulh v14.8h, v12.8h, v15.h[3]
add v25.8h, v25.8h, v2.8h
mls v17.8h, v20.8h, v15.h[0]
sqrdmulh v20.8h, v23.8h, v7.h[5]
mls v11.8h, v9.8h, v15.h[0]
mul v23.8h, v3.8h, v7.h[6]
str q25, [x1, #144]
sqrdmulh v25.8h, v3.8h, v7.h[7]
mul v12.8h, v27.8h, v6.h[0]
sub v2.8h, v31.8h, v18.8h
mls v29.8h, v14.8h, v15.h[0]
sqrdmulh v27.8h, v27.8h, v6.h[1]
mls v13.8h, v20.8h, v15.h[0]
mul v20.8h, v28.8h, v6.h[2]
add v21.8h, v21.8h, v17.8h
sub v17.8h, v4.8h, v17.8h
sqrdmulh v28.8h, v28.8h, v6.h[3]
mul v9.8h, v8.8h, v6.h[4]
sub v3.8h, v18.8h, v11.8h
add v18.8h, v18.8h, v11.8h
sub v7.8h, v31.8h, v11.8h
mls v23.8h, v25.8h, v15.h[0]
sqrdmulh v25.8h, v8.8h, v6.h[5]
str q21, [x1, #112]
mul v21.8h, v3.8h, v15.h[2]
sqrdmulh v14.8h, v3.8h, v15.h[3]
add v4.8h, v0.8h, v29.8h
sub v24.8h, v24.8h, v29.8h
str q17, [x1, #128]
mls v12.8h, v27.8h, v15.h[0]
mul v29.8h, v26.8h, v6.h[6]
add v17.8h, v31.8h, v18.8h
sub v27.8h, v19.8h, v13.8h
mls v20.8h, v28.8h, v15.h[0]
sqrdmulh v31.8h, v26.8h, v6.h[7]
str q4, [x1, #160]
mls v9.8h, v25.8h, v15.h[0]
mul v18.8h, v10.8h, v1.h[0]
sub v28.8h, v13.8h, v23.8h
add v26.8h, v13.8h, v23.8h
sub v25.8h, v19.8h, v23.8h
str q24, [x1, #176]
mls v21.8h, v14.8h, v15.h[0]
sqrdmulh v23.8h, v10.8h, v1.h[1]
str q17, [x1, #192]
mul v24.8h, v28.8h, v15.h[2]
sqrdmulh v17.8h, v28.8h, v15.h[3]
add v28.8h, v19.8h, v26.8h
sub v19.8h, v22.8h, v12.8h
mls v29.8h, v31.8h, v15.h[0]
mul v26.8h, v30.8h, v1.h[2]
sub v31.8h, v12.8h, v20.8h
add v11.8h, v12.8h, v20.8h
sub v20.8h, v22.8h, v20.8h
sqrdmulh v30.8h, v30.8h, v1.h[3]
sub v8.8h, v5.8h, v9.8h
str q28, [x1, #240]
mul v28.8h, v31.8h, v15.h[2]
sqrdmulh v31.8h, v31.8h, v15.h[3]
add v3.8h, v7.8h, v21.8h
sub v21.8h, v2.8h, v21.8h
mls v24.8h, v17.8h, v15.h[0]
mls v18.8h, v23.8h, v15.h[0]
add v17.8h, v22.8h, v11.8h
sub v23.8h, v9.8h, v29.8h
add v22.8h, v9.8h, v29.8h
sub v29.8h, v5.8h, v29.8h
str q3, [x1, #208]
mls v26.8h, v30.8h, v15.h[0]
str q21, [x1, #224]
mls v28.8h, v31.8h, v15.h[0]
mul v30.8h, v23.8h, v15.h[2]
add v21.8h, v5.8h, v22.8h
str q17, [x1, #288]
sqrdmulh v31.8h, v23.8h, v15.h[3]
add v22.8h, v25.8h, v24.8h
sub v17.8h, v27.8h, v24.8h
sub v23.8h, v16.8h, v18.8h
str q21, [x1, #336]
sub v25.8h, v18.8h, v26.8h
add v27.8h, v18.8h, v26.8h
sub v24.8h, v16.8h, v26.8h
str q22, [x1, #256]
add v21.8h, v20.8h, v28.8h
sub v18.8h, v19.8h, v28.8h
str q17, [x1, #272]
mls v30.8h, v31.8h, v15.h[0]
mul v26.8h, v25.8h, v15.h[2]
add v22.8h, v16.8h, v27.8h
sqrdmulh v20.8h, v25.8h, v15.h[3]
str q21, [x1, #304]
str q18, [x1, #320]
str q22, [x1, #384]
add v19.8h, v29.8h, v30.8h
sub v28.8h, v8.8h, v30.8h
mls v26.8h, v20.8h, v15.h[0]
str q19, [x1, #352]
str q28, [x1, #368]
add v17.8h, v24.8h, v26.8h
sub v31.8h, v23.8h, v26.8h
str q17, [x1, #400]
str q31, [x1, #416]
ret
9:
br x17
