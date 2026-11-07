.text
.p2align 4
.globl sms_slot_neon
.type sms_slot_neon, %function
sms_slot_neon:
cbz x3, .Lsms_slot_neon_return
sub sp, sp, #32
str d12, [sp, #0]
str d13, [sp, #8]
str d14, [sp, #16]
mov w9, #41472
ldp q16, q17, [x2, #0]
ldp q18, q19, [x2, #32]
cbnz w4, .Lsms_slot_neon_rows03_fresh
ldp q20, q21, [x2, #64]
ldp q22, q23, [x2, #96]
ldp q24, q25, [x2, #128]
ldp q26, q27, [x2, #160]
ldp q28, q29, [x2, #192]
ldp q30, q31, [x2, #224]
b .Lsms_slot_neon_rows03_setup
.Lsms_slot_neon_rows03_fresh:
movi v20.16b, #0
movi v21.16b, #0
movi v22.16b, #0
movi v23.16b, #0
movi v24.16b, #0
movi v25.16b, #0
movi v26.16b, #0
movi v27.16b, #0
movi v28.16b, #0
movi v29.16b, #0
movi v30.16b, #0
movi v31.16b, #0
.Lsms_slot_neon_rows03_setup:
mov x5, x0
mov x6, x1
lsr x7, x3, #1
cbz x7, .Lsms_slot_neon_rows03_tail
.p2align 4
.Lsms_slot_neon_rows03_loop:
// MODEL-BEGIN rows03
ldr q0, [x6, #0]
ldr q1, [x6, #32]
ldr q2, [x6, #64]
ldr q3, [x6, #96]
ldr q4, [x5, #0]
ldr q5, [x5, #10368]
ldr q6, [x5, #20736]
ldr q7, [x5, #31104]
add x5, x5, x9
dup v12.2d, v0.d[0]
smull v13.4s, v4.4h, v12.4h
smull2 v14.4s, v4.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v16.2d, v16.2d, v13.2d
smull v13.4s, v5.4h, v12.4h
smull2 v14.4s, v5.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v17.2d, v17.2d, v13.2d
smull v13.4s, v6.4h, v12.4h
smull2 v14.4s, v6.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v18.2d, v18.2d, v13.2d
smull v13.4s, v7.4h, v12.4h
smull2 v14.4s, v7.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v19.2d, v19.2d, v13.2d
dup v12.2d, v1.d[0]
smull v13.4s, v4.4h, v12.4h
smull2 v14.4s, v4.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v20.2d, v20.2d, v13.2d
smull v13.4s, v5.4h, v12.4h
smull2 v14.4s, v5.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v21.2d, v21.2d, v13.2d
smull v13.4s, v6.4h, v12.4h
smull2 v14.4s, v6.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v22.2d, v22.2d, v13.2d
smull v13.4s, v7.4h, v12.4h
smull2 v14.4s, v7.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v23.2d, v23.2d, v13.2d
dup v12.2d, v2.d[0]
smull v13.4s, v4.4h, v12.4h
smull2 v14.4s, v4.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v24.2d, v24.2d, v13.2d
smull v13.4s, v5.4h, v12.4h
smull2 v14.4s, v5.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v25.2d, v25.2d, v13.2d
smull v13.4s, v6.4h, v12.4h
smull2 v14.4s, v6.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v26.2d, v26.2d, v13.2d
smull v13.4s, v7.4h, v12.4h
smull2 v14.4s, v7.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v27.2d, v27.2d, v13.2d
dup v12.2d, v3.d[0]
smull v13.4s, v4.4h, v12.4h
smull2 v14.4s, v4.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v28.2d, v28.2d, v13.2d
smull v13.4s, v5.4h, v12.4h
smull2 v14.4s, v5.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v29.2d, v29.2d, v13.2d
smull v13.4s, v6.4h, v12.4h
smull2 v14.4s, v6.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v30.2d, v30.2d, v13.2d
smull v13.4s, v7.4h, v12.4h
smull2 v14.4s, v7.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v31.2d, v31.2d, v13.2d
ldr q4, [x5, #0]
ldr q5, [x5, #10368]
ldr q6, [x5, #20736]
ldr q7, [x5, #31104]
add x5, x5, x9
dup v12.2d, v0.d[1]
smull v13.4s, v4.4h, v12.4h
smull2 v14.4s, v4.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v16.2d, v16.2d, v13.2d
smull v13.4s, v5.4h, v12.4h
smull2 v14.4s, v5.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v17.2d, v17.2d, v13.2d
smull v13.4s, v6.4h, v12.4h
smull2 v14.4s, v6.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v18.2d, v18.2d, v13.2d
smull v13.4s, v7.4h, v12.4h
smull2 v14.4s, v7.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v19.2d, v19.2d, v13.2d
dup v12.2d, v1.d[1]
smull v13.4s, v4.4h, v12.4h
smull2 v14.4s, v4.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v20.2d, v20.2d, v13.2d
smull v13.4s, v5.4h, v12.4h
smull2 v14.4s, v5.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v21.2d, v21.2d, v13.2d
smull v13.4s, v6.4h, v12.4h
smull2 v14.4s, v6.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v22.2d, v22.2d, v13.2d
smull v13.4s, v7.4h, v12.4h
smull2 v14.4s, v7.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v23.2d, v23.2d, v13.2d
dup v12.2d, v2.d[1]
smull v13.4s, v4.4h, v12.4h
smull2 v14.4s, v4.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v24.2d, v24.2d, v13.2d
smull v13.4s, v5.4h, v12.4h
smull2 v14.4s, v5.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v25.2d, v25.2d, v13.2d
smull v13.4s, v6.4h, v12.4h
smull2 v14.4s, v6.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v26.2d, v26.2d, v13.2d
smull v13.4s, v7.4h, v12.4h
smull2 v14.4s, v7.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v27.2d, v27.2d, v13.2d
dup v12.2d, v3.d[1]
smull v13.4s, v4.4h, v12.4h
smull2 v14.4s, v4.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v28.2d, v28.2d, v13.2d
smull v13.4s, v5.4h, v12.4h
smull2 v14.4s, v5.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v29.2d, v29.2d, v13.2d
smull v13.4s, v6.4h, v12.4h
smull2 v14.4s, v6.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v30.2d, v30.2d, v13.2d
smull v13.4s, v7.4h, v12.4h
smull2 v14.4s, v7.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v31.2d, v31.2d, v13.2d
ldr q0, [x6, #16]
ldr q1, [x6, #48]
ldr q2, [x6, #80]
ldr q3, [x6, #112]
ldr q4, [x5, #0]
ldr q5, [x5, #10368]
ldr q6, [x5, #20736]
ldr q7, [x5, #31104]
add x5, x5, x9
dup v12.2d, v0.d[0]
smull v13.4s, v4.4h, v12.4h
smull2 v14.4s, v4.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v16.2d, v16.2d, v13.2d
smull v13.4s, v5.4h, v12.4h
smull2 v14.4s, v5.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v17.2d, v17.2d, v13.2d
smull v13.4s, v6.4h, v12.4h
smull2 v14.4s, v6.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v18.2d, v18.2d, v13.2d
smull v13.4s, v7.4h, v12.4h
smull2 v14.4s, v7.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v19.2d, v19.2d, v13.2d
dup v12.2d, v1.d[0]
smull v13.4s, v4.4h, v12.4h
smull2 v14.4s, v4.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v20.2d, v20.2d, v13.2d
smull v13.4s, v5.4h, v12.4h
smull2 v14.4s, v5.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v21.2d, v21.2d, v13.2d
smull v13.4s, v6.4h, v12.4h
smull2 v14.4s, v6.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v22.2d, v22.2d, v13.2d
smull v13.4s, v7.4h, v12.4h
smull2 v14.4s, v7.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v23.2d, v23.2d, v13.2d
dup v12.2d, v2.d[0]
smull v13.4s, v4.4h, v12.4h
smull2 v14.4s, v4.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v24.2d, v24.2d, v13.2d
smull v13.4s, v5.4h, v12.4h
smull2 v14.4s, v5.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v25.2d, v25.2d, v13.2d
smull v13.4s, v6.4h, v12.4h
smull2 v14.4s, v6.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v26.2d, v26.2d, v13.2d
smull v13.4s, v7.4h, v12.4h
smull2 v14.4s, v7.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v27.2d, v27.2d, v13.2d
dup v12.2d, v3.d[0]
smull v13.4s, v4.4h, v12.4h
smull2 v14.4s, v4.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v28.2d, v28.2d, v13.2d
smull v13.4s, v5.4h, v12.4h
smull2 v14.4s, v5.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v29.2d, v29.2d, v13.2d
smull v13.4s, v6.4h, v12.4h
smull2 v14.4s, v6.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v30.2d, v30.2d, v13.2d
smull v13.4s, v7.4h, v12.4h
smull2 v14.4s, v7.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v31.2d, v31.2d, v13.2d
ldr q4, [x5, #0]
ldr q5, [x5, #10368]
ldr q6, [x5, #20736]
ldr q7, [x5, #31104]
add x5, x5, x9
dup v12.2d, v0.d[1]
smull v13.4s, v4.4h, v12.4h
smull2 v14.4s, v4.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v16.2d, v16.2d, v13.2d
smull v13.4s, v5.4h, v12.4h
smull2 v14.4s, v5.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v17.2d, v17.2d, v13.2d
smull v13.4s, v6.4h, v12.4h
smull2 v14.4s, v6.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v18.2d, v18.2d, v13.2d
smull v13.4s, v7.4h, v12.4h
smull2 v14.4s, v7.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v19.2d, v19.2d, v13.2d
dup v12.2d, v1.d[1]
smull v13.4s, v4.4h, v12.4h
smull2 v14.4s, v4.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v20.2d, v20.2d, v13.2d
smull v13.4s, v5.4h, v12.4h
smull2 v14.4s, v5.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v21.2d, v21.2d, v13.2d
smull v13.4s, v6.4h, v12.4h
smull2 v14.4s, v6.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v22.2d, v22.2d, v13.2d
smull v13.4s, v7.4h, v12.4h
smull2 v14.4s, v7.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v23.2d, v23.2d, v13.2d
dup v12.2d, v2.d[1]
smull v13.4s, v4.4h, v12.4h
smull2 v14.4s, v4.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v24.2d, v24.2d, v13.2d
smull v13.4s, v5.4h, v12.4h
smull2 v14.4s, v5.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v25.2d, v25.2d, v13.2d
smull v13.4s, v6.4h, v12.4h
smull2 v14.4s, v6.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v26.2d, v26.2d, v13.2d
smull v13.4s, v7.4h, v12.4h
smull2 v14.4s, v7.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v27.2d, v27.2d, v13.2d
dup v12.2d, v3.d[1]
smull v13.4s, v4.4h, v12.4h
smull2 v14.4s, v4.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v28.2d, v28.2d, v13.2d
smull v13.4s, v5.4h, v12.4h
smull2 v14.4s, v5.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v29.2d, v29.2d, v13.2d
smull v13.4s, v6.4h, v12.4h
smull2 v14.4s, v6.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v30.2d, v30.2d, v13.2d
smull v13.4s, v7.4h, v12.4h
smull2 v14.4s, v7.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v31.2d, v31.2d, v13.2d
add x6, x6, #224
subs x7, x7, #1
b.ne .Lsms_slot_neon_rows03_loop
// MODEL-END rows03
.Lsms_slot_neon_rows03_tail:
tbz x3, #0, .Lsms_slot_neon_rows03_store
ldr q0, [x6, #0]
ldr q1, [x6, #32]
ldr q2, [x6, #64]
ldr q3, [x6, #96]
ldr q4, [x5, #0]
ldr q5, [x5, #10368]
ldr q6, [x5, #20736]
ldr q7, [x5, #31104]
add x5, x5, x9
dup v12.2d, v0.d[0]
smull v13.4s, v4.4h, v12.4h
smull2 v14.4s, v4.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v16.2d, v16.2d, v13.2d
smull v13.4s, v5.4h, v12.4h
smull2 v14.4s, v5.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v17.2d, v17.2d, v13.2d
smull v13.4s, v6.4h, v12.4h
smull2 v14.4s, v6.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v18.2d, v18.2d, v13.2d
smull v13.4s, v7.4h, v12.4h
smull2 v14.4s, v7.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v19.2d, v19.2d, v13.2d
dup v12.2d, v1.d[0]
smull v13.4s, v4.4h, v12.4h
smull2 v14.4s, v4.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v20.2d, v20.2d, v13.2d
smull v13.4s, v5.4h, v12.4h
smull2 v14.4s, v5.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v21.2d, v21.2d, v13.2d
smull v13.4s, v6.4h, v12.4h
smull2 v14.4s, v6.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v22.2d, v22.2d, v13.2d
smull v13.4s, v7.4h, v12.4h
smull2 v14.4s, v7.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v23.2d, v23.2d, v13.2d
dup v12.2d, v2.d[0]
smull v13.4s, v4.4h, v12.4h
smull2 v14.4s, v4.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v24.2d, v24.2d, v13.2d
smull v13.4s, v5.4h, v12.4h
smull2 v14.4s, v5.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v25.2d, v25.2d, v13.2d
smull v13.4s, v6.4h, v12.4h
smull2 v14.4s, v6.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v26.2d, v26.2d, v13.2d
smull v13.4s, v7.4h, v12.4h
smull2 v14.4s, v7.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v27.2d, v27.2d, v13.2d
dup v12.2d, v3.d[0]
smull v13.4s, v4.4h, v12.4h
smull2 v14.4s, v4.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v28.2d, v28.2d, v13.2d
smull v13.4s, v5.4h, v12.4h
smull2 v14.4s, v5.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v29.2d, v29.2d, v13.2d
smull v13.4s, v6.4h, v12.4h
smull2 v14.4s, v6.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v30.2d, v30.2d, v13.2d
smull v13.4s, v7.4h, v12.4h
smull2 v14.4s, v7.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v31.2d, v31.2d, v13.2d
ldr q4, [x5, #0]
ldr q5, [x5, #10368]
ldr q6, [x5, #20736]
ldr q7, [x5, #31104]
add x5, x5, x9
dup v12.2d, v0.d[1]
smull v13.4s, v4.4h, v12.4h
smull2 v14.4s, v4.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v16.2d, v16.2d, v13.2d
smull v13.4s, v5.4h, v12.4h
smull2 v14.4s, v5.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v17.2d, v17.2d, v13.2d
smull v13.4s, v6.4h, v12.4h
smull2 v14.4s, v6.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v18.2d, v18.2d, v13.2d
smull v13.4s, v7.4h, v12.4h
smull2 v14.4s, v7.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v19.2d, v19.2d, v13.2d
dup v12.2d, v1.d[1]
smull v13.4s, v4.4h, v12.4h
smull2 v14.4s, v4.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v20.2d, v20.2d, v13.2d
smull v13.4s, v5.4h, v12.4h
smull2 v14.4s, v5.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v21.2d, v21.2d, v13.2d
smull v13.4s, v6.4h, v12.4h
smull2 v14.4s, v6.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v22.2d, v22.2d, v13.2d
smull v13.4s, v7.4h, v12.4h
smull2 v14.4s, v7.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v23.2d, v23.2d, v13.2d
dup v12.2d, v2.d[1]
smull v13.4s, v4.4h, v12.4h
smull2 v14.4s, v4.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v24.2d, v24.2d, v13.2d
smull v13.4s, v5.4h, v12.4h
smull2 v14.4s, v5.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v25.2d, v25.2d, v13.2d
smull v13.4s, v6.4h, v12.4h
smull2 v14.4s, v6.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v26.2d, v26.2d, v13.2d
smull v13.4s, v7.4h, v12.4h
smull2 v14.4s, v7.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v27.2d, v27.2d, v13.2d
dup v12.2d, v3.d[1]
smull v13.4s, v4.4h, v12.4h
smull2 v14.4s, v4.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v28.2d, v28.2d, v13.2d
smull v13.4s, v5.4h, v12.4h
smull2 v14.4s, v5.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v29.2d, v29.2d, v13.2d
smull v13.4s, v6.4h, v12.4h
smull2 v14.4s, v6.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v30.2d, v30.2d, v13.2d
smull v13.4s, v7.4h, v12.4h
smull2 v14.4s, v7.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v31.2d, v31.2d, v13.2d
.Lsms_slot_neon_rows03_store:
stp q16, q17, [x2, #0]
stp q18, q19, [x2, #32]
stp q20, q21, [x2, #64]
stp q22, q23, [x2, #96]
stp q24, q25, [x2, #128]
stp q26, q27, [x2, #160]
stp q28, q29, [x2, #192]
stp q30, q31, [x2, #224]
cbnz w4, .Lsms_slot_neon_rows46_fresh
ldp q16, q17, [x2, #256]
ldp q18, q19, [x2, #288]
ldp q20, q21, [x2, #320]
ldp q22, q23, [x2, #352]
ldp q24, q25, [x2, #384]
ldp q26, q27, [x2, #416]
b .Lsms_slot_neon_rows46_setup
.Lsms_slot_neon_rows46_fresh:
movi v16.16b, #0
movi v17.16b, #0
movi v18.16b, #0
movi v19.16b, #0
movi v20.16b, #0
movi v21.16b, #0
movi v22.16b, #0
movi v23.16b, #0
movi v24.16b, #0
movi v25.16b, #0
movi v26.16b, #0
movi v27.16b, #0
.Lsms_slot_neon_rows46_setup:
mov x5, x0
mov x6, x1
lsr x7, x3, #1
cbz x7, .Lsms_slot_neon_rows46_tail
.p2align 4
.Lsms_slot_neon_rows46_loop:
// MODEL-BEGIN rows46
ldr q0, [x6, #128]
ldr q1, [x6, #160]
ldr q2, [x6, #192]
ldr q4, [x5, #0]
ldr q5, [x5, #10368]
ldr q6, [x5, #20736]
ldr q7, [x5, #31104]
add x5, x5, x9
dup v12.2d, v0.d[0]
smull v13.4s, v4.4h, v12.4h
smull2 v14.4s, v4.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v16.2d, v16.2d, v13.2d
smull v13.4s, v5.4h, v12.4h
smull2 v14.4s, v5.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v17.2d, v17.2d, v13.2d
smull v13.4s, v6.4h, v12.4h
smull2 v14.4s, v6.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v18.2d, v18.2d, v13.2d
smull v13.4s, v7.4h, v12.4h
smull2 v14.4s, v7.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v19.2d, v19.2d, v13.2d
dup v12.2d, v1.d[0]
smull v13.4s, v4.4h, v12.4h
smull2 v14.4s, v4.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v20.2d, v20.2d, v13.2d
smull v13.4s, v5.4h, v12.4h
smull2 v14.4s, v5.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v21.2d, v21.2d, v13.2d
smull v13.4s, v6.4h, v12.4h
smull2 v14.4s, v6.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v22.2d, v22.2d, v13.2d
smull v13.4s, v7.4h, v12.4h
smull2 v14.4s, v7.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v23.2d, v23.2d, v13.2d
dup v12.2d, v2.d[0]
smull v13.4s, v4.4h, v12.4h
smull2 v14.4s, v4.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v24.2d, v24.2d, v13.2d
smull v13.4s, v5.4h, v12.4h
smull2 v14.4s, v5.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v25.2d, v25.2d, v13.2d
smull v13.4s, v6.4h, v12.4h
smull2 v14.4s, v6.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v26.2d, v26.2d, v13.2d
smull v13.4s, v7.4h, v12.4h
smull2 v14.4s, v7.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v27.2d, v27.2d, v13.2d
ldr q4, [x5, #0]
ldr q5, [x5, #10368]
ldr q6, [x5, #20736]
ldr q7, [x5, #31104]
add x5, x5, x9
dup v12.2d, v0.d[1]
smull v13.4s, v4.4h, v12.4h
smull2 v14.4s, v4.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v16.2d, v16.2d, v13.2d
smull v13.4s, v5.4h, v12.4h
smull2 v14.4s, v5.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v17.2d, v17.2d, v13.2d
smull v13.4s, v6.4h, v12.4h
smull2 v14.4s, v6.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v18.2d, v18.2d, v13.2d
smull v13.4s, v7.4h, v12.4h
smull2 v14.4s, v7.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v19.2d, v19.2d, v13.2d
dup v12.2d, v1.d[1]
smull v13.4s, v4.4h, v12.4h
smull2 v14.4s, v4.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v20.2d, v20.2d, v13.2d
smull v13.4s, v5.4h, v12.4h
smull2 v14.4s, v5.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v21.2d, v21.2d, v13.2d
smull v13.4s, v6.4h, v12.4h
smull2 v14.4s, v6.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v22.2d, v22.2d, v13.2d
smull v13.4s, v7.4h, v12.4h
smull2 v14.4s, v7.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v23.2d, v23.2d, v13.2d
dup v12.2d, v2.d[1]
smull v13.4s, v4.4h, v12.4h
smull2 v14.4s, v4.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v24.2d, v24.2d, v13.2d
smull v13.4s, v5.4h, v12.4h
smull2 v14.4s, v5.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v25.2d, v25.2d, v13.2d
smull v13.4s, v6.4h, v12.4h
smull2 v14.4s, v6.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v26.2d, v26.2d, v13.2d
smull v13.4s, v7.4h, v12.4h
smull2 v14.4s, v7.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v27.2d, v27.2d, v13.2d
ldr q0, [x6, #144]
ldr q1, [x6, #176]
ldr q2, [x6, #208]
ldr q4, [x5, #0]
ldr q5, [x5, #10368]
ldr q6, [x5, #20736]
ldr q7, [x5, #31104]
add x5, x5, x9
dup v12.2d, v0.d[0]
smull v13.4s, v4.4h, v12.4h
smull2 v14.4s, v4.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v16.2d, v16.2d, v13.2d
smull v13.4s, v5.4h, v12.4h
smull2 v14.4s, v5.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v17.2d, v17.2d, v13.2d
smull v13.4s, v6.4h, v12.4h
smull2 v14.4s, v6.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v18.2d, v18.2d, v13.2d
smull v13.4s, v7.4h, v12.4h
smull2 v14.4s, v7.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v19.2d, v19.2d, v13.2d
dup v12.2d, v1.d[0]
smull v13.4s, v4.4h, v12.4h
smull2 v14.4s, v4.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v20.2d, v20.2d, v13.2d
smull v13.4s, v5.4h, v12.4h
smull2 v14.4s, v5.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v21.2d, v21.2d, v13.2d
smull v13.4s, v6.4h, v12.4h
smull2 v14.4s, v6.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v22.2d, v22.2d, v13.2d
smull v13.4s, v7.4h, v12.4h
smull2 v14.4s, v7.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v23.2d, v23.2d, v13.2d
dup v12.2d, v2.d[0]
smull v13.4s, v4.4h, v12.4h
smull2 v14.4s, v4.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v24.2d, v24.2d, v13.2d
smull v13.4s, v5.4h, v12.4h
smull2 v14.4s, v5.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v25.2d, v25.2d, v13.2d
smull v13.4s, v6.4h, v12.4h
smull2 v14.4s, v6.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v26.2d, v26.2d, v13.2d
smull v13.4s, v7.4h, v12.4h
smull2 v14.4s, v7.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v27.2d, v27.2d, v13.2d
ldr q4, [x5, #0]
ldr q5, [x5, #10368]
ldr q6, [x5, #20736]
ldr q7, [x5, #31104]
add x5, x5, x9
dup v12.2d, v0.d[1]
smull v13.4s, v4.4h, v12.4h
smull2 v14.4s, v4.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v16.2d, v16.2d, v13.2d
smull v13.4s, v5.4h, v12.4h
smull2 v14.4s, v5.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v17.2d, v17.2d, v13.2d
smull v13.4s, v6.4h, v12.4h
smull2 v14.4s, v6.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v18.2d, v18.2d, v13.2d
smull v13.4s, v7.4h, v12.4h
smull2 v14.4s, v7.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v19.2d, v19.2d, v13.2d
dup v12.2d, v1.d[1]
smull v13.4s, v4.4h, v12.4h
smull2 v14.4s, v4.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v20.2d, v20.2d, v13.2d
smull v13.4s, v5.4h, v12.4h
smull2 v14.4s, v5.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v21.2d, v21.2d, v13.2d
smull v13.4s, v6.4h, v12.4h
smull2 v14.4s, v6.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v22.2d, v22.2d, v13.2d
smull v13.4s, v7.4h, v12.4h
smull2 v14.4s, v7.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v23.2d, v23.2d, v13.2d
dup v12.2d, v2.d[1]
smull v13.4s, v4.4h, v12.4h
smull2 v14.4s, v4.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v24.2d, v24.2d, v13.2d
smull v13.4s, v5.4h, v12.4h
smull2 v14.4s, v5.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v25.2d, v25.2d, v13.2d
smull v13.4s, v6.4h, v12.4h
smull2 v14.4s, v6.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v26.2d, v26.2d, v13.2d
smull v13.4s, v7.4h, v12.4h
smull2 v14.4s, v7.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v27.2d, v27.2d, v13.2d
add x6, x6, #224
subs x7, x7, #1
b.ne .Lsms_slot_neon_rows46_loop
// MODEL-END rows46
.Lsms_slot_neon_rows46_tail:
tbz x3, #0, .Lsms_slot_neon_rows46_store
ldr q0, [x6, #128]
ldr q1, [x6, #160]
ldr q2, [x6, #192]
ldr q4, [x5, #0]
ldr q5, [x5, #10368]
ldr q6, [x5, #20736]
ldr q7, [x5, #31104]
add x5, x5, x9
dup v12.2d, v0.d[0]
smull v13.4s, v4.4h, v12.4h
smull2 v14.4s, v4.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v16.2d, v16.2d, v13.2d
smull v13.4s, v5.4h, v12.4h
smull2 v14.4s, v5.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v17.2d, v17.2d, v13.2d
smull v13.4s, v6.4h, v12.4h
smull2 v14.4s, v6.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v18.2d, v18.2d, v13.2d
smull v13.4s, v7.4h, v12.4h
smull2 v14.4s, v7.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v19.2d, v19.2d, v13.2d
dup v12.2d, v1.d[0]
smull v13.4s, v4.4h, v12.4h
smull2 v14.4s, v4.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v20.2d, v20.2d, v13.2d
smull v13.4s, v5.4h, v12.4h
smull2 v14.4s, v5.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v21.2d, v21.2d, v13.2d
smull v13.4s, v6.4h, v12.4h
smull2 v14.4s, v6.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v22.2d, v22.2d, v13.2d
smull v13.4s, v7.4h, v12.4h
smull2 v14.4s, v7.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v23.2d, v23.2d, v13.2d
dup v12.2d, v2.d[0]
smull v13.4s, v4.4h, v12.4h
smull2 v14.4s, v4.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v24.2d, v24.2d, v13.2d
smull v13.4s, v5.4h, v12.4h
smull2 v14.4s, v5.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v25.2d, v25.2d, v13.2d
smull v13.4s, v6.4h, v12.4h
smull2 v14.4s, v6.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v26.2d, v26.2d, v13.2d
smull v13.4s, v7.4h, v12.4h
smull2 v14.4s, v7.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v27.2d, v27.2d, v13.2d
ldr q4, [x5, #0]
ldr q5, [x5, #10368]
ldr q6, [x5, #20736]
ldr q7, [x5, #31104]
add x5, x5, x9
dup v12.2d, v0.d[1]
smull v13.4s, v4.4h, v12.4h
smull2 v14.4s, v4.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v16.2d, v16.2d, v13.2d
smull v13.4s, v5.4h, v12.4h
smull2 v14.4s, v5.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v17.2d, v17.2d, v13.2d
smull v13.4s, v6.4h, v12.4h
smull2 v14.4s, v6.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v18.2d, v18.2d, v13.2d
smull v13.4s, v7.4h, v12.4h
smull2 v14.4s, v7.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v19.2d, v19.2d, v13.2d
dup v12.2d, v1.d[1]
smull v13.4s, v4.4h, v12.4h
smull2 v14.4s, v4.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v20.2d, v20.2d, v13.2d
smull v13.4s, v5.4h, v12.4h
smull2 v14.4s, v5.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v21.2d, v21.2d, v13.2d
smull v13.4s, v6.4h, v12.4h
smull2 v14.4s, v6.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v22.2d, v22.2d, v13.2d
smull v13.4s, v7.4h, v12.4h
smull2 v14.4s, v7.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v23.2d, v23.2d, v13.2d
dup v12.2d, v2.d[1]
smull v13.4s, v4.4h, v12.4h
smull2 v14.4s, v4.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v24.2d, v24.2d, v13.2d
smull v13.4s, v5.4h, v12.4h
smull2 v14.4s, v5.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v25.2d, v25.2d, v13.2d
smull v13.4s, v6.4h, v12.4h
smull2 v14.4s, v6.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v26.2d, v26.2d, v13.2d
smull v13.4s, v7.4h, v12.4h
smull2 v14.4s, v7.8h, v12.8h
saddlp v13.2d, v13.4s
saddlp v14.2d, v14.4s
addp v13.2d, v13.2d, v14.2d
add v27.2d, v27.2d, v13.2d
.Lsms_slot_neon_rows46_store:
stp q16, q17, [x2, #256]
stp q18, q19, [x2, #288]
stp q20, q21, [x2, #320]
stp q22, q23, [x2, #352]
stp q24, q25, [x2, #384]
stp q26, q27, [x2, #416]
ldr d12, [sp, #0]
ldr d13, [sp, #8]
ldr d14, [sp, #16]
add sp, sp, #32
.Lsms_slot_neon_return:
ret
.size sms_slot_neon, .-sms_slot_neon
.section .note.GNU-stack,"",%progbits
