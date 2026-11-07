#!/usr/bin/env python3
"""Generate the fixed SVE128 sink/front schedule in sms_pipeline.rs.

Only instruction scheduling is generated: arithmetic and layout follow indices,
pass_lut::<1>, and slot_sink in the adjoining Rust sources. Run with Python 3.
"""
from pathlib import Path


def interleave(a, b):
    out = []
    i = j = 0
    while i < len(a) or j < len(b):
        if j == len(b) or (i < len(a) and i * len(b) <= j * len(a)):
            out.append(a[i]); i += 1
        else:
            out.append(b[j]); j += 1
    return out


def init_rows(row, label):
    a = [f"ldp q16, q17, [{{a}}, #{row * 64}]", f"ldp q18, q19, [{{a}}, #{row * 64 + 32}]"] if row == 0 else []
    regs = range(20, 24) if row == 0 else range(16, 24 if row < 6 else 20)
    a += [f"cbnz {{fresh}}, {label}f"]
    for v in regs:
        a.append(f"ldr q{v}, [{{a}}, #{row * 64 + (v - 16) * 16}]")
    a += [f"b {label + 1}f", f"{label}:"]
    a += [f"movi v{v}.16b, #0" for v in regs]
    return a + [f"{label + 1}:", "mov {zp}, {z}", "mov {kp}, {k}"]


def store_rows(row):
    return [f"stp q{v}, q{v+1}, [{{a}}, #{row * 64 + (v - 16) * 16}]" for v in range(16, 24 if row < 6 else 20, 2)]


def tile(row):
    a = [f"ldp q0, q1, [{{kp}}, #{row * 16}]"] if row < 6 else ["ldr q0, [{kp}, #96]"]
    for half in range(2):
        a += [f"ldr q{4+q}, [{{zp}}, #{q * 10368}]" for q in range(4)]
        a += ["add {zp}, {zp}, {xs4}"]
        for r in range(2 if row < 6 else 1):
            a += [f"sdot z{16+r*4+q}.d, z{4+q}.h, z{r}.h[{half}]" for q in range(4)]
    return a + ["add {kp}, {kp}, {ks}"]


def index_front():
    a = []
    # Transpose each word independently, spilling 4 x 9 byte pairs. This keeps
    # the index front inside the 18 registers left by the two-row sink.
    for k in range(4):
        for c in range(8):
            a += [f"ldr {{wp}}, [{{words}}, #{c*8}]", f"ldr q{8+c}, [{{wp}}, #{16*k}]"]
        for i in range(8):
            a += [f"zip{1 if i < 4 else 2} v{24+i}.16b, v{8+2*(i%4)}.16b, v{9+2*(i%4)}.16b"]
        for i in range(8):
            x = 24 + 4*(i//4) + 2*((i%4)//2)
            a += [f"zip{1+i%2} v{8+i}.8h, v{x}.8h, v{x+1}.8h"]
        for v in range(8):
            x = 8 + 4*(v//4) + (v%4)//2
            a += [f"zip{1+v%2} v{24+v}.4s, v{x}.4s, v{x+2}.4s"]
        a += [f"str q{24+v}, [{{tmp}}, #{(9*k+v)*16}]" for v in range(8)]
        a += ["movi v31.16b, #0", f"str q31, [{{tmp}}, #{(9*k+8)*16}]"]
    a += ["movi v2.16b, #85", "movi v3.16b, #51", "movi v29.16b, #0", "movi v30.16b, #30", "movi v31.16b, #1"]
    def load(v, k, row):
        return f"ldr q{v}, [{{tmp}}, #{(9*k+row)*16}]"
    for k in range(4):
        other = k ^ 2
        for g in range(3):
            a += [load(8,k,g), load(12,other,2+g), load(13,other,3+g), "ext v9.16b, v12.16b, v13.16b, #8"]
            if k >= 2:
                a += ["ushr v9.16b, v9.16b, #1", "shl v13.16b, v13.16b, #7", "orr v9.16b, v9.16b, v13.16b"]
            a += [load(12,k,5+g), load(13,k,6+g), "ext v13.16b, v12.16b, v13.16b, #8", "ushr v10.16b, v12.16b, #1", "shl v13.16b, v13.16b, #7", "orr v10.16b, v10.16b, v13.16b"]
            if g == 0:
                a += [load(11,other,7), "ext v11.16b, v11.16b, v29.16b, #8", f"ushr v11.16b, v11.16b, #{1 if k < 2 else 2}"]
            else:
                a += ["movi v11.16b, #0"]
            for x,y,d,mask in [(8,9,1,2),(10,11,1,2),(8,10,2,3),(9,11,2,3)]:
                a += [f"shl v24.16b, v{y}.16b, #{d}", f"mov v25.16b, v{mask}.16b", f"bsl v25.16b, v{x}.16b, v24.16b", f"ushr v24.16b, v{x}.16b, #{d}", f"mov v26.16b, v{mask}.16b", f"bsl v26.16b, v24.16b, v{y}.16b", f"mov v{x}.16b, v25.16b", f"mov v{y}.16b, v26.16b"]
            for r,v in enumerate(range(8,12)):
                a += [f"shl v12.16b, v{v}.16b, #1", "and v12.16b, v12.16b, v30.16b", f"ushr v13.16b, v{v}.16b, #3", "and v13.16b, v13.16b, v30.16b", "orr v14.16b, v12.16b, v31.16b", "orr v15.16b, v13.16b, v31.16b"]
                for q,lo,odd,half in [(r,12,14,1),(r+4,13,15,1),(r+8,12,14,2),(r+12,13,15,2)]:
                    j = 4*(16*g+q)+k
                    if j < 162:
                        a += [f"zip{half} v24.16b, v{lo}.16b, v{odd}.16b", f"str q24, [{{nib}}, #{16*j}]"]
    return a


def lut_front():
    a = []
    for i,off in enumerate([0,81,27,108,54,135]):
        a += [f"ldr q{24+i}, [{{ip}}, #{off*16}]", f"ldp q2, q3, [{{tp}}, #{i*32}]", f"tbl v{24+i}.16b, {{v2.16b, v3.16b}}, v{24+i}.16b"]
    for i in range(3):
        a += [f"add v{8+2*i}.8h, v{24+2*i}.8h, v{25+2*i}.8h", f"sub v{9+2*i}.8h, v{24+2*i}.8h, v{25+2*i}.8h"]
    a += ["add v24.8h, v10.8h, v12.8h", "sub v25.8h, v10.8h, v12.8h", "mul v26.8h, v25.8h, v15.h[2]", "sqrdmulh v25.8h, v25.8h, v15.h[3]", "mls v26.8h, v25.8h, v15.h[0]", "add v27.8h, v8.8h, v24.8h", "sub v28.8h, v8.8h, v12.8h", "add v28.8h, v28.8h, v26.8h", "sub v29.8h, v8.8h, v10.8h", "sub v29.8h, v29.8h, v26.8h", "str q27, [{op}]", "str q28, [{op}, #432]", "str q29, [{op}, #864]", "add v24.8h, v11.8h, v13.8h", "mul v26.8h, v24.8h, v15.h[2]", "sqrdmulh v24.8h, v24.8h, v15.h[3]", "mls v26.8h, v24.8h, v15.h[0]", "add v27.8h, v9.8h, v11.8h", "add v27.8h, v27.8h, v26.8h", "sub v28.8h, v9.8h, v11.8h", "add v28.8h, v28.8h, v13.8h", "sub v14.8h, v9.8h, v13.8h", "sub v14.8h, v14.8h, v26.8h", "str q27, [{op}, #1296]", "str q28, [{op}, #1728]", "str q14, [{op}, #2160]"]
    return a


def asm(lines, operands):
    # Conditional accumulator initialization is atomic: a branch must never
    # jump over index instructions placed between its source and target.
    lines = [s for line in lines for s in (line if isinstance(line, list) else [line])]
    # Double the architectural register-list braces for Rust asm templates.
    lines = [l.replace('{v2.16b, v3.16b}', '{{v2.16b, v3.16b}}') for l in lines]
    return '    core::arch::asm!(\n' + ''.join(f'        "{l}",\n' for l in lines) + operands + ''.join(f'        out("v{i}") _,\n' for i in range(32)) + '        options(nostack)\n    );\n'


common = '''        a = in(reg) acc, z = in(reg) z, k = in(reg) keys,
        fresh = in(reg) usize::from(FRESH),
        xs4 = in(reg) 4 * N * 16, ks = in(reg) N * ROWS * LOCK * 2,
        zp = out(reg) _, kp = out(reg) _,
'''
sink = []
for row in [0,2,4,6]:
    sink += [init_rows(row, 20+row*2)]
    sink += tile(row)*8
    sink += store_rows(row)

source = '''// Generated by sms_pipeline_gen.py; edit the generator, then regenerate.
// SVE indexed halfword dot products consume the low 128 bits, as in slot_sink.
use super::*;

#[inline(never)]
unsafe fn indices_sink<const FRESH: bool>(
    z: *const int16x8_t, keys: *const i16, acc: *mut int64x2_t,
    words: &[*const u8; LANES], nib: *mut uint8x16_t, tmp: *mut int16x8_t,
) {
'''
source += asm(interleave(sink, index_front()), common + '''        words = in(reg) words.as_ptr(), nib = in(reg) nib, tmp = in(reg) tmp,
        wp = out(reg) _,
''') + '}\n\n'

source += '''/// One slot's seven-row sink plus twelve successive pass_lut steps.
#[inline(never)]
unsafe fn lut_sink<const FRESH: bool>(
    z: *const int16x8_t, keys: *const i16, acc: *mut int64x2_t,
    nib: *const uint8x16_t, x: *mut int16x8_t, kern: &Kernel, step: usize,
) {
    let (node, j) = (step / 27, step % 27);
'''
lines = ['ldr q15, [{constants}]']
for row in [0,2,4]:
    lines += init_rows(row,20+row*2) + ['mov {count}, #4',f'{40+row}:']
    lines += interleave(tile(row)*2, lut_front())
    lines += ['add {ip}, {ip}, #16', 'add {op}, {op}, #16', 'subs {left}, {left}, #1', f'b.ne {50+row}f', 'mov {left}, #27', 'sub {ip}, {ip}, #432', 'add {op}, {op}, #2160', 'add {tp}, {tp}, #192',f'{50+row}:', 'subs {count}, {count}, #1',f'b.ne {40+row}b']
    lines += store_rows(row)
lines += init_rows(6,32) + ['mov {count}, #8','46:'] + tile(6) + ['subs {count}, {count}, #1','b.ne 46b'] + store_rows(6)
source += asm(lines, common + '''        constants = in(reg) kern.k.as_ptr(),
        ip = inout(reg) nib.add(j) => _, op = inout(reg) x.add(node * 162 + j) => _,
        tp = inout(reg) kern.lut.as_ptr().add(node) => _, left = inout(reg) 27 - j => _,
        count = out(reg) _,
''') + '}\n\n'
source += '''/// Consume one full current sweep while preparing the next sweep's levels 0-3.
/// Ten slots per next transform: one for indices and nine for twelve LUT steps.
/// The final eight slots have no front work left and use the original sink.
pub(super) unsafe fn sink_front<const FRESH: bool>(
    xs: &[[int16x8_t; N]; SWEEP], keys: *const i16, acc: &mut [int64x2_t],
    words: &[(*const u8, usize); LANES], elements: usize, next: usize,
    ys: &mut [[int16x8_t; N]; SWEEP], nib: &mut [uint8x16_t; GROUPS],
    tmp: &mut [int16x8_t; BITS], kern: &Kernel,
) {
    const { assert!(SINK == 8 && 10 * SWEEP <= N); }
    let z = xs.as_ptr().cast::<int16x8_t>();
    for t in 0..SWEEP {
        let at = pass_words(words, elements, next, t);
        let s = 10 * t;
        indices_sink::<FRESH>(z.add(s), keys.add(s * ROWS * LOCK), acc.as_mut_ptr().add(idx(s, 0)), &at, nib.as_mut_ptr(), tmp.as_mut_ptr());
        for b in 0..9 {
            let s = s + 1 + b;
            lut_sink::<FRESH>(z.add(s), keys.add(s * ROWS * LOCK), acc.as_mut_ptr().add(idx(s, 0)), nib.as_ptr(), ys[t].as_mut_ptr(), kern, 12 * b);
        }
    }
    for s in 10 * SWEEP..N {
        if FRESH {
            slot_sink_fresh(z.add(s), keys.add(s * ROWS * LOCK), acc.as_mut_ptr().add(idx(s, 0)), SINK);
        } else {
            slot_sink(z.add(s), keys.add(s * ROWS * LOCK), acc.as_mut_ptr().add(idx(s, 0)), SINK);
        }
    }
}
'''
Path(__file__).with_name('sms_pipeline.rs').write_text(source)
