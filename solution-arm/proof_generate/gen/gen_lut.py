"""pass_lut::<1> (levels 0-3 of the 9721 binary transform from the nibble tables) as scheduled asm.
Tables of node n live in v16-v27 (pairs for tbl), k in v15; the scheduler gets v0-v14 and v28-v31.
Indices come packed (ntt_neon.rs packed_slot) and are zero-extended by ld1b; p7 = ptrue.h."""
import os
from sched import Ins, schedule, allocate
RESERVED = tuple(range(15, 28))
def packed_slot(s):
    # ntt_neon.rs packed_slot: byte offset of slot s in the packed index buffer
    k, t = s % 4, s // 4
    g, q = t // 16, t % 16
    return 16 * (2 * ((4 * g + q % 4) * 4 + k) + (q // 4) % 2) + 8 * (q // 8)
BASES = set()
def ld1b(s):
    # 8 bytes zero-extended to 8 u16 lanes; base m = ix + 64 + 128 m, imm in 8-byte units (-8..7)
    off = packed_slot(s)
    m = off // 128
    imm = (off - 128 * m - 64) // 8
    assert -8 <= imm <= 7
    BASES.add(m)
    return f'ld1b LBz{{d0}}.hRB, p7/z, [IB{m}X, #{imm}, mul vl]'
def build_node(n, steps):
    ins = []; cnt = [0]
    def v(nm):
        cnt[0] += 1; return f'{nm}{cnt[0]}'
    def A(op, a, b):
        d = v(op); ins.append(Ins('A', [d], [a, b], f'{op} v{{d0}}.8h, v{{s0}}.8h, v{{s1}}.8h')); return d
    def mulc_eps(a):
        lo, hi, r = v('lo'), v('hi'), v('r')
        ins.append(Ins('M', [lo], [a, 'k'], 'mul v{d0}.8h, v{s0}.8h, v{s1}.h[2]'))
        ins.append(Ins('M', [hi], [a, 'k'], 'sqrdmulh v{d0}.8h, v{s0}.8h, v{s1}.h[3]'))
        ins.append(Ins('M', [r], [lo, hi, 'k'], 'mls v{d0}.8h, v{s1}.8h, v{s2}.h[0]', tie=0))
        return r
    g = 0
    for j in steps:
        start = len(ins)
        offs = [j, j + 81, j + 27, j + 108, j + 54, j + 135]
        look = []
        for t, o in enumerate(offs):
            ix, val = v('ix'), v('lk')
            ins.append(Ins('L', [ix], [], ld1b(o)))
            ins.append(Ins('A', [val], [ix], f'tbl z{{d0}}.h, LBz{16+2*t}.h, z{17+2*t}.hRB, z{{s0}}.h'))
            look.append(val)
        x0, y0, x1, y1, x2, y2 = look
        a0, b0 = A('add', x0, y0), A('sub', x0, y0)
        a1, b1 = A('add', x1, y1), A('sub', x1, y1)
        a2, b2 = A('add', x2, y2), A('sub', x2, y2)
        # radix3_twiddled::<0>(a0, a1, a2)
        s = A('add', a1, a2); d = A('sub', a1, a2); e = mulc_eps(d)
        u0 = A('add', a0, s); u1 = A('add', A('sub', a0, a2), e); u2 = A('sub', A('sub', a0, a1), e)
        # radix3_sixth::<0>(b0, b1, b2)
        d2 = A('add', b1, b2); e2 = mulc_eps(d2)
        v0 = A('add', A('add', b0, b1), e2); v1 = A('add', A('sub', b0, b1), b2); v2 = A('sub', A('sub', b0, b2), e2)
        at = 162 * n + j
        for o, val in zip((0, 27, 54, 81, 108, 135), (u0, u1, u2, v0, v1, v2)):
            ins.append(Ins('T', [], [val], f'str q{{s0}}, [XPTR, #{16*(at+o)}]'))
        for x in ins[start:]: x.g = g
        g += 1
    return ins
def base_names(a):
    for m in range(12):
        a = a.replace(f'IB{m}X', f'{{i{m}}}')
    return a
def emit(per=27, K=8, B=17, width=6, ls=2):
    out = ['"ldr q15, [{kp}]"', '"ptrue p7.h"']
    for n in range(4):
        for t in range(6):
            out.append(f'"ldp q{16+2*t}, q{17+2*t}, [{{lut}}, #{n*192 + 32*t}]"')
        for c in range(0, 27, per):
            ins = build_node(n, list(range(c, min(27, c + per))))
            seq, _ = schedule(ins, skew=K, budget=B, cap=18, width=width, ls=ls)
            asm, _ = allocate(seq, {'k': 15}, set(), reserved=RESERVED)
            out += [f'"{base_names(a).replace("XPTR", "{x}").replace("LB", "{{").replace("RB", "}}")}"' for a in asm]
    return ',\n        '.join(out)
