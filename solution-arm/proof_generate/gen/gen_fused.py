"""Levels 4, 5 and 6 fused ([`pass_c_asm`] then [`pass_d_out`]) as one list-scheduled loop over pairs of
level-4 nodes; writes ../src/ntt_neon_fused.rs. Run `python3 gen_fused.py` here.

Same instructions per value as the two passes, so the same values bit for bit. The node-dependent
reductions of pass_c (lut_node_kind) become `mls` by a per-node flag lane (p or 0): a skipped reduction
computes x - sh * 0 = x. Of the three 9-slot groups j of a node, the outputs of the groups in SPILL go
through the scratch row (as pass_c stores them now); the others go straight into level 6 in registers.
Level 6 outputs are stored to `y` with stp pairs."""
import os, sys
os.environ.setdefault('MFIRST', '1')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sched import Ins, schedule, allocate
from gen_pc import kind

P = 9721
CLOB = ', '.join(f'out("v{r}") _' for r in range(32))

def build(spill=(0,), gl6=None, U=2, n0=None):
    def ya(a):
        # stp reaches [-1024, 1008]: y, y + 1024 (YPTR2) and y + 2048 (YPTR3) cover six nodes
        off = 16 * a
        return f"{['YPTR', 'YPTR2', 'YPTR3'][off // 1024]}, #{off % 1024}"
    ins, low, cnt = [], set(), [0]
    def v(n):
        cnt[0] += 1; return f'{n}{cnt[0]}'
    def mulc(a, t, w, tt):
        lo, hi, r = v('lo'), v('hi'), v('r')
        ins.append(Ins('M', [lo], [a, t], f'mul v{{d0}}.8h, v{{s0}}.8h, v{{s1}}.h[{w}]'))
        ins.append(Ins('M', [hi], [a, t], f'sqrdmulh v{{d0}}.8h, v{{s0}}.8h, v{{s1}}.h[{tt}]'))
        ins.append(Ins('M', [r], [lo, hi, 'k'], 'mls v{d0}.8h, v{s1}.8h, v{s2}.h[0]', tie=0))
        return r
    def A(op, a, b):
        d = v(op); ins.append(Ins('A', [d], [a, b], f'{op} v{{d0}}.8h, v{{s0}}.8h, v{{s1}}.8h')); return d
    def red(a, f='k', lane=0):
        sh, r = v('sh'), v('red')
        ins.append(Ins('S', [sh], [a], 'srshr v{d0}.8h, v{s0}.8h, #13'))
        ins.append(Ins('M', [r], [a, sh, f], f'mls v{{d0}}.8h, v{{s1}}.8h, v{{s2}}.h[{lane}]', tie=0))
        return r
    def bfly(a, b, c, t, o):
        b1 = mulc(b, t, o, o + 1); c1 = mulc(c, t, o + 2, o + 3)
        s = A('add', b1, c1); d = A('sub', b1, c1)
        e = mulc(d, 'k', 2, 3)
        return [A('add', a, s), A('add', A('sub', a, c1), e), A('sub', A('sub', a, b1), e)]
    def grp(start, g):
        for x in ins[start:]: x.g = g
    G = 0
    for u in range(U):
        s0 = len(ins)
        t0, t1, fl = v('t0_'), v('t1_'), v('fl'); low |= {t0, t1, fl}
        ins.append(Ins('L', [t0], [], f'ldr q{{d0}}, [CPTR, #{32*u}]'))
        ins.append(Ins('L', [t1], [], f'ldr q{{d0}}, [CPTR, #{32*u+16}]'))
        if n0 is None: ins.append(Ins('L', [fl], [], f'ldr q{{d0}}, [FPTR, #{16*u}]'))
        grp(s0, G)
        out = {}   # slot within node -> value (or ('mem', token))
        for j in range(3):
            s0 = len(ins)
            vv = [None] * 9
            for i in range(9):
                d = v('in'); vv[i] = d
                ins.append(Ins('L', [d], [], f'ldr q{{d0}}, [XPTR, #{16*(27*u + j + 3*i)}]'))
            # pass_c: r4 reduces inputs 0..2 when the node kind says so (flag lane 0 = p or 0)
            for i in range(3):
                if n0 is None: vv[i] = red(vv[i], fl, 0)
                elif kind(n0 + u)[0]: vv[i] = red(vv[i])
            for i in range(3):
                vv[i], vv[i+3], vv[i+6] = bfly(vv[i], vv[i+3], vv[i+6], t0, 0)
            for q in range(3):
                if n0 is None: vv[3*q] = red(vv[3*q], fl, 1 + q)
                elif not (kind(n0 + u)[1] >> q & 1): vv[3*q] = red(vv[3*q])
            outs = [bfly(vv[0], vv[1], vv[2], t0, 4), bfly(vv[3], vv[4], vv[5], t1, 0), bfly(vv[6], vv[7], vv[8], t1, 4)]
            for q, trip in enumerate(outs):
                for o, val in zip((0, 3, 6), trip):
                    s = j + 9*q + o
                    if j in spill:
                        tok = v('m')
                        x = Ins('T', [], [val], f'str q{{s0}}, [XPTR, #{16*(27*u + s)}]'); x.mdst = [tok]
                        ins.append(x); out[s] = ('mem', tok)
                    else:
                        out[s] = val
            grp(s0, G + j)
        # level 6 on the node's 9 triples (slots 3m', 3m'+1, 3m'+2), pass_d's schedule: FIRST13 input
        tws, res, done = {}, {}, set()
        base = 27*u
        for m in range(9):
            s0 = len(ins)
            T = 9*u + m
            gi, h = T // 2, T % 2
            if gi not in tws:
                t = v('t'); low.add(t); tws[gi] = t
                ins.append(Ins('L', [t], [], f'ldr q{{d0}}, [DPTR, #{16*gi}]'))
            y = []
            for e in range(3):
                val = out[3*m + e]
                if isinstance(val, tuple):
                    d = v('rl'); x = Ins('L', [d], [], f'ldr q{{d0}}, [XPTR, #{16*(27*u + 3*m + e)}]'); x.msrc = [val[1]]
                    ins.append(x); val = d
                y.append(val)
            r = bfly(red(y[0]), y[1], y[2], tws[gi], 4*h)
            for e in range(3): res[3*m + e] = r[e]
            # stores as soon as a stp pair (even absolute slot, its successor) is complete; node u
            # holds slots 27u..27u+26, so an odd node's first slot and an even node's last go alone
            for s in sorted(res):
                if s in done: continue
                a = base + s
                if a % 2 == 0 and s + 1 < 27:
                    if s + 1 in res:
                        ins.append(Ins('T', [], [res[s], res[s+1]], f'stp q{{s0}}, q{{s1}}, [{ya(a)}]')); done |= {s, s + 1}
                elif a % 2 == 0 or s == 0:
                    ins.append(Ins('T', [], [res[s]], f'str q{{s0}}, [{ya(a)}]')); done.add(s)
            grp(s0, G + 2 + 0.1 * (m + 1))
        assert len(done) == 27
        G += 3
    return ins, low

def flags():
    rows = []
    for n in range(24):
        r4, skip = kind(n)
        rows.append([P if r4 else 0] + [0 if (skip >> q & 1) else P for q in range(3)] + [0] * 4)
    return rows

def emit(spill=(0,), K=22, B=27, gl6=None, width=6, ls=2, lres=4, U=2):
    ins, low = build(spill, gl6, U)
    seq, cyc = schedule(ins, width=width, ls=ls, skew=K, budget=B, lres=lres)
    asm, live = allocate(seq, {'k': 15}, low)
    print(f"fused spill={spill} K={K} B={B} lres={lres}: {len(ins)} ins, {cyc} sched cycles per 2 nodes, live {live}", file=sys.stderr)
    lines = ['"ldr q15, [{kp}]"', f'"mov {{n}}, #{24 // U}"', '"2:"', '"add {y2}, {y}, #1024"', '"add {y3}, {y}, #2048"']
    lines += [f'"{a.replace("XPTR", "{x}").replace("YPTR3", "{y3}").replace("YPTR2", "{y2}").replace("YPTR", "{y}").replace("CPTR", "{c}").replace("DPTR", "{d}").replace("FPTR", "{f}")}"' for a in asm]
    lines += [f'"add {{x}}, {{x}}, #{432*U}"', f'"add {{y}}, {{y}}, #{432*U}"', f'"add {{c}}, {{c}}, #{32*U}"', f'"add {{d}}, {{d}}, #{72*U}"', f'"add {{f}}, {{f}}, #{16*U}"',
              '"subs {n}, {n}, #1"', '"b.ne 2b"']
    body = ',\n        '.join(lines)
    fl = ',\n    '.join('[' + ', '.join(map(str, r)) + ']' for r in flags())
    return f"""// Generated by gen/gen_fused.py. Do not edit by hand.
use super::*;

#[repr(C, align(16))]
struct Flags([[i16; 8]; 24]);
/// Per level-4 node: `[p or 0 (pass_c's input reduction), p or 0 (its three level-5 reductions), 0..]`.
static FLAGS: Flags = Flags([
    {fl}
]);

/// [`pass_c_asm`] then [`pass_d_out`] fused (levels 4, 5 and 6, two level-4 nodes per iteration):
/// reads `x` (also used as spill space) and writes `y`; the same values as the two passes.
#[inline(never)]
pub unsafe fn pass_cd_out(x: &mut [int16x8_t; N], y: &mut [int16x8_t; N], kern: &Kernel) {{
    core::arch::asm!(
        {body},
        x = inout(reg) x.as_mut_ptr() => _,
        y = inout(reg) y.as_mut_ptr() => _,
        c = inout(reg) kern.c.as_ptr() => _,
        d = inout(reg) kern.d.as_ptr() => _,
        f = inout(reg) FLAGS.0.as_ptr() => _,
        kp = in(reg) kern.k.as_ptr(),
        n = out(reg) _,
        y2 = out(reg) _,
        y3 = out(reg) _,
        {CLOB},
        options(nostack)
    );
}}
"""

def emit_lin(spill=(0, 1), K=400, B=30, width=6, ls=2, lres=3, U=2):
    """Straight-line form: 24/U chunks, each with the exact per-node reductions of `kind` (no flag
    lanes), list-scheduled separately, pointers advanced between chunks."""
    lines = ['"ldr q15, [{kp}]"']
    for c in range(24 // U):
        ins, low = build(spill, None, U, n0=U * c)
        seq, cyc = schedule(ins, width=width, ls=ls, skew=K, budget=B, lres=lres)
        asm, live = allocate(seq, {'k': 15}, low)
        lines += ['"add {y2}, {y}, #1024"', '"add {y3}, {y}, #2048"']
        lines += [f'"{a.replace("XPTR", "{x}").replace("YPTR3", "{y3}").replace("YPTR2", "{y2}").replace("YPTR", "{y}").replace("CPTR", "{c}").replace("DPTR", "{d}")}"' for a in asm]
        lines += [f'"add {{x}}, {{x}}, #{432*U}"', f'"add {{y}}, {{y}}, #{432*U}"', f'"add {{c}}, {{c}}, #{32*U}"', f'"add {{d}}, {{d}}, #{72*U}"']
    body = ',\n        '.join(lines)
    return f"""
/// [`pass_cd_out`] as straight-line code with the exact per-node reductions of pass_c (no flag
/// lanes): the same values as [`pass_c_asm`] then [`pass_d_out`], {U} level-4 nodes per scheduled chunk.
#[inline(never)]
pub unsafe fn pass_cd_lin(x: &mut [int16x8_t; N], y: &mut [int16x8_t; N], kern: &Kernel) {{
    core::arch::asm!(
        {body},
        x = inout(reg) x.as_mut_ptr() => _,
        y = inout(reg) y.as_mut_ptr() => _,
        c = inout(reg) kern.c.as_ptr() => _,
        d = inout(reg) kern.d.as_ptr() => _,
        kp = in(reg) kern.k.as_ptr(),
        y2 = out(reg) _,
        y3 = out(reg) _,
        {CLOB},
        options(nostack)
    );
}}
"""
if __name__ == '__main__':
    a = sys.argv[1:]
    spill = tuple(int(c) for c in a[0]) if a else (0,)
    K = int(a[1]) if len(a) > 1 else 22
    B = int(a[2]) if len(a) > 2 else 27
    open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src', 'ntt_neon_fused.rs'), 'w').write(emit(spill, K, B))
