"""pass_c::<4> (levels 4 and 5 after the 9721 LUT front) as straight-line scheduled asm."""
import sys, os
from sched import Ins, schedule, allocate
NODES = [1, 11, 15, 16, 17, 19, 20, 22, 23, 25, 28, 30, 31, 32, 34, 35, 36, 37, 44, 50, 56, 59, 61, 65, 67, 70, 71]
def skip_lut5(n): return n in NODES
def kind(n):
    skip = sum(1 << g for g in range(3) if skip_lut5(3 * n + g))
    return (n != 1 and n != 15, skip)
def build(nodes):
    ins = []; low = set(); cnt = [0]
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
    def red13(a):
        sh, r = v('sh'), v('red')
        ins.append(Ins('S', [sh], [a], 'srshr v{d0}.8h, v{s0}.8h, #13'))
        ins.append(Ins('M', [r], [a, sh, 'k'], 'mls v{d0}.8h, v{s1}.8h, v{s2}.h[0]', tie=0))
        return r
    def bfly(a, b, c, t, o):
        b1 = mulc(b, t, o, o + 1); c1 = mulc(c, t, o + 2, o + 3)
        s = A('add', b1, c1); d = A('sub', b1, c1)
        e = mulc(d, 'k', 2, 3)
        return [A('add', a, s), A('add', A('sub', a, c1), e), A('sub', A('sub', a, b1), e)]
    g = 0
    for n in nodes:
        r4, skip = kind(n)
        t0, t1 = v('t0_'), v('t1_'); low |= {t0, t1}
        start = len(ins)
        ins.append(Ins('L', [t0], [], f'ldr q{{d0}}, [CPTR, #{32*n}]'))
        ins.append(Ins('L', [t1], [], f'ldr q{{d0}}, [CPTR, #{32*n+16}]'))
        for x in ins[start:]: x.g = g
        base = 27 * n
        for j in range(3):
            start = len(ins)
            at = base + j
            vv = []
            for i in range(9):
                d = v('in'); vv.append(d)
                ins.append(Ins('L', [d], [], f'ldr q{{d0}}, [XPTR, #{16*(at+3*i)}]'))
            if r4:
                for i in range(3): vv[i] = red13(vv[i])
            for i in range(3):
                vv[i], vv[i+3], vv[i+6] = bfly(vv[i], vv[i+3], vv[i+6], t0, 0)
            for q in range(3):
                if not (skip >> q & 1): vv[3*q] = red13(vv[3*q])
            outs = [bfly(vv[0], vv[1], vv[2], t0, 4), bfly(vv[3], vv[4], vv[5], t1, 0), bfly(vv[6], vv[7], vv[8], t1, 4)]
            for q, (a, b, c) in enumerate(outs):
                for o, val in zip((0, 3, 6), (a, b, c)):
                    ins.append(Ins('T', [], [val], f'str q{{s0}}, [XPTR, #{16*(at+9*q+o)}]'))
            for x in ins[start:]: x.g = g
            g += 1
    return ins, low
def emit(per=2, K=8, B=27):
    out = ['"ldr q15, [{kp}]",']
    for c in range(0, 24, per):
        ins, low = build(list(range(c, c + per)))
        seq, cyc = schedule(ins, skew=K, budget=B)
        asm, live = allocate(seq, {'k': 15}, low)
        out += [f'"{a.replace("XPTR", "{x}").replace("CPTR", "{c}")}",' for a in asm]
    return '\n'.join(out)
if __name__ == '__main__':
    print(emit(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])))
