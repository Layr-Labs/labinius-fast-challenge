"""pass_d (level 6, prime 9721 schedule: FIRST13 inputs 0 and 3, no reduction of d) for U groups of six."""
import sys
from sched import Ins, schedule, allocate
def build(U):
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
        y0 = A('add', a, s)
        y1 = A('add', A('sub', a, c1), e)
        y2 = A('sub', A('sub', a, b1), e)
        return [y0, y1, y2]
    def xaddr(byte):
        # ldp/stp immediate offsets end at 1008; the second half of a wide
        # unroll addresses through XPTR2 = XPTR + 1024.
        return ('XPTR2', byte - 1024) if byte >= 1024 else ('XPTR', byte)
    for g in range(U):
        start = len(ins)
        vin = [v('in') for _ in range(6)]
        for p in range(3):
            ptr, off = xaddr(96*g + 32*p)
            ins.append(Ins('L', [vin[2*p], vin[2*p+1]], [], f'ldp q{{d0}}, q{{d1}}, [{ptr}, #{off}]'))
        t = v('t'); low.add(t)
        ins.append(Ins('L', [t], [], f'ldr q{{d0}}, [TPTR, #{16*g}]'))
        ys = bfly(red13(vin[0]), vin[1], vin[2], t, 0) + bfly(red13(vin[3]), vin[4], vin[5], t, 4)
        for p in range(3):
            ptr, off = xaddr(96*g + 32*p)
            ins.append(Ins('T', [], [ys[2*p], ys[2*p+1]], f'stp q{{s0}}, q{{s1}}, [{ptr}, #{off}]'))
        for x in ins[start:]:
            x.g = g
    return ins, low
if __name__ == '__main__':
    U = int(sys.argv[1])
    ins, low = build(U)
    import os
    seq, cyc = schedule(ins, skew=int(os.environ.get('SKEW', 8)))
    asm, live = allocate(seq, {'k': 15}, low)
    print(f'U={U}: {len(ins)} instrs, scheduled {cyc} cycles ({cyc/U:.1f}/group), max live {live}', file=sys.stderr)
    print('\n'.join(asm))
