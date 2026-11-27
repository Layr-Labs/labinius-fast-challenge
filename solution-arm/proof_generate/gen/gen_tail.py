"""Levels 4 to 6 of the 9721 transform fused per level-3 node (27 slots): the instructions of
pass_c::<4> (gen_pc) and of pass_d (gen_pd) for one node, reading the table front's output `x` once
and writing level 6 straight to `y`, so the node's 27 values stay in registers between the levels
instead of going through `x` (648 stores and 648 loads fewer per transform), and level 6's stores
into the sweep buffer are spread over the whole kernel instead of one short burst. Every value is
computed by the same instruction from the same operands as in the two passes, so the outputs are
the same.

Node n holds slots 27 n .. 27 n + 26 (p = slot - 27 n):
  level 4: butterflies (p, p + 9, p + 18), p < 9, twiddle kern.c[n] lanes 0..3; first inputs
           reduced (red13) unless the node is 1 or 15 (gen_pc.kind);
  level 5: butterflies (9 q + j, 9 q + j + 3, 9 q + j + 6), j < 3, twiddle kern.c[n] lanes 4..8
           (q = 0), 8..12 (q = 1), 12..16 (q = 2); first inputs reduced unless skip bit q;
  level 6: butterflies (3 m, 3 m + 1, 3 m + 2), m < 9, twiddle kern.d[M / 2] lanes 4 (M % 2),
           M = 9 n + m, every first input reduced (FIRST13), here read from kern.t6[n][m / 2]
           lanes 4 (m % 2) (the same pairs, laid out per node).
Registers: `k` (v15) holds kern.kc[n] = [p, v, eps, eps', the level-4 twiddle pair]: level 4
multiplies by lanes 4..7 of `k`, so only one twiddle register is live next to the 27 values.
"""
import sys
from sched import Ins, schedule, allocate
import gen_pc


def build(n, rel=False, omit6=False):
    """Node n's instructions. `rel`: addresses relative to per-node base pointers (x, y = slot
    27 n, c = kern.c[n], t = kern.t6[n]) so that every node of one reduction kind runs the same
    code; else absolute offsets from the transform's base pointers."""
    ins = []; low = set(); cnt = [0]; grp = [0]
    def v(nm):
        cnt[0] += 1; return f'{nm}{cnt[0]}'
    def emit(i):
        i.g = grp[0]; ins.append(i)
    def mulc(a, t, w, tt):
        lo, hi, r = v('lo'), v('hi'), v('r')
        emit(Ins('M', [lo], [a, t], f'mul v{{d0}}.8h, v{{s0}}.8h, v{{s1}}.h[{w}]'))
        emit(Ins('M', [hi], [a, t], f'sqrdmulh v{{d0}}.8h, v{{s0}}.8h, v{{s1}}.h[{tt}]'))
        emit(Ins('M', [r], [lo, hi, 'k'], 'mls v{d0}.8h, v{s1}.8h, v{s2}.h[0]', tie=0))
        return r
    def A(op, a, b, guard=False):
        d = v(op); insn = 'sq' + op if guard else op; emit(Ins('A', [d], [a, b], f'{insn} v{{d0}}.8h, v{{s0}}.8h, v{{s1}}.8h')); return d
    def red13(a):
        sh, r = v('sh'), v('red')
        emit(Ins('S', [sh], [a], 'srshr v{d0}.8h, v{s0}.8h, #13'))
        emit(Ins('M', [r], [a, sh, 'k'], 'mls v{d0}.8h, v{s1}.8h, v{s2}.h[0]', tie=0))
        return r
    def bfly(a, b, c, t, o, guard=False):
        b1 = mulc(b, t, o, o + 1); c1 = mulc(c, t, o + 2, o + 3)
        s = A('add', b1, c1); d = A('sub', b1, c1)
        e = mulc(d, 'k', 2, 3)
        return [A('add', a, s, guard), A('add', A('sub', a, c1, guard), e, guard),
                A('sub', A('sub', a, b1, guard), e, guard)]
    xb = 0 if rel else 16 * 27 * n
    cb = 0 if rel else 32 * n
    tb = 0 if rel else 80 * n
    def load(p):
        d = v('in'); emit(Ins('L', [d], [], f'ldr q{{d0}}, [XPTR, #{xb + 16*p}]')); return d
    def tw(fmt):
        d = v('tw'); low.add(d); emit(Ins('L', [d], [], fmt)); return d
    r4, skip = gen_pc.kind(n)
    x = [None] * 27
    # level 4 (twiddle in lanes 4..7 of k)
    for p in range(9):
        a = load(p); b = load(p + 9); c = load(p + 18)
        if r4: a = red13(a)
        x[p], x[p + 9], x[p + 18] = bfly(a, b, c, 'k', 4)
        grp[0] += 1
    # level 5
    t0 = t1 = None
    for q in range(3):
        for j in range(3):
            if q == 0 and t0 is None: t0 = tw(f'ldr q{{d0}}, [CPTR, #{cb}]')
            if q == 1 and t1 is None: t1 = tw(f'ldr q{{d0}}, [CPTR, #{cb + 16}]')
            t, o = ((t0, 4), (t1, 0), (t1, 4))[q]
            p = 9 * q + j
            a = x[p] if (skip >> q & 1) else red13(x[p])
            x[p], x[p + 3], x[p + 6] = bfly(a, x[p + 3], x[p + 6], t, o)
            grp[0] += 1
    # level 6
    dv = {}
    for m in range(9):
        if m // 2 not in dv:
            dv[m // 2] = tw(f'ldr q{{d0}}, [DPTR, #{tb + 16*(m//2)}]')
        a = x[3 * m] if omit6 else red13(x[3 * m])
        ys = bfly(a, x[3 * m + 1], x[3 * m + 2], dv[m // 2], 4 * (m % 2), guard=omit6)
        for i in range(3):
            emit(Ins('T', [], [ys[i]], f'str q{{s0}}, [YPTR, #{xb + 16*(3*m+i)}]'))
        grp[0] += 1
    return ins, low


def node_asm(n, rel, K, B, width=6, ls=2, cap=31, reserve=1, omit6=False):
    ins, low = build(n, rel)
    seq, cyc = schedule(ins, width=width, ls=ls, budget=B, skew=K, cap=cap, reserve=reserve)
    if omit6:
        # Keep the frozen order of every remaining operation. Each RED13 result
        # aliases its input. Guarded adds produce the same value when QC is clear.
        guarded, low = build(n, rel, omit6=True)
        wanted = iter(guarded)
        shifts = set()
        names = {'k': 'k'}
        kept = {}
        for old in ins:
            if old.g >= 18 and old.cls == 'S':
                shifts.add(old.dsts[0])
            elif old.tie == 0 and old.srcs[1] in shifts:
                names[old.dsts[0]] = names[old.srcs[0]]
            else:
                new = next(wanted)
                assert old.cls == new.cls and old.g == new.g and old.tie == new.tie
                assert [names[s] for s in old.srcs] == new.srcs
                names.update(zip(old.dsts, new.dsts))
                kept[old] = new
        assert len(shifts) == 9 and len(kept) == len(guarded) and next(wanted, None) is None
        seq = [kept[op] for op in seq if op in kept]
    asm, live = allocate(seq, {'k': 15}, low)
    return asm, cyc, live


def subst(a):
    return a.replace("XPTR", "{x}").replace("YPTR", "{y}").replace("CPTR", "{c}").replace("DPTR", "{t}")


def emit_straight(K=8, B=26, reserve=1):
    """All 24 nodes as straight-line code (absolute offsets)."""
    out = []
    for n in range(24):
        asm, _, _ = node_asm(n, False, K, B, reserve=reserve)
        out.append(f'"ldr q15, [{{kc}}, #{16*n}]"')
        out += [f'"{subst(a)}"' for a in asm]
    return out


def kinds():
    order = []
    for n in range(24):
        k = gen_pc.kind(n)
        if k not in order: order.append(k)
    return order


def emit_calls(K=8, B=26, reserve=1, width=6, ls=2, omit6=None):
    """One body per node reduction kind, called per node with the node's base pointers."""
    ks = kinds()
    out = []
    for n in range(24):
        out.append(f'"ldr q15, [{{kc}}, #{16*n}]"')
        out.append(f'"bl {20 + ks.index(gen_pc.kind(n))}f"')
        out += ['"add {x}, {x}, #432"', '"add {y}, {y}, #432"', '"add {c}, {c}, #32"', '"add {t}, {t}, #80"']
    out.append('"b 9f"')
    for i, k in enumerate(ks):
        n = next(m for m in range(24) if gen_pc.kind(m) == k)
        asm, _, _ = node_asm(n, True, K, B, width=width, ls=ls, reserve=reserve,
                              omit6=omit6(n) if omit6 else False)
        out.append(f'"{20 + i}:"')
        out += [f'"{subst(a)}"' for a in asm]
        out.append('"ret"')
    out.append('"9:"')
    return out


if __name__ == '__main__':
    K, B = int(sys.argv[1]), int(sys.argv[2])
    tot = 0; mx = 0
    for n in range(24):
        _, cyc, live = node_asm(n, False, K, B)
        tot += cyc; mx = max(mx, live)
    print(f'sched cycles {tot}, max live {mx}, kinds {len(kinds())}', file=sys.stderr)
    body = emit_calls(K, B) if len(sys.argv) > 3 and sys.argv[3] == 'calls' else emit_straight(K, B)
    print('\n'.join(a.strip('"') for a in body))
