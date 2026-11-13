"""The lazy inverse transform of prime 9721 ([`inverse_lazy_io2`]'s arithmetic on one row) as four
list-scheduled loops; writes ../src/ntt_neon_inv.rs. Run `python3 gen_inv.py` here.

Same operations as the Rust: ib3 = (r13(y0+y1+y2), mulc<0,1>((y0-y2)+e), mulc<2,3>((y0-y1)-e)) with
e = mulc<4,5>(y1-y2); ib2 = (r13(y0+y1), mulc<0,1>(y0-y1)); unsplit b = mulc<0,1>(u-z, uv),
a = r13(u - mulc<2,3>(b, uv)). mulc<W,T>(a,t) = mul(a,t[W]) - sqrdmulh(a,t[T]) * p (one mls)."""
import os, sys
os.environ.setdefault('MFIRST', '1')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sched import Ins, schedule, allocate

CLOB = ', '.join(f'out("v{r}") _' for r in range(32))

class B:
    def __init__(self):
        self.ins, self.low, self.cnt = [], set(), 0
    def v(self, n):
        self.cnt += 1
        return f'{n}{self.cnt}'
    def A(self, op, a, b):
        d = self.v(op); self.ins.append(Ins('A', [d], [a, b], f'{op} v{{d0}}.8h, v{{s0}}.8h, v{{s1}}.8h')); return d
    def mulc(self, a, t, w, tt):
        lo, hi, r = self.v('lo'), self.v('hi'), self.v('r')
        self.ins.append(Ins('M', [lo], [a, t], f'mul v{{d0}}.8h, v{{s0}}.8h, v{{s1}}.h[{w}]'))
        self.ins.append(Ins('M', [hi], [a, t], f'sqrdmulh v{{d0}}.8h, v{{s0}}.8h, v{{s1}}.h[{tt}]'))
        self.ins.append(Ins('M', [r], [lo, hi, 'k'], 'mls v{d0}.8h, v{s1}.8h, v{s2}.h[0]', tie=0))
        return r
    def r13(self, a):
        sh, r = self.v('sh'), self.v('red')
        self.ins.append(Ins('S', [sh], [a], 'srshr v{d0}.8h, v{s0}.8h, #13'))
        self.ins.append(Ins('M', [r], [a, sh, 'k'], 'mls v{d0}.8h, v{s1}.8h, v{s2}.h[0]', tie=0))
        return r
    def ib3(self, y0, y1, y2, t):
        a = self.r13(self.A('add', self.A('add', y0, y1), y2))
        e = self.mulc(self.A('sub', y1, y2), t, 4, 5)
        b = self.mulc(self.A('add', self.A('sub', y0, y2), e), t, 0, 1)
        c = self.mulc(self.A('sub', self.A('sub', y0, y1), e), t, 2, 3)
        return a, b, c
    def ib2(self, y0, y1, t):
        return self.r13(self.A('add', y0, y1)), self.mulc(self.A('sub', y0, y1), t, 0, 1)
    def load(self, ptr, off):
        d = self.v('in'); self.ins.append(Ins('L', [d], [], f'ldr q{{d0}}, [{ptr}, #{off}]')); return d
    def tw(self, ptr, off):
        d = self.v('t'); self.low.add(d); self.ins.append(Ins('L', [d], [], f'ldr q{{d0}}, [{ptr}, #{off}]')); return d
    def store(self, val, ptr, off):
        self.ins.append(Ins('T', [], [val], f'str q{{s0}}, [{ptr}, #{off}]'))
    def group(self, start, g):
        for x in self.ins[start:]: x.g = g

def pass1(U):
    """levels 6 and 5 for U level-4 nodes m (slots 9 m .. 9 m + 9): twiddles r3[3 m + g], r3[216 + m]"""
    b = B()
    for u in range(U):
        s0 = len(b.ins)
        w = [b.load('XP', 16 * (9 * u + i)) for i in range(9)]
        for g in range(3):
            t = b.tw('T1', 16 * (3 * u + g))
            w[3 * g], w[3 * g + 1], w[3 * g + 2] = b.ib3(w[3 * g], w[3 * g + 1], w[3 * g + 2], t)
        t = b.tw('T2', 16 * u)
        for j in range(3):
            w[j], w[j + 3], w[j + 6] = b.ib3(w[j], w[j + 3], w[j + 6], t)
        for i in range(9):
            b.store(w[i], 'XP', 16 * (9 * u + i))
        b.group(s0, u)
    return b

def pass_stride(U, span, stride, per):
    """one radix-3 level: for U nodes (span slots each, twiddle per node), (j, j + stride, j + 2 stride),
    j < stride; scheduling groups of `per` j"""
    b = B()
    g = 0
    for u in range(U):
        t = b.tw('T1', 16 * u)
        for j0 in range(0, stride, per):
            s0 = len(b.ins)
            for j in range(j0, j0 + per):
                base = span * u + j
                y = [b.load('XP', 16 * (base + stride * i)) for i in range(3)]
                y = b.ib3(*y, t)
                for i in range(3):
                    b.store(y[i], 'XP', 16 * (base + stride * i))
            b.group(s0, g); g += 1
    return b

def pass4(U):
    """levels 2 and 1 and the unsplit for U consecutive j (slots j + 81 i, i < 8)"""
    b = B()
    t2 = [b.tw('R2', 16 * n) for n in range(4)]
    t1 = [b.tw('R2', 16 * (4 + n)) for n in range(2)]
    uv = b.tw('UP', 0)
    for u in range(U):
        s0 = len(b.ins)
        w = [b.load('XP', 16 * (u + 81 * i)) for i in range(8)]
        for n in range(4):
            w[2 * n], w[2 * n + 1] = b.ib2(w[2 * n], w[2 * n + 1], t2[n])
        for n in range(2):
            for h in range(2):
                w[4 * n + h], w[4 * n + h + 2] = b.ib2(w[4 * n + h], w[4 * n + h + 2], t1[n])
        for i in range(4):
            uu, z = w[i], w[i + 4]
            bb = b.mulc(b.A('sub', uu, z), uv, 0, 1)
            a = b.r13(b.A('sub', uu, b.mulc(bb, uv, 2, 3)))
            w[i], w[i + 4] = a, bb
        for i in range(8):
            b.store(w[i], 'XP', 16 * (u + 81 * i))
        b.group(s0, u)
    return b

def sched_asm(b, K, Bud):
    seq, _ = schedule(b.ins, skew=K, budget=Bud)
    asm, _ = allocate(seq, {'k': 15}, b.low)
    return asm

def loop(asm, label, count, adv):
    lines = [f'"mov {{n}}, #{count}"', f'"{label}:"']
    lines += [f'"{a.replace("XP", "{x}").replace("T1", "{t1}").replace("T2", "{t2}").replace("R2", "{r2}").replace("UP", "{u}")}"' for a in asm]
    lines += [f'"add {{{r}}}, {{{r}}}, #{v}"' for r, v in adv] + ['"subs {n}, {n}, #1"', f'"b.ne {label}b"']
    return lines

def emit(U1=4, U2=2, U3=1, U4=3, K=22, Bud=27):
    lines = ['"ldr q15, [{kp}]"', '"mov {t1}, {r3}"', '"add {t2}, {r3}, #3456"']
    lines += loop(sched_asm(pass1(U1), K, Bud), 2, 72 // U1, [('x', 144 * U1), ('t1', 48 * U1), ('t2', 16 * U1)])
    lines += ['"sub {x}, {x}, #2592"', '"sub {x}, {x}, #2592"', '"sub {x}, {x}, #2592"', '"sub {x}, {x}, #2592"']
    lines += ['"add {t1}, {r3}, #1, lsl #12"', '"add {t1}, {t1}, #512"']
    lines += loop(sched_asm(pass_stride(U2, 27, 9, 3), K, Bud), 3, 24 // U2, [('x', 432 * U2), ('t1', 16 * U2)])
    lines += ['"sub {x}, {x}, #2592"', '"sub {x}, {x}, #2592"', '"sub {x}, {x}, #2592"', '"sub {x}, {x}, #2592"']
    lines += ['"add {t1}, {r3}, #1, lsl #12"', '"add {t1}, {t1}, #896"']
    lines += loop(sched_asm(pass_stride(U3, 81, 27, 3), K, Bud), 4, 8 // U3, [('x', 1296 * U3), ('t1', 16 * U3)])
    lines += ['"sub {x}, {x}, #2592"', '"sub {x}, {x}, #2592"', '"sub {x}, {x}, #2592"', '"sub {x}, {x}, #2592"']
    lines += loop(sched_asm(pass4(U4), K, Bud), 5, 81 // U4, [('x', 16 * U4)])
    body = ',\n        '.join(lines)
    return f"""// Generated by gen/gen_inv.py. Do not edit by hand.
use super::*;

/// [`inverse_lazy_io2`]'s arithmetic for prime 9721 on one row, in place, as four list-scheduled
/// loops (levels 6-5, level 4, level 3, levels 2-1 and the unsplit); generated by `gen/gen_inv.py`.
/// Inputs `|x| <= p` (as [`inverse_lazy`]); outputs as the Rust version's, bit for bit.
#[inline(never)]
pub unsafe fn inverse_asm(x: &mut [int16x8_t; N], inv: &LazyInverse, kern: &Kernel) {{
    core::arch::asm!(
        {body},
        x = inout(reg) x.as_mut_ptr() => _,
        r3 = in(reg) inv.r3.as_ptr(),
        r2 = in(reg) inv.r2.as_ptr(),
        u = in(reg) inv.u.as_ptr(),
        kp = in(reg) kern.k.as_ptr(),
        t1 = out(reg) _,
        t2 = out(reg) _,
        n = out(reg) _,
        {CLOB},
        options(nostack)
    );
}}
"""

if __name__ == '__main__':
    open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src', 'ntt_neon_inv.rs'), 'w').write(emit())
