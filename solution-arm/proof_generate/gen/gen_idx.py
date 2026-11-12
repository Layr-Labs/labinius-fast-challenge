"""indices (the nibble table indices of one element of eight lanes) as list-scheduled asm: the
same operations as `indices_pair::<0>` and `indices_pair::<1>` with their `index_block`s.
Constants pinned: v28 = 0x0f, v29 = 0x55, v30 = 0x33, v31 = 0 (all bytes). Output packed
(ntt_neon.rs packed_slot), as kaankolcu's index_block writes it."""
from sched import Ins
GROUPS = 162
PIN = {'m55': 29, 'm33': 30, 'zero': 31, 'm0f': 28}
def build(only=None):
    ins = []; cnt = [0]
    def v(nm):
        cnt[0] += 1; return f'{nm}{cnt[0]}'
    def op(cls, fmt, *srcs, tie=None):
        d = v('t'); ins.append(Ins(cls, [d], list(srcs), fmt, tie)); return d
    zip1 = lambda w, x, y: op('A', f'zip1 v{{d0}}.{w}, v{{s0}}.{w}, v{{s1}}.{w}', x, y)
    zip2 = lambda w, x, y: op('A', f'zip2 v{{d0}}.{w}, v{{s0}}.{w}, v{{s1}}.{w}', x, y)
    ext8 = lambda x, y: op('A', 'ext v{d0}.16b, v{s0}.16b, v{s1}.16b, #8', x, y)
    ushr = lambda x, n: op('S', f'ushr v{{d0}}.16b, v{{s0}}.16b, #{n}', x)
    shl = lambda x, n: op('S', f'shl v{{d0}}.16b, v{{s0}}.16b, #{n}', x)
    orr = lambda x, y: op('A', 'orr v{d0}.16b, v{s0}.16b, v{s1}.16b', x, y)
    andv = lambda x, y: op('A', 'and v{d0}.16b, v{s0}.16b, v{s1}.16b', x, y)
    def swap(a, b, d, mask):
        # (bsl(mask, a, b << d), bsl(mask, a >> d, b))
        if b == 'zero':
            # bsl(mask, a, 0) = a & mask, bsl(mask, a >> d, 0) = (a >> d) & mask
            return andv(a, mask), andv(ushr(a, d), mask)
        sb = shl(b, d); ua = ushr(a, d)
        o0 = op('A', 'bit v{d0}.16b, v{s1}.16b, v{s2}.16b', sb, a, mask, tie=0)
        o1 = op('A', 'bif v{d0}.16b, v{s1}.16b, v{s2}.16b', ua, b, mask, tie=0)
        return o0, o1
    stores = {}
    for P in ((0, 1) if only is None else (only,)):
        g = P
        start = len(ins)
        rows = []
        for i in range(2):
            k = P + 2 * i
            r = []
            for c in range(8):
                d = v('ld'); ins.append(Ins('L', [d], [], f'ldr q{{d0}}, [W{c}, #{16*k}]')); r.append(d)
            a = [(zip1 if i_ < 4 else zip2)('16b', r[2 * (i_ % 4)], r[2 * (i_ % 4) + 1]) for i_ in range(8)]
            b = []
            for i_ in range(8):
                h = i_ // 4; x = a[4 * h + 2 * ((i_ % 4) // 2)]; y = a[4 * h + 2 * ((i_ % 4) // 2) + 1]
                b.append((zip1 if i_ % 2 == 0 else zip2)('8h', x, y))
            row = []
            for vv in range(8):
                h = vv // 4; z = (vv % 4) // 2; x = b[4 * h + z]; y = b[4 * h + 2 + z]
                row.append((zip1 if vv % 2 == 0 else zip2)('4s', x, y))
            row.append('zero')
            rows.append(row)
        for G in range(3):
            for kk, K in enumerate((0, 2) if P == 0 else (1, 3)):
                gstart = len(ins)
                own, other = rows[K // 2], rows[1 - K // 2]
                a = own[G]
                b = ext8(other[2 + G], other[3 + G])
                if K >= 2:
                    b = orr(ushr(b, 1), shl(other[3 + G], 7))
                c0 = own[5 + G]
                c1 = ext8(c0, own[6 + G])
                c = orr(ushr(c0, 1), shl(c1, 7))
                if G == 0:
                    last = ext8(other[7], 'zero')
                    dd = ushr(last, 1 if K < 2 else 2)
                else:
                    dd = 'zero'
                a, b = swap(a, b, 1, 'm55')
                c, dd = swap(c, dd, 1, 'm55')
                a, c = swap(a, c, 2, 'm33')
                b, dd = swap(b, dd, 2, 'm33')
                for vec, r_ in ((a, 0), (b, 1), (c, 2), (dd, 3)):
                    # packed (ntt_neon.rs packed_slot): low nibbles at 2 at, high nibbles at 2 at + 1
                    at = 2 * ((G * 4 + r_) * 4 + K)
                    lo = andv(vec, 'm0f'); hi = ushr(vec, 4)
                    ins.append(Ins('T', [], [lo], f'str q{{s0}}, [IPTR, #{16*at}]'))
                    ins.append(Ins('T', [], [hi], f'str q{{s0}}, [IPTR, #{16*(at+1)}]'))
                for x in ins[gstart:]: x.g = 7 * P + 2 * G + kk + 1
        for x in ins[start:]:
            if not hasattr(x, 'g'): x.g = 7 * P
    # remove ops whose results are never used (zips of unstored indices), repeatedly
    while True:
        used = set(s for x in ins for s in x.srcs)
        keep = [x for x in ins if x.cls == 'T' or any(d in used for d in x.dsts)]
        if len(keep) == len(ins): break
        ins = keep
    return ins
