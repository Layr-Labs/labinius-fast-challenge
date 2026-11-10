import os
"""List scheduler + linear-scan register allocator for NEON kernels on Neoverse V2.
Ins(cls, dsts, srcs, fmt, tie): fmt placeholders {d0} {d1} {s0} {s1} ... are register NUMBERS,
write them as v{d0}.8h or q{d0}. Classes: M mul-class (V0/V2), S shift (V1/V3), A any vector pipe,
L load, T store. tie: index of the source whose register d0 reuses (destructive ops like mls)."""
LAT = {'M': 4, 'S': 2, 'A': 2, 'L': 6, 'T': 1}

class Ins:
    def __init__(self, cls, dsts, srcs, fmt, tie=None):
        self.cls, self.dsts, self.srcs, self.fmt, self.tie = cls, list(dsts), list(srcs), fmt, tie

def schedule(ins, width=6, ls=2, budget=27, skew=8, reserve=1, cap=31):
    n = len(ins)
    prod = {}
    deps = []
    for i, x in enumerate(ins):
        deps.append(sorted({prod[s] for s in x.srcs if s in prod}))
        for d in x.dsts:
            prod[d] = i
    # memory: keep stores in program order relative to earlier loads of the same group (disjoint
    # addresses across groups, so only order T after all earlier L to be safe w.r.t. aliasing of x)
    succ = [[] for _ in range(n)]
    for i, d in enumerate(deps):
        for j in d:
            succ[j].append(i)
    prio = [0] * n
    for i in reversed(range(n)):
        prio[i] = LAT[ins[i].cls] + max([prio[j] for j in succ[i]], default=0)
    done = [False] * n
    finish = [0] * n
    order = []
    cycle = 0
    left = n
    uses = {}
    for x in ins:
        for s_ in x.srcs:
            uses[s_] = uses.get(s_, 0) + 1
    live = 0
    pending = [len(d) for d in deps]
    ready = [i for i in range(n) if pending[i] == 0]   # deps issued; still need finish <= cycle
    gleft = {}
    for x in ins:
        gleft[getattr(x, 'g', 0)] = gleft.get(getattr(x, 'g', 0), 0) + 1
    def key(i):
        return (-prio[i] + skew * getattr(ins[i], 'g', 0), i)
    stall = 0
    while left:
        avail = [i for i in ready if all(finish[j] <= cycle for j in deps[i])]
        avail.sort(key=key)
        oldest = min(g for g, c in gleft.items() if c > 0)
        m = s = a = l = w = 0
        issued = []
        for i in avail:
            c = ins[i].cls
            if w >= width:
                break
            if c == 'M':
                if m >= 2: continue
            elif c == 'S':
                if s >= 2 or m + s + a >= 4: continue
            elif c == 'A':
                if m + s + a >= 4: continue
            else:
                if l >= ls: continue
            x = ins[i]
            new = len(x.dsts) - (1 if x.tie is not None else 0)
            frees = sum(1 for s_ in set(x.srcs) if s_ in uses and uses[s_] == x.srcs.count(s_) and (x.tie is None or s_ != x.srcs[x.tie]))
            if new > 0 and live + new - frees > budget and not (getattr(x, 'g', 0) == oldest and live + new - frees <= cap - reserve):
                continue
            if c in 'LT':
                l += 1
            elif c == 'M':
                m += 1
            elif c == 'S':
                s += 1
            else:
                a += 1
            live += new - frees
            for s_ in x.srcs:
                if s_ in uses: uses[s_] -= 1
            w += 1
            done[i] = True
            finish[i] = cycle + LAT[c]
            order.append(i)
            issued.append(i)
            gleft[getattr(x, 'g', 0)] -= 1
            left -= 1
        for i in issued:
            ready.remove(i)
            for j in succ[i]:
                pending[j] -= 1
                if pending[j] == 0:
                    ready.append(j)
        if os.environ.get('MFIRST') and issued:
            k0 = len(order) - len(issued)
            grp = order[k0:]
            rank = {'M': 0, 'S': 1, 'A': 1, 'L': 2, 'T': 2} if os.environ.get('MFIRST') == '1' else {'L': 0, 'M': 1, 'S': 2, 'A': 2, 'T': 3}
            grp.sort(key=lambda i: rank[ins[i].cls])
            order[k0:] = grp
        stall = 0 if issued else stall + 1
        cycle += 1
        if stall > 100:
            raise RuntimeError('schedule deadlock')
    return [ins[i] for i in order], cycle

def allocate(seq, pinned, low, reserved=()):
    """pinned: virtual -> register number. low: virtuals needing v0-v15 (by-element operands)."""
    last = {}
    for i, x in enumerate(seq):
        for s in x.srcs:
            last[s] = i
    used = set(pinned.values()) | set(reserved)
    free = [r for r in range(32) if r not in used]
    phys = dict(pinned)
    out = []
    maxlive = 0
    for i, x in enumerate(seq):
        regs = {f's{k}': phys[s] for k, s in enumerate(x.srcs)}
        tied = x.srcs[x.tie] if x.tie is not None else None
        if tied is not None:
            assert last[tied] == i, f'tied source {tied} used later'
        for s in dict.fromkeys(x.srcs):
            if last.get(s) == i and s not in pinned and s != tied:
                free.append(phys[s])
        for k, d in enumerate(x.dsts):
            if k == 0 and tied is not None:
                r = phys[tied]
            else:
                cand = [r for r in free if r < 16] if d in low else ([r for r in free if r >= 16] or free)
                if not cand:
                    raise RuntimeError(f'out of registers at {i}: {x.fmt}')
                r = cand[0]
                free.remove(r)
            phys[d] = r
            regs[f'd{k}'] = r
        maxlive = max(maxlive, 32 - len(free))
        out.append(x.fmt.format(**regs))
    return out, maxlive
