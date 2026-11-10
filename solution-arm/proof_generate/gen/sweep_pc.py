"""Rank pass_c candidates by the total cycles of the whole pass.

The per-group figure divides the pass's cycles by the number of scheduled
blocks, so it favours a smaller `per` (more blocks) without changing the
pass's real cost.  The comparable quantity is the total over the 24 nodes.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MFIRST", "1")

from sched import schedule, allocate
import gen_pc

WIDTHS = ((6, 2), (4, 2), (5, 2), (6, 1), (7, 2), (8, 2), (8, 1))


def total_pc(per, K, B, width, ls):
    total = live = 0
    for c in range(0, 24, per):
        ins, low = gen_pc.build(list(range(c, c + per)))
        seq, cyc = schedule(ins, skew=K, budget=B, width=width, ls=ls)
        asm, l = allocate(seq, {"k": 15}, low)
        total += cyc
        live = max(live, l)
    return total, live


if __name__ == "__main__":
    base, bl = total_pc(6, 40, 28, 6, 2)
    print(f"current pass_c per=6 K=40 B=28 w=6 ls=2: total={base} live={bl}", flush=True)
    rows = []
    for per in (4, 6, 8):
        for K in (16, 24, 32, 40, 48, 56, 64, 72):
            for B in (16, 20, 24, 28, 32):
                for w, ls in WIDTHS:
                    try:
                        t, l = total_pc(per, K, B, w, ls)
                    except Exception:
                        continue
                    rows.append((t, per, K, B, l, w, ls))
    rows.sort()
    for r in rows[:20]:
        print(f"total={r[0]:6d} per={r[1]:2d} K={r[2]:2d} B={r[3]:2d} live={r[4]:2d} w={r[5]} ls={r[6]}", flush=True)
