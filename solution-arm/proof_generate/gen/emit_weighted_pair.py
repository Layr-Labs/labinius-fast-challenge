"""Generate the two-bank SVE2 schedule for the weighted nibble front."""
from collections import Counter
from functools import lru_cache
from pathlib import Path
import re

from emit_weighted import build
import gen_lut
from sched import Ins, allocate

TABLE_REGS = tuple(range(10, 32))
TEMP_COUNT = 32 - len(TABLE_REGS)


def build_pair():
    """Duplicate the V12 graph with shared X0, Y0 and X1 index loads."""
    paired = []
    shared = set()
    index = 0
    for ins in build():
        if ins.cls == "L":
            if index < 3:
                shared.add(ins.dsts[0])
            index = (index + 1) % 6
        for side in (0, 1):
            if side == 1 and ins.cls == "L" and ins.dsts[0] in shared:
                continue

            def name(value):
                return f"{value}_0" if value in shared else f"{value}_{side}"

            fmt = ins.fmt
            if ins.cls == "A":
                bias = 11 * side - 6
                fmt = re.sub(r'z(\d+)\.h', lambda m: f'z{int(m[1]) + bias}.h', fmt)
            elif ins.cls == "T" and side == 1:
                fmt = re.sub(r'#(\d+)\]', lambda m: f'#{int(m[1]) + 2592}]', fmt)
            step = Ins(ins.cls, [name(v) for v in ins.dsts], [name(v) for v in ins.srcs], fmt)
            step.g = ins.g
            paired.append(step)
    return paired


def dependencies(ins):
    producers = {d: i for i, step in enumerate(ins) for d in step.dsts}
    return [tuple(sorted({producers[s] for s in step.srcs})) for step in ins]


def feasible_order(ins):
    """Find one legal step order under the ten-vector budget."""
    deps = dependencies(ins)
    required = [sum(1 << d for d in row) for row in deps]
    users = [sum(1 << j for j, row in enumerate(deps) if i in row) for i in range(len(ins))]
    full = (1 << len(ins)) - 1

    @lru_cache(None)
    def choose(done, live):
        if done == full:
            return ()
        choices = []
        for i, step in enumerate(ins):
            if done >> i & 1 or required[i] & done != required[i]:
                continue
            delta = len(step.dsts) - sum(users[d] & ~done == 1 << i for d in deps[i])
            if live + delta <= TEMP_COUNT:
                choices.append((delta, i))
        for delta, i in sorted(choices):
            rest = choose(done | (1 << i), live + delta)
            if rest is not None:
                return (i, *rest)
            # A non-growing instruction can run before the others. Any inputs
            # it releases have no remaining users except this instruction.
            if delta <= 0:
                break
        return None

    order = choose(0, 0)
    assert order is not None
    return order


def issue_order(ins, first_step, *, window=24):
    """Fill issue slots without breaking the feasible fallback order."""
    count = len(first_step)
    active = [j + i for j in range(0, len(ins), count) for i in first_step]
    deps = dependencies(ins)
    uses = Counter(d for row in deps for d in row)
    finish = {}
    out = []
    cycle = live = 0

    def growth(i, taken):
        return len(ins[i].dsts) - sum(uses[d] - taken[d] == 1 for d in deps[i])

    def can_move(i, before, delta):
        if delta <= 0:
            return True
        taken = Counter(deps[i])
        for other in before:
            delta += growth(other, taken)
            if live + delta > TEMP_COUNT:
                return False
            taken.update(deps[other])
        return True

    while active:
        vector = load = store = issued = 0
        while issued < 6:
            for k, i in enumerate(active[:window]):
                cls = ins[i].cls
                if ((cls == "A" and vector == 4)
                    or (cls == "L" and load + store == 3)
                    or (cls == "T" and (store == 2 or load + store == 3))):
                    continue
                if any(finish.get(d, cycle + 1) > cycle for d in deps[i]):
                    continue
                delta = growth(i, Counter())
                if live + delta > TEMP_COUNT or not can_move(i, active[:k], delta):
                    continue
                break
            else:
                break
            vector += cls == "A"
            load += cls == "L"
            store += cls == "T"
            issued += 1
            live += delta
            finish[i] = cycle + {"A": 2, "L": 6, "T": 1}[cls]
            uses.subtract(deps[i])
            out.append(ins[i])
            active.pop(k)
        cycle += 1
    return out, cycle


def emit():
    ins = build_pair()
    first_step = feasible_order([step for step in ins if step.g == 0])
    ordered, cycles = issue_order(ins, first_step)
    lines, live = allocate(ordered, {}, set(), reserved=TABLE_REGS)
    assert live <= 32

    asm = ["ptrue p7.h", "mov {n}, #2", "3:"]
    for r in (10, 21):
        asm += [f"ldp q{i}, q{i + 1}, [{{lut}}, #{16 * (i - 10)}]" for i in range(r, r + 10, 2)]
        asm += [f"ldr q{r + 10}, [{{lut}}, #{16 * r}]"]
    asm += [gen_lut.base_names(line).replace("XPTR", "{x}").replace("LB", "{{").replace("RB", "}}")
            for line in lines]
    asm += ["add {x}, {x}, #2592", "add {x}, {x}, #2592", "add {lut}, {lut}, #352", "subs {n}, {n}, #1", "b.ne 3b"]
    body = "\n".join(f'        "{line}",' for line in asm)
    bases = "\n".join(
        f"        i{m} = in(reg) (idx.as_ptr() as *const u8).add({64 + 128 * m}),"
        for m in sorted(gen_lut.BASES)
    )
    clobbers = ", ".join(f'out("v{r}") _' for r in range(32))
    print(f"V15 hot-body schedule {cycles} model cycles, {live} peak vector registers")
    return f"""// Generated by gen/emit_weighted_pair.py.
use super::*;

/// Compute two weighted table banks per loop iteration, sharing three packed indices.
/// The expressions and signed lookup representatives are those of `pass_lut_weighted`.
#[inline(never)]
pub(super) unsafe fn pass_lut_weighted_pair(idx: &[uint8x16_t; GROUPS], x: &mut [int16x8_t; N]) {{
    core::arch::asm!(
{body}
        x = inout(reg) x.as_mut_ptr() => _,
{bases}
        lut = inout(reg) super::weighted::LUT.as_ptr() => _,
        n = out(reg) _,
        {clobbers},
        out("p7") _,
        options(nostack)
    );
}}
"""


if __name__ == "__main__":
    path = Path(__file__).resolve().parents[1] / "src" / "ntt_neon_weighted_pair.rs"
    path.write_text(emit())
