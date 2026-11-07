#!/usr/bin/env python3
"""Generates `src/sms_fused.rs`: the SMS key sink of one slot with a slice of the NEXT sweep's table
index build (`ntt_neon::indices`) interleaved into the same instruction stream.

Why: on Neoverse V2 the 16-bit SVE `sdot` issues only on pipes V0/V2, so during the sink V1/V3 are
idle. The index build (byte transposes, shifts, bit selects, zips, stores) needs no V0/V2-only op,
so its instructions can fill the idle pipes. An out-of-order core only overlaps them if they sit
between the `sdot`s in program order (a long `sdot` run fills the V0/V2 issue queues and blocks
dispatch), hence one fused, fully unrolled stream per slot.

Layout: a sweep has `SINK` tiles of 8 transforms (`8 SINK` transforms) and 648 slots. The index
build of one transform is cut into `PIECES = 648 // (8 SINK)` pieces at points where no vector
register is live; slot `s < PIECES * 8 SINK` runs piece `s % PIECES` of transform `s // PIECES`.
Slots past that run the plain sink.

The index code mirrors `index_block` / `indices_pair` operation for operation (same values, same
stored bytes); the transposed byte rows go through a 256-byte L1 scratch instead of registers. Its
vector registers come from the pool v6-v15 (the sink keeps keys in v0-v3, transforms in v4-v5 and
the sums in v16-v31). `--check` prints statistics. Usage:
    python3 gen/sms_fused.py > src/sms_fused.rs
"""
import sys

N = 648
GROUPS = N // 4
SINK = 16
LOCK = 8
TRANSFORMS = SINK * LOCK
PIECES = N // TRANSFORMS            # 5 for SINK = 16
ROWS = 7
XS = N * 16                          # bytes between transforms of one slot
POOL = [f"v{i}" for i in range(6, 16)]
COLS = [f"w{c}" for c in range(8)]   # GPR operand names of the eight column word pointers


class Prog:
    """A straight-line program over virtual vector registers."""

    def __init__(self):
        self.ops = []   # (template, dst, srcs, tied)
        self.n = 0
        self.cuts = [0]  # op indices where no virtual register is live

    def new(self):
        self.n += 1
        return f"%{self.n}"

    def op(self, template, srcs=(), tied=None, dst=True):
        d = self.new() if dst else None
        self.ops.append((template, d, list(srcs), tied))
        return d

    def cut(self):
        self.cuts.append(len(self.ops))

    # instruction helpers: {d} destination, {0}, {1}.. sources
    def ldr(self, ptr, off):
        return self.op(f"ldr q{{d}}, [{{{ptr}}}, #{off}]")

    def str_(self, v, ptr, off):
        self.op(f"str q{{0}}, [{{{ptr}}}, #{off}]", [v], dst=False)

    def zip(self, which, t, a, b):
        return self.op(f"{which} v{{d}}.{t}, v{{0}}.{t}, v{{1}}.{t}", [a, b])

    def ext8(self, a, b):
        return self.op("ext v{d}.16b, v{0}.16b, v{1}.16b, #8", [a, b])

    def high_half(self, a):
        # ext8(a, 0): the upper doubleword moved down, upper half zero (scalar writes zero it)
        return self.op("mov d{d}, v{0}.d[1]", [a])

    def ushr(self, a, n):
        return self.op(f"ushr v{{d}}.16b, v{{0}}.16b, #{n}", [a])

    def shl(self, a, n):
        return self.op(f"shl v{{d}}.16b, v{{0}}.16b, #{n}", [a])

    def orr(self, a, b):
        return self.op("orr v{d}.16b, v{0}.16b, v{1}.16b", [a, b])

    def and_(self, a, b):
        return self.op("and v{d}.16b, v{0}.16b, v{1}.16b", [a, b])

    def movi(self, imm):
        return self.op(f"movi v{{d}}.16b, #{imm:#x}")

    def bit(self, d, n, m):
        # d = (d & ~m) | (n & m); d is consumed (tied)
        return self.op("bit v{d}.16b, v{1}.16b, v{2}.16b", [d, n, m], tied=0)

    def bif(self, d, n, m):
        # d = (d & m) | (n & ~m); d is consumed (tied)
        return self.op("bif v{d}.16b, v{1}.16b, v{2}.16b", [d, n, m], tied=0)


ZERO = "ZERO"  # a known all-zero vector (folded away)


def swap(p, x, y, d, mask):
    """index_swap::<D>: (bsl(mask, x, y << D), bsl(mask, x >> D, y)); y may be ZERO."""
    if y == ZERO:
        # (x & mask, (x >> D) & mask)
        u = p.ushr(x, d)
        return p.and_(x, mask), p.and_(u, mask)
    t = p.shl(y, d)
    u = p.ushr(x, d)
    x2 = p.bit(t, x, mask)
    y2 = p.bif(u, y, mask)
    return x2, y2


def transpose(p, P, i):
    """indices_pair::<P>'s byte rows for word k = P + 2 i, stored to scratch rows 8 i .. 8 i + 7."""
    k = P + 2 * i
    r = [p.ldr(COLS[c], 16 * k) for c in range(8)]
    a = [None] * 8
    for j in range(4):
        x, y = r[2 * j], r[2 * j + 1]
        a[j] = p.zip("zip1", "16b", x, y)
        a[j + 4] = p.zip("zip2", "16b", x, y)
    b = [None] * 8
    for j in range(8):
        h = j // 4
        x, y = a[4 * h + 2 * ((j % 4) // 2)], a[4 * h + 2 * ((j % 4) // 2) + 1]
        b[j] = p.zip("zip1" if j % 2 == 0 else "zip2", "8h", x, y)
    for v in range(8):
        h, z = v // 4, (v % 4) // 2
        x, y = b[4 * h + z], b[4 * h + 2 + z]
        row = p.zip("zip1" if v % 2 == 0 else "zip2", "4s", x, y)
        p.str_(row, "scr", (8 * i + v) * 16)


def block(p, K, G):
    """index_block::<K, G>: sixteen index vectors (fewer for G = 2) from the scratch rows."""
    R = 8 * (K // 2)          # scratch row base of bytes[K / 2]
    O = 8 * (1 - K // 2)      # of bytes[1 - K / 2]
    row = lambda base, v: p.ldr("scr", (base + v) * 16)
    a = row(R, G)
    o2, o3 = row(O, 2 + G), row(O, 3 + G)
    b = p.ext8(o2, o3)
    if K >= 2:
        b = p.orr(p.ushr(b, 1), p.shl(o3, 7))
    c0 = row(R, 5 + G)
    c1 = p.high_half(c0) if 6 + G == 8 else p.ext8(c0, row(R, 6 + G))
    c = p.orr(p.ushr(c0, 1), p.shl(c1, 7))
    if G == 0:
        last = p.high_half(row(O, 7))
        d = p.ushr(last, 1 if K < 2 else 2)
    else:
        d = ZERO
    m55 = p.movi(0x55)
    a, b = swap(p, a, b, 1, m55)
    c, d = swap(p, c, d, 1, m55)
    m33 = p.movi(0x33)
    a, c = swap(p, a, c, 2, m33)
    b, d = swap(p, b, d, 2, m33)
    c1e = p.movi(0x1E)
    c01 = p.movi(0x01)

    def put(q, z):
        j = 4 * (16 * G + q) + K
        p.str_(z, "nib", j * 16)

    def wanted(q):
        return 4 * (16 * G + q) + K < GROUPS

    for v, r in ((a, 0), (b, 1), (c, 2), (d, 3)):
        if wanted(r) or wanted(r + 8):
            lo = p.and_(p.shl(v, 1), c1e)
            lo_odd = p.orr(lo, c01)
            if wanted(r):
                put(r, p.zip("zip1", "16b", lo, lo_odd))
            if wanted(r + 8):
                put(r + 8, p.zip("zip2", "16b", lo, lo_odd))
        if wanted(r + 4) or wanted(r + 12):
            hi = p.and_(p.ushr(v, 3), c1e)
            hi_odd = p.orr(hi, c01)
            if wanted(r + 4):
                put(r + 4, p.zip("zip1", "16b", hi, hi_odd))
            if wanted(r + 12):
                put(r + 12, p.zip("zip2", "16b", hi, hi_odd))


def prefetch(p):
    """The input words of transform t + 4 (the same two columns, the next four elements: 256
    bytes on), so that no word load of the fused stream waits on memory with `sdot`s behind it
    in the reorder buffer. Harmless past the column's end (prefetches never fault)."""
    for c in range(8):
        p.op(f"prfm pldl1keep, [{{{COLS[c]}}}, #256]", dst=False)


def index_program():
    p = Prog()
    prefetch(p)
    for P in (0, 1):
        for i in (0, 1):
            transpose(p, P, i)
            p.cut()
        for K in (P, P + 2):
            for G in range(3):
                block(p, K, G)
                p.cut()
    return p


def allocate(p):
    """Linear scan over the straight-line program onto POOL; asserts it never runs out. Returns
    the instruction texts and the per-op live count."""
    last = {}
    for i, (_, d, srcs, _) in enumerate(p.ops):
        for s in srcs:
            last[s] = i
    phys = {}
    free = list(POOL)
    out, live = [], []
    for i, (tmpl, d, srcs, tied) in enumerate(p.ops):
        regs = [phys[s] for s in srcs]
        dying = [s for s in dict.fromkeys(srcs) if last[s] == i]
        if tied is not None:
            assert last[srcs[tied]] == i, "a tied operand must die at its use"
        for s in dying:
            if tied is not None and s == srcs[tied]:
                continue
            free.append(phys.pop(s))
        if d is not None:
            if tied is not None:
                phys[d] = phys.pop(srcs[tied])
            else:
                assert free, f"out of vector registers at op {i}"
                phys[d] = free.pop(0)
            if d not in last:
                raise AssertionError(f"dead value at op {i}: {tmpl}")
        num = lambda r: r[1:]
        text = tmpl.replace("{d}", num(phys[d]) if d else "")
        for k, r in enumerate(regs):
            text = text.replace("{" + str(k) + "}", num(r))
        out.append(text)
        live.append(len(phys))
    assert not phys, "values live at the end"
    return out, live


def pieces(p, code, live):
    """Cut the program into PIECES runs of whole units with balanced instruction counts."""
    cuts = p.cuts
    assert all(c == 0 or live[c - 1] == 0 for c in cuts), "a cut with live values"
    total = len(code)
    bounds = [0]
    for k in range(1, PIECES):
        target = total * k / PIECES
        best = min(cuts, key=lambda c: abs(c - target))
        assert best > bounds[-1]
        bounds.append(best)
    bounds.append(total)
    return [code[bounds[k]:bounds[k + 1]] for k in range(PIECES)]


def sink_code():
    """The sink of one slot over SINK tiles (rows 0-3, then rows 4-6), fully unrolled: the
    sums of `slot_sink`, the same `sdot` per accumulator in the same order. Row 0 starts from
    `[a]`, rows 1-6 from `[src]` (the accumulators, or zeros for a chunk's first sweep); all
    rows are stored to `[a]`."""
    o = []
    # rows 0-3
    o += ["ldp q16, q17, [{a}, #0]", "ldp q18, q19, [{a}, #32]"]
    o += [f"ldp q{20 + 2 * i}, q{21 + 2 * i}, [{{src}}, #{64 + 32 * i}]" for i in range(6)]
    for part, (rows, keys) in enumerate(((4, ["ldp q0, q1, [{kp}]", "ldp q2, q3, [{kp}, #32]"]),
                                         (3, ["ldp q0, q1, [{kp}, #64]", "ldr q2, [{kp}, #96]"]))):
        if part == 1:
            o += ["mov {zp}, {z}", "mov {kp}, {k}"]
            o += [f"ldp q{16 + 2 * i}, q{17 + 2 * i}, [{{src}}, #{256 + 32 * i}]" for i in range(6)]
        for g in range(SINK):
            o += keys
            for h in range(2):
                for q in range(4):
                    x = 4 + q % 2
                    o.append(f"ldr q{x}, [{{zp}}, #{q * XS}]" if q else f"ldr q{x}, [{{zp}}]")
                    for r in range(rows):
                        o.append(f"sdot z{16 + 4 * r + q}.d, z{x}.h, z{r}.h[{h}]")
                o.append("add {zp}, {zp}, {xs4}")
            o.append("add {kp}, {kp}, {ks}")
        base = 0 if part == 0 else 256
        o += [f"stp q{16 + 2 * i}, q{17 + 2 * i}, [{{a}}, #{base + 32 * i}]" for i in range(8 if part == 0 else 6)]
    return o


def interleave(a, b, lead=24):
    """Merge b into a proportionally, keeping each stream's order; b starts after `lead` ops of a."""
    out = a[:lead]
    a = a[lead:]
    i = j = 0
    while i < len(a) or j < len(b):
        if j >= len(b) or (i < len(a) and i * len(b) <= j * len(a)):
            out.append(a[i]); i += 1
        else:
            out.append(b[j]); j += 1
    return out


VREG_CLOBBERS = [f"v{i}" for i in range(32)]


def rust_asm(lines, indent):
    pad = " " * indent
    return "\n".join(f'{pad}"{l}",' for l in lines)


def gpr_operands(piece):
    """The index stream's pointer operands that `piece` uses (asm! rejects unused ones)."""
    text = "\n".join(piece)
    out = ""
    for c in range(8):
        if "{w%d}" % c in text:
            out += f"        w{c} = in(reg) w[{c}],\n"
    for name in ("scr", "nib"):
        if "{%s}" % name in text:
            out += f"        {name} = in(reg) {name},\n"
    return out


def emit():
    p = index_program()
    code, live = allocate(p)
    parts = pieces(p, code, live)
    sink = sink_code()
    w = sys.stdout.write
    w("// @generated by gen/sms_fused.py; do not edit by hand.\n")
    w("//\n")
    w(f"// SINK = {SINK}, {PIECES} pieces of the index build per transform; index instructions per piece:\n")
    w("// " + ", ".join(str(len(x)) for x in parts) + f"; sink instructions per slot: {len(sink)}.\n")
    w(f"// Peak live index vectors: {max(live)} of {len(POOL)} (v6-v15).\n\n")
    w(f"/// Tiles per sweep the fused slots assume.\npub const FUSED_SINK: usize = {SINK};\n")
    w(f"/// Index-build pieces per transform (slots per transform).\npub const PIECES: usize = {PIECES};\n\n")
    # index-only pieces (tests and non-SVE hosts can run them)
    for k, piece in enumerate(parts):
        w(f"/// Piece {k} of the index build alone (the same instructions as in [`fused_slot_{k}`]).\n")
        w("#[allow(dead_code)]\n#[inline(never)]\n")
        w(f"pub unsafe fn index_piece_{k}(w: &[*const u8; 8], scr: *mut u8, nib: *mut u8) {{\n")
        w("    let _ = (w, scr, nib);\n")
        w("    core::arch::asm!(\n")
        w(rust_asm(piece, 8) + "\n")
        w(gpr_operands(piece))
        for v in POOL:
            w(f'        out("{v}") _,\n')
        w("        options(nostack, preserves_flags)\n    );\n}\n\n")
    for k, piece in enumerate(parts):
        fused = interleave(sink, piece)
        w(f"/// The sink of one slot over {SINK} tiles with piece {k} of a transform's index build.\n")
        w("#[cfg(target_feature = \"sve\")]\n#[inline(never)]\n")
        w(f"pub unsafe fn fused_slot_{k}(z: *const u8, k: *const u8, a: *mut u8, src: *const u8, w: &[*const u8; 8], scr: *mut u8, nib: *mut u8) {{\n")
        w("    let _ = (w, scr, nib);\n")
        w("    core::arch::asm!(\n")
        w(rust_asm(fused, 8) + "\n")
        w("        a = in(reg) a,\n        src = in(reg) src,\n        z = in(reg) z,\n        k = in(reg) k,\n")
        w("        zp = inout(reg) z => _,\n        kp = inout(reg) k => _,\n")
        w(f"        xs4 = in(reg) {4 * XS}usize,\n        ks = in(reg) {N * ROWS * LOCK * 2}usize,\n")
        w(gpr_operands(piece))
        for v in VREG_CLOBBERS:
            w(f'        out("{v}") _,\n')
        w("        options(nostack, preserves_flags)\n    );\n}\n\n")
    w("/// The fused slots by piece.\n#[cfg(target_feature = \"sve\")]\n")
    w("pub const FUSED: [unsafe fn(*const u8, *const u8, *mut u8, *const u8, &[*const u8; 8], *mut u8, *mut u8); PIECES] = [\n")
    w("    " + ", ".join(f"fused_slot_{k}" for k in range(PIECES)) + ",\n];\n\n")
    w("/// The index-only pieces in order.\n#[allow(dead_code)]\n")
    w("pub const INDEX_PIECES: [unsafe fn(&[*const u8; 8], *mut u8, *mut u8); PIECES] = [\n")
    w("    " + ", ".join(f"index_piece_{k}" for k in range(PIECES)) + ",\n];\n")


def check():
    p = index_program()
    code, live = allocate(p)
    parts = pieces(p, code, live)
    sink = sink_code()
    print("index ops", len(code), "peak live", max(live), "pieces", [len(x) for x in parts])
    print("sink ops per slot", len(sink), "sdot", sum(1 for l in sink if l.startswith("sdot")))


if __name__ == "__main__":
    if "--check" in sys.argv:
        check()
    elif "--mca" in sys.argv:
        # plain llvm-mca inputs: fused piece k, the sink alone, the index pieces alone
        k = int(sys.argv[sys.argv.index("--mca") + 1])
        p = index_program()
        code, live = allocate(p)
        parts = pieces(p, code, live)
        sink = sink_code()
        regs = {"a": "x0", "src": "x1", "zp": "x2", "kp": "x3", "z": "x16", "k": "x17", "xs4": "x4",
                "ks": "x5", "scr": "x14", "nib": "x15"}
        regs.update({f"w{c}": f"x{6 + c}" for c in range(8)})
        def fix(l):
            for n, r in regs.items():
                l = l.replace("{" + n + "}", r)
            return l
        which = sys.argv[sys.argv.index("--mca") + 2]
        body = {"fused": interleave(sink, parts[k]), "sink": sink, "index": parts[k],
                "indexall": code}[which]
        print("\n".join(fix(l) for l in body))
    else:
        emit()
