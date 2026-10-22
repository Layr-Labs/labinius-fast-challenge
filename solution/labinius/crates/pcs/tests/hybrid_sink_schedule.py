"""Scalar execution of the actual hybrid sink assembly, runnable without AVX-512.

Run: python3 solution/labinius/crates/pcs/tests/hybrid_sink_schedule.py
The native fused_register_sink_matches_staged_sink test remains the end-to-end
NTT/sink check. This test independently checks the changed register schedule,
including shuffles, key addresses, wrapping sums, and both fold-back modes.
"""
import random
import re
import struct
import unittest
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "src/simd/ntt/bin_asm.rs"
ROWS = [
    [0, 2, 1, 6, 8, 7, 3, 5],
    [4, 18, 20, 19, 24, 26, 25, 21],
    [23, 22, 9, 11, 10, 15, 17, 16],
    [12, 14, 13, 13],
]


def pack32(values):
    return struct.pack("<16I", *(x & 0xffffffff for x in values))


def add32(a, b):
    return pack32(x + y for x, y in zip(struct.unpack("<16i", a), struct.unpack("<16i", b)))


def dot_pairs(a, b):
    x, y = struct.unpack("<32h", a), struct.unpack("<32h", b)
    return pack32(x[i] * y[i] + x[i + 1] * y[i + 1] for i in range(0, 32, 2))


def fold(v, q):
    weight = 65536 % q - q
    # Signed low word, signed high word, and the low-word sign correction.
    out = []
    for x in struct.unpack("<16i", v):
        low = (x & 0xffff) - (65536 if x & 0x8000 else 0)
        out.append(low + ((x >> 16) + bool(x & 0x8000)) * weight)
    return pack32(out)


def instructions():
    source = SOURCE.read_text()
    start = source.index("macro_rules! hybrid_register_sink")
    source = source[start:source.index("#[repr(C, align(64))]", start)]
    for line in source.splitlines():
        asm = re.fullmatch(r'\s*"(.*)\\n",', line)
        store = re.fullmatch(r"\s*hybrid_store!\(\$mode, (\d+), (\d+)\),", line)
        if asm:
            yield asm[1]
        elif store:
            yield "store " + store[1] + " " + store[2]


def execute(initial, keys, accum, q, reduce):
    regs = list(initial)
    out = list(accum)
    for ins in instructions():
        if ins.startswith("store "):
            _, reg, off = ins.split()
            v = regs[int(reg)]
            if reduce:
                regs[31] = fold(v, q)
                v = regs[31]
            out[int(off) // 64] = v
            continue
        op, args = ins.split(" ", 1)
        args = args.split(", ")
        dst = int(args[0][3:])
        if op == "vmovdqa64":
            off = int(re.fullmatch(r"\[\{acc\}\+(\d+)\]", args[1])[1])
            regs[dst] = out[off // 64]
        elif op in ("vpmaddwd", "vpdpwssd"):
            x = regs[int(args[1][3:])]
            off = int(re.fullmatch(r"zmmword ptr \[\{key\}\+(\d+)\]", args[2])[1])
            product = dot_pairs(x, keys[off // 64])
            regs[dst] = add32(regs[dst], product) if op == "vpdpwssd" else product
        elif op == "vpaddd":
            regs[dst] = add32(regs[int(args[1][3:])], regs[int(args[2][3:])])
        elif op in ("vpunpcklqdq", "vpunpckhqdq"):
            x, y = regs[int(args[1][3:])], regs[int(args[2][3:])]
            offset = 8 if op == "vpunpckhqdq" else 0
            regs[dst] = b"".join(x[i + offset:i + offset + 8] + y[i + offset:i + offset + 8]
                                 for i in range(0, 64, 16))
        elif op == "vshufi64x2":
            x, y = regs[int(args[1][3:])], regs[int(args[2][3:])]
            imm = int(args[3])
            regs[dst] = b"".join(v[16 * ((imm >> (2 * i)) & 3):16 * (((imm >> (2 * i)) & 3) + 1)]
                                 for i, v in enumerate([x, x, y, y]))
        else:
            raise AssertionError("Unmodelled instruction: " + ins)
    return out


def reference(initial, keys, accum, q, reduce):
    out = []
    for group, rows in enumerate(ROWS):
        width = 64 // len(rows)
        key_start = 16 * group
        for stream in range(2):
            value = accum[2 * group + stream]
            for chunk in range(len(rows)):
                x = b"".join(initial[r][width * chunk:width * (chunk + 1)] for r in rows)
                value = add32(value, dot_pairs(x, keys[key_start + stream * len(rows) + chunk]))
            out.append(fold(value, q) if reduce else value)
    return out


class HybridSinkSchedule(unittest.TestCase):
    def test_signed_pair_overflow_and_reduction_scratch(self):
        words = struct.pack("<32h", *([-32768] * 32))
        regs = [words for _ in range(32)]
        keys = [words for _ in range(56)]
        accum = [pack32([0x7fffffff, 0x80000000] * 8) for _ in range(8)]
        for q in (3889, 9721):
            for reduce in (False, True):
                self.assertEqual(execute(regs, keys, accum, q, reduce),
                                 reference(regs, keys, accum, q, reduce))

    def test_actual_schedule_matches_independent_dot_products(self):
        rng = random.Random(0xa3e41d7b92c650f1)
        cases = [0, 0xff, 0x80, 0x7f, None]
        for case in cases:
            for trial in range(12):
                data = lambda: bytes([case] * 64) if case is not None else rng.randbytes(64)
                regs = [data() for _ in range(32)]
                keys = [rng.randbytes(64) for _ in range(56)]
                accum = [rng.randbytes(64) for _ in range(8)]
                for q in (3889, 9721):
                    for reduce in (False, True):
                        with self.subTest(case=case, trial=trial, q=q, reduce=reduce):
                            self.assertEqual(execute(regs, keys, accum, q, reduce),
                                             reference(regs, keys, accum, q, reduce))


if __name__ == "__main__":
    unittest.main()
