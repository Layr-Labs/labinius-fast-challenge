//! Byte transposes of 64-word tiles, so that `vgf2p8affineqb` — `out.byte[t].bit[i] =
//! <A.byte[7 - i], x.byte[t]>`, an 8x8x8 block product per qword — sees eight different words in
//! each qword. Each `vpermt2b` index comes from the two bit layouts around it ([`round_idx`]);
//! the byte order inside a tile is the scramble [`word_at`].
use core::arch::x86_64::*;

/// `affine(ident, D)` is the bit transpose of each qword of `D`.
pub const GF_IDENT: i64 = 0x8040_2010_0804_0201u64 as i64;

const E: [u8; 6] = [0, 1, 2, 3, 4, 5];
const K: [u8; 4] = [10, 11, 12, 13];
const C: [u8; 3] = [20, 21, 22];

/// `zmm[s]` (`s < n`): the logical bit in vector-index bit `s` from the top; `pos[b]`: in
/// position bit `5 - b`.
#[derive(Clone, Copy)]
struct Layout {
    zmm: [u8; 4],
    n: usize,
    pos: [u8; 6],
}

impl Layout {
    const fn value(&self, id: u8, z: usize, p: usize) -> usize {
        let mut s = 0;
        while s < self.n {
            if self.zmm[s] == id {
                return (z >> (self.n - 1 - s)) & 1;
            }
            s += 1;
        }
        let mut b = 0;
        while b < 6 {
            if self.pos[b] == id {
                return (p >> (5 - b)) & 1;
            }
            b += 1;
        }
        panic!("a placed bit")
    }
}

/// One round moves one bit between a vector slot and a byte position; the two vectors differing
/// in that slot feed the two outputs, and this is the index for the output whose new bit is `v`.
const fn round_idx(from: &Layout, to: &Layout, v: usize) -> [u8; 64] {
    let mut slot = 0;
    while slot < to.n && from.zmm[slot] == to.zmm[slot] {
        slot += 1;
    }
    assert!(slot < to.n, "one slot moves");
    let mut idx = [0u8; 64];
    let z = v << (to.n - 1 - slot);
    let mut p = 0;
    while p < 64 {
        let mut q = 0;
        let mut b = 0;
        while b < 6 {
            q |= to.value(from.pos[b], z, p) << (5 - b);
            b += 1;
        }
        let s = to.value(from.zmm[slot], z, p);
        idx[p] = ((s << 6) | q) as u8;
        p += 1;
    }
    idx
}

const fn rounds<const N: usize>(mut layout: Layout, steps: [(usize, usize); N]) -> Rounds<N> {
    let mut out = [[[0u8; 64]; 2]; N];
    let mut i = 0;
    while i < N {
        let (slot, pos_bit) = steps[i];
        let mut next = layout;
        let moved = next.zmm[slot];
        next.zmm[slot] = next.pos[pos_bit];
        next.pos[pos_bit] = moved;
        out[i] = [round_idx(&layout, &next, 0), round_idx(&layout, &next, 1)];
        layout = next;
        i += 1;
    }
    Rounds(out)
}

/// The `vpermt2b` index pairs of a transpose's rounds, one cache line each.
#[repr(C, align(64))]
struct Rounds<const N: usize>([[[u8; 64]; 2]; N]);

#[repr(C, align(64))]
struct Fix([u8; 64]);

// The tables are built at compile time from the bit layouts, so a tile neither checks a lazy
// initialization nor reads a table list of run-time length: the rounds unroll with constant slot
// bits and a tile's vectors stay in registers through them.
static INPUT_ROUNDS: Rounds<4> = rounds(
    Layout {
        zmm: [E[5], E[4], E[3], E[2]],
        n: 4,
        pos: [E[1], E[0], K[3], K[2], K[1], K[0]],
    },
    [(0, 2), (1, 3), (2, 4), (3, 5)],
);
static OUTPUT_ROUNDS: Rounds<3> = rounds(
    Layout {
        zmm: [C[2], C[1], C[0], 0],
        n: 3,
        pos: [E[1], E[0], E[5], E[4], E[3], E[2]],
    },
    [(0, 2), (1, 3), (2, 4)],
);
// After the output rounds pos is [E1, E0, C2, C1, C0, E2]; the qword order is
// [E2, E1, E0, C2, C1, C0].
static OUTPUT_FIX: Fix = {
    let mut f = [0u8; 64];
    let mut p = 0;
    while p < 64 {
        let (e2, e1, e0, c) = ((p >> 5) & 1, (p >> 4) & 1, (p >> 3) & 1, p & 7);
        f[p] = ((e1 << 5) | (e0 << 4) | (c << 1) | e2) as u8;
        p += 1;
    }
    Fix(f)
};

/// The word at byte position `p` of a transposed tile.
#[inline]
pub fn word_at(p: usize) -> usize {
    (((p >> 5) & 1) << 1) | ((p >> 4) & 1) | ((p & 0xf) << 2)
}

#[inline(always)]
unsafe fn idx(v: &[u8; 64]) -> __m512i {
    _mm512_loadu_si512(v.as_ptr() as *const __m512i)
}

#[inline(always)]
unsafe fn apply_round(z: &mut [__m512i], slot_bit: usize, r: &[[u8; 64]; 2]) {
    let (i0, i1) = (idx(&r[0]), idx(&r[1]));
    for z0 in 0..z.len() {
        if z0 & slot_bit != 0 {
            continue;
        }
        let z1 = z0 | slot_bit;
        let (a, b) = (z[z0], z[z1]);
        z[z0] = _mm512_permutex2var_epi8(a, i0, b);
        z[z1] = _mm512_permutex2var_epi8(a, i1, b);
    }
}

/// `out[k].byte[p]` = byte `k` of word `word_at(p)` of the 64 words at `src`.
#[inline(always)]
pub unsafe fn transpose_tile(src: *const u8, out: &mut [__m512i; 16]) {
    for (i, o) in out.iter_mut().enumerate() {
        *o = _mm512_loadu_si512(src.add(64 * i) as *const __m512i);
    }
    for (r, round) in INPUT_ROUNDS.0.iter().enumerate() {
        apply_round(out, 8 >> r, round);
    }
}

/// `planes[c].byte[p]` = byte `c` of word `word_at(p)` becomes `out[g].qword[i].byte[c]` of word
/// `8g + i`.
#[inline(always)]
pub unsafe fn untranspose_planes(mut planes: [__m512i; 8]) -> [__m512i; 8] {
    for (r, round) in OUTPUT_ROUNDS.0.iter().enumerate() {
        apply_round(&mut planes, 4 >> r, round);
    }
    let fix = idx(&OUTPUT_FIX.0);
    planes.map(|z| _mm512_permutexvar_epi8(fix, z))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tile_transpose_and_back() {
        let words: Vec<u128> = (0..64u128)
            .map(|i| (i * 0x0123_4567_89AB_CDEF_1122_3344_5566_7788u128) ^ (i << 100))
            .collect();
        let mut t = [unsafe { _mm512_setzero_si512() }; 16];
        unsafe { transpose_tile(words.as_ptr() as *const u8, &mut t) };
        let bytes: Vec<[u8; 64]> = t
            .iter()
            .map(|&z| unsafe { core::mem::transmute::<__m512i, [u8; 64]>(z) })
            .collect();
        for k in 0..16 {
            for p in 0..64 {
                assert_eq!(bytes[k][p], words[word_at(p)].to_le_bytes()[k], "k {k} p {p}");
            }
        }
        let planes: [__m512i; 8] = core::array::from_fn(|c| t[c]);
        let back = unsafe { untranspose_planes(planes) };
        for g in 0..8 {
            let q: [u64; 8] = unsafe { core::mem::transmute(back[g]) };
            for i in 0..8 {
                assert_eq!(q[i], words[8 * g + i] as u64, "g {g} i {i}");
            }
        }
    }
}
