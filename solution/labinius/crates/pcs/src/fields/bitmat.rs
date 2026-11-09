//! Byte transposes of 64-word tiles, so that `vgf2p8affineqb` — `out.byte[t].bit[i] =
//! <A.byte[7 - i], x.byte[t]>`, an 8x8x8 block product per qword — sees eight different words in
//! each qword. Each `vpermt2b` index comes from the two bit layouts around it ([`round_idx`]);
//! the byte order inside a tile is the scramble [`word_at`].
use core::arch::x86_64::*;

/// `affine(ident, D)` is the bit transpose of each qword of `D`.
pub const GF_IDENT: i64 = 0x8040_2010_0804_0201u64 as i64;

#[allow(dead_code)]
const E: [u8; 6] = [0, 1, 2, 3, 4, 5];
const K: [u8; 4] = [10, 11, 12, 13];
const C: [u8; 3] = [20, 21, 22];

/// `zmm[s]`: the logical bit in vector-index bit `s` from the top; `pos[b]`: in position bit `5 - b`.
#[allow(dead_code)]
#[derive(Clone)]
struct Layout {
    zmm: Vec<u8>,
    pos: [u8; 6],
}

impl Layout {
    fn slot_of(&self, id: u8) -> Option<usize> {
        self.zmm.iter().position(|&x| x == id)
    }

    fn value(&self, id: u8, z: usize, p: usize) -> usize {
        if let Some(s) = self.slot_of(id) {
            (z >> (self.zmm.len() - 1 - s)) & 1
        } else {
            let b = self.pos.iter().position(|&x| x == id).expect("a placed bit");
            (p >> (5 - b)) & 1
        }
    }
}

/// One round moves one bit between a vector slot and a byte position; the two vectors differing
/// in that slot feed the two outputs, and this is the index for the output whose new bit is `v`.
fn round_idx(from: &Layout, to: &Layout, v: usize) -> [u8; 64] {
    let slot = (0..to.zmm.len())
        .find(|&s| from.zmm[s] != to.zmm[s])
        .expect("one slot moves");
    let mut idx = [0u8; 64];
    let z = v << (to.zmm.len() - 1 - slot);
    for (p, i) in idx.iter_mut().enumerate() {
        let mut q = 0;
        for b in 0..6 {
            q |= to.value(from.pos[b], z, p) << (5 - b);
        }
        let s = to.value(from.zmm[slot], z, p);
        *i = ((s << 6) | q) as u8;
    }
    idx
}

fn rounds(mut layout: Layout, steps: &[(usize, usize)]) -> Vec<[[u8; 64]; 2]> {
    steps
        .iter()
        .map(|&(slot, pos_bit)| {
            let mut next = layout.clone();
            let out = next.zmm[slot];
            next.zmm[slot] = next.pos[pos_bit];
            next.pos[pos_bit] = out;
            let r = [round_idx(&layout, &next, 0), round_idx(&layout, &next, 1)];
            layout = next;
            r
        })
        .collect()
}

static INPUT_ROUNDS: [[[u8; 64]; 2]; 4] = [
        [
            [0, 1, 2, 3, 4, 5, 6, 7, 64, 65, 66, 67, 68, 69, 70, 71, 16, 17, 18, 19, 20, 21, 22, 23, 80, 81, 82, 83, 84, 85, 86, 87, 32, 33, 34, 35, 36, 37, 38, 39, 96, 97, 98, 99, 100, 101, 102, 103, 48, 49, 50, 51, 52, 53, 54, 55, 112, 113, 114, 115, 116, 117, 118, 119],
            [8, 9, 10, 11, 12, 13, 14, 15, 72, 73, 74, 75, 76, 77, 78, 79, 24, 25, 26, 27, 28, 29, 30, 31, 88, 89, 90, 91, 92, 93, 94, 95, 40, 41, 42, 43, 44, 45, 46, 47, 104, 105, 106, 107, 108, 109, 110, 111, 56, 57, 58, 59, 60, 61, 62, 63, 120, 121, 122, 123, 124, 125, 126, 127]
        ],
        [
            [0, 1, 2, 3, 64, 65, 66, 67, 8, 9, 10, 11, 72, 73, 74, 75, 16, 17, 18, 19, 80, 81, 82, 83, 24, 25, 26, 27, 88, 89, 90, 91, 32, 33, 34, 35, 96, 97, 98, 99, 40, 41, 42, 43, 104, 105, 106, 107, 48, 49, 50, 51, 112, 113, 114, 115, 56, 57, 58, 59, 120, 121, 122, 123],
            [4, 5, 6, 7, 68, 69, 70, 71, 12, 13, 14, 15, 76, 77, 78, 79, 20, 21, 22, 23, 84, 85, 86, 87, 28, 29, 30, 31, 92, 93, 94, 95, 36, 37, 38, 39, 100, 101, 102, 103, 44, 45, 46, 47, 108, 109, 110, 111, 52, 53, 54, 55, 116, 117, 118, 119, 60, 61, 62, 63, 124, 125, 126, 127]
        ],
        [
            [0, 1, 64, 65, 4, 5, 68, 69, 8, 9, 72, 73, 12, 13, 76, 77, 16, 17, 80, 81, 20, 21, 84, 85, 24, 25, 88, 89, 28, 29, 92, 93, 32, 33, 96, 97, 36, 37, 100, 101, 40, 41, 104, 105, 44, 45, 108, 109, 48, 49, 112, 113, 52, 53, 116, 117, 56, 57, 120, 121, 60, 61, 124, 125],
            [2, 3, 66, 67, 6, 7, 70, 71, 10, 11, 74, 75, 14, 15, 78, 79, 18, 19, 82, 83, 22, 23, 86, 87, 26, 27, 90, 91, 30, 31, 94, 95, 34, 35, 98, 99, 38, 39, 102, 103, 42, 43, 106, 107, 46, 47, 110, 111, 50, 51, 114, 115, 54, 55, 118, 119, 58, 59, 122, 123, 62, 63, 126, 127]
        ],
        [
            [0, 64, 2, 66, 4, 68, 6, 70, 8, 72, 10, 74, 12, 76, 14, 78, 16, 80, 18, 82, 20, 84, 22, 86, 24, 88, 26, 90, 28, 92, 30, 94, 32, 96, 34, 98, 36, 100, 38, 102, 40, 104, 42, 106, 44, 108, 46, 110, 48, 112, 50, 114, 52, 116, 54, 118, 56, 120, 58, 122, 60, 124, 62, 126],
            [1, 65, 3, 67, 5, 69, 7, 71, 9, 73, 11, 75, 13, 77, 15, 79, 17, 81, 19, 83, 21, 85, 23, 87, 25, 89, 27, 91, 29, 93, 31, 95, 33, 97, 35, 99, 37, 101, 39, 103, 41, 105, 43, 107, 45, 109, 47, 111, 49, 113, 51, 115, 53, 117, 55, 119, 57, 121, 59, 123, 61, 125, 63, 127]
        ]
    ];

static OUTPUT_ROUNDS: [[[u8; 64]; 2]; 3] = [
        [
            [0, 1, 2, 3, 4, 5, 6, 7, 64, 65, 66, 67, 68, 69, 70, 71, 16, 17, 18, 19, 20, 21, 22, 23, 80, 81, 82, 83, 84, 85, 86, 87, 32, 33, 34, 35, 36, 37, 38, 39, 96, 97, 98, 99, 100, 101, 102, 103, 48, 49, 50, 51, 52, 53, 54, 55, 112, 113, 114, 115, 116, 117, 118, 119],
            [8, 9, 10, 11, 12, 13, 14, 15, 72, 73, 74, 75, 76, 77, 78, 79, 24, 25, 26, 27, 28, 29, 30, 31, 88, 89, 90, 91, 92, 93, 94, 95, 40, 41, 42, 43, 44, 45, 46, 47, 104, 105, 106, 107, 108, 109, 110, 111, 56, 57, 58, 59, 60, 61, 62, 63, 120, 121, 122, 123, 124, 125, 126, 127]
        ],
        [
            [0, 1, 2, 3, 64, 65, 66, 67, 8, 9, 10, 11, 72, 73, 74, 75, 16, 17, 18, 19, 80, 81, 82, 83, 24, 25, 26, 27, 88, 89, 90, 91, 32, 33, 34, 35, 96, 97, 98, 99, 40, 41, 42, 43, 104, 105, 106, 107, 48, 49, 50, 51, 112, 113, 114, 115, 56, 57, 58, 59, 120, 121, 122, 123],
            [4, 5, 6, 7, 68, 69, 70, 71, 12, 13, 14, 15, 76, 77, 78, 79, 20, 21, 22, 23, 84, 85, 86, 87, 28, 29, 30, 31, 92, 93, 94, 95, 36, 37, 38, 39, 100, 101, 102, 103, 44, 45, 46, 47, 108, 109, 110, 111, 52, 53, 54, 55, 116, 117, 118, 119, 60, 61, 62, 63, 124, 125, 126, 127]
        ],
        [
            [0, 1, 64, 65, 4, 5, 68, 69, 8, 9, 72, 73, 12, 13, 76, 77, 16, 17, 80, 81, 20, 21, 84, 85, 24, 25, 88, 89, 28, 29, 92, 93, 32, 33, 96, 97, 36, 37, 100, 101, 40, 41, 104, 105, 44, 45, 108, 109, 48, 49, 112, 113, 52, 53, 116, 117, 56, 57, 120, 121, 60, 61, 124, 125],
            [2, 3, 66, 67, 6, 7, 70, 71, 10, 11, 74, 75, 14, 15, 78, 79, 18, 19, 82, 83, 22, 23, 86, 87, 26, 27, 90, 91, 30, 31, 94, 95, 34, 35, 98, 99, 38, 39, 102, 103, 42, 43, 106, 107, 46, 47, 110, 111, 50, 51, 114, 115, 54, 55, 118, 119, 58, 59, 122, 123, 62, 63, 126, 127]
        ]
    ];

static OUTPUT_FIX: [u8; 64] = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63];

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
    for (r, round) in INPUT_ROUNDS.iter().enumerate() {
        apply_round(out, 8 >> r, round);
    }
}

/// `planes[c].byte[p]` = byte `c` of word `word_at(p)` becomes `out[g].qword[i].byte[c]` of word
/// `8g + i`.
#[inline(always)]
pub unsafe fn untranspose_planes(mut planes: [__m512i; 8]) -> [__m512i; 8] {
    for (r, round) in OUTPUT_ROUNDS.iter().enumerate() {
        apply_round(&mut planes, 4 >> r, round);
    }
    let fix = idx(&OUTPUT_FIX);
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
