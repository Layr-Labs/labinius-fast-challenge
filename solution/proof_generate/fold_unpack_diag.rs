// Temporary build-only adapter. Remove this file after the native comparison.
// All variants use the active schedule, output, and task code.
#![allow(dead_code)]
use labinius::fields::scalar::B128;
use labinius::ring::{Representation, RingElement};
use labinius::challenge::ShortChallenge;
use labinius::fold::fold_witness_streaming;

pub fn check_carry4_pair() { native_pair_check::run(); }

pub fn fold_witness_bitsliced_unpack(
    words: &[B128],
    columns: usize,
    challenges: &[ShortChallenge],
    bpc: usize,
    q: u16,
    workers: usize,
) -> Vec<RingElement> {
    assert_eq!(challenges.len(), columns, "one challenge per column");
    assert!(bpc > 0 && words.len() == columns * 128 * bpc, "the words are not whole columns");
    if bpc % 4 != 0 || !bitsliced_unpack::fits(challenges) {
        return fold_witness_streaming(words, columns, challenges, bpc, q, workers);
    }
    bitsliced_unpack::fold_words_with::<true, false>(words, challenges, 128 * bpc, workers)
}

pub fn fold_witness_bitsliced_carry4(
    words: &[B128],
    columns: usize,
    challenges: &[ShortChallenge],
    bpc: usize,
    q: u16,
    workers: usize,
) -> Vec<RingElement> {
    assert_eq!(challenges.len(), columns, "one challenge per column");
    assert!(bpc > 0 && words.len() == columns * 128 * bpc, "the words are not whole columns");
    if bpc % 4 != 0 || !bitsliced_unpack::fits(challenges) {
        return fold_witness_streaming(words, columns, challenges, bpc, q, workers);
    }
    bitsliced_unpack::fold_words_with::<false, true>(words, challenges, 128 * bpc, workers)
}

pub fn fold_witness_bitsliced_unpack_carry4(
    words: &[B128],
    columns: usize,
    challenges: &[ShortChallenge],
    bpc: usize,
    q: u16,
    workers: usize,
) -> Vec<RingElement> {
    assert_eq!(challenges.len(), columns, "one challenge per column");
    assert!(bpc > 0 && words.len() == columns * 128 * bpc, "the words are not whole columns");
    if bpc % 4 != 0 || !bitsliced_unpack::fits(challenges) {
        return fold_witness_streaming(words, columns, challenges, bpc, q, workers);
    }
    bitsliced_unpack::fold_words_with::<true, true>(words, challenges, 128 * bpc, workers)
}

mod bitsliced_unpack {
    //! Integer convolution in Z[Y]/(Y^243 + 1), followed by reduction modulo
    //! Y^162 - Y^81 + 1. Since (Y^81 + 1)(Y^162 - Y^81 + 1) = Y^243 + 1,
    //! this gives exactly the original ring product, with no probabilistic reconstruction.
    //!
    //! A negative bit contribution is accumulated as its complement, with a scalar bias removed
    //! at the end: -b = (1-b)-1. The complement is never stored: a slot pair's negative terms
    //! are scheduled apart from its positive ones, and their leaf full adders take the carry of
    //! the complemented inputs (the sum of two complemented inputs is the sum of the direct
    //! ones). Every negative entry, guard and padding reads included, adds one to the bias of
    //! both its slots. The counter's low six bits use a 64-input carry-save tree (two ternlogs
    //! per full adder); the carry into bit six ripples through the remaining nine bits once per
    //! 64 inputs. The 15-bit counter cannot overflow when the total challenge weight plus the
    //! padding ([`fits`]) is at most 32767.
    use super::{B128, Representation, RingElement, ShortChallenge};
    use labinius::fields::bitmat;
#[inline(always)]
pub(crate) unsafe fn transpose8x8_q(r: [__m512i; 8]) -> [__m512i; 8] {
    let a0 = _mm512_unpacklo_epi64(r[0], r[1]);
    let a1 = _mm512_unpackhi_epi64(r[0], r[1]);
    let a2 = _mm512_unpacklo_epi64(r[2], r[3]);
    let a3 = _mm512_unpackhi_epi64(r[2], r[3]);
    let a4 = _mm512_unpacklo_epi64(r[4], r[5]);
    let a5 = _mm512_unpackhi_epi64(r[4], r[5]);
    let a6 = _mm512_unpacklo_epi64(r[6], r[7]);
    let a7 = _mm512_unpackhi_epi64(r[6], r[7]);
    let b0 = _mm512_shuffle_i64x2::<0x88>(a0, a2);
    let b1 = _mm512_shuffle_i64x2::<0xDD>(a0, a2);
    let b2 = _mm512_shuffle_i64x2::<0x88>(a4, a6);
    let b3 = _mm512_shuffle_i64x2::<0xDD>(a4, a6);
    let b4 = _mm512_shuffle_i64x2::<0x88>(a1, a3);
    let b5 = _mm512_shuffle_i64x2::<0xDD>(a1, a3);
    let b6 = _mm512_shuffle_i64x2::<0x88>(a5, a7);
    let b7 = _mm512_shuffle_i64x2::<0xDD>(a5, a7);
    [
        _mm512_shuffle_i64x2::<0x88>(b0, b2),
        _mm512_shuffle_i64x2::<0x88>(b4, b6),
        _mm512_shuffle_i64x2::<0x88>(b1, b3),
        _mm512_shuffle_i64x2::<0x88>(b5, b7),
        _mm512_shuffle_i64x2::<0xDD>(b0, b2),
        _mm512_shuffle_i64x2::<0xDD>(b4, b6),
        _mm512_shuffle_i64x2::<0xDD>(b1, b3),
        _mm512_shuffle_i64x2::<0xDD>(b5, b7),
    ]
}
    use core::arch::x86_64::*;

    const LANES: usize = 512;
    const SLOTS: usize = 243;
    const BITS: usize = 15;
    /// Columns per group: 32 * 129 planes, 264 KB per worker (two hyperthread siblings fit
    /// comfortably inside one 2 MB L2 cache, while halving the outer group loops, partial CSA
    /// trees, and 16-entry padding waste vs 16 columns). The planes of a group
    /// (`GUARD + COL_GROUP * STRIDE`) keep the qword offsets in u16 (`33040 <= 65535`).
    const COL_GROUP: usize = 32;
    /// Parts of the column groups per block in the fold's dynamic task hand-out.
    const PARTS: usize = 2;
    /// Planes per column: the 128 direct planes and a zero guard, which is also the next
    /// column's leading guard. Every term's 128-plane run is flanked by zero planes, so a window
    /// of two consecutive slots reads a zero wherever the term does not reach, with no edge
    /// cases in the schedule.
    const STRIDE: usize = 129;
    /// Plane index of column `j`'s first direct plane; `GUARD + j * STRIDE - 1` is its leading
    /// zero guard, and planes 0 and 1 (the padding pair of a short chunk) are both zero.
    const GUARD: usize = 2;
    /// Slot pairs `(2q, 2q + 1)`; the odd slot of the last pair (243) is a dummy.
    const PAIRS: usize = SLOTS.div_ceil(2);
    /// Entries a group's schedule may add to a slot pair beyond one per term: padding of its
    /// positive and of its negative entries to multiples of 2.
    const PAD_PER_GROUP: usize = 2;
    // The offsets use x86's scale-8 addressing directly, avoiding a shift per input.
    const _: () = assert!((GUARD + COL_GROUP * STRIDE) * 8 <= u16::MAX as usize);

    /// Whether the counters cannot overflow: a slot's counter adds at most one entry per
    /// challenge term and the padding of every group.
    pub(super) fn fits(challenges: &[ShortChallenge]) -> bool {
        let weight: usize = challenges.iter().map(|c| c.weight).sum();
        weight + PAD_PER_GROUP * challenges.len().div_ceil(COL_GROUP) <= i16::MAX as usize
    }

    struct Group {
        first: usize,
        columns: usize,
        /// `terms[offsets[q] .. mid[q]]` and `terms[mid[q] .. offsets[q + 1]]` are the
        /// positive and the negative entries of slot pair `q`, each padded to a multiple of 2
        /// with offset 0 (planes 0 and 1 are zero): whole 64-input chunks, then binary tail
        /// trees of sizes 32, 16, 8, 4, 2.
        offsets: [usize; PAIRS + 1],
        mid: [usize; PAIRS],
        terms: Vec<u16>,
    }

    /// The schedule of column group `g` (columns `cs`), and per slot pair the number of its
    /// negative entries (padding included): each adds one to the bias of both slots.
    fn plan_group<const GROUP: usize>(g: usize, cs: &[ShortChallenge]) -> (Group, [i32; PAIRS]) {
        // A term's run `[m, end)` of slots meets slot pair `q` when `2q <= end - 1` and
        // `2q + 1 >= m`, i.e. `q` in `[m / 2, (end - 1) / 2]`; the wrapped run `[0, wrap)`
        // meets pairs `[0, (wrap - 1) / 2]`, with the opposite sign (`Y^243 = -1`).
        // `count_diff[negative]`: the entries of each sign per pair, as range differences.
        let mut count_diff = [[0i32; PAIRS + 1]; 2];
        for c in cs.iter() {
            assert!(c.weight <= c.positions.len());
            for t in 0..c.weight {
                let m = c.positions[t] as usize;
                assert!(m < 162);
                let negative = ((c.signs >> t) & 1 != 0) ^ (m % 2 != 0);
                let end = (m + 128).min(SLOTS);
                count_diff[negative as usize][m / 2] += 1;
                count_diff[negative as usize][(end + 1) / 2] -= 1;
                let wrap = (m + 128).saturating_sub(SLOTS);
                if wrap != 0 {
                    count_diff[!negative as usize][0] += 1;
                    count_diff[!negative as usize][(wrap + 1) / 2] -= 1;
                }
            }
        }
        let mut group = Group {
            first: g * GROUP,
            columns: cs.len(),
            offsets: [0; PAIRS + 1],
            mid: [0; PAIRS],
            terms: Vec::new(),
        };
        let mut negatives = [0i32; PAIRS];
        let mut count = [0i32; 2];
        for q in 0..PAIRS {
            for (c, diff) in count.iter_mut().zip(&count_diff) {
                *c += diff[q];
                debug_assert!(*c >= 0);
            }
            let (pos, neg) = ((count[0] as usize + 1) & !1, (count[1] as usize + 1) & !1);
            group.mid[q] = group.offsets[q] + pos;
            group.offsets[q + 1] = group.mid[q] + neg;
            negatives[q] = neg as i32;
        }
        // Offset 0 is the padding pair (planes 0 and 1 are zero), not a column's plane.
        // Only the last entry of a non-empty sign range can be padding (at most 1 entry for
        // even alignment); presetting that slot to 0 avoids zeroing the whole schedule vector.
        let total_terms = group.offsets[PAIRS];
        group.terms = Vec::with_capacity(total_terms);
        unsafe {
            let terms_ptr = group.terms.as_mut_ptr();
            for q in 0..PAIRS {
                if group.mid[q] > group.offsets[q] {
                    *terms_ptr.add(group.mid[q] - 1) = 0;
                }
                if group.offsets[q + 1] > group.mid[q] {
                    *terms_ptr.add(group.offsets[q + 1] - 1) = 0;
                }
            }
            let mut at: [[u32; PAIRS]; 2] = [
                core::array::from_fn(|q| group.offsets[q] as u32),
                core::array::from_fn(|q| group.mid[q] as u32),
            ];
            for (j, c) in cs.iter().enumerate() {
                let column = GUARD + j * STRIDE;
                for t in 0..c.weight {
                    let m = c.positions[t] as usize;
                    let negative = ((c.signs >> t) & 1 != 0) ^ (m % 2 != 0);
                    let end = (m + 128).min(SLOTS);
                    // The plane of slot `2q` for the run starting at slot `m`: `column + 2q - m`, which
                    // is the guard before the run when `2q = m - 1`; the odd slot reads one plane
                    // further, the guard after the run when `2q + 1 = end`.
                    let q0 = m / 2;
                    let q1 = (end + 1) / 2;
                    let at_ptr = at[negative as usize].as_mut_ptr();
                    let mut val = (8 * (column + 2 * q0).wrapping_sub(m)) as u16;
                    for q in q0..q1 {
                        let p = at_ptr.add(q);
                        let idx = *p as usize;
                        debug_assert!(idx < if negative { group.offsets[q + 1] } else { group.mid[q] });
                        *p = idx as u32 + 1;
                        *terms_ptr.add(idx) = val;
                        val = val.wrapping_add(16);
                    }
                    let wrap = (m + 128).saturating_sub(SLOTS);
                    if wrap != 0 {
                        let q1 = (wrap + 1) / 2;
                        let at_ptr = at[!negative as usize].as_mut_ptr();
                        let mut val = (8 * (column + (end - m))) as u16;
                        for q in 0..q1 {
                            let p = at_ptr.add(q);
                            let idx = *p as usize;
                            debug_assert!(idx < if !negative { group.offsets[q + 1] } else { group.mid[q] });
                            *p = idx as u32 + 1;
                            *terms_ptr.add(idx) = val;
                            val = val.wrapping_add(16);
                        }
                    }
                }
            }
            group.terms.set_len(total_terms);
        }
        (group, negatives)
    }

    fn plan<const GROUP: usize>(challenges: &[ShortChallenge], workers: usize) -> (Vec<Group>, [i32; SLOTS]) {
        let jobs = challenges.len().div_ceil(GROUP);
        let group = |g: usize| {
            let first = g * GROUP;
            let last = (first + GROUP).min(challenges.len());
            plan_group::<GROUP>(g, &challenges[first..last])
        };
        // The schedule is independent for each public column group; the pool's parked
        // workers take the groups, and the merge keeps column order.
        let built: Vec<(Group, [i32; PAIRS])> = if jobs < 4 || workers <= 1 {
            (0..jobs).map(group).collect()
        } else {
            labinius::par::pool().map(jobs, group)
        };
        let mut groups = Vec::with_capacity(jobs);
        let mut negatives = [0i32; PAIRS];
        for (group, n) in built {
            groups.push(group);
            for (dst, x) in negatives.iter_mut().zip(n) { *dst += x; }
        }
        let bias = core::array::from_fn(|s| negatives[s / 2]);
        (groups, bias)
    }

    #[repr(C, align(64))]
    struct Aligned<T>(T);

    const fn swap_index(mask: usize, value: usize) -> [u8; 64] {
        let mut out = [0; 64];
        let mut p = 0;
        while p < 64 {
            out[p] = (64 * ((p & mask != 0) as usize) + (p & !mask) + value) as u8;
            p += 1;
        }
        out
    }

    static SWAPS: Aligned<[[[u8; 64]; 2]; 4]> = Aligned([
        [swap_index(8, 0), swap_index(8, 8)],
        [swap_index(4, 0), swap_index(4, 4)],
        [swap_index(2, 0), swap_index(2, 2)],
        [swap_index(1, 0), swap_index(1, 1)],
    ]);

    /// Exactly bitmat's byte transpose, specialized to four fixed rounds. Both the vector-slot
    /// bit and the byte-position bit exchanged in round r are 3-r. Constant pair indices let
    /// the compiler keep all sixteen vectors in registers instead of indexing a stack array.
    #[inline(always)]
unsafe fn transpose_tile(src: *const B128, dst: *mut __m512i) {
        let mut z: [__m512i; 16] = core::array::from_fn(|i| {
            _mm512_loadu_si512((src as *const u8).add(64 * i) as *const __m512i)
        });
        macro_rules! pair {
            ($a:literal, $b:literal, $i0:ident, $i1:ident) => {{
                let (a, b) = (z[$a], z[$b]);
                z[$a] = _mm512_permutex2var_epi8(a, $i0, b);
                z[$b] = _mm512_permutex2var_epi8(a, $i1, b);
            }};
        }
        let i0 = _mm512_load_si512(SWAPS.0[0][0].as_ptr() as *const __m512i);
        let i1 = _mm512_load_si512(SWAPS.0[0][1].as_ptr() as *const __m512i);
        pair!(0, 8, i0, i1); pair!(1, 9, i0, i1); pair!(2, 10, i0, i1); pair!(3, 11, i0, i1);
        pair!(4, 12, i0, i1); pair!(5, 13, i0, i1); pair!(6, 14, i0, i1); pair!(7, 15, i0, i1);
        let i0 = _mm512_load_si512(SWAPS.0[1][0].as_ptr() as *const __m512i);
        let i1 = _mm512_load_si512(SWAPS.0[1][1].as_ptr() as *const __m512i);
        pair!(0, 4, i0, i1); pair!(1, 5, i0, i1); pair!(2, 6, i0, i1); pair!(3, 7, i0, i1);
        pair!(8, 12, i0, i1); pair!(9, 13, i0, i1); pair!(10, 14, i0, i1); pair!(11, 15, i0, i1);
        let i0 = _mm512_load_si512(SWAPS.0[2][0].as_ptr() as *const __m512i);
        let i1 = _mm512_load_si512(SWAPS.0[2][1].as_ptr() as *const __m512i);
        pair!(0, 2, i0, i1); pair!(1, 3, i0, i1); pair!(4, 6, i0, i1); pair!(5, 7, i0, i1);
        pair!(8, 10, i0, i1); pair!(9, 11, i0, i1); pair!(12, 14, i0, i1); pair!(13, 15, i0, i1);
        let i0 = _mm512_load_si512(SWAPS.0[3][0].as_ptr() as *const __m512i);
        let i1 = _mm512_load_si512(SWAPS.0[3][1].as_ptr() as *const __m512i);
        pair!(0, 1, i0, i1); pair!(2, 3, i0, i1); pair!(4, 5, i0, i1); pair!(6, 7, i0, i1);
        pair!(8, 9, i0, i1); pair!(10, 11, i0, i1); pair!(12, 13, i0, i1); pair!(14, 15, i0, i1);
        for i in 0..16 {
            _mm512_store_si512(dst.add(i), z[i]);
        }
    }

    /// Byte transpose with lane-local unpack stages and restored column order.
    #[inline(always)]
    unsafe fn transpose_tile_unpack(src: *const B128, dst: *mut __m512i) {
    let mut z: [__m512i; 16] = core::array::from_fn(|i| {
        _mm512_loadu_si512((src as *const u8).add(64 * i) as *const __m512i)
    });
    macro_rules! pair {
        ($a:literal, $b:literal, $lo:ident, $hi:ident) => {{
            let (a, b) = (z[$a], z[$b]);
            z[$a] = $lo(a, b);
            z[$b] = $hi(a, b);
        }};
    }
    pair!(0, 1, _mm512_unpacklo_epi8, _mm512_unpackhi_epi8);
    pair!(2, 3, _mm512_unpacklo_epi8, _mm512_unpackhi_epi8);
    pair!(4, 5, _mm512_unpacklo_epi8, _mm512_unpackhi_epi8);
    pair!(6, 7, _mm512_unpacklo_epi8, _mm512_unpackhi_epi8);
    pair!(8, 9, _mm512_unpacklo_epi8, _mm512_unpackhi_epi8);
    pair!(10, 11, _mm512_unpacklo_epi8, _mm512_unpackhi_epi8);
    pair!(12, 13, _mm512_unpacklo_epi8, _mm512_unpackhi_epi8);
    pair!(14, 15, _mm512_unpacklo_epi8, _mm512_unpackhi_epi8);
    pair!(0, 2, _mm512_unpacklo_epi16, _mm512_unpackhi_epi16);
    pair!(1, 3, _mm512_unpacklo_epi16, _mm512_unpackhi_epi16);
    pair!(4, 6, _mm512_unpacklo_epi16, _mm512_unpackhi_epi16);
    pair!(5, 7, _mm512_unpacklo_epi16, _mm512_unpackhi_epi16);
    pair!(8, 10, _mm512_unpacklo_epi16, _mm512_unpackhi_epi16);
    pair!(9, 11, _mm512_unpacklo_epi16, _mm512_unpackhi_epi16);
    pair!(12, 14, _mm512_unpacklo_epi16, _mm512_unpackhi_epi16);
    pair!(13, 15, _mm512_unpacklo_epi16, _mm512_unpackhi_epi16);
    pair!(0, 4, _mm512_unpacklo_epi32, _mm512_unpackhi_epi32);
    pair!(1, 5, _mm512_unpacklo_epi32, _mm512_unpackhi_epi32);
    pair!(2, 6, _mm512_unpacklo_epi32, _mm512_unpackhi_epi32);
    pair!(3, 7, _mm512_unpacklo_epi32, _mm512_unpackhi_epi32);
    pair!(8, 12, _mm512_unpacklo_epi32, _mm512_unpackhi_epi32);
    pair!(9, 13, _mm512_unpacklo_epi32, _mm512_unpackhi_epi32);
    pair!(10, 14, _mm512_unpacklo_epi32, _mm512_unpackhi_epi32);
    pair!(11, 15, _mm512_unpacklo_epi32, _mm512_unpackhi_epi32);
    pair!(0, 8, _mm512_unpacklo_epi64, _mm512_unpackhi_epi64);
    pair!(1, 9, _mm512_unpacklo_epi64, _mm512_unpackhi_epi64);
    pair!(2, 10, _mm512_unpacklo_epi64, _mm512_unpackhi_epi64);
    pair!(3, 11, _mm512_unpacklo_epi64, _mm512_unpackhi_epi64);
    pair!(4, 12, _mm512_unpacklo_epi64, _mm512_unpackhi_epi64);
    pair!(5, 13, _mm512_unpacklo_epi64, _mm512_unpackhi_epi64);
    pair!(6, 14, _mm512_unpacklo_epi64, _mm512_unpackhi_epi64);
    pair!(7, 15, _mm512_unpacklo_epi64, _mm512_unpackhi_epi64);
    // The unpack tree reverses the four column-index bits.
    // Reorder stores; no instruction is needed to reorder vector contents.
    for i in 0usize..16 {
        _mm512_store_si512(dst.add(i), z[i.reverse_bits() >> (usize::BITS - 4)]);
    }
}

    // Byte transpose inside each vector, and its inverse with reversed GFNI matrix rows and
    // qwords pre-ordered `[0, 4, 1, 5, 2, 6, 3, 7]` so `vpunpcklbw`/`vpunpckhbw` emit the
    // natural 64-lane order directly without a follow-up `vpermt2q`.
    const PERM: Aligned<[u8; 64]> = Aligned({
        let mut a = [0; 64];
        let mut i = 0;
        while i < 64 {
            a[i] = (8 * (i % 8) + i / 8) as u8;
            i += 1;
        }
        a
    });
    const BACK: Aligned<[u8; 64]> = Aligned({
        let mut a = [0; 64];
        let mut i = 0;
        while i < 64 {
            let q = i / 8;
            let src_q = (q / 2) + 4 * (q % 2);
            a[i] = (8 * (7 - i % 8) + src_q) as u8;
            i += 1;
        }
        a
    });

    /// Lane l corresponds to this source word. GFNI reverses the eight matrix rows; the byte
    /// transpose uses bitmat's public word permutation. No permutation is needed in the hot loop;
    /// [`write_block`] relies on its shape (lane `64 g + 16 k + r` is word `64 g + 4 (r ^ 7) + k`).
    #[cfg_attr(not(test), allow(dead_code))]
    fn word_of(l: usize) -> usize {
        64 * (l / 64) + bitmat::word_at((l % 64 & !7) + 7 - l % 8)
    }

    /// The 128 planes of a block of 512 words to `out`; the guard planes around them are the
    /// caller's.
    ///
    /// The 8 KB the next call slices (`next`, any readable address) are prefetched a tile at a
    /// time: the blocks of a group are 256 KB apart, two fresh pages each, which the hardware
    /// prefetchers only start on after the first misses, and the table comes from DRAM.
    #[inline(always)]
    unsafe fn slice_with<const UNPACK: bool>(words: *const B128, out: *mut __m512i, next: *const B128) {
        let mut tiles = core::mem::MaybeUninit::<[[__m512i; 16]; 8]>::uninit();
        let tiles_ptr = tiles.as_mut_ptr() as *mut __m512i;
        for t in 0..8 {
            for i in 0..16 {
                _mm_prefetch::<{ _MM_HINT_T0 }>((next as *const i8).add(1024 * t + 64 * i));
            }
            if UNPACK {
                transpose_tile_unpack(words.add(64 * t), tiles_ptr.add(16 * t));
            } else {
                transpose_tile(words.add(64 * t), tiles_ptr.add(16 * t));
            }
        }
        let perm = _mm512_load_si512(PERM.0.as_ptr() as *const __m512i);
        let ident = _mm512_set1_epi64(bitmat::GF_IDENT);
        for b in 0..16 {
            let rows = core::array::from_fn(|t| {
                _mm512_permutexvar_epi8(
                    perm,
                    _mm512_gf2p8affine_epi64_epi8::<0>(ident, _mm512_load_si512(tiles_ptr.add(16 * t + b))),
                )
            });
            let planes = transpose8x8_q(rows);
            for p in 0..8 {
                _mm512_store_si512(out.add(8 * b + p), planes[p]);
            }
        }
    }
    #[inline(always)]
    unsafe fn slice(words: *const B128, out: *mut __m512i, next: *const B128) {
        slice_with::<true>(words, out, next);
    }


    /// Three equally weighted bit planes -> sum and double-weight carry.
    #[inline(always)]
    unsafe fn csa(sum: &mut __m512i, a: __m512i, b: __m512i) -> __m512i {
        // `vpternlogq` overwrites its first operand, so taking both outputs from the three
        // originals costs a register copy per full adder. Instead the sum is formed in place
        // and the carry is read off `(a, sum', b)`: with `sum = sum' ^ a ^ b`, the majority is
        // `a` when `a == b` and `!sum'` otherwise, truth table 0xb2. The carry can then land on
        // `a` or `b`, both dead afterwards.
        *sum = _mm512_ternarylogic_epi64::<0x96>(*sum, a, b);
        _mm512_ternarylogic_epi64::<0xb2>(a, *sum, b)
    }

    /// [`csa`] of `sum` and the complements of `a` and `b`: the sum is the same (the two
    /// complements cancel), and the carry `maj(sum, !a, !b)` is `!a` when `a == b` and the
    /// complement of the new sum otherwise, truth table 0x17 on `(a, sum', b)`.
    #[inline(always)]
    unsafe fn csa_neg(sum: &mut __m512i, a: __m512i, b: __m512i) -> __m512i {
        *sum = _mm512_ternarylogic_epi64::<0x96>(*sum, a, b);
        _mm512_ternarylogic_epi64::<0x17>(a, *sum, b)
    }

    // Isolate the counter's register allocation from the transpose's many vector constants.
    #[allow(dead_code)]
    #[inline(never)]
    unsafe fn accumulate(planes: *const __m512i, terms: &[u16], acc: *mut __m512i) {
        let planes = planes as *const u64;
        let mut b0 = _mm512_load_si512(acc);
        let mut b1 = _mm512_load_si512(acc.add(1));
        let mut b2 = _mm512_load_si512(acc.add(2));
        let mut b3 = _mm512_load_si512(acc.add(3));
        let mut b4 = _mm512_load_si512(acc.add(4));
        let mut b5 = _mm512_load_si512(acc.add(5));
        let mut b6 = _mm512_load_si512(acc.add(6));
        let mut b7 = _mm512_load_si512(acc.add(7));
        let mut b8 = _mm512_load_si512(acc.add(8));
        let mut b9 = _mm512_load_si512(acc.add(9));
        let mut b10 = _mm512_load_si512(acc.add(10));
        let mut b11 = _mm512_load_si512(acc.add(11));
        let mut b12 = _mm512_load_si512(acc.add(12));
        let mut b13 = _mm512_load_si512(acc.add(13));
        let mut b14 = _mm512_load_si512(acc.add(14));
        for input in terms.chunks_exact(64) {
            let t = input.as_ptr();
            macro_rules! two {
                ($off:expr) => {{
                    let a = _mm512_load_si512(planes.add(*t.add($off) as usize) as *const __m512i);
                    let b = _mm512_load_si512(planes.add(*t.add($off + 1) as usize) as *const __m512i);
                    csa(&mut b0, a, b)
                }};
            }
            macro_rules! four {
                ($off:expr) => {{ let a = two!($off); let b = two!($off + 2); csa(&mut b1, a, b) }};
            }
            macro_rules! eight {
                ($off:expr) => {{ let a = four!($off); let b = four!($off + 4); csa(&mut b2, a, b) }};
            }
            macro_rules! sixteen {
                ($off:expr) => {{ let a = eight!($off); let b = eight!($off + 8); csa(&mut b3, a, b) }};
            }
            macro_rules! thirty_two {
                ($off:expr) => {{ let a = sixteen!($off); let b = sixteen!($off + 16); csa(&mut b4, a, b) }};
            }
            let a = thirty_two!(0);
            let b = thirty_two!(32);
            let mut carry = csa(&mut b5, a, b);
            macro_rules! ripple {
                ($bit:ident) => {{
                    let old = $bit;
                    $bit = _mm512_xor_si512(old, carry);
                    carry = _mm512_and_si512(old, carry);
                }};
            }
            ripple!(b6); ripple!(b7); ripple!(b8); ripple!(b9);
            ripple!(b10); ripple!(b11); ripple!(b12); ripple!(b13);
            b14 = _mm512_xor_si512(b14, carry);
        }
        _mm512_store_si512(acc, b0);
        _mm512_store_si512(acc.add(1), b1);
        _mm512_store_si512(acc.add(2), b2);
        _mm512_store_si512(acc.add(3), b3);
        _mm512_store_si512(acc.add(4), b4);
        _mm512_store_si512(acc.add(5), b5);
        _mm512_store_si512(acc.add(6), b6);
        _mm512_store_si512(acc.add(7), b7);
        _mm512_store_si512(acc.add(8), b8);
        _mm512_store_si512(acc.add(9), b9);
        _mm512_store_si512(acc.add(10), b10);
        _mm512_store_si512(acc.add(11), b11);
        _mm512_store_si512(acc.add(12), b12);
        _mm512_store_si512(acc.add(13), b13);
        _mm512_store_si512(acc.add(14), b14);
    }

    /// Two consecutive slots at once: every scheduled offset is the plane of the even slot, and
    /// the odd slot's plane is the next one, so one scalar offset serves two vector loads. Each
    /// slot keeps the six low counter bits of a 64-input tree in registers, plus a three-bit
    /// counter of the trees' weight-64 carries; every seventh chunk that counter is added into
    /// the nine upper counter bits in memory. The `pos` entries add their planes, the `neg`
    /// entries their complements (leaf full adders by [`csa_neg`]).
    #[target_feature(enable = "avx512f")]
    #[inline(never)]
    pub(super) unsafe fn accumulate_pair(planes: *const __m512i, pos: &[u16], neg: &[u16], acc: *mut __m512i) {
        let planes = planes as *const u64;
        let (even, odd) = (acc, acc.add(BITS));
        let mut s0: [__m512i; 6] = core::array::from_fn(|b| _mm512_load_si512(even.add(b)));
        let mut s1: [__m512i; 6] = core::array::from_fn(|b| _mm512_load_si512(odd.add(b)));
        // The weight-64 carries of up to seven chunks, in registers at weights 64, 128 and 256;
        // every seventh chunk they are added into the nine upper counter bits in memory.
        let zero = _mm512_setzero_si512();
        let (mut h0, mut h1) = ([zero; 3], [zero; 3]);
        let mut pending = 0usize;
        macro_rules! flush {
            ($slot:ident, $h:ident) => {{
                let p = $slot.add(6);
                let m6 = _mm512_load_si512(p);
                _mm512_store_si512(p, _mm512_xor_si512(m6, $h[0]));
                let mut carry = _mm512_and_si512(m6, $h[0]);
                let p = $slot.add(7);
                let m7 = _mm512_load_si512(p);
                let x = _mm512_ternarylogic_epi64::<0x96>(m7, $h[1], carry);
                _mm512_store_si512(p, x);
                carry = _mm512_ternarylogic_epi64::<0xb2>($h[1], x, carry);
                let p = $slot.add(8);
                let m8 = _mm512_load_si512(p);
                let x = _mm512_ternarylogic_epi64::<0x96>(m8, $h[2], carry);
                _mm512_store_si512(p, x);
                carry = _mm512_ternarylogic_epi64::<0xb2>($h[2], x, carry);
                for b in 9..BITS - 1 {
                    let p = $slot.add(b);
                    let old = _mm512_load_si512(p);
                    _mm512_store_si512(p, _mm512_xor_si512(old, carry));
                    carry = _mm512_and_si512(old, carry);
                }
                let p = $slot.add(BITS - 1);
                _mm512_store_si512(p, _mm512_xor_si512(_mm512_load_si512(p), carry));
            }};
        }
        let mut t: *const u16;
        macro_rules! two {
            ($off:expr, $leaf:ident) => {{
                let pa = planes.add(*t.add($off) as usize);
                let pb = planes.add(*t.add($off + 1) as usize);
                let a0 = _mm512_load_si512(pa as *const __m512i);
                let b0 = _mm512_load_si512(pb as *const __m512i);
                let a1 = _mm512_load_si512(pa.add(8) as *const __m512i);
                let b1 = _mm512_load_si512(pb.add(8) as *const __m512i);
                ($leaf(&mut s0[0], a0, b0), $leaf(&mut s1[0], a1, b1))
            }};
        }
        macro_rules! level {
            ($bit:literal, $lower:ident, $off:expr, $leaf:ident) => {{
                let (x0, x1) = $lower!($off, $leaf);
                let (y0, y1) = $lower!($off + (1 << $bit), $leaf);
                (csa(&mut s0[$bit], x0, y0), csa(&mut s1[$bit], x1, y1))
            }};
        }
        macro_rules! four { ($off:expr, $leaf:ident) => { level!(1, two, $off, $leaf) }; }
        macro_rules! eight { ($off:expr, $leaf:ident) => { level!(2, four, $off, $leaf) }; }
        macro_rules! sixteen { ($off:expr, $leaf:ident) => { level!(3, eight, $off, $leaf) }; }
        macro_rules! thirty_two { ($off:expr, $leaf:ident) => { level!(4, sixteen, $off, $leaf) }; }
        // Three-bit register counters of the weight-64 carries: seven fit before a flush.
        macro_rules! add_carry64 {
            ($c0:expr, $c1:expr) => {{
                let (c0, c1) = ($c0, $c1);
                let u0 = _mm512_and_si512(h0[0], c0);
                let v0 = _mm512_and_si512(h0[1], u0);
                h0[0] = _mm512_xor_si512(h0[0], c0);
                h0[1] = _mm512_xor_si512(h0[1], u0);
                h0[2] = _mm512_xor_si512(h0[2], v0);
                let u1 = _mm512_and_si512(h1[0], c1);
                let v1 = _mm512_and_si512(h1[1], u1);
                h1[0] = _mm512_xor_si512(h1[0], c1);
                h1[1] = _mm512_xor_si512(h1[1], u1);
                h1[2] = _mm512_xor_si512(h1[2], v1);
            }};
        }
        macro_rules! step_pending {
            () => {{
                pending += 1;
                if pending == 7 {
                    flush!(even, h0);
                    flush!(odd, h1);
                    h0 = [zero; 3];
                    h1 = [zero; 3];
                    pending = 0;
                }
            }};
        }
        // A carry of weight `2^$bit` from a partial tree, through half adders at that level and
        // above: what leaves level 5 is a weight-64 carry like a whole chunk's. Because the
        // partial trees of a remainder total at most 62 < 64 inputs and `s0[0..6] <= 63`,
        // `63 + 62 = 125 < 128`, so across the entire remainder `s0[0..6]` wraps modulo 64 at
        // most once per lane (`pending` advances by 1 for the whole non-empty remainder).
        macro_rules! half_adders {
            ($c0:expr, $c1:expr, $($bit:literal),+) => {{
                let (mut c0, mut c1) = ($c0, $c1);
                $(
                    let (u0, u1) = (_mm512_and_si512(s0[$bit], c0), _mm512_and_si512(s1[$bit], c1));
                    s0[$bit] = _mm512_xor_si512(s0[$bit], c0);
                    s1[$bit] = _mm512_xor_si512(s1[$bit], c1);
                    (c0, c1) = (u0, u1);
                )+
                add_carry64!(c0, c1);
            }};
        }
        macro_rules! entries {
            ($terms:expr, $leaf:ident) => {{
                let terms: &[u16] = $terms;
                for input in terms.chunks_exact(64) {
                    t = input.as_ptr();
                    let (c0, c1) = level!(5, thirty_two, 0, $leaf);
                    add_carry64!(c0, c1);
                    step_pending!();
                }
                // The schedule pads each sign of a slot pair to a multiple of 2: the remainder
                // (at most 62 inputs) decomposes into binary partial trees of sizes 32, 16, 8, 4, 2.
                let rest = terms.chunks_exact(64).remainder();
                debug_assert!(rest.len() % 2 == 0);
                if !rest.is_empty() {
                    t = rest.as_ptr();
                    let n = rest.len();
                    if n & 32 != 0 {
                        let (c0, c1) = thirty_two!(0, $leaf);
                        half_adders!(c0, c1, 5);
                        t = t.add(32);
                    }
                    if n & 16 != 0 {
                        let (c0, c1) = sixteen!(0, $leaf);
                        half_adders!(c0, c1, 4, 5);
                        t = t.add(16);
                    }
                    if n & 8 != 0 {
                        let (c0, c1) = eight!(0, $leaf);
                        half_adders!(c0, c1, 3, 4, 5);
                        t = t.add(8);
                    }
                    if n & 4 != 0 {
                        let (c0, c1) = four!(0, $leaf);
                        half_adders!(c0, c1, 2, 3, 4, 5);
                        t = t.add(4);
                    }
                    if n & 2 != 0 {
                        let (c0, c1) = two!(0, $leaf);
                        half_adders!(c0, c1, 1, 2, 3, 4, 5);
                    }
                    step_pending!();
                }
            }};
        }
        entries!(pos, csa);
        entries!(neg, csa_neg);
        if pending != 0 {
            flush!(even, h0);
            flush!(odd, h1);
        }
        for b in 0..6 {
            _mm512_store_si512(even.add(b), s0[b]);
            _mm512_store_si512(odd.add(b), s1[b]);
        }
    }

    /// Inverse bit transpose to 512 ordinary 16-bit lane counts.
    /// `out` addresses 512 writable `i16` values; they need not be initialized.
    #[inline(always)]
    unsafe fn unslice(acc: *const __m512i, out: *mut i16) {
        let zero = _mm512_setzero_si512();
        let lo = transpose8x8_q(core::array::from_fn(|b| _mm512_load_si512(acc.add(b))));
        let hi = transpose8x8_q(core::array::from_fn(|b| {
            if b == 7 { zero } else { _mm512_load_si512(acc.add(8 + b)) }
        }));
        let perm = _mm512_load_si512(BACK.0.as_ptr() as *const __m512i);
        let ident = _mm512_set1_epi64(bitmat::GF_IDENT);
        for t in 0..8 {
            let l = _mm512_gf2p8affine_epi64_epi8::<0>(ident, _mm512_permutexvar_epi8(perm, lo[t]));
            let h = _mm512_gf2p8affine_epi64_epi8::<0>(ident, _mm512_permutexvar_epi8(perm, hi[t]));
            let dst = out.add(64 * t) as *mut __m512i;
            _mm512_storeu_si512(dst, _mm512_unpacklo_epi8(l, h));
            _mm512_storeu_si512(dst.add(1), _mm512_unpackhi_epi8(l, h));
        }
    }

    /// Coefficient slots of `R_162`: slot `s` of the output takes counter `s` minus counter
    /// `162 + s` below 81 and counter `s` plus counter `s + 81` from 81 on.
    const OUT_SLOTS: usize = 162;

    /// `vpermt2w` indices interleaving a 64-lane group's four components: `[word_of]` puts
    /// component `k` of element `16 g + (r ^ 7)` on lane `64 g + 16 k + r`, so half `h` of the
    /// result holds, in qword `j`, the four components of element `16 g + 8 h + 7 - j`.
    static INTERLEAVE: Aligned<[[u16; 32]; 2]> = Aligned({
        let mut a = [[0u16; 32]; 2];
        let mut h = 0;
        while h < 2 {
            let mut p = 0;
            while p < 32 {
                a[h][p] = (16 * (p % 4) + 8 * h + p / 4) as u16;
                p += 1;
            }
            h += 1;
        }
        a
    });

    /// A block's 243 counters, reduced modulo `Y^162 - Y^81 + 1` and with the bias removed, as
    /// its 128 ring elements at `output` (uninitialised memory, every field written).
    ///
    /// Per output slot and 64-lane group the four components interleave into one qword per
    /// element, and eight consecutive slots transpose into one 64-byte store per element. Each
    /// counter minus its bias is exact in an i16 lane (both are at most 32767); the combination
    /// is formed both wrapping and saturating, and any difference is the overflow the scalar
    /// reference reports.
    ///
    /// # Safety
    /// `acc` holds the 243 slots' counters; `output` addresses 128 writable elements.
    #[cfg(test)]
    #[inline(never)]
    unsafe fn write_block(
        acc: *const __m512i,
        counts: &mut [[i16; LANES]],
        bias: &[i32; SLOTS],
        output: *mut RingElement,
    ) {
        debug_assert!(counts.len() >= SLOTS);
        for s in 0..SLOTS {
            unslice(acc.add(s * BITS), counts[s].as_mut_ptr());
        }
        write_counts(counts, None, bias, output);
    }

    /// The output stage of [`write_block`] from counts already unsliced (and optionally summed
    /// with `other` from the other part of the column groups).
    #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,gfni")]
    unsafe fn write_counts(
        counts: &[[i16; LANES]],
        other: Option<&[[i16; LANES]]>,
        bias: &[i32; SLOTS],
        output: *mut RingElement,
    ) {
        debug_assert!(counts.len() >= SLOTS && other.map_or(true, |o| o.len() >= SLOTS));
        for e in 0..LANES / 4 {
            core::ptr::addr_of_mut!((*output.add(e)).representation).write(Representation::Coefficients);
        }
        let idx0 = _mm512_load_si512(INTERLEAVE.0[0].as_ptr() as *const __m512i);
        let idx1 = _mm512_load_si512(INTERLEAVE.0[1].as_ptr() as *const __m512i);
        let counts_ptr = counts.as_ptr();
        let other_ptr = other.map_or(core::ptr::null(), |o| o.as_ptr());
        let mut overflow = 0u32;
        for s0 in (0..OUT_SLOTS).step_by(8) {
            let n = (OUT_SLOTS - s0).min(8);
            let mask = ((1u32 << n) - 1) as __mmask8;
            for g in 0..LANES / 64 {
                macro_rules! row {
                    ($ds:literal) => {{
                        let s = (s0 + $ds).min(OUT_SLOTS - 1);
                        let below = s < 81;
                        let t = if below { 162 + s } else { s + 81 };
                        let bs = _mm512_set1_epi16(*bias.get_unchecked(s) as i16);
                        let bt = _mm512_set1_epi16(*bias.get_unchecked(t) as i16);
                        let cs_base = (*counts_ptr.add(s)).as_ptr().add(64 * g);
                        let ct_base = (*counts_ptr.add(t)).as_ptr().add(64 * g);
                        let mut cs0 = _mm512_loadu_si512(cs_base as *const __m512i);
                        let mut ct0 = _mm512_loadu_si512(ct_base as *const __m512i);
                        let mut cs1 = _mm512_loadu_si512(cs_base.add(32) as *const __m512i);
                        let mut ct1 = _mm512_loadu_si512(ct_base.add(32) as *const __m512i);
                        if !other_ptr.is_null() {
                            let os_base = (*other_ptr.add(s)).as_ptr().add(64 * g);
                            let ot_base = (*other_ptr.add(t)).as_ptr().add(64 * g);
                            cs0 = _mm512_add_epi16(cs0, _mm512_loadu_si512(os_base as *const __m512i));
                            ct0 = _mm512_add_epi16(ct0, _mm512_loadu_si512(ot_base as *const __m512i));
                            cs1 = _mm512_add_epi16(cs1, _mm512_loadu_si512(os_base.add(32) as *const __m512i));
                            ct1 = _mm512_add_epi16(ct1, _mm512_loadu_si512(ot_base.add(32) as *const __m512i));
                        }
                        let x0 = _mm512_sub_epi16(cs0, bs);
                        let y0 = if below { _mm512_sub_epi16(bt, ct0) } else { _mm512_sub_epi16(ct0, bt) };
                        let v0 = _mm512_add_epi16(x0, y0);
                        overflow |= _mm512_cmpneq_epi16_mask(v0, _mm512_adds_epi16(x0, y0));

                        let x1 = _mm512_sub_epi16(cs1, bs);
                        let y1 = if below { _mm512_sub_epi16(bt, ct1) } else { _mm512_sub_epi16(ct1, bt) };
                        let v1 = _mm512_add_epi16(x1, y1);
                        overflow |= _mm512_cmpneq_epi16_mask(v1, _mm512_adds_epi16(x1, y1));

                        (
                            _mm512_permutex2var_epi16(v0, idx0, v1),
                            _mm512_permutex2var_epi16(v0, idx1, v1),
                        )
                    }};
                }
                let (r0_0, r1_0) = row!(0);
                let (r0_1, r1_1) = row!(1);
                let (r0_2, r1_2) = row!(2);
                let (r0_3, r1_3) = row!(3);
                let (r0_4, r1_4) = row!(4);
                let (r0_5, r1_5) = row!(5);
                let (r0_6, r1_6) = row!(6);
                let (r0_7, r1_7) = row!(7);
                for (h, rows) in [
                    [r0_0, r0_1, r0_2, r0_3, r0_4, r0_5, r0_6, r0_7],
                    [r1_0, r1_1, r1_2, r1_3, r1_4, r1_5, r1_6, r1_7],
                ]
                .into_iter()
                .enumerate()
                {
                    let t = transpose8x8_q(rows);
                    for j in 0..8 {
                        let e = 16 * g + 8 * h + 7 - j;
                        let dst = (core::ptr::addr_of_mut!((*output.add(e)).v) as *mut i16).add(4 * s0);
                        if n == 8 {
                            _mm512_storeu_si512(dst as *mut __m512i, t[j]);
                        } else {
                            _mm512_mask_storeu_epi64(dst as *mut i64, mask, t[j]);
                        }
                    }
                }
            }
        }
        assert!(overflow == 0, "integer fold overflow");
    }

    /// Allocate capacity for `SLOTS` counts; `unslice` initializes every lane before `set_len`.
    #[inline]
    fn uninit_counts() -> Vec<[i16; LANES]> {
        Vec::with_capacity(SLOTS)
    }

    #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,gfni")]
    unsafe fn run_worker<const GROUP: usize, const UNPACK: bool, const CARRY4: bool>(
        worker: usize,
        words: &[B128],
        column_words: usize,
        groups: &[Group],
        bias: &[i32; SLOTS],
        cuts: &[usize],
        parts: usize,
        tasks: usize,
        pending: &[std::sync::Mutex<Option<Vec<[i16; LANES]>>>],
        free: &std::sync::Mutex<Vec<Vec<[i16; LANES]>>>,
        next: &std::sync::atomic::AtomicUsize,
        out_addr: usize,
        #[cfg(test)] profile: bool,
    ) {
        #[cfg(test)]
        let mut times = [std::time::Duration::ZERO; 3];
        #[cfg(not(test))]
        let _ = worker;
        let zero = _mm512_setzero_si512();
        // Zero everywhere the columns do not write: the guards and the padding pair.
        let mut planes: Vec<__m512i> = Vec::with_capacity(GUARD + GROUP * STRIDE);
        planes.as_mut_ptr().write(zero);
        planes.as_mut_ptr().add(1).write(zero);
        for j in 0..GROUP {
            planes.as_mut_ptr().add(GUARD + j * STRIDE + 128).write(zero);
        }
        let mut acc: Vec<__m512i> = Vec::with_capacity(2 * PAIRS * BITS);
        let mut counts = uninit_counts();
        loop {
            let task = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if task >= tasks {
                break;
            }
            let (block, part) = (task / parts, task % parts);
            for p in 0..2 * PAIRS * BITS {
                _mm512_store_si512(acc.as_mut_ptr().add(p), zero);
            }
            let range = cuts[part]..cuts[part + 1];
            for gi in range.clone() {
                let group = &groups[gi];
                #[cfg(test)]
                let start = profile.then(std::time::Instant::now);
                for j in 0..group.columns {
                    let src = words.as_ptr().add((group.first + j) * column_words + block * LANES);
                    // What the next call slices: this group's next column, else the next
                    // group's first, else this block's start (harmless).
                    let next_src = if j + 1 < group.columns {
                        src.add(column_words)
                    } else if gi + 1 < range.end {
                        words.as_ptr().add(groups[gi + 1].first * column_words + block * LANES)
                    } else {
                        src
                    };
                    slice_with::<UNPACK>(src, planes.as_mut_ptr().add(GUARD + j * STRIDE), next_src);
                }
                #[cfg(test)]
                if let Some(start) = start { times[0] += start.elapsed(); }
                #[cfg(test)]
                let start = profile.then(std::time::Instant::now);
                for q in 0..PAIRS {
                    if CARRY4 {
                        super::carry4_kernel::accumulate_pair_carry4_sol(
                            planes.as_ptr(),
                            &group.terms[group.offsets[q]..group.mid[q]],
                            &group.terms[group.mid[q]..group.offsets[q + 1]],
                            acc.as_mut_ptr().add(2 * q * BITS),
                        );
                    } else {
                        accumulate_pair(
                        planes.as_ptr(),
                        &group.terms[group.offsets[q]..group.mid[q]],
                        &group.terms[group.mid[q]..group.offsets[q + 1]],
                        acc.as_mut_ptr().add(2 * q * BITS),
                    );
                    }
                }
                #[cfg(test)]
                if let Some(start) = start { times[1] += start.elapsed(); }
            }
            #[cfg(test)]
            let start = profile.then(std::time::Instant::now);
            for s in 0..SLOTS {
                unslice(acc.as_ptr().add(s * BITS), counts.as_mut_ptr().add(s).cast());
            }
            counts.set_len(SLOTS);
            let output = (out_addr as *mut RingElement).add(block * (LANES / 4));
            if parts == 1 {
                write_counts(&counts, None, bias, output);
                #[cfg(test)]
                if let Some(start) = start { times[2] += start.elapsed(); }
                continue;
            }
            let first = {
                let mut slot = pending[block].lock().unwrap();
                match slot.take() {
                    None => {
                        let next_buf = free.lock().unwrap().pop().unwrap_or_else(uninit_counts);
                        *slot = Some(core::mem::replace(&mut counts, next_buf));
                        #[cfg(test)]
                        if let Some(start) = start { times[2] += start.elapsed(); }
                        continue;
                    }
                    Some(first) => first,
                }
            };
            write_counts(&counts, Some(&first), bias, output);
            free.lock().unwrap().push(first);
            #[cfg(test)]
            if let Some(start) = start { times[2] += start.elapsed(); }
        }
        #[cfg(test)]
        if profile {
            eprintln!("bitfold worker={} slice_us={} add_us={} output_us={}",
                worker, times[0].as_micros(), times[1].as_micros(), times[2].as_micros());
        }
    }

    pub(super) fn fold_words(
        words: &[B128],
        challenges: &[ShortChallenge],
        column_words: usize,
        workers: usize,
    ) -> Vec<RingElement> {
        fold_config::<COL_GROUP>(words, challenges, column_words, workers)
    }

    pub(super) fn fold_words_with<const UNPACK: bool, const CARRY4: bool>(
        words: &[B128],
        challenges: &[ShortChallenge],
        column_words: usize,
        workers: usize,
    ) -> Vec<RingElement> {
        fold_config_with::<COL_GROUP, UNPACK, CARRY4>(words, challenges, column_words, workers)
    }

    fn fold_config<const GROUP: usize>(
        words: &[B128], challenges: &[ShortChallenge], column_words: usize, workers: usize,
    ) -> Vec<RingElement> {
        fold_config_with::<GROUP, true, false>(words, challenges, column_words, workers)
    }

    fn fold_config_with<const GROUP: usize, const UNPACK: bool, const CARRY4: bool>(
        words: &[B128],
        challenges: &[ShortChallenge],
        column_words: usize,
        workers: usize,
    ) -> Vec<RingElement> {
        assert!(GROUP > 0 && (GUARD + GROUP * STRIDE) * 8 <= u16::MAX as usize);
        #[cfg(test)]
        let profile = std::env::var_os("LABINIUS_BITFOLD_PROFILE").is_some();
        #[cfg(test)]
        let plan_start = std::time::Instant::now();
        let (groups, bias) = plan::<GROUP>(challenges, workers);
        #[cfg(test)]
        if profile {
            eprintln!("bitfold plan_us={}", plan_start.elapsed().as_micros());
        }
        let blocks = column_words / LANES;
        let len = column_words / 4;
        // Counters and bias are at most the total weight (checked by the caller), which is
        // what lets the output stage subtract them in i16 lanes.
        assert!(bias.iter().all(|&b| (0..=i16::MAX as i32).contains(&b)));
        // Tasks are (block, part of the column groups), handed out dynamically so a busy or
        // shared core does fewer of them; the second part of a block to finish adds its counts
        // to the first's in `write_counts` (counts are additive and stay within the total weight)
        // and writes.
        let parts = PARTS.min(groups.len().max(1));
        let cuts: Vec<usize> = (0..=parts).map(|i| groups.len() * i / parts).collect();
        let tasks = blocks * parts;
        let pending: Vec<std::sync::Mutex<Option<Vec<[i16; LANES]>>>> =
            (0..blocks).map(|_| std::sync::Mutex::new(None)).collect();
        let free: std::sync::Mutex<Vec<Vec<[i16; LANES]>>> =
            std::sync::Mutex::new(Vec::new());
        let next = std::sync::atomic::AtomicUsize::new(0);
        let mut out: Vec<RingElement> = Vec::with_capacity(len);
        let out_addr = out.as_mut_ptr() as usize;
        let (groups, bias, cuts, pending, free, next) = (&groups, &bias, &cuts, &pending, &free, &next);
        labinius::par::pool().run(workers.clamp(1, tasks), |worker| unsafe {
            run_worker::<GROUP, UNPACK, CARRY4>(
                worker,
                words,
                column_words,
                groups,
                bias,
                cuts,
                parts,
                tasks,
                pending,
                free,
                next,
                out_addr,
                #[cfg(test)]
                profile,
            );
        });
        // Every task wrote its whole range.
        unsafe { out.set_len(len) };
        out
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use labinius::challenge::{sample_short_challenge, Transcript, DEFAULT_BOUND, DEFAULT_WEIGHT};

        fn table(n: usize) -> Vec<B128> {
            let mut h = blake3::Hasher::new_keyed(&[81; 32]);
            h.update(b"labinius-fast/binary-table/v1");
            let mut out = vec![B128(0); n];
            let bytes = unsafe { core::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, 16 * n) };
            h.finalize_xof().fill(bytes);
            out
        }

        fn challenges(n: usize) -> Vec<ShortChallenge> {
            let mut t = Transcript::new(b"labinius/test/bitfold");
            (0..n).map(|_| sample_short_challenge(&mut t, DEFAULT_WEIGHT, DEFAULT_BOUND).0).collect()
        }

        #[test]
        fn bitfold_transpose_and_counter() {
            let words = table(LANES);
            unsafe {
                let zero = _mm512_setzero_si512();
                let mut planes = vec![zero; STRIDE];
                slice(words.as_ptr(), planes.as_mut_ptr(), words.as_ptr());
                for p in 0..128 {
                    let bits: [u64; 8] = core::mem::transmute(planes[p]);
                    for lane in 0..LANES {
                        let expected = (words[word_of(lane)].0 >> p) as u64 & 1;
                        assert_eq!(bits[lane / 64] >> (lane % 64) & 1, expected, "plane {p}, lane {lane}");
                    }
                }
                assert!((128..STRIDE).all(|p| core::mem::transmute::<_, [u64; 8]>(planes[p]) == [0u64; 8]));
                let mut acc = vec![zero; BITS];
                let terms: Vec<u16> = (0..128u16).cycle().take(32704).collect();
                let offsets: Vec<u16> = terms.iter().map(|p| 8 * p).collect();
                accumulate(planes.as_ptr(), &offsets, acc.as_mut_ptr());
                let mut counts = [0i16; LANES];
                unslice(acc.as_ptr(), counts.as_mut_ptr());
                for lane in 0..LANES {
                    let expected: usize = terms.iter().map(|&p| (words[word_of(lane)].0 >> p & 1) as usize).sum();
                    assert_eq!(counts[lane] as usize, expected, "lane {lane}");
                }
                // Exercise the very top of the declared counter range, including zero padding.
                acc.fill(zero);
                planes[0] = zero;
                planes[1] = _mm512_set1_epi32(-1);
                let mut terms = vec![8; 32768];
                terms[32767] = 0;
                accumulate(planes.as_ptr(), &terms, acc.as_mut_ptr());
                unslice(acc.as_ptr(), counts.as_mut_ptr());
                assert_eq!(counts, [32767; LANES]);
            }
        }

        /// The slot-pair kernel on schedules of every length class the plan pads to (whole
        /// chunks plus a 16-, 32- or 48-input tail), in pieces as a block's groups call it, and
        /// up to the top of the counter range.
        #[test]
        fn bitfold_pair_kernel_partial_chunks() {
            let mut x = 0x0123_4567_89AB_CDEFu64;
            let mut next = || {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                x
            };
            unsafe {
                let zero = _mm512_setzero_si512();
                let mut planes = vec![zero; 66];
                for p in planes.iter_mut().skip(2) {
                    let w: [u64; 8] = core::array::from_fn(|_| next() & next());
                    *p = core::mem::transmute(w);
                }
                let bits: Vec<[u64; 8]> = planes.iter().map(|p| core::mem::transmute(*p)).collect();
                let bit = |p: usize, l: usize| (bits[p][l / 64] >> (l % 64) & 1) as usize;
                // Each call takes `n` positive entries and the next length's worth of negative
                // ones (`lens` cycled), so both signs meet every length class, alone and mixed.
                for lens in [
                    vec![16usize],
                    vec![32],
                    vec![48],
                    vec![2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62],
                    vec![80, 112, 144],
                    vec![64 * 7 + 48; 9],
                    vec![16; 40],
                    vec![16000, 752, 15232],
                    vec![0, 64],
                ] {
                    let mut acc = vec![zero; 2 * BITS];
                    let mut want = [[0usize; LANES]; 2];
                    for (i, &n) in lens.iter().enumerate() {
                        let m = lens[(i + 1) % lens.len()];
                        let mut terms = |n: usize| -> Vec<u16> {
                            (0..n).map(|i| {
                                // a sprinkling of padding entries (offset 0: two zero planes)
                                if i % 13 == 5 { 0 } else { (8 * (2 + next() % 63)) as u16 }
                            }).collect()
                        };
                        let (pos, neg) = (terms(n), terms(m));
                        for (list, complement) in [(&pos, 0), (&neg, 1)] {
                            for &o in list {
                                let p = o as usize / 8;
                                for l in 0..LANES {
                                    want[0][l] += bit(p, l) ^ complement;
                                    want[1][l] += bit(p + 1, l) ^ complement;
                                }
                            }
                        }
                        accumulate_pair(planes.as_ptr(), &pos, &neg, acc.as_mut_ptr());
                    }
                    for slot in 0..2 {
                        let mut got = [0i16; LANES];
                        unslice(acc.as_ptr().add(slot * BITS), got.as_mut_ptr());
                        for l in 0..LANES {
                            assert_eq!(got[l] as usize, want[slot][l] & 0x7fff, "lens {lens:?} slot {slot} lane {l}");
                        }
                    }
                }
            }
        }

        /// The vector output stage against the scalar reduction it replaced, on random
        /// counters (the full 15-bit range) and biases chosen so that every value fits, and its
        /// overflow report.
        #[test]
        fn bitfold_write_block_matches_scalar() {
            for l in 0..LANES {
                let (g, k, r) = (l / 64, l % 64 / 16, l % 16);
                assert_eq!(word_of(l), 64 * g + 4 * (r ^ 7) + k, "lane {l}");
            }
            let mut x = 0x9E37_79B9_7F4A_7C15u64;
            let mut next = || {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                x
            };
            unsafe {
                let zero = _mm512_setzero_si512();
                for round in 0..3 {
                    let mut acc = vec![zero; SLOTS * BITS + BITS];
                    for v in acc.iter_mut() {
                        let w: [u64; 8] = core::array::from_fn(|_| next());
                        *v = core::mem::transmute(w);
                    }
                    let mut counts = vec![[0i16; LANES]; SLOTS];
                    for s in 0..SLOTS {
                        unslice(acc.as_ptr().add(s * BITS), counts[s].as_mut_ptr());
                    }
                    // Slots s and their partner t: bias such that the reference fits i16.
                    let mut bias = [0i32; SLOTS];
                    for s in 0..SLOTS {
                        bias[s] = (next() % 32768) as i32;
                    }
                    if round > 0 {
                        for s in 0..81 {
                            // x - z = (a - b_s) - (c - b_t): keep it in range by matching biases
                            // to the counters' means.
                            bias[s] = 16384;
                            bias[162 + s] = 16384;
                            bias[81 + s] = 16384 + (next() % 8) as i32;
                        }
                    }
                    let fits = (0..81).all(|i| {
                        (0..LANES).all(|l| {
                            let (a, b, c) = (counts[i][l] as i32, counts[81 + i][l] as i32, counts[162 + i][l] as i32);
                            let lo = (a - bias[i]) - (c - bias[162 + i]);
                            let hi = (b - bias[81 + i]) + (c - bias[162 + i]);
                            i16::try_from(lo).is_ok() && i16::try_from(hi).is_ok()
                        })
                    });
                    let mut want = vec![RingElement::zero(Representation::Coefficients); LANES / 4];
                    if fits {
                        for i in 0..81 {
                            for l in 0..LANES {
                                let w = word_of(l);
                                let (a, b, c) = (counts[i][l] as i32, counts[81 + i][l] as i32, counts[162 + i][l] as i32);
                                want[w / 4].v[4 * i + w % 4] = ((a - bias[i]) - (c - bias[162 + i])) as i16;
                                want[w / 4].v[4 * (81 + i) + w % 4] = ((b - bias[81 + i]) + (c - bias[162 + i])) as i16;
                            }
                        }
                    }
                    let mut got: Vec<RingElement> = Vec::with_capacity(LANES / 4);
                    let mut scratch = vec![[0i16; LANES]; SLOTS];
                    let result = std::panic::catch_unwind(core::panic::AssertUnwindSafe(|| {
                        write_block(acc.as_ptr(), &mut scratch, &bias, got.as_mut_ptr());
                    }));
                    assert_eq!(result.is_ok(), fits, "round {round}: overflow report");
                    if fits {
                        got.set_len(LANES / 4);
                        assert_eq!(got, want, "round {round}");
                    }
                }
            }
        }

        // Independent schoolbook reduction, including the high terms requiring two reductions.
        fn scalar(words: &[B128], cs: &[ShortChallenge], column_words: usize) -> Vec<RingElement> {
            (0..column_words / 4).map(|r| {
                let mut raw = [0i32; 4 * (128 + 162)];
                for (j, c) in cs.iter().enumerate() {
                    for t in 0..c.weight {
                        let m = c.positions[t] as usize;
                        let sign = if ((c.signs >> t) & 1 != 0) ^ (m % 2 != 0) { -1 } else { 1 };
                        for k in 0..4 {
                            let w = words[j * column_words + 4 * r + k].0;
                            for p in 0..128 {
                                raw[4 * (p + m) + k] += sign * ((w >> p) & 1) as i32;
                            }
                        }
                    }
                }
                for d in (648..raw.len()).rev() {
                    let value = raw[d];
                    raw[d - 324] += value;
                    raw[d - 648] -= value;
                }
                let mut out = RingElement::zero(Representation::Coefficients);
                for (v, x) in out.v.iter_mut().zip(raw) { *v = i16::try_from(x).unwrap(); }
                out
            }).collect()
        }

        #[test]
        fn bitfold_matches_schoolbook_and_ntt() {
            let columns = (COL_GROUP + 2).next_multiple_of(2);
            let words = table(columns * LANES);
            let cs = challenges(columns);
            let exact = fold_words(&words, &cs, LANES, 2);
            assert_eq!(exact, scalar(&words, &cs, LANES));
            let multi = table(columns * 3 * LANES);
            assert_eq!(fold_words(&multi, &cs, 3 * LANES, 2), scalar(&multi, &cs, 3 * LANES));
            for q in [3889, 2917, 4861, 9721, 12637, 17497, 19441] {
                assert_eq!(exact, super::super::fold_witness_streaming(&words, columns, &cs, 4, q, 2), "q {q}");
            }
            // All degrees, both signs, all four X components, and zero/all-one input patterns.
            for sign in [0, 1] {
                let cs: Vec<_> = (0..162).map(|m| {
                    let mut c = ShortChallenge::zero();
                    c.positions[0] = m;
                    c.weight = 1;
                    c.signs = sign;
                    c
                }).collect();
                let words: Vec<_> = (0..cs.len() * LANES).map(|i| B128(if i % 7 == 0 { 0 } else { u128::MAX })).collect();
                assert_eq!(fold_words(&words, &cs, LANES, 2), scalar(&words, &cs, LANES));
            }
        }

        #[test]
        #[ignore = "full 2^24-word timing; run explicitly on the box"]
        fn bitfold_bench_full() {
            let words = table(1 << 24);
            let cs = challenges(1024);
            let workers = labinius::par::threads();
            let mut first: Option<Vec<RingElement>> = None;
            let mut times = Vec::new();
            for _ in 0..7 {
                let start = std::time::Instant::now();
                let v = fold_words(&words, &cs, 1 << 14, workers);
                times.push(start.elapsed().as_micros());
                if let Some(f) = &first { assert_eq!(&v, f); } else { first = Some(v); }
            }
            eprintln!("bitfold bench workers={} us={:?}", workers, times);
        }

        #[test]
        #[ignore = "full 2^24-word challenge instance; run explicitly on the box"]
        fn bitfold_full_instance_matches_streaming() {
            let words = table(1 << 24);
            let cs = challenges(1024);
            let mut exact = None;
            for (group, workers) in [(16, 4), (31, 4), (31, 2), (16, 2)] {
                let start = std::time::Instant::now();
                let v = match group {
                    16 => fold_config::<16>(&words, &cs, 1 << 14, workers),
                    31 => fold_config::<31>(&words, &cs, 1 << 14, workers),
                    _ => unreachable!(),
                };
                eprintln!("bitfold group={} workers={} exact_us={}", group, workers, start.elapsed().as_micros());
                if let Some(previous) = &exact {
                    assert_eq!(&v, previous);
                } else {
                    exact = Some(v);
                }
            }
            let start = std::time::Instant::now();
            let old = super::super::fold_witness_streaming(&words, 1024, &cs, 128, 3889, 4);
            eprintln!("bitfold ntt_us={}", start.elapsed().as_micros());
            assert_eq!(exact.unwrap(), old);
        }
    }

}

mod carry4_kernel {
// Generated by generate.py. Keep all 32 vector registers inside this block.
use core::arch::{asm, x86_64::__m512i};

/// Add even and odd planes. Both lists have even lengths. All vectors are 64-byte aligned.
/// Offset entries use u64 units. Each offset and its next vector must be readable.
/// All counts must remain in 0..=32767. Negative entries add complemented planes.
#[target_feature(enable = "avx512f")]
#[inline(never)]
#[no_mangle]
pub unsafe fn accumulate_pair_carry4_sol(
    planes: *const __m512i, pos: &[u16], neg: &[u16], acc: *mut __m512i,
) {
    debug_assert_eq!(pos.len() % 2, 0);
    debug_assert_eq!(neg.len() % 2, 0);
    asm!(
        "vmovdqa64 zmm0, zmmword ptr [r9 + 0]",
        "vmovdqa64 zmm1, zmmword ptr [r9 + 64]",
        "vmovdqa64 zmm2, zmmword ptr [r9 + 128]",
        "vmovdqa64 zmm3, zmmword ptr [r9 + 192]",
        "vmovdqa64 zmm4, zmmword ptr [r9 + 256]",
        "vmovdqa64 zmm5, zmmword ptr [r9 + 320]",
        "vmovdqa64 zmm6, zmmword ptr [r9 + 960]",
        "vmovdqa64 zmm7, zmmword ptr [r9 + 1024]",
        "vmovdqa64 zmm8, zmmword ptr [r9 + 1088]",
        "vmovdqa64 zmm9, zmmword ptr [r9 + 1152]",
        "vmovdqa64 zmm10, zmmword ptr [r9 + 1216]",
        "vmovdqa64 zmm11, zmmword ptr [r9 + 1280]",
        "vpxorq zmm12, zmm12, zmm12",
        "vpxorq zmm13, zmm13, zmm13",
        "vpxorq zmm14, zmm14, zmm14",
        "vpxorq zmm15, zmm15, zmm15",
        "vpxorq zmm16, zmm16, zmm16",
        "vpxorq zmm17, zmm17, zmm17",
        "vpxorq zmm18, zmm18, zmm18",
        "vpxorq zmm19, zmm19, zmm19",
        "xor r12d, r12d",
        "20:",
        "cmp rdx, 64",
        "jb 21f",
        "movzx eax, word ptr [rsi + 0]",
        "movzx r10d, word ptr [rsi + 2]",
        "vmovdqa64 zmm20, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm22, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm21, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm23, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm20, zmm22, 0x96",
        "vpternlogq zmm20, zmm0, zmm22, 0xb2",
        "vpternlogq zmm6, zmm21, zmm23, 0x96",
        "vpternlogq zmm21, zmm6, zmm23, 0xb2",
        "movzx eax, word ptr [rsi + 4]",
        "movzx r10d, word ptr [rsi + 6]",
        "vmovdqa64 zmm22, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm24, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm23, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm25, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm22, zmm24, 0x96",
        "vpternlogq zmm22, zmm0, zmm24, 0xb2",
        "vpternlogq zmm6, zmm23, zmm25, 0x96",
        "vpternlogq zmm23, zmm6, zmm25, 0xb2",
        "vpternlogq zmm1, zmm20, zmm22, 0x96",
        "vpternlogq zmm20, zmm1, zmm22, 0xb2",
        "vpternlogq zmm7, zmm21, zmm23, 0x96",
        "vpternlogq zmm21, zmm7, zmm23, 0xb2",
        "movzx eax, word ptr [rsi + 8]",
        "movzx r10d, word ptr [rsi + 10]",
        "vmovdqa64 zmm22, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm24, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm23, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm25, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm22, zmm24, 0x96",
        "vpternlogq zmm22, zmm0, zmm24, 0xb2",
        "vpternlogq zmm6, zmm23, zmm25, 0x96",
        "vpternlogq zmm23, zmm6, zmm25, 0xb2",
        "movzx eax, word ptr [rsi + 12]",
        "movzx r10d, word ptr [rsi + 14]",
        "vmovdqa64 zmm24, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm26, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm25, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm27, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm24, zmm26, 0x96",
        "vpternlogq zmm24, zmm0, zmm26, 0xb2",
        "vpternlogq zmm6, zmm25, zmm27, 0x96",
        "vpternlogq zmm25, zmm6, zmm27, 0xb2",
        "vpternlogq zmm1, zmm22, zmm24, 0x96",
        "vpternlogq zmm22, zmm1, zmm24, 0xb2",
        "vpternlogq zmm7, zmm23, zmm25, 0x96",
        "vpternlogq zmm23, zmm7, zmm25, 0xb2",
        "vpternlogq zmm2, zmm20, zmm22, 0x96",
        "vpternlogq zmm20, zmm2, zmm22, 0xb2",
        "vpternlogq zmm8, zmm21, zmm23, 0x96",
        "vpternlogq zmm21, zmm8, zmm23, 0xb2",
        "movzx eax, word ptr [rsi + 16]",
        "movzx r10d, word ptr [rsi + 18]",
        "vmovdqa64 zmm22, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm24, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm23, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm25, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm22, zmm24, 0x96",
        "vpternlogq zmm22, zmm0, zmm24, 0xb2",
        "vpternlogq zmm6, zmm23, zmm25, 0x96",
        "vpternlogq zmm23, zmm6, zmm25, 0xb2",
        "movzx eax, word ptr [rsi + 20]",
        "movzx r10d, word ptr [rsi + 22]",
        "vmovdqa64 zmm24, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm26, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm25, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm27, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm24, zmm26, 0x96",
        "vpternlogq zmm24, zmm0, zmm26, 0xb2",
        "vpternlogq zmm6, zmm25, zmm27, 0x96",
        "vpternlogq zmm25, zmm6, zmm27, 0xb2",
        "vpternlogq zmm1, zmm22, zmm24, 0x96",
        "vpternlogq zmm22, zmm1, zmm24, 0xb2",
        "vpternlogq zmm7, zmm23, zmm25, 0x96",
        "vpternlogq zmm23, zmm7, zmm25, 0xb2",
        "movzx eax, word ptr [rsi + 24]",
        "movzx r10d, word ptr [rsi + 26]",
        "vmovdqa64 zmm24, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm26, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm25, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm27, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm24, zmm26, 0x96",
        "vpternlogq zmm24, zmm0, zmm26, 0xb2",
        "vpternlogq zmm6, zmm25, zmm27, 0x96",
        "vpternlogq zmm25, zmm6, zmm27, 0xb2",
        "movzx eax, word ptr [rsi + 28]",
        "movzx r10d, word ptr [rsi + 30]",
        "vmovdqa64 zmm26, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm28, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm27, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm29, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm26, zmm28, 0x96",
        "vpternlogq zmm26, zmm0, zmm28, 0xb2",
        "vpternlogq zmm6, zmm27, zmm29, 0x96",
        "vpternlogq zmm27, zmm6, zmm29, 0xb2",
        "vpternlogq zmm1, zmm24, zmm26, 0x96",
        "vpternlogq zmm24, zmm1, zmm26, 0xb2",
        "vpternlogq zmm7, zmm25, zmm27, 0x96",
        "vpternlogq zmm25, zmm7, zmm27, 0xb2",
        "vpternlogq zmm2, zmm22, zmm24, 0x96",
        "vpternlogq zmm22, zmm2, zmm24, 0xb2",
        "vpternlogq zmm8, zmm23, zmm25, 0x96",
        "vpternlogq zmm23, zmm8, zmm25, 0xb2",
        "vpternlogq zmm3, zmm20, zmm22, 0x96",
        "vpternlogq zmm20, zmm3, zmm22, 0xb2",
        "vpternlogq zmm9, zmm21, zmm23, 0x96",
        "vpternlogq zmm21, zmm9, zmm23, 0xb2",
        "movzx eax, word ptr [rsi + 32]",
        "movzx r10d, word ptr [rsi + 34]",
        "vmovdqa64 zmm22, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm24, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm23, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm25, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm22, zmm24, 0x96",
        "vpternlogq zmm22, zmm0, zmm24, 0xb2",
        "vpternlogq zmm6, zmm23, zmm25, 0x96",
        "vpternlogq zmm23, zmm6, zmm25, 0xb2",
        "movzx eax, word ptr [rsi + 36]",
        "movzx r10d, word ptr [rsi + 38]",
        "vmovdqa64 zmm24, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm26, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm25, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm27, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm24, zmm26, 0x96",
        "vpternlogq zmm24, zmm0, zmm26, 0xb2",
        "vpternlogq zmm6, zmm25, zmm27, 0x96",
        "vpternlogq zmm25, zmm6, zmm27, 0xb2",
        "vpternlogq zmm1, zmm22, zmm24, 0x96",
        "vpternlogq zmm22, zmm1, zmm24, 0xb2",
        "vpternlogq zmm7, zmm23, zmm25, 0x96",
        "vpternlogq zmm23, zmm7, zmm25, 0xb2",
        "movzx eax, word ptr [rsi + 40]",
        "movzx r10d, word ptr [rsi + 42]",
        "vmovdqa64 zmm24, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm26, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm25, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm27, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm24, zmm26, 0x96",
        "vpternlogq zmm24, zmm0, zmm26, 0xb2",
        "vpternlogq zmm6, zmm25, zmm27, 0x96",
        "vpternlogq zmm25, zmm6, zmm27, 0xb2",
        "movzx eax, word ptr [rsi + 44]",
        "movzx r10d, word ptr [rsi + 46]",
        "vmovdqa64 zmm26, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm28, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm27, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm29, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm26, zmm28, 0x96",
        "vpternlogq zmm26, zmm0, zmm28, 0xb2",
        "vpternlogq zmm6, zmm27, zmm29, 0x96",
        "vpternlogq zmm27, zmm6, zmm29, 0xb2",
        "vpternlogq zmm1, zmm24, zmm26, 0x96",
        "vpternlogq zmm24, zmm1, zmm26, 0xb2",
        "vpternlogq zmm7, zmm25, zmm27, 0x96",
        "vpternlogq zmm25, zmm7, zmm27, 0xb2",
        "vpternlogq zmm2, zmm22, zmm24, 0x96",
        "vpternlogq zmm22, zmm2, zmm24, 0xb2",
        "vpternlogq zmm8, zmm23, zmm25, 0x96",
        "vpternlogq zmm23, zmm8, zmm25, 0xb2",
        "movzx eax, word ptr [rsi + 48]",
        "movzx r10d, word ptr [rsi + 50]",
        "vmovdqa64 zmm24, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm26, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm25, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm27, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm24, zmm26, 0x96",
        "vpternlogq zmm24, zmm0, zmm26, 0xb2",
        "vpternlogq zmm6, zmm25, zmm27, 0x96",
        "vpternlogq zmm25, zmm6, zmm27, 0xb2",
        "movzx eax, word ptr [rsi + 52]",
        "movzx r10d, word ptr [rsi + 54]",
        "vmovdqa64 zmm26, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm28, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm27, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm29, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm26, zmm28, 0x96",
        "vpternlogq zmm26, zmm0, zmm28, 0xb2",
        "vpternlogq zmm6, zmm27, zmm29, 0x96",
        "vpternlogq zmm27, zmm6, zmm29, 0xb2",
        "vpternlogq zmm1, zmm24, zmm26, 0x96",
        "vpternlogq zmm24, zmm1, zmm26, 0xb2",
        "vpternlogq zmm7, zmm25, zmm27, 0x96",
        "vpternlogq zmm25, zmm7, zmm27, 0xb2",
        "movzx eax, word ptr [rsi + 56]",
        "movzx r10d, word ptr [rsi + 58]",
        "vmovdqa64 zmm26, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm28, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm27, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm29, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm26, zmm28, 0x96",
        "vpternlogq zmm26, zmm0, zmm28, 0xb2",
        "vpternlogq zmm6, zmm27, zmm29, 0x96",
        "vpternlogq zmm27, zmm6, zmm29, 0xb2",
        "movzx eax, word ptr [rsi + 60]",
        "movzx r10d, word ptr [rsi + 62]",
        "vmovdqa64 zmm28, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm30, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm29, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm31, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm28, zmm30, 0x96",
        "vpternlogq zmm28, zmm0, zmm30, 0xb2",
        "vpternlogq zmm6, zmm29, zmm31, 0x96",
        "vpternlogq zmm29, zmm6, zmm31, 0xb2",
        "vpternlogq zmm1, zmm26, zmm28, 0x96",
        "vpternlogq zmm26, zmm1, zmm28, 0xb2",
        "vpternlogq zmm7, zmm27, zmm29, 0x96",
        "vpternlogq zmm27, zmm7, zmm29, 0xb2",
        "vpternlogq zmm2, zmm24, zmm26, 0x96",
        "vpternlogq zmm24, zmm2, zmm26, 0xb2",
        "vpternlogq zmm8, zmm25, zmm27, 0x96",
        "vpternlogq zmm25, zmm8, zmm27, 0xb2",
        "vpternlogq zmm3, zmm22, zmm24, 0x96",
        "vpternlogq zmm22, zmm3, zmm24, 0xb2",
        "vpternlogq zmm9, zmm23, zmm25, 0x96",
        "vpternlogq zmm23, zmm9, zmm25, 0xb2",
        "vpternlogq zmm4, zmm20, zmm22, 0x96",
        "vpternlogq zmm20, zmm4, zmm22, 0xb2",
        "vpternlogq zmm10, zmm21, zmm23, 0x96",
        "vpternlogq zmm21, zmm10, zmm23, 0xb2",
        "movzx eax, word ptr [rsi + 64]",
        "movzx r10d, word ptr [rsi + 66]",
        "vmovdqa64 zmm22, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm24, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm23, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm25, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm22, zmm24, 0x96",
        "vpternlogq zmm22, zmm0, zmm24, 0xb2",
        "vpternlogq zmm6, zmm23, zmm25, 0x96",
        "vpternlogq zmm23, zmm6, zmm25, 0xb2",
        "movzx eax, word ptr [rsi + 68]",
        "movzx r10d, word ptr [rsi + 70]",
        "vmovdqa64 zmm24, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm26, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm25, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm27, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm24, zmm26, 0x96",
        "vpternlogq zmm24, zmm0, zmm26, 0xb2",
        "vpternlogq zmm6, zmm25, zmm27, 0x96",
        "vpternlogq zmm25, zmm6, zmm27, 0xb2",
        "vpternlogq zmm1, zmm22, zmm24, 0x96",
        "vpternlogq zmm22, zmm1, zmm24, 0xb2",
        "vpternlogq zmm7, zmm23, zmm25, 0x96",
        "vpternlogq zmm23, zmm7, zmm25, 0xb2",
        "movzx eax, word ptr [rsi + 72]",
        "movzx r10d, word ptr [rsi + 74]",
        "vmovdqa64 zmm24, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm26, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm25, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm27, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm24, zmm26, 0x96",
        "vpternlogq zmm24, zmm0, zmm26, 0xb2",
        "vpternlogq zmm6, zmm25, zmm27, 0x96",
        "vpternlogq zmm25, zmm6, zmm27, 0xb2",
        "movzx eax, word ptr [rsi + 76]",
        "movzx r10d, word ptr [rsi + 78]",
        "vmovdqa64 zmm26, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm28, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm27, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm29, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm26, zmm28, 0x96",
        "vpternlogq zmm26, zmm0, zmm28, 0xb2",
        "vpternlogq zmm6, zmm27, zmm29, 0x96",
        "vpternlogq zmm27, zmm6, zmm29, 0xb2",
        "vpternlogq zmm1, zmm24, zmm26, 0x96",
        "vpternlogq zmm24, zmm1, zmm26, 0xb2",
        "vpternlogq zmm7, zmm25, zmm27, 0x96",
        "vpternlogq zmm25, zmm7, zmm27, 0xb2",
        "vpternlogq zmm2, zmm22, zmm24, 0x96",
        "vpternlogq zmm22, zmm2, zmm24, 0xb2",
        "vpternlogq zmm8, zmm23, zmm25, 0x96",
        "vpternlogq zmm23, zmm8, zmm25, 0xb2",
        "movzx eax, word ptr [rsi + 80]",
        "movzx r10d, word ptr [rsi + 82]",
        "vmovdqa64 zmm24, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm26, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm25, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm27, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm24, zmm26, 0x96",
        "vpternlogq zmm24, zmm0, zmm26, 0xb2",
        "vpternlogq zmm6, zmm25, zmm27, 0x96",
        "vpternlogq zmm25, zmm6, zmm27, 0xb2",
        "movzx eax, word ptr [rsi + 84]",
        "movzx r10d, word ptr [rsi + 86]",
        "vmovdqa64 zmm26, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm28, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm27, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm29, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm26, zmm28, 0x96",
        "vpternlogq zmm26, zmm0, zmm28, 0xb2",
        "vpternlogq zmm6, zmm27, zmm29, 0x96",
        "vpternlogq zmm27, zmm6, zmm29, 0xb2",
        "vpternlogq zmm1, zmm24, zmm26, 0x96",
        "vpternlogq zmm24, zmm1, zmm26, 0xb2",
        "vpternlogq zmm7, zmm25, zmm27, 0x96",
        "vpternlogq zmm25, zmm7, zmm27, 0xb2",
        "movzx eax, word ptr [rsi + 88]",
        "movzx r10d, word ptr [rsi + 90]",
        "vmovdqa64 zmm26, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm28, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm27, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm29, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm26, zmm28, 0x96",
        "vpternlogq zmm26, zmm0, zmm28, 0xb2",
        "vpternlogq zmm6, zmm27, zmm29, 0x96",
        "vpternlogq zmm27, zmm6, zmm29, 0xb2",
        "movzx eax, word ptr [rsi + 92]",
        "movzx r10d, word ptr [rsi + 94]",
        "vmovdqa64 zmm28, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm30, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm29, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm31, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm28, zmm30, 0x96",
        "vpternlogq zmm28, zmm0, zmm30, 0xb2",
        "vpternlogq zmm6, zmm29, zmm31, 0x96",
        "vpternlogq zmm29, zmm6, zmm31, 0xb2",
        "vpternlogq zmm1, zmm26, zmm28, 0x96",
        "vpternlogq zmm26, zmm1, zmm28, 0xb2",
        "vpternlogq zmm7, zmm27, zmm29, 0x96",
        "vpternlogq zmm27, zmm7, zmm29, 0xb2",
        "vpternlogq zmm2, zmm24, zmm26, 0x96",
        "vpternlogq zmm24, zmm2, zmm26, 0xb2",
        "vpternlogq zmm8, zmm25, zmm27, 0x96",
        "vpternlogq zmm25, zmm8, zmm27, 0xb2",
        "vpternlogq zmm3, zmm22, zmm24, 0x96",
        "vpternlogq zmm22, zmm3, zmm24, 0xb2",
        "vpternlogq zmm9, zmm23, zmm25, 0x96",
        "vpternlogq zmm23, zmm9, zmm25, 0xb2",
        "movzx eax, word ptr [rsi + 96]",
        "movzx r10d, word ptr [rsi + 98]",
        "vmovdqa64 zmm24, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm26, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm25, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm27, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm24, zmm26, 0x96",
        "vpternlogq zmm24, zmm0, zmm26, 0xb2",
        "vpternlogq zmm6, zmm25, zmm27, 0x96",
        "vpternlogq zmm25, zmm6, zmm27, 0xb2",
        "movzx eax, word ptr [rsi + 100]",
        "movzx r10d, word ptr [rsi + 102]",
        "vmovdqa64 zmm26, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm28, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm27, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm29, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm26, zmm28, 0x96",
        "vpternlogq zmm26, zmm0, zmm28, 0xb2",
        "vpternlogq zmm6, zmm27, zmm29, 0x96",
        "vpternlogq zmm27, zmm6, zmm29, 0xb2",
        "vpternlogq zmm1, zmm24, zmm26, 0x96",
        "vpternlogq zmm24, zmm1, zmm26, 0xb2",
        "vpternlogq zmm7, zmm25, zmm27, 0x96",
        "vpternlogq zmm25, zmm7, zmm27, 0xb2",
        "movzx eax, word ptr [rsi + 104]",
        "movzx r10d, word ptr [rsi + 106]",
        "vmovdqa64 zmm26, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm28, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm27, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm29, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm26, zmm28, 0x96",
        "vpternlogq zmm26, zmm0, zmm28, 0xb2",
        "vpternlogq zmm6, zmm27, zmm29, 0x96",
        "vpternlogq zmm27, zmm6, zmm29, 0xb2",
        "movzx eax, word ptr [rsi + 108]",
        "movzx r10d, word ptr [rsi + 110]",
        "vmovdqa64 zmm28, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm30, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm29, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm31, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm28, zmm30, 0x96",
        "vpternlogq zmm28, zmm0, zmm30, 0xb2",
        "vpternlogq zmm6, zmm29, zmm31, 0x96",
        "vpternlogq zmm29, zmm6, zmm31, 0xb2",
        "vpternlogq zmm1, zmm26, zmm28, 0x96",
        "vpternlogq zmm26, zmm1, zmm28, 0xb2",
        "vpternlogq zmm7, zmm27, zmm29, 0x96",
        "vpternlogq zmm27, zmm7, zmm29, 0xb2",
        "vpternlogq zmm2, zmm24, zmm26, 0x96",
        "vpternlogq zmm24, zmm2, zmm26, 0xb2",
        "vpternlogq zmm8, zmm25, zmm27, 0x96",
        "vpternlogq zmm25, zmm8, zmm27, 0xb2",
        "movzx eax, word ptr [rsi + 112]",
        "movzx r10d, word ptr [rsi + 114]",
        "vmovdqa64 zmm26, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm28, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm27, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm29, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm26, zmm28, 0x96",
        "vpternlogq zmm26, zmm0, zmm28, 0xb2",
        "vpternlogq zmm6, zmm27, zmm29, 0x96",
        "vpternlogq zmm27, zmm6, zmm29, 0xb2",
        "movzx eax, word ptr [rsi + 116]",
        "movzx r10d, word ptr [rsi + 118]",
        "vmovdqa64 zmm28, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm30, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm29, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm31, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm28, zmm30, 0x96",
        "vpternlogq zmm28, zmm0, zmm30, 0xb2",
        "vpternlogq zmm6, zmm29, zmm31, 0x96",
        "vpternlogq zmm29, zmm6, zmm31, 0xb2",
        "vpternlogq zmm1, zmm26, zmm28, 0x96",
        "vpternlogq zmm26, zmm1, zmm28, 0xb2",
        "vpternlogq zmm7, zmm27, zmm29, 0x96",
        "vpternlogq zmm27, zmm7, zmm29, 0xb2",
        "movzx eax, word ptr [rsi + 120]",
        "movzx r10d, word ptr [rsi + 122]",
        "vmovdqa64 zmm28, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm30, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm29, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm31, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm28, zmm30, 0x96",
        "vpternlogq zmm28, zmm0, zmm30, 0xb2",
        "vpternlogq zmm6, zmm29, zmm31, 0x96",
        "vpternlogq zmm29, zmm6, zmm31, 0xb2",
        "movzx eax, word ptr [rsi + 124]",
        "movzx r10d, word ptr [rsi + 126]",
        "vmovdqa64 zmm30, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm31, zmmword ptr [rdi + r10*8 + 0]",
        "vpternlogq zmm0, zmm30, zmm31, 0x96",
        "vpternlogq zmm30, zmm0, zmm31, 0xb2",
        "vpternlogq zmm1, zmm28, zmm30, 0x96",
        "vpternlogq zmm28, zmm1, zmm30, 0xb2",
        "vmovdqa64 zmm30, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm31, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm6, zmm30, zmm31, 0x96",
        "vpternlogq zmm30, zmm6, zmm31, 0xb2",
        "vpternlogq zmm7, zmm29, zmm30, 0x96",
        "vpternlogq zmm29, zmm7, zmm30, 0xb2",
        "vpternlogq zmm2, zmm26, zmm28, 0x96",
        "vpternlogq zmm26, zmm2, zmm28, 0xb2",
        "vpternlogq zmm8, zmm27, zmm29, 0x96",
        "vpternlogq zmm27, zmm8, zmm29, 0xb2",
        "vpternlogq zmm3, zmm24, zmm26, 0x96",
        "vpternlogq zmm24, zmm3, zmm26, 0xb2",
        "vpternlogq zmm9, zmm25, zmm27, 0x96",
        "vpternlogq zmm25, zmm9, zmm27, 0xb2",
        "vpternlogq zmm4, zmm22, zmm24, 0x96",
        "vpternlogq zmm22, zmm4, zmm24, 0xb2",
        "vpternlogq zmm10, zmm23, zmm25, 0x96",
        "vpternlogq zmm23, zmm10, zmm25, 0xb2",
        "vpternlogq zmm5, zmm20, zmm22, 0x96",
        "vpternlogq zmm20, zmm5, zmm22, 0xb2",
        "vpternlogq zmm11, zmm21, zmm23, 0x96",
        "vpternlogq zmm21, zmm11, zmm23, 0xb2",
        "vpandq zmm22, zmm12, zmm20",
        "vpxorq zmm12, zmm12, zmm20",
        "vpandq zmm20, zmm13, zmm22",
        "vpxorq zmm13, zmm13, zmm22",
        "vpandq zmm22, zmm14, zmm20",
        "vpxorq zmm14, zmm14, zmm20",
        "vpxorq zmm15, zmm15, zmm22",
        "vpandq zmm22, zmm16, zmm21",
        "vpxorq zmm16, zmm16, zmm21",
        "vpandq zmm21, zmm17, zmm22",
        "vpxorq zmm17, zmm17, zmm22",
        "vpandq zmm22, zmm18, zmm21",
        "vpxorq zmm18, zmm18, zmm21",
        "vpxorq zmm19, zmm19, zmm22",
        "inc r12",
        "cmp r12, 15",
        "jne 23f",
        "lea r13, [rip + 23f]",
        "jmp 90f",
        "23:",
        "add rsi, 128",
        "sub rdx, 64",
        "jmp 20b",
        "21:",
        "test rdx, rdx",
        "jz 22f",
        "test rdx, 32",
        "jz 24f",
        "movzx eax, word ptr [rsi + 0]",
        "movzx r10d, word ptr [rsi + 2]",
        "vmovdqa64 zmm20, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm22, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm21, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm23, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm20, zmm22, 0x96",
        "vpternlogq zmm20, zmm0, zmm22, 0xb2",
        "vpternlogq zmm6, zmm21, zmm23, 0x96",
        "vpternlogq zmm21, zmm6, zmm23, 0xb2",
        "movzx eax, word ptr [rsi + 4]",
        "movzx r10d, word ptr [rsi + 6]",
        "vmovdqa64 zmm22, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm24, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm23, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm25, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm22, zmm24, 0x96",
        "vpternlogq zmm22, zmm0, zmm24, 0xb2",
        "vpternlogq zmm6, zmm23, zmm25, 0x96",
        "vpternlogq zmm23, zmm6, zmm25, 0xb2",
        "vpternlogq zmm1, zmm20, zmm22, 0x96",
        "vpternlogq zmm20, zmm1, zmm22, 0xb2",
        "vpternlogq zmm7, zmm21, zmm23, 0x96",
        "vpternlogq zmm21, zmm7, zmm23, 0xb2",
        "movzx eax, word ptr [rsi + 8]",
        "movzx r10d, word ptr [rsi + 10]",
        "vmovdqa64 zmm22, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm24, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm23, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm25, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm22, zmm24, 0x96",
        "vpternlogq zmm22, zmm0, zmm24, 0xb2",
        "vpternlogq zmm6, zmm23, zmm25, 0x96",
        "vpternlogq zmm23, zmm6, zmm25, 0xb2",
        "movzx eax, word ptr [rsi + 12]",
        "movzx r10d, word ptr [rsi + 14]",
        "vmovdqa64 zmm24, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm26, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm25, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm27, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm24, zmm26, 0x96",
        "vpternlogq zmm24, zmm0, zmm26, 0xb2",
        "vpternlogq zmm6, zmm25, zmm27, 0x96",
        "vpternlogq zmm25, zmm6, zmm27, 0xb2",
        "vpternlogq zmm1, zmm22, zmm24, 0x96",
        "vpternlogq zmm22, zmm1, zmm24, 0xb2",
        "vpternlogq zmm7, zmm23, zmm25, 0x96",
        "vpternlogq zmm23, zmm7, zmm25, 0xb2",
        "vpternlogq zmm2, zmm20, zmm22, 0x96",
        "vpternlogq zmm20, zmm2, zmm22, 0xb2",
        "vpternlogq zmm8, zmm21, zmm23, 0x96",
        "vpternlogq zmm21, zmm8, zmm23, 0xb2",
        "movzx eax, word ptr [rsi + 16]",
        "movzx r10d, word ptr [rsi + 18]",
        "vmovdqa64 zmm22, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm24, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm23, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm25, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm22, zmm24, 0x96",
        "vpternlogq zmm22, zmm0, zmm24, 0xb2",
        "vpternlogq zmm6, zmm23, zmm25, 0x96",
        "vpternlogq zmm23, zmm6, zmm25, 0xb2",
        "movzx eax, word ptr [rsi + 20]",
        "movzx r10d, word ptr [rsi + 22]",
        "vmovdqa64 zmm24, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm26, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm25, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm27, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm24, zmm26, 0x96",
        "vpternlogq zmm24, zmm0, zmm26, 0xb2",
        "vpternlogq zmm6, zmm25, zmm27, 0x96",
        "vpternlogq zmm25, zmm6, zmm27, 0xb2",
        "vpternlogq zmm1, zmm22, zmm24, 0x96",
        "vpternlogq zmm22, zmm1, zmm24, 0xb2",
        "vpternlogq zmm7, zmm23, zmm25, 0x96",
        "vpternlogq zmm23, zmm7, zmm25, 0xb2",
        "movzx eax, word ptr [rsi + 24]",
        "movzx r10d, word ptr [rsi + 26]",
        "vmovdqa64 zmm24, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm26, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm25, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm27, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm24, zmm26, 0x96",
        "vpternlogq zmm24, zmm0, zmm26, 0xb2",
        "vpternlogq zmm6, zmm25, zmm27, 0x96",
        "vpternlogq zmm25, zmm6, zmm27, 0xb2",
        "movzx eax, word ptr [rsi + 28]",
        "movzx r10d, word ptr [rsi + 30]",
        "vmovdqa64 zmm26, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm28, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm27, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm29, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm26, zmm28, 0x96",
        "vpternlogq zmm26, zmm0, zmm28, 0xb2",
        "vpternlogq zmm6, zmm27, zmm29, 0x96",
        "vpternlogq zmm27, zmm6, zmm29, 0xb2",
        "vpternlogq zmm1, zmm24, zmm26, 0x96",
        "vpternlogq zmm24, zmm1, zmm26, 0xb2",
        "vpternlogq zmm7, zmm25, zmm27, 0x96",
        "vpternlogq zmm25, zmm7, zmm27, 0xb2",
        "vpternlogq zmm2, zmm22, zmm24, 0x96",
        "vpternlogq zmm22, zmm2, zmm24, 0xb2",
        "vpternlogq zmm8, zmm23, zmm25, 0x96",
        "vpternlogq zmm23, zmm8, zmm25, 0xb2",
        "vpternlogq zmm3, zmm20, zmm22, 0x96",
        "vpternlogq zmm20, zmm3, zmm22, 0xb2",
        "vpternlogq zmm9, zmm21, zmm23, 0x96",
        "vpternlogq zmm21, zmm9, zmm23, 0xb2",
        "movzx eax, word ptr [rsi + 32]",
        "movzx r10d, word ptr [rsi + 34]",
        "vmovdqa64 zmm22, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm24, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm23, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm25, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm22, zmm24, 0x96",
        "vpternlogq zmm22, zmm0, zmm24, 0xb2",
        "vpternlogq zmm6, zmm23, zmm25, 0x96",
        "vpternlogq zmm23, zmm6, zmm25, 0xb2",
        "movzx eax, word ptr [rsi + 36]",
        "movzx r10d, word ptr [rsi + 38]",
        "vmovdqa64 zmm24, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm26, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm25, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm27, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm24, zmm26, 0x96",
        "vpternlogq zmm24, zmm0, zmm26, 0xb2",
        "vpternlogq zmm6, zmm25, zmm27, 0x96",
        "vpternlogq zmm25, zmm6, zmm27, 0xb2",
        "vpternlogq zmm1, zmm22, zmm24, 0x96",
        "vpternlogq zmm22, zmm1, zmm24, 0xb2",
        "vpternlogq zmm7, zmm23, zmm25, 0x96",
        "vpternlogq zmm23, zmm7, zmm25, 0xb2",
        "movzx eax, word ptr [rsi + 40]",
        "movzx r10d, word ptr [rsi + 42]",
        "vmovdqa64 zmm24, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm26, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm25, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm27, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm24, zmm26, 0x96",
        "vpternlogq zmm24, zmm0, zmm26, 0xb2",
        "vpternlogq zmm6, zmm25, zmm27, 0x96",
        "vpternlogq zmm25, zmm6, zmm27, 0xb2",
        "movzx eax, word ptr [rsi + 44]",
        "movzx r10d, word ptr [rsi + 46]",
        "vmovdqa64 zmm26, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm28, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm27, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm29, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm26, zmm28, 0x96",
        "vpternlogq zmm26, zmm0, zmm28, 0xb2",
        "vpternlogq zmm6, zmm27, zmm29, 0x96",
        "vpternlogq zmm27, zmm6, zmm29, 0xb2",
        "vpternlogq zmm1, zmm24, zmm26, 0x96",
        "vpternlogq zmm24, zmm1, zmm26, 0xb2",
        "vpternlogq zmm7, zmm25, zmm27, 0x96",
        "vpternlogq zmm25, zmm7, zmm27, 0xb2",
        "vpternlogq zmm2, zmm22, zmm24, 0x96",
        "vpternlogq zmm22, zmm2, zmm24, 0xb2",
        "vpternlogq zmm8, zmm23, zmm25, 0x96",
        "vpternlogq zmm23, zmm8, zmm25, 0xb2",
        "movzx eax, word ptr [rsi + 48]",
        "movzx r10d, word ptr [rsi + 50]",
        "vmovdqa64 zmm24, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm26, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm25, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm27, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm24, zmm26, 0x96",
        "vpternlogq zmm24, zmm0, zmm26, 0xb2",
        "vpternlogq zmm6, zmm25, zmm27, 0x96",
        "vpternlogq zmm25, zmm6, zmm27, 0xb2",
        "movzx eax, word ptr [rsi + 52]",
        "movzx r10d, word ptr [rsi + 54]",
        "vmovdqa64 zmm26, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm28, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm27, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm29, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm26, zmm28, 0x96",
        "vpternlogq zmm26, zmm0, zmm28, 0xb2",
        "vpternlogq zmm6, zmm27, zmm29, 0x96",
        "vpternlogq zmm27, zmm6, zmm29, 0xb2",
        "vpternlogq zmm1, zmm24, zmm26, 0x96",
        "vpternlogq zmm24, zmm1, zmm26, 0xb2",
        "vpternlogq zmm7, zmm25, zmm27, 0x96",
        "vpternlogq zmm25, zmm7, zmm27, 0xb2",
        "movzx eax, word ptr [rsi + 56]",
        "movzx r10d, word ptr [rsi + 58]",
        "vmovdqa64 zmm26, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm28, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm27, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm29, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm26, zmm28, 0x96",
        "vpternlogq zmm26, zmm0, zmm28, 0xb2",
        "vpternlogq zmm6, zmm27, zmm29, 0x96",
        "vpternlogq zmm27, zmm6, zmm29, 0xb2",
        "movzx eax, word ptr [rsi + 60]",
        "movzx r10d, word ptr [rsi + 62]",
        "vmovdqa64 zmm28, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm30, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm29, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm31, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm28, zmm30, 0x96",
        "vpternlogq zmm28, zmm0, zmm30, 0xb2",
        "vpternlogq zmm6, zmm29, zmm31, 0x96",
        "vpternlogq zmm29, zmm6, zmm31, 0xb2",
        "vpternlogq zmm1, zmm26, zmm28, 0x96",
        "vpternlogq zmm26, zmm1, zmm28, 0xb2",
        "vpternlogq zmm7, zmm27, zmm29, 0x96",
        "vpternlogq zmm27, zmm7, zmm29, 0xb2",
        "vpternlogq zmm2, zmm24, zmm26, 0x96",
        "vpternlogq zmm24, zmm2, zmm26, 0xb2",
        "vpternlogq zmm8, zmm25, zmm27, 0x96",
        "vpternlogq zmm25, zmm8, zmm27, 0xb2",
        "vpternlogq zmm3, zmm22, zmm24, 0x96",
        "vpternlogq zmm22, zmm3, zmm24, 0xb2",
        "vpternlogq zmm9, zmm23, zmm25, 0x96",
        "vpternlogq zmm23, zmm9, zmm25, 0xb2",
        "vpternlogq zmm4, zmm20, zmm22, 0x96",
        "vpternlogq zmm20, zmm4, zmm22, 0xb2",
        "vpternlogq zmm10, zmm21, zmm23, 0x96",
        "vpternlogq zmm21, zmm10, zmm23, 0xb2",
        "vpandq zmm22, zmm5, zmm20",
        "vpxorq zmm5, zmm5, zmm20",
        "vmovdqa64 zmm20, zmm22",
        "vpandq zmm22, zmm11, zmm21",
        "vpxorq zmm11, zmm11, zmm21",
        "vmovdqa64 zmm21, zmm22",
        "vpandq zmm22, zmm12, zmm20",
        "vpxorq zmm12, zmm12, zmm20",
        "vpandq zmm20, zmm13, zmm22",
        "vpxorq zmm13, zmm13, zmm22",
        "vpandq zmm22, zmm14, zmm20",
        "vpxorq zmm14, zmm14, zmm20",
        "vpxorq zmm15, zmm15, zmm22",
        "vpandq zmm22, zmm16, zmm21",
        "vpxorq zmm16, zmm16, zmm21",
        "vpandq zmm21, zmm17, zmm22",
        "vpxorq zmm17, zmm17, zmm22",
        "vpandq zmm22, zmm18, zmm21",
        "vpxorq zmm18, zmm18, zmm21",
        "vpxorq zmm19, zmm19, zmm22",
        "add rsi, 64",
        "24:",
        "test rdx, 16",
        "jz 25f",
        "movzx eax, word ptr [rsi + 0]",
        "movzx r10d, word ptr [rsi + 2]",
        "vmovdqa64 zmm20, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm22, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm21, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm23, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm20, zmm22, 0x96",
        "vpternlogq zmm20, zmm0, zmm22, 0xb2",
        "vpternlogq zmm6, zmm21, zmm23, 0x96",
        "vpternlogq zmm21, zmm6, zmm23, 0xb2",
        "movzx eax, word ptr [rsi + 4]",
        "movzx r10d, word ptr [rsi + 6]",
        "vmovdqa64 zmm22, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm24, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm23, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm25, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm22, zmm24, 0x96",
        "vpternlogq zmm22, zmm0, zmm24, 0xb2",
        "vpternlogq zmm6, zmm23, zmm25, 0x96",
        "vpternlogq zmm23, zmm6, zmm25, 0xb2",
        "vpternlogq zmm1, zmm20, zmm22, 0x96",
        "vpternlogq zmm20, zmm1, zmm22, 0xb2",
        "vpternlogq zmm7, zmm21, zmm23, 0x96",
        "vpternlogq zmm21, zmm7, zmm23, 0xb2",
        "movzx eax, word ptr [rsi + 8]",
        "movzx r10d, word ptr [rsi + 10]",
        "vmovdqa64 zmm22, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm24, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm23, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm25, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm22, zmm24, 0x96",
        "vpternlogq zmm22, zmm0, zmm24, 0xb2",
        "vpternlogq zmm6, zmm23, zmm25, 0x96",
        "vpternlogq zmm23, zmm6, zmm25, 0xb2",
        "movzx eax, word ptr [rsi + 12]",
        "movzx r10d, word ptr [rsi + 14]",
        "vmovdqa64 zmm24, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm26, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm25, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm27, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm24, zmm26, 0x96",
        "vpternlogq zmm24, zmm0, zmm26, 0xb2",
        "vpternlogq zmm6, zmm25, zmm27, 0x96",
        "vpternlogq zmm25, zmm6, zmm27, 0xb2",
        "vpternlogq zmm1, zmm22, zmm24, 0x96",
        "vpternlogq zmm22, zmm1, zmm24, 0xb2",
        "vpternlogq zmm7, zmm23, zmm25, 0x96",
        "vpternlogq zmm23, zmm7, zmm25, 0xb2",
        "vpternlogq zmm2, zmm20, zmm22, 0x96",
        "vpternlogq zmm20, zmm2, zmm22, 0xb2",
        "vpternlogq zmm8, zmm21, zmm23, 0x96",
        "vpternlogq zmm21, zmm8, zmm23, 0xb2",
        "movzx eax, word ptr [rsi + 16]",
        "movzx r10d, word ptr [rsi + 18]",
        "vmovdqa64 zmm22, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm24, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm23, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm25, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm22, zmm24, 0x96",
        "vpternlogq zmm22, zmm0, zmm24, 0xb2",
        "vpternlogq zmm6, zmm23, zmm25, 0x96",
        "vpternlogq zmm23, zmm6, zmm25, 0xb2",
        "movzx eax, word ptr [rsi + 20]",
        "movzx r10d, word ptr [rsi + 22]",
        "vmovdqa64 zmm24, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm26, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm25, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm27, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm24, zmm26, 0x96",
        "vpternlogq zmm24, zmm0, zmm26, 0xb2",
        "vpternlogq zmm6, zmm25, zmm27, 0x96",
        "vpternlogq zmm25, zmm6, zmm27, 0xb2",
        "vpternlogq zmm1, zmm22, zmm24, 0x96",
        "vpternlogq zmm22, zmm1, zmm24, 0xb2",
        "vpternlogq zmm7, zmm23, zmm25, 0x96",
        "vpternlogq zmm23, zmm7, zmm25, 0xb2",
        "movzx eax, word ptr [rsi + 24]",
        "movzx r10d, word ptr [rsi + 26]",
        "vmovdqa64 zmm24, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm26, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm25, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm27, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm24, zmm26, 0x96",
        "vpternlogq zmm24, zmm0, zmm26, 0xb2",
        "vpternlogq zmm6, zmm25, zmm27, 0x96",
        "vpternlogq zmm25, zmm6, zmm27, 0xb2",
        "movzx eax, word ptr [rsi + 28]",
        "movzx r10d, word ptr [rsi + 30]",
        "vmovdqa64 zmm26, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm28, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm27, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm29, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm26, zmm28, 0x96",
        "vpternlogq zmm26, zmm0, zmm28, 0xb2",
        "vpternlogq zmm6, zmm27, zmm29, 0x96",
        "vpternlogq zmm27, zmm6, zmm29, 0xb2",
        "vpternlogq zmm1, zmm24, zmm26, 0x96",
        "vpternlogq zmm24, zmm1, zmm26, 0xb2",
        "vpternlogq zmm7, zmm25, zmm27, 0x96",
        "vpternlogq zmm25, zmm7, zmm27, 0xb2",
        "vpternlogq zmm2, zmm22, zmm24, 0x96",
        "vpternlogq zmm22, zmm2, zmm24, 0xb2",
        "vpternlogq zmm8, zmm23, zmm25, 0x96",
        "vpternlogq zmm23, zmm8, zmm25, 0xb2",
        "vpternlogq zmm3, zmm20, zmm22, 0x96",
        "vpternlogq zmm20, zmm3, zmm22, 0xb2",
        "vpternlogq zmm9, zmm21, zmm23, 0x96",
        "vpternlogq zmm21, zmm9, zmm23, 0xb2",
        "vpandq zmm22, zmm4, zmm20",
        "vpxorq zmm4, zmm4, zmm20",
        "vpandq zmm20, zmm5, zmm22",
        "vpxorq zmm5, zmm5, zmm22",
        "vpandq zmm22, zmm10, zmm21",
        "vpxorq zmm10, zmm10, zmm21",
        "vpandq zmm21, zmm11, zmm22",
        "vpxorq zmm11, zmm11, zmm22",
        "vpandq zmm22, zmm12, zmm20",
        "vpxorq zmm12, zmm12, zmm20",
        "vpandq zmm20, zmm13, zmm22",
        "vpxorq zmm13, zmm13, zmm22",
        "vpandq zmm22, zmm14, zmm20",
        "vpxorq zmm14, zmm14, zmm20",
        "vpxorq zmm15, zmm15, zmm22",
        "vpandq zmm22, zmm16, zmm21",
        "vpxorq zmm16, zmm16, zmm21",
        "vpandq zmm21, zmm17, zmm22",
        "vpxorq zmm17, zmm17, zmm22",
        "vpandq zmm22, zmm18, zmm21",
        "vpxorq zmm18, zmm18, zmm21",
        "vpxorq zmm19, zmm19, zmm22",
        "add rsi, 32",
        "25:",
        "test rdx, 8",
        "jz 26f",
        "movzx eax, word ptr [rsi + 0]",
        "movzx r10d, word ptr [rsi + 2]",
        "vmovdqa64 zmm20, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm22, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm21, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm23, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm20, zmm22, 0x96",
        "vpternlogq zmm20, zmm0, zmm22, 0xb2",
        "vpternlogq zmm6, zmm21, zmm23, 0x96",
        "vpternlogq zmm21, zmm6, zmm23, 0xb2",
        "movzx eax, word ptr [rsi + 4]",
        "movzx r10d, word ptr [rsi + 6]",
        "vmovdqa64 zmm22, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm24, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm23, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm25, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm22, zmm24, 0x96",
        "vpternlogq zmm22, zmm0, zmm24, 0xb2",
        "vpternlogq zmm6, zmm23, zmm25, 0x96",
        "vpternlogq zmm23, zmm6, zmm25, 0xb2",
        "vpternlogq zmm1, zmm20, zmm22, 0x96",
        "vpternlogq zmm20, zmm1, zmm22, 0xb2",
        "vpternlogq zmm7, zmm21, zmm23, 0x96",
        "vpternlogq zmm21, zmm7, zmm23, 0xb2",
        "movzx eax, word ptr [rsi + 8]",
        "movzx r10d, word ptr [rsi + 10]",
        "vmovdqa64 zmm22, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm24, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm23, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm25, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm22, zmm24, 0x96",
        "vpternlogq zmm22, zmm0, zmm24, 0xb2",
        "vpternlogq zmm6, zmm23, zmm25, 0x96",
        "vpternlogq zmm23, zmm6, zmm25, 0xb2",
        "movzx eax, word ptr [rsi + 12]",
        "movzx r10d, word ptr [rsi + 14]",
        "vmovdqa64 zmm24, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm26, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm25, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm27, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm24, zmm26, 0x96",
        "vpternlogq zmm24, zmm0, zmm26, 0xb2",
        "vpternlogq zmm6, zmm25, zmm27, 0x96",
        "vpternlogq zmm25, zmm6, zmm27, 0xb2",
        "vpternlogq zmm1, zmm22, zmm24, 0x96",
        "vpternlogq zmm22, zmm1, zmm24, 0xb2",
        "vpternlogq zmm7, zmm23, zmm25, 0x96",
        "vpternlogq zmm23, zmm7, zmm25, 0xb2",
        "vpternlogq zmm2, zmm20, zmm22, 0x96",
        "vpternlogq zmm20, zmm2, zmm22, 0xb2",
        "vpternlogq zmm8, zmm21, zmm23, 0x96",
        "vpternlogq zmm21, zmm8, zmm23, 0xb2",
        "vpandq zmm22, zmm3, zmm20",
        "vpxorq zmm3, zmm3, zmm20",
        "vpandq zmm20, zmm4, zmm22",
        "vpxorq zmm4, zmm4, zmm22",
        "vpandq zmm22, zmm5, zmm20",
        "vpxorq zmm5, zmm5, zmm20",
        "vmovdqa64 zmm20, zmm22",
        "vpandq zmm22, zmm9, zmm21",
        "vpxorq zmm9, zmm9, zmm21",
        "vpandq zmm21, zmm10, zmm22",
        "vpxorq zmm10, zmm10, zmm22",
        "vpandq zmm22, zmm11, zmm21",
        "vpxorq zmm11, zmm11, zmm21",
        "vmovdqa64 zmm21, zmm22",
        "vpandq zmm22, zmm12, zmm20",
        "vpxorq zmm12, zmm12, zmm20",
        "vpandq zmm20, zmm13, zmm22",
        "vpxorq zmm13, zmm13, zmm22",
        "vpandq zmm22, zmm14, zmm20",
        "vpxorq zmm14, zmm14, zmm20",
        "vpxorq zmm15, zmm15, zmm22",
        "vpandq zmm22, zmm16, zmm21",
        "vpxorq zmm16, zmm16, zmm21",
        "vpandq zmm21, zmm17, zmm22",
        "vpxorq zmm17, zmm17, zmm22",
        "vpandq zmm22, zmm18, zmm21",
        "vpxorq zmm18, zmm18, zmm21",
        "vpxorq zmm19, zmm19, zmm22",
        "add rsi, 16",
        "26:",
        "test rdx, 4",
        "jz 27f",
        "movzx eax, word ptr [rsi + 0]",
        "movzx r10d, word ptr [rsi + 2]",
        "vmovdqa64 zmm20, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm22, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm21, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm23, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm20, zmm22, 0x96",
        "vpternlogq zmm20, zmm0, zmm22, 0xb2",
        "vpternlogq zmm6, zmm21, zmm23, 0x96",
        "vpternlogq zmm21, zmm6, zmm23, 0xb2",
        "movzx eax, word ptr [rsi + 4]",
        "movzx r10d, word ptr [rsi + 6]",
        "vmovdqa64 zmm22, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm24, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm23, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm25, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm22, zmm24, 0x96",
        "vpternlogq zmm22, zmm0, zmm24, 0xb2",
        "vpternlogq zmm6, zmm23, zmm25, 0x96",
        "vpternlogq zmm23, zmm6, zmm25, 0xb2",
        "vpternlogq zmm1, zmm20, zmm22, 0x96",
        "vpternlogq zmm20, zmm1, zmm22, 0xb2",
        "vpternlogq zmm7, zmm21, zmm23, 0x96",
        "vpternlogq zmm21, zmm7, zmm23, 0xb2",
        "vpandq zmm22, zmm2, zmm20",
        "vpxorq zmm2, zmm2, zmm20",
        "vpandq zmm20, zmm3, zmm22",
        "vpxorq zmm3, zmm3, zmm22",
        "vpandq zmm22, zmm4, zmm20",
        "vpxorq zmm4, zmm4, zmm20",
        "vpandq zmm20, zmm5, zmm22",
        "vpxorq zmm5, zmm5, zmm22",
        "vpandq zmm22, zmm8, zmm21",
        "vpxorq zmm8, zmm8, zmm21",
        "vpandq zmm21, zmm9, zmm22",
        "vpxorq zmm9, zmm9, zmm22",
        "vpandq zmm22, zmm10, zmm21",
        "vpxorq zmm10, zmm10, zmm21",
        "vpandq zmm21, zmm11, zmm22",
        "vpxorq zmm11, zmm11, zmm22",
        "vpandq zmm22, zmm12, zmm20",
        "vpxorq zmm12, zmm12, zmm20",
        "vpandq zmm20, zmm13, zmm22",
        "vpxorq zmm13, zmm13, zmm22",
        "vpandq zmm22, zmm14, zmm20",
        "vpxorq zmm14, zmm14, zmm20",
        "vpxorq zmm15, zmm15, zmm22",
        "vpandq zmm22, zmm16, zmm21",
        "vpxorq zmm16, zmm16, zmm21",
        "vpandq zmm21, zmm17, zmm22",
        "vpxorq zmm17, zmm17, zmm22",
        "vpandq zmm22, zmm18, zmm21",
        "vpxorq zmm18, zmm18, zmm21",
        "vpxorq zmm19, zmm19, zmm22",
        "add rsi, 8",
        "27:",
        "test rdx, 2",
        "jz 28f",
        "movzx eax, word ptr [rsi + 0]",
        "movzx r10d, word ptr [rsi + 2]",
        "vmovdqa64 zmm20, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm22, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm21, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm23, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm20, zmm22, 0x96",
        "vpternlogq zmm20, zmm0, zmm22, 0xb2",
        "vpternlogq zmm6, zmm21, zmm23, 0x96",
        "vpternlogq zmm21, zmm6, zmm23, 0xb2",
        "vpandq zmm22, zmm1, zmm20",
        "vpxorq zmm1, zmm1, zmm20",
        "vpandq zmm20, zmm2, zmm22",
        "vpxorq zmm2, zmm2, zmm22",
        "vpandq zmm22, zmm3, zmm20",
        "vpxorq zmm3, zmm3, zmm20",
        "vpandq zmm20, zmm4, zmm22",
        "vpxorq zmm4, zmm4, zmm22",
        "vpandq zmm22, zmm5, zmm20",
        "vpxorq zmm5, zmm5, zmm20",
        "vmovdqa64 zmm20, zmm22",
        "vpandq zmm22, zmm7, zmm21",
        "vpxorq zmm7, zmm7, zmm21",
        "vpandq zmm21, zmm8, zmm22",
        "vpxorq zmm8, zmm8, zmm22",
        "vpandq zmm22, zmm9, zmm21",
        "vpxorq zmm9, zmm9, zmm21",
        "vpandq zmm21, zmm10, zmm22",
        "vpxorq zmm10, zmm10, zmm22",
        "vpandq zmm22, zmm11, zmm21",
        "vpxorq zmm11, zmm11, zmm21",
        "vmovdqa64 zmm21, zmm22",
        "vpandq zmm22, zmm12, zmm20",
        "vpxorq zmm12, zmm12, zmm20",
        "vpandq zmm20, zmm13, zmm22",
        "vpxorq zmm13, zmm13, zmm22",
        "vpandq zmm22, zmm14, zmm20",
        "vpxorq zmm14, zmm14, zmm20",
        "vpxorq zmm15, zmm15, zmm22",
        "vpandq zmm22, zmm16, zmm21",
        "vpxorq zmm16, zmm16, zmm21",
        "vpandq zmm21, zmm17, zmm22",
        "vpxorq zmm17, zmm17, zmm22",
        "vpandq zmm22, zmm18, zmm21",
        "vpxorq zmm18, zmm18, zmm21",
        "vpxorq zmm19, zmm19, zmm22",
        "add rsi, 4",
        "28:",
        "inc r12",
        "cmp r12, 15",
        "jne 29f",
        "lea r13, [rip + 29f]",
        "jmp 90f",
        "29:",
        "22:",
        "mov rsi, rcx",
        "mov rdx, r8",
        "30:",
        "cmp rdx, 64",
        "jb 31f",
        "movzx eax, word ptr [rsi + 0]",
        "movzx r10d, word ptr [rsi + 2]",
        "vmovdqa64 zmm20, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm22, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm21, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm23, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm20, zmm22, 0x96",
        "vpternlogq zmm20, zmm0, zmm22, 0x17",
        "vpternlogq zmm6, zmm21, zmm23, 0x96",
        "vpternlogq zmm21, zmm6, zmm23, 0x17",
        "movzx eax, word ptr [rsi + 4]",
        "movzx r10d, word ptr [rsi + 6]",
        "vmovdqa64 zmm22, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm24, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm23, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm25, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm22, zmm24, 0x96",
        "vpternlogq zmm22, zmm0, zmm24, 0x17",
        "vpternlogq zmm6, zmm23, zmm25, 0x96",
        "vpternlogq zmm23, zmm6, zmm25, 0x17",
        "vpternlogq zmm1, zmm20, zmm22, 0x96",
        "vpternlogq zmm20, zmm1, zmm22, 0xb2",
        "vpternlogq zmm7, zmm21, zmm23, 0x96",
        "vpternlogq zmm21, zmm7, zmm23, 0xb2",
        "movzx eax, word ptr [rsi + 8]",
        "movzx r10d, word ptr [rsi + 10]",
        "vmovdqa64 zmm22, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm24, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm23, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm25, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm22, zmm24, 0x96",
        "vpternlogq zmm22, zmm0, zmm24, 0x17",
        "vpternlogq zmm6, zmm23, zmm25, 0x96",
        "vpternlogq zmm23, zmm6, zmm25, 0x17",
        "movzx eax, word ptr [rsi + 12]",
        "movzx r10d, word ptr [rsi + 14]",
        "vmovdqa64 zmm24, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm26, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm25, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm27, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm24, zmm26, 0x96",
        "vpternlogq zmm24, zmm0, zmm26, 0x17",
        "vpternlogq zmm6, zmm25, zmm27, 0x96",
        "vpternlogq zmm25, zmm6, zmm27, 0x17",
        "vpternlogq zmm1, zmm22, zmm24, 0x96",
        "vpternlogq zmm22, zmm1, zmm24, 0xb2",
        "vpternlogq zmm7, zmm23, zmm25, 0x96",
        "vpternlogq zmm23, zmm7, zmm25, 0xb2",
        "vpternlogq zmm2, zmm20, zmm22, 0x96",
        "vpternlogq zmm20, zmm2, zmm22, 0xb2",
        "vpternlogq zmm8, zmm21, zmm23, 0x96",
        "vpternlogq zmm21, zmm8, zmm23, 0xb2",
        "movzx eax, word ptr [rsi + 16]",
        "movzx r10d, word ptr [rsi + 18]",
        "vmovdqa64 zmm22, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm24, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm23, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm25, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm22, zmm24, 0x96",
        "vpternlogq zmm22, zmm0, zmm24, 0x17",
        "vpternlogq zmm6, zmm23, zmm25, 0x96",
        "vpternlogq zmm23, zmm6, zmm25, 0x17",
        "movzx eax, word ptr [rsi + 20]",
        "movzx r10d, word ptr [rsi + 22]",
        "vmovdqa64 zmm24, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm26, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm25, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm27, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm24, zmm26, 0x96",
        "vpternlogq zmm24, zmm0, zmm26, 0x17",
        "vpternlogq zmm6, zmm25, zmm27, 0x96",
        "vpternlogq zmm25, zmm6, zmm27, 0x17",
        "vpternlogq zmm1, zmm22, zmm24, 0x96",
        "vpternlogq zmm22, zmm1, zmm24, 0xb2",
        "vpternlogq zmm7, zmm23, zmm25, 0x96",
        "vpternlogq zmm23, zmm7, zmm25, 0xb2",
        "movzx eax, word ptr [rsi + 24]",
        "movzx r10d, word ptr [rsi + 26]",
        "vmovdqa64 zmm24, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm26, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm25, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm27, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm24, zmm26, 0x96",
        "vpternlogq zmm24, zmm0, zmm26, 0x17",
        "vpternlogq zmm6, zmm25, zmm27, 0x96",
        "vpternlogq zmm25, zmm6, zmm27, 0x17",
        "movzx eax, word ptr [rsi + 28]",
        "movzx r10d, word ptr [rsi + 30]",
        "vmovdqa64 zmm26, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm28, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm27, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm29, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm26, zmm28, 0x96",
        "vpternlogq zmm26, zmm0, zmm28, 0x17",
        "vpternlogq zmm6, zmm27, zmm29, 0x96",
        "vpternlogq zmm27, zmm6, zmm29, 0x17",
        "vpternlogq zmm1, zmm24, zmm26, 0x96",
        "vpternlogq zmm24, zmm1, zmm26, 0xb2",
        "vpternlogq zmm7, zmm25, zmm27, 0x96",
        "vpternlogq zmm25, zmm7, zmm27, 0xb2",
        "vpternlogq zmm2, zmm22, zmm24, 0x96",
        "vpternlogq zmm22, zmm2, zmm24, 0xb2",
        "vpternlogq zmm8, zmm23, zmm25, 0x96",
        "vpternlogq zmm23, zmm8, zmm25, 0xb2",
        "vpternlogq zmm3, zmm20, zmm22, 0x96",
        "vpternlogq zmm20, zmm3, zmm22, 0xb2",
        "vpternlogq zmm9, zmm21, zmm23, 0x96",
        "vpternlogq zmm21, zmm9, zmm23, 0xb2",
        "movzx eax, word ptr [rsi + 32]",
        "movzx r10d, word ptr [rsi + 34]",
        "vmovdqa64 zmm22, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm24, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm23, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm25, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm22, zmm24, 0x96",
        "vpternlogq zmm22, zmm0, zmm24, 0x17",
        "vpternlogq zmm6, zmm23, zmm25, 0x96",
        "vpternlogq zmm23, zmm6, zmm25, 0x17",
        "movzx eax, word ptr [rsi + 36]",
        "movzx r10d, word ptr [rsi + 38]",
        "vmovdqa64 zmm24, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm26, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm25, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm27, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm24, zmm26, 0x96",
        "vpternlogq zmm24, zmm0, zmm26, 0x17",
        "vpternlogq zmm6, zmm25, zmm27, 0x96",
        "vpternlogq zmm25, zmm6, zmm27, 0x17",
        "vpternlogq zmm1, zmm22, zmm24, 0x96",
        "vpternlogq zmm22, zmm1, zmm24, 0xb2",
        "vpternlogq zmm7, zmm23, zmm25, 0x96",
        "vpternlogq zmm23, zmm7, zmm25, 0xb2",
        "movzx eax, word ptr [rsi + 40]",
        "movzx r10d, word ptr [rsi + 42]",
        "vmovdqa64 zmm24, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm26, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm25, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm27, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm24, zmm26, 0x96",
        "vpternlogq zmm24, zmm0, zmm26, 0x17",
        "vpternlogq zmm6, zmm25, zmm27, 0x96",
        "vpternlogq zmm25, zmm6, zmm27, 0x17",
        "movzx eax, word ptr [rsi + 44]",
        "movzx r10d, word ptr [rsi + 46]",
        "vmovdqa64 zmm26, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm28, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm27, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm29, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm26, zmm28, 0x96",
        "vpternlogq zmm26, zmm0, zmm28, 0x17",
        "vpternlogq zmm6, zmm27, zmm29, 0x96",
        "vpternlogq zmm27, zmm6, zmm29, 0x17",
        "vpternlogq zmm1, zmm24, zmm26, 0x96",
        "vpternlogq zmm24, zmm1, zmm26, 0xb2",
        "vpternlogq zmm7, zmm25, zmm27, 0x96",
        "vpternlogq zmm25, zmm7, zmm27, 0xb2",
        "vpternlogq zmm2, zmm22, zmm24, 0x96",
        "vpternlogq zmm22, zmm2, zmm24, 0xb2",
        "vpternlogq zmm8, zmm23, zmm25, 0x96",
        "vpternlogq zmm23, zmm8, zmm25, 0xb2",
        "movzx eax, word ptr [rsi + 48]",
        "movzx r10d, word ptr [rsi + 50]",
        "vmovdqa64 zmm24, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm26, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm25, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm27, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm24, zmm26, 0x96",
        "vpternlogq zmm24, zmm0, zmm26, 0x17",
        "vpternlogq zmm6, zmm25, zmm27, 0x96",
        "vpternlogq zmm25, zmm6, zmm27, 0x17",
        "movzx eax, word ptr [rsi + 52]",
        "movzx r10d, word ptr [rsi + 54]",
        "vmovdqa64 zmm26, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm28, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm27, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm29, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm26, zmm28, 0x96",
        "vpternlogq zmm26, zmm0, zmm28, 0x17",
        "vpternlogq zmm6, zmm27, zmm29, 0x96",
        "vpternlogq zmm27, zmm6, zmm29, 0x17",
        "vpternlogq zmm1, zmm24, zmm26, 0x96",
        "vpternlogq zmm24, zmm1, zmm26, 0xb2",
        "vpternlogq zmm7, zmm25, zmm27, 0x96",
        "vpternlogq zmm25, zmm7, zmm27, 0xb2",
        "movzx eax, word ptr [rsi + 56]",
        "movzx r10d, word ptr [rsi + 58]",
        "vmovdqa64 zmm26, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm28, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm27, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm29, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm26, zmm28, 0x96",
        "vpternlogq zmm26, zmm0, zmm28, 0x17",
        "vpternlogq zmm6, zmm27, zmm29, 0x96",
        "vpternlogq zmm27, zmm6, zmm29, 0x17",
        "movzx eax, word ptr [rsi + 60]",
        "movzx r10d, word ptr [rsi + 62]",
        "vmovdqa64 zmm28, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm30, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm29, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm31, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm28, zmm30, 0x96",
        "vpternlogq zmm28, zmm0, zmm30, 0x17",
        "vpternlogq zmm6, zmm29, zmm31, 0x96",
        "vpternlogq zmm29, zmm6, zmm31, 0x17",
        "vpternlogq zmm1, zmm26, zmm28, 0x96",
        "vpternlogq zmm26, zmm1, zmm28, 0xb2",
        "vpternlogq zmm7, zmm27, zmm29, 0x96",
        "vpternlogq zmm27, zmm7, zmm29, 0xb2",
        "vpternlogq zmm2, zmm24, zmm26, 0x96",
        "vpternlogq zmm24, zmm2, zmm26, 0xb2",
        "vpternlogq zmm8, zmm25, zmm27, 0x96",
        "vpternlogq zmm25, zmm8, zmm27, 0xb2",
        "vpternlogq zmm3, zmm22, zmm24, 0x96",
        "vpternlogq zmm22, zmm3, zmm24, 0xb2",
        "vpternlogq zmm9, zmm23, zmm25, 0x96",
        "vpternlogq zmm23, zmm9, zmm25, 0xb2",
        "vpternlogq zmm4, zmm20, zmm22, 0x96",
        "vpternlogq zmm20, zmm4, zmm22, 0xb2",
        "vpternlogq zmm10, zmm21, zmm23, 0x96",
        "vpternlogq zmm21, zmm10, zmm23, 0xb2",
        "movzx eax, word ptr [rsi + 64]",
        "movzx r10d, word ptr [rsi + 66]",
        "vmovdqa64 zmm22, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm24, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm23, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm25, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm22, zmm24, 0x96",
        "vpternlogq zmm22, zmm0, zmm24, 0x17",
        "vpternlogq zmm6, zmm23, zmm25, 0x96",
        "vpternlogq zmm23, zmm6, zmm25, 0x17",
        "movzx eax, word ptr [rsi + 68]",
        "movzx r10d, word ptr [rsi + 70]",
        "vmovdqa64 zmm24, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm26, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm25, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm27, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm24, zmm26, 0x96",
        "vpternlogq zmm24, zmm0, zmm26, 0x17",
        "vpternlogq zmm6, zmm25, zmm27, 0x96",
        "vpternlogq zmm25, zmm6, zmm27, 0x17",
        "vpternlogq zmm1, zmm22, zmm24, 0x96",
        "vpternlogq zmm22, zmm1, zmm24, 0xb2",
        "vpternlogq zmm7, zmm23, zmm25, 0x96",
        "vpternlogq zmm23, zmm7, zmm25, 0xb2",
        "movzx eax, word ptr [rsi + 72]",
        "movzx r10d, word ptr [rsi + 74]",
        "vmovdqa64 zmm24, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm26, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm25, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm27, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm24, zmm26, 0x96",
        "vpternlogq zmm24, zmm0, zmm26, 0x17",
        "vpternlogq zmm6, zmm25, zmm27, 0x96",
        "vpternlogq zmm25, zmm6, zmm27, 0x17",
        "movzx eax, word ptr [rsi + 76]",
        "movzx r10d, word ptr [rsi + 78]",
        "vmovdqa64 zmm26, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm28, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm27, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm29, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm26, zmm28, 0x96",
        "vpternlogq zmm26, zmm0, zmm28, 0x17",
        "vpternlogq zmm6, zmm27, zmm29, 0x96",
        "vpternlogq zmm27, zmm6, zmm29, 0x17",
        "vpternlogq zmm1, zmm24, zmm26, 0x96",
        "vpternlogq zmm24, zmm1, zmm26, 0xb2",
        "vpternlogq zmm7, zmm25, zmm27, 0x96",
        "vpternlogq zmm25, zmm7, zmm27, 0xb2",
        "vpternlogq zmm2, zmm22, zmm24, 0x96",
        "vpternlogq zmm22, zmm2, zmm24, 0xb2",
        "vpternlogq zmm8, zmm23, zmm25, 0x96",
        "vpternlogq zmm23, zmm8, zmm25, 0xb2",
        "movzx eax, word ptr [rsi + 80]",
        "movzx r10d, word ptr [rsi + 82]",
        "vmovdqa64 zmm24, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm26, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm25, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm27, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm24, zmm26, 0x96",
        "vpternlogq zmm24, zmm0, zmm26, 0x17",
        "vpternlogq zmm6, zmm25, zmm27, 0x96",
        "vpternlogq zmm25, zmm6, zmm27, 0x17",
        "movzx eax, word ptr [rsi + 84]",
        "movzx r10d, word ptr [rsi + 86]",
        "vmovdqa64 zmm26, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm28, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm27, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm29, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm26, zmm28, 0x96",
        "vpternlogq zmm26, zmm0, zmm28, 0x17",
        "vpternlogq zmm6, zmm27, zmm29, 0x96",
        "vpternlogq zmm27, zmm6, zmm29, 0x17",
        "vpternlogq zmm1, zmm24, zmm26, 0x96",
        "vpternlogq zmm24, zmm1, zmm26, 0xb2",
        "vpternlogq zmm7, zmm25, zmm27, 0x96",
        "vpternlogq zmm25, zmm7, zmm27, 0xb2",
        "movzx eax, word ptr [rsi + 88]",
        "movzx r10d, word ptr [rsi + 90]",
        "vmovdqa64 zmm26, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm28, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm27, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm29, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm26, zmm28, 0x96",
        "vpternlogq zmm26, zmm0, zmm28, 0x17",
        "vpternlogq zmm6, zmm27, zmm29, 0x96",
        "vpternlogq zmm27, zmm6, zmm29, 0x17",
        "movzx eax, word ptr [rsi + 92]",
        "movzx r10d, word ptr [rsi + 94]",
        "vmovdqa64 zmm28, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm30, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm29, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm31, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm28, zmm30, 0x96",
        "vpternlogq zmm28, zmm0, zmm30, 0x17",
        "vpternlogq zmm6, zmm29, zmm31, 0x96",
        "vpternlogq zmm29, zmm6, zmm31, 0x17",
        "vpternlogq zmm1, zmm26, zmm28, 0x96",
        "vpternlogq zmm26, zmm1, zmm28, 0xb2",
        "vpternlogq zmm7, zmm27, zmm29, 0x96",
        "vpternlogq zmm27, zmm7, zmm29, 0xb2",
        "vpternlogq zmm2, zmm24, zmm26, 0x96",
        "vpternlogq zmm24, zmm2, zmm26, 0xb2",
        "vpternlogq zmm8, zmm25, zmm27, 0x96",
        "vpternlogq zmm25, zmm8, zmm27, 0xb2",
        "vpternlogq zmm3, zmm22, zmm24, 0x96",
        "vpternlogq zmm22, zmm3, zmm24, 0xb2",
        "vpternlogq zmm9, zmm23, zmm25, 0x96",
        "vpternlogq zmm23, zmm9, zmm25, 0xb2",
        "movzx eax, word ptr [rsi + 96]",
        "movzx r10d, word ptr [rsi + 98]",
        "vmovdqa64 zmm24, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm26, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm25, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm27, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm24, zmm26, 0x96",
        "vpternlogq zmm24, zmm0, zmm26, 0x17",
        "vpternlogq zmm6, zmm25, zmm27, 0x96",
        "vpternlogq zmm25, zmm6, zmm27, 0x17",
        "movzx eax, word ptr [rsi + 100]",
        "movzx r10d, word ptr [rsi + 102]",
        "vmovdqa64 zmm26, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm28, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm27, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm29, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm26, zmm28, 0x96",
        "vpternlogq zmm26, zmm0, zmm28, 0x17",
        "vpternlogq zmm6, zmm27, zmm29, 0x96",
        "vpternlogq zmm27, zmm6, zmm29, 0x17",
        "vpternlogq zmm1, zmm24, zmm26, 0x96",
        "vpternlogq zmm24, zmm1, zmm26, 0xb2",
        "vpternlogq zmm7, zmm25, zmm27, 0x96",
        "vpternlogq zmm25, zmm7, zmm27, 0xb2",
        "movzx eax, word ptr [rsi + 104]",
        "movzx r10d, word ptr [rsi + 106]",
        "vmovdqa64 zmm26, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm28, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm27, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm29, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm26, zmm28, 0x96",
        "vpternlogq zmm26, zmm0, zmm28, 0x17",
        "vpternlogq zmm6, zmm27, zmm29, 0x96",
        "vpternlogq zmm27, zmm6, zmm29, 0x17",
        "movzx eax, word ptr [rsi + 108]",
        "movzx r10d, word ptr [rsi + 110]",
        "vmovdqa64 zmm28, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm30, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm29, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm31, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm28, zmm30, 0x96",
        "vpternlogq zmm28, zmm0, zmm30, 0x17",
        "vpternlogq zmm6, zmm29, zmm31, 0x96",
        "vpternlogq zmm29, zmm6, zmm31, 0x17",
        "vpternlogq zmm1, zmm26, zmm28, 0x96",
        "vpternlogq zmm26, zmm1, zmm28, 0xb2",
        "vpternlogq zmm7, zmm27, zmm29, 0x96",
        "vpternlogq zmm27, zmm7, zmm29, 0xb2",
        "vpternlogq zmm2, zmm24, zmm26, 0x96",
        "vpternlogq zmm24, zmm2, zmm26, 0xb2",
        "vpternlogq zmm8, zmm25, zmm27, 0x96",
        "vpternlogq zmm25, zmm8, zmm27, 0xb2",
        "movzx eax, word ptr [rsi + 112]",
        "movzx r10d, word ptr [rsi + 114]",
        "vmovdqa64 zmm26, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm28, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm27, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm29, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm26, zmm28, 0x96",
        "vpternlogq zmm26, zmm0, zmm28, 0x17",
        "vpternlogq zmm6, zmm27, zmm29, 0x96",
        "vpternlogq zmm27, zmm6, zmm29, 0x17",
        "movzx eax, word ptr [rsi + 116]",
        "movzx r10d, word ptr [rsi + 118]",
        "vmovdqa64 zmm28, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm30, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm29, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm31, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm28, zmm30, 0x96",
        "vpternlogq zmm28, zmm0, zmm30, 0x17",
        "vpternlogq zmm6, zmm29, zmm31, 0x96",
        "vpternlogq zmm29, zmm6, zmm31, 0x17",
        "vpternlogq zmm1, zmm26, zmm28, 0x96",
        "vpternlogq zmm26, zmm1, zmm28, 0xb2",
        "vpternlogq zmm7, zmm27, zmm29, 0x96",
        "vpternlogq zmm27, zmm7, zmm29, 0xb2",
        "movzx eax, word ptr [rsi + 120]",
        "movzx r10d, word ptr [rsi + 122]",
        "vmovdqa64 zmm28, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm30, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm29, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm31, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm28, zmm30, 0x96",
        "vpternlogq zmm28, zmm0, zmm30, 0x17",
        "vpternlogq zmm6, zmm29, zmm31, 0x96",
        "vpternlogq zmm29, zmm6, zmm31, 0x17",
        "movzx eax, word ptr [rsi + 124]",
        "movzx r10d, word ptr [rsi + 126]",
        "vmovdqa64 zmm30, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm31, zmmword ptr [rdi + r10*8 + 0]",
        "vpternlogq zmm0, zmm30, zmm31, 0x96",
        "vpternlogq zmm30, zmm0, zmm31, 0x17",
        "vpternlogq zmm1, zmm28, zmm30, 0x96",
        "vpternlogq zmm28, zmm1, zmm30, 0xb2",
        "vmovdqa64 zmm30, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm31, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm6, zmm30, zmm31, 0x96",
        "vpternlogq zmm30, zmm6, zmm31, 0x17",
        "vpternlogq zmm7, zmm29, zmm30, 0x96",
        "vpternlogq zmm29, zmm7, zmm30, 0xb2",
        "vpternlogq zmm2, zmm26, zmm28, 0x96",
        "vpternlogq zmm26, zmm2, zmm28, 0xb2",
        "vpternlogq zmm8, zmm27, zmm29, 0x96",
        "vpternlogq zmm27, zmm8, zmm29, 0xb2",
        "vpternlogq zmm3, zmm24, zmm26, 0x96",
        "vpternlogq zmm24, zmm3, zmm26, 0xb2",
        "vpternlogq zmm9, zmm25, zmm27, 0x96",
        "vpternlogq zmm25, zmm9, zmm27, 0xb2",
        "vpternlogq zmm4, zmm22, zmm24, 0x96",
        "vpternlogq zmm22, zmm4, zmm24, 0xb2",
        "vpternlogq zmm10, zmm23, zmm25, 0x96",
        "vpternlogq zmm23, zmm10, zmm25, 0xb2",
        "vpternlogq zmm5, zmm20, zmm22, 0x96",
        "vpternlogq zmm20, zmm5, zmm22, 0xb2",
        "vpternlogq zmm11, zmm21, zmm23, 0x96",
        "vpternlogq zmm21, zmm11, zmm23, 0xb2",
        "vpandq zmm22, zmm12, zmm20",
        "vpxorq zmm12, zmm12, zmm20",
        "vpandq zmm20, zmm13, zmm22",
        "vpxorq zmm13, zmm13, zmm22",
        "vpandq zmm22, zmm14, zmm20",
        "vpxorq zmm14, zmm14, zmm20",
        "vpxorq zmm15, zmm15, zmm22",
        "vpandq zmm22, zmm16, zmm21",
        "vpxorq zmm16, zmm16, zmm21",
        "vpandq zmm21, zmm17, zmm22",
        "vpxorq zmm17, zmm17, zmm22",
        "vpandq zmm22, zmm18, zmm21",
        "vpxorq zmm18, zmm18, zmm21",
        "vpxorq zmm19, zmm19, zmm22",
        "inc r12",
        "cmp r12, 15",
        "jne 33f",
        "lea r13, [rip + 33f]",
        "jmp 90f",
        "33:",
        "add rsi, 128",
        "sub rdx, 64",
        "jmp 30b",
        "31:",
        "test rdx, rdx",
        "jz 32f",
        "test rdx, 32",
        "jz 34f",
        "movzx eax, word ptr [rsi + 0]",
        "movzx r10d, word ptr [rsi + 2]",
        "vmovdqa64 zmm20, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm22, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm21, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm23, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm20, zmm22, 0x96",
        "vpternlogq zmm20, zmm0, zmm22, 0x17",
        "vpternlogq zmm6, zmm21, zmm23, 0x96",
        "vpternlogq zmm21, zmm6, zmm23, 0x17",
        "movzx eax, word ptr [rsi + 4]",
        "movzx r10d, word ptr [rsi + 6]",
        "vmovdqa64 zmm22, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm24, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm23, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm25, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm22, zmm24, 0x96",
        "vpternlogq zmm22, zmm0, zmm24, 0x17",
        "vpternlogq zmm6, zmm23, zmm25, 0x96",
        "vpternlogq zmm23, zmm6, zmm25, 0x17",
        "vpternlogq zmm1, zmm20, zmm22, 0x96",
        "vpternlogq zmm20, zmm1, zmm22, 0xb2",
        "vpternlogq zmm7, zmm21, zmm23, 0x96",
        "vpternlogq zmm21, zmm7, zmm23, 0xb2",
        "movzx eax, word ptr [rsi + 8]",
        "movzx r10d, word ptr [rsi + 10]",
        "vmovdqa64 zmm22, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm24, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm23, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm25, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm22, zmm24, 0x96",
        "vpternlogq zmm22, zmm0, zmm24, 0x17",
        "vpternlogq zmm6, zmm23, zmm25, 0x96",
        "vpternlogq zmm23, zmm6, zmm25, 0x17",
        "movzx eax, word ptr [rsi + 12]",
        "movzx r10d, word ptr [rsi + 14]",
        "vmovdqa64 zmm24, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm26, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm25, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm27, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm24, zmm26, 0x96",
        "vpternlogq zmm24, zmm0, zmm26, 0x17",
        "vpternlogq zmm6, zmm25, zmm27, 0x96",
        "vpternlogq zmm25, zmm6, zmm27, 0x17",
        "vpternlogq zmm1, zmm22, zmm24, 0x96",
        "vpternlogq zmm22, zmm1, zmm24, 0xb2",
        "vpternlogq zmm7, zmm23, zmm25, 0x96",
        "vpternlogq zmm23, zmm7, zmm25, 0xb2",
        "vpternlogq zmm2, zmm20, zmm22, 0x96",
        "vpternlogq zmm20, zmm2, zmm22, 0xb2",
        "vpternlogq zmm8, zmm21, zmm23, 0x96",
        "vpternlogq zmm21, zmm8, zmm23, 0xb2",
        "movzx eax, word ptr [rsi + 16]",
        "movzx r10d, word ptr [rsi + 18]",
        "vmovdqa64 zmm22, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm24, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm23, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm25, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm22, zmm24, 0x96",
        "vpternlogq zmm22, zmm0, zmm24, 0x17",
        "vpternlogq zmm6, zmm23, zmm25, 0x96",
        "vpternlogq zmm23, zmm6, zmm25, 0x17",
        "movzx eax, word ptr [rsi + 20]",
        "movzx r10d, word ptr [rsi + 22]",
        "vmovdqa64 zmm24, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm26, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm25, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm27, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm24, zmm26, 0x96",
        "vpternlogq zmm24, zmm0, zmm26, 0x17",
        "vpternlogq zmm6, zmm25, zmm27, 0x96",
        "vpternlogq zmm25, zmm6, zmm27, 0x17",
        "vpternlogq zmm1, zmm22, zmm24, 0x96",
        "vpternlogq zmm22, zmm1, zmm24, 0xb2",
        "vpternlogq zmm7, zmm23, zmm25, 0x96",
        "vpternlogq zmm23, zmm7, zmm25, 0xb2",
        "movzx eax, word ptr [rsi + 24]",
        "movzx r10d, word ptr [rsi + 26]",
        "vmovdqa64 zmm24, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm26, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm25, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm27, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm24, zmm26, 0x96",
        "vpternlogq zmm24, zmm0, zmm26, 0x17",
        "vpternlogq zmm6, zmm25, zmm27, 0x96",
        "vpternlogq zmm25, zmm6, zmm27, 0x17",
        "movzx eax, word ptr [rsi + 28]",
        "movzx r10d, word ptr [rsi + 30]",
        "vmovdqa64 zmm26, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm28, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm27, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm29, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm26, zmm28, 0x96",
        "vpternlogq zmm26, zmm0, zmm28, 0x17",
        "vpternlogq zmm6, zmm27, zmm29, 0x96",
        "vpternlogq zmm27, zmm6, zmm29, 0x17",
        "vpternlogq zmm1, zmm24, zmm26, 0x96",
        "vpternlogq zmm24, zmm1, zmm26, 0xb2",
        "vpternlogq zmm7, zmm25, zmm27, 0x96",
        "vpternlogq zmm25, zmm7, zmm27, 0xb2",
        "vpternlogq zmm2, zmm22, zmm24, 0x96",
        "vpternlogq zmm22, zmm2, zmm24, 0xb2",
        "vpternlogq zmm8, zmm23, zmm25, 0x96",
        "vpternlogq zmm23, zmm8, zmm25, 0xb2",
        "vpternlogq zmm3, zmm20, zmm22, 0x96",
        "vpternlogq zmm20, zmm3, zmm22, 0xb2",
        "vpternlogq zmm9, zmm21, zmm23, 0x96",
        "vpternlogq zmm21, zmm9, zmm23, 0xb2",
        "movzx eax, word ptr [rsi + 32]",
        "movzx r10d, word ptr [rsi + 34]",
        "vmovdqa64 zmm22, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm24, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm23, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm25, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm22, zmm24, 0x96",
        "vpternlogq zmm22, zmm0, zmm24, 0x17",
        "vpternlogq zmm6, zmm23, zmm25, 0x96",
        "vpternlogq zmm23, zmm6, zmm25, 0x17",
        "movzx eax, word ptr [rsi + 36]",
        "movzx r10d, word ptr [rsi + 38]",
        "vmovdqa64 zmm24, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm26, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm25, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm27, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm24, zmm26, 0x96",
        "vpternlogq zmm24, zmm0, zmm26, 0x17",
        "vpternlogq zmm6, zmm25, zmm27, 0x96",
        "vpternlogq zmm25, zmm6, zmm27, 0x17",
        "vpternlogq zmm1, zmm22, zmm24, 0x96",
        "vpternlogq zmm22, zmm1, zmm24, 0xb2",
        "vpternlogq zmm7, zmm23, zmm25, 0x96",
        "vpternlogq zmm23, zmm7, zmm25, 0xb2",
        "movzx eax, word ptr [rsi + 40]",
        "movzx r10d, word ptr [rsi + 42]",
        "vmovdqa64 zmm24, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm26, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm25, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm27, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm24, zmm26, 0x96",
        "vpternlogq zmm24, zmm0, zmm26, 0x17",
        "vpternlogq zmm6, zmm25, zmm27, 0x96",
        "vpternlogq zmm25, zmm6, zmm27, 0x17",
        "movzx eax, word ptr [rsi + 44]",
        "movzx r10d, word ptr [rsi + 46]",
        "vmovdqa64 zmm26, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm28, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm27, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm29, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm26, zmm28, 0x96",
        "vpternlogq zmm26, zmm0, zmm28, 0x17",
        "vpternlogq zmm6, zmm27, zmm29, 0x96",
        "vpternlogq zmm27, zmm6, zmm29, 0x17",
        "vpternlogq zmm1, zmm24, zmm26, 0x96",
        "vpternlogq zmm24, zmm1, zmm26, 0xb2",
        "vpternlogq zmm7, zmm25, zmm27, 0x96",
        "vpternlogq zmm25, zmm7, zmm27, 0xb2",
        "vpternlogq zmm2, zmm22, zmm24, 0x96",
        "vpternlogq zmm22, zmm2, zmm24, 0xb2",
        "vpternlogq zmm8, zmm23, zmm25, 0x96",
        "vpternlogq zmm23, zmm8, zmm25, 0xb2",
        "movzx eax, word ptr [rsi + 48]",
        "movzx r10d, word ptr [rsi + 50]",
        "vmovdqa64 zmm24, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm26, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm25, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm27, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm24, zmm26, 0x96",
        "vpternlogq zmm24, zmm0, zmm26, 0x17",
        "vpternlogq zmm6, zmm25, zmm27, 0x96",
        "vpternlogq zmm25, zmm6, zmm27, 0x17",
        "movzx eax, word ptr [rsi + 52]",
        "movzx r10d, word ptr [rsi + 54]",
        "vmovdqa64 zmm26, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm28, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm27, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm29, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm26, zmm28, 0x96",
        "vpternlogq zmm26, zmm0, zmm28, 0x17",
        "vpternlogq zmm6, zmm27, zmm29, 0x96",
        "vpternlogq zmm27, zmm6, zmm29, 0x17",
        "vpternlogq zmm1, zmm24, zmm26, 0x96",
        "vpternlogq zmm24, zmm1, zmm26, 0xb2",
        "vpternlogq zmm7, zmm25, zmm27, 0x96",
        "vpternlogq zmm25, zmm7, zmm27, 0xb2",
        "movzx eax, word ptr [rsi + 56]",
        "movzx r10d, word ptr [rsi + 58]",
        "vmovdqa64 zmm26, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm28, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm27, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm29, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm26, zmm28, 0x96",
        "vpternlogq zmm26, zmm0, zmm28, 0x17",
        "vpternlogq zmm6, zmm27, zmm29, 0x96",
        "vpternlogq zmm27, zmm6, zmm29, 0x17",
        "movzx eax, word ptr [rsi + 60]",
        "movzx r10d, word ptr [rsi + 62]",
        "vmovdqa64 zmm28, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm30, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm29, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm31, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm28, zmm30, 0x96",
        "vpternlogq zmm28, zmm0, zmm30, 0x17",
        "vpternlogq zmm6, zmm29, zmm31, 0x96",
        "vpternlogq zmm29, zmm6, zmm31, 0x17",
        "vpternlogq zmm1, zmm26, zmm28, 0x96",
        "vpternlogq zmm26, zmm1, zmm28, 0xb2",
        "vpternlogq zmm7, zmm27, zmm29, 0x96",
        "vpternlogq zmm27, zmm7, zmm29, 0xb2",
        "vpternlogq zmm2, zmm24, zmm26, 0x96",
        "vpternlogq zmm24, zmm2, zmm26, 0xb2",
        "vpternlogq zmm8, zmm25, zmm27, 0x96",
        "vpternlogq zmm25, zmm8, zmm27, 0xb2",
        "vpternlogq zmm3, zmm22, zmm24, 0x96",
        "vpternlogq zmm22, zmm3, zmm24, 0xb2",
        "vpternlogq zmm9, zmm23, zmm25, 0x96",
        "vpternlogq zmm23, zmm9, zmm25, 0xb2",
        "vpternlogq zmm4, zmm20, zmm22, 0x96",
        "vpternlogq zmm20, zmm4, zmm22, 0xb2",
        "vpternlogq zmm10, zmm21, zmm23, 0x96",
        "vpternlogq zmm21, zmm10, zmm23, 0xb2",
        "vpandq zmm22, zmm5, zmm20",
        "vpxorq zmm5, zmm5, zmm20",
        "vmovdqa64 zmm20, zmm22",
        "vpandq zmm22, zmm11, zmm21",
        "vpxorq zmm11, zmm11, zmm21",
        "vmovdqa64 zmm21, zmm22",
        "vpandq zmm22, zmm12, zmm20",
        "vpxorq zmm12, zmm12, zmm20",
        "vpandq zmm20, zmm13, zmm22",
        "vpxorq zmm13, zmm13, zmm22",
        "vpandq zmm22, zmm14, zmm20",
        "vpxorq zmm14, zmm14, zmm20",
        "vpxorq zmm15, zmm15, zmm22",
        "vpandq zmm22, zmm16, zmm21",
        "vpxorq zmm16, zmm16, zmm21",
        "vpandq zmm21, zmm17, zmm22",
        "vpxorq zmm17, zmm17, zmm22",
        "vpandq zmm22, zmm18, zmm21",
        "vpxorq zmm18, zmm18, zmm21",
        "vpxorq zmm19, zmm19, zmm22",
        "add rsi, 64",
        "34:",
        "test rdx, 16",
        "jz 35f",
        "movzx eax, word ptr [rsi + 0]",
        "movzx r10d, word ptr [rsi + 2]",
        "vmovdqa64 zmm20, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm22, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm21, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm23, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm20, zmm22, 0x96",
        "vpternlogq zmm20, zmm0, zmm22, 0x17",
        "vpternlogq zmm6, zmm21, zmm23, 0x96",
        "vpternlogq zmm21, zmm6, zmm23, 0x17",
        "movzx eax, word ptr [rsi + 4]",
        "movzx r10d, word ptr [rsi + 6]",
        "vmovdqa64 zmm22, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm24, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm23, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm25, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm22, zmm24, 0x96",
        "vpternlogq zmm22, zmm0, zmm24, 0x17",
        "vpternlogq zmm6, zmm23, zmm25, 0x96",
        "vpternlogq zmm23, zmm6, zmm25, 0x17",
        "vpternlogq zmm1, zmm20, zmm22, 0x96",
        "vpternlogq zmm20, zmm1, zmm22, 0xb2",
        "vpternlogq zmm7, zmm21, zmm23, 0x96",
        "vpternlogq zmm21, zmm7, zmm23, 0xb2",
        "movzx eax, word ptr [rsi + 8]",
        "movzx r10d, word ptr [rsi + 10]",
        "vmovdqa64 zmm22, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm24, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm23, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm25, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm22, zmm24, 0x96",
        "vpternlogq zmm22, zmm0, zmm24, 0x17",
        "vpternlogq zmm6, zmm23, zmm25, 0x96",
        "vpternlogq zmm23, zmm6, zmm25, 0x17",
        "movzx eax, word ptr [rsi + 12]",
        "movzx r10d, word ptr [rsi + 14]",
        "vmovdqa64 zmm24, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm26, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm25, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm27, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm24, zmm26, 0x96",
        "vpternlogq zmm24, zmm0, zmm26, 0x17",
        "vpternlogq zmm6, zmm25, zmm27, 0x96",
        "vpternlogq zmm25, zmm6, zmm27, 0x17",
        "vpternlogq zmm1, zmm22, zmm24, 0x96",
        "vpternlogq zmm22, zmm1, zmm24, 0xb2",
        "vpternlogq zmm7, zmm23, zmm25, 0x96",
        "vpternlogq zmm23, zmm7, zmm25, 0xb2",
        "vpternlogq zmm2, zmm20, zmm22, 0x96",
        "vpternlogq zmm20, zmm2, zmm22, 0xb2",
        "vpternlogq zmm8, zmm21, zmm23, 0x96",
        "vpternlogq zmm21, zmm8, zmm23, 0xb2",
        "movzx eax, word ptr [rsi + 16]",
        "movzx r10d, word ptr [rsi + 18]",
        "vmovdqa64 zmm22, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm24, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm23, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm25, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm22, zmm24, 0x96",
        "vpternlogq zmm22, zmm0, zmm24, 0x17",
        "vpternlogq zmm6, zmm23, zmm25, 0x96",
        "vpternlogq zmm23, zmm6, zmm25, 0x17",
        "movzx eax, word ptr [rsi + 20]",
        "movzx r10d, word ptr [rsi + 22]",
        "vmovdqa64 zmm24, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm26, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm25, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm27, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm24, zmm26, 0x96",
        "vpternlogq zmm24, zmm0, zmm26, 0x17",
        "vpternlogq zmm6, zmm25, zmm27, 0x96",
        "vpternlogq zmm25, zmm6, zmm27, 0x17",
        "vpternlogq zmm1, zmm22, zmm24, 0x96",
        "vpternlogq zmm22, zmm1, zmm24, 0xb2",
        "vpternlogq zmm7, zmm23, zmm25, 0x96",
        "vpternlogq zmm23, zmm7, zmm25, 0xb2",
        "movzx eax, word ptr [rsi + 24]",
        "movzx r10d, word ptr [rsi + 26]",
        "vmovdqa64 zmm24, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm26, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm25, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm27, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm24, zmm26, 0x96",
        "vpternlogq zmm24, zmm0, zmm26, 0x17",
        "vpternlogq zmm6, zmm25, zmm27, 0x96",
        "vpternlogq zmm25, zmm6, zmm27, 0x17",
        "movzx eax, word ptr [rsi + 28]",
        "movzx r10d, word ptr [rsi + 30]",
        "vmovdqa64 zmm26, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm28, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm27, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm29, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm26, zmm28, 0x96",
        "vpternlogq zmm26, zmm0, zmm28, 0x17",
        "vpternlogq zmm6, zmm27, zmm29, 0x96",
        "vpternlogq zmm27, zmm6, zmm29, 0x17",
        "vpternlogq zmm1, zmm24, zmm26, 0x96",
        "vpternlogq zmm24, zmm1, zmm26, 0xb2",
        "vpternlogq zmm7, zmm25, zmm27, 0x96",
        "vpternlogq zmm25, zmm7, zmm27, 0xb2",
        "vpternlogq zmm2, zmm22, zmm24, 0x96",
        "vpternlogq zmm22, zmm2, zmm24, 0xb2",
        "vpternlogq zmm8, zmm23, zmm25, 0x96",
        "vpternlogq zmm23, zmm8, zmm25, 0xb2",
        "vpternlogq zmm3, zmm20, zmm22, 0x96",
        "vpternlogq zmm20, zmm3, zmm22, 0xb2",
        "vpternlogq zmm9, zmm21, zmm23, 0x96",
        "vpternlogq zmm21, zmm9, zmm23, 0xb2",
        "vpandq zmm22, zmm4, zmm20",
        "vpxorq zmm4, zmm4, zmm20",
        "vpandq zmm20, zmm5, zmm22",
        "vpxorq zmm5, zmm5, zmm22",
        "vpandq zmm22, zmm10, zmm21",
        "vpxorq zmm10, zmm10, zmm21",
        "vpandq zmm21, zmm11, zmm22",
        "vpxorq zmm11, zmm11, zmm22",
        "vpandq zmm22, zmm12, zmm20",
        "vpxorq zmm12, zmm12, zmm20",
        "vpandq zmm20, zmm13, zmm22",
        "vpxorq zmm13, zmm13, zmm22",
        "vpandq zmm22, zmm14, zmm20",
        "vpxorq zmm14, zmm14, zmm20",
        "vpxorq zmm15, zmm15, zmm22",
        "vpandq zmm22, zmm16, zmm21",
        "vpxorq zmm16, zmm16, zmm21",
        "vpandq zmm21, zmm17, zmm22",
        "vpxorq zmm17, zmm17, zmm22",
        "vpandq zmm22, zmm18, zmm21",
        "vpxorq zmm18, zmm18, zmm21",
        "vpxorq zmm19, zmm19, zmm22",
        "add rsi, 32",
        "35:",
        "test rdx, 8",
        "jz 36f",
        "movzx eax, word ptr [rsi + 0]",
        "movzx r10d, word ptr [rsi + 2]",
        "vmovdqa64 zmm20, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm22, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm21, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm23, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm20, zmm22, 0x96",
        "vpternlogq zmm20, zmm0, zmm22, 0x17",
        "vpternlogq zmm6, zmm21, zmm23, 0x96",
        "vpternlogq zmm21, zmm6, zmm23, 0x17",
        "movzx eax, word ptr [rsi + 4]",
        "movzx r10d, word ptr [rsi + 6]",
        "vmovdqa64 zmm22, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm24, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm23, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm25, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm22, zmm24, 0x96",
        "vpternlogq zmm22, zmm0, zmm24, 0x17",
        "vpternlogq zmm6, zmm23, zmm25, 0x96",
        "vpternlogq zmm23, zmm6, zmm25, 0x17",
        "vpternlogq zmm1, zmm20, zmm22, 0x96",
        "vpternlogq zmm20, zmm1, zmm22, 0xb2",
        "vpternlogq zmm7, zmm21, zmm23, 0x96",
        "vpternlogq zmm21, zmm7, zmm23, 0xb2",
        "movzx eax, word ptr [rsi + 8]",
        "movzx r10d, word ptr [rsi + 10]",
        "vmovdqa64 zmm22, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm24, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm23, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm25, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm22, zmm24, 0x96",
        "vpternlogq zmm22, zmm0, zmm24, 0x17",
        "vpternlogq zmm6, zmm23, zmm25, 0x96",
        "vpternlogq zmm23, zmm6, zmm25, 0x17",
        "movzx eax, word ptr [rsi + 12]",
        "movzx r10d, word ptr [rsi + 14]",
        "vmovdqa64 zmm24, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm26, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm25, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm27, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm24, zmm26, 0x96",
        "vpternlogq zmm24, zmm0, zmm26, 0x17",
        "vpternlogq zmm6, zmm25, zmm27, 0x96",
        "vpternlogq zmm25, zmm6, zmm27, 0x17",
        "vpternlogq zmm1, zmm22, zmm24, 0x96",
        "vpternlogq zmm22, zmm1, zmm24, 0xb2",
        "vpternlogq zmm7, zmm23, zmm25, 0x96",
        "vpternlogq zmm23, zmm7, zmm25, 0xb2",
        "vpternlogq zmm2, zmm20, zmm22, 0x96",
        "vpternlogq zmm20, zmm2, zmm22, 0xb2",
        "vpternlogq zmm8, zmm21, zmm23, 0x96",
        "vpternlogq zmm21, zmm8, zmm23, 0xb2",
        "vpandq zmm22, zmm3, zmm20",
        "vpxorq zmm3, zmm3, zmm20",
        "vpandq zmm20, zmm4, zmm22",
        "vpxorq zmm4, zmm4, zmm22",
        "vpandq zmm22, zmm5, zmm20",
        "vpxorq zmm5, zmm5, zmm20",
        "vmovdqa64 zmm20, zmm22",
        "vpandq zmm22, zmm9, zmm21",
        "vpxorq zmm9, zmm9, zmm21",
        "vpandq zmm21, zmm10, zmm22",
        "vpxorq zmm10, zmm10, zmm22",
        "vpandq zmm22, zmm11, zmm21",
        "vpxorq zmm11, zmm11, zmm21",
        "vmovdqa64 zmm21, zmm22",
        "vpandq zmm22, zmm12, zmm20",
        "vpxorq zmm12, zmm12, zmm20",
        "vpandq zmm20, zmm13, zmm22",
        "vpxorq zmm13, zmm13, zmm22",
        "vpandq zmm22, zmm14, zmm20",
        "vpxorq zmm14, zmm14, zmm20",
        "vpxorq zmm15, zmm15, zmm22",
        "vpandq zmm22, zmm16, zmm21",
        "vpxorq zmm16, zmm16, zmm21",
        "vpandq zmm21, zmm17, zmm22",
        "vpxorq zmm17, zmm17, zmm22",
        "vpandq zmm22, zmm18, zmm21",
        "vpxorq zmm18, zmm18, zmm21",
        "vpxorq zmm19, zmm19, zmm22",
        "add rsi, 16",
        "36:",
        "test rdx, 4",
        "jz 37f",
        "movzx eax, word ptr [rsi + 0]",
        "movzx r10d, word ptr [rsi + 2]",
        "vmovdqa64 zmm20, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm22, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm21, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm23, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm20, zmm22, 0x96",
        "vpternlogq zmm20, zmm0, zmm22, 0x17",
        "vpternlogq zmm6, zmm21, zmm23, 0x96",
        "vpternlogq zmm21, zmm6, zmm23, 0x17",
        "movzx eax, word ptr [rsi + 4]",
        "movzx r10d, word ptr [rsi + 6]",
        "vmovdqa64 zmm22, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm24, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm23, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm25, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm22, zmm24, 0x96",
        "vpternlogq zmm22, zmm0, zmm24, 0x17",
        "vpternlogq zmm6, zmm23, zmm25, 0x96",
        "vpternlogq zmm23, zmm6, zmm25, 0x17",
        "vpternlogq zmm1, zmm20, zmm22, 0x96",
        "vpternlogq zmm20, zmm1, zmm22, 0xb2",
        "vpternlogq zmm7, zmm21, zmm23, 0x96",
        "vpternlogq zmm21, zmm7, zmm23, 0xb2",
        "vpandq zmm22, zmm2, zmm20",
        "vpxorq zmm2, zmm2, zmm20",
        "vpandq zmm20, zmm3, zmm22",
        "vpxorq zmm3, zmm3, zmm22",
        "vpandq zmm22, zmm4, zmm20",
        "vpxorq zmm4, zmm4, zmm20",
        "vpandq zmm20, zmm5, zmm22",
        "vpxorq zmm5, zmm5, zmm22",
        "vpandq zmm22, zmm8, zmm21",
        "vpxorq zmm8, zmm8, zmm21",
        "vpandq zmm21, zmm9, zmm22",
        "vpxorq zmm9, zmm9, zmm22",
        "vpandq zmm22, zmm10, zmm21",
        "vpxorq zmm10, zmm10, zmm21",
        "vpandq zmm21, zmm11, zmm22",
        "vpxorq zmm11, zmm11, zmm22",
        "vpandq zmm22, zmm12, zmm20",
        "vpxorq zmm12, zmm12, zmm20",
        "vpandq zmm20, zmm13, zmm22",
        "vpxorq zmm13, zmm13, zmm22",
        "vpandq zmm22, zmm14, zmm20",
        "vpxorq zmm14, zmm14, zmm20",
        "vpxorq zmm15, zmm15, zmm22",
        "vpandq zmm22, zmm16, zmm21",
        "vpxorq zmm16, zmm16, zmm21",
        "vpandq zmm21, zmm17, zmm22",
        "vpxorq zmm17, zmm17, zmm22",
        "vpandq zmm22, zmm18, zmm21",
        "vpxorq zmm18, zmm18, zmm21",
        "vpxorq zmm19, zmm19, zmm22",
        "add rsi, 8",
        "37:",
        "test rdx, 2",
        "jz 38f",
        "movzx eax, word ptr [rsi + 0]",
        "movzx r10d, word ptr [rsi + 2]",
        "vmovdqa64 zmm20, zmmword ptr [rdi + rax*8 + 0]",
        "vmovdqa64 zmm22, zmmword ptr [rdi + r10*8 + 0]",
        "vmovdqa64 zmm21, zmmword ptr [rdi + rax*8 + 64]",
        "vmovdqa64 zmm23, zmmword ptr [rdi + r10*8 + 64]",
        "vpternlogq zmm0, zmm20, zmm22, 0x96",
        "vpternlogq zmm20, zmm0, zmm22, 0x17",
        "vpternlogq zmm6, zmm21, zmm23, 0x96",
        "vpternlogq zmm21, zmm6, zmm23, 0x17",
        "vpandq zmm22, zmm1, zmm20",
        "vpxorq zmm1, zmm1, zmm20",
        "vpandq zmm20, zmm2, zmm22",
        "vpxorq zmm2, zmm2, zmm22",
        "vpandq zmm22, zmm3, zmm20",
        "vpxorq zmm3, zmm3, zmm20",
        "vpandq zmm20, zmm4, zmm22",
        "vpxorq zmm4, zmm4, zmm22",
        "vpandq zmm22, zmm5, zmm20",
        "vpxorq zmm5, zmm5, zmm20",
        "vmovdqa64 zmm20, zmm22",
        "vpandq zmm22, zmm7, zmm21",
        "vpxorq zmm7, zmm7, zmm21",
        "vpandq zmm21, zmm8, zmm22",
        "vpxorq zmm8, zmm8, zmm22",
        "vpandq zmm22, zmm9, zmm21",
        "vpxorq zmm9, zmm9, zmm21",
        "vpandq zmm21, zmm10, zmm22",
        "vpxorq zmm10, zmm10, zmm22",
        "vpandq zmm22, zmm11, zmm21",
        "vpxorq zmm11, zmm11, zmm21",
        "vmovdqa64 zmm21, zmm22",
        "vpandq zmm22, zmm12, zmm20",
        "vpxorq zmm12, zmm12, zmm20",
        "vpandq zmm20, zmm13, zmm22",
        "vpxorq zmm13, zmm13, zmm22",
        "vpandq zmm22, zmm14, zmm20",
        "vpxorq zmm14, zmm14, zmm20",
        "vpxorq zmm15, zmm15, zmm22",
        "vpandq zmm22, zmm16, zmm21",
        "vpxorq zmm16, zmm16, zmm21",
        "vpandq zmm21, zmm17, zmm22",
        "vpxorq zmm17, zmm17, zmm22",
        "vpandq zmm22, zmm18, zmm21",
        "vpxorq zmm18, zmm18, zmm21",
        "vpxorq zmm19, zmm19, zmm22",
        "add rsi, 4",
        "38:",
        "inc r12",
        "cmp r12, 15",
        "jne 39f",
        "lea r13, [rip + 39f]",
        "jmp 90f",
        "39:",
        "32:",
        "test r12, r12",
        "jz 91f",
        "lea r13, [rip + 91f]",
        "jmp 90f",
        "91:",
        "vmovdqa64 zmmword ptr [r9 + 0], zmm0",
        "vmovdqa64 zmmword ptr [r9 + 64], zmm1",
        "vmovdqa64 zmmword ptr [r9 + 128], zmm2",
        "vmovdqa64 zmmword ptr [r9 + 192], zmm3",
        "vmovdqa64 zmmword ptr [r9 + 256], zmm4",
        "vmovdqa64 zmmword ptr [r9 + 320], zmm5",
        "vmovdqa64 zmmword ptr [r9 + 960], zmm6",
        "vmovdqa64 zmmword ptr [r9 + 1024], zmm7",
        "vmovdqa64 zmmword ptr [r9 + 1088], zmm8",
        "vmovdqa64 zmmword ptr [r9 + 1152], zmm9",
        "vmovdqa64 zmmword ptr [r9 + 1216], zmm10",
        "vmovdqa64 zmmword ptr [r9 + 1280], zmm11",
        "jmp 92f",
        "90:",
        "vmovdqa64 zmm20, zmmword ptr [r9 + 384]",
        "vpxorq zmm21, zmm20, zmm12",
        "vmovdqa64 zmmword ptr [r9 + 384], zmm21",
        "vpandq zmm20, zmm20, zmm12",
        "vmovdqa64 zmm21, zmmword ptr [r9 + 448]",
        "vpternlogq zmm21, zmm13, zmm20, 0x96",
        "vmovdqa64 zmmword ptr [r9 + 448], zmm21",
        "vpternlogq zmm13, zmm21, zmm20, 0xb2",
        "vmovdqa64 zmm21, zmmword ptr [r9 + 512]",
        "vpternlogq zmm21, zmm14, zmm13, 0x96",
        "vmovdqa64 zmmword ptr [r9 + 512], zmm21",
        "vpternlogq zmm14, zmm21, zmm13, 0xb2",
        "vmovdqa64 zmm21, zmmword ptr [r9 + 576]",
        "vpternlogq zmm21, zmm15, zmm14, 0x96",
        "vmovdqa64 zmmword ptr [r9 + 576], zmm21",
        "vpternlogq zmm15, zmm21, zmm14, 0xb2",
        "vmovdqa64 zmm20, zmmword ptr [r9 + 640]",
        "vpxorq zmm21, zmm20, zmm15",
        "vmovdqa64 zmmword ptr [r9 + 640], zmm21",
        "vpandq zmm15, zmm15, zmm20",
        "vmovdqa64 zmm20, zmmword ptr [r9 + 704]",
        "vpxorq zmm21, zmm20, zmm15",
        "vmovdqa64 zmmword ptr [r9 + 704], zmm21",
        "vpandq zmm15, zmm15, zmm20",
        "vmovdqa64 zmm20, zmmword ptr [r9 + 768]",
        "vpxorq zmm21, zmm20, zmm15",
        "vmovdqa64 zmmword ptr [r9 + 768], zmm21",
        "vpandq zmm15, zmm15, zmm20",
        "vmovdqa64 zmm20, zmmword ptr [r9 + 832]",
        "vpxorq zmm21, zmm20, zmm15",
        "vmovdqa64 zmmword ptr [r9 + 832], zmm21",
        "vpandq zmm15, zmm15, zmm20",
        "vpxorq zmm15, zmm15, zmmword ptr [r9 + 896]",
        "vmovdqa64 zmmword ptr [r9 + 896], zmm15",
        "vmovdqa64 zmm20, zmmword ptr [r9 + 1344]",
        "vpxorq zmm21, zmm20, zmm16",
        "vmovdqa64 zmmword ptr [r9 + 1344], zmm21",
        "vpandq zmm20, zmm20, zmm16",
        "vmovdqa64 zmm21, zmmword ptr [r9 + 1408]",
        "vpternlogq zmm21, zmm17, zmm20, 0x96",
        "vmovdqa64 zmmword ptr [r9 + 1408], zmm21",
        "vpternlogq zmm17, zmm21, zmm20, 0xb2",
        "vmovdqa64 zmm21, zmmword ptr [r9 + 1472]",
        "vpternlogq zmm21, zmm18, zmm17, 0x96",
        "vmovdqa64 zmmword ptr [r9 + 1472], zmm21",
        "vpternlogq zmm18, zmm21, zmm17, 0xb2",
        "vmovdqa64 zmm21, zmmword ptr [r9 + 1536]",
        "vpternlogq zmm21, zmm19, zmm18, 0x96",
        "vmovdqa64 zmmword ptr [r9 + 1536], zmm21",
        "vpternlogq zmm19, zmm21, zmm18, 0xb2",
        "vmovdqa64 zmm20, zmmword ptr [r9 + 1600]",
        "vpxorq zmm21, zmm20, zmm19",
        "vmovdqa64 zmmword ptr [r9 + 1600], zmm21",
        "vpandq zmm19, zmm19, zmm20",
        "vmovdqa64 zmm20, zmmword ptr [r9 + 1664]",
        "vpxorq zmm21, zmm20, zmm19",
        "vmovdqa64 zmmword ptr [r9 + 1664], zmm21",
        "vpandq zmm19, zmm19, zmm20",
        "vmovdqa64 zmm20, zmmword ptr [r9 + 1728]",
        "vpxorq zmm21, zmm20, zmm19",
        "vmovdqa64 zmmword ptr [r9 + 1728], zmm21",
        "vpandq zmm19, zmm19, zmm20",
        "vmovdqa64 zmm20, zmmword ptr [r9 + 1792]",
        "vpxorq zmm21, zmm20, zmm19",
        "vmovdqa64 zmmword ptr [r9 + 1792], zmm21",
        "vpandq zmm19, zmm19, zmm20",
        "vpxorq zmm19, zmm19, zmmword ptr [r9 + 1856]",
        "vmovdqa64 zmmword ptr [r9 + 1856], zmm19",
        "vpxorq zmm12, zmm12, zmm12",
        "vpxorq zmm13, zmm13, zmm13",
        "vpxorq zmm14, zmm14, zmm14",
        "vpxorq zmm15, zmm15, zmm15",
        "vpxorq zmm16, zmm16, zmm16",
        "vpxorq zmm17, zmm17, zmm17",
        "vpxorq zmm18, zmm18, zmm18",
        "vpxorq zmm19, zmm19, zmm19",
        "xor r12d, r12d",
        "jmp r13",
        "92:",
        in("rdi") planes,
        inout("rsi") pos.as_ptr() => _,
        inout("rdx") pos.len() => _,
        in("rcx") neg.as_ptr(),
        in("r8") neg.len(),
        in("r9") acc,
        out("rax") _, out("r10") _, out("r12") _, out("r13") _,
        out("zmm0") _,
        out("zmm1") _,
        out("zmm2") _,
        out("zmm3") _,
        out("zmm4") _,
        out("zmm5") _,
        out("zmm6") _,
        out("zmm7") _,
        out("zmm8") _,
        out("zmm9") _,
        out("zmm10") _,
        out("zmm11") _,
        out("zmm12") _,
        out("zmm13") _,
        out("zmm14") _,
        out("zmm15") _,
        out("zmm16") _,
        out("zmm17") _,
        out("zmm18") _,
        out("zmm19") _,
        out("zmm20") _,
        out("zmm21") _,
        out("zmm22") _,
        out("zmm23") _,
        out("zmm24") _,
        out("zmm25") _,
        out("zmm26") _,
        out("zmm27") _,
        out("zmm28") _,
        out("zmm29") _,
        out("zmm30") _,
        out("zmm31") _,
        options(nostack),
    );
}

}

mod native_pair_check {
use super::carry4_kernel as carry4;
use super::bitsliced_unpack as baseline;
use core::arch::x86_64::__m512i;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C, align(64))]
struct Plane([u64; 8]);

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

fn check_case(p: usize, n: usize, fill: u8, initial: u8, rng: &mut Rng) {
    assert_eq!((p | n) & 1, 0);
    assert!(p + n <= 32767);
    let mut planes = vec![Plane([0; 8]); 130];
    for plane in &mut planes[2..] {
        for word in &mut plane.0 {
            *word = match fill { 0 => 0, 1 => !0, _ => rng.next() };
        }
    }
    let mut terms = |len: usize| -> Vec<u16> {
        (0..len).map(|i| if i % 13 == 5 { 0 } else { 8 * (2 + rng.next() % 127) as u16 }).collect()
    };
    let pos = terms(p);
    let neg = terms(n);
    let mut got = vec![Plane([0; 8]); 30];
    let max = 32767 - p - n;
    for slot in 0..2 {
        for lane in 0..512 {
            let count = match initial {
                0 => 0,
                1 => max,
                _ => rng.next() as usize % (max + 1),
            };
            for bit in 0..15 {
                got[slot * 15 + bit].0[lane / 64] |= ((count >> bit & 1) as u64) << (lane % 64);
            }
        }
    }
    let mut old = got.clone();
    let mut want = got.clone();
    // Each reference step adds one input directly to a 15-plane counter.
    for slot in 0..2 {
        for (list, complement) in [(&pos, 0), (&neg, u64::MAX)] {
            for &term in list {
                for word in 0..8 {
                    let mut carry = planes[term as usize / 8 + slot].0[word] ^ complement;
                    for bit in 0..15 {
                        let at = slot * 15 + bit;
                        let value = want[at].0[word];
                        want[at].0[word] ^= carry;
                        carry &= value;
                    }
                    assert_eq!(carry, 0);
                }
            }
        }
    }
    unsafe {
        baseline::accumulate_pair(
            planes.as_ptr().cast::<__m512i>(), &pos, &neg, old.as_mut_ptr().cast::<__m512i>());
        carry4::accumulate_pair_carry4_sol(
            planes.as_ptr().cast::<__m512i>(), &pos, &neg, got.as_mut_ptr().cast::<__m512i>());
    }
    assert_eq!(old, want, "baseline p={p} n={n} fill={fill} initial={initial}");
    assert_eq!(got, want, "carry4 p={p} n={n} fill={fill} initial={initial}");
}

pub(super) fn run() {
    assert!(std::is_x86_feature_detected!("avx512f"), "native AVX-512 is required");
    let mut rng = Rng(0x5c7a9f002031);
    let mut cases = 0usize;
    for p in (0..64).step_by(2) {
        for n in (0..64).step_by(2) {
            check_case(p, n, 2, (cases % 3) as u8, &mut rng);
            cases += 1;
        }
    }
    let edges = [
        (0, 0), (0, 64), (64, 0), (448, 0), (450, 0), (896, 0),
        (2, 894), (450, 448), (2, 896), (64, 832), (2, 1024),
        (514, 512), (960, 2), (1024, 0), (0, 1024),
        (16000, 0), (0, 16000), (16000, 16766), (32766, 0),
    ];
    for (p, n) in edges {
        for fill in 0..3 {
            for initial in 0..3 {
                check_case(p, n, fill, initial, &mut rng);
                cases += 1;
            }
        }
    }
    for _ in 0..600 {
        let p = 2 * (rng.next() as usize % 514);
        let n = 2 * (rng.next() as usize % ((1026 - p) / 2 + 1));
        check_case(p, n, 2, 2, &mut rng);
        cases += 1;
    }
    println!("cargo:warning=kernel-diag native_pair_cases={cases} lanes_per_case=1024 baseline_equal=true scalar_equal=true");
}

}
