// Re-pair Y1/X2 and X1/Y2 so lookup tables absorb the first radix-3 roots.
// Each pair is a balanced triple (P0,P1,-P0-P1), with |entry| <=6436.
// The public key is unchanged. Prefix normalization gives a proved tail bound31073.

const fn r3pair_balance(a: i32, b: i32) -> (i16, i16) {
    let a = ((a % 9721) + 14581) % 9721 - 4860;
    let b = ((b % 9721) + 14581) % 9721 - 4860;
    let mut best = (0, 0);
    let mut bm = i32::MAX;
    let mut bn = i64::MAX;
    let mut i = -1;
    while i <= 1 {
        let mut j = -1;
        while j <= 1 {
            let x = a + i * 9721;
            let y = b + j * 9721;
            let z = -x - y;
            let mut m = x.abs();
            if y.abs() > m {
                m = y.abs();
            }
            if z.abs() > m {
                m = z.abs();
            }
            let n = x as i64 * x as i64 + y as i64 * y as i64 + z as i64 * z as i64;
            if m < bm || (m == bm && n < bn) {
                bm = m;
                bn = n;
                best = (x as i16, y as i16);
            }
            j += 1;
        }
        i += 1;
    }
    best
}
const fn sms_r3pair_lut() -> SmsJointPairLut {
    let old = sms_band_lut();
    let mut out = [[[[[0i16; 64]; 2]; 2]; 4]; 3];
    let mut band = 0;
    while band < 3 {
        let mut k = 0;
        while k < 4 {
            let mut rr = 0;
            while rr < 2 {
                let mut n = 0;
                while n < 64 {
                    let x = (n & 1) | ((n >> 1) & 2) | ((n >> 2) & 4);
                    let y = ((n >> 1) & 1) | ((n >> 2) & 2) | ((n >> 3) & 4);
                    let tx = 12 * k + if rr == 0 { 3 } else { 2 };
                    let ty = 12 * k + if rr == 0 { 4 } else { 5 };
                    let vx = (old.0[band][tx][x] as u16 | ((old.0[band][tx][16 + x] as u16) << 8))
                        as i16 as i32;
                    let vy = (old.0[band][ty][y] as u16 | ((old.0[band][ty][16 + y] as u16) << 8))
                        as i16 as i32;
                    let (a, b) = r3pair_balance(vx + vy, 7796 * vx + 1924 * vy);
                    out[band][k][rr][0][n] = a;
                    out[band][k][rr][1][n] = b;
                    n += 1;
                }
                rr += 1;
            }
            k += 1;
        }
        band += 1;
    }
    SmsJointPairLut(out)
}
static SMS_R3PAIR_LUT: SmsJointPairLut = sms_r3pair_lut();
/// Per quarter, the band-0 rows whose `v2` (tail block 3) the prefix reduces.
const V2_BARRETT: [u32; 4] = [1 << 3, 0, 0, 0];
/// Whether the prefix Barrett-reduces tail block `block`'s row `row` of band `band` in quarter
/// `k` (blocks 1, 2, 5 are never reduced). The kernel and r3pair_tail_interval_proof both read
/// this, so the proof covers exactly the reductions the kernel performs.
#[inline(always)]
const fn prefix_reduces(block: usize, band: usize, k: usize, row: usize) -> bool {
    match block {
        0 | 4 => band != 0,
        3 => band == 0 && V2_BARRETT[k] >> row & 1 == 1,
        _ => false,
    }
}
#[inline(never)]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vnni,gfni")]
pub(crate) unsafe fn ntt_bin_batch32_r3pair_source<
    const Q: u16,
    const H: usize,
    const FUSED: bool,
    const SPARSE512: bool,
    S: SliceSink<H>,
    F: FnMut(usize, *mut crate::simd::transpose_f162::SmsR3PairIndex32),
>(
    idx: *mut crate::simd::transpose_f162::SmsR3PairIndex32,
    wbuf: *mut i16,
    sink: &mut S,
    fill: &mut F,
) {
    assert!(
        Q == 9721 && SPARSE512,
        "private joint producer is only the fixed packed-binary path"
    );
    if !FUSED {
        for h in 0..H {
            fill(h, idx.add(h));
        }
    }
    let t = tables::<Q>();
    let c = C {
        q: bc(&t.qd),
        om: bc(&t.om[0]),
        omp: bc(&t.om[1]),
    };
    let mut bv = bc(&t.bv);
    // Prefix blocks use the callback buffer's block-major layout. A leaf loads
    // all 27 input vectors before storing any output, so its pointers may alias.
    let buf = |kk: usize| -> [*mut i16; H] {
        core::array::from_fn(|h| wbuf.add(32 * 27 * (H * (kk % 6) + h)))
    };
    let ro = |w: [*mut i16; H]| -> [*const i16; H] { core::array::from_fn(|h| w[h] as *const i16) };

    for k in 0..4 {
        if FUSED && k == 0 {
            for h in 0..core::hint::black_box(H) {
                fill(h, idx.add(h));
            }
        }
        for band in 0..core::hint::black_box(3usize) {
            let lut = SMS_JOINT_R0_LUT
                .0
                .get_unchecked(band)
                .get_unchecked(k)
                .as_ptr() as *const u8;
            let l = |r: usize, ab: usize| {
                _mm512_load_si512(lut.add(64 * (r * 2 + ab)) as *const __m512i)
            };
            let (l000, l001) = (l(0, 0), l(0, 1));
            let pair_band = SMS_R3PAIR_LUT.0.get_unchecked(band).get_unchecked(k);
            let pair = |r: usize, s: usize, h: usize| {
                _mm512_load_si512(pair_band[r][s].as_ptr().add(32 * h).cast())
            };
            let (a10, a11, b10, b11) = (pair(0, 0, 0), pair(0, 0, 1), pair(0, 1, 0), pair(0, 1, 1));
            let (a20, a21, b20, b21) = (pair(1, 0, 0), pair(1, 0, 1), pair(1, 1, 0), pair(1, 1, 1));

            core::arch::asm!("/* {0} */", inout(zmm_reg) bv, options(pure, nomem, nostack, preserves_flags));
            for h in 0..core::hint::black_box(H) {
                let ip = core::ptr::addr_of!((*idx.add(h)).rows).cast::<u8>();
                let bp = wbuf.add(32 * 27 * h);
                for i in 9 * band..9 * band + 9 {
                    let nx = ldb(ip, i);
                    let ny = _mm512_srli_epi16::<4>(nx);
                    let n1 = ldb(ip, i + 27);
                    let n2 = ldb(ip, i + 54);
                    let x = _mm512_permutexvar_epi8(nx, l000);
                    let y = _mm512_permutexvar_epi8(ny, l001);
                    let a0 = _mm512_add_epi16(x, y);
                    let b0 = _mm512_sub_epi16(x, y);
                    let a1 = _mm512_permutex2var_epi16(a10, n1, a11);
                    let b1 = _mm512_permutex2var_epi16(b10, n1, b11);
                    let a2 = _mm512_permutex2var_epi16(a20, n2, a21);
                    let b2 = _mm512_permutex2var_epi16(b20, n2, b21);

                    let t0 = _mm512_add_epi16(a1, a2);
                    let t1 = _mm512_add_epi16(b1, b2);
                    let d0 = _mm512_sub_epi16(a1, a2);
                    let d1 = _mm512_sub_epi16(b1, b2);
                    let mut u0 = _mm512_add_epi16(a0, t0);
                    let u1 = _mm512_add_epi16(a0, t1);
                    let u2 = _mm512_sub_epi16(a0, _mm512_add_epi16(t0, t1));
                    let mut v0 = _mm512_add_epi16(b0, d0);
                    let v1 = _mm512_add_epi16(b0, d1);
                    let mut v2 = _mm512_sub_epi16(b0, _mm512_add_epi16(d0, d1));
                    // Of block 3's band-0 rows only row 3 of quarter 0 needs this reduction: the
                    // scaled leaf keeps every word and output under SMS_SOURCE_BOUND without the
                    // others (r3pair_tail_interval_proof).
                    if prefix_reduces(3, band, k, i - 9 * band) {
                        v2 = _mm512_sub_epi16(
                            v2,
                            _mm512_mullo_epi16(_mm512_mulhrs_epi16(v2, bv), c.q),
                        );
                    }
                    // Blocks 0 and 4 (unit-twiddle leaves) need reduced band-1/2 rows only.
                    if prefix_reduces(0, band, k, i - 9 * band) {
                        u0 = _mm512_sub_epi16(u0, _mm512_mullo_epi16(_mm512_mulhrs_epi16(u0, bv), c.q));
                    }
                    if prefix_reduces(4, band, k, i - 9 * band) {
                        v0 = _mm512_sub_epi16(v0, _mm512_mullo_epi16(_mm512_mulhrs_epi16(v0, bv), c.q));
                    }
                    st(bp, i, u0);
                    st(bp, i + 27 * H, u1);
                    st(bp, i + 54 * H, u2);
                    st(bp, i + 81 * H, v2);
                    st(bp, i + 108 * H, v0);
                    st(bp, i + 135 * H, v1);
                }
            }
        }
        let cv = t.cv.as_ptr() as *const i16;
        let tail_blocks = core::hint::black_box(6usize);
        for j in 0..tail_blocks {
            let kk = 6 * k + j;
            let (t4, t5, t6) = if Q == 9721 {
                (
                    SMS_BAND_T4.0.as_ptr().add(4 * kk).cast::<u32>(),
                    SMS_TWIDDLES.t5.as_ptr().add(12 * kk).cast::<u32>(),
                    SMS_TWIDDLES.t6.as_ptr().add(36 * kk).cast::<u32>(),
                )
            } else {
                (
                    t.tw4.as_ptr().add(kk).cast::<u32>(),
                    t.tw5.as_ptr().add(3 * kk).cast::<u32>(),
                    t.tw6.as_ptr().add(9 * kk).cast::<u32>(),
                )
            };
            for h in 0..core::hint::black_box(H) {
                let bpj = wbuf.add(32 * 27 * (H * j + h));
                if Q == 9721 {
                    if SPARSE512 {
                        if j == 0 {
                            tail27_sms_band0_leaf(bpj, bpj, t4, t5, t6, cv);
                        } else if j == 4 {
                            tail27_sms_band4_leaf(bpj, bpj, t4, t5, t6, cv);
                        } else {
                            tail27_scaled_leaf(bpj, bpj, t4, t5, t6, cv);
                        }
                    } else {
                        tail27_relocated_leaf(bpj, bpj, t4, t5, t6, cv);
                    }
                } else {
                    tail27_out_of_line::<Q>(bpj, bpj, t4, t5, t6, cv);
                }
            }
        }
        for j in 0..6 {
            let kk = 6 * k + j;
            sink.rows(kk, ro(buf(kk)), 0, 27);
        }
    }
}

#[cfg(test)]
mod sms_r3pair_source_guards {
    use super::*;
    use crate::fields::scalar::B128;
    use crate::simd::transpose_f162::{
        slice_b128_cols16_into, slice_b128_cols16_r3pair_into, SmsR3PairIndex32,
    };
    #[repr(C, align(64))]
    struct Out([i16; 648 * 32]);
    struct Collect<const H: usize> {
        output: Vec<Out>,
        next: usize,
    }
    impl<const H: usize> SliceSink<H> for Collect<H> {
        unsafe fn rows(&mut self, blk: usize, w: [*const i16; H], r0: usize, r1: usize) {
            assert_eq!(blk, self.next);
            self.next += 1;
            assert_eq!((r0, r1), (0, 27));
            for h in 0..H {
                for i in r0..r1 {
                    core::ptr::copy_nonoverlapping(
                        w[h].add(32 * i),
                        self.output[h].0.as_mut_ptr().add((27 * blk + i) * 32),
                        32,
                    );
                }
            }
        }
    }
    unsafe fn check<const H: usize, const FUSED: bool>() {
        let poison = _mm512_set1_epi32(0x1759acf3);
        let mut state = 0x531789fdd8712ac3u128;
        for fixture in 0..20 {
            let mut old: Vec<BinaryIndex32> = (0..H)
                .map(|_| BinaryIndex32 {
                    rows: [[0; 64]; 162],
                })
                .collect();
            let mut joint: Vec<SmsR3PairIndex32> = (0..H)
                .map(|_| SmsR3PairIndex32 {
                    rows: [[0; 32]; 81],
                })
                .collect();
            for h in 0..H {
                let mut columns = [[B128(0); 8]; 16];
                for lane in 0..32 {
                    let col = lane / 2;
                    let e = lane % 2;
                    if fixture < 16 {
                        let bit = fixture * 32 + lane;
                        columns[col][4 * e + bit % 4].0 = 1u128 << (bit / 4);
                    } else {
                        for word in 0..4 {
                            state ^= state << 13;
                            state ^= state >> 7;
                            state ^= state << 17;
                            columns[col][4 * e + word].0 = match fixture {
                                16 => 0,
                                17 => u128::MAX,
                                18 => {
                                    if (h + lane + word) & 1 == 0 {
                                        u128::MAX
                                    } else {
                                        0
                                    }
                                }
                                _ => state,
                            };
                        }
                    }
                }
                let ptrs = core::array::from_fn(|i| columns[i].as_ptr());
                slice_b128_cols16_into(&ptrs, &mut old[h]);
                slice_b128_cols16_r3pair_into(&ptrs, &mut joint[h]);
            }
            let mut ib: Vec<BinaryIndex32> = (0..H)
                .map(|_| BinaryIndex32 {
                    rows: [[0x75; 64]; 162],
                })
                .collect();
            let mut ij: Vec<SmsR3PairIndex32> = (0..H)
                .map(|_| SmsR3PairIndex32 {
                    rows: [[0x7531; 32]; 81],
                })
                .collect();
            let mut wb = vec![poison; 6 * H * 27 + 2];
            let mut wj = vec![poison; 6 * H * 27 + 2];
            let mut expected = Collect::<H> {
                output: (0..H).map(|_| Out([0; 648 * 32])).collect(),
                next: 0,
            };
            let mut got = Collect::<H> {
                output: (0..H).map(|_| Out([0; 648 * 32])).collect(),
                next: 0,
            };
            let mut bnext = 0;
            let mut jnext = 0;
            ntt_bin_batch32_batched_source::<9721, H, FUSED, true, _, _>(
                ib.as_mut_ptr(),
                wb.as_mut_ptr().add(1).cast(),
                &mut expected,
                &mut |h, out| {
                    assert_eq!(h, bnext);
                    bnext += 1;
                    core::ptr::copy_nonoverlapping(&old[h], out, 1);
                },
            );
            ntt_bin_batch32_r3pair_source::<9721, H, FUSED, true, _, _>(
                ij.as_mut_ptr(),
                wj.as_mut_ptr().add(1).cast(),
                &mut got,
                &mut |h, out| {
                    assert_eq!(h, jnext);
                    jnext += 1;
                    core::ptr::copy_nonoverlapping(&joint[h], out, 1);
                },
            );
            assert_eq!((bnext, jnext, expected.next, got.next), (H, H, 24, 24));
            for h in 0..H {
                for (i, (&a, &b)) in got.output[h]
                    .0
                    .iter()
                    .zip(&expected.output[h].0)
                    .enumerate()
                {
                    assert!(a.abs() <= 31296, "bound {a}");
                    assert_eq!(
                        (a as i32 - b as i32).rem_euclid(9721),
                        0,
                        "fixture={fixture} H={H} fused={FUSED} h={h} i={i}"
                    );
                }
            }
            for w in [&wb, &wj] {
                assert_eq!(_mm512_cmpeq_epi32_mask(w[0], poison), u16::MAX);
                assert_eq!(_mm512_cmpeq_epi32_mask(w[w.len() - 1], poison), u16::MAX);
            }
        }
    }
    #[test]
    fn sms_r3pair_source_all512basis_residues_bound() {
        unsafe {
            check::<1, true>();
            check::<2, true>();
            check::<3, true>();
            check::<4, true>();
            check::<8, true>();
            check::<16, true>();
            check::<1, false>();
            check::<2, false>();
            check::<3, false>();
            check::<4, false>();
            check::<8, false>();
            check::<16, false>();
        }
    }
}

#[cfg(test)]
mod sms_r3pair_lut_guards {
    use super::*;
    #[test]
    fn all_public_pair_entries_match_field_identities_and_bounds() {
        let old = sms_band_lut();
        assert_eq!((7796i64 * 7796 + 7796 + 1) % 9721, 0);
        for band in 0..3 {
            for k in 0..4 {
                for rr in 0..2 {
                    for n in 0..64 {
                        let x = (n & 1) | ((n >> 1) & 2) | ((n >> 2) & 4);
                        let y = ((n >> 1) & 1) | ((n >> 2) & 2) | ((n >> 3) & 4);
                        let tx = 12 * k + if rr == 0 { 3 } else { 2 };
                        let ty = 12 * k + if rr == 0 { 4 } else { 5 };
                        let read = |t: usize, i: usize| {
                            (old.0[band][t][i] as u16 | ((old.0[band][t][16 + i] as u16) << 8))
                                as i16 as i32
                        };
                        let vx = read(tx, x);
                        let vy = read(ty, y);
                        let a = SMS_R3PAIR_LUT.0[band][k][rr][0][n] as i32;
                        let b = SMS_R3PAIR_LUT.0[band][k][rr][1][n] as i32;
                        assert_eq!((a - vx - vy).rem_euclid(9721), 0);
                        assert_eq!((b - 7796 * vx - 1924 * vy).rem_euclid(9721), 0);
                        assert!(a.abs() <= 6436 && b.abs() <= 6436 && (-a - b).abs() <= 6436);
                    }
                }
            }
        }
    }
}

/// Interval proof of the prefix's reductions: the exact range of every prefix output per block,
/// band and quarter (from every index word the slicer writes) is pushed through the tail leaf's
/// own asm text with per-lane interval semantics. All 24 leaves keep every word in i16 and every
/// output within SMS_SOURCE_BOUND with the reductions the kernel performs (u0/v0 on bands 1-2,
/// v2 on the [`V2_BARRETT`] rows) as well as with the old prefix's.
#[cfg(test)]
mod r3pair_tail_interval_proof {
    use super::*;
    use crate::fields::scalar::B128;
    use crate::simd::transpose_f162::{slice_b128_cols16_r3pair_into, SmsR3PairIndex32};
    use std::collections::{BTreeSet, HashMap};

    const TAIL: &str = tail27_scaled!(lut, plain);
    type V = [(i64, i64); 32];

    fn words_seen() -> [BTreeSet<u16>; 3] {
        let mut sets: [BTreeSet<u16>; 3] = Default::default();
        let mut s = 0x9e3779b97f4a7c15u64;
        let mut rnd = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        for _ in 0..4000 {
            let mut columns = [[B128(0); 8]; 16];
            for c in columns.iter_mut() {
                for w in c.iter_mut() {
                    *w = B128(((rnd() as u128) << 64) | rnd() as u128);
                }
            }
            let ptrs = core::array::from_fn(|i| columns[i].as_ptr());
            let mut out = SmsR3PairIndex32 { rows: [[0; 32]; 81] };
            unsafe { slice_b128_cols16_r3pair_into(&ptrs, &mut out) };
            for r in 0..81 {
                for &w in &out.rows[r] {
                    sets[r / 27].insert(w);
                }
            }
        }
        sets
    }

    fn barrett(v: i64, bv: i64, q: i64) -> i64 {
        let m = ((v * bv + 0x4000) >> 15) as i16 as i64;
        let p = (m * q) as i16 as i64;
        v - p
    }

    fn prefix_raw_red(band: usize, k: usize, sets: &[BTreeSet<u16>; 3]) -> ([(i64, i64); 6], [(i64, i64); 6]) {
        let t = tables::<9721>();
        let bv = t.bv as u16 as i16 as i64;
        let r0 = &SMS_JOINT_R0_LUT.0[band][k];
        // vpermt2w reads the low 6 bits of each index word.
        let pl = &SMS_R3PAIR_LUT.0[band][k];
        let word = |lo: &[u8; 64], hi: &[u8; 64], w: u16| -> i64 {
            (lo[(w & 63) as usize] as u16 | ((hi[((w >> 8) & 63) as usize] as u16) << 8)) as i16 as i64
        };
        let g0: Vec<(i64, i64)> = sets[0].iter().map(|&w| { let x = word(&r0[0], &r0[0], w); let y = word(&r0[1], &r0[1], w >> 4); (x + y, x - y) }).collect();
        let pair = |r: usize, w: u16| (pl[r][0][(w & 63) as usize] as i64, pl[r][1][(w & 63) as usize] as i64);
        let g1: Vec<(i64, i64)> = sets[1].iter().map(|&w| pair(0, w)).collect();
        let g2: Vec<(i64, i64)> = sets[2].iter().map(|&w| pair(1, w)).collect();
        let (mut lo, mut hi, mut rlo, mut rhi) = ([i64::MAX; 6], [i64::MIN; 6], [i64::MAX; 6], [i64::MIN; 6]);
        for &(a0, b0) in &g0 { for &(a1, b1) in &g1 { for &(a2, b2) in &g2 {
            let (t0, t1, d0, d1) = (a1 + a2, b1 + b2, a1 - a2, b1 - b2);
            let out = [a0 + t0, a0 + t1, a0 - (t0 + t1), b0 - (d0 + d1), b0 + d0, b0 + d1];
            for v in [a0, b0, t0, t1, d0, d1, t0 + t1, d0 + d1].into_iter().chain(out) {
                assert!(v.abs() < 32768, "prefix word overflow");
            }
            for j in 0..6 {
                let r = barrett(out[j], bv, 9721);
                lo[j] = lo[j].min(out[j]); hi[j] = hi[j].max(out[j]);
                rlo[j] = rlo[j].min(r); rhi[j] = rhi[j].max(r);
            }
        }}}
        (core::array::from_fn(|j| (lo[j], hi[j])), core::array::from_fn(|j| (rlo[j], rhi[j])))
    }

    fn eval(expr: &str) -> usize {
        expr.split('+').filter(|s| !s.is_empty()).map(|t| t.split('*').map(|f| f.trim().parse::<usize>().unwrap()).product::<usize>()).sum()
    }

    /// The leaf on interval inputs: returns (largest |intermediate|, largest |output|).
    fn run_leaf(text: &str, rows: &[(i64, i64); 27], ptrs: &HashMap<&str, *const u8>) -> (i64, i64) {
        let mut regs: HashMap<String, V> = HashMap::new();
        let mut worst = 0i64;
        let mut out_worst = 0i64;
        let lines: Vec<&str> = text.lines().map(str::trim).filter(|l| !l.is_empty()).collect();
        let mem = |op: &str| -> Option<V> {
            let op = op.trim_start_matches("zmmword ptr ").trim();
            let inner = op.strip_prefix('[')?.strip_suffix(']')?;
            let (base, rest) = inner.split_once('}').unwrap();
            let base = base.trim_start_matches('{');
            let off = eval(rest);
            if base == "i" {
                assert_eq!(off % 64, 0);
                return Some([rows[off / 64]; 32]);
            }
            let p = unsafe { ptrs[base].add(off) as *const i16 };
            Some(core::array::from_fn(|l| { let v = unsafe { *p.add(l) } as i64; (v, v) }))
        };
        regs.insert("zmm27".to_string(), mem("[{c}+0]").unwrap());
        let mut i = 0;
        while i < lines.len() {
            let l = lines[i];
            let (mn, ops) = l.split_once(' ').unwrap();
            let ops: Vec<&str> = ops.split(", ").map(str::trim).collect();
            let src = |regs: &HashMap<String, V>, o: &str| -> V { if o.starts_with("zmm") && !o.starts_with("zmmword") { regs[o] } else { mem(o).unwrap() } };
            match mn {
                "vmovdqa64" => {
                    if ops[0].starts_with('[') {
                        let v = regs[ops[1]];
                        for &(a, b) in &v { out_worst = out_worst.max(a.abs()).max(b.abs()); }
                    } else {
                        let v = mem(ops[1]).unwrap();
                        regs.insert(ops[0].to_string(), v);
                    }
                }
                "vpaddw" | "vpsubw" => {
                    let (a, b) = (src(&regs, ops[1]), src(&regs, ops[2]));
                    let r: V = core::array::from_fn(|l| if mn == "vpaddw" { (a[l].0 + b[l].0, a[l].1 + b[l].1) } else { (a[l].0 - b[l].1, a[l].1 - b[l].0) });
                    for &(x, y) in &r { worst = worst.max(x.abs()).max(y.abs()); }
                    regs.insert(ops[0].to_string(), r);
                }
                "vpmullw" => {
                    let (a, b) = (src(&regs, ops[1]), src(&regs, ops[2]));
                    let r: V = core::array::from_fn(|l| if a[l].0 == a[l].1 && b[l].0 == b[l].1 { let v = (a[l].0 * b[l].0) as i16 as i64; (v, v) } else { (-32768, 32767) });
                    regs.insert(ops[0].to_string(), r);
                }
                "vpmulhw" => {
                    let (a, b) = (src(&regs, ops[1]), src(&regs, ops[2]));
                    let r: V = core::array::from_fn(|l| {
                        let c = [a[l].0 * b[l].0, a[l].0 * b[l].1, a[l].1 * b[l].0, a[l].1 * b[l].1];
                        (c.iter().min().unwrap() >> 16, c.iter().max().unwrap() >> 16)
                    });
                    regs.insert(ops[0].to_string(), r);
                }
                "vpxord" => {
                    assert!(ops[1] == ops[2]);
                    regs.insert(ops[0].to_string(), [(0, 0); 32]);
                }
                "vpmulhrsw" => {
                    let (a, b) = (src(&regs, ops[1]), src(&regs, ops[2]));
                    let r: V = core::array::from_fn(|l| {
                        let c = [a[l].0 * b[l].0, a[l].0 * b[l].1, a[l].1 * b[l].0, a[l].1 * b[l].1];
                        ((c.iter().min().unwrap() + 0x4000) >> 15, (c.iter().max().unwrap() + 0x4000) >> 15)
                    });
                    regs.insert(ops[0].to_string(), r);
                }
                "vpsrlw" => {
                    // Fused lookup reduction: vpsrlw t, a, 11; vpermw t, t, [c+512]; vpaddw a, a, t.
                    let (t, a) = (ops[0], ops[1]);
                    assert_eq!(ops[2], "11");
                    let perm = lines[i + 1];
                    let add = lines[i + 2];
                    assert!(perm.starts_with(&format!("vpermw {t}, {t}, ")), "{perm}");
                    assert_eq!(add, format!("vpaddw {a}, {a}, {t}"));
                    let table = mem(perm.rsplit(", ").next().unwrap()).unwrap();
                    let av = regs[a];
                    let r: V = core::array::from_fn(|l| {
                        let (mut lo, mut hi) = (i64::MAX, i64::MIN);
                        for v in av[l].0..=av[l].1 {
                            assert!((-32768..32768).contains(&v));
                            let w = ((v as i16 as u16) >> 11) as usize & 31;
                            let r = v + table[w].0;
                            lo = lo.min(r);
                            hi = hi.max(r);
                        }
                        (lo, hi)
                    });
                    for &(x, y) in &r { worst = worst.max(x.abs()).max(y.abs()); }
                    regs.insert(a.to_string(), r);
                    i += 3;
                    continue;
                }
                other => panic!("unsupported {other}: {l}"),
            }
            i += 1;
        }
        (worst, out_worst)
    }

    #[test]
    fn prefix_reductions_keep_every_tail_word_bounded() {
        let sets = words_seen();
        // Complete by construction: R0 tables of 16 and 8 entries (16 x 8 nx encodings), 6-bit n1/n2.
        assert_eq!((sets[0].len(), sets[1].len(), sets[2].len()), (128, 64, 64), "index encodings");
        let t = tables::<9721>();
        // Per block j and band b: is that band's row r reduced? (old prefix, kernel)
        let old = |j: usize, b: usize, _k: usize, _r: usize| j == 0 || j == 4 || (j == 3 && b == 0);
        let new = |j: usize, b: usize, k: usize, r: usize| prefix_reduces(j, b, k, r);
        let configs: [(&str, &dyn Fn(usize, usize, usize, usize) -> bool); 2] = [("old", &old), ("kernel", &new)];
        for (name, reduced) in configs {
            let mut worst = (0, 0);
            for k in 0..4 {
                let rr: Vec<_> = (0..3).map(|b| prefix_raw_red(b, k, &sets)).collect();
                for j in 0..6 {
                    let kk = 6 * k + j;
                    let mut ptrs: HashMap<&str, *const u8> = HashMap::new();
                    ptrs.insert("t4", SMS_BAND_T4.0.as_ptr().wrapping_add(4 * kk).cast());
                    ptrs.insert("t5", SMS_TWIDDLES.t5.as_ptr().wrapping_add(12 * kk).cast());
                    ptrs.insert("t6", SMS_TWIDDLES.t6.as_ptr().wrapping_add(36 * kk).cast());
                    ptrs.insert("c", t.cv.as_ptr().cast());
                    let rows: [(i64, i64); 27] = core::array::from_fn(|r| {
                        let b = r / 9;
                        if reduced(j, b, k, r % 9) { rr[b].1[j] } else { rr[b].0[j] }
                    });
                    let text = match j { 0 => tail27_sms_band0!(), 4 => tail27_sms_band4!(), _ => TAIL };
                    let (w, o) = run_leaf(text, &rows, &ptrs);
                    assert!(w <= 32767 && o <= SMS_SOURCE_BOUND, "{name} k={k} j={j}: ({w}, {o})");
                    worst = (worst.0.max(w), worst.1.max(o));
                }
            }
            eprintln!("{name}: worst word {} worst output {}", worst.0, worst.1);
        }
    }
}
