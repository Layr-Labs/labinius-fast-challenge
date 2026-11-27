//! Diagnostic driver with the leader's scratch, sink and reconstruction.
use super::*;

#[inline(always)]
unsafe fn candidate_transform<const V: u8, const PAIRED: bool>(
    nib: &[uint8x16_t; N / 4], sc: &mut [int16x8_t; N], x: &mut [int16x8_t; N], kern: &Kernel,
) {
    if V == 16 {
        super::super::signed::transform(nib, sc, x, kern);
        return;
    }

    #[cfg(target_feature = "sve")]
    if V == 15 {
        super::super::weighted_pair::pass_lut_weighted_pair(nib, sc);
        super::tail::<4>(sc, x, kern);
        return;
    }

    if PAIRED {
        lut_pass2(nib, sc, kern);
    } else {
        lut_pass(nib, sc, kern);
    }
    match V {
        19 => super::super::pass_tail_calls_tuned(sc, x, kern),
        20 => super::super::tail_both(sc, x, kern),
        _ => super::tail::<4>(sc, x, kern),
    }
}

#[inline(never)]
unsafe fn sums<const V: u8>(
    words: &[(*const u8, usize); LANES],
    a: &SmsMatrix,
    m: &mut [int16x8_t; BITS],
    nib: &mut [uint8x16_t; N / 4],
    xs: &mut [[int16x8_t; N]; SWEEP],
    sc: &mut [int16x8_t; N],
    acc: &mut [int64x2_t],
    y: &mut [int16x8_t; N],
    dev: &mut [int32x4_t],
) {
    let kern = &kernels()[PB];
    for s in 0..N {
        acc[idx(s, 0)..][..4].fill(vdupq_n_s64(0));
    }
    let mut e = 0usize;
    while e < a.elements {
        let end = ((e / CHUNK + 1) * CHUNK).min(a.elements);
        while e < end {
            // Up to eight tiles per sweep, never across a chunk end (chunks are whole tiles). A short
            // tile only occurs at the end of the column, where the missing elements read zero
            // words and their key lanes are zero, so those lanes contribute nothing.
            let l = SWEEP.min(end - e);
            let groups = l.div_ceil(LOCK);
            for i in 0..groups * LOCK {
                let at = pass_words(words, a.elements, e, i);
                indices_asm(&at, nib);
                let x = &mut xs[i];
                candidate_transform::<V, false>(nib, sc, x, kern);
            }
            let keys = a.data.as_ptr().add(e / LOCK * N * ROWS * LOCK);
            // the chunk's first sweep starts the digit rows' sums afresh
            if e % CHUNK == 0 {
                sink::<true>(xs, keys, acc, groups);
            } else {
                sink::<false>(xs, keys, acc, groups);
            }
            e += l;
        }
        let c = (end - 1) / CHUNK;
        const _: () = assert!(DIGITS % 2 == 0);
        let dv = dev.as_mut_ptr();
        for d in (0..DIGITS).step_by(2) {
            let mr = [a.mean_residues(c, d), a.mean_residues(c, d + 1)];
            let rows = [dv.add(d * 2 * N), dv.add((d + 1) * 2 * N)];
            chunk_row2(acc, 1 + d, mr, rows, [&mut *y, &mut xs[0]], &Repair { words, a, c });
        }
    }
    narrow(acc, 0, y);
    super::inverse_folded::<PB>(y);
}

/// [`sums`] for two batches against one sweep of the key (after IvanLudvig's submission
/// ce686fcc): per sweep, batch A's transforms into `xs` and their sink, then batch B's into the
/// same `xs` and their sink against the same key lines, which are then still cached.
#[inline(never)]
unsafe fn sums2<const V: u8>(
    words: [&[(*const u8, usize); LANES]; 2],
    a: &SmsMatrix,
    nib: &mut [uint8x16_t; N / 4],
    xs: &mut [[int16x8_t; N]; SWEEP],
    sc: &mut [int16x8_t; N],
    acc: [&mut [int64x2_t]; 2],
    y: [&mut [int16x8_t; N]; 2],
    dev: [&mut [int32x4_t]; 2],
    dev16: [&mut [int16x8_t]; 2],
) {
    let kern = &kernels()[PB];
    let mut dev16 = dev16;
    for sums in dev16.iter_mut() {
        sums.fill(vdupq_n_s16(0));
    }
    let [acc0, acc1] = acc;
    let [y0, y1] = y;
    let [dev0, dev1] = dev;
    let accs: [&mut [int64x2_t]; 2] = [acc0, acc1];
    let ys: [&mut [int16x8_t; N]; 2] = [y0, y1];
    let devs: [&mut [int32x4_t]; 2] = [dev0, dev1];
    let (mut accs, mut ys, mut devs) = (accs, ys, devs);
    for t in 0..2 {
        for s in 0..N {
            accs[t][idx(s, 0)..][..4].fill(vdupq_n_s64(0));
        }
    }
    let mut e = 0usize;
    while e < a.elements {
        let end = ((e / CHUNK + 1) * CHUNK).min(a.elements);
        while e < end {
            let l = SWEEP.min(end - e);
            let groups = l.div_ceil(LOCK);
            let keys = a.data.as_ptr().add(e / LOCK * N * ROWS * LOCK);
            for t in 0..2 {
                let count = groups * LOCK;
                for i in 0..count {
                    // The input words [`PF_T`] transforms on: later in this sweep, or the start of
                    // the sweep that follows it (the other batch's, over the same elements for
                    // the first batch and over the next ones for the second). A counter profile
                    // on Neoverse-V2 had 30% of the table's lines arriving late (L2 refills in
                    // the index packing, 41% of its cycles stalled).
                    if i + PF_L1 < count {
                        prefetch_words_l1(words[t], e, i + PF_L1);
                    }
                    let ahead = i + PF_T;
                    if ahead < count {
                        prefetch_words(words[t], e, ahead);
                    } else if t == 0 {
                        prefetch_words(words[1], e, ahead - count);
                    } else {
                        prefetch_words(words[0], e + l, ahead - count);
                    }
                    let at = pass_words(words[t], a.elements, e, i);
                    indices_asm(&at, nib);
                    let x = &mut xs[i];
                    candidate_transform::<V, true>(nib, sc, x, kern);
                }
                if e % CHUNK == 0 {
                    sink::<true>(xs, keys, accs[t], groups);
                } else {
                    sink::<false>(xs, keys, accs[t], groups);
                }
            }
            e += l;
        }
        let c = (end - 1) / CHUNK;
        for t in 0..2 {
            let dv = devs[t].as_mut_ptr();
            let dv16 = dev16[t].as_mut_ptr();
            for d in (0..DIGITS).step_by(2) {
                let mr = [a.mean_residues(c, d), a.mean_residues(c, d + 1)];
                let rows = [dv.add(d * 2 * N), dv.add((d + 1) * 2 * N)];
                let rows16 = [dv16.add(d * N), dv16.add((d + 1) * N)];
                chunk_row2_fast(accs[t], 1 + d, mr, rows, rows16, [&mut *ys[t], &mut xs[0]], &Repair { words: words[t], a, c });
            }
            if (c + 1) % FLUSH == 0 || end == a.elements {
                flush16(devs[t], dev16[t]);
            }
        }
    }
    for t in 0..2 {
        narrow(accs[t], 0, ys[t]);
        super::inverse_folded::<PB>(ys[t]);
    }
    let _ = (&mut accs, &mut ys, &mut devs);
}

pub fn commit_candidate<const V: u8>(values: &[B128], a: &SmsMatrix, workers: usize, q: u128, out: &mut [u8], hook: bool) {
    let len = 4 * a.elements;
    assert_eq!(values.len() % len, 0);
    let columns = values.len() / len;
    // `MODULUS < 2^40`: five wire bytes per coefficient, a constant so the stores are fixed-size.
    const W: usize = 5;
    const _: () = assert!(MODULUS < 1 << (8 * W) && MODULUS >= 1 << (8 * (W - 1)));
    assert_eq!(crate::rq::coefficient_bytes(q), W);
    let width = W;
    assert_eq!(out.len(), columns * N * width);
    for_blocks(out, N * width, LANES, workers, |first, block| unsafe {
        pin_worker(first * workers / columns);
        let arena = Arena::new();
        let Scratch { xs, sc, acc, dev, acc2, dev2, dev16, y2, m, nib, y, tail } = &mut *arena.0;
        let p = P1 as i16;
        let count = block.len() / (N * width);
        let batch_words = |b: usize| -> [(*const u8, usize); LANES] {
            let col0 = first + LANES * b;
            let here = (count - LANES * b).min(LANES);
            core::array::from_fn(|c| {
                if c < here {
                    (values[(col0 + c) * len..].as_ptr() as *const u8, 64)
                } else {
                    (ZERO_WORDS.as_ptr() as *const u8, 0)
                }
            })
        };
        let batches = count.div_ceil(LANES);
        let mut targets: Vec<&mut [u8]> = block.chunks_mut(LANES * N * width).collect();
        for b in 0..batches {
            // pairs of batches share one key sweep; the second one's results wait in `*2`
            if b % 2 == 0 {
                dev.fill(vdupq_n_s32(0));
                if b + 1 < batches {
                    dev2.fill(vdupq_n_s32(0));
                    let (w0, w1) = (batch_words(b), batch_words(b + 1));
                    let [d16a, d16b] = dev16;
                    sums2::<V>([&w0, &w1], a, nib, xs, sc, [&mut acc[..], &mut acc2[..]], [&mut *y, &mut *y2], [&mut dev[..], &mut dev2[..]], [&mut d16a[..], &mut d16b[..]]);
                } else {
                    sums::<V>(&batch_words(b), a, m, nib, xs, sc, acc, y, dev);
                }
            } else {
                core::mem::swap(dev, dev2);
                core::mem::swap(y, y2);
            }
            let target = &mut *targets[b];
            // The base row's centered residues (`|x| <= (p - 1) / 2`, [`super::inverse`]) to
            // `[0, p)` lane-wise, eight columns per vector: `+ p` where negative, as `rem_euclid`.
            let pv = vdupq_n_s16(p);
            let mut base = [[0u16; LANES]; N];
            for (k, b) in base.iter_mut().enumerate() {
                let v = y[k];
                let v = vaddq_s16(v, vandq_s16(vreinterpretq_s16_u16(vcltzq_s16(v)), pv));
                vst1q_u16(b.as_mut_ptr(), vreinterpretq_u16_s16(v));
            }
            // `tail = mu + sum_d W_d dev_d` ([`add_weighted`]'s map) for the eight columns at
            let tail = &mut *tail;
            for k in 0..N / 2 {
                let h = k + N / 2;
                let mut lo = [vdupq_n_s64(a.mu[k] as i64); 4];
                let mut hi = [vdupq_n_s64(a.mu[h] as i64); 4];
                for d in 0..DIGITS {
                    let (u, v) = lattice::WEIGHTS[d];
                    let w = u + v;
                    for half in 0..2 {
                        let x = dev[d * 2 * N + 2 * k + half];
                        let y = dev[d * 2 * N + 2 * h + half];
                        let (xl, yl) = (vget_low_s32(x), vget_low_s32(y));
                        let (i, j) = (2 * half, 2 * half + 1);
                        lo[i] = vmlal_n_s32(lo[i], xl, u);
                        lo[j] = vmlal_high_n_s32(lo[j], x, u);
                        lo[i] = vmlsl_n_s32(lo[i], yl, v);
                        lo[j] = vmlsl_high_n_s32(lo[j], y, v);
                        hi[i] = vmlal_n_s32(hi[i], xl, v);
                        hi[j] = vmlal_high_n_s32(hi[j], x, v);
                        hi[i] = vmlal_n_s32(hi[i], yl, w);
                        hi[j] = vmlal_high_n_s32(hi[j], y, w);
                    }
                }
                for q in 0..4 {
                    vst1q_s64(tail[k].as_mut_ptr().add(2 * q), lo[q]);
                    vst1q_s64(tail[h].as_mut_ptr().add(2 * q), hi[q]);
                }
            }
            // a partial batch has fewer than eight columns in `target`: only those lanes are written
            for (c, column) in target.chunks_exact_mut(N * width).enumerate() {
                let (slots, _) = column.as_chunks_mut::<W>();
                for (k, slot) in slots.iter_mut().enumerate() {
                    let v = crt(base[k][c] as u32, tail[k][c]);
                    slot[..4].copy_from_slice(&(v as u32).to_le_bytes());
                    slot[4] = (v >> 32) as u8;
                }
            }
            if b % 2 == 1 && hook {
                if let Some(between) = BETWEEN.get() {
                    between(first * workers / columns, (b + 1) / 2);
                }
            }
        }
    });
}

