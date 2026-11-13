#[cfg(test)]
mod phi24_source_bounds16752_proof {
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
    fn run_leaf(text: &str, rows: &[(i64, i64); 27], ptrs: &HashMap<&str, *const u8>) -> (i64, i64, [(i64,i64);27]) {
        let mut output=[(i64::MAX,i64::MIN);27];let mut seen=[false;27];
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
        regs.insert("zmm31".to_string(), mem("[{c}+128]").unwrap());
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
                        let offset=ops[0].strip_prefix("[{o}").unwrap().strip_suffix(']').unwrap();let offset=eval(offset);assert_eq!(offset%64,0);let row=offset/64;assert!(row<27);seen[row]=true;for &(a,b)in &v{out_worst=out_worst.max(a.abs()).max(b.abs());output[row].0=output[row].0.min(a);output[row].1=output[row].1.max(b);}
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
        assert!(seen.iter().all(|x|*x));
        (worst, out_worst,output)
    }


    #[test]fn phi24_proved_source_intervals_match_constants(){let sets=words_seen();assert_eq!((sets[0].len(),sets[1].len(),sets[2].len()),(128,64,64));let t=tables::<9721>();let mut all=Vec::new();let mut peak=0;for k in 0..4{let rr:Vec<_>=(0..3).map(|b|prefix_raw_red(b,k,&sets)).collect();for j in 0..6{let kk=6*k+j;let mut ptrs:HashMap<&str,*const u8>=HashMap::new();ptrs.insert("t4",SMS_BAND_T4.0.as_ptr().wrapping_add(4*kk).cast());ptrs.insert("t5",SMS_TWIDDLES.t5.as_ptr().wrapping_add(12*kk).cast());ptrs.insert("t6",SMS_TWIDDLES.t6.as_ptr().wrapping_add(36*kk).cast());ptrs.insert("c",t.cv.as_ptr().cast());let rows:[(i64,i64);27]=core::array::from_fn(|r|{let b=r/9;if prefix_reduces(j,b,k,r%9){rr[b].1[j]}else{rr[b].0[j]}});let text=match j{0=>tail27_sms_band0!(),4=>tail27_sms_band4!(),_=>TAIL};let(w,o,bounds)=run_leaf(text,&rows,&ptrs);assert!(w<=32767&&o<=SMS_SOURCE_BOUND);peak=peak.max(o);for (lo,hi)in bounds{assert!(lo<=hi&&lo>=-SMS_SOURCE_BOUND&&hi<=SMS_SOURCE_BOUND);let coefficient=all.len();assert_eq!((lo,hi),crate::simd::commit::sms::source_bounds::INTERVALS[coefficient]);assert_eq!((-lo).max(hi),crate::simd::commit::sms::source_bounds::BOUNDS[coefficient]);all.push(format!("[{lo},{hi}]"));}}}assert_eq!(all.len(),648);let text=format!("{{\"scope\":\"Exact per-coefficient integer source intervals from canonical actual asm and complete slicer domains; no kernel change\",\"global_peak\":{peak},\"intervals\":[{}]}}\n",all.join(","));let _ = text;println!("PUBLIC SOURCE INTERVAL PROOF PASS coefficients=648 exact_global_peak={peak}");}
}

#[cfg(test)]
mod sms_phi24_source_interval_guards {
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
                    let(lo,hi)=crate::simd::commit::sms::source_bounds::INTERVALS[i/32];assert!((lo..=hi).contains(&(a as i64)),"coefficient={} value={a} interval={lo}..{hi}",i/32);
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
    fn phi24_source_all512basis_with_exact_coefficient_intervals() {
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

