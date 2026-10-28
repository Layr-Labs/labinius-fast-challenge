#![allow(clippy::needless_range_loop)]
#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
include!(concat!(env!("OUT_DIR"), "/fixture-lib.rs"));

#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
mod candidate_a_checks {
    fn next(s: &mut u64) -> u64 {
        *s ^= *s << 13;
        *s ^= *s >> 7;
        *s ^= *s << 17;
        *s
    }
    #[test]
    fn inverse_9721_and_sms_equivalence() {
        assert!(
            is_x86_feature_detected!("avx512f")
                && is_x86_feature_detected!("avx512bw")
                && is_x86_feature_detected!("avx512vl")
                && is_x86_feature_detected!("avx512vbmi")
                && is_x86_feature_detected!("avx512vbmi2")
                && is_x86_feature_detected!("avx512vnni")
                && is_x86_feature_detected!("gfni"),
            "required native features"
        );
        use labinius::{
            params::N,
            ring::{Batch32, Representation},
            simd::ntt::gen_small as candidate,
        };
        use labinius_control::simd::ntt::gen_small as control;
        assert_eq!(candidate::TwI::<9721>::IN_BOUND, 22359);
        assert_eq!(candidate::TwI::<9721>::OUT_BOUND, 4860);
        let mut state = 0x9721_1944_6480_0032u64;
        for case in 0..7 {
            let mut a = Batch32::zero(Representation::Ntt);
            let mut b =
                labinius_control::ring::Batch32::zero(labinius_control::ring::Representation::Ntt);
            for j in 0..N {
                for lane in 0..32 {
                    let x = match case {
                        0 => 0,
                        1 => 22359,
                        2 => -22359,
                        3 => {
                            if j % 2 == 0 {
                                22359
                            } else {
                                -22359
                            }
                        }
                        4 => {
                            if lane % 2 == 0 {
                                22359
                            } else {
                                -22359
                            }
                        }
                        5 => {
                            if (j % 3 + lane) % 2 == 0 {
                                22359
                            } else {
                                -22359
                            }
                        }
                        _ => ((next(&mut state) % 44719) as i32 - 22359) as i16,
                    };
                    a.v[j][lane] = x;
                    b.v[j][lane] = x;
                }
            }
            let input = a.clone();
            unsafe {
                candidate::intt_gen_batch32::<9721>(&mut a);
                control::intt_gen_batch32::<9721>(&mut b);
            }
            assert_eq!(a.v, b.v, "inverse case {case}");
            assert_eq!(a.representation, Representation::Coefficients);
            assert_eq!(
                b.representation,
                labinius_control::ring::Representation::Coefficients
            );
            for lane in 0..32 {
                let col = core::array::from_fn(|j| input.v[j][lane]);
                let oracle =
                    labinius::scalar::intt::<9721>(&labinius::scalar::normalize_i16(&col, 9721));
                for j in 0..N {
                    assert!((a.v[j][lane] as i32).abs() <= 4860);
                    assert_eq!((a.v[j][lane] as i32).rem_euclid(9721) as u32, oracle[j]);
                }
            }
        }
        for (name, elements, columns) in [
            ("one-step", 32, 1),
            ("short", 64, 15),
            ("short-tail", 544, 17),
            ("scored", 4096, 16),
        ] {
            let matrix: Vec<u128> = (0..elements * N)
                .map(|_| {
                    (((next(&mut state) as u128) << 64) | next(&mut state) as u128)
                        % crate::simd::commit::sms::MODULUS
                })
                .collect();
            let data = crate::simd::commit::sms::SmsKeyData::build(&matrix, elements);
            let words: Vec<u128> = (0..4 * elements * columns)
                .map(|_| ((next(&mut state) as u128) << 64) | next(&mut state) as u128)
                .collect();
            let a = crate::simd::commit::sms::diagnostic_sms(
                elements,
                columns,
                &data.pairs,
                &data.mur,
                &data.mu,
                &words,
            );
            let b = labinius_control::simd::commit::sms::diagnostic_sms(
                elements,
                columns,
                &data.pairs,
                &data.mur,
                &data.mu,
                &words,
            );
            assert_eq!(a.0.len(), b.0.len());
            assert!(!a.0.is_empty());
            for (x, y) in a.0.iter().zip(&b.0) {
                assert_eq!(
                    (*x as i32 - *y as i32).rem_euclid(9721),
                    0,
                    "pending {name}"
                );
            }
            assert_eq!(a.1, b.1, "centered inverse {name}");
            assert_eq!(a.2.len(), b.2.len());
            for (x, y) in a.2.iter().zip(&b.2) {
                assert_eq!((y - x).rem_euclid(9721), 32768 % 9721, "bias {name}");
            }
            if elements == 32 {
                assert!(a.2.iter().all(|&x| x == 0));
            }
            assert_eq!(a.3, b.3, "full rows {name}");
            assert_eq!(a.4, b.4, "half rows {name}");
            eprintln!("focused shape={name} elements={elements} columns={columns} pending={} inverse={} rows={} mode=0 result=PASS",a.0.len(),a.1.len(),a.3.len());
        }
    }
}
#[cfg(not(all(target_arch = "x86_64", target_os = "linux")))]
compile_error!("focused test requires native Linux x86-64");
