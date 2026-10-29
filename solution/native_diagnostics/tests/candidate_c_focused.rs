#![allow(clippy::needless_range_loop)]
#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
include!(concat!(env!("OUT_DIR"), "/fixture-lib.rs"));
#[path = "../src/support.rs"]
mod support;
#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
mod mean_static {
    include!(concat!(env!("OUT_DIR"), "/mean-static.rs"));
}
#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
mod candidate_c_checks {
    use labinius::{fields::scalar::B128, params::N, simd::commit::sms};
    use labinius_control::simd::commit::sms as old;
    use std::{fs, path::Path};
    fn next(s: &mut u64) -> u64 {
        *s ^= *s << 13; *s ^= *s >> 7; *s ^= *s << 17; *s
    }
    fn canonical<const LEN: usize>(mur: &[i16]) -> Vec<i16> {
        assert_eq!(2 * mur.len(), LEN);
        let mut bytes = [0u8; LEN];
        for (at, value) in mur.iter().enumerate() {
            assert!((-4860..=4860).contains(value));
            bytes[2 * at..2 * at + 2].copy_from_slice(&value.to_le_bytes());
        }
        let bytes = crate::mean_static::normalize(bytes);
        mur.iter().zip(bytes.chunks_exact(2)).enumerate().map(|(at, (&old, b))| {
            let value = i16::from_le_bytes(b.try_into().unwrap());
            assert_eq!(value as i32, (old as i32).rem_euclid(9721), "canonical mean {at}");
            assert!((0..=9720).contains(&value)); value
        }).collect()
    }
    fn equal_rows(name: &str, path: &str, wanted: &[[u32; N]], actual: &[[u32; N]], columns: usize) {
        assert_eq!(actual.len(), 2 * columns, "{name} {path} row count");
        assert_eq!(wanted.len(), actual.len());
        for (row, (want, got)) in wanted.iter().zip(actual).enumerate() {
            for m in 0..N {
                assert_eq!(got[m], want[m], "{name} {path} row={row} coefficient={m}");
                assert!(got[m] < if row % 2 == 0 { 9721 } else { 1 << 22 }, "{name} {path} row={row} coefficient={m} bound");
            }
        }
    }
    fn real_mean_image() {
        let out = Path::new(env!("OUT_DIR"));
        let raw_path = out.join("sms-mur.bin");
        let raw_hash = crate::support::sha256(&raw_path);
        let raw = crate::mean_static::raw();
        let canonical = crate::mean_static::canonical();
        assert_eq!(raw.len(), 145_152); assert_eq!(canonical.len(), raw.len());
        assert_eq!(fs::read(&raw_path).unwrap(), raw);
        assert_eq!(canonical.as_ptr() as usize % 64, 0);
        for (at, (a, b)) in raw.chunks_exact(2).zip(canonical.chunks_exact(2)).enumerate() {
            let a = i16::from_le_bytes(a.try_into().unwrap());
            let b = i16::from_le_bytes(b.try_into().unwrap());
            assert!((-4860..=4860).contains(&a), "raw native mean {at}");
            assert!((0..=9720).contains(&b), "canonical native mean {at}");
            assert_eq!(b as i32, (a as i32).rem_euclid(9721), "native mean {at}");
        }
        let derived_path = out.join("canonical-mean-static.bin");
        assert!(!derived_path.exists()); fs::write(&derived_path, canonical).unwrap();
        assert_eq!(crate::support::sha256(&raw_path), raw_hash);
        let view_hash = crate::support::sha256(&out.join("mean-static.rs"));
        fs::write(out.join("mean-image-check.tsv"), format!("{}\t{}\t{}\t{}\t{}\t64\t{}\n", raw.len(), raw_hash,
            canonical.len(), crate::support::sha256(&derived_path), raw.len() / 2, view_hash)).unwrap();
    }
    #[test]
    fn canonical_mean_equivalence() {
        assert!(is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512vl") && is_x86_feature_detected!("avx512vbmi")
            && is_x86_feature_detected!("avx512vbmi2") && is_x86_feature_detected!("avx512vnni")
            && is_x86_feature_detected!("gfni"), "required native features");
        assert_eq!(sms::MODULUS, 40_772_829_184); assert_eq!(sms::LOCK, 8);
        real_mean_image();
        let mut state = 0x9721_1944_6480_0032u64;
        let (mut publics, mut bridges, mut arrays) = (0, 0, 0);
        for (name, elements, columns) in [("one-step",32,1),("short",64,15),("short-tail",544,17),("scored",4096,16)] {
            let matrix: Vec<u128> = (0..elements * N).map(|_| (((next(&mut state) as u128) << 64) | next(&mut state) as u128) % sms::MODULUS).collect();
            let data = sms::SmsKeyData::build(&matrix, elements);
            let words: Vec<u128> = (0..4 * elements * columns).map(|_| ((next(&mut state) as u128) << 64) | next(&mut state) as u128).collect();
            let canonical = match elements { 32 | 64 => canonical::<18144>(&data.mur), 544 => canonical::<36288>(&data.mur), 4096 => canonical::<145152>(&data.mur), _ => unreachable!() };
            let input: Vec<B128> = words.iter().copied().map(B128).collect();
            let cols: Vec<_> = input.chunks_exact(4 * elements).collect();
            let old_input: Vec<labinius_control::fields::scalar::B128> = words.iter().copied().map(labinius_control::fields::scalar::B128).collect();
            let old_cols: Vec<_> = old_input.chunks_exact(4 * elements).collect();
            let centered_keys = data.keys();
            let canonical_keys = sms::SmsKeys { pairs: &data.pairs, mur: &canonical, mu: &data.mu, elements };
            let old_keys = old::SmsKeys { pairs: &data.pairs, mur: &data.mur, mu: &data.mu, elements };
            let centered = sms::commit_columns_rows(&centered_keys, &cols, 1);
            let control = old::commit_columns_rows(&old_keys, &old_cols, 1);
            let positive = sms::commit_columns_rows_canonical_means(&canonical_keys, &cols, 1);
            publics += 3;
            equal_rows(name, "candidate centered", &control, &centered, columns);
            equal_rows(name, "candidate canonical", &control, &positive, columns);
            let paths = [
                ("control", old::diagnostic_sms_rows(elements, columns, &data.pairs, &data.mur, &data.mu, &words, false)),
                ("candidate centered", crate::simd::commit::sms::diagnostic_sms_rows(elements, columns, &data.pairs, &data.mur, &data.mu, &words, false)),
                ("candidate canonical", crate::simd::commit::sms::diagnostic_sms_rows(elements, columns, &data.pairs, &canonical, &data.mu, &words, true)),
            ];
            for (path, results) in paths {
                bridges += 1; assert_eq!(results.len(), 3);
                for (pass, rows) in results.iter().enumerate() {
                    equal_rows(name, &format!("{path} pass={pass}"), &control, rows, columns); arrays += 1;
                }
            }
            eprintln!("focused shape={name} elements={elements} columns={columns} public_calls=3 bridges=3 arrays=9 mode=0 result=PASS");
        }
        assert_eq!((publics, bridges, arrays), (12, 12, 36));
    }
}
#[cfg(not(all(target_arch = "x86_64", target_os = "linux")))]
compile_error!("focused test requires native Linux x86-64");
