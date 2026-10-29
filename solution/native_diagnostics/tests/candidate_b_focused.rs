#![allow(clippy::needless_range_loop)]
#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
extern crate labinius as _;
#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
include!(concat!(env!("OUT_DIR"), "/fixture-lib.rs"));

#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
mod candidate_b_checks {
    type Field = [u64; 3];
    const MASK34: u64 = (1 << 34) - 1;
    const SEED: u64 = 0x4b1d_5eed_3a9c_6027;
    const ONE: Field = [1, 0, 0];
    const SPARSE_ROWS: [usize; 8] = [0, 1, 6, 7, 8, 63, 64, 16383];

    fn next(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    fn field(state: &mut u64) -> Field {
        [next(state), next(state), next(state) & MASK34]
    }

    fn word(state: &mut u64) -> u128 {
        (next(state) as u128) | ((next(state) as u128) << 64)
    }

    fn xor(a: Field, b: Field) -> Field {
        core::array::from_fn(|i| a[i] ^ b[i])
    }

    // Independent polynomial convolution and descending long reduction.
    fn multiply(a: Field, b: Field) -> Field {
        assert_eq!(a[2] & !MASK34, 0);
        assert_eq!(b[2] & !MASK34, 0);
        let mut product = [0u64; 6];
        for j in 0..3 {
            let mut bits = b[j];
            while bits != 0 {
                let shift = bits.trailing_zeros() as usize;
                for i in 0..3 {
                    product[i + j] ^= a[i] << shift;
                    if shift != 0 {
                        product[i + j + 1] ^= a[i] >> (64 - shift);
                    }
                }
                bits &= bits - 1;
            }
        }
        for degree in (162..=322).rev() {
            if (product[degree / 64] >> (degree % 64)) & 1 != 0 {
                for bit in [degree, degree - 81, degree - 162] {
                    product[bit / 64] ^= 1u64 << (bit % 64);
                }
            }
        }
        assert_eq!(product[2] & !MASK34, 0);
        assert_eq!(product[3..], [0; 3]);
        [product[0], product[1], product[2]]
    }

    fn lift(word: u128) -> Field {
        [word as u64, (word >> 64) as u64, 0]
    }

    fn weight(point: &[Field], index: usize) -> Field {
        point.iter().enumerate().fold(ONE, |value, (k, &p)| {
            multiply(
                value,
                if index & (1 << k) != 0 {
                    p
                } else {
                    xor(ONE, p)
                },
            )
        })
    }

    fn oracle_row(point: &[Field], words: &[u128]) -> Vec<Field> {
        let n = 1usize << point.len();
        assert_eq!(words.len() % n, 0);
        let weights: Vec<_> = (0..n).map(|i| weight(point, i)).collect();
        words
            .chunks_exact(n)
            .map(|column| {
                column
                    .iter()
                    .zip(&weights)
                    .fold([0; 3], |sum, (&w, &a)| xor(sum, multiply(a, lift(w))))
            })
            .collect()
    }

    fn special_words() -> [u128; 8] {
        [
            0,
            1,
            1 << 63,
            1 << 64,
            1 << 127,
            u128::MAX,
            0xaaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa,
            0x5555_5555_5555_5555_5555_5555_5555_5555,
        ]
    }

    fn replace_words(words: &mut [u128], column: usize) {
        let special = special_words();
        for (i, word) in words.iter_mut().take(8).enumerate() {
            *word = special[(i + column) % 8];
        }
    }

    fn private_inputs(n: usize) -> (Vec<Field>, [Vec<u128>; 4]) {
        let mut state = SEED;
        let mut a: Vec<_> = (0..n).map(|_| field(&mut state)).collect();
        let special = [
            [0, 0, 0],
            ONE,
            [1 << 63, 0, 0],
            [0, 1, 0],
            [0, 1 << 63, 0],
            [0, 0, 1],
            [0, 0, 1 << 33],
            [u64::MAX, u64::MAX, MASK34],
            [0xaaaa_aaaa_aaaa_aaaa, 0xaaaa_aaaa_aaaa_aaaa, 0x2_aaaa_aaaa],
            [0x5555_5555_5555_5555, 0x5555_5555_5555_5555, 0x1_5555_5555],
        ];
        for (value, replacement) in a.iter_mut().zip(special) {
            *value = replacement;
        }
        let words = core::array::from_fn(|j| {
            let mut column: Vec<_> = (0..n).map(|_| word(&mut state)).collect();
            replace_words(&mut column, j);
            column
        });
        (a, words)
    }

    fn public_inputs(log: usize, columns: usize) -> (Vec<Field>, Vec<u128>) {
        let mut state = SEED;
        let point = (0..log).map(|_| field(&mut state)).collect();
        let n = 1usize << log;
        let mut words: Vec<_> = (0..n * columns).map(|_| word(&mut state)).collect();
        for (j, column) in words.chunks_exact_mut(n).enumerate() {
            replace_words(column, j);
        }
        (point, words)
    }

    fn candidate_public(words: &[u128], point: &[Field], workers: usize) -> Vec<Field> {
        let point: Vec<_> = point.iter().copied().map(labinius::F162).collect();
        // The frozen B128 source declares repr(transparent) over u128.
        let words = unsafe {
            core::slice::from_raw_parts(words.as_ptr().cast::<labinius::B128>(), words.len())
        };
        labinius::eval::row_evaluate_words(words, &point, workers)
            .into_iter()
            .map(|value| value.0)
            .collect()
    }

    fn control_public(words: &[u128], point: &[Field], workers: usize) -> Vec<Field> {
        let point: Vec<_> = point.iter().copied().map(labinius_control::F162).collect();
        // The frozen B128 source declares repr(transparent) over u128.
        let words = unsafe {
            core::slice::from_raw_parts(
                words.as_ptr().cast::<labinius_control::B128>(),
                words.len(),
            )
        };
        labinius_control::eval::row_evaluate_words(words, &point, workers)
            .into_iter()
            .map(|value| value.0)
            .collect()
    }

    fn check_public(
        point: &[Field],
        words: &[u128],
        workers: usize,
        expected: &[Field],
        sparse: bool,
    ) {
        let candidate = candidate_public(words, point, workers);
        let control = control_public(words, point, workers);
        assert_eq!(
            candidate.len(),
            expected.len(),
            "candidate row length log={} workers={workers}",
            point.len()
        );
        assert_eq!(
            control.len(),
            expected.len(),
            "control row length log={} workers={workers}",
            point.len()
        );
        let n = 1usize << point.len();
        for j in 0..expected.len() {
            if candidate[j] != expected[j]
                || control[j] != expected[j]
                || candidate[j][2] & !MASK34 != 0
                || control[j][2] & !MASK34 != 0
            {
                let input: Vec<_> = if sparse {
                    SPARSE_ROWS.iter().map(|&i| (i, words[n * j + i])).collect()
                } else {
                    words[n * j..n * (j + 1)]
                        .iter()
                        .copied()
                        .enumerate()
                        .collect()
                };
                panic!("first row failure log={} columns={} workers={workers} sparse={sparse} output={j} point={point:?} words={input:?} expected={:?} candidate={:?} control={:?}", point.len(), expected.len(), expected[j], candidate[j], control[j]);
            }
        }
    }

    #[test]
    fn row_equivalence() {
        assert!(
            is_x86_feature_detected!("avx512f")
                && is_x86_feature_detected!("avx512bw")
                && is_x86_feature_detected!("avx512vl")
                && is_x86_feature_detected!("avx512dq")
                && is_x86_feature_detected!("avx512vbmi")
                && is_x86_feature_detected!("avx512vbmi2")
                && is_x86_feature_detected!("avx512vnni")
                && is_x86_feature_detected!("vpclmulqdq")
                && is_x86_feature_detected!("pclmulqdq")
                && is_x86_feature_detected!("bmi2")
                && is_x86_feature_detected!("gfni"),
            "required native features"
        );
        let (mut cases, mut outputs) = (0, 0);
        for n in (0..=17).chain([63, 64, 65]) {
            let (a, words) = private_inputs(n);
            let candidate = crate::eval::diagnostic_quad(&a, &words);
            let control = labinius_control::eval::diagnostic_singles(&a, &words);
            for j in 0..4 {
                let expected = a
                    .iter()
                    .zip(&words[j])
                    .fold([0; 3], |sum, (&a, &w)| xor(sum, multiply(a, lift(w))));
                assert!(candidate[j] == expected && control[j] == expected
                    && candidate[j][2] & !MASK34 == 0 && control[j][2] & !MASK34 == 0,
                    "first private failure n={n} output={j} coefficients={a:?} words={:?} expected={expected:?} candidate={:?} control={:?}", words[j], candidate[j], control[j]);
            }
            cases += 1;
            outputs += 4;
        }
        for log in [0, 1, 2, 3, 6] {
            for columns in (0..=8).chain([1023, 1024, 1025]) {
                let (point, words) = public_inputs(log, columns);
                let expected = oracle_row(&point, &words);
                for workers in [1, 3, 8] {
                    check_public(&point, &words, workers, &expected, false);
                    cases += 1;
                    outputs += columns;
                }
            }
        }
        let mut state = SEED;
        let point: Vec<_> = (0..14).map(|_| field(&mut state)).collect();
        let n = 1usize << 14;
        let mut words = vec![0u128; n * 1024];
        let weights = SPARSE_ROWS.map(|i| weight(&point, i));
        let mut expected = vec![[0; 3]; 1024];
        for j in 0..1024 {
            for (k, &i) in SPARSE_ROWS.iter().enumerate() {
                let w = word(&mut state) | 1;
                words[n * j + i] = w;
                expected[j] = xor(expected[j], multiply(weights[k], lift(w)));
            }
        }
        check_public(&point, &words, 8, &expected, true);
        cases += 1;
        outputs += 1024;
        assert_eq!(cases, 202);
        assert_eq!(outputs, 47_728);
        eprintln!(
            "focused fixture=candidate-b-row cases={cases} full_f162_outputs={outputs} result=PASS"
        );
    }
}

#[cfg(not(all(target_arch = "x86_64", target_os = "linux")))]
compile_error!("focused test requires native Linux x86-64");
