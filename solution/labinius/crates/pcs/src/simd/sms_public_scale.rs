//! Public diagonal preconditioning of the six auxiliary NTT streams.
//!
//! A slot's nonzero scale multiplies both its public keys and mean subtraction.
//! Its inverse is applied before the existing inverse NTT, so the reconstructed
//! integer coefficient banks and their statistical margins remain unchanged.
//! The scale is selected from public keys alone to certify sixteen-pair signed
//! accumulation windows. Base keys and their eight-pair schedule stay unchanged.
use super::{P, PP, LOCK, LOCK_WORDS, N, STREAMS, DIGITS, CHUNK_PAIRS,
    acc_after_reduce_c, bias, w_bound, fold_bias, fold_bias_period, chunk_count, chunk_range, qinv16};

const PERIOD: usize = 2 * PP;
const WINDOW_COEFFICIENTS: usize = 2 * PERIOD;

fn center(x: i32) -> i16 {
    let r = x.rem_euclid(P as i32);
    (if r > (P / 2) as i32 { r - P as i32 } else { r }) as i16
}

fn key_limit() -> i64 {
    // The extra 32768 is conservative; regular folds already satisfy the
    // smaller acc_after_reduce_c bound, including signed intermediate values.
    (bias(P) - acc_after_reduce_c(P) - 32768) / w_bound(P)
}

fn find_scale(canonical: &[i32], limit: i64) -> Option<i16> {
    debug_assert_eq!(canonical.len() % WINDOW_COEFFICIENTS, 0);
    // Opposite units have identical centered absolute values. For this fixed
    // prime every candidate is a unit; inverse() checks that independently.
    for c in 1..=P as i32 / 2 {
        if canonical.chunks_exact(WINDOW_COEFFICIENTS).all(|window| {
            let sum: i64 = window.iter().map(|&x| {
                let r = x * c % P as i32;
                if r > (P / 2) as i32 { (P as i32 - r) as i64 } else { r as i64 }
            }).sum();
            sum <= limit
        }) { return Some(c as i16); }
    }
    None
}

fn inverse(a: i16) -> Option<i16> {
    let (mut r, mut next_r) = (P as i32, a as i32);
    let (mut t, mut next_t) = (0i32, 1i32);
    while next_r != 0 {
        let q = r / next_r;
        (r, next_r) = (next_r, r - q * next_r);
        (t, next_t) = (next_t, t - q * next_t);
    }
    (r == 1).then(|| center(t))
}

fn inverse_pair(a: i16, b: i16) -> Option<[i16; 16]> {
    let mut out = [0i16; 16];
    for (row, c) in [a, b].into_iter().enumerate() {
        let w = center(inverse(c)? as i32 * 65536);
        let wp = (w as u16 as u32).wrapping_mul(qinv16(P)) as u16 as i16;
        out[4 * row..4 * row + 4].fill(w);
        out[8 + 4 * row..12 + 4 * row].fill(wp);
    }
    Some(out)
}

fn certificate(pairs: &[u32]) -> bool {
    pairs.chunks_exact(2 * LOCK_WORDS).all(|window| {
        (0..N).all(|j| (1..STREAMS).all(|s| {
            let mut sum = 0i64;
            for step in 0..2 { for h in 0..LOCK {
                let v = window[step * LOCK_WORDS + (j * STREAMS + s) * LOCK + h];
                sum += (v as i16 as i64).abs() + ((v >> 16) as i16 as i64).abs();
            }}
            sum <= key_limit()
        }))
    })
}

/// Returns empty without modifying either image when the generic schedule is
/// needed. A nonempty result certifies every complete auxiliary sixteen-pair
/// window; whole half-columns and reconstruction chunks also end on its boundary.
pub(super) fn precondition(pairs: &mut [u32], mur: &mut [i32], elements: usize) -> Vec<[i16; 16]> {
    assert_eq!(LOCK, PP);
    assert_eq!(PP, 8);
    if elements % (4 * PERIOD) != 0 || CHUNK_PAIRS % PERIOD != 0 { return Vec::new(); }
    assert_eq!(pairs.len(), elements / (2 * LOCK) * LOCK_WORDS);
    assert_eq!(mur.len(), chunk_count(elements) * N * DIGITS);
    let mut scales = vec![1i16; DIGITS * N];
    let mut canonical = Vec::with_capacity(elements);
    for s in 1..STREAMS { for j in 0..N {
        canonical.clear();
        for step in pairs.chunks_exact(LOCK_WORDS) { for h in 0..LOCK {
            let word = step[(j * STREAMS + s) * LOCK + h];
            for v in [word as i16, (word >> 16) as i16] {
                canonical.push((v as i32).rem_euclid(P as i32));
            }
        }}
        let Some(c) = find_scale(&canonical, key_limit()) else { return Vec::new(); };
        scales[(s - 1) * N + j] = c;
    }}
    // Finish all fallible selection before mutating either public image.
    let mut inverses = Vec::with_capacity(DIGITS * (N / 2));
    for s in 0..DIGITS { for j in (0..N).step_by(2) {
        let Some(pair) = inverse_pair(scales[s * N + j], scales[s * N + j + 1]) else { return Vec::new(); };
        inverses.push(pair);
    }}
    for step in pairs.chunks_exact_mut(LOCK_WORDS) { for j in 0..N { for s in 1..STREAMS { for h in 0..LOCK {
        let slot = &mut step[(j * STREAMS + s) * LOCK + h];
        let c = scales[(s - 1) * N + j] as i32;
        let lo = center(*slot as i16 as i32 * c) as u16 as u32;
        let hi = center((*slot >> 16) as i16 as i32 * c) as u16 as u32;
        *slot = lo | (hi << 16);
    }}}}
    assert!(certificate(pairs), "public auxiliary accumulation certificate");
    for chunk in 0..chunk_count(elements) {
        let span = chunk_range(elements / 2, chunk).len();
        let (old_bias, new_bias) = (fold_bias(span), fold_bias_period(span, PERIOD));
        for j in 0..N { for d in 0..DIGITS {
            let value = &mut mur[(chunk * N + j) * DIGITS + d];
            let mean = old_bias - *value;
            *value = new_bias - center(mean * scales[d * N + j] as i32) as i32;
        }}
    }
    inverses
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn search_and_inverse_preserve_residues() {
        assert_eq!(find_scale(&[4860; 32], key_limit()), Some(2));
        assert_eq!(find_scale(&[1; 32], 0), None);
        for a in 1..=P as i16 / 2 {
            let b = 1 + ((a as i32 * 37) % (P as i32 / 2)) as i16;
            let pair = inverse_pair(a, b).unwrap();
            for (row, c) in [a, b].into_iter().enumerate() {
                let w = pair[4 * row] as i32;
                let wp = pair[8 + 4 * row] as i32;
                for x in [-9720i32, -4860, -1, 0, 1, 4860, 9720] {
                    let m = (x * wp) as i16 as i32;
                    let y = (x * w >> 16) - (m * P as i32 >> 16);
                    assert_eq!((y * c as i32 - x).rem_euclid(P as i32), 0);
                    assert!(y.abs() <= P as i32 - 1);
                }
                assert!(pair[4 * row..4 * row + 4].iter().all(|&x| x as i32 == w));
                assert!(pair[8 + 4 * row..12 + 4 * row].iter().all(|&x| x as i32 == wp));
            }
        }
    }

    #[test]
    fn public_images_certified_and_base_unchanged() {
        let elements = 64;
        let mut pairs = vec![0u32; elements / (2 * LOCK) * LOCK_WORDS];
        for step in pairs.chunks_exact_mut(LOCK_WORDS) { for j in 0..N { for h in 0..LOCK {
            step[(j * STREAMS + 1) * LOCK + h] = 4860 | ((-4860i16 as u16 as u32) << 16);
            step[j * STREAMS * LOCK + h] = 123 | (456 << 16);
        }}}
        assert!(!certificate(&pairs));
        let original = pairs.clone();
        let old_bias = fold_bias(16);
        let mut mur = vec![old_bias - 37; chunk_count(elements) * N * DIGITS];
        let inverse = precondition(&mut pairs, &mut mur, elements);
        assert_eq!(inverse.len(), DIGITS * N / 2);
        assert!(certificate(&pairs));
        for (old, new) in original.chunks_exact(LOCK_WORDS).zip(pairs.chunks_exact(LOCK_WORDS)) {
            for j in 0..N { assert_eq!(&old[j * STREAMS * LOCK..j * STREAMS * LOCK + LOCK], &new[j * STREAMS * LOCK..j * STREAMS * LOCK + LOCK]); }
        }
        for (i, &value) in mur.iter().enumerate() {
            let scale = if i % DIGITS == 0 { 2 } else { 1 };
            assert_eq!(value, fold_bias_period(16, PERIOD) - 37 * scale);
        }
    }

    #[test]
    fn partial_windows_keep_generic_images() {
        let mut pairs = vec![7; 96 / (2 * LOCK) * LOCK_WORDS];
        let mut mur = vec![19; chunk_count(96) * N * DIGITS];
        let original = (pairs.clone(), mur.clone());
        assert!(precondition(&mut pairs, &mut mur, 96).is_empty());
        assert_eq!((pairs, mur), original);
    }
}
