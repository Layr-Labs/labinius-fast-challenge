//! Portable checks: `rustc --test -O centered_crt_tests.rs -o /tmp/centered-crt-tests`.

const P0: i64 = 3889;
const P1: i64 = 9721;
const PRODUCT: i64 = P0 * P1;
const INV: i64 = 3242;
const W: i64 = -4185;
const WP: i64 = -21857;

pub fn reference_crt(a: i64, b: i64) -> i64 {
    let u = ((b - a) * INV).rem_euclid(P1);
    (a + P0 * u + PRODUCT / 2).rem_euclid(PRODUCT) - PRODUCT / 2
}

// Scalar form of the SIMD signed-word Montgomery product.
fn montgomery(d: i64) -> i64 {
    let m = (d * WP) as i16 as i64;
    ((d * W) >> 16) - ((m * P1) >> 16)
}

fn centered_crt(a: i64, b: i64) -> i64 {
    let mut u = montgomery(b - a);
    if u > P1 / 2 {
        u -= P1;
    }
    if u < -(P1 / 2) {
        u += P1;
    }
    a + P0 * u
}

#[test]
fn montgomery_covers_every_centered_difference() {
    assert_eq!((P0 * INV) % P1, 1);
    assert_eq!((W - 65536 * INV).rem_euclid(P1), 0);
    assert_eq!((WP * P1 - W).rem_euclid(65536), 0);
    let bound = P0 / 2 + P1 / 2;
    for d in -bound..=bound {
        let u = montgomery(d);
        assert!(u.abs() < P1);
        assert_eq!((P0 * u - d).rem_euclid(P1), 0);
    }
}

#[test]
fn centered_crt_matches_all_residue_pairs() {
    let mut count = 0u64;
    for a in -(P0 / 2)..=P0 / 2 {
        for b in -(P1 / 2)..=P1 / 2 {
            let got = centered_crt(a, b);
            assert!(got.abs() <= PRODUCT / 2);
            assert_eq!(got, reference_crt(a, b), "a={a}, b={b}");
            count += 1;
        }
    }
    assert_eq!(count, 37_804_969);
}
