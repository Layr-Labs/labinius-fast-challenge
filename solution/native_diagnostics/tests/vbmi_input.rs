//! Native equality fixture. Run only on the required x86-64 VBMI/GFNI host.
#![allow(dead_code)]
#[cfg(not(target_arch = "x86_64"))]
compile_error!("This fixture must execute the actual x86-64 slicer; no portable substitute is allowed.");

pub use labinius::fields;
use labinius::fields::scalar::B128;
use labinius::simd::transpose_f162 as candidate;
#[path = "../src/control_transpose_f162_pr221.rs"]
mod control;

fn oracle(words: &[B128; 128]) -> [[u8; 64]; 162] {
    let mut rows = [[0; 64]; 162];
    for i in 0..162 {
        for p in 0..32 {
            let mut n = 0;
            for b in 0..4 {
                let c = i + 162 * b;
                if c < 512 {
                    n |= (((words[4 * p + c % 4].0 >> (c / 4)) & 1) as u8) << b;
                }
            }
            rows[i][2 * p] = n;
            rows[i][2 * p + 1] = 16 + n;
        }
    }
    rows
}

fn compare(got: &candidate::BinaryIndex32, old: &control::BinaryIndex32, want: &[[u8; 64]; 162], label: &str) {
    for i in 0..162 {
        assert_eq!(got.rows[i], old.rows[i], "{label}: control row {i}");
        assert_eq!(got.rows[i], want[i], "{label}: coefficient oracle row {i}");
    }
}

fn check_cols(words: &[B128; 128], cols: &[*const B128; 16], label: &str) -> candidate::BinaryIndex32 {
    let mut got = candidate::BinaryIndex32::zero();
    let mut old = control::BinaryIndex32::zero();
    unsafe {
        candidate::slice_b128_cols16_into(cols, &mut got);
        control::slice_b128_cols16_into(cols, &mut old);
    }
    compare(&got, &old, &oracle(words), label);
    got
}

fn check(words: &[B128; 128], label: &str, sibling_entries: bool) -> candidate::BinaryIndex32 {
    let cols = core::array::from_fn(|c| unsafe { words.as_ptr().add(8 * c) });
    let got = check_cols(words, &cols, label);
    if sibling_entries {
        let want = oracle(words);
        let mut a = candidate::BinaryIndex32::zero();
        let mut b = control::BinaryIndex32::zero();
        let lanes = core::array::from_fn(|p| unsafe { words.as_ptr().add(4 * p) });
        unsafe {
            candidate::slice_b128_lanes_into(&lanes, &mut a);
            control::slice_b128_lanes_into(&lanes, &mut b);
        }
        compare(&a, &b, &want, "lanes entry");
        unsafe {
            candidate::slice_b128_into(words, &mut a);
            control::slice_b128_into(words, &mut b);
        }
        compare(&a, &b, &want, "contiguous entry");
        unsafe {
            candidate::slice_b128_staged_into(words, &mut a);
            control::slice_b128_staged_into(words, &mut b);
        }
        compare(&a, &b, &want, "unchanged F162 staged entry");
    }
    got
}

fn next(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

#[test]
fn native_vbmi_input_equivalence() {
    assert!(std::is_x86_feature_detected!("avx512f"));
    assert!(std::is_x86_feature_detected!("avx512bw"));
    assert!(std::is_x86_feature_detected!("avx512vl"));
    assert!(std::is_x86_feature_detected!("avx512vbmi"));
    assert!(std::is_x86_feature_detected!("gfni"));
    let mut words = [B128(0); 128];
    check(&words, "zero", true);
    // Every input bit, including every lane, component, ag/bg end and zero-tail transition.
    for word in 0..128 {
        for bit in 0..128 {
            words[word] = B128(1u128 << bit);
            check(&words, "one input-bit basis", false);
            words[word] = B128(0);
        }
    }
    words[0] = B128(1u128 << 91);
    let got = check(&words, "bit91 row40 counterexample", true);
    assert_eq!(got.rows[40][0], 4);
    assert_eq!(got.rows[40][1], 20);
    words = [B128(u128::MAX); 128];
    check(&words, "all one", true);
    for bit in [0, 1, 40, 41, 63, 64, 80, 81, 91, 120, 121, 122, 127] {
        words = [B128(1u128 << bit); 128];
        check(&words, "all lanes at boundary bit", true);
        words = [B128(u128::MAX ^ (1u128 << bit)); 128];
        check(&words, "all lanes except boundary bit", true);
    }
    for value in [0xaaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaau128, 0x5555_5555_5555_5555_5555_5555_5555_5555] {
        words = core::array::from_fn(|i| B128(if i & 1 == 0 { value } else { !value }));
        check(&words, "alternating words/bits", true);
    }
    let mut state = 0x63bd_e20a_791c_45f1;
    for _ in 0..32 {
        words = core::array::from_fn(|_| B128(next(&mut state) as u128 | ((next(&mut state) as u128) << 64)));
        check(&words, "deterministic mixed", true);
    }
    // Disjoint columns with an extra B128 prefix exercise unaligned 64-byte input loads.
    let mut padded = [[B128(0); 9]; 16];
    for c in 0..16 {
        padded[c][1..9].copy_from_slice(&words[8 * c..8 * c + 8]);
    }
    let cols = core::array::from_fn(|c| unsafe { padded[c].as_ptr().add(1) });
    check_cols(&words, &cols, "noncontiguous unaligned columns");
    let zero = [B128(0); 8];
    for live in 0..=16 {
        let cols = core::array::from_fn(|c| if c < live { unsafe { padded[c].as_ptr().add(1) } } else { zero.as_ptr() });
        let mut want = words;
        want[8 * live..].fill(B128(0));
        check_cols(&want, &cols, "missing-column zero pointers");
    }
}
