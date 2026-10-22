use labinius::fields::scalar::B128;
use labinius::simd::ntt::bin_asm;
use labinius::simd::transpose_f162::{slice_b128_cols16_into, BinaryIndex32};

#[repr(C, align(128))]
struct Acc([i32; 648 * 32]);

fn initialize(acc: &mut Acc, case: usize) {
    for (index, value) in acc.0.iter_mut().enumerate() {
        *value = if case == 0 { 0 } else if index % 2 == 0 { 10_000_000 } else { -10_000_000 };
    }
}

#[inline(never)]
unsafe fn check_prime<const Q: u16>(keys: &[u32], columns: &[&[B128]]) {
    let mut indices: [BinaryIndex32; 2] = core::array::from_fn(|_| BinaryIndex32 { rows: [[0; 64]; 162] });
    let mut reference = Acc([0; 648 * 32]);
    let mut output = Acc([0; 648 * 32]);
    for batch in [0, 128, 1024, 2046] {
        for (half, index) in indices.iter_mut().enumerate() {
            let pointers = core::array::from_fn(|column| columns[column].as_ptr().add(8 * (batch + half)));
            slice_b128_cols16_into(&pointers, index);
        }
        let ins = [&indices[0], &indices[1]];
        let prime_offset = if Q == 3889 { 0 } else { 648 * 4 };
        let key = keys.as_ptr().add(batch * 648 * 4 + prime_offset);
        for red in [false, true] {
            for initial in 0..2 {
                initialize(&mut reference, initial);
                bin_asm::ntt_bin_batch32_pair::<Q>(ins, key, reference.0.as_mut_ptr(), red);
                initialize(&mut output, initial);
                crate::omega_path::omega::ntt_bin_batch32_pair::<Q>(ins, key, output.0.as_mut_ptr(), red);
                assert!(output.0 == reference.0,
                    "native omega pair mismatch q={Q} batch={batch} red={red} initial={initial}");
            }
        }
    }
    println!("native-diag phase=pair-validation q={Q} input_cases=4 red_modes=2 initial_accumulators=2 baseline_calls=16 candidate_calls=16 candidate_checks=16 omega=exact_i32 equality=ok");
}

#[inline(never)]
pub fn run(keys: &[u32], columns: &[&[B128]]) {
    unsafe {
        check_prime::<3889>(keys, columns);
        check_prime::<9721>(keys, columns);
    }
    println!("native-diag phase=pair-validation total_calls=64 candidate_checks=32 timing=excluded buffers=stack_released_before_calibration");
}
