use std::hint::black_box;
use std::time::Instant;

// Stop each timer before the complete-output check and logging.
pub fn run<T: PartialEq>(case: &str, variants: &mut [(&str, &mut dyn FnMut() -> T); 4]) {
    const ORDERS: [[usize; 4]; 8] = [
        [0, 1, 3, 2], [1, 2, 0, 3], [2, 3, 1, 0], [3, 0, 2, 1],
        [3, 0, 2, 1], [2, 3, 1, 0], [1, 2, 0, 3], [0, 1, 3, 2],
    ];
    let mut expected = None;
    let mut samples = [[0u128; 8]; 4];
    println!("native-diag case={case} variants=4 validation_passes=1 warmups=2 samples=8");
    for (index, (name, run)) in variants.iter_mut().enumerate() {
        let clock = Instant::now();
        let output = black_box(run());
        let ns = clock.elapsed().as_nanos();
        if let Some(reference) = &expected {
            assert!(reference == &output, "native-diag mismatch case={case} variant={name}");
        } else {
            expected = Some(output);
        }
        let phase = if index == 0 { "calibration" } else { "validation" };
        println!("native-diag case={case} variant={name} phase={phase} elapsed_ns={ns} equality=ok");
    }
    for round in 0..10 {
        for position in 0..4 {
            let index = match round {
                0 => position,
                1 => 3 - position,
                _ => ORDERS[round - 2][position],
            };
            let (name, run) = &mut variants[index];
            let clock = Instant::now();
            let output = black_box(run());
            let ns = clock.elapsed().as_nanos();
            assert!(expected.as_ref() == Some(&output), "native-diag mismatch case={case} variant={name}");
            let phase = if round < 2 { "warmup" } else { "sample" };
            let trial = if round < 2 { round + 1 } else { round - 1 };
            println!("native-diag case={case} variant={name} phase={phase} round={trial} position={} elapsed_ns={ns} equality=ok", position + 1);
            if round >= 2 {
                samples[index][round - 2] = ns;
            }
        }
    }
    for ((name, _), times) in variants.iter().zip(&mut samples) {
        times.sort_unstable();
        let median = (times[3] + times[4]) as f64 / 2.0;
        println!("native-diag case={case} variant={name} samples=8 median_ns={median:.1}");
    }
}
