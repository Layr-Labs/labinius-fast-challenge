use std::hint::black_box;
use std::time::Instant;

fn timed<T>(run: &mut dyn FnMut() -> T) -> (T, u128) {
    let clock = Instant::now();
    let output = black_box(run());
    let ns = clock.elapsed().as_nanos();
    (output, ns)
}

// Stop each timer before the complete-output check and logging.
pub fn run<T: PartialEq>(
    case: &str,
    baseline: &mut dyn FnMut() -> T,
    candidate: &mut dyn FnMut() -> T,
) {
    println!("native-diag case={case} variants=2 initial_calls=2 warmups=2 samples=16 total_calls=38");
    let (expected, ns) = timed(baseline);
    println!("native-diag case={case} variant=baseline phase=calibration elapsed_ns={ns} equality=reference");
    let (output, ns) = timed(candidate);
    assert!(expected == output, "native-diag mismatch case={case} variant=parts1-fused-output");
    println!("native-diag case={case} variant=parts1-fused-output phase=validation elapsed_ns={ns} equality=ok");
    drop(output);
    let mut variants: [(&str, &mut dyn FnMut() -> T); 2] = [
        ("baseline", &mut *baseline), ("parts1-fused-output", &mut *candidate),
    ];
    let mut samples = [[0u128; 16]; 2];
    for round in 0..18 {
        let order = if round % 2 == 0 { "AB" } else { "BA" };
        for position in 0..2 {
            let index = if round % 2 == 0 { position } else { 1 - position };
            let (name, run) = &mut variants[index];
            let (output, ns) = timed(*run);
            assert!(expected == output, "native-diag mismatch case={case} variant={name}");
            let phase = if round < 2 { "warmup" } else { "sample" };
            let trial = if round < 2 { round + 1 } else { round - 1 };
            println!("native-diag case={case} variant={name} phase={phase} round={trial} order={order} position={} elapsed_ns={ns} equality=ok", position + 1);
            if round >= 2 {
                samples[index][round - 2] = ns;
            }
            drop(output);
        }
    }
    for ((name, _), times) in variants.iter().zip(&mut samples) {
        times.sort_unstable();
        let median = (times[7] + times[8]) as f64 / 2.0;
        println!("native-diag case={case} variant={name} samples=16 median_ns={median:.1}");
    }
}
