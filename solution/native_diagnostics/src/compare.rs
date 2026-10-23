use std::hint::black_box;
use std::time::Instant;

// Counter setup, control, reads, output checks and drops stay outside this timer.
fn timed<T>(run: &mut dyn FnMut() -> T) -> (T, u128) {
    let clock = Instant::now();
    let output = black_box(run());
    (output, clock.elapsed().as_nanos())
}

pub fn run<T: PartialEq>(
    case: &str,
    run: &mut dyn FnMut() -> T,
    profile: &mut dyn FnMut(usize, &mut dyn FnMut() -> (T, u128)) -> (T, u128),
) {
    println!("native-diag case={case} variant=baseline full_calls=13 output_checks=12 reference=1 warmup_calls=2 unprofiled_samples=5 counter_requested_samples=5");
    let (reference, ns) = timed(run);
    println!("native-diag case={case} variant=baseline phase=calibration call=1 elapsed_ns={ns} equality=reference counters_requested=false");
    let mut samples = [[0u128; 5]; 2];
    for index in 0..12 {
        let (phase, sample, requested) = match index {
            0..=1 => ("warmup", index + 1, false),
            2..=6 => ("unprofiled", index - 1, false),
            _ => ("counter-requested", index - 6, true),
        };
        let (output, ns) = if requested {
            profile(sample, &mut || timed(run))
        } else {
            timed(run)
        };
        assert!(reference == output, "native-diag mismatch case={case} phase={phase} sample={sample}");
        println!("native-diag case={case} variant=baseline phase={phase} call={} sample={sample} elapsed_ns={ns} equality=ok counters_requested={requested}", index + 2);
        if index >= 2 {
            samples[usize::from(requested)][sample - 1] = ns;
        }
        drop(output);
    }
    for (phase, times) in ["unprofiled", "counter-requested"].iter().zip(&mut samples) {
        times.sort_unstable();
        println!("native-diag case={case} variant=baseline phase={phase} samples=5 median_ns={}", times[2]);
    }
}
