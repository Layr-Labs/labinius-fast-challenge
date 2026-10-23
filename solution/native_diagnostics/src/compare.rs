pub const NAMES: [&str; 2] = ["baseline", "prefix64-parallel"];
const CASE: &str = "partials-switch-context";

pub fn phase_ns(stderr: &str) -> u128 {
    let lines: Vec<_> = stderr.lines().filter(|s| s.starts_with("labinius-profile ")).collect();
    assert_eq!(lines.len(), 1, "one actual child profile is required");
    let tokens: Vec<_> = lines[0].split_whitespace().skip(1).collect();
    assert_eq!(tokens.len(), 14, "complete current production profile");
    let fields: std::collections::BTreeMap<_, _> = tokens.into_iter()
        .map(|s| s.split_once('=').expect("profile field")).collect();
    assert_eq!(fields.len(), 14, "unique production profile fields");
    assert_eq!(fields["threads"], "8", "actual child uses the eight-thread Pool");
    assert!(matches!(fields["fold_gfni"], "true" | "false"));
    for name in ["load_us", "matrix_us", "setup_us", "commit_us", "recombine_us", "partials_us",
        "switch_us", "row_us", "challenges_us", "fold_us", "encode_us", "total_us"] {
        fields[name].parse::<u128>().expect("integer production profile duration");
    }
    (fields["partials_us"].parse::<u128>().unwrap() + fields["switch_us"].parse::<u128>().unwrap()) * 1000
}

// The runner returns the actual child's phase metric and complete proof after normal exit.
pub fn run(call: &mut dyn FnMut(usize, usize) -> (Vec<u8>, u128)) {
    println!("native-diag case={CASE} variants=2 initial_calls=2 warmups=2 samples=16 total_calls=38 metric=child-profile-partials-plus-switch");
    let mut ordinal = 1;
    let (expected, ns) = call(0, ordinal);
    println!("native-diag case={CASE} call={ordinal} variant=baseline phase=calibration elapsed_ns={ns} equality=reference");
    ordinal += 1;
    let (output, ns) = call(1, ordinal);
    assert!(expected == output, "native-diag complete proof mismatch call={ordinal}");
    println!("native-diag case={CASE} call={ordinal} variant=prefix64-parallel phase=validation elapsed_ns={ns} equality=ok");
    drop(output);
    let mut samples = [[0u128; 16]; 2];
    for round in 0..18 {
        let order = if round % 2 == 0 { "AB" } else { "BA" };
        for position in 0..2 {
            let index = if round % 2 == 0 { position } else { 1 - position };
            ordinal += 1;
            let (output, ns) = call(index, ordinal);
            assert!(expected == output, "native-diag complete proof mismatch call={ordinal}");
            let phase = if round < 2 { "warmup" } else { "sample" };
            let trial = if round < 2 { round + 1 } else { round - 1 };
            let name = NAMES[index];
            println!("native-diag case={CASE} call={ordinal} variant={name} phase={phase} round={trial} order={order} position={} elapsed_ns={ns} equality=ok", position + 1);
            if round >= 2 { samples[index][round - 2] = ns; }
            drop(output);
        }
    }
    assert_eq!(ordinal, 38);
    for (name, times) in NAMES.iter().zip(&mut samples) {
        times.sort_unstable();
        let median = (times[7] + times[8]) as f64 / 2.0;
        println!("native-diag case={CASE} variant={name} samples=16 median_ns={median:.1}");
    }
}
