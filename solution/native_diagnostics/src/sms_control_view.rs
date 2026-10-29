// Appended only to the complete generated SMS module; no production hook.
pub fn diagnostic_sms_rows(
    elements: usize, columns: usize, pairs: &[u32], mur: &[i16], mu: &[u32],
    words: &[u128], canonical: bool,
) -> Vec<Vec<[u32; N]>> {
    assert_eq!(bench_mode(), 0); assert_eq!(LOCK, 8); assert_eq!(PP, 8);
    assert!(!canonical, "old centered control");
    let keys = SmsKeys { pairs, mur, mu, elements };
    let words: Vec<B128> = words.iter().copied().map(B128).collect();
    assert_eq!(words.len(), 4 * elements * columns);
    let cols: Vec<&[B128]> = words.chunks_exact(4 * elements).collect();
    let np = elements / 2;
    let mut ws = SmsWorker::new();
    let mut results = Vec::new();
    for pass in 0..3 {
        let mut rows = vec![[0u32; N]; 2 * columns];
        for c in (0..columns).step_by(16) {
            let end = (c + 16).min(columns);
            let dst = unsafe { rows.as_mut_ptr().add(2 * c) };
            unsafe {
                if pass == 0 {
                    group(&cols[c..end], &keys, np, 0..np, RowsOut::Full, &mut ws, dst, |_| {});
                } else {
                    let state = std::sync::Mutex::new(0u8);
                    for half in if pass == 1 { [0, 1] } else { [1, 0] } {
                        group(&cols[c..end], &keys, np, half * np / 2..(half + 1) * np / 2,
                            RowsOut::Half(&state), &mut ws, dst, |_| {});
                    }
                    assert_eq!(*state.lock().unwrap(), 2);
                }
            }
        }
        results.push(rows);
    }
    results
}
