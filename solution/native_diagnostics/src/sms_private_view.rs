// Appended only to hashed test views of the actual SMS source. No production hook.
type DiagnosticTrace = (Vec<i16>, Vec<i16>, Vec<i32>);
std::thread_local! {
    static DIAGNOSTIC_TRACE: std::cell::RefCell<DiagnosticTrace> = const { std::cell::RefCell::new((Vec::new(), Vec::new(), Vec::new())) };
}
fn diagnostic_bias(value: i32) {
    DIAGNOSTIC_TRACE.with(|t| t.borrow_mut().2.push(value));
}
fn diagnostic_pending(pend: &[Batch32], half: usize) {
    DIAGNOSTIC_TRACE.with(|t| {
        let mut t = t.borrow_mut();
        for pd in pend {
            for row in &pd.v {
                for &x in &row[16 * half..16 * half + 16] {
                    assert!((x as i32).abs() < 2 * P as i32);
                    t.0.push(x);
                }
            }
        }
    });
}
fn diagnostic_inverse(pd: &Batch32, _: usize, both: bool) {
    assert_eq!(pd.representation, Representation::Coefficients);
    DIAGNOSTIC_TRACE.with(|t| {
        let mut t = t.borrow_mut();
        for row in &pd.v {
            for &x in &row[..if both { 32 } else { 16 }] {
                assert!((x as i32).abs() <= (P as i32 - 1) / 2);
                t.1.push(x);
            }
        }
    });
}
fn diagnostic_worker() -> SmsWorker {
    let mut ws = SmsWorker::new();
    // FIRST must overwrite every digit accumulator before a read.
    unsafe {
        for s in 1..STREAMS {
            fill_stream(&mut ws.acc[0], s, 0x12345678);
        }
    }
    for pd in ws.pend.iter_mut() {
        for row in &mut pd.v {
            row.fill(P as i16);
        }
    }
    ws
}

pub fn diagnostic_sms(
    elements: usize,
    columns: usize,
    pairs: &[u32],
    mur: &[i16],
    mu: &[u32],
    words: &[u128],
) -> (Vec<i16>, Vec<i16>, Vec<i32>, Vec<u32>, Vec<u32>) {
    assert_eq!(bench_mode(), 0);
    assert_eq!(LOCK, 8);
    assert_eq!(PP, 8);
    let keys = SmsKeys {
        pairs,
        mur,
        mu,
        elements,
    };
    let words: Vec<B128> = words.iter().copied().map(B128).collect();
    assert_eq!(words.len(), 4 * elements * columns);
    let cols: Vec<&[B128]> = words.chunks_exact(4 * elements).collect();
    let np = elements / 2;
    let mut full = vec![[0u32; N]; 2 * columns];
    let mut halves = full.clone();
    DIAGNOSTIC_TRACE.with(|t| *t.borrow_mut() = (Vec::new(), Vec::new(), Vec::new()));
    for c in (0..columns).step_by(16) {
        let end = (c + 16).min(columns);
        unsafe {
            group(
                &cols[c..end],
                &keys,
                np,
                0..np,
                RowsOut::Full,
                &mut diagnostic_worker(),
                full.as_mut_ptr().add(2 * c),
                |_| {},
            );
            let state = std::sync::Mutex::new(0u8);
            group(
                &cols[c..end],
                &keys,
                np,
                0..np / 2,
                RowsOut::Half(&state),
                &mut diagnostic_worker(),
                halves.as_mut_ptr().add(2 * c),
                |_| {},
            );
            group(
                &cols[c..end],
                &keys,
                np,
                np / 2..np,
                RowsOut::Half(&state),
                &mut diagnostic_worker(),
                halves.as_mut_ptr().add(2 * c),
                |_| {},
            );
            assert_eq!(*state.lock().unwrap(), 2);
        }
    }
    assert_eq!(full, halves, "full and half work units");
    let (pending, inverse, bias) = DIAGNOSTIC_TRACE.with(|t| std::mem::take(&mut *t.borrow_mut()));
    (
        pending,
        inverse,
        bias,
        full.into_iter().flatten().collect(),
        halves.into_iter().flatten().collect(),
    )
}
