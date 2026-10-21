//! Scalar, public-key-only certificate for the parent's unbiased i32 fold.
//! A returned byte applies to one complete batch: bit 0 folds every P0 vector,
//! bit 1 every P1 vector. It is valid only for this exact ordered key slice and
//! its complete batch count, with the parent's seven-batch tail reset points.
//! Never use a prefix for a shorter invocation: its last tail flush differs.
use super::{KeyBatch, KBLK, ABLK, CHUNK_BATCHES, P0, P1, acc_after_reduce_c, bias};
use crate::simd::ntt::bin_asm::proved_binary_row_bounds;

type Bounds = [[i64; 16]; 24 * ABLK];
const LIMIT: i64 = i32::MAX as i64;

fn growth(key: &KeyBatch, q: u16, rows: &[i32; 648]) -> Bounds {
    let mut out = [[0; 16]; 24 * ABLK];
    for blk in 0..24 { for group in 0..4 { for stream in 0..2 {
        let count = if group < 3 { 8 } else { 4 };
        let offset = if group < 3 { 16 * group + 8 * stream } else { 48 + 4 * stream };
        for lane in 0..16 {
            let row = if group < 3 { 8 * group + lane / 2 } else { 24 + lane / 4 };
            let mut sum = 0;
            for vector in 0..count { for half in 0..2 {
                let k = key.v[blk * KBLK + offset + vector][2 * lane + half] as i64;
                assert!(k.abs() <= (q as i64 - 1) / 2,
                    "fold certificate: non-centered arranged key");
                if row == 27 { assert_eq!(k, 0, "fold certificate: nonzero pad key"); }
                sum += k.abs();
            }}
            if row < 27 {
                assert!((0..=32768).contains(&rows[27 * blk + row]));
                out[blk * ABLK + 2 * group + stream][lane] =
                    sum * rows[27 * blk + row] as i64;
            }
        }
    }}}
    out
}
fn add(state: &mut Bounds, delta: &Bounds) {
    for v in 0..24 * ABLK { for lane in 0..16 {
        state[v][lane] += delta[v][lane];
        // The sum of absolute products bounds every partial vpdpwssd sum,
        // including its individual two-product contribution.
        assert!(state[v][lane] <= LIMIT, "fold certificate: i32 accumulation not certified");
    }}
}
fn future_overflow(state: &Bounds, next: &Bounds, clear_tails: bool) -> bool {
    (0..24 * ABLK).any(|v| (0..16).any(|lane| {
        let current = if clear_tails && v % 2 == 1 { 0 } else { state[v][lane] };
        current + next[v][lane] > LIMIT
    }))
}
fn tail_max(state: &Bounds) -> i64 {
    let mut max = 0;
    for blk in 0..24 { for group in 0..4 {
        let width = if group < 3 { 2 } else { 4 };
        let v = blk * ABLK + 2 * group + 1;
        for start in (0..16).step_by(width) {
            let row = state[v][start..start + width].iter().sum();
            max = max.max(row);
        }
    }}
    max
}
fn clear_tails(state: &mut Bounds) {
    for v in (1..24 * ABLK).step_by(2) { state[v] = [0; 16]; }
}
fn certify_prime(keys: &[KeyBatch], prime: usize, rows: &[i32; 648], flags: &mut [u8]) {
    let q = [P0, P1][prime];
    let reduced = acc_after_reduce_c(q);
    let mut state = [[0; 16]; 24 * ABLK];
    if flags.is_empty() { return; }
    let mut current = growth(&keys[prime], q, rows);
    for b in 0..flags.len() {
        add(&mut state, &current);
        let end = (b + 1) % CHUNK_BATCHES == 0 || b + 1 == flags.len();
        let next = if b + 1 < flags.len() {
            Some(growth(&keys[2 * (b + 1) + prime], q, rows))
        } else { None };
        let needs_next = next.as_ref().is_some_and(|n| future_overflow(&state, n, end));
        let needs_tail = end && tail_max(&state) > bias(q);
        if needs_next || needs_tail {
            flags[b] |= 1 << prime;
            state.fill([reduced; 16]);
        }
        if end {
            assert!(tail_max(&state) <= bias(q), "fold certificate: raw tail finish not certified");
            clear_tails(&mut state);
        }
        if let Some(n) = next {
            // Catch a key batch too large even immediately after folding,
            // before returning any schedule to a caller.
            assert!(!future_overflow(&state, &n, false),
                "fold certificate: next batch not certified after reduction");
            current = n;
        }
    }
    // finish_base already folds individual lanes before adding at most four;
    // the parent's existing bound proves that final path for every i32 lane.
}

/// Certify a complete invocation before use; panic on invalid public keys or
/// an unsupported bound. A caller choosing a generic fallback must do so for
/// the entire invocation, before any witness accumulation starts.
///
/// This assumes zero initial accumulators and the original unbiased
/// reduce_vec_c. It must not be combined with an affine/seeded fold without
/// a new proof that incorporates its seeds and dynamic reduction counts.
pub fn certify_global_fold_schedule(keys: &[KeyBatch]) -> Vec<u8> {
    assert_eq!(keys.len() % 2, 0, "fold certificate: incomplete prime pair");
    assert_eq!(CHUNK_BATCHES, 7, "fold certificate: tail chunk contract changed");
    assert_eq!((KBLK, ABLK), (56, 8));
    let mut flags = vec![0; keys.len() / 2];
    if flags.is_empty() { return flags; }
    certify_prime(keys, 0, proved_binary_row_bounds::<P0>(), &mut flags);
    certify_prime(keys, 1, proved_binary_row_bounds::<P1>(), &mut flags);
    flags
}
