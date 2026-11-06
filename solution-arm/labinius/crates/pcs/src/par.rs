//! The worker count and a scoped parallel loop for the prover's data-parallel passes.
//!
//! Every pass splits its index range into contiguous blocks, one per worker, on boundaries the
//! kernel asks for; workers never share an output element, and passes that reduce (XOR sums over
//! GF(2)) combine their per-worker partial sums afterwards, so the result does not depend on the
//! worker count.

use rayon::prelude::*;
use std::sync::OnceLock;

/// Worker threads: `LABINIUS_THREADS` when it holds a positive integer, otherwise the available
/// parallelism. The variable is a development knob for matching the scored machine's CPU count
/// on a larger host; the scoring sandbox starts the prover with a cleared environment.
pub fn threads() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("LABINIUS_THREADS")
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .filter(|&n| n > 0)
            .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get()))
    })
}

fn pool() -> &'static rayon::ThreadPool {
    static POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();
    POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads())
            .build()
            .expect("create prover workers")
    })
}

/// Process borrowed, disjoint work items and wait for every item before returning.
pub fn for_each<T: Send>(items: impl IntoIterator<Item = T>, f: impl Fn(T) + Sync) {
    let items: Vec<T> = items.into_iter().collect();
    if items.len() <= 1 {
        items.into_iter().for_each(f);
    } else {
        let f = &f;
        pool().install(move || items.into_par_iter().for_each(f));
    }
}

/// `[start, end)` of block `t` of `parts` over `n` items, block edges on multiples of `align`.
#[inline]
pub fn block(n: usize, parts: usize, t: usize, align: usize) -> (usize, usize) {
    let units = n / align;
    let lo = units * t / parts;
    let hi = units * (t + 1) / parts;
    let end = if t + 1 == parts { n } else { hi * align };
    (lo * align, end)
}

/// `s` cut into the disjoint, ascending `[start, end)` ranges of `edges`.
pub fn split_mut<'a, T>(s: &'a mut [T], edges: &[(usize, usize)]) -> Vec<&'a mut [T]> {
    let mut out = Vec::with_capacity(edges.len());
    let mut rest = s;
    let mut base = 0;
    for &(lo, hi) in edges {
        let (_, tail) = core::mem::take(&mut rest).split_at_mut(lo - base);
        let (mid, tail) = tail.split_at_mut(hi - lo);
        out.push(mid);
        rest = tail;
        base = hi;
    }
    out
}

/// Workers for `n` items when a block below `min` items is not worth a thread.
#[inline]
pub fn workers(n: usize, min: usize) -> usize {
    (n / min.max(1)).clamp(1, threads())
}

/// Run each block on reusable workers and collect results in block order.
pub fn map_blocks<R: Send>(
    n: usize,
    parts: usize,
    align: usize,
    f: impl Fn(usize, usize, usize) -> R + Sync,
) -> Vec<R> {
    if parts <= 1 {
        return vec![f(0, 0, n)];
    }
    let f = &f;
    pool().install(|| {
        (0..parts)
            .into_par_iter()
            .map(|t| {
                let (s, e) = block(n, parts, t, align);
                f(t, s, e)
            })
            .collect()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repeated_passes_preserve_results_and_borrowed_outputs() {
        let mut values = vec![0usize; 1003];
        for round in 1..=20 {
            for_each(values.chunks_mut(64).enumerate(), |(i, chunk)| {
                for (j, value) in chunk.iter_mut().enumerate() {
                    *value += round * (64 * i + j);
                }
            });
            let results = map_blocks(values.len(), 9, 64, |t, start, end| {
                (t, values[start..end].iter().sum::<usize>())
            });
            let expected: Vec<_> = (0..9)
                .map(|t| {
                    let (start, end) = block(values.len(), 9, t, 64);
                    (t, (start..end).sum::<usize>() * round * (round + 1) / 2)
                })
                .collect();
            assert_eq!(results, expected);
        }
    }

    #[test]
    fn nested_and_concurrent_dispatches_complete() {
        std::thread::scope(|scope| {
            let handles: Vec<_> = (0..3)
                .map(|caller| {
                    scope.spawn(move || {
                        map_blocks(16, 4, 1, |t, _, _| {
                            map_blocks(8, 4, 1, |u, _, _| caller + t + u)
                        })
                    })
                })
                .collect();
            for (caller, handle) in handles.into_iter().enumerate() {
                let expected: Vec<Vec<_>> = (0..4)
                    .map(|t| (0..4).map(|u| caller + t + u).collect())
                    .collect();
                assert_eq!(handle.join().unwrap(), expected);
            }
        });
    }

    #[test]
    fn worker_panic_propagates_and_pool_remains_usable() {
        let result = std::panic::catch_unwind(|| {
            map_blocks(8, 4, 1, |t, _, _| {
                assert_ne!(t, 2, "test worker panic");
                t
            })
        });
        assert!(result.is_err());
        assert_eq!(map_blocks(8, 4, 1, |t, _, _| t), vec![0, 1, 2, 3]);
    }

    #[test]
    fn blocks_tile_the_range() {
        for n in [0usize, 1, 7, 8, 63, 64, 65, 1000, 4096] {
            for parts in 1..=9 {
                for align in [1usize, 8, 64] {
                    let mut next = 0;
                    for t in 0..parts {
                        let (s, e) = block(n, parts, t, align);
                        assert_eq!(s, next);
                        assert!(e >= s);
                        if t + 1 < parts {
                            assert_eq!(e % align, 0);
                        }
                        next = e;
                    }
                    assert_eq!(next, n);
                }
            }
        }
    }
}
