//! The little the parallel paths share: how many threads to use and how to cut a range.

/// `available_parallelism`, or 1.
pub fn threads() -> usize {
    std::thread::available_parallelism().map_or(1, |n| n.get())
}

/// `n` items cut into at most `parts` contiguous ranges whose starts are multiples of `align`;
/// the last range takes the remainder. Empty ranges are not returned.
pub fn ranges(n: usize, parts: usize, align: usize) -> Vec<core::ops::Range<usize>> {
    let align = align.max(1);
    let parts = parts.max(1);
    let per = n.div_ceil(parts).div_ceil(align) * align;
    let mut out = Vec::with_capacity(parts);
    let mut start = 0;
    while start < n {
        let end = (start + per).min(n);
        out.push(start..end);
        start = end;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::ranges;

    #[test]
    fn ranges_cover_once() {
        for (n, parts, align) in [(0, 4, 64), (1, 8, 64), (1000, 8, 64), (4096, 3, 32), (5, 8, 1)] {
            let r = ranges(n, parts, align);
            assert!(r.len() <= parts);
            let mut at = 0;
            for x in &r {
                assert_eq!(x.start, at);
                assert!(x.start % align == 0);
                assert!(x.end > x.start);
                at = x.end;
            }
            assert_eq!(at, n);
        }
    }
}
