//! Helpers for the large buffers the timed prover allocates.

/// Capacity for `count` elements, length set, contents uninitialized.
///
/// # Safety
/// Every element must be written before it is read. `T` must be safe to drop
/// uninitialized: it is `Copy` at every call site, so drop does not read it.
pub unsafe fn uninit<T>(count: usize) -> Vec<T> {
    let mut v = Vec::with_capacity(count);
    if count == 0 {
        return v;
    }
    unsafe { v.set_len(count) };
    v
}

/// A zeroed buffer. Large polynomial tables are cleared once and then overwritten
/// only in part, so the zeros are load-bearing.
pub fn zeroed<T: Copy>(count: usize) -> Vec<T> {
    let mut v = Vec::with_capacity(count);
    if count == 0 {
        return v;
    }
    unsafe {
        core::ptr::write_bytes(v.as_mut_ptr(), 0, count);
        v.set_len(count);
    }
    v
}

/// How many workers to use for `items` independent chunks of work.
pub fn workers(items: usize) -> usize {
    if items < 2 {
        return 1;
    }
    let n = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(1);
    n.max(1).min(items)
}
