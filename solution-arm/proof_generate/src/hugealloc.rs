//! Large buffers on transparent huge pages, with medium blocks recycled (Linux only).
//!
//! The prover allocates roughly 1.5 GB of fresh buffers per run. Three tiers:
//!
//! - **Huge** (>= 1 MiB, alignment <= 2 MiB): mapped directly, 2 MiB aligned, `MADV_HUGEPAGE`.
//! - **Medium** (64 KiB .. 1 MiB, alignment <= 64 KiB): carved from 2 MiB huge-page chunks and
//!   recycled through per-class free lists (quarter-octave size classes). glibc would serve these
//!   from per-thread arenas it trims back to the kernel (`MADV_DONTNEED`) when freed, so the next
//!   allocation re-faults every page; the free list keeps the pages resident and avoids that
//!   fault storm.
//! - **Small** (< 64 KiB): delegated to the system allocator.
//!
//! Blocks are carved with a 64-byte gap between them (not at power-of-two offsets) so that a set
//! of buffers written together never share one L1 set. This mirrors the x86 track's allocator,
//! which the hot-path notes credit with removing a measurable allocation stall.

use std::alloc::{GlobalAlloc, Layout, System};

pub struct Allocator;

#[cfg(target_os = "linux")]
mod imp {
    use super::*;
    use core::cell::UnsafeCell;
    use core::sync::atomic::{AtomicBool, Ordering};

    pub const HUGE: usize = 2 << 20;
    pub const THRESHOLD: usize = 1 << 20;
    /// Medium blocks span from this size up to `THRESHOLD`.
    pub const MEDIUM: usize = 1 << 16;
    /// Gap between consecutively carved blocks (avoids L1-set conflicts).
    const GAP: usize = 64;
    /// One free list per quarter-octave size class in `[MEDIUM, THRESHOLD)`.
    const LISTS: usize = 4 * ((THRESHOLD / MEDIUM).trailing_zeros() as usize + 1);

    const PROT_READ_WRITE: i32 = 0x1 | 0x2;
    const MAP_PRIVATE_ANONYMOUS: i32 = 0x02 | 0x20;
    const MAP_NORESERVE: i32 = 0x4000;
    const MADV_HUGEPAGE: i32 = 14;

    extern "C" {
        fn mmap(addr: *mut u8, len: usize, prot: i32, flags: i32, fd: i32, off: i64) -> *mut u8;
        fn munmap(addr: *mut u8, len: usize) -> i32;
        fn madvise(addr: *mut u8, len: usize, advice: i32) -> i32;
    }

    #[inline]
    pub fn is_big(layout: &Layout) -> bool {
        layout.size() >= THRESHOLD && layout.align() <= HUGE
    }

    #[inline]
    pub fn is_medium(layout: &Layout) -> bool {
        layout.size() >= MEDIUM && layout.size() < THRESHOLD && layout.align() <= MEDIUM
    }

    #[inline]
    pub fn span(size: usize) -> usize {
        (size + HUGE - 1) & !(HUGE - 1)
    }

    /// A medium request's rounded block size and its free-list index.
    #[inline]
    pub fn class(size: usize) -> (usize, usize) {
        let k = (usize::BITS - 1 - (size - 1).leading_zeros()) as usize;
        let step = 1usize << (k - 2);
        let bytes = (size + step - 1) & !(step - 1);
        (bytes, 4 * (k - MEDIUM.trailing_zeros() as usize + 1) + (bytes >> (k - 2)) - 5)
    }

    struct SpinLock(AtomicBool);

    impl SpinLock {
        #[inline]
        fn lock(&self) {
            while self
                .0
                .compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed)
                .is_err()
            {
                core::hint::spin_loop();
            }
        }
        #[inline]
        fn unlock(&self) {
            self.0.store(false, Ordering::Release);
        }
    }

    /// A class's free blocks: an intrusive LIFO list (a free block's first word is the next).
    struct FreeList {
        lock: SpinLock,
        head: UnsafeCell<*mut u8>,
    }

    /// The huge-page chunk that medium blocks are carved from.
    struct Carve {
        lock: SpinLock,
        at: UnsafeCell<usize>,
        end: UnsafeCell<usize>,
    }

    // SAFETY: both fields are only mutated while the spinlock is held; the raw pointers are
    // treated as opaque storage and never dereferenced outside their lock's critical section.
    unsafe impl Sync for FreeList {}
    // SAFETY: as above; the carve cursor is only read/written under `lock`.
    unsafe impl Sync for Carve {}

    static FREE: [FreeList; LISTS] = [const {
        FreeList {
            lock: SpinLock(AtomicBool::new(false)),
            head: UnsafeCell::new(core::ptr::null_mut()),
        }
    }; LISTS];
    static CARVE: Carve = Carve {
        lock: SpinLock(AtomicBool::new(false)),
        at: UnsafeCell::new(0),
        end: UnsafeCell::new(0),
    };

    /// A medium block for `size` bytes at `align`, plus whether it is fresh (still zeroed).
    ///
    /// # Safety
    /// The returned pointer (when non-null) is valid for `size` bytes and matches `align`.
    unsafe fn medium_alloc(size: usize, align: usize) -> (*mut u8, bool) {
        let (bytes, i) = class(size);
        let list = &FREE[i];
        list.lock.lock();
        let head = *list.head.get();
        if !head.is_null() && head as usize % align == 0 {
            *list.head.get() = *(head as *mut *mut u8);
            list.lock.unlock();
            return (head, false);
        }
        list.lock.unlock();
        let align = align.max(GAP);
        CARVE.lock.lock();
        let mut at = (*CARVE.at.get() + align - 1) & !(align - 1);
        if *CARVE.end.get() == 0 || at + bytes > *CARVE.end.get() {
            let chunk = map(HUGE) as usize;
            if chunk == 0 {
                CARVE.lock.unlock();
                return (core::ptr::null_mut(), false);
            }
            // Fault the fresh chunk before publishing blocks from it to several workers; the
            // volatile store preserves `alloc_zeroed`'s fresh-zeroed contract.
            core::ptr::write_volatile(chunk as *mut u8, 0);
            at = chunk;
            *CARVE.end.get() = chunk + HUGE;
        }
        *CARVE.at.get() = at + bytes + GAP;
        CARVE.lock.unlock();
        (at as *mut u8, true)
    }

    /// # Safety
    /// `p` must be a block returned by `medium_alloc` for `size`.
    unsafe fn medium_free(p: *mut u8, size: usize) {
        let list = &FREE[class(size).1];
        list.lock.lock();
        *(p as *mut *mut u8) = *list.head.get();
        *list.head.get() = p;
        list.lock.unlock();
    }

    /// A fresh (zeroed) 2 MiB-aligned mapping of `span(size)` bytes, or null.
    ///
    /// # Safety
    /// Caller must ensure the returned pointer (if non-null) is used only for writes of the
    /// requested size and freed with `unmap`.
    pub unsafe fn map(size: usize) -> *mut u8 {
        let len = span(size);
        let raw = mmap(
            core::ptr::null_mut(),
            len + HUGE,
            PROT_READ_WRITE,
            MAP_PRIVATE_ANONYMOUS | MAP_NORESERVE,
            -1,
            0,
        );
        if raw as isize == -1 || raw.is_null() {
            return core::ptr::null_mut();
        }
        let raw = raw as usize;
        let aligned = (raw + HUGE - 1) & !(HUGE - 1);
        let head = aligned - raw;
        if head > 0 {
            munmap(raw as *mut _, head);
        }
        let tail = HUGE - head;
        if tail > 0 {
            munmap((aligned + len) as *mut _, tail);
        }
        // Advisory only: without THP the block is ordinary pages.
        madvise(aligned as *mut u8, len, MADV_HUGEPAGE);
        aligned as *mut u8
    }

    /// # Safety
    /// `ptr`/`size` must match a prior `map`.
    pub unsafe fn unmap(ptr: *mut u8, size: usize) {
        munmap(ptr, span(size));
    }

    /// # Safety
    /// See `medium_alloc`.
    pub unsafe fn med_alloc(size: usize, align: usize) -> (*mut u8, bool) {
        medium_alloc(size, align)
    }

    /// # Safety
    /// See `medium_free`.
    pub unsafe fn med_free(ptr: *mut u8, size: usize) {
        medium_free(ptr, size);
    }
}

#[cfg(target_os = "linux")]
unsafe impl GlobalAlloc for Allocator {
    #[inline]
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if imp::is_big(&layout) {
            imp::map(layout.size())
        } else if imp::is_medium(&layout) {
            imp::med_alloc(layout.size(), layout.align()).0
        } else {
            System.alloc(layout)
        }
    }

    #[inline]
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if imp::is_big(&layout) {
            imp::map(layout.size())
        } else if imp::is_medium(&layout) {
            let (p, fresh) = imp::med_alloc(layout.size(), layout.align());
            if !p.is_null() && !fresh {
                core::ptr::write_bytes(p, 0, layout.size());
            }
            p
        } else {
            System.alloc_zeroed(layout)
        }
    }

    #[inline]
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if imp::is_big(&layout) {
            imp::unmap(ptr, layout.size())
        } else if imp::is_medium(&layout) {
            imp::med_free(ptr, layout.size())
        } else {
            System.dealloc(ptr, layout)
        }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new = Layout::from_size_align_unchecked(new_size, layout.align());
        let (old_big, new_big) = (imp::is_big(&layout), imp::is_big(&new));
        let (old_med, new_med) = (imp::is_medium(&layout), imp::is_medium(&new));
        if !old_big && !new_big && !old_med && !new_med {
            return System.realloc(ptr, layout, new_size);
        }
        // Same huge span: keep the mapping in place.
        if old_big && new_big && imp::span(new_size) == imp::span(layout.size()) {
            return ptr;
        }
        // Otherwise allocate fresh and copy (medium class changes just allocate a new block).
        let out = self.alloc(new);
        if !out.is_null() {
            core::ptr::copy_nonoverlapping(ptr, out, layout.size().min(new_size));
            self.dealloc(ptr, layout);
        }
        out
    }
}

#[cfg(not(target_os = "linux"))]
unsafe impl GlobalAlloc for Allocator {
    #[inline]
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        System.alloc(layout)
    }
    #[inline]
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        System.alloc_zeroed(layout)
    }
    #[inline]
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        System.realloc(ptr, layout, new_size)
    }
}
