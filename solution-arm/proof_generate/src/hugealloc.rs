//! Large buffers on transparent huge pages (Linux only).
//!
//! The prover allocates about 1.5 GB of fresh buffers per run (the witness table, the switch
//! polynomials, the transform scratch), and with 4 KiB pages every one of them is faulted in
//! and zeroed a page at a time, with a TLB entry per page during the transforms and sumcheck.
//! Allocations of at least [`THRESHOLD`] bytes are therefore mapped directly, 2 MiB aligned, and
//! marked `MADV_HUGEPAGE`, so the kernel backs them with 2 MiB pages where it can. Anonymous
//! mappings are zero-filled, so `alloc_zeroed` costs nothing extra. Smaller requests, and every
//! request on other targets, go to the system allocator unchanged. The advice is a hint: if the
//! kernel has no huge page at hand it falls back to normal pages, so behaviour never changes.

use std::alloc::{GlobalAlloc, Layout, System};

pub struct Allocator;

#[cfg(target_os = "linux")]
mod imp {
    use super::*;

    pub const HUGE: usize = 2 << 20;
    pub const THRESHOLD: usize = 1 << 20;
    const PROT_READ_WRITE: i32 = 0x1 | 0x2;
    const MAP_PRIVATE_ANONYMOUS: i32 = 0x02 | 0x20;
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
    fn span(size: usize) -> usize {
        (size + HUGE - 1) & !(HUGE - 1)
    }

    use core::sync::atomic::{AtomicBool, Ordering};
    static LOCK: AtomicBool = AtomicBool::new(false);
    static mut FREE: [(usize, usize); 16] = [(0, 0); 16];

    #[inline]
    unsafe fn take_recycled(size: usize) -> *mut u8 {
        let need = span(size);
        while LOCK.swap(true, Ordering::Acquire) {
            core::hint::spin_loop();
        }
        let mut out = core::ptr::null_mut();
        for i in 0..16 {
            let (addr, rem) = FREE[i];
            if rem >= need {
                out = addr as *mut u8;
                FREE[i] = (addr + need, rem - need);
                break;
            }
        }
        LOCK.store(false, Ordering::Release);
        out
    }

    pub unsafe fn map(size: usize) -> *mut u8 {
        let recycled = take_recycled(size);
        if !recycled.is_null() {
            return recycled;
        }
        let len = span(size);
        let raw = mmap(core::ptr::null_mut(), len + HUGE, PROT_READ_WRITE, MAP_PRIVATE_ANONYMOUS, -1, 0);
        if raw as isize == -1 {
            return core::ptr::null_mut();
        }
        let start = raw as usize;
        let aligned = (start + HUGE - 1) & !(HUGE - 1);
        let head = aligned - start;
        if head > 0 {
            munmap(raw, head);
        }
        let tail = HUGE - head;
        if tail > 0 {
            munmap((aligned + len) as *mut u8, tail);
        }
        madvise(aligned as *mut u8, len, MADV_HUGEPAGE);
        aligned as *mut u8
    }

    pub unsafe fn map_zeroed(size: usize) -> *mut u8 {
        let recycled = take_recycled(size);
        if !recycled.is_null() {
            core::ptr::write_bytes(recycled, 0, size);
            return recycled;
        }
        map(size)
    }

    pub unsafe fn unmap(ptr: *mut u8, size: usize) {
        let len = span(size);
        let p = ptr as usize;
        while LOCK.swap(true, Ordering::Acquire) {
            core::hint::spin_loop();
        }
        let mut stored = false;
        for i in 0..16 {
            let (addr, rem) = FREE[i];
            if rem > 0 && addr + rem == p {
                FREE[i] = (addr, rem + len);
                stored = true;
                break;
            } else if rem > 0 && p + len == addr {
                FREE[i] = (p, rem + len);
                stored = true;
                break;
            }
        }
        if !stored {
            for i in 0..16 {
                if FREE[i].1 == 0 {
                    FREE[i] = (p, len);
                    stored = true;
                    break;
                }
            }
        }
        LOCK.store(false, Ordering::Release);
        if !stored {
            munmap(ptr, len);
        }
    }
}

#[cfg(target_os = "linux")]
unsafe impl GlobalAlloc for Allocator {
    #[inline]
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if imp::is_big(&layout) {
            imp::map(layout.size())
        } else {
            System.alloc(layout)
        }
    }

    #[inline]
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if imp::is_big(&layout) {
            imp::map_zeroed(layout.size())
        } else {
            System.alloc_zeroed(layout)
        }
    }

    #[inline]
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if imp::is_big(&layout) {
            imp::unmap(ptr, layout.size())
        } else {
            System.dealloc(ptr, layout)
        }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new = Layout::from_size_align_unchecked(new_size, layout.align());
        if !imp::is_big(&layout) && !imp::is_big(&new) {
            return System.realloc(ptr, layout, new_size);
        }
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
    #[inline]
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        System.realloc(ptr, layout, new_size)
    }
}
