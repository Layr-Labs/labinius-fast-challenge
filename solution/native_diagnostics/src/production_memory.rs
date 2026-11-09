#[cfg(target_os = "linux")]
mod sys {
    use core::ffi::c_void;
    extern "C" {
        pub fn mmap(
            addr: *mut c_void,
            len: usize,
            prot: i32,
            flags: i32,
            fd: i32,
            offset: i64,
        ) -> *mut c_void;
        pub fn munmap(addr: *mut c_void, len: usize) -> i32;
        pub fn madvise(addr: *mut c_void, len: usize, advice: i32) -> i32;
        pub fn fallocate(fd: i32, mode: i32, offset: i64, len: i64) -> i32;
    }
    pub const PROT_NONE: i32 = 0;
    pub const PROT_READ: i32 = 1;
    pub const PROT_WRITE: i32 = 2;
    pub const MAP_SHARED: i32 = 1;
    pub const MAP_FIXED: i32 = 0x10;
    pub const MAP_PRIVATE: i32 = 2;
    pub const MAP_ANONYMOUS: i32 = 0x20;
    pub const MAP_NORESERVE: i32 = 0x4000;
    pub const MADV_HUGEPAGE: i32 = 14;
    /// Linux 5.14+: fault the range in now, read-only, without blocking a later writer.
    pub const MADV_POPULATE_READ: i32 = 22;
    /// Linux 5.14+: fault the range in now, writable (for a shared file mapping: page-cache pages
    /// allocated and their blocks reserved, as a first write would).
    pub const MADV_POPULATE_WRITE: i32 = 23;
}

// =============================================================================================
// memory: large allocations on huge pages
// =============================================================================================

/// The process's allocator: glibc's for everything below [`huge::LARGE`], and for every larger
/// block a mapping of its own, 2 MB aligned, with transparent huge pages requested.
///
/// The prover's big buffers (the commitment rows, the commitment and fold outputs, the proof's
/// byte fields, the switch's tables) are written once and freed whole. On 4 KB pages each of
/// them costs one fault and one zeroed page per 4 KB when it is first written, and the same
/// number of page frees when it is unmapped — and whatever is still mapped at the end is unmapped
/// by the kernel in the exit path, after the proof is written, inside the measured time (about
/// 0.25 us per 4 KB page: 2.2 ms for the ~30 MB the baseline still held). A huge page is one fault,
/// one 2 MB clear and one free.
#[cfg(target_os = "linux")]
mod huge {
    use super::sys;
    use core::cell::UnsafeCell;
    use core::sync::atomic::{AtomicBool, Ordering};
    use std::alloc::{GlobalAlloc, Layout, System};

    const HUGE: usize = 1 << 21;
    /// Blocks at least this large get a mapping of their own.
    pub const LARGE: usize = 1 << 20;
    /// Blocks from this size up to [`LARGE`] (alignment at most this) come from huge-page chunks
    /// in quarter-octave classes, each class recycling its blocks through a free list (never
    /// returned to the kernel). glibc served them from per-thread arenas that it trimmed back to
    /// the kernel (`MADV_DONTNEED`) when they were freed, and the next allocation faulted the
    /// pages in again; larger per-dispatch buffers (the fold's ~840 KB per worker) were mmapped
    /// and unmapped each time. Blocks are carved with a 64-byte gap between them, not at
    /// power-of-two offsets: the commitment's sixty-four 6 KB accumulators at 8 KB-aligned
    /// addresses all fell into the same cache sets.
    const MEDIUM: usize = 1 << 16;
    const LISTS: usize = 4 * ((LARGE / MEDIUM).trailing_zeros() as usize + 1);
    /// Gap between consecutively carved blocks.
    const GAP: usize = 64;

    pub struct Alloc;

    #[inline]
    fn large(size: usize, align: usize) -> bool {
        size >= LARGE && align <= HUGE
    }

    #[inline]
    fn medium(size: usize, align: usize) -> bool {
        size >= MEDIUM && size < LARGE && align <= MEDIUM
    }

    /// A medium request's block size (`size` rounded up to a quarter of its octave) and the
    /// free list of that size.
    #[inline]
    fn class(size: usize) -> (usize, usize) {
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

    /// The chunk medium blocks are carved from.
    struct Carve {
        lock: SpinLock,
        at: UnsafeCell<usize>,
        end: UnsafeCell<usize>,
    }

    // Guarded by their locks.
    unsafe impl Sync for FreeList {}
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

    /// A medium block for `size` bytes at `align`, and whether it is fresh (still zero).
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
            at = chunk;
            *CARVE.end.get() = chunk + HUGE;
        }
        *CARVE.at.get() = at + bytes + GAP;
        CARVE.lock.unlock();
        (at as *mut u8, true)
    }

    unsafe fn medium_free(p: *mut u8, size: usize) {
        let list = &FREE[class(size).1];
        list.lock.lock();
        *(p as *mut *mut u8) = *list.head.get();
        *list.head.get() = p;
        list.lock.unlock();
    }

    #[inline]
    fn span(size: usize) -> usize {
        (size + HUGE - 1) & !(HUGE - 1)
    }

    /// A fresh (zeroed) 2 MB-aligned mapping of `span(size)` bytes, or null.
    unsafe fn map(size: usize) -> *mut u8 {
        let len = span(size);
        let raw = sys::mmap(
            core::ptr::null_mut(),
            len + HUGE,
            sys::PROT_READ | sys::PROT_WRITE,
            sys::MAP_PRIVATE | sys::MAP_ANONYMOUS | sys::MAP_NORESERVE,
            -1,
            0,
        );
        if raw as isize == -1 || raw.is_null() {
            return core::ptr::null_mut();
        }
        let raw = raw as usize;
        let start = (raw + HUGE - 1) & !(HUGE - 1);
        if start > raw {
            sys::munmap(raw as *mut _, start - raw);
        }
        let (end, raw_end) = (start + len, raw + len + HUGE);
        if raw_end > end {
            sys::munmap(end as *mut _, raw_end - end);
        }
        // Advisory: on a kernel without THP the block is ordinary pages.
        sys::madvise(start as *mut _, len, sys::MADV_HUGEPAGE);
        start as *mut u8
    }

    unsafe impl GlobalAlloc for Alloc {
        #[inline]
        unsafe fn alloc(&self, l: Layout) -> *mut u8 {
            if large(l.size(), l.align()) {
                map(l.size())
            } else if medium(l.size(), l.align()) {
                medium_alloc(l.size(), l.align()).0
            } else {
                System.alloc(l)
            }
        }

        #[inline]
        unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
            if large(l.size(), l.align()) {
                map(l.size())
            } else if medium(l.size(), l.align()) {
                let (p, fresh) = medium_alloc(l.size(), l.align());
                if !p.is_null() && !fresh {
                    core::ptr::write_bytes(p, 0, l.size());
                }
                p
            } else {
                System.alloc_zeroed(l)
            }
        }

        #[inline]
        unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
            if large(l.size(), l.align()) {
                sys::munmap(p as *mut _, span(l.size()));
            } else if medium(l.size(), l.align()) {
                medium_free(p, l.size())
            } else {
                System.dealloc(p, l)
            }
        }

        unsafe fn realloc(&self, p: *mut u8, l: Layout, new_size: usize) -> *mut u8 {
            let (old_large, new_large) = (large(l.size(), l.align()), large(new_size, l.align()));
            let (old_medium, new_medium) = (medium(l.size(), l.align()), medium(new_size, l.align()));
            if !old_large && !new_large && !old_medium && !new_medium {
                return System.realloc(p, l, new_size);
            }
            if old_large && new_large && span(new_size) == span(l.size()) {
                return p;
            }
            if old_medium && new_medium && class(new_size).0 == class(l.size()).0 {
                return p;
            }
            let q = self.alloc(Layout::from_size_align_unchecked(new_size, l.align()));
            if !q.is_null() {
                core::ptr::copy_nonoverlapping(p, q, l.size().min(new_size));
                self.dealloc(p, l);
            }
            q
        }
    }
}

#[cfg(target_os = "linux")]
#[global_allocator]
static ALLOC: huge::Alloc = huge::Alloc;

enum Table {
    #[cfg(target_os = "linux")]
    Mapped {
        ptr: *const B128,
        bytes: usize,
    },
    Owned(Vec<B128>),
}

impl Table {
    const WORDS: usize = 1 << LOG_PACKED;
    const BYTES: usize = (1 << LOG_VARS) / 8;

    fn load(path: &str) -> Table {
        assert!(cfg!(target_endian = "little"), "the table is little-endian");
        assert_eq!(core::mem::size_of::<B128>(), 16, "B128 is 16 bytes");
        let file = std::fs::File::open(path).expect("open witness table");
        let len = file.metadata().expect("stat witness table").len();
        assert_eq!(len, Self::BYTES as u64, "witness table has the wrong length");
        #[cfg(target_os = "linux")]
        {
            use std::os::fd::AsRawFd;
            let ptr = unsafe {
                sys::mmap(
                    core::ptr::null_mut(),
                    Self::BYTES,
                    sys::PROT_READ,
                    sys::MAP_PRIVATE,
                    file.as_raw_fd(),
                    0,
                )
            };
            if ptr as isize != -1 && !ptr.is_null() && (ptr as usize) % 16 == 0 {
                return Table::Mapped {
                    ptr: ptr as *const B128,
                    bytes: Self::BYTES,
                };
            }
        }
        let mut words = vec![B128::ZERO; Self::WORDS];
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(words.as_mut_ptr() as *mut u8, Self::BYTES)
        };
        let mut file = file;
        std::io::Read::read_exact(&mut file, bytes).expect("witness table has the wrong length");
        let mut extra = [0u8; 1];
        let extra_n = std::io::Read::read(&mut file, &mut extra).expect("read witness table");
        assert_eq!(extra_n, 0, "witness table has the wrong length");
        Table::Owned(words)
    }

    fn words(&self) -> &[B128] {
        match self {
            #[cfg(target_os = "linux")]
            Table::Mapped { ptr, .. } => unsafe { std::slice::from_raw_parts(*ptr, Self::WORDS) },
            Table::Owned(v) => v,
        }
    }

}

/// Fault the words' pages in, page-cache pages into page tables: `MADV_POPULATE_READ` where the
/// kernel has it (older kernels refuse it, harmlessly), one read per page otherwise (the
/// fault-around maps a run of pages per fault). A read-only touch, so it runs on another thread
/// while the key is built.
fn populate_words(words: &[B128], keys: &[u8], threads: usize) {
    let runs = labinius::par::ranges(words.len(), threads, 4096 / 16);
    let kruns = labinius::par::ranges(keys.len(), threads, 4096);
    labinius::par::pool().run(runs.len().max(kruns.len()), |i| {
        if let Some(r) = kruns.get(i) {
            populate_part(keys[r.clone()].as_ptr() as usize, r.len());
        }
        if let Some(r) = runs.get(i) {
            populate_part(words[r.clone()].as_ptr() as usize, r.len() * 16);
        }
    });
}

fn populate_part(addr: usize, bytes: usize) {
    #[cfg(target_os = "linux")]
    {
        // whole pages: the range need not start on one
        let start = addr & !4095;
        let done = unsafe {
            sys::madvise(start as *mut core::ffi::c_void, bytes + (addr - start), sys::MADV_POPULATE_READ)
        };
        if done == 0 {
            return;
        }
    }
    let mut off = 0;
    let mut sink = 0u8;
    while off < bytes {
        sink ^= unsafe { core::ptr::read_volatile((addr as *const u8).add(off)) };
        off += 4096;
    }
    std::hint::black_box(sink);
}

// A read-only mapping: any thread may read or unmap it.
#[cfg(target_os = "linux")]
unsafe impl Send for Table {}

impl Drop for Table {
    fn drop(&mut self) {
        #[cfg(target_os = "linux")]
        if let Table::Mapped { ptr, bytes } = self {
            unsafe { sys::munmap(*ptr as *mut core::ffi::c_void, *bytes) };
        }
    }
}

