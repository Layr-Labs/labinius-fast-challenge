//! The fixed proof wire layout, encoded directly into preallocated file pages.
//! The layout and idle-work design follow the x86 track's Output implementation.
use super::{rq, FoldedWitness, COLUMNS, LIMIT, LOG_PACKED, RING_ELEMENTS};
use labinius::par;
use labinius::ring::Representation;
use std::sync::{Arc, Mutex, OnceLock};

pub(super) struct ProofParts {
    pub modulus: u128,
    pub commitment: Vec<u8>,
    pub partials: Vec<u128>,
    pub rounds: Vec<[[u64; 3]; 2]>,
    pub opened: [u64; 3],
    pub row: Vec<u8>,
    pub folded: FoldedWitness,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Layout {
    commitment: usize,
    partials: usize,
    rounds: usize,
    row: usize,
    folded: usize,
}

impl Layout {
    const HEAD: usize = 4 + 16 + 8;
    fn instance(q: u128) -> Self {
        Self {
            commitment: COLUMNS * rq::N * rq::coefficient_bytes(q),
            partials: 128,
            rounds: LOG_PACKED,
            row: labinius::wire::f162_bytes(COLUMNS),
            folded: 4 * rq::N * RING_ELEMENTS,
        }
    }
    fn parts(p: &ProofParts) -> Self {
        Self {
            commitment: p.commitment.len(),
            partials: p.partials.len(),
            rounds: p.rounds.len(),
            row: p.row.len(),
            folded: 4 * rq::N * p.folded.elements().len(),
        }
    }
    fn middle_at(self) -> usize { Self::HEAD + self.commitment }
    fn folded_at(self) -> usize {
        self.middle_at() + 8 + 16 * self.partials + 8 + 48 * self.rounds + 24 + 8 + self.row + 8
    }
    fn len(self) -> usize { self.folded_at() + self.folded }
}

/// # Safety
/// `out` is writable for exactly `layout.len()` bytes and does not overlap `p`'s storage.
/// The caller must supply Layout::parts(p). Only disjoint output ranges are sent to workers.
unsafe fn encode_into(p: &ProofParts, layout: Layout, out: *mut u8) {
    assert_eq!(layout, Layout::parts(p));
    assert!(layout.len() as u64 <= LIMIT, "proof exceeds the size limit");
    let put = |at: &mut usize, bytes: &[u8]| {
        core::ptr::copy_nonoverlapping(bytes.as_ptr(), out.add(*at), bytes.len());
        *at += bytes.len();
    };
    let mut at = 0;
    put(&mut at, &1u32.to_le_bytes());
    put(&mut at, &p.modulus.to_le_bytes());
    put(&mut at, &(p.commitment.len() as u64).to_le_bytes());
    assert_eq!(at, Layout::HEAD);
    let mut at = layout.middle_at();
    put(&mut at, &(p.partials.len() as u64).to_le_bytes());
    for x in &p.partials { put(&mut at, &x.to_le_bytes()); }
    put(&mut at, &(p.rounds.len() as u64).to_le_bytes());
    for x in p.rounds.iter().flatten().flatten() { put(&mut at, &x.to_le_bytes()); }
    for x in &p.opened { put(&mut at, &x.to_le_bytes()); }
    put(&mut at, &(p.row.len() as u64).to_le_bytes());
    put(&mut at, &p.row);
    put(&mut at, &(layout.folded as u64).to_le_bytes());
    assert_eq!(at, layout.folded_at());

    let elements = p.folded.elements();
    assert!(elements.iter().all(|e| e.representation == Representation::Coefficients));
    let addr = out as usize;
    let ctasks = if p.commitment.is_empty() { 0 } else { 4 };
    let ftasks = if elements.is_empty() { 0 } else { 4 * par::threads() };
    par::pool().run(ctasks + ftasks, |i| {
        let out = addr as *mut u8;
        if i < ctasks {
            let (s, e) = par::block(p.commitment.len(), ctasks, i, 4096);
            core::ptr::copy_nonoverlapping(p.commitment.as_ptr().add(s), out.add(Layout::HEAD + s), e - s);
        } else {
            let (s, e) = par::block(elements.len(), ftasks, i - ctasks, 1);
            let mut dst = out.add(layout.folded_at() + 4 * rq::N * s);
            // The widening loop vectorizes on NEON; the temporary gives it aligned i32 stores.
            let mut wide = [0i32; rq::N];
            for element in &elements[s..e] {
                for (w, &c) in wide.iter_mut().zip(&element.v) { *w = (c as i32).to_le(); }
                core::ptr::copy_nonoverlapping(wide.as_ptr() as *const u8, dst, 4 * rq::N);
                dst = dst.add(4 * rq::N);
            }
        }
    });
}

#[cfg(target_os = "linux")]
mod sys {
    unsafe extern "C" {
        pub fn fallocate(fd: i32, mode: i32, offset: i64, len: i64) -> i32;
        pub fn mmap(addr: *mut u8, len: usize, prot: i32, flags: i32, fd: i32, off: i64) -> *mut u8;
        pub fn munmap(addr: *mut u8, len: usize) -> i32;
        pub fn madvise(addr: *mut u8, len: usize, advice: i32) -> i32;
    }
}

struct Mapped {
    _file: std::fs::File,
    base: usize,
    span: usize,
    addr: usize,
}
impl Drop for Mapped {
    fn drop(&mut self) {
        #[cfg(target_os = "linux")]
        unsafe { sys::munmap(self.base as *mut u8, self.span); }
    }
}
const CHUNK: usize = 2 << 20;

struct State {
    path: String,
    layout: Layout,
    map: OnceLock<Option<Mapped>>,
}
impl State {
    #[cfg(target_os = "linux")]
    fn create_and_map(&self) -> Option<Mapped> {
        use std::os::fd::AsRawFd;
        let len = self.layout.len();
        let file = std::fs::OpenOptions::new().read(true).write(true).create(true).truncate(true)
            .open(&self.path).ok()?;
        // Preallocate BEFORE mapping: a sparse, delayed-allocation file can SIGBUS on a fault.
        // Filesystems without fallocate use the ordinary write fallback.
        if unsafe { sys::fallocate(file.as_raw_fd(), 0, 0, len as i64) } != 0 { return None; }
        let span = len.div_ceil(CHUNK) * CHUNK + CHUNK;
        let base = unsafe { sys::mmap(core::ptr::null_mut(), span, 0, 2 | 0x20 | 0x4000, -1, 0) };
        if base as isize == -1 { return None; }
        let addr = (base as usize + CHUNK - 1) & !(CHUNK - 1);
        let ptr = unsafe { sys::mmap(addr as *mut u8, len, 1 | 2, 1 | 0x10, file.as_raw_fd(), 0) };
        if ptr as isize == -1 {
            unsafe { sys::munmap(base, span); }
            return None;
        }
        unsafe { sys::madvise(ptr, len, 14); }
        Some(Mapped { _file: file, base: base as usize, span, addr })
    }
    #[cfg(not(target_os = "linux"))]
    fn create_and_map(&self) -> Option<Mapped> { None }
    fn mapped(&self) -> Option<&Mapped> { self.map.get_or_init(|| self.create_and_map()).as_ref() }
    fn populate(&self, i: usize) {
        let Some(map) = self.mapped() else { return; };
        let at = i * CHUNK;
        let len = CHUNK.min(self.layout.len() - at);
        #[cfg(target_os = "linux")]
        unsafe { sys::madvise((map.addr + at) as *mut u8, len, 23); }
    }
}

pub(super) struct Output {
    state: Arc<State>,
    setup: Mutex<Option<std::thread::JoinHandle<()>>>,
    populate: OnceLock<&'static par::IdleJob>,
    commitment: Mutex<Option<Vec<u8>>>,
}
impl Output {
    pub fn start(path: &str, q: u128) -> Self {
        let layout = Layout::instance(q);
        let state = Arc::new(State { path: path.to_owned(), layout, map: OnceLock::new() });
        let mut commitment = vec![0u8; layout.commitment];
        let addr = commitment.as_mut_ptr() as usize;
        let bytes = commitment.len();
        let worker = Arc::downgrade(&state);
        // Kernel-owned code still uses scoped threads. Starting seven spinning workers now
        let setup = std::thread::spawn(move || {
            let state = worker.upgrade();
            if let Some(state) = &state { let _ = state.mapped(); }
            labinius::challenge::warm();
            for offset in (0..bytes).step_by(4096) {
                unsafe { core::ptr::write_volatile((addr + offset) as *mut u8, 0); }
            }
            // Fault the whole proof mapping while the table read still runs on the other
            // cores: the write phase then only has to encode, not populate file pages.
            if let Some(state) = &state {
                for i in 0..state.layout.len().div_ceil(CHUNK) { state.populate(i); }
            }
        });
        Self { state, setup: Mutex::new(Some(setup)), populate: OnceLock::new(), commitment: Mutex::new(Some(commitment)) }
    }
    pub fn wait_setup(&self) {
        self.finish_setup();
    }
    fn finish_setup(&self) {
        if let Some(thread) = self.setup.lock().unwrap().take() {
            thread.join().expect("output preparation thread panicked");
        }
    }
    pub fn commitment_buffer(&self) -> Vec<u8> {
        self.finish_setup();
        self.commitment.lock().unwrap().take().expect("commitment buffer used once")
    }
    /// Called before transcript absorb, while the caller is about to run alone. Avoid file
    /// population chunks at commitment dispatch: they can delay a coarse commitment task.
    pub fn populate(&self) {
        let worker = Arc::downgrade(&self.state);
        let chunks = self.state.layout.len().div_ceil(CHUNK);
        let job = par::pool().idle(chunks, move |i| {
            if let Some(state) = worker.upgrade() { state.populate(i); }
        });
        assert!(self.populate.set(job).is_ok(), "output populated once");
    }
    pub fn write(&self, p: &ProofParts) {
        self.quiesce();
        let layout = Layout::parts(p);
        if layout == self.state.layout {
            if let Some(map) = self.state.mapped() {
                unsafe { encode_into(p, layout, map.addr as *mut u8); }
                return;
            }
        }
        // A generic modulus/layout or unsupported file mapping still gets exactly the codec.
        // Ordinary write shares the harness's persistence contract: neither path calls fsync.
        std::fs::write(&self.state.path, encode_vec(p)).expect("write proof");
    }
    fn quiesce(&self) {
        self.finish_setup();
        if let Some(job) = self.populate.get() { par::pool().quiesce(job); }
    }
}
impl Drop for Output { fn drop(&mut self) { self.quiesce(); } }

fn encode_vec(p: &ProofParts) -> Vec<u8> {
    let layout = Layout::parts(p);
    let mut bytes = vec![0; layout.len()];
    unsafe { encode_into(p, layout, bytes.as_mut_ptr()); }
    bytes
}

/// Pack commitment coefficients into the buffer faulted in by idle workers. The ring math and
/// transcript absorb stay unchanged; chunks cover disjoint, whole coefficient encodings.
pub(super) fn pack_commitment(values: &[u128], q: u128, mut bytes: Vec<u8>) -> Vec<u8> {
    let width = rq::coefficient_bytes(q);
    assert_eq!(bytes.len(), width * values.len());
    let per = values.len().div_ceil(par::threads()).max(1);
    std::thread::scope(|scope| {
        for (dst, values) in bytes.chunks_mut(per * width).zip(values.chunks(per)) {
            scope.spawn(move || {
                for (slot, &x) in dst.chunks_exact_mut(width).zip(values) {
                    debug_assert!(x < q, "a coefficient is not reduced");
                    slot.copy_from_slice(&x.to_le_bytes()[..width]);
                }
            });
        }
    });
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{codec, pack_folded, Proof, RingElement648};
    use bincode::Options;

    fn fixture(n: usize) -> (ProofParts, Vec<u8>) {
        let mut state = 0x123456789abcdef0u64;
        let mut next = || {
            state ^= state << 13; state ^= state >> 7; state ^= state << 17; state
        };
        let mut elements: Vec<_> = (0..n).map(|_| {
            let mut e = RingElement648::zero(Representation::Coefficients);
            for c in &mut e.v { *c = next() as i16; }
            e
        }).collect();
        if let Some(first) = elements.first_mut() {
            first.v[..5].copy_from_slice(&[i16::MIN, -1, 0, 1, i16::MAX]);
        }
        let p = ProofParts {
            modulus: 35039150080,
            commitment: (0..n * 648 + 3).map(|_| next() as u8).collect(),
            partials: (0..n + 2).map(|_| (next() as u128) << 64 | next() as u128).collect(),
            rounds: (0..n + 1).map(|_| [[next(), next(), next()], [next(), next(), next()]]).collect(),
            opened: [next(), next(), next()],
            row: (0..n * 13 + 1).map(|_| next() as u8).collect(),
            folded: FoldedWitness::of(elements),
        };
        let reference = Proof {
            version: 1, modulus: p.modulus, commitment: p.commitment.clone(),
            partials: p.partials.clone(), rounds: p.rounds.clone(), opened: p.opened,
            row: p.row.clone(), folded: pack_folded(&p.folded),
        };
        let bytes = codec().serialize(&reference).unwrap();
        (p, bytes)
    }

    #[test]
    fn parallel_encoder_matches_bincode_and_preserves_guards() {
        for n in [0, 1, 3, 37, 129] {
            let (p, want) = fixture(n);
            let layout = Layout::parts(&p);
            assert_eq!(layout.len(), want.len());
            assert_eq!(encode_vec(&p), want, "elements {n}");
            let mut guarded = vec![0x5a; layout.len() + 34];
            unsafe { encode_into(&p, layout, guarded.as_mut_ptr().add(17)); }
            assert_eq!(&guarded[..17], &[0x5a; 17]);
            assert_eq!(&guarded[17..17 + want.len()], &want);
            assert_eq!(&guarded[17 + want.len()..], &[0x5a; 17]);
        }
    }

    #[test]
    fn parallel_commitment_pack_matches_existing_wire_for_all_widths() {
        for width in 1..=12 {
            let q = (1u128 << (8 * width - 1)) + 17;
            let values: Vec<_> = (0..1003u128).map(|x| (x * x * 7919 + 1) % q).collect();
            let old = rq::pack_ring_elements(&values, q);
            assert_eq!(pack_commitment(&values, q, vec![0; old.len()]), old, "width {width}");
        }
        assert_eq!(Layout::instance(35039150080).len(), 13958612);
    }

    fn test_output(path: &str, layout: Layout, force_write: bool) -> Output {
        let map = OnceLock::new();
        if force_write { assert!(map.set(None).is_ok()); }
        Output {
            state: Arc::new(State { path: path.to_owned(), layout, map }),
            setup: Mutex::new(None),
            populate: OnceLock::new(),
            commitment: Mutex::new(None),
        }
    }

    #[test]
    fn file_mapping_and_write_fallback_match_bincode() {
        let path = std::env::temp_dir().join(format!("labinius-tail-output-{}", std::process::id()));
        for n in [0, 1, 37, 1000] {
            let (p, want) = fixture(n);
            for force_write in [false, true] {
                let output = test_output(path.to_str().unwrap(), Layout::parts(&p), force_write);
                output.populate();
                output.write(&p);
                #[cfg(target_os = "linux")]
                if !force_write { assert!(output.state.mapped().is_some(), "test filesystem must support mapping"); }
                assert_eq!(std::fs::read(&path).unwrap(), want, "n {n}, fallback {force_write}");
                drop(output);
            }
            // A different runtime shape must also use the generic write path.
            let output = test_output(path.to_str().unwrap(), Layout::instance(p.modulus), false);
            output.write(&p);
            assert_eq!(std::fs::read(&path).unwrap(), want);
        }
        std::fs::remove_file(path).unwrap();
    }
}
