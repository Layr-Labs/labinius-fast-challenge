//! Local native falsifier footer, not production code or validated native execution.
//! C[p2][factor6][N2][form3][stream2][col16][coef16] i32.
//! Coupled-lift aux is flushed every256 rings:16 complete chunks; ordinary persists4096 rings.
//! Footer::new, returned48rows, memory faults and destruction are timed by driver.
use core::arch::x86_64::*;
use labinius::params::{inv_mod, Params};
use labinius::ring::{Batch32, Representation};
use labinius::simd::ntt::gen_small;
mod forward_adapter;
const P0: u16 = 3889;
const P1: u16 = 9721;
const PRODUCT: i64 = P0 as i64 * P1 as i64;
const N: usize = 648;
const COLS: usize = 16;
const CHUNKS: usize = 16;
include!("arithmetic.rs");

/// Owns aligned, zero-initialized memory. Zero is valid for its actual uses:
/// i16 words, and Batch32 whose Coefficients representation discriminant is0.
struct Aligned<T> {
    ptr: *mut T,
    len: usize,
    layout: std::alloc::Layout,
}
impl<T> Aligned<T> {
    fn zeroed(len: usize) -> Self {
        let layout = std::alloc::Layout::from_size_align(
            core::mem::size_of::<T>().checked_mul(len).unwrap(),
            core::mem::align_of::<T>().max(64),
        )
        .unwrap();
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) } as *mut T;
        if ptr.is_null() {
            std::alloc::handle_alloc_error(layout)
        }
        Self { ptr, len, layout }
    }
    fn slice(&self) -> &[T] {
        unsafe { core::slice::from_raw_parts(self.ptr, self.len) }
    }
    fn slice_mut(&mut self) -> &mut [T] {
        unsafe { core::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}
impl<T> Drop for Aligned<T> {
    fn drop(&mut self) {
        unsafe {
            core::ptr::drop_in_place(core::ptr::slice_from_raw_parts_mut(self.ptr, self.len));
            std::alloc::dealloc(self.ptr as *mut u8, self.layout)
        }
    }
}
// Only moved to an owning worker; no concurrent access to one Footer.
unsafe impl<T: Send> Send for Aligned<T> {}

pub struct Footer {
    // pending[p2][chunk16][coef648][col16], ordinary[p2][coef648][col16].
    pending: Aligned<i16>,
    ordinary: Aligned<i16>,
    // scratch[coef32][col16], actual M16-by-N16 -> coefficientmajor transpose.
    scratch: Aligned<i16>,
    batches: Aligned<Batch32>,
    total: Aligned<Batch32>,
    aux_done: [[bool; CHUNKS]; 4],
    ordinary_done: [bool; 4],
}
impl Footer {
    pub fn new() -> Self {
        Self {
            pending: Aligned::zeroed(2 * CHUNKS * N * COLS),
            ordinary: Aligned::zeroed(2 * N * COLS),
            scratch: Aligned::zeroed(32 * COLS),
            batches: Aligned::zeroed(2),
            total: Aligned::zeroed(1),
            aux_done: [[false; CHUNKS]; 4],
            ordinary_done: [false; 4],
        }
    }
    /// Caller flushes br3,7,...,63 with chunk=br/4 (16 chunks perquarter).
    /// Caller must finish all AMX updates for this quarter/chunk before calling.
    /// No other thread may access C or Footer during this call; C is64B aligned.
    #[target_feature(enable = "avx512f,avx512bw,avx512vl")]
    pub unsafe fn flush_aux(&mut self, quarter: usize, chunk: usize, c: *mut i32) {
        assert!(quarter < 4 && chunk < CHUNKS && !self.aux_done[quarter][chunk]);
        self.extract::<P0>(0, quarter, Some(chunk), c);
        self.extract::<P1>(1, quarter, Some(chunk), c);
        self.aux_done[quarter][chunk] = true;
    }
    /// Complete4096rings before finishing ordinary; aux must already be flushed.
    #[target_feature(enable = "avx512f,avx512bw,avx512vl")]
    pub unsafe fn finish_quarter(&mut self, quarter: usize, c: *mut i32) {
        assert!(quarter < 4 && !self.ordinary_done[quarter]);
        assert!(self.aux_done[quarter].iter().all(|&v| v));
        self.extract::<P0>(0, quarter, None, c);
        self.extract::<P1>(1, quarter, None, c);
        self.ordinary_done[quarter] = true;
    }
    /// Normalize each raw form independently BEFORE applying base128 weights.
    /// RawC i32 bounds and exact coupled256ringlift are established by prior gates, not here.
    #[target_feature(enable = "avx512f,avx512bw,avx512vl")]
    unsafe fn extract<const Q: u16>(
        &mut self,
        p: usize,
        quarter: usize,
        chunk: Option<usize>,
        c: *mut i32,
    ) {
        assert_eq!((c as usize) & 63, 0);
        let stream = usize::from(chunk.is_some());
        for f in 0..6 {
            for col in 0..COLS {
                for n in 0..2 {
                    let address = |form: usize| {
                        c.add((((((p * 6 + f) * 2 + n) * 3 + form) * 2 + stream) * COLS + col) * 16)
                    };
                    let ll = red16::<Q>(_mm512_load_si512(address(0) as *const __m512i));
                    let hh = red16::<Q>(_mm512_load_si512(address(1) as *const __m512i));
                    let ss = red16::<Q>(_mm512_load_si512(address(2) as *const __m512i));
                    // rLL +128*(rSS-rLL-rHH) +16384*rHH, <321006862 in abs.
                    let cross = _mm512_sub_epi32(_mm512_sub_epi32(ss, ll), hh);
                    let weighted = _mm512_add_epi32(
                        ll,
                        _mm512_add_epi32(
                            _mm512_slli_epi32::<7>(cross),
                            _mm512_slli_epi32::<14>(hh),
                        ),
                    );
                    let normalized = red16::<Q>(weighted);
                    let mut words = [0i32; 16];
                    _mm512_storeu_si512(words.as_mut_ptr() as *mut __m512i, normalized);
                    for j in 0..16 {
                        self.scratch.slice_mut()[(16 * n + j) * COLS + col] = words[j] as i16;
                    }
                    // Only the selected stream is cleared; fullN16 padding is also cleared.
                    for form in 0..3 {
                        _mm512_store_si512(address(form) as *mut __m512i, _mm512_setzero_si512());
                    }
                }
            }
            for j in 0..27 {
                let coefficient = 27 * (quarter * 6 + f) + j;
                let src = self.scratch.ptr.add(j * COLS) as *const __m256i;
                if let Some(ch) = chunk {
                    let dst = self
                        .pending
                        .ptr
                        .add(((p * CHUNKS + ch) * N + coefficient) * COLS)
                        as *mut __m256i;
                    //32B percoef,54coeff per factorpair=27fulllines. This is a real NT
                    //store; pairing, cacheeffects and fence cost are included in native walltime.
                    _mm256_stream_si256(dst, _mm256_load_si256(src));
                } else {
                    let dst = self.ordinary.ptr.add((p * N + coefficient) * COLS) as *mut __m256i;
                    _mm256_store_si256(dst, _mm256_load_si256(src));
                }
            }
        }
    }
    /// Returns3rows percolumn: ordinary q3889 fullysplitNTT, ordinary q9721
    /// fullysplitNTT, centered-exact auxiliary coefficient sum modulo1024.
    /// Ordinary inverse belongs to unchanged main encoding, outside commit timer.
    #[target_feature(enable = "avx512f,avx512bw,avx512vl")]
    pub unsafe fn finish(&mut self) -> Vec<[u32; 648]> {
        assert!(self.ordinary_done.iter().all(|&v| v));
        assert!(self.aux_done.iter().flatten().all(|&v| v));
        // All non-temporal pending writes must be globally ordered before any read.
        _mm_sfence();
        core::ptr::write_bytes(
            self.total.ptr as *mut u8,
            0,
            core::mem::size_of::<Batch32>(),
        );
        for pair in 0..CHUNKS / 2 {
            for p in 0..2 {
                let b = &mut *self.batches.ptr.add(p);
                for j in 0..N {
                    for half in 0..2 {
                        let src = self
                            .pending
                            .ptr
                            .add(((p * CHUNKS + 2 * pair + half) * N + j) * COLS)
                            as *const __m256i;
                        _mm256_store_si256(
                            b.v[j].as_mut_ptr().add(half * COLS) as *mut __m256i,
                            _mm256_load_si256(src),
                        );
                    }
                }
            }
            forward_adapter::partial_to_full::<P0>(&mut *self.batches.ptr);
            forward_adapter::partial_to_full::<P1>(&mut *self.batches.ptr.add(1));
            gen_small::intt_gen_batch32::<P0>(&mut *self.batches.ptr);
            gen_small::intt_gen_batch32::<P1>(&mut *self.batches.ptr.add(1));
            crt_accumulate(
                &*self.batches.ptr,
                &*self.batches.ptr.add(1),
                &mut *self.total.ptr,
            );
        }
        // This allocation and all writes are within the timed finish call.
        let mut rows = vec![[0u32; N]; 3 * COLS];
        for col in 0..COLS {
            for j in 0..N {
                let lo = (*self.total.ptr).v[j][col] as u16;
                let hi = (*self.total.ptr).v[j][COLS + col] as u16;
                rows[3 * col + 2][j] = (lo.wrapping_add(hi) & 1023) as u32;
            }
        }
        for p in 0..2 {
            let b = &mut *self.batches.ptr.add(p);
            // Clear the otherwise unused upper16 lanes to make the native workload
            //deterministic and avoid carrying prior aux data into ordinary operations.
            for j in 0..N {
                let src = self.ordinary.ptr.add((p * N + j) * COLS) as *const __m256i;
                _mm256_store_si256(b.v[j].as_mut_ptr() as *mut __m256i, _mm256_load_si256(src));
                _mm256_store_si256(
                    b.v[j].as_mut_ptr().add(COLS) as *mut __m256i,
                    _mm256_setzero_si256(),
                );
            }
        }
        forward_adapter::partial_to_full::<P0>(&mut *self.batches.ptr);
        forward_adapter::partial_to_full::<P1>(&mut *self.batches.ptr.add(1));
        for p in 0..2 {
            let q = if p == 0 { P0 } else { P1 } as i32;
            for col in 0..COLS {
                for j in 0..N {
                    rows[3 * col + p][j] =
                        ((*self.batches.ptr.add(p)).v[j][col] as i32).rem_euclid(q) as u32;
                }
            }
        }
        self.aux_done = [[false; CHUNKS]; 4];
        self.ordinary_done = [false; 4];
        rows
    }
}
