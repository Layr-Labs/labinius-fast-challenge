// AMX-INT8 accumulation for the one-transform commitment: included by `sms_commit.rs` as
// `sms::amx`, so the parent's items are in scope.
//
// Per output slot `j` the eight streams' accumulation over one transform call is a matrix
// product on the tiles, `C (16 x 16 i32) += A (16 x K i8) * B (K x 16 u8)` (`TDPBSUD`):
//
// - `A` row `2 s + p` is plane `p` of stream `s`'s centered key at slot `j` (`k = lo + 128 hi`,
//   `lo in [-64, 64)`, `hi in [-38, 38]`), one byte per ring element of the call;
// - `B` is the call's witness transforms plus `OFF` (so `0 <= v + OFF < 2^16`), byte-split: dword
//   `c < 8` holds the low bytes of column `c`'s four ring elements of a transform, dword `8 + c`
//   their high bytes; row `h` of `B` is transform `h` of the call (one `vpermb` of its lane);
// - so `sum_e k (v + OFF) = (C[2s][c] + 128 C[2s+1][c]) + 256 (C[2s][8+c] + 128 C[2s+1][8+c])`.
//
// A column group is 8 columns; a transform's 32 lanes are 8 columns x 4 consecutive ring elements
// (lane `4 c + i`). The accumulators live in memory between calls (1 KB per slot, 648 KB per
// worker) and start from zero at every reconstruction chunk: at a chunk's end every stream is
// flushed, stream 0 into a running residue, the digits into the inverse transform's batch (four
// chunks of 8 columns per batch). The public `OFF * sum_e k` term is folded into the digits' chunk
// means (through the inverse transform, which is linear) and subtracted from stream 0's rows, so
// the rows are exactly the ones the VNNI path writes. There is no fold-back: a chunk's sums stay
// far inside `i32`.
use super::*;

/// Transforms per NTT call; each covers 8 columns x 4 ring elements.
pub const AL: usize = 8;
/// Ring elements of a column per call: the tiles' `K`.
pub const K: usize = 4 * AL;
/// Columns per group.
pub const G: usize = 8;
/// Bias that makes every transform lane non-negative (`|v| <= w_bound`).
const OFF: i16 = w_bound(P) as i16;
/// Ring-element quads (one transform lane group) per reconstruction chunk.
const CQ: usize = CHUNK_PAIRS / 2;
/// Chunks per inverse transform: 4 x 8 columns fill its 32 lanes.
const BATCH_CHUNKS: usize = 4;
const _: () = {
    assert!(CQ % AL == 0 && K % 4 == 0 && K <= 64);
    assert!(OFF as i32 >= 0 && 2 * (OFF as i32) < 65536);
    assert!(STREAMS * 2 == 16);
};

// =============================================================================================
// public parameters
// =============================================================================================

/// The tile image and the means and corrections the AMX path reads.
pub struct AmxKeys<'a> {
    /// `tiles[((l * N + j) * 16 + 2 s + p) * K + i]`: plane `p` of stream `s`'s key at slot `j`
    /// for ring element `K l + i` of every column.
    pub tiles: &'a [i8],
    /// The digits' chunk means as [`SmsKeys::mur`], with the bias term of each chunk added.
    pub mur: &'a [i16],
    /// `corr0[h * N + j]`: stream 0's bias term over half `h` of the column, canonical.
    pub corr0: &'a [i16],
}

/// Owned [`AmxKeys`].
pub struct AmxKeyData {
    pub tiles: Vec<i8>,
    pub mur: Vec<i16>,
    pub corr0: Vec<i16>,
}

impl AmxKeyData {
    pub fn keys(&self) -> AmxKeys<'_> {
        AmxKeys { tiles: &self.tiles, mur: &self.mur, corr0: &self.corr0 }
    }

    /// Bytes of the tile image for `elements` ring elements per column.
    pub const fn tiles_len(elements: usize) -> usize {
        (elements / K) * N * 16 * K
    }

    /// The AMX image of `keys` (the lockstep pair layout), with the bias terms.
    pub fn build(keys: &SmsKeys) -> AmxKeyData {
        let elements = keys.elements;
        assert!(elements % K == 0 && (elements / 2) % 2 == 0);
        let np = elements / 2;
        let (hl, cph) = (np / 2, chunks_per_half(np));
        let chunks = chunk_count(elements);
        let key = |s: usize, j: usize, e: usize| -> i32 {
            let (l, h, hi) = (e / (2 * LOCK), (e % (2 * LOCK)) / 2, e % 2);
            let w = keys.pairs[(l * N + j) * STREAMS * LOCK + LOCK * s + h];
            (w >> (16 * hi)) as u16 as i16 as i32
        };
        let chunk_of = |e: usize| -> usize {
            let pair = e / 2;
            (pair / hl) * cph + (pair % hl) / CHUNK_PAIRS
        };
        let mut tiles = vec![0i8; Self::tiles_len(elements)];
        let mut sums = vec![0i64; STREAMS * chunks * N];
        for e in 0..elements {
            let (l, i, c) = (e / K, e % K, chunk_of(e));
            for j in 0..N {
                for s in 0..STREAMS {
                    let k = key(s, j, e);
                    let lo = (k + 64).rem_euclid(128) - 64;
                    let hi = (k - lo) / 128;
                    debug_assert!((-38..=38).contains(&hi));
                    let at = ((l * N + j) * 16 + 2 * s) * K + i;
                    tiles[at] = lo as i8;
                    tiles[at + K] = hi as i8;
                    sums[(s * chunks + c) * N + j] += k as i64;
                }
            }
        }
        let p = P as i64;
        let bias = |s: usize, c: usize, j: usize| -> i64 { (OFF as i64 * sums[(s * chunks + c) * N + j]).rem_euclid(p) };
        let mut corr0 = vec![0i16; 2 * N];
        for c in 0..chunks {
            let h = c / cph;
            for j in 0..N {
                let v = &mut corr0[h * N + j];
                *v = ((*v as i64 + bias(0, c, j)) % p) as i16;
            }
        }
        // The digits' bias vectors are in the NTT domain; their coefficients go into the means.
        let mut mur = keys.mur.to_vec();
        let vectors = DIGITS * chunks;
        let mut b = zeroed_slice::<Batch32>(1);
        for v0 in (0..vectors).step_by(32) {
            let lanes = 32.min(vectors - v0);
            for j in 0..N {
                for l in 0..32 {
                    b[0].v[j][l] = if l < lanes {
                        let (d, c) = ((v0 + l) / chunks, (v0 + l) % chunks);
                        let r = bias(1 + d, c, j);
                        (if r > (p - 1) / 2 { r - p } else { r }) as i16
                    } else {
                        0
                    };
                }
            }
            b[0].representation = Representation::Ntt;
            unsafe { gen_small::intt_gen_batch32::<P>(&mut b[0]) };
            for l in 0..lanes {
                for m in 0..N {
                    let at = (v0 + l) * N + m;
                    let r = (mur[at] as i64 + b[0].v[m][l] as i64).rem_euclid(p);
                    mur[at] = (if r > (p - 1) / 2 { r - p } else { r }) as i16;
                }
            }
        }
        AmxKeyData { tiles, mur, corr0 }
    }
}

// =============================================================================================
// AMX support
// =============================================================================================

/// Whether the commitment can run on AMX-INT8: the CPU reports AMX-TILE and AMX-INT8 and the
/// kernel grants this process the tile data state. `LABINIUS_AMX=0` turns it off;
/// `LABINIUS_AMX=force` skips the permission request's result (for emulators whose host kernel
/// does not know the state).
pub fn available() -> bool {
    static A: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *A.get_or_init(|| {
        let mode = std::env::var("LABINIUS_AMX").ok();
        if mode.as_deref() == Some("0") {
            return false;
        }
        if !(std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512bw")
            && std::is_x86_feature_detected!("avx512vl")
            && std::is_x86_feature_detected!("avx512vbmi")
            && std::is_x86_feature_detected!("avx512vbmi2")
            && std::is_x86_feature_detected!("gfni"))
        {
            return false;
        }
        let l7 = core::arch::x86_64::__cpuid_count(7, 0);
        if (l7.edx >> 24) & 1 == 0 || (l7.edx >> 25) & 1 == 0 {
            return false;
        }
        let granted = unsafe { request_tile_data() } == 0;
        granted || mode.as_deref() == Some("force")
    })
}

/// `arch_prctl(ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)`: 0 on success.
#[cfg(target_os = "linux")]
unsafe fn request_tile_data() -> i64 {
    const SYS_ARCH_PRCTL: i64 = 158;
    const ARCH_REQ_XCOMP_PERM: u64 = 0x1023;
    const XFEATURE_XTILEDATA: u64 = 18;
    let ret: i64;
    core::arch::asm!("syscall", inlateout("rax") SYS_ARCH_PRCTL => ret, in("rdi") ARCH_REQ_XCOMP_PERM,
        in("rsi") XFEATURE_XTILEDATA, lateout("rcx") _, lateout("r11") _, options(nostack));
    ret
}
#[cfg(not(target_os = "linux"))]
unsafe fn request_tile_data() -> i64 {
    -1
}

/// Palette 1: two sets of (C 16 x 64 B, A 16 x K B, B K/4 x 64 B) in tiles 0-2 and 3-5.
#[repr(C, align(64))]
struct TileCfg([u8; 64]);
static CFG: TileCfg = TileCfg({
    let mut c = [0u8; 64];
    c[0] = 1;
    let mut set = 0;
    while set < 2 {
        let t = 3 * set;
        c[16 + 2 * t] = 64;
        c[48 + t] = 16;
        c[16 + 2 * (t + 1)] = K as u8;
        c[48 + t + 1] = 16;
        c[16 + 2 * (t + 2)] = 64;
        c[48 + t + 2] = (K / 4) as u8;
        set += 1;
    }
    c
});

/// The tile configuration for this thread, released when dropped.
struct Tiles;
impl Tiles {
    unsafe fn load() -> Tiles {
        core::arch::asm!("ldtilecfg [{}]", in(reg) CFG.0.as_ptr(), options(nostack, readonly, preserves_flags));
        Tiles
    }
}
impl Drop for Tiles {
    fn drop(&mut self) {
        unsafe { core::arch::asm!("tilerelease", options(nostack, nomem, preserves_flags)) };
    }
}

/// One slot of a call: `C[j] (+)= A * B` on tile set `set`, from zero when `zero`.
#[inline(always)]
unsafe fn slot_step(set: usize, zero: bool, kp: *const i8, bp: *const u8, cp: *mut i32) {
    use core::arch::asm;
    match (set, zero) {
        (0, false) => asm!(
            "tileloadd tmm1, [{k} + {ks}]",
            "tileloadd tmm2, [{b} + {bs}]",
            "tileloadd tmm0, [{c} + {cs}]",
            "tdpbsud tmm0, tmm1, tmm2",
            "tilestored [{c} + {cs}], tmm0",
            k = in(reg) kp, ks = in(reg) K, b = in(reg) bp, bs = in(reg) 64usize, c = in(reg) cp, cs = in(reg) 64usize,
            options(nostack, preserves_flags)),
        (0, true) => asm!(
            "tileloadd tmm1, [{k} + {ks}]",
            "tileloadd tmm2, [{b} + {bs}]",
            "tilezero tmm0",
            "tdpbsud tmm0, tmm1, tmm2",
            "tilestored [{c} + {cs}], tmm0",
            k = in(reg) kp, ks = in(reg) K, b = in(reg) bp, bs = in(reg) 64usize, c = in(reg) cp, cs = in(reg) 64usize,
            options(nostack, preserves_flags)),
        (_, false) => asm!(
            "tileloadd tmm4, [{k} + {ks}]",
            "tileloadd tmm5, [{b} + {bs}]",
            "tileloadd tmm3, [{c} + {cs}]",
            "tdpbsud tmm3, tmm4, tmm5",
            "tilestored [{c} + {cs}], tmm3",
            k = in(reg) kp, ks = in(reg) K, b = in(reg) bp, bs = in(reg) 64usize, c = in(reg) cp, cs = in(reg) 64usize,
            options(nostack, preserves_flags)),
        (_, true) => asm!(
            "tileloadd tmm4, [{k} + {ks}]",
            "tileloadd tmm5, [{b} + {bs}]",
            "tilezero tmm3",
            "tdpbsud tmm3, tmm4, tmm5",
            "tilestored [{c} + {cs}], tmm3",
            k = in(reg) kp, ks = in(reg) K, b = in(reg) bp, bs = in(reg) 64usize, c = in(reg) cp, cs = in(reg) 64usize,
            options(nostack, preserves_flags)),
    }
}

// =============================================================================================
// the kernel's sink
// =============================================================================================

/// `vpermb` index: the low bytes of the 32 lanes into bytes 0..32, the high bytes into 32..64.
#[repr(C, align(64))]
struct Perm([u8; 64]);
static PERM: Perm = Perm({
    let mut p = [0u8; 64];
    let mut o = 0;
    while o < 64 {
        p[o] = (2 * (o % 32) + o / 32) as u8;
        o += 1;
    }
    p
});

/// One slot's accumulators: 16 rows (stream, key plane) of 16 lanes (8 columns' low then high
/// witness bytes).
#[repr(C, align(64))]
struct CTile([i32; 256]);

/// Two staging tiles for `B`.
#[repr(C, align(64))]
struct Stage([u8; 2 * (K / 4) * 64]);

/// Stream 0's running residues: `v[j][c]`, lanes `0..8`.
#[repr(C, align(64))]
struct Acc0 {
    v: [[i32; 16]; N],
}

struct AmxSink {
    /// This call's key tiles: slot `j` at `k + 16 K j`.
    k: *const i8,
    c: *mut CTile,
    b: *mut Stage,
    zero: bool,
}
impl vb::SliceSink<AL> for AmxSink {
    #[inline(always)]
    unsafe fn rows(&mut self, blk: usize, w: [*const i16; AL], r0: usize, r1: usize) {
        amx_rows(self.k, self.c, self.b, self.zero, blk, w, r0, r1);
    }
}

/// Byte-split each transform's slot into a `B` row and multiply-accumulate the slot on the tiles.
#[inline(never)]
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi")]
unsafe fn amx_rows(k: *const i8, c: *mut CTile, b: *mut Stage, zero: bool, blk: usize, w: [*const i16; AL], r0: usize, r1: usize) {
    let off = _mm512_set1_epi16(OFF);
    let perm = _mm512_load_si512(PERM.0.as_ptr() as *const __m512i);
    let stage = (*b).0.as_mut_ptr();
    for i in r0..r1 {
        let j = 27 * blk + i;
        let set = j & 1;
        let bs = stage.add(set * (K / 4) * 64);
        for h in 0..AL {
            let x = _mm512_load_si512((w[h] as *const __m512i).add(i));
            let y = _mm512_permutexvar_epi8(perm, _mm512_add_epi16(x, off));
            _mm512_store_si512(bs.add(64 * h) as *mut __m512i, y);
        }
        slot_step(set, zero, k.add(16 * K * j), bs, (*c.add(j)).0.as_mut_ptr());
    }
}

/// Per-worker scratch, allocated once and reused by every group the worker takes.
struct AmxWorker {
    c: Box<[CTile]>,
    b: Box<[Stage]>,
    pend: Box<[Batch32]>,
    total: Box<[Total]>,
    acc0: Box<[Acc0]>,
    idx: Box<[BinaryIndex32]>,
    w: Box<[Blk27]>,
}
impl AmxWorker {
    fn new() -> Self {
        AmxWorker {
            c: zeroed_slice(N),
            b: zeroed_slice(1),
            pend: zeroed_slice(DIGITS),
            total: zeroed_slice(1),
            acc0: zeroed_slice(1),
            idx: zeroed_slice(AL),
            w: zeroed_slice(2 * AL),
        }
    }
}

/// A chunk's accumulators into residues: stream 0 added to `acc0`, digit `d` into lanes
/// `8 q .. 8 q + 8` of `pend[d]` (`|r| < 2q`).
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
unsafe fn flush(c: &[CTile], acc0: &mut Acc0, pend: &mut [Batch32], q: usize) {
    for j in 0..N {
        let cp = c[j].0.as_ptr() as *const __m512i;
        for s in 0..STREAMS {
            let a = _mm512_load_si512(cp.add(2 * s));
            let hb = _mm512_load_si512(cp.add(2 * s + 1));
            // lanes 0..8: the low witness bytes' sum, 8..16 the high ones'
            let t = red16::<P>(_mm512_add_epi32(a, _mm512_slli_epi32::<7>(hb)));
            let th = _mm512_alignr_epi32::<8>(t, t);
            let v = red16::<P>(_mm512_add_epi32(t, _mm512_slli_epi32::<8>(th)));
            if s == 0 {
                let p = acc0.v[j].as_mut_ptr() as *mut __m512i;
                let old = _mm512_load_si512(p);
                _mm512_store_si512(p, _mm512_mask_add_epi32(old, 0x00FF, old, v));
            } else {
                let v16 = _mm512_cvtepi32_epi16(v);
                _mm_storeu_si128(pend[s - 1].v[j].as_mut_ptr().add(8 * q) as *mut __m128i, _mm256_castsi256_si128(v16));
            }
        }
    }
}

/// [`reconstruct`] for the AMX batch layout: chunks `c0 .. c0 + nq` of digit `d` in lane groups
/// of 8 columns, centered around their (bias-corrected) means and added to the total with weight
/// `9^d`.
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
unsafe fn reconstruct_amx(b: &Batch32, d: usize, mur: &[i16], chunks: usize, c0: usize, nq: usize, total: &mut Total) {
    let at = d * chunks + c0;
    let q = _mm512_set1_epi16(P as i16);
    let hq = _mm512_set1_epi16(((P - 1) / 2) as i16);
    let nhq = _mm512_set1_epi16(-(((P - 1) / 2) as i16));
    let wd = _mm512_set1_epi32(digit_weight(d) as i32);
    let live: u32 = if nq >= BATCH_CHUNKS { !0 } else { (1u32 << (8 * nq)) - 1 };
    for m in 0..N {
        let x = _mm512_load_si512(b.v[m].as_ptr() as *const __m512i);
        let mut mu = _mm512_set1_epi16(mur[at * N + m]);
        for g in 1..nq.min(BATCH_CHUNKS) {
            mu = _mm512_mask_mov_epi16(mu, 0xFFu32 << (8 * g), _mm512_set1_epi16(mur[(at + g) * N + m]));
        }
        let mut y = _mm512_sub_epi16(x, mu);
        y = _mm512_mask_sub_epi16(y, _mm512_cmpgt_epi16_mask(y, hq), y, q);
        y = _mm512_mask_add_epi16(y, _mm512_cmplt_epi16_mask(y, nhq), y, q);
        y = _mm512_maskz_mov_epi16(live, y);
        let lo = _mm512_mullo_epi32(_mm512_cvtepi16_epi32(_mm512_castsi512_si256(y)), wd);
        let hi = _mm512_mullo_epi32(_mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(y)), wd);
        let t = total.t[m].as_mut_ptr() as *mut __m512i;
        _mm512_store_si512(t, _mm512_add_epi32(_mm512_load_si512(t), lo));
        _mm512_store_si512(t.add(1), _mm512_add_epi32(_mm512_load_si512(t.add(1)), hi));
    }
}

/// [`write_rows`] for the AMX layout: stream 0 from `acc0` less the halves' bias terms, the tail
/// from the four lane groups of `total`; both cleared for the next group.
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
unsafe fn write_rows_amx(acc0: &mut Acc0, total: &mut Total, keys: &SmsKeys, ak: &AmxKeys, halves: core::ops::Range<usize>, gc: usize, dst: *mut [u32; N], mode: RowsMode) {
    let zero = _mm512_setzero_si512();
    let mask = _mm512_set1_epi32(TAIL_MASK as i32);
    let q = _mm512_set1_epi32(P as i32);
    let pt = total.t.as_mut_ptr() as *mut __m512i;
    let mut j0 = 0;
    while j0 < N {
        let n = 16.min(N - j0);
        let mut r0 = [zero; 16];
        let mut rt = [zero; 16];
        for s in 0..n {
            let j = j0 + s;
            let pa = acc0.v[j].as_mut_ptr() as *mut __m512i;
            let mut v0 = _mm512_load_si512(pa);
            _mm512_store_si512(pa, zero);
            for h in halves.clone() {
                v0 = _mm512_sub_epi32(v0, _mm512_set1_epi32(ak.corr0[h * N + j] as i32));
            }
            r0[s] = canon_red16::<P>(v0);
            let a = _mm512_load_si512(pt.add(2 * j));
            let b = _mm512_load_si512(pt.add(2 * j + 1));
            _mm512_store_si512(pt.add(2 * j), zero);
            _mm512_store_si512(pt.add(2 * j + 1), zero);
            let s1 = _mm512_add_epi32(a, b);
            let s2 = _mm512_add_epi32(s1, _mm512_alignr_epi32::<8>(s1, s1));
            let mu = if mode == RowsMode::First { 0 } else { keys.mu[j] as i32 };
            rt[s] = _mm512_add_epi32(s2, _mm512_set1_epi32(mu));
        }
        let c0 = transpose16x16_u32(&r0);
        let ct = transpose16x16_u32(&rt);
        let m = if n == 16 { 0xFFFF } else { (1u16 << n) - 1 };
        for c in 0..gc {
            let (db, dt) = ((dst.add(2 * c) as *mut i32).add(j0), (dst.add(2 * c + 1) as *mut i32).add(j0));
            let (mut b, mut t) = (c0[c], ct[c]);
            if mode == RowsMode::Second {
                b = _mm512_add_epi32(b, _mm512_maskz_loadu_epi32(m, db));
                b = _mm512_mask_sub_epi32(b, _mm512_cmp_epi32_mask(b, q, _MM_CMPINT_NLT), b, q);
                t = _mm512_add_epi32(t, _mm512_maskz_loadu_epi32(m, dt));
            }
            if mode != RowsMode::First {
                t = _mm512_and_si512(t, mask);
            }
            _mm512_mask_storeu_epi32(db, m, b);
            _mm512_mask_storeu_epi32(dt, m, t);
        }
        j0 += 16;
    }
}

/// Commits a group of up to 8 columns over the ring-element quads `range` (a half or the whole
/// column of `nq` quads) and writes its `2 g` rows to `dst`. `sync(quads_done)` is called after
/// every transform call so hyperthread siblings can share the key stream in L2. The calling
/// thread's tile configuration must be loaded.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vnni,gfni")]
unsafe fn group_amx<F: FnMut(usize)>(cols: &[&[B128]], keys: &SmsKeys, ak: &AmxKeys, nq: usize, range: core::ops::Range<usize>, out: RowsOut, ws: &mut AmxWorker, dst: *mut [u32; N], ph: &mut Phases, mut sync: F) {
    let mut last = std::time::Instant::now();
    let mut lap = |k: usize| {
        let now = std::time::Instant::now();
        ph.0[k] += (now - last).as_nanos() as u64;
        last = now;
    };
    let g = cols.len();
    let hq = nq / 2;
    assert!(g >= 1 && g <= G && hq % AL == 0 && 4 * nq == keys.elements);
    assert!(range.start % hq == 0 && (range.len() == nq || range.len() == hq), "whole halves of the column");
    static ZERO8: [B128; 8] = [B128::ZERO; 8];
    let (np, chunks) = (2 * nq, chunk_count(keys.elements));
    let cph = chunks_per_half(np);
    let (mut bq, mut first) = (0usize, 0usize);
    let mut zero = true;
    for b in range.clone().step_by(AL) {
        for h in 0..AL {
            let cols16: [*const B128; 16] = core::array::from_fn(|l| {
                let (c, e) = (l / 2, l % 2);
                if c < g { cols[c].as_ptr().add(16 * (b + h) + 8 * e) } else { ZERO8.as_ptr() }
            });
            crate::simd::transpose_f162::slice_b128_cols16_into(&cols16, &mut ws.idx[h]);
        }
        lap(0);
        let ins: [&BinaryIndex32; AL] = core::array::from_fn(|h| &ws.idx[h]);
        let mut sink = AmxSink { k: ak.tiles.as_ptr().add((b / AL) * N * 16 * K), c: ws.c.as_mut_ptr(), b: ws.b.as_mut_ptr(), zero };
        vb::ntt_bin_batch32_pipelined::<P, AL, _>(ins, ws.w.as_mut_ptr() as *mut i16, &mut sink);
        zero = false;
        lap(1);
        sync(b + AL);
        lap(4);
        // Chunk ends: every CQ quads into a half, and the half's end.
        let e = b + AL;
        let (eh, local) = ((e - 1) / hq, e - (e - 1) / hq * hq);
        if local % CQ == 0 || local == hq {
            let chunk = eh * cph + (local - 1) / CQ;
            if bq == 0 {
                first = chunk;
            }
            flush(&ws.c, &mut ws.acc0[0], &mut ws.pend, bq);
            lap(2);
            bq += 1;
            zero = true;
            if bq == BATCH_CHUNKS || e == range.end {
                for d in 0..DIGITS {
                    let pd = &mut ws.pend[d];
                    if bq < BATCH_CHUNKS {
                        for m in 0..N {
                            for l in 8 * bq..32 {
                                pd.v[m][l] = 0;
                            }
                        }
                    }
                    pd.representation = Representation::Ntt;
                    gen_small::intt_gen_batch32::<P>(pd);
                    reconstruct_amx(pd, d, ak.mur, chunks, first, bq, &mut ws.total[0]);
                }
                lap(3);
                bq = 0;
            }
        }
    }
    let halves = range.start / hq..range.end / hq;
    match out {
        RowsOut::Full => write_rows_amx(&mut ws.acc0[0], &mut ws.total[0], keys, ak, halves, g, dst, RowsMode::Full),
        RowsOut::Half(state) => {
            let mut done = state.lock().unwrap();
            let mode = if *done == 0 { RowsMode::First } else { RowsMode::Second };
            write_rows_amx(&mut ws.acc0[0], &mut ws.total[0], keys, ak, halves, g, dst, mode);
            *done += 1;
        }
    }
    lap(5);
}

/// Per-worker phase times of the last AMX commitment (nanoseconds), for the profile line only:
/// slicing, transform + tiles, chunk flush, inverse transform + reconstruction, sibling waits,
/// row output.
pub static PHASE_NS: [[std::sync::atomic::AtomicU64; 6]; 16] =
    [const { [const { std::sync::atomic::AtomicU64::new(0) }; 6] }; 16];

/// A worker's running phase times.
#[derive(Default)]
struct Phases([u64; 6]);
impl Phases {
    fn publish(&self, worker: usize) {
        if worker < 16 {
            for (a, v) in PHASE_NS[worker].iter().zip(self.0) {
                a.store(v, std::sync::atomic::Ordering::Relaxed);
            }
        }
    }
}

/// The phase times as ` amx_ms=slice/ntt/flush/recon/wait/rows`, averaged over `workers`.
pub fn phase_report(workers: usize) -> String {
    let w = workers.clamp(1, 16);
    let avg: Vec<String> = (0..6)
        .map(|k| {
            let sum: u64 = (0..w).map(|i| PHASE_NS[i][k].load(std::sync::atomic::Ordering::Relaxed)).sum();
            format!("{:.1}", sum as f64 / w as f64 / 1e6)
        })
        .collect();
    format!(" amx_ms={}", avg.join("/"))
}

/// Maximum lead (in quads) between two hyperthread siblings on one physical core.
const AMX_LEAD: usize = 2 * AL;

/// [`commit_paired`] for the AMX path: sibling pairs take 8-column half units, coupled within
/// [`AMX_LEAD`] quads so both read every key tile from their core's L2.
fn commit_paired_amx(keys: &SmsKeys, ak: &AmxKeys, columns: &[&[B128]], nq: usize, workers: usize, placement: Option<&crate::par::CorePlacement>, dst: usize) {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    let g = G;
    let pairs = workers / 2;
    let groups = columns.len().div_ceil(g);
    let units = groups * HALVES;
    let locks: Vec<std::sync::Mutex<u8>> = (0..groups).map(|_| std::sync::Mutex::new(0)).collect();
    let next = AtomicUsize::new(0);
    let pending: Vec<std::sync::Mutex<Option<usize>>> = (0..pairs).map(|_| std::sync::Mutex::new(None)).collect();
    let done: Vec<Progress> = (0..2 * pairs).map(|_| Progress(AtomicUsize::new(0))).collect();
    let coupled: Vec<AtomicBool> = (0..pairs).map(|_| AtomicBool::new(true)).collect();
    let t0 = std::time::Instant::now();
    crate::par::pool().run(2 * pairs, |worker| {
        let _guard = placement.and_then(|p| p.pin(worker));
        let _tiles = unsafe { Tiles::load() };
        let pair = worker / 2;
        let mine = &done[worker].0;
        let other = &done[worker ^ 1].0;
        let coupled = &coupled[pair];
        let mut ws: Option<AmxWorker> = None;
        let mut ph = Phases::default();
        let mut ran = 0u32;
        let mut progressed = 0usize;
        loop {
            let unit = {
                let mut left = pending[pair].lock().unwrap();
                match left.take() {
                    Some(u) => Some(u),
                    None => {
                        let two = coupled.load(Ordering::Relaxed);
                        let u = next.fetch_add(if two { 2 } else { 1 }, Ordering::Relaxed);
                        if two && u + 1 < units {
                            *left = Some(u + 1);
                        }
                        (u < units).then_some(u)
                    }
                }
            };
            let Some(u) = unit else { break };
            let (part, gi) = (u / groups, u % groups);
            let range = part * nq / HALVES..(part + 1) * nq / HALVES;
            let lo = gi * g;
            let n = g.min(columns.len() - lo);
            let ws = ws.get_or_insert_with(AmxWorker::new);
            ran += 1;
            let (from, base) = (range.start, progressed);
            unsafe {
                group_amx(&columns[lo..lo + n], keys, ak, nq, range.clone(), RowsOut::Half(&locks[gi]), ws, (dst as *mut [u32; N]).add(2 * lo), &mut ph, |q_done| {
                    let d = base + (q_done - from);
                    mine.store(d, Ordering::Release);
                    if !coupled.load(Ordering::Relaxed) || other.load(Ordering::Acquire) + AMX_LEAD >= d {
                        return;
                    }
                    let start = std::time::Instant::now();
                    let mut spins = 0u32;
                    while other.load(Ordering::Acquire) + AMX_LEAD < d {
                        core::hint::spin_loop();
                        spins += 1;
                        if spins % 64 == 0 && start.elapsed() > SIBLING_WAIT {
                            coupled.store(false, Ordering::Relaxed);
                            let k = DECOUPLED_PAIRS.fetch_add(1, Ordering::Relaxed) as usize;
                            if k < 16 {
                                DECOUPLED_AT[k].store(((t0.elapsed().as_micros() as u32) << 4) | worker as u32, Ordering::Relaxed);
                            }
                            return;
                        }
                    }
                })
            };
            progressed += range.len();
            mine.store(progressed, Ordering::Release);
        }
        mine.store(usize::MAX / 2, Ordering::Release);
        ph.publish(worker);
        if worker < 16 {
            WORKER_END_US[worker].store(t0.elapsed().as_micros() as u32, Ordering::Relaxed);
            WORKER_GROUPS[worker].store(ran, Ordering::Relaxed);
        }
    });
}

/// Whether the last commitment ran on AMX (for the profile line).
pub static USED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// [`commit_columns_rows`] on AMX-INT8; the caller checks [`available`]. Same rows, bit for bit.
pub fn commit_columns_rows(keys: &SmsKeys, ak: &AmxKeys, columns: &[&[B128]], workers: usize) -> Vec<[u32; N]> {
    assert!(available(), "AMX-INT8 is not available");
    assert!(!columns.is_empty(), "at least one witness column");
    let len = columns[0].len();
    for c in columns {
        assert_eq!(c.len(), len);
    }
    assert_eq!(len % 16, 0);
    let nq = len / 16;
    assert_eq!(keys.elements, 4 * nq);
    assert_eq!(ak.tiles.len(), AmxKeyData::tiles_len(keys.elements));
    assert_eq!(ak.mur.len(), keys.mur.len());
    assert_eq!(ak.corr0.len(), 2 * N);
    USED.store(true, std::sync::atomic::Ordering::Relaxed);
    let groups = columns.len().div_ceil(G);
    let mut rows: Vec<[u32; N]> = Vec::with_capacity(columns.len() * 2);
    let dst = rows.as_mut_ptr() as usize;
    let paired = if workers >= 2 && workers % 2 == 0 && groups > 1 { crate::par::CorePlacement::for_sibling_pairs(workers) } else { None };
    if let Some(placement) = paired {
        commit_paired_amx(keys, ak, columns, nq, workers, Some(&placement), dst);
    } else {
        let workers = workers.clamp(1, groups);
        let next = std::sync::atomic::AtomicUsize::new(0);
        let placement = crate::par::CorePlacement::for_logical(workers).or_else(|| crate::par::CorePlacement::for_workers(workers));
        crate::par::pool().run(workers, |worker| {
            let _guard = placement.as_ref().and_then(|p| p.pin(worker));
            let _tiles = unsafe { Tiles::load() };
            let mut ws: Option<AmxWorker> = None;
            let mut ph = Phases::default();
            loop {
                let gi = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if gi >= groups {
                    ph.publish(worker);
                    break;
                }
                let c0 = gi * G;
                let gc = G.min(columns.len() - c0);
                let ws = ws.get_or_insert_with(AmxWorker::new);
                unsafe { group_amx(&columns[c0..c0 + gc], keys, ak, nq, 0..nq, RowsOut::Full, ws, (dst as *mut [u32; N]).add(2 * c0), &mut ph, |_| {}) };
            }
        });
    }
    unsafe { rows.set_len(columns.len() * 2) };
    rows
}

#[cfg(test)]
mod tests {
    use super::*;

    fn xorshift(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    #[test]
    fn tile_planes_reconstruct_keys() {
        for k in -4860i32..=4860 {
            let lo = (k + 64).rem_euclid(128) - 64;
            let hi = (k - lo) / 128;
            assert!((-64..64).contains(&lo) && (-38..=38).contains(&hi));
            assert_eq!(lo + 128 * hi, k);
        }
    }

    #[test]
    fn perm_splits_bytes() {
        for o in 0..64 {
            let src = PERM.0[o] as usize;
            let (lane, byte) = (src / 2, src % 2);
            assert_eq!(lane, o % 32);
            assert_eq!(byte, o / 32);
        }
    }

    /// The AMX rows are the VNNI path's rows, bit for bit, for every worker count.
    #[test]
    fn amx_rows_match_vnni() {
        if !available() {
            eprintln!("AMX-INT8 not available: skipped (LABINIUS_AMX=force under an emulator)");
            return;
        }
        for elements in [512usize, 1024] {
            let ncols = 21usize;
            let mut s = 0x9e3779b97f4a7c15u64 ^ elements as u64;
            let a: Vec<u128> = (0..elements * N).map(|_| ((xorshift(&mut s) as u128) << 64 | xorshift(&mut s) as u128) % MODULUS).collect();
            let data = SmsKeyData::build(&a, elements);
            let keys = data.keys();
            let amx = AmxKeyData::build(&keys);
            let ak = amx.keys();
            let len = 4 * elements;
            let words: Vec<B128> = (0..len * ncols).map(|_| B128(((xorshift(&mut s) as u128) << 64) | xorshift(&mut s) as u128)).collect();
            let cols: Vec<&[B128]> = words.chunks_exact(len).collect();
            let want = super::super::commit_columns_rows(&keys, &cols, 1);
            for workers in [1usize, 3, 4] {
                let got = commit_columns_rows(&keys, &ak, &cols, workers);
                for c in 0..ncols {
                    assert!(got[2 * c] == want[2 * c], "base row of column {c}, {elements} elements, {workers} workers");
                    assert!(got[2 * c + 1] == want[2 * c + 1], "tail row of column {c}, {elements} elements, {workers} workers");
                }
            }
        }
    }
}
