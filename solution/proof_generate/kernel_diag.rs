//! Temporary native arithmetic diagnostics. This module is used only by build.rs.
use std::hint::black_box;
use std::time::Instant;

// Each adapter returns its complete output. Checks and logging are outside its timer.
pub fn compare<T: PartialEq>(label: &str, variants: &mut [(&str, &mut dyn FnMut() -> T); 4]) {
    const WARMUPS: usize = 2;
    const SAMPLES: usize = 8;
    const ORDERS: [[usize; 4]; SAMPLES] = [
        [0, 1, 3, 2], [1, 2, 0, 3], [2, 3, 1, 0], [3, 0, 2, 1],
        [3, 0, 2, 1], [2, 3, 1, 0], [1, 2, 0, 3], [0, 1, 3, 2],
    ];
    let mut expected = None;
    let mut samples = vec![Vec::new(); variants.len()];
    println!("cargo:warning=kernel-diag case={label} variants={} warmups={WARMUPS} samples={SAMPLES}", variants.len());
    for (name, run) in variants.iter_mut() {
        let output = black_box(run());
        if let Some(reference) = &expected {
            assert!(reference == &output, "kernel-diag mismatch case={label} variant={name}");
        } else {
            expected = Some(output);
        }
        println!("cargo:warning=kernel-diag case={label} variant={name} phase=check equality=ok");
    }
    for round in 0..WARMUPS + SAMPLES {
        for position in 0..variants.len() {
            let index = if round < WARMUPS {
                if round == 0 { position } else { variants.len() - 1 - position }
            } else {
                ORDERS[round - WARMUPS][position]
            };
            let (name, run) = &mut variants[index];
            let clock = Instant::now();
            let output = black_box(run());
            let ns = clock.elapsed().as_nanos();
            assert!(expected.as_ref() == Some(&output), "kernel-diag mismatch case={label} variant={name}");
            let phase = if round < WARMUPS { "warmup" } else { "sample" };
            let trial = if round < WARMUPS { round + 1 } else { round + 1 - WARMUPS };
            println!("cargo:warning=kernel-diag case={label} variant={name} phase={phase} round={trial} position={} elapsed_ns={ns} equality=ok", position + 1);
            if round >= WARMUPS {
                samples[index].push(ns);
            }
        }
    }
    for ((name, _), times) in variants.iter().zip(&mut samples) {
        times.sort_unstable();
        let middle = times.len() / 2;
        let median = (times[middle - 1] + times[middle]) as f64 / 2.0;
        println!("cargo:warning=kernel-diag case={label} variant={name} samples={} median_ns={median:.1}", times.len());
    }
}

fn fixed_words() -> impl Iterator<Item = u128> {
    let mut state = 0x6b65_726e_656c_6469u64;
    std::iter::from_fn(move || {
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        Some(next() as u128 | ((next() as u128) << 64))
    })
}

#[cfg(target_arch = "x86_64")]
pub fn check_transpose() {
    use core::arch::x86_64::__m512i;
    type Tile = unsafe extern "C" fn(*const u128, *mut __m512i);
    type Slice = unsafe extern "C" fn(*const u128, *mut __m512i, *const u128);
    #[repr(C, align(64))]
    #[derive(Clone, PartialEq)]
    struct Block<const N: usize>([u128; N]);

    let words: Vec<u128> = fixed_words().take(128 * 512).collect();
    let tiles = |kernel: Tile| {
        let kernel = black_box(kernel);
        let mut out = vec![Block([0u128; 64]); words.len() / 64];
        for (input, output) in words.chunks_exact(64).zip(&mut out) {
            unsafe { kernel(input.as_ptr(), output.0.as_mut_ptr().cast()); }
        }
        out
    };
    assert!(tiles(fold_transpose::fold_tile_baseline) == tiles(fold_transpose::fold_tile_unpack), "kernel-diag tile mismatch");
    println!("cargo:warning=kernel-diag case=fold-tile phase=check equality=ok words={}", words.len());

    let slices = |kernel: Slice| {
        let kernel = black_box(kernel);
        let mut out = vec![Block([0u128; 512]); 128];
        for (index, (input, output)) in words.chunks_exact(512).zip(&mut out).enumerate() {
            let next = words.as_ptr().wrapping_add(512 * ((index + 1) % 128));
            unsafe { kernel(input.as_ptr(), output.0.as_mut_ptr().cast(), next); }
        }
        out
    };
    assert!(slices(fold_transpose::fold_slice_baseline) == slices(fold_transpose::fold_slice_unpack), "kernel-diag slice mismatch");
    println!("cargo:warning=kernel-diag case=fold-slice phase=check equality=ok words={}", words.len());
}

#[cfg(target_arch = "x86_64")]
mod fold_transpose {
// Standalone experiment. The baseline is taken from the active fold source.
// Compile only for x86-64 with Sapphire Rapids or equivalent AVX-512 features.
use core::arch::x86_64::*;
#[repr(C, align(64))]
struct Aligned<T>(T);
const GF_IDENT: i64 = 0x8040_2010_0804_0201u64 as i64;

    const fn swap_index(mask: usize, value: usize) -> [u8; 64] {
        let mut out = [0; 64];
        let mut p = 0;
        while p < 64 {
            out[p] = (64 * ((p & mask != 0) as usize) + (p & !mask) + value) as u8;
            p += 1;
        }
        out
    }

    static SWAPS: Aligned<[[[u8; 64]; 2]; 4]> = Aligned([
        [swap_index(8, 0), swap_index(8, 8)],
        [swap_index(4, 0), swap_index(4, 4)],
        [swap_index(2, 0), swap_index(2, 2)],
        [swap_index(1, 0), swap_index(1, 1)],
    ]);


#[inline(always)]
    unsafe fn transpose_tile_baseline(src: *const u128, dst: *mut __m512i) {
        let mut z: [__m512i; 16] = core::array::from_fn(|i| {
            _mm512_loadu_si512((src as *const u8).add(64 * i) as *const __m512i)
        });
        macro_rules! pair {
            ($a:literal, $b:literal, $i0:ident, $i1:ident) => {{
                let (a, b) = (z[$a], z[$b]);
                z[$a] = _mm512_permutex2var_epi8(a, $i0, b);
                z[$b] = _mm512_permutex2var_epi8(a, $i1, b);
            }};
        }
        let i0 = _mm512_load_si512(SWAPS.0[0][0].as_ptr() as *const __m512i);
        let i1 = _mm512_load_si512(SWAPS.0[0][1].as_ptr() as *const __m512i);
        pair!(0, 8, i0, i1); pair!(1, 9, i0, i1); pair!(2, 10, i0, i1); pair!(3, 11, i0, i1);
        pair!(4, 12, i0, i1); pair!(5, 13, i0, i1); pair!(6, 14, i0, i1); pair!(7, 15, i0, i1);
        let i0 = _mm512_load_si512(SWAPS.0[1][0].as_ptr() as *const __m512i);
        let i1 = _mm512_load_si512(SWAPS.0[1][1].as_ptr() as *const __m512i);
        pair!(0, 4, i0, i1); pair!(1, 5, i0, i1); pair!(2, 6, i0, i1); pair!(3, 7, i0, i1);
        pair!(8, 12, i0, i1); pair!(9, 13, i0, i1); pair!(10, 14, i0, i1); pair!(11, 15, i0, i1);
        let i0 = _mm512_load_si512(SWAPS.0[2][0].as_ptr() as *const __m512i);
        let i1 = _mm512_load_si512(SWAPS.0[2][1].as_ptr() as *const __m512i);
        pair!(0, 2, i0, i1); pair!(1, 3, i0, i1); pair!(4, 6, i0, i1); pair!(5, 7, i0, i1);
        pair!(8, 10, i0, i1); pair!(9, 11, i0, i1); pair!(12, 14, i0, i1); pair!(13, 15, i0, i1);
        let i0 = _mm512_load_si512(SWAPS.0[3][0].as_ptr() as *const __m512i);
        let i1 = _mm512_load_si512(SWAPS.0[3][1].as_ptr() as *const __m512i);
        pair!(0, 1, i0, i1); pair!(2, 3, i0, i1); pair!(4, 5, i0, i1); pair!(6, 7, i0, i1);
        pair!(8, 9, i0, i1); pair!(10, 11, i0, i1); pair!(12, 13, i0, i1); pair!(14, 15, i0, i1);
        for i in 0..16 {
            _mm512_store_si512(dst.add(i), z[i]);
        }
    }

#[inline(always)]
unsafe fn transpose_tile_unpack(src: *const u128, dst: *mut __m512i) {
    let mut z: [__m512i; 16] = core::array::from_fn(|i| {
        _mm512_loadu_si512((src as *const u8).add(64 * i) as *const __m512i)
    });
    macro_rules! pair {
        ($a:literal, $b:literal, $lo:ident, $hi:ident) => {{
            let (a, b) = (z[$a], z[$b]);
            z[$a] = $lo(a, b);
            z[$b] = $hi(a, b);
        }};
    }
    pair!(0, 1, _mm512_unpacklo_epi8, _mm512_unpackhi_epi8);
    pair!(2, 3, _mm512_unpacklo_epi8, _mm512_unpackhi_epi8);
    pair!(4, 5, _mm512_unpacklo_epi8, _mm512_unpackhi_epi8);
    pair!(6, 7, _mm512_unpacklo_epi8, _mm512_unpackhi_epi8);
    pair!(8, 9, _mm512_unpacklo_epi8, _mm512_unpackhi_epi8);
    pair!(10, 11, _mm512_unpacklo_epi8, _mm512_unpackhi_epi8);
    pair!(12, 13, _mm512_unpacklo_epi8, _mm512_unpackhi_epi8);
    pair!(14, 15, _mm512_unpacklo_epi8, _mm512_unpackhi_epi8);
    pair!(0, 2, _mm512_unpacklo_epi16, _mm512_unpackhi_epi16);
    pair!(1, 3, _mm512_unpacklo_epi16, _mm512_unpackhi_epi16);
    pair!(4, 6, _mm512_unpacklo_epi16, _mm512_unpackhi_epi16);
    pair!(5, 7, _mm512_unpacklo_epi16, _mm512_unpackhi_epi16);
    pair!(8, 10, _mm512_unpacklo_epi16, _mm512_unpackhi_epi16);
    pair!(9, 11, _mm512_unpacklo_epi16, _mm512_unpackhi_epi16);
    pair!(12, 14, _mm512_unpacklo_epi16, _mm512_unpackhi_epi16);
    pair!(13, 15, _mm512_unpacklo_epi16, _mm512_unpackhi_epi16);
    pair!(0, 4, _mm512_unpacklo_epi32, _mm512_unpackhi_epi32);
    pair!(1, 5, _mm512_unpacklo_epi32, _mm512_unpackhi_epi32);
    pair!(2, 6, _mm512_unpacklo_epi32, _mm512_unpackhi_epi32);
    pair!(3, 7, _mm512_unpacklo_epi32, _mm512_unpackhi_epi32);
    pair!(8, 12, _mm512_unpacklo_epi32, _mm512_unpackhi_epi32);
    pair!(9, 13, _mm512_unpacklo_epi32, _mm512_unpackhi_epi32);
    pair!(10, 14, _mm512_unpacklo_epi32, _mm512_unpackhi_epi32);
    pair!(11, 15, _mm512_unpacklo_epi32, _mm512_unpackhi_epi32);
    pair!(0, 8, _mm512_unpacklo_epi64, _mm512_unpackhi_epi64);
    pair!(1, 9, _mm512_unpacklo_epi64, _mm512_unpackhi_epi64);
    pair!(2, 10, _mm512_unpacklo_epi64, _mm512_unpackhi_epi64);
    pair!(3, 11, _mm512_unpacklo_epi64, _mm512_unpackhi_epi64);
    pair!(4, 12, _mm512_unpacklo_epi64, _mm512_unpackhi_epi64);
    pair!(5, 13, _mm512_unpacklo_epi64, _mm512_unpackhi_epi64);
    pair!(6, 14, _mm512_unpacklo_epi64, _mm512_unpackhi_epi64);
    pair!(7, 15, _mm512_unpacklo_epi64, _mm512_unpackhi_epi64);
    // The unpack tree reverses the four column-index bits.
    // Reorder stores; no instruction is needed to reorder vector contents.
    for i in 0usize..16 {
        _mm512_store_si512(dst.add(i), z[i.reverse_bits() >> (usize::BITS - 4)]);
    }
}

#[inline(always)]
pub(crate) unsafe fn transpose8x8_q(r: [__m512i; 8]) -> [__m512i; 8] {
    let a0 = _mm512_unpacklo_epi64(r[0], r[1]);
    let a1 = _mm512_unpackhi_epi64(r[0], r[1]);
    let a2 = _mm512_unpacklo_epi64(r[2], r[3]);
    let a3 = _mm512_unpackhi_epi64(r[2], r[3]);
    let a4 = _mm512_unpacklo_epi64(r[4], r[5]);
    let a5 = _mm512_unpackhi_epi64(r[4], r[5]);
    let a6 = _mm512_unpacklo_epi64(r[6], r[7]);
    let a7 = _mm512_unpackhi_epi64(r[6], r[7]);
    let b0 = _mm512_shuffle_i64x2::<0x88>(a0, a2);
    let b1 = _mm512_shuffle_i64x2::<0xDD>(a0, a2);
    let b2 = _mm512_shuffle_i64x2::<0x88>(a4, a6);
    let b3 = _mm512_shuffle_i64x2::<0xDD>(a4, a6);
    let b4 = _mm512_shuffle_i64x2::<0x88>(a1, a3);
    let b5 = _mm512_shuffle_i64x2::<0xDD>(a1, a3);
    let b6 = _mm512_shuffle_i64x2::<0x88>(a5, a7);
    let b7 = _mm512_shuffle_i64x2::<0xDD>(a5, a7);
    [
        _mm512_shuffle_i64x2::<0x88>(b0, b2),
        _mm512_shuffle_i64x2::<0x88>(b4, b6),
        _mm512_shuffle_i64x2::<0x88>(b1, b3),
        _mm512_shuffle_i64x2::<0x88>(b5, b7),
        _mm512_shuffle_i64x2::<0xDD>(b0, b2),
        _mm512_shuffle_i64x2::<0xDD>(b4, b6),
        _mm512_shuffle_i64x2::<0xDD>(b1, b3),
        _mm512_shuffle_i64x2::<0xDD>(b5, b7),
    ]
}
    const PERM: Aligned<[u8; 64]> = Aligned({
        let mut a = [0; 64];
        let mut i = 0;
        while i < 64 {
            a[i] = (8 * (i % 8) + i / 8) as u8;
            i += 1;
        }
        a
    });


#[no_mangle]
#[inline(never)]
pub unsafe extern "C" fn fold_tile_baseline(src: *const u128, dst: *mut __m512i) {
    transpose_tile_baseline(src, dst);
}

#[no_mangle]
#[inline(never)]
    pub unsafe extern "C" fn fold_slice_baseline(words: *const u128, out: *mut __m512i, next: *const u128) {
        let mut tiles = core::mem::MaybeUninit::<[[__m512i; 16]; 8]>::uninit();
        let tiles_ptr = tiles.as_mut_ptr() as *mut __m512i;
        for t in 0..8 {
            for i in 0..16 {
                _mm_prefetch::<{ _MM_HINT_T0 }>((next as *const i8).add(1024 * t + 64 * i));
            }
            transpose_tile_baseline(words.add(64 * t), tiles_ptr.add(16 * t));
        }
        let perm = _mm512_load_si512(PERM.0.as_ptr() as *const __m512i);
        let ident = _mm512_set1_epi64(GF_IDENT);
        for b in 0..16 {
            let rows = core::array::from_fn(|t| {
                _mm512_permutexvar_epi8(
                    perm,
                    _mm512_gf2p8affine_epi64_epi8::<0>(ident, _mm512_load_si512(tiles_ptr.add(16 * t + b))),
                )
            });
            let planes = transpose8x8_q(rows);
            for p in 0..8 {
                _mm512_store_si512(out.add(8 * b + p), planes[p]);
            }
        }
    }
#[no_mangle]
#[inline(never)]
pub unsafe extern "C" fn fold_tile_unpack(src: *const u128, dst: *mut __m512i) {
    transpose_tile_unpack(src, dst);
}

#[no_mangle]
#[inline(never)]
    pub unsafe extern "C" fn fold_slice_unpack(words: *const u128, out: *mut __m512i, next: *const u128) {
        let mut tiles = core::mem::MaybeUninit::<[[__m512i; 16]; 8]>::uninit();
        let tiles_ptr = tiles.as_mut_ptr() as *mut __m512i;
        for t in 0..8 {
            for i in 0..16 {
                _mm_prefetch::<{ _MM_HINT_T0 }>((next as *const i8).add(1024 * t + 64 * i));
            }
            transpose_tile_unpack(words.add(64 * t), tiles_ptr.add(16 * t));
        }
        let perm = _mm512_load_si512(PERM.0.as_ptr() as *const __m512i);
        let ident = _mm512_set1_epi64(GF_IDENT);
        for b in 0..16 {
            let rows = core::array::from_fn(|t| {
                _mm512_permutexvar_epi8(
                    perm,
                    _mm512_gf2p8affine_epi64_epi8::<0>(ident, _mm512_load_si512(tiles_ptr.add(16 * t + b))),
                )
            });
            let planes = transpose8x8_q(rows);
            for p in 0..8 {
                _mm512_store_si512(out.add(8 * b + p), planes[p]);
            }
        }
    }
}


#[cfg(target_arch = "x86_64")]
pub fn run(keys: &[u32]) {
    use labinius::challenge::{sample_short_challenges_parallel, Transcript, DEFAULT_BOUND, DEFAULT_WEIGHT};
    use labinius::fields::scalar::B128;
    let clock = Instant::now();
    println!("cargo:warning=kernel-diag diagnostic_only=true configured_build_opt_level=3 build_codegen_units=unavailable configured_prover_codegen_units=1 configured_prover_lto=fat");
    let flags = std::env::var("CARGO_ENCODED_RUSTFLAGS")
        .or_else(|_| std::env::var("RUSTFLAGS"))
        .unwrap_or_else(|_| "unavailable".into())
        .replace('\u{1f}', " ");
    println!("cargo:warning=kernel-diag rustflags={flags:?}");
    check_transpose();
    let check_clock = Instant::now();
    fold_unpack::check_carry4_pair();
    println!("cargo:warning=kernel-diag case=carry4-pair phase=check elapsed_ms={}", check_clock.elapsed().as_millis());

    let workers = labinius::par::threads();
    if workers != 8 {
        println!("cargo:warning=kernel-diag full_tests_skipped=true workers={workers} required_workers=8");
        return;
    }
    const COLUMNS: usize = 1024;
    const COLUMN_LEN: usize = 16384;
    let words: Vec<B128> = fixed_words().take(COLUMNS * COLUMN_LEN).map(B128).collect();
    let mut transcript = Transcript::new(b"labinius-fast/kernel-diagnostic/v1");
    let challenges = sample_short_challenges_parallel(&mut transcript, COLUMNS, DEFAULT_WEIGHT, DEFAULT_BOUND, workers);
    println!("cargo:warning=kernel-diag full_input_words={} columns={COLUMNS} column_len={COLUMN_LEN} workers={workers} weight={DEFAULT_WEIGHT}", words.len());
    let mut fold_baseline = || labinius::fold::fold_witness_bitsliced(&words, COLUMNS, &challenges, COLUMN_LEN / 128, 3889, workers);
    let mut fold_unpack = || fold_unpack::fold_witness_bitsliced_unpack(&words, COLUMNS, &challenges, COLUMN_LEN / 128, 3889, workers);
    let mut fold_carry4 = || fold_unpack::fold_witness_bitsliced_carry4(&words, COLUMNS, &challenges, COLUMN_LEN / 128, 3889, workers);
    let mut fold_combined = || fold_unpack::fold_witness_bitsliced_unpack_carry4(&words, COLUMNS, &challenges, COLUMN_LEN / 128, 3889, workers);
    compare("fold-full", &mut [
        ("baseline", &mut fold_baseline),
        ("unpack", &mut fold_unpack),
        ("carry4", &mut fold_carry4),
        ("unpack-carry4", &mut fold_combined),
    ]);

    let columns: Vec<&[B128]> = words.chunks_exact(COLUMN_LEN).collect();
    let mut commit_baseline = || labinius::simd::commit::hybrid::commit_columns_rows_pairs(keys, &columns, workers);
    let mut commit_lock4 = || commit_lock4::commit_columns_rows_pairs::<4, false>(keys, &columns, workers);
    let mut commit_slicer = || commit_lock4::commit_columns_rows_pairs::<2, true>(keys, &columns, workers);
    let mut commit_combined = || commit_lock4::commit_columns_rows_pairs::<4, true>(keys, &columns, workers);
    compare("commit-full", &mut [
        ("lock2-current", &mut commit_baseline),
        ("lock4-current", &mut commit_lock4),
        ("lock2-register", &mut commit_slicer),
        ("lock4-register", &mut commit_combined),
    ]);
    println!("cargo:warning=kernel-diag complete=true elapsed_ms={}", clock.elapsed().as_millis());
}

#[cfg(target_arch = "x86_64")]
#[path = "fold_unpack_diag.rs"]
mod fold_unpack;

#[cfg(target_arch = "x86_64")]
#[path = "commit_lock4_diag.rs"]
mod commit_lock4;
