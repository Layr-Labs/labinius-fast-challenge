use super::*;

// The descriptor owns no witness memory. Every current column contains one
// full 128-transform span. Next pointers are used only by non-faulting hints.
#[repr(C)]
struct Burst {
    words: [*const u8; LANES],
    next: [*const u8; LANES],
    nib: *mut uint8x16_t,
    sc: *mut int16x8_t,
    out: *mut int16x8_t,
    lut: *const i16,
    kc: *const i16,
    c: *const i16,
    t: *const i16,
    hints: u64,
}

const _: () = assert!(CHUNK == 128 && SWEEP == 128 && LOCK == 8 && N == 648);
const _: () = assert!(PF_L1 == 4 && PF_T == 16);

// Full chunks have no padded element lanes. Require all real columns in
// both paired batches so the inherited prefetch offsets also have stride 64.
#[inline(never)]
pub(super) unsafe fn produce(
    words: &[(*const u8, usize); LANES],
    next: Option<(&[(*const u8, usize); LANES], usize)>,
    length: usize, start: usize, elements: usize,
    nib: &mut [uint8x16_t; N / 4], sc: &mut [int16x8_t; N],
    xs: &mut [[int16x8_t; N]; SWEEP], kern: &Kernel,
) -> bool {
    if length != CHUNK || words.iter().any(|x| x.1 != 64)
        || next.is_some_and(|(cols, _)| cols.iter().any(|x| x.1 != 64)) {
        return false;
    }
    let desc = Burst {
        words: (*words).map(|x| x.0.add(64 * start)),
        next: next.map_or([core::ptr::null(); LANES], |(cols, at)|
            (*cols).map(|x| x.0.wrapping_add(64 * at))),
        nib: nib.as_mut_ptr(),
        sc: sc.as_mut_ptr(),
        out: xs.as_mut_ptr().cast(),
        lut: super::super::weighted::LUT.as_ptr().cast(),
        kc: kern.kc.as_ptr().cast(),
        c: kern.c.as_ptr().cast(),
        t: kern.t6.as_ptr().cast(),
        hints: u64::from(next.is_some()),
    };
    if run(&desc, kern) {
        replay_tail_sweep(words, elements, start, nib, sc, xs, kern);
    }
    true
}

const _: () = assert!(core::mem::size_of::<Burst>() == 192
    && core::mem::offset_of!(Burst, words) == 0
    && core::mem::offset_of!(Burst, next) == 64
    && core::mem::offset_of!(Burst, nib) == 128
    && core::mem::offset_of!(Burst, sc) == 136
    && core::mem::offset_of!(Burst, out) == 144
    && core::mem::offset_of!(Burst, lut) == 152
    && core::mem::offset_of!(Burst, kc) == 160
    && core::mem::offset_of!(Burst, c) == 168
    && core::mem::offset_of!(Burst, t) == 176
    && core::mem::offset_of!(Burst, hints) == 184);

#[cfg(all(target_os = "linux", target_feature = "sve"))]
core::arch::global_asm!(include_str!("forward_sealed_keep10_ntt_guard_burst_linux.s"), options(raw));
#[cfg(all(target_os = "macos", target_feature = "sve"))]
core::arch::global_asm!(include_str!("forward_sealed_keep10_ntt_guard_burst_darwin-sve.s"), options(raw));
#[cfg(all(target_os = "macos", not(target_feature = "sve")))]
core::arch::global_asm!(include_str!("forward_sealed_keep10_ntt_guard_burst_darwin-neon.s"), options(raw));

// Native verification uses the original weighted NEON front. Its linked
// body is separately checked to contain no calls or FPSR-changing opcode.
#[cfg(all(target_os = "macos", not(target_feature = "sve")))]
#[no_mangle]
unsafe extern "C" fn labinius_guard_native_front(
    nib: *const u8, sc: *mut u8, _lut: *const i16,
) {
    super::super::weighted::pass_lut_weighted_neon(&*nib.cast(), &mut *sc.cast());
}

#[cfg(any(target_os = "macos", all(target_os = "linux", target_feature = "sve")))]
unsafe fn run(d: &Burst, _kern: &Kernel) -> bool {
    unsafe extern "C" {
        fn labinius_guard_burst(d: *const u8, first: usize, end: usize) -> u64;
    }
    labinius_guard_burst(core::ptr::from_ref(d).cast(), 0, SWEEP) != 0
}

#[cfg(not(any(target_os = "macos", all(target_os = "linux", target_feature = "sve"))))]
#[inline(always)]
unsafe fn hint<const L1: bool>(p: *const u8) {
    macro_rules! four {
        ($kind:literal) => {
            core::arch::asm!(
                concat!("prfm ", $kind, ", [{p}]"),
                concat!("prfm ", $kind, ", [{p}, #64]"),
                concat!("prfm ", $kind, ", [{p}, #128]"),
                concat!("prfm ", $kind, ", [{p}, #192]"),
                p = in(reg) p, options(nostack, preserves_flags, readonly)
            )
        };
    }
    if L1 { four!("pldl1keep") } else { four!("pldl2keep") }
}

#[cfg(not(any(target_os = "macos", all(target_os = "linux", target_feature = "sve"))))]
unsafe fn run(d: &Burst, kern: &Kernel) -> bool {
    let saved = begin_tail_guard();
    let mut status = 0;
    for i in 0..SWEEP {
        let h = i / 4;
        let q = 2 * (i % 4);
        let p = (*d.words.get_unchecked(q)).add(256 * h);
        let r = (*d.words.get_unchecked(q + 1)).add(256 * h);
        if d.hints != 0 {
            if i + PF_L1 < SWEEP {
                hint::<true>(p.wrapping_add(256));
                hint::<true>(r.wrapping_add(256));
            }
            let (p2, r2) = if i + PF_T < SWEEP {
                (p.wrapping_add(1024), r.wrapping_add(1024))
            } else {
                let offset = (h + 4 - SWEEP / 4) * 256;
                (d.next.get_unchecked(q).wrapping_add(offset),
                 d.next.get_unchecked(q + 1).wrapping_add(offset))
            };
            hint::<false>(p2);
            hint::<false>(r2);
        }
        let words = [p, p.add(64), p.add(128), p.add(192), r, r.add(64), r.add(128), r.add(192)];
        let nib = &mut *d.nib.cast::<[uint8x16_t; N / 4]>();
        let sc = &mut *d.sc.cast::<[int16x8_t; N]>();
        let out = &mut *d.out.add(i * N).cast::<[int16x8_t; N]>();
        indices_asm(&words, nib);
        lut_pass(nib, sc, kern);
        status |= pass_tail_guarded(sc, out, kern);
    }
    finish_tail_guard(saved);
    status & (1 << 27) != 0
}

