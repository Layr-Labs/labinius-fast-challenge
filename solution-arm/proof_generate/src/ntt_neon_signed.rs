//! The signed role bands of the 9721 witness transform.
use super::*;

#[path = "ntt_neon_signed_asm.rs"]
mod asm;

const BETA: [usize; 4] = [9, 495, 1125, 99];

struct Constants {
    lut: [[[[i16; 16]; 6]; 3]; 4],
    kc: [[i16; 8]; 24],
}

fn constants(kernel: &Kernel) -> &'static Constants {
    static CONSTANTS: OnceLock<Constants> = OnceLock::new();
    CONSTANTS.get_or_init(|| {
        let plan = &primes()[1];
        let p = plan.p;
        let lut = core::array::from_fn(|n| {
            core::array::from_fn(|band| {
                let weight = plan.root(BETA[n] * band) as i64;
                kernel.lut[n]
                    .map(|table| table.map(|v| super::super::centered(v as i64 * weight, p)))
            })
        });
        let mut kc = kernel.kc;
        for (n, row) in kc.iter_mut().enumerate() {
            let w = plan.exps[3][n] as usize / 3 + 1944 - BETA[n / 6];
            row[4..6].copy_from_slice(&twiddle(p, plan.root(w)));
            row[6..8].copy_from_slice(&twiddle(p, plan.root(2 * w)));
        }
        Constants { lut, kc }
    })
}

#[cfg(not(target_feature = "sve"))]
unsafe fn front(idx: &[uint8x16_t; GROUPS], x: &mut [int16x8_t; N], k: &Kernel) {
    let bytes = idx.as_ptr().cast::<u8>();
    let kvec = vld1q_s16(k.k.as_ptr());
    for (n, bands) in constants(k).lut.iter().enumerate() {
        for (band, tables) in bands.iter().enumerate() {
            let t = tables.each_ref().map(|t| table(t));
            for j in (9 * band)..(9 * (band + 1)) {
                let val = |r: usize, b: usize| {
                    let v = vld1_u8(bytes.add(packed_slot(j + 27 * r + 81 * b)));
                    let lo = vshl_n_u8::<1>(v);
                    let hi = vorr_u8(lo, vdup_n_u8(1));
                    look(
                        t[2 * r + b],
                        vzip1q_u8(vcombine_u8(lo, lo), vcombine_u8(hi, hi)),
                    )
                };
                let [(a0, b0), (a1, b1), (a2, b2)] = core::array::from_fn(|r| {
                    let u = val(r, 0);
                    let v = val(r, 1);
                    (vaddq_s16(u, v), vsubq_s16(u, v))
                });
                let (u0, u1, u2) = radix3_twiddled::<0>(a0, a1, a2, kvec);
                let (v0, v1, v2) = radix3_sixth::<0>(b0, b1, b2, kvec);
                for (i, v) in [u0, u1, u2, v0, v1, v2].into_iter().enumerate() {
                    x[162 * n + 27 * i + j] = v;
                }
            }
        }
    }
}

#[inline(always)]
pub(super) unsafe fn transform(
    idx: &[uint8x16_t; GROUPS],
    x: &mut [int16x8_t; N],
    y: &mut [int16x8_t; N],
    k: &Kernel,
) {
    #[cfg(target_feature = "sve")]
    asm::front(idx, x, k);
    #[cfg(not(target_feature = "sve"))]
    front(idx, x, k);
    asm::tail(x, y, k);
}
