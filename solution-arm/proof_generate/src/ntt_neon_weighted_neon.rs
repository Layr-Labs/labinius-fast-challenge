use super::{super::*, LUT};

/// The same scaled table representatives and producer order as the SVE2 path.
pub(in super::super) unsafe fn pass_lut_weighted_neon(
    packed: &[uint8x16_t; GROUPS],
    out: &mut [int16x8_t; N],
) {
    let bytes = packed.as_ptr().cast::<u8>();
    let idx: [uint8x16_t; GROUPS] = core::array::from_fn(|s| {
        let a = vld1_u8(bytes.add(packed_slot(s)));
        let low = vshl_n_u8::<1>(a);
        let high = vorr_u8(low, vdup_n_u8(1));
        vzip1q_u8(vcombine_u8(low, low), vcombine_u8(high, high))
    });
    for (n, tables) in LUT.iter().enumerate() {
        let rows: [uint8x16_t; 11] = core::array::from_fn(|i| vld1q_u8(tables[i].as_ptr().cast()));
        for j in 0..27 {
            let look = |row: usize, offset: usize| {
                vreinterpretq_s16_u8(vqtbl1q_u8(rows[row], idx[j + offset]))
            };
            let x0 = vreinterpretq_s16_u8(vqtbl2q_u8(uint8x16x2_t(rows[0], rows[1]), idx[j]));
            let y0 = look(2, 81);
            let a = vaddq_s16(x0, y0);
            let b = vsubq_s16(x0, y0);
            let p0 = vaddq_s16(look(4, 108), look(5, 54));
            let p1 = vaddq_s16(look(8, 108), look(9, 54));
            let q0 = vaddq_s16(look(3, 27), look(6, 135));
            let q1 = vaddq_s16(look(7, 27), look(10, 135));
            let (s0, s1) = (vaddq_s16(p0, q0), vaddq_s16(p1, q1));
            let (d0, d1) = (vsubq_s16(p0, q0), vsubq_s16(p1, q1));
            let values = [
                vaddq_s16(a, s0),
                vaddq_s16(a, s1),
                vsubq_s16(vsubq_s16(a, s0), s1),
                vsubq_s16(vsubq_s16(b, d0), d1),
                vaddq_s16(b, d0),
                vaddq_s16(b, d1),
            ];
            for (k, value) in values.into_iter().enumerate() {
                out[162 * n + j + 27 * k] = value;
            }
        }
    }
}
