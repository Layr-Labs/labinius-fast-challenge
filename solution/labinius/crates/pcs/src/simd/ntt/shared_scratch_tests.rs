//! Each leaf has a canary vector immediately before and after its exact callback-sized
//! block. Unlike whole-buffer guards, these catch writes across any leaf boundary.
use super::*;

#[repr(C, align(64))]
struct Guarded {
    before: [i16; 32],
    data: [i16; 27 * 32],
    after: [i16; 32],
}

fn guarded(poison: i16) -> Box<Guarded> {
    Box::new(Guarded {
        before: [0x6b6b; 32],
        data: [poison; 27 * 32],
        after: [-0x3535; 32],
    })
}

fn guards(g: &Guarded) {
    assert_eq!(g.before, [0x6b6b; 32]);
    assert_eq!(g.after, [-0x3535; 32]);
}

unsafe fn check<const Q: u16>() {
    let t = tables::<Q>();
    let cv = t.cv.as_ptr().cast::<i16>();
    let mut rng = crate::rng::Rng::new(0x27ca_0000 ^ Q as u64);
    for kk in 0..24 {
        let (t4, t5, t6) = if Q == 9721 {
            (
                SMS_TWIDDLES.t4.as_ptr().add(4 * kk).cast::<u32>(),
                SMS_TWIDDLES.t5.as_ptr().add(12 * kk).cast::<u32>(),
                SMS_TWIDDLES.t6.as_ptr().add(36 * kk).cast::<u32>(),
            )
        } else {
            (
                t.tw4.as_ptr().add(kk).cast::<u32>(),
                t.tw5.as_ptr().add(3 * kk).cast::<u32>(),
                t.tw6.as_ptr().add(9 * kk).cast::<u32>(),
            )
        };
        for fixture in 0..10 {
            let mut input = guarded(0);
            for (i, x) in input.data.iter_mut().enumerate() {
                // Alias equivalence also holds for arbitrary i16 patterns, not just
                // valid prefixes. Complete NTT correctness uses binary fixtures separately.
                *x = match fixture {
                    0 => 0,
                    1 => 1,
                    2 => {
                        if i % 2 == 0 {
                            i16::MIN
                        } else {
                            i16::MAX
                        }
                    }
                    3 => {
                        if i % 32 == kk {
                            29000
                        } else {
                            -29000
                        }
                    }
                    _ => rng.next_u64() as i16,
                };
            }
            let before = input.data;
            let mut separate = guarded(0x5a5a);
            let mut inplace = guarded(-0x2525);
            inplace.data.copy_from_slice(&input.data);
            tail27_out_of_line::<Q>(
                input.data.as_ptr(),
                separate.data.as_mut_ptr(),
                t4,
                t5,
                t6,
                cv,
            );
            let p = inplace.data.as_mut_ptr();
            tail27_out_of_line::<Q>(p, p, t4, t5, t6, cv);
            assert_eq!(
                inplace.data, separate.data,
                "Q={Q} block={kk} fixture={fixture}"
            );
            assert_eq!(input.data, before);
            guards(&input);
            guards(&separate);
            guards(&inplace);
        }
    }
}

#[test]
fn leaf_inplace_canaries_3889() {
    unsafe {
        check::<3889>();
    }
}

#[test]
fn leaf_inplace_canaries_9721() {
    unsafe {
        check::<9721>();
    }
}
