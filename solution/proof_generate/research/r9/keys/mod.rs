//! Fixed public matrix -> compact signed-byte E windows for the native falsifier.
//!
//! This module contains no instance data and performs no timing. Call `build_image`
//! outside the candidate timer, then copy its result into a 64-byte-aligned buffer.
//! The caller supplies the production `crate::rq` and `crate::parameters` modules.
use labinius::params::{subring_exp, Params};

const BR64: usize = 64;
const QUARTERS: usize = 4;
const PRIMES: usize = 2;
const FACTORS: usize = 6;
const FORMS: usize = 3;
const STREAMS: usize = 2;
const RING4_ROWS: usize = 16;
const E_PAD_BYTES: usize = 256;
const DEGREE: usize = 27;
const COEFFICIENTS: usize = 648;
const MODULUS: u128 = 38_712_288_256;

/// Exact `[64][4][2][6][3][2][16][256]` image size, in bytes.
pub const IMAGE_BYTES: usize =
    BR64 * QUARTERS * PRIMES * FACTORS * FORMS * STREAMS * RING4_ROWS * E_PAD_BYTES;
const _: () = assert!(IMAGE_BYTES == 75_497_472);

#[inline]
fn row_offset(
    br: usize,
    quarter: usize,
    prime: usize,
    factor: usize,
    form: usize,
    stream: usize,
    ring4_row: usize,
) -> usize {
    ((((((br * QUARTERS + quarter) * PRIMES + prime) * FACTORS + factor) * FORMS + form) * STREAMS
        + stream)
        * RING4_ROWS
        + ring4_row)
        * E_PAD_BYTES
}

#[inline]
fn center(x: i64, p: i64) -> i32 {
    let residue = x.rem_euclid(p);
    (if residue > p / 2 {
        residue - p
    } else {
        residue
    }) as i32
}

/// Base 128, balanced low digit, with the third form ready for Karatsuba.
/// Centering must occur BEFORE this function; weighted E entries are centered
/// independently rather than multiplying already decomposed digits by `a`.
#[inline]
fn forms(centered: i32, p: i64) -> [i8; FORMS] {
    assert!(centered.abs() <= (p / 2) as i32);
    let hi = (centered + 64).div_euclid(128);
    let lo = centered - 128 * hi;
    debug_assert!((-64..=63).contains(&lo));
    debug_assert_eq!(lo + 128 * hi, centered);
    [
        i8::try_from(lo).expect("centered low digit fits i8"),
        i8::try_from(hi).expect("centered high digit fits i8"),
        i8::try_from(lo + hi).expect("centered digit sum fits i8"),
    ]
}

// Imported public-key-only coupled lift, present in both PR126 (52f94604)
// and PR127 (86c9e9aa). Their lift_tail_pair implementations are identical
// after whitespace normalization. Strict comparison preserves centered ties.
fn lift_tail_pair(a: i16, b: i16) -> (i16, i16) {
    let (a, b) = (a as i32, b as i32);
    let norm = |a: i32, b: i32| a.abs() + b.abs() + (a + b).abs();
    let mut best = (a, b);
    let mut bound = norm(a, b);
    for u in -1..=1 {
        for v in -1..=1 {
            let (x, y) = (a + 1024 * u, b + 1024 * v);
            let next = norm(x, y);
            if next < bound {
                best = (x, y);
                bound = next;
            }
        }
    }
    (best.0 as i16, best.1 as i16)
}

fn coupled_tail_coefficients(a: &[u128]) -> [i16; 648] {
    assert_eq!(a.len(), 648, "one complete ring element is required");
    let mut out = core::array::from_fn(|j| {
        let v = (a[j] & 1023) as i16;
        if v >= 512 {
            v - 1024
        } else {
            v
        }
    });
    for j in 0..324 {
        (out[j], out[j + 324]) = lift_tail_pair(out[j], out[j + 324]);
    }
    out
}

fn emit_prime<const P: u16>(
    raw: &[u128],
    tail: &[[i16; COEFFICIENTS]],
    prime_index: usize,
    image: &mut [i8],
) {
    let p = i64::from(P);
    // `ZETA_L4` is the NEXT radix-three split twiddle. The degree-27
    // quotient is X^27-a with a=psi^subring_exp(4,blk), not ZETA_L4[blk].
    let a: [i64; 24] =
        core::array::from_fn(|blk| i64::from(Params::<P>::psi_pow(subring_exp(4, blk))));
    for ring in 0..BR64 * 64 {
        let source = &raw[ring * COEFFICIENTS..(ring + 1) * COEFFICIENTS];
        let mut residues = [[0i64; COEFFICIENTS]; STREAMS];
        for (j, &value) in source.iter().enumerate() {
            residues[0][j] = (value % u128::from(P)) as i64;
            // Use the SAME coupled rawT for both primes; do not re-center mod1024 here.
            residues[1][j] = i64::from(tail[ring][j]).rem_euclid(p);
        }
        let br = ring / 64;
        let ring4_row = ring % 64 / 4;
        let ring4_lane = ring % 4;
        for blk in 0..24 {
            let quarter = blk / FACTORS;
            let factor = blk % FACTORS;
            for stream in 0..STREAMS {
                for j in 0..DEGREE {
                    // Ai(X) mod (X^27-a): sum_t Ai[j+27*t] * a^t mod p.
                    // Horner form uses exact small integer arithmetic, not an NTT
                    // approximation or the fully split arranged key image.
                    let mut sum = 0i64;
                    for t in (0..24).rev() {
                        sum = (sum * a[blk] + residues[stream][j + DEGREE * t]) % p;
                    }
                    let entries = [center(a[blk] * sum, p), center(sum, p)];
                    for (half, &entry) in entries.iter().enumerate() {
                        let digit_forms = forms(entry, p);
                        for form in 0..FORMS {
                            let base = row_offset(
                                br,
                                quarter,
                                prime_index,
                                factor,
                                form,
                                stream,
                                ring4_row,
                            );
                            image[base + 4 * (half * DEGREE + j) + ring4_lane] = digit_forms[form];
                        }
                    }
                }
            }
        }
    }
}

/// Build the complete image from the ACTUAL production fixed public matrix.
///
/// Axes: `[br64][quarter][prime][factor][form][stream][ring4_row][E_byte]`.
/// Prime order is 3889,9721; form order lo,hi,lo+hi; stream order ordinary A,
/// coupled-mod-1024 auxiliary T. The adjacent form/stream axis is form*2+stream.
/// Each row contains interleaved four-ring E=[center(a*A),center(A)] coefficients
/// in its first 216 bytes and zero padding in bytes 216..256. An output tile uses
/// byte base `(27-i+o)*4`, including pass1's o=16 (offset64), and positive row
/// stride256. All loads for the padded last output columns remain within the row.
///
/// `Vec<i8>` carries no special alignment guarantee. The native harness must
/// copy/align the buffer and KEEP THIS PRECOMPUTATION OUTSIDE THE CANDIDATE TIMER.
/// This is a large public-data precomputation and is not run on the development Mac.
pub fn build_image() -> Vec<i8> {
    assert_eq!(
        crate::parameters::MODULUS,
        MODULUS,
        "frozen two-prime hybrid Q"
    );
    assert_eq!(
        crate::parameters::RING_ELEMENTS,
        4096,
        "fixed public key shape"
    );
    let raw = crate::rq::ajtai_matrix(MODULUS, crate::parameters::RING_ELEMENTS);
    assert_eq!(raw.len(), 4096 * COEFFICIENTS);
    let tail: Vec<[i16; COEFFICIENTS]> = raw
        .chunks_exact(COEFFICIENTS)
        .map(coupled_tail_coefficients)
        .collect();
    // Fixed public-data precomputation only, never fresh-witness work. The
    // actual certificate accepts an explicit eight Batch32 = 256-ring chunk.
    let bound = labinius::simd::commit::hybrid::certify(&tail, 8);
    assert_eq!(
        bound, 17_639_740,
        "frozen actual-public-key coupled256 certificate"
    );
    let mut image = vec![0i8; IMAGE_BYTES];
    emit_prime::<3889>(&raw, &tail, 0, &mut image);
    emit_prime::<9721>(&raw, &tail, 1, &mut image);
    image
}
