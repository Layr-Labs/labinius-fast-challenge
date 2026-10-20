//! Fixed shape parameters must match the checker; solutions may change MODULUS.
//! The field modulus is x^128 + x^7 + x^2 + x + 1 (GHASH).
//! MODULUS describes the independent integer commitment modulus, not GHASH.
pub const LOG_VARS: usize = 31;
pub const LOG_PACKED: usize = LOG_VARS - labinius::fields::crossfield::LOG_PACK;
pub const COLUMN_LOG: u32 = 10;
/// Columns of the witness, and so the number of commitment columns and folding challenges.
pub const COLUMNS: usize = 1 << COLUMN_LOG;
/// One column in `F162` elements.
pub const COLUMN_LEN: usize = 1 << (LOG_PACKED - COLUMN_LOG as usize);
/// Ring elements of one column: four `F162` make one element of `R_648`.
pub const RING_ELEMENTS: usize = COLUMN_LEN / 4;
/// The product of three limbs: no pair of the supported primes reaches the checker's 2^35
/// threshold. 3889 * 9721 * 2917 (about 36.7 bits): the two fully splitting primes below 2^14
/// run the hand-scheduled `ntt::bin_asm`, 2917 is the cheaper of the quadratic-slot limbs, and
/// a measured sweep over all triples on the scored CPU (submission d1c3cf12) put this one
/// first — the quadratic kernels cost more there than the baseline's 3889 * 2917 * 4861 assumed.
/// 3889 stays the base, so the fold's regenerated transform is `ntt::bin_asm`. Five bytes per
/// commitment coefficient either way.
pub const MODULUS: u128 = 110_277_094_573;
