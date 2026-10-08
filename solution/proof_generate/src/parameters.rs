//! Fixed shape parameters must match the checker; solutions may change PRIMES.
//! The field modulus is x^128 + x^7 + x^2 + x + 1 (GHASH).
//! PRIMES describes the independent integer commitment modulus, not GHASH.
pub const LOG_VARS: usize = 31;
pub const LOG_PACKED: usize = LOG_VARS - labinius::fields::crossfield::LOG_PACK;
pub const COLUMN_LOG: u32 = 10;
/// Three limbs, the minimum the checker's minimum-modulus threshold accepts: no pair
/// of the supported primes reaches 2^35. This is the cheapest passing triple, product
/// 55144219393 at about 101.9 bits. 3889 is a fully splitting base, so the fold stays
/// on `ntt::bin_asm`; 2917 and 4861 are the quadratic extra limbs.
pub const PRIMES: &[u16] = &[3889, 2917, 4861];
