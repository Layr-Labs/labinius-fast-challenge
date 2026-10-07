//! Fixed shape parameters must match the checker; solutions may change PRIMES.
//! The field modulus is x^128 + x^7 + x^2 + x + 1 (GHASH).
//! PRIMES describes the independent integer commitment modulus, not GHASH.
pub const LOG_VARS: usize = 24;
pub const LOG_PACKED: usize = LOG_VARS - labinius::fields::crossfield::LOG_PACK;
pub const COLUMN_LOG: u32 = 6;
/// Two limbs, the minimum the checker's 100-bit table accepts.
/// 9721 is a fully splitting base, so the fold stays on `ntt::bin_asm`.
/// 12637 is the quadratic extra limb. Product 122844277 is about 102 bits.
pub const PRIMES: &[u16] = &[9721, 12637];
