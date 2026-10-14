//! Fixed shape parameters must match the checker; solutions may change PRIMES.
//! The field modulus is x^128 + x^7 + x^2 + x + 1 (GHASH).
//! PRIMES describes the independent integer commitment modulus, not GHASH.
pub const LOG_VARS: usize = 24;
pub const LOG_PACKED: usize = LOG_VARS - labinius::fields::crossfield::LOG_PACK;
pub const COLUMN_LOG: u32 = 6;
pub const PRIMES: &[u16] = &[3889, 2917, 9721];
