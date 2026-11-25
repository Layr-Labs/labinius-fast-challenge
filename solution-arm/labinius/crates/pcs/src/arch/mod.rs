//! The vector instruction set the kernels are written against.
//!
//! On x86-64 this is `core::arch::x86_64` itself, so the AVX-512 kernels compile to exactly the
//! instructions they name. On every other target it is [`portable`]: the same intrinsic names and
//! types with their Intel-documented lane semantics in plain Rust (NEON where it maps one to one),
//! so the kernels — and everything built on them — run unchanged, bit for bit, at lower speed.

#[cfg(target_arch = "x86_64")]
pub use core::arch::x86_64::*;

#[cfg(not(target_arch = "x86_64"))]
mod portable;
#[cfg(not(target_arch = "x86_64"))]
pub use portable::*;
