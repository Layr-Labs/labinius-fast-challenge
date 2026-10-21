//! Precompute the commitment key of the default modulus. `A` is public parameter data — a
//! function of the fixed matrix seed and the modulus alone, never of the instance seed — so the
//! prover may precompute it, and here it is computed once at build time with the library's own
//! kernels and embedded in the binary. On a host without those kernels (anything but x86-64) an
//! empty image is written and the prover builds the key when it runs.
#![allow(dead_code)]
use std::env;
use std::path::PathBuf;

#[cfg(target_arch = "x86_64")]
#[path = "src/keygen.rs"]
mod keygen;
#[cfg(target_arch = "x86_64")]
#[path = "src/modulus_choice.rs"]
mod modulus_choice;
#[cfg(target_arch = "x86_64")]
#[path = "src/parameters.rs"]
mod parameters;
#[cfg(target_arch = "x86_64")]
#[path = "src/rq.rs"]
mod rq;

fn main() {
    let out = PathBuf::from(env::var("OUT_DIR").unwrap()).join("key.bin");
    println!("cargo:rerun-if-changed=src/keygen.rs");
    println!("cargo:rerun-if-changed=src/modulus_choice.rs");
    println!("cargo:rerun-if-changed=src/parameters.rs");
    println!("cargo:rerun-if-changed=src/rq.rs");
    println!("cargo:rerun-if-changed=build.rs");
    std::fs::write(&out, pair_image()).expect("write the key image");
}

#[cfg(target_arch = "x86_64")]
fn host_can_run() -> bool {
    is_x86_feature_detected!("avx512f")
        && is_x86_feature_detected!("avx512bw")
        && is_x86_feature_detected!("avx512vl")
        && is_x86_feature_detected!("avx512vbmi")
        && is_x86_feature_detected!("avx512vbmi2")
        && is_x86_feature_detected!("avx512vnni")
        && is_x86_feature_detected!("gfni")
}

/// The column-pair image of the default (hybrid) modulus: both primes' ordinary and 1024-lift
/// keys, as the flat `u32` buffer `hybrid::pair_keys` builds, little-endian.
#[cfg(target_arch = "x86_64")]
fn pair_image() -> Vec<u8> {
    if !host_can_run() || parameters::MODULUS != modulus_choice::HYBRID_MODULUS {
        return Vec::new();
    }
    let key = keygen::build_key(parameters::MODULUS);
    let tail = keygen::build_tail_key(parameters::MODULUS);
    let words = labinius::simd::commit::hybrid::pair_keys(&key, &tail, labinius::par::threads());
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}
#[cfg(not(target_arch = "x86_64"))]
fn pair_image() -> Vec<u8> {
    Vec::new()
}

#[cfg(target_arch = "x86_64")]
fn key_image() -> Vec<u8> {
    // The library's kernels need these features at run time. A build host without them (for
    // instance a cross-build for the scored CPU) writes an empty image and the prover builds the
    // key when it runs; the scored runner has every one of them.
    if !host_can_run() {
        return Vec::new();
    }
    let key = keygen::build_key(parameters::MODULUS);
    keygen::key_image(&key)
}

#[cfg(not(target_arch = "x86_64"))]
fn key_image() -> Vec<u8> {
    Vec::new()
}

#[cfg(target_arch="x86_64")]
fn tail_image()->Vec<u8> {
 // Same host guard as `key_image`: an empty image makes the prover build the tail key at run time.
 if !host_can_run() { return Vec::new(); }
 if parameters::MODULUS==modulus_choice::HYBRID_MODULUS {
  keygen::key_image(&keygen::build_tail_key(parameters::MODULUS))
 } else {Vec::new()}
}
#[cfg(not(target_arch="x86_64"))]
fn tail_image()->Vec<u8> {Vec::new()}
