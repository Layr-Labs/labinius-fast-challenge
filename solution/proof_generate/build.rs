//! Precompute the commitment key of the default modulus. `A` is public parameter data — a
//! function of the fixed matrix seed and the modulus alone, never of the instance seed — so the
//! prover may precompute it, and here it is computed once at build time with the library's own
//! kernels and embedded in the binary. On a host without those kernels (anything but x86-64) an
//! empty image is written and the prover builds the key when it runs.
#![allow(dead_code)]
use std::env;
use std::path::PathBuf;

#[cfg(target_arch = "x86_64")]
mod kernel_diag;

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
    let dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    println!("cargo:rerun-if-changed=src/keygen.rs");
    println!("cargo:rerun-if-changed=src/modulus_choice.rs");
    println!("cargo:rerun-if-changed=src/parameters.rs");
    println!("cargo:rerun-if-changed=src/rq.rs");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=kernel_diag.rs");
    println!("cargo:rerun-if-changed=fold_unpack_diag.rs");
    println!("cargo:rerun-if-changed=commit_lock4_diag.rs");
    println!("cargo:rerun-if-changed=input_slicer_diag.rs");
    let (image, tail, hybrid, pairs) = images();
    std::fs::write(dir.join("pair-key.bin"), pairs).expect("write the column-pair key image");
    std::fs::write(dir.join("key.bin"), image).expect("write the key image");
    std::fs::write(dir.join("tail-key.bin"), tail).expect("write the auxiliary key image");
    std::fs::write(dir.join("hybrid-key.bin"), hybrid).expect("write the arranged hybrid key image");
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

/// For the hybrid modulus, precomputes only `pair-key.bin` (leaving `key.bin`, `tail-key.bin`,
/// and `hybrid-key.bin` empty so the binary does not embed 41 MiB of unused keys); for any other
/// default modulus, precomputes `key.bin`.
#[cfg(target_arch = "x86_64")]
fn images() -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>) {
    // The library's kernels need these features at run time. A build host without them (for
    // instance a cross-build for the scored CPU) writes empty images and the prover builds the
    // keys when it runs; the scored runner has every one of them.
    if !host_can_run() {
        return (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    }
    let key = keygen::build_key(parameters::MODULUS);
    if parameters::MODULUS != modulus_choice::HYBRID_MODULUS {
        return (keygen::key_image(&key), Vec::new(), Vec::new(), Vec::new());
    }
    let tail = keygen::build_tail_key(parameters::MODULUS);
    let pairs = labinius::simd::commit::hybrid::pair_keys(&key, &tail, labinius::par::threads());
    kernel_diag::run(&pairs);
    let pairs: Vec<u8> = pairs.iter().flat_map(|w| w.to_le_bytes()).collect();
    (Vec::new(), Vec::new(), Vec::new(), pairs)
}

#[cfg(not(target_arch = "x86_64"))]
fn images() -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>) {
    (Vec::new(), Vec::new(), Vec::new(), Vec::new())
}
