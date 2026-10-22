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
    let dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    println!("cargo:rerun-if-changed=src/keygen.rs");
    println!("cargo:rerun-if-changed=src/modulus_choice.rs");
    println!("cargo:rerun-if-changed=src/parameters.rs");
    println!("cargo:rerun-if-changed=src/rq.rs");
    println!("cargo:rerun-if-changed=build.rs");
    let (image, tail, hybrid, pairs) = images();
    std::fs::write(dir.join("pair-key.bin"), pairs).expect("write the column-pair key image");
    std::fs::write(dir.join("key.bin"), image).expect("write the key image");
    std::fs::write(dir.join("tail-key.bin"), tail).expect("write the auxiliary key image");
    std::fs::write(dir.join("hybrid-key.bin"), hybrid).expect("write the arranged hybrid key image");
    full_process_comparison(&dir);
}

// Build-stage diagnostic only: the official prover's main and runtime stay unchanged.
fn full_process_comparison(dir: &std::path::Path) {
    println!("cargo:rerun-if-changed=research");
    println!("cargo:rerun-if-changed=.cargo/config.toml");
    println!("cargo:rerun-if-changed=../labinius/crates/pcs/src/fields/bitmat.rs");
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    {
        assert_ne!(env::var("LABINIUS_COMPARE_CHILD").as_deref(), Ok("1"), "recursive diagnostic build");
        assert_eq!(env::var("HOST").unwrap(), env::var("TARGET").unwrap(), "native host=target required");
        assert_eq!(env::var("NUM_JOBS").as_deref(), Ok("1"), "diagnostic requires single-job parent Cargo");
        assert!(host_can_run(), "unsupported diagnostic CPU features");
        let manifest = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
        let status = std::process::Command::new("/usr/bin/python3")
            .arg(manifest.join("research/compare_driver.py"))
            .arg("--solution")
            .arg(manifest.parent().unwrap())
            .arg("--out")
            .arg(dir)
            .status()
            .expect("run offline full-process comparison");
        assert!(status.success(), "full-process comparison environment/correctness failed");
    }
    #[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
    {
        let _ = dir;
        println!("cargo:warning=labinius-full-compare unsupported_host=true native_runs=0 official_score=false");
    }
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
    let pairs: Vec<u8> = pairs.iter().flat_map(|w| w.to_le_bytes()).collect();
    (Vec::new(), Vec::new(), Vec::new(), pairs)
}

#[cfg(not(target_arch = "x86_64"))]
fn images() -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>) {
    (Vec::new(), Vec::new(), Vec::new(), Vec::new())
}
