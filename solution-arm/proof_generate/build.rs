//! Precompute the commitment key of the default modulus. `A` is public parameter data — a
//! function of the fixed matrix seed and the modulus alone, never of the instance seed — so the
//! prover may precompute it, and here it is computed once at build time with the library's own
//! kernels and embedded in the binary: on the scored Graviton the native AArch64 build runs the
//! `arch` port of the kernels here. An x86-64 build host without the AVX-512 features the
//! kernels need writes an empty image and the prover builds the key when it runs.
#![allow(dead_code)]
use std::env;
use std::path::PathBuf;

#[path = "src/keygen.rs"]
mod keygen;
#[path = "src/modulus_choice.rs"]
mod modulus_choice;
#[path = "src/parameters.rs"]
mod parameters;
#[path = "src/rq.rs"]
mod rq;

fn main() {
    let dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    println!("cargo:rerun-if-changed=src/keygen.rs");
    println!("cargo:rerun-if-changed=src/modulus_choice.rs");
    println!("cargo:rerun-if-changed=src/parameters.rs");
    println!("cargo:rerun-if-changed=src/rq.rs");
    println!("cargo:rerun-if-changed=build.rs");
    // 2 MiB segment alignment, so the 2 MiB-aligned key image (main.rs) can be mapped with PMD
    // entries instead of ~11.7k 4 KiB ones (e-man07, submission 721d445d).
    println!("cargo:rustc-link-arg-bins=-Wl,-z,max-page-size=0x200000");
    let (image, tail, hybrid, pairs) = images();
    std::fs::write(dir.join("pair-key.bin"), pairs).expect("write the column-pair key image");
    std::fs::write(dir.join("key.bin"), image).expect("write the key image");
    std::fs::write(dir.join("tail-key.bin"), tail).expect("write the auxiliary key image");
    std::fs::write(dir.join("hybrid-key.bin"), hybrid).expect("write the arranged hybrid key image");
    let (pairs, mur, mu) = sms_images();
    std::fs::write(dir.join("sms-pairs.bin"), pairs).expect("write the single-transform key image");
    std::fs::write(dir.join("sms-mur.bin"), mur).expect("write the chunk mean image");
    std::fs::write(dir.join("sms-mu.bin"), mu).expect("write the mean total image");
}

/// For the single-transform modulus: the seven streams' keys in the lockstep layout, the digit
/// chunks' mean residues and the means' total (`labinius::simd::commit::sms::SmsKeyData`), with
/// the reconstruction margin checked here, at build time, from the public matrix alone.
fn sms_images() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    if !host_can_run() || parameters::MODULUS != modulus_choice::SMS_MODULUS {
        return (Vec::new(), Vec::new(), Vec::new());
    }
    let matrix = rq::ajtai_matrix(parameters::MODULUS, parameters::RING_ELEMENTS);
    let data = labinius::simd::commit::sms::SmsKeyData::build(&matrix, parameters::RING_ELEMENTS);
    println!("cargo:warning=single-transform reconstruction margin {:.3} sigma", data.margin);
    for (d, m) in data.margins.iter().enumerate() {
        println!("cargo:warning=digit {d}: margin {:.3} at chunk {} coefficient {} |mean| {}", m.0, m.1, m.2, m.3);
    }
    // Public lattice coefficient banks: certify every public chunk, digit and coefficient.
    // No instance data enters the key image or this reconstruction-margin check.
    assert!(data.margin >= 7.5, "the public matrix leaves too small a reconstruction margin");
    (
        data.pairs.iter().flat_map(|w| w.to_le_bytes()).collect(),
        data.mur.iter().flat_map(|w| w.to_le_bytes()).collect(),
        data.mu.iter().flat_map(|w| w.to_le_bytes()).collect(),
    )
}

/// The build script runs on the build host, so this is about the host: off x86-64 the kernels
/// are the portable `arch` port and always run.
#[cfg(not(target_arch = "x86_64"))]
fn host_can_run() -> bool {
    true
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
fn images() -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>) {
    // The library's kernels need these features at run time. A build host without them (for
    // instance a cross-build for the scored CPU) writes empty images and the prover builds the
    // keys when it runs; the scored runner has every one of them.
    if !host_can_run() || parameters::MODULUS == modulus_choice::SMS_MODULUS {
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
