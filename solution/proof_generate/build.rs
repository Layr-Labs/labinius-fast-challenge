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
    // A matching file offset and address alignment lets the public key use huge pages.
    if env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux") {
        println!("cargo:rustc-link-arg-bins=-Wl,-z,max-page-size=0x200000");
    }
    let dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    println!("cargo:rerun-if-changed=src/keygen.rs");
    println!("cargo:rerun-if-changed=src/modulus_choice.rs");
    println!("cargo:rerun-if-changed=src/parameters.rs");
    println!("cargo:rerun-if-changed=src/rq.rs");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=../native_diagnostics");
    let (image, tail, hybrid, pairs) = images();
    std::fs::write(dir.join("pair-key.bin"), pairs).expect("write the column-pair key image");
    std::fs::write(dir.join("key.bin"), image).expect("write the key image");
    std::fs::write(dir.join("tail-key.bin"), tail).expect("write the auxiliary key image");
    std::fs::write(dir.join("hybrid-key.bin"), hybrid).expect("write the arranged hybrid key image");
    let (pairs, mur, biases, mu) = sms_images();
    std::fs::write(dir.join("sms-pairs.bin"), pairs).expect("write the single-transform key image");
    std::fs::write(dir.join("sms-mur.bin"), mur).expect("write the chunk mean image");
    std::fs::write(dir.join("sms-biases.bin"), biases).expect("write the inverse offset image");
    std::fs::write(dir.join("sms-mu.bin"), mu).expect("write the mean total image");
}

/// For the single-transform modulus: the nine streams' keys in the lockstep layout, the digit
/// chunks' mean residues and the means' total (`labinius::simd::commit::sms::SmsKeyData`), with
/// the reconstruction margin checked here, at build time, from the public matrix alone.
#[cfg(target_arch = "x86_64")]
fn sms_images() -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>) {
    if !host_can_run() || parameters::MODULUS != modulus_choice::SMS_MODULUS {
        return (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    }
    let matrix = rq::ajtai_matrix(parameters::MODULUS, parameters::RING_ELEMENTS);
    let data = labinius::simd::commit::sms::SmsKeyData::build(&matrix, parameters::RING_ELEMENTS);
    println!("cargo:warning=single-transform reconstruction margin {:.3} sigma", data.margin);
    for (d, m) in data.margins.iter().enumerate() {
        println!("cargo:warning=digit {d}: margin {:.3} at chunk {} coefficient {} |mean| {}", m.0, m.1, m.2, m.3);
    }
    assert!(data.margin >= 8.5, "the public matrix leaves too small a reconstruction margin");
    let diagnostic = unsafe { labinius::simd::commit::sms::affine_inverse_diagnostics(&data) };
    println!("cargo:warning={diagnostic}");
    let paired_commit = unsafe { labinius::simd::commit::sms::commit_affine_diagnostics(&data) };
    println!("cargo:warning={paired_commit}");
    (
        data.pairs.iter().flat_map(|w| w.to_le_bytes()).collect(),
        data.mur.iter().flat_map(|w| w.to_le_bytes()).collect(),
        data.biases.iter().flat_map(|w| w.to_le_bytes()).collect(),
        data.mu.iter().flat_map(|w| w.to_le_bytes()).collect(),
    )
}

#[cfg(not(target_arch = "x86_64"))]
fn sms_images() -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>) {
    (Vec::new(), Vec::new(), Vec::new(), Vec::new())
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

#[cfg(not(target_arch = "x86_64"))]
fn images() -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>) {
    (Vec::new(), Vec::new(), Vec::new(), Vec::new())
}

// This separate release executable records fixed synthetic tests before proving.
#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
fn native_diagnostics(dir: &std::path::Path) {
    use std::process::Command;
    let manifest = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let helper = manifest.parent().unwrap().join("native_diagnostics");
    let target = dir.join("native-diagnostics-target");
    let native = dir.join("native-diagnostics-labrador");
    copy_native_source(&manifest.join("../labinius/crates/pcs/labrador"), &native);
    let host = env::var("HOST").unwrap();
    let build = Command::new(env::var_os("CARGO").unwrap())
        .args(["build", "--locked", "--offline", "--release", "--jobs", "1", "--target", &host, "-vv", "--manifest-path"])
        .arg(helper.join("Cargo.toml"))
        .env("CARGO_TARGET_DIR", &target)
        .env("LABINIUS_LABRADOR_DIR", &native)
        .env("LABINIUS_DIAGNOSTIC_PAIR_KEY", dir.join("pair-key.bin"))
        .env("LABINIUS_DIAGNOSTIC_RUSTFLAGS", env::var("RUSTFLAGS").unwrap_or_default())
        .env("LABINIUS_DIAGNOSTIC_ENCODED_RUSTFLAGS", env::var("CARGO_ENCODED_RUSTFLAGS").unwrap_or_default())
        .env_remove("CARGO_MAKEFLAGS")
        .env_remove("MAKEFLAGS")
        .env_remove("MFLAGS")
        .output().expect("build the separate native diagnostic offline");
    let log = String::from_utf8_lossy(&build.stderr);
    for line in log.lines().filter(|line| line.contains("Running `") &&
        (line.contains("--crate-name labinius_native_diagnostics ") || line.contains("--crate-name labinius "))) {
        println!("cargo:warning=native-diag compiler={line}");
    }
    assert!(build.status.success(), "native diagnostic build failed:\n{}\n{}", String::from_utf8_lossy(&build.stdout), log);
    let run = Command::new(target.join(host).join("release/labinius-native-diagnostics"))
        .arg(dir.join("native-diagnostics-input.bin"))
        .output().expect("run the separate native diagnostic");
    for line in String::from_utf8_lossy(&run.stdout).lines() {
        println!("cargo:warning={line}");
    }
    assert!(run.status.success(), "native diagnostic failed:\n{}", String::from_utf8_lossy(&run.stderr));
}

#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
fn copy_native_source(from: &std::path::Path, to: &std::path::Path) {
    std::fs::create_dir_all(to).expect("create private native dependency directory");
    for entry in std::fs::read_dir(from).expect("read native dependency source") {
        let entry = entry.unwrap();
        let name = entry.file_name();
        if name == "libobj" || name == "liblabrador.a" {
            continue;
        }
        let target = to.join(name);
        if entry.file_type().unwrap().is_dir() {
            copy_native_source(&entry.path(), &target);
        } else {
            std::fs::copy(entry.path(), target).expect("copy private native dependency source");
        }
    }
}
