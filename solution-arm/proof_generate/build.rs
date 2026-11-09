//! Precompute the commitment matrix of the default (single-transform SMS) modulus: its seven
//! transformed rows per element, the digit chunks' mean residues, the means' weighted total, and
//! the reconstruction margin and failure bound, certified here from the public matrix alone.
//! `A` is public parameter data: a function of the fixed matrix seed, the modulus and the element
//! count alone, never of the instance seed. So it is computed here, once, at build time, with the
//! prover's own NEON kernels, and embedded in the binary (the x86 track embeds its key the same
//! way). On a build host that is not AArch64, an empty image is written and the prover builds the
//! matrix when it runs.
#![allow(dead_code, unused_imports)]
use std::env;
use std::path::PathBuf;

#[cfg(target_arch = "aarch64")]
#[path = "src/ntt.rs"]
mod ntt;
#[cfg(target_arch = "aarch64")]
#[path = "src/rq.rs"]
mod rq;

/// Ring elements per column (`parameters`: 2^24 words / 1024 columns / 4); the prover checks the
/// image's length before it uses it.
const RING_ELEMENTS: usize = 4096;

fn main() {
    let dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    for f in ["build.rs", "src/ntt.rs", "src/ntt_neon.rs", "src/ntt_hybrid.rs", "src/ntt_sms.rs", "src/sms_lattice.rs", "src/rq.rs"] {
        println!("cargo:rerun-if-changed={f}");
    }
    println!("cargo:rerun-if-env-changed=CARGO_CFG_TARGET_ARCH");
    // The hybrid matrix is no longer the default's: its image stays empty and `--modulus` with
    // the hybrid modulus builds it at run time.
    std::fs::write(dir.join("hybrid-matrix.bin"), Vec::<u8>::new()).expect("write the matrix image");
    let [data, means, mu, meta] = sms_images();
    for (name, bytes) in [("sms-data.bin", data), ("sms-means.bin", means), ("sms-mu.bin", mu), ("sms-meta.bin", meta)] {
        std::fs::write(dir.join(name), bytes).expect("write an SMS image");
    }
}

#[cfg(target_arch = "aarch64")]
fn sms_images() -> [Vec<u8>; 4] {
    if env::var("CARGO_CFG_TARGET_ARCH").as_deref() != Ok("aarch64") {
        return Default::default();
    }
    let workers = std::thread::available_parallelism().map_or(1, |n| n.get());
    // `matrix` asserts the margin and the column failure bound.
    let m = ntt::neon::sms::matrix(RING_ELEMENTS, workers);
    println!(
        "cargo:warning=SMS reconstruction margin {:.3} sigma, Hoeffding failure bound {:.3e} per column, {:.3e} per 1024-column proof",
        m.margin, m.column_failure, 1024.0 * m.column_failure
    );
    m.images()
}

#[cfg(not(target_arch = "aarch64"))]
fn sms_images() -> [Vec<u8>; 4] {
    Default::default()
}

#[cfg(target_arch = "aarch64")]
fn image() -> Vec<u8> {
    // The build script runs on the host; embed only for an AArch64 target as well.
    if env::var("CARGO_CFG_TARGET_ARCH").as_deref() != Ok("aarch64") {
        return Vec::new();
    }
    let workers = std::thread::available_parallelism().map_or(1, |n| n.get());
    let m = ntt::neon::hybrid::matrix(RING_ELEMENTS, workers);
    m.data.iter().flat_map(|v| v.to_le_bytes()).collect()
}

#[cfg(not(target_arch = "aarch64"))]
fn image() -> Vec<u8> {
    Vec::new()
}
