//! Fixed synthetic tests. This executable is separate from the prover.
mod compare;

#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
mod commit_adapter;
#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
mod omega_path;
#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
mod pair_checks;
#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
mod native;
#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
mod inverse_support;
#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
pub use inverse_support::{fields, par, simd};

fn main() {
    #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
    native::run();
    #[cfg(not(all(target_arch = "x86_64", target_os = "linux")))]
    {
        println!("native-diag phase=smoke native_kernels=not_run");
        let mut baseline = || vec![1u32, 2, 3];
        let mut omega = || vec![1u32, 2, 3];
        compare::run("host-smoke", &mut baseline, &mut omega);
    }
}
