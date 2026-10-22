//! Fixed synthetic tests. This executable is separate from the prover.
mod compare;

#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
mod commit_adapter;
#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
mod ntt_radix_order;
#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
mod native;

fn main() {
    #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
    native::run();
    #[cfg(not(all(target_arch = "x86_64", target_os = "linux")))]
    {
        println!("native-diag phase=smoke native_kernels=not_run");
        let mut baseline = || vec![1u32, 2, 3];
        let mut candidate = || vec![1u32, 2, 3];
        compare::run("host-smoke", &mut baseline, &mut candidate);
    }
}
