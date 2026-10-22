//! Fixed synthetic tests. This executable is separate from the prover.
mod compare;

#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
mod commit_adapter;
#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
mod four_paths;
#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
mod pair_checks;
#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
mod native;

fn main() {
    #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
    native::run();
    #[cfg(not(all(target_arch = "x86_64", target_os = "linux")))]
    {
        println!("native-diag phase=smoke native_kernels=not_run");
        let mut baseline = || vec![1u32, 2, 3];
        let mut omega = || vec![1u32, 2, 3];
        let mut lut6 = || vec![1u32, 2, 3];
        let mut combined = || vec![1u32, 2, 3];
        compare::run("host-smoke", &mut [
            ("baseline", &mut baseline), ("omega-only", &mut omega),
            ("lut6-only", &mut lut6), ("combined", &mut combined),
        ]);
    }
}
