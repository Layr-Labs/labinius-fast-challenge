//! Fixed synthetic PMU test. This executable is separate from the prover.
mod compare;

#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
mod pmu_probe;
#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
mod native;

fn main() {
    #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
    native::run();
    #[cfg(not(all(target_arch = "x86_64", target_os = "linux")))]
    {
        println!("native-diag phase=smoke native_kernels=not_run counters=not_run");
        let mut run = || vec![1u32, 2, 3];
        let mut profile = |_: usize, call: &mut dyn FnMut() -> (Vec<u32>, u128)| call();
        compare::run("host-smoke", &mut run, &mut profile);
    }
}
