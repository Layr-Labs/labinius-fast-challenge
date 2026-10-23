mod compare;

#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
mod native;

fn main() {
    #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
    native::run();
    #[cfg(not(all(target_arch = "x86_64", target_os = "linux")))]
    println!("native-context phase=smoke native_execution=not_run");
}
