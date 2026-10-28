#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
mod native;
mod support;

fn main() {
    #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
    native::run();
    #[cfg(not(all(target_arch = "x86_64", target_os = "linux")))]
    panic!("candidate-a-native-v1: native Linux x86-64 execution is unavailable");
}
