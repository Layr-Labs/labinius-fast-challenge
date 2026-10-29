#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
include!(concat!(env!("OUT_DIR"), "/candidate-c.rs"));
#[cfg(not(all(target_arch = "x86_64", target_os = "linux")))]
fn main() {
    panic!("native Linux x86-64 execution is unavailable");
}
