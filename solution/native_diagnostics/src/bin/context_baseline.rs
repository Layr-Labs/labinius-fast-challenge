#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
include!(concat!(env!("OUT_DIR"), "/context-baseline.rs"));

#[cfg(not(all(target_arch = "x86_64", target_os = "linux")))]
fn main() { println!("native-context baseline native_execution=not_run"); }
