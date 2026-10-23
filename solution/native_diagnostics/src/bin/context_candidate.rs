#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
include!(concat!(env!("OUT_DIR"), "/context-candidate.rs"));

#[cfg(not(all(target_arch = "x86_64", target_os = "linux")))]
fn main() { println!("native-context candidate native_execution=not_run"); }
