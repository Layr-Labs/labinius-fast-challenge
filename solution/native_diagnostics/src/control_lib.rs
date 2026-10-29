#![allow(clippy::needless_range_loop)]
#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
extern crate labinius as _;
#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
include!(concat!(env!("OUT_DIR"), "/control-lib.rs"));
