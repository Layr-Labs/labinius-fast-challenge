//! Compile-only mirror of the build script's x86 image-generation arm: `cargo check --target
//! x86_64-apple-darwin` compiles build scripts for the *build* host, so from an ARM development
//! host the x86 arm of `build.rs` is never type-checked. Including it as a module under
//! `cfg(all(target_arch = "x86_64", test))` makes a cross-check fail this build on a typo or
//! signature drift instead of failing the scored runner's build. Nothing here is executed.

#![allow(dead_code)]

#[path = "../build.rs"]
mod build;

