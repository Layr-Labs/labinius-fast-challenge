// Temporary omega NTT check. No production caller.
#![allow(dead_code)]
macro_rules! r3_twiddles {
    ($n:literal, $pair:path, $zetas:expr) => {{
        let mut t = [[0u32; 4]; $n];
        let mut i = 0;
        while i < $n {
            t[i] = $pair($zetas[i]);
            i += 1;
        }
        t
    }};
}
#[path="omega.rs"] pub mod omega;
