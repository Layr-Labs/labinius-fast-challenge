#![allow(dead_code)]
pub use labinius::par;
pub mod simd {
    pub mod transpose_f162 { include!("inverse/native/transpose.rs"); }
}
pub mod fields {
    pub use labinius::fields::{scalar, bitmat};
    pub mod f162 { include!("inverse/source/f162.rs"); }
    pub mod sumcheck { include!("inverse/source/sumcheck.rs"); }
    pub mod candidate { include!("inverse/source/crossfield_candidate.rs"); }
    pub mod baseline_counted { include!("inverse/source/crossfield_baseline_counted.rs"); }
    pub mod candidate_counted { include!("inverse/source/crossfield_candidate_counted.rs"); }
}
#[path="inverse/native/compare.rs"]
pub mod compare;
