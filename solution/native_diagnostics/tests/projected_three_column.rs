#![allow(dead_code)]
#[cfg(not(target_arch = "x86_64"))]
compile_error!("Run the actual x86-64 row functions; no portable substitute is allowed.");
pub use labinius::{challenge, fields, params, ring, par};
include!(concat!(env!("OUT_DIR"), "/projected-three-column-eval.rs"));
mod old_control {
    include!(concat!(env!("OUT_DIR"), "/projected-three-column-control-eval.rs"));
}
mod projected_three_column_checks {
use super::*;
use crate::fields::scalar::B128;
fn rng(s:&mut u64)->u64{*s^=*s<<13;*s^=*s>>7;*s^=*s<<17;*s}
fn field(s:&mut u64)->F162{F162([rng(s),rng(s),rng(s)&((1<<34)-1)])}
fn basis(i:usize)->F162{let mut x=[0;3];x[i/64]=1<<(i%64);F162(x)}
fn point(log:usize,mode:usize)->Vec<F162>{let mut s=0x12ab6790cdef5478;(0..log).map(|i|match mode{0=>F162::ZERO,1=>F162::ONE,2=>F162([1<<63,1u64<<(i%64),1<<33]),3=>F162([u64::MAX,u64::MAX,(1<<34)-1]),_=>field(&mut s)}).collect()}
fn words(n:usize)->Vec<B128>{let mut s=0xfedc0123456789ab;(0..n).map(|i|B128(match i%9{0=>0,1=>u128::MAX,2=>1u128<<63,3=>1u128<<64,4=>1u128<<127,_=>rng(&mut s)as u128|((rng(&mut s)as u128)<<64)})).collect()}
fn reference_dot(eq:&[F162],w:&[B128])->F162{eq.iter().zip(w).fold(F162::ZERO,|a,(&e,&b)|a+reference_mul(e,F162::from_b128(b)))}
fn clmul(a:u64,b:u64)->u128{let mut z=0u128;for i in 0..64{if(b>>i)&1!=0{z^=(a as u128)<<i;}}z}
fn reference_mul(a:F162,r:F162)->F162{let mut p=[0u64;6];for i in 0..3{for j in 0..3{let z=clmul(a.0[i],r.0[j]);p[i+j]^=z as u64;p[i+j+1]^=(z>>64)as u64;}}for d in(162..384).rev(){if(p[d/64]>>(d%64))&1!=0{for e in[d,d-81,d-162]{p[e/64]^=1u64<<(e%64);}}}F162([p[0],p[1],p[2]])}

fn oracle_eq(p: &[F162]) -> Vec<F162> {
    let mut out = vec![F162::ONE];
    for &v in p {
        let n = out.len();
        out.resize(2 * n, F162::ZERO);
        for i in 0..n {
            let product = reference_mul(out[i], v);
            out[i + n] = product;
            out[i] += product;
        }
    }
    out
}
fn check_private(n: usize, mode: usize, alias: bool) {
    let max = F162([u64::MAX, u64::MAX, (1 << 34) - 1]);
    let mut state = 0x67121fcca5194b03;
    let e: Vec<_> = (0..n).map(|i| match mode {
        0 => F162::ZERO, 1 => max, 2 => basis([63,64,127,161][i % 4]),
        3 => F162::ONE, _ => field(&mut state),
    }).collect();
    let w0: Vec<_> = (0..n).map(|i| B128(match mode {
        0 => 0, 1 => u128::MAX, 2 => 1u128 << [63,64,127][i % 3],
        3 => 1, _ => rng(&mut state) as u128 | ((rng(&mut state) as u128) << 64),
    })).collect();
    let w1: Vec<_> = (0..n).map(|i| if mode == 5 {w0[i]} else {B128(w0[i].0.rotate_left(17) ^ (i + 1) as u128)}).collect();
    let w2: Vec<_> = (0..n).map(|i| if mode == 5 {w0[i]} else {B128(w0[i].0.rotate_left(41) ^ (i + 3) as u128)}).collect();
    let columns: [&[B128]; 3] = if alias {[&w0,&w0,&w0]} else {[&w0,&w1,&w2]};
    let p = Poly::from_scalars(&e);
    let expected = columns.map(|w| reference_dot(&e,w));
    unsafe {
        let got = dot_words3(&p,columns[0].as_ptr(),columns[1].as_ptr(),columns[2].as_ptr(),n);
        let pair = dot_words2(&p,columns[0].as_ptr(),columns[1].as_ptr(),n);
        let old = [pair[0],pair[1],dot_words(&p,columns[2].as_ptr(),n)];
        assert_eq!(old,expected,"old tails n={n} mode={mode} alias={alias}");
        assert_eq!(got,expected,"triple n={n} mode={mode} alias={alias}");
        for j in 0..3 {for k in 0..3 {assert_eq!(got[j].0[k],expected[j].0[k]);}}
    }
}
fn check_outputs(w: &[B128], p: &[F162], workers: usize, expected: &[F162]) {
    let view = row_evaluate_words(w,p,workers);
    let installed = labinius::eval::row_evaluate_words(w,p,workers);
    let control = old_control::row_evaluate_words(w,p,workers);
    assert_eq!(view,expected,"private-view public row workers={workers}");
    assert_eq!(installed,expected,"installed public row workers={workers}");
    assert_eq!(control,expected,"complete old public row workers={workers}");
    for j in 0..expected.len() {for k in 0..3 {
        assert_eq!(installed[j].0[k],expected[j].0[k],"column={j} limb={k}");
    }}
}
fn check_row(log: usize, columns: usize, workers: usize) {
    let p = point(log,4);
    let n = 1usize << log;
    let w = words(n * columns);
    let eq = oracle_eq(&p);
    let expected: Vec<_> = w.chunks_exact(n).map(|c| reference_dot(&eq,c)).collect();
    check_outputs(&w,&p,workers,&expected);
    let actual_eq = eq_poly(&p,workers);
    assert_eq!((0..n).map(|i| actual_eq.get(i)).collect::<Vec<_>>(),eq);
}
fn oracle_weight(p: &[F162], index: usize) -> F162 {
    p.iter().enumerate().fold(F162::ONE, |a,(bit,&v)| {
        reference_mul(a,if (index >> bit) & 1 == 1 {v} else {F162::ONE + v})
    })
}
fn check_sparse_scored_row() {
    let p = point(14,4);
    let n = 1usize << 14;
    let mut w = vec![B128::ZERO;n * 1024];
    let support = [0,n / 2,n - 1];
    let eq = support.map(|index| oracle_weight(&p,index));
    let mut expected = Vec::with_capacity(1024);
    for column in 0..1024 {
        let values = [B128((column + 1) as u128), B128(((column + 1) as u128) << 64), B128((1u128 << 127) ^ column as u128)];
        let mut total = F162::ZERO;
        for k in 0..3 {
            w[column * n + support[k]] = values[k];
            total += reference_mul(eq[k],F162::from_b128(values[k]));
        }
        expected.push(total);
    }
    check_outputs(&w,&p,8,&expected);
}
#[test]
fn native_projected_three_column_equivalence() {
    assert!(cfg!(target_endian="little"));
    assert!(std::is_x86_feature_detected!("avx512f"));
    assert!(std::is_x86_feature_detected!("avx512bw"));
    assert!(std::is_x86_feature_detected!("avx512vl"));
    assert!(std::is_x86_feature_detected!("vpclmulqdq"));
    assert!(std::is_x86_feature_detected!("pclmulqdq"));
    for n in (0..=17).chain(63..=65) {for mode in 0..6 {for alias in [false,true] {check_private(n,mode,alias);}}}
    for log in [0usize,1,3,6] {for columns in 0..=8 {for workers in [1usize,3,8] {check_row(log,columns,workers);}}}
    for columns in [1023usize,1024,1025] {for workers in [1usize,3,8] {check_row(6,columns,workers);}}
    check_sparse_scored_row();
    par::pool().run(2,|i| if i == 0 {check_row(6,7,8)});
    let p = point(6,4);
    let w = words(63);
    assert!(std::panic::catch_unwind(|| row_evaluate_words(&w,&p,8)).is_err());
    assert!(std::panic::catch_unwind(|| labinius::eval::row_evaluate_words(&w,&p,8)).is_err());
    assert!(std::panic::catch_unwind(|| old_control::row_evaluate_words(&w,&p,8)).is_err());
}
}
