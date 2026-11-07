//! Full-stream moments for development proofs; no sampling. Each worker owns
//! its counters and writes them when it exits, before the proof is returned.
use super::*;
use std::{cell::RefCell,io::Write,sync::atomic::{AtomicUsize,Ordering}};
#[derive(Clone,Copy)]
struct Moment {count:u64,min:i32,max:i32,sum:[f64;4]}
impl Default for Moment {
    fn default()->Self {Self{count:0,min:i32::MAX,max:i32::MIN,sum:[0.0;4]}}
}
struct Stats(Vec<Moment>);
thread_local! {static STATS:RefCell<Stats>=RefCell::new(Stats(vec![Moment::default();4*N]));}
static NEXT:AtomicUsize=AtomicUsize::new(0);
impl Drop for Stats {
    fn drop(&mut self) {
        let Ok(dir)=std::env::var("LABINIUS_STAT_DIR") else {return};
        let id=NEXT.fetch_add(1,Ordering::Relaxed);
        let mut out=std::fs::File::create(format!("{dir}/worker-{id}.csv")).unwrap();
        for (i,m) in self.0.iter().enumerate() {if m.count!=0 {
            writeln!(out,"{},{},{},{},{},{:.17e},{:.17e},{:.17e},{:.17e}",3+i/N,i%N,m.count,m.min,m.max,m.sum[0],m.sum[1],m.sum[2],m.sum[3]).unwrap();
        }}
    }
}
pub unsafe fn record<const L:usize>(stage:usize,at:usize,stride:usize,v:&[int16x8_t;L]) {
    STATS.with(|s| {
        let mut s=s.borrow_mut();
        for (j,&v) in v.iter().enumerate() {
            let values:[i16;8]=core::mem::transmute(v);
            let m=&mut s.0[(stage-3)*N+at+stride*j];
            for x in values {
                let x=x as i32;m.count+=1;m.min=m.min.min(x);m.max=m.max.max(x);
                let x=x as f64;let x2=x*x;
                m.sum[0]+=x;m.sum[1]+=x2;m.sum[2]+=x2*x;m.sum[3]+=x2*x2;
            }
        }
    });
}
