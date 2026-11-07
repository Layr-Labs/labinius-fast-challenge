//! Discrete FIRST13 model. All convolutions use positive terms: no FFT cancellation,
//! normal approximation, independent-e approximation, or sampled tails.
//! Inputs of a butterfly use disjoint original coefficient positions. Siblings
//! are correlated, but never combined by a later butterfly, so marginals suffice.
use super::*;
use std::io::Write;
use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};

const P: i32 = 9721;
const LANES_PER_PROOF: f64 = (1u64 << 22) as f64;
const LIMIT: f64 = 1e-7 / (LANES_PER_PROOF * 1.001);
type Dist = Arc<Histogram>;
fn report_path(name:&str)->std::path::PathBuf {
    let dir=std::path::PathBuf::from(std::env::var("STAT_MODEL_DIR").unwrap_or_else(|_|"target/stat-model".into()));
    std::fs::create_dir_all(&dir).unwrap();dir.join(name)
}

#[derive(Debug)]
struct Histogram { lo: i32, p: Vec<f64> }
impl Histogram {
    fn new(lo: i32, p: Vec<f64>) -> Dist {
        let first = p.iter().position(|&p| p != 0.0).unwrap();
        let last = p.iter().rposition(|&p| p != 0.0).unwrap();
        Arc::new(Self { lo: lo + first as i32, p: p[first..=last].to_vec() })
    }
    fn hi(&self) -> i32 { self.lo + self.p.len() as i32 - 1 }
    fn map(&self, f: impl Fn(i32) -> i32) -> Dist {
        let vs: Vec<_> = (self.lo..=self.hi()).map(f).collect();
        let lo = *vs.iter().min().unwrap();
        let hi = *vs.iter().max().unwrap();
        let mut out = vec![0.0; (hi-lo+1) as usize];
        for (&v, &p) in vs.iter().zip(&self.p) { out[(v-lo) as usize] += p; }
        Self::new(lo, out)
    }
    fn conv(&self, b: &Histogram) -> Dist {
        let mut out = vec![0.0; self.p.len()+b.p.len()-1];
        // Contiguous positive accumulation; the inner loop is vectorizable.
        for (i, &a) in self.p.iter().enumerate() {
            if a == 0.0 { continue; }
            for (y, &b) in out[i..i+b.p.len()].iter_mut().zip(&b.p) { *y += a*b; }
        }
        Self::new(self.lo+b.lo, out)
    }
    fn neg(&self) -> Dist { Self::new(-self.hi(), self.p.iter().rev().copied().collect()) }
    fn tail(&self) -> f64 {
        self.p.iter().enumerate().filter(|(i,_)| !(-32768..=32767).contains(&(self.lo+*i as i32))).map(|(_,p)|p).sum()
    }
    fn valid(&self) -> Dist {
        let lo = self.lo.max(-32768); let hi = self.hi().min(32767);
        Self::new(lo, self.p[(lo-self.lo) as usize..=(hi-self.lo) as usize].to_vec())
    }
    fn moments(&self) -> (f64,f64,f64) {
        let mass: f64 = self.p.iter().sum();
        let mean = self.p.iter().enumerate().map(|(i,p)| (self.lo+i as i32) as f64*p).sum::<f64>()/mass;
        let central = |n| self.p.iter().enumerate().map(|(i,p)| ((self.lo+i as i32) as f64-mean).powi(n)*p).sum::<f64>()/mass;
        let var = central(2); (mean, var.sqrt(), central(4)/(var*var))
    }
}

fn mul(x: i32, t: &[i16]) -> i32 { x*t[0] as i32-((x*t[1] as i32+16384)>>15)*P }
fn red(x: i32) -> i32 { x-((x+4096)>>13)*P }

/// Probability that a+b overflows, without building its entire convolution.
fn sum_tail(a: &Histogram, b: &Histogram) -> f64 {
    if a.lo+b.lo >= -32768 && a.hi()+b.hi() <= 32767 { return 0.0; }
    // Separate prefix and suffix avoid subtraction of nearly equal probabilities.
    let mut pre = vec![0.0; b.p.len()+1]; let mut suf = pre.clone();
    for i in 0..b.p.len() { pre[i+1]=pre[i]+b.p[i]; }
    for i in (0..b.p.len()).rev() { suf[i]=suf[i+1]+b.p[i]; }
    let clamp = |v: i32| v.clamp(0,b.p.len() as i32) as usize;
    a.p.iter().enumerate().map(|(i,p)| {
        let x=a.lo+i as i32;
        p*(pre[clamp(-32768-x-b.lo)]+suf[clamp(32768-x-b.lo)])
    }).sum()
}

/// Enumerate (b,c) jointly, keeping eps(b-c)'s exact dependence. For the sixth
/// root front use eps(b+c), as in radix3_sixth. Charge all i16 intermediates.
fn butterfly(a: &Dist, b: &Dist, c: &Dist, eps: &[i16], sixth: bool) -> ([Dist;3], f64) {
    let mut terms: [Vec<f64>;3] = core::array::from_fn(|_| vec![0.0;131073]);
    let mut failure = 0.0;
    for (ib,&pb) in b.p.iter().enumerate() {
        if pb==0.0 { continue; }
        let bv=b.lo+ib as i32;
        for (ic,&pc) in c.p.iter().enumerate() {
            if pc==0.0 { continue; }
            let cv=c.lo+ic as i32; let weight=pb*pc;
            let d=if sixth {bv+cv} else {bv-cv};
            if !(-32768..=32767).contains(&d) || !(-32768..=32767).contains(&(bv+cv)) {
                failure+=weight; continue;
            }
            let e=mul(d,eps);
            let ts=if sixth { [bv+e,-bv+cv,-cv-e] } else { [bv+cv,-cv+e,-bv-e] };
            for i in 0..3 { terms[i][(ts[i]+65536) as usize]+=weight; }
        }
    }
    // Each emitted arithmetic instruction is checked, even if its final output fits.
    failure+=sum_tail(a,&b.neg())+sum_tail(a,&c.neg());
    if sixth { failure+=sum_tail(a,b); }
    let out = terms.map(|t| {
        let d=a.conv(&Histogram::new(-65536,t));
        failure+=d.tail();
        d.valid()
    });
    (out,failure)
}

fn r3(v: [&Dist;3], t: &[i16], k: &Kernel, reduce: bool) -> ([Dist;3],f64) {
    let a=if reduce {v[0].map(red)} else {Arc::clone(v[0])};
    let b=v[1].map(|x|mul(x,t)); let c=v[2].map(|x|mul(x,&t[2..]));
    butterfly(&a,&b,&c,&k.k[2..],false)
}

fn lut(k: &Kernel, n: usize, t: usize, pos: usize) -> Dist {
    let patterns: Vec<_>=(0..16).filter(|&v| (0..4).all(|g| v>>g&1==0 || pos+162*g<BITS)).collect();
    let mut p=vec![0.0;9721];
    for &v in &patterns {p[(k.lut[n][t][v] as i32+4860) as usize]+=1.0/patterns.len() as f64;}
    Histogram::new(-4860,p)
}

/// Two position classes at each level: all positions except the last, and the
/// last. Only coefficient 512's boundary distinguishes them. Multiplicities
/// are 26/1 at L3, 8/1 at L4, 2/1 at L5, and 1 at L6.
fn level3(k:&Kernel) -> Vec<[Dist;2]> {
    let zero=Histogram::new(0,vec![1.0]);
    let mut out=vec![[zero.clone(),zero];24];
    for n in 0..4 {for h in 0..2 {for class in 0..2 {
        let j=if class==0 {0} else {26};
        let v: [Dist;3]=core::array::from_fn(|r| {
            let a=lut(k,n,2*r,j+27*r); let b=lut(k,n,2*r+1,j+27*r+81);
            let b=if h==0 {b} else {b.neg()};
            a.conv(&b)
        });
        let (ys,fail)=butterfly(&v[0],&v[1],&v[2],&k.k[2..],h==1);
        assert_eq!(fail,0.0,"L3 overflow");
        for (g,y) in ys.into_iter().enumerate() {out[6*n+3*h+g][class]=y;}
    }}}
    out
}

fn node(k:&Kernel,n:usize,input:&[Dist;2],mode:&str) -> f64 {
    let mut log=Vec::new();
    let selected=mode=="selected";
    let reduce4=if selected {stat_schedule::NODES[n].0} else {mode=="record" && n!=1 && n!=15};
    let mut total=0.0;
    let l4: [[Dist;3];2]=core::array::from_fn(|cl| {
        let (ys,f)=r3([&input[0],&input[0],&input[cl]],&k.c[n],k,reduce4);
        total+=f*if cl==0 {8.0} else {1.0}; ys
    });
    for j in 0..9 {for g in 0..3 {log_moment(&mut log,4,27*n+9*g+j,&l4[usize::from(j==8)][g]);}}
    for g in 0..3 {
        let reduce5=if selected {stat_schedule::NODES[n].1>>g&1!=0} else if mode=="l5all" {true} else {!skip_lut5(3*n+g)};
        let t=&k.c[n][4+4*g..];
        let l5: [[Dist;3];2]=core::array::from_fn(|cl| {
            let (ys,f)=r3([&l4[0][g],&l4[0][g],&l4[cl][g]],t,k,reduce5);
            total+=f*if cl==0 {2.0} else {1.0}; ys
        });
        for j in 0..3 {for h in 0..3 {log_moment(&mut log,5,27*n+9*g+3*h+j,&l5[usize::from(j==2)][h]);}}
        for h in 0..3 {
            let nn=9*n+3*g+h; let t=&k.d[nn/2][4*(nn%2)..];
            let reduce6=if selected {stat_schedule::level6(nn)} else {mode=="record"};
            let (ys,f)=r3([&l5[0][h],&l5[0][h],&l5[1][h]],t,k,reduce6);
            total+=f;
            for (slot,y) in ys.iter().enumerate() {
                log_moment(&mut log,6,3*nn+slot,y);
                let (mean,sd,kurt)=y.moments();
                writeln!(log,"STAT_MODEL mode={mode} n={n} g={g} h={h} slot={slot} mean={mean:.6} sd={sd:.6} kurt={kurt:.6} support={}..{}",y.lo,y.hi()).unwrap();
            }
        }
    }
    std::fs::write(report_path(&format!("stat-model-{mode}-{n}.txt")),log).unwrap();
    println!("STAT_NODE mode={mode} n={n} r4={reduce4} proof_bound={:.12e}",total*LANES_PER_PROOF);
    total
}

fn log_moment(log:&mut Vec<u8>,stage:usize,slot:usize,y:&Histogram) {
    let (mean,sd,kurt)=y.moments();
    writeln!(log,"MOMENT stage={stage} slot={slot} mean={mean:.12e} sd={sd:.12e} kurt={kurt:.12e} lo={} hi={}",y.lo,y.hi()).unwrap();
}

fn selected_certificate() {
    assert_eq!(stat_schedule::REDUCTIONS,394);
    let k=&kernels()[1];let front=level3(k);
    let mut log=Vec::new();
    for (n,ds) in front.iter().enumerate() {for j in 0..27 {log_moment(&mut log,3,27*n+j,&ds[usize::from(j==26)]);}}
    std::fs::write(report_path("stat-model-selected-front.txt"),log).unwrap();
    let next=AtomicUsize::new(0);
    let sum=std::thread::scope(|scope| {
        let handles:Vec<_>=(0..8).map(|_|scope.spawn(|| {
            let mut sum=0.0;
            loop {let n=next.fetch_add(1,Ordering::Relaxed);if n>=24 {break;}sum+=node(k,n,&front[n],"selected");}sum
        })).collect();handles.into_iter().map(|h|h.join().unwrap()).sum::<f64>()
    });
    let bound=sum*LANES_PER_PROOF*1.001;
    println!("STAT_SELECTED reductions={} proof_upper={bound:.12e}",stat_schedule::REDUCTIONS);
    assert!(bound<=1e-7,"selected FIRST13 schedule fails: {bound:e}");
}

#[test]
fn stat_selected_certificate() {selected_certificate();}

#[derive(Clone, Copy, Debug)]
struct Choice { failure: f64, r4: bool, r5: u8, r6: u16 }
type Frontier = Vec<Option<Choice>>;
fn singleton(failure:f64) -> Frontier {
    vec![Some(Choice {failure,r4:false,r5:0,r6:0})]
}
fn insert(front:&mut Frontier,cost:usize,c:Choice) {
    if c.failure>LIMIT {return;}
    if cost>=front.len() {front.resize(cost+1,None);}
    if front[cost].map_or(true,|old|c.failure<old.failure) {front[cost]=Some(c);}
}
fn combine(a:&Frontier,b:&Frontier) -> Frontier {
    let mut out=Vec::new();
    for (i,x) in a.iter().enumerate() {if let Some(x)=x {
        for (j,y) in b.iter().enumerate() {if let Some(y)=y {
            insert(&mut out,i+j,Choice {failure:x.failure+y.failure,
                r4:x.r4||y.r4,r5:x.r5|y.r5,r6:x.r6|y.r6});
        }}
    }} out
}

/// Exhaust all 2^(1+3+9) reduction schedules in each length-27 subtree.
/// Reuse distributions: L6 choices do not affect siblings or earlier levels.
fn search_node(k:&Kernel,n:usize,input:&[Dist;2]) -> (Frontier,usize) {
    let mut best=Vec::new(); let mut mandatory_l4=0;
    let mut log=Vec::new();
    for r4 in [false,true] {
        let mut failure4=0.0;
        let l4:[[Dist;3];2]=core::array::from_fn(|cl| {
            let (ys,f)=r3([&input[0],&input[0],&input[cl]],&k.c[n],k,r4);
            let multiplicity=if cl==0 {8} else {1};
            failure4+=f*multiplicity as f64;
            // At most six events are charged by a radix3 call. Hence f/6 is
            // a LOWER bound on the union, without assuming disjoint tails.
            // L3 has no lost mass. Distinct ring-element lanes use disjoint
            // witness bits, so lift the guarded per-lane bound with 1-(1-p)^L.
            let lower=-((-(f/6.0/1.001)).ln_1p()*LANES_PER_PROOF).exp_m1();
            if !r4 && lower>1e-7 {mandatory_l4+=multiplicity;}
            writeln!(log,"L4 n={n} class={cl} red={r4} union_upper={:.12e} union_lower={lower:.12e}",f*LANES_PER_PROOF).unwrap();
            ys
        });
        if failure4>LIMIT {continue;}
        let mut root=singleton(failure4);
        if r4 {root=vec![None;10];root[9]=Some(Choice{failure:failure4,r4:true,r5:0,r6:0});}
        for g in 0..3 {
            let mut child_best=Vec::new();
            for r5 in [false,true] {
                let mut failure5=0.0;
                let l5:[[Dist;3];2]=core::array::from_fn(|cl| {
                    let (ys,f)=r3([&l4[0][g],&l4[0][g],&l4[cl][g]],&k.c[n][4+4*g..],k,r5);
                    failure5+=f*if cl==0 {2.0} else {1.0};ys
                });
                writeln!(log,"L5 n={n} g={g} r4={r4} red={r5} proof_upper={:.12e}",failure5*LANES_PER_PROOF).unwrap();
                if failure4+failure5>LIMIT {continue;}
                let mut child=singleton(failure5);
                if r5 {child=vec![None;4];child[3]=Some(Choice{failure:failure5,r4:false,r5:1<<g,r6:0});}
                for h in 0..3 {
                    let nn=9*n+3*g+h;let t=&k.d[nn/2][4*(nn%2)..];
                    let mut leaf=Vec::new();
                    for r6 in [false,true] {
                        let (_,f)=r3([&l5[0][h],&l5[0][h],&l5[1][h]],t,k,r6);
                        writeln!(log,"L6 n={n} g={g} h={h} r4={r4} r5={r5} red={r6} proof_upper={:.12e}",f*LANES_PER_PROOF).unwrap();
                        insert(&mut leaf,usize::from(r6),Choice{failure:f,r4:false,r5:0,r6:if r6 {1<<(3*g+h)} else {0}});
                    }
                    child=combine(&child,&leaf);
                }
                for (cost,c) in child.into_iter().enumerate() {if let Some(c)=c {insert(&mut child_best,cost,c);}}
            }
            root=combine(&root,&child_best);
        }
        for (cost,c) in root.into_iter().enumerate() {if let Some(c)=c {insert(&mut best,cost,c);}}
    }
    std::fs::write(report_path(&format!("stat-search-{n}.txt")),log).unwrap();
    println!("STAT_SEARCH n={n} mandatory_l4={mandatory_l4} best={:?}",best.iter().position(Option::is_some));
    (best,mandatory_l4)
}

fn search(k:&Kernel,front:&[[Dist;2]]) {
    let next=AtomicUsize::new(0);
    let mut results=std::thread::scope(|scope| {
        let handles:Vec<_>=(0..8).map(|_|scope.spawn(|| {
            let mut rows=Vec::new();
            loop {let n=next.fetch_add(1,Ordering::Relaxed);if n>=24 {break;}
                rows.push((n,search_node(k,n,&front[n])));
            }rows
        })).collect();
        handles.into_iter().flat_map(|h|h.join().unwrap()).collect::<Vec<_>>()
    });
    results.sort_by_key(|r|r.0);
    let mut states:Vec<Option<(f64,Vec<Choice>)>>=vec![Some((0.0,Vec::new()))];
    let mut mandatory=0;
    for (_,(choices,required)) in results {
        mandatory+=required;
        let mut out=vec![None;states.len()+choices.len()];
        for (i,x) in states.iter().enumerate() {if let Some((p,path))=x {
            for (j,c) in choices.iter().enumerate() {if let Some(c)=c {
                let sum=p+c.failure;
                if sum<=LIMIT && out[i+j].as_ref().map_or(true,|old:&(f64,Vec<Choice>)|sum<old.0) {
                    let mut path=path.clone();path.push(*c);out[i+j]=Some((sum,path));
                }
            }}
        }}states=out;
    }
    println!("STAT_LOWER_BOUND mandatory_l4={mandatory} target=135");
    let mut log=Vec::new();
    for (cost,state) in states.iter().enumerate() {if let Some((p,path))=state {
        writeln!(log,"STAT_FRONTIER cost={cost} proof_upper={:.12e} path={path:?}",p*LANES_PER_PROOF*1.001).unwrap();
    }}
    std::fs::write(report_path("stat-search-frontier.txt"),&log).unwrap();
    let (cost,(p,path))=states.into_iter().enumerate().find_map(|(c,s)|s.map(|s|(c,s))).expect("no certified schedule");
    println!("STAT_OPTIMUM reductions={cost} proof_upper={:.12e} path={path:?}",p*LANES_PER_PROOF*1.001);
}

#[test]
fn stat_histogram_checks() {
    let a=Histogram::new(-1,vec![0.25,0.5,0.25]);
    let c=a.conv(&a);assert_eq!(c.lo,-2);assert_eq!(c.p,vec![0.0625,0.25,0.375,0.25,0.0625]);
    let edge=Histogram::new(32767,vec![1.0]);
    assert_eq!(sum_tail(&edge,&a),0.25);
    let edge=Histogram::new(-32768,vec![1.0]);
    assert_eq!(sum_tail(&edge,&a),0.25);
    let k=&kernels()[1];let (out,f)=butterfly(&a,&a,&a,&k.k[2..],false);
    assert_eq!(f,0.0);
    let mut want:[std::collections::BTreeMap<i32,f64>;3]=core::array::from_fn(|_|Default::default());
    for x in -1..=1 {for b in -1..=1 {for c in -1..=1 {
        let prob=|v|if v==0 {0.5} else {0.25};
        let weight=prob(x)*prob(b)*prob(c);
        let e=mul(b-c,&k.k[2..]);
        for (i,v) in [x+b+c,x-c+e,x-b-e].into_iter().enumerate() {*want[i].entry(v).or_default()+=weight;}
    }}}
    for (i,o) in out.into_iter().enumerate() {
        assert!((o.p.iter().sum::<f64>()-1.0).abs()<1e-15);
        for (v,p) in &want[i] {assert_eq!(o.p[(*v-o.lo) as usize],*p);}
    }
}

#[test]
#[ignore = "full positive discrete convolution, queue only"]
fn stat_discrete_model() {
    let mode=std::env::var("STAT_MODE").unwrap_or_else(|_|"search".into());
    let k=&kernels()[1]; let front=level3(k);
    if mode=="search" {search(k,&front);return;}
    let next=AtomicUsize::new(0);
    let sums=std::thread::scope(|scope| {
        let handles: Vec<_>=(0..8).map(|_|scope.spawn(|| {
            let mut sum=0.0;
            loop {let n=next.fetch_add(1,Ordering::Relaxed);if n>=24 {break;}
                sum+=node(k,n,&front[n],&mode);
            }sum
        })).collect();
        handles.into_iter().map(|h|h.join().unwrap()).sum::<f64>()
    });
    // Binary64 only adds nonnegative terms. No branch has >2^32 rounding steps
    // per coefficient per level. 1.001 is larger than the cumulative gamma_n
    // envelope for all four levels, including pair histograms and tail sums.
    let bound=sums*LANES_PER_PROOF*1.001;
    println!("STAT_CERT mode={mode} proof_overflow_bound={bound:.12e} limit=1e-7");
    assert!(bound<=1e-7,"candidate fails the discrete overflow certificate: {bound:e}");
}
