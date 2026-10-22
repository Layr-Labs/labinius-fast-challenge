//! Fixed complete-phase comparison. Counted copies run only before timing.
use crate::inverse_support::compare::{prepare,run_baseline,run_candidate,Output,Prepared};
use crate::inverse_support::fields::{baseline_counted as counted_base,candidate_counted as counted_candidate};
use labinius::fields::crossfield as baseline;
use std::time::Instant;

fn count_gate(p:&Prepared<32>,reference:&Output){
 macro_rules! counted_run {
  ($module:ident)=>{{
   let (sets,eq)=$module::eq_expand_and_partials_factored_blocks::<32>(&p.hi,&p.words,8);
   $module::reset_inverse_count();
   let partials=$module::partials_from_sets(&sets,&p.hi);
   let mut sw=$module::LeanSwitch::with_sets_factored(&p.words,eq,&p.hi,&p.batch,8,
       $module::LeanSwitch::planes(p.words.len()),sets.into()).unwrap();
   let mut transcript=p.transcript.clone();
   let mut messages=Vec::with_capacity(24);let mut challenges=Vec::with_capacity(24);
   for _ in 0..24{
    let msg=sw.msg();transcript.absorb_f162(&msg);
    let r=transcript.sample_f162(b"switch-round",1)[0];
    messages.push(msg);challenges.push(r);sw.fold(r);
   }
   let opened=sw.final_eval();transcript.absorb_f162(&[opened]);
   let following_challenge=transcript.sample_f162(b"following-round",1)[0];
   let output=Output{messages,challenges,opened,following_challenge};
   assert_eq!(&output,reference,"counted complete-output gate");
   (partials,$module::inverse_count())
  }};
 }
 let (partials_a,count_a)=counted_run!(counted_base);
 let (partials_b,count_b)=counted_run!(counted_candidate);
 assert_eq!(partials_a,partials_b);
 assert_eq!(partials_a,baseline::partials_from_sets(&p.sets,&p.hi));
 assert_eq!(count_a,39);assert_eq!(count_b,6);
 println!("switch-inverse count_gate=pass baseline_inverse_calls={count_a} candidate_inverse_calls={count_b}");
}

/// One fixed input, full equality/count gates, two warmups and eight paired samples.
/// Construction, cache allocation, all rounds, transcript work, and drop are timed.
pub fn run(){
 const SEED:u128=0x123456789abcdef013572468abcdef01;
 let p=prepare::<32>(24,8,SEED);
 super::populate_words(&p.words,&[],8);
 assert_eq!(p.words.len(),1<<24);assert_eq!(p.hi.len(),24);assert_eq!(p.sets.len(),32);
 let reference=run_baseline(&p,8);
 assert_eq!(run_candidate(&p,8),reference,"untimed complete-output gate");
 count_gate(&p,&reference);
 println!("switch-inverse shape_words={} prefix_sets=32 workers=8 seed={SEED:032x} output_gate=pass",p.words.len());
 for round in 0..10{
  let mut pair=[0u128;2];
  for slot in 0..2{
   let candidate=(round+slot)%2!=0;
   let start=Instant::now();
   let output=if candidate{run_candidate(&p,8)}else{run_baseline(&p,8)};
   let ns=start.elapsed().as_nanos();
   assert_eq!(output,reference,"round {round} candidate {candidate}");
   pair[candidate as usize]=ns;
  }
  let phase=if round<2{"warmup"}else{"sample"};
  let number=if round<2{round}else{round-2};
  let first=if round%2==0{"baseline"}else{"batch-inverse"};
  println!("switch-inverse phase={phase} pair={number} first={first} baseline_ns={} candidate_ns={} delta_ns={}",pair[0],pair[1],pair[1]as i128-pair[0]as i128);
 }
}
