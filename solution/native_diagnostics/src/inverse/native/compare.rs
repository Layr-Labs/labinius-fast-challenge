use labinius::{B128,F162,Transcript};
use labinius::fields::crossfield as baseline;
use crate::fields::candidate;

#[derive(Clone,Debug,PartialEq,Eq)]
pub struct Output {
 pub messages:Vec<[F162;2]>,
 pub challenges:Vec<F162>,
 pub opened:F162,
 pub following_challenge:F162,
}

pub struct Prepared<const G:usize> {
 pub words:Vec<B128>,pub hi:Vec<B128>,pub batch:Vec<F162>,pub transcript:Transcript,
 pub sets:[Vec<B128>;G],pub baseline_eq:baseline::EqFactors,pub candidate_eq:candidate::EqFactors,
}

// Input generation and partial-set preparation are outside the comparison timer.
pub fn prepare<const G:usize>(log_words:usize,workers:usize,seed:u128)->Prepared<G>{
 let mut state=seed;
 let mut next=||{state^=state<<13;state^=state>>7;state^=state<<17;state};
 let words:Vec<_>=(0..1usize<<log_words).map(|_|B128(next())).collect();
 let mut transcript=Transcript::new(b"switch-inverse-comparison");
 transcript.absorb_bytes(&seed.to_le_bytes());
 let hi=transcript.sample_b128(b"hi",log_words);
 assert!(hi.iter().all(|&r|r!=B128::ONE));
 let batch=baseline::eq_expand_f162(&transcript.sample_f162(b"switch-batch",7));
 let (sets,baseline_eq)=baseline::eq_expand_and_partials_factored_blocks::<G>(&hi,&words,workers);
 let (check,candidate_eq)=candidate::eq_expand_and_partials_factored_blocks::<G>(&hi,&words,workers);
 assert_eq!(sets,check);
 assert_eq!(baseline_eq.to_vec(),candidate_eq.to_vec());
 Prepared{words,hi,batch,transcript,sets,baseline_eq,candidate_eq}
}

macro_rules! full_run {
 ($name:ident,$module:ident,$eq:ident)=>{
  #[inline(never)]
  pub fn $name<const G:usize>(p:&Prepared<G>,workers:usize)->Output{
   let mut sw=$module::LeanSwitch::with_sets_factored(&p.words,p.$eq.clone(),&p.hi,&p.batch,
       workers,$module::LeanSwitch::planes(p.words.len()),p.sets.clone().into()).unwrap();
   let mut transcript=p.transcript.clone();
   let mut messages=Vec::with_capacity(p.hi.len());let mut challenges=Vec::with_capacity(p.hi.len());
   for _ in 0..p.hi.len(){
    let msg=sw.msg();transcript.absorb_f162(&msg);
    let r=transcript.sample_f162(b"switch-round",1)[0];
    messages.push(msg);challenges.push(r);sw.fold(r);
   }
   let opened=sw.final_eval();transcript.absorb_f162(&[opened]);
   let following_challenge=transcript.sample_f162(b"following-round",1)[0];
   Output{messages,challenges,opened,following_challenge}
  }
 };
}
full_run!(run_baseline,baseline,baseline_eq);
full_run!(run_candidate,candidate,candidate_eq);

// Non-generic exports force complete scored-shape paths through release code generation.
#[no_mangle]
pub fn full_baseline_32(p:&Prepared<32>,workers:usize)->Output{run_baseline(p,workers)}
#[no_mangle]
pub fn full_candidate_32(p:&Prepared<32>,workers:usize)->Output{run_candidate(p,workers)}
