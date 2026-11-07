//! FIRST13 schedule selected by the positive discrete certificate in ntt_stat.
//! (reduce before L4, three L5 bits, nine L6 bits), one length-27 subtree.
pub const NODES: [(bool, u8, u16); 24] = [
    (true,5,56), (false,7,65), (true,5,56), (true,3,448),
    (true,7,0), (false,7,16), (true,0,511), (false,7,33),
    (true,5,56), (true,5,56), (true,0,511), (false,7,1),
    (true,0,511), (true,7,0), (true,3,448), (false,7,4),
    (true,3,448), (true,3,448), (true,3,448), (true,3,448),
    (true,5,56), (true,1,504), (true,4,63), (true,1,504),
];
pub const OUTPUT_BOUND: u64 = 32768;
pub const fn level6(n: usize) -> bool { NODES[n/9].2 >> (n%9) & 1 != 0 }
pub const fn pair(q: usize) -> u8 { level6(2*q) as u8 | ((level6(2*q+1) as u8)<<1) }
pub const REDUCTIONS: usize = {
    let mut count=0;let mut n=0;
    while n<24 {
        count+=9*NODES[n].0 as usize+3*NODES[n].1.count_ones() as usize+NODES[n].2.count_ones() as usize;
        n+=1;
    }count
};
const _: () = assert!(REDUCTIONS==394);
