//! Public-input certificate for the exact lazy binary NTT in the parent module.
//! This models all 648 independent input bits, not sampled witnesses. It reads
//! the live LUT and twiddle tables; the interval program follows the parent's
//! level-3 folding and physical level-4/5/6 register permutation.
use super::{tables, Tables};
use std::collections::HashMap;
use std::sync::OnceLock;

#[derive(Clone, Copy, Debug)]
struct Interval { lo: i32, hi: i32 }
impl Interval {
    fn new(lo: i32, hi: i32) -> Self {
        assert!(lo <= hi && lo >= -32768 && hi <= 32767,
            "binary NTT certificate: i16 interval {lo}..{hi}");
        Self { lo, hi }
    }
    fn empty() -> Self { Self { lo: i32::MAX, hi: i32::MIN } }
    fn see(&mut self, x: i32) { self.lo = self.lo.min(x); self.hi = self.hi.max(x); }
    fn checked(self) -> Self { Self::new(self.lo, self.hi) }
    fn add(self, b: Self) -> Self { Self::new(self.lo + b.lo, self.hi + b.hi) }
    fn sub(self, b: Self) -> Self { Self::new(self.lo - b.hi, self.hi - b.lo) }
    fn abs(self) -> i32 { (-self.lo).max(self.hi) }
}
fn set_bound(xs: &[i32]) -> Interval {
    Interval::new(*xs.iter().min().unwrap(), *xs.iter().max().unwrap())
}

// Each distinct signed-Montgomery constant is enumerated over the entire i16
// domain, including negative low-word multiplication and signed high halves.
// These maps are temporary proof data, released when the bounds are cached.
struct Maps { q: u16, maps: HashMap<(i16, i16), Vec<i16>> }
impl Maps {
    fn new(q: u16) -> Self { Self { q, maps: HashMap::new() } }
    fn get(&mut self, w: u32, wp: u32) -> &[i16] {
        assert_eq!(w as u16, (w >> 16) as u16);
        assert_eq!(wp as u16, (wp >> 16) as u16);
        let key = (w as i16, wp as i16);
        let q = self.q;
        self.maps.entry(key).or_insert_with(|| {
            (-32768..=32767).map(|a: i32| {
                let m = (a * key.1 as i32) as i16 as i64;
                let exact = ((a as i64 * key.0 as i64) >> 16)
                    - ((m * q as i64) >> 16);
                assert!((-32768..=32767).contains(&exact));
                assert_eq!(exact as i16,
                    crate::params::mont_mul_i16(a as i16, key.0, key.1, q));
                exact as i16
            }).collect()
        })
    }
    fn range(&mut self, a: Interval, w: u32, wp: u32) -> Interval {
        let m = self.get(w, wp);
        let mut r = Interval::empty();
        for x in a.lo..=a.hi { r.see(m[(x + 32768) as usize] as i32); }
        r.checked()
    }
}

// The only Barrett operations used by this binary kernel: level 4 uses the
// live 32-entry correction vector; level 6 uses the live vpmulhrsw constant.
fn reduce(a: Interval, t: &Tables, lookup: bool) -> Interval {
    let mut r = Interval::empty();
    for x in a.lo..=a.hi {
        let y = if lookup {
            (x as i16).wrapping_add(t.cv[8][((x >> 11) & 31) as usize])
        } else {
            let h = ((x * t.cv[3][0] as i32 * 2 + 32768) >> 16) as i16;
            (x as i16).wrapping_sub(h.wrapping_mul(t.cv[0][0]))
        };
        r.see(y as i32);
    }
    r.checked()
}

// a enters additively. Enumerate all finite (b,c) possibilities at level 3,
// preserving the dependence of omega*(b-c) on both inputs.
fn folded_sets(s: &[Vec<i32>; 3], maps: &mut Maps, om: [u32; 2]) -> [Interval; 3] {
    let [a, b, c] = [set_bound(&s[0]), set_bound(&s[1]), set_bound(&s[2])];
    let _ = [b.sub(c), b.add(c), a.sub(c), a.sub(b)];
    let mut p = Interval::empty();
    let mut n = Interval::empty();
    let m = maps.get(om[0], om[1]);
    for &x in &s[1] { for &y in &s[2] {
        let u = m[(x - y + 32768) as usize] as i32;
        p.see(-y + u); n.see(-x - u);
    }}
    [a.add(b.add(c)), a.add(p.checked()), a.add(n.checked())]
}

// For an interval of d=b-c, feasible c is exactly the interval below. Thus
// endpoints give exact marginal extrema over these conservative input boxes.
fn folded_intervals(a: Interval, b: Interval, c: Interval,
    maps: &mut Maps, om: [u32; 2]) -> [Interval; 3]
{
    let d = b.sub(c);
    let _ = [b.add(c), a.sub(c), a.sub(b)];
    let mut p = Interval::empty();
    let mut n = Interval::empty();
    let m = maps.get(om[0], om[1]);
    for delta in d.lo..=d.hi {
        let lo = c.lo.max(b.lo - delta);
        let hi = c.hi.min(b.hi - delta);
        assert!(lo <= hi);
        let u = m[(delta + 32768) as usize] as i32;
        p.see(-hi + u); p.see(-lo + u);
        n.see(-hi - delta - u); n.see(-lo - delta - u);
    }
    [a.add(b.add(c)), a.add(p.checked()), a.add(n.checked())]
}
fn butterfly(a: Interval, b: Interval, c: Interval, tw: [u32; 4],
    maps: &mut Maps, t: &Tables, reduction: u8) -> [Interval; 3]
{
    let b = maps.range(b, tw[0], tw[1]);
    let c = maps.range(c, tw[2], tw[3]);
    let a = if reduction == 0 { a } else { reduce(a, t, reduction == 1) };
    folded_intervals(a, b, c, maps, t.om)
}
fn put(p: &mut [Interval], idx: [usize; 3], y: [Interval; 3]) {
    // The assembler leaves Y0,Y2,Y1 in a0,a1,a2.
    p[idx[0]] = y[0]; p[idx[1]] = y[2]; p[idx[2]] = y[1];
}

fn prove<const Q: u16>() -> [i32; 648] {
    assert!(Q == 3889 || Q == 9721);
    let t = tables::<Q>();
    // The scalar maps use the same per-lane constants as the assembler.
    for lane in 0..32 {
        assert_eq!(t.cv[0][lane], Q as i16);
        assert_eq!(t.cv[1][lane], t.om[0] as i16);
        assert_eq!(t.cv[2][lane], t.om[1] as i16);
        assert_eq!(t.cv[3][lane], t.bv as i16);
    }
    let mut maps = Maps::new(Q);
    let mut out = [0; 648];
    for k in 0..4 {
        let mut plus: [Vec<i32>; 3] = std::array::from_fn(|_| Vec::new());
        let mut minus: [Vec<i32>; 3] = std::array::from_fn(|_| Vec::new());
        for r in 0..3 {
            let x = &t.lut[12 * k + 2 * r];
            let y = &t.lut[12 * k + 2 * r + 1];
            for nx in 0..16 { for ny in 0..16 {
                let a = i16::from_le_bytes([x[nx], x[16 + nx]]) as i32;
                let b = i16::from_le_bytes([y[ny], y[16 + ny]]) as i32;
                plus[r].push(a + b);
                minus[r].push(if r == 1 { b - a } else { a - b });
            }}
            plus[r].sort_unstable(); plus[r].dedup();
            minus[r].sort_unstable(); minus[r].dedup();
        }
        let u = folded_sets(&plus, &mut maps, t.om);
        let v = folded_sets(&minus, &mut maps, t.om);
        // Every one of the 27 coefficient indices has the same full binary
        // domain; sharing its level-3 bound loses no marginal precision.
        let scratch = [u[0], u[1], u[2], v[2], v[0], v[1]];
        for j in 0..6 {
            let kk = 6 * k + j;
            let mut p = [scratch[j]; 27];
            for i in 0..9 {
                let idx = [i, i + 9, i + 18];
                let y = butterfly(p[idx[0]], p[idx[1]], p[idx[2]], t.tw4[kk],
                    &mut maps, t, if Q == 9721 { 1 } else { 0 });
                put(&mut p, idx, y);
            }
            for (g, base) in [0, 18, 9].into_iter().enumerate() { for i in 0..3 {
                let idx = [base + i, base + i + 3, base + i + 6];
                let y = butterfly(p[idx[0]], p[idx[1]], p[idx[2]], t.tw5[3 * kk + g],
                    &mut maps, t, 0);
                put(&mut p, idx, y);
            }}
            for (g, base) in [0, 6, 3, 18, 24, 21, 9, 15, 12].into_iter().enumerate() {
                let y = butterfly(p[base], p[base + 1], p[base + 2], t.tw6[9 * kk + g],
                    &mut maps, t, if Q == 9721 { 2 } else { 0 });
                for (i, bound) in y.into_iter().enumerate() {
                    out[27 * kk + 3 * g + i] = bound.abs();
                }
            }
        }
    }
    out
}

/// Proven absolute bounds in the 648 canonical output rows. Initialization is
/// scalar and depends only on public transform tables, never on a witness.
pub fn proved_binary_row_bounds<const Q: u16>() -> &'static [i32; 648] {
    static P0: OnceLock<[i32; 648]> = OnceLock::new();
    static P1: OnceLock<[i32; 648]> = OnceLock::new();
    match Q {
        3889 => P0.get_or_init(prove::<3889>),
        9721 => P1.get_or_init(prove::<9721>),
        _ => panic!("binary NTT certificate: unsupported prime"),
    }
}
