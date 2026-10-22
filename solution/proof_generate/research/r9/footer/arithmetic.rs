// Exact current red16 and CRT helpers; imports supplied by parent module.
const fn bias(q: u16) -> i64 {
    ((1i64 << 31) / q as i64) * q as i64
}
/// `q^-1 mod 2^16` (Newton).
const fn qinv16(q: u16) -> u32 {
    let q = q as u32;
    let mut x = q;
    let mut i = 0;
    while i < 4 {
        x = x.wrapping_mul(2u32.wrapping_sub(q.wrapping_mul(x))) & 0xffff;
        i += 1;
    }
    x
}
/// [`red16`]'s word constants: `(W, W q^-1 mod 2^16)` per i32 lane, `W = 2^16 mod q` in the low
/// word and `2^32 mod q` in the high word.
const fn red16_w(q: u16) -> (i32, i32) {
    let wl = (1u32 << 16) % q as u32;
    let wh = ((1u64 << 32) % q as u64) as u32;
    let qi = qinv16(q);
    (
        (wl | (wh << 16)) as i32,
        (((wl * qi) & 0xffff) | (((wh * qi) & 0xffff) << 16)) as i32,
    )
}
/// A representative of `x mod q` with `|r| < 2q`, lane-wise, for `|x| <= bias(q)`: `y = x + bias`
/// is `2^16 yh + yl` with both words unsigned, and an unsigned Montgomery product per word against
/// `2^16` (low) and `2^32` (high) mod q leaves each word in `(-q, q)` congruent to `yl` and
/// `2^16 yh`; `vpmaddwd` adds them. Six uops, none of them a shuffle.
#[inline(always)]
unsafe fn red16<const Q: u16>(x: __m512i) -> __m512i {
    let (w, wp) = red16_w(Q);
    let y = _mm512_add_epi32(x, _mm512_set1_epi32(bias(Q) as i32));
    let m = _mm512_mullo_epi16(y, _mm512_set1_epi32(wp));
    let r = _mm512_sub_epi16(
        _mm512_mulhi_epu16(y, _mm512_set1_epi32(w)),
        _mm512_mulhi_epu16(m, _mm512_set1_epi16(Q as i16)),
    );
    _mm512_madd_epi16(r, _mm512_set1_epi16(1))
}

#[inline(always)]
unsafe fn canonical(x: __m512i, q: i16) -> __m512i {
    _mm512_mask_add_epi16(
        x,
        _mm512_cmp_epi16_mask(x, _mm512_setzero_si512(), _MM_CMPINT_LT),
        x,
        _mm512_set1_epi16(q),
    )
}
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn crt_accumulate(a: &Batch32, b: &Batch32, tail: &mut Batch32) {
    const INV: u16 = inv_mod(P0 as u64, P1 as u64) as u16;
    const W: i16 = Params::<P1>::to_mont(INV);
    const WP: i16 = Params::<P1>::mont_pre(W);
    let qv = _mm512_set1_epi16(P1 as i16);
    for j in 0..648 {
        let av = canonical(
            _mm512_load_si512(a.v[j].as_ptr() as *const __m512i),
            P0 as i16,
        );
        let bv = canonical(
            _mm512_load_si512(b.v[j].as_ptr() as *const __m512i),
            P1 as i16,
        );
        let d = _mm512_sub_epi16(bv, av);
        let m = _mm512_mullo_epi16(d, _mm512_set1_epi16(WP));
        let h = _mm512_mulhi_epi16(d, _mm512_set1_epi16(W));
        let u = canonical(_mm512_sub_epi16(h, _mm512_mulhi_epi16(m, qv)), P1 as i16);
        let gt = _mm512_cmp_epi16_mask(u, _mm512_set1_epi16((P1 / 2) as i16), _MM_CMPINT_NLE);
        let eq = _mm512_cmp_epi16_mask(u, _mm512_set1_epi16((P1 / 2) as i16), _MM_CMPINT_EQ);
        let agt = _mm512_cmp_epi16_mask(av, _mm512_set1_epi16((P0 / 2) as i16), _MM_CMPINT_NLE);
        let mut low = _mm512_add_epi16(av, _mm512_mullo_epi16(u, _mm512_set1_epi16(P0 as i16)));
        low = _mm512_mask_sub_epi16(
            low,
            gt | (eq & agt),
            low,
            _mm512_set1_epi16((PRODUCT & 65535) as u16 as i16),
        );
        let old = _mm512_load_si512(tail.v[j].as_ptr() as *const __m512i);
        _mm512_store_si512(
            tail.v[j].as_mut_ptr() as *mut __m512i,
            _mm512_add_epi16(old, low),
        );
    }
}
