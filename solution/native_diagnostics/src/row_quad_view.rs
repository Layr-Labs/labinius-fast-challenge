// Appended only to a generated full candidate eval.rs view.
pub fn diagnostic_quad(a: &[[u64; 3]], words: &[Vec<u128>; 4]) -> [[u64; 3]; 4] {
    assert!(words.iter().all(|column| column.len() == a.len()));
    let a: Vec<F162> = a.iter().copied().map(F162).collect();
    let poly = Poly::from_scalars(&a);
    // F162Word is B128, whose bound source declares repr(transparent) over u128.
    let ptrs = words
        .each_ref()
        .map(|column| column.as_ptr().cast::<F162Word>());
    unsafe { dot_words4(&poly, ptrs[0], ptrs[1], ptrs[2], ptrs[3], a.len()) }.map(|value| value.0)
}
