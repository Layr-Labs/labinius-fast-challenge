// Appended only to a generated full control eval.rs view.
pub fn diagnostic_singles(a: &[[u64; 3]], words: &[Vec<u128>; 4]) -> [[u64; 3]; 4] {
    assert!(words.iter().all(|column| column.len() == a.len()));
    let a: Vec<F162> = a.iter().copied().map(F162).collect();
    let poly = Poly::from_scalars(&a);
    // F162Word is B128, whose bound source declares repr(transparent) over u128.
    core::array::from_fn(|j| unsafe {
        dot_words(&poly, words[j].as_ptr().cast::<F162Word>(), a.len()).0
    })
}
