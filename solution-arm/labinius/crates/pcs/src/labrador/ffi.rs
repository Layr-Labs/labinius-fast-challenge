//! Raw declarations for `liblabrador.a` (built at `LOGQ = 48`, symbol prefix `labrador48_`)
//! and for the `csrc/bn_labrador.c` shim.
//!
//! Only the symbols LaBRADOR marks `visibility("default")` carry the `labrador48_` prefix;
//! everything else (`init_sparsecnst_half`, `polxvec_*`, `sparsecnst_eval`, ...) keeps its
//! plain name and is reached through the shim rather than from here.

use std::ffi::c_void;
use std::os::raw::c_int;

#[cfg(target_arch = "x86_64")]
extern "C" {
    pub fn labrador48_init_comkey(n: usize);
    pub fn labrador48_free_comkey();
    pub fn labrador48_init_witness_raw(wt: *mut c_void, r: usize, n: *const usize);
    pub fn labrador48_free_witness(wt: *mut c_void);
    pub fn labrador48_init_smplstmnt_raw(
        st: *mut c_void,
        r: usize,
        n: *const usize,
        betasq: *const u64,
        k: usize,
    ) -> c_int;
    pub fn labrador48_free_smplstmnt(st: *mut c_void);
    pub fn labrador48_free_commitment(com: *mut c_void);
    pub fn labrador48_free_composite(p: *mut c_void);
    pub fn labrador48_simple_verify(st: *const c_void, wt: *const c_void) -> c_int;
    pub fn labrador48_composite_prove_simple(
        p: *mut c_void,
        com: *mut c_void,
        st: *const c_void,
        wt: *const c_void,
    ) -> c_int;
    pub fn labrador48_composite_verify_simple(
        p: *const c_void,
        com: *const c_void,
        st: *const c_void,
    ) -> c_int;

    pub fn bn_sizeof_polx() -> usize;
    pub fn bn_polx_align() -> usize;
    pub fn bn_ring_degree() -> usize;
    pub fn bn_logq() -> usize;
    pub fn bn_compiled_q() -> u64;
    pub fn bn_extlen(len: usize, deg: usize) -> usize;
    pub fn bn_comkey_len() -> usize;

    pub fn bn_polx_alloc(len: usize) -> *mut c_void;
    pub fn bn_polx_free(p: *mut c_void);
    pub fn bn_polx_setzero(r: *mut c_void, len: usize);
    pub fn bn_polx_copy(r: *mut c_void, a: *const c_void, len: usize);
    pub fn bn_polx_eq(a: *const c_void, b: *const c_void, len: usize) -> c_int;
    pub fn bn_polx_from_int64(r: *mut c_void, len: usize, a: *const i64);
    pub fn bn_polx_from_int16(r: *mut c_void, len: usize, a: *const i16);
    pub fn bn_polx_expand(r: *mut c_void, len: usize, seed: *const u8, nonce: u64);

    pub fn bn_sx_new(r: usize, n: *const usize, s: *const *const i16) -> *mut c_void;
    pub fn bn_sx_free(sx: *mut c_void);
    pub fn bn_sx_ptr(sx: *const c_void, i: usize, off: usize) -> *const c_void;

    pub fn bn_commit_polx(
        out: *mut c_void,
        key: *const c_void,
        s: *const c_void,
        len: usize,
        deg: usize,
    );
    pub fn bn_commit_i16(
        out: *mut c_void,
        key: *const c_void,
        s: *const i16,
        len: usize,
        deg: usize,
    );
    pub fn bn_commit_blocks(
        out: *mut c_void,
        key: *const c_void,
        deg: usize,
        nb: usize,
        len: *const usize,
        s: *const *const i16,
    );
    pub fn bn_polx_table_sum(
        out: *mut c_void,
        len: usize,
        table: *const c_void,
        terms: usize,
        idx: *const u16,
        sign: *const i8,
    );
    pub fn bn_sis_secure(rank: usize, norm: f64) -> c_int;

    pub fn bn_eval_blocks(
        out: *mut c_void,
        deg: usize,
        nz: usize,
        idx: *const usize,
        off: *const usize,
        len: *const usize,
        phi: *const *const c_void,
        sphi: *const *const i16,
        soff: *const usize,
        swid: *const usize,
        sx: *const c_void,
    );

    pub fn bn_alloc_smplstmnt() -> *mut c_void;
    pub fn bn_alloc_witness() -> *mut c_void;
    pub fn bn_alloc_composite() -> *mut c_void;
    pub fn bn_alloc_commitment() -> *mut c_void;
    pub fn bn_free(p: *mut c_void);

    pub fn bn_smplstmnt_set_digest(st: *mut c_void, digest: *const u8);
    pub fn bn_smplstmnt_set_constraint(
        st: *mut c_void,
        ci: usize,
        deg: usize,
        nz: usize,
        idx: *const usize,
        off: *const usize,
        len: *const usize,
        phi: *const *const c_void,
        sphi: *const *const i16,
        soff: *const usize,
        swid: *const usize,
        b: *const c_void,
    ) -> c_int;

    pub fn bn_set_witness_i16(wt: *mut c_void, i: usize, n: usize, s: *const i16) -> c_int;
    pub fn bn_witness_normsq(wt: *const c_void, i: usize) -> u64;
    pub fn bn_composite_size(p: *const c_void) -> f64;

    pub fn bn_mute_stdout();
    pub fn bn_unmute_stdout();
}

/// Stand-ins for the C library off x86-64 (see `build.rs`): the same signatures, failing loudly
/// if the recursive opening is ever reached.
#[cfg(not(target_arch = "x86_64"))]
mod unsupported {
    use super::*;

    fn unsupported() -> ! {
        panic!("the LaBRADOR recursive opening needs the vendored x86-64 C library")
    }

    pub unsafe fn labrador48_init_comkey(_n: usize) {
        unsupported()
    }

    pub unsafe fn labrador48_free_comkey() {
        unsupported()
    }

    pub unsafe fn labrador48_init_witness_raw(_wt: *mut c_void, _r: usize, _n: *const usize) {
        unsupported()
    }

    pub unsafe fn labrador48_free_witness(_wt: *mut c_void) {
        unsupported()
    }

    pub unsafe fn labrador48_init_smplstmnt_raw(_st: *mut c_void, _r: usize, _n: *const usize, _betasq: *const u64, _k: usize) -> c_int {
        unsupported()
    }

    pub unsafe fn labrador48_free_smplstmnt(_st: *mut c_void) {
        unsupported()
    }

    pub unsafe fn labrador48_free_commitment(_com: *mut c_void) {
        unsupported()
    }

    pub unsafe fn labrador48_free_composite(_p: *mut c_void) {
        unsupported()
    }

    pub unsafe fn labrador48_simple_verify(_st: *const c_void, _wt: *const c_void) -> c_int {
        unsupported()
    }

    pub unsafe fn labrador48_composite_prove_simple(_p: *mut c_void, _com: *mut c_void, _st: *const c_void, _wt: *const c_void) -> c_int {
        unsupported()
    }

    pub unsafe fn labrador48_composite_verify_simple(_p: *const c_void, _com: *const c_void, _st: *const c_void) -> c_int {
        unsupported()
    }

    pub unsafe fn bn_sizeof_polx() -> usize {
        unsupported()
    }

    pub unsafe fn bn_polx_align() -> usize {
        unsupported()
    }

    pub unsafe fn bn_ring_degree() -> usize {
        unsupported()
    }

    pub unsafe fn bn_logq() -> usize {
        unsupported()
    }

    pub unsafe fn bn_compiled_q() -> u64 {
        unsupported()
    }

    pub unsafe fn bn_extlen(_len: usize, _deg: usize) -> usize {
        unsupported()
    }

    pub unsafe fn bn_comkey_len() -> usize {
        unsupported()
    }

    pub unsafe fn bn_polx_alloc(_len: usize) -> *mut c_void {
        unsupported()
    }

    pub unsafe fn bn_polx_free(_p: *mut c_void) {
        unsupported()
    }

    pub unsafe fn bn_polx_setzero(_r: *mut c_void, _len: usize) {
        unsupported()
    }

    pub unsafe fn bn_polx_copy(_r: *mut c_void, _a: *const c_void, _len: usize) {
        unsupported()
    }

    pub unsafe fn bn_polx_eq(_a: *const c_void, _b: *const c_void, _len: usize) -> c_int {
        unsupported()
    }

    pub unsafe fn bn_polx_from_int64(_r: *mut c_void, _len: usize, _a: *const i64) {
        unsupported()
    }

    pub unsafe fn bn_polx_from_int16(_r: *mut c_void, _len: usize, _a: *const i16) {
        unsupported()
    }

    pub unsafe fn bn_polx_expand(_r: *mut c_void, _len: usize, _seed: *const u8, _nonce: u64) {
        unsupported()
    }

    pub unsafe fn bn_sx_new(_r: usize, _n: *const usize, _s: *const *const i16) -> *mut c_void {
        unsupported()
    }

    pub unsafe fn bn_sx_free(_sx: *mut c_void) {
        unsupported()
    }

    pub unsafe fn bn_sx_ptr(_sx: *const c_void, _i: usize, _off: usize) -> *const c_void {
        unsupported()
    }

    pub unsafe fn bn_commit_polx(_out: *mut c_void, _key: *const c_void, _s: *const c_void, _len: usize, _deg: usize) {
        unsupported()
    }

    pub unsafe fn bn_commit_i16(_out: *mut c_void, _key: *const c_void, _s: *const i16, _len: usize, _deg: usize) {
        unsupported()
    }

    pub unsafe fn bn_commit_blocks(_out: *mut c_void, _key: *const c_void, _deg: usize, _nb: usize, _len: *const usize, _s: *const *const i16) {
        unsupported()
    }

    pub unsafe fn bn_polx_table_sum(_out: *mut c_void, _len: usize, _table: *const c_void, _terms: usize, _idx: *const u16, _sign: *const i8) {
        unsupported()
    }

    pub unsafe fn bn_sis_secure(_rank: usize, _norm: f64) -> c_int {
        unsupported()
    }

    pub unsafe fn bn_eval_blocks(_out: *mut c_void, _deg: usize, _nz: usize, _idx: *const usize, _off: *const usize, _len: *const usize, _phi: *const *const c_void, _sphi: *const *const i16, _soff: *const usize, _swid: *const usize, _sx: *const c_void) {
        unsupported()
    }

    pub unsafe fn bn_alloc_smplstmnt() -> *mut c_void {
        unsupported()
    }

    pub unsafe fn bn_alloc_witness() -> *mut c_void {
        unsupported()
    }

    pub unsafe fn bn_alloc_composite() -> *mut c_void {
        unsupported()
    }

    pub unsafe fn bn_alloc_commitment() -> *mut c_void {
        unsupported()
    }

    pub unsafe fn bn_free(_p: *mut c_void) {
        unsupported()
    }

    pub unsafe fn bn_smplstmnt_set_digest(_st: *mut c_void, _digest: *const u8) {
        unsupported()
    }

    pub unsafe fn bn_smplstmnt_set_constraint(_st: *mut c_void, _ci: usize, _deg: usize, _nz: usize, _idx: *const usize, _off: *const usize, _len: *const usize, _phi: *const *const c_void, _sphi: *const *const i16, _soff: *const usize, _swid: *const usize, _b: *const c_void) -> c_int {
        unsupported()
    }

    pub unsafe fn bn_set_witness_i16(_wt: *mut c_void, _i: usize, _n: usize, _s: *const i16) -> c_int {
        unsupported()
    }

    pub unsafe fn bn_witness_normsq(_wt: *const c_void, _i: usize) -> u64 {
        unsupported()
    }

    pub unsafe fn bn_composite_size(_p: *const c_void) -> f64 {
        unsupported()
    }

    pub unsafe fn bn_mute_stdout() {
        unsupported()
    }

    pub unsafe fn bn_unmute_stdout() {
        unsupported()
    }
}
#[cfg(not(target_arch = "x86_64"))]
pub use unsupported::*;
