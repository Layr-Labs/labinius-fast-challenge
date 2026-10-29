// Appended to the exact const function and initializer extracted from C main.
pub fn normalize<const LEN: usize>(bytes: [u8; LEN]) -> [u8; LEN] {
    sms_canonical_means(bytes)
}
pub fn raw() -> &'static [u8] {
    include_bytes!(concat!(env!("OUT_DIR"), "/sms-mur.bin"))
}
pub fn canonical() -> &'static [u8] {
    &SMS_MUR.0
}
