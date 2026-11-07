//! Exact signed14 key images in `[sweep][slot][group pair][row]` order.
//! Each 28-byte record holds sixteen values, with the adjacent groups' eight lanes first.

pub const SLOTS: usize = 648;
pub const ROWS: usize = 7;
pub const LANES: usize = 8;
pub const SINK: usize = 16;
pub const PACKED_RECORD: usize = 28;
pub const DENSE_RECORD: usize = 2 * LANES;

#[cfg(all(target_os = "linux", target_endian = "little", target_feature = "sve2-bitperm"))]
pub type KeyWord = u8;
#[cfg(not(all(target_os = "linux", target_endian = "little", target_feature = "sve2-bitperm")))]
pub type KeyWord = i16;

pub const RECORD_WORDS: usize = if cfg!(all(target_os = "linux", target_endian = "little", target_feature = "sve2-bitperm")) {
    PACKED_RECORD
} else {
    DENSE_RECORD
};

pub fn packed_len(elements: usize) -> usize {
    elements.div_ceil(LANES).div_ceil(2) * SLOTS * ROWS * PACKED_RECORD
}

pub fn runtime_len(elements: usize) -> usize {
    elements.div_ceil(LANES).div_ceil(2) * SLOTS * ROWS * RECORD_WORDS
}

pub fn sweep_offset(first_group: usize) -> usize {
    debug_assert_eq!(first_group % SINK, 0);
    first_group / 2 * SLOTS * ROWS * RECORD_WORDS
}

fn pack_record(values: &[i16; DENSE_RECORD]) -> [u8; PACKED_RECORD] {
    let mut out = [0; PACKED_RECORD];
    for (chunk, values) in values.chunks_exact(4).enumerate() {
        let mut word = 0u64;
        for (lane, &value) in values.iter().enumerate() {
            assert!((-8192..8192).contains(&value));
            word |= u64::from(value as u16 & 0x3fff) << (14 * lane);
        }
        out[chunk * 7..chunk * 7 + 7].copy_from_slice(&word.to_le_bytes()[..7]);
    }
    out
}

fn unpack_record(record: &[u8; PACKED_RECORD]) -> [i16; DENSE_RECORD] {
    let mut out = [0; DENSE_RECORD];
    for chunk in 0..4 {
        let mut bytes = [0u8; 8];
        bytes[..7].copy_from_slice(&record[chunk * 7..chunk * 7 + 7]);
        let word = u64::from_le_bytes(bytes);
        for lane in 0..4 {
            out[4 * chunk + lane] = (((word >> (14 * lane)) as u16) << 2) as i16 >> 2;
        }
    }
    out
}

pub fn pack_group_major(values: &[i16], elements: usize) -> Vec<u8> {
    let groups = elements.div_ceil(LANES);
    assert_eq!(values.len(), groups * SLOTS * ROWS * LANES);
    let mut image = Vec::with_capacity(packed_len(elements));
    for first in (0..groups).step_by(SINK) {
        let here = (groups - first).min(SINK);
        for slot in 0..SLOTS {
            for pair in 0..here.div_ceil(2) {
                for row in 0..ROWS {
                    let mut record = [0; DENSE_RECORD];
                    for half in 0..2 {
                        if 2 * pair + half < here {
                            let at = (((first + 2 * pair + half) * SLOTS + slot) * ROWS + row) * LANES;
                            record[half * LANES..(half + 1) * LANES].copy_from_slice(&values[at..at + LANES]);
                        }
                    }
                    image.extend_from_slice(&pack_record(&record));
                }
            }
        }
    }
    assert_eq!(image.len(), packed_len(elements));
    image
}

pub fn expand_blocked(image: &[u8], elements: usize) -> Vec<i16> {
    assert_eq!(image.len(), packed_len(elements));
    let mut values = Vec::with_capacity(image.len() / PACKED_RECORD * DENSE_RECORD);
    for record in image.chunks_exact(PACKED_RECORD) {
        values.extend_from_slice(&unpack_record(record.try_into().unwrap()));
    }
    values
}

pub fn from_packed(image: Vec<u8>, elements: usize) -> Vec<KeyWord> {
    assert_eq!(image.len(), packed_len(elements));
    #[cfg(all(target_os = "linux", target_endian = "little", target_feature = "sve2-bitperm"))]
    { image }
    #[cfg(not(all(target_os = "linux", target_endian = "little", target_feature = "sve2-bitperm")))]
    { expand_blocked(&image, elements) }
}

pub fn from_group_major(values: &[i16], elements: usize) -> Vec<KeyWord> {
    from_packed(pack_group_major(values, elements), elements)
}

pub fn image(values: &[KeyWord], elements: usize) -> Vec<u8> {
    assert_eq!(values.len(), runtime_len(elements));
    #[cfg(all(target_os = "linux", target_endian = "little", target_feature = "sve2-bitperm"))]
    { values.to_vec() }
    #[cfg(not(all(target_os = "linux", target_endian = "little", target_feature = "sve2-bitperm")))]
    {
        let mut image = Vec::with_capacity(packed_len(elements));
        for record in values.chunks_exact(DENSE_RECORD) {
            image.extend_from_slice(&pack_record(record.try_into().unwrap()));
        }
        image
    }
}

#[cfg(test)]
pub fn group_major(image: &[u8], elements: usize) -> Vec<i16> {
    assert_eq!(image.len(), packed_len(elements));
    let groups = elements.div_ceil(LANES);
    let mut values = vec![0; groups * SLOTS * ROWS * LANES];
    let mut records = image.chunks_exact(PACKED_RECORD);
    for first in (0..groups).step_by(SINK) {
        let here = (groups - first).min(SINK);
        for slot in 0..SLOTS {
            for pair in 0..here.div_ceil(2) {
                for row in 0..ROWS {
                    let record = unpack_record(records.next().unwrap().try_into().unwrap());
                    for half in 0..2 {
                        if 2 * pair + half < here {
                            let at = (((first + 2 * pair + half) * SLOTS + slot) * ROWS + row) * LANES;
                            values[at..at + LANES].copy_from_slice(&record[half * LANES..(half + 1) * LANES]);
                        } else {
                            assert_eq!(&record[half * LANES..(half + 1) * LANES], &[0; LANES]);
                        }
                    }
                }
            }
        }
    }
    assert!(records.next().is_none());
    values
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random(state: &mut u64) -> i16 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        (*state as u16 & 0x3fff) as i16 - 8192
    }

    #[test]
    fn signed14_is_exact_in_every_lane() {
        let mut state = 73013;
        for lane in 0..DENSE_RECORD {
            for value in -8192..8192 {
                let mut values = core::array::from_fn(|_| random(&mut state));
                values[lane] = value;
                assert_eq!(unpack_record(&pack_record(&values)), values);
            }
        }
    }

    #[test]
    fn image_round_trip_and_tail_layout_sizes() {
        let mut state = 19;
        for elements in [0usize, 1, 8, 16, 127, 128, 129, 229, 232, 301, 4096] {
            let groups = elements.div_ceil(LANES);
            let values: Vec<_> = (0..groups * SLOTS * ROWS * LANES).map(|_| random(&mut state)).collect();
            let packed = pack_group_major(&values, elements);
            assert_eq!(packed.len(), groups.div_ceil(2) * SLOTS * ROWS * 28);
            assert_eq!(group_major(&packed, elements), values);
            let runtime = from_packed(packed.clone(), elements);
            assert_eq!(runtime.len(), runtime_len(elements));
            assert_eq!(image(&runtime, elements), packed);
            if elements == 4096 {
                assert_eq!(packed.len(), 32_514_048);
                assert_eq!(expand_blocked(&packed, elements).len() * 2, 37_158_912);
            }
            let expanded = expand_blocked(&packed, elements);
            for first in (0..groups).step_by(SINK) {
                let here = (groups - first).min(SINK);
                let pairs = here.div_ceil(2);
                for slot in 0..SLOTS {
                    for group in 0..here {
                        for row in 0..ROWS {
                            let record = first / 2 * SLOTS * ROWS + (slot * pairs + group / 2) * ROWS + row;
                            let at = (((first + group) * SLOTS + slot) * ROWS + row) * LANES;
                            assert_eq!(&expanded[record * DENSE_RECORD + group % 2 * LANES..][..LANES], &values[at..at + LANES]);
                        }
                    }
                }
            }
        }
    }

    #[test]
    #[should_panic]
    fn out_of_range_values_are_rejected() {
        pack_record(&[8192; DENSE_RECORD]);
    }
}
