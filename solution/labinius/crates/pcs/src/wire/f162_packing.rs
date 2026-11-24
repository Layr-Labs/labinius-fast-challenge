//! Little-endian 162-bit field elements with no gaps between adjacent elements.

pub(super) fn pack(values: impl ExactSizeIterator<Item = [u64; 3]>) -> Vec<u8> {
    let mut bytes = Vec::with_capacity((values.len() * 162).div_ceil(8));
    let mut offset = 0;
    let mut carry = 0u8;
    for [low, mid, high] in values {
        // Each element contributes 20 full bytes plus two bits. The previous
        // element's 0, 2, 4 or 6 leftover bits occupy the bottom of the first byte.
        let low128 = (u128::from(mid) << 64) | u128::from(low);
        let upper = (u128::from(high & ((1 << 34) - 1)) << 64) | u128::from(mid);
        let tail = (upper >> (64u32 - offset)) as u64;
        bytes.extend_from_slice(&((low128 << offset) | u128::from(carry)).to_le_bytes());
        bytes.extend_from_slice(&(tail as u32).to_le_bytes());
        carry = (tail >> 32) as u8;
        offset += 2;
        if offset == 8 {
            bytes.push(carry);
            offset = 0;
            carry = 0;
        }
    }
    if offset != 0 {
        bytes.push(carry);
    }
    bytes
}

#[cfg(test)]
mod tests {
    use super::pack;

    fn bit_reference(values: &[[u64; 3]]) -> Vec<u8> {
        let mut bytes = vec![0; (values.len() * 162).div_ceil(8)];
        for (index, limbs) in values.iter().enumerate() {
            for bit in 0..162 {
                let pos = index * 162 + bit;
                let value = (limbs[bit / 64] >> (bit % 64)) & 1;
                bytes[pos / 8] |= (value as u8) << (pos % 8);
            }
        }
        bytes
    }

    #[test]
    fn every_input_bit_at_each_alignment() {
        for slot in 0..4 {
            for bit in 0..192 {
                let mut values = [[0; 3]; 4];
                values[slot][bit / 64] = 1 << (bit % 64);
                assert_eq!(pack(values.into_iter()), bit_reference(&values));
            }
        }
    }

    #[test]
    fn empty_partial_and_full_rows_match_the_bit_stream() {
        let mut state = 0x55aa1234abcd6789u64;
        let values: Vec<_> = (0..1025)
            .map(|_| std::array::from_fn(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state
            }))
            .collect();
        for count in [0, 1, 2, 3, 4, 5, 6, 7, 8, 1023, 1024, 1025] {
            let row = &values[..count];
            assert_eq!(pack(row.iter().copied()), bit_reference(row));
        }
    }
}
