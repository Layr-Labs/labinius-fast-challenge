//! The moduli this prover can commit under.
//!
//! The checker takes any `u128` that clears its size conditions. This prover goes through the
//! library's limb kernels, whose arithmetic is instantiated per prime at compile time, so the
//! modulus it can reach is a product of distinct supported primes — it commits modulo each of
//! them and recombines the residues into coefficients modulo the product. A modulus with any
//! other factorisation, a power of two included, is one the checker would accept and this prover
//! cannot build; that is a property of this solution, not of the scheme.
//!
//! The checker, not this parser, applies the minimum-modulus threshold and rejects a smaller
//! product.

/// The primes the library has opening arithmetic for.
pub const SUPPORTED_PRIMES: [u16; 7] = [2917, 3889, 4861, 9721, 12637, 17497, 19441];

/// The order the limbs are listed in once a modulus has been factored. The first entry present
/// becomes the base limb — the one whose transform the commitment keeps and whose domain the
/// fold runs in — so the fully splitting primes below `2^14`, which run the hand-scheduled
/// `ntt::bin_asm`, come first.
const LIMB_ORDER: [u16; 7] = [3889, 9721, 17497, 19441, 2917, 4861, 12637];

/// [`LIMB_ORDER`] is [`SUPPORTED_PRIMES`] reordered, so the two lists admit the same moduli.
const _: () = {
    let mut i = 0;
    while i < LIMB_ORDER.len() {
        let mut found = false;
        let mut j = 0;
        while j < SUPPORTED_PRIMES.len() {
            found |= LIMB_ORDER[i] == SUPPORTED_PRIMES[j];
            j += 1;
        }
        assert!(found, "a limb order entry is not a supported prime");
        i += 1;
    }
    assert!(LIMB_ORDER.len() == SUPPORTED_PRIMES.len());
};

/// The limbs of `modulus`, base first, if it is a product of distinct supported primes.
pub fn limbs(modulus: u128) -> Result<Vec<u16>, &'static str> {
    if modulus < 2 {
        return Err("a modulus is at least 2");
    }
    let mut rest = modulus;
    let mut chosen = Vec::new();
    for q in LIMB_ORDER {
        if rest % q as u128 == 0 {
            rest /= q as u128;
            chosen.push(q);
        }
    }
    if rest != 1 {
        return Err("this prover only commits modulo a product of distinct supported primes");
    }
    Ok(chosen)
}

/// `--modulus` takes the modulus itself in decimal, not a factor list: the proof carries the
/// modulus and the checker never sees the factorisation.
pub fn parse(value: &str) -> Result<u128, &'static str> {
    let modulus: u128 = value.trim().parse().map_err(|_| "invalid modulus")?;
    limbs(modulus)?;
    Ok(modulus)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn supported_choices() {
        for q in SUPPORTED_PRIMES {
            assert_eq!(limbs(q as u128), Ok(vec![q]));
        }
        assert_eq!(
            limbs(3889u128 * 2917 * 4861),
            Ok(vec![3889, 2917, 4861]),
            "the base limb comes first, not the smallest prime"
        );
        assert_eq!(limbs(9721u128 * 2917 * 4861), Ok(vec![9721, 2917, 4861]));
        let all: u128 = SUPPORTED_PRIMES.iter().map(|&q| q as u128).product();
        assert_eq!(limbs(all).unwrap().len(), 7);
        assert_eq!(parse(" 55144219393 ").unwrap(), 55_144_219_393);
    }
    #[test]
    fn invalid_choices() {
        for modulus in [
            0u128,
            1,
            2,
            1 << 36,
            3889 * 3889,
            55_144_219_393 * 2,
            u128::MAX,
        ] {
            assert!(limbs(modulus).is_err(), "{modulus} is not a limb product");
        }
        for s in ["", "abc", "-1", "3889,2917", "0x10"] {
            assert!(parse(s).is_err());
        }
    }
}
