//! Supported integer commitment factors.
//! The checker, not this parser, applies the minimum-modulus threshold and rejects a smaller product.
pub const SUPPORTED_PRIMES: [u16; 7] = [2917, 3889, 4861, 9721, 12637, 17497, 19441];

pub fn validate(primes: &[u16]) -> Result<u128, &'static str> {
    if primes.is_empty() || primes.len() > SUPPORTED_PRIMES.len() {
        return Err("choose between one and seven distinct supported prime factors");
    }
    let mut product = 1u128;
    for (i, q) in primes.iter().enumerate() {
        if !SUPPORTED_PRIMES.contains(q) {
            return Err("unsupported commitment prime");
        }
        if primes[..i].contains(q) {
            return Err("duplicate commitment prime");
        }
        product = product.checked_mul(*q as u128).ok_or("modulus overflow")?;
    }
    Ok(product)
}

pub fn parse(value: &str) -> Result<Vec<u16>, &'static str> {
    let primes: Vec<u16> = value
        .split(',')
        .map(|s| s.trim().parse().map_err(|_| "invalid prime list"))
        .collect::<Result<_, _>>()?;
    validate(&primes)?;
    Ok(primes)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn supported_choices() {
        for q in SUPPORTED_PRIMES {
            assert_eq!(validate(&[q]), Ok(q as u128));
        }
        assert_eq!(validate(&[9721, 2917, 4861]), Ok(9721u128 * 2917 * 4861));
        assert!(validate(&SUPPORTED_PRIMES).is_ok());
        assert_eq!(parse("9721, 2917,4861").unwrap(), vec![9721, 2917, 4861]);
    }
    #[test]
    fn invalid_choices() {
        for qs in [
            vec![],
            vec![3889, 3889],
            vec![2],
            vec![65535],
            vec![3889; 8],
        ] {
            assert!(validate(&qs).is_err());
        }
        for s in ["", "3889,", "abc", "-1", "65536", "3889,3889"] {
            assert!(parse(s).is_err());
        }
    }
}
