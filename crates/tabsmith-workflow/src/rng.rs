//! A tiny, versioned PRNG. Splits and minibatch orders must stay reproducible
//! across dependency upgrades, so they do not depend on an external RNG crate
//! whose algorithms may change between releases.

/// SplitMix64 (Steele, Lea & Flood 2014).
#[derive(Debug, Clone)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Derives an independent stream from a seed and a label.
    pub fn derive(seed: u64, label: &str) -> Self {
        let mut h = 0xcbf2_9ce4_8422_2325u64; // FNV-1a over the label
        for b in label.as_bytes() {
            h ^= u64::from(*b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        Self::new(seed ^ h.rotate_left(17))
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform integer in `0..n` (Lemire's multiply-shift with rejection).
    pub fn below(&mut self, n: u64) -> u64 {
        assert!(n > 0);
        loop {
            let x = self.next_u64();
            let m = (x as u128) * (n as u128);
            let low = m as u64;
            if low >= n.wrapping_neg() % n {
                return (m >> 64) as u64;
            }
        }
    }

    /// Uniform float in [0, 1).
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Standard normal via Box–Muller.
    pub fn normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(f64::MIN_POSITIVE);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }

    /// Fisher–Yates shuffle.
    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            let j = self.below(i as u64 + 1) as usize;
            items.swap(i, j);
        }
    }
}

/// Order of training rows for one epoch; depends only on (seed, epoch), so the
/// sampler state to persist for exact resume is just (epoch, batch index).
pub fn epoch_permutation(seed: u64, epoch: usize, n: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..n).collect();
    SplitMix64::derive(seed, &format!("epoch-{epoch}")).shuffle(&mut idx);
    idx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_values_are_stable() {
        // Pinned outputs: changing them changes every split and minibatch order.
        let mut r = SplitMix64::new(1234567);
        assert_eq!(r.next_u64(), 6457827717110365317);
        assert_eq!(r.next_u64(), 3203168211198807973);
    }

    #[test]
    fn permutation_is_deterministic_and_complete() {
        let a = epoch_permutation(7, 3, 100);
        let b = epoch_permutation(7, 3, 100);
        assert_eq!(a, b);
        assert_ne!(a, epoch_permutation(7, 4, 100));
        let mut s = a.clone();
        s.sort_unstable();
        assert_eq!(s, (0..100).collect::<Vec<_>>());
    }

    #[test]
    fn below_is_in_range() {
        let mut r = SplitMix64::new(1);
        for n in [1u64, 2, 3, 10, 1000] {
            for _ in 0..100 {
                assert!(r.below(n) < n);
            }
        }
    }
}
