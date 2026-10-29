//! Tabsmith models compiled into `no_std` Rust, for firmware.
//!
//! Two of the repository's fixture bundles become plain Rust modules when this crate
//! compiles: `defects` (gradient-boosted trees) and `defects_mlp` (a small neural
//! network), both predicting equipment defects. The crate has no allocator, no
//! standard library and no `unsafe` code; it builds for a Cortex-M4F:
//!
//! ```sh
//! rustup target add thumbv7em-none-eabihf
//! cargo build --release --target thumbv7em-none-eabihf
//! ```
//!
//! On the device, [`power_on_self_test`] replays the bundles' examples and checks every
//! prediction bit for bit against what `tabsmith-runtime` predicted when the crate was
//! compiled, and [`defect_risk`] predicts. Under the Kani model checker
//! (`cargo kani`), the proofs at the end of this file show that prediction never
//! panics, for every possible input: each stage of `predict` (features, outputs,
//! decode) is proved total on its own.
#![cfg_attr(not(test), no_std)]

tabsmith_macros::model!(defects, "../../fixtures/bundles/binary-gbdt", kani);
tabsmith_macros::model!(defects_mlp, "../../fixtures/bundles/binary-mlp", kani);

pub use tabsmith_embedded::{InputError, SelfCheckError};

/// Checks both models against the runtime's predictions recorded at compile time.
/// Run it once at start-up: it fails on a platform whose floating point differs.
pub fn power_on_self_test() -> Result<(), SelfCheckError> {
    defects::self_check()?;
    defects_mlp::self_check()
}

/// Probability of a defect, from the trees and the network averaged.
pub fn defect_risk(input: &defects::Input, same_reading: &defects_mlp::Input) -> Result<f64, InputError> {
    let trees = defects::predict(input)?;
    let network = defects_mlp::predict(same_reading)?;
    Ok(0.5 * (trees.probability + network.probability))
}

/// Prediction never panics, for every input. A model's `predict` is
/// `decode(&outputs(&features(input)?))`, and each stage is proved total for every value
/// of its argument's type: a superset of what the stage before it can return. Together
/// the three proofs cover every input (any number, NaN and infinities included, any
/// category, any missing value). Proving the stages apart keeps each proof small: no
/// array index depends on the network's arithmetic, so Kani slices it away, and `exp`
/// is checked once, in `decode`. The trees' `outputs` is proved one walk at a time:
/// all 14 walks in one proof ran out of memory with 14 GB, and a CI runner has 16 GB.
#[cfg(kani)]
mod proofs {
    use super::*;

    /// Encoding any input returns features or an `InputError`.
    #[kani::proof]
    #[kani::unwind(40)]
    fn tree_features_are_total() {
        let input: defects::Input = kani::any();
        let _ = defects::features(&input);
    }

    /// Walking any tree with any feature values stays inside the arrays and ends at a
    /// leaf. `outputs` adds one walk per tree to each output's starting score: `value`
    /// has one entry per node, and the compiler checks that there is one starting score
    /// per output, so with every walk in range the sum cannot panic either.
    #[kani::proof]
    #[kani::unwind(10)]
    fn tree_walk_is_total() {
        let t: usize = kani::any();
        kani::assume(t < defects::TREES.roots.len());
        let x: [f32; defects::N_FEATURES] = kani::any();
        let leaf = defects::TREES.leaf(t, &x);
        assert!(defects::TREES.feature[leaf] < 0);
    }

    /// The decision for any logit: the sigmoid (and its `exp`) never panics.
    #[kani::proof]
    fn tree_decode_is_total() {
        let z: [f32; defects::N_OUTPUTS] = kani::any();
        let _ = defects::decode(&z);
    }

    /// Encoding any input returns features or an `InputError`.
    #[kani::proof]
    #[kani::unwind(40)]
    fn network_features_are_total() {
        let input: defects_mlp::Input = kani::any();
        let _ = defects_mlp::features(&input);
    }

    /// The dense layers stay inside their weight arrays, for any feature values.
    #[kani::proof]
    #[kani::unwind(40)]
    fn network_layers_are_total() {
        let x: [f32; defects_mlp::N_FEATURES] = kani::any();
        let _ = defects_mlp::outputs(&x);
    }

    /// The decision for any logit.
    #[kani::proof]
    fn network_decode_is_total() {
        let z: [f32; defects_mlp::N_OUTPUTS] = kani::any();
        let _ = defects_mlp::decode(&z);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_self_test_passes_on_this_machine() {
        power_on_self_test().unwrap();
    }

    #[test]
    fn risk_is_a_probability() {
        let risk = defect_risk(&defects::EXAMPLES[0], &defects_mlp::EXAMPLES[0]).unwrap();
        assert!((0.0..=1.0).contains(&risk));
    }
}
