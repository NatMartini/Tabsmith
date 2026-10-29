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
//! compiled, and [`defect_risk`] predicts.
//!
//! Before the firmware runs, the compiler already knows three things about each model:
//! the bounds of its outputs ([`TREE_PROBABILITY_BOUNDS`], [`NETWORK_LOGIT_BOUNDS`]), the
//! most arithmetic one prediction can take (`defects::WORST_CASE`), and, under the Kani
//! model checker (`cargo kani`), that prediction never panics, for every possible input.
//! The `kani` option of `model!` makes each module carry its own proofs
//! (`defects::proofs`, `defects_mlp::proofs`): each stage of `predict` (features,
//! outputs, decode) is proved total on its own, with loop bounds taken from the model's
//! shape. Nothing in this file is written by hand for the proofs.
#![cfg_attr(not(test), no_std)]

tabsmith_macros::model!(defects, "../../fixtures/bundles/binary-gbdt", kani);
tabsmith_macros::model!(defects_mlp, "../../fixtures/bundles/binary-mlp", kani);

pub use tabsmith_embedded::{InputError, SelfCheckError};

/// The trees' probability of a defect lies within these bounds for every possible
/// input: the compiler summed the smallest and the largest leaf of every tree.
pub const TREE_PROBABILITY_BOUNDS: [f64; 2] = defects::PROBABILITY_BOUNDS;

/// The network's logit for every plausible reading: each numeric feature within four
/// standard deviations of its training mean, each category slot anywhere between 0
/// and 1. Interval arithmetic through the layers, widened for rounding, evaluated by
/// the compiler. (Over all inputs a network is unbounded, unlike the trees.)
pub const NETWORK_LOGIT_BOUNDS: [f32; 2] = {
    let [lo, hi] = defects_mlp::feature_box(4.0);
    defects_mlp::output_bounds(&lo, &hi)[0]
};

// The bounds are finite, so the firmware can size its fixed-point output from them,
// and one walk never takes more comparisons than the whole prediction allows.
const _: () = {
    assert!(TREE_PROBABILITY_BOUNDS[0] > 0.0 && TREE_PROBABILITY_BOUNDS[1] < 1.0);
    assert!(NETWORK_LOGIT_BOUNDS[0] > -1e4 && NETWORK_LOGIT_BOUNDS[1] < 1e4);
    assert!(defects::WORST_CASE.comparisons >= defects::MAX_TREE_DEPTH);
};

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

    #[test]
    fn the_examples_stay_inside_what_the_compiler_promised() {
        let [lo, hi] = TREE_PROBABILITY_BOUNDS;
        for input in &defects::EXAMPLES {
            let p = defects::predict(input).unwrap().probability;
            assert!(lo <= p && p <= hi, "{p} outside {lo}..{hi}");
        }
        let [lo, hi] = NETWORK_LOGIT_BOUNDS;
        let [box_lo, box_hi] = defects_mlp::feature_box(4.0);
        for input in &defects_mlp::EXAMPLES {
            let x = defects_mlp::features(input).unwrap();
            if x.iter().enumerate().all(|(j, v)| box_lo[j] <= *v && *v <= box_hi[j]) {
                let [z] = defects_mlp::outputs(&x);
                assert!(lo <= z && z <= hi, "{z} outside {lo}..{hi}");
            }
        }
    }
}
