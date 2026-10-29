//! What the compiler knows about a compiled model beyond its weights holds at run time:
//! the raw outputs of every row stay inside `OUTPUT_BOUNDS`, the outputs of the rows of
//! a box stay inside `output_bounds` of that box, the probability and value bounds
//! derived from them hold, `WORST_CASE` describes the model, and `Model` works through
//! the trait, for a compiled model and for the runtime alike.

use proptest::prelude::*;
use proptest::strategy::ValueTree;
use tabsmith_embedded::Model;
use tabsmith_runtime::{LoadOptions, Predictor, Record};

tabsmith_macros::model!(binary_mlp, "../../fixtures/bundles/binary-mlp");
tabsmith_macros::model!(binary_gbdt, "../../fixtures/bundles/binary-gbdt");
tabsmith_macros::model!(multiclass_linear, "../../fixtures/bundles/multiclass-linear");
tabsmith_macros::model!(multiclass_gbdt, "../../fixtures/bundles/multiclass-gbdt");
tabsmith_macros::model!(regression_mlp, "../../fixtures/bundles/regression-mlp");
tabsmith_macros::model!(regression_gbdt, "../../fixtures/bundles/regression-gbdt");
tabsmith_macros::model!(policies_linear, "../../fixtures/bundles/policies-linear");

fn inside(bounds: [f32; 2], v: f32) -> bool {
    bounds[0] <= v && v <= bounds[1]
}

macro_rules! bounds_hold {
    ($test:ident, $m:ident) => {
        mod $test {
            use super::$m as m;
            use super::*;

            /// A dense network multiplies its features; the bounds of a box with infinite
            /// ends cannot cover `inf - inf`. Trees only compare, so for them every row,
            /// NaN and infinities included, lies inside the bounds over everything.
            const DENSE: bool = m::WORST_CASE.multiply_adds > 0;

            fn any_row() -> impl Strategy<Value = [f32; m::N_FEATURES]> {
                prop::collection::vec(any::<f32>(), m::N_FEATURES).prop_map(|v| {
                    let mut x = [0f32; m::N_FEATURES];
                    x.copy_from_slice(&v);
                    x
                })
            }

            fn row_in(lo: [f32; m::N_FEATURES], hi: [f32; m::N_FEATURES]) -> impl Strategy<Value = [f32; m::N_FEATURES]> {
                prop::collection::vec(0f32..=1f32, m::N_FEATURES).prop_map(move |u| {
                    let mut x = [0f32; m::N_FEATURES];
                    for j in 0..m::N_FEATURES {
                        x[j] = (lo[j] + (hi[j] - lo[j]) * u[j]).clamp(lo[j], hi[j]);
                    }
                    x
                })
            }

            #[test]
            fn the_shape_constants_describe_the_model() {
                let w = m::WORST_CASE;
                assert!(w.comparisons + w.multiply_adds > 0);
                assert!(w.additions > 0);
                assert_eq!(
                    w.exponentials,
                    match m::TASK {
                        "binary" => 1,
                        "multiclass" => m::N_OUTPUTS,
                        _ => 0,
                    }
                );
                for b in m::OUTPUT_BOUNDS {
                    assert!(b[0] <= b[1], "{b:?}");
                    if !DENSE {
                        assert!(b[0].is_finite() && b[1].is_finite(), "trees have finite outputs: {b:?}");
                    }
                }
                let [lo, hi] = m::feature_box(1.0);
                for j in 0..m::N_FEATURES {
                    assert!(lo[j] <= hi[j] && lo[j] >= -1.0 && hi[j] <= 1.0);
                }
            }

            proptest! {
                #![proptest_config(ProptestConfig::with_cases(2000))]
                #[test]
                fn every_row_stays_inside_the_bounds_over_everything(x in any_row()) {
                    if DENSE {
                        prop_assume!(x.iter().all(|v| v.is_finite()));
                    }
                    let out = m::outputs(&x);
                    for o in 0..m::N_OUTPUTS {
                        prop_assert!(inside(m::OUTPUT_BOUNDS[o], out[o]), "output {o} = {} outside {:?}", out[o], m::OUTPUT_BOUNDS[o]);
                    }
                }

                #[test]
                fn the_rows_of_a_box_stay_inside_its_bounds(sd in 0.05f32..8.0, u in 0f32..1f32) {
                    let [lo, mut hi] = m::feature_box(sd);
                    // Shrink the category slots of the box too, so that some are fixed.
                    for j in 0..m::N_FEATURES {
                        if lo[j] == 0.0 && hi[j] == 1.0 && (j as f32 * 0.37 + u) % 1.0 < 0.5 {
                            hi[j] = 0.0;
                        }
                    }
                    let bounds = m::output_bounds(&lo, &hi);
                    let mut runner = proptest::test_runner::TestRunner::deterministic();
                    for _ in 0..50 {
                        let x = row_in(lo, hi).new_tree(&mut runner).unwrap().current();
                        let out = m::outputs(&x);
                        for o in 0..m::N_OUTPUTS {
                            prop_assert!(inside(bounds[o], out[o]), "output {o} = {} outside {:?} for the box ±{sd} sd", out[o], bounds[o]);
                            prop_assert!(inside(m::OUTPUT_BOUNDS[o], out[o]));
                        }
                    }
                }
            }
        }
    };
}

bounds_hold!(binary_mlp_bounds, binary_mlp);
bounds_hold!(binary_gbdt_bounds, binary_gbdt);
bounds_hold!(multiclass_linear_bounds, multiclass_linear);
bounds_hold!(multiclass_gbdt_bounds, multiclass_gbdt);
bounds_hold!(regression_mlp_bounds, regression_mlp);
bounds_hold!(regression_gbdt_bounds, regression_gbdt);
bounds_hold!(policies_linear_bounds, policies_linear);

#[test]
fn the_tree_models_bound_their_probability_and_value() {
    // Binary trees: the probability of every example lies inside PROBABILITY_BOUNDS.
    let [lo, hi] = binary_gbdt::PROBABILITY_BOUNDS;
    assert!(0.0 < lo && lo < hi && hi < 1.0, "{lo} {hi}");
    for input in &binary_gbdt::EXAMPLES {
        let p = binary_gbdt::predict(input).unwrap().probability;
        assert!(lo <= p && p <= hi);
    }
    // Regression trees: the value of every example lies inside VALUE_BOUNDS.
    let [lo, hi] = regression_gbdt::VALUE_BOUNDS;
    assert!(lo.is_finite() && hi.is_finite() && lo < hi);
    for input in &regression_gbdt::EXAMPLES {
        let v = regression_gbdt::predict(input).unwrap().value;
        assert!(lo <= v && v <= hi);
    }
    // The networks are unbounded over everything, so their derived bounds are the
    // whole range.
    assert_eq!(binary_mlp::PROBABILITY_BOUNDS, [0.0, 1.0]);
    assert_eq!(regression_mlp::VALUE_BOUNDS, [f64::NEG_INFINITY, f64::INFINITY]);
    // ...but a box of plausible readings bounds them.
    let [lo, hi] = binary_mlp::feature_box(4.0);
    let [b] = binary_mlp::output_bounds(&lo, &hi);
    assert!(b[0].is_finite() && b[1].is_finite() && b[0] < b[1], "{b:?}");
}

// The shape constants are constants: the compiler checks them, not a test.
const _: () = {
    assert!(binary_gbdt::MAX_TREE_DEPTH >= 1);
    assert!(binary_gbdt::WORST_CASE.comparisons >= binary_gbdt::MAX_TREE_DEPTH);
    assert!(binary_gbdt::WORST_CASE.multiply_adds == 0);
    assert!(binary_mlp::WORST_CASE.multiply_adds == 35 * 32 + 32 * 16 + 16);
    assert!(binary_mlp::WORST_CASE.additions == 32 + 16 + 1);
    assert!(binary_mlp::WORST_CASE.comparisons == 32 + 16);
};

#[test]
fn fixtures_without_a_calibration_have_no_intervals_or_sets() {
    // The fixture bundles predate conformal.json: the constants say so, honestly.
    assert!(regression_mlp::CONFORMAL.is_none() && binary_gbdt::CONFORMAL.is_none());
    let p = regression_mlp::predict(&regression_mlp::EXAMPLES[0]).unwrap();
    assert_eq!(regression_mlp::interval(&p), None);
    let p = binary_gbdt::predict(&binary_gbdt::EXAMPLES[0]).unwrap();
    assert_eq!(binary_gbdt::prediction_set(&p), None);
    let p = multiclass_gbdt::predict(&multiclass_gbdt::EXAMPLES[0]).unwrap();
    assert_eq!(multiclass_gbdt::prediction_set(&p), None);
}

/// Code written against the trait takes a compiled model or a loaded bundle alike.
fn score<M: Model>(model: &M, input: &M::Input) -> Result<M::Prediction, M::Error> {
    model.predict(input)
}

#[test]
fn a_compiled_model_and_the_runtime_share_the_model_trait() {
    let compiled = score(&binary_mlp::Model, &binary_mlp::EXAMPLES[0]).unwrap();
    let dir = format!("{}/../../fixtures/bundles/binary-mlp", env!("CARGO_MANIFEST_DIR"));
    let runtime = Predictor::load(dir, LoadOptions::cpu()).unwrap();
    let record = Record::from_json(&runtime.bundle().examples.records[0]).unwrap();
    let interpreted = score(&runtime, &record).unwrap();
    assert_eq!(compiled.probability.to_bits(), interpreted.probability().unwrap().to_bits());
    assert_eq!(compiled.label.as_str(), interpreted.label);
}
