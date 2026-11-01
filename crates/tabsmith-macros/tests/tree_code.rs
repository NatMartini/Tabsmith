//! Trees as generated code predict exactly what trees as arrays predict, bit for bit:
//! at every split on its exact threshold, at every leaf, on NaN and infinities, and in
//! `outputs_batch`, which can run them the other way round. The compiler can run them too.
//!
//! A number written wrongly in the generated code, by a single bit, fails here: every
//! node is reached by a row built for it, and each tree is compared on its own, so that
//! the sum of the trees cannot round the difference away.

use proptest::prelude::*;
use tabsmith_embedded::kernels::Trees;

tabsmith_macros::model!(binary_code, "../../fixtures/bundles/binary-gbdt", tree_code);
tabsmith_macros::model!(binary_arrays, "../../fixtures/bundles/binary-gbdt", tree_arrays);
tabsmith_macros::model!(multiclass_code, "../../fixtures/bundles/multiclass-gbdt", tree_code);
tabsmith_macros::model!(multiclass_arrays, "../../fixtures/bundles/multiclass-gbdt", tree_arrays);
tabsmith_macros::model!(regression_code, "../../fixtures/bundles/regression-gbdt", tree_code);
tabsmith_macros::model!(regression_arrays, "../../fixtures/bundles/regression-gbdt", tree_arrays);

/// A value of each feature inside its interval `(lo, hi]`.
fn row_inside(bounds: &[(f32, f32)]) -> Vec<f32> {
    bounds
        .iter()
        .map(|&(lo, hi)| {
            if hi.is_finite() {
                hi
            } else if lo.is_finite() {
                lo.next_up()
            } else {
                0.0
            }
        })
        .collect()
}

/// `(tree, row)` pairs that reach every node of every tree: for a split, one row on each
/// side of its exact threshold; for a leaf, one row. `bounds` holds the interval of each
/// feature on the way to node `i`.
fn visit(t: &Trees<'_>, tree: usize, i: usize, bounds: &mut Vec<(f32, f32)>, out: &mut Vec<(usize, Vec<f32>)>) {
    if t.feature[i] < 0 {
        out.push((tree, row_inside(bounds)));
        return;
    }
    let (f, threshold) = (t.feature[i] as usize, t.threshold[i]);
    let (lo, hi) = bounds[f];
    for edge in [threshold, threshold.next_up(), threshold.next_down()] {
        if lo < edge && edge <= hi {
            let mut row = row_inside(bounds);
            row[f] = edge;
            out.push((tree, row));
        }
    }
    if lo < threshold {
        bounds[f] = (lo, hi.min(threshold));
        visit(t, tree, t.left[i] as usize, bounds, out);
        bounds[f] = (lo, hi);
    }
    if threshold < hi {
        bounds[f] = (lo.max(threshold), hi);
        visit(t, tree, t.right[i] as usize, bounds, out);
        bounds[f] = (lo, hi);
    }
}

fn rows_reaching_every_node(t: &Trees<'_>, n_features: usize) -> Vec<(usize, Vec<f32>)> {
    let mut out = Vec::new();
    for (tree, &root) in t.roots.iter().enumerate() {
        let mut bounds = vec![(f32::NEG_INFINITY, f32::INFINITY); n_features];
        visit(t, tree, root as usize, &mut bounds, &mut out);
    }
    out
}

macro_rules! same_bits {
    ($test:ident, $code:ident, $arrays:ident) => {
        mod $test {
            use super::*;
            use super::{$arrays as arrays, $code as code};

            type Row = [f32; code::N_FEATURES];
            type Out = [f32; code::N_OUTPUTS];

            fn bits(o: &Out) -> Vec<u32> {
                o.iter().map(|v| v.to_bits()).collect()
            }

            fn nodes() -> Vec<(usize, Row)> {
                let rows = rows_reaching_every_node(&arrays::TREES, code::N_FEATURES);
                // At least one row per node: the trees have no node out of reach.
                assert!(rows.len() >= arrays::TREES.feature.len(), "{} rows", rows.len());
                rows.into_iter().map(|(tree, row)| (tree, Row::try_from(row).unwrap())).collect()
            }

            #[test]
            fn every_tree_gives_the_same_leaf_at_every_node() {
                for (tree, row) in &nodes() {
                    let want = arrays::TREES.value[arrays::TREES.leaf(*tree, row)];
                    let got = code::TREE_FUNCTIONS[*tree](row);
                    assert_eq!(got.to_bits(), want.to_bits(), "tree {tree}, row {row:?}");
                }
            }

            #[test]
            fn every_output_is_the_same_at_every_node() {
                for (_, row) in nodes() {
                    assert_eq!(bits(&code::outputs(&row)), bits(&arrays::outputs(&row)), "{row:?}");
                }
            }

            #[test]
            fn a_batch_gives_what_single_rows_give() {
                let rows: Vec<Row> = nodes().into_iter().map(|(_, row)| row).collect();
                for n in [0, 1, 31, 32, 33, 64, 100, rows.len()] {
                    let n = n.min(rows.len());
                    let want: Vec<Vec<u32>> = rows[..n].iter().map(|r| bits(&arrays::outputs(r))).collect();
                    let check = |out: &[Out]| {
                        let got: Vec<Vec<u32>> = out.iter().map(bits).collect();
                        assert_eq!(got, want, "{n} rows");
                    };
                    let mut out = vec![[0f32; code::N_OUTPUTS]; n];
                    assert_eq!(code::outputs_batch(&rows[..n], &mut out), n);
                    check(&out);
                    code::outputs_by_tree(&rows[..n], &mut out);
                    check(&out);
                    assert_eq!(arrays::outputs_batch(&rows[..n], &mut out), n);
                    check(&out);
                }
            }

            #[test]
            fn a_batch_stops_at_the_shorter_slice() {
                let rows = vec![[0.5f32; code::N_FEATURES]; 40];
                let empty = [-1f32; code::N_OUTPUTS];
                let one = bits(&arrays::outputs(&rows[0]));

                let mut out = vec![empty; 35];
                assert_eq!(code::outputs_batch(&rows, &mut out), 35);
                assert!(out.iter().all(|o| bits(o) == one));
                let mut out = vec![empty; 35];
                code::outputs_by_tree(&rows, &mut out);
                assert!(out.iter().all(|o| bits(o) == one));

                let mut out = vec![empty; 45];
                assert_eq!(code::outputs_batch(&rows, &mut out), 40);
                assert!(out[..40].iter().all(|o| bits(o) == one));
                assert!(out[40..].iter().all(|o| o == &empty), "rows past the input stay as they were");
                let mut out = vec![empty; 45];
                code::outputs_by_tree(&rows, &mut out);
                assert!(out[..40].iter().all(|o| bits(o) == one));
                assert!(out[40..].iter().all(|o| o == &empty));
            }

            #[test]
            fn the_whole_model_is_the_same() {
                assert_eq!(code::self_check(), Ok(()));
                assert_eq!(code::BUNDLE_ID, arrays::BUNDLE_ID);
                assert_eq!(code::WORST_CASE, arrays::WORST_CASE);
                assert_eq!(format!("{:?}", code::OUTPUT_BOUNDS), format!("{:?}", arrays::OUTPUT_BOUNDS));
            }

            proptest! {
                #![proptest_config(ProptestConfig::with_cases(3000))]
                #[test]
                fn any_numbers_give_the_same_bits(
                    row in proptest::array::uniform::<_, { code::N_FEATURES }>(prop_oneof![
                        1 => proptest::num::f32::ANY,
                        1 => proptest::sample::select(vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.0, 0.0, f32::MIN_POSITIVE, 1e-45]),
                        4 => -4.0f32..4.0,
                    ])
                ) {
                    prop_assert_eq!(bits(&code::outputs(&row)), bits(&arrays::outputs(&row)));
                }
            }
        }
    };
}

same_bits!(binary, binary_code, binary_arrays);
same_bits!(multiclass, multiclass_code, multiclass_arrays);
same_bits!(regression, regression_code, regression_arrays);

// The compiler itself evaluates the model here, with its own software floating point.
const MULTICLASS_AT_COMPILE_TIME: [f32; multiclass_code::N_OUTPUTS] = multiclass_code::outputs(&[0.25; multiclass_code::N_FEATURES]);
const REGRESSION_AT_COMPILE_TIME: [f32; regression_code::N_OUTPUTS] = regression_code::outputs(&[0.25; regression_code::N_FEATURES]);

#[test]
fn the_compiler_computes_what_the_program_computes() {
    let bits = |o: &[f32]| o.iter().map(|v| v.to_bits()).collect::<Vec<u32>>();

    let row = std::hint::black_box([0.25; multiclass_code::N_FEATURES]);
    assert_eq!(bits(&MULTICLASS_AT_COMPILE_TIME), bits(&multiclass_code::outputs(&row)));
    assert_eq!(bits(&MULTICLASS_AT_COMPILE_TIME), bits(&multiclass_arrays::outputs(&row)));

    let row = std::hint::black_box([0.25; regression_code::N_FEATURES]);
    assert_eq!(bits(&REGRESSION_AT_COMPILE_TIME), bits(&regression_code::outputs(&row)));
    assert_eq!(bits(&REGRESSION_AT_COMPILE_TIME), bits(&regression_arrays::outputs(&row)));
}
