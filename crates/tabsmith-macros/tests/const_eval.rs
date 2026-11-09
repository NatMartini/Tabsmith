//! The compiler predicts what the program predicts, bit for bit: every model, both forms of
//! trees, on rows the compiler makes up itself, not only on the bundle's examples (which the
//! generated code already checks while it compiles). `tests/equivalence.rs` shows that the
//! program predicts what `tabsmith-runtime` predicts, so the compiler does too.

tabsmith_macros::model!(binary_mlp, "../../fixtures/bundles/binary-mlp");
tabsmith_macros::model!(binary_gbdt, "../../fixtures/bundles/binary-gbdt");
tabsmith_macros::model!(binary_gbdt_arrays, "../../fixtures/bundles/binary-gbdt", tree_arrays);
tabsmith_macros::model!(multiclass_linear, "../../fixtures/bundles/multiclass-linear");
tabsmith_macros::model!(multiclass_gbdt, "../../fixtures/bundles/multiclass-gbdt");
tabsmith_macros::model!(multiclass_gbdt_arrays, "../../fixtures/bundles/multiclass-gbdt", tree_arrays);
tabsmith_macros::model!(regression_mlp, "../../fixtures/bundles/regression-mlp");
tabsmith_macros::model!(regression_gbdt, "../../fixtures/bundles/regression-gbdt");
tabsmith_macros::model!(regression_gbdt_arrays, "../../fixtures/bundles/regression-gbdt", tree_arrays);
tabsmith_macros::model!(policies_linear, "../../fixtures/bundles/policies-linear");
tabsmith_macros::model!(imported_xgboost, "../../fixtures/bundles/imported-xgboost-binary");
tabsmith_macros::model!(imported_lightgbm, "../../fixtures/bundles/imported-lightgbm-multiclass");

/// Rows the compiler computes: xorshift values in [-3, 3), a few of them NaN or infinite.
const fn rows<const N: usize, const K: usize>(seed: u64) -> [[f32; N]; K] {
    let mut out = [[0f32; N]; K];
    let mut s = seed;
    let mut r = 0;
    while r < K {
        let mut j = 0;
        while j < N {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            out[r][j] = match s % 97 {
                0 => f32::NAN,
                1 => f32::INFINITY,
                2 => f32::NEG_INFINITY,
                _ => ((s >> 40) as f32 / 16_777_216.0) * 6.0 - 3.0,
            };
            j += 1;
        }
        r += 1;
    }
    out
}

/// The bits of a prediction the tests compare: a difference in any bit changes them.
macro_rules! prediction_bits {
    (binary, $p:ident) => {
        $p.probability.to_bits() ^ $p.label as u64
    };
    (multiclass, $p:ident) => {
        super::every_bit(&$p.probabilities, $p.class as usize)
    };
    (regression, $p:ident) => {
        $p.value.to_bits()
    };
}

macro_rules! same_bits {
    ($test:ident, $m:ident, $task:ident) => {
        mod $test {
            use super::$m as m;

            const fn bits(p: &m::Prediction) -> u64 {
                prediction_bits!($task, p)
            }

            const K: usize = 48;
            const ROWS: [[f32; m::N_FEATURES]; K] = super::rows(0x9E37_79B9_7F4A_7C15 ^ m::N_FEATURES as u64);

            // What the compiler computes: the outputs of every row, and the decision on them.
            const AT_COMPILE_TIME: [([u32; m::N_OUTPUTS], u64); K] = {
                let mut out = [([0u32; m::N_OUTPUTS], 0u64); K];
                let mut r = 0;
                while r < K {
                    let z = m::outputs(&ROWS[r]);
                    let mut o = 0;
                    while o < m::N_OUTPUTS {
                        out[r].0[o] = z[o].to_bits();
                        o += 1;
                    }
                    out[r].1 = bits(&m::decode(&z));
                    r += 1;
                }
                out
            };

            // Every example, predicted by the compiler.
            const EXAMPLES_AT_COMPILE_TIME: [u64; m::EXAMPLES.len()] = {
                let mut out = [0u64; m::EXAMPLES.len()];
                let mut i = 0;
                while i < out.len() {
                    out[i] = match m::predict(&m::EXAMPLES[i]) {
                        Ok(p) => bits(&p),
                        Err(_) => panic!("an example is refused"),
                    };
                    i += 1;
                }
                out
            };

            #[test]
            fn the_compiler_and_the_program_compute_the_same_bits() {
                for (r, row) in ROWS.iter().enumerate() {
                    let z = m::outputs(std::hint::black_box(row));
                    let got: Vec<u32> = z.iter().map(|v| v.to_bits()).collect();
                    assert_eq!(got, AT_COMPILE_TIME[r].0, "outputs of row {r}");
                    assert_eq!(bits(&m::decode(&z)), AT_COMPILE_TIME[r].1, "decision on row {r}");
                }
                for (i, input) in m::EXAMPLES.iter().enumerate() {
                    let p = m::predict(std::hint::black_box(input)).unwrap();
                    assert_eq!(bits(&p), EXAMPLES_AT_COMPILE_TIME[i], "example {i}");
                }
            }
        }
    };
}

/// Every class's probability and the class, mixed into one number.
const fn every_bit(probabilities: &[f64], class: usize) -> u64 {
    let mut h = class as u64;
    let mut k = 0;
    while k < probabilities.len() {
        h = h.rotate_left(13) ^ probabilities[k].to_bits();
        k += 1;
    }
    h
}

same_bits!(binary_mlp_bits, binary_mlp, binary);
same_bits!(binary_gbdt_bits, binary_gbdt, binary);
same_bits!(binary_gbdt_arrays_bits, binary_gbdt_arrays, binary);
same_bits!(multiclass_linear_bits, multiclass_linear, multiclass);
same_bits!(multiclass_gbdt_bits, multiclass_gbdt, multiclass);
same_bits!(multiclass_gbdt_arrays_bits, multiclass_gbdt_arrays, multiclass);
same_bits!(regression_mlp_bits, regression_mlp, regression);
same_bits!(regression_gbdt_bits, regression_gbdt, regression);
same_bits!(regression_gbdt_arrays_bits, regression_gbdt_arrays, regression);
same_bits!(policies_linear_bits, policies_linear, binary);
same_bits!(imported_xgboost_bits, imported_xgboost, binary);
same_bits!(imported_lightgbm_bits, imported_lightgbm, multiclass);

// The derived bounds come from the compiler too, with the same kernels as `decode`.
const _: () = {
    assert!(binary_gbdt::PROBABILITY_BOUNDS[0] > 0.0 && binary_gbdt::PROBABILITY_BOUNDS[1] < 1.0);
    assert!(regression_gbdt::VALUE_BOUNDS[0] < regression_gbdt::VALUE_BOUNDS[1]);
};
