//! Predictions are the same bits on every platform. For each fixture bundle, the hash of
//! the predictions for 20,000 pseudo-random records, from the compiled model and from
//! the runtime, must equal the value recorded on Windows x86_64. CI runs this test on
//! Linux x86_64, macOS arm64 and Windows x86_64.
//!
//! It holds because a prediction is plain IEEE 754 arithmetic in a fixed order, with
//! `exp` from `libm` (pure Rust) instead of the platform's C library.

mod sample;

use sample::{FNV_OFFSET, Rng, fnv, record, row, runtime_numbers};
use tabsmith_embedded::FieldValue;
use tabsmith_runtime::{LoadOptions, Predictor};

tabsmith_macros::model!(binary_mlp, "../../fixtures/bundles/binary-mlp");
tabsmith_macros::model!(binary_gbdt, "../../fixtures/bundles/binary-gbdt");
tabsmith_macros::model!(multiclass_linear, "../../fixtures/bundles/multiclass-linear");
tabsmith_macros::model!(multiclass_gbdt, "../../fixtures/bundles/multiclass-gbdt");
tabsmith_macros::model!(regression_mlp, "../../fixtures/bundles/regression-mlp");
tabsmith_macros::model!(regression_gbdt, "../../fixtures/bundles/regression-gbdt");
tabsmith_macros::model!(policies_linear, "../../fixtures/bundles/policies-linear");

const RECORDS: usize = 20_000;

/// Hashed in place of the numbers when both refuse a record.
const REFUSED: u64 = u64::MAX;

/// One test per line: `test: module, bundle, recorded hash, |prediction| its numbers;`.
macro_rules! portable {
    ($($test:ident: $m:ident, $bundle:literal, $expected:literal, |$p:ident| $numbers:expr;)*) => {$(
        #[test]
        fn $test() {
            let dir = format!("{}/../../fixtures/bundles/{}", env!("CARGO_MANIFEST_DIR"), $bundle);
            let rt = Predictor::load(dir, LoadOptions::cpu()).unwrap();
            let mut rng = Rng(7);
            let (mut compiled, mut runtime) = (FNV_OFFSET, FNV_OFFSET);
            for _ in 0..RECORDS {
                let values: [FieldValue<'_>; $m::N_FIELDS] = row(&$m::FIELDS, &mut rng).try_into().unwrap();
                match ($m::predict_values(&values), rt.predict_one(&record(&$m::FIELDS, &values))) {
                    (Ok($p), Ok(want)) => {
                        for v in $numbers {
                            fnv(&mut compiled, v.to_bits());
                        }
                        for v in runtime_numbers(&want.outcome) {
                            fnv(&mut runtime, v.to_bits());
                        }
                    }
                    (Err(_), Err(_)) => {
                        fnv(&mut compiled, REFUSED);
                        fnv(&mut runtime, REFUSED);
                    }
                    (c, r) => panic!("{}: compiled ok = {}, runtime ok = {}", $bundle, c.is_ok(), r.is_ok()),
                }
            }
            assert_eq!(compiled, runtime, "{}: the compiled model and the runtime disagree", $bundle);
            assert_eq!(
                compiled, $expected,
                "{}: the predictions hash to {compiled:#018x}, not to the value recorded on Windows x86_64",
                $bundle
            );
        }
    )*};
}

portable! {
    binary_mlp_is_portable: binary_mlp, "binary-mlp", 0x701e_62b9_c428_2f3e, |p| [p.probability];
    binary_gbdt_is_portable: binary_gbdt, "binary-gbdt", 0xb2c0_bcb1_4f60_92bb, |p| [p.probability];
    multiclass_linear_is_portable: multiclass_linear, "multiclass-linear", 0x0a62_55b8_8476_3cee, |p| p.probabilities;
    multiclass_gbdt_is_portable: multiclass_gbdt, "multiclass-gbdt", 0x025d_9931_9064_a824, |p| p.probabilities;
    regression_mlp_is_portable: regression_mlp, "regression-mlp", 0xd38d_3051_1c7d_e789, |p| [p.value];
    regression_gbdt_is_portable: regression_gbdt, "regression-gbdt", 0x62e5_a5f8_89b1_be08, |p| [p.value];
    policies_linear_is_portable: policies_linear, "policies-linear", 0x1a0c_ca3d_1d7d_bb76, |p| [p.probability];
}
