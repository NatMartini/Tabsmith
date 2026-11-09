//! A model imported from XGBoost and compiled by `model!` computes XGBoost's own margins,
//! bit for bit, with its trees as generated code and as arrays, and refuses a missing value
//! where the bundle refuses it. The rows and XGBoost 3.4.1's margins for them are in
//! `fixtures/import/xgboost-binary.reference.json`.

use tabsmith_embedded::{FieldValue, InputError};

tabsmith_macros::model!(code, "../../fixtures/bundles/imported-xgboost-binary", tree_code);
tabsmith_macros::model!(arrays, "../../fixtures/bundles/imported-xgboost-binary", tree_arrays);

#[test]
fn compiled_trees_compute_xgboosts_margins() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../fixtures/import/xgboost-binary.reference.json");
    let reference: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let rows = reference["rows"].as_array().unwrap();
    let margins = reference["margin_bits"].as_array().unwrap();
    let (mut same, mut refused) = (0, 0);
    for (row, margin) in rows.iter().zip(margins) {
        let values: [FieldValue<'_>; code::N_FIELDS] = std::array::from_fn(|i| match row[code::FIELDS[i].name].as_f64() {
            Some(x) => FieldValue::Float(x),
            None => FieldValue::Missing,
        });
        let want = u32::try_from(margin[0].as_u64().unwrap()).unwrap();
        match code::Input::from_values(&values).and_then(|input| code::features(&input)) {
            Ok(x) => {
                assert_eq!(code::outputs(&x)[0].to_bits(), want, "{row}");
                assert_eq!(arrays::outputs(&x)[0].to_bits(), want, "{row}");
                same += 1;
            }
            Err(e) => {
                assert!(
                    matches!(
                        e,
                        InputError::Missing {
                            field: "humidity" | "voltage_drop"
                        }
                    ),
                    "{e}"
                );
                refused += 1;
            }
        }
    }
    assert!(same >= 60 && refused > 0, "{same} margins, {refused} refused");
}

/// The compiler adds the leaves in f32 too.
const AT_COMPILE_TIME: [f32; 1] = code::outputs(&[1.0; code::N_FEATURES]);

#[test]
fn the_compiler_adds_in_f32_like_the_program() {
    let row = std::hint::black_box([1.0; code::N_FEATURES]);
    assert_eq!(AT_COMPILE_TIME[0].to_bits(), code::outputs(&row)[0].to_bits());
    assert_eq!(AT_COMPILE_TIME[0].to_bits(), arrays::outputs(&row)[0].to_bits());
}
