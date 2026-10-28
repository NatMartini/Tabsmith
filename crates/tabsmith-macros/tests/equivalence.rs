//! Every model that `model!` compiles predicts exactly what `tabsmith-runtime`
//! predicts, bit for bit: on the bundle's examples and on random records. A random
//! record that one of them refuses, the other refuses too.
//!
//! The random records come from each model's `FIELDS`: values seen in training, but
//! also missing values, missing-value markers, unseen categories, NaN, infinity,
//! numbers too large to standardise, and values of the wrong type.

use std::sync::OnceLock;

use proptest::prelude::*;
use proptest::sample::select;
use tabsmith_embedded::{FieldInfo, FieldKind, FieldValue};
use tabsmith_runtime::{LoadOptions, Outcome, Prediction, Predictor, Record, Value};

tabsmith_macros::model!(binary_mlp, "../../fixtures/bundles/binary-mlp");
tabsmith_macros::model!(binary_gbdt, "../../fixtures/bundles/binary-gbdt");
tabsmith_macros::model!(multiclass_linear, "../../fixtures/bundles/multiclass-linear");
tabsmith_macros::model!(multiclass_gbdt, "../../fixtures/bundles/multiclass-gbdt");
tabsmith_macros::model!(regression_mlp, "../../fixtures/bundles/regression-mlp");
tabsmith_macros::model!(regression_gbdt, "../../fixtures/bundles/regression-gbdt");
tabsmith_macros::model!(policies_linear, "../../fixtures/bundles/policies-linear");

fn runtime(name: &str) -> Predictor {
    let dir = format!("{}/../../fixtures/bundles/{name}", env!("CARGO_MANIFEST_DIR"));
    Predictor::load(dir, LoadOptions::cpu()).unwrap()
}

/// A prediction as the label and the bits of every probability (or of the value).
#[derive(Debug, PartialEq)]
struct Bits {
    label: String,
    numbers: Vec<u64>,
}

fn runtime_bits(p: &Prediction) -> Bits {
    let numbers = match &p.outcome {
        Outcome::Binary { probability, .. } => vec![probability.to_bits()],
        Outcome::Multiclass { probabilities, .. } => probabilities.iter().map(|v| v.to_bits()).collect(),
        Outcome::Regression { value } => vec![value.to_bits()],
    };
    Bits {
        label: p.label.clone(),
        numbers,
    }
}

/// An owned raw value (`FieldValue` borrows its text).
#[derive(Debug, Clone)]
enum Raw {
    Missing,
    Bool(bool),
    Int(i64),
    Float(f64),
    Text(String),
}

impl Raw {
    fn field(&self) -> FieldValue<'_> {
        match self {
            Raw::Missing => FieldValue::Missing,
            Raw::Bool(b) => FieldValue::Bool(*b),
            Raw::Int(i) => FieldValue::Int(*i),
            Raw::Float(f) => FieldValue::Float(*f),
            Raw::Text(s) => FieldValue::Text(s),
        }
    }

    fn value(&self) -> Value {
        match self {
            Raw::Missing => Value::Null,
            Raw::Bool(b) => Value::Bool(*b),
            Raw::Int(i) => Value::Int(*i),
            Raw::Float(f) => Value::Float(*f),
            Raw::Text(s) => Value::Text(s.clone()),
        }
    }
}

fn text(s: &str) -> Raw {
    Raw::Text(s.to_string())
}

/// Values for one field: mostly plausible ones, and every kind of awkward one.
fn raw(kind: FieldKind) -> BoxedStrategy<Raw> {
    match kind {
        FieldKind::Numeric { mean, std } => prop_oneof![
            6 => (-4.0..4.0f64).prop_map(move |z| Raw::Float(mean + z * std)),
            2 => (-3i64..=3).prop_map(move |k| Raw::Int((mean + k as f64 * std).round() as i64)),
            2 => Just(Raw::Missing),
            1 => Just(Raw::Float(f64::NAN)),
            1 => Just(Raw::Float(f64::INFINITY)),
            1 => Just(Raw::Float(-999.0)),
            1 => Just(Raw::Int(-999)),
            1 => Just(Raw::Float(1e300)),
            1 => select(vec!["NA", "", "-999", "n/a"]).prop_map(text),
            1 => Just(Raw::Bool(true)),
        ]
        .boxed(),
        FieldKind::Text { categories } => prop_oneof![
            6 => select(categories.to_vec()).prop_map(text),
            1 => Just(text("never seen")),
            1 => Just(Raw::Missing),
            1 => select(vec!["NA", ""]).prop_map(text),
            1 => Just(Raw::Int(3)),
            1 => Just(Raw::Float(f64::NAN)),
        ]
        .boxed(),
        FieldKind::Bool => prop_oneof![
            6 => any::<bool>().prop_map(Raw::Bool),
            1 => Just(Raw::Missing),
            1 => Just(text("NA")),
            1 => Just(Raw::Int(1)),
        ]
        .boxed(),
    }
}

fn record(fields: &[FieldInfo]) -> Vec<BoxedStrategy<Raw>> {
    fields.iter().map(|f| raw(f.kind)).collect()
}

fn to_record(fields: &[FieldInfo], row: &[Raw]) -> Record {
    fields.iter().zip(row).fold(Record::new(), |r, (f, v)| r.with(f.name, v.value()))
}

macro_rules! compiled_equals_runtime {
    ($test:ident, $m:ident, $bundle:literal, |$p:ident| $bits:expr) => {
        mod $test {
            use super::*;

            fn rt() -> &'static Predictor {
                static RT: OnceLock<Predictor> = OnceLock::new();
                RT.get_or_init(|| runtime($bundle))
            }

            #[test]
            fn examples_predict_as_in_the_runtime() {
                assert_eq!(super::$m::self_check(), Ok(()));
                assert_eq!(super::$m::BUNDLE_ID, rt().bundle_id());
                let records: Vec<Record> = rt()
                    .bundle()
                    .examples
                    .records
                    .iter()
                    .map(|r| Record::from_json(r).unwrap())
                    .collect();
                let expected = rt().predict(&records).unwrap();
                assert_eq!(super::$m::EXAMPLES.len(), expected.len());
                for (input, want) in super::$m::EXAMPLES.iter().zip(&expected) {
                    let $p = super::$m::predict(input).unwrap();
                    assert_eq!($bits, runtime_bits(want));
                    // The typed input and its raw values describe the same record.
                    let values = input.values();
                    assert_eq!(super::$m::Input::from_values(&values), Ok(*input));
                }
            }

            proptest! {
                #![proptest_config(ProptestConfig::with_cases(3000))]
                #[test]
                fn random_records_predict_or_fail_as_in_the_runtime(row in record(&super::$m::FIELDS)) {
                    let values: [FieldValue<'_>; super::$m::N_FIELDS] = std::array::from_fn(|i| row[i].field());
                    let compiled = super::$m::predict_values(&values);
                    let interpreted = rt().predict_one(&to_record(&super::$m::FIELDS, &row));
                    match (compiled, interpreted) {
                        (Ok($p), Ok(want)) => prop_assert_eq!($bits, runtime_bits(&want)),
                        (Err(_), Err(_)) => {}
                        (c, i) => prop_assert!(false, "compiled {:?}, runtime {:?}, record {:?}", c, i.map(|p| p.label), row),
                    }
                }
            }
        }
    };
}

compiled_equals_runtime!(binary_mlp_matches_the_runtime, binary_mlp, "binary-mlp", |p| Bits {
    label: p.label.as_str().to_string(),
    numbers: vec![p.probability.to_bits()],
});
compiled_equals_runtime!(binary_gbdt_matches_the_runtime, binary_gbdt, "binary-gbdt", |p| Bits {
    label: p.label.as_str().to_string(),
    numbers: vec![p.probability.to_bits()],
});
compiled_equals_runtime!(multiclass_linear_matches_the_runtime, multiclass_linear, "multiclass-linear", |p| {
    Bits {
        label: p.class.as_str().to_string(),
        numbers: p.probabilities.iter().map(|v| v.to_bits()).collect(),
    }
});
compiled_equals_runtime!(multiclass_gbdt_matches_the_runtime, multiclass_gbdt, "multiclass-gbdt", |p| Bits {
    label: p.class.as_str().to_string(),
    numbers: p.probabilities.iter().map(|v| v.to_bits()).collect(),
});
compiled_equals_runtime!(regression_mlp_matches_the_runtime, regression_mlp, "regression-mlp", |p| Bits {
    label: p.value.to_string(),
    numbers: vec![p.value.to_bits()],
});
compiled_equals_runtime!(regression_gbdt_matches_the_runtime, regression_gbdt, "regression-gbdt", |p| Bits {
    label: p.value.to_string(),
    numbers: vec![p.value.to_bits()],
});
compiled_equals_runtime!(policies_linear_matches_the_runtime, policies_linear, "policies-linear", |p| Bits {
    label: p.label.as_str().to_string(),
    numbers: vec![p.probability.to_bits()],
});

#[test]
fn the_policies_bundle_types_its_fields_strictly() {
    use policies_linear::{Grade, Input, InputError, Mode, Zone, predict};
    // `unknown_category = "error"`: categories never missing in training are plain
    // values, not options, and there is no `Other`, so an unseen zone cannot even be
    // written. Numbers declared categorical became text categories ("2", "3.0").
    let input = Input {
        zone: Zone::V2,
        grade: Grade::V3_0,
        flag: true,
        mode: Some(Mode::Service),
        temp: None,
        load: Some(1.4),
    };
    assert!(predict(&input).is_ok());
    let mut values = input.values();
    values[0] = FieldValue::Text("9");
    assert_eq!(Input::from_values(&values), Err(InputError::UnknownCategory { field: "zone" }));
    values[0] = FieldValue::Int(2);
    assert!(Input::from_values(&values).is_err(), "a number where the data held text");
    assert!(
        predict(&Input {
            temp: Some(f64::NAN),
            ..input
        })
        .is_err(),
        "NaN is refused (nan = error)"
    );
    assert_eq!(
        predict(&Input {
            temp: Some(-999.0),
            ..input
        }),
        predict(&Input { temp: None, ..input }),
        "-999 is a missing-value marker"
    );
}
