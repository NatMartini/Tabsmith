//! A large batch runs on several threads and gives what one thread gives: the same
//! predictions in the same order, and the same error when a record is wrong.
//! Every bundle kind is covered, because the outputs and their decoding differ by task.

use tabsmith_runtime::{ErrorCode, LoadOptions, Predictor, Record, Value};

fn load(bundle: &str, threads: usize) -> Predictor {
    let dir = format!("{}/../../fixtures/bundles/{bundle}", env!("CARGO_MANIFEST_DIR"));
    Predictor::load(dir, LoadOptions::cpu().with_threads(threads)).unwrap()
}

/// `n` records: the bundle's examples with their numbers moved a little, row by row.
fn records(bundle: &str, n: usize) -> Vec<Record> {
    let path = format!("{}/../../fixtures/bundles/{bundle}/examples/input.json", env!("CARGO_MANIFEST_DIR"));
    let text = std::fs::read_to_string(path).unwrap();
    let examples: serde_json::Value = serde_json::from_str(&text).unwrap();
    let examples = examples["records"].as_array().unwrap();
    (0..n)
        .map(|i| {
            let mut row = examples[i % examples.len()].clone();
            for (k, v) in row.as_object_mut().unwrap().iter_mut() {
                if let Some(x) = v.as_f64() {
                    let shift = ((i * 2_654_435_761 + k.len()) % 1000) as f64 / 500.0;
                    *v = serde_json::json!(x * (0.5 + shift));
                }
            }
            Record::from_json(&row).unwrap()
        })
        .collect()
}

const BUNDLES: [&str; 7] = [
    "binary-mlp",
    "binary-gbdt",
    "multiclass-linear",
    "multiclass-gbdt",
    "regression-mlp",
    "regression-gbdt",
    "policies-linear",
];

#[test]
fn the_predictions_do_not_depend_on_the_number_of_threads() {
    for bundle in BUNDLES {
        let rows = records(bundle, 5000);
        let one = load(bundle, 1);
        let reference = one.predict(&rows).unwrap();
        assert_eq!(reference.len(), rows.len());
        let table = one.records_table(&rows, &[]).unwrap();
        for threads in [2, 7, 0] {
            let p = load(bundle, threads);
            assert_eq!(p.predict(&rows).unwrap(), reference, "{bundle}: records, {threads} threads");
            assert_eq!(p.predict_table(&table).unwrap(), reference, "{bundle}: table, {threads} threads");
            let (a, b) = (one.outputs(&table).unwrap(), p.outputs(&table).unwrap());
            let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
            assert_eq!((a.rows, a.cols), (b.rows, b.cols));
            assert_eq!(bits(&a.data), bits(&b.data), "{bundle}: outputs, {threads} threads");
        }
    }
}

#[test]
fn a_batch_predicts_what_single_records_predict() {
    for bundle in BUNDLES {
        let rows = records(bundle, 4200);
        let p = load(bundle, 0);
        let batch = p.predict(&rows).unwrap();
        for i in [0, 1, 1023, 1024, 2100, 4199] {
            assert_eq!(batch[i], p.predict_one(&rows[i]).unwrap(), "{bundle}: row {i}");
        }
    }
}

#[test]
fn a_table_from_records_carries_the_keep_columns() {
    let bundle = "binary-gbdt";
    let p = load(bundle, 1);
    let rows: Vec<Record> = records(bundle, 3).into_iter().map(|r| r.with("id", "row")).collect();
    let keep = ["id".to_string()];
    let table = p.records_table(&rows, &keep).unwrap();
    let plain = p.records_table(&records(bundle, 3), &[]).unwrap();
    assert_eq!(table.columns().len(), plain.columns().len() + 1);
    assert!(table.names().contains(&"id"));
    for name in plain.names() {
        for row in 0..3 {
            assert_eq!(
                table.column(name).unwrap().get(row),
                plain.column(name).unwrap().get(row),
                "{name}, row {row}"
            );
        }
    }
    assert_eq!(p.predict_table(&table).unwrap(), p.predict_table(&plain).unwrap());
}

#[test]
fn the_error_is_the_one_a_single_thread_reports() {
    let bundle = "binary-gbdt";
    let rows = records(bundle, 5000);

    // A wrong type deep in a later piece of the batch.
    let mut wrong_type = rows.clone();
    wrong_type[4500] = wrong_type[4500].clone().with("cycles", Value::Text("many".into()));
    let one = load(bundle, 1).predict(&wrong_type).unwrap_err();
    let many = load(bundle, 8).predict(&wrong_type).unwrap_err();
    assert_eq!(one.code, ErrorCode::SchemaTypeMismatch);
    assert_eq!(many.code, one.code);
    assert_eq!(many.message, one.message, "the message names the row of the whole batch");
    assert!(many.message.contains("4501"), "{}", many.message);
    assert_eq!(many.field, one.field);

    // A missing field deep in a later piece of the batch.
    let field = load(bundle, 1)
        .input_schema()
        .fields
        .iter()
        .find(|f| f.required)
        .unwrap()
        .name
        .clone();
    let mut without = Record::new();
    for name in rows[4700].names().filter(|n| *n != field) {
        without.insert(name, rows[4700].get(name).unwrap().clone());
    }
    let mut missing = rows.clone();
    missing[4700] = without;
    let one = load(bundle, 1).predict(&missing).unwrap_err();
    let many = load(bundle, 8).predict(&missing).unwrap_err();
    assert_eq!(one.code, ErrorCode::SchemaMissingColumn);
    assert_eq!(many.code, one.code);
    assert_eq!(many.message, one.message);
    assert!(many.message.contains(&field), "{}", many.message);
}
