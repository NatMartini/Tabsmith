//! `tabsmith import` against the libraries' own outputs. `fixtures/import` holds small
//! XGBoost 3.4.1 and LightGBM 4.7.0 models, rows with missing values and the outputs the
//! libraries gave for them (made by `scratch/import/reference.py`). XGBoost's margins must
//! come out bit for bit, LightGBM's raw scores within the bound the import reports, and a
//! row must be refused exactly when it lacks a value its field refuses to go without.
//!
//! `evidence` (ignored) runs the same check over a folder of larger references, given by
//! `TABSMITH_IMPORT_EVIDENCE`, and prints the numbers of `reports/import/`.

use std::path::{Path, PathBuf};

use serde::Deserialize;
use tabsmith_runtime::{LoadOptions, Predictor, Record, Value};
use tabsmith_workflow::import::{ImportOptions, ImportReport, import_model};

#[derive(Deserialize)]
struct Reference {
    model: String,
    library: String,
    labels: Option<Vec<String>>,
    /// The rows, or the name of a file next to the reference that holds them.
    rows: serde_json::Value,
    #[serde(default)]
    margin_bits: Vec<Vec<u32>>,
    #[serde(default)]
    raw_scores: Vec<Vec<f64>>,
    #[serde(default)]
    raw_scores_f64_inputs: Vec<Vec<f64>>,
}

/// What one reference showed.
#[derive(Debug, Default)]
struct Outcome {
    name: String,
    trees: usize,
    rows: usize,
    compared: usize,
    /// Rows whose outputs equal the library's, bit for bit.
    same_bits: usize,
    refused: usize,
    /// Rows with a number beyond the f32 range: XGBoost rounds it to an infinity, LightGBM
    /// compares it as it is, the bundle refuses it.
    out_of_range: usize,
    refusing_fields: Vec<String>,
    max_difference: f64,
    bound: f64,
    /// LightGBM given the rows as they are (not rounded to f32): rows beyond the bound.
    beyond_bound_f64_inputs: usize,
    max_difference_f64_inputs: f64,
}

fn record(row: &serde_json::Map<String, serde_json::Value>) -> Record {
    let mut r = Record::new();
    for (k, v) in row {
        match v.as_f64() {
            Some(x) => r.insert(k.clone(), x),
            None => r.insert(k.clone(), Value::Null),
        }
    }
    r
}

fn check(reference: &Path, out: &Path) -> Outcome {
    let dir = reference.parent().unwrap();
    let r: Reference = serde_json::from_str(&std::fs::read_to_string(reference).unwrap()).unwrap();
    let rows: Vec<serde_json::Map<String, serde_json::Value>> = match &r.rows {
        serde_json::Value::String(file) => serde_json::from_str(&std::fs::read_to_string(dir.join(file)).unwrap()).unwrap(),
        v => serde_json::from_value(v.clone()).unwrap(),
    };
    let report: ImportReport = import_model(
        &dir.join(&r.model),
        &ImportOptions {
            out: out.to_path_buf(),
            labels: r.labels.clone().unwrap_or_default(),
            ..Default::default()
        },
    )
    .unwrap_or_else(|e| panic!("{}: {e:?}", r.model));
    assert!(report.check.passed, "{}: self-check", r.model);
    assert_eq!(report.model.library.as_str(), r.library);
    let p = Predictor::load(out, LoadOptions::cpu()).unwrap();
    let refusing: Vec<String> = p
        .input_schema()
        .fields
        .iter()
        .filter(|f| !f.nullable)
        .map(|f| f.name.clone())
        .collect();
    let refuses = |row: &serde_json::Map<String, serde_json::Value>| refusing.iter().any(|f| row.get(f).is_some_and(|v| v.is_null()));
    let too_large =
        |row: &serde_json::Map<String, serde_json::Value>| row.values().any(|v| v.as_f64().is_some_and(|x| (x as f32).is_infinite()));
    let mut o = Outcome {
        name: r.model.clone(),
        trees: report.model.trees,
        rows: rows.len(),
        refusing_fields: refusing.clone(),
        bound: report.model.max_output_difference,
        ..Default::default()
    };
    // Rows with a missing value where it is refused, or a number beyond f32: each an error.
    for row in rows.iter().filter(|row| refuses(row) || too_large(row)) {
        let result = p.records_table(&[record(row)], &[]).and_then(|t| p.outputs(&t));
        let e = result.expect_err("a refused row");
        assert_eq!(e.code, tabsmith_contracts::ErrorCode::InvalidValue, "{}: {e:?}", r.model);
        if refuses(row) {
            o.refused += 1;
        } else {
            o.out_of_range += 1;
        }
    }
    // The others in one batch, against the library's outputs.
    let kept: Vec<usize> = (0..rows.len()).filter(|&i| !refuses(&rows[i]) && !too_large(&rows[i])).collect();
    let records: Vec<Record> = kept.iter().map(|&i| record(&rows[i])).collect();
    let outputs = p.outputs(&p.records_table(&records, &[]).unwrap()).unwrap();
    let k = outputs.cols;
    for (n, &i) in kept.iter().enumerate() {
        let got = &outputs.data[n * k..(n + 1) * k];
        o.compared += 1;
        if r.library == "xgboost" {
            let bits: Vec<u32> = got.iter().map(|v| v.to_bits()).collect();
            if bits == r.margin_bits[i] {
                o.same_bits += 1;
            }
            continue;
        }
        let diff = |want: &[f64]| got.iter().zip(want).map(|(&g, &w)| (f64::from(g) - w).abs()).fold(0.0, f64::max);
        let d = diff(&r.raw_scores[i]);
        o.max_difference = o.max_difference.max(d);
        if got.iter().zip(&r.raw_scores[i]).all(|(&g, &w)| f64::from(g) == w) {
            o.same_bits += 1;
        }
        if let Some(want) = r.raw_scores_f64_inputs.get(i) {
            let d = diff(want);
            o.max_difference_f64_inputs = o.max_difference_f64_inputs.max(d);
            o.beyond_bound_f64_inputs += usize::from(d > o.bound);
        }
    }
    o
}

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/import")
}

fn run(name: &str) -> Outcome {
    let tmp = tempfile::tempdir().unwrap();
    check(&fixtures().join(format!("{name}.reference.json")), &tmp.path().join("bundle"))
}

#[test]
fn xgboost_margins_come_out_bit_for_bit() {
    for task in ["binary", "multiclass", "regression"] {
        let o = run(&format!("xgboost-{task}"));
        assert_eq!(o.same_bits, o.compared, "{o:?}");
        assert!(o.compared + o.refused + o.out_of_range == o.rows && o.compared >= 60, "{o:?}");
    }
}

#[test]
fn lightgbm_raw_scores_stay_within_the_reported_bound() {
    for task in ["binary", "multiclass", "regression"] {
        let o = run(&format!("lightgbm-{task}"));
        assert!(o.bound > 0.0 && o.max_difference <= o.bound, "{o:?}");
        assert!(o.compared + o.refused + o.out_of_range == o.rows && o.compared >= 60, "{o:?}");
    }
}

#[test]
fn a_missing_value_is_imputed_where_one_value_takes_its_way_and_refused_elsewhere() {
    // humidity and voltage_drop had missing values in training: each library learned a
    // direction per split, left at some and right at others. The other fields had none, and
    // every split on one of them sends a missing value the same way.
    let o = run("xgboost-binary");
    assert_eq!(o.refusing_fields, ["humidity", "voltage_drop"]);
    assert!(o.refused > 0 && o.refused < o.rows, "{o:?}");
    // Rows with a missing value in an accepting field are among the bit-for-bit ones.
    let r: Reference = serde_json::from_str(&std::fs::read_to_string(fixtures().join("xgboost-binary.reference.json")).unwrap()).unwrap();
    let rows: Vec<serde_json::Map<String, serde_json::Value>> = serde_json::from_value(r.rows).unwrap();
    let imputed = rows
        .iter()
        .filter(|row| row.iter().any(|(k, v)| v.is_null() && k != "humidity" && k != "voltage_drop"))
        .filter(|row| !row["humidity"].is_null() && !row["voltage_drop"].is_null())
        .count();
    assert!(imputed >= 10, "{imputed} rows with an imputed missing value");
}

#[test]
fn an_imported_bundle_is_the_same_bundle_twice() {
    let tmp = tempfile::tempdir().unwrap();
    let opts = ImportOptions {
        out: tmp.path().join("b"),
        ..Default::default()
    };
    let model = fixtures().join("xgboost-regression.json");
    let a = import_model(&model, &opts).unwrap();
    let b = import_model(&model, &opts).unwrap();
    assert!(!a.already_existed && b.already_existed);
    assert_eq!(a.manifest.bundle_id, b.manifest.bundle_id);
    // Another name is another bundle; the directory is taken.
    let other = ImportOptions {
        name: Some("other".into()),
        ..opts
    };
    assert_eq!(
        import_model(&model, &other).unwrap_err().code,
        tabsmith_contracts::ErrorCode::BundleExists
    );
}

#[test]
fn options_and_refusals() {
    let tmp = tempfile::tempdir().unwrap();
    let out = |n: &str| tmp.path().join(n);
    let model = fixtures().join("xgboost-multiclass.json");
    let labels = |l: &[&str]| ImportOptions {
        out: out("m"),
        labels: l.iter().map(|s| s.to_string()).collect(),
        ..Default::default()
    };
    let e = import_model(&model, &labels(&["a", "b"])).unwrap_err();
    assert_eq!(e.code, tabsmith_contracts::ErrorCode::ConfigInvalid);
    let r = import_model(&model, &labels(&["bearing", "none", "overheat", "seal"])).unwrap();
    let p = Predictor::load(&r.path, LoadOptions::cpu()).unwrap();
    assert_eq!(p.output_schema().classes, ["bearing", "none", "overheat", "seal"]);
    // Feature names: the right count, no duplicates.
    let features = |f: Vec<String>| ImportOptions {
        out: out("f"),
        features: f,
        ..Default::default()
    };
    let e = import_model(&fixtures().join("xgboost-binary.json"), &features(vec!["a".into(); 15])).unwrap_err();
    assert_eq!(e.code, tabsmith_contracts::ErrorCode::ConfigInvalid);
    let names: Vec<String> = (0..15).map(|i| format!("x{i}")).collect();
    let r = import_model(&fixtures().join("xgboost-binary.json"), &features(names)).unwrap();
    assert_eq!(
        Predictor::load(&r.path, LoadOptions::cpu()).unwrap().input_schema().fields[3].name,
        "x3"
    );
    // Not a model at all.
    let junk = tmp.path().join("junk.json");
    std::fs::write(&junk, "{\"hello\": 1}").unwrap();
    let e = import_model(&junk, &features(Vec::new())).unwrap_err();
    assert_eq!(e.code, tabsmith_contracts::ErrorCode::ImportInvalid);
    // A LightGBM model with a log link.
    let text = std::fs::read_to_string(fixtures().join("lightgbm-regression.txt"))
        .unwrap()
        .replace("objective=regression", "objective=poisson");
    let poisson = tmp.path().join("poisson.txt");
    std::fs::write(&poisson, text).unwrap();
    let e = import_model(&poisson, &features(Vec::new())).unwrap_err();
    assert_eq!(e.code, tabsmith_contracts::ErrorCode::ImportUnsupported);
}

/// `TABSMITH_IMPORT_EVIDENCE=<folder> cargo test -p tabsmith-workflow --test import --release -- --ignored --nocapture`
#[test]
#[ignore = "needs the evidence folder of scratch/import/reference.py"]
fn evidence() {
    let dir = PathBuf::from(std::env::var("TABSMITH_IMPORT_EVIDENCE").expect("TABSMITH_IMPORT_EVIDENCE"));
    let mut refs: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.to_string_lossy().ends_with(".reference.json"))
        .collect();
    refs.sort();
    let tmp = tempfile::tempdir().unwrap();
    println!(
        "| model | trees | rows | compared | same bits | refused: missing | refused: beyond f32 | max difference | bound | f64 inputs: beyond bound, max difference |"
    );
    println!("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|");
    for (i, r) in refs.iter().enumerate() {
        let o = check(r, &tmp.path().join(format!("b{i}")));
        println!(
            "| {} | {} | {} | {} | {} | {} | {} | {:.2e} | {:.2e} | {}, {:.2e} |",
            o.name,
            o.trees,
            o.rows,
            o.compared,
            o.same_bits,
            o.refused,
            o.out_of_range,
            o.max_difference,
            o.bound,
            o.beyond_bound_f64_inputs,
            o.max_difference_f64_inputs
        );
        if o.name.starts_with("xgboost") {
            assert_eq!(o.same_bits, o.compared, "{o:?}");
        } else {
            assert!(o.max_difference <= o.bound, "{o:?}");
        }
    }
}

/// The parsers answer every model file, damaged or not: an error or a model, never a
/// panic, and never a size taken from a count the file declares but does not back.
mod damaged_models {
    use proptest::prelude::*;
    use tabsmith_workflow::import::describe;

    const NUMBERS: [&str; 14] = [
        "0",
        "-1",
        "1",
        "3",
        "0.5",
        "1e400",
        "-1e400",
        "nan",
        "inf",
        "99999999999999999999",
        "18446744073709551615",
        "9223372036854775807",
        "-9223372036854775808",
        "4294967295",
    ];

    fn models() -> Vec<String> {
        [
            "xgboost-binary.json",
            "xgboost-multiclass.json",
            "xgboost-regression.json",
            "lightgbm-binary.txt",
            "lightgbm-multiclass.txt",
            "lightgbm-regression.txt",
        ]
        .iter()
        .map(|f| std::fs::read_to_string(super::fixtures().join(f)).unwrap())
        .collect()
    }

    /// The byte ranges of the numbers in `text`: maximal runs of digits, signs, dots and
    /// exponents that hold a digit.
    fn numbers(text: &str) -> Vec<(usize, usize)> {
        let b = text.as_bytes();
        let mut out = Vec::new();
        let mut i = 0;
        while i < b.len() {
            let is_num = |c: u8| c.is_ascii_digit() || matches!(c, b'.' | b'-' | b'+' | b'e' | b'E');
            if is_num(b[i]) {
                let start = i;
                while i < b.len() && is_num(b[i]) {
                    i += 1;
                }
                if b[start..i].iter().any(u8::is_ascii_digit) {
                    out.push((start, i));
                }
            } else {
                i += 1;
            }
        }
        out
    }

    /// Applies one mutation: cut a range, replace a number, drop or double a line, truncate.
    fn mutate(text: &str, kind: u8, a: usize, b: usize, n: usize) -> String {
        if text.is_empty() {
            return text.to_string();
        }
        match kind {
            0 => {
                let (x, y) = (a % text.len(), b % text.len());
                let (x, y) = (x.min(y), x.max(y).min(x + 64));
                let (x, y) = (floor_char(text, x), floor_char(text, y));
                format!("{}{}", &text[..x], &text[y..])
            }
            1 => {
                let spots = numbers(text);
                match spots.get(a % spots.len().max(1)) {
                    Some(&(x, y)) => format!("{}{}{}", &text[..x], NUMBERS[n % NUMBERS.len()], &text[y..]),
                    None => text.to_string(),
                }
            }
            2 | 3 => {
                let mut lines: Vec<&str> = text.split_inclusive('\n').collect();
                let i = a % lines.len();
                if kind == 2 {
                    lines.remove(i);
                } else {
                    lines.insert(i, lines[i]);
                }
                lines.concat()
            }
            _ => text[..floor_char(text, a % text.len())].to_string(),
        }
    }

    fn floor_char(text: &str, mut i: usize) -> usize {
        while !text.is_char_boundary(i) {
            i -= 1;
        }
        i
    }

    #[test]
    fn the_fixtures_themselves_are_read() {
        for m in models() {
            let d = describe(&m).unwrap();
            assert!(d.trees >= 20 && d.features == 15, "{d:?}");
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(400))]
        #[test]
        fn a_damaged_model_gets_an_answer(
            which in 0usize..6,
            steps in prop::collection::vec((0u8..5, any::<usize>(), any::<usize>(), any::<usize>()), 1..=4),
        ) {
            let models = models();
            let mut text = models[which].clone();
            for (kind, a, b, n) in steps {
                text = mutate(&text, kind, a, b, n);
            }
            let _ = describe(&text);
        }
    }
}
