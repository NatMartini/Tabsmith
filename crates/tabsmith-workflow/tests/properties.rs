//! Property-based tests: invariants that must hold for every input, not only for the
//! examples in the unit tests. More cases: `PROPTEST_CASES=2048 cargo test --test properties`.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;
use std::sync::OnceLock;

use proptest::prelude::*;
use proptest::sample::Index;
use tabsmith_contracts::ErrorCode;
use tabsmith_contracts::config::{Candidate, DataConfig, SplitStrategy};
use tabsmith_contracts::data::SplitName;
use tabsmith_contracts::model::ModelConfig;
use tabsmith_runtime::model::{DenseLayer, DenseModel};
use tabsmith_runtime::{Bundle, Column, ColumnData, LoadOptions, Predictor, Record, Table, Value, ValueRef};
use tabsmith_workflow::bundle::BundleOptions;
use tabsmith_workflow::data::TargetInfo;
use tabsmith_workflow::exposure::entity_key;
use tabsmith_workflow::metrics::{binary_metrics, roc_auc};
use tabsmith_workflow::pipeline::RunOptions;
use tabsmith_workflow::rng::epoch_permutation;
use tabsmith_workflow::sample::{SampleVariant, write_defects_csv};
use tabsmith_workflow::setup::{BindRequest, init_project};
use tabsmith_workflow::split::{check_group_leakage, make_split};
use tabsmith_workflow::{ConfigRequest, Workspace};

// ---------------------------------------------------------------------------
// Splits

/// A table with a 0/1 target `y`, a group column `g` and a unique numeric `x`.
fn split_table(labels: &[bool], groups: &[u8]) -> (Table, TargetInfo) {
    let n = labels.len();
    let table = Table::new(vec![
        Column {
            name: "x".into(),
            data: ColumnData::Int((0..n as i64).map(Some).collect()),
        },
        Column {
            name: "g".into(),
            data: ColumnData::Text(groups.iter().map(|g| Some(format!("dev-{g}"))).collect()),
        },
        Column {
            name: "y".into(),
            data: ColumnData::Int(labels.iter().map(|&b| Some(i64::from(b))).collect()),
        },
    ])
    .unwrap();
    let positives = labels.iter().filter(|&&b| b).count();
    let target = TargetInfo::binary("y", "1", "0", positives, n - positives);
    (table, target)
}

fn rows_with_groups() -> impl Strategy<Value = (Vec<bool>, Vec<u8>)> {
    (80usize..400, 0.12f64..0.5, 8u8..60).prop_flat_map(|(n, rate, groups)| {
        (
            prop::collection::vec(prop::bool::weighted(rate), n),
            prop::collection::vec(0..groups, n),
        )
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(96))]

    /// Every row lands in exactly one split, the counts add up, each split holds
    /// both classes, the same inputs give the same split, and a grouped split never
    /// puts one entity in two splits. Inputs that cannot be split are refused with
    /// a split error, never with a panic or a wrong split.
    #[test]
    fn splits_partition_rows_and_respect_groups((labels, groups) in rows_with_groups(), seed: u64, grouped: bool) {
        let (table, target) = split_table(&labels, &groups);
        let cfg = DataConfig {
            seed,
            split: if grouped { SplitStrategy::Group } else { SplitStrategy::Stratified },
            group_column: grouped.then(|| "g".to_string()),
            ..DataConfig::default()
        };
        match make_split("s-prop", &table, &target, &cfg) {
            Err(e) => prop_assert!(
                matches!(e.code, ErrorCode::SplitDegenerate | ErrorCode::SplitUnsupported),
                "unexpected refusal {:?}: {}", e.code, e.message
            ),
            Ok(split) => {
                let n = labels.len();
                prop_assert_eq!(split.assignment.len(), n);
                let parts = [SplitName::Train, SplitName::Valid, SplitName::Test].map(|s| split.rows(s));
                prop_assert_eq!(parts.iter().map(Vec::len).sum::<usize>(), n);
                let mut seen = vec![false; n];
                for rows in &parts {
                    prop_assert!(rows.iter().any(|&r| labels[r]) && rows.iter().any(|&r| !labels[r]), "a split lacks a class");
                    for &r in rows {
                        prop_assert!(!seen[r], "row {} is in two splits", r);
                        seen[r] = true;
                    }
                }
                let counts = [&split.train, &split.valid, &split.test];
                prop_assert_eq!(counts.iter().map(|c| c.positives).sum::<usize>(), target.positives);
                prop_assert_eq!(make_split("s-prop", &table, &target, &cfg).unwrap(), split.clone());
                if grouped {
                    prop_assert!(check_group_leakage(&table, &split).is_ok());
                    let mut split_of: BTreeMap<u8, u8> = BTreeMap::new();
                    for (r, code) in split.assignment.bytes().enumerate() {
                        let first = *split_of.entry(groups[r]).or_insert(code);
                        prop_assert_eq!(first, code, "group dev-{} spans two splits", groups[r]);
                    }
                }
            }
        }
    }

    /// Id and group keys: spacing and zero padding never make two entities out of one.
    #[test]
    fn entity_keys_are_normalised(v in 0u64..u64::MAX / 2, pad in 0usize..6, spaces in 0usize..3, text in "[ -~]{0,12}") {
        let padded = format!("{}{}{v}{}", " ".repeat(spaces), "0".repeat(pad), " ".repeat(spaces));
        prop_assert_eq!(entity_key(ValueRef::Text(&padded)), entity_key(ValueRef::Text(&v.to_string())));
        prop_assert_eq!(entity_key(ValueRef::Text(&v.to_string())), Some(v.to_string()));
        match entity_key(ValueRef::Text(&text)) {
            None => prop_assert!(text.trim().is_empty()),
            Some(k) => prop_assert_eq!(entity_key(ValueRef::Text(&k)), Some(k.clone()), "not idempotent"),
        }
    }

    /// The minibatch order of every epoch is a permutation, fixed by (seed, epoch).
    #[test]
    fn epoch_order_is_a_reproducible_permutation(seed: u64, epoch in 0usize..1000, n in 0usize..3000) {
        let p = epoch_permutation(seed, epoch, n);
        prop_assert_eq!(p.iter().copied().collect::<BTreeSet<_>>(), (0..n).collect::<BTreeSet<_>>());
        prop_assert_eq!(p.len(), n);
        prop_assert_eq!(epoch_permutation(seed, epoch, n), p);
    }

    /// AUC treats ties as half-correct, so negating the scores gives exactly 1 - AUC;
    /// metrics do not depend on the order of the rows.
    #[test]
    fn metrics_are_order_free_and_auc_complements(
        rows in prop::collection::vec((0u32..50, any::<bool>()), 2..300),
        rotate in any::<Index>(),
    ) {
        let p: Vec<f64> = rows.iter().map(|&(k, _)| f64::from(k) / 50.0).collect();
        let y: Vec<bool> = rows.iter().map(|&(_, b)| b).collect();
        let flipped: Vec<f64> = p.iter().map(|v| 1.0 - v).collect();
        if let (Some(a), Some(b)) = (roc_auc(&p, &y), roc_auc(&flipped, &y)) {
            prop_assert!((a + b - 1.0).abs() < 1e-12, "AUC {} + {} != 1", a, b);
        } else {
            prop_assert!(y.iter().all(|&b| b) || y.iter().all(|&b| !b), "AUC undefined with both classes present");
        }
        let k = rotate.index(rows.len());
        let (mut p2, mut y2) = (p.clone(), y.clone());
        p2.rotate_left(k);
        y2.rotate_left(k);
        let (m1, m2) = (binary_metrics(&p, &y, 0.5), binary_metrics(&p2, &y2, 0.5));
        prop_assert_eq!(m1.confusion.clone(), m2.confusion.clone());
        prop_assert_eq!(m1.balanced_accuracy, m2.balanced_accuracy);
        prop_assert_eq!(m1.roc_auc, m2.roc_auc);
        // Summation order changes with the rows, so the log loss may differ in the last bits.
        prop_assert!((m1.log_loss - m2.log_loss).abs() <= 1e-9 * m1.log_loss.abs().max(1.0));
        prop_assert!((0.0..=1.0).contains(&m1.balanced_accuracy));
    }
}

// ---------------------------------------------------------------------------
// Runtime values and weights

fn value() -> impl Strategy<Value = Value> {
    prop_oneof![
        Just(Value::Null),
        any::<bool>().prop_map(Value::Bool),
        any::<i64>().prop_map(Value::Int),
        any::<f64>().prop_filter("finite", |f| f.is_finite()).prop_map(Value::Float),
        any::<String>().prop_map(Value::Text),
    ]
}

fn dense_model() -> impl Strategy<Value = DenseModel> {
    (1usize..12, prop::collection::vec(1usize..10, 0..3)).prop_flat_map(|(input, hidden)| {
        let config = ModelConfig::dense("small-mlp", input, &hidden);
        let sizes: Vec<(usize, usize)> = config.layers.iter().map(|l| (l.in_dim, l.out_dim)).collect();
        let weights = sizes
            .iter()
            .map(|&(i, o)| (prop::collection::vec(-4.0f32..4.0, i * o), prop::collection::vec(-4.0f32..4.0, o)))
            .collect::<Vec<_>>();
        (Just(config), weights).prop_map(|(config, weights)| {
            let layers = config
                .layers
                .iter()
                .zip(weights)
                .map(|(spec, (weight, bias))| DenseLayer {
                    in_dim: spec.in_dim,
                    out_dim: spec.out_dim,
                    weight,
                    bias,
                    activation: spec.activation,
                })
                .collect();
            DenseModel::new(config, layers).unwrap()
        })
    })
}

proptest! {
    /// A record survives the JSON form applications send.
    #[test]
    fn records_roundtrip_through_json(fields in prop::collection::btree_map("[a-z_]{1,10}", value(), 0..12)) {
        let record = fields.into_iter().fold(Record::new(), |r, (k, v)| r.with(k, v));
        prop_assert_eq!(Record::from_json(&record.to_json()).unwrap(), record);
    }

    /// Weights survive the safetensors codec bit for bit.
    #[test]
    fn weights_roundtrip_through_safetensors(model in dense_model()) {
        let bytes = model.to_safetensors().unwrap();
        prop_assert_eq!(DenseModel::from_safetensors(model.config.clone(), &bytes).unwrap(), model);
    }
}

// ---------------------------------------------------------------------------
// A real bundle: prediction and loading

struct Fixture {
    _dir: tempfile::TempDir,
    files: BTreeMap<String, Vec<u8>>,
    predictor: Predictor,
}

fn read_tree(base: &Path, dir: &Path, out: &mut BTreeMap<String, Vec<u8>>) {
    for e in fs::read_dir(dir).unwrap() {
        let path = e.unwrap().path();
        if path.is_dir() {
            read_tree(base, &path, out);
        } else {
            let rel = path.strip_prefix(base).unwrap().to_string_lossy().replace('\\', "/");
            out.insert(rel, fs::read(&path).unwrap());
        }
    }
}

/// One bundle per test process: a linear model on 1,200 synthetic rows.
fn fixture() -> &'static Fixture {
    static FIXTURE: OnceLock<Fixture> = OnceLock::new();
    FIXTURE.get_or_init(|| {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        init_project(root, "tabular-binary", false).unwrap();
        write_defects_csv(&root.join("data/defects.csv"), 1200, 11, SampleVariant::Base).unwrap();
        let ws = Workspace::open(root).unwrap();
        ws.bind_data(&BindRequest {
            file: root.join("data/defects.csv"),
            target: "defect".into(),
            positive: Some("1".into()),
            group: Some("device_id".into()),
            id: Some("unit_id".into()),
            ..Default::default()
        })
        .unwrap();
        let config = ConfigRequest {
            profile: None,
            overrides: vec!["train.linear.epochs=5".into()],
            override_tables: vec![],
        };
        let run_opts = RunOptions {
            candidates: Some(vec![Candidate::Linear]),
            ..Default::default()
        };
        let run = ws.run(&config, &run_opts, &mut |_| {}).unwrap().runs.remove(0);
        let out = root.join("bundles/prop");
        ws.build_bundle(
            &run.run_id,
            &BundleOptions {
                allow_unevaluated: true,
                out: Some(out.clone()),
            },
        )
        .unwrap();
        let mut files = BTreeMap::new();
        read_tree(&out, &out, &mut files);
        let predictor = Predictor::from_memory("prop", files.clone(), LoadOptions::cpu().with_self_check(true)).unwrap();
        Fixture {
            _dir: dir,
            files,
            predictor,
        }
    })
}

/// Records built from the bundle's own examples, field by field, with some values
/// missing and some categories never seen in training.
fn records() -> impl Strategy<Value = Vec<(Vec<Index>, Vec<u8>)>> {
    prop::collection::vec(
        (prop::collection::vec(any::<Index>(), 64), prop::collection::vec(any::<u8>(), 64)),
        1..6,
    )
}

fn build_record(examples: &[Record], choice: &[Index], noise: &[u8]) -> Record {
    let fx = fixture();
    let mut record = Record::new();
    for (i, field) in fx.predictor.input_schema().fields.iter().enumerate() {
        let from = &examples[choice[i % choice.len()].index(examples.len())];
        let value = from.get(&field.name).cloned().unwrap_or(Value::Null);
        let value = match (noise[i % noise.len()] % 8, &value) {
            (0, _) if field.nullable => Value::Null,
            (1, Value::Text(_)) => Value::Text(format!("never-seen-{}", noise[(i + 1) % noise.len()])),
            _ => value,
        };
        record.insert(field.name.clone(), value);
    }
    record
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// For well-typed records the predictor always answers, with probabilities in
    /// [0, 1], labels that follow the threshold, and the same answer for a record
    /// alone or inside a batch.
    #[test]
    fn predictions_are_total_and_batch_independent(spec in records()) {
        let fx = fixture();
        let examples: Vec<Record> = fx.predictor.bundle().examples.records.iter().map(|v| Record::from_json(v).unwrap()).collect();
        let batch: Vec<Record> = spec.iter().map(|(c, n)| build_record(&examples, c, n)).collect();
        let out = fx.predictor.predict(&batch).unwrap();
        let schema = fx.predictor.output_schema();
        prop_assert_eq!(out.len(), batch.len());
        for (record, p) in batch.iter().zip(&out) {
            let prob = p.probability().unwrap();
            prop_assert!(prob.is_finite() && (0.0..=1.0).contains(&prob));
            prop_assert_eq!(p.is_positive(), prob >= schema.threshold);
            let label = if p.is_positive() { &schema.positive_label } else { &schema.negative_label };
            prop_assert_eq!(&p.label, label);
            prop_assert_eq!(&fx.predictor.predict_one(record).unwrap(), p);
        }
    }
}

#[derive(Debug, Clone)]
enum Damage {
    FlipBit { file: Index, at: Index, bit: u8 },
    Truncate { file: Index, keep: Index },
    Remove { file: Index },
    AppendGarbage { file: Index, bytes: Vec<u8> },
    AddFile { name: String, bytes: Vec<u8> },
    ListPath { path: String },
    EditManifest { at: Index, value: serde_json::Value },
}

fn json_value() -> impl Strategy<Value = serde_json::Value> {
    prop_oneof![
        Just(serde_json::Value::Null),
        any::<bool>().prop_map(serde_json::Value::from),
        any::<i64>().prop_map(serde_json::Value::from),
        Just(serde_json::Value::from(-1)),
        "[ -~]{0,16}".prop_map(serde_json::Value::from),
        Just(serde_json::json!([])),
        Just(serde_json::json!({})),
    ]
}

fn damage() -> impl Strategy<Value = Damage> {
    prop_oneof![
        (any::<Index>(), any::<Index>(), 0u8..8).prop_map(|(file, at, bit)| Damage::FlipBit { file, at, bit }),
        (any::<Index>(), any::<Index>()).prop_map(|(file, keep)| Damage::Truncate { file, keep }),
        any::<Index>().prop_map(|file| Damage::Remove { file }),
        (any::<Index>(), prop::collection::vec(any::<u8>(), 1..32)).prop_map(|(file, bytes)| Damage::AppendGarbage { file, bytes }),
        ("[a-z]{1,8}(/[a-z]{1,8})?", prop::collection::vec(any::<u8>(), 0..16)).prop_map(|(name, bytes)| Damage::AddFile { name, bytes }),
        prop_oneof![
            Just("../escape.json".to_string()),
            Just("/etc/passwd".to_string()),
            Just("examples/../../x".to_string()),
            Just("C:\\x".to_string()),
            Just(String::new()),
            "[a-z./\\\\]{1,12}",
        ]
        .prop_map(|path| Damage::ListPath { path }),
        (any::<Index>(), json_value()).prop_map(|(at, value)| Damage::EditManifest { at, value }),
    ]
}

/// Every leaf of a JSON document, as a path of keys and indices.
fn leaves(v: &serde_json::Value, path: &mut Vec<serde_json::Value>, out: &mut Vec<Vec<serde_json::Value>>) {
    match v {
        serde_json::Value::Object(m) => {
            for (k, child) in m {
                path.push(k.clone().into());
                leaves(child, path, out);
                path.pop();
            }
        }
        serde_json::Value::Array(a) => {
            for (i, child) in a.iter().enumerate() {
                path.push(i.into());
                leaves(child, path, out);
                path.pop();
            }
        }
        _ => out.push(path.clone()),
    }
}

fn apply(files: &mut BTreeMap<String, Vec<u8>>, d: &Damage) {
    let names: Vec<String> = files.keys().cloned().collect();
    let pick = |i: &Index| names[i.index(names.len())].clone();
    match d {
        Damage::FlipBit { file, at, bit } => {
            let b = files.get_mut(&pick(file)).unwrap();
            if !b.is_empty() {
                let i = at.index(b.len());
                b[i] ^= 1 << bit;
            }
        }
        Damage::Truncate { file, keep } => {
            let b = files.get_mut(&pick(file)).unwrap();
            let n = keep.index(b.len().max(1));
            b.truncate(n);
        }
        Damage::Remove { file } => {
            files.remove(&pick(file));
        }
        Damage::AppendGarbage { file, bytes } => files.get_mut(&pick(file)).unwrap().extend_from_slice(bytes),
        Damage::AddFile { name, bytes } => {
            files.insert(name.clone(), bytes.clone());
        }
        Damage::ListPath { path } => edit_manifest(files, |m| {
            if let Some(list) = m.get_mut("files").and_then(|f| f.as_array_mut()) {
                list.push(serde_json::json!({"path": path, "sha256": "00", "bytes": 0}));
            }
        }),
        Damage::EditManifest { at, value } => edit_manifest(files, |m| {
            let mut all = Vec::new();
            leaves(m, &mut Vec::new(), &mut all);
            if all.is_empty() {
                return;
            }
            let mut cur = &mut *m;
            for key in &all[at.index(all.len())] {
                cur = match key {
                    serde_json::Value::String(k) => &mut cur[k.as_str()],
                    other => &mut cur[other.as_u64().unwrap() as usize],
                };
            }
            *cur = value.clone();
        }),
    }
}

/// Same files with the same bytes, except that the manifest (which is not hashed)
/// only needs the same JSON content: rewriting it changes its layout.
fn same_content(a: &BTreeMap<String, Vec<u8>>, b: &BTreeMap<String, Vec<u8>>) -> bool {
    let json = |m: &BTreeMap<String, Vec<u8>>| {
        m.get("manifest.json")
            .and_then(|v| serde_json::from_slice::<serde_json::Value>(v).ok())
    };
    let rest = |m: &BTreeMap<String, Vec<u8>>| {
        m.iter()
            .filter(|(k, _)| *k != "manifest.json")
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect::<Vec<_>>()
    };
    a.contains_key("manifest.json") == b.contains_key("manifest.json") && json(a) == json(b) && json(a).is_some() && rest(a) == rest(b)
}

fn edit_manifest(files: &mut BTreeMap<String, Vec<u8>>, f: impl FnOnce(&mut serde_json::Value)) {
    if let Some(bytes) = files.get_mut("manifest.json")
        && let Ok(mut m) = serde_json::from_slice::<serde_json::Value>(bytes)
    {
        f(&mut m);
        *bytes = serde_json::to_vec(&m).unwrap();
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// Damaged bundles are refused, never half-loaded: with hash verification any
    /// change is an error, and without it (where the parsers see the damaged bytes)
    /// loading and predicting may fail but never panic.
    #[test]
    fn damaged_bundles_are_refused_without_panicking(damages in prop::collection::vec(damage(), 1..4)) {
        let fx = fixture();
        let mut files = fx.files.clone();
        for d in &damages {
            apply(&mut files, d);
        }
        let verified = Bundle::from_memory("damaged", files.clone(), &LoadOptions::cpu());
        if same_content(&files, &fx.files) {
            prop_assert!(verified.is_ok());
        } else {
            let e = verified.expect_err("a damaged bundle was accepted");
            prop_assert!(
                matches!(e.code, ErrorCode::BundleCorrupt | ErrorCode::BundleMissingFile | ErrorCode::BundleIncompatible | ErrorCode::DeviceUnsupported),
                "unexpected error code {:?}: {}", e.code, e.message
            );
        }
        let unverified = LoadOptions::cpu().with_hash_verification(false);
        if let Ok(p) = Predictor::from_memory("damaged", files, unverified) {
            let _ = p.self_check();
        }
    }
}
