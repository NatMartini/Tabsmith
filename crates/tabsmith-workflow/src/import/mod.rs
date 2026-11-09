//! Models trained by XGBoost and LightGBM, turned into bundles (`tabsmith import`).
//!
//! An imported model is a tree ensemble in Tabsmith's format with an identity transform:
//! each input field is one feature, used as given and rounded to f32. The bundle then
//! loads, predicts, serves, compiles into Rust code (`model!`) and answers rules
//! (`tabsmith audit`) like a bundle Tabsmith trained.
//!
//! XGBoost's outputs come out bit for bit. It sends `x < s` left; Tabsmith sends `x <= t`
//! left, so `t` is the f32 just below `s`, which splits every f32 input the same way. Its
//! leaves are f32 and it adds them in f32, in tree order, from the starting margin; the
//! bundle says so ([`LeafSum::F32`]) and the runtime adds them the same way.
//!
//! LightGBM's leaves are f64; the bundle holds the nearest f32, so its outputs come out
//! within a bound the import computes and reports. Its thresholds are rounded down to f32,
//! which splits every input that is an f32 value as LightGBM does.
//!
//! Missing values: each split of a library's tree sends a missing value left or right. When
//! one value takes the missing value's way at every split on its feature, the import imputes
//! that value, which reproduces the library exactly. When none does (left at some splits,
//! right at others), the field refuses missing values ([`MissingRule::Refuse`]) rather than
//! predict differently from the library.

mod lightgbm;
mod logf;
mod xgboost;

use std::fs;
use std::path::{Path, PathBuf};

use serde::Serialize;
use tabsmith_contracts::config::{ExtraColumnPolicy, NanPolicy, Task, UnknownCategoryPolicy};
use tabsmith_contracts::hash::sha256_hex;
use tabsmith_contracts::manifest::{
    BundleEvaluation, BundleManifest, BundleProvenance, EVALUATION_FILE, EXAMPLES_EXPECTED_FILE, EXAMPLES_INPUT_FILE, ExampleExpected,
    ExampleInputs, ExpectedPrediction, FileEntry, INPUT_SCHEMA_FILE, MANIFEST_FILE, MODEL_CONFIG_FILE, MODEL_WEIGHTS_FILE,
    OUTPUT_SCHEMA_FILE, RuntimeRequirements, TRANSFORM_FILE,
};
use tabsmith_contracts::model::{LeafSum, ModelConfig, TreesSpec};
use tabsmith_contracts::schema::{FieldKind, InputField, InputSchema, OutputSchema, RawDtype, TargetScale};
use tabsmith_contracts::time::now_rfc3339;
use tabsmith_contracts::transform::{FeatureTransform, FittedTransform, MissingRule, TRANSFORM_VERSION};
use tabsmith_contracts::{BUNDLE_FORMAT_VERSION, ErrorCode, Phase, RECIPE_ID, Result, TABSMITH_VERSION, TabsmithError};
use tabsmith_runtime::model::{Model, TreeEnsemble};
use tabsmith_runtime::{CheckReport, LoadOptions, Predictor, Record, Value};

use crate::bundle::file_entry;
use crate::project::{fresh_dir, read_json, write_atomic, write_json};

/// `evaluation_status` of an imported bundle: Tabsmith did not evaluate the model.
pub const IMPORTED: &str = "imported";

/// The most input fields an imported model may declare. A file may declare counts it does
/// not back; nothing is sized by a declared count beyond this, or beyond what the file holds.
pub const MAX_FEATURES: usize = 1 << 20;

/// What `tabsmith import` was asked for.
#[derive(Debug, Clone, Default)]
pub struct ImportOptions {
    /// Bundle directory to create.
    pub out: PathBuf,
    /// Bundle name (default: the model file's name without its extension).
    pub name: Option<String>,
    /// Name of the predicted column (default `target`).
    pub target: Option<String>,
    /// Class labels in the model's order: negative and positive (binary) or one per class
    /// (multiclass). Default: `0`, `1`, ….
    pub labels: Vec<String>,
    /// Input field names in the model's feature order (default: the model's own names, or
    /// XGBoost's `f0`, `f1`, … when it has none).
    pub features: Vec<String>,
    /// Binary: the decision threshold on the probability (default 0.5).
    pub threshold: Option<f64>,
}

/// The library a model came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum Library {
    #[serde(rename = "xgboost")]
    XGBoost,
    #[serde(rename = "lightgbm")]
    LightGBM,
}

impl Library {
    pub fn as_str(self) -> &'static str {
        match self {
            Library::XGBoost => "xgboost",
            Library::LightGBM => "lightgbm",
        }
    }

    /// `XGBoost 3.4.1` or `LightGBM, model format v4`: a LightGBM model file records the
    /// version of its format, not of the library.
    pub fn source(self, version: &str) -> String {
        match self {
            Library::XGBoost => format!("XGBoost {version}"),
            Library::LightGBM => format!("LightGBM, model format {version}"),
        }
    }
}

/// What was imported and how close its outputs are to the library's.
#[derive(Debug, Clone, Serialize)]
pub struct ImportedModel {
    pub library: Library,
    /// The library version that saved the model (XGBoost) or the model file's format (LightGBM).
    pub version: String,
    pub objective: String,
    pub task: Task,
    pub features: usize,
    pub trees: usize,
    pub nodes: usize,
    /// True when the bundle's outputs (margins, raw scores) are the library's, bit for bit.
    pub exact: bool,
    /// A bound on |the bundle's output − the library's| for inputs that are f32 values
    /// (0 when exact): leaves rounded to f32, the sums' rounding, the output's rounding.
    pub max_output_difference: f64,
    /// Fields that refuse missing values (their splits send one left at some and right at others).
    pub refuses_missing: Vec<String>,
    /// SHA-256 of the model file.
    pub sha256: String,
    pub notes: Vec<String>,
}

/// The result of `tabsmith import`.
#[derive(Debug, Clone, Serialize)]
pub struct ImportReport {
    pub path: String,
    pub manifest: BundleManifest,
    pub check: CheckReport,
    /// True when an identical bundle already existed at the path.
    pub already_existed: bool,
    pub model: ImportedModel,
}

pub(crate) fn invalid(message: impl Into<String>) -> TabsmithError {
    TabsmithError::new(ErrorCode::ImportInvalid, Phase::Import, message)
        .next("pass a model file saved by XGBoost (`save_model(\"model.json\")`) or LightGBM (`save_model(\"model.txt\")`)")
}

pub(crate) fn unsupported(message: impl Into<String>) -> TabsmithError {
    TabsmithError::new(ErrorCode::ImportUnsupported, Phase::Import, message)
        .next("see docs/import.md for the models Tabsmith imports: gradient-boosted trees over numeric features")
}

/// One node of a library's tree, by the library's own node ids.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Node {
    /// Rows with `x <= threshold` go to `left`; a missing value goes left when `missing_left`.
    Split {
        feature: usize,
        threshold: f32,
        missing_left: bool,
        left: usize,
        right: usize,
    },
    Leaf(f32),
}

/// Where a feature's missing value goes, gathered over every split on it.
#[derive(Debug, Clone, Copy, Default)]
struct MissingWay {
    /// The lowest threshold of a split that sends a missing value left.
    left_below: Option<f32>,
    /// The highest threshold of a split that sends a missing value right.
    right_above: Option<f32>,
    /// A split sends a missing value where no finite value goes.
    stranded: bool,
}

impl MissingWay {
    /// A value that takes the missing value's way at every split: 0 when it does (what
    /// LightGBM gives a missing value at splits without a missing type), else the lowest
    /// left-sending threshold, else the f32 just above the highest right-sending one.
    fn value(&self) -> Option<f32> {
        if self.stranded {
            return None;
        }
        let fits = |z: f32| self.left_below.is_none_or(|l| z <= l) && self.right_above.is_none_or(|r| z > r);
        [Some(0.0), self.left_below, self.right_above.map(f32::next_up)]
            .into_iter()
            .flatten()
            .find(|&z| z.is_finite() && fits(z))
    }
}

/// A library's ensemble in Tabsmith's layout: every tree's nodes in preorder, children after
/// their parent; tree `t` adds to output `t % n_outputs`.
#[derive(Debug, Clone)]
pub(crate) struct Forest {
    pub library: Library,
    pub version: String,
    pub objective: String,
    pub task: Task,
    pub feature_names: Vec<String>,
    pub n_outputs: usize,
    /// The starting score of each output.
    pub base: Vec<f64>,
    pub sum: LeafSum,
    pub roots: Vec<u32>,
    pub feature: Vec<i32>,
    pub threshold: Vec<f32>,
    pub left: Vec<u32>,
    pub right: Vec<u32>,
    pub value: Vec<f32>,
    pub max_depth: usize,
    missing: Vec<MissingWay>,
    pub exact: bool,
    pub max_output_difference: f64,
    pub notes: Vec<String>,
}

impl Forest {
    pub(crate) fn new(
        library: Library,
        version: String,
        objective: String,
        task: Task,
        feature_names: Vec<String>,
        n_outputs: usize,
    ) -> Self {
        let n = feature_names.len();
        Self {
            library,
            version,
            objective,
            task,
            feature_names,
            n_outputs,
            base: vec![0.0; n_outputs],
            sum: LeafSum::F64,
            roots: Vec::new(),
            feature: Vec::new(),
            threshold: Vec::new(),
            left: Vec::new(),
            right: Vec::new(),
            value: Vec::new(),
            max_depth: 0,
            missing: vec![MissingWay::default(); n],
            exact: false,
            max_output_difference: 0.0,
            notes: Vec::new(),
        }
    }

    /// Appends tree `tree` (the library's numbering), renumbered in preorder from `root`. A
    /// split no finite value takes to the left (threshold −∞) keeps only its right side.
    pub(crate) fn push_tree(&mut self, nodes: &[Node], root: usize, tree: usize) -> Result<()> {
        self.roots
            .push(u32::try_from(self.feature.len()).map_err(|_| unsupported("the model has more than 2^32 nodes"))?);
        let mut seen = vec![false; nodes.len()];
        // (library node, Tabsmith parent and side to point at it, depth)
        let mut stack = vec![(root, None::<(usize, bool)>, 0usize)];
        while let Some((id, parent, depth)) = stack.pop() {
            let node = *nodes
                .get(id)
                .ok_or_else(|| invalid(format!("tree {tree} points to node {id}, which it does not have")))?;
            if std::mem::replace(&mut seen[id], true) {
                return Err(invalid(format!("tree {tree} reaches node {id} twice")));
            }
            if let Node::Split {
                feature,
                threshold,
                missing_left,
                right,
                ..
            } = node
                && threshold == f32::NEG_INFINITY
            {
                if missing_left {
                    self.missing_way(feature, tree)?.stranded = true;
                }
                stack.push((right, parent, depth));
                continue;
            }
            let here = self.feature.len();
            if let Some((p, is_left)) = parent {
                if is_left {
                    self.left[p] = here as u32;
                } else {
                    self.right[p] = here as u32;
                }
            }
            match node {
                Node::Leaf(v) => {
                    if !v.is_finite() {
                        return Err(invalid(format!("tree {tree} has a leaf value that is not a finite f32")));
                    }
                    self.feature.push(-1);
                    self.threshold.push(0.0);
                    self.left.push(0);
                    self.right.push(0);
                    self.value.push(v);
                    self.max_depth = self.max_depth.max(depth);
                }
                Node::Split {
                    feature,
                    threshold,
                    missing_left,
                    left,
                    right,
                } => {
                    if threshold.is_nan() {
                        return Err(invalid(format!("tree {tree} has a split without a threshold")));
                    }
                    let way = self.missing_way(feature, tree)?;
                    if missing_left {
                        way.left_below = Some(way.left_below.map_or(threshold, |l| l.min(threshold)));
                    } else {
                        way.right_above = Some(way.right_above.map_or(threshold, |r| r.max(threshold)));
                    }
                    self.feature.push(feature as i32);
                    // x <= f32::MAX holds for every finite x: a threshold of +∞ splits alike.
                    self.threshold.push(threshold.min(f32::MAX));
                    self.left.push(0);
                    self.right.push(0);
                    self.value.push(0.0);
                    stack.push((right, Some((here, false)), depth + 1));
                    stack.push((left, Some((here, true)), depth + 1));
                }
            }
        }
        Ok(())
    }

    fn missing_way(&mut self, feature: usize, tree: usize) -> Result<&mut MissingWay> {
        let n = self.missing.len();
        self.missing
            .get_mut(feature)
            .ok_or_else(|| invalid(format!("tree {tree} splits on feature {feature}, but the model has {n} features")))
    }

    /// Puts trees in Tabsmith's order, tree `t` for output `t % n_outputs`, from trees listed
    /// with their output; each output keeps its trees' order, which fixes the order of its sum.
    pub(crate) fn interleave<T>(trees: Vec<(usize, T)>, n_outputs: usize) -> Result<Vec<T>> {
        // Each output needs a tree, so the outputs are at most the trees: no allocation
        // follows a count the file declares and does not back.
        if n_outputs == 0 || n_outputs > trees.len() {
            return Err(unsupported(format!(
                "{n_outputs} outputs for {} trees: each output needs at least one tree",
                trees.len()
            )));
        }
        let mut by_output: Vec<std::collections::VecDeque<T>> = (0..n_outputs).map(|_| Default::default()).collect();
        for (o, t) in trees {
            by_output
                .get_mut(o)
                .ok_or_else(|| invalid(format!("a tree adds to output {o}, but the model has {n_outputs}")))?
                .push_back(t);
        }
        let rounds = by_output[0].len();
        if by_output.iter().any(|q| q.len() != rounds) || rounds == 0 {
            return Err(unsupported("the outputs have different numbers of trees (or none)"));
        }
        let mut out = Vec::with_capacity(rounds * n_outputs);
        for _ in 0..rounds {
            for q in &mut by_output {
                out.push(q.pop_front().expect("counted"));
            }
        }
        Ok(out)
    }
}

/// Reads an XGBoost JSON or LightGBM text model and writes it as a bundle at `opts.out`.
pub fn import_model(model_path: &Path, opts: &ImportOptions) -> Result<ImportReport> {
    let bytes = fs::read(model_path).map_err(|e| TabsmithError::io(Phase::Import, model_path, e))?;
    let sha256 = sha256_hex(&bytes);
    let text = std::str::from_utf8(&bytes).map_err(|_| {
        let ubj = model_path.extension().is_some_and(|e| e.eq_ignore_ascii_case("ubj"));
        if ubj {
            unsupported("XGBoost's binary UBJSON format").next("save the model as JSON: `booster.save_model(\"model.json\")`")
        } else {
            invalid(format!("{} is not text", model_path.display()))
        }
    })?;
    let forest = read(text, &model_path.display().to_string())?;
    let name = opts
        .name
        .clone()
        .unwrap_or_else(|| model_path.file_stem().and_then(|s| s.to_str()).unwrap_or("imported").to_string());
    write_bundle(forest, opts, &name, sha256)
}

/// Reads a model file's text: XGBoost's JSON or LightGBM's text.
fn read(text: &str, what: &str) -> Result<Forest> {
    let trimmed = text.trim_start();
    if trimmed.starts_with('{') {
        xgboost::parse(text)
    } else if trimmed.lines().next().is_some_and(|l| l.trim() == "tree") {
        lightgbm::parse(text)
    } else {
        Err(invalid(format!(
            "{what} is neither an XGBoost JSON model nor a LightGBM text model"
        )))
    }
}

/// What [`import_model`] would make of a model file's text, without writing anything: the
/// model is read, its trees are checked as the runtime checks a bundle's, and the summary
/// comes back, with the model's own feature names. A test feeds this damaged models.
pub fn describe(text: &str) -> Result<ImportedModel> {
    let f = read(text, "the text")?;
    build_ensemble(&f)?;
    let (_, refuses) = missing_rules(&f);
    Ok(summary(&f, refusing(&f.feature_names, &refuses), sha256_hex(text.as_bytes())))
}

/// Each feature's missing value: what to impute, or that it is refused; and the refusals.
fn missing_rules(f: &Forest) -> (Vec<(f64, MissingRule)>, Vec<bool>) {
    let rules: Vec<(f64, MissingRule)> = f
        .missing
        .iter()
        .map(|w| match w.value() {
            Some(z) => (f64::from(z), MissingRule::Impute),
            None => (0.0, MissingRule::Refuse),
        })
        .collect();
    let refuses = rules.iter().map(|r| r.1 == MissingRule::Refuse).collect();
    (rules, refuses)
}

fn refusing(names: &[String], refuses: &[bool]) -> Vec<String> {
    names.iter().zip(refuses).filter(|(_, r)| **r).map(|(n, _)| n.clone()).collect()
}

/// The model's trees as the runtime holds them, checked as it checks a bundle's.
fn build_ensemble(f: &Forest) -> Result<(ModelConfig, TreeEnsemble)> {
    let spec = TreesSpec {
        n_trees: f.roots.len(),
        n_nodes: f.feature.len(),
        max_depth: f.max_depth,
        base_score: if f.n_outputs == 1 { f.base[0] } else { 0.0 },
        n_outputs: f.n_outputs,
        base_scores: if f.n_outputs == 1 { Vec::new() } else { f.base.clone() },
        monotone: Vec::new(),
        sum: f.sum,
    };
    let config = ModelConfig::trees_for(f.feature_names.len(), f.task, Some(spec));
    let ensemble = TreeEnsemble::new(
        config.clone(),
        f.roots.clone(),
        f.feature.clone(),
        f.threshold.clone(),
        f.left.clone(),
        f.right.clone(),
        f.value.clone(),
    )
    .map_err(|e| invalid(format!("the trees do not form a valid ensemble: {}", e.message)))?;
    Ok((config, ensemble))
}

fn summary(f: &Forest, refuses_missing: Vec<String>, sha256: String) -> ImportedModel {
    ImportedModel {
        library: f.library,
        version: f.version.clone(),
        objective: f.objective.clone(),
        task: f.task,
        features: f.feature_names.len(),
        trees: f.roots.len(),
        nodes: f.feature.len(),
        exact: f.exact,
        max_output_difference: f.max_output_difference,
        refuses_missing,
        sha256,
        notes: f.notes.clone(),
    }
}

fn usage(field: &str, message: impl Into<String>) -> TabsmithError {
    TabsmithError::new(ErrorCode::ConfigInvalid, Phase::Import, message).field(field)
}

fn check_names(names: &[String], what: &str) -> std::result::Result<(), String> {
    let mut seen = std::collections::BTreeSet::new();
    for n in names {
        if n.trim().is_empty() {
            return Err(format!("{what} has an empty name"));
        }
        if !seen.insert(n.as_str()) {
            return Err(format!("{what} names `{n}` twice"));
        }
    }
    Ok(())
}

/// Six significant digits, inside the f32 range: short example values the bundle accepts.
fn short(x: f64) -> f64 {
    let max = f64::from(f32::MAX);
    format!("{x:.5e}").parse().unwrap_or(x).clamp(-max, max)
}

/// Example records: for each feature the middle threshold of its splits, values below and
/// above all of them, a mix, and a missing value where the field accepts one.
fn examples(f: &Forest, names: &[String], refuses: &[bool]) -> Vec<Record> {
    let mut thresholds: Vec<Vec<f32>> = vec![Vec::new(); names.len()];
    for (i, &feat) in f.feature.iter().enumerate() {
        if feat >= 0 {
            thresholds[feat as usize].push(f.threshold[i]);
        }
    }
    let mut out = vec![Record::new(); 5];
    for (j, (name, ts)) in names.iter().zip(&mut thresholds).enumerate() {
        ts.sort_by(f32::total_cmp);
        let (lo, mid, hi) = match (ts.first(), ts.last()) {
            (Some(&a), Some(&b)) => {
                let (a, b) = (f64::from(a), f64::from(b));
                (
                    short(a - 1.0 - a.abs() * 0.1),
                    short(f64::from(ts[ts.len() / 2])),
                    short(b + 1.0 + b.abs() * 0.1),
                )
            }
            _ => (0.0, 0.0, 0.0),
        };
        out[0].insert(name.clone(), mid);
        out[1].insert(name.clone(), lo);
        out[2].insert(name.clone(), hi);
        out[3].insert(name.clone(), if j % 2 == 0 { lo } else { hi });
        if refuses[j] {
            out[4].insert(name.clone(), mid);
        } else {
            out[4].insert(name.clone(), Value::Null);
        }
    }
    out
}

fn write_bundle(f: Forest, opts: &ImportOptions, name: &str, sha256: String) -> Result<ImportReport> {
    let n = f.feature_names.len();
    let names = if opts.features.is_empty() {
        check_names(&f.feature_names, "the model").map_err(|m| invalid(m).next("name the features with --features"))?;
        f.feature_names.clone()
    } else {
        if opts.features.len() != n {
            return Err(usage(
                "--features",
                format!("the model has {n} features; --features names {}", opts.features.len()),
            ));
        }
        check_names(&opts.features, "--features").map_err(|m| usage("--features", m))?;
        opts.features.clone()
    };
    let target = opts.target.clone().unwrap_or_else(|| "target".into());
    if opts.threshold.is_some() && f.task != Task::Binary {
        return Err(usage("--threshold", "a decision threshold is for binary models"));
    }
    let labels = |k: usize| -> Result<Vec<String>> {
        if opts.labels.is_empty() {
            return Ok((0..k).map(|i| i.to_string()).collect());
        }
        if opts.labels.len() != k {
            return Err(usage(
                "--labels",
                format!("the model has {k} classes; --labels names {}", opts.labels.len()),
            ));
        }
        check_names(&opts.labels, "--labels").map_err(|m| usage("--labels", m))?;
        Ok(opts.labels.clone())
    };
    let output_schema = match f.task {
        Task::Binary => {
            let l = labels(2)?;
            let threshold = opts.threshold.unwrap_or(0.5);
            if !(threshold > 0.0 && threshold < 1.0) {
                return Err(usage("--threshold", "the threshold must be inside (0, 1)"));
            }
            OutputSchema::new(&target, &l[0], &l[1], threshold)
        }
        Task::Multiclass => OutputSchema::multiclass(&target, labels(f.n_outputs)?),
        Task::Regression => {
            if !opts.labels.is_empty() {
                return Err(usage("--labels", "a regression model has no classes"));
            }
            OutputSchema::regression(&target, TargetScale { mean: 0.0, std: 1.0 })
        }
    };
    let (rules, refuses) = missing_rules(&f);
    let transform = FittedTransform {
        version: TRANSFORM_VERSION.into(),
        features: names
            .iter()
            .zip(&rules)
            .enumerate()
            .map(|(i, (name, &(impute, missing)))| FeatureTransform::Numeric {
                name: name.clone(),
                impute,
                mean: 0.0,
                std: 1.0,
                offset: i,
                missing,
            })
            .collect(),
        dropped: Vec::new(),
        output_dim: n,
        output_names: names.clone(),
        nan: NanPolicy::Missing,
        unknown_category: UnknownCategoryPolicy::Ignore,
        missing_tokens: vec![String::new(), "NA".into(), "null".into()],
        fitted_on_split: String::new(),
        fitted_rows: 0,
    };
    transform.check_consistency().map_err(invalid)?;
    let input_schema = InputSchema {
        fields: names
            .iter()
            .zip(&refuses)
            .map(|(name, &refuse)| InputField {
                name: name.clone(),
                kind: FieldKind::Numeric,
                raw_dtype: RawDtype::Float,
                required: true,
                nullable: !refuse,
                note: refuse.then(|| "refuses missing values: the model's splits on it send one left at some and right at others".into()),
            })
            .collect(),
        missing_column: "error".into(),
        extra_columns: ExtraColumnPolicy::Ignore,
        unknown_category: UnknownCategoryPolicy::Ignore,
        nan: NanPolicy::Missing,
        infinite: "error".into(),
        csv_delimiter: Some(",".into()),
        passthrough_id: None,
    };
    let refuses_missing = refusing(&names, &refuses);
    let (config, ensemble) = build_ensemble(&f)?;
    let weights = Model::Trees(ensemble).to_safetensors()?;
    let model = summary(&f, refuses_missing.clone(), sha256.clone());
    let mut notes = vec![format!(
        "imported from {} ({}), model file sha256 {sha256}; Tabsmith did not evaluate it",
        f.library.source(&f.version),
        f.objective
    )];
    notes.extend(f.notes.iter().cloned());
    if !refuses_missing.is_empty() {
        notes.push(format!(
            "refuses missing values in {}: the model's splits on each send one left at some and right at others",
            refuses_missing.join(", ")
        ));
    }
    let evaluation = BundleEvaluation {
        status: IMPORTED.into(),
        test: None,
        valid: None,
        notes,
    };
    let records = examples(&f, &names, &refuses);

    let out = opts.out.clone();
    let parent = out
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."))
        .to_path_buf();
    fs::create_dir_all(&parent).map_err(|e| TabsmithError::io(Phase::Import, &parent, e))?;
    let tmp = parent.join(format!(
        ".{}.tmp-{}",
        out.file_name().and_then(|n| n.to_str()).unwrap_or("bundle"),
        std::process::id()
    ));
    fresh_dir(&tmp, &["examples"], Phase::Import)?;
    let cleanup = |e: TabsmithError| {
        let _ = fs::remove_dir_all(&tmp);
        e
    };
    let manifest = (|| -> Result<BundleManifest> {
        write_json(&tmp.join(INPUT_SCHEMA_FILE), &input_schema)?;
        write_json(&tmp.join(OUTPUT_SCHEMA_FILE), &output_schema)?;
        write_json(&tmp.join(TRANSFORM_FILE), &transform)?;
        write_json(&tmp.join(MODEL_CONFIG_FILE), &config)?;
        write_atomic(&tmp.join(MODEL_WEIGHTS_FILE), &weights)?;
        write_json(&tmp.join(EVALUATION_FILE), &evaluation)?;
        write_json(
            &tmp.join(EXAMPLES_INPUT_FILE),
            &ExampleInputs {
                records: records.iter().map(Record::to_json).collect(),
                note: "synthetic records built from the model's split thresholds".into(),
            },
        )?;
        let expected = |predictions: Vec<ExpectedPrediction>| ExampleExpected {
            predictions,
            tolerance: 1e-6,
        };
        // Placeholder of the right length, so the bundle loads once to compute the
        // expected outputs with the runtime itself; replaced right after.
        let blank = ExpectedPrediction {
            label: String::new(),
            probability: 0.0,
            probabilities: Vec::new(),
            value: None,
        };
        write_json(&tmp.join(EXAMPLES_EXPECTED_FILE), &expected(vec![blank; records.len()]))?;
        let mut manifest = BundleManifest {
            bundle_format: BUNDLE_FORMAT_VERSION,
            bundle_id: String::new(),
            name: name.to_string(),
            created_at: now_rfc3339(),
            recipe: RECIPE_ID.to_string(),
            tabsmith_version: TABSMITH_VERSION.to_string(),
            provenance: BundleProvenance {
                run_id: "import".into(),
                candidate: f.library.as_str().into(),
                snapshot_id: String::new(),
                split_id: String::new(),
                transform_id: String::new(),
                config_hash: sha256.clone(),
                evaluation_id: None,
            },
            runtime: RuntimeRequirements {
                // `sum` and `missing` are new in this version: older runtimes refuse them.
                min_runtime_version: TABSMITH_VERSION.into(),
                transform_version: TRANSFORM_VERSION.into(),
                model_format: config.format.clone(),
                devices: vec!["cpu".into()],
                precision: "fp32".into(),
            },
            files: Vec::new(),
            self_contained: true,
            contains_training_data: false,
            contains_optimizer_state: false,
            evaluation_status: IMPORTED.into(),
        };
        let listed = [
            INPUT_SCHEMA_FILE,
            OUTPUT_SCHEMA_FILE,
            TRANSFORM_FILE,
            MODEL_CONFIG_FILE,
            MODEL_WEIGHTS_FILE,
            EVALUATION_FILE,
            EXAMPLES_INPUT_FILE,
            EXAMPLES_EXPECTED_FILE,
        ];
        let write_manifest = |m: &mut BundleManifest| -> Result<()> {
            m.files = listed.iter().map(|rel| file_entry(&tmp, rel)).collect::<Result<Vec<FileEntry>>>()?;
            m.bundle_id = m.expected_id();
            write_json(&tmp.join(MANIFEST_FILE), m)
        };
        write_manifest(&mut manifest)?;
        let predictions = Predictor::load(&tmp, LoadOptions::cpu())?.predict(&records)?;
        write_json(
            &tmp.join(EXAMPLES_EXPECTED_FILE),
            &expected(
                predictions
                    .iter()
                    .map(|p| ExpectedPrediction {
                        label: p.label.clone(),
                        probability: p.probability().unwrap_or(0.0),
                        probabilities: p.probabilities().map(<[f64]>::to_vec).unwrap_or_default(),
                        value: p.value(),
                    })
                    .collect(),
            ),
        )?;
        write_manifest(&mut manifest)?;
        Ok(manifest)
    })()
    .map_err(cleanup)?;
    let check = Predictor::load(&tmp, LoadOptions::cpu().with_self_check(true))
        .and_then(|p| p.self_check())
        .map_err(cleanup)?;
    let report = |manifest, check, already_existed| ImportReport {
        path: out.display().to_string(),
        manifest,
        check,
        already_existed,
        model: model.clone(),
    };
    if out.exists() {
        let _ = fs::remove_dir_all(&tmp);
        return match read_json::<BundleManifest>(&out.join(MANIFEST_FILE), Phase::Import).ok() {
            Some(m) if m.bundle_id == manifest.bundle_id => {
                let check = Predictor::load(&out, LoadOptions::cpu().with_self_check(true))?.self_check()?;
                Ok(report(m, check, true))
            }
            _ => Err(TabsmithError::new(
                ErrorCode::BundleExists,
                Phase::Import,
                format!("`{}` already exists and holds something else; bundles are immutable", out.display()),
            )
            .field(out.display().to_string())
            .next("choose another directory with --out")),
        };
    }
    fs::rename(&tmp, &out).map_err(|e| {
        let _ = fs::remove_dir_all(&tmp);
        TabsmithError::io(Phase::Import, &out, e)
    })?;
    let mut check = check;
    check.bundle_dir = out.display().to_string();
    Ok(report(manifest, check, false))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn way(left: Option<f32>, right: Option<f32>) -> MissingWay {
        MissingWay {
            left_below: left,
            right_above: right,
            stranded: false,
        }
    }

    #[test]
    fn a_missing_value_takes_its_way_at_every_split_or_is_refused() {
        // No split: 0. Only right-sending splits: 0 when it is above them, else just above.
        assert_eq!(way(None, None).value(), Some(0.0));
        assert_eq!(way(None, Some(-2.0)).value(), Some(0.0));
        assert_eq!(way(None, Some(3.0)).value(), Some(3.0f32.next_up()));
        // Only left-sending splits: 0 when it is below them, else the lowest threshold.
        assert_eq!(way(Some(1.0), None).value(), Some(0.0));
        assert_eq!(way(Some(-1.5), None).value(), Some(-1.5));
        // Both: a value in (right, left] exists only when right < left.
        assert_eq!(way(Some(5.0), Some(2.0)).value(), Some(5.0));
        assert_eq!(way(Some(2.0), Some(2.0)).value(), None);
        assert_eq!(way(Some(1.0), Some(4.0)).value(), None);
        assert_eq!(way(None, Some(f32::MAX)).value(), None);
        let stranded = MissingWay {
            stranded: true,
            ..Default::default()
        };
        assert_eq!(stranded.value(), None);
    }

    #[test]
    fn trees_are_renumbered_in_preorder_and_dead_sides_dropped() {
        let mut f = Forest::new(
            Library::XGBoost,
            "t".into(),
            "t".into(),
            Task::Regression,
            vec!["a".into(), "b".into()],
            1,
        );
        // 0: a <= 1 ? 2 : 1;  1: leaf 10;  2: b <= -inf (no finite value) ? 3 : 4;  3: leaf 20;  4: leaf 30
        let nodes = [
            Node::Split {
                feature: 0,
                threshold: 1.0,
                missing_left: false,
                left: 2,
                right: 1,
            },
            Node::Leaf(10.0),
            Node::Split {
                feature: 1,
                threshold: f32::NEG_INFINITY,
                missing_left: true,
                left: 3,
                right: 4,
            },
            Node::Leaf(20.0),
            Node::Leaf(30.0),
        ];
        f.push_tree(&nodes, 0, 0).unwrap();
        assert_eq!(f.feature, vec![0, -1, -1]);
        assert_eq!((f.left[0], f.right[0]), (1, 2));
        assert_eq!(f.value, vec![0.0, 30.0, 10.0]);
        assert_eq!(f.max_depth, 1);
        // b's missing value must go left where no finite value goes: refused.
        assert_eq!(f.missing[1].value(), None);
        assert_eq!(f.missing[0].value(), Some(1.0f32.next_up()));
        // A cycle is refused.
        let cycle = [Node::Split {
            feature: 0,
            threshold: 0.0,
            missing_left: false,
            left: 0,
            right: 0,
        }];
        assert!(f.push_tree(&cycle, 0, 1).is_err());
    }

    #[test]
    fn interleaving_keeps_each_outputs_order() {
        let trees = vec![(0, "a0"), (1, "b0"), (1, "b1"), (0, "a1"), (2, "c0"), (2, "c1")];
        assert_eq!(Forest::interleave(trees, 3).unwrap(), vec!["a0", "b0", "c0", "a1", "b1", "c1"]);
        assert!(Forest::interleave(vec![(0, 1), (1, 2), (0, 3)], 2).is_err());
    }
}
