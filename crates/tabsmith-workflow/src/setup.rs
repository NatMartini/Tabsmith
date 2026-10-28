//! Project creation, data binding and the plan report.

use std::fs;
use std::path::{Path, PathBuf};

use serde::Serialize;
use tabsmith_contracts::config::{Candidate, ConfigLayer, ProjectFile, SplitStrategy, Task, ValueSource, resolve};
use tabsmith_contracts::data::DataSnapshot;
use tabsmith_contracts::device::DevicePlan;
use tabsmith_contracts::transform::FeatureTransform;
use tabsmith_contracts::{ErrorCode, Phase, Result, TabsmithError};
use toml_edit::{Array, DocumentMut, Item, Table as EditTable, value};

use crate::data::{ColumnDecision, ColumnRole, TargetInfo, plan_columns, snapshot_source};
use crate::pipeline::Workspace;
use crate::project::{CONFIG_FILE, ConfigRequest, write_atomic};
use crate::split::make_split;
use crate::train::model_config;

/// The `tabsmith.toml` of a new project for `task`.
fn template(task: Task) -> String {
    TEMPLATE.replace("tabular-binary@0.1", task.recipe()).replace(
        "metric = \"balanced_accuracy\"",
        &format!("metric = \"{}\"", task.default_metric().as_str()),
    )
}

const TEMPLATE: &str = r#"# Tabsmith project configuration (recipe tabular-binary@0.1).
# Settings not written here use the recipe defaults. `tabsmith config` lists every
# effective setting and where it came from; unknown keys are errors.
schema_version = 1
recipe = "tabular-binary@0.1"

[data]
# Filled in by `tabsmith data bind <file> --target <column>`:
# source = "data/train.csv"
# target = "defect"
# positive_label = "1"
# split = "stratified"           # rows independent; use "group" + group_column for repeated entities

[selection]
candidates = ["linear", "small-mlp", "gbdt"]
metric = "balanced_accuracy"     # validation metric used to rank candidates
threshold = 0.5                  # decision threshold on P(positive), fixed before evaluation

[execution]
baseline_device = "cpu"
candidate_device = "auto-certified"   # best certified device in this build; never a silent fallback
fallback = "error"
precision = "fp32"

# Profiles are named override sets: `tabsmith run --profile quick`
[profiles.quick.train.linear]
epochs = 10

[profiles.quick.train.small-mlp]
epochs = 12

[profiles.quick.train.gbdt]
epochs = 60
"#;

#[derive(Debug, Clone, Serialize)]
pub struct InitReport {
    pub project: String,
    pub config: String,
    pub sample_data: Option<String>,
}

/// Creates a new project directory.
pub fn init_project(dir: &Path, recipe: &str, sample_data: bool) -> Result<InitReport> {
    let Some(task) = Task::from_recipe(recipe) else {
        return Err(
            TabsmithError::new(ErrorCode::ConfigInvalid, Phase::Config, format!("unknown recipe `{recipe}`"))
                .field("--recipe")
                .expected("tabular-binary, tabular-multiclass or tabular-regression")
                .next("pass one of the three recipes"),
        );
    };
    if dir.join(CONFIG_FILE).exists() {
        return Err(TabsmithError::new(
            ErrorCode::ConfigInvalid,
            Phase::Config,
            format!("`{}` already contains a {CONFIG_FILE}", dir.display()),
        )
        .next("use the existing project, or choose another directory"));
    }
    let sample_file = dir.join("data").join(match task {
        Task::Binary => "defects.csv",
        Task::Multiclass => "failure_modes.csv",
        Task::Regression => "remaining_life.csv",
    });
    if sample_data && sample_file.exists() {
        return Err(TabsmithError::new(
            ErrorCode::ConfigInvalid,
            Phase::Config,
            format!("`{}` already exists; the sample data would overwrite it", sample_file.display()),
        )
        .next("run `tabsmith init` without --sample-data, or in an empty directory"));
    }
    fs::create_dir_all(dir.join("data")).map_err(|e| TabsmithError::io(Phase::Config, dir, e))?;
    write_atomic(&dir.join(CONFIG_FILE), template(task).as_bytes())?;
    // Never overwrite user files: add the state directory to an existing .gitignore.
    let gitignore = dir.join(".gitignore");
    let existing = fs::read_to_string(&gitignore).unwrap_or_default();
    if !existing
        .lines()
        .any(|l| l.trim() == "/.tabsmith/" || l.trim() == ".tabsmith/" || l.trim() == ".tabsmith")
    {
        let mut text = existing.clone();
        if !text.is_empty() && !text.ends_with('\n') {
            text.push('\n');
        }
        text.push_str("/.tabsmith/\n");
        write_atomic(&gitignore, text.as_bytes())?;
    }
    let mut report = InitReport {
        project: dir.display().to_string(),
        config: dir.join(CONFIG_FILE).display().to_string(),
        sample_data: None,
    };
    if sample_data {
        let file = sample_file;
        crate::sample::write_sample_csv(&file, 10_000, 42, crate::sample::SampleVariant::Base, task)?;
        let ws = Workspace::open(dir)?;
        ws.bind_data(&BindRequest {
            // Absolute: never resolved against the caller's current directory.
            file: crate::project::canonical(&file).unwrap_or(file.clone()),
            target: crate::sample::sample_target(task).into(),
            positive: (task == Task::Binary).then(|| "1".into()),
            split: Some("group".into()),
            group: Some("device_id".into()),
            id: Some("unit_id".into()),
            exclude: vec![],
            task: Some(task),
        })?;
        report.sample_data = Some(file.display().to_string());
    }
    Ok(report)
}

#[derive(Debug, Clone, Default)]
pub struct BindRequest {
    pub file: PathBuf,
    pub target: String,
    pub positive: Option<String>,
    pub split: Option<String>,
    pub group: Option<String>,
    pub id: Option<String>,
    pub exclude: Vec<String>,
    /// Sets the project's recipe to this task's (`None` keeps the current one).
    pub task: Option<Task>,
}

#[derive(Debug, Clone, Serialize)]
pub struct BindReport {
    pub source: String,
    pub snapshot: DataSnapshot,
    pub target: TargetInfo,
    pub columns: Vec<ColumnDecision>,
    pub warnings: Vec<String>,
    pub split: String,
    pub split_counts: [(String, usize, usize); 3],
}

fn data_table(doc: &mut DocumentMut) -> &mut EditTable {
    // `data = { ... }` inline tables are converted (keeping their keys), not replaced.
    if let Some(Item::Value(toml_edit::Value::InlineTable(t))) = doc.get("data") {
        let table = t.clone().into_table();
        doc.insert("data", Item::Table(table));
    }
    if !doc.contains_table("data") {
        doc.insert("data", Item::Table(EditTable::new()));
    }
    doc["data"].as_table_mut().expect("data is a table")
}

/// Maps an error raised while validating `data bind` flags back to the flag.
fn as_flag_error(mut e: TabsmithError) -> TabsmithError {
    let flag = match e.field.as_deref() {
        Some("data.target") => Some("--target"),
        Some("data.positive_label") => Some("--positive"),
        Some("data.group_column") => Some("--group"),
        Some("data.id_column") => Some("--id"),
        Some("data.split") => Some("--split"),
        Some("data.exclude") => Some("--exclude"),
        _ => None,
    };
    if let Some(flag) = flag {
        e.field = Some(flag.to_string());
        if let Some(n) = &e.next_action {
            e.next_action = Some(n.replace("in tabsmith.toml", &format!("with {flag}")));
        }
    }
    e
}

/// Path written to `data.source`: relative to the project when the file is inside
/// it or at most two levels above it (the project can move together with its
/// data), absolute otherwise.
fn project_relative(root: &Path, file: &Path) -> String {
    let mut up = 0;
    let mut base = root.to_path_buf();
    loop {
        if let Ok(rest) = file.strip_prefix(&base) {
            let rest = rest.to_string_lossy().replace('\\', "/");
            return "../".repeat(up) + &rest;
        }
        if up == 2 || !base.pop() {
            return file.display().to_string();
        }
        up += 1;
    }
}

/// Removes the template's "Filled in by `tabsmith data bind`" comment block once
/// the real values are written.
fn strip_bind_hint(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut skipping = false;
    for line in text.split_inclusive('\n') {
        if line.starts_with("# Filled in by `tabsmith data bind") {
            skipping = true;
            continue;
        }
        if skipping && line.starts_with("# ") {
            continue;
        }
        skipping = false;
        out.push_str(line);
    }
    out
}

impl Workspace {
    /// Records the data source and target in `tabsmith.toml` after validating them
    /// against the data. The file is only changed if validation succeeds.
    pub fn bind_data(&self, req: &BindRequest) -> Result<BindReport> {
        let project = &self.project;
        let text = fs::read_to_string(project.config_path()).map_err(|e| TabsmithError::io(Phase::Config, project.config_path(), e))?;
        let mut doc: DocumentMut = text.parse().map_err(|e: toml_edit::TomlError| {
            TabsmithError::new(ErrorCode::ConfigInvalid, Phase::Config, format!("{CONFIG_FILE}: {e}"))
        })?;
        let abs = if req.file.is_absolute() {
            req.file.clone()
        } else {
            std::env::current_dir()
                .map_err(|e| TabsmithError::io(Phase::Config, ".", e))?
                .join(&req.file)
        };
        let abs = if abs.exists() {
            abs
        } else if project.root.join(&req.file).exists() {
            project.root.join(&req.file)
        } else {
            abs
        };
        // Resolve the directory, not the file: a symlinked file (data/current.csv ->
        // exports/2026-10.csv) stays bound by its own path, so re-pointing it works.
        let abs = match (abs.parent().and_then(|p| crate::project::canonical(p).ok()), abs.file_name()) {
            (Some(dir), Some(name)) => dir.join(name),
            _ => abs,
        };
        if !abs.is_file() {
            return Err(TabsmithError::new(
                ErrorCode::DataSourceNotFound,
                Phase::Data,
                format!("data file `{}` does not exist", req.file.display()),
            )
            .field(req.file.display().to_string())
            .next("check the path (relative paths are resolved from the current directory)"));
        }
        let root = crate::project::canonical(&project.root).unwrap_or(project.root.clone());
        let source = project_relative(&root, &abs);
        if let Some(sp) = &req.split
            && !matches!(sp.as_str(), "stratified" | "group" | "time")
        {
            return Err(
                TabsmithError::new(ErrorCode::ConfigInvalid, Phase::Config, format!("unknown split `{sp}`"))
                    .field("--split")
                    .expected("stratified (independent rows) or group (with --group <column>)"),
            );
        }
        // Only the flags given change the file; a new source or target invalidates the
        // keys that describe the old one (they are re-inferred or must be passed again).
        let old_source = doc
            .get("data")
            .and_then(|d| d.get("source"))
            .and_then(|v| v.as_str())
            .map(str::to_string);
        let old_target = doc
            .get("data")
            .and_then(|d| d.get("target"))
            .and_then(|v| v.as_str())
            .map(str::to_string);
        let rebinding = old_source.as_deref() != Some(source.as_str()) || old_target.as_deref() != Some(req.target.as_str());
        if let Some(task) = req.task {
            doc["recipe"] = value(task.recipe());
            // A metric of another task would be refused: use this task's default.
            let metric = doc
                .get("selection")
                .and_then(|s| s.get("metric"))
                .and_then(|v| v.as_str())
                .and_then(tabsmith_contracts::config::SelectionMetric::parse);
            if metric.is_some_and(|m| !m.applies_to(task))
                && let Some(sel) = doc.get_mut("selection").and_then(|s| s.as_table_like_mut())
            {
                sel.insert("metric", value(task.default_metric().as_str()));
            }
        }
        {
            let data = data_table(&mut doc);
            data["source"] = value(source.clone());
            data["target"] = value(req.target.clone());
            if old_target.as_deref() != Some(req.target.as_str()) {
                // A different target means different labels.
                data.remove("positive_label");
            }
            match (&req.split, &req.group) {
                (Some(sp), _) => data["split"] = value(sp.clone()),
                (None, Some(_)) => data["split"] = value("group"),
                (None, None) => {}
            }
            if let Some(p) = &req.positive {
                data["positive_label"] = value(p.clone());
            }
            if let Some(g) = &req.group {
                data["group_column"] = value(g.clone());
            }
            if let Some(i) = &req.id {
                data["id_column"] = value(i.clone());
            }
            if !req.exclude.is_empty() {
                let mut arr = match data.get("exclude").and_then(|i| i.as_array()) {
                    Some(a) => a.clone(),
                    None => Array::new(),
                };
                for c in &req.exclude {
                    if !arr.iter().any(|v| v.as_str() == Some(c)) {
                        arr.push(c.as_str());
                    }
                }
                data["exclude"] = value(arr);
            }
        }
        // Validate: config, snapshot, positive label, columns, split.
        let resolve_doc = |doc: &DocumentMut| -> Result<tabsmith_contracts::config::ResolvedConfig> {
            let pf = ProjectFile::parse(&doc.to_string(), CONFIG_FILE)?;
            resolve(&[pf.project])
        };
        let mut resolved = resolve_doc(&doc).map_err(as_flag_error)?;
        let mut snap = snapshot_source(project, &resolved.config.data).map_err(as_flag_error)?;
        let mut kept_notes = Vec::new();
        if rebinding {
            // Keys that name columns of the previous file are kept only if the new file
            // has those columns; the report lists what was kept or dropped.
            let has = |c: &str| snap.snapshot.column(c).is_some();
            let data = data_table(&mut doc);
            for key in ["id_column", "group_column", "time_column"] {
                let given = (key == "id_column" && req.id.is_some()) || (key == "group_column" && req.group.is_some());
                if let Some(col) = data.get(key).and_then(|v| v.as_str()).map(str::to_string) {
                    if given {
                        continue;
                    }
                    if has(&col) {
                        kept_notes.push(format!("kept data.{key} = \"{col}\" from the previous binding"));
                    } else {
                        data.remove(key);
                        kept_notes.push(format!("dropped data.{key} = \"{col}\" (no such column in the new file)"));
                        if key == "group_column" && data.get("split").and_then(|v| v.as_str()) == Some("group") && req.split.is_none() {
                            data.remove("split");
                            kept_notes.push("split reset to the default (stratified) because the group column is gone".into());
                        }
                    }
                }
            }
            if let Some(arr) = data.get("exclude").and_then(|v| v.as_array()).cloned() {
                let mut kept = toml_edit::Array::new();
                for v in arr.iter() {
                    if let Some(c) = v.as_str() {
                        if has(c) {
                            kept.push(c);
                        } else {
                            kept_notes.push(format!("dropped `{c}` from data.exclude (no such column)"));
                        }
                    }
                }
                data["exclude"] = value(kept);
            }
            if let Some(cols) = data
                .get("columns")
                .and_then(|v| v.as_table_like())
                .map(|t| t.iter().map(|(k, _)| k.to_string()).collect::<Vec<_>>())
            {
                for c in cols {
                    if !has(&c) {
                        if let Some(t) = data.get_mut("columns").and_then(|v| v.as_table_like_mut()) {
                            t.remove(&c);
                        }
                        kept_notes.push(format!("dropped data.columns.{c} (no such column)"));
                    }
                }
            }
            resolved = resolve_doc(&doc).map_err(as_flag_error)?;
            // The snapshot identity depends on the id column: take it again.
            snap = snapshot_source(project, &resolved.config.data).map_err(as_flag_error)?;
        }
        if resolved.config.task() != Task::Binary {
            // Only a binary target has a positive label.
            if data_table(&mut doc).remove("positive_label").is_some() {
                resolved = resolve_doc(&doc).map_err(as_flag_error)?;
            }
        } else if resolved.config.data.positive_label.is_none() {
            let positive = infer_positive(&snap.snapshot, &snap.table, &req.target)?;
            data_table(&mut doc)["positive_label"] = value(positive);
            resolved = resolve_doc(&doc).map_err(as_flag_error)?;
        }
        let cfg = &resolved.config;
        let features =
            plan_columns(&snap.snapshot, &snap.table, &cfg.data, cfg.task(), cfg.preprocess.max_categories).map_err(as_flag_error)?;
        let split = make_split(&snap.snapshot.snapshot_id, &snap.table, &features.target, &cfg.data).map_err(as_flag_error)?;
        let earlier: Vec<&String> = cfg.data.exclude.iter().filter(|c| !req.exclude.contains(c)).collect();
        if !earlier.is_empty() {
            kept_notes.push(format!(
                "data.exclude is {:?}: {:?} kept from the previous binding (--exclude adds to the list; edit data.exclude in tabsmith.toml to remove entries)",
                cfg.data.exclude, earlier
            ));
        }
        write_atomic(&project.config_path(), strip_bind_hint(&doc.to_string()).as_bytes())?;
        Ok(BindReport {
            source,
            snapshot: snap.snapshot,
            target: features.target.clone(),
            columns: features.columns.clone(),
            warnings: kept_notes
                .iter()
                .chain(&features.warnings)
                .chain(&split.warnings)
                .cloned()
                .collect(),
            split: format!("{} (seed {})", split.strategy, split.seed),
            split_counts: [
                ("train".into(), split.train.rows, split.train.positives),
                ("valid".into(), split.valid.rows, split.valid.positives),
                ("test".into(), split.test.rows, split.test.positives),
            ],
        })
    }

    /// Everything that `tabsmith run` would do, without training.
    pub fn plan(&self, req: &ConfigRequest) -> Result<PlanReport> {
        let resolved = self.resolve(req)?;
        let effective = resolved.effective_values();
        let p = self.prepare(resolved, false)?;
        let cfg = &p.resolved.config;
        let candidates = cfg.selection.candidates.clone();
        let device_plans = self.device_plans(&p, &candidates)?;
        let features = p
            .fitted
            .transform
            .features
            .iter()
            .map(|f| match f {
                FeatureTransform::Numeric {
                    name, impute, mean, std, ..
                } => PlannedFeature {
                    name: name.clone(),
                    encoding: format!("numeric: missing -> {impute:.4}, z = (x - {mean:.4}) / {std:.4}"),
                    width: 1,
                },
                FeatureTransform::Categorical {
                    name,
                    categories,
                    missing_slot,
                    ..
                } => PlannedFeature {
                    name: name.clone(),
                    encoding: format!(
                        "one-hot over {} categories{}; unseen -> {}",
                        categories.len(),
                        if *missing_slot { " + missing" } else { "" },
                        match cfg.preprocess.unknown_category {
                            tabsmith_contracts::config::UnknownCategoryPolicy::Ignore => "all zeros",
                            tabsmith_contracts::config::UnknownCategoryPolicy::Error => "error",
                        }
                    ),
                    width: f.width(),
                },
            })
            .collect();
        let mut warnings = p.features.warnings.clone();
        warnings.extend(p.split.warnings.iter().cloned());
        warnings.extend(p.snapshot.limitations.iter().cloned());
        warnings.extend(self.data_change_note(&p.snapshot));
        for d in &p.fitted.transform.dropped {
            warnings.push(format!("feature `{}` dropped: {}", d.name, d.reason));
        }
        if cfg.data.split == SplitStrategy::Stratified {
            warnings.push(
                "split = stratified assumes independent rows; if rows repeat an entity (device, user), set data.group_column and split = \"group\""
                    .into(),
            );
        }
        let planned_candidates = candidates
            .iter()
            .map(|&c| {
                let t = cfg.train.for_candidate(c);
                let m = model_config(c, p.fitted.transform.output_dim, t);
                PlannedCandidate {
                    candidate: c,
                    parameters: m.parameter_count(),
                    hidden: t.hidden.clone(),
                    epochs: t.epochs,
                    batch_size: t.batch_size,
                    learning_rate: t.learning_rate,
                    l2: t.l2,
                    class_weight: format!("{:?}", t.class_weight).to_lowercase(),
                    steps: (p.split.train.rows.div_ceil(t.batch_size) * t.epochs) as u64,
                }
            })
            .collect();
        Ok(PlanReport {
            config_hash: p.resolved.config_hash.clone(),
            effective,
            snapshot: p.snapshot.clone(),
            columns: p.features.columns.clone(),
            target: p.features.target.clone(),
            split_id: p.split.split_id.clone(),
            split_strategy: p.split.strategy.to_string(),
            split_counts: [
                ("train".into(), p.split.train.rows, p.split.train.positives),
                ("valid".into(), p.split.valid.rows, p.split.valid.positives),
                ("test".into(), p.split.test.rows, p.split.test.positives),
            ],
            transform_id: p.fitted.transform_id.clone(),
            feature_dim: p.fitted.transform.output_dim,
            features,
            candidates: planned_candidates,
            devices: device_plans.into_iter().map(|(c, planned)| (c, planned.plan)).collect(),
            selection: format!(
                "rank by validation {} at threshold {}; test split reserved for the final candidate",
                cfg.metric().as_str(),
                cfg.selection.threshold
            ),
            reused: p.reused.clone(),
            warnings,
        })
    }
}

fn infer_positive(snapshot: &DataSnapshot, table: &tabsmith_runtime::Table, target: &str) -> Result<String> {
    let Some(col) = table.column(target) else {
        let names: Vec<&str> = snapshot.columns.iter().map(|c| c.name.as_str()).collect();
        let mut e = TabsmithError::new(
            ErrorCode::SchemaMissingColumn,
            Phase::Data,
            format!("target column `{target}` is not in the data"),
        )
        .field("--target")
        .actual(format!("columns: {}", names.join(", ")));
        if let Some(s) = crate::project::did_you_mean(target, names.iter().copied()) {
            e = e.next(format!("did you mean `{s}`?"));
        }
        return Err(e);
    };
    let mut labels = std::collections::BTreeSet::new();
    for i in 0..table.n_rows() {
        if let Ok(Some(l)) = tabsmith_runtime::coerce_category(col.get(i), tabsmith_contracts::config::NanPolicy::Missing) {
            labels.insert(l.into_owned());
            if labels.len() > 20 {
                break;
            }
        }
    }
    let v: Vec<&str> = labels.iter().map(String::as_str).collect();
    // Two labels that read as negative/positive in any letter case: 0/1, false/true, no/yes, n/y.
    let guess = match v.as_slice() {
        [a, b] => {
            let lower = |s: &str| s.to_ascii_lowercase();
            let positive = |s: &str| matches!(lower(s).as_str(), "1" | "true" | "yes" | "y");
            let negative = |s: &str| matches!(lower(s).as_str(), "0" | "false" | "no" | "n");
            if negative(a) && positive(b) {
                Some(*b)
            } else if positive(a) && negative(b) {
                Some(*a)
            } else {
                None
            }
        }
        _ => None,
    };
    guess.map(str::to_string).ok_or_else(|| {
        TabsmithError::new(
            ErrorCode::TargetInvalid,
            Phase::Data,
            format!("cannot tell which value of `{target}` is the positive class"),
        )
        .field("--positive")
        .actual(format!("labels: {:?}", v.iter().take(10).collect::<Vec<_>>()))
        .next("pass --positive <label> (the class you want to detect)")
    })
}

#[derive(Debug, Clone, Serialize)]
pub struct PlannedFeature {
    pub name: String,
    pub encoding: String,
    pub width: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct PlannedCandidate {
    pub candidate: Candidate,
    pub parameters: usize,
    pub hidden: Vec<usize>,
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub l2: f64,
    pub class_weight: String,
    pub steps: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct PlanReport {
    pub config_hash: String,
    pub effective: Vec<(String, String, ValueSource)>,
    pub snapshot: DataSnapshot,
    pub columns: Vec<ColumnDecision>,
    pub target: TargetInfo,
    pub split_id: String,
    pub split_strategy: String,
    pub split_counts: [(String, usize, usize); 3],
    pub transform_id: String,
    pub feature_dim: usize,
    pub features: Vec<PlannedFeature>,
    pub candidates: Vec<PlannedCandidate>,
    pub devices: Vec<(Candidate, DevicePlan)>,
    pub selection: String,
    pub reused: Vec<String>,
    pub warnings: Vec<String>,
}

impl PlanReport {
    pub fn excluded(&self) -> Vec<(&str, &str)> {
        self.columns
            .iter()
            .filter_map(|c| match &c.role {
                ColumnRole::Excluded { reason } => Some((c.name.as_str(), reason.as_str())),
                _ => None,
            })
            .collect()
    }
}

/// SDK helper: an override layer from `key=value` strings.
pub fn override_layer(pairs: &[&str]) -> Result<ConfigLayer> {
    let mut table = toml::Table::new();
    for p in pairs {
        let t = tabsmith_contracts::config::parse_override(p)?;
        for (k, v) in t {
            merge(&mut table, k, v);
        }
    }
    Ok(ConfigLayer {
        source: ValueSource::Override,
        table,
    })
}

fn merge(t: &mut toml::Table, k: String, v: toml::Value) {
    match (t.get_mut(&k), v) {
        (Some(toml::Value::Table(a)), toml::Value::Table(b)) => {
            for (k2, v2) in b {
                merge(a, k2, v2);
            }
        }
        (_, v) => {
            t.insert(k, v);
        }
    }
}
