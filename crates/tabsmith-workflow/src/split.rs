//! Deterministic train/valid/test splits, fixed before any statistic is fitted.

use std::collections::{BTreeMap, BTreeSet};

use tabsmith_contracts::config::{DataConfig, SplitStrategy, Task};
use tabsmith_contracts::data::{SPLIT_TEST, SPLIT_TRAIN, SPLIT_VALID, SplitCounts, SplitManifest, SplitName};
use tabsmith_contracts::hash::{canonical_hash, short};
use tabsmith_contracts::{ErrorCode, Phase, Result, TabsmithError};
use tabsmith_runtime::Table;

use crate::data::{TargetInfo, Targets, targets};
use crate::exposure::{content_columns, entity_key, row_fingerprint};
use crate::rng::SplitMix64;

/// Version of the positional stratified split algorithm (no longer produced; kept
/// so older split manifests stay readable).
pub const SPLIT_ALGORITHM: &str = "tabsmith.split.v1";
/// Stratified split without an id column: rows are ordered by a hash of their
/// full content (every column of the file) plus their occurrence number among
/// identical rows, so a row keeps its split when data is appended or re-ordered,
/// and changing which columns are features does not re-split the data.
pub const SPLIT_ALGORITHM_CONTENT: &str = "tabsmith.split.v3-content-hash";
/// Version of the stable, key-hashed assignment (grouped splits, or any split with
/// an id column): a row/group keeps its split when the data grows or is re-ordered.
/// Keys are [`entity_key`]s, so `00000001` and `1` are the same entity (v4; v2
/// hashed the key text as written).
pub const SPLIT_ALGORITHM_STABLE: &str = "tabsmith.split.v4-stable-key";
// Stratified with an id: per-class ordering by the stable hash, then exact cuts.
/// Chronological split: rows ordered by the time column, the earliest `train_fraction`
/// for training, then validation, the latest for test; rows with the same time stay
/// together.
pub const SPLIT_ALGORITHM_TIME: &str = "tabsmith.split.v1-time";

/// Deterministic value in [0, 1) from (seed, domain, key): FNV-1a + SplitMix64 finaliser.
pub fn stable_unit(seed: u64, domain: &str, key: &str) -> f64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for b in domain.as_bytes().iter().chain([0u8].iter()).chain(key.as_bytes()) {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    let mut z = h ^ seed.wrapping_mul(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^= z >> 31;
    (z >> 11) as f64 / (1u64 << 53) as f64
}

fn by_hash(u: f64, fractions: &[f64; 3]) -> u8 {
    if u < fractions[2] {
        SPLIT_TEST
    } else if u < fractions[2] + fractions[1] {
        SPLIT_VALID
    } else {
        SPLIT_TRAIN
    }
}

/// The rows of each stratum, in a fixed order: the negative then the positive
/// class (binary; the order, labels and seeds of earlier versions, so binary
/// splits are unchanged), the classes in order (multiclass), or up to ten
/// quantile bins of the target (regression).
fn strata(t: &Targets, target: &TargetInfo) -> Vec<(String, Vec<usize>)> {
    match t {
        Targets::Binary(y) => [false, true]
            .into_iter()
            .map(|class| {
                let label = if class { &target.positive_label } else { &target.negative_label };
                (label.clone(), (0..y.len()).filter(|&i| y[i] == class).collect())
            })
            .collect(),
        Targets::Classes { y, k } => (0..*k)
            .map(|c| (target.classes[c].clone(), (0..y.len()).filter(|&i| y[i] as usize == c).collect()))
            .collect(),
        Targets::Values(y) => {
            let mut sorted = y.clone();
            sorted.sort_by(f64::total_cmp);
            let bins = (y.len() / 10).clamp(1, 10);
            let cuts: Vec<f64> = (1..bins).map(|b| sorted[b * y.len() / bins]).collect();
            let mut rows = vec![Vec::new(); bins];
            for (i, &v) in y.iter().enumerate() {
                rows[cuts.partition_point(|&c| c <= v)].push(i);
            }
            rows.into_iter()
                .enumerate()
                .filter(|(_, r)| !r.is_empty())
                .map(|(b, r)| (format!("target quantile bin {} of {bins}", b + 1), r))
                .collect()
        }
    }
}

/// A time split needs its column, and a declared time column needs a time split:
/// Tabsmith never falls back to a random split for time-dependent data.
fn check_time_config(cfg: &DataConfig) -> Result<()> {
    match (cfg.split, &cfg.time_column) {
        (SplitStrategy::Time, None) => Err(TabsmithError::new(
            ErrorCode::ConfigInvalid,
            Phase::Split,
            "split = \"time\" needs data.time_column",
        )
        .field("data.time_column")
        .next("set data.time_column to the column that orders the rows (a timestamp, a date, or a counter), or pass --time <column>")),
        (split, Some(_)) if split != SplitStrategy::Time => Err(TabsmithError::new(
            ErrorCode::SplitUnsupported,
            Phase::Split,
            "data.time_column is declared but data.split is not \"time\": a random split would put future rows into training",
        )
        .field("data.split")
        .expected("time")
        .actual(split.to_string())
        .next("set data.split = \"time\" (or pass --time <column>) for a chronological split, or remove data.time_column if rows are truly independent")),
        _ => Ok(()),
    }
}

/// How a time column orders rows: numbers (an epoch, a counter) or text (ISO 8601
/// dates sort as text). One column holds one kind.
#[derive(Debug, Clone, PartialEq)]
enum TimeKey {
    Number(f64),
    Text(String),
}

impl TimeKey {
    fn of(v: tabsmith_runtime::ValueRef<'_>) -> Option<Self> {
        use tabsmith_runtime::ValueRef;
        match v {
            ValueRef::Float(f) if f.is_finite() => Some(TimeKey::Number(f)),
            ValueRef::Int(i) => Some(TimeKey::Number(i as f64)),
            ValueRef::Text(t) if !t.trim().is_empty() => Some(TimeKey::Text(t.trim().to_string())),
            _ => None,
        }
    }

    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match (self, other) {
            (TimeKey::Number(a), TimeKey::Number(b)) => a.total_cmp(b),
            (TimeKey::Text(a), TimeKey::Text(b)) => a.cmp(b),
            (TimeKey::Number(_), TimeKey::Text(_)) => std::cmp::Ordering::Less,
            (TimeKey::Text(_), TimeKey::Number(_)) => std::cmp::Ordering::Greater,
        }
    }
}

/// Computes the split manifest for a snapshot.
pub fn make_split(snapshot_id: &str, table: &Table, target: &TargetInfo, cfg: &DataConfig) -> Result<SplitManifest> {
    check_time_config(cfg)?;
    if cfg.group_column.is_some() && cfg.split == SplitStrategy::Stratified {
        return Err(TabsmithError::new(
            ErrorCode::LeakageDetected,
            Phase::Split,
            "data.group_column is declared but data.split is not \"group\": rows of one entity could land in both training and test",
        )
        .field("data.split")
        .expected("group")
        .actual(cfg.split.to_string())
        .next("set data.split = \"group\" (or remove data.group_column if rows are independent)"));
    }
    let t = targets(table, target)?;
    let n = t.len();
    let strata = strata(&t, target);
    let fractions = [cfg.train_fraction, cfg.valid_fraction, cfg.test_fraction];
    let mut assignment = vec![SPLIT_TRAIN; n];
    let mut group_counts = None;
    let mut warnings = Vec::new();
    let id_column = cfg.id_column.as_deref().and_then(|c| table.column(c));
    // Without an id, a row is keyed by all of its columns (sorted by name, so
    // column order does not matter) plus its occurrence number among identical
    // rows: duplicates are spread like any other rows, and the key does not depend
    // on which columns are features.
    let keyed_by: Vec<String> = if id_column.is_none() {
        let mut names: Vec<String> = table.names().into_iter().map(str::to_string).collect();
        names.sort();
        names
    } else {
        Vec::new()
    };
    let content_keys: Option<Vec<String>> = content_columns(table, &keyed_by).map(|cols| {
        let mut seen: std::collections::HashMap<u64, usize> = std::collections::HashMap::new();
        (0..n)
            .map(|i| {
                let fp = row_fingerprint(&cols, i);
                let k = seen.entry(fp).or_insert(0);
                let key = format!("{fp:016x}#{k}");
                *k += 1;
                key
            })
            .collect()
    });
    let row_key = |i: usize| -> String {
        match (id_column, &content_keys) {
            (Some(ids), _) => entity_key(ids.get(i)).unwrap_or_default(),
            (None, Some(keys)) => keys[i].clone(),
            (None, None) => i.to_string(),
        }
    };
    match (cfg.split, id_column.is_some() || content_keys.is_some()) {
        (SplitStrategy::Stratified, true) => {
            // Stratified and stable: within each class, rows are ordered by a hash of
            // (seed, row key) and cut at the configured fractions. The row key is the
            // id, or the row's content without one. Re-ordering the file never
            // changes the split; appending data moves at most a few rows near the cuts
            // (the row-level evaluation checks catch any row that moves).
            for (label, rows) in &strata {
                let mut idx: Vec<(f64, usize)> = rows.iter().map(|&i| (stable_unit(cfg.seed, "row", &row_key(i)), i)).collect();
                idx.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
                let nc = idx.len();
                let mut n_test = (nc as f64 * cfg.test_fraction).round() as usize;
                let mut n_valid = (nc as f64 * cfg.valid_fraction).round() as usize;
                if nc >= 3 {
                    n_test = n_test.max(1);
                    n_valid = n_valid.max(1);
                }
                // Never cut between rows with the same key: identical rows share a split.
                let extend = |mut cut: usize| {
                    while cut > 0 && cut < nc && idx[cut].0 == idx[cut - 1].0 {
                        cut += 1;
                    }
                    cut
                };
                n_test = extend(n_test);
                n_valid = extend(n_test + n_valid).saturating_sub(n_test);
                if n_test + n_valid >= nc {
                    return Err(degenerate(target.task, label, nc));
                }
                for &(_, i) in &idx[..n_test] {
                    assignment[i] = SPLIT_TEST;
                }
                for &(_, i) in &idx[n_test..n_test + n_valid] {
                    assignment[i] = SPLIT_VALID;
                }
            }
        }
        (SplitStrategy::Stratified, false) => {
            for (label, rows) in &strata {
                let mut idx = rows.clone();
                SplitMix64::derive(cfg.seed, &format!("stratified/{label}")).shuffle(&mut idx);
                let nc = idx.len();
                let mut n_test = (nc as f64 * cfg.test_fraction).round() as usize;
                let mut n_valid = (nc as f64 * cfg.valid_fraction).round() as usize;
                if nc >= 3 {
                    n_test = n_test.max(1);
                    n_valid = n_valid.max(1);
                }
                if n_test + n_valid >= nc {
                    return Err(degenerate(target.task, label, nc));
                }
                for &i in &idx[..n_test] {
                    assignment[i] = SPLIT_TEST;
                }
                for &i in &idx[n_test..n_test + n_valid] {
                    assignment[i] = SPLIT_VALID;
                }
            }
        }
        (SplitStrategy::Group, _) => {
            let gname = cfg.group_column.as_deref().ok_or_else(|| {
                TabsmithError::new(ErrorCode::ConfigInvalid, Phase::Split, "split = \"group\" needs data.group_column")
                    .field("data.group_column")
                    .next("set data.group_column to the entity column (e.g. device_id)")
            })?;
            let col = table.column(gname).ok_or_else(|| {
                TabsmithError::new(
                    ErrorCode::SchemaMissingColumn,
                    Phase::Split,
                    format!("group column `{gname}` not found"),
                )
                .field("data.group_column")
            })?;
            let mut rows_by_group: BTreeMap<String, Vec<usize>> = BTreeMap::new();
            for i in 0..n {
                let key = entity_key(col.get(i)).ok_or_else(|| {
                    TabsmithError::new(
                        ErrorCode::DataUnsupported,
                        Phase::Split,
                        format!("group column `{gname}` is missing or invalid in row {i}"),
                    )
                    .field("data.group_column")
                    .next("every row needs a group id for a grouped split")
                })?;
                rows_by_group.entry(key).or_default().push(i);
            }
            if rows_by_group.len() < 3 {
                return Err(TabsmithError::new(
                    ErrorCode::SplitDegenerate,
                    Phase::Split,
                    format!("only {} groups in `{gname}`; a grouped split needs at least 3", rows_by_group.len()),
                )
                .field("data.group_column"));
            }
            // Stable assignment by group key: a group keeps its split across data versions.
            let mut counts = [0usize; 3];
            for (key, rows) in &rows_by_group {
                let code = by_hash(stable_unit(cfg.seed, "group", key), &fractions);
                counts[(code - SPLIT_TRAIN) as usize] += 1;
                for &i in rows {
                    assignment[i] = code;
                }
            }
            group_counts = Some(counts);
        }
        (SplitStrategy::Time, _) => {
            let tname = cfg.time_column.as_deref().expect("checked above");
            let col = table.column(tname).ok_or_else(|| {
                TabsmithError::new(
                    ErrorCode::SchemaMissingColumn,
                    Phase::Split,
                    format!("time column `{tname}` not found"),
                )
                .field("data.time_column")
            })?;
            let mut keys = Vec::with_capacity(n);
            for i in 0..n {
                let key = TimeKey::of(col.get(i)).ok_or_else(|| {
                    TabsmithError::new(
                        ErrorCode::DataUnsupported,
                        Phase::Split,
                        format!("time column `{tname}` is missing or not a time in row {i}"),
                    )
                    .field("data.time_column")
                    .next("every row needs a time for a chronological split: a number or a date as text (ISO 8601 sorts as text)")
                })?;
                keys.push(key);
            }
            if keys.iter().any(|k| matches!(k, TimeKey::Number(_))) && keys.iter().any(|k| matches!(k, TimeKey::Text(_))) {
                return Err(TabsmithError::new(
                    ErrorCode::DataUnsupported,
                    Phase::Split,
                    format!("time column `{tname}` mixes numbers and text"),
                )
                .field("data.time_column")
                .next("use one kind of time: numbers (an epoch, a counter) or dates as text"));
            }
            // Earliest rows train, the latest are the test; the same time never
            // straddles a cut, so no row is predicted from rows of its own moment.
            let mut idx: Vec<usize> = (0..n).collect();
            idx.sort_by(|&a, &b| keys[a].cmp(&keys[b]).then(a.cmp(&b)));
            let mut n_train = (n as f64 * cfg.train_fraction).round() as usize;
            let mut n_valid = (n as f64 * cfg.valid_fraction).round() as usize;
            if n >= 3 {
                n_train = n_train.max(1);
                n_valid = n_valid.max(1);
            }
            let extend = |mut cut: usize| {
                while cut > 0 && cut < n && keys[idx[cut]] == keys[idx[cut - 1]] {
                    cut += 1;
                }
                cut
            };
            n_train = extend(n_train);
            n_valid = extend(n_train + n_valid).saturating_sub(n_train);
            if n_train + n_valid >= n || n_train == 0 || n_valid == 0 {
                return Err(TabsmithError::new(
                    ErrorCode::SplitDegenerate,
                    Phase::Split,
                    format!(
                        "{n} rows at {} distinct times: not enough for train, validation and test in time order",
                        {
                            let mut d = keys.clone();
                            d.sort_by(TimeKey::cmp);
                            d.dedup();
                            d.len()
                        }
                    ),
                )
                .field("data.time_column")
                .next("add more rows, or use finer times"));
            }
            for &i in &idx[n_train..n_train + n_valid] {
                assignment[i] = SPLIT_VALID;
            }
            for &i in &idx[n_train + n_valid..] {
                assignment[i] = SPLIT_TEST;
            }
            warnings.push(format!(
                "chronological split on `{tname}`: the split moves as the data grows (later rows push earlier test rows into validation and training), and the classes are not balanced across splits"
            ));
            if let Some(g) = &cfg.group_column {
                warnings.push(format!(
                    "rows of one `{g}` can be in several splits, ordered by time: the test rows of an entity come after its training rows, as in a forecast"
                ));
            }
        }
    }
    // Hash-based group assignment only approximates the fractions; check how far off it is.
    if cfg.split == SplitStrategy::Group {
        let actual: Vec<f64> = [SPLIT_TRAIN, SPLIT_VALID, SPLIT_TEST]
            .iter()
            .map(|c| assignment.iter().filter(|a| *a == c).count() as f64 / n.max(1) as f64)
            .collect();
        let names = ["train", "valid", "test"];
        let shown = format!(
            "train {:.1}% / valid {:.1}% / test {:.1}% (configured {:.0}% / {:.0}% / {:.0}%)",
            actual[0] * 100.0,
            actual[1] * 100.0,
            actual[2] * 100.0,
            fractions[0] * 100.0,
            fractions[1] * 100.0,
            fractions[2] * 100.0
        );
        if actual[0] < fractions[0] * 0.5 || actual[1] == 0.0 || actual[2] == 0.0 {
            return Err(TabsmithError::new(
                ErrorCode::SplitDegenerate,
                Phase::Split,
                format!("the split is far from the configured fractions: {shown}"),
            )
            .field("data.split")
            .next("with few or very unequal groups a grouped split cannot match the fractions; add groups/rows or try another data.seed"));
        }
        for k in 0..3 {
            if (actual[k] - fractions[k]).abs() > 0.05 {
                warnings.push(format!(
                    "the {} split differs from its configured fraction by more than 5 points: {shown}",
                    names[k]
                ));
                break;
            }
        }
    }
    let counts = |code: u8, g: Option<usize>| {
        let rows: Vec<usize> = (0..n).filter(|&i| assignment[i] == code).collect();
        let (positives, negatives, classes) = match &t {
            Targets::Binary(y) => {
                let p = rows.iter().filter(|&&i| y[i]).count();
                (p, rows.len() - p, Vec::new())
            }
            Targets::Classes { y, k } => {
                let mut c = vec![0usize; *k];
                for &i in &rows {
                    c[y[i] as usize] += 1;
                }
                (0, 0, c)
            }
            Targets::Values(_) => (0, 0, Vec::new()),
        };
        SplitCounts {
            rows: rows.len(),
            positives,
            negatives,
            classes,
            groups: g,
        }
    };
    let train = counts(SPLIT_TRAIN, group_counts.map(|c| c[0]));
    let valid = counts(SPLIT_VALID, group_counts.map(|c| c[1]));
    let test = counts(SPLIT_TEST, group_counts.map(|c| c[2]));
    for (name, c) in [("train", &train), ("valid", &valid), ("test", &test)] {
        match target.task {
            Task::Binary if c.positives == 0 || c.negatives == 0 => {
                return Err(TabsmithError::new(
                    ErrorCode::SplitDegenerate,
                    Phase::Split,
                    format!(
                        "the {name} split has {} positive and {} negative rows; every split needs both classes",
                        c.positives, c.negatives
                    ),
                )
                .field("data.split")
                .next("use more data, change the fractions, or (for grouped splits) check that groups contain both classes"));
            }
            Task::Multiclass => {
                for (k, &count) in c.classes.iter().enumerate() {
                    if count > 0 {
                        continue;
                    }
                    if name == "train" {
                        return Err(TabsmithError::new(
                            ErrorCode::SplitDegenerate,
                            Phase::Split,
                            format!("class {:?} has no training rows", target.classes[k]),
                        )
                        .field("data.split")
                        .next("add rows of this class or merge it with a related class (with a grouped split, its rows may all be in one group)"));
                    }
                    warnings.push(format!(
                        "class {:?} has no {name} rows: metrics on that split average over the other classes",
                        target.classes[k]
                    ));
                }
            }
            Task::Regression if c.rows < 2 => {
                return Err(TabsmithError::new(
                    ErrorCode::SplitDegenerate,
                    Phase::Split,
                    format!("the {name} split has {} rows", c.rows),
                )
                .field("data.split")
                .next("use more data or change the fractions"));
            }
            _ => {}
        }
    }
    let assignment = String::from_utf8(assignment).expect("ascii digits");
    let algorithm = if cfg.split == SplitStrategy::Time {
        SPLIT_ALGORITHM_TIME
    } else if cfg.split == SplitStrategy::Group || id_column.is_some() {
        SPLIT_ALGORITHM_STABLE
    } else if content_keys.is_some() {
        SPLIT_ALGORITHM_CONTENT
    } else {
        SPLIT_ALGORITHM
    };
    let base = (
        snapshot_id,
        cfg.split,
        cfg.seed,
        fractions,
        &target.name,
        &target.positive_label,
        &cfg.group_column,
        &cfg.id_column,
        algorithm,
    );
    // The key columns are part of the identity only for the content-keyed split, and
    // the time column only for the time split (id/group split ids stay what they were).
    let identity = if algorithm == SPLIT_ALGORITHM_CONTENT {
        canonical_hash(&(base, &keyed_by))
    } else if algorithm == SPLIT_ALGORITHM_TIME {
        canonical_hash(&(base, &cfg.time_column))
    } else {
        canonical_hash(&base)
    };
    let manifest = SplitManifest {
        split_id: format!("sp-{}", short(&identity)),
        snapshot_id: snapshot_id.to_string(),
        strategy: cfg.split,
        seed: cfg.seed,
        fractions,
        target: target.name.clone(),
        positive_label: target.positive_label.clone(),
        group_column: cfg.group_column.clone(),
        time_column: if cfg.split == SplitStrategy::Time {
            cfg.time_column.clone()
        } else {
            None
        },
        assignment,
        train,
        valid,
        test,
        algorithm: algorithm.to_string(),
        warnings,
    };
    debug_assert!(manifest.rows(SplitName::Train).len() == manifest.train.rows);
    Ok(manifest)
}

fn degenerate(task: Task, label: &str, n: usize) -> TabsmithError {
    let what = if task == Task::Regression {
        label.to_string()
    } else {
        format!("class {label:?}")
    };
    TabsmithError::new(
        ErrorCode::SplitDegenerate,
        Phase::Split,
        format!("{what} has only {n} rows: not enough for train, validation and test"),
    )
    .next("add more rows (at least 3 per class; realistically many more)")
}

/// Verifies that no group appears in two splits (used by tests and `plan`).
pub fn check_group_leakage(table: &Table, split: &SplitManifest) -> Result<()> {
    let Some(g) = &split.group_column else { return Ok(()) };
    let col = table.column(g).expect("group column exists");
    let mut by_group: BTreeMap<String, BTreeSet<u8>> = BTreeMap::new();
    for (i, code) in split.assignment.bytes().enumerate() {
        if let Some(k) = entity_key(col.get(i)) {
            by_group.entry(k).or_default().insert(code);
        }
    }
    if let Some((g, _)) = by_group.iter().find(|(_, s)| s.len() > 1) {
        return Err(TabsmithError::new(
            ErrorCode::LeakageDetected,
            Phase::Split,
            format!("group {g:?} appears in more than one split"),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tabsmith_runtime::{Column, ColumnData, Value};

    fn table(n: usize) -> (Table, TargetInfo) {
        let y: Vec<Option<i64>> = (0..n).map(|i| Some(i64::from(i % 5 == 0))).collect();
        let g: Vec<Option<String>> = (0..n).map(|i| Some(format!("dev{}", i / 4))).collect();
        let t = Table::new(vec![
            Column {
                name: "y".into(),
                data: ColumnData::Int(y),
            },
            Column {
                name: "g".into(),
                data: ColumnData::Text(g),
            },
        ])
        .unwrap();
        let info = TargetInfo::binary("y", "1", "0", n / 5, n - n / 5);
        (t, info)
    }

    #[test]
    fn stratified_is_deterministic_and_balanced() {
        let (t, info) = table(1000);
        let cfg = DataConfig::default();
        let a = make_split("s-1", &t, &info, &cfg).unwrap();
        let b = make_split("s-1", &t, &info, &cfg).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.train.rows + a.valid.rows + a.test.rows, 1000);
        assert_eq!(a.test.positives, 30);
        assert_eq!(a.valid.positives, 30);
        let mut other = cfg.clone();
        other.seed = 43;
        assert_ne!(make_split("s-1", &t, &info, &other).unwrap().assignment, a.assignment);
    }

    #[test]
    fn content_keyed_split_is_stable_under_append_and_reorder() {
        let build = |xs: &[i64]| {
            let y: Vec<Option<i64>> = xs.iter().map(|x| Some(i64::from(x % 5 == 0))).collect();
            Table::new(vec![
                Column {
                    name: "x".into(),
                    data: ColumnData::Int(xs.iter().map(|&x| Some(x)).collect()),
                },
                Column {
                    name: "y".into(),
                    data: ColumnData::Int(y),
                },
            ])
            .unwrap()
        };
        let (_, info) = table(10);
        let cfg = DataConfig::default();
        let base: Vec<i64> = (0..2000).collect();
        let s1 = make_split("s-1", &build(&base), &info, &cfg).unwrap();
        assert!(s1.algorithm.contains("content"), "{}", s1.algorithm);
        assert_eq!(s1.test.rows, 300);
        let a1 = s1.assignment.as_bytes();
        // Reversed and extended by 25%: almost every original row keeps its split.
        let mut grown: Vec<i64> = (0..2500).collect();
        grown.reverse();
        let s2 = make_split("s-2", &build(&grown), &info, &cfg).unwrap();
        let a2 = s2.assignment.as_bytes();
        let moved = (0..2000).filter(|&x| a1[x] != a2[2499 - x]).count();
        assert!(moved < 60, "{moved} of 2000 rows changed split");
    }

    #[test]
    fn duplicate_heavy_data_keeps_fractions_and_strata() {
        // Four distinct rows repeated 250 times each (a/b x 0/1 labels).
        let n = 1000;
        let a: Vec<Option<String>> = (0..n).map(|i| Some(if i % 2 == 0 { "x" } else { "y" }.to_string())).collect();
        let y: Vec<Option<i64>> = (0..n).map(|i| Some(i64::from(i % 4 < 2))).collect();
        let t = Table::new(vec![
            Column {
                name: "a".into(),
                data: ColumnData::Text(a),
            },
            Column {
                name: "y".into(),
                data: ColumnData::Int(y),
            },
        ])
        .unwrap();
        let (_, info) = table(10);
        let s = make_split("s-1", &t, &info, &DataConfig::default()).unwrap();
        assert_eq!((s.train.rows, s.valid.rows, s.test.rows), (700, 150, 150));
        assert_eq!((s.test.positives, s.valid.positives), (75, 75));
        // Both feature values appear in every split.
        let code = s.assignment.as_bytes();
        for split in *b"012" {
            assert!((0..n).any(|i| code[i] == split && i % 2 == 0) && (0..n).any(|i| code[i] == split && i % 2 == 1));
        }
    }

    #[test]
    fn group_split_has_no_leakage() {
        let (t, info) = table(1000);
        let cfg = DataConfig {
            split: SplitStrategy::Group,
            group_column: Some("g".into()),
            ..Default::default()
        };
        let s = make_split("s-1", &t, &info, &cfg).unwrap();
        check_group_leakage(&t, &s).unwrap();
        assert!(s.test.groups.unwrap() > 0);
    }

    #[test]
    fn refuses_a_random_split_of_time_data_and_undeclared_grouping() {
        let (t, info) = table(100);
        let cfg = DataConfig {
            time_column: Some("ts".into()),
            ..Default::default()
        };
        assert_eq!(make_split("s", &t, &info, &cfg).unwrap_err().code, ErrorCode::SplitUnsupported);
        let cfg = DataConfig {
            split: SplitStrategy::Time,
            ..Default::default()
        };
        assert_eq!(make_split("s", &t, &info, &cfg).unwrap_err().code, ErrorCode::ConfigInvalid);
        let cfg = DataConfig {
            group_column: Some("g".into()),
            ..Default::default()
        };
        assert_eq!(make_split("s", &t, &info, &cfg).unwrap_err().code, ErrorCode::LeakageDetected);
    }

    /// A table whose `ts` column holds `time(i)`, with a label every fifth row.
    fn timed(n: usize, time: impl Fn(usize) -> ColumnData) -> (Table, TargetInfo) {
        let (base, info) = table(n);
        let mut columns = vec![Column {
            name: "ts".into(),
            data: time(n),
        }];
        columns.extend(["y", "g"].iter().map(|c| Column {
            name: (*c).into(),
            data: base.column(c).unwrap().clone(),
        }));
        (Table::new(columns).unwrap(), info)
    }

    #[test]
    fn time_split_puts_the_earliest_rows_in_training_and_the_latest_in_test() {
        let cfg = DataConfig {
            split: SplitStrategy::Time,
            time_column: Some("ts".into()),
            group_column: Some("g".into()),
            ..Default::default()
        };
        // Numbers, in a scrambled row order; two rows share each time.
        let numbers = |n: usize| ColumnData::Int((0..n).map(|i| Some(((i * 7919) % n / 2) as i64)).collect());
        let (t, info) = timed(1000, numbers);
        let s = make_split("s-1", &t, &info, &cfg).unwrap();
        assert_eq!(s.algorithm, SPLIT_ALGORITHM_TIME);
        assert_eq!(s.time_column.as_deref(), Some("ts"));
        let code = s.assignment.as_bytes();
        let time = |i: usize| ((i * 7919) % 1000 / 2) as i64;
        let latest = |split: u8| (0..1000).filter(|&i| code[i] == split).map(time).max().unwrap();
        let earliest = |split: u8| (0..1000).filter(|&i| code[i] == split).map(time).min().unwrap();
        assert!(latest(SPLIT_TRAIN) < earliest(SPLIT_VALID));
        assert!(latest(SPLIT_VALID) < earliest(SPLIT_TEST));
        assert_eq!((s.train.rows, s.valid.rows, s.test.rows), (700, 150, 150));
        assert!(s.warnings.iter().any(|w| w.contains("chronological")) && s.warnings.iter().any(|w| w.contains("`g`")));
        // Dates as text sort the same way.
        let dates = |n: usize| ColumnData::Text((0..n).map(|i| Some(format!("2026-{:02}-{:02}", 1 + i % 12, 1 + i % 28))).collect());
        let (t, info) = timed(1000, dates);
        let s = make_split("s-1", &t, &info, &cfg).unwrap();
        let code = s.assignment.as_bytes();
        let date = |i: usize| format!("2026-{:02}-{:02}", 1 + i % 12, 1 + i % 28);
        let latest_train = (0..1000).filter(|&i| code[i] == SPLIT_TRAIN).map(date).max().unwrap();
        let earliest_test = (0..1000).filter(|&i| code[i] == SPLIT_TEST).map(date).min().unwrap();
        assert!(latest_train < earliest_test, "{latest_train} {earliest_test}");
        // A missing time, or times of two kinds, is refused.
        let (t, info) = timed(100, |n| ColumnData::Int((0..n).map(|i| (i != 7).then_some(i as i64)).collect()));
        assert_eq!(make_split("s", &t, &info, &cfg).unwrap_err().code, ErrorCode::DataUnsupported);
        let (t, info) = timed(100, |n| {
            ColumnData::Mixed(
                (0..n)
                    .map(|i| if i == 7 { Value::Text("x".into()) } else { Value::Int(i as i64) })
                    .collect(),
            )
        });
        assert_eq!(make_split("s", &t, &info, &cfg).unwrap_err().code, ErrorCode::DataUnsupported);
    }
}
