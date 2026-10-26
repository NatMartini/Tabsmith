//! Deterministic train/valid/test splits, fixed before any statistic is fitted.

use std::collections::{BTreeMap, BTreeSet};

use rustpy_contracts::config::{DataConfig, NanPolicy, SplitStrategy};
use rustpy_contracts::data::{SplitCounts, SplitManifest, SplitName, SPLIT_TEST, SPLIT_TRAIN, SPLIT_VALID};
use rustpy_contracts::hash::{canonical_hash, short};
use rustpy_contracts::{ErrorCode, Phase, Result, RustpyError};
use rustpy_runtime::{coerce_category, Table};

use crate::data::TargetInfo;
use crate::rng::SplitMix64;

/// Version of the split algorithm; part of the split id.
pub const SPLIT_ALGORITHM: &str = "rustpy.split.v1";

/// Binary label per row (true = positive).
pub fn labels(table: &Table, target: &TargetInfo) -> Vec<bool> {
    let col = table.column(&target.name).expect("target exists");
    (0..table.n_rows())
        .map(|i| {
            coerce_category(col.get(i), NanPolicy::Missing)
                .ok()
                .flatten()
                .is_some_and(|l| l == target.positive_label.as_str())
        })
        .collect()
}

fn refuse_time_split(cfg: &DataConfig) -> Result<()> {
    if cfg.split == SplitStrategy::Time || cfg.time_column.is_some() {
        return Err(RustpyError::new(
            ErrorCode::SplitUnsupported,
            Phase::Split,
            "a chronological (time) split is required but not implemented in tabular-binary@0.1",
        )
        .field(if cfg.split == SplitStrategy::Time { "data.split" } else { "data.time_column" })
        .expected("split = \"stratified\" (independent rows) or \"group\" (repeated entities)")
        .next("RustPy will not fall back to a random split for time-dependent data; split the file by time yourself into train/test files, or remove data.time_column if rows are truly independent"));
    }
    Ok(())
}

/// Computes the split manifest for a snapshot.
pub fn make_split(
    snapshot_id: &str,
    table: &Table,
    target: &TargetInfo,
    cfg: &DataConfig,
) -> Result<SplitManifest> {
    refuse_time_split(cfg)?;
    if cfg.group_column.is_some() && cfg.split != SplitStrategy::Group {
        return Err(RustpyError::new(
            ErrorCode::LeakageDetected,
            Phase::Split,
            "data.group_column is declared but data.split is not \"group\": rows of one entity could land in both training and test",
        )
        .field("data.split")
        .expected("group")
        .actual(cfg.split.to_string())
        .next("set data.split = \"group\" (or remove data.group_column if rows are independent)"));
    }
    let y = labels(table, target);
    let n = y.len();
    let fractions = [cfg.train_fraction, cfg.valid_fraction, cfg.test_fraction];
    let mut assignment = vec![SPLIT_TRAIN; n];
    let mut group_counts = None;
    match cfg.split {
        SplitStrategy::Stratified => {
            for class in [false, true] {
                let mut idx: Vec<usize> = (0..n).filter(|&i| y[i] == class).collect();
                let label = if class { &target.positive_label } else { &target.negative_label };
                SplitMix64::derive(cfg.seed, &format!("stratified/{label}")).shuffle(&mut idx);
                let nc = idx.len();
                let mut n_test = (nc as f64 * cfg.test_fraction).round() as usize;
                let mut n_valid = (nc as f64 * cfg.valid_fraction).round() as usize;
                if nc >= 3 {
                    n_test = n_test.max(1);
                    n_valid = n_valid.max(1);
                }
                if n_test + n_valid >= nc {
                    return Err(degenerate(label, nc));
                }
                for &i in &idx[..n_test] {
                    assignment[i] = SPLIT_TEST;
                }
                for &i in &idx[n_test..n_test + n_valid] {
                    assignment[i] = SPLIT_VALID;
                }
            }
        }
        SplitStrategy::Group => {
            let gname = cfg.group_column.as_deref().ok_or_else(|| {
                RustpyError::new(ErrorCode::ConfigInvalid, Phase::Split, "split = \"group\" needs data.group_column")
                    .field("data.group_column")
                    .next("set data.group_column to the entity column (e.g. device_id)")
            })?;
            let col = table.column(gname).ok_or_else(|| {
                RustpyError::new(ErrorCode::SchemaMissingColumn, Phase::Split, format!("group column `{gname}` not found"))
                    .field("data.group_column")
            })?;
            let mut rows_by_group: BTreeMap<String, Vec<usize>> = BTreeMap::new();
            for i in 0..n {
                let key = coerce_category(col.get(i), NanPolicy::Missing)
                    .ok()
                    .flatten()
                    .ok_or_else(|| {
                        RustpyError::new(
                            ErrorCode::DataUnsupported,
                            Phase::Split,
                            format!("group column `{gname}` is missing or invalid in row {i}"),
                        )
                        .field("data.group_column")
                        .next("every row needs a group id for a grouped split")
                    })?;
                rows_by_group.entry(key.into_owned()).or_default().push(i);
            }
            let mut groups: Vec<(String, Vec<usize>)> = rows_by_group.into_iter().collect();
            if groups.len() < 3 {
                return Err(RustpyError::new(
                    ErrorCode::SplitDegenerate,
                    Phase::Split,
                    format!("only {} groups in `{gname}`; a grouped split needs at least 3", groups.len()),
                )
                .field("data.group_column"));
            }
            SplitMix64::derive(cfg.seed, "group").shuffle(&mut groups);
            let test_target = (n as f64 * cfg.test_fraction).round() as usize;
            let valid_target = (n as f64 * cfg.valid_fraction).round() as usize;
            let (mut test_rows, mut valid_rows) = (0usize, 0usize);
            let mut counts = [0usize; 3];
            for (_, rows) in &groups {
                let code = if test_rows < test_target.max(1) {
                    test_rows += rows.len();
                    counts[2] += 1;
                    SPLIT_TEST
                } else if valid_rows < valid_target.max(1) {
                    valid_rows += rows.len();
                    counts[1] += 1;
                    SPLIT_VALID
                } else {
                    counts[0] += 1;
                    SPLIT_TRAIN
                };
                for &i in rows {
                    assignment[i] = code;
                }
            }
            group_counts = Some(counts);
            // Leakage check by construction: verify no group spans two splits.
            let mut seen: BTreeMap<&str, u8> = BTreeMap::new();
            for (g, rows) in &groups {
                for &i in rows {
                    if let Some(prev) = seen.insert(g.as_str(), assignment[i]) {
                        if prev != assignment[i] {
                            return Err(RustpyError::internal(Phase::Split, format!("group {g} spans splits")));
                        }
                    }
                }
            }
        }
        SplitStrategy::Time => unreachable!("refused above"),
    }
    let counts = |code: u8, g: Option<usize>| {
        let rows: Vec<usize> = (0..n).filter(|&i| assignment[i] == code).collect();
        let positives = rows.iter().filter(|&&i| y[i]).count();
        SplitCounts {
            rows: rows.len(),
            positives,
            negatives: rows.len() - positives,
            groups: g,
        }
    };
    let train = counts(SPLIT_TRAIN, group_counts.map(|c| c[0]));
    let valid = counts(SPLIT_VALID, group_counts.map(|c| c[1]));
    let test = counts(SPLIT_TEST, group_counts.map(|c| c[2]));
    for (name, c) in [("train", &train), ("valid", &valid), ("test", &test)] {
        if c.positives == 0 || c.negatives == 0 {
            return Err(RustpyError::new(
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
    }
    let assignment = String::from_utf8(assignment).expect("ascii digits");
    let identity = canonical_hash(&(
        snapshot_id,
        cfg.split,
        cfg.seed,
        fractions,
        &target.name,
        &target.positive_label,
        &cfg.group_column,
        SPLIT_ALGORITHM,
    ));
    let manifest = SplitManifest {
        split_id: format!("sp-{}", short(&identity)),
        snapshot_id: snapshot_id.to_string(),
        strategy: cfg.split,
        seed: cfg.seed,
        fractions,
        target: target.name.clone(),
        positive_label: target.positive_label.clone(),
        group_column: cfg.group_column.clone(),
        assignment,
        train,
        valid,
        test,
        algorithm: SPLIT_ALGORITHM.to_string(),
    };
    debug_assert!(manifest.rows(SplitName::Train).len() == manifest.train.rows);
    Ok(manifest)
}

fn degenerate(label: &str, n: usize) -> RustpyError {
    RustpyError::new(
        ErrorCode::SplitDegenerate,
        Phase::Split,
        format!("class {label:?} has only {n} rows: not enough for train, validation and test"),
    )
    .next("add more labelled rows of this class (at least 3; realistically many more)")
}

/// Verifies that no group appears in two splits (used by tests and `plan`).
pub fn check_group_leakage(table: &Table, split: &SplitManifest) -> Result<()> {
    let Some(g) = &split.group_column else { return Ok(()) };
    let col = table.column(g).expect("group column exists");
    let mut by_group: BTreeMap<String, BTreeSet<u8>> = BTreeMap::new();
    for (i, code) in split.assignment.bytes().enumerate() {
        if let Ok(Some(k)) = coerce_category(col.get(i), NanPolicy::Missing) {
            by_group.entry(k.into_owned()).or_default().insert(code);
        }
    }
    if let Some((g, _)) = by_group.iter().find(|(_, s)| s.len() > 1) {
        return Err(RustpyError::new(
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
    use rustpy_runtime::{Column, ColumnData};

    fn table(n: usize) -> (Table, TargetInfo) {
        let y: Vec<Option<i64>> = (0..n).map(|i| Some(i64::from(i % 5 == 0))).collect();
        let g: Vec<Option<String>> = (0..n).map(|i| Some(format!("dev{}", i / 4))).collect();
        let t = Table::new(vec![
            Column { name: "y".into(), data: ColumnData::Int(y) },
            Column { name: "g".into(), data: ColumnData::Text(g) },
        ])
        .unwrap();
        let info = TargetInfo {
            name: "y".into(),
            positive_label: "1".into(),
            negative_label: "0".into(),
            positives: n / 5,
            negatives: n - n / 5,
        };
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
    fn group_split_has_no_leakage() {
        let (t, info) = table(1000);
        let mut cfg = DataConfig::default();
        cfg.split = SplitStrategy::Group;
        cfg.group_column = Some("g".into());
        let s = make_split("s-1", &t, &info, &cfg).unwrap();
        check_group_leakage(&t, &s).unwrap();
        assert!(s.test.groups.unwrap() > 0);
    }

    #[test]
    fn refuses_time_and_undeclared_grouping() {
        let (t, info) = table(100);
        let mut cfg = DataConfig::default();
        cfg.time_column = Some("ts".into());
        assert_eq!(make_split("s", &t, &info, &cfg).unwrap_err().code, ErrorCode::SplitUnsupported);
        let mut cfg = DataConfig::default();
        cfg.group_column = Some("g".into());
        assert_eq!(make_split("s", &t, &info, &cfg).unwrap_err().code, ErrorCode::LeakageDetected);
    }
}
