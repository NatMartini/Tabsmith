//! Deterministic train/valid/test splits, fixed before any statistic is fitted.

use std::collections::{BTreeMap, BTreeSet};

use rustpy_contracts::config::{DataConfig, NanPolicy, SplitStrategy};
use rustpy_contracts::data::{SplitCounts, SplitManifest, SplitName, SPLIT_TEST, SPLIT_TRAIN, SPLIT_VALID};
use rustpy_contracts::hash::{canonical_hash, short};
use rustpy_contracts::{ErrorCode, Phase, Result, RustpyError};
use rustpy_runtime::{coerce_category, Table};

use crate::data::TargetInfo;
use crate::rng::SplitMix64;

/// Version of the positional stratified split algorithm; part of the split id.
pub const SPLIT_ALGORITHM: &str = "rustpy.split.v1";
/// Version of the stable, key-hashed assignment (grouped splits, or any split with
/// an id column): a row/group keeps its split when the data grows or is re-ordered.
pub const SPLIT_ALGORITHM_STABLE: &str = "rustpy.split.v2-stable-hash";
// v2 stratified-with-id: per-class ordering by the stable hash, then exact cuts.

fn key_text(v: rustpy_runtime::ValueRef<'_>) -> Option<String> {
    coerce_category(v, NanPolicy::Missing).ok().flatten().map(|c| c.into_owned())
}

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
    let mut warnings = Vec::new();
    let id_column = cfg.id_column.as_deref().and_then(|c| table.column(c));
    match (cfg.split, id_column) {
        (SplitStrategy::Stratified, Some(ids)) => {
            // Stratified and stable: within each class, rows are ordered by a hash of
            // (seed, id) and cut at the configured fractions. Re-ordering the file never
            // changes the split; appending data moves at most a few rows near the cuts
            // (the row-level evaluation checks catch any row that moves).
            for class in [false, true] {
                let mut idx: Vec<(f64, usize)> = (0..n)
                    .filter(|&i| y[i] == class)
                    .map(|i| (stable_unit(cfg.seed, "row", &key_text(ids.get(i)).unwrap_or_default()), i))
                    .collect();
                idx.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
                let label = if class { &target.positive_label } else { &target.negative_label };
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
                for &(_, i) in &idx[..n_test] {
                    assignment[i] = SPLIT_TEST;
                }
                for &(_, i) in &idx[n_test..n_test + n_valid] {
                    assignment[i] = SPLIT_VALID;
                }
            }
        }
        (SplitStrategy::Stratified, None) => {
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
        (SplitStrategy::Group, _) => {
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
                let key = key_text(col.get(i)).ok_or_else(|| {
                    RustpyError::new(
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
                return Err(RustpyError::new(
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
        (SplitStrategy::Time, _) => unreachable!("refused above"),
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
            return Err(RustpyError::new(
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
    let algorithm = if cfg.split == SplitStrategy::Group || id_column.is_some() {
        SPLIT_ALGORITHM_STABLE
    } else {
        SPLIT_ALGORITHM
    };
    let identity = canonical_hash(&(
        snapshot_id,
        cfg.split,
        cfg.seed,
        fractions,
        &target.name,
        &target.positive_label,
        &cfg.group_column,
        &cfg.id_column,
        algorithm,
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
        algorithm: algorithm.to_string(),
        warnings,
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
