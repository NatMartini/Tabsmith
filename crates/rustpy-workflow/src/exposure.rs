//! Which rows a model has seen, and which rows an evaluation used.
//!
//! Rows are identified by the id column when the project declares one (stable
//! across data versions). Without one, rows are identified by position within one
//! exact snapshot content and, across data versions, by a fingerprint of their
//! content (the model's input columns + the target): identical rows count as the
//! same row, an edited row counts as a new row. Overlap between two row sets is
//! known only when they can be matched.

use std::collections::BTreeSet;

use rustpy_contracts::config::NanPolicy;
use rustpy_contracts::data::{DataSnapshot, RowIdentity};
use rustpy_contracts::hash::sha256_hex;
use rustpy_runtime::{coerce_category, ColumnData, Table, ValueRef};
use serde::{Deserialize, Serialize};

/// Content fingerprints of rows over a fixed list of columns.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RowContents {
    pub columns: Vec<String>,
    pub hashes: BTreeSet<u64>,
}

/// Normalised value text: numbers by value (`3`, `3.0` and `"3.00"` match), text
/// trimmed, missing (null/NaN) as one marker.
fn value_key(v: ValueRef<'_>, out: &mut String) {
    use std::fmt::Write;
    let num = |f: f64, out: &mut String| {
        let f = if f == 0.0 { 0.0 } else { f };
        let _ = write!(out, "n{f:?}");
    };
    match v {
        ValueRef::Null => out.push_str("\u{0}"),
        ValueRef::Float(f) if f.is_nan() => out.push_str("\u{0}"),
        ValueRef::Float(f) => num(f, out),
        ValueRef::Int(i) => num(i as f64, out),
        ValueRef::Bool(b) => out.push_str(if b { "btrue" } else { "bfalse" }),
        ValueRef::Text(t) => match t.trim().parse::<f64>() {
            Ok(f) if f.is_finite() => num(f, out),
            _ => {
                out.push('t');
                out.push_str(t.trim());
            }
        },
    }
}

/// Fingerprint of row `r` over `columns` (all must be present).
pub fn row_fingerprint(columns: &[(&str, &ColumnData)], r: usize) -> u64 {
    let mut text = String::new();
    for (name, col) in columns {
        text.push_str(name);
        text.push('\u{1f}');
        value_key(col.get(r), &mut text);
        text.push('\u{1e}');
    }
    u64::from_str_radix(&sha256_hex(text.as_bytes())[..16], 16).expect("hex")
}

/// Content columns of a model: its input fields plus the target.
pub fn schema_content_columns(input: &rustpy_contracts::schema::InputSchema, target: &str) -> Vec<String> {
    input.fields.iter().map(|f| f.name.clone()).chain([target.to_string()]).collect()
}

/// The named columns of `table`, or `None` if one is missing.
pub fn content_columns<'a>(table: &'a Table, names: &'a [String]) -> Option<Vec<(&'a str, &'a ColumnData)>> {
    if names.is_empty() {
        return None;
    }
    names.iter().map(|n| table.column(n).map(|c| (n.as_str(), c))).collect()
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RowKeys {
    /// Values of the id column.
    Ids { column: String, ids: BTreeSet<String> },
    /// Row positions within one snapshot content (identified by its SHA-256), plus
    /// content fingerprints for matching against other data versions.
    Positions {
        snapshot_sha256: String,
        rows: BTreeSet<usize>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        contents: Option<RowContents>,
    },
}

impl RowKeys {
    /// Keys of `rows` (positions in `table`, which holds the snapshot's rows).
    /// `content` names the columns fingerprinted when there is no id column
    /// (the model's input columns + the target).
    pub fn of(snapshot: &DataSnapshot, table: &Table, rows: &[usize], content: &[String]) -> Self {
        if let RowIdentity::IdColumn { column } = &snapshot.row_identity {
            if let Some(col) = table.column(column) {
                let ids = rows
                    .iter()
                    .filter_map(|&r| coerce_category(col.get(r), NanPolicy::Missing).ok().flatten())
                    .map(|c| c.into_owned())
                    .collect();
                return RowKeys::Ids {
                    column: column.clone(),
                    ids,
                };
            }
        }
        let contents = content_columns(table, content).map(|cols| RowContents {
            columns: content.to_vec(),
            hashes: rows.iter().map(|&r| row_fingerprint(&cols, r)).collect(),
        });
        RowKeys::Positions {
            snapshot_sha256: snapshot.source.sha256.clone(),
            rows: rows.iter().copied().collect(),
            contents,
        }
    }

    /// Whether row `r` of `table` (a snapshot's rows) is in this set; `None` when the
    /// row cannot be matched with these keys.
    pub fn contains_row(&self, snapshot: &DataSnapshot, table: &Table, r: usize) -> Option<bool> {
        self.matcher(snapshot, table).map(|m| m(r))
    }

    /// A per-row membership test for rows of `table`, or `None` when its rows
    /// cannot be matched with these keys.
    pub fn matcher<'a>(&'a self, snapshot: &DataSnapshot, table: &'a Table) -> Option<Box<dyn Fn(usize) -> bool + 'a>> {
        match (self, &snapshot.row_identity) {
            (RowKeys::Ids { ids, column }, RowIdentity::IdColumn { column: current }) if column == current => {
                let col = table.column(current)?;
                Some(Box::new(move |r| {
                    coerce_category(col.get(r), NanPolicy::Missing)
                        .ok()
                        .flatten()
                        .is_some_and(|id| ids.contains(id.as_ref()))
                }))
            }
            (RowKeys::Positions { snapshot_sha256, rows, .. }, _) if *snapshot_sha256 == snapshot.source.sha256 => {
                Some(Box::new(move |r| rows.contains(&r)))
            }
            (RowKeys::Positions { contents: Some(c), .. }, _) => {
                let cols = content_columns(table, &c.columns)?;
                Some(Box::new(move |r| c.hashes.contains(&row_fingerprint(&cols, r))))
            }
            _ => None,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            RowKeys::Ids { ids, .. } => ids.len(),
            RowKeys::Positions { rows, .. } => rows.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Number of shared rows, or `None` when the two sets cannot be matched
    /// (different id columns, ids vs positions, or positions in different
    /// snapshots without comparable content fingerprints).
    pub fn overlap(&self, other: &RowKeys) -> Option<usize> {
        match (self, other) {
            (RowKeys::Ids { ids: a, column: ca }, RowKeys::Ids { ids: b, column: cb }) if ca == cb => {
                Some(a.intersection(b).count())
            }
            (
                RowKeys::Positions { snapshot_sha256: sa, rows: a, .. },
                RowKeys::Positions { snapshot_sha256: sb, rows: b, .. },
            ) if sa == sb => Some(a.intersection(b).count()),
            (
                RowKeys::Positions { contents: Some(a), .. },
                RowKeys::Positions { contents: Some(b), .. },
            ) if a.columns == b.columns => Some(a.hashes.intersection(&b.hashes).count()),
            _ => None,
        }
    }
}

/// Result of checking evaluated rows against what the model has seen.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ExposureCheck {
    /// Evaluated rows that the model (or an ancestor it was warm-started from)
    /// used for training or model selection.
    pub overlapping_rows: usize,
    /// Sources (run ids) whose rows could not be matched with the evaluated rows.
    pub unverifiable: Vec<String>,
}
