//! Which rows a model has seen, and which rows an evaluation used.
//!
//! Rows are identified by the id column when the project declares one (stable
//! across data versions), otherwise by position within one exact snapshot
//! content. Overlap between two row sets is known only when they can be matched.

use std::collections::BTreeSet;

use rustpy_contracts::config::NanPolicy;
use rustpy_contracts::data::{DataSnapshot, RowIdentity};
use rustpy_runtime::{coerce_category, Table};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RowKeys {
    /// Values of the id column.
    Ids { column: String, ids: BTreeSet<String> },
    /// Row positions within one snapshot content (identified by its SHA-256).
    Positions { snapshot_sha256: String, rows: BTreeSet<usize> },
}

impl RowKeys {
    /// Keys of `rows` (positions in `table`, which holds the snapshot's rows).
    pub fn of(snapshot: &DataSnapshot, table: &Table, rows: &[usize]) -> Self {
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
        RowKeys::Positions {
            snapshot_sha256: snapshot.source.sha256.clone(),
            rows: rows.iter().copied().collect(),
        }
    }

    /// Whether row `r` of `table` (a snapshot's rows) is in this set; `None` when the
    /// row cannot be matched with these keys.
    pub fn contains_row(&self, snapshot: &DataSnapshot, table: &Table, r: usize) -> Option<bool> {
        match (self, &snapshot.row_identity) {
            (RowKeys::Ids { ids, .. }, RowIdentity::IdColumn { column }) => {
                let col = table.column(column)?;
                let id = coerce_category(col.get(r), NanPolicy::Missing).ok().flatten()?;
                Some(ids.contains(id.as_ref()))
            }
            (RowKeys::Positions { snapshot_sha256, rows }, _) if *snapshot_sha256 == snapshot.source.sha256 => {
                Some(rows.contains(&r))
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
    /// (positions in different snapshots, or ids vs positions).
    pub fn overlap(&self, other: &RowKeys) -> Option<usize> {
        match (self, other) {
            (RowKeys::Ids { ids: a, .. }, RowKeys::Ids { ids: b, .. }) => Some(a.intersection(b).count()),
            (
                RowKeys::Positions { snapshot_sha256: sa, rows: a },
                RowKeys::Positions { snapshot_sha256: sb, rows: b },
            ) if sa == sb => Some(a.intersection(b).count()),
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
