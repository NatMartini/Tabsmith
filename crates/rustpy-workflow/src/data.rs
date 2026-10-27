//! Reading CSV/Parquet with Polars into raw-value tables, profiling columns and
//! creating immutable data snapshots.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use polars::prelude::*;
use rustpy_contracts::config::{ColumnKindOverride, DataConfig, DataFormat, SnapshotMode};
use rustpy_contracts::data::{
    ColumnProfile, DataSnapshot, NumericSummary, RowIdentity, SnapshotStorage, SourceFormat, SourceRef,
};
use rustpy_contracts::hash::{canonical_hash, sha256_file, short};
use rustpy_contracts::schema::RawDtype;
use rustpy_contracts::time::now_rfc3339;
use rustpy_contracts::{ErrorCode, Phase, Result, RustpyError};
use rustpy_runtime::{Column as RtColumn, ColumnData, Table};
use serde::{Deserialize, Serialize};

use crate::project::{did_you_mean, read_json, write_json, Project};

/// Options that change how a file is interpreted; part of the snapshot identity.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReadOptions {
    pub format: SourceFormat,
    pub delimiter: u8,
    pub null_values: Vec<String>,
    /// Columns forced to text (declared categorical), e.g. to keep leading zeros.
    pub text_columns: Vec<String>,
    pub reader: String,
}

impl ReadOptions {
    pub fn from_config(path: &Path, cfg: &DataConfig) -> Result<Self> {
        let format = match cfg.format {
            DataFormat::Csv => SourceFormat::Csv,
            DataFormat::Parquet => SourceFormat::Parquet,
            DataFormat::Auto => match path.extension().and_then(|e| e.to_str()).map(str::to_lowercase).as_deref() {
                Some("csv") | Some("txt") | Some("tsv") => SourceFormat::Csv,
                Some("parquet") | Some("pq") => SourceFormat::Parquet,
                other => {
                    return Err(RustpyError::new(
                        ErrorCode::DataUnsupported,
                        Phase::Data,
                        format!("cannot tell the format of `{}` (extension {:?})", path.display(), other.unwrap_or("")),
                    )
                    .field("data.format")
                    .expected("a .csv or .parquet file")
                    .next("set data.format = \"csv\" or \"parquet\" in rustpy.toml"));
                }
            },
        };
        let mut text_columns: Vec<String> = cfg
            .columns
            .iter()
            .filter(|(_, k)| **k == ColumnKindOverride::Categorical)
            .map(|(n, _)| n.clone())
            .collect();
        text_columns.sort();
        Ok(Self {
            format,
            delimiter: cfg.csv_delimiter.as_bytes()[0],
            null_values: cfg.null_values.clone(),
            text_columns,
            reader: format!("polars-{}", "0.55.2"),
        })
    }
}

/// Raw table plus the engine schema it was read with.
pub struct LoadedData {
    pub table: Table,
    pub raw_dtypes: Vec<(String, Option<RawDtype>, String)>,
}

fn read_err(path: &Path, msg: impl Into<String>) -> RustpyError {
    RustpyError::new(ErrorCode::DataReadFailed, Phase::Data, msg).field(path.display().to_string())
}

/// Result of the CSV structure pre-pass.
#[derive(Debug, Clone, PartialEq)]
pub struct CsvStructure {
    /// Header names exactly as the parser will name them (BOM removed, not trimmed).
    pub header: Vec<String>,
    /// Number of data records (blank lines excluded).
    pub records: usize,
    /// Blank lines after the last record (each becomes an all-missing row in the parser).
    pub trailing_blank_lines: usize,
}

/// Scans a CSV file's structure (RFC 4180 quoting): header names and the field
/// count of every record. Polars fills short rows and blank lines with missing
/// values silently, so this pre-pass turns ragged rows, blank lines inside the
/// data, CR-only line endings and duplicate headers into explicit errors.
pub fn check_csv_structure(path: &Path, delimiter: u8) -> Result<CsvStructure> {
    let raw = fs::read(path).map_err(|e| RustpyError::io(Phase::Data, path, e))?;
    let bytes: &[u8] = raw.strip_prefix(b"\xEF\xBB\xBF").unwrap_or(&raw);
    if bytes.iter().all(|b| b.is_ascii_whitespace()) {
        return Err(read_err(path, format!("`{}` is empty", path.display()))
            .next("point data.source at a CSV file with a header row and data rows"));
    }
    let mut header: Vec<String> = Vec::new();
    let mut field: Vec<u8> = Vec::new();
    let mut fields = 0usize;
    let mut in_quotes = false;
    let mut records = 0usize; // data records seen
    let mut have_header = false;
    let mut line = 1usize;
    let mut record_start_line = 1usize;
    let mut has_content = false;
    let mut pending_blank: Vec<usize> = Vec::new(); // blank lines not yet followed by a record
    let mut i = 0usize;
    while i < bytes.len() {
        let b = bytes[i];
        if in_quotes {
            if b == b'"' {
                if bytes.get(i + 1) == Some(&b'"') {
                    field.push(b'"');
                    i += 1;
                } else {
                    in_quotes = false;
                }
            } else {
                if b == b'\n' {
                    line += 1;
                }
                field.push(b);
            }
            i += 1;
            continue;
        }
        if b == b'\r' && bytes.get(i + 1) != Some(&b'\n') {
            return Err(read_err(path, format!("line {line} ends with a bare carriage return (CR-only line endings)"))
                .next("convert the file to LF or CRLF line endings"));
        }
        if b == b'"' {
            in_quotes = true;
            has_content = true;
        } else if b == delimiter {
            if !have_header {
                header.push(String::from_utf8_lossy(&field).to_string());
            }
            field.clear();
            fields += 1;
            has_content = true;
        } else if b == b'\n' || b == b'\r' {
            if b == b'\r' {
                i += 1; // CRLF
            }
            if has_content {
                fields += 1;
                if !have_header {
                    header.push(String::from_utf8_lossy(&field).to_string());
                    have_header = true;
                } else {
                    if let Some(&blank) = pending_blank.first() {
                        return Err(blank_line_error(path, blank));
                    }
                    if fields != header.len() {
                        return Err(ragged_error(path, record_start_line, fields, header.len()));
                    }
                    records += 1;
                }
            } else if have_header {
                pending_blank.push(line);
            } else {
                return Err(read_err(path, format!("line {line} is blank before the header row"))
                    .next("remove blank lines at the start of the file"));
            }
            field.clear();
            fields = 0;
            has_content = false;
            line += 1;
            record_start_line = line;
        } else {
            field.push(b);
            has_content = true;
        }
        i += 1;
    }
    if in_quotes {
        return Err(read_err(path, "CSV file ends inside a quoted field (unbalanced quote)")
            .next("fix the quoting and bind again"));
    }
    if has_content {
        fields += 1;
        if !have_header {
            header.push(String::from_utf8_lossy(&field).to_string());
            have_header = true;
        } else {
            if let Some(&blank) = pending_blank.first() {
                return Err(blank_line_error(path, blank));
            }
            if fields != header.len() {
                return Err(ragged_error(path, record_start_line, fields, header.len()));
            }
            records += 1;
        }
    }
    let _ = have_header;
    let mut seen = HashSet::new();
    for h in &header {
        if h.trim().is_empty() {
            return Err(read_err(path, "the CSV header has an empty column name")
                .next("give every column a name in the header row"));
        }
        if !seen.insert(h.as_str()) {
            return Err(read_err(path, format!("the CSV header repeats column `{h}`"))
                .field(h.clone())
                .next("rename the duplicate column"));
        }
    }
    if records == 0 {
        return Err(read_err(path, "the CSV file has a header but no data rows")
            .next("bind a file that contains data rows"));
    }
    Ok(CsvStructure {
        header,
        records,
        trailing_blank_lines: pending_blank.len(),
    })
}

fn blank_line_error(path: &Path, line: usize) -> RustpyError {
    read_err(path, format!("line {line} is blank; a blank line inside the data would be read as a row of missing values"))
        .next(format!("remove the blank line {line} (and any others) from the file"))
}

fn ragged_error(path: &Path, line: usize, fields: usize, expected: usize) -> RustpyError {
    read_err(path, format!("CSV record starting on line {line} has {fields} fields, the header has {expected}"))
        .expected(format!("{expected} fields per row"))
        .actual(format!("{fields} fields"))
        .next("fix the row (missing or extra delimiters / unbalanced quotes) and bind again")
}

fn raw_of(dtype: &DataType) -> Option<RawDtype> {
    match dtype {
        DataType::Float32 | DataType::Float64 => Some(RawDtype::Float),
        DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::Int64
        | DataType::UInt8
        | DataType::UInt16
        | DataType::UInt32
        | DataType::UInt64 => Some(RawDtype::Int),
        DataType::Boolean => Some(RawDtype::Bool),
        DataType::String => Some(RawDtype::Text),
        other => {
            // Parquet dictionary/categorical and decimal columns map to text / float.
            let name = format!("{other:?}");
            if name.starts_with("Categorical") || name.starts_with("Enum") {
                Some(RawDtype::Text)
            } else if name.starts_with("Decimal") {
                Some(RawDtype::Float)
            } else {
                None
            }
        }
    }
}

fn polars_of(raw: RawDtype) -> DataType {
    match raw {
        RawDtype::Float => DataType::Float64,
        RawDtype::Int => DataType::Int64,
        RawDtype::Bool => DataType::Boolean,
        RawDtype::Text => DataType::String,
    }
}

fn polars_err(path: &Path, e: PolarsError) -> RustpyError {
    let msg = e.to_string();
    let first = msg.lines().next().unwrap_or("").to_string();
    read_err(path, format!("could not read `{}`: {first}", path.display()))
        .next("check the file format and values; numeric columns must contain only numbers or missing values")
}

/// Reads a file into a raw table. With `frozen`, column types are fixed (names and
/// order must match the header exactly); otherwise they are inferred over all rows.
pub fn read_table(path: &Path, opts: &ReadOptions, frozen: Option<&[(String, Option<RawDtype>, String)]>) -> Result<LoadedData> {
    if !path.is_file() {
        return Err(RustpyError::new(
            ErrorCode::DataSourceNotFound,
            Phase::Data,
            format!("data file `{}` does not exist", path.display()),
        )
        .field(path.display().to_string())
        .next("check data.source in rustpy.toml or bind a file with `rustpy data bind <file>`"));
    }
    let mut expected_rows: Option<CsvStructure> = None;
    let df = match opts.format {
        SourceFormat::Csv => {
            let structure = check_csv_structure(path, opts.delimiter)?;
            expected_rows = Some(structure.clone());
            let header = structure.header;
            let nulls: Vec<PlSmallStr> = opts.null_values.iter().map(|s| PlSmallStr::from(s.as_str())).collect();
            let delimiter = opts.delimiter;
            let mut read = CsvReadOptions::default()
                .with_has_header(true)
                .with_infer_schema_length(None)
                .with_rechunk(true)
                .with_ignore_errors(false)
                .map_parse_options(move |p| {
                    p.with_separator(delimiter)
                        .with_null_values(Some(NullValues::AllColumns(nulls.clone())))
                        .with_missing_is_null(true)
                });
            if let Some(frozen) = frozen {
                let names: Vec<&str> = frozen.iter().map(|(n, _, _)| n.as_str()).collect();
                if names != header.iter().map(String::as_str).collect::<Vec<_>>() {
                    return Err(RustpyError::new(
                        ErrorCode::DataSourceChanged,
                        Phase::Data,
                        "the file's columns differ from the snapshot schema",
                    )
                    .expected(names.join(", "))
                    .actual(header.join(", ")));
                }
                let schema: Schema = frozen
                    .iter()
                    .map(|(n, raw, _)| Field::new(n.as_str().into(), raw.map_or(DataType::String, polars_of)))
                    .collect();
                read = read.with_schema(Some(Arc::new(schema)));
            } else if !opts.text_columns.is_empty() {
                let schema: Schema = opts
                    .text_columns
                    .iter()
                    .filter(|c| header.contains(c))
                    .map(|n| Field::new(n.as_str().into(), DataType::String))
                    .collect();
                read = read.with_schema_overwrite(Some(Arc::new(schema)));
            }
            read.try_into_reader_with_file_path(Some(path.to_path_buf()))
                .and_then(|r| r.finish())
                .map_err(|e| polars_err(path, e))?
        }
        SourceFormat::Parquet => {
            let file = fs::File::open(path).map_err(|e| RustpyError::io(Phase::Data, path, e))?;
            ParquetReader::new(file).finish().map_err(|e| polars_err(path, e))?
        }
    };
    // Reconcile the parser's row count with the structure pre-pass: trailing blank
    // lines become all-missing rows in the parser and are dropped here; any other
    // difference is an error rather than silently misaligned rows.
    let df = match &expected_rows {
        Some(st) if df.height() == st.records => df,
        Some(st) if df.height() == st.records + st.trailing_blank_lines => df.slice(0, st.records),
        Some(st) => {
            return Err(read_err(
                path,
                format!(
                    "the CSV parser produced {} rows but the file has {} data records",
                    df.height(),
                    st.records
                ),
            )
            .next("check the file for unusual line endings or quoting"));
        }
        None => df,
    };
    if df.height() == 0 {
        return Err(read_err(path, "the data has no rows").next("bind a file that contains data rows"));
    }
    let mut columns = Vec::with_capacity(df.width());
    let mut raw_dtypes = Vec::with_capacity(df.width());
    for col in df.columns() {
        let name = col.name().to_string();
        let engine = format!("{:?}", col.dtype());
        let mut raw = raw_of(col.dtype());
        if opts.text_columns.contains(&name) && raw.is_some() && raw != Some(RawDtype::Text) {
            // Parquet columns declared categorical: use the canonical text form.
            raw = Some(RawDtype::Text);
        }
        let data = match raw {
            Some(RawDtype::Float) => {
                let c = col.cast(&DataType::Float64).map_err(|e| polars_err(path, e))?;
                ColumnData::Float(c.f64().map_err(|e| polars_err(path, e))?.iter().collect())
            }
            Some(RawDtype::Int) => {
                let c = col.strict_cast(&DataType::Int64).map_err(|e| {
                    read_err(path, format!("integer column `{name}` does not fit in i64: {e}"))
                })?;
                ColumnData::Int(c.i64().map_err(|e| polars_err(path, e))?.iter().collect())
            }
            Some(RawDtype::Bool) => ColumnData::Bool(col.bool().map_err(|e| polars_err(path, e))?.iter().collect()),
            Some(RawDtype::Text) => {
                let c = col.cast(&DataType::String).map_err(|e| polars_err(path, e))?;
                ColumnData::Text(
                    c.str()
                        .map_err(|e| polars_err(path, e))?
                        .iter()
                        .map(|o| o.map(str::to_owned))
                        .collect(),
                )
            }
            None => ColumnData::Mixed(vec![rustpy_runtime::Value::Null; df.height()]),
        };
        raw_dtypes.push((name.clone(), raw, engine));
        columns.push(RtColumn { name, data });
    }
    Ok(LoadedData {
        table: Table::new(columns)?,
        raw_dtypes,
    })
}

/// Profiles every column (whole snapshot; informational, not used for fitting).
pub fn profile_columns(data: &LoadedData) -> Vec<ColumnProfile> {
    let mut out = Vec::new();
    for (col, (name, raw, engine)) in data.table.columns().iter().zip(&data.raw_dtypes) {
        let mut p = ColumnProfile {
            name: name.clone(),
            raw_dtype: *raw,
            engine_dtype: engine.clone(),
            null_count: 0,
            nan_count: 0,
            inf_count: 0,
            distinct: 0,
            numeric: None,
            top_values: Vec::new(),
            all_integral: false,
            numeric_text: 0,
            non_numeric_examples: Vec::new(),
        };
        match &col.data {
            ColumnData::Float(v) => {
                let mut distinct = HashSet::new();
                let (mut min, mut max, mut sum, mut n) = (f64::INFINITY, f64::NEG_INFINITY, 0.0, 0u64);
                for x in v {
                    match x {
                        None => p.null_count += 1,
                        Some(f) if f.is_nan() => p.nan_count += 1,
                        Some(f) if f.is_infinite() => p.inf_count += 1,
                        Some(f) => {
                            distinct.insert(f.to_bits());
                            min = min.min(*f);
                            max = max.max(*f);
                            sum += f;
                            n += 1;
                        }
                    }
                }
                p.distinct = distinct.len() as u64;
                if n > 0 {
                    p.numeric = Some(NumericSummary { min, max, mean: sum / n as f64 });
                    p.all_integral = v.iter().flatten().filter(|f| f.is_finite()).all(|f| f.fract() == 0.0);
                }
            }
            ColumnData::Int(v) => {
                let mut counts: HashMap<i64, u64> = HashMap::new();
                let (mut min, mut max, mut sum, mut n) = (f64::INFINITY, f64::NEG_INFINITY, 0.0, 0u64);
                for x in v {
                    match x {
                        None => p.null_count += 1,
                        Some(i) => {
                            *counts.entry(*i).or_default() += 1;
                            let f = *i as f64;
                            min = min.min(f);
                            max = max.max(f);
                            sum += f;
                            n += 1;
                        }
                    }
                }
                p.distinct = counts.len() as u64;
                if n > 0 {
                    p.numeric = Some(NumericSummary { min, max, mean: sum / n as f64 });
                }
                if counts.len() <= 20 {
                    p.top_values = top_n(counts.into_iter().map(|(k, c)| (k.to_string(), c)));
                }
            }
            ColumnData::Bool(v) => {
                let mut counts: HashMap<String, u64> = HashMap::new();
                for x in v {
                    match x {
                        None => p.null_count += 1,
                        Some(b) => *counts.entry(b.to_string()).or_default() += 1,
                    }
                }
                p.distinct = counts.len() as u64;
                p.top_values = top_n(counts);
            }
            ColumnData::Text(v) => {
                let mut counts: HashMap<&str, u64> = HashMap::new();
                for x in v {
                    match x {
                        None => p.null_count += 1,
                        Some(s) => *counts.entry(s.as_str()).or_default() += 1,
                    }
                }
                p.distinct = counts.len() as u64;
                let mut non_numeric: Vec<(&str, u64)> = Vec::new();
                for (k, c) in &counts {
                    if k.trim().parse::<f64>().is_ok() {
                        p.numeric_text += c;
                    } else {
                        non_numeric.push((k, *c));
                    }
                }
                if p.numeric_text * 2 > v.len() as u64 - p.null_count {
                    non_numeric.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));
                    p.non_numeric_examples = non_numeric.iter().take(5).map(|(k, _)| k.to_string()).collect();
                }
                p.top_values = top_n(counts.into_iter().map(|(k, c)| (k.to_string(), c)));
            }
            ColumnData::Mixed(v) => {
                p.null_count = v.len() as u64;
            }
        }
        out.push(p);
    }
    out
}

fn top_n(counts: impl IntoIterator<Item = (String, u64)>) -> Vec<(String, u64)> {
    let mut v: Vec<(String, u64)> = counts.into_iter().collect();
    v.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    v.truncate(10);
    v
}

/// Snapshot plus the loaded table.
pub struct SnapshotData {
    pub snapshot: DataSnapshot,
    pub table: Table,
    pub reused: bool,
}

/// Extra fields stored next to the snapshot (not part of the public contract).
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SnapshotInternal {
    read_options: ReadOptions,
    raw_dtypes: Vec<(String, Option<RawDtype>, String)>,
}

pub fn source_path(project: &Project, cfg: &DataConfig) -> Result<PathBuf> {
    let source = cfg.source.as_deref().ok_or_else(|| {
        RustpyError::new(ErrorCode::ConfigInvalid, Phase::Config, "`data.source` is not set")
            .field("data.source")
            .next("bind a dataset: rustpy data bind <file> --target <column> --positive <value>")
    })?;
    let p = Path::new(source);
    Ok(if p.is_absolute() { p.to_path_buf() } else { project.root.join(p) })
}

/// Creates (or reuses) the snapshot of the configured data source.
pub fn snapshot_source(project: &Project, cfg: &DataConfig) -> Result<SnapshotData> {
    let path = source_path(project, cfg)?;
    let opts = ReadOptions::from_config(&path, cfg)?;
    if !path.is_file() {
        return Err(RustpyError::new(
            ErrorCode::DataSourceNotFound,
            Phase::Data,
            format!("data file `{}` does not exist", path.display()),
        )
        .field("data.source")
        .next("check data.source in rustpy.toml or bind a file with `rustpy data bind <file>`"));
    }
    let (sha, bytes) = sha256_file(&path).map_err(|e| RustpyError::io(Phase::Data, &path, e))?;
    // Reference snapshots point at a file: its location is part of the identity.
    let location = match cfg.snapshot {
        SnapshotMode::Reference => Some(path.display().to_string()),
        SnapshotMode::Copy => None,
    };
    let identity = canonical_hash(&(sha.as_str(), &opts, cfg.snapshot, cfg.id_column.as_deref(), location));
    let snapshot_id = format!("s-{}", short(&identity));
    let dir = project.dir("snapshots").join(&snapshot_id);
    if dir.join("snapshot.json").is_file() {
        let loaded = load_snapshot(project, &snapshot_id)?;
        return Ok(SnapshotData {
            snapshot: loaded.0,
            table: loaded.1,
            reused: true,
        });
    }
    fs::create_dir_all(&dir).map_err(|e| RustpyError::io(Phase::Data, &dir, e))?;
    let ext = match opts.format {
        SourceFormat::Csv => "csv",
        SourceFormat::Parquet => "parquet",
    };
    let (read_path, storage) = match cfg.snapshot {
        SnapshotMode::Copy => {
            let copy = dir.join(format!("data.{ext}"));
            fs::copy(&path, &copy).map_err(|e| RustpyError::io(Phase::Data, &copy, e))?;
            let (copy_sha, _) = sha256_file(&copy).map_err(|e| RustpyError::io(Phase::Data, &copy, e))?;
            if copy_sha != sha {
                let _ = fs::remove_dir_all(&dir);
                return Err(changed_error(&path));
            }
            (copy, SnapshotStorage::Copy { file: format!("data.{ext}") })
        }
        SnapshotMode::Reference => (path.clone(), SnapshotStorage::Reference),
    };
    let loaded = match read_table(&read_path, &opts, None) {
        Ok(l) => l,
        Err(e) => {
            let _ = fs::remove_dir_all(&dir);
            return Err(e);
        }
    };
    if matches!(storage, SnapshotStorage::Reference) {
        let (after, _) = sha256_file(&path).map_err(|e| RustpyError::io(Phase::Data, &path, e))?;
        if after != sha {
            let _ = fs::remove_dir_all(&dir);
            return Err(changed_error(&path));
        }
    }
    let n_rows = loaded.table.n_rows();
    let columns = profile_columns(&loaded);
    let mut limitations = Vec::new();
    let row_identity = match &cfg.id_column {
        Some(id) => RowIdentity::IdColumn { column: id.clone() },
        None => {
            limitations.push("rows are identified by position in this exact file content".to_string());
            RowIdentity::RowIndex
        }
    };
    if matches!(storage, SnapshotStorage::Reference) {
        limitations.push("reference snapshot: replay needs the unchanged source file".to_string());
    }
    let snapshot = DataSnapshot {
        snapshot_id: snapshot_id.clone(),
        created_at: now_rfc3339(),
        source: SourceRef {
            path: cfg.source.clone().unwrap_or_default(),
            resolved_path: path.display().to_string(),
            format: opts.format,
            sha256: sha,
            bytes,
        },
        storage,
        n_rows,
        columns,
        row_identity,
        limitations,
    };
    write_json(
        &dir.join("internal.json"),
        &SnapshotInternal {
            read_options: opts,
            raw_dtypes: loaded.raw_dtypes.clone(),
        },
    )?;
    write_json(&dir.join("snapshot.json"), &snapshot)?;
    Ok(SnapshotData {
        snapshot,
        table: loaded.table,
        reused: false,
    })
}

fn changed_error(path: &Path) -> RustpyError {
    RustpyError::new(
        ErrorCode::DataSourceChanged,
        Phase::Data,
        format!("`{}` changed while it was being read", path.display()),
    )
    .field(path.display().to_string())
    .next("wait until the file is fully written, then run the command again (a new snapshot will be created)")
}

/// Loads an existing snapshot, verifying its content hash.
pub fn load_snapshot(project: &Project, snapshot_id: &str) -> Result<(DataSnapshot, Table)> {
    let dir = project.dir("snapshots").join(snapshot_id);
    let snapshot: DataSnapshot = read_json(&dir.join("snapshot.json"), Phase::Data)?;
    let internal: SnapshotInternal = read_json(&dir.join("internal.json"), Phase::Data)?;
    let path = match &snapshot.storage {
        SnapshotStorage::Copy { file } => dir.join(file),
        SnapshotStorage::Reference => PathBuf::from(&snapshot.source.resolved_path),
    };
    if !path.is_file() {
        return Err(RustpyError::new(
            ErrorCode::DataSourceNotFound,
            Phase::Data,
            format!("snapshot {snapshot_id} data is missing at `{}`", path.display()),
        )
        .next("bind the data again to create a new snapshot (`rustpy data bind <file>`)"));
    }
    let (sha, _) = sha256_file(&path).map_err(|e| RustpyError::io(Phase::Data, &path, e))?;
    if sha != snapshot.source.sha256 {
        return Err(RustpyError::new(
            ErrorCode::DataSourceChanged,
            Phase::Data,
            format!("the data of snapshot {snapshot_id} changed since it was created"),
        )
        .field(path.display().to_string())
        .expected(format!("sha256 {}", snapshot.source.sha256))
        .actual(format!("sha256 {sha}"))
        .retained(format!("snapshot record {snapshot_id} (provenance only; cannot be replayed)"))
        .next("run `rustpy run` to create a new snapshot from the current file; runs on the old snapshot cannot be resumed"));
    }
    let loaded = read_table(&path, &internal.read_options, Some(&internal.raw_dtypes))?;
    Ok((snapshot, loaded.table))
}

/// Reads a CSV/Parquet file for offline prediction with a bundle: categorical
/// fields that were text in training are read as text (keeps e.g. leading zeros).
pub fn read_for_prediction(
    path: &Path,
    schema: &rustpy_contracts::schema::InputSchema,
    transform: &rustpy_contracts::transform::FittedTransform,
) -> Result<Table> {
    // Read with the training data's delimiter and missing-value tokens.
    let mut cfg = DataConfig::default();
    if let Some(d) = &schema.csv_delimiter {
        cfg.csv_delimiter = d.clone();
    }
    cfg.null_values = transform.missing_tokens.clone();
    let mut opts = ReadOptions::from_config(path, &cfg)?;
    opts.text_columns = schema
        .fields
        .iter()
        .filter(|f| f.kind == rustpy_contracts::schema::FieldKind::Categorical && f.raw_dtype == RawDtype::Text)
        .map(|f| f.name.clone())
        .collect();
    Ok(read_table(path, &opts, None)?.table)
}

// ---------------------------------------------------------------------------
// Column roles
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "role", rename_all = "snake_case")]
pub enum ColumnRole {
    Feature { kind: rustpy_contracts::schema::FieldKind },
    Target,
    Id,
    Group,
    Excluded { reason: String },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ColumnDecision {
    pub name: String,
    pub role: ColumnRole,
    pub raw_dtype: Option<RawDtype>,
    /// `inferred` or `config`.
    pub decided_by: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TargetInfo {
    pub name: String,
    pub positive_label: String,
    pub negative_label: String,
    pub positives: usize,
    pub negatives: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FeaturePlan {
    pub columns: Vec<ColumnDecision>,
    pub target: TargetInfo,
    pub warnings: Vec<String>,
}

impl FeaturePlan {
    pub fn features(&self) -> impl Iterator<Item = (&ColumnDecision, rustpy_contracts::schema::FieldKind)> {
        self.columns.iter().filter_map(|c| match c.role {
            ColumnRole::Feature { kind } => Some((c, kind)),
            _ => None,
        })
    }
}

fn looks_like_date(s: &str) -> bool {
    let b = s.as_bytes();
    b.len() >= 8
        && b[..4].iter().all(u8::is_ascii_digit)
        && (b[4] == b'-' || b[4] == b'/')
        && b[5..7].iter().all(u8::is_ascii_digit)
        && (b[7] == b'-' || b[7] == b'/')
}

/// Decides the role of every column and validates the target.
pub fn plan_columns(snapshot: &DataSnapshot, table: &Table, cfg: &DataConfig, max_categories: usize) -> Result<FeaturePlan> {
    use rustpy_contracts::schema::FieldKind;
    let target = cfg.target.as_deref().ok_or_else(|| {
        RustpyError::new(ErrorCode::ConfigInvalid, Phase::Config, "`data.target` is not set")
            .field("data.target")
            .next("rustpy data bind <file> --target <column> --positive <value>")
    })?;
    let positive = cfg.positive_label.as_deref().ok_or_else(|| {
        RustpyError::new(ErrorCode::ConfigInvalid, Phase::Config, "`data.positive_label` is not set")
            .field("data.positive_label")
            .next("set data.positive_label to the target value that means \"positive\" (e.g. \"1\")")
    })?;
    let names: Vec<&str> = snapshot.columns.iter().map(|c| c.name.as_str()).collect();
    let must_exist = |col: &str, key: &str| -> Result<()> {
        if names.contains(&col) {
            return Ok(());
        }
        let mut e = RustpyError::new(
            ErrorCode::SchemaMissingColumn,
            Phase::Data,
            format!("{key} refers to column `{col}`, which is not in the data"),
        )
        .field(key.to_string())
        .actual(format!("columns: {}", names.join(", ")));
        e = match did_you_mean(col, names.iter().copied()) {
            Some(s) => e.next(format!("did you mean `{s}`? fix {key} in rustpy.toml")),
            None => e.next(format!("fix {key} in rustpy.toml")),
        };
        Err(e)
    };
    must_exist(target, "data.target")?;
    for (key, col) in [("data.id_column", &cfg.id_column), ("data.group_column", &cfg.group_column), ("data.time_column", &cfg.time_column)] {
        if let Some(c) = col {
            must_exist(c, key)?;
        }
    }
    for c in &cfg.exclude {
        must_exist(c, "data.exclude")?;
    }
    for c in cfg.columns.keys() {
        must_exist(c, &format!("data.columns.{c}"))?;
    }

    for (key, col) in [
        ("data.target", Some(target)),
        ("data.id_column", cfg.id_column.as_deref()),
        ("data.group_column", cfg.group_column.as_deref()),
    ] {
        if let Some(c) = col {
            if let Some(prof) = snapshot.column(c) {
                if prof.raw_dtype.is_none() {
                    return Err(RustpyError::new(
                        ErrorCode::DataUnsupported,
                        Phase::Data,
                        format!("{key} `{c}` has the unsupported type {}", prof.engine_dtype),
                    )
                    .field(key)
                    .next("convert the column to text, integer or boolean values"));
                }
            }
        }
    }

    // Target: exactly two labels, no missing values, positive label present.
    let tcol = table.column(target).expect("checked above");
    let mut label_counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut missing_targets = 0usize;
    for i in 0..table.n_rows() {
        match rustpy_runtime::coerce_category(tcol.get(i), rustpy_contracts::config::NanPolicy::Missing) {
            Ok(Some(l)) => *label_counts.entry(l.into_owned()).or_default() += 1,
            Ok(None) => missing_targets += 1,
            Err(_) => {
                return Err(RustpyError::new(
                    ErrorCode::TargetInvalid,
                    Phase::Data,
                    format!("target `{target}` row {i} has a non-label value {}", tcol.get(i).preview()),
                )
                .field(target)
                .next("the target must hold two class labels (text, bool or integers)"));
            }
        }
    }
    if missing_targets > 0 {
        return Err(RustpyError::new(
            ErrorCode::TargetInvalid,
            Phase::Data,
            format!("target `{target}` is missing in {missing_targets} rows"),
        )
        .field(target)
        .next("remove or label those rows; rows without a label cannot be used for training or evaluation"));
    }
    let labels: Vec<&String> = label_counts.keys().collect();
    let shown = label_counts.iter().map(|(l, c)| format!("{l:?}: {c}")).collect::<Vec<_>>().join(", ");
    if label_counts.len() != 2 {
        return Err(RustpyError::new(
            ErrorCode::TargetInvalid,
            Phase::Data,
            format!("target `{target}` must have exactly 2 classes, found {}", label_counts.len()),
        )
        .field(target)
        .expected("2 distinct labels")
        .actual(shown)
        .next("tabular-binary@0.1 supports binary targets only; map the labels to two classes"));
    }
    if !label_counts.contains_key(positive) {
        return Err(RustpyError::new(
            ErrorCode::TargetInvalid,
            Phase::Data,
            format!("positive label {positive:?} does not occur in target `{target}`"),
        )
        .field("data.positive_label")
        .expected(format!("one of {:?}", labels))
        .actual(positive.to_string())
        .next("set data.positive_label to one of the labels above"));
    }
    let negative = labels.iter().find(|l| l.as_str() != positive).expect("two labels").to_string();
    let target_info = TargetInfo {
        name: target.to_string(),
        positive_label: positive.to_string(),
        negative_label: negative.clone(),
        positives: label_counts[positive],
        negatives: label_counts[&negative],
    };

    // Id column must be unique and complete.
    if let Some(id) = &cfg.id_column {
        let col = table.column(id).expect("checked");
        let mut seen = HashSet::new();
        for i in 0..table.n_rows() {
            let key = match rustpy_runtime::coerce_category(col.get(i), rustpy_contracts::config::NanPolicy::Missing) {
                Ok(Some(k)) => k.into_owned(),
                _ => {
                    return Err(RustpyError::new(
                        ErrorCode::DataUnsupported,
                        Phase::Data,
                        format!("id column `{id}` is missing or invalid in row {i}"),
                    )
                    .field("data.id_column")
                    .next("every row needs an id; fix the data or unset data.id_column"));
                }
            };
            if !seen.insert(key.clone()) {
                return Err(RustpyError::new(
                    ErrorCode::DataUnsupported,
                    Phase::Data,
                    format!("id column `{id}` repeats the value {key:?}; it cannot identify rows"),
                )
                .field("data.id_column")
                .next("use data.group_column for entity ids that repeat (e.g. device id), or unset data.id_column"));
            }
        }
    }

    let mut warnings = Vec::new();
    let mut columns = Vec::new();
    for prof in &snapshot.columns {
        let name = prof.name.as_str();
        let (role, by) = if name == target {
            (ColumnRole::Target, "config")
        } else if cfg.id_column.as_deref() == Some(name) {
            (ColumnRole::Id, "config")
        } else if cfg.group_column.as_deref() == Some(name) {
            (ColumnRole::Group, "config")
        } else if cfg.time_column.as_deref() == Some(name) {
            (ColumnRole::Excluded { reason: "declared time column".into() }, "config")
        } else if cfg.exclude.iter().any(|c| c == name) {
            (ColumnRole::Excluded { reason: "listed in data.exclude".into() }, "config")
        } else if let Some(kind) = cfg.columns.get(name) {
            let role = match kind {
                ColumnKindOverride::Ignore => ColumnRole::Excluded { reason: "data.columns = ignore".into() },
                ColumnKindOverride::Categorical => ColumnRole::Feature { kind: FieldKind::Categorical },
                ColumnKindOverride::Numeric => {
                    if !matches!(prof.raw_dtype, Some(RawDtype::Float) | Some(RawDtype::Int)) {
                        return Err(RustpyError::new(
                            ErrorCode::SchemaTypeMismatch,
                            Phase::Data,
                            format!(
                                "column `{name}` is declared numeric but holds {} values",
                                prof.raw_dtype.map_or("unsupported", RawDtype::as_str)
                            ),
                        )
                        .field(format!("data.columns.{name}"))
                        .next(format!("declare it categorical or clean the column; first values: {:?}", prof.top_values.iter().take(3).map(|v| &v.0).collect::<Vec<_>>())));
                    }
                    ColumnRole::Feature { kind: FieldKind::Numeric }
                }
            };
            (role, "config")
        } else {
            let non_missing = (snapshot.n_rows as u64).saturating_sub(prof.null_count + prof.nan_count);
            let role = match prof.raw_dtype {
                None => ColumnRole::Excluded {
                    reason: format!("unsupported type {} (dates/nested types are not features in 0.1)", prof.engine_dtype),
                },
                Some(_) if non_missing == 0 => ColumnRole::Excluded { reason: "all values are missing".into() },
                Some(RawDtype::Float | RawDtype::Int)
                    if prof.distinct == non_missing
                        && non_missing >= 20
                        && (prof.raw_dtype == Some(RawDtype::Int) || prof.all_integral) =>
                {
                    ColumnRole::Excluded {
                        reason: format!(
                            "every value is a unique integer (looks like an identifier); set data.columns.{name} = \"numeric\" to use it as a feature"
                        ),
                    }
                }
                Some(RawDtype::Float | RawDtype::Int) => ColumnRole::Feature { kind: FieldKind::Numeric },
                Some(RawDtype::Bool) => ColumnRole::Feature { kind: FieldKind::Categorical },
                Some(RawDtype::Text) => {
                    let sample_dates = prof.top_values.iter().filter(|(v, _)| looks_like_date(v)).count();
                    if !prof.non_numeric_examples.is_empty() && prof.numeric_text * 10 >= non_missing * 9 {
                        // A numeric column with a few stray tokens ("?", "n/a", ...): never guess.
                        return Err(RustpyError::new(
                            ErrorCode::SchemaTypeMismatch,
                            Phase::Data,
                            format!(
                                "column `{name}` looks numeric ({} of {non_missing} values are numbers) but contains non-numeric values",
                                prof.numeric_text
                            ),
                        )
                        .field(name)
                        .actual(format!("non-numeric values: {:?}", prof.non_numeric_examples))
                        .next(format!(
                            "if these mean \"missing\", add them to data.null_values (e.g. null_values = [\"\", \"NA\", \"null\", {:?}]); otherwise fix the data or set data.columns.{name} = \"categorical\" or \"ignore\"",
                            prof.non_numeric_examples[0]
                        )));
                    } else if !prof.top_values.is_empty() && sample_dates * 10 >= prof.top_values.len() * 9 {
                        if cfg.time_column.is_none() {
                            warnings.push(format!(
                                "column `{name}` holds dates: if rows depend on time (drift, seasonality), declare data.time_column = \"{name}\"; RustPy then refuses the random split instead of leaking future rows into training"
                            ));
                        }
                        ColumnRole::Excluded { reason: "date/time text is not a feature in 0.1".into() }
                    } else if prof.distinct as usize > max_categories {
                        if prof.distinct * 2 > non_missing {
                            ColumnRole::Excluded {
                                reason: format!(
                                    "{} distinct values in {} rows (identifier or free text)",
                                    prof.distinct, non_missing
                                ),
                            }
                        } else {
                            let repeats = non_missing as f64 / prof.distinct.max(1) as f64;
                            let mut e = RustpyError::new(
                                ErrorCode::CardinalityBudget,
                                Phase::Data,
                                format!(
                                    "text column `{name}` has {} distinct values (each repeated ~{repeats:.0} times), above preprocess.max_categories = {max_categories}",
                                    prof.distinct
                                ),
                            )
                            .field(name)
                            .expected(format!("at most {max_categories} categories"))
                            .actual(prof.distinct.to_string());
                            e = if cfg.group_column.is_none() && prof.distinct >= 20 {
                                // Repeated ids are often entities, which matter for the split.
                                e.next(format!(
                                    "if `{name}` identifies an entity (device, customer, patient) whose rows are related, split by it: rustpy data bind <file> --target <col> --group {name}; otherwise exclude it (data.exclude = [\"{name}\"]) or raise preprocess.max_categories"
                                ))
                            } else {
                                e.next(format!(
                                    "exclude it (data.exclude = [\"{name}\"]), group rare values upstream, or raise preprocess.max_categories"
                                ))
                            };
                            return Err(e);
                        }
                    } else {
                        ColumnRole::Feature { kind: FieldKind::Categorical }
                    }
                }
            };
            (role, "inferred")
        };
        if let (ColumnRole::Excluded { reason }, "inferred") = (&role, by) {
            warnings.push(format!("column `{name}` excluded: {reason}"));
        }
        columns.push(ColumnDecision {
            name: name.to_string(),
            role,
            raw_dtype: prof.raw_dtype,
            decided_by: by.to_string(),
        });
    }
    if !columns.iter().any(|c| matches!(c.role, ColumnRole::Feature { .. })) {
        return Err(RustpyError::new(
            ErrorCode::DataUnsupported,
            Phase::Data,
            "no usable feature columns remain after excluding target, id, group and unsupported columns",
        )
        .next("check data.exclude / data.columns, or add numeric/categorical columns"));
    }
    Ok(FeaturePlan {
        columns,
        target: target_info,
        warnings,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(dir: &Path, name: &str, text: &str) -> PathBuf {
        let p = dir.join(name);
        fs::write(&p, text).unwrap();
        p
    }

    #[test]
    fn csv_structure_errors() {
        let d = tempfile::tempdir().unwrap();
        let ok = write(d.path(), "ok.csv", "a,b\n1,\"x,y\"\n2,\"q\"\"z\"\n");
        let st = check_csv_structure(&ok, b',').unwrap();
        assert_eq!((st.header, st.records, st.trailing_blank_lines), (vec!["a".into(), "b".into()], 2, 0));
        let bom = write(d.path(), "bom.csv", "\u{feff}x, y\n1,2\n");
        assert_eq!(check_csv_structure(&bom, b',').unwrap().header, vec!["x", " y"]);
        let blank = write(d.path(), "blank.csv", "a,b\n1,2\n\n3,4\n");
        assert!(check_csv_structure(&blank, b',').unwrap_err().message.contains("line 3 is blank"));
        let trailing = write(d.path(), "t.csv", "a,b\n1,2\n3,4\n\n\n");
        let st = check_csv_structure(&trailing, b',').unwrap();
        assert_eq!((st.records, st.trailing_blank_lines), (2, 2));
        let cr = write(d.path(), "cr.csv", "a,b\r1,2\r");
        assert!(check_csv_structure(&cr, b',').unwrap_err().message.contains("carriage return"));
        let quoted_newline = write(d.path(), "q.csv", "a,b\n1,\"line1\n\nline3\"\n");
        assert_eq!(check_csv_structure(&quoted_newline, b',').unwrap().records, 1);
        let short = write(d.path(), "s.csv", "a,b,c\n1,2,3\n4,5\n");
        assert_eq!(check_csv_structure(&short, b',').unwrap_err().code, ErrorCode::DataReadFailed);
        let dup = write(d.path(), "d.csv", "a,a\n1,2\n");
        assert!(check_csv_structure(&dup, b',').unwrap_err().message.contains("repeats"));
        let crlf = write(d.path(), "c.csv", "a,b\r\n1,2\r\n3,4\r\n");
        assert!(check_csv_structure(&crlf, b',').is_ok());
        let empty = write(d.path(), "e.csv", "");
        assert!(check_csv_structure(&empty, b',').is_err());
    }

    #[test]
    fn reads_csv_with_null_and_nan() {
        let d = tempfile::tempdir().unwrap();
        let p = write(d.path(), "x.csv", "x,c,y\n1.5,a,1\n,NA,0\nNaN,b,1\n2,a,0\n");
        let cfg = DataConfig::default();
        let opts = ReadOptions::from_config(&p, &cfg).unwrap();
        let data = read_table(&p, &opts, None).unwrap();
        let x = data.table.column("x").unwrap();
        assert_eq!(x.get(0), rustpy_runtime::ValueRef::Float(1.5));
        assert_eq!(x.get(1), rustpy_runtime::ValueRef::Null);
        assert!(matches!(x.get(2), rustpy_runtime::ValueRef::Float(f) if f.is_nan()));
        assert_eq!(data.table.column("c").unwrap().get(1), rustpy_runtime::ValueRef::Null);
        let prof = profile_columns(&data);
        assert_eq!(prof[0].null_count, 1);
        assert_eq!(prof[0].nan_count, 1);
        // Frozen re-read gives the same table.
        let again = read_table(&p, &opts, Some(&data.raw_dtypes)).unwrap();
        // NaN != NaN, so compare the debug rendering (bit-identical values print identically).
        assert_eq!(format!("{:?}", again.table), format!("{:?}", data.table));
    }
}
