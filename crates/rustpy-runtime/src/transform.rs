//! The single implementation of value coercion and of the standard tabular
//! transform. Training, evaluation, offline prediction and application prediction
//! all go through these functions.

use std::borrow::Cow;
use std::collections::HashMap;

use rustpy_contracts::config::{NanPolicy, UnknownCategoryPolicy};
use rustpy_contracts::schema::{FieldKind, InputSchema, RawDtype};
use rustpy_contracts::transform::{FeatureTransform, FittedTransform};
use rustpy_contracts::{ErrorCode, Phase, Result, RustpyError};

use crate::value::{Table, ValueRef};

/// Why a raw value could not be coerced.
#[derive(Debug, Clone, PartialEq)]
pub enum CoerceError {
    TypeMismatch { expected: &'static str, actual: &'static str },
    NaN,
    Infinite,
    NonIntegralCategory(f64),
}

impl CoerceError {
    pub fn into_error(self, field: &str, row: usize, value: ValueRef<'_>, phase: Phase) -> RustpyError {
        let at = format!("field `{field}`, row {row}");
        match self {
            CoerceError::TypeMismatch { expected, actual } => RustpyError::new(
                ErrorCode::SchemaTypeMismatch,
                phase,
                format!("{at}: expected a {expected} value, got {actual} {}", value.preview()),
            )
            .field(field)
            .expected(expected)
            .actual(format!("{actual} {}", value.preview()))
            .next(format!("send `{field}` as a {expected} (no implicit conversion is done)")),
            CoerceError::NaN => RustpyError::new(
                ErrorCode::InvalidValue,
                phase,
                format!("{at}: NaN is not accepted (preprocess.nan = \"error\")"),
            )
            .field(field)
            .expected("a finite number or null")
            .actual("NaN")
            .next("send null for missing values, or set preprocess.nan = \"missing\" and retrain"),
            CoerceError::Infinite => RustpyError::new(
                ErrorCode::InvalidValue,
                phase,
                format!("{at}: infinite values are not accepted"),
            )
            .field(field)
            .expected("a finite number or null")
            .actual(value.preview())
            .next("clean or clip the value before prediction/training"),
            CoerceError::NonIntegralCategory(f) if f.is_finite() && f.abs() >= 9.0e15 => RustpyError::new(
                ErrorCode::SchemaTypeMismatch,
                phase,
                format!("{at}: categorical value {f} is too large to be represented exactly as a number"),
            )
            .field(field)
            .expected("an integer below 2^53, or the value as text")
            .actual(f.to_string())
            .next(format!("send `{field}` as text if the bundle was trained on text codes")),
            CoerceError::NonIntegralCategory(f) => RustpyError::new(
                ErrorCode::SchemaTypeMismatch,
                phase,
                format!("{at}: categorical value {f} is a non-integral number"),
            )
            .field(field)
            .expected("text, bool or an integral number")
            .actual(f.to_string()),
        }
    }
}

/// Numeric coercion: `Ok(None)` means missing.
pub fn coerce_numeric(v: ValueRef<'_>, nan: NanPolicy) -> Result<Option<f64>, CoerceError> {
    match v {
        ValueRef::Null => Ok(None),
        ValueRef::Int(i) => Ok(Some(i as f64)),
        ValueRef::Float(f) if f.is_nan() => match nan {
            NanPolicy::Missing => Ok(None),
            NanPolicy::Error => Err(CoerceError::NaN),
        },
        ValueRef::Float(f) if f.is_infinite() => Err(CoerceError::Infinite),
        ValueRef::Float(f) => Ok(Some(f)),
        other => Err(CoerceError::TypeMismatch {
            expected: "number",
            actual: other.type_name(),
        }),
    }
}

/// Categorical coercion to the canonical text form: `Ok(None)` means missing.
/// Integers and integral floats map to their decimal form (`3`, `3.0` -> "3"),
/// booleans to "true"/"false".
pub fn coerce_category<'a>(v: ValueRef<'a>, nan: NanPolicy) -> Result<Option<Cow<'a, str>>, CoerceError> {
    match v {
        ValueRef::Null => Ok(None),
        ValueRef::Text(s) => Ok(Some(Cow::Borrowed(s))),
        ValueRef::Bool(b) => Ok(Some(Cow::Borrowed(if b { "true" } else { "false" }))),
        ValueRef::Int(i) => Ok(Some(Cow::Owned(i.to_string()))),
        ValueRef::Float(f) if f.is_nan() => match nan {
            NanPolicy::Missing => Ok(None),
            NanPolicy::Error => Err(CoerceError::NaN),
        },
        ValueRef::Float(f) if f.is_finite() && f.fract() == 0.0 && f.abs() < 9.0e15 => {
            Ok(Some(Cow::Owned((f as i64).to_string())))
        }
        ValueRef::Float(f) => Err(CoerceError::NonIntegralCategory(f)),
    }
}

/// Dense row-major f32 matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

impl Matrix {
    pub fn row(&self, i: usize) -> &[f32] {
        &self.data[i * self.cols..(i + 1) * self.cols]
    }
}

/// Checks that a table satisfies the input contract (columns present, extra-column policy).
pub fn check_table_columns(schema: &InputSchema, table: &Table, phase: Phase) -> Result<()> {
    let names = table.names();
    for field in schema.fields.iter().filter(|f| f.required) {
        if table.column(&field.name).is_none() {
            let mut present: Vec<&str> = names.clone();
            present.sort_unstable();
            return Err(missing_column_error(&field.name, field.kind, &present, None, phase));
        }
    }
    if schema.extra_columns == rustpy_contracts::config::ExtraColumnPolicy::Error {
        for name in &names {
            let known = schema.field(name).is_some() || schema.passthrough_id.as_deref() == Some(*name);
            if !known {
                return Err(extra_column_error(name, phase));
            }
        }
    }
    Ok(())
}

pub(crate) fn missing_column_error(
    name: &str,
    kind: FieldKind,
    present: &[&str],
    record: Option<usize>,
    phase: Phase,
) -> RustpyError {
    let shown: Vec<&str> = present.iter().take(30).copied().collect();
    let where_ = record.map(|r| format!("record {r} ")).unwrap_or_default();
    RustpyError::new(
        ErrorCode::SchemaMissingColumn,
        phase,
        format!("{where_}is missing required field `{name}`"),
    )
    .field(name)
    .expected(format!("field `{name}` ({}; use null for a missing value)", kind.as_str()))
    .actual(format!("fields present: [{}]", shown.join(", ")))
    .next(format!(
        "add `{name}` to the input (null is allowed and is imputed); the bundle's input.schema.json lists every field"
    ))
}

pub(crate) fn extra_column_error(name: &str, phase: Phase) -> RustpyError {
    RustpyError::new(
        ErrorCode::SchemaExtraColumn,
        phase,
        format!("field `{name}` is not part of the model input"),
    )
    .field(name)
    .next("remove the field, or retrain with preprocess.extra_columns = \"ignore\"")
}

/// Applies the frozen transform to a table. Statistics are never re-estimated here.
pub fn apply_transform(t: &FittedTransform, table: &Table, phase: Phase) -> Result<Matrix> {
    let rows = table.n_rows();
    let cols = t.output_dim;
    let mut data = vec![0f32; rows * cols];
    let tokens = MissingTokens::new(&t.missing_tokens);
    for feature in &t.features {
        let name = feature.name();
        let column = table.column(name).ok_or_else(|| {
            missing_column_error(name, feature_kind(feature), &table.names(), None, phase)
        })?;
        match feature {
            FeatureTransform::Numeric {
                impute,
                mean,
                std,
                offset,
                ..
            } => {
                for row in 0..rows {
                    let v = column.get(row);
                    let v = tokens.numeric(v);
                    let x = coerce_numeric(v, t.nan)
                        .map_err(|e| e.into_error(name, row, v, phase))?
                        .unwrap_or(*impute);
                    let z = ((x - mean) / std) as f32;
                    if !z.is_finite() {
                        return Err(RustpyError::new(
                            ErrorCode::InvalidValue,
                            phase,
                            format!("field `{name}`, row {row}: value {x} is too large to encode (out of the f32 range after scaling)"),
                        )
                        .field(name)
                        .expected(format!("a value within the f32 range after standardization (training mean {mean:.4}, std {std:.4})"))
                        .actual(x.to_string())
                        .next("check the units / clean the value before prediction"));
                    }
                    data[row * cols + offset] = z;
                }
            }
            FeatureTransform::Categorical {
                categories,
                missing_slot,
                offset,
                raw_dtype,
                ..
            } => {
                let index: HashMap<&str, usize> =
                    categories.iter().enumerate().map(|(i, c)| (c.as_str(), i)).collect();
                for row in 0..rows {
                    let v = column.get(row);
                    let v = tokens.categorical(v);
                    check_category_type(*raw_dtype, v).map_err(|e| e.into_error(name, row, v, phase))?;
                    let slot = match coerce_category(v, t.nan).map_err(|e| e.into_error(name, row, v, phase))? {
                        Some(c) => index.get(c.as_ref()).copied(),
                        None if *missing_slot => Some(categories.len()),
                        None => None,
                    };
                    match slot {
                        Some(s) => data[row * cols + offset + s] = 1.0,
                        None if t.unknown_category == UnknownCategoryPolicy::Ignore => {}
                        None => return Err(unknown_category_error(name, row, v, categories, phase)),
                    }
                }
            }
        }
    }
    Ok(Matrix { rows, cols, data })
}

/// The project's missing-value tokens, applied identically when fitting and when
/// transforming (CSV reading applies them too; Parquet and application records do not).
#[derive(Debug, Clone, Default)]
pub struct MissingTokens {
    text: Vec<String>,
    numeric: Vec<f64>,
}

impl MissingTokens {
    pub fn new(tokens: &[String]) -> Self {
        Self {
            text: tokens.to_vec(),
            numeric: tokens.iter().filter_map(|s| s.trim().parse::<f64>().ok()).filter(|f| !f.is_nan()).collect(),
        }
    }

    /// For numeric features: a token text or a number equal to a numeric token is missing.
    pub fn numeric<'a>(&self, v: ValueRef<'a>) -> ValueRef<'a> {
        match v {
            ValueRef::Text(s) if self.text.iter().any(|t| t == s) => ValueRef::Null,
            ValueRef::Int(i) if self.numeric.contains(&(i as f64)) => ValueRef::Null,
            ValueRef::Float(f) if self.numeric.contains(&f) => ValueRef::Null,
            other => other,
        }
    }

    /// For categorical features (and labels): a token text is missing.
    pub fn categorical<'a>(&self, v: ValueRef<'a>) -> ValueRef<'a> {
        match v {
            ValueRef::Text(s) if self.text.iter().any(|t| t == s) => ValueRef::Null,
            other => other,
        }
    }
}

/// A categorical value must have the raw type seen in training (null always allowed).
pub fn check_category_type(trained: Option<RawDtype>, v: ValueRef<'_>) -> Result<(), CoerceError> {
    let ok = match (trained, v) {
        (None, _) | (_, ValueRef::Null) => true,
        (Some(RawDtype::Text), ValueRef::Text(_)) => true,
        (Some(RawDtype::Bool), ValueRef::Bool(_)) => true,
        (Some(RawDtype::Int), ValueRef::Int(_)) => true,
        (Some(RawDtype::Int), ValueRef::Float(f)) => f.is_nan() || f.fract() == 0.0,
        (Some(RawDtype::Float), ValueRef::Float(_) | ValueRef::Int(_)) => true,
        _ => false,
    };
    if ok {
        Ok(())
    } else {
        Err(CoerceError::TypeMismatch {
            expected: match trained {
                Some(RawDtype::Text) => "text",
                Some(RawDtype::Bool) => "bool",
                Some(RawDtype::Int) => "integer",
                _ => "number",
            },
            actual: v.type_name(),
        })
    }
}

fn feature_kind(f: &FeatureTransform) -> FieldKind {
    match f {
        FeatureTransform::Numeric { .. } => FieldKind::Numeric,
        FeatureTransform::Categorical { .. } => FieldKind::Categorical,
    }
}

fn unknown_category_error(
    name: &str,
    row: usize,
    v: ValueRef<'_>,
    categories: &[String],
    phase: Phase,
) -> RustpyError {
    let known: Vec<&str> = categories.iter().take(12).map(String::as_str).collect();
    let more = if categories.len() > 12 { ", …" } else { "" };
    RustpyError::new(
        ErrorCode::UnknownCategory,
        phase,
        format!("field `{name}`, row {row}: category {} was not seen in training", v.preview()),
    )
    .field(name)
    .expected(format!("one of [{}{more}]", known.join(", ")))
    .actual(v.preview())
    .next(format!(
        "map the value to a known category, or retrain on data that contains it (or with preprocess.unknown_category = \"ignore\")"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::{Column, ColumnData, Value};

    fn transform(unknown: UnknownCategoryPolicy) -> FittedTransform {
        FittedTransform {
            version: rustpy_contracts::transform::TRANSFORM_VERSION.into(),
            features: vec![
                FeatureTransform::Numeric {
                    name: "x".into(),
                    impute: 2.0,
                    mean: 2.0,
                    std: 0.5,
                    offset: 0,
                },
                FeatureTransform::Categorical {
                    name: "c".into(),
                    categories: vec!["a".into(), "b".into()],
                    missing_slot: true,
                    offset: 1,
                    raw_dtype: Some(RawDtype::Text),
                },
            ],
            dropped: vec![],
            output_dim: 4,
            output_names: vec!["x".into(), "c=a".into(), "c=b".into(), "c=<missing>".into()],
            nan: NanPolicy::Missing,
            unknown_category: unknown,
            missing_tokens: vec!["".into(), "NA".into(), "-999".into()],
            fitted_on_split: "s".into(),
            fitted_rows: 3,
        }
    }

    fn table(x: Vec<Value>, c: Vec<Value>) -> Table {
        Table::new(vec![
            Column { name: "c".into(), data: ColumnData::Mixed(c) },
            Column { name: "x".into(), data: ColumnData::Mixed(x) },
        ])
        .unwrap()
    }

    #[test]
    fn encodes_with_frozen_statistics() {
        let t = transform(UnknownCategoryPolicy::Ignore);
        let tab = table(
            vec![Value::Float(3.0), Value::Null, Value::Float(f64::NAN), Value::Int(1)],
            vec![Value::from("b"), Value::Null, Value::from("zzz"), Value::from("a")],
        );
        let m = apply_transform(&t, &tab, Phase::Predict).unwrap();
        assert_eq!(m.row(0), &[2.0, 0.0, 1.0, 0.0]);
        assert_eq!(m.row(1), &[0.0, 0.0, 0.0, 1.0]);
        assert_eq!(m.row(2), &[0.0, 0.0, 0.0, 0.0]);
        assert_eq!(m.row(3), &[-2.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn unknown_category_error_policy() {
        let t = transform(UnknownCategoryPolicy::Error);
        let tab = table(vec![Value::Float(1.0)], vec![Value::from("zzz")]);
        let e = apply_transform(&t, &tab, Phase::Predict).unwrap_err();
        assert_eq!(e.code, ErrorCode::UnknownCategory);
        assert_eq!(e.field.as_deref(), Some("c"));
    }

    #[test]
    fn missing_tokens_types_and_overflow() {
        let t = transform(UnknownCategoryPolicy::Error);
        // "NA" text and the numeric sentinel -999 are missing values, like in the CSV reader.
        let tab = table(vec![Value::Int(-999), Value::Float(-999.0)], vec![Value::from("NA"), Value::from("")]);
        let m = apply_transform(&t, &tab, Phase::Predict).unwrap();
        assert_eq!(m.row(0), &[0.0, 0.0, 0.0, 1.0]);
        assert_eq!(m.row(1), &[0.0, 0.0, 0.0, 1.0]);
        // A field trained as text does not accept numbers (no silent "012" vs 12 mapping).
        let tab = table(vec![Value::Float(1.0)], vec![Value::Int(12)]);
        assert_eq!(apply_transform(&t, &tab, Phase::Predict).unwrap_err().code, ErrorCode::SchemaTypeMismatch);
        // Finite but huge inputs that overflow f32 after scaling are rejected.
        let tab = table(vec![Value::Float(1e39)], vec![Value::from("a")]);
        assert_eq!(apply_transform(&t, &tab, Phase::Predict).unwrap_err().code, ErrorCode::InvalidValue);
    }

    #[test]
    fn strict_types() {
        assert!(coerce_numeric(ValueRef::Text("1.0"), NanPolicy::Missing).is_err());
        assert!(coerce_numeric(ValueRef::Float(f64::INFINITY), NanPolicy::Missing).is_err());
        assert_eq!(coerce_numeric(ValueRef::Float(f64::NAN), NanPolicy::Missing), Ok(None));
        assert_eq!(coerce_numeric(ValueRef::Float(f64::NAN), NanPolicy::Error), Err(CoerceError::NaN));
        assert_eq!(coerce_category(ValueRef::Float(3.0), NanPolicy::Missing).unwrap().unwrap(), "3");
        assert_eq!(coerce_category(ValueRef::Int(3), NanPolicy::Missing).unwrap().unwrap(), "3");
        assert!(coerce_category(ValueRef::Float(3.5), NanPolicy::Missing).is_err());
    }
}
