//! Raw input values, records and columnar tables.

use std::collections::BTreeMap;

use tabsmith_contracts::{ErrorCode, Phase, Result, TabsmithError};

/// A raw input value before coercion to a feature.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    /// A missing value; imputed as in training.
    Null,
    /// A boolean.
    Bool(bool),
    /// An integer.
    Int(i64),
    /// A floating-point number.
    Float(f64),
    /// Text: a category, or a number or date written as text.
    Text(String),
}

/// Borrowed view of a [`Value`]; this is what the transform consumes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ValueRef<'a> {
    /// A missing value.
    Null,
    /// A boolean.
    Bool(bool),
    /// An integer.
    Int(i64),
    /// A floating-point number.
    Float(f64),
    /// Borrowed text.
    Text(&'a str),
}

impl Value {
    /// Borrows this value as a [`ValueRef`].
    pub fn as_ref(&self) -> ValueRef<'_> {
        match self {
            Value::Null => ValueRef::Null,
            Value::Bool(b) => ValueRef::Bool(*b),
            Value::Int(i) => ValueRef::Int(*i),
            Value::Float(f) => ValueRef::Float(*f),
            Value::Text(s) => ValueRef::Text(s),
        }
    }
}

impl ValueRef<'_> {
    /// The type name used in error messages: `null`, `bool`, `int`, `float` or `text`.
    pub fn type_name(&self) -> &'static str {
        match self {
            ValueRef::Null => "null",
            ValueRef::Bool(_) => "bool",
            ValueRef::Int(_) => "int",
            ValueRef::Float(_) => "float",
            ValueRef::Text(_) => "text",
        }
    }

    /// Short display form for error messages (text truncated to 40 chars).
    pub fn preview(&self) -> String {
        match self {
            ValueRef::Null => "null".into(),
            ValueRef::Bool(b) => b.to_string(),
            ValueRef::Int(i) => i.to_string(),
            ValueRef::Float(f) => f.to_string(),
            ValueRef::Text(s) => {
                let mut t: String = s.chars().take(40).collect();
                if s.chars().count() > 40 {
                    t.push('…');
                }
                format!("{t:?}")
            }
        }
    }
}

macro_rules! value_from {
    ($($t:ty => $variant:ident as $conv:ty),* $(,)?) => {
        $(impl From<$t> for Value {
            fn from(v: $t) -> Self { Value::$variant(v as $conv) }
        })*
    };
}

value_from!(f64 => Float as f64, f32 => Float as f64, i64 => Int as i64, i32 => Int as i64,
            i16 => Int as i64, i8 => Int as i64, u32 => Int as i64, u16 => Int as i64, u8 => Int as i64);

impl From<bool> for Value {
    fn from(v: bool) -> Self {
        Value::Bool(v)
    }
}

impl From<&str> for Value {
    fn from(v: &str) -> Self {
        Value::Text(v.to_string())
    }
}

impl From<String> for Value {
    fn from(v: String) -> Self {
        Value::Text(v)
    }
}

impl<T: Into<Value>> From<Option<T>> for Value {
    fn from(v: Option<T>) -> Self {
        v.map_or(Value::Null, Into::into)
    }
}

/// One input row: field name -> value. An explicit [`Value::Null`] is a missing
/// value; an absent field is a schema error.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Record {
    fields: BTreeMap<String, Value>,
}

impl Record {
    /// An empty record.
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder-style insert: `Record::new().with("temp", 71.5).with("line", "B")`.
    pub fn with(mut self, name: impl Into<String>, value: impl Into<Value>) -> Self {
        self.fields.insert(name.into(), value.into());
        self
    }

    /// Inserts a field, replacing any previous value of the same name.
    pub fn insert(&mut self, name: impl Into<String>, value: impl Into<Value>) {
        self.fields.insert(name.into(), value.into());
    }

    /// The value of a field, if the record has it.
    pub fn get(&self, name: &str) -> Option<&Value> {
        self.fields.get(name)
    }

    /// Whether the record has the field; a field holding [`Value::Null`] counts.
    pub fn contains(&self, name: &str) -> bool {
        self.fields.contains_key(name)
    }

    /// Field names, in sorted order.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.fields.keys().map(String::as_str)
    }

    /// Number of fields.
    pub fn len(&self) -> usize {
        self.fields.len()
    }

    /// Whether the record has no fields.
    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }

    /// Converts a flat JSON object. Nested arrays/objects are rejected.
    pub fn from_json(value: &serde_json::Value) -> Result<Self> {
        let serde_json::Value::Object(map) = value else {
            return Err(TabsmithError::new(
                ErrorCode::SchemaTypeMismatch,
                Phase::Predict,
                "a record must be a JSON object of field -> value",
            )
            .actual(json_kind(value).to_string()));
        };
        let mut rec = Record::new();
        for (k, v) in map {
            let value = match v {
                serde_json::Value::Null => Value::Null,
                serde_json::Value::Bool(b) => Value::Bool(*b),
                serde_json::Value::Number(n) => {
                    if let Some(i) = n.as_i64() {
                        Value::Int(i)
                    } else if let Some(f) = n.as_f64() {
                        Value::Float(f)
                    } else {
                        return Err(TabsmithError::new(
                            ErrorCode::InvalidValue,
                            Phase::Predict,
                            format!("number for field `{k}` is out of range"),
                        )
                        .field(k.clone()));
                    }
                }
                serde_json::Value::String(s) => Value::Text(s.clone()),
                other => {
                    return Err(TabsmithError::new(
                        ErrorCode::SchemaTypeMismatch,
                        Phase::Predict,
                        format!(
                            "field `{k}` holds a nested {}; only flat scalar values are supported",
                            json_kind(other)
                        ),
                    )
                    .field(k.clone())
                    .expected("null, bool, number or string")
                    .actual(json_kind(other).to_string()));
                }
            };
            rec.fields.insert(k.clone(), value);
        }
        Ok(rec)
    }

    /// Converts to a flat JSON object; a non-finite float becomes `null`.
    pub fn to_json(&self) -> serde_json::Value {
        let mut map = serde_json::Map::new();
        for (k, v) in &self.fields {
            let jv = match v {
                Value::Null => serde_json::Value::Null,
                Value::Bool(b) => serde_json::Value::Bool(*b),
                Value::Int(i) => serde_json::Value::from(*i),
                Value::Float(f) => serde_json::Number::from_f64(*f)
                    .map(serde_json::Value::Number)
                    .unwrap_or(serde_json::Value::Null),
                Value::Text(s) => serde_json::Value::String(s.clone()),
            };
            map.insert(k.clone(), jv);
        }
        serde_json::Value::Object(map)
    }
}

fn json_kind(v: &serde_json::Value) -> &'static str {
    match v {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "bool",
        serde_json::Value::Number(_) => "number",
        serde_json::Value::String(_) => "string",
        serde_json::Value::Array(_) => "array",
        serde_json::Value::Object(_) => "object",
    }
}

/// Typed column storage. Typed variants avoid per-cell enums for file data; `Mixed`
/// holds values coming from application records.
#[derive(Debug, Clone, PartialEq)]
pub enum ColumnData {
    /// Floating-point values; `None` is missing.
    Float(Vec<Option<f64>>),
    /// Integers; `None` is missing.
    Int(Vec<Option<i64>>),
    /// Booleans; `None` is missing.
    Bool(Vec<Option<bool>>),
    /// Text; `None` is missing.
    Text(Vec<Option<String>>),
    /// Values of any type, as application records provide them.
    Mixed(Vec<Value>),
}

impl ColumnData {
    /// Number of rows.
    pub fn len(&self) -> usize {
        match self {
            ColumnData::Float(v) => v.len(),
            ColumnData::Int(v) => v.len(),
            ColumnData::Bool(v) => v.len(),
            ColumnData::Text(v) => v.len(),
            ColumnData::Mixed(v) => v.len(),
        }
    }

    /// Whether the column has no rows.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The value in `row`. Panics if `row` is out of bounds.
    pub fn get(&self, row: usize) -> ValueRef<'_> {
        match self {
            ColumnData::Float(v) => v[row].map_or(ValueRef::Null, ValueRef::Float),
            ColumnData::Int(v) => v[row].map_or(ValueRef::Null, ValueRef::Int),
            ColumnData::Bool(v) => v[row].map_or(ValueRef::Null, ValueRef::Bool),
            ColumnData::Text(v) => v[row].as_deref().map_or(ValueRef::Null, ValueRef::Text),
            ColumnData::Mixed(v) => v[row].as_ref(),
        }
    }

    /// Keeps only the given rows (in the given order).
    pub fn take(&self, rows: &[usize]) -> ColumnData {
        match self {
            ColumnData::Float(v) => ColumnData::Float(rows.iter().map(|&r| v[r]).collect()),
            ColumnData::Int(v) => ColumnData::Int(rows.iter().map(|&r| v[r]).collect()),
            ColumnData::Bool(v) => ColumnData::Bool(rows.iter().map(|&r| v[r]).collect()),
            ColumnData::Text(v) => ColumnData::Text(rows.iter().map(|&r| v[r].clone()).collect()),
            ColumnData::Mixed(v) => ColumnData::Mixed(rows.iter().map(|&r| v[r].clone()).collect()),
        }
    }
}

/// A named column of a [`Table`].
#[derive(Debug, Clone, PartialEq)]
pub struct Column {
    /// Column name, as in the bundle's input schema.
    pub name: String,
    /// The values, one per row.
    pub data: ColumnData,
}

/// A columnar table of raw values.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Table {
    columns: Vec<Column>,
    rows: usize,
}

impl Table {
    /// Builds a table from columns of equal length; columns of different
    /// lengths are an error.
    pub fn new(columns: Vec<Column>) -> Result<Self> {
        let rows = columns.first().map_or(0, |c| c.data.len());
        if let Some(bad) = columns.iter().find(|c| c.data.len() != rows) {
            return Err(TabsmithError::internal(
                Phase::Data,
                format!("column `{}` has {} rows, expected {rows}", bad.name, bad.data.len()),
            ));
        }
        Ok(Self { columns, rows })
    }

    /// Builds a table holding exactly the named columns from records. Records must
    /// already have been checked for presence of these fields.
    pub fn from_records(records: &[Record], names: &[&str]) -> Self {
        let columns = names
            .iter()
            .map(|name| Column {
                name: name.to_string(),
                data: ColumnData::Mixed(records.iter().map(|r| r.get(name).cloned().unwrap_or(Value::Null)).collect()),
            })
            .collect();
        Self {
            columns,
            rows: records.len(),
        }
    }

    /// Number of rows.
    pub fn n_rows(&self) -> usize {
        self.rows
    }

    /// The columns, in order.
    pub fn columns(&self) -> &[Column] {
        &self.columns
    }

    /// The column with this name, if any.
    pub fn column(&self, name: &str) -> Option<&ColumnData> {
        self.columns.iter().find(|c| c.name == name).map(|c| &c.data)
    }

    /// Column names, in order.
    pub fn names(&self) -> Vec<&str> {
        self.columns.iter().map(|c| c.name.as_str()).collect()
    }

    /// Keeps only the named columns that exist (in the given order).
    pub fn select(&self, names: &[&str]) -> Table {
        Table {
            columns: names
                .iter()
                .filter_map(|n| self.columns.iter().find(|c| c.name == *n).cloned())
                .collect(),
            rows: self.rows,
        }
    }

    /// Keeps only the given rows (in the given order).
    pub fn take(&self, rows: &[usize]) -> Table {
        Table {
            columns: self
                .columns
                .iter()
                .map(|c| Column {
                    name: c.name.clone(),
                    data: c.data.take(rows),
                })
                .collect(),
            rows: rows.len(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_builder_and_json() {
        let r = Record::new().with("a", 1.5).with("b", "x").with("c", Option::<f64>::None);
        assert_eq!(r.get("a"), Some(&Value::Float(1.5)));
        assert_eq!(r.get("c"), Some(&Value::Null));
        let back = Record::from_json(&r.to_json()).unwrap();
        assert_eq!(back, r);
        let nested = serde_json::json!({"a": [1, 2]});
        assert_eq!(Record::from_json(&nested).unwrap_err().code, ErrorCode::SchemaTypeMismatch);
    }

    #[test]
    fn table_take() {
        let t = Table::new(vec![Column {
            name: "x".into(),
            data: ColumnData::Int(vec![Some(1), None, Some(3)]),
        }])
        .unwrap();
        let s = t.take(&[2, 0]);
        assert_eq!(s.n_rows(), 2);
        assert_eq!(s.column("x").unwrap().get(0), ValueRef::Int(3));
    }
}
