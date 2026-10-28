use core::fmt;

/// One raw input value, as an application reads it (a CSV cell, a JSON value, a
/// sensor register). A compiled model's `Input::from_values` turns a row of these into
/// its typed input, applying the same rules as the runtime does to JSON records.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FieldValue<'a> {
    /// No value (null, an empty cell).
    Missing,
    /// A boolean.
    Bool(bool),
    /// An integer.
    Int(i64),
    /// A floating-point number.
    Float(f64),
    /// Text.
    Text(&'a str),
}

impl FieldValue<'_> {
    /// The name of the value's type, as error messages show it.
    pub fn type_name(&self) -> &'static str {
        match self {
            FieldValue::Missing => "missing",
            FieldValue::Bool(_) => "bool",
            FieldValue::Int(_) => "integer",
            FieldValue::Float(_) => "number",
            FieldValue::Text(_) => "text",
        }
    }
}

/// What a compiled model knows about one of its input fields, from its bundle.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FieldInfo {
    /// The field's name in the data.
    pub name: &'static str,
    /// How the model encodes it.
    pub kind: FieldKind,
}

/// How a compiled model encodes an input field.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FieldKind {
    /// A number, standardised with the training rows' statistics.
    Numeric {
        /// Mean of the training rows.
        mean: f64,
        /// Standard deviation of the training rows.
        std: f64,
    },
    /// A text category, one-hot over the categories seen in training.
    Text {
        /// The categories, in training order.
        categories: &'static [&'static str],
    },
    /// A boolean category.
    Bool,
}

/// Why a compiled model refused an input. The runtime refuses the same inputs, with
/// the matching error code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputError {
    /// A value of the wrong type (`E_SCHEMA_TYPE_MISMATCH`).
    TypeMismatch {
        /// The input field.
        field: &'static str,
        /// The type the field takes.
        expected: &'static str,
    },
    /// NaN where the bundle's `nan` policy is `error` (`E_INVALID_VALUE`).
    NotANumber {
        /// The input field.
        field: &'static str,
    },
    /// An infinite number (`E_INVALID_VALUE`).
    Infinite {
        /// The input field.
        field: &'static str,
    },
    /// A number outside the f32 range once standardised (`E_INVALID_VALUE`).
    OutOfRange {
        /// The input field.
        field: &'static str,
    },
    /// A category the model has no slot for, with `unknown_category = "error"`
    /// (`E_UNKNOWN_CATEGORY`).
    UnknownCategory {
        /// The input field.
        field: &'static str,
    },
}

impl InputError {
    /// The input field the error is about.
    pub fn field(&self) -> &'static str {
        match *self {
            InputError::TypeMismatch { field, .. }
            | InputError::NotANumber { field }
            | InputError::Infinite { field }
            | InputError::OutOfRange { field }
            | InputError::UnknownCategory { field } => field,
        }
    }
}

impl fmt::Display for InputError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InputError::TypeMismatch { field, expected } => write!(f, "field `{field}`: expected {expected}"),
            InputError::NotANumber { field } => write!(f, "field `{field}`: NaN is not accepted (the bundle's nan policy is error)"),
            InputError::Infinite { field } => write!(f, "field `{field}`: infinite values are not accepted"),
            InputError::OutOfRange { field } => write!(f, "field `{field}`: the value is out of the f32 range once standardised"),
            InputError::UnknownCategory { field } => write!(f, "field `{field}`: a category the model was not trained with"),
        }
    }
}

impl core::error::Error for InputError {}

/// A compiled model's self-check found an example whose prediction differs, in any
/// bit, from the runtime's prediction recorded when the model was compiled: the
/// platform computes differently (for example, it flushes subnormal numbers to zero).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SelfCheckError {
    /// Index of the first example that differs.
    pub example: usize,
}

impl fmt::Display for SelfCheckError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "example {} predicts differently from the runtime; this platform's floating point differs",
            self.example
        )
    }
}

impl core::error::Error for SelfCheckError {}
