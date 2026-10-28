//! Pseudo-random records for the `portable` test and the `compiled_vs_runtime` example:
//! the same records, from the same seed, on every platform.

use tabsmith_embedded::{FieldInfo, FieldKind, FieldValue};
use tabsmith_runtime::{Outcome, Record, Value};

/// SplitMix64: the same pseudo-random numbers on every platform.
pub struct Rng(pub u64);

impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    pub fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// A plausible random value for each field: numbers within four standard deviations,
/// categories seen in training, and now and then a missing value.
pub fn row(fields: &[FieldInfo], rng: &mut Rng) -> Vec<FieldValue<'static>> {
    fields
        .iter()
        .map(|f| {
            if rng.unit() < 0.05 {
                return FieldValue::Missing;
            }
            match f.kind {
                FieldKind::Numeric { mean, std } => FieldValue::Float(mean + std * (8.0 * rng.unit() - 4.0)),
                FieldKind::Text { categories } => FieldValue::Text(categories[(rng.next() % categories.len() as u64) as usize]),
                FieldKind::Bool => FieldValue::Bool(rng.next().is_multiple_of(2)),
            }
        })
        .collect()
}

/// The same values as a runtime `Record`.
pub fn record(fields: &[FieldInfo], values: &[FieldValue<'_>]) -> Record {
    fields.iter().zip(values).fold(Record::new(), |r, (f, v)| {
        let value = match *v {
            FieldValue::Missing => Value::Null,
            FieldValue::Bool(b) => Value::Bool(b),
            FieldValue::Int(i) => Value::Int(i),
            FieldValue::Float(x) => Value::Float(x),
            FieldValue::Text(s) => Value::Text(s.to_string()),
        };
        r.with(f.name, value)
    })
}

/// The FNV-1a offset basis: the hash of nothing.
pub const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;

/// FNV-1a over the bits of every number.
pub fn fnv(hash: &mut u64, bits: u64) {
    for b in bits.to_le_bytes() {
        *hash ^= u64::from(b);
        *hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
}

/// The numbers of a runtime prediction: its probabilities, or its value.
pub fn runtime_numbers(outcome: &Outcome) -> Vec<f64> {
    match outcome {
        Outcome::Binary { probability, .. } => vec![*probability],
        Outcome::Multiclass { probabilities, .. } => probabilities.clone(),
        Outcome::Regression { value } => vec![*value],
    }
}
