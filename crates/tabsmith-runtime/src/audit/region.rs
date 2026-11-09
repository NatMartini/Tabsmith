//! From a rule's conditions on input fields to a box of feature values, and back from a box
//! to a record: the region a search starts from, and the witness it ends with.

use tabsmith_contracts::config::UnknownCategoryPolicy;
use tabsmith_contracts::schema::RawDtype;
use tabsmith_contracts::transform::{FeatureTransform, FittedTransform, MissingRule};
use tabsmith_contracts::{Phase, Result};
use tabsmith_embedded::kernels;
use tabsmith_embedded::rules::{Domain, NUMERIC};

use super::parse::{Literal, OTHER_FIELDS, Rule, Test, invalid};
use crate::Predictor;
use crate::transform::{CoerceError, MissingTokens, apply_transform, check_category_type, coerce_category};
use crate::value::{Record, Value, ValueRef};

/// The rows of features a bundle's transform produces: which features are numbers and which
/// are slots of a category, in the form [`tabsmith_embedded::rules`] reads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FeatureDomain {
    /// For each feature: [`NUMERIC`], or the index of its category in `slots`.
    pub category: Vec<u32>,
    /// For each category: its slots, the features `first..end`.
    pub slots: Vec<[u32; 2]>,
    /// Whether a category may have all its slots at 0: a value the model never saw, which a
    /// bundle with `unknown_category = "ignore"` accepts.
    pub none: bool,
}

impl FeatureDomain {
    /// The domain of a bundle's transform.
    pub fn of(t: &FittedTransform) -> Self {
        let mut category = vec![NUMERIC; t.output_dim];
        let mut slots = Vec::new();
        for f in &t.features {
            if let FeatureTransform::Categorical { offset, .. } = f {
                let c = slots.len() as u32;
                for j in *offset..offset + f.width() {
                    category[j] = c;
                }
                slots.push([*offset as u32, (offset + f.width()) as u32]);
            }
        }
        Self {
            category,
            slots,
            none: t.unknown_category == UnknownCategoryPolicy::Ignore,
        }
    }

    /// Borrowed, as [`tabsmith_embedded::rules`] takes it.
    pub fn as_domain(&self) -> Domain<'_> {
        Domain {
            category: &self.category,
            slots: &self.slots,
        }
    }
}

/// What a rule allows a numeric field to be, in its own units.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Raw {
    /// A number in `[lo, hi]`.
    Range(f64, f64),
    /// A missing value, encoded as the imputed one.
    Missing,
}

/// A rule's region: the box of features its inputs encode to, and the raw limits a witness
/// must keep.
#[derive(Debug, Clone)]
pub(crate) struct Region {
    pub lo: Vec<f32>,
    pub hi: Vec<f32>,
    pub none: Vec<bool>,
    /// Per feature; only numeric features' entries are read.
    pub raw: Vec<Raw>,
}

/// A rule's value as a record would hold it.
fn literal_value(v: &Literal) -> Value {
    match v {
        Literal::Number(x) if x.fract() == 0.0 && x.abs() < 9.0e15 => Value::Int(*x as i64),
        Literal::Number(x) => Value::Float(*x),
        Literal::Text(s) => Value::Text(s.clone()),
        Literal::Bool(b) => Value::Bool(*b),
        Literal::Null => Value::Null,
    }
}

/// The canonical text of a category value, as the transform compares it.
fn category_text(v: &Literal) -> Option<String> {
    coerce_category(literal_value(v).as_ref(), tabsmith_contracts::config::NanPolicy::Missing)
        .ok()
        .flatten()
        .map(|c| c.into_owned())
}

impl Region {
    /// The region of `rule` for the bundle `p` loaded, with its conditions checked against the
    /// bundle's fields.
    pub fn of(p: &Predictor, rule: &Rule, text: &str) -> Result<Region> {
        let b = p.bundle();
        let t = &b.transform;
        let domain = FeatureDomain::of(t);
        let n = t.output_dim;
        // A feature is a finite number: the transform refuses the others.
        let mut lo = vec![-f32::MAX; n];
        let mut hi = vec![f32::MAX; n];
        let mut none = vec![domain.none; domain.slots.len()];
        let mut raw = vec![Raw::Range(f64::NEG_INFINITY, f64::INFINITY); n];
        for c in &domain.slots {
            for j in c[0] as usize..c[1] as usize {
                (lo[j], hi[j]) = (0.0, 1.0);
            }
        }
        // Which choices each category allows: its slots, then `none`.
        let mut allowed: Vec<Option<Vec<bool>>> = vec![None; domain.slots.len()];
        let mut named = vec![false; n];
        let mut others = None;
        // Standard deviations of training: an imported model has none to measure with.
        if !t.has_statistics()
            && let Some(cond) = rule.conditions.iter().find(|c| matches!(c.test, Test::Within(_)))
        {
            return Err(invalid(
                text,
                format!(
                    "`{} within … sd`: the bundle was imported and holds no training statistics; give the field a range with `in a..=b`",
                    cond.field
                ),
            ));
        }
        for cond in &rule.conditions {
            if cond.field == OTHER_FIELDS {
                let Test::Within(k) = cond.test else {
                    unreachable!("the parser only reads `_ within k sd`")
                };
                others = Some(others.map_or(k, |o: f64| o.min(k)));
                continue;
            }
            let feature = t.features.iter().find(|f| f.name() == cond.field);
            let Some(feature) = feature else {
                let e = if b.input_schema.field(&cond.field).is_some() {
                    invalid(
                        text,
                        format!(
                            "the model does not use `{}` (dropped when it was fitted): a rule on it limits nothing",
                            cond.field
                        ),
                    )
                } else {
                    let names: Vec<&str> = t.features.iter().map(|f| f.name()).collect();
                    invalid(text, format!("the bundle has no input field `{}`", cond.field)).actual(format!("fields: {}", names.join(", ")))
                };
                return Err(e);
            };
            match feature {
                FeatureTransform::Numeric {
                    offset,
                    mean,
                    std,
                    missing,
                    ..
                } => {
                    named[*offset] = true;
                    let limit = match &cond.test {
                        Test::Within(k) => Raw::Range(mean - k * std, mean + k * std),
                        Test::Range(a, b) => Raw::Range(*a, *b),
                        Test::AtMost(v) => Raw::Range(f64::NEG_INFINITY, *v),
                        Test::AtLeast(v) => Raw::Range(*v, f64::INFINITY),
                        Test::Equals(Literal::Number(v)) => Raw::Range(*v, *v),
                        Test::Equals(Literal::Null) if *missing == MissingRule::Refuse => {
                            return Err(invalid(text, format!("`{}` refuses missing values", cond.field)));
                        }
                        Test::Equals(Literal::Null) => Raw::Missing,
                        _ => {
                            return Err(invalid(
                                text,
                                format!(
                                    "`{}` is a number: limit it with `in a..=b`, `<=`, `>=`, `== value`, `== null` or `within k sd`",
                                    cond.field
                                ),
                            ));
                        }
                    };
                    raw[*offset] = match (raw[*offset], limit) {
                        (Raw::Range(a, b), Raw::Range(c, d)) if a.max(c) <= b.min(d) => Raw::Range(a.max(c), b.min(d)),
                        (Raw::Range(a, b), Raw::Missing) if a == f64::NEG_INFINITY && b == f64::INFINITY => Raw::Missing,
                        (Raw::Missing, Raw::Range(a, b)) if a == f64::NEG_INFINITY && b == f64::INFINITY => Raw::Missing,
                        _ => return Err(invalid(text, format!("the conditions on `{}` allow no value together", cond.field))),
                    };
                }
                FeatureTransform::Categorical {
                    categories,
                    missing_slot,
                    offset,
                    raw_dtype,
                    ..
                } => {
                    let values: Vec<&Literal> = match &cond.test {
                        Test::Equals(v) => vec![v],
                        Test::OneOf(v) => v.iter().collect(),
                        _ => {
                            return Err(invalid(
                                text,
                                format!("`{}` is a category: limit it with `== value` or `in [value, …]`", cond.field),
                            ));
                        }
                    };
                    let c = domain.category[*offset] as usize;
                    let width = feature.width();
                    let mut set = vec![false; width + 1];
                    let raw = raw_dtype.or_else(|| b.input_schema.field(&cond.field).map(|f| f.raw_dtype));
                    for v in values {
                        // A value of another type than the field's is refused by `predict`.
                        if let Err(CoerceError::TypeMismatch { expected, .. }) = check_category_type(raw, literal_value(v).as_ref()) {
                            return Err(invalid(text, format!("`{}` holds {expected} values, not {v}", cond.field)));
                        }
                        let slot = match category_text(v) {
                            None if *missing_slot => Some(categories.len()),
                            None if domain.none => None,
                            None => {
                                return Err(invalid(
                                    text,
                                    format!("`{}` refuses missing values: the bundle has no missing slot for it", cond.field),
                                ));
                            }
                            Some(s) => match categories.iter().position(|k| *k == s) {
                                Some(k) => Some(k),
                                None if domain.none => None,
                                None => {
                                    let shown: Vec<&str> = categories.iter().take(12).map(String::as_str).collect();
                                    return Err(invalid(
                                        text,
                                        format!(
                                            "`{}` never held {v} in training, and the bundle refuses unseen categories",
                                            cond.field
                                        ),
                                    )
                                    .actual(format!("categories: {}", shown.join(", "))));
                                }
                            },
                        };
                        set[slot.unwrap_or(width)] = true;
                    }
                    allowed[c] = Some(match allowed[c].take() {
                        None => set,
                        Some(old) => old.iter().zip(&set).map(|(a, b)| *a && *b).collect(),
                    });
                    if !allowed[c].as_ref().is_some_and(|s| s.iter().any(|&a| a)) {
                        return Err(invalid(text, format!("the conditions on `{}` allow no value together", cond.field)));
                    }
                }
            }
        }
        if let Some(k) = others {
            for f in &t.features {
                if let FeatureTransform::Numeric { mean, std, offset, .. } = f
                    && !named[*offset]
                {
                    raw[*offset] = Raw::Range(mean - k * std, mean + k * std);
                }
            }
        }
        // Numbers in raw units become standardised features; standardisation never decreases,
        // so the box holds every value of the range.
        for f in &t.features {
            if let FeatureTransform::Numeric {
                name,
                impute,
                mean,
                std,
                offset,
                ..
            } = f
            {
                let (a, b) = match raw[*offset] {
                    Raw::Range(a, b) => (a, b),
                    Raw::Missing => (*impute, *impute),
                };
                lo[*offset] = kernels::standardize(a, *mean, *std).max(-f32::MAX);
                hi[*offset] = kernels::standardize(b, *mean, *std).min(f32::MAX);
                if lo[*offset] > hi[*offset] {
                    return Err(invalid(
                        text,
                        format!("the model refuses every value of `{name}` the rule allows: they are too large to encode"),
                    ));
                }
            }
        }
        for (c, set) in allowed.iter().enumerate() {
            let Some(set) = set else { continue };
            let [first, end] = domain.slots[c];
            for (k, j) in (first as usize..end as usize).enumerate() {
                if !set[k] {
                    hi[j] = 0.0;
                }
            }
            none[c] = set[set.len() - 1];
            // One possible slot and no other value: that slot is 1.
            let open: Vec<usize> = (first as usize..end as usize).filter(|&j| hi[j] == 1.0).collect();
            if open.len() == 1 && !none[c] {
                tabsmith_embedded::rules::narrow(&domain.as_domain(), &mut lo, &mut hi, &mut none, open[0], 0.0, false);
            }
        }
        Ok(Region { lo, hi, none, raw })
    }
}

/// The f64 values in total order, as unsigned keys: `key(a) < key(b)` exactly when `a < b`
/// (with `-0.0` before `0.0`).
fn key(x: f64) -> u64 {
    let b = x.to_bits();
    if b >> 63 == 1 { !b } else { b | (1 << 63) }
}

fn from_key(k: u64) -> f64 {
    f64::from_bits(if k >> 63 == 1 { k & !(1 << 63) } else { !k })
}

/// The smallest `x` in `[a, b]` with `f(x) >= z`, for a non-decreasing `f`.
fn first_at_least(f: &dyn Fn(f64) -> f32, a: f64, b: f64, z: f32) -> Option<f64> {
    if f(a) >= z {
        return Some(a);
    }
    if f(b) < z {
        return None;
    }
    let (mut lo, mut hi) = (key(a), key(b));
    while hi - lo > 1 {
        let mid = lo + (hi - lo) / 2;
        if f(from_key(mid)) >= z { hi = mid } else { lo = mid }
    }
    Some(from_key(hi))
}

/// The largest `x` in `[a, b]` with `f(x) <= z`, for a non-decreasing `f`.
fn last_at_most(f: &dyn Fn(f64) -> f32, a: f64, b: f64, z: f32) -> Option<f64> {
    if f(b) <= z {
        return Some(b);
    }
    if f(a) > z {
        return None;
    }
    let (mut lo, mut hi) = (key(a), key(b));
    while hi - lo > 1 {
        let mid = lo + (hi - lo) / 2;
        if f(from_key(mid)) <= z { lo = mid } else { hi = mid }
    }
    Some(from_key(lo))
}

/// A value for a numeric feature: in the rule's range, encoding into `[lo, hi]`, and never one
/// of the bundle's missing-value markers. `keep` is kept when it qualifies; otherwise the value
/// is as close to `target` as the box allows, with as few decimals as that allows, a whole
/// number when the field held whole numbers in training.
#[allow(clippy::too_many_arguments)]
fn number_in(
    (lo, hi): (f32, f32),
    (a, b): (f64, f64),
    mean: f64,
    std: f64,
    target: f64,
    whole: bool,
    keep: Option<&Value>,
    tokens: &MissingTokens,
) -> Option<Value> {
    let s = |x: f64| kernels::standardize(x, mean, std);
    let (a, b) = (a.max(-f64::MAX), b.min(f64::MAX));
    let lo = lo.max(-f32::MAX);
    let hi = hi.min(f32::MAX);
    let first = first_at_least(&s, a, b, lo)?;
    let last = last_at_most(&s, a, b, hi)?;
    if first > last {
        return None;
    }
    let refused = |v: &Value| !matches!(tokens.numeric(v.as_ref()), ValueRef::Int(_) | ValueRef::Float(_));
    let kept = match keep {
        Some(Value::Int(i)) => Some(*i as f64),
        Some(Value::Float(x)) => Some(*x),
        _ => None,
    };
    if let (Some(v), Some(x)) = (keep, kept)
        && (first..=last).contains(&x)
        && !refused(v)
    {
        return Some(v.clone());
    }
    let x = kept.unwrap_or(target).clamp(first, last);
    if whole {
        let (c, d) = (first.ceil(), last.floor());
        if c <= d && c.abs() < 9.0e15 && d.abs() < 9.0e15 {
            let r = x.round().clamp(c, d);
            for v in [r, r + 1.0, r - 1.0, c, d] {
                let value = Value::Int(v as i64);
                if (c..=d).contains(&v) && !refused(&value) {
                    return Some(value);
                }
            }
        }
    }
    // The number with the fewest decimals in the interval, the closest to the target among those.
    for decimals in 0..=17 {
        let scale = 10f64.powi(decimals);
        let (c, d) = ((first * scale).ceil(), (last * scale).floor());
        if c > d || !c.is_finite() || !d.is_finite() {
            continue;
        }
        let r = (x * scale).round().clamp(c, d);
        for v in [r, r + 1.0, r - 1.0] {
            let v = v / scale;
            let value = Value::Float(v);
            if (first..=last).contains(&v) && !refused(&value) {
                return Some(value);
            }
        }
    }
    for v in [x, x.next_up(), x.next_down(), first, last] {
        let value = Value::Float(v);
        if (first..=last).contains(&v) && !refused(&value) {
            return Some(value);
        }
    }
    None
}

/// A record whose features lie in the box (`lo`, `hi`, `none`), within the rule's raw limits.
/// With `near`, it starts as that record and keeps every value the box allows; the numbers it
/// must move go as close to their old values as the box allows. Otherwise the numbers go as
/// close to `target` as the box allows (the training means without one). A category takes the
/// value `choice` asks (`Some(feature)` for the slot at that feature, `None` for a value it
/// never saw) or, without one, its first possible category. `None` when no input encodes into
/// the box. The record is checked by encoding it with the bundle's transform.
pub(crate) fn record_in(
    p: &Predictor,
    region: &Region,
    (lo, hi, none): (&[f32], &[f32], &[bool]),
    target: Option<&[f32]>,
    choice: Option<&[Option<usize>]>,
    near: Option<&Record>,
) -> Result<Option<(Record, Vec<f32>)>> {
    let b = p.bundle();
    let t = &b.transform;
    let tokens = MissingTokens::new(&t.missing_tokens);
    let mut record = match (near, b.examples.records.first()) {
        (Some(r), _) => r.clone(),
        (None, Some(r)) => Record::from_json(r)?,
        (None, None) => Record::new(),
    };
    let mut c = 0;
    for f in &t.features {
        match f {
            FeatureTransform::Numeric {
                name, mean, std, offset, ..
            } => {
                let j = *offset;
                let value = match region.raw[j] {
                    Raw::Missing => Some(Value::Null),
                    Raw::Range(a, bb) => {
                        let whole = b.input_schema.field(name).is_some_and(|f| f.raw_dtype == RawDtype::Int);
                        let goal = target.map_or(*mean, |z| mean + std * f64::from(z[j]));
                        let keep = near.and_then(|r| r.get(name));
                        number_in((lo[j], hi[j]), (a, bb), *mean, *std, goal, whole, keep, &tokens)
                    }
                };
                let Some(value) = value else { return Ok(None) };
                record.insert(name.clone(), value);
            }
            FeatureTransform::Categorical {
                name,
                categories,
                missing_slot,
                offset,
                raw_dtype,
            } => {
                let width = f.width();
                let can_be_one = |k: usize| lo[offset + k] <= 1.0 && 1.0 <= hi[offset + k];
                let forced = (0..width).find(|&k| lo[offset + k] > 0.0);
                // A choice names a slot by its feature index.
                let wanted: Option<Option<usize>> = choice.map(|ch| ch[c].map(|s| s - offset));
                // The value `near` holds, where the box allows it.
                let kept: Option<Option<usize>> = near.and_then(|r| r.get(name)).and_then(|v| {
                    let text = coerce_category(tokens.categorical(v.as_ref()), tabsmith_contracts::config::NanPolicy::Missing).ok()?;
                    match text.and_then(|s| categories.iter().position(|k| *k == s)) {
                        Some(k) => can_be_one(k).then_some(Some(k)),
                        None if *missing_slot && matches!(v, Value::Null) => can_be_one(categories.len()).then_some(Some(categories.len())),
                        None => none[c].then_some(None),
                    }
                });
                let wanted = kept.or(wanted);
                let pick: Option<Option<usize>> = match (forced, wanted) {
                    (Some(k), _) => Some(Some(k)),
                    (None, Some(Some(k))) if can_be_one(k) => Some(Some(k)),
                    (None, Some(None)) if none[c] => Some(None),
                    _ => (0..categories.len())
                        .find(|&k| can_be_one(k))
                        .or_else(|| (*missing_slot && can_be_one(categories.len())).then_some(categories.len()))
                        .map(Some)
                        .or(none[c].then_some(None)),
                };
                let Some(pick) = pick else { return Ok(None) };
                let raw = raw_dtype.or_else(|| b.input_schema.field(name).map(|f| f.raw_dtype));
                let value = match pick {
                    Some(k) if k == categories.len() => Value::Null,
                    Some(k) => typed(&categories[k], raw),
                    None if !missing_slot => Value::Null,
                    None => match unseen(categories, raw) {
                        Some(v) => v,
                        None => return Ok(None),
                    },
                };
                record.insert(name.clone(), value);
                c += 1;
            }
        }
    }
    let table = p.records_table(std::slice::from_ref(&record), &[])?;
    let row = match apply_transform(t, &table, Phase::Audit) {
        Ok(m) => m.data,
        Err(_) => return Ok(None),
    };
    let inside = (0..row.len()).all(|j| lo[j] <= row[j] && row[j] <= hi[j]);
    Ok(inside.then_some((record, row)))
}

/// A category as the type its field had in training.
fn typed(category: &str, raw: Option<RawDtype>) -> Value {
    match raw {
        Some(RawDtype::Bool) => Value::Bool(category == "true"),
        Some(RawDtype::Int) => category.parse::<i64>().map_or_else(|_| Value::Text(category.into()), Value::Int),
        Some(RawDtype::Float) => category.parse::<f64>().map_or_else(|_| Value::Text(category.into()), Value::Float),
        _ => Value::Text(category.into()),
    }
}

/// A value of the field's type that training never saw, if its type has one.
fn unseen(categories: &[String], raw: Option<RawDtype>) -> Option<Value> {
    match raw {
        Some(RawDtype::Bool) => [false, true]
            .into_iter()
            .find(|b| !categories.iter().any(|c| *c == b.to_string()))
            .map(Value::Bool),
        Some(RawDtype::Int | RawDtype::Float) => (0..)
            .map(|i: i64| -1 - i)
            .find(|i| !categories.iter().any(|c| *c == i.to_string()))
            .map(Value::Int),
        _ => {
            let mut s = "(unseen)".to_string();
            while categories.contains(&s) {
                s.insert(0, '~');
            }
            Some(Value::Text(s))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_order_every_f64() {
        let values = [
            f64::NEG_INFINITY,
            -f64::MAX,
            -1.0,
            -f64::MIN_POSITIVE,
            -0.0,
            0.0,
            5e-324,
            1.0,
            f64::MAX,
            f64::INFINITY,
        ];
        for w in values.windows(2) {
            assert!(key(w[0]) < key(w[1]), "{} {}", w[0], w[1]);
        }
        for v in values {
            assert_eq!(from_key(key(v)).to_bits(), v.to_bits());
        }
    }

    #[test]
    fn a_number_lands_in_its_box() {
        let tokens = MissingTokens::new(&["-999".to_string()]);
        let s = |x: f64| kernels::standardize(x, 38.6, 13.7);
        // A box between two thresholds, from a range in raw units.
        let (lo, hi) = (s(20.0).next_up(), s(21.5));
        let v = number_in((lo, hi), (17.0, 25.0), 38.6, 13.7, 38.6, true, None, &tokens).unwrap();
        assert_eq!(v, Value::Int(21));
        let Value::Float(x) = number_in((lo, hi), (17.0, 25.0), 38.6, 13.7, 38.6, false, None, &tokens).unwrap() else {
            panic!()
        };
        assert!(lo <= s(x) && s(x) <= hi && x <= 25.0, "{x}");
        assert_eq!(x, 21.0, "the fewest decimals, closest to the mean");
        let Value::Float(x) = number_in((s(20.11), s(20.13)), (17.0, 25.0), 38.6, 13.7, 38.6, false, None, &tokens).unwrap() else {
            panic!()
        };
        assert!(s(20.11) <= s(x) && s(x) <= s(20.13), "{x}");
        assert_eq!(x, 20.13, "two decimals, the end nearest the mean");
        // No whole number inside: a fraction.
        let (lo, hi) = (s(20.2), s(20.8));
        assert!(matches!(
            number_in((lo, hi), (17.0, 25.0), 38.6, 13.7, 0.0, true, None, &tokens),
            Some(Value::Float(_))
        ));
        // Outside the rule's range: nothing.
        assert_eq!(
            number_in((s(30.0), s(40.0)), (17.0, 25.0), 38.6, 13.7, 38.6, false, None, &tokens),
            None
        );
        // A missing-value marker is never the witness.
        let v = number_in((s(-1000.0), s(-998.0)), (-1000.0, -998.0), 38.6, 13.7, -999.0, true, None, &tokens).unwrap();
        assert_ne!(v, Value::Int(-999));
    }
}
