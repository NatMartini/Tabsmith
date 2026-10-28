//! Fitting the standard transform on training rows only.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use tabsmith_contracts::config::{ImputeStrategy, PreprocessConfig, ScaleStrategy, Task, UnknownCategoryPolicy, ZeroVariancePolicy};
use tabsmith_contracts::hash::{canonical_hash, short};
use tabsmith_contracts::schema::{FieldKind, InputField, InputSchema, OutputSchema, RawDtype, TargetScale};
use tabsmith_contracts::transform::{DroppedField, FeatureTransform, FittedTransform, TRANSFORM_VERSION};
use tabsmith_contracts::{ErrorCode, Phase, Result, TabsmithError};
use tabsmith_runtime::{MissingTokens, Table, check_category_type, coerce_category, coerce_numeric};

use crate::data::{ColumnRole, FeaturePlan, TargetInfo, Targets, targets};

/// Fitted transform with the input/output contracts derived from it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FittedArtifacts {
    pub transform_id: String,
    pub transform: FittedTransform,
    pub input_schema: InputSchema,
    pub output_schema: OutputSchema,
}

/// Identity of a fitted transform: split + preprocessing config + column roles + version.
pub fn transform_id(split_id: &str, pre: &PreprocessConfig, plan: &FeaturePlan, threshold: f64, missing_tokens: &[String]) -> String {
    let roles: Vec<(&str, &ColumnRole)> = plan.columns.iter().map(|c| (c.name.as_str(), &c.role)).collect();
    let h = canonical_hash(&(split_id, pre, roles, TRANSFORM_VERSION, threshold.to_bits(), missing_tokens));
    format!("t-{}", short(&h))
}

/// Fits on `train` (a table holding only training rows). `row_ids` maps table rows
/// back to snapshot row numbers for error messages.
#[allow(clippy::too_many_arguments)]
pub fn fit_transform(
    train: &Table,
    row_ids: &[usize],
    plan: &FeaturePlan,
    pre: &PreprocessConfig,
    split_id: &str,
    threshold: f64,
    id_column: Option<&str>,
    missing_tokens: &[String],
    csv_delimiter: Option<&str>,
) -> Result<FittedArtifacts> {
    let tokens = MissingTokens::new(missing_tokens);
    let mut features = Vec::new();
    let mut dropped = Vec::new();
    let mut fields = Vec::new();
    let mut names = Vec::new();
    let mut offset = 0usize;
    for (decision, kind) in plan.features() {
        let name = decision.name.as_str();
        let col = train.column(name).expect("feature column exists");
        let raw_dtype = decision.raw_dtype.unwrap_or(RawDtype::Text);
        let mut drop_reason: Option<String> = None;
        let mut nullable = true;
        match kind {
            FieldKind::Numeric => {
                let mut values = Vec::with_capacity(train.n_rows());
                for r in 0..train.n_rows() {
                    let v = tokens.numeric(col.get(r));
                    match coerce_numeric(v, pre.nan) {
                        Ok(Some(x)) => values.push(x),
                        Ok(None) => {}
                        Err(e) => return Err(e.into_error(name, row_ids[r], v, Phase::Fit)),
                    }
                }
                if values.is_empty() {
                    drop_reason = Some("all values missing in the training rows".into());
                } else {
                    let n = values.len() as f64;
                    let mean = values.iter().sum::<f64>() / n;
                    let var = values.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n;
                    let std = var.sqrt();
                    // Negated on purpose: a NaN standard deviation counts as zero variance.
                    #[allow(clippy::neg_cmp_op_on_partial_ord)]
                    let constant = !(std > 1e-12 * mean.abs().max(1.0));
                    if constant {
                        drop_reason = Some(format!("zero variance in the training rows (constant {mean})"));
                    } else {
                        let impute = match pre.numeric_impute {
                            ImputeStrategy::Mean => mean,
                            ImputeStrategy::Median => median(&mut values),
                        };
                        let (m, s) = match pre.numeric_scale {
                            ScaleStrategy::Standard => (mean, std),
                            ScaleStrategy::None => (0.0, 1.0),
                        };
                        features.push(FeatureTransform::Numeric {
                            name: name.to_string(),
                            impute,
                            mean: m,
                            std: s,
                            offset,
                        });
                        names.push(name.to_string());
                        offset += 1;
                    }
                }
            }
            FieldKind::Categorical => {
                let mut counts: HashMap<String, usize> = HashMap::new();
                let mut missing = 0usize;
                for r in 0..train.n_rows() {
                    let v = tokens.categorical(col.get(r));
                    if let Err(e) = check_category_type(Some(raw_dtype), v) {
                        return Err(e.into_error(name, row_ids[r], v, Phase::Fit));
                    }
                    match coerce_category(v, pre.nan) {
                        Ok(Some(c)) => *counts.entry(c.into_owned()).or_default() += 1,
                        Ok(None) => missing += 1,
                        Err(e) => return Err(e.into_error(name, row_ids[r], v, Phase::Fit)),
                    }
                }
                if counts.len() > pre.max_categories {
                    return Err(TabsmithError::new(
                        ErrorCode::CardinalityBudget,
                        Phase::Fit,
                        format!(
                            "categorical column `{name}` has {} distinct training values, above preprocess.max_categories = {}",
                            counts.len(),
                            pre.max_categories
                        ),
                    )
                    .field(name)
                    .expected(format!("at most {} categories", pre.max_categories))
                    .actual(counts.len().to_string())
                    .next(format!(
                        "exclude it (data.exclude = [\"{name}\"]), group rare values upstream, or raise preprocess.max_categories"
                    )));
                }
                let mut cats: Vec<(String, usize)> = counts.into_iter().collect();
                cats.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
                let slots = cats.len() + usize::from(missing > 0);
                if slots <= 1 {
                    drop_reason = Some("a single value in the training rows (zero variance)".into());
                } else {
                    let categories: Vec<String> = cats.into_iter().map(|(c, _)| c).collect();
                    for c in &categories {
                        names.push(format!("{name}={c}"));
                    }
                    if missing > 0 {
                        names.push(format!("{name}=<missing>"));
                    }
                    nullable = missing > 0 || pre.unknown_category == UnknownCategoryPolicy::Ignore;
                    features.push(FeatureTransform::Categorical {
                        name: name.to_string(),
                        categories,
                        missing_slot: missing > 0,
                        offset,
                        raw_dtype: Some(raw_dtype),
                    });
                    offset += slots;
                }
            }
        }
        if let Some(reason) = &drop_reason {
            if pre.zero_variance == ZeroVariancePolicy::Error {
                return Err(TabsmithError::new(
                    ErrorCode::DataUnsupported,
                    Phase::Fit,
                    format!("feature `{name}` cannot be used: {reason}"),
                )
                .field(name)
                .next(format!(
                    "exclude it (data.exclude = [\"{name}\"]) or set preprocess.zero_variance = \"drop\""
                )));
            }
            dropped.push(DroppedField {
                name: name.to_string(),
                reason: reason.clone(),
            });
        }
        fields.push(InputField {
            name: name.to_string(),
            kind,
            raw_dtype,
            required: drop_reason.is_none(),
            nullable,
            note: drop_reason.map(|r| format!("not used by the model: {r}")),
        });
    }
    if features.is_empty() {
        return Err(TabsmithError::new(
            ErrorCode::DataUnsupported,
            Phase::Fit,
            "every feature was dropped (zero variance or all missing in the training rows)",
        )
        .next("check the data; at least one informative feature is needed"));
    }
    let transform = FittedTransform {
        version: TRANSFORM_VERSION.to_string(),
        features,
        dropped,
        output_dim: offset,
        output_names: names,
        nan: pre.nan,
        unknown_category: pre.unknown_category,
        missing_tokens: missing_tokens.to_vec(),
        fitted_on_split: split_id.to_string(),
        fitted_rows: train.n_rows(),
    };
    transform.check_consistency().map_err(|m| TabsmithError::internal(Phase::Fit, m))?;
    let input_schema = InputSchema {
        fields,
        missing_column: "error".to_string(),
        extra_columns: pre.extra_columns,
        unknown_category: pre.unknown_category,
        nan: pre.nan,
        infinite: "error".to_string(),
        csv_delimiter: csv_delimiter.map(str::to_string),
        passthrough_id: id_column.map(str::to_string),
    };
    let output_schema = match plan.target.task {
        Task::Binary => OutputSchema::new(
            &plan.target.name,
            &plan.target.negative_label,
            &plan.target.positive_label,
            threshold,
        ),
        Task::Multiclass => OutputSchema::multiclass(&plan.target.name, plan.target.classes.clone()),
        Task::Regression => OutputSchema::regression(&plan.target.name, target_scale(train, &plan.target)?),
    };
    Ok(FittedArtifacts {
        transform_id: transform_id(split_id, pre, plan, threshold, missing_tokens),
        transform,
        input_schema,
        output_schema,
    })
}

/// Mean and standard deviation of the training rows' regression targets: the model
/// learns `(value - mean) / std`, so its loss and learning rate do not depend on
/// the target's units.
fn target_scale(train: &Table, target: &TargetInfo) -> Result<TargetScale> {
    let Targets::Values(y) = targets(train, target)? else {
        return Err(TabsmithError::internal(Phase::Fit, "regression target is not numeric"));
    };
    let n = y.len().max(1) as f64;
    let mean = y.iter().sum::<f64>() / n;
    let std = (y.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n).sqrt();
    if !(std.is_finite() && std > 0.0) {
        return Err(TabsmithError::new(
            ErrorCode::TargetInvalid,
            Phase::Fit,
            format!("target `{}` is constant on the training rows", target.name),
        )
        .field(target.name.clone())
        .next("a regression needs a target that varies; check the data or the split"));
    }
    Ok(TargetScale { mean, std })
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(|a, b| a.total_cmp(b));
    let n = values.len();
    if n % 2 == 1 {
        values[n / 2]
    } else {
        (values[n / 2 - 1] + values[n / 2]) / 2.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{ColumnDecision, TargetInfo};
    use tabsmith_runtime::{Column, ColumnData, apply_transform};

    fn plan() -> FeaturePlan {
        let f = |n: &str, k| ColumnDecision {
            name: n.into(),
            role: ColumnRole::Feature { kind: k },
            raw_dtype: Some(if k == FieldKind::Categorical {
                RawDtype::Text
            } else {
                RawDtype::Float
            }),
            decided_by: "inferred".into(),
        };
        FeaturePlan {
            columns: vec![
                f("x", FieldKind::Numeric),
                f("c", FieldKind::Categorical),
                f("k", FieldKind::Numeric),
            ],
            target: TargetInfo::binary("y", "1", "0", 1, 1),
            warnings: vec![],
        }
    }

    #[test]
    fn statistics_come_from_training_rows_only() {
        let train = Table::new(vec![
            Column {
                name: "x".into(),
                data: ColumnData::Float(vec![Some(1.0), Some(3.0), None]),
            },
            Column {
                name: "c".into(),
                data: ColumnData::Text(vec![Some("b".into()), Some("a".into()), Some("b".into())]),
            },
            Column {
                name: "k".into(),
                data: ColumnData::Float(vec![Some(5.0), Some(5.0), Some(5.0)]),
            },
        ])
        .unwrap();
        let fa = fit_transform(
            &train,
            &[0, 1, 2],
            &plan(),
            &PreprocessConfig::default(),
            "sp",
            0.5,
            None,
            &[],
            None,
        )
        .unwrap();
        let t = &fa.transform;
        assert_eq!(t.output_dim, 3);
        assert_eq!(t.output_names, vec!["x", "c=b", "c=a"]);
        match &t.features[0] {
            FeatureTransform::Numeric { mean, std, impute, .. } => {
                assert_eq!((*mean, *std, *impute), (2.0, 1.0, 2.0));
            }
            _ => panic!(),
        }
        assert_eq!(t.dropped.len(), 1);
        assert!(!fa.input_schema.field("k").unwrap().required);

        // A test table with a very different mean is encoded with the training statistics.
        let test = Table::new(vec![
            Column {
                name: "x".into(),
                data: ColumnData::Float(vec![Some(100.0)]),
            },
            Column {
                name: "c".into(),
                data: ColumnData::Text(vec![Some("a".into())]),
            },
        ])
        .unwrap();
        let m = apply_transform(t, &test, Phase::Predict).unwrap();
        assert_eq!(m.data, vec![98.0, 0.0, 1.0]);
    }

    #[test]
    fn cardinality_budget() {
        let vals: Vec<Option<String>> = (0..10).map(|i| Some(format!("v{i}"))).collect();
        let train = Table::new(vec![
            Column {
                name: "x".into(),
                data: ColumnData::Float((0..10).map(|i| Some(i as f64)).collect()),
            },
            Column {
                name: "c".into(),
                data: ColumnData::Text(vals),
            },
            Column {
                name: "k".into(),
                data: ColumnData::Float((0..10).map(|i| Some(i as f64)).collect()),
            },
        ])
        .unwrap();
        let pre = PreprocessConfig {
            max_categories: 5,
            ..Default::default()
        };
        let e = fit_transform(&train, &(0..10).collect::<Vec<_>>(), &plan(), &pre, "sp", 0.5, None, &[], None).unwrap_err();
        assert_eq!(e.code, ErrorCode::CardinalityBudget);
    }
}
