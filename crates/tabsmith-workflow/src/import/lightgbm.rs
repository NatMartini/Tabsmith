//! LightGBM's text model (`booster.save_model("model.txt")`).
//!
//! LightGBM compares f64 inputs with f64 thresholds and adds f64 leaves in f64. The bundle
//! holds f32 values: each threshold rounded down, which splits every f32 input as LightGBM
//! does, and each leaf rounded to the nearest f32, which moves the outputs by a bound the
//! import computes.

use std::collections::HashMap;

use tabsmith_contracts::Result;
use tabsmith_contracts::config::Task;

use super::{Forest, Library, MAX_FEATURES, Node, invalid, unsupported};

/// LightGBM's `kZeroThreshold` (`1e-35f`): inputs this close to 0 count as zero.
const ZERO: f64 = 1e-35f32 as f64;

/// The largest f32 at most `t`: `x <= t` and `x <= round_down(t)` agree for every finite f32
/// `x` (above the f32 range, `f32::MAX` lets every one through, like `t`).
fn round_down(t: f64) -> f32 {
    let f = t as f32;
    let f = if f64::from(f) > t { f.next_down() } else { f };
    f.min(f32::MAX)
}

fn list<T: std::str::FromStr>(kv: &HashMap<&str, &str>, key: &str, len: usize, t: usize) -> Result<Vec<T>> {
    let text = kv.get(key).ok_or_else(|| invalid(format!("tree {t} has no {key}")))?;
    let v = text
        .split_whitespace()
        .map(|s| s.parse::<T>().map_err(|_| invalid(format!("tree {t}'s {key} holds `{s}`"))))
        .collect::<Result<Vec<T>>>()?;
    if v.len() != len {
        return Err(invalid(format!("tree {t}'s {key} has {} values, {len} expected", v.len())));
    }
    Ok(v)
}

pub(super) fn parse(text: &str) -> Result<Forest> {
    // A header of key=value lines and flags, then one block per tree, then "end of trees".
    let mut header: HashMap<&str, &str> = HashMap::new();
    let mut flags: Vec<&str> = Vec::new();
    let mut trees: Vec<HashMap<&str, &str>> = Vec::new();
    let mut ended = false;
    for line in text.lines().skip(1) {
        let line = line.trim();
        if line == "end of trees" {
            ended = true;
            break;
        }
        if line.is_empty() {
            continue;
        }
        if line.starts_with("Tree=") {
            trees.push(HashMap::new());
            continue;
        }
        match line.split_once('=') {
            Some((k, v)) => {
                trees.last_mut().unwrap_or(&mut header).insert(k, v);
            }
            None if trees.is_empty() => flags.push(line),
            None => {}
        }
    }
    if !ended {
        return Err(invalid("the model ends before `end of trees`"));
    }
    let version = header.get("version").copied().unwrap_or("unknown").to_string();
    let objective = header.get("objective").copied().unwrap_or("").to_string();
    if flags.contains(&"average_output") {
        return Err(unsupported("a random forest (boosting=rf averages its trees)"));
    }
    let num = |key: &str| -> Result<usize> {
        header
            .get(key)
            .ok_or_else(|| invalid(format!("the header has no {key}")))?
            .trim()
            .parse()
            .map_err(|_| invalid(format!("the header's {key} is not a count")))
    };
    let n_features = num("max_feature_idx")?
        .checked_add(1)
        .filter(|&n| n <= MAX_FEATURES)
        .ok_or_else(|| unsupported(format!("more than {MAX_FEATURES} features")))?;
    let n_class = num("num_class")?;
    let per_round = num("num_tree_per_iteration")?;
    let mut words = objective.split_whitespace();
    let name = words.next().unwrap_or("");
    let params: HashMap<&str, &str> = words.filter_map(|w| w.split_once(':')).collect();
    let plain = objective
        .split_whitespace()
        .skip(1)
        .filter(|w| !w.contains(':'))
        .collect::<Vec<_>>();
    let task = match name {
        "binary" => {
            let sigmoid = params.get("sigmoid").map_or(Ok(1.0), |s| s.parse::<f64>()).unwrap_or(f64::NAN);
            if sigmoid != 1.0 {
                return Err(unsupported(format!(
                    "a binary model with sigmoid {sigmoid} (Tabsmith's probability is sigmoid(output))"
                )));
            }
            Task::Binary
        }
        "multiclass" => Task::Multiclass,
        "regression" | "regression_l1" | "huber" | "fair" | "quantile" | "mape" if plain.is_empty() => Task::Regression,
        "regression" if plain.contains(&"sqrt") => {
            return Err(unsupported(
                "a regression trained on the square root of its target (reg_sqrt): its outputs are squared",
            ));
        }
        o => {
            let why = match o {
                "poisson" | "gamma" | "tweedie" => {
                    "its outputs are the exponential of the raw score, and a Tabsmith regression output is linear"
                }
                "multiclassova" => "one sigmoid per class rather than a softmax",
                "cross_entropy" | "cross_entropy_lambda" => "a probability target rather than labels",
                "lambdarank" | "rank_xendcg" => "it ranks rather than predicts a label or a value",
                _ => "Tabsmith does not know how it turns a raw score into a prediction",
            };
            return Err(unsupported(format!("objective `{objective}`: {why}")).next(
                "train with binary, multiclass or a regression with an identity link (regression, regression_l1, huber, fair, quantile, mape)",
            ));
        }
    };
    let n_outputs = if task == Task::Multiclass { n_class } else { 1 };
    if task == Task::Multiclass && n_outputs < 3 {
        return Err(unsupported(format!("a multiclass model of {n_outputs} classes")).next("train two classes with objective=binary"));
    }
    if per_round != n_outputs {
        return Err(invalid(format!("{per_round} trees per round for {n_outputs} outputs")));
    }
    let names: Vec<String> = match header.get("feature_names") {
        Some(s) => s.split_whitespace().map(str::to_string).collect(),
        None => (0..n_features).map(|i| format!("Column_{i}")).collect(),
    };
    if names.len() != n_features {
        return Err(invalid(format!("the model names {} features but has {n_features}", names.len())));
    }
    if trees.is_empty() {
        return Err(invalid("the model has no trees"));
    }
    if n_outputs > trees.len() {
        return Err(unsupported(format!(
            "{n_outputs} classes for {} trees: each class needs at least one tree",
            trees.len()
        )));
    }

    let mut forest = Forest::new(Library::LightGBM, version, objective.clone(), task, names, n_outputs);
    // Per output: the leaves' rounding, and the largest leaf, summed over its trees.
    let (mut rounding, mut largest) = (vec![0f64; n_outputs], vec![0f64; n_outputs]);
    let mut listed = Vec::with_capacity(trees.len());
    for (t, kv) in trees.iter().enumerate() {
        let (nodes, round, big) = tree(kv, t, n_features)?;
        rounding[t % n_outputs] += round;
        largest[t % n_outputs] += big;
        listed.push((t % n_outputs, (t, nodes)));
    }
    for (t, nodes) in Forest::interleave(listed, n_outputs)? {
        forest.push_tree(&nodes, 0, t)?;
    }
    // |Σ f32 leaves − Σ f64 leaves| plus both sums' roundings (n additions each, every partial
    // sum at most Σ |leaf|) plus rounding the output to f32.
    let n_trees = trees.len() as f64;
    forest.max_output_difference = rounding
        .iter()
        .zip(&largest)
        .map(|(r, l)| r + l * (n_trees * f64::EPSILON + f64::from(f32::EPSILON) / 2.0))
        .fold(0.0, f64::max);
    forest.notes = vec![
        format!(
            "LightGBM's leaves are f64; the bundle holds the nearest f32, so its outputs differ from LightGBM's raw scores by at most {:.2e} for inputs that are f32 values",
            forest.max_output_difference
        ),
        "thresholds are rounded down to f32: every input that is an f32 value takes LightGBM's branch".into(),
    ];
    Ok(forest)
}

/// One tree's nodes (internal nodes first, then leaves), the largest rounding of a leaf to
/// f32 and the largest |leaf|.
fn tree(kv: &HashMap<&str, &str>, t: usize, n_features: usize) -> Result<(Vec<Node>, f64, f64)> {
    let leaves: usize = kv
        .get("num_leaves")
        .and_then(|s| s.trim().parse().ok())
        .filter(|&n| n >= 1)
        .ok_or_else(|| invalid(format!("tree {t} has no num_leaves")))?;
    if kv.get("num_cat").is_some_and(|s| s.trim() != "0") {
        return Err(unsupported(format!("tree {t} has categorical splits")).next("one-hot encode categorical features before training"));
    }
    if kv.get("is_linear").is_some_and(|s| s.trim() != "0") {
        return Err(unsupported(format!("tree {t} is a linear tree (linear_tree=true)")));
    }
    let values: Vec<f64> = list(kv, "leaf_value", leaves, t)?;
    let mut nodes = Vec::with_capacity(2 * leaves - 1);
    let (mut round, mut big) = (0f64, 0f64);
    let mut leaf_nodes = Vec::with_capacity(leaves);
    for &v in &values {
        let v32 = v as f32;
        if !v.is_finite() || !v32.is_finite() {
            return Err(invalid(format!("tree {t} has leaf value {v}")));
        }
        round = round.max((f64::from(v32) - v).abs());
        big = big.max(v.abs());
        leaf_nodes.push(Node::Leaf(v32));
    }
    if leaves == 1 {
        return Ok((leaf_nodes, round, big));
    }
    let inner = leaves - 1;
    let feature: Vec<usize> = list(kv, "split_feature", inner, t)?;
    let threshold: Vec<f64> = list(kv, "threshold", inner, t)?;
    let decision: Vec<u32> = list(kv, "decision_type", inner, t)?;
    let left: Vec<i64> = list(kv, "left_child", inner, t)?;
    let right: Vec<i64> = list(kv, "right_child", inner, t)?;
    // Internal node i is node i; leaf j (written ~j) is node inner + j.
    let child = |c: i64| -> Result<usize> {
        let id = if c >= 0 { c as usize } else { inner + (!c) as usize };
        if (c >= 0 && id < inner) || (c < 0 && id < inner + leaves) {
            Ok(id)
        } else {
            Err(invalid(format!("tree {t} has a child {c}")))
        }
    };
    for i in 0..inner {
        let (f, th, d) = (feature[i], threshold[i], decision[i]);
        if f >= n_features {
            return Err(invalid(format!("tree {t} splits on feature {f}")));
        }
        if d & 1 != 0 {
            return Err(
                unsupported(format!("tree {t} has a categorical split")).next("one-hot encode categorical features before training")
            );
        }
        if th.is_nan() {
            return Err(invalid(format!("tree {t} has a split without a threshold")));
        }
        let default_left = d & 2 != 0;
        // A missing value: as 0 (missing type None), by the default (NaN), or as zero, which
        // goes by the default however it compares (Zero): kept only when the default is where
        // the threshold sends values near 0 anyway.
        let missing_left = match (d >> 2) & 3 {
            0 => 0.0 <= th,
            2 => default_left,
            1 if (default_left && th >= ZERO) || (!default_left && th < -ZERO) => default_left,
            1 => {
                return Err(unsupported(format!(
                    "tree {t} sends zero against its threshold (zero_as_missing=true); Tabsmith compares every value with the threshold"
                )));
            }
            m => return Err(invalid(format!("tree {t} has missing type {m}"))),
        };
        nodes.push(Node::Split {
            feature: f,
            threshold: round_down(th),
            missing_left,
            left: child(left[i])?,
            right: child(right[i])?,
        });
    }
    nodes.extend(leaf_nodes);
    Ok((nodes, round, big))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thresholds_round_down() {
        for t in [0.1f64, -0.1, 73.94295000000001, 1e-40, -1e-40, 3.0, 1e300, f64::INFINITY] {
            let r = round_down(t);
            assert!(f64::from(r) <= t, "{t}");
            assert!(f64::from(r.next_up()) > t || r == f32::MAX, "{t}");
        }
        assert_eq!(round_down(-1e300), f32::NEG_INFINITY);
        assert_eq!(round_down(2.5), 2.5);
    }

    #[test]
    fn a_tiny_model_reads_and_refuses_what_it_cannot_reproduce() {
        let model = |decision: &str, objective: &str| {
            format!(
                "tree\nversion=v4\nnum_class=1\nnum_tree_per_iteration=1\nlabel_index=0\nmax_feature_idx=1\nobjective={objective}\n\
                 feature_names=a b\n\nTree=0\nnum_leaves=2\nnum_cat=0\nsplit_feature=1\nthreshold=0.5\ndecision_type={decision}\n\
                 left_child=-1\nright_child=-2\nleaf_value=0.10000000000000001 -0.20000000000000001\nis_linear=0\nshrinkage=1\n\n\nend of trees\n"
            )
        };
        let f = parse(&model("2", "binary sigmoid:1")).unwrap();
        assert_eq!(f.feature, vec![1, -1, -1]);
        assert_eq!(f.value, vec![0.0, 0.1f32, -0.2f32]);
        assert_eq!(f.threshold[0], round_down(0.5));
        // Missing type None: a missing value goes as 0 does (left of 0.5).
        assert_eq!(f.missing[1].value(), Some(0.0));
        // 0.2 rounds by 3e-9; the f32 output's rounding adds up to 0.2 * 2^-24.
        assert!(
            f.max_output_difference > 1.4e-8 && f.max_output_difference < 1.6e-8,
            "{}",
            f.max_output_difference
        );
        // Missing type NaN, default right: just above 0.5.
        assert_eq!(
            parse(&model("8", "binary sigmoid:1")).unwrap().missing[1].value(),
            Some(0.5f32.next_up())
        );
        // Zero as missing, default left with zero left of the threshold: kept.
        assert!(parse(&model("6", "regression")).is_ok());
        // Zero as missing, default right although zero is left of the threshold: refused.
        assert!(parse(&model("4", "regression")).is_err());
        assert!(parse(&model("2", "binary sigmoid:2")).is_err());
        assert!(parse(&model("2", "poisson")).is_err());
        assert!(parse(&model("3", "regression")).is_err());
        assert!(parse(&model("2", "regression sqrt")).is_err());
        // Counts the file declares but does not back are refused, and cannot overflow.
        let text = model("2", "regression");
        for (old, new) in [
            ("max_feature_idx=1", "max_feature_idx=18446744073709551615"),
            ("max_feature_idx=1", "max_feature_idx=1048576"),
            (
                "num_class=1\nnum_tree_per_iteration=1",
                "num_class=99999999\nnum_tree_per_iteration=99999999",
            ),
        ] {
            let e = parse(
                &text
                    .replace(old, new)
                    .replace("objective=regression", "objective=multiclass num_class:99999999"),
            )
            .unwrap_err();
            assert_eq!(e.code, tabsmith_contracts::ErrorCode::ImportUnsupported, "{new}: {e:?}");
        }
    }
}
