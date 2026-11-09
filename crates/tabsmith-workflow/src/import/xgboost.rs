//! XGBoost's JSON model (`booster.save_model("model.json")`), read so that the bundle
//! computes XGBoost's margins bit for bit.
//!
//! XGBoost keeps thresholds and leaves as f32 and writes them as the shortest decimal that
//! reads back to the same f32. They are parsed from their text straight to f32: reading them
//! as f64 first and then rounding to f32 rounds twice, which can land on the other f32.

use serde::Deserialize;
use serde_json::value::RawValue;
use tabsmith_contracts::config::Task;
use tabsmith_contracts::model::LeafSum;
use tabsmith_contracts::{Result, TabsmithError};

use super::logf::logf;
use super::{Forest, Library, MAX_FEATURES, Node, invalid, unsupported};

#[derive(Deserialize)]
struct File<'a> {
    #[serde(borrow)]
    learner: Learner<'a>,
    #[serde(default)]
    version: Vec<u64>,
}

#[derive(Deserialize)]
struct Learner<'a> {
    #[serde(default)]
    feature_names: Vec<String>,
    #[serde(default)]
    feature_types: Vec<String>,
    #[serde(borrow)]
    gradient_booster: Booster<'a>,
    learner_model_param: ModelParam,
    objective: Objective,
}

#[derive(Deserialize)]
struct ModelParam {
    base_score: String,
    #[serde(default)]
    num_class: Option<String>,
    num_feature: String,
    #[serde(default)]
    num_target: Option<String>,
}

#[derive(Deserialize)]
struct Objective {
    name: String,
}

#[derive(Deserialize)]
struct Booster<'a> {
    name: String,
    #[serde(borrow, default)]
    model: Option<&'a RawValue>,
}

#[derive(Deserialize)]
struct GbTree<'a> {
    #[serde(default)]
    gbtree_model_param: Option<GbTreeParam>,
    #[serde(borrow)]
    trees: Vec<Tree<'a>>,
    tree_info: Vec<i64>,
}

#[derive(Deserialize)]
struct GbTreeParam {
    #[serde(default)]
    num_parallel_tree: Option<String>,
}

#[derive(Deserialize)]
struct Tree<'a> {
    #[serde(default)]
    tree_param: Option<TreeParam>,
    left_children: Vec<i64>,
    right_children: Vec<i64>,
    split_indices: Vec<i64>,
    #[serde(borrow)]
    split_conditions: &'a RawValue,
    default_left: Vec<serde_json::Value>,
    #[serde(default)]
    split_type: Vec<i64>,
}

#[derive(Deserialize)]
struct TreeParam {
    #[serde(default)]
    size_leaf_vector: Option<String>,
}

/// f32 values from a JSON array's text (`[1.4375E0,-2.5E-1]`) or one number, each parsed
/// straight from its decimal to the nearest f32.
fn f32s(text: &str, what: &str) -> Result<Vec<f32>> {
    let t = text.trim();
    let inner = t.strip_prefix('[').and_then(|s| s.strip_suffix(']')).unwrap_or(t).trim();
    if inner.is_empty() {
        return Ok(Vec::new());
    }
    inner
        .split(',')
        .map(|v| {
            v.trim()
                .parse::<f32>()
                .map_err(|_| invalid(format!("{what} holds `{}`, which is not a number", v.trim())))
        })
        .collect()
}

fn count(text: &Option<String>, what: &str, default: usize) -> Result<usize> {
    match text {
        None => Ok(default),
        Some(s) => s.trim().parse().map_err(|_| invalid(format!("{what} is `{s}`, not a count"))),
    }
}

/// Objectives whose margin is the model's raw output (identity link).
const IDENTITY: [&str; 6] = [
    "reg:squarederror",
    "reg:linear",
    "reg:pseudohubererror",
    "reg:absoluteerror",
    "reg:quantileerror",
    "reg:squaredlogerror",
];

pub(super) fn parse(text: &str) -> Result<Forest> {
    let file: File<'_> = serde_json::from_str(text).map_err(|e| invalid(format!("not an XGBoost JSON model: {e}")))?;
    let l = file.learner;
    let version = if file.version.is_empty() {
        "unknown".to_string()
    } else {
        file.version.iter().map(u64::to_string).collect::<Vec<_>>().join(".")
    };
    let objective = l.objective.name.clone();
    let task = match objective.as_str() {
        "binary:logistic" | "binary:logitraw" => Task::Binary,
        "multi:softprob" | "multi:softmax" => Task::Multiclass,
        o if IDENTITY.contains(&o) => Task::Regression,
        o => {
            let why = match o {
                "reg:logistic" => "its outputs pass through a sigmoid, and a Tabsmith regression output is linear",
                "binary:hinge" => "it outputs 0 or 1, not a probability",
                o if o.starts_with("count:") || o.starts_with("survival:") || o == "reg:gamma" || o == "reg:tweedie" => {
                    "its outputs are the exponential of the margin, and a Tabsmith regression output is linear"
                }
                o if o.starts_with("rank:") => "it ranks rather than predicts a label or a value",
                _ => "Tabsmith does not know how it turns a margin into a prediction",
            };
            return Err(unsupported(format!("objective `{o}`: {why}")).next(
                "train with binary:logistic, binary:logitraw, multi:softprob, multi:softmax or a regression with an identity link (reg:squarederror, reg:absoluteerror, reg:pseudohubererror, reg:quantileerror, reg:squaredlogerror)",
            ));
        }
    };
    let p = &l.learner_model_param;
    if count(&p.num_target, "num_target", 1)? > 1 {
        return Err(unsupported("a model of several targets (one tree output per target)"));
    }
    let n_features = count(&Some(p.num_feature.clone()), "num_feature", 0)?;
    if n_features == 0 {
        return Err(invalid("the model has no features"));
    }
    if n_features > MAX_FEATURES {
        return Err(unsupported(format!("more than {MAX_FEATURES} features")));
    }
    let n_outputs = match task {
        Task::Multiclass => count(&p.num_class, "num_class", 0)?,
        _ => 1,
    };
    if task == Task::Multiclass && n_outputs < 3 {
        return Err(unsupported(format!("a multiclass model of {n_outputs} classes")).next("train two classes with binary:logistic"));
    }
    if l.feature_types.iter().any(|t| t == "c") {
        return Err(unsupported("categorical features (feature type `c`)").next("one-hot encode them before training"));
    }
    match l.gradient_booster.name.as_str() {
        "gbtree" => {}
        "dart" => return Err(unsupported("the dart booster: it weighs each tree when it predicts").next("train with booster=gbtree")),
        "gblinear" => return Err(unsupported("the gblinear booster: a linear model, not trees")),
        other => return Err(unsupported(format!("booster `{other}`"))),
    }
    let raw = l.gradient_booster.model.ok_or_else(|| invalid("the booster has no model"))?;
    let gb: GbTree<'_> = serde_json::from_str(raw.get()).map_err(|e| invalid(format!("unreadable trees: {e}")))?;
    let parallel = gb
        .gbtree_model_param
        .as_ref()
        .map_or(Ok(1), |g| count(&g.num_parallel_tree, "num_parallel_tree", 1))?;
    if parallel != 1 {
        return Err(unsupported(format!(
            "{parallel} parallel trees per round (a boosted random forest)"
        )));
    }
    if gb.tree_info.len() != gb.trees.len() {
        return Err(invalid("tree_info does not list every tree"));
    }
    // Counts the file declares are backed by what it holds before anything is sized by them.
    if n_outputs > gb.trees.len() {
        return Err(unsupported(format!(
            "{n_outputs} classes for {} trees: each class needs at least one tree",
            gb.trees.len()
        )));
    }
    let names = if l.feature_names.is_empty() {
        (0..n_features).map(|i| format!("f{i}")).collect()
    } else if l.feature_names.len() == n_features {
        l.feature_names.clone()
    } else {
        return Err(invalid(format!(
            "the model names {} features but has {n_features}",
            l.feature_names.len()
        )));
    };

    // The starting margin XGBoost adds the trees to, in f32.
    let scores = f32s(&p.base_score, "base_score")?;
    let single = |what: &str| -> Result<f32> {
        match scores.as_slice() {
            [s] => Ok(*s),
            _ => Err(invalid(format!("{what} needs one base_score, the model has {}", scores.len()))),
        }
    };
    let base: Vec<f32> = match task {
        Task::Binary => {
            // ProbToMargin of the logistic objectives: -logf(1/p - 1), with glibc's logf.
            let prob = single("a binary model")?;
            if !(prob > 0.0 && prob < 1.0) {
                return Err(invalid(format!("base_score {prob} of a logistic model is not inside (0, 1)")));
            }
            vec![-logf(1.0f32 / prob - 1.0f32)]
        }
        Task::Multiclass if scores.len() == n_outputs => scores.clone(),
        Task::Multiclass => vec![single("a multiclass model")?; n_outputs],
        Task::Regression => vec![single("a regression model")?],
    };
    if base.iter().any(|b| !b.is_finite()) {
        return Err(invalid("the starting margin is not finite"));
    }

    let mut forest = Forest::new(Library::XGBoost, version, objective, task, names, n_outputs);
    forest.base = base.iter().map(|&b| f64::from(b)).collect();
    forest.sum = LeafSum::F32;
    forest.exact = true;
    let trees: Vec<(usize, (usize, Vec<Node>))> = gb
        .trees
        .iter()
        .zip(&gb.tree_info)
        .enumerate()
        .map(|(t, (tree, &output))| {
            Ok((
                usize::try_from(output).map_err(|_| invalid(format!("tree {t} adds to output {output}")))?,
                (t, nodes(tree, t, n_features)?),
            ))
        })
        .collect::<Result<_>>()?;
    for (t, nodes) in Forest::interleave(trees, n_outputs)? {
        forest.push_tree(&nodes, 0, t)?;
    }
    forest.notes = vec![
        "outputs are XGBoost's margins bit for bit: thresholds are the f32 just below XGBoost's (it sends x < s left, Tabsmith x <= t), leaves are its f32 values, added in f32 in its tree order".into(),
        "inputs are rounded to f32, as XGBoost rounds them; a missing value is null, NaN or a missing token (XGBoost's default missing value is NaN)".into(),
    ];
    if task == Task::Binary {
        forest.notes.push(format!(
            "the starting margin is -logf(1/p - 1) for base_score p = {}, with glibc's logf, as XGBoost computes it on Linux",
            scores[0]
        ));
    }
    Ok(forest)
}

fn flag(v: &serde_json::Value, t: usize) -> Result<bool> {
    match v {
        serde_json::Value::Bool(b) => Ok(*b),
        serde_json::Value::Number(n) => Ok(n.as_i64().is_some_and(|i| i != 0)),
        _ => Err(invalid(format!("tree {t} has a default_left that is neither 0/1 nor a boolean"))),
    }
}

fn nodes(tree: &Tree<'_>, t: usize, n_features: usize) -> Result<Vec<Node>> {
    let leaf_size = count(
        &tree.tree_param.as_ref().and_then(|p| p.size_leaf_vector.clone()),
        "size_leaf_vector",
        1,
    )?;
    if leaf_size > 1 {
        return Err(unsupported(format!(
            "tree {t} has leaves of {leaf_size} values (multi_strategy=multi_output_tree)"
        )));
    }
    let n = tree.left_children.len();
    let cond = f32s(tree.split_conditions.get(), "split_conditions")?;
    if [
        tree.right_children.len(),
        tree.split_indices.len(),
        cond.len(),
        tree.default_left.len(),
    ]
    .iter()
    .any(|&len| len != n)
        || n == 0
    {
        return Err(invalid(format!("tree {t}'s node arrays have different lengths")));
    }
    let child = |c: i64| usize::try_from(c).map_err(|_| invalid(format!("tree {t} has a child {c}")));
    (0..n)
        .map(|i| {
            if tree.left_children[i] == -1 {
                if tree.right_children[i] != -1 {
                    return Err(invalid(format!("tree {t} node {i} has one child")));
                }
                return Ok(Node::Leaf(cond[i]));
            }
            if tree.split_type.get(i).is_some_and(|&s| s != 0) {
                return Err(
                    unsupported(format!("tree {t} has a categorical split")).next("one-hot encode categorical features before training")
                );
            }
            let feature = usize::try_from(tree.split_indices[i])
                .ok()
                .filter(|&f| f < n_features)
                .ok_or_else(|| invalid(format!("tree {t} node {i} splits on feature {}", tree.split_indices[i])))?;
            let s = cond[i];
            if !s.is_finite() {
                return Err(invalid(format!("tree {t} node {i} has threshold {s}")));
            }
            Ok(Node::Split {
                feature,
                // x < s for an f32 x is x <= the f32 just below s.
                threshold: s.next_down(),
                missing_left: flag(&tree.default_left[i], t)?,
                left: child(tree.left_children[i])?,
                right: child(tree.right_children[i])?,
            })
        })
        .collect::<std::result::Result<Vec<Node>, TabsmithError>>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tabsmith_contracts::ErrorCode;

    /// A one-split regression model in the JSON layout of XGBoost 3.4.
    fn tiny(num_feature: &str, objective: &str, num_class: &str) -> String {
        format!(
            r#"{{"learner":{{"attributes":{{}},"feature_names":[],"feature_types":[],"gradient_booster":{{"model":{{"gbtree_model_param":{{"num_parallel_tree":"1","num_trees":"1"}},"iteration_indptr":[0,1],"tree_info":[0],"trees":[{{"base_weights":[0,0,0],"categories":[],"categories_nodes":[],"categories_segments":[],"categories_sizes":[],"default_left":[1,0,0],"id":0,"left_children":[1,-1,-1],"loss_changes":[0,0,0],"parents":[2147483647,0,0],"right_children":[2,-1,-1],"split_conditions":[5E-1,1E0,-1E0],"split_indices":[0,0,0],"split_type":[0,0,0],"sum_hessian":[0,0,0],"tree_param":{{"num_deleted":"0","num_feature":"2","num_nodes":"3","size_leaf_vector":"1"}}}}]}},"name":"gbtree"}},"learner_model_param":{{"base_score":"[5E-1]","boost_from_average":"1","num_class":"{num_class}","num_feature":"{num_feature}","num_target":"1"}},"objective":{{"name":"{objective}"}}}},"version":[3,4,1]}}"#
        )
    }

    #[test]
    fn a_declared_count_the_file_does_not_back_is_refused_before_anything_is_sized_by_it() {
        let f = parse(&tiny("2", "reg:squarederror", "0")).unwrap();
        assert_eq!((f.roots.len(), f.feature.len(), f.feature_names.len()), (1, 3, 2));
        assert_eq!(f.threshold[0], 0.5f32.next_down());
        // More features than any model has, and more classes than the one tree can serve.
        for (num_feature, objective, num_class) in [
            ("99999999999999", "reg:squarederror", "0"),
            ("1048577", "reg:squarederror", "0"),
            ("2", "multi:softprob", "4000000000000"),
            ("2", "multi:softprob", "4"),
        ] {
            let e = parse(&tiny(num_feature, objective, num_class)).unwrap_err();
            assert_eq!(e.code, ErrorCode::ImportUnsupported, "{num_feature} {objective} {num_class}: {e:?}");
        }
        // A count that is not a count.
        assert_eq!(
            parse(&tiny("two", "reg:squarederror", "0")).unwrap_err().code,
            ErrorCode::ImportInvalid
        );
    }

    #[test]
    fn numbers_are_read_straight_to_f32() {
        assert_eq!(f32s("[1.4375E0, -2.5E-1,3E0]", "x").unwrap(), vec![1.4375, -0.25, 3.0]);
        assert_eq!(f32s("[1.27E-1]", "x").unwrap(), vec![0.127]);
        assert_eq!(f32s("5E-1", "x").unwrap(), vec![0.5]);
        assert!(f32s("[]", "x").unwrap().is_empty());
        assert!(f32s("[1,banana]", "x").is_err());
        // A decimal halfway between two f32s, plus a hair: the nearest f32 directly. Read as
        // f64 first, the hair is lost and the tie goes to the even f32.
        let tie_plus = "1.000000059604644775390625000000001";
        assert_eq!(f32s(tie_plus, "x").unwrap(), vec![1.0f32.next_up()]);
        assert_eq!(tie_plus.parse::<f64>().unwrap() as f32, 1.0);
    }
}
