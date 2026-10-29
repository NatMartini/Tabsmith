//! Engine-neutral description of the model architecture stored in a Bundle.

use serde::{Deserialize, Serialize};

use crate::config::Task;

/// Weight file format understood by `tabsmith-runtime`.
///
/// Tensors are stored in `model.safetensors` as F32, little endian:
/// `layers.{i}.weight` with shape `[out_dim, in_dim]` (PyTorch convention) and
/// `layers.{i}.bias` with shape `[out_dim]`.
pub const MODEL_FORMAT: &str = "tabsmith.dense.v1";

/// Tree-ensemble format understood by `tabsmith-runtime` (the `gbdt` candidate).
///
/// All trees' nodes are concatenated in `model.safetensors`: `trees.roots` (I32,
/// one root index per tree), `trees.feature` (I32; -1 marks a leaf),
/// `trees.threshold` (F32; rows with `x[feature] <= threshold` go left),
/// `trees.left` / `trees.right` (I32 child indices, always greater than the
/// parent's and inside the same tree) and `trees.value` (F32 leaf values, already
/// scaled by the learning rate). With one output, output = base_score + the sum
/// of one leaf per tree; with K outputs (multiclass), tree `t` adds to output
/// `t % K`, starting from `base_scores[k]`.
pub const TREES_FORMAT: &str = "tabsmith.trees.v1";

/// The output kind of each task: one logit (sigmoid), K logits (softmax), one value.
pub fn output_kind(task: Task) -> &'static str {
    match task {
        Task::Binary => "logit",
        Task::Multiclass => "logits",
        Task::Regression => "value",
    }
}

fn one() -> usize {
    1
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Activation {
    Identity,
    Relu,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LayerSpec {
    pub in_dim: usize,
    pub out_dim: usize,
    pub activation: Activation,
}

/// Size of a tree ensemble.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TreesSpec {
    pub n_trees: usize,
    /// Nodes of all trees together.
    pub n_nodes: usize,
    pub max_depth: usize,
    /// Output before the first tree: the class-weighted log-odds of the training
    /// labels (binary), 0 (regression; targets are standardised).
    pub base_score: f64,
    /// Outputs per row (K for multiclass, otherwise 1).
    #[serde(default = "one")]
    pub n_outputs: usize,
    /// Multiclass: the starting output of each class (log of its weighted share).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub base_scores: Vec<f64>,
    /// Monotone constraints the trees were trained under, one entry per feature:
    /// `1` (the output never decreases in the feature), `-1` (never increases) or `0`.
    /// Empty when there are none. A loader checks that every tree keeps them.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub monotone: Vec<i8>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelConfig {
    /// [`MODEL_FORMAT`] (dense networks) or [`TREES_FORMAT`] (tree ensembles).
    pub format: String,
    /// `linear`, `small-mlp` or `gbdt`.
    pub family: String,
    pub input_dim: usize,
    /// Dense layers, input first (dense format only).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub layers: Vec<LayerSpec>,
    /// Ensemble size (tree format only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trees: Option<TreesSpec>,
    /// The final layer outputs a single logit; probability = sigmoid(logit).
    pub output: String,
    pub precision: String,
}

impl ModelConfig {
    /// A binary network (one logit).
    pub fn dense(family: &str, input_dim: usize, hidden: &[usize]) -> Self {
        Self::dense_for(family, input_dim, hidden, Task::Binary, 1)
    }

    /// A network for `task` with `n_outputs` outputs (the number of classes for multiclass).
    pub fn dense_for(family: &str, input_dim: usize, hidden: &[usize], task: Task, n_outputs: usize) -> Self {
        let mut layers = Vec::new();
        let mut d = input_dim;
        for &h in hidden {
            layers.push(LayerSpec {
                in_dim: d,
                out_dim: h,
                activation: Activation::Relu,
            });
            d = h;
        }
        layers.push(LayerSpec {
            in_dim: d,
            out_dim: n_outputs,
            activation: Activation::Identity,
        });
        Self {
            format: MODEL_FORMAT.to_string(),
            family: family.to_string(),
            input_dim,
            layers,
            trees: None,
            output: output_kind(task).to_string(),
            precision: "fp32".to_string(),
        }
    }

    /// A binary tree ensemble; `trees` is `None` until training has grown it.
    pub fn trees(input_dim: usize, trees: Option<TreesSpec>) -> Self {
        Self::trees_for(input_dim, Task::Binary, trees)
    }

    /// A tree ensemble for `task`.
    pub fn trees_for(input_dim: usize, task: Task, trees: Option<TreesSpec>) -> Self {
        Self {
            format: TREES_FORMAT.to_string(),
            family: "gbdt".to_string(),
            input_dim,
            layers: Vec::new(),
            trees,
            output: output_kind(task).to_string(),
            precision: "fp32".to_string(),
        }
    }

    /// Outputs per row: 1, or K logits for multiclass.
    pub fn n_outputs(&self) -> usize {
        match &self.trees {
            Some(t) => t.n_outputs,
            None => self.layers.last().map_or(1, |l| l.out_dim),
        }
    }

    /// Weights and biases of a network, or nodes of a tree ensemble.
    pub fn parameter_count(&self) -> usize {
        match &self.trees {
            Some(t) => t.n_nodes,
            None => self.layers.iter().map(|l| l.in_dim * l.out_dim + l.out_dim).sum(),
        }
    }

    pub fn check_consistency(&self) -> Result<(), String> {
        let single = match self.output.as_str() {
            "logit" | "value" => true,
            "logits" => false,
            other => return Err(format!("unknown model output `{other}`")),
        };
        if self.precision != "fp32" {
            return Err(format!("unsupported precision `{}`", self.precision));
        }
        if self.format == TREES_FORMAT {
            let Some(t) = &self.trees else {
                return Err("tree model without its ensemble size".into());
            };
            if !self.layers.is_empty() || self.input_dim == 0 || t.n_trees == 0 || t.n_nodes < t.n_trees || !t.base_score.is_finite() {
                return Err("tree model sizes are inconsistent".into());
            }
            let outputs_ok = if single {
                t.n_outputs == 1 && t.base_scores.is_empty()
            } else {
                t.n_outputs >= 3 && t.base_scores.len() == t.n_outputs && t.base_scores.iter().all(|b| b.is_finite())
            };
            if !outputs_ok || t.n_trees % t.n_outputs != 0 {
                return Err("tree model outputs are inconsistent".into());
            }
            if !t.monotone.is_empty() && (t.monotone.len() != self.input_dim || t.monotone.iter().any(|&m| !(-1..=1).contains(&m))) {
                return Err("tree model monotone constraints do not match its features".into());
            }
            return Ok(());
        }
        if self.format != MODEL_FORMAT {
            return Err(format!("unsupported model format `{}`", self.format));
        }
        if self.trees.is_some() {
            return Err("a dense model has no trees".into());
        }
        let Some(first) = self.layers.first() else {
            return Err("model has no layers".into());
        };
        if first.in_dim != self.input_dim {
            return Err("first layer input does not match input_dim".into());
        }
        for w in self.layers.windows(2) {
            if w[0].out_dim != w[1].in_dim {
                return Err("layer dimensions are not chained".into());
            }
        }
        let last = self.layers.last().expect("non-empty");
        if last.activation != Activation::Identity || (single && last.out_dim != 1) || (!single && last.out_dim < 3) {
            return Err(format!("last layer does not match the `{}` output", self.output));
        }
        Ok(())
    }
}
