//! Minimal FP32 execution of `tabsmith.dense.v1` networks and `tabsmith.trees.v1`
//! tree ensembles, and their weight file codec.
//!
//! The runtime does not depend on the training engine; it executes the
//! engine-neutral architecture in `model.config.json` with weights from
//! `model.safetensors`. Agreement with the training engine's forward pass is
//! covered by tests in `tabsmith-workflow`. The arithmetic itself is
//! `tabsmith_embedded::kernels`, which models compiled into Rust code
//! (`tabsmith_macros::model!`) call too: both compute the same bits.

use std::collections::HashMap;

use safetensors::tensor::TensorView;
use safetensors::{Dtype, SafeTensors};
use tabsmith_contracts::model::{Activation, ModelConfig, TREES_FORMAT};
use tabsmith_contracts::{ErrorCode, Phase, Result, TabsmithError};
use tabsmith_embedded::kernels;
pub use tabsmith_embedded::kernels::sigmoid;

use crate::transform::Matrix;

/// Weights of one dense layer, `weight` row-major `[out_dim, in_dim]`.
#[derive(Debug, Clone, PartialEq)]
pub struct DenseLayer {
    /// Input width.
    pub in_dim: usize,
    /// Output width.
    pub out_dim: usize,
    /// `out_dim * in_dim` weights, row-major (one row per output).
    pub weight: Vec<f32>,
    /// `out_dim` biases.
    pub bias: Vec<f32>,
    /// Activation applied to this layer's output.
    pub activation: Activation,
}

/// A feed-forward network of [`DenseLayer`]s, validated against its [`ModelConfig`].
#[derive(Debug, Clone, PartialEq)]
pub struct DenseModel {
    /// The engine-neutral architecture (`model.config.json`).
    pub config: ModelConfig,
    /// The layers, input first.
    pub layers: Vec<DenseLayer>,
}

fn corrupt(msg: impl Into<String>) -> TabsmithError {
    TabsmithError::new(ErrorCode::BundleCorrupt, Phase::Load, msg)
        .field("model.safetensors")
        .next("rebuild the bundle with `tabsmith bundle build`, or restore it from a known-good copy")
}

impl DenseModel {
    /// Builds a model from per-layer weights, validating shapes against the config.
    pub fn new(config: ModelConfig, layers: Vec<DenseLayer>) -> Result<Self> {
        config.check_consistency().map_err(corrupt)?;
        if layers.len() != config.layers.len() {
            return Err(corrupt(format!(
                "config has {} layers, weights have {}",
                config.layers.len(),
                layers.len()
            )));
        }
        for (i, (spec, l)) in config.layers.iter().zip(&layers).enumerate() {
            if spec.in_dim != l.in_dim
                || spec.out_dim != l.out_dim
                || l.weight.len() != l.in_dim * l.out_dim
                || l.bias.len() != l.out_dim
                || spec.activation != l.activation
            {
                return Err(corrupt(format!("layer {i} weights do not match model.config.json")));
            }
            if l.weight.iter().chain(&l.bias).any(|v| !v.is_finite()) {
                return Err(corrupt(format!("layer {i} contains non-finite weights")));
            }
        }
        Ok(Self { config, layers })
    }

    /// Serializes weights as safetensors (F32, `[out, in]` weights).
    pub fn to_safetensors(&self) -> Result<Vec<u8>> {
        let mut buffers: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::new();
        for (i, l) in self.layers.iter().enumerate() {
            buffers.push((format!("layers.{i}.weight"), vec![l.out_dim, l.in_dim], f32_bytes(&l.weight)));
            buffers.push((format!("layers.{i}.bias"), vec![l.out_dim], f32_bytes(&l.bias)));
        }
        let mut views = Vec::new();
        for (name, shape, bytes) in &buffers {
            let view = TensorView::new(Dtype::F32, shape.clone(), bytes)
                .map_err(|e| TabsmithError::internal(Phase::Bundle, format!("tensor view: {e}")))?;
            views.push((name.clone(), view));
        }
        let mut meta = HashMap::new();
        meta.insert("format".to_string(), self.config.format.clone());
        safetensors::serialize(views, Some(meta)).map_err(|e| TabsmithError::internal(Phase::Bundle, format!("safetensors serialize: {e}")))
    }

    /// Loads weights for `config` from safetensors bytes.
    pub fn from_safetensors(config: ModelConfig, bytes: &[u8]) -> Result<Self> {
        config.check_consistency().map_err(corrupt)?;
        let st = SafeTensors::deserialize(bytes).map_err(|e| corrupt(format!("unreadable weights: {e}")))?;
        let mut layers = Vec::new();
        for (i, spec) in config.layers.iter().enumerate() {
            let weight = read_f32(&st, &format!("layers.{i}.weight"), &[spec.out_dim, spec.in_dim])?;
            let bias = read_f32(&st, &format!("layers.{i}.bias"), &[spec.out_dim])?;
            layers.push(DenseLayer {
                in_dim: spec.in_dim,
                out_dim: spec.out_dim,
                weight,
                bias,
                activation: spec.activation,
            });
        }
        let expected = 2 * config.layers.len();
        if st.names().len() != expected {
            return Err(corrupt(format!(
                "weights file has {} tensors, expected {expected}",
                st.names().len()
            )));
        }
        Self::new(config, layers)
    }

    /// The outputs of every row: `rows x n_outputs`.
    pub fn outputs(&self, x: &Matrix) -> Matrix {
        assert_eq!(x.cols, self.config.input_dim, "input width checked by the predictor");
        let k = self.config.n_outputs();
        let mut out = Vec::with_capacity(x.rows * k);
        let max_width = self.layers.iter().map(|l| l.out_dim.max(l.in_dim)).max().unwrap_or(1);
        let mut cur = vec![0f32; max_width];
        let mut next = vec![0f32; max_width];
        for r in 0..x.rows {
            let row = x.row(r);
            cur[..row.len()].copy_from_slice(row);
            for l in &self.layers {
                let relu = matches!(l.activation, Activation::Relu);
                kernels::dense(&l.weight, &l.bias, &cur[..l.in_dim], relu, &mut next[..l.out_dim]);
                std::mem::swap(&mut cur, &mut next);
            }
            out.extend_from_slice(&cur[..k]);
        }
        Matrix {
            rows: x.rows,
            cols: k,
            data: out,
        }
    }
}

/// A gradient-boosted tree ensemble ([`TREES_FORMAT`]); all trees' nodes in one
/// set of arrays. Node `i` is a leaf when `feature[i] < 0`; otherwise rows with
/// `x[feature[i]] <= threshold[i]` continue at `left[i]`, the others at `right[i]`.
#[derive(Debug, Clone, PartialEq)]
pub struct TreeEnsemble {
    /// The engine-neutral description (`model.config.json`).
    pub config: ModelConfig,
    /// Index of each tree's root node, ascending; a tree ends where the next begins.
    pub roots: Vec<u32>,
    /// Split feature of each node, or -1 for a leaf.
    pub feature: Vec<i32>,
    /// Split threshold of each node (unused for leaves).
    pub threshold: Vec<f32>,
    /// Left child of each split node.
    pub left: Vec<u32>,
    /// Right child of each split node.
    pub right: Vec<u32>,
    /// Value of each leaf, already scaled by the learning rate (0 for split nodes).
    pub value: Vec<f32>,
}

impl TreeEnsemble {
    /// Builds an ensemble, checking that every traversal ends inside its own tree:
    /// children come after their parent and before the next tree's root.
    pub fn new(
        config: ModelConfig,
        roots: Vec<u32>,
        feature: Vec<i32>,
        threshold: Vec<f32>,
        left: Vec<u32>,
        right: Vec<u32>,
        value: Vec<f32>,
    ) -> Result<Self> {
        config.check_consistency().map_err(corrupt)?;
        let spec = config
            .trees
            .as_ref()
            .ok_or_else(|| corrupt("tree model without its ensemble size"))?;
        let n = feature.len();
        if [threshold.len(), left.len(), right.len(), value.len()].iter().any(|&l| l != n) || n != spec.n_nodes {
            return Err(corrupt("tree arrays have different lengths"));
        }
        if roots.len() != spec.n_trees || roots.first() != Some(&0) || roots.windows(2).any(|w| w[0] >= w[1]) {
            return Err(corrupt("tree roots are not ascending from 0"));
        }
        for (t, &root) in roots.iter().enumerate() {
            let end = roots.get(t + 1).map_or(n, |&r| r as usize);
            if root as usize >= n || end > n {
                return Err(corrupt(format!("tree {t} lies beyond the last node")));
            }
            for i in root as usize..end {
                if feature[i] < 0 {
                    if !value[i].is_finite() {
                        return Err(corrupt(format!("tree {t} has a non-finite leaf value")));
                    }
                } else {
                    let (l, r) = (left[i] as usize, right[i] as usize);
                    if feature[i] as usize >= config.input_dim || !threshold[i].is_finite() || l <= i || r <= i || l >= end || r >= end {
                        return Err(corrupt(format!("tree {t} has an invalid split at node {i}")));
                    }
                }
            }
        }
        // The bundle claims monotone constraints: every tree must keep them. At each
        // split on a constrained feature, every leaf of the lesser side is at most
        // every leaf of the greater side; then the sum of the trees is monotone too.
        // Children come after their parent, so one pass backwards gives every
        // subtree's smallest and largest leaf.
        if !spec.monotone.is_empty() {
            let (mut lo, mut hi) = (vec![0f32; n], vec![0f32; n]);
            for i in (0..n).rev() {
                if feature[i] < 0 {
                    (lo[i], hi[i]) = (value[i], value[i]);
                    continue;
                }
                let (l, r) = (left[i] as usize, right[i] as usize);
                (lo[i], hi[i]) = (lo[l].min(lo[r]), hi[l].max(hi[r]));
                let broken = match spec.monotone[feature[i] as usize] {
                    1 => hi[l] > lo[r],
                    -1 => lo[l] < hi[r],
                    _ => false,
                };
                if broken {
                    return Err(corrupt(format!(
                        "the split at node {i} breaks the monotone constraint on feature {}",
                        feature[i]
                    )));
                }
            }
        }
        Ok(Self {
            config,
            roots,
            feature,
            threshold,
            left,
            right,
            value,
        })
    }

    /// The outputs of one feature row into `out` (one per output): the base score
    /// plus one leaf per tree of that output, summed in f64 in tree order (as the
    /// trainer does) and rounded to f32 at the end.
    pub fn outputs_row(&self, row: &[f32], out: &mut [f32]) {
        let spec = self.config.trees.as_ref().expect("checked when loaded");
        let base: &[f64] = if spec.n_outputs == 1 {
            std::slice::from_ref(&spec.base_score)
        } else {
            &spec.base_scores
        };
        self.kernel(base).outputs(row, out);
    }

    fn kernel<'a>(&'a self, base: &'a [f64]) -> kernels::Trees<'a> {
        kernels::Trees {
            roots: &self.roots,
            feature: &self.feature,
            threshold: &self.threshold,
            left: &self.left,
            right: &self.right,
            value: &self.value,
            base,
        }
    }

    fn to_safetensors(&self) -> Result<Vec<u8>> {
        let indices = |v: &[u32]| v.iter().map(|&x| x as i32).collect::<Vec<i32>>();
        serialize(
            &[
                ("trees.roots", Dtype::I32, i32_bytes(&indices(&self.roots))),
                ("trees.feature", Dtype::I32, i32_bytes(&self.feature)),
                ("trees.threshold", Dtype::F32, f32_bytes(&self.threshold)),
                ("trees.left", Dtype::I32, i32_bytes(&indices(&self.left))),
                ("trees.right", Dtype::I32, i32_bytes(&indices(&self.right))),
                ("trees.value", Dtype::F32, f32_bytes(&self.value)),
            ],
            &self.config.format,
        )
    }

    fn from_safetensors(config: ModelConfig, st: &SafeTensors<'_>) -> Result<Self> {
        let spec = config
            .trees
            .clone()
            .ok_or_else(|| corrupt("tree model without its ensemble size"))?;
        let (t, n) = (spec.n_trees, spec.n_nodes);
        if st.names().len() != 6 {
            return Err(corrupt(format!("weights file has {} tensors, expected 6", st.names().len())));
        }
        let index = |name: &str, len: usize| -> Result<Vec<u32>> {
            read_i32(st, name, &[len])?
                .into_iter()
                .map(|v| u32::try_from(v).map_err(|_| corrupt(format!("tensor `{name}` holds a negative index"))))
                .collect()
        };
        let roots = index("trees.roots", t)?;
        let feature = read_i32(st, "trees.feature", &[n])?;
        let threshold = read_f32(st, "trees.threshold", &[n])?;
        let left = index("trees.left", n)?;
        let right = index("trees.right", n)?;
        let value = read_f32(st, "trees.value", &[n])?;
        Self::new(config, roots, feature, threshold, left, right, value)
    }
}

/// The model of a bundle: a dense network or a tree ensemble.
#[derive(Debug, Clone, PartialEq)]
pub enum Model {
    /// `linear` and `small-mlp`.
    Dense(DenseModel),
    /// `gbdt`.
    Trees(TreeEnsemble),
}

impl Model {
    /// The engine-neutral description (`model.config.json`).
    pub fn config(&self) -> &ModelConfig {
        match self {
            Model::Dense(m) => &m.config,
            Model::Trees(m) => &m.config,
        }
    }

    /// The outputs of every row: `rows x n_outputs` (logits, or standardised values).
    pub fn outputs(&self, x: &Matrix) -> Matrix {
        match self {
            Model::Dense(m) => m.outputs(x),
            Model::Trees(m) => {
                assert_eq!(x.cols, m.config.input_dim, "input width checked by the predictor");
                let k = m.config.n_outputs();
                let mut data = vec![0f32; x.rows * k];
                for r in 0..x.rows {
                    m.outputs_row(x.row(r), &mut data[r * k..(r + 1) * k]);
                }
                Matrix {
                    rows: x.rows,
                    cols: k,
                    data,
                }
            }
        }
    }

    /// Serializes the weights (`model.safetensors`).
    pub fn to_safetensors(&self) -> Result<Vec<u8>> {
        match self {
            Model::Dense(m) => m.to_safetensors(),
            Model::Trees(m) => m.to_safetensors(),
        }
    }

    /// Loads the model `config` describes from safetensors bytes.
    pub fn from_safetensors(config: ModelConfig, bytes: &[u8]) -> Result<Self> {
        if config.format == TREES_FORMAT {
            config.check_consistency().map_err(corrupt)?;
            let st = SafeTensors::deserialize(bytes).map_err(|e| corrupt(format!("unreadable weights: {e}")))?;
            Ok(Model::Trees(TreeEnsemble::from_safetensors(config, &st)?))
        } else {
            Ok(Model::Dense(DenseModel::from_safetensors(config, bytes)?))
        }
    }
}

fn serialize(buffers: &[(&str, Dtype, Vec<u8>)], format: &str) -> Result<Vec<u8>> {
    let mut views = Vec::new();
    for (name, dtype, bytes) in buffers {
        let view = TensorView::new(*dtype, vec![bytes.len() / 4], bytes)
            .map_err(|e| TabsmithError::internal(Phase::Bundle, format!("tensor view: {e}")))?;
        views.push((name.to_string(), view));
    }
    let mut meta = HashMap::new();
    meta.insert("format".to_string(), format.to_string());
    safetensors::serialize(views, Some(meta)).map_err(|e| TabsmithError::internal(Phase::Bundle, format!("safetensors serialize: {e}")))
}

/// Dot product with eight independent accumulators (vectorisable). The summation
/// order is fixed, so results are deterministic for a given input.
/// Softmax of one row of logits, in f64 (shifted by the maximum for stability).
pub fn softmax(z: &[f32]) -> Vec<f64> {
    let mut p = vec![0.0; z.len()];
    kernels::softmax(z, &mut p);
    p
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn i32_bytes(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn read_i32(st: &SafeTensors<'_>, name: &str, shape: &[usize]) -> Result<Vec<i32>> {
    let t = st.tensor(name).map_err(|_| corrupt(format!("tensor `{name}` is missing")))?;
    if t.dtype() != Dtype::I32 {
        return Err(corrupt(format!("tensor `{name}` has dtype {:?}, expected I32", t.dtype())));
    }
    if t.shape() != shape {
        return Err(corrupt(format!("tensor `{name}` has shape {:?}, expected {shape:?}", t.shape())));
    }
    Ok(t.data().as_chunks::<4>().0.iter().map(|c| i32::from_le_bytes(*c)).collect())
}

fn read_f32(st: &SafeTensors<'_>, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
    let t = st.tensor(name).map_err(|_| corrupt(format!("tensor `{name}` is missing")))?;
    if t.dtype() != Dtype::F32 {
        return Err(corrupt(format!("tensor `{name}` has dtype {:?}, expected F32", t.dtype())));
    }
    if t.shape() != shape {
        return Err(corrupt(format!("tensor `{name}` has shape {:?}, expected {shape:?}", t.shape())));
    }
    Ok(t.data().as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny() -> DenseModel {
        let config = ModelConfig::dense("small-mlp", 2, &[2]);
        DenseModel::new(
            config,
            vec![
                DenseLayer {
                    in_dim: 2,
                    out_dim: 2,
                    weight: vec![1.0, -1.0, 0.5, 0.5],
                    bias: vec![0.0, -1.0],
                    activation: Activation::Relu,
                },
                DenseLayer {
                    in_dim: 2,
                    out_dim: 1,
                    weight: vec![2.0, 3.0],
                    bias: vec![0.25],
                    activation: Activation::Identity,
                },
            ],
        )
        .unwrap()
    }

    #[test]
    fn forward_matches_hand_computation() {
        let m = tiny();
        let x = Matrix {
            rows: 2,
            cols: 2,
            data: vec![3.0, 1.0, 0.0, 4.0],
        };
        // row0: h = relu([3-1, 1.5+0.5-1]) = [2, 1]; y = 4 + 3 + 0.25
        // row1: h = relu([-4, 2-1]) = [0, 1]; y = 3 + 0.25
        assert_eq!(m.outputs(&x).data, vec![7.25, 3.25]);
    }

    #[test]
    fn safetensors_roundtrip_and_shape_checks() {
        let m = tiny();
        let bytes = m.to_safetensors().unwrap();
        let back = DenseModel::from_safetensors(m.config.clone(), &bytes).unwrap();
        assert_eq!(back, m);
        let wrong = ModelConfig::dense("small-mlp", 3, &[2]);
        let e = DenseModel::from_safetensors(wrong, &bytes).unwrap_err();
        assert_eq!(e.code, ErrorCode::BundleCorrupt);
    }

    #[test]
    fn sigmoid_is_stable() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-15);
        assert!(sigmoid(-1000.0) >= 0.0 && sigmoid(1000.0) <= 1.0);
    }
}
