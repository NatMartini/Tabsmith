//! Minimal FP32 execution of `rustpy.dense.v1` models and the weight file codec.
//!
//! The runtime does not depend on the training engine; it executes the
//! engine-neutral architecture in `model.config.json` with weights from
//! `model.safetensors`. Agreement with the training engine's forward pass is
//! covered by tests in `rustpy-workflow`.

use std::collections::HashMap;

use rustpy_contracts::model::{Activation, ModelConfig};
use rustpy_contracts::{ErrorCode, Phase, Result, RustpyError};
use safetensors::tensor::TensorView;
use safetensors::{Dtype, SafeTensors};

use crate::transform::Matrix;

/// Weights of one dense layer, `weight` row-major `[out_dim, in_dim]`.
#[derive(Debug, Clone, PartialEq)]
pub struct DenseLayer {
    pub in_dim: usize,
    pub out_dim: usize,
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
    pub activation: Activation,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DenseModel {
    pub config: ModelConfig,
    pub layers: Vec<DenseLayer>,
}

fn corrupt(msg: impl Into<String>) -> RustpyError {
    RustpyError::new(ErrorCode::BundleCorrupt, Phase::Load, msg)
        .field("model.safetensors")
        .next("rebuild the bundle with `rustpy bundle build`, or restore it from a known-good copy")
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
                .map_err(|e| RustpyError::internal(Phase::Bundle, format!("tensor view: {e}")))?;
            views.push((name.clone(), view));
        }
        let mut meta = HashMap::new();
        meta.insert("format".to_string(), self.config.format.clone());
        safetensors::serialize(views, Some(meta)).map_err(|e| RustpyError::internal(Phase::Bundle, format!("safetensors serialize: {e}")))
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

    /// One logit per row.
    pub fn logits(&self, x: &Matrix) -> Vec<f32> {
        assert_eq!(x.cols, self.config.input_dim, "input width checked by the predictor");
        let mut out = Vec::with_capacity(x.rows);
        let max_width = self.layers.iter().map(|l| l.out_dim.max(l.in_dim)).max().unwrap_or(1);
        let mut cur = vec![0f32; max_width];
        let mut next = vec![0f32; max_width];
        for r in 0..x.rows {
            let row = x.row(r);
            cur[..row.len()].copy_from_slice(row);
            for l in &self.layers {
                for o in 0..l.out_dim {
                    let w = &l.weight[o * l.in_dim..(o + 1) * l.in_dim];
                    let v = dot(w, &cur[..l.in_dim]) + l.bias[o];
                    next[o] = match l.activation {
                        Activation::Identity => v,
                        Activation::Relu => v.max(0.0),
                    };
                }
                std::mem::swap(&mut cur, &mut next);
            }
            out.push(cur[0]);
        }
        out
    }
}

/// Dot product with eight independent accumulators (vectorisable). The summation
/// order is fixed, so results are deterministic for a given input.
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let chunks = a.len() / 8;
    for c in 0..chunks {
        let (x, y) = (&a[c * 8..c * 8 + 8], &b[c * 8..c * 8 + 8]);
        for k in 0..8 {
            acc[k] += x[k] * y[k];
        }
    }
    let mut tail = 0f32;
    for i in chunks * 8..a.len() {
        tail += a[i] * b[i];
    }
    ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7])) + tail
}

/// Numerically stable logistic function.
pub fn sigmoid(z: f64) -> f64 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let e = z.exp();
        e / (1.0 + e)
    }
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn read_f32(st: &SafeTensors<'_>, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
    let t = st.tensor(name).map_err(|_| corrupt(format!("tensor `{name}` is missing")))?;
    if t.dtype() != Dtype::F32 {
        return Err(corrupt(format!("tensor `{name}` has dtype {:?}, expected F32", t.dtype())));
    }
    if t.shape() != shape {
        return Err(corrupt(format!("tensor `{name}` has shape {:?}, expected {shape:?}", t.shape())));
    }
    Ok(t.data()
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
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
        assert_eq!(m.logits(&x), vec![7.25, 3.25]);
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
