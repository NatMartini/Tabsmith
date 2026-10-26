//! Engine-neutral description of the model architecture stored in a Bundle.

use serde::{Deserialize, Serialize};

/// Weight file format understood by `rustpy-runtime`.
///
/// Tensors are stored in `model.safetensors` as F32, little endian:
/// `layers.{i}.weight` with shape `[out_dim, in_dim]` (PyTorch convention) and
/// `layers.{i}.bias` with shape `[out_dim]`.
pub const MODEL_FORMAT: &str = "rustpy.dense.v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Activation {
    Identity,
    Relu,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LayerSpec {
    pub in_dim: usize,
    pub out_dim: usize,
    pub activation: Activation,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelConfig {
    pub format: String,
    /// `linear` or `small-mlp`.
    pub family: String,
    pub input_dim: usize,
    pub layers: Vec<LayerSpec>,
    /// The final layer outputs a single logit; probability = sigmoid(logit).
    pub output: String,
    pub precision: String,
}

impl ModelConfig {
    pub fn dense(family: &str, input_dim: usize, hidden: &[usize]) -> Self {
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
            out_dim: 1,
            activation: Activation::Identity,
        });
        Self {
            format: MODEL_FORMAT.to_string(),
            family: family.to_string(),
            input_dim,
            layers,
            output: "logit".to_string(),
            precision: "fp32".to_string(),
        }
    }

    pub fn parameter_count(&self) -> usize {
        self.layers.iter().map(|l| l.in_dim * l.out_dim + l.out_dim).sum()
    }

    pub fn check_consistency(&self) -> Result<(), String> {
        if self.format != MODEL_FORMAT {
            return Err(format!("unsupported model format `{}`", self.format));
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
        if last.out_dim != 1 || last.activation != Activation::Identity || self.output != "logit" {
            return Err("last layer must output a single logit".into());
        }
        if self.precision != "fp32" {
            return Err(format!("unsupported precision `{}`", self.precision));
        }
        Ok(())
    }
}
