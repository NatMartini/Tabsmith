//! Training of the `linear` and `small-mlp` candidates with Burn (CPU: Flex backend).
//!
//! Design for exact resume (R2):
//! * weights are initialised from Tabsmith's own versioned PRNG, not the engine's;
//! * the minibatch order is a pure function of (seed, epoch);
//! * no stochastic layers (no dropout);
//! * a checkpoint stores model record + optimizer record (Adam moments, step
//!   counts) + the host-side loop state (epoch, batch cursor, partial epoch loss,
//!   early-stopping state, best weights, history).
//!
//! Continuing from a checkpoint therefore reproduces the uninterrupted run bit for
//! bit on the same binary and CPU feature set (verified by tests).

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use burn::module::{Module, Param};
use burn::nn::{Initializer, Linear, LinearConfig, Relu};
use burn::optim::{AdamConfig, GradientsParams, ModuleOptimizer};
use burn::store::ModuleRecord;
use burn::tensor::activation::{log_sigmoid, log_softmax};
use burn::tensor::{Bytes, Device, Tensor, TensorData};
use serde::{Deserialize, Serialize};
use tabsmith_contracts::config::{Candidate, CandidateTrainConfig, ClassWeight, SelectionMetric, Task};
use tabsmith_contracts::hash::{canonical_hash, sha256_file};
use tabsmith_contracts::model::ModelConfig;
use tabsmith_contracts::run::EpochRecord;
use tabsmith_contracts::schema::OutputSchema;
use tabsmith_contracts::{ErrorCode, Phase, Result, TabsmithError};
use tabsmith_runtime::Matrix;
use tabsmith_runtime::model::{DenseLayer, DenseModel, Model};

use crate::data::Targets;
use crate::device::EngineDevice;
use crate::metrics::task_metrics;
use crate::project::{write_atomic, write_json};
use crate::rng::{SplitMix64, epoch_permutation};

/// Fixed training choices of tabular-binary@0.1 that are not configurable; recorded
/// in every Run and in the exact-resume key so changes are visible and refused.
pub const RECIPE_CONSTANTS: &str = "optimizer=adam(beta1=0.9,beta2=0.999,eps=1e-8); init=tabsmith.init.v1(linear:zeros; mlp:uniform(+-1/sqrt(fan_in)) from SplitMix64(seed,\"init\")); loss=class-weighted mean BCE-with-logits + l2/2*sum(W^2) (biases excluded); batches=tabsmith.rng.v1 epoch permutation";

/// Engine-side network: dense layers with ReLU between them.
#[derive(Module, Debug)]
pub struct DenseNet {
    layers: Vec<Linear>,
    act: Relu,
}

fn train_err(msg: impl Into<String>) -> TabsmithError {
    TabsmithError::internal(Phase::Train, msg)
}

impl DenseNet {
    /// Initialises weights from Tabsmith's PRNG: zeros for the linear baseline,
    /// Kaiming-uniform U(-1/sqrt(fan_in), 1/sqrt(fan_in)) for MLP weights and biases.
    pub fn init(config: &ModelConfig, seed: u64, device: &Device) -> Self {
        let zero = config.layers.len() == 1;
        let mut rng = SplitMix64::derive(seed, "init");
        let layers = config
            .layers
            .iter()
            .map(|spec| {
                let bound = 1.0 / (spec.in_dim as f64).sqrt();
                let mut draw = |n: usize| -> Vec<f32> {
                    (0..n)
                        .map(|_| if zero { 0.0 } else { ((rng.next_f64() * 2.0 - 1.0) * bound) as f32 })
                        .collect()
                };
                // Burn stores Linear weights as [d_in, d_out].
                let w = draw(spec.in_dim * spec.out_dim);
                let b = draw(spec.out_dim);
                make_linear(spec.in_dim, spec.out_dim, w, b, device)
            })
            .collect();
        Self { layers, act: Relu::new() }
    }

    /// Initialises from exported weights (warm start).
    pub fn from_dense(model: &DenseModel, device: &Device) -> Self {
        let layers = model
            .layers
            .iter()
            .map(|l| {
                let mut w = vec![0f32; l.in_dim * l.out_dim];
                for o in 0..l.out_dim {
                    for i in 0..l.in_dim {
                        w[i * l.out_dim + o] = l.weight[o * l.in_dim + i];
                    }
                }
                make_linear(l.in_dim, l.out_dim, w, l.bias.clone(), device)
            })
            .collect();
        Self { layers, act: Relu::new() }
    }

    pub fn forward(&self, mut x: Tensor<2>) -> Tensor<2> {
        let last = self.layers.len() - 1;
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(x);
            if i < last {
                x = self.act.forward(x);
            }
        }
        x
    }

    /// Sum of squared weights (biases excluded).
    pub fn weight_sq_sum(&self) -> Tensor<1> {
        let mut total: Option<Tensor<1>> = None;
        for l in &self.layers {
            let s = l.weight.val().powf_scalar(2.0).sum();
            total = Some(match total {
                Some(t) => t + s,
                None => s,
            });
        }
        total.expect("at least one layer")
    }

    /// Exports weights in the runtime format (`[out, in]`).
    pub fn export(&self, config: &ModelConfig) -> Result<DenseModel> {
        let mut layers = Vec::new();
        for (spec, l) in config.layers.iter().zip(&self.layers) {
            let w: Vec<f32> = l
                .weight
                .val()
                .into_data()
                .try_to_vec::<f32>()
                .map_err(|e| train_err(format!("weight export: {e:?}")))?;
            let b: Vec<f32> = match &l.bias {
                Some(b) => b
                    .val()
                    .into_data()
                    .try_to_vec::<f32>()
                    .map_err(|e| train_err(format!("bias export: {e:?}")))?,
                None => vec![0.0; spec.out_dim],
            };
            let mut wt = vec![0f32; w.len()];
            for i in 0..spec.in_dim {
                for o in 0..spec.out_dim {
                    wt[o * spec.in_dim + i] = w[i * spec.out_dim + o];
                }
            }
            layers.push(DenseLayer {
                in_dim: spec.in_dim,
                out_dim: spec.out_dim,
                weight: wt,
                bias: b,
                activation: spec.activation,
            });
        }
        DenseModel::new(config.clone(), layers).map_err(|e| e.in_phase(Phase::Train))
    }
}

fn make_linear(in_dim: usize, out_dim: usize, w: Vec<f32>, b: Vec<f32>, device: &Device) -> Linear {
    let mut lin = LinearConfig::new(in_dim, out_dim).with_initializer(Initializer::Zeros).init(device);
    lin.weight = Param::from_tensor(Tensor::<2>::from_data(TensorData::new(w, [in_dim, out_dim]), device));
    lin.bias = Some(Param::from_tensor(Tensor::<1>::from_data(TensorData::new(b, [out_dim]), device)));
    lin
}

/// A binary model of `candidate`.
pub fn model_config(candidate: Candidate, input_dim: usize, cfg: &CandidateTrainConfig) -> ModelConfig {
    model_config_for(candidate, input_dim, cfg, Task::Binary, 1)
}

/// The model of `candidate` for `task`, with `n_outputs` outputs.
pub fn model_config_for(candidate: Candidate, input_dim: usize, cfg: &CandidateTrainConfig, task: Task, n_outputs: usize) -> ModelConfig {
    match candidate {
        Candidate::Linear => ModelConfig::dense_for("linear", input_dim, &[], task, n_outputs),
        Candidate::SmallMlp => ModelConfig::dense_for("small-mlp", input_dim, &cfg.hidden, task, n_outputs),
        Candidate::Gbdt => ModelConfig::trees_for(input_dim, task, None),
    }
}

/// Inputs that must match exactly for an R2 resume.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResumeKey {
    pub snapshot_id: String,
    pub split_id: String,
    pub transform_id: String,
    pub candidate: Candidate,
    pub train_config: CandidateTrainConfig,
    pub seed: u64,
    pub threshold: f64,
    pub model_config: ModelConfig,
    pub engine: String,
    pub tabsmith_version: String,
    pub n_train: usize,
    /// OS / architecture / SIMD features: bit-identical continuation is only claimed
    /// on the same platform (the matrix kernels are chosen by CPU features).
    #[serde(default)]
    pub platform: String,
    #[serde(default)]
    pub recipe_constants: String,
}

/// Platform identity relevant to floating-point results.
pub fn platform_fingerprint() -> String {
    #[allow(unused_mut)]
    let mut features: Vec<&str> = Vec::new();
    #[cfg(target_arch = "x86_64")]
    {
        for (name, on) in [
            ("sse2", std::arch::is_x86_feature_detected!("sse2")),
            ("avx", std::arch::is_x86_feature_detected!("avx")),
            ("avx2", std::arch::is_x86_feature_detected!("avx2")),
            ("fma", std::arch::is_x86_feature_detected!("fma")),
            ("avx512f", std::arch::is_x86_feature_detected!("avx512f")),
        ] {
            if on {
                features.push(name);
            }
        }
    }
    format!("{}/{} [{}]", std::env::consts::OS, std::env::consts::ARCH, features.join(","))
}

/// Platform identity of a training device: the CPU platform, plus the GPU model
/// and driver (or compute capability) when training runs on a GPU.
pub fn platform_for(device: &EngineDevice) -> String {
    match device.identity() {
        Some(gpu) => format!("{} + {gpu}", platform_fingerprint()),
        None => platform_fingerprint(),
    }
}

impl ResumeKey {
    /// Field-by-field differences, for an actionable refusal.
    pub fn differences(&self, other: &ResumeKey) -> Vec<String> {
        let mut d = Vec::new();
        macro_rules! cmp {
            ($f:ident, $label:expr) => {
                if self.$f != other.$f {
                    d.push(format!("{} ({:?} -> {:?})", $label, self.$f, other.$f));
                }
            };
        }
        cmp!(snapshot_id, "data snapshot");
        cmp!(split_id, "split");
        cmp!(transform_id, "fitted transform");
        cmp!(candidate, "candidate");
        cmp!(seed, "train seed");
        cmp!(engine, "training engine");
        cmp!(tabsmith_version, "tabsmith version");
        cmp!(n_train, "training rows");
        cmp!(platform, "platform (OS/CPU features)");
        cmp!(recipe_constants, "recipe constants (optimizer/initialisation/loss)");
        if self.train_config != other.train_config {
            d.push("training configuration".into());
        }
        if self.model_config != other.model_config {
            d.push("model architecture".into());
        }
        if self.threshold.to_bits() != other.threshold.to_bits() {
            d.push("decision threshold".into());
        }
        d
    }
}

/// Host-side loop state saved with each checkpoint.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrainState {
    pub epoch: usize,
    pub next_batch: usize,
    pub step: u64,
    pub epoch_loss_sum: f64,
    pub epoch_batches: usize,
    pub best_valid_loss: Option<f64>,
    pub best_epoch: usize,
    pub bad_epochs: usize,
    pub history: Vec<EpochRecord>,
    pub stop_reason: Option<String>,
    pub elapsed_ms: f64,
}

impl TrainState {
    pub(crate) fn fresh() -> Self {
        Self {
            epoch: 0,
            next_batch: 0,
            step: 0,
            epoch_loss_sum: 0.0,
            epoch_batches: 0,
            best_valid_loss: None,
            best_epoch: 0,
            bad_epochs: 0,
            history: Vec::new(),
            stop_reason: None,
            elapsed_ms: 0.0,
        }
    }
}

/// Cancellation and test hooks.
#[derive(Debug, Clone, Default)]
pub struct RunControl {
    /// Set (e.g. by Ctrl-C) to stop at the next optimizer-step boundary with a checkpoint.
    pub cancel: Arc<AtomicBool>,
    /// Test hook: abort the process right after this global step, without cleanup,
    /// to simulate a crash (`tabsmith run --simulate-crash-after-steps N`, hidden).
    pub crash_after_steps: Option<u64>,
    /// Test hook: request cancellation once this global step is reached (a clean
    /// stop with a mid-epoch checkpoint, like Ctrl-C at a deterministic point).
    pub cancel_after_steps: Option<u64>,
}

pub struct TrainInputs<'a> {
    pub candidate: Candidate,
    pub cfg: &'a CandidateTrainConfig,
    pub seed: u64,
    pub metric: SelectionMetric,
    pub max_train_seconds: u64,
    /// How outputs are read: task, classes or target scale, binary threshold.
    pub output: &'a OutputSchema,
    pub x_train: &'a Matrix,
    pub y_train: &'a Targets,
    pub x_valid: &'a Matrix,
    pub y_valid: &'a Targets,
    pub checkpoint_dir: PathBuf,
    pub key: ResumeKey,
    /// Where to train, as planned by [`crate::device::plan_device`].
    pub device: EngineDevice,
    /// Monotone constraints of the `gbdt` candidate, one sign per feature
    /// ([`crate::gbdt::monotone_constraints`]); empty for the networks or without any.
    pub monotone: Vec<i8>,
}

// Returned once per run and immediately taken apart; boxing the model buys nothing.
#[allow(clippy::large_enum_variant)]
pub enum TrainOutcome {
    Completed {
        best: Model,
        state: TrainState,
        parity_max_abs_diff: f64,
    },
    Cancelled {
        state: TrainState,
        checkpoint: PathBuf,
    },
}

/// Where training starts from.
// Built once per run and consumed at once; boxing the warm-start weights buys nothing.
#[allow(clippy::large_enum_variant)]
pub enum StartFrom {
    Scratch,
    /// R2: continue exactly from a checkpoint.
    Checkpoint(PathBuf),
    /// Fresh optimizer, weights copied from a previous model.
    WarmStart(DenseModel),
}

/// Loss weight of each class from the training targets: `[negative, positive]`
/// (binary), one per class (multiclass), `[1]` (regression). Balanced weights are
/// n / (classes * rows of the class), so every class weighs the same in total.
pub fn class_weights_of(train: &Targets, mode: ClassWeight) -> Vec<f32> {
    match (train, mode) {
        (Targets::Binary(y), _) => {
            let (w_neg, w_pos) = class_weights(y, mode);
            vec![w_neg, w_pos]
        }
        (Targets::Classes { y, k }, ClassWeight::Balanced) => {
            let mut counts = vec![0usize; *k];
            for &c in y {
                counts[c as usize] += 1;
            }
            let n = y.len() as f64;
            counts.iter().map(|&c| (n / (*k as f64 * c.max(1) as f64)) as f32).collect()
        }
        (Targets::Classes { k, .. }, ClassWeight::None) => vec![1.0; *k],
        (Targets::Values(_), _) => vec![1.0],
    }
}

/// What the network fits for each row: 0/1 (binary), the class index
/// (multiclass) or the target standardised with the output schema's scale
/// (regression); and the row's loss weight.
pub fn encode_targets(t: &Targets, output: &OutputSchema, weights: &[f32]) -> (Vec<f32>, Vec<f32>) {
    match t {
        Targets::Binary(y) => (
            y.iter().map(|&v| if v { 1.0 } else { 0.0 }).collect(),
            y.iter().map(|&v| weights[usize::from(v)]).collect(),
        ),
        Targets::Classes { y, .. } => (
            y.iter().map(|&c| c as f32).collect(),
            y.iter().map(|&c| weights[c as usize]).collect(),
        ),
        Targets::Values(y) => {
            let s = output.target_scale.expect("regression schema");
            (y.iter().map(|&v| ((v - s.mean) / s.std) as f32).collect(), vec![1.0; y.len()])
        }
    }
}

/// The weighted training loss of model outputs (without the L2 term), in f64:
/// log loss (binary), softmax cross-entropy (multiclass), squared error of the
/// standardised target (regression).
pub fn task_loss(outputs: &Matrix, t: &Targets, output: &OutputSchema, weights: &[f32]) -> f64 {
    match t {
        Targets::Binary(y) => weighted_bce_from_logits(&outputs.data, y, weights[0], weights[1]),
        Targets::Classes { y, .. } => {
            let mut s = 0.0;
            for (r, &c) in y.iter().enumerate() {
                let z: Vec<f64> = outputs.row(r).iter().map(|&v| f64::from(v)).collect();
                let max = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let lse = max + libm::log(z.iter().map(|v| libm::exp(v - max)).sum::<f64>());
                s += (lse - z[c as usize]) * f64::from(weights[c as usize]);
            }
            s / y.len().max(1) as f64
        }
        Targets::Values(y) => {
            let sc = output.target_scale.expect("regression schema");
            let s: f64 = y
                .iter()
                .zip(&outputs.data)
                .map(|(&v, &z)| {
                    let d = f64::from(z) - (v - sc.mean) / sc.std;
                    d * d
                })
                .sum();
            s / y.len().max(1) as f64
        }
    }
}

/// Per-sample weights for the training loss.
pub(crate) fn class_weights(y: &[bool], mode: ClassWeight) -> (f32, f32) {
    match mode {
        ClassWeight::None => (1.0, 1.0),
        ClassWeight::Balanced => {
            let n = y.len() as f64;
            let pos = y.iter().filter(|&&v| v).count() as f64;
            let neg = n - pos;
            ((n / (2.0 * neg)) as f32, (n / (2.0 * pos)) as f32)
        }
    }
}

/// Class-weighted mean binary cross-entropy computed from logits (stable). `exp` and
/// `log1p` come from `libm`, so the loss, and the early stopping it drives, are the
/// same on every operating system.
pub fn weighted_bce_from_logits(logits: &[f32], y: &[bool], w_neg: f32, w_pos: f32) -> f64 {
    let mut s = 0.0;
    for (&z, &t) in logits.iter().zip(y) {
        let z = f64::from(z);
        let l = z.max(0.0) - if t { z } else { 0.0 } + libm::log1p(libm::exp(-z.abs()));
        s += l * f64::from(if t { w_pos } else { w_neg });
    }
    s / y.len().max(1) as f64
}

/// Training objective for one minibatch: class-weighted mean BCE-with-logits plus
/// `l2 / 2 * sum(W^2)` over weights (biases excluded). Returns `(objective, data_loss)`.
pub fn batch_objective(
    model: &DenseNet,
    xb: Vec<f32>,
    yb: Vec<f32>,
    wb: Vec<f32>,
    d: usize,
    l2: f32,
    device: &Device,
) -> (Tensor<1>, Tensor<1>) {
    let b = yb.len();
    let x = Tensor::<2>::from_data(TensorData::new(xb, [b, d]), device);
    let y = Tensor::<1>::from_data(TensorData::new(yb, [b]), device);
    let w = Tensor::<1>::from_data(TensorData::new(wb, [b]), device);
    let z: Tensor<1> = model.forward(x).reshape([b]);
    // BCE with logits: (1 - y) * z - log_sigmoid(z)
    let per_row = y.neg().add_scalar(1.0) * z.clone() - log_sigmoid(z);
    let data_loss = (per_row * w).mean();
    let loss = if l2 > 0.0 {
        data_loss.clone() + model.weight_sq_sum().mul_scalar(l2 / 2.0)
    } else {
        data_loss.clone()
    };
    (loss, data_loss)
}

/// Training objective of one minibatch for the task of `output`: the binary
/// objective of [`batch_objective`], softmax cross-entropy over `k` classes
/// (`yb` holds class indices), or the squared error of the standardised target;
/// each class-weighted, plus `l2 / 2 * sum(W^2)`. Returns `(objective, data_loss)`.
#[allow(clippy::too_many_arguments)]
pub fn task_objective(
    model: &DenseNet,
    task: Task,
    k: usize,
    xb: Vec<f32>,
    yb: Vec<f32>,
    wb: Vec<f32>,
    d: usize,
    l2: f32,
    device: &Device,
) -> (Tensor<1>, Tensor<1>) {
    let b = yb.len();
    let data_loss = match task {
        Task::Binary => return batch_objective(model, xb, yb, wb, d, l2, device),
        Task::Multiclass => {
            let mut onehot = vec![0f32; b * k];
            for (i, &c) in yb.iter().enumerate() {
                onehot[i * k + c as usize] = 1.0;
            }
            let x = Tensor::<2>::from_data(TensorData::new(xb, [b, d]), device);
            let y = Tensor::<2>::from_data(TensorData::new(onehot, [b, k]), device);
            let w = Tensor::<1>::from_data(TensorData::new(wb, [b]), device);
            let per_row: Tensor<1> = (y * log_softmax(model.forward(x), 1)).sum_dim(1).reshape([b]).neg();
            (per_row * w).mean()
        }
        Task::Regression => {
            let x = Tensor::<2>::from_data(TensorData::new(xb, [b, d]), device);
            let y = Tensor::<1>::from_data(TensorData::new(yb, [b]), device);
            let w = Tensor::<1>::from_data(TensorData::new(wb, [b]), device);
            let diff = model.forward(x).reshape([b]) - y;
            (diff.clone() * diff * w).mean()
        }
    };
    let loss = if l2 > 0.0 {
        data_loss.clone() + model.weight_sq_sum().mul_scalar(l2 / 2.0)
    } else {
        data_loss.clone()
    };
    (loss, data_loss)
}

pub fn adam() -> AdamConfig {
    AdamConfig::new().with_beta_1(0.9).with_beta_2(0.999).with_epsilon(1e-8)
}

/// Adds the training losses of consecutive steps (the first being `first_step`) to
/// the epoch's running sum, in step order; the first non-finite one is an error.
fn record_losses(state: &mut TrainState, losses: &[f32], first_step: u64, checkpoint_dir: &Path) -> Result<()> {
    for (i, &value) in losses.iter().enumerate() {
        if !value.is_finite() {
            let mut e = TabsmithError::new(
                ErrorCode::TrainingDiverged,
                Phase::Train,
                format!("training loss became {value} at step {}", first_step + i as u64),
            )
            .next("lower train.<candidate>.learning_rate (e.g. by 10x) and run again");
            if let Some(k) = latest_checkpoint(checkpoint_dir) {
                e = e.retained(format!("last valid checkpoint {}", k.display()));
            }
            return Err(e);
        }
        state.epoch_loss_sum += f64::from(value);
        state.epoch_batches += 1;
    }
    Ok(())
}

/// The runtime's outputs for every row of `x`, computed on all cores. Rows are
/// independent, and each one is computed exactly as `f(x)` would compute it, so the
/// result is bit-identical to `f(x)`. Only matrices of several thousand rows are
/// split: below that, threads cost more than they save.
pub fn par_outputs(x: &Matrix, f: impl Fn(&Matrix) -> Matrix + Sync) -> Matrix {
    const MIN_ROWS_PER_THREAD: usize = 4096;
    let threads = std::thread::available_parallelism()
        .map_or(1, |n| n.get())
        .min(x.rows / MIN_ROWS_PER_THREAD);
    if threads <= 1 {
        return f(x);
    }
    let per = x.rows.div_ceil(threads);
    let f = &f;
    let parts: Vec<Matrix> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..x.rows)
            .step_by(per)
            .map(|lo| {
                let hi = (lo + per).min(x.rows);
                s.spawn(move || {
                    f(&Matrix {
                        rows: hi - lo,
                        cols: x.cols,
                        data: x.data[lo * x.cols..hi * x.cols].to_vec(),
                    })
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().unwrap_or_else(|panic| std::panic::resume_unwind(panic)))
            .collect()
    });
    Matrix {
        rows: x.rows,
        cols: parts[0].cols,
        data: parts.into_iter().flat_map(|m| m.data).collect(),
    }
}

/// Reads back the losses of the steps run since the last read, in one device
/// synchronisation (GPU only; the CPU path records every step as it goes).
fn flush_losses(state: &mut TrainState, pending: &mut Vec<Tensor<1>>, checkpoint_dir: &Path) -> Result<()> {
    if pending.is_empty() {
        return Ok(());
    }
    let first_step = state.step + 1 - pending.len() as u64;
    let values: Vec<f32> = Tensor::cat(std::mem::take(pending), 0)
        .into_data()
        .try_to_vec::<f32>()
        .map_err(|e| train_err(format!("loss read-back: {e:?}")))?;
    record_losses(state, &values, first_step, checkpoint_dir)
}

/// Runs (or continues) training. `on_epoch` is called after each completed epoch.
pub fn train(
    inputs: &TrainInputs<'_>,
    control: &RunControl,
    start: StartFrom,
    on_epoch: &mut dyn FnMut(&TrainState),
) -> Result<TrainOutcome> {
    let base = inputs.device.burn_device()?;
    let device = base.clone().autodiff();
    let eager_losses = inputs.device.reads_losses_every_step();
    let mut pending: Vec<Tensor<1>> = Vec::new();
    let config = &inputs.key.model_config;
    let cfg = inputs.cfg;
    let n = inputs.x_train.rows;
    let d = inputs.x_train.cols;
    let batches_per_epoch = n.div_ceil(cfg.batch_size);
    let task = inputs.output.task;
    let k = inputs.output.n_outputs();
    let weights = class_weights_of(inputs.y_train, cfg.class_weight);
    let (enc_y, enc_w) = encode_targets(inputs.y_train, inputs.output, &weights);

    let (mut model, mut optim, mut state, mut best): (DenseNet, ModuleOptimizer, TrainState, Option<DenseModel>) = match start {
        StartFrom::Scratch => (
            DenseNet::init(config, inputs.seed, &device),
            adam().init(),
            TrainState::fresh(),
            None,
        ),
        StartFrom::WarmStart(m) => (DenseNet::from_dense(&m, &device), adam().init(), TrainState::fresh(), None),
        StartFrom::Checkpoint(dir) => {
            let ck = load_checkpoint(&dir, &inputs.key, &device)?;
            (ck.model, ck.optim, ck.state, ck.best)
        }
    };

    let started = Instant::now();
    let mut last_checkpoint: Option<Instant> = None;
    let elapsed_before = state.elapsed_ms;
    let l2 = cfg.l2 as f32;
    while state.epoch < cfg.epochs && state.stop_reason.is_none() {
        let perm = epoch_permutation(inputs.seed, state.epoch, n);
        while state.next_batch < batches_per_epoch {
            if control.cancel.load(Ordering::SeqCst) {
                flush_losses(&mut state, &mut pending, &inputs.checkpoint_dir)?;
                state.elapsed_ms = elapsed_before + started.elapsed().as_secs_f64() * 1e3;
                let ck = save_checkpoint(&inputs.checkpoint_dir, &inputs.key, &model, &optim, &state, best.as_ref())?;
                return Ok(TrainOutcome::Cancelled { state, checkpoint: ck });
            }
            let lo = state.next_batch * cfg.batch_size;
            let hi = (lo + cfg.batch_size).min(n);
            let rows = &perm[lo..hi];
            let b = rows.len();
            let mut xb = Vec::with_capacity(b * d);
            let mut yb = Vec::with_capacity(b);
            let mut wb = Vec::with_capacity(b);
            for &r in rows {
                xb.extend_from_slice(inputs.x_train.row(r));
                yb.push(enc_y[r]);
                wb.push(enc_w[r]);
            }
            let (loss, data_loss) = task_objective(&model, task, k, xb, yb, wb, d, l2, &device);
            if eager_losses {
                // Checked before the optimizer step: a diverged step never updates the weights.
                let step = state.step + 1;
                record_losses(&mut state, &[data_loss.into_scalar::<f32>()], step, &inputs.checkpoint_dir)?;
            } else {
                pending.push(data_loss.without_autodiff());
            }
            let grads = GradientsParams::from_grads(loss.backward(), &model);
            model = optim.step(cfg.learning_rate, model, grads);
            state.step += 1;
            state.next_batch += 1;
            if control.cancel_after_steps == Some(state.step) {
                control.cancel.store(true, Ordering::SeqCst);
            }
            if control.crash_after_steps == Some(state.step) {
                eprintln!("[test hook] simulating a crash after step {}", state.step);
                std::process::exit(137);
            }
        }

        // End of epoch: validation through the runtime forward pass (what will ship).
        flush_losses(&mut state, &mut pending, &inputs.checkpoint_dir)?;
        let exported = model.export(config)?;
        let v_out = par_outputs(inputs.x_valid, |x| exported.outputs(x));
        let valid_loss = task_loss(&v_out, inputs.y_valid, inputs.output, &weights);
        let metric = task_metrics(&v_out, inputs.y_valid, inputs.output)
            .get(inputs.metric.as_str())
            .unwrap_or(0.0);
        if !valid_loss.is_finite() {
            return Err(TabsmithError::new(
                ErrorCode::TrainingDiverged,
                Phase::Train,
                format!("validation loss became {valid_loss} after epoch {}", state.epoch + 1),
            )
            .next("lower the learning rate and run again"));
        }
        state.history.push(EpochRecord {
            epoch: state.epoch + 1,
            step: state.step,
            train_loss: state.epoch_loss_sum / state.epoch_batches.max(1) as f64,
            valid_loss,
            valid_metric: metric,
        });
        let improved = state.best_valid_loss.is_none_or(|b| valid_loss < b);
        if improved {
            state.best_valid_loss = Some(valid_loss);
            state.best_epoch = state.epoch + 1;
            state.bad_epochs = 0;
            best = Some(exported);
        } else {
            state.bad_epochs += 1;
        }
        state.epoch += 1;
        state.next_batch = 0;
        state.epoch_loss_sum = 0.0;
        state.epoch_batches = 0;
        state.elapsed_ms = elapsed_before + started.elapsed().as_secs_f64() * 1e3;
        if cfg.early_stopping_patience > 0 && state.bad_epochs >= cfg.early_stopping_patience {
            state.stop_reason = Some("early_stopping".into());
        } else if inputs.max_train_seconds > 0 && state.elapsed_ms >= inputs.max_train_seconds as f64 * 1e3 {
            state.stop_reason = Some("time_budget".into());
        } else if state.epoch >= cfg.epochs {
            state.stop_reason = Some("epochs".into());
        }
        let due = state.epoch % cfg.checkpoint_every_epochs == 0
            && last_checkpoint.is_none_or(|t: Instant| t.elapsed().as_secs_f64() >= cfg.checkpoint_min_interval_seconds);
        if due || state.stop_reason.is_some() {
            save_checkpoint(&inputs.checkpoint_dir, &inputs.key, &model, &optim, &state, best.as_ref())?;
            last_checkpoint = Some(Instant::now());
        }
        on_epoch(&state);
    }
    let best = match best {
        Some(b) => b,
        None => model.export(config)?,
    };
    // Engine/runtime parity on the validation matrix (same weights, two implementations).
    let parity = {
        let engine = DenseNet::from_dense(&best, &base);
        let x = Tensor::<2>::from_data(TensorData::new(inputs.x_valid.data.clone(), [inputs.x_valid.rows, d]), &base);
        let burn_logits: Vec<f32> = engine
            .forward(x)
            .into_data()
            .try_to_vec::<f32>()
            .map_err(|e| train_err(format!("parity export: {e:?}")))?;
        let rt = par_outputs(inputs.x_valid, |x| best.outputs(x)).data;
        burn_logits
            .iter()
            .zip(&rt)
            .map(|(a, b)| f64::from((a - b).abs()))
            .fold(0.0, f64::max)
    };
    Ok(TrainOutcome::Completed {
        best: Model::Dense(best),
        state,
        parity_max_abs_diff: parity,
    })
}

// ---------------------------------------------------------------------------
// Checkpoints
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CheckpointFiles {
    files: Vec<(String, String, u64)>,
}

pub struct LoadedCheckpoint {
    pub model: DenseNet,
    pub optim: ModuleOptimizer,
    pub state: TrainState,
    pub best: Option<DenseModel>,
}

fn ck_err(code: ErrorCode, msg: impl Into<String>) -> TabsmithError {
    TabsmithError::new(code, Phase::Resume, msg)
}

/// Writes a checkpoint atomically (temp dir, fsync, rename) and keeps the last two.
pub fn save_checkpoint(
    root: &Path,
    key: &ResumeKey,
    model: &DenseNet,
    optim: &ModuleOptimizer,
    state: &TrainState,
    best: Option<&DenseModel>,
) -> Result<PathBuf> {
    let (tmp, final_dir) = checkpoint_dirs(root, state.step)?;
    let model_bytes = model
        .clone()
        .into_record()
        .into_bytes()
        .map_err(|e| train_err(format!("model record: {e:?}")))?;
    write_atomic(&tmp.join("model.bpk"), &model_bytes)?;
    let optim_bytes = optim.into_bytes().map_err(|e| train_err(format!("optimizer record: {e:?}")))?;
    write_atomic(&tmp.join("optim.bpk"), &optim_bytes)?;
    write_json(&tmp.join("state.json"), state)?;
    write_json(&tmp.join("key.json"), key)?;
    if let Some(b) = best {
        write_atomic(&tmp.join("best.safetensors"), &b.to_safetensors()?)?;
    }
    seal_checkpoint(
        root,
        &tmp,
        &final_dir,
        &["model.bpk", "optim.bpk", "state.json", "key.json", "best.safetensors"],
    )
}

/// A fresh temporary directory for the checkpoint of `step`, and its final name.
pub(crate) fn checkpoint_dirs(root: &Path, step: u64) -> Result<(PathBuf, PathBuf)> {
    fs::create_dir_all(root).map_err(|e| TabsmithError::io(Phase::Train, root, e))?;
    let name = format!("step-{step:09}");
    let tmp = root.join(format!(".{name}.tmp"));
    if tmp.exists() {
        fs::remove_dir_all(&tmp).map_err(|e| TabsmithError::io(Phase::Train, &tmp, e))?;
    }
    fs::create_dir_all(&tmp).map_err(|e| TabsmithError::io(Phase::Train, &tmp, e))?;
    Ok((tmp, root.join(name)))
}

/// Records the hashes of the files `names` that exist in `tmp`, renames it to
/// `final_dir`, and keeps the two most recent checkpoints.
pub(crate) fn seal_checkpoint(root: &Path, tmp: &Path, final_dir: &Path, names: &[&str]) -> Result<PathBuf> {
    let mut files = Vec::new();
    for f in names {
        let p = tmp.join(f);
        if p.is_file() {
            let (sha, bytes) = sha256_file(&p).map_err(|e| TabsmithError::io(Phase::Train, &p, e))?;
            files.push((f.to_string(), sha, bytes));
        }
    }
    write_json(&tmp.join("files.json"), &CheckpointFiles { files })?;
    let final_dir = final_dir.to_path_buf();
    if final_dir.exists() {
        fs::remove_dir_all(&final_dir).map_err(|e| TabsmithError::io(Phase::Train, &final_dir, e))?;
    }
    fs::rename(tmp, &final_dir).map_err(|e| TabsmithError::io(Phase::Train, &final_dir, e))?;
    // Keep the two most recent valid checkpoints.
    let mut all = list_checkpoints(root);
    while all.len() > 2 {
        let old = all.remove(0);
        let _ = fs::remove_dir_all(old);
    }
    Ok(final_dir)
}

/// Checkpoint directories, oldest first (temp dirs ignored).
pub fn list_checkpoints(root: &Path) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = fs::read_dir(root)
        .map(|it| {
            it.flatten()
                .map(|e| e.path())
                .filter(|p| p.is_dir() && p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with("step-")))
                .collect()
        })
        .unwrap_or_default();
    v.sort();
    v
}

pub fn latest_checkpoint(root: &Path) -> Option<PathBuf> {
    list_checkpoints(root).pop()
}

/// Verifies every file of a checkpoint against its recorded hash.
pub fn verify_checkpoint(dir: &Path) -> Result<()> {
    let files: CheckpointFiles = crate::project::read_json(&dir.join("files.json"), Phase::Resume).map_err(|_| {
        ck_err(
            ErrorCode::CheckpointCorrupt,
            format!("checkpoint `{}` has no file list", dir.display()),
        )
    })?;
    for (f, sha, bytes) in &files.files {
        let p = dir.join(f);
        let ok = sha256_file(&p).map(|(s, b)| s == *sha && b == *bytes).unwrap_or(false);
        if !ok {
            return Err(ck_err(
                ErrorCode::CheckpointCorrupt,
                format!("checkpoint file `{}` is missing or damaged", p.display()),
            )
            .field(p.display().to_string())
            .next("an older checkpoint is used automatically when one verifies; otherwise start a new run (`tabsmith run`)"));
        }
    }
    Ok(())
}

/// Verifies hashes and compatibility, then restores model, optimizer and loop state.
pub fn load_checkpoint(dir: &Path, key: &ResumeKey, device: &Device) -> Result<LoadedCheckpoint> {
    verify_checkpoint(dir)?;
    let saved: ResumeKey = crate::project::read_json(&dir.join("key.json"), Phase::Resume)?;
    let diffs = saved.differences(key);
    if !diffs.is_empty() {
        return Err(ck_err(
            ErrorCode::ResumeIncompatible,
            format!("cannot continue exactly (R2): {}", diffs.join("; ")),
        )
        .field(dir.display().to_string())
        .next("start a new run (`tabsmith run`), or continue from the old weights with `tabsmith resume <run> --warm-start` (a new training, not a resume)"));
    }
    let state: TrainState = crate::project::read_json(&dir.join("state.json"), Phase::Resume)?;
    let model_bytes = fs::read(dir.join("model.bpk")).map_err(|e| TabsmithError::io(Phase::Resume, dir, e))?;
    let record = ModuleRecord::from_bytes(Bytes::from_bytes_vec(model_bytes))
        .map_err(|e| ck_err(ErrorCode::CheckpointCorrupt, format!("model record: {e:?}")))?;
    let model = DenseNet::init(&key.model_config, key.seed, device).load_record(record);
    let optim_bytes = fs::read(dir.join("optim.bpk")).map_err(|e| TabsmithError::io(Phase::Resume, dir, e))?;
    let optim = adam()
        .init()
        .from_bytes(Bytes::from_bytes_vec(optim_bytes))
        .map_err(|e| ck_err(ErrorCode::CheckpointCorrupt, format!("optimizer record: {e:?}")))?;
    let best = match fs::read(dir.join("best.safetensors")) {
        Ok(bytes) => Some(DenseModel::from_safetensors(key.model_config.clone(), &bytes).map_err(|e| e.in_phase(Phase::Resume))?),
        Err(_) => None,
    };
    Ok(LoadedCheckpoint { model, optim, state, best })
}

/// Hash of a model's weights (bitwise identity check).
pub fn weights_hash(model: &Model) -> String {
    match model {
        Model::Dense(m) => {
            let bits: Vec<Vec<u32>> = m
                .layers
                .iter()
                .flat_map(|l| {
                    [
                        l.weight.iter().map(|v| v.to_bits()).collect(),
                        l.bias.iter().map(|v| v.to_bits()).collect(),
                    ]
                })
                .collect();
            canonical_hash(&bits)
        }
        Model::Trees(t) => canonical_hash(&(
            &t.roots,
            &t.feature,
            t.threshold.iter().map(|v| v.to_bits()).collect::<Vec<u32>>(),
            &t.left,
            &t.right,
            t.value.iter().map(|v| v.to_bits()).collect::<Vec<u32>>(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tabsmith_runtime::model::DenseModel;

    fn data(n: usize, d: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut rng = SplitMix64::new(11);
        let x: Vec<f32> = (0..n * d).map(|_| rng.normal() as f32).collect();
        let y: Vec<f32> = (0..n)
            .map(|i| if x[i * d] + 0.5 * x[i * d + 1] > 0.2 { 1.0 } else { 0.0 })
            .collect();
        let w: Vec<f32> = y.iter().map(|&t| if t > 0.5 { 1.7 } else { 0.6 }).collect();
        (x, y, w)
    }

    /// Independent f64 objective from exported weights.
    fn objective_f64(m: &DenseModel, x: &[f32], y: &[f32], w: &[f32], d: usize, l2: f64) -> f64 {
        let n = y.len();
        let mut total = 0.0;
        for i in 0..n {
            let mut h: Vec<f64> = x[i * d..(i + 1) * d].iter().map(|&v| f64::from(v)).collect();
            for l in &m.layers {
                let mut out = vec![0.0; l.out_dim];
                for o in 0..l.out_dim {
                    let mut acc = f64::from(l.bias[o]);
                    for k in 0..l.in_dim {
                        acc += f64::from(l.weight[o * l.in_dim + k]) * h[k];
                    }
                    out[o] = if l.activation == tabsmith_contracts::model::Activation::Relu {
                        acc.max(0.0)
                    } else {
                        acc
                    };
                }
                h = out;
            }
            let z = h[0];
            let t = f64::from(y[i]);
            total += f64::from(w[i]) * (z.max(0.0) - t * z + libm::log1p(libm::exp(-z.abs())));
        }
        let reg: f64 = m
            .layers
            .iter()
            .flat_map(|l| l.weight.iter())
            .map(|&v| f64::from(v) * f64::from(v))
            .sum();
        total / n as f64 + 0.5 * l2 * reg
    }

    fn grads_of(model: &DenseNet, x: &[f32], y: &[f32], w: &[f32], d: usize, l2: f32) -> (f64, Vec<Vec<f32>>) {
        let device = Device::flex().autodiff();
        let (loss, _) = batch_objective(model, x.to_vec(), y.to_vec(), w.to_vec(), d, l2, &device);
        let value = f64::from(loss.clone().into_scalar::<f32>());
        let grads = loss.backward();
        let mut out = Vec::new();
        for l in &model.layers {
            out.push(l.weight.val().grad(&grads).unwrap().into_data().try_to_vec::<f32>().unwrap());
            out.push(
                l.bias
                    .as_ref()
                    .unwrap()
                    .val()
                    .grad(&grads)
                    .unwrap()
                    .into_data()
                    .try_to_vec::<f32>()
                    .unwrap(),
            );
        }
        (value, out)
    }

    #[test]
    fn loss_and_gradients_match_independent_f64_reference() {
        let (n, d, l2) = (64, 5, 0.01f32);
        let (x, y, w) = data(n, d);
        let device = Device::flex().autodiff();
        for hidden in [vec![], vec![7]] {
            let family = if hidden.is_empty() { "linear" } else { "small-mlp" };
            let config = ModelConfig::dense(family, d, &hidden);
            // Non-zero weights for the linear case too, so the L2 term matters.
            let mut model = DenseNet::init(&config, 3, &device);
            if hidden.is_empty() {
                let exported = DenseModel::new(
                    config.clone(),
                    vec![tabsmith_runtime::model::DenseLayer {
                        in_dim: d,
                        out_dim: 1,
                        weight: vec![0.3, -0.2, 0.1, 0.05, -0.4],
                        bias: vec![0.1],
                        activation: tabsmith_contracts::model::Activation::Identity,
                    }],
                )
                .unwrap();
                model = DenseNet::from_dense(&exported, &device);
            }
            let exported = model.export(&config).unwrap();
            let (burn_loss, burn_grads) = grads_of(&model, &x, &y, &w, d, l2);
            let ref_loss = objective_f64(&exported, &x, &y, &w, d, f64::from(l2));
            assert!(
                (burn_loss - ref_loss).abs() < 1e-5 * ref_loss.max(1.0),
                "{family}: loss {burn_loss} vs {ref_loss}"
            );
            // Central finite differences of the f64 objective, per parameter.
            let h = 1e-4f32;
            for (li, layer) in exported.layers.iter().enumerate() {
                for (pi, is_bias) in [(0usize, false), (1usize, true)] {
                    let n_params = if is_bias { layer.out_dim } else { layer.in_dim * layer.out_dim };
                    let burn_g = &burn_grads[li * 2 + pi];
                    for k in 0..n_params {
                        let mut plus = exported.clone();
                        let mut minus = exported.clone();
                        // Runtime layout is [out, in]; Burn's is [in, out].
                        let (idx_rt, idx_burn) = if is_bias {
                            (k, k)
                        } else {
                            let (o, i) = (k / layer.in_dim, k % layer.in_dim);
                            (k, i * layer.out_dim + o)
                        };
                        let step = if is_bias {
                            plus.layers[li].bias[idx_rt] += h;
                            minus.layers[li].bias[idx_rt] -= h;
                            f64::from(plus.layers[li].bias[idx_rt]) - f64::from(minus.layers[li].bias[idx_rt])
                        } else {
                            plus.layers[li].weight[idx_rt] += h;
                            minus.layers[li].weight[idx_rt] -= h;
                            f64::from(plus.layers[li].weight[idx_rt]) - f64::from(minus.layers[li].weight[idx_rt])
                        };
                        // Divide by the exact (f32-rounded) step.
                        let fd = (objective_f64(&plus, &x, &y, &w, d, f64::from(l2)) - objective_f64(&minus, &x, &y, &w, d, f64::from(l2)))
                            / step;
                        let g = f64::from(burn_g[idx_burn]);
                        assert!(
                            (g - fd).abs() <= 2e-3 * fd.abs().max(1e-2),
                            "{family} layer {li} {} param {k}: autodiff {g} vs finite difference {fd}",
                            if is_bias { "bias" } else { "weight" }
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn first_adam_step_matches_the_update_rule() {
        let (n, d, l2) = (32, 4, 0.0f32);
        let (x, y, w) = data(n, d);
        let device = Device::flex().autodiff();
        let config = ModelConfig::dense("small-mlp", d, &[3]);
        let model = DenseNet::init(&config, 5, &device);
        let before = model.export(&config).unwrap();
        let (_, g) = grads_of(&model, &x, &y, &w, d, l2);
        let (loss, _) = batch_objective(&model, x.clone(), y.clone(), w.clone(), d, l2, &device);
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        let lr = 0.01;
        let after = adam().init().step(lr, model, grads).export(&config).unwrap();
        // Adam, first step: m_hat = g, v_hat = g^2  =>  delta = -lr * g / (|g| + eps)
        for (li, (b, a)) in before.layers.iter().zip(&after.layers).enumerate() {
            for o in 0..b.out_dim {
                for i in 0..b.in_dim {
                    let gk = f64::from(g[li * 2][i * b.out_dim + o]);
                    let expected = f64::from(b.weight[o * b.in_dim + i]) - lr * gk / (gk.abs() + 1e-8);
                    let got = f64::from(a.weight[o * b.in_dim + i]);
                    assert!((got - expected).abs() < 1e-6, "layer {li} w[{o},{i}]: {got} vs {expected}");
                }
                let gb = f64::from(g[li * 2 + 1][o]);
                let expected = f64::from(b.bias[o]) - lr * gb / (gb.abs() + 1e-8);
                assert!((f64::from(a.bias[o]) - expected).abs() < 1e-6);
            }
        }
    }
}
