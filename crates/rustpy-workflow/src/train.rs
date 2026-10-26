//! Training of the `linear` and `small-mlp` candidates with Burn (CPU, NdArray).
//!
//! Design for exact resume (R2):
//! * weights are initialised from RustPy's own versioned PRNG, not the engine's;
//! * the minibatch order is a pure function of (seed, epoch);
//! * no stochastic layers (no dropout);
//! * a checkpoint stores model record + optimizer record (Adam moments, step
//!   counts) + the host-side loop state (epoch, batch cursor, partial epoch loss,
//!   early-stopping state, best weights, history).
//!
//! Continuing from a checkpoint therefore reproduces the uninterrupted run bit for
//! bit on the same binary and CPU feature set (verified by tests).
#![allow(deprecated)] // `Device::ndarray()` is deprecated in Burn 0.22 in favour of Flex.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use burn::module::{Module, Param};
use burn::nn::{Initializer, Linear, LinearConfig, Relu};
use burn::optim::{AdamConfig, GradientsParams, ModuleOptimizer};
use burn::store::ModuleRecord;
use burn::tensor::activation::log_sigmoid;
use burn::tensor::{Bytes, Device, Tensor, TensorData};
use rustpy_contracts::config::{Candidate, CandidateTrainConfig, ClassWeight, SelectionMetric};
use rustpy_contracts::hash::{canonical_hash, sha256_file};
use rustpy_contracts::model::ModelConfig;
use rustpy_contracts::run::EpochRecord;
use rustpy_contracts::{ErrorCode, Phase, Result, RustpyError};
use rustpy_runtime::model::{DenseLayer, DenseModel};
use rustpy_runtime::Matrix;
use serde::{Deserialize, Serialize};

use crate::metrics::binary_metrics;
use crate::project::{write_atomic, write_json};
use crate::rng::{epoch_permutation, SplitMix64};

/// Engine-side network: dense layers with ReLU between them.
#[derive(Module, Debug)]
pub struct DenseNet {
    layers: Vec<Linear>,
    act: Relu,
}

fn train_err(msg: impl Into<String>) -> RustpyError {
    RustpyError::internal(Phase::Train, msg)
}

impl DenseNet {
    /// Initialises weights from RustPy's PRNG: zeros for the linear baseline,
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
                .to_vec::<f32>()
                .map_err(|e| train_err(format!("weight export: {e:?}")))?;
            let b: Vec<f32> = match &l.bias {
                Some(b) => b.val().into_data().to_vec::<f32>().map_err(|e| train_err(format!("bias export: {e:?}")))?,
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
    let mut lin = LinearConfig::new(in_dim, out_dim)
        .with_initializer(Initializer::Zeros)
        .init(device);
    lin.weight = Param::from_tensor(Tensor::<2>::from_data(TensorData::new(w, [in_dim, out_dim]), device));
    lin.bias = Some(Param::from_tensor(Tensor::<1>::from_data(TensorData::new(b, [out_dim]), device)));
    lin
}

pub fn model_config(candidate: Candidate, input_dim: usize, cfg: &CandidateTrainConfig) -> ModelConfig {
    match candidate {
        Candidate::Linear => ModelConfig::dense("linear", input_dim, &[]),
        Candidate::SmallMlp => ModelConfig::dense("small-mlp", input_dim, &cfg.hidden),
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
    pub rustpy_version: String,
    pub n_train: usize,
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
        cmp!(rustpy_version, "rustpy version");
        cmp!(n_train, "training rows");
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
    fn fresh() -> Self {
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
    /// to simulate a crash (`rustpy run --simulate-crash-after-steps N`, hidden).
    pub crash_after_steps: Option<u64>,
}

pub struct TrainInputs<'a> {
    pub candidate: Candidate,
    pub cfg: &'a CandidateTrainConfig,
    pub seed: u64,
    pub threshold: f64,
    pub metric: SelectionMetric,
    pub max_train_seconds: u64,
    pub x_train: &'a Matrix,
    pub y_train: &'a [bool],
    pub x_valid: &'a Matrix,
    pub y_valid: &'a [bool],
    pub checkpoint_dir: PathBuf,
    pub key: ResumeKey,
}

pub enum TrainOutcome {
    Completed { best: DenseModel, state: TrainState, parity_max_abs_diff: f64 },
    Cancelled { state: TrainState, checkpoint: PathBuf },
}

/// Where training starts from.
pub enum StartFrom {
    Scratch,
    /// R2: continue exactly from a checkpoint.
    Checkpoint(PathBuf),
    /// Fresh optimizer, weights copied from a previous model.
    WarmStart(DenseModel),
}

/// Per-sample weights for the training loss.
fn class_weights(y: &[bool], mode: ClassWeight) -> (f32, f32) {
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

/// Class-weighted mean binary cross-entropy computed from logits (stable).
pub fn weighted_bce_from_logits(logits: &[f32], y: &[bool], w_neg: f32, w_pos: f32) -> f64 {
    let mut s = 0.0;
    for (&z, &t) in logits.iter().zip(y) {
        let z = f64::from(z);
        let l = z.max(0.0) - if t { z } else { 0.0 } + (-z.abs()).exp().ln_1p();
        s += l * f64::from(if t { w_pos } else { w_neg });
    }
    s / y.len().max(1) as f64
}

fn adam() -> AdamConfig {
    AdamConfig::new().with_beta_1(0.9).with_beta_2(0.999).with_epsilon(1e-8)
}

/// Runs (or continues) training. `on_epoch` is called after each completed epoch.
pub fn train(
    inputs: &TrainInputs<'_>,
    control: &RunControl,
    start: StartFrom,
    on_epoch: &mut dyn FnMut(&TrainState),
) -> Result<TrainOutcome> {
    let base = Device::ndarray();
    let device = base.clone().autodiff();
    let config = &inputs.key.model_config;
    let cfg = inputs.cfg;
    let n = inputs.x_train.rows;
    let d = inputs.x_train.cols;
    let batches_per_epoch = n.div_ceil(cfg.batch_size);
    let (w_neg, w_pos) = class_weights(inputs.y_train, cfg.class_weight);

    let (mut model, mut optim, mut state, mut best): (DenseNet, ModuleOptimizer, TrainState, Option<DenseModel>) =
        match start {
            StartFrom::Scratch => (DenseNet::init(config, inputs.seed, &device), adam().init(), TrainState::fresh(), None),
            StartFrom::WarmStart(m) => (DenseNet::from_dense(&m, &device), adam().init(), TrainState::fresh(), None),
            StartFrom::Checkpoint(dir) => {
                let ck = load_checkpoint(&dir, &inputs.key, &device)?;
                (ck.model, ck.optim, ck.state, ck.best)
            }
        };

    let started = Instant::now();
    let elapsed_before = state.elapsed_ms;
    let l2 = cfg.l2 as f32;
    while state.epoch < cfg.epochs && state.stop_reason.is_none() {
        let perm = epoch_permutation(inputs.seed, state.epoch, n);
        while state.next_batch < batches_per_epoch {
            if control.cancel.load(Ordering::SeqCst) {
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
                let t = inputs.y_train[r];
                yb.push(if t { 1f32 } else { 0f32 });
                wb.push(if t { w_pos } else { w_neg });
            }
            let x = Tensor::<2>::from_data(TensorData::new(xb, [b, d]), &device);
            let y = Tensor::<1>::from_data(TensorData::new(yb, [b]), &device);
            let w = Tensor::<1>::from_data(TensorData::new(wb, [b]), &device);
            let z: Tensor<1> = model.forward(x).reshape([b]);
            // BCE with logits: (1 - y) * z - log_sigmoid(z)
            let per_row = y.neg().add_scalar(1.0) * z.clone() - log_sigmoid(z);
            let data_loss = (per_row * w).mean();
            let loss = if l2 > 0.0 {
                data_loss.clone() + model.weight_sq_sum().mul_scalar(l2 / 2.0)
            } else {
                data_loss.clone()
            };
            let loss_value = data_loss.into_scalar::<f32>();
            if !loss_value.is_finite() {
                let kept = latest_checkpoint(&inputs.checkpoint_dir).map(|p| p.display().to_string());
                let mut e = RustpyError::new(
                    ErrorCode::TrainingDiverged,
                    Phase::Train,
                    format!("training loss became {loss_value} at step {}", state.step + 1),
                )
                .next("lower train.<candidate>.learning_rate (e.g. by 10x) and run again");
                if let Some(k) = kept {
                    e = e.retained(format!("last valid checkpoint {k}"));
                }
                return Err(e);
            }
            let grads = GradientsParams::from_grads(loss.backward(), &model);
            model = optim.step(cfg.learning_rate, model, grads);
            state.step += 1;
            state.next_batch += 1;
            state.epoch_loss_sum += f64::from(loss_value);
            state.epoch_batches += 1;
            if control.crash_after_steps == Some(state.step) {
                eprintln!("[test hook] simulating a crash after step {}", state.step);
                std::process::exit(137);
            }
        }

        // End of epoch: validation through the runtime forward pass (what will ship).
        let exported = model.export(config)?;
        let v_logits = exported.logits(inputs.x_valid);
        let valid_loss = weighted_bce_from_logits(&v_logits, inputs.y_valid, w_neg, w_pos);
        let v_probs: Vec<f64> = v_logits.iter().map(|&z| rustpy_runtime::model::sigmoid(f64::from(z))).collect();
        let vm = binary_metrics(&v_probs, inputs.y_valid, inputs.threshold);
        let metric = vm.get(inputs.metric.as_str()).unwrap_or(0.0);
        if !valid_loss.is_finite() {
            return Err(RustpyError::new(
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
        if state.epoch % cfg.checkpoint_every_epochs == 0 || state.stop_reason.is_some() {
            save_checkpoint(&inputs.checkpoint_dir, &inputs.key, &model, &optim, &state, best.as_ref())?;
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
            .to_vec::<f32>()
            .map_err(|e| train_err(format!("parity export: {e:?}")))?;
        let rt = best.logits(inputs.x_valid);
        burn_logits
            .iter()
            .zip(&rt)
            .map(|(a, b)| f64::from((a - b).abs()))
            .fold(0.0, f64::max)
    };
    Ok(TrainOutcome::Completed {
        best,
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

fn ck_err(code: ErrorCode, msg: impl Into<String>) -> RustpyError {
    RustpyError::new(code, Phase::Resume, msg)
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
    fs::create_dir_all(root).map_err(|e| RustpyError::io(Phase::Train, root, e))?;
    let name = format!("step-{:09}", state.step);
    let final_dir = root.join(&name);
    let tmp = root.join(format!(".{name}.tmp"));
    if tmp.exists() {
        fs::remove_dir_all(&tmp).map_err(|e| RustpyError::io(Phase::Train, &tmp, e))?;
    }
    fs::create_dir_all(&tmp).map_err(|e| RustpyError::io(Phase::Train, &tmp, e))?;
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
    let mut files = Vec::new();
    for f in ["model.bpk", "optim.bpk", "state.json", "key.json", "best.safetensors"] {
        let p = tmp.join(f);
        if p.is_file() {
            let (sha, bytes) = sha256_file(&p).map_err(|e| RustpyError::io(Phase::Train, &p, e))?;
            files.push((f.to_string(), sha, bytes));
        }
    }
    write_json(&tmp.join("files.json"), &CheckpointFiles { files })?;
    if final_dir.exists() {
        fs::remove_dir_all(&final_dir).map_err(|e| RustpyError::io(Phase::Train, &final_dir, e))?;
    }
    fs::rename(&tmp, &final_dir).map_err(|e| RustpyError::io(Phase::Train, &final_dir, e))?;
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
                .filter(|p| {
                    p.is_dir()
                        && p.file_name()
                            .and_then(|n| n.to_str())
                            .is_some_and(|n| n.starts_with("step-"))
                })
                .collect()
        })
        .unwrap_or_default();
    v.sort();
    v
}

pub fn latest_checkpoint(root: &Path) -> Option<PathBuf> {
    list_checkpoints(root).pop()
}

/// Verifies hashes and compatibility, then restores model, optimizer and loop state.
pub fn load_checkpoint(dir: &Path, key: &ResumeKey, device: &Device) -> Result<LoadedCheckpoint> {
    let files: CheckpointFiles = crate::project::read_json(&dir.join("files.json"), Phase::Resume)
        .map_err(|_| ck_err(ErrorCode::CheckpointCorrupt, format!("checkpoint `{}` has no file list", dir.display())))?;
    for (f, sha, bytes) in &files.files {
        let p = dir.join(f);
        let ok = sha256_file(&p).map(|(s, b)| s == *sha && b == *bytes).unwrap_or(false);
        if !ok {
            return Err(ck_err(
                ErrorCode::CheckpointCorrupt,
                format!("checkpoint file `{}` is missing or damaged", p.display()),
            )
            .field(p.display().to_string())
            .next("resume from an earlier checkpoint is not possible automatically; start a new run (`rustpy run`)"));
        }
    }
    let saved: ResumeKey = crate::project::read_json(&dir.join("key.json"), Phase::Resume)?;
    let diffs = saved.differences(key);
    if !diffs.is_empty() {
        return Err(ck_err(
            ErrorCode::ResumeIncompatible,
            format!("cannot continue exactly (R2): {}", diffs.join("; ")),
        )
        .field(dir.display().to_string())
        .next("start a new run (`rustpy run`), or continue from the old weights with `rustpy resume <run> --warm-start` (a new training, not a resume)"));
    }
    let state: TrainState = crate::project::read_json(&dir.join("state.json"), Phase::Resume)?;
    let model_bytes = fs::read(dir.join("model.bpk")).map_err(|e| RustpyError::io(Phase::Resume, dir, e))?;
    let record = ModuleRecord::from_bytes(Bytes::from_bytes_vec(model_bytes))
        .map_err(|e| ck_err(ErrorCode::CheckpointCorrupt, format!("model record: {e:?}")))?;
    let model = DenseNet::init(&key.model_config, key.seed, device).load_record(record);
    let optim_bytes = fs::read(dir.join("optim.bpk")).map_err(|e| RustpyError::io(Phase::Resume, dir, e))?;
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
pub fn weights_hash(model: &DenseModel) -> String {
    let bits: Vec<Vec<u32>> = model
        .layers
        .iter()
        .flat_map(|l| [l.weight.iter().map(|v| v.to_bits()).collect(), l.bias.iter().map(|v| v.to_bits()).collect()])
        .collect();
    canonical_hash(&bits)
}
