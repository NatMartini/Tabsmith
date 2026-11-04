//! The `gbdt` candidate: gradient-boosted decision trees on the fitted feature matrix (the same
//! transform as the networks). Trees grow best-first, the leaf with the largest gain splitting
//! next. The loss is the task's: class-weighted log loss (binary), class-weighted softmax
//! cross-entropy with one tree per class and round (multiclass), squared error of the
//! standardised target (regression); early stopping watches the validation loss.
//!
//! Exact by construction. A row's gradient and hessian become integers at a scale fixed per tree,
//! and a histogram cell is a sum of integers. Integer addition has no rounding and no order, so a
//! cell holds the same value whichever thread adds which row, and one child's histogram is exactly
//! its parent's minus its sibling's. Every decision is a pure function of those sums, and of
//! nothing else: the split with the largest gain, first in (feature, bin) order on ties, the leaf
//! with the largest gain, first in creation order on ties, a leaf's value. So how the sums are
//! computed cannot change a byte of the model. The number of threads cannot, nor the order of the
//! rows, nor which features are summed in one pass; tests train with each of these changed.
//! Thresholds are a pure function of counts of values in cells, in the same way.
//!
//! The float arithmetic that remains (a row's gradient, a split's gain) uses only IEEE operations
//! and `libm`, so the same data gives the same model on every operating system.
//!
//! One boosting round is one "epoch" of the shared [`TrainState`], so checkpoints, cancellation,
//! crash recovery and exact resume (R2) work as for the networks.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering::Relaxed;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use tabsmith_contracts::config::{CandidateTrainConfig, Task};
use tabsmith_contracts::model::{ModelConfig, TreesSpec};
use tabsmith_contracts::run::EpochRecord;
use tabsmith_contracts::schema::OutputSchema;
use tabsmith_contracts::transform::{FeatureTransform, FittedTransform};
use tabsmith_contracts::{ErrorCode, Phase, Result, TabsmithError};
use tabsmith_embedded::kernels;
use tabsmith_runtime::Matrix;
use tabsmith_runtime::model::{Model, TreeEnsemble, sigmoid};

use self::bins::Bins;
use self::grow::{FinishedLeaf, FreeHists, Growth, Units, exponent_for, grow_tree, nearest, two_pow};
use self::rows::{PIECE_ROWS, Rows, lock, pack, piece_range, section};
use crate::data::Targets;
use crate::metrics::selected_metric;
use crate::parallel::{install, threads};
use crate::pool::{Pool, with_pool};
use crate::project::{read_json, write_atomic, write_json};
use crate::train::{
    ResumeKey, RunControl, StartFrom, TrainInputs, TrainOutcome, TrainState, checkpoint_dirs, class_weights_of, encode_targets,
    par_outputs, row_loss, seal_checkpoint, verify_checkpoint,
};

mod bins;
mod grow;
mod rows;
#[cfg(test)]
mod tests;

/// The training loss of a round is the loss of at most about this many rows, always the same ones.
const TRAIN_LOSS_ROWS: usize = 1 << 12;
/// Fixed-point scale of a row's loss in the sum that decides early stopping.
const LOSS_SCALE: f64 = (1u64 << 30) as f64;

/// The per-feature constraints of `train.gbdt.trees.monotone`, resolved against the
/// fitted transform: a numeric column becomes the sign of its feature. Empty without
/// constraints. A categorical column, or one the model does not use, is refused.
pub fn monotone_constraints(cfg: &CandidateTrainConfig, transform: &FittedTransform) -> Result<Vec<i8>> {
    let Some(p) = &cfg.trees else { return Ok(Vec::new()) };
    if p.monotone.is_empty() {
        return Ok(Vec::new());
    }
    let mut out = vec![0i8; transform.output_dim];
    for (column, m) in &p.monotone {
        let refuse = |why: &str, next: &str| {
            TabsmithError::new(
                ErrorCode::ConfigInvalid,
                Phase::Train,
                format!("monotone constraint on `{column}`: {why}"),
            )
            .field(format!("train.gbdt.trees.monotone.{column}"))
            .next(next)
        };
        match transform.features.iter().find(|f| f.name() == column) {
            Some(FeatureTransform::Numeric { offset, .. }) => out[*offset] = m.sign(),
            Some(_) => {
                return Err(refuse(
                    "the column is categorical, and a constraint needs an order",
                    "constrain numeric columns only",
                ));
            }
            None => {
                return Err(refuse(
                    "the model has no such feature",
                    "check the spelling; the column may be the target, an id, excluded, or dropped when the transform was fitted",
                ));
            }
        }
    }
    Ok(out)
}

/// A tree node, numbered from the tree's root: what the workers need to apply a tree.
#[derive(Debug, Clone, Copy)]
struct Node {
    feature: i32,
    threshold: f32,
    left: u32,
    right: u32,
    value: f32,
}

/// The value of the leaf `row` reaches.
#[inline]
fn walk(nodes: &[Node], row: &[f32]) -> f32 {
    let mut i = 0;
    loop {
        let node = &nodes[i];
        if node.feature < 0 {
            return node.value;
        }
        i = if row[node.feature as usize] <= node.threshold {
            node.left
        } else {
            node.right
        } as usize;
    }
}

/// The trees grown so far, as the runtime stores them.
#[derive(Debug, Clone, Default)]
struct Forest {
    roots: Vec<u32>,
    feature: Vec<i32>,
    threshold: Vec<f32>,
    left: Vec<u32>,
    right: Vec<u32>,
    value: Vec<f32>,
    max_depth: usize,
    /// One sign per feature, or empty: recorded in the ensemble's config.
    monotone: Vec<i8>,
}

impl Forest {
    /// The first `n_trees` trees as a runtime ensemble; `base` holds one starting
    /// score per output.
    fn ensemble(&self, input_dim: usize, task: Task, base: &[f64], n_trees: usize) -> Result<TreeEnsemble> {
        let end = self.roots.get(n_trees).map_or(self.feature.len(), |&r| r as usize);
        let k = base.len();
        let config = ModelConfig::trees_for(
            input_dim,
            task,
            Some(TreesSpec {
                n_trees,
                n_nodes: end,
                max_depth: self.max_depth,
                base_score: if k == 1 { base[0] } else { 0.0 },
                n_outputs: k,
                base_scores: if k == 1 { Vec::new() } else { base.to_vec() },
                monotone: self.monotone.clone(),
            }),
        );
        TreeEnsemble::new(
            config,
            self.roots[..n_trees].to_vec(),
            self.feature[..end].to_vec(),
            self.threshold[..end].to_vec(),
            self.left[..end].to_vec(),
            self.right[..end].to_vec(),
            self.value[..end].to_vec(),
        )
        .map_err(|e| e.in_phase(Phase::Train))
    }

    fn from_ensemble(e: &TreeEnsemble) -> Self {
        Self {
            roots: e.roots.clone(),
            feature: e.feature.clone(),
            threshold: e.threshold.clone(),
            left: e.left.clone(),
            right: e.right.clone(),
            value: e.value.clone(),
            max_depth: e.config.trees.as_ref().map_or(0, |t| t.max_depth),
            monotone: e.config.trees.as_ref().map_or(Vec::new(), |t| t.monotone.clone()),
        }
    }

    /// Tree `t`'s nodes, numbered from its root.
    fn nodes(&self, t: usize) -> Vec<Node> {
        let start = self.roots[t] as usize;
        let end = self.roots.get(t + 1).map_or(self.feature.len(), |&r| r as usize);
        (start..end)
            .map(|i| Node {
                feature: self.feature[i],
                threshold: self.threshold[i],
                left: self.left[i].wrapping_sub(start as u32),
                right: self.right[i].wrapping_sub(start as u32),
                value: self.value[i],
            })
            .collect()
    }
}

fn save_checkpoint(root: &Path, key: &ResumeKey, state: &TrainState, ensemble: Option<TreeEnsemble>) -> Result<PathBuf> {
    let (tmp, final_dir) = checkpoint_dirs(root, state.step)?;
    if let Some(e) = ensemble {
        write_json(&tmp.join("trees.config.json"), &e.config)?;
        write_atomic(&tmp.join("trees.safetensors"), &Model::Trees(e).to_safetensors()?)?;
    }
    write_json(&tmp.join("state.json"), state)?;
    write_json(&tmp.join("key.json"), key)?;
    seal_checkpoint(
        root,
        &tmp,
        &final_dir,
        &["trees.config.json", "trees.safetensors", "state.json", "key.json"],
    )
}

fn load_checkpoint(dir: &Path, key: &ResumeKey, k: usize) -> Result<(TrainState, Forest)> {
    verify_checkpoint(dir)?;
    let saved: ResumeKey = read_json(&dir.join("key.json"), Phase::Resume)?;
    let diffs = saved.differences(key);
    if !diffs.is_empty() {
        return Err(TabsmithError::new(
            ErrorCode::ResumeIncompatible,
            Phase::Resume,
            format!("cannot continue exactly (R2): {}", diffs.join("; ")),
        )
        .field(dir.display().to_string())
        .next("start a new run (`tabsmith run`)"));
    }
    let state: TrainState = read_json(&dir.join("state.json"), Phase::Resume)?;
    let forest = match fs::read(dir.join("trees.safetensors")) {
        Ok(bytes) => {
            let config: ModelConfig = read_json(&dir.join("trees.config.json"), Phase::Resume)?;
            match Model::from_safetensors(config, &bytes).map_err(|e| e.in_phase(Phase::Resume))? {
                Model::Trees(e) => Forest::from_ensemble(&e),
                Model::Dense(_) => return Err(TabsmithError::internal(Phase::Resume, "a gbdt checkpoint holds a dense model")),
            }
        }
        Err(_) => Forest::default(),
    };
    if forest.roots.len() != state.epoch * k {
        return Err(TabsmithError::new(
            ErrorCode::CheckpointCorrupt,
            Phase::Resume,
            format!("checkpoint holds {} trees for {} rounds", forest.roots.len(), state.epoch),
        )
        .field(dir.display().to_string()));
    }
    Ok((state, forest))
}

/// Starting score of each output: the class-weighted log-odds of the positive
/// label (binary), the log of each class's weighted share (multiclass), 0 for the
/// standardised target (regression). The logarithm comes from `libm`, like every
/// transcendental function the trees' training uses, so that the same data gives the
/// same trees on every operating system (the platform C libraries differ in the
/// last bit).
fn base_scores(t: &Targets, weights: &[f32]) -> Vec<f64> {
    match t {
        Targets::Binary(y) => {
            let (wp, wn) = y.iter().fold((0.0, 0.0), |(a, b), &v| {
                let w = f64::from(weights[usize::from(v)]);
                if v { (a + w, b) } else { (a, b + w) }
            });
            vec![libm::log(wp / wn)]
        }
        Targets::Classes { y, k } => {
            let mut w = vec![0.0; *k];
            for &c in y {
                w[c as usize] += f64::from(weights[c as usize]);
            }
            let total: f64 = w.iter().sum();
            w.iter().map(|&c| libm::log(c.max(f64::MIN_POSITIVE) / total)).collect()
        }
        Targets::Values(_) => vec![0.0],
    }
}

/// An f64 that the items of a section read and write through a shared reference; see [`Slot`].
struct SharedF64(AtomicU64);

impl SharedF64 {
    fn new(v: f64) -> Self {
        Self(AtomicU64::new(v.to_bits()))
    }

    #[inline(always)]
    fn get(&self) -> f64 {
        f64::from_bits(self.0.load(Relaxed))
    }

    #[inline(always)]
    fn set(&self, v: f64) {
        self.0.store(v.to_bits(), Relaxed);
    }

    #[inline(always)]
    fn add(&self, v: f64) {
        self.set(self.get() + v);
    }
}

fn shared_f64s(rows: usize, v: f64) -> Vec<SharedF64> {
    (0..rows).map(|_| SharedF64::new(v)).collect()
}

/// The scores of the rows, one array per output, and what a round computes from them.
struct Boosting<'a> {
    task: Task,
    k: usize,
    x: &'a Matrix,
    x_valid: &'a Matrix,
    /// Encoded target and weight of every training row.
    y: &'a [f32],
    w: &'a [f32],
    largest_weight: f64,
    y_train: &'a Targets,
    y_valid: &'a Targets,
    schema: &'a OutputSchema,
    weights: &'a [f32],
    scores: Vec<Vec<SharedF64>>,
    valid_scores: Vec<Vec<SharedF64>>,
    /// Probabilities of the training rows under the current scores: one array for a binary task,
    /// one per class for a multiclass task, none for a regression.
    probs: Vec<Vec<SharedF64>>,
    /// The largest |score − target| over the training rows, for a regression.
    largest_residual: Mutex<f64>,
    valid_loss_sum: Mutex<i128>,
    valid_loss_broken: AtomicBool,
}

impl<'a> Boosting<'a> {
    /// Probabilities from the current scores, once per round: the gradients of all the round's
    /// trees come from them, and their exponentials are the slow part of a gradient.
    fn link_pass<'env>(&'env self, pool: &Pool<'env>)
    where
        'a: 'env,
    {
        let n = self.x.rows;
        match self.task {
            Task::Binary => section(pool, n, n.div_ceil(PIECE_ROWS), move |piece| {
                for r in piece_range(piece, n) {
                    self.probs[0][r].set(sigmoid(self.scores[0][r].get()));
                }
            }),
            Task::Multiclass => {
                let k = self.k;
                section(pool, n * k, n.div_ceil(PIECE_ROWS), move |piece| {
                    let (mut z, mut p) = (vec![0f32; k], vec![0f64; k]);
                    for r in piece_range(piece, n) {
                        for c in 0..k {
                            z[c] = self.scores[c][r].get() as f32;
                        }
                        kernels::softmax(&z, &mut p);
                        for c in 0..k {
                            self.probs[c][r].set(p[c]);
                        }
                    }
                });
            }
            Task::Regression => {
                *lock(&self.largest_residual) = 0.0;
                section(pool, n, n.div_ceil(PIECE_ROWS), move |piece| {
                    let mut largest = 0f64;
                    for r in piece_range(piece, n) {
                        largest = largest.max((self.scores[0][r].get() - f64::from(self.y[r])).abs());
                    }
                    let mut shared = lock(&self.largest_residual);
                    *shared = shared.max(largest);
                });
            }
        }
    }

    /// The gradients of output `c` as integers in `rows.gh`, and their sums. The bound of the
    /// gradients and hessians gives the scale, before any row is converted.
    fn gradients<'env>(&'env self, pool: &Pool<'env>, rows: &'env Rows, c: usize) -> Units
    where
        'a: 'env,
    {
        let n = self.x.rows;
        let (bound_g, bound_h) = match self.task {
            Task::Binary | Task::Multiclass => (self.largest_weight, self.largest_weight * 0.25),
            Task::Regression => (self.largest_weight * *lock(&self.largest_residual), self.largest_weight),
        };
        let units = Units {
            g: exponent_for(bound_g),
            h: exponent_for(bound_h),
        };
        let (scale_g, scale_h) = (two_pow(units.g), two_pow(units.h));
        rows.total_g.store(0, Relaxed);
        rows.total_h.store(0, Relaxed);
        section(pool, n, n.div_ceil(PIECE_ROWS), move |piece| {
            let (mut total_g, mut total_h) = (0i64, 0i64);
            for r in piece_range(piece, n) {
                let w = f64::from(self.w[r]);
                let (g, h) = match self.task {
                    Task::Binary => {
                        let prob = self.probs[0][r].get();
                        (w * (prob - f64::from(self.y[r])), w * prob * (1.0 - prob))
                    }
                    Task::Multiclass => {
                        let prob = self.probs[c][r].get();
                        let t = if self.y[r] as usize == c { 1.0 } else { 0.0 };
                        (w * (prob - t), w * prob * (1.0 - prob))
                    }
                    Task::Regression => (w * (self.scores[0][r].get() - f64::from(self.y[r])), w),
                };
                let (gi, hi) = (nearest(g * scale_g), nearest(h * scale_h).max(1));
                rows.gh[r].store(pack(gi, hi), Relaxed);
                total_g += i64::from(gi);
                total_h += i64::from(hi);
            }
            rows.total_g.fetch_add(total_g, Relaxed);
            rows.total_h.fetch_add(total_h, Relaxed);
        });
        units
    }

    /// Adds a finished tree's leaves to output `c` of the training rows, whose order holds them.
    fn add_leaves<'env>(&'env self, pool: &Pool<'env>, rows: &'env Rows, c: usize, leaves: Vec<FinishedLeaf>)
    where
        'a: 'env,
    {
        let n = self.x.rows;
        // A large leaf is several items, so that it does not stay on one thread.
        let mut items = Vec::with_capacity(leaves.len());
        for leaf in leaves {
            let mut start = leaf.start;
            let end = leaf.start + leaf.len;
            while start < end {
                let stop = (start + 4 * PIECE_ROWS).min(end);
                items.push(FinishedLeaf {
                    start,
                    len: stop - start,
                    value: leaf.value,
                });
                start = stop;
            }
        }
        let count = items.len();
        section(pool, n, count, move |i| {
            let FinishedLeaf { start, len, value } = items[i];
            let scores = &self.scores[c];
            for at in &rows.order[start..start + len] {
                scores[at.load(Relaxed) as usize].add(f64::from(value));
            }
        });
    }

    /// Adds a tree to output `c` of the validation rows, as the runtime walks it.
    fn add_tree_to_valid<'env>(&'env self, pool: &Pool<'env>, nodes: Vec<Node>, c: usize)
    where
        'a: 'env,
    {
        let n = self.x_valid.rows;
        section(pool, n, n.div_ceil(PIECE_ROWS), move |piece| {
            for r in piece_range(piece, n) {
                self.valid_scores[c][r].add(f64::from(walk(&nodes, self.x_valid.row(r))));
            }
        });
    }

    /// Adds a tree to output `c` of the training rows by walking it: for a resumed run, the same
    /// f64 additions as when the tree was grown.
    fn add_tree_to_train<'env>(&'env self, pool: &Pool<'env>, nodes: Vec<Node>, c: usize)
    where
        'a: 'env,
    {
        let n = self.x.rows;
        section(pool, n, n.div_ceil(PIECE_ROWS), move |piece| {
            for r in piece_range(piece, n) {
                self.scores[c][r].add(f64::from(walk(&nodes, self.x.row(r))));
            }
        });
    }

    /// The validation outputs as the runtime computes them.
    fn valid_outputs(&self) -> Matrix {
        let rows = self.x_valid.rows;
        let mut data = vec![0f32; rows * self.k];
        for (c, scores) in self.valid_scores.iter().enumerate() {
            for r in 0..rows {
                data[r * self.k + c] = scores[r].get() as f32;
            }
        }
        Matrix { rows, cols: self.k, data }
    }

    /// The mean loss of the validation outputs, which decides early stopping. Each row's loss goes
    /// through fixed point, so the sum is exact and has no order.
    fn valid_loss<'env>(&'env self, pool: &Pool<'env>, outputs: Arc<Matrix>) -> f64
    where
        'a: 'env,
    {
        let n = outputs.rows;
        *lock(&self.valid_loss_sum) = 0;
        self.valid_loss_broken.store(false, Relaxed);
        let (y, schema, weights) = (self.y_valid, self.schema, self.weights);
        section(pool, n * self.k, n.div_ceil(PIECE_ROWS), move |piece| {
            let mut sum = 0i128;
            for r in piece_range(piece, n) {
                let loss = row_loss(y, schema, weights, r, outputs.row(r));
                if !loss.is_finite() {
                    self.valid_loss_broken.store(true, Relaxed);
                }
                let scaled = loss * LOSS_SCALE;
                sum += i128::from((scaled + 0.5f64.copysign(scaled)) as i64);
            }
            *lock(&self.valid_loss_sum) += sum;
        });
        if self.valid_loss_broken.load(Relaxed) {
            return f64::NAN;
        }
        *lock(&self.valid_loss_sum) as f64 / LOSS_SCALE / n.max(1) as f64
    }

    /// The training loss of a round, on the same few rows every round: a learning curve needs no
    /// more, and on all the rows it would cost as much as the gradients.
    fn train_loss(&self) -> f64 {
        let n = self.x.rows;
        let stride = n.div_ceil(TRAIN_LOSS_ROWS).max(1);
        let mut z = vec![0f32; self.k];
        let (mut sum, mut count) = (0.0, 0usize);
        for r in (0..n).step_by(stride) {
            for c in 0..self.k {
                z[c] = self.scores[c][r].get() as f32;
            }
            sum += row_loss(self.y_train, self.schema, self.weights, r, &z);
            count += 1;
        }
        sum / count.max(1) as f64
    }
}

/// How the boosting loop ended.
enum Ended {
    Cancelled(PathBuf),
    /// The validation outputs of the best round.
    Finished(Arc<Matrix>),
}

/// Trains (or continues) the `gbdt` candidate. `on_epoch` runs after each round.
pub fn train_gbdt(
    inputs: &TrainInputs<'_>,
    control: &RunControl,
    start: StartFrom,
    on_epoch: &mut dyn FnMut(&TrainState),
) -> Result<TrainOutcome> {
    let cfg = inputs.cfg;
    let p = cfg.trees.clone().unwrap_or_default();
    let task = inputs.output.task;
    let (x, x_valid) = (inputs.x_train, inputs.x_valid);
    let (y_train, y_valid, output) = (inputs.y_train, inputs.y_valid, inputs.output);
    let metric_name = inputs.metric.as_str();
    let (n, d) = (x.rows, x.cols);
    let (l2, shrinkage) = (cfg.l2, cfg.learning_rate);
    let weights = class_weights_of(y_train, cfg.class_weight);
    let (enc_y, enc_w) = encode_targets(y_train, output, &weights);
    let base = base_scores(y_train, &weights);
    let k = base.len();

    let monotone = inputs.monotone.as_slice();
    let (mut state, mut forest) = match start {
        StartFrom::Scratch => (TrainState::fresh(), Forest::default()),
        StartFrom::Checkpoint(dir) => load_checkpoint(&dir, &inputs.key, k)?,
        StartFrom::WarmStart(_) => {
            return Err(TabsmithError::new(
                ErrorCode::ResumeIncompatible,
                Phase::Resume,
                "gbdt runs cannot warm-start from network weights",
            )
            .next("continue a gbdt run exactly with `tabsmith resume <run>`, or train a new one"));
        }
    };
    // The resume key covers the configuration, so a checkpoint's constraints are these.
    forest.monotone = monotone.to_vec();
    let rows = Rows::new(install(|| Bins::new(x, p.max_bins, true)), p.max_leaves);
    let boosting = Boosting {
        task,
        k,
        x,
        x_valid,
        y: &enc_y,
        w: &enc_w,
        largest_weight: enc_w.iter().copied().fold(0f32, f32::max).into(),
        y_train,
        y_valid,
        schema: output,
        weights: &weights,
        scores: base.iter().map(|&b| shared_f64s(n, b)).collect(),
        valid_scores: base.iter().map(|&b| shared_f64s(x_valid.rows, b)).collect(),
        probs: (0..if task == Task::Regression { 0 } else { k })
            .map(|_| shared_f64s(n, 0.0))
            .collect(),
        largest_residual: Mutex::new(0.0),
        valid_loss_sum: Mutex::new(0),
        valid_loss_broken: AtomicBool::new(false),
    };
    let started = Instant::now();
    let elapsed_before = state.elapsed_ms;

    let ended = with_pool(threads(), |pool| -> Result<Ended> {
        // Scores are recomputed from the trees in order: the same f64 additions as when the trees
        // were grown, so a resumed run continues bit for bit.
        for t in 0..forest.roots.len() {
            boosting.add_tree_to_train(pool, forest.nodes(t), t % k);
            boosting.add_tree_to_valid(pool, forest.nodes(t), t % k);
        }
        let mut best_valid = Arc::new(boosting.valid_outputs());
        // Counted from the start: a run shorter than the interval checkpoints only at its end.
        let mut last_checkpoint = started;
        let mut free = FreeHists::default();
        let ensemble_now = |forest: &Forest| -> Result<Option<TreeEnsemble>> {
            if forest.roots.is_empty() {
                Ok(None)
            } else {
                forest.ensemble(d, task, &base, forest.roots.len()).map(Some)
            }
        };
        boosting.link_pass(pool);
        while state.epoch < cfg.epochs && state.stop_reason.is_none() {
            if control.cancel.load(Ordering::SeqCst) {
                state.elapsed_ms = elapsed_before + started.elapsed().as_secs_f64() * 1e3;
                let ck = save_checkpoint(&inputs.checkpoint_dir, &inputs.key, &state, ensemble_now(&forest)?)?;
                return Ok(Ended::Cancelled(ck));
            }
            // One tree per output, all fitted to the gradients at the start of the round.
            for c in 0..k {
                let units = boosting.gradients(pool, &rows, c);
                let growth = Growth::new(&p, monotone, l2, shrinkage, units);
                let leaves = grow_tree(pool, &rows, growth, &mut forest, &mut free);
                boosting.add_leaves(pool, &rows, c, leaves);
                boosting.add_tree_to_valid(pool, forest.nodes(forest.roots.len() - 1), c);
            }
            boosting.link_pass(pool);
            let train_loss = boosting.train_loss();
            let v_out = Arc::new(boosting.valid_outputs());
            let valid_loss = boosting.valid_loss(pool, v_out.clone());
            let metric = selected_metric(&v_out, y_valid, output, metric_name).unwrap_or(0.0);
            state.step += 1;
            state.epoch += 1;

            let residual_ok = task != Task::Regression || lock(&boosting.largest_residual).is_finite();
            if !(train_loss.is_finite() && valid_loss.is_finite() && residual_ok) {
                return Err(TabsmithError::new(
                    ErrorCode::TrainingDiverged,
                    Phase::Train,
                    format!("loss became non-finite after round {}", state.epoch),
                )
                .next("lower train.gbdt.learning_rate or raise train.gbdt.l2, and run again"));
            }
            state.history.push(EpochRecord {
                epoch: state.epoch,
                step: state.step,
                train_loss,
                valid_loss,
                valid_metric: metric,
            });
            if state.best_valid_loss.is_none_or(|b| valid_loss < b) {
                state.best_valid_loss = Some(valid_loss);
                state.best_epoch = state.epoch;
                state.bad_epochs = 0;
                best_valid = v_out;
            } else {
                state.bad_epochs += 1;
            }
            state.elapsed_ms = elapsed_before + started.elapsed().as_secs_f64() * 1e3;
            if cfg.early_stopping_patience > 0 && state.bad_epochs >= cfg.early_stopping_patience {
                state.stop_reason = Some("early_stopping".into());
            } else if inputs.max_train_seconds > 0 && state.elapsed_ms >= inputs.max_train_seconds as f64 * 1e3 {
                state.stop_reason = Some("time_budget".into());
            } else if state.epoch >= cfg.epochs {
                state.stop_reason = Some("epochs".into());
            }
            let due = state.epoch % cfg.checkpoint_every_epochs == 0
                && last_checkpoint.elapsed().as_secs_f64() >= cfg.checkpoint_min_interval_seconds;
            if due || state.stop_reason.is_some() {
                save_checkpoint(&inputs.checkpoint_dir, &inputs.key, &state, ensemble_now(&forest)?)?;
                last_checkpoint = Instant::now();
            }
            on_epoch(&state);
            if control.cancel_after_steps == Some(state.step) {
                control.cancel.store(true, Ordering::SeqCst);
            }
            if control.crash_after_steps == Some(state.step) {
                eprintln!("[test hook] simulating a crash after step {}", state.step);
                std::process::exit(137);
            }
        }
        Ok(Ended::Finished(best_valid))
    })?;
    let best_valid = match ended {
        Ended::Cancelled(checkpoint) => return Ok(TrainOutcome::Cancelled { state, checkpoint }),
        Ended::Finished(best_valid) => best_valid,
    };
    let best = forest.ensemble(d, task, &base, state.best_epoch.max(1) * k)?;
    // The trainer's scores and the runtime's evaluation of the kept trees (exactly 0
    // by construction; recorded like the networks' engine/runtime parity).
    let runtime = Model::Trees(best);
    let parity = par_outputs(x_valid, |x| runtime.outputs(x))
        .data
        .iter()
        .zip(&best_valid.data)
        .map(|(a, b)| f64::from((a - b).abs()))
        .fold(0.0, f64::max);
    Ok(TrainOutcome::Completed {
        best: runtime,
        state,
        parity_max_abs_diff: parity,
    })
}
