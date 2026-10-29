//! The `gbdt` candidate: gradient-boosted decision trees on the fitted feature
//! matrix (the same transform as the networks). Split points come from the
//! training rows, up to `max_bins` per feature; trees grow best-first, the leaf
//! with the largest gain splitting next. The loss is the task's: class-weighted
//! log loss (binary), class-weighted softmax cross-entropy with one tree per class
//! and round (multiclass), squared error of the standardised target (regression);
//! early stopping watches the validation loss.
//!
//! Deterministic by construction: every histogram sum runs in row order inside fixed
//! chunks of `CHUNK` rows and the chunk sums are merged in chunk order, so the
//! result does not depend on how many threads built the chunks; ties between splits
//! are broken by (feature, bin) and between leaves by creation order, and leaf values
//! are rounded to f32 before they move the scores. The trainer's validation scores
//! are therefore exactly what the runtime computes for the same trees. Binning,
//! histograms and the application of a grown tree use every core; a node with at
//! most one chunk of rows is summed exactly as on one thread.
//!
//! One boosting round is one "epoch" of the shared [`TrainState`], so checkpoints,
//! cancellation, crash recovery and exact resume (R2) work as for the networks.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::time::Instant;

use tabsmith_contracts::config::{CandidateTrainConfig, Task, TreeParams};
use tabsmith_contracts::model::{ModelConfig, TreesSpec};
use tabsmith_contracts::run::EpochRecord;
use tabsmith_contracts::transform::{FeatureTransform, FittedTransform};
use tabsmith_contracts::{ErrorCode, Phase, Result, TabsmithError};
use tabsmith_runtime::Matrix;
use tabsmith_runtime::model::{Model, TreeEnsemble, sigmoid, softmax};

use crate::data::Targets;
use crate::metrics::task_metrics;
use crate::project::{read_json, write_atomic, write_json};
use crate::train::{
    ResumeKey, RunControl, StartFrom, TrainInputs, TrainOutcome, TrainState, checkpoint_dirs, class_weights_of, encode_targets,
    par_outputs, seal_checkpoint, task_loss, verify_checkpoint,
};

/// Histogram slots per feature (bins are `u8`).
const SLOTS: usize = 256;
/// Smallest gain worth a split; below it the split only fits rounding noise.
const MIN_GAIN: f64 = 1e-9;
/// Rows per histogram chunk. The chunking is a function of the row count alone, so
/// the sums are the same on one core and on thirty-two; a node with at most this many
/// rows is summed in plain row order.
const CHUNK: usize = 16_384;

/// The threads the trainer may use for the parts that parallelise.
pub(crate) fn threads() -> usize {
    std::thread::available_parallelism().map_or(1, |n| n.get()).max(1)
}

/// Runs `f(i)` for `i < n` on every core (worker `w` takes `w, w + workers, …`) and
/// returns the results in order `0..n`. Each `f(i)` is a pure function of `i`, so the
/// result does not depend on the number of workers.
fn par_map<T: Send>(n: usize, f: impl Fn(usize) -> T + Sync) -> Vec<T> {
    let workers = threads().min(n);
    if workers <= 1 {
        return (0..n).map(f).collect();
    }
    let f = &f;
    let parts: Vec<Vec<(usize, T)>> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..workers)
            .map(|w| s.spawn(move || (w..n).step_by(workers).map(|i| (i, f(i))).collect::<Vec<_>>()))
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().unwrap_or_else(|panic| std::panic::resume_unwind(panic)))
            .collect()
    });
    let mut slots: Vec<Option<T>> = (0..n).map(|_| None).collect();
    for (i, v) in parts.into_iter().flatten() {
        slots[i] = Some(v);
    }
    slots.into_iter().map(|v| v.expect("every index was computed")).collect()
}

/// Candidate split points of every feature, ascending. Bin `k` of a feature holds
/// the values `<= thresholds[k]` (above the previous threshold); the last bin the rest.
struct Bins {
    thresholds: Vec<Vec<f32>>,
    /// Bin of every training row and feature, row-major.
    codes: Vec<u8>,
    width: usize,
}

impl Bins {
    fn new(x: &Matrix, max_bins: usize) -> Self {
        let (n, d) = (x.rows, x.cols);
        // Every feature's thresholds depend on that feature's values alone.
        let thresholds: Vec<Vec<f32>> = par_map(d, |j| {
            let mut v: Vec<f32> = (0..n).map(|r| x.data[r * d + j]).collect();
            v.sort_by(f32::total_cmp);
            let mut unique = v.clone();
            unique.dedup_by(|a, b| a == b);
            let mut t: Vec<f32> = if unique.len() <= max_bins {
                unique
            } else {
                // Quantiles of the training values: each bin holds about n / max_bins rows.
                let mut q: Vec<f32> = (1..max_bins).map(|i| v[(i * n / max_bins).min(n - 1)]).collect();
                q.dedup_by(|a, b| a == b);
                q
            };
            // A split at the largest value would leave its right side empty.
            if let Some(&max) = v.last() {
                t.retain(|&x| x < max);
            }
            t
        });
        let code_chunks = par_map(n.div_ceil(CHUNK).max(1), |c| {
            let rows = c * CHUNK..((c + 1) * CHUNK).min(n);
            let mut codes = Vec::with_capacity(rows.len() * d);
            for r in rows {
                for j in 0..d {
                    let x = x.data[r * d + j];
                    codes.push(thresholds[j].partition_point(|&t| t < x) as u8);
                }
            }
            codes
        });
        Self {
            thresholds,
            codes: code_chunks.concat(),
            width: d,
        }
    }
}

struct Histogram {
    g: Vec<f64>,
    h: Vec<f64>,
    n: Vec<u32>,
}

impl Histogram {
    fn zero(d: usize) -> Self {
        Histogram {
            g: vec![0.0; d * SLOTS],
            h: vec![0.0; d * SLOTS],
            n: vec![0; d * SLOTS],
        }
    }

    /// Gradient and hessian sums per (feature, bin) over `rows`: in row order within
    /// each chunk of [`CHUNK`] rows, the chunks summed in order. One chunk (at most
    /// `CHUNK` rows) is plain row order.
    fn of(bins: &Bins, rows: &[u32], grad: &[f64], hess: &[f64]) -> Self {
        let d = bins.width;
        let chunk = |rows: &[u32]| {
            let mut hist = Histogram::zero(d);
            for &r in rows {
                let r = r as usize;
                for (j, &code) in bins.codes[r * d..(r + 1) * d].iter().enumerate() {
                    let k = j * SLOTS + code as usize;
                    hist.g[k] += grad[r];
                    hist.h[k] += hess[r];
                    hist.n[k] += 1;
                }
            }
            hist
        };
        if rows.len() <= CHUNK {
            return chunk(rows);
        }
        let parts = par_map(rows.len().div_ceil(CHUNK), |c| {
            chunk(&rows[c * CHUNK..((c + 1) * CHUNK).min(rows.len())])
        });
        let mut total = Histogram::zero(d);
        for part in &parts {
            for k in 0..d * SLOTS {
                total.g[k] += part.g[k];
                total.h[k] += part.h[k];
                total.n[k] += part.n[k];
            }
        }
        total
    }
}

#[derive(Debug, Clone, Copy)]
struct Split {
    feature: usize,
    bin: usize,
    gain: f64,
}

/// The objective reduction a leaf with sums (g, h) buys, and its weight: G²/(H+λ) at
/// the free optimum −G/(H+λ), or the reduction at the clipped weight when a monotone
/// constraint bounds the leaf to `[lower, upper]`. Without constraints the bounds are
/// infinite and the arithmetic is exactly the unconstrained formula.
fn score(g: f64, h: f64, l2: f64, lower: f64, upper: f64) -> (f64, f64) {
    let w = -g / (h + l2);
    if w >= lower && w <= upper {
        (g * g / (h + l2), w)
    } else {
        let w = if w < lower { lower } else { upper };
        (-(2.0 * g * w + (h + l2) * w * w), w)
    }
}

/// The best split of a node with sums (g, h) over n rows and leaf bounds
/// `[lower, upper]`: the largest gain score(L) + score(R) − score(parent), first in
/// (feature, bin) order on ties. A split on a constrained feature (`monotone[j]` is
/// 1 or −1) is allowed only when the lesser side's weight is at most the greater side's.
#[allow(clippy::too_many_arguments)]
fn best_split(
    bins: &Bins,
    hist: &Histogram,
    g: f64,
    h: f64,
    n: usize,
    p: &TreeParams,
    l2: f64,
    monotone: &[i8],
    lower: f64,
    upper: f64,
) -> Option<Split> {
    let (parent, _) = score(g, h, l2, lower, upper);
    let mut best: Option<Split> = None;
    for (j, thresholds) in bins.thresholds.iter().enumerate() {
        let c = monotone.get(j).copied().unwrap_or(0);
        let (mut gl, mut hl, mut nl) = (0.0, 0.0, 0usize);
        for k in 0..thresholds.len() {
            let idx = j * SLOTS + k;
            gl += hist.g[idx];
            hl += hist.h[idx];
            nl += hist.n[idx] as usize;
            if nl < p.min_samples_leaf {
                continue;
            }
            if n - nl < p.min_samples_leaf {
                break;
            }
            let (gr, hr) = (g - gl, h - hl);
            let ((sl, wl), (sr, wr)) = (score(gl, hl, l2, lower, upper), score(gr, hr, l2, lower, upper));
            if (c > 0 && wl > wr) || (c < 0 && wl < wr) {
                continue;
            }
            let gain = sl + sr - parent;
            if gain > best.map_or(MIN_GAIN, |b| b.gain) {
                best = Some(Split { feature: j, bin: k, gain });
            }
        }
    }
    best
}

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

    /// Adds tree `t` to output `t % k` of the scores (`rows x k`) of the rows of `x`.
    /// Every row's walk is independent, so the rows are split between the cores.
    fn add_tree(&self, t: usize, x: &Matrix, scores: &mut [f64], k: usize) {
        let walk = |row: &[f32]| {
            let mut i = self.roots[t] as usize;
            while self.feature[i] >= 0 {
                i = if row[self.feature[i] as usize] <= self.threshold[i] {
                    self.left[i]
                } else {
                    self.right[i]
                } as usize;
            }
            f64::from(self.value[i])
        };
        let workers = threads().min(x.rows.div_ceil(CHUNK)).max(1);
        if workers <= 1 {
            for r in 0..x.rows {
                scores[r * k + t % k] += walk(x.row(r));
            }
            return;
        }
        let per = x.rows.div_ceil(workers);
        let walk = &walk;
        std::thread::scope(|s| {
            for (w, chunk) in scores.chunks_mut(per * k).enumerate() {
                let first = w * per;
                s.spawn(move || {
                    for (i, row_scores) in chunk.chunks_mut(k).enumerate() {
                        row_scores[t % k] += walk(x.row(first + i));
                    }
                });
            }
        });
    }
}

struct Leaf {
    rows: Vec<u32>,
    depth: usize,
    g: f64,
    h: f64,
    node: usize,
    split: Option<Split>,
    /// Bounds on the leaf's weight from the monotone splits above it; infinite
    /// without constraints.
    lower: f64,
    upper: f64,
}

/// Grows one tree best-first over `n` training rows and adds it to `forest`.
/// Returns the (f32-rounded) value of the leaf each training row lands in.
///
/// Monotone constraints (`monotone[j]` is 1 or −1) are kept by construction, as in
/// XGBoost: a split on a constrained feature is taken only when the lesser side's
/// weight is at most the greater side's, and from then on the lesser subtree's leaves
/// are capped at the midpoint of the two weights while the greater subtree's leaves
/// are floored at it. So at every constrained split, every leaf on one side is at most
/// every leaf on the other, and the tree is monotone in the feature.
#[allow(clippy::too_many_arguments)]
fn grow_tree(
    bins: &Bins,
    grad: &[f64],
    hess: &[f64],
    p: &TreeParams,
    l2: f64,
    shrinkage: f64,
    monotone: &[i8],
    forest: &mut Forest,
) -> Vec<f64> {
    let d = bins.width;
    let n = grad.len();
    let base = forest.feature.len();
    let mut nodes: Vec<(i32, f32, u32, u32)> = vec![(-1, 0.0, 0, 0)];
    let sums = |rows: &[u32]| {
        rows.iter()
            .fold((0.0, 0.0), |(g, h), &r| (g + grad[r as usize], h + hess[r as usize]))
    };
    let leaf = |rows: Vec<u32>, depth: usize, node: usize, lower: f64, upper: f64| {
        let (g, h) = sums(&rows);
        let split = (depth < p.max_depth && rows.len() >= 2 * p.min_samples_leaf)
            .then(|| {
                let hist = Histogram::of(bins, &rows, grad, hess);
                best_split(bins, &hist, g, h, rows.len(), p, l2, monotone, lower, upper)
            })
            .flatten();
        Leaf {
            rows,
            depth,
            g,
            h,
            node,
            split,
            lower,
            upper,
        }
    };
    let mut leaves = vec![leaf((0..n as u32).collect(), 0, 0, f64::NEG_INFINITY, f64::INFINITY)];
    while leaves.len() < p.max_leaves {
        let mut pick: Option<usize> = None;
        for (i, l) in leaves.iter().enumerate() {
            let Some(s) = l.split else { continue };
            let better = pick.is_none_or(|b| {
                let (bs, bn) = (leaves[b].split.expect("picked").gain, leaves[b].node);
                s.gain > bs || (s.gain == bs && l.node < bn)
            });
            if better {
                pick = Some(i);
            }
        }
        let Some(i) = pick else { break };
        let parent = leaves.swap_remove(i);
        let s = parent.split.expect("picked");
        let (mut lrows, mut rrows) = (Vec::new(), Vec::new());
        for &r in &parent.rows {
            if (bins.codes[r as usize * d + s.feature] as usize) <= s.bin {
                lrows.push(r);
            } else {
                rrows.push(r);
            }
        }
        let (ln, rn) = (nodes.len(), nodes.len() + 1);
        nodes.push((-1, 0.0, 0, 0));
        nodes.push((-1, 0.0, 0, 0));
        nodes[parent.node] = (
            s.feature as i32,
            bins.thresholds[s.feature][s.bin],
            (base + ln) as u32,
            (base + rn) as u32,
        );
        let (pl, pu) = (parent.lower, parent.upper);
        let ((ll, lu), (rl, ru)) = match monotone.get(s.feature).copied().unwrap_or(0) {
            0 => ((pl, pu), (pl, pu)),
            c => {
                let ((gl, hl), (gr, hr)) = (sums(&lrows), sums(&rrows));
                let (wl, wr) = (score(gl, hl, l2, pl, pu).1, score(gr, hr, l2, pl, pu).1);
                let mid = (wl + wr) / 2.0;
                if c > 0 {
                    ((pl, pu.min(mid)), (pl.max(mid), pu))
                } else {
                    ((pl.max(mid), pu), (pl, pu.min(mid)))
                }
            }
        };
        leaves.push(leaf(lrows, parent.depth + 1, ln, ll, lu));
        leaves.push(leaf(rrows, parent.depth + 1, rn, rl, ru));
    }
    let mut values = vec![0f32; nodes.len()];
    let mut moved = vec![0.0; n];
    for l in &leaves {
        let (_, w) = score(l.g, l.h, l2, l.lower, l.upper);
        let v = (w * shrinkage) as f32;
        values[l.node] = v;
        for &r in &l.rows {
            moved[r as usize] = f64::from(v);
        }
        forest.max_depth = forest.max_depth.max(l.depth);
    }
    forest.roots.push(base as u32);
    for (i, (f, t, l, r)) in nodes.into_iter().enumerate() {
        forest.feature.push(f);
        forest.threshold.push(t);
        forest.left.push(l);
        forest.right.push(r);
        forest.value.push(values[i]);
    }
    moved
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
    let x = inputs.x_train;
    let (n, d) = (x.rows, x.cols);
    let l2 = cfg.l2;
    let weights = class_weights_of(inputs.y_train, cfg.class_weight);
    let (enc_y, enc_w) = encode_targets(inputs.y_train, inputs.output, &weights);
    let base = base_scores(inputs.y_train, &weights);
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
    let bins = Bins::new(x, p.max_bins);
    // Scores (`rows x k`) are recomputed from the trees in order: the same f64
    // additions as when the trees were grown, so a resumed run continues bit for bit.
    let start_scores = |rows: usize| (0..rows).flat_map(|_| base.iter().copied()).collect::<Vec<f64>>();
    let mut scores = start_scores(n);
    let mut valid_scores = start_scores(inputs.x_valid.rows);
    for t in 0..forest.roots.len() {
        forest.add_tree(t, x, &mut scores, k);
        forest.add_tree(t, inputs.x_valid, &mut valid_scores, k);
    }
    let as_outputs = |s: &[f64], rows: usize| Matrix {
        rows,
        cols: k,
        data: s.iter().map(|&v| v as f32).collect(),
    };
    let mut best_valid = as_outputs(&valid_scores, inputs.x_valid.rows).data;
    let (mut grad, mut hess) = (vec![0.0; n], vec![0.0; n]);
    let started = Instant::now();
    let mut last_checkpoint: Option<Instant> = None;
    let elapsed_before = state.elapsed_ms;
    let ensemble_now = |forest: &Forest| -> Result<Option<TreeEnsemble>> {
        if forest.roots.is_empty() {
            Ok(None)
        } else {
            forest.ensemble(d, task, &base, forest.roots.len()).map(Some)
        }
    };
    while state.epoch < cfg.epochs && state.stop_reason.is_none() {
        if control.cancel.load(Ordering::SeqCst) {
            state.elapsed_ms = elapsed_before + started.elapsed().as_secs_f64() * 1e3;
            let ck = save_checkpoint(&inputs.checkpoint_dir, &inputs.key, &state, ensemble_now(&forest)?)?;
            return Ok(TrainOutcome::Cancelled { state, checkpoint: ck });
        }
        // One tree per output, all fitted to the gradients at the start of the round.
        let probs: Vec<Vec<f64>> = match task {
            Task::Multiclass => (0..n)
                .map(|r| softmax(&scores[r * k..(r + 1) * k].iter().map(|&v| v as f32).collect::<Vec<f32>>()))
                .collect(),
            _ => Vec::new(),
        };
        for c in 0..k {
            for r in 0..n {
                let w = f64::from(enc_w[r]);
                let (g, h) = match task {
                    Task::Binary => {
                        let prob = sigmoid(scores[r]);
                        (w * (prob - f64::from(enc_y[r])), w * prob * (1.0 - prob))
                    }
                    Task::Multiclass => {
                        let pc = probs[r][c];
                        let t = if enc_y[r] as usize == c { 1.0 } else { 0.0 };
                        (w * (pc - t), w * pc * (1.0 - pc))
                    }
                    Task::Regression => (w * (scores[r] - f64::from(enc_y[r])), w),
                };
                grad[r] = g;
                hess[r] = h.max(1e-16);
            }
            let moved = grow_tree(&bins, &grad, &hess, &p, l2, cfg.learning_rate, monotone, &mut forest);
            for r in 0..n {
                scores[r * k + c] += moved[r];
            }
            forest.add_tree(forest.roots.len() - 1, inputs.x_valid, &mut valid_scores, k);
        }
        state.step += 1;
        state.epoch += 1;

        let train_out = as_outputs(&scores, n);
        let train_loss = task_loss(&train_out, inputs.y_train, inputs.output, &weights);
        let v_out = as_outputs(&valid_scores, inputs.x_valid.rows);
        let valid_loss = task_loss(&v_out, inputs.y_valid, inputs.output, &weights);
        if !(train_loss.is_finite() && valid_loss.is_finite()) {
            return Err(TabsmithError::new(
                ErrorCode::TrainingDiverged,
                Phase::Train,
                format!("loss became non-finite after round {}", state.epoch),
            )
            .next("lower train.gbdt.learning_rate or raise train.gbdt.l2, and run again"));
        }
        let metric = task_metrics(&v_out, inputs.y_valid, inputs.output)
            .get(inputs.metric.as_str())
            .unwrap_or(0.0);
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
            best_valid = v_out.data;
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
            && last_checkpoint.is_none_or(|t: Instant| t.elapsed().as_secs_f64() >= cfg.checkpoint_min_interval_seconds);
        if due || state.stop_reason.is_some() {
            save_checkpoint(&inputs.checkpoint_dir, &inputs.key, &state, ensemble_now(&forest)?)?;
            last_checkpoint = Some(Instant::now());
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
    let best = forest.ensemble(d, task, &base, state.best_epoch.max(1) * k)?;
    // The trainer's scores and the runtime's evaluation of the kept trees (exactly 0
    // by construction; recorded like the networks' engine/runtime parity).
    let runtime = Model::Trees(best);
    let parity = par_outputs(inputs.x_valid, |x| runtime.outputs(x))
        .data
        .iter()
        .zip(&best_valid)
        .map(|(a, b)| f64::from((a - b).abs()))
        .fold(0.0, f64::max);
    Ok(TrainOutcome::Completed {
        best: runtime,
        state,
        parity_max_abs_diff: parity,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunked_histograms_and_tree_application_are_the_fixed_order_sums() {
        // 40,000 rows of 3 features: three chunks, built on several cores, must equal
        // the chunk sums folded in order on one thread, bit for bit.
        let (n, d) = (40_000usize, 3usize);
        let mut rng = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            (rng >> 11) as f64 / (1u64 << 53) as f64
        };
        let x = Matrix {
            rows: n,
            cols: d,
            data: (0..n * d).map(|_| next() as f32).collect(),
        };
        let grad: Vec<f64> = (0..n).map(|_| next() - 0.5).collect();
        let hess: Vec<f64> = (0..n).map(|_| next() * 0.25).collect();
        let bins = Bins::new(&x, 64);
        let rows: Vec<u32> = (0..n as u32).collect();
        let parallel = Histogram::of(&bins, &rows, &grad, &hess);
        let mut reference = Histogram::zero(d);
        for chunk in rows.chunks(CHUNK) {
            let part = Histogram::of(&bins, chunk, &grad, &hess);
            for k in 0..d * SLOTS {
                reference.g[k] += part.g[k];
                reference.h[k] += part.h[k];
                reference.n[k] += part.n[k];
            }
        }
        assert!(parallel.g.iter().zip(&reference.g).all(|(a, b)| a.to_bits() == b.to_bits()));
        assert!(parallel.h.iter().zip(&reference.h).all(|(a, b)| a.to_bits() == b.to_bits()));
        assert_eq!(parallel.n, reference.n);
        // Binning in chunks equals binning row by row.
        for r in 0..n {
            for j in 0..d {
                let code = bins.thresholds[j].partition_point(|&t| t < x.data[r * d + j]) as u8;
                assert_eq!(bins.codes[r * d + j], code);
            }
        }
        // Applying a tree on several cores equals applying it row by row.
        let mut forest = Forest::default();
        grow_tree(&bins, &grad, &hess, &stump(), 1.0, 1.0, &[], &mut forest);
        let mut parallel = vec![0.0; n];
        forest.add_tree(0, &x, &mut parallel, 1);
        let e = forest.ensemble(d, Task::Regression, &[0.0], 1).unwrap();
        let serial = Model::Trees(e).outputs(&x).data;
        assert!(parallel.iter().zip(&serial).all(|(a, b)| (*a as f32).to_bits() == b.to_bits()));
    }

    #[test]
    fn thresholds_bin_values_like_the_runtime_compares_them() {
        let x = Matrix {
            rows: 6,
            cols: 1,
            data: vec![3.0, 1.0, 2.0, 2.0, 5.0, 1.0],
        };
        let bins = Bins::new(&x, 255);
        assert_eq!(bins.thresholds[0], vec![1.0, 2.0, 3.0]);
        // bin k <=> x <= thresholds[k]; 5.0 is above every threshold.
        assert_eq!(bins.codes, vec![2, 0, 1, 1, 3, 0]);
        let coarse = Bins::new(
            &Matrix {
                rows: 1000,
                cols: 1,
                data: (0..1000).map(|i| i as f32).collect(),
            },
            4,
        );
        assert_eq!(coarse.thresholds[0], vec![250.0, 500.0, 750.0]);
    }

    fn stump() -> TreeParams {
        TreeParams {
            max_leaves: 2,
            max_depth: 1,
            min_samples_leaf: 10,
            max_bins: 255,
            monotone: Default::default(),
        }
    }

    #[test]
    fn a_stump_learns_a_threshold_rule() {
        // y = x > 0.5; one split separates the classes.
        let n = 200;
        let x = Matrix {
            rows: n,
            cols: 2,
            data: (0..n).flat_map(|i| [i as f32 / n as f32, (i % 7) as f32]).collect(),
        };
        let y: Vec<bool> = (0..n).map(|i| i as f32 / n as f32 > 0.5).collect();
        let bins = Bins::new(&x, 255);
        let grad: Vec<f64> = y.iter().map(|&t| 0.5 - if t { 1.0 } else { 0.0 }).collect();
        let hess = vec![0.25; n];
        let mut forest = Forest::default();
        let moved = grow_tree(&bins, &grad, &hess, &stump(), 1.0, 1.0, &[], &mut forest);
        assert_eq!(forest.feature[0], 0, "splits on the informative feature");
        assert_eq!(forest.threshold[0], 0.5);
        assert!(moved.iter().zip(&y).all(|(&s, &t)| (s > 0.0) == t));
        let e = forest.ensemble(2, Task::Binary, &[0.0], 1).unwrap();
        let rt = Model::Trees(e).outputs(&x);
        assert!(
            rt.data.iter().zip(&moved).all(|(&a, &b)| a == b as f32),
            "runtime agrees with the trainer"
        );
    }

    #[test]
    fn a_regression_stump_moves_towards_each_side_mean() {
        // Standardised target -1 below x = 0.5 and +1 above; squared error: g = F - y, h = 1.
        let n = 100;
        let x = Matrix {
            rows: n,
            cols: 1,
            data: (0..n).map(|i| i as f32 / n as f32).collect(),
        };
        let y: Vec<f64> = (0..n).map(|i| if i < n / 2 { -1.0 } else { 1.0 }).collect();
        let grad: Vec<f64> = y.iter().map(|v| -v).collect();
        let mut forest = Forest::default();
        let moved = grow_tree(&Bins::new(&x, 255), &grad, &vec![1.0; n], &stump(), 0.0, 1.0, &[], &mut forest);
        assert!(moved.iter().zip(&y).all(|(&m, &t)| (m - t).abs() < 1e-6), "{moved:?}");
    }

    /// Every tree of `forest` keeps `monotone`: at each constrained split, every leaf
    /// of the lesser side is at most every leaf of the greater side.
    fn is_monotone(forest: &Forest, monotone: &[i8]) -> bool {
        let n = forest.feature.len();
        let (mut lo, mut hi) = (vec![0f32; n], vec![0f32; n]);
        for i in (0..n).rev() {
            if forest.feature[i] < 0 {
                (lo[i], hi[i]) = (forest.value[i], forest.value[i]);
                continue;
            }
            let (l, r) = (forest.left[i] as usize, forest.right[i] as usize);
            (lo[i], hi[i]) = (lo[l].min(lo[r]), hi[l].max(hi[r]));
            let broken = match monotone[forest.feature[i] as usize] {
                1 => hi[l] > lo[r],
                -1 => lo[l] < hi[r],
                _ => false,
            };
            if broken {
                return false;
            }
        }
        true
    }

    #[test]
    fn a_monotone_constraint_is_kept_even_when_the_signal_is_not() {
        // A bump: the target rises with x until 0.6 and falls after it, plus noise
        // from a second feature. Unconstrained trees follow the bump; constrained ones
        // may not decrease in x.
        let n = 600;
        let x = Matrix {
            rows: n,
            cols: 2,
            data: (0..n)
                .flat_map(|i| [i as f32 / n as f32, ((i * 7919) % 97) as f32 / 97.0])
                .collect(),
        };
        let target: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / n as f64;
                let noise = (((i * 7919) % 97) as f64 / 97.0 - 0.5) * 0.4;
                (if t < 0.6 { 2.0 * t } else { 2.4 - 2.0 * t }) + noise
            })
            .collect();
        let bins = Bins::new(&x, 255);
        let deep = TreeParams {
            max_leaves: 16,
            max_depth: 5,
            min_samples_leaf: 5,
            max_bins: 255,
            monotone: Default::default(),
        };
        let train = |monotone: &[i8]| {
            let mut forest = Forest::default();
            let mut scores = vec![0.0; n];
            for _ in 0..30 {
                let grad: Vec<f64> = scores.iter().zip(&target).map(|(s, y)| s - y).collect();
                let moved = grow_tree(&bins, &grad, &vec![1.0; n], &deep, 0.0, 0.3, monotone, &mut forest);
                for r in 0..n {
                    scores[r] += moved[r];
                }
            }
            (forest, scores)
        };
        let (free, free_scores) = train(&[]);
        assert!(!is_monotone(&free, &[1, 0]), "the unconstrained trees follow the bump");
        assert!(free_scores[n - 1] < free_scores[n * 3 / 5], "...and come down after it");

        let (kept, _) = train(&[1, 0]);
        assert!(is_monotone(&kept, &[1, 0]));
        // With the other feature fixed, the output never decreases in x, and the fit
        // still tracks the rise.
        let ensemble = Model::Trees(kept.ensemble(2, Task::Regression, &[0.0], kept.roots.len()).unwrap());
        for other in [0.1f32, 0.5, 0.9] {
            let sweep = Matrix {
                rows: 101,
                cols: 2,
                data: (0..101).flat_map(|i| [i as f32 / 100.0, other]).collect(),
            };
            let out = ensemble.outputs(&sweep).data;
            for r in 1..101 {
                assert!(out[r] >= out[r - 1], "x2 = {other}, step {r}: {} < {}", out[r], out[r - 1]);
            }
            assert!(out[25] > out[5] + 0.2, "{} vs {}", out[25], out[5]);
        }
        // Without constraints, every bound is infinite and the arithmetic is the
        // unconstrained formula, so zero constraints train the same trees as none.
        let (zeros, _) = train(&[0, 0]);
        assert_eq!(zeros.value, free.value);
    }
}
