//! The `gbdt` candidate: gradient-boosted decision trees on the fitted feature
//! matrix (the same transform as the networks). Split points come from the
//! training rows, up to `max_bins` per feature; trees grow best-first, the leaf
//! with the largest gain splitting next. The loss is the task's: class-weighted
//! log loss (binary), class-weighted softmax cross-entropy with one tree per class
//! and round (multiclass), squared error of the standardised target (regression);
//! early stopping watches the validation loss.
//!
//! Deterministic by construction, on any number of threads. A histogram is built one
//! feature at a time, and each feature sums its rows in row order on one thread, so
//! every (feature, bin) sees the same additions in the same order however the features
//! are spread over threads. One child's histogram is the parent's minus its sibling's,
//! and which child that is depends on the row counts alone. Everything else that runs
//! on several threads is a pure function of its index (a row, a piece of rows), combined
//! in index order. Ties between splits are broken by (feature, bin) and between leaves
//! by creation order, and leaf values are summed from the leaf's rows and rounded to f32
//! before they move the scores. The trainer's validation scores are therefore exactly
//! what the runtime computes for the same trees.
//!
//! One boosting round is one "epoch" of the shared [`TrainState`], so checkpoints,
//! cancellation, crash recovery and exact resume (R2) work as for the networks.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::time::Instant;

use rayon::prelude::*;
use tabsmith_contracts::config::{CandidateTrainConfig, Task, TreeParams};
use tabsmith_contracts::model::{ModelConfig, TreesSpec};
use tabsmith_contracts::run::EpochRecord;
use tabsmith_contracts::transform::{FeatureTransform, FittedTransform};
use tabsmith_contracts::{ErrorCode, Phase, Result, TabsmithError};
use tabsmith_embedded::kernels;
use tabsmith_runtime::Matrix;
use tabsmith_runtime::model::{Model, TreeEnsemble, sigmoid};

use crate::data::Targets;
use crate::metrics::selected_metric;
use crate::parallel::install;
use crate::project::{read_json, write_atomic, write_json};
use crate::train::{
    ResumeKey, RunControl, StartFrom, TrainInputs, TrainOutcome, TrainState, checkpoint_dirs, class_weights_of, encode_targets,
    par_outputs, seal_checkpoint, task_loss, verify_checkpoint,
};

/// Smallest gain worth a split; below it the split only fits rounding noise.
const MIN_GAIN: f64 = 1e-9;
/// A bin holding at least this share of the training rows is not summed: its sums are
/// what the other bins leave of the node's totals. One-hot columns are almost all one bin.
const DOMINANT_SHARE: f64 = 0.8;
/// Below this many (row, feature) visits a histogram stays on the calling thread, and one
/// piece of parallel work holds at least this many: waking a thread costs more than less.
const PAR_MIN_CELLS: usize = 1 << 18;
/// A node with less than 1 / this of the training rows copies its gradients next to each
/// other first; a larger one reads them in place, nearly in order.
const GATHER_BELOW: usize = 8;
/// A histogram with fewer slots than this is searched for a split on the calling thread.
const PAR_MIN_SLOTS: usize = 1 << 15;
/// Rows of one piece when a large node's rows are partitioned on several threads.
const PARTITION_PIECE: usize = 1 << 16;
/// Applying a tree to at most this many rows stays on the calling thread.
const PAR_MIN_ROWS: usize = 1 << 14;

/// Candidate split points of every feature, ascending. Bin `k` of a feature holds
/// the values `<= thresholds[k]` (above the previous threshold); the last bin the rest.
struct Bins {
    thresholds: Vec<Vec<f32>>,
    /// Bin of every training row, one column per feature.
    codes: Vec<u8>,
    rows: usize,
    width: usize,
    /// First histogram slot of each feature, then the total: feature `j` has
    /// `thresholds[j].len() + 1` slots.
    offsets: Vec<usize>,
    /// The bin of each feature that holds at least [`DOMINANT_SHARE`] of the rows, if any.
    dominant: Vec<Option<u8>>,
}

impl Bins {
    fn new(x: &Matrix, max_bins: usize) -> Self {
        let (n, d) = (x.rows, x.cols);
        // Every feature's thresholds and codes depend on that feature's values alone.
        let columns: Vec<(Vec<f32>, Vec<u8>, Option<u8>)> = (0..d)
            .into_par_iter()
            .map(|j| {
                let values: Vec<f32> = (0..n).map(|r| x.data[r * d + j]).collect();
                let mut v = values.clone();
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
                let codes: Vec<u8> = values.iter().map(|&x| t.partition_point(|&t| t < x) as u8).collect();
                let mut counts = [0u32; 256];
                for &c in &codes {
                    counts[c as usize] += 1;
                }
                let (top, top_count) = counts
                    .iter()
                    .enumerate()
                    .fold((0, 0), |b, (i, &c)| if c > b.1 { (i, c) } else { b });
                let dominant = (f64::from(top_count) >= DOMINANT_SHARE * n as f64).then_some(top as u8);
                (t, codes, dominant)
            })
            .collect();
        let mut offsets = Vec::with_capacity(d + 1);
        let mut total = 0;
        for (t, _, _) in &columns {
            offsets.push(total);
            total += t.len() + 1;
        }
        offsets.push(total);
        let (mut thresholds, mut codes, mut dominant) = (Vec::with_capacity(d), Vec::with_capacity(n * d), Vec::with_capacity(d));
        for (t, c, top) in columns {
            thresholds.push(t);
            codes.extend_from_slice(&c);
            dominant.push(top);
        }
        Self {
            thresholds,
            codes,
            rows: n,
            width: d,
            offsets,
            dominant,
        }
    }

    /// The bin of every training row in feature `j`.
    fn column(&self, j: usize) -> &[u8] {
        &self.codes[j * self.rows..(j + 1) * self.rows]
    }

    /// The histogram slots of all features.
    fn slots(&self) -> usize {
        self.offsets[self.width]
    }
}

/// Gradient sum, hessian sum and row count of one (feature, bin).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct Slot {
    g: f64,
    h: f64,
    n: u32,
}

/// One feature's slots: every row in row order, except that a dominant bin's sums are
/// what the other bins leave of the node's totals `g` and `h`.
#[inline(always)]
fn fill_slots(slots: &mut [Slot], col: &[u8], rows: &[u32], top: Option<u8>, g: f64, h: f64, pair: impl Fn(usize, u32) -> (f64, f64)) {
    slots.fill(Slot::default());
    match top {
        None => {
            for (i, &r) in rows.iter().enumerate() {
                let (a, b) = pair(i, r);
                let s = &mut slots[col[r as usize] as usize];
                s.g += a;
                s.h += b;
                s.n += 1;
            }
        }
        Some(top) => {
            let (mut sum_g, mut sum_h, mut sum_n) = (0.0, 0.0, 0u32);
            for (i, &r) in rows.iter().enumerate() {
                let code = col[r as usize];
                if code != top {
                    let (a, b) = pair(i, r);
                    let s = &mut slots[code as usize];
                    s.g += a;
                    s.h += b;
                    s.n += 1;
                    sum_g += a;
                    sum_h += b;
                    sum_n += 1;
                }
            }
            slots[top as usize] = Slot {
                g: g - sum_g,
                h: h - sum_h,
                n: rows.len() as u32 - sum_n,
            };
        }
    }
}

/// The sums of every (feature, bin) over `rows`, into `hist`; `g` and `h` are the rows'
/// totals. Each feature sums its rows in row order on one thread, so the result does not
/// depend on the number of threads.
fn build_histogram(bins: &Bins, rows: &[u32], g: f64, h: f64, gh: &[(f64, f64)], gathered: &mut Vec<(f64, f64)>, hist: &mut [Slot]) {
    let d = bins.width;
    let gather = rows.len() * GATHER_BELOW < bins.rows;
    if gather {
        gathered.clear();
        gathered.extend(rows.iter().map(|&r| gh[r as usize]));
    }
    let ordered: &[(f64, f64)] = gathered;
    let mut parts: Vec<&mut [Slot]> = Vec::with_capacity(d);
    let mut rest = hist;
    for j in 0..d {
        let (a, b) = rest.split_at_mut(bins.offsets[j + 1] - bins.offsets[j]);
        parts.push(a);
        rest = b;
    }
    let fill = |(j, slots): (usize, &mut [Slot])| {
        let (col, top) = (bins.column(j), bins.dominant[j]);
        if gather {
            fill_slots(slots, col, rows, top, g, h, |i, _| ordered[i]);
        } else {
            fill_slots(slots, col, rows, top, g, h, |_, r| gh[r as usize]);
        }
    };
    if rows.len() * d >= PAR_MIN_CELLS {
        let features_per_piece = PAR_MIN_CELLS.div_ceil(rows.len()).max(1);
        parts.into_par_iter().enumerate().with_min_len(features_per_piece).for_each(fill);
    } else {
        parts.into_iter().enumerate().for_each(fill);
    }
}

/// The histogram of the larger child: the parent's, in place, minus the smaller child's.
fn subtract(parent: &mut [Slot], smaller: &[Slot]) {
    for (a, b) in parent.iter_mut().zip(smaller) {
        a.g -= b.g;
        a.h -= b.h;
        a.n -= b.n;
    }
}

/// Moves the rows that go left (`col[row] <= bin`) to the front of `rows`, keeping the
/// order on both sides, and returns how many there are. `spare` is as long as `rows`.
fn partition_piece(col: &[u8], bin: usize, rows: &mut [u32], spare: &mut [u32]) -> usize {
    let (mut left, mut right) = (0, 0);
    for i in 0..rows.len() {
        let row = rows[i];
        let goes_left = usize::from(col[row as usize] as usize <= bin);
        rows[left] = row;
        spare[right] = row;
        left += goes_left;
        right += 1 - goes_left;
    }
    rows[left..].copy_from_slice(&spare[..right]);
    left
}

/// [`partition_piece`], on several threads for a large node: each piece partitions
/// itself, then the pieces' two sides are joined in order, so the result is the same.
fn partition(col: &[u8], bin: usize, rows: &mut [u32], spare: &mut [u32]) -> usize {
    if rows.len() <= 2 * PARTITION_PIECE {
        return partition_piece(col, bin, rows, spare);
    }
    let lefts: Vec<usize> = rows
        .par_chunks_mut(PARTITION_PIECE)
        .zip(spare.par_chunks_mut(PARTITION_PIECE))
        .map(|(r, s)| partition_piece(col, bin, r, s))
        .collect();
    let total: usize = lefts.iter().sum();
    let (mut at_left, mut at_right) = (0, total);
    for (piece, &left) in rows.chunks(PARTITION_PIECE).zip(&lefts) {
        spare[at_left..at_left + left].copy_from_slice(&piece[..left]);
        spare[at_right..at_right + piece.len() - left].copy_from_slice(&piece[left..]);
        at_left += left;
        at_right += piece.len() - left;
    }
    let n = rows.len();
    rows.copy_from_slice(&spare[..n]);
    total
}

#[derive(Debug, Clone, Copy)]
struct Split {
    feature: usize,
    bin: usize,
    gain: f64,
    /// Sums of the rows that go left.
    left_g: f64,
    left_h: f64,
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
    hist: &[Slot],
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
    let unbounded = lower == f64::NEG_INFINITY && upper == f64::INFINITY;
    let feature = |j: usize| -> Option<Split> {
        let thresholds = &bins.thresholds[j];
        let slots = &hist[bins.offsets[j]..bins.offsets[j + 1]];
        let c = monotone.get(j).copied().unwrap_or(0);
        let free = unbounded && c == 0;
        let (mut gl, mut hl, mut nl) = (0.0, 0.0, 0usize);
        let mut best: Option<Split> = None;
        let mut best_gain = MIN_GAIN;
        for (k, slot) in slots[..thresholds.len()].iter().enumerate() {
            gl += slot.g;
            hl += slot.h;
            nl += slot.n as usize;
            if nl < p.min_samples_leaf {
                continue;
            }
            if n - nl < p.min_samples_leaf {
                break;
            }
            let (gr, hr) = (g - gl, h - hl);
            // Differences of histograms can leave a side with a hessian sum of rounding
            // noise; with no L2 that would divide by next to nothing.
            if hl + l2 <= 0.0 || hr + l2 <= 0.0 {
                continue;
            }
            let gain = if free {
                // What `score` computes when no bound applies, without the weights.
                gl * gl / (hl + l2) + gr * gr / (hr + l2) - parent
            } else {
                let ((sl, wl), (sr, wr)) = (score(gl, hl, l2, lower, upper), score(gr, hr, l2, lower, upper));
                if (c > 0 && wl > wr) || (c < 0 && wl < wr) {
                    continue;
                }
                sl + sr - parent
            };
            if gain > best_gain {
                best_gain = gain;
                best = Some(Split {
                    feature: j,
                    bin: k,
                    gain,
                    left_g: gl,
                    left_h: hl,
                });
            }
        }
        best
    };
    let per_feature: Vec<Option<Split>> = if hist.len() >= PAR_MIN_SLOTS {
        (0..bins.width).into_par_iter().map(feature).collect()
    } else {
        (0..bins.width).map(feature).collect()
    };
    // The first feature wins a tie, as in one scan over all of them.
    per_feature.into_iter().flatten().fold(
        None,
        |best: Option<Split>, s| if best.is_none_or(|b| s.gain > b.gain) { Some(s) } else { best },
    )
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
    /// Every row's walk is independent, so the rows are split between the threads.
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
        if x.rows <= PAR_MIN_ROWS {
            for r in 0..x.rows {
                scores[r * k + t % k] += walk(x.row(r));
            }
            return;
        }
        scores
            .par_chunks_mut(k)
            .enumerate()
            .with_min_len(PAR_MIN_ROWS / 8)
            .for_each(|(r, row_scores)| row_scores[t % k] += walk(x.row(r)));
    }
}

/// A leaf of the tree being grown: a range of the tree's row order.
struct Leaf {
    start: usize,
    len: usize,
    depth: usize,
    /// Sums of the leaf's rows, from the histogram that split its parent.
    g: f64,
    h: f64,
    node: usize,
    split: Option<Split>,
    /// Bounds on the leaf's weight from the monotone splits above it; infinite
    /// without constraints.
    lower: f64,
    upper: f64,
    /// Kept while the leaf can still split, for its children.
    hist: Option<Vec<Slot>>,
}

/// What growing a tree needs and the next tree can reuse: no allocation per node.
#[derive(Default)]
struct Scratch {
    hists: Vec<Vec<Slot>>,
    gathered: Vec<(f64, f64)>,
    order: Vec<u32>,
    spare: Vec<u32>,
    moved: Vec<f64>,
}

/// Grows one tree best-first over the training rows and adds it to `forest`. `gh` holds
/// each row's gradient and hessian. Returns the (f32-rounded) value of the leaf each
/// training row lands in.
///
/// Monotone constraints (`monotone[j]` is 1 or −1) are kept by construction, as in
/// XGBoost: a split on a constrained feature is taken only when the lesser side's
/// weight is at most the greater side's, and from then on the lesser subtree's leaves
/// are capped at the midpoint of the two weights while the greater subtree's leaves
/// are floored at it. So at every constrained split, every leaf on one side is at most
/// every leaf on the other, and the tree is monotone in the feature.
#[allow(clippy::too_many_arguments)]
fn grow_tree<'s>(
    bins: &Bins,
    gh: &[(f64, f64)],
    p: &TreeParams,
    l2: f64,
    shrinkage: f64,
    monotone: &[i8],
    forest: &mut Forest,
    scratch: &'s mut Scratch,
) -> &'s [f64] {
    let n = gh.len();
    let base = forest.feature.len();
    let mut nodes: Vec<(i32, f32, u32, u32)> = vec![(-1, 0.0, 0, 0)];
    let Scratch {
        hists,
        gathered,
        order,
        spare,
        moved,
    } = scratch;
    order.clear();
    order.extend(0..n as u32);
    spare.resize(n, 0);

    let can_split = |rows: usize, depth: usize| depth < p.max_depth && rows >= 2 * p.min_samples_leaf;
    let take = |hists: &mut Vec<Vec<Slot>>| hists.pop().unwrap_or_else(|| vec![Slot::default(); bins.slots()]);
    let sums = |rows: &[u32]| {
        rows.iter().fold((0.0, 0.0), |(g, h), &r| {
            let (a, b) = gh[r as usize];
            (g + a, h + b)
        })
    };
    let leaf = |start: usize,
                len: usize,
                depth: usize,
                node: usize,
                g: f64,
                h: f64,
                lower: f64,
                upper: f64,
                hist: Option<Vec<Slot>>,
                hists: &mut Vec<Vec<Slot>>| {
        let split = hist
            .as_ref()
            .and_then(|hist| best_split(bins, hist, g, h, len, p, l2, monotone, lower, upper));
        // A leaf that cannot split needs no histogram any more.
        let hist = match (split, hist) {
            (Some(_), hist) => hist,
            (None, Some(unused)) => {
                hists.push(unused);
                None
            }
            (None, None) => None,
        };
        Leaf {
            start,
            len,
            depth,
            g,
            h,
            node,
            split,
            lower,
            upper,
            hist,
        }
    };

    let (root_g, root_h) = sums(order);
    let root_hist = can_split(n, 0).then(|| {
        let mut hist = take(hists);
        build_histogram(bins, order, root_g, root_h, gh, gathered, &mut hist);
        hist
    });
    let mut leaves = vec![leaf(0, n, 0, 0, root_g, root_h, f64::NEG_INFINITY, f64::INFINITY, root_hist, hists)];
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
        let left_len = partition(
            bins.column(s.feature),
            s.bin,
            &mut order[parent.start..parent.start + parent.len],
            &mut spare[..parent.len],
        );
        let (left_start, right_start, right_len) = (parent.start, parent.start + left_len, parent.len - left_len);
        let (ln, rn) = (nodes.len(), nodes.len() + 1);
        nodes.push((-1, 0.0, 0, 0));
        nodes.push((-1, 0.0, 0, 0));
        nodes[parent.node] = (
            s.feature as i32,
            bins.thresholds[s.feature][s.bin],
            (base + ln) as u32,
            (base + rn) as u32,
        );
        let (gl, hl) = (s.left_g, s.left_h);
        let (gr, hr) = (parent.g - gl, parent.h - hl);
        let (pl, pu) = (parent.lower, parent.upper);
        let ((ll, lu), (rl, ru)) = match monotone.get(s.feature).copied().unwrap_or(0) {
            0 => ((pl, pu), (pl, pu)),
            c => {
                let (wl, wr) = (score(gl, hl, l2, pl, pu).1, score(gr, hr, l2, pl, pu).1);
                let mid = (wl + wr) / 2.0;
                if c > 0 {
                    ((pl, pu.min(mid)), (pl.max(mid), pu))
                } else {
                    ((pl.max(mid), pu), (pl, pu.min(mid)))
                }
            }
        };
        // The smaller child is summed from its rows, the larger one follows from its
        // parent; which is which depends on the row counts alone.
        let depth = parent.depth + 1;
        let left_is_smaller = left_len <= right_len;
        let (small_start, small_len, small_g, small_h, large_len) = if left_is_smaller {
            (left_start, left_len, gl, hl, right_len)
        } else {
            (right_start, right_len, gr, hr, left_len)
        };
        let (split_small, split_large) = (can_split(small_len, depth), can_split(large_len, depth));
        let mut parent_hist = parent.hist.expect("a leaf that can split keeps its histogram");
        let (mut hist_small, mut hist_large) = (None, None);
        if split_small || split_large {
            let mut small = take(hists);
            build_histogram(
                bins,
                &order[small_start..small_start + small_len],
                small_g,
                small_h,
                gh,
                gathered,
                &mut small,
            );
            if split_large {
                subtract(&mut parent_hist, &small);
                hist_large = Some(parent_hist);
            } else {
                hists.push(parent_hist);
            }
            if split_small {
                hist_small = Some(small);
            } else {
                hists.push(small);
            }
        } else {
            hists.push(parent_hist);
        }
        let (hist_left, hist_right) = if left_is_smaller {
            (hist_small, hist_large)
        } else {
            (hist_large, hist_small)
        };
        leaves.push(leaf(left_start, left_len, depth, ln, gl, hl, ll, lu, hist_left, hists));
        leaves.push(leaf(right_start, right_len, depth, rn, gr, hr, rl, ru, hist_right, hists));
    }
    let mut values = vec![0f32; nodes.len()];
    moved.resize(n, 0.0);
    for l in leaves {
        // From the leaf's rows: differences of histograms carry rounding noise, which
        // a split can bear but a leaf with a small hessian sum and no L2 could not.
        let rows = &order[l.start..l.start + l.len];
        let (g, h) = sums(rows);
        let (_, w) = score(g, h, l2, l.lower, l.upper);
        let v = (w * shrinkage) as f32;
        values[l.node] = v;
        for &r in rows {
            moved[r as usize] = f64::from(v);
        }
        forest.max_depth = forest.max_depth.max(l.depth);
        if let Some(unused) = l.hist {
            hists.push(unused);
        }
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
    let bins = install(|| Bins::new(x, p.max_bins));
    // Scores (`rows x k`) are recomputed from the trees in order: the same f64
    // additions as when the trees were grown, so a resumed run continues bit for bit.
    let start_scores = |rows: usize| (0..rows).flat_map(|_| base.iter().copied()).collect::<Vec<f64>>();
    let mut scores = start_scores(n);
    let mut valid_scores = start_scores(x_valid.rows);
    install(|| {
        for t in 0..forest.roots.len() {
            forest.add_tree(t, x, &mut scores, k);
            forest.add_tree(t, x_valid, &mut valid_scores, k);
        }
    });
    let as_outputs = |s: &[f64], rows: usize| Matrix {
        rows,
        cols: k,
        data: s.iter().map(|&v| v as f32).collect(),
    };
    let mut best_valid = as_outputs(&valid_scores, x_valid.rows).data;
    let mut gh = vec![(0.0f64, 0.0f64); n];
    let mut probs = vec![0.0f64; if task == Task::Multiclass { n * k } else { 0 }];
    let mut scratch = Scratch::default();
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
        let (train_loss, valid_loss, metric, v_out) = install(|| {
            // One tree per output, all fitted to the gradients at the start of the round.
            if task == Task::Multiclass {
                probs.par_chunks_mut(k).zip(scores.par_chunks(k)).with_min_len(1024).for_each_init(
                    || vec![0f32; k],
                    |z, (p, s)| {
                        for (a, &b) in z.iter_mut().zip(s) {
                            *a = b as f32;
                        }
                        kernels::softmax(z, p);
                    },
                );
            }
            for c in 0..k {
                gh.par_iter_mut().enumerate().with_min_len(4096).for_each(|(r, slot)| {
                    let w = f64::from(enc_w[r]);
                    let (g, h) = match task {
                        Task::Binary => {
                            let prob = sigmoid(scores[r]);
                            (w * (prob - f64::from(enc_y[r])), w * prob * (1.0 - prob))
                        }
                        Task::Multiclass => {
                            let pc = probs[r * k + c];
                            let t = if enc_y[r] as usize == c { 1.0 } else { 0.0 };
                            (w * (pc - t), w * pc * (1.0 - pc))
                        }
                        Task::Regression => (w * (scores[r] - f64::from(enc_y[r])), w),
                    };
                    *slot = (g, h.max(1e-16));
                });
                let moved = grow_tree(&bins, &gh, &p, l2, shrinkage, monotone, &mut forest, &mut scratch);
                scores
                    .par_chunks_mut(k)
                    .zip(moved.par_iter())
                    .with_min_len(4096)
                    .for_each(|(s, &m)| s[c] += m);
                forest.add_tree(forest.roots.len() - 1, x_valid, &mut valid_scores, k);
            }
            let train_out = as_outputs(&scores, n);
            let train_loss = task_loss(&train_out, y_train, output, &weights);
            let v_out = as_outputs(&valid_scores, x_valid.rows);
            let valid_loss = task_loss(&v_out, y_valid, output, &weights);
            let metric = selected_metric(&v_out, y_valid, output, metric_name).unwrap_or(0.0);
            (train_loss, valid_loss, metric, v_out)
        });
        state.step += 1;
        state.epoch += 1;

        if !(train_loss.is_finite() && valid_loss.is_finite()) {
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
    let parity = par_outputs(x_valid, |x| runtime.outputs(x))
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
    use rayon::ThreadPoolBuilder;

    use super::*;

    /// A pseudo-random generator for test data, uniform on [0, 1).
    fn uniform(seed: u64) -> impl FnMut() -> f64 {
        let mut state = seed;
        move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    /// `n` rows of `dense` uniform features and `sparse` one-hot-like ones (1 in 5% of
    /// the rows, which makes bin 0 dominant).
    fn table(n: usize, dense: usize, sparse: usize) -> Matrix {
        let mut next = uniform(0x9e37_79b9_7f4a_7c15);
        let d = dense + sparse;
        let mut data = Vec::with_capacity(n * d);
        for _ in 0..n {
            for _ in 0..dense {
                data.push(next() as f32);
            }
            for _ in 0..sparse {
                data.push(if next() < 0.05 { 1.0 } else { 0.0 });
            }
        }
        Matrix { rows: n, cols: d, data }
    }

    fn pairs(grad: &[f64], hess: &[f64]) -> Vec<(f64, f64)> {
        grad.iter().copied().zip(hess.iter().copied()).collect()
    }

    #[allow(clippy::too_many_arguments)]
    fn grow(
        bins: &Bins,
        grad: &[f64],
        hess: &[f64],
        p: &TreeParams,
        l2: f64,
        shrinkage: f64,
        monotone: &[i8],
        forest: &mut Forest,
    ) -> Vec<f64> {
        let mut scratch = Scratch::default();
        grow_tree(bins, &pairs(grad, hess), p, l2, shrinkage, monotone, forest, &mut scratch).to_vec()
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

    /// The sums of every (feature, bin) one row at a time, with no shortcut.
    fn plain_histogram(bins: &Bins, rows: &[u32], gh: &[(f64, f64)]) -> Vec<Slot> {
        let mut hist = vec![Slot::default(); bins.slots()];
        for j in 0..bins.width {
            for &r in rows {
                let s = &mut hist[bins.offsets[j] + bins.column(j)[r as usize] as usize];
                s.g += gh[r as usize].0;
                s.h += gh[r as usize].1;
                s.n += 1;
            }
        }
        hist
    }

    fn assert_close(a: &[Slot], b: &[Slot]) {
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b) {
            assert_eq!(x.n, y.n);
            assert!((x.g - y.g).abs() < 1e-9, "{} vs {}", x.g, y.g);
            assert!((x.h - y.h).abs() < 1e-9, "{} vs {}", x.h, y.h);
        }
    }

    #[test]
    fn the_trees_do_not_depend_on_the_number_of_threads() {
        // Large enough that nodes take every parallel path: histograms, partition, tree
        // application. One round of a few trees on pools of different sizes.
        let (n, dense, sparse) = (150_000usize, 3usize, 3usize);
        let x = table(n, dense, sparse);
        assert!(n * (dense + sparse) >= PAR_MIN_CELLS && n > 2 * PARTITION_PIECE && n > PAR_MIN_ROWS);
        let target: Vec<f64> = (0..n)
            .map(|r| {
                let row = x.row(r);
                f64::from(row[0]) * 2.0 + if row[4] > 0.5 { 1.0 } else { 0.0 } + (f64::from(row[1]) * 25.0).sin()
            })
            .collect();
        let params = TreeParams {
            max_leaves: 15,
            max_depth: 6,
            min_samples_leaf: 20,
            max_bins: 255,
            monotone: Default::default(),
        };
        let train = |threads: usize| {
            let pool = ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
            pool.install(|| {
                let bins = Bins::new(&x, params.max_bins);
                let mut forest = Forest::default();
                let mut scratch = Scratch::default();
                let mut scores = vec![0.0; n];
                for _ in 0..3 {
                    let gh: Vec<(f64, f64)> = scores.iter().zip(&target).map(|(s, y)| (s - y, 1.0)).collect();
                    let moved = grow_tree(&bins, &gh, &params, 1.0, 0.1, &[], &mut forest, &mut scratch);
                    for (s, m) in scores.iter_mut().zip(moved) {
                        *s += m;
                    }
                }
                let mut applied = vec![0.0; n];
                for t in 0..forest.roots.len() {
                    forest.add_tree(t, &x, &mut applied, 1);
                }
                (forest, scores, applied)
            })
        };
        let (reference, scores, applied) = train(1);
        assert_eq!(reference.roots.len(), 3);
        assert!(reference.feature.len() > 30, "the trees really split");
        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<u64>>();
        assert_eq!(bits(&scores), bits(&applied), "the rows' leaves are the ones the walk reaches");
        for threads in [2, 7] {
            let (forest, s, a) = train(threads);
            assert_eq!(forest.feature, reference.feature);
            assert_eq!(forest.left, reference.left);
            assert_eq!(forest.right, reference.right);
            let f32_bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
            assert_eq!(f32_bits(&forest.threshold), f32_bits(&reference.threshold));
            assert_eq!(f32_bits(&forest.value), f32_bits(&reference.value));
            assert_eq!(bits(&s), bits(&scores));
            assert_eq!(bits(&a), bits(&applied));
        }
    }

    #[test]
    fn shortcuts_match_summing_every_row() {
        // The dominant bin, the gathered gradients and the parent minus its sibling all
        // give what plain sums give, within rounding, and the same row counts exactly.
        let n = 20_000usize;
        let x = table(n, 2, 2);
        let bins = Bins::new(&x, 64);
        assert_eq!(bins.dominant[0], None);
        assert_eq!(bins.dominant[2], Some(0), "one-hot columns are almost all one bin");
        assert_eq!(bins.slots(), bins.thresholds.iter().map(|t| t.len() + 1).sum::<usize>());
        let mut next = uniform(7);
        let gh: Vec<(f64, f64)> = (0..n).map(|_| (next() - 0.5, next() * 0.25 + 1e-3)).collect();
        let totals = |rows: &[u32]| {
            rows.iter().fold((0.0, 0.0), |(g, h), &r| {
                let (a, b) = gh[r as usize];
                (g + a, h + b)
            })
        };
        let all: Vec<u32> = (0..n as u32).collect();
        let few: Vec<u32> = all.iter().copied().filter(|r| r % 11 == 0).collect();
        let others: Vec<u32> = all.iter().copied().filter(|r| r % 11 != 0).collect();
        assert!(few.len() * GATHER_BELOW < n, "this node gathers its gradients");
        let mut gathered = Vec::new();
        let build = |rows: &[u32], gathered: &mut Vec<(f64, f64)>| {
            let mut hist = vec![Slot::default(); bins.slots()];
            let (g, h) = totals(rows);
            build_histogram(&bins, rows, g, h, &gh, gathered, &mut hist);
            hist
        };
        let mut parent = build(&all, &mut gathered);
        assert_close(&parent, &plain_histogram(&bins, &all, &gh));
        let small = build(&few, &mut gathered);
        assert_close(&small, &plain_histogram(&bins, &few, &gh));
        subtract(&mut parent, &small);
        assert_close(&parent, &plain_histogram(&bins, &others, &gh));
    }

    #[test]
    fn the_partition_keeps_the_order_on_both_sides() {
        // Below and above the size that runs on several threads.
        for n in [1000usize, 2 * PARTITION_PIECE + 12_345] {
            let mut next = uniform(11);
            let col: Vec<u8> = (0..n).map(|_| (next() * 8.0) as u8).collect();
            let mut rows: Vec<u32> = (0..n as u32).collect();
            let mut spare = vec![0; n];
            let left = partition(&col, 3, &mut rows, &mut spare);
            let expected: Vec<u32> = (0..n as u32)
                .filter(|&r| col[r as usize] <= 3)
                .chain((0..n as u32).filter(|&r| col[r as usize] > 3))
                .collect();
            assert_eq!(left, (0..n).filter(|&r| col[r] <= 3).count());
            assert_eq!(rows, expected);
        }
    }

    #[test]
    fn binning_and_applying_a_tree_match_the_runtime() {
        let (n, d) = (40_000usize, 3usize);
        let x = table(n, d, 0);
        let mut next = uniform(3);
        let grad: Vec<f64> = (0..n).map(|_| next() - 0.5).collect();
        let hess: Vec<f64> = (0..n).map(|_| next() * 0.25).collect();
        let bins = Bins::new(&x, 64);
        for r in 0..n {
            for j in 0..d {
                let code = bins.thresholds[j].partition_point(|&t| t < x.data[r * d + j]) as u8;
                assert_eq!(bins.column(j)[r], code);
            }
        }
        // Applying a tree on several threads equals the runtime's walk, row by row.
        let mut forest = Forest::default();
        grow(&bins, &grad, &hess, &stump(), 1.0, 1.0, &[], &mut forest);
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
        let moved = grow(&bins, &grad, &hess, &stump(), 1.0, 1.0, &[], &mut forest);
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
        let moved = grow(&Bins::new(&x, 255), &grad, &vec![1.0; n], &stump(), 0.0, 1.0, &[], &mut forest);
        assert!(moved.iter().zip(&y).all(|(&m, &t)| (m - t).abs() < 1e-6), "{moved:?}");
    }

    #[test]
    fn saturated_rows_with_no_l2_stay_finite() {
        // Hessians at the floor next to large ones: a histogram difference can leave a
        // side with rounding noise for its hessian sum. With no L2 nothing may blow up.
        let n = 4000usize;
        let x = table(n, 2, 0);
        let mut next = uniform(5);
        let gh: Vec<(f64, f64)> = (0..n)
            .map(|r| if r % 4 == 0 { (next() - 0.5, 1.0) } else { (1e-16, 1e-16) })
            .collect();
        let params = TreeParams {
            max_leaves: 31,
            max_depth: 8,
            min_samples_leaf: 5,
            max_bins: 255,
            monotone: Default::default(),
        };
        let mut forest = Forest::default();
        let mut scratch = Scratch::default();
        let moved = grow_tree(&Bins::new(&x, 255), &gh, &params, 0.0, 0.1, &[], &mut forest, &mut scratch).to_vec();
        assert!(forest.value.iter().all(|v| v.is_finite()));
        assert!(moved.iter().all(|v| v.is_finite()));
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
                let moved = grow(&bins, &grad, &vec![1.0; n], &deep, 0.0, 0.3, monotone, &mut forest);
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
