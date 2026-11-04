//! Growing one tree best-first from the sums in [`Rows`]. Nothing here sees a row.

use tabsmith_contracts::config::TreeParams;

use super::Forest;
use super::bins::Bins;
use super::rows::{Cell, Rows, Slot};
use crate::pool::Pool;

/// Smallest gain worth a split; below it the split only fits noise.
const MIN_GAIN: f64 = 1e-9;
/// The largest gradient or hessian of a tree is scaled to at most 2^30, so it fits an `i32`, and
/// the sum of up to 2^32 rows fits a cell's `i64`. No data can overflow a cell.
pub(super) const ROW_BITS: i32 = 30;

#[derive(Debug, Clone, Copy)]
pub(super) struct Split {
    pub(super) feature: usize,
    pub(super) bin: usize,
    pub(super) gain: f64,
    /// Sums of the rows that go left.
    pub(super) left: Cell,
}

/// The objective reduction a leaf with sums (g, h) buys, and its weight: G²/(H+λ) at the free
/// optimum −G/(H+λ), or the reduction at the clipped weight when a monotone constraint bounds the
/// leaf to `[lower, upper]`. Without constraints the bounds are infinite and the arithmetic is
/// exactly the unconstrained formula. All of it is at the scale of the tree: see [`Units`].
pub(super) fn score(g: f64, h: f64, l2: f64, lower: f64, upper: f64) -> (f64, f64) {
    let w = -g / (h + l2);
    if w >= lower && w <= upper {
        (g * g / (h + l2), w)
    } else {
        let w = if w < lower { lower } else { upper };
        (-(2.0 * g * w + (h + l2) * w * w), w)
    }
}

/// 2^e, exactly.
pub(super) fn two_pow(e: i32) -> f64 {
    f64::from_bits(((e + 1023) as u64) << 52)
}

/// The exponent of the scale of a tree: `bound * 2^e` is at most 2^[`ROW_BITS`]. Larger scales add
/// nothing a tree can use, so `e` stops there for small bounds.
pub(super) fn exponent_for(bound: f64) -> i32 {
    let binary = ((bound.to_bits() >> 52) & 0x7ff) as i32 - 1023;
    (ROW_BITS - 1 - binary).min(ROW_BITS)
}

/// The nearest integer, halves away from zero. Truncation after adding a half: without SSE4.1,
/// `f64::round` is a function call.
#[inline(always)]
pub(super) fn nearest(x: f64) -> i32 {
    (x + 0.5f64.copysign(x)) as i32
}

/// The scale of one tree: its rows' gradients are multiplied by 2^g, their hessians by 2^h.
#[derive(Debug, Clone, Copy)]
pub(super) struct Units {
    pub(super) g: i32,
    pub(super) h: i32,
}

/// What every node of a tree needs to decide.
#[derive(Clone, Copy)]
pub(super) struct Growth<'a> {
    pub(super) p: &'a TreeParams,
    pub(super) monotone: &'a [i8],
    /// λ, in units of the tree's hessians.
    l2: f64,
    /// [`MIN_GAIN`], in the units of a gain: 2^(2g − h).
    pub(super) min_gain: f64,
    /// What a leaf's scaled weight is multiplied by: 2^(h − g), and the shrinkage.
    pub(super) leaf_scale: f64,
}

impl<'a> Growth<'a> {
    pub(super) fn new(p: &'a TreeParams, monotone: &'a [i8], l2: f64, shrinkage: f64, units: Units) -> Self {
        Self {
            p,
            monotone,
            l2: l2 * two_pow(units.h),
            min_gain: MIN_GAIN * two_pow(units.g) * two_pow(units.g) / two_pow(units.h),
            leaf_scale: two_pow(units.h) / two_pow(units.g) * shrinkage,
        }
    }

    pub(super) fn can_split(&self, rows: usize, depth: usize) -> bool {
        depth < self.p.max_depth && rows >= 2 * self.p.min_samples_leaf
    }
}

/// The best split of one feature, or none above the smallest gain. A split on a constrained
/// feature (`monotone[j]` is 1 or −1) is allowed only when the lesser side's weight is at most the
/// greater side's.
pub(super) fn feature_split(bins: &Bins, hist: &[Slot], j: usize, total: Cell, grow: &Growth<'_>, lower: f64, upper: f64) -> Option<Split> {
    let (g, h, l2) = (total.g as f64, total.h as f64, grow.l2);
    let (parent, _) = score(g, h, l2, lower, upper);
    let unbounded = lower == f64::NEG_INFINITY && upper == f64::INFINITY;
    let c = grow.monotone.get(j).copied().unwrap_or(0);
    let free = unbounded && c == 0;
    let min_leaf = grow.p.min_samples_leaf as i64;
    let slots = &hist[bins.offsets[j]..bins.offsets[j + 1]];
    let mut left = Cell::default();
    let mut best: Option<Split> = None;
    let mut best_gain = grow.min_gain;
    for (bin, slot) in slots[..bins.thresholds[j].len()].iter().enumerate() {
        left += slot.get();
        if left.n < min_leaf {
            continue;
        }
        if total.n - left.n < min_leaf {
            break;
        }
        let (gl, hl, gr, hr) = (left.g as f64, left.h as f64, (total.g - left.g) as f64, (total.h - left.h) as f64);
        // With no L2 and no row on a side, there would be nothing to divide by.
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
                bin,
                gain,
                left,
            });
        }
    }
    best
}

/// A leaf of the tree being grown: a range of the tree's row order.
pub(super) struct Leaf {
    pub(super) start: usize,
    pub(super) len: usize,
    pub(super) depth: usize,
    /// Sums of the leaf's rows, from the histogram that split its parent.
    pub(super) total: Cell,
    pub(super) node: usize,
    pub(super) split: Option<Split>,
    /// Bounds on the leaf's weight from the monotone splits above it; infinite without
    /// constraints. At the scale of the tree.
    pub(super) lower: f64,
    pub(super) upper: f64,
    /// Kept while the leaf can still split, for its children.
    pub(super) hist: Option<usize>,
}

/// Which histograms are free for the next node, kept from tree to tree.
#[derive(Default)]
pub(super) struct FreeHists {
    pub(super) free: Vec<usize>,
    pub(super) made: usize,
}

impl FreeHists {
    pub(super) fn take(&mut self) -> usize {
        self.free.pop().unwrap_or_else(|| {
            self.made += 1;
            self.made - 1
        })
    }
}

/// A finished leaf: its range of the tree's row order, and its value.
#[derive(Debug, Clone, Copy)]
pub(super) struct FinishedLeaf {
    pub(super) start: usize,
    pub(super) len: usize,
    pub(super) value: f32,
}

/// Grows one tree best-first over the rows' gradients in `rows.gh` and adds it to `forest`. Returns
/// its leaves.
///
/// Monotone constraints (`monotone[j]` is 1 or −1) are kept by construction, as in XGBoost: a split
/// on a constrained feature is taken only when the lesser side's weight is at most the greater
/// side's, and from then on the lesser subtree's leaves are capped at the midpoint of the two
/// weights while the greater subtree's leaves are floored at it. So at every constrained split,
/// every leaf on one side is at most every leaf on the other, and the tree is monotone in the feature.
pub(super) fn grow_tree<'env>(
    pool: &Pool<'env>,
    rows: &'env Rows,
    grow: Growth<'env>,
    forest: &mut Forest,
    free: &mut FreeHists,
) -> Vec<FinishedLeaf> {
    let n = rows.bins.rows;
    let l2 = grow.l2;
    let base = forest.feature.len();
    let mut nodes: Vec<(i32, f32, u32, u32)> = vec![(-1, 0.0, 0, 0)];
    let total = rows.begin_tree(pool);
    let mut hist = None;
    if grow.can_split(n, 0) {
        let at = free.take();
        rows.fill_root(pool, total, at);
        hist = Some(at);
    }

    let leaf = |start: usize,
                len: usize,
                depth: usize,
                node: usize,
                total: Cell,
                lower: f64,
                upper: f64,
                hist: Option<usize>,
                free: &mut FreeHists| {
        let split = hist.and_then(|at| rows.best_split(pool, at, total, grow, lower, upper));
        // A leaf that cannot split needs no histogram any more.
        let hist = match (split, hist) {
            (Some(_), hist) => hist,
            (None, Some(unused)) => {
                free.free.push(unused);
                None
            }
            (None, None) => None,
        };
        Leaf {
            start,
            len,
            depth,
            total,
            node,
            split,
            lower,
            upper,
            hist,
        }
    };

    let mut leaves = vec![leaf(0, n, 0, 0, total, f64::NEG_INFINITY, f64::INFINITY, hist, free)];
    while leaves.len() < grow.p.max_leaves {
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
        let left_len = rows.partition(pool, s.feature, s.bin, parent.start, parent.len);
        debug_assert_eq!(left_len as i64, s.left.n);
        let (left_start, right_start, right_len) = (parent.start, parent.start + left_len, parent.len - left_len);
        let (ln, rn) = (nodes.len(), nodes.len() + 1);
        nodes.push((-1, 0.0, 0, 0));
        nodes.push((-1, 0.0, 0, 0));
        nodes[parent.node] = (
            s.feature as i32,
            rows.bins.thresholds[s.feature][s.bin],
            (base + ln) as u32,
            (base + rn) as u32,
        );
        let (left_total, right_total) = (s.left, parent.total - s.left);
        let (pl, pu) = (parent.lower, parent.upper);
        let ((ll, lu), (rl, ru)) = match grow.monotone.get(s.feature).copied().unwrap_or(0) {
            0 => ((pl, pu), (pl, pu)),
            c => {
                let w = |t: Cell| score(t.g as f64, t.h as f64, l2, pl, pu).1;
                let mid = (w(left_total) + w(right_total)) / 2.0;
                if c > 0 {
                    ((pl, pu.min(mid)), (pl.max(mid), pu))
                } else {
                    ((pl.max(mid), pu), (pl, pu.min(mid)))
                }
            }
        };
        // The smaller child is summed from its rows, the larger one follows from its parent;
        // which is which depends on the row counts alone.
        let depth = parent.depth + 1;
        let left_is_smaller = left_len <= right_len;
        let (small_start, small_len, small_total, large_len) = if left_is_smaller {
            (left_start, left_len, left_total, right_len)
        } else {
            (right_start, right_len, right_total, left_len)
        };
        let (split_small, split_large) = (grow.can_split(small_len, depth), grow.can_split(large_len, depth));
        let parent_hist = parent.hist.expect("a leaf that can split keeps its histogram");
        let (mut hist_small, mut hist_large) = (None, None);
        if split_small || split_large {
            let small = free.take();
            rows.fill_node(pool, small_start, small_len, small_total, small);
            if split_large {
                rows.subtract(pool, parent_hist, small);
                hist_large = Some(parent_hist);
            } else {
                free.free.push(parent_hist);
            }
            if split_small {
                hist_small = Some(small);
            } else {
                free.free.push(small);
            }
        } else {
            free.free.push(parent_hist);
        }
        let (hist_left, hist_right) = if left_is_smaller {
            (hist_small, hist_large)
        } else {
            (hist_large, hist_small)
        };
        let new_left = leaf(left_start, left_len, depth, ln, left_total, ll, lu, hist_left, free);
        let new_right = leaf(right_start, right_len, depth, rn, right_total, rl, ru, hist_right, free);
        leaves.push(new_left);
        leaves.push(new_right);
    }
    let mut values = vec![0f32; nodes.len()];
    let mut finished = Vec::with_capacity(leaves.len());
    for l in leaves {
        // The sums are exact: differences of histograms carry no noise.
        let (_, w) = score(l.total.g as f64, l.total.h as f64, l2, l.lower, l.upper);
        let value = (w * grow.leaf_scale) as f32;
        values[l.node] = value;
        finished.push(FinishedLeaf {
            start: l.start,
            len: l.len,
            value,
        });
        forest.max_depth = forest.max_depth.max(l.depth);
        if let Some(unused) = l.hist {
            free.free.push(unused);
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
    finished
}
