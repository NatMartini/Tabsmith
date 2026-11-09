//! The search over boxes of features for a proof that a claim holds, or for an input where it
//! fails.
//!
//! A tree ensemble is searched exactly: a box where every tree reaches leaves of one value is
//! one prediction, so the search ends with a proof or a counterexample unless the budget runs
//! out. A network is searched with interval bounds: a proof or a counterexample when one is
//! found, undecided otherwise. Both keep every box they split as a node of the proof, and both
//! close a box only when [`rules::is_empty`] or [`Claim::holds_within`] on the same bounds the
//! checker computes says so.

use std::time::Instant;

use tabsmith_contracts::Result;
use tabsmith_embedded::kernels;
use tabsmith_embedded::rules::{self, Claim, Domain, NUMERIC, Step};

use crate::model::{DenseModel, TreeEnsemble};

/// The rows with `lo <= x <= hi` that the domain accepts, `none` per category.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Bx {
    pub lo: Vec<f32>,
    pub hi: Vec<f32>,
    pub none: Vec<bool>,
}

impl Bx {
    fn narrowed(&self, domain: &Domain<'_>, feature: usize, threshold: f32, left: bool) -> Bx {
        let mut b = self.clone();
        rules::narrow(domain, &mut b.lo, &mut b.hi, &mut b.none, feature, threshold, left);
        b
    }

    fn is_empty(&self, domain: &Domain<'_>) -> bool {
        rules::is_empty(domain, &self.lo, &self.hi, &self.none)
    }
}

/// What a candidate counterexample turned out to be, once made an input and predicted.
pub(crate) enum Check<W> {
    /// An input in the box whose prediction breaks the claim.
    Fails(W),
    /// No input encodes into the box.
    NoInput,
    /// The input made for the box gets a prediction that satisfies the claim.
    Holds,
}

/// How a search ended.
pub(crate) enum Found<W> {
    /// The claim holds on every box: the proof, and the room its walk needs.
    Proof { steps: Vec<Step>, depth: usize },
    /// A confirmed counterexample.
    Fails(W),
    /// Neither, and why.
    Undecided(String),
}

/// Makes an input of a box where the claim fails, near a point of it (numbers, and each
/// category's value) when one is given, and predicts it.
pub(crate) type Confirm<'c, W> = dyn FnMut(&Bx, Option<(&[f32], &[Option<usize>])>) -> Result<Check<W>> + 'c;

/// When a search stops.
pub(crate) struct Limits {
    pub boxes: usize,
    pub deadline: Option<Instant>,
}

impl Limits {
    fn reached(&self, visited: usize) -> Option<String> {
        if visited > self.boxes {
            return Some(format!("the budget of {} boxes ran out", self.boxes));
        }
        if visited.is_multiple_of(256) && self.deadline.is_some_and(|d| Instant::now() > d) {
            return Some("the time limit ran out".to_string());
        }
        None
    }
}

/// A proof's nodes while it grows.
#[derive(Debug, Clone, Copy)]
enum Node {
    Open,
    Close,
    Split {
        feature: u32,
        threshold: f32,
        left: u32,
        right: u32,
    },
}

/// The proof in the order of a walk (left part first), and its deepest path in splits.
fn serialize(nodes: &[Node]) -> (Vec<Step>, usize) {
    let mut steps = Vec::with_capacity(nodes.len());
    let mut depth = 0;
    let mut stack = vec![(0u32, 0usize)];
    while let Some((id, d)) = stack.pop() {
        match nodes[id as usize] {
            Node::Split {
                feature,
                threshold,
                left,
                right,
            } => {
                steps.push(Step::Split { feature, threshold });
                depth = depth.max(d + 1);
                stack.push((right, d + 1));
                stack.push((left, d + 1));
            }
            Node::Close | Node::Open => steps.push(Step::Close),
        }
    }
    (steps, depth)
}

/// How far the bounds are from settling the claim: the smaller, the likelier a counterexample.
fn slack(claim: &Claim, lo: &[f32], hi: &[f32]) -> f64 {
    let rival = |of: &[f32], k: usize| {
        (0..of.len())
            .filter(|&j| j != k)
            .map(|j| f64::from(of[j]))
            .fold(f64::NEG_INFINITY, f64::max)
    };
    match *claim {
        Claim::ProbabilityAtMost(_) | Claim::Negative { .. } | Claim::ValueAtMost { .. } => -f64::from(hi[0]),
        Claim::ProbabilityAtLeast(_) | Claim::Positive { .. } | Claim::ValueAtLeast { .. } => f64::from(lo[0]),
        Claim::Class(k) => f64::from(lo[k]) - rival(hi, k),
        Claim::NotClass(k) => rival(lo, k) - f64::from(hi[k]),
    }
}

/// Which way each output should move to break the claim: 1 up, -1 down, 0 either.
fn directions(claim: &Claim, lo: &[f32], hi: &[f32]) -> Vec<i8> {
    let k = lo.len();
    match *claim {
        Claim::ProbabilityAtMost(_) | Claim::Negative { .. } | Claim::ValueAtMost { .. } => vec![1],
        Claim::ProbabilityAtLeast(_) | Claim::Positive { .. } | Claim::ValueAtLeast { .. } => vec![-1],
        Claim::Class(c) => {
            let rival = (0..k).filter(|&j| j != c).max_by(|&a, &b| hi[a].total_cmp(&hi[b])).unwrap_or(c);
            let mut d = vec![0; k];
            d[c] = -1;
            d[rival] = 1;
            d
        }
        Claim::NotClass(c) => {
            let mut d = vec![-1; k];
            d[c] = 1;
            d
        }
    }
}

/// The search over a tree ensemble.
pub(crate) struct TreeSearch<'a> {
    m: &'a TreeEnsemble,
    base: &'a [f64],
    domain: Domain<'a>,
    claim: Claim,
    k: usize,
    /// For each feature, the trees that split on it.
    feature_trees: Vec<Vec<u32>>,
    /// For each category, the trees that split on any of its slots.
    category_trees: Vec<Vec<u32>>,
    stack: Vec<u32>,
    p: Vec<f64>,
}

/// For each of the `n` features, the trees that split on it, in order.
fn trees_by_feature(m: &TreeEnsemble, n: usize) -> Vec<Vec<u32>> {
    let mut feature_trees: Vec<Vec<u32>> = vec![Vec::new(); n];
    let mut seen = vec![usize::MAX; n];
    for t in 0..m.roots.len() {
        let (start, end) = span(m, t);
        for i in start..end {
            if let Ok(f) = usize::try_from(m.feature[i])
                && seen[f] != t
            {
                seen[f] = t;
                feature_trees[f].push(t as u32);
            }
        }
    }
    feature_trees
}

/// The bounds [`kernels::Trees::bounds`] gives, over one box after another: each tree's
/// [`kernels::Trees::leaf_range`] is kept and computed again only when a feature it reads
/// changed bounds, and the ranges are added in the order `bounds` adds them.
pub(crate) struct Ranges {
    feature_trees: Vec<Vec<u32>>,
    ranges: Vec<[f32; 2]>,
    last_lo: Vec<f32>,
    last_hi: Vec<f32>,
    stale: Vec<bool>,
    reachable: Vec<bool>,
}

impl Ranges {
    pub fn new(m: &TreeEnsemble, n: usize) -> Self {
        Ranges {
            feature_trees: trees_by_feature(m, n),
            ranges: vec![[0.0; 2]; m.roots.len()],
            last_lo: Vec::new(),
            last_hi: Vec::new(),
            stale: vec![true; m.roots.len()],
            reachable: vec![false; m.feature.len()],
        }
    }

    pub fn bounds(&mut self, m: &TreeEnsemble, lo: &[f32], hi: &[f32]) -> (Vec<f32>, Vec<f32>) {
        if self.last_lo.len() == lo.len() {
            for j in 0..lo.len() {
                if lo[j].to_bits() != self.last_lo[j].to_bits() || hi[j].to_bits() != self.last_hi[j].to_bits() {
                    for &t in &self.feature_trees[j] {
                        self.stale[t as usize] = true;
                    }
                }
            }
        }
        self.last_lo = lo.to_vec();
        self.last_hi = hi.to_vec();
        let base = m.base();
        let kernel = m.kernel(base);
        for t in 0..self.ranges.len() {
            if self.stale[t] {
                self.ranges[t] = kernel.leaf_range(t, lo, hi, &mut self.reachable);
                self.stale[t] = false;
            }
        }
        sum_ranges(base, &self.ranges, m.sums_in_f32())
    }
}

/// The trees' ranges added up per output as [`kernels::Trees::bounds`] adds them (in f64,
/// rounded at the end) or, with `in_f32`, as `bounds_in_f32` does (in f32).
fn sum_ranges(base: &[f64], ranges: &[[f32; 2]], in_f32: bool) -> (Vec<f32>, Vec<f32>) {
    let k = base.len();
    let (mut lo, mut hi) = (vec![0f32; k], vec![0f32; k]);
    for o in 0..k {
        if in_f32 {
            let (mut min_sum, mut max_sum) = (base[o] as f32, base[o] as f32);
            for r in ranges.iter().skip(o).step_by(k) {
                min_sum += r[0];
                max_sum += r[1];
            }
            (lo[o], hi[o]) = (min_sum, max_sum);
        } else {
            let (mut min_sum, mut max_sum) = (base[o], base[o]);
            for r in ranges.iter().skip(o).step_by(k) {
                min_sum += f64::from(r[0]);
                max_sum += f64::from(r[1]);
            }
            (lo[o], hi[o]) = (min_sum as f32, max_sum as f32);
        }
    }
    (lo, hi)
}

impl<'a> TreeSearch<'a> {
    pub fn new(m: &'a TreeEnsemble, base: &'a [f64], domain: Domain<'a>, claim: Claim) -> Self {
        let feature_trees = trees_by_feature(m, domain.category.len());
        let category_trees = domain
            .slots
            .iter()
            .map(|&[first, end]| {
                let mut trees: Vec<u32> = (first as usize..end as usize)
                    .flat_map(|j| feature_trees[j].iter().copied())
                    .collect();
                trees.sort_unstable();
                trees.dedup();
                trees
            })
            .collect();
        let k = base.len();
        TreeSearch {
            m,
            base,
            domain,
            claim,
            k,
            feature_trees,
            category_trees,
            stack: Vec::with_capacity(256),
            p: vec![0.0; k],
        }
    }

    /// The smallest and the largest leaf below `node` that a row of the box reaches: what
    /// [`kernels::Trees::leaf_range`] computes for a whole tree, visiting only reachable nodes.
    fn extremes(&mut self, node: u32, b: &Bx) -> [f32; 2] {
        let m = self.m;
        let (mut min, mut max) = (f32::INFINITY, f32::NEG_INFINITY);
        self.stack.clear();
        self.stack.push(node);
        while let Some(i) = self.stack.pop() {
            let i = i as usize;
            let Ok(f) = usize::try_from(m.feature[i]) else {
                let v = m.value[i];
                if v < min {
                    min = v;
                }
                if v > max {
                    max = v;
                }
                continue;
            };
            let t = m.threshold[i];
            if b.hi[f] > t {
                self.stack.push(m.right[i]);
            }
            if b.lo[f] <= t {
                self.stack.push(m.left[i]);
            }
        }
        [min, max]
    }

    fn ranges(&mut self, b: &Bx) -> Vec<[f32; 2]> {
        (0..self.m.roots.len()).map(|t| self.extremes(self.m.roots[t], b)).collect()
    }

    /// `ranges` after the box changed by a split on `feature`.
    fn update(&mut self, b: &Bx, ranges: &mut [[f32; 2]], feature: usize) {
        let c = self.domain.category[feature];
        let trees = if c == NUMERIC {
            std::mem::take(&mut self.feature_trees[feature])
        } else {
            std::mem::take(&mut self.category_trees[c as usize])
        };
        for &t in &trees {
            ranges[t as usize] = self.extremes(self.m.roots[t as usize], b);
        }
        if c == NUMERIC {
            self.feature_trees[feature] = trees;
        } else {
            self.category_trees[c as usize] = trees;
        }
    }

    /// The bounds [`kernels::Trees::bounds`] (or `bounds_in_f32`) gives for the box, from its
    /// trees' ranges: the same sums in the same order.
    fn bounds(&self, ranges: &[[f32; 2]]) -> (Vec<f32>, Vec<f32>) {
        sum_ranges(self.base, ranges, self.m.sums_in_f32())
    }

    /// The first split of tree `t` that the box reaches on both sides.
    fn split_of(&self, t: usize, b: &Bx) -> Option<(usize, f32)> {
        let m = self.m;
        let mut i = m.roots[t] as usize;
        loop {
            let f = usize::try_from(m.feature[i]).ok()?;
            let th = m.threshold[i];
            match (b.lo[f] <= th, b.hi[f] > th) {
                (true, true) => return Some((f, th)),
                (true, false) => i = m.left[i] as usize,
                _ => i = m.right[i] as usize,
            }
        }
    }

    /// A box where every tree reaches leaves of one value, found greedily: each tree, widest
    /// first, follows the side that moves its output the way that breaks the claim.
    fn dive(&mut self, b: &Bx, ranges: &[[f32; 2]]) -> Option<Bx> {
        let (lo_out, hi_out) = self.bounds(ranges);
        let dir = directions(&self.claim, &lo_out, &hi_out);
        let mut order: Vec<usize> = (0..ranges.len()).filter(|&t| ranges[t][1] > ranges[t][0]).collect();
        order.sort_by(|&a, &c| (ranges[c][1] - ranges[c][0]).total_cmp(&(ranges[a][1] - ranges[a][0])));
        let mut d = b.clone();
        for t in order {
            let m = self.m;
            let up = dir[t % self.k];
            let mut i = m.roots[t] as usize;
            while let Ok(f) = usize::try_from(m.feature[i]) {
                let th = m.threshold[i];
                let left = match (d.lo[f] <= th, d.hi[f] > th) {
                    (true, true) if up != 0 => {
                        let l = self.extremes(m.left[i], &d);
                        let r = self.extremes(m.right[i], &d);
                        if up > 0 { l[1] >= r[1] } else { l[0] <= r[0] }
                    }
                    (true, _) => true,
                    _ => false,
                };
                rules::narrow(&self.domain, &mut d.lo, &mut d.hi, &mut d.none, f, th, left);
                i = if left { m.left[i] } else { m.right[i] } as usize;
            }
            if d.is_empty(&self.domain) {
                return None;
            }
        }
        Some(d)
    }

    /// Searches the region for a proof or a counterexample. `confirm` makes an input of a box
    /// where the claim fails and predicts it.
    pub fn run<W>(&mut self, region: Bx, limits: &Limits, confirm: &mut Confirm<'_, W>) -> Result<(Found<W>, usize)> {
        let mut nodes = vec![Node::Open];
        let ranges = self.ranges(&region);
        let mut stack = vec![(region, ranges, 0u32)];
        let mut visited = 0usize;
        while let Some((b, ranges, id)) = stack.pop() {
            visited += 1;
            if let Some(why) = limits.reached(visited) {
                return Ok((Found::Undecided(why), visited - 1));
            }
            if b.is_empty(&self.domain) {
                nodes[id as usize] = Node::Close;
                continue;
            }
            let (lo_out, hi_out) = self.bounds(&ranges);
            if self.claim.holds_within(&lo_out, &hi_out, &mut self.p) {
                nodes[id as usize] = Node::Close;
                continue;
            }
            let widest = (0..ranges.len()).filter(|&t| ranges[t][1] > ranges[t][0]).max_by(|&a, &c| {
                (ranges[a][1] - ranges[a][0])
                    .total_cmp(&(ranges[c][1] - ranges[c][0]))
                    .then(c.cmp(&a))
            });
            let Some(widest) = widest else {
                // One value per tree: every row of the box gets these outputs, and they break the claim.
                return match confirm(&b, None)? {
                    Check::Fails(w) => Ok((Found::Fails(w), visited)),
                    Check::NoInput => Ok((
                        Found::Undecided("the claim fails only on rows of features that no input encodes to".to_string()),
                        visited,
                    )),
                    Check::Holds => Err(tabsmith_contracts::TabsmithError::internal(
                        tabsmith_contracts::Phase::Audit,
                        "an input of a box where every tree's leaf is known predicts something else than the box's bounds",
                    )),
                };
            };
            if visited % 64 == 1
                && let Some(d) = self.dive(&b, &ranges)
            {
                let r = self.ranges(&d);
                let (lo_d, hi_d) = self.bounds(&r);
                if !self.claim.holds_within(&lo_d, &hi_d, &mut self.p)
                    && let Check::Fails(w) = confirm(&d, None)?
                {
                    return Ok((Found::Fails(w), visited));
                }
            }
            let (f, th) = self.split_of(widest, &b).expect("a tree whose leaves differ splits the box");
            let mut children = Vec::with_capacity(2);
            for left in [true, false] {
                let c = b.narrowed(&self.domain, f, th, left);
                let mut r = ranges.clone();
                self.update(&c, &mut r, f);
                let (lo_c, hi_c) = self.bounds(&r);
                let s = slack(&self.claim, &lo_c, &hi_c);
                let cid = nodes.len() as u32;
                nodes.push(Node::Open);
                children.push((s, c, r, cid));
            }
            nodes[id as usize] = Node::Split {
                feature: f as u32,
                threshold: th,
                left: children[0].3,
                right: children[1].3,
            };
            // The likelier counterexample is searched first.
            children.sort_by(|a, c| c.0.total_cmp(&a.0));
            for (_, c, r, cid) in children {
                stack.push((c, r, cid));
            }
        }
        let (steps, depth) = serialize(&nodes);
        Ok((Found::Proof { steps, depth }, visited))
    }
}

/// The node range of tree `t`.
fn span(m: &TreeEnsemble, t: usize) -> (usize, usize) {
    let start = m.roots[t] as usize;
    let end = m.roots.get(t + 1).map_or(m.feature.len(), |&r| r as usize);
    (start, end)
}

/// The bounds of a network's outputs over a box: [`kernels::dense_bounds`] through the
/// layers, as a compiled model's `output_bounds` computes them.
pub(crate) fn network_bounds(m: &DenseModel, lo: &[f32], hi: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let (mut lo, mut hi) = (lo.to_vec(), hi.to_vec());
    for l in &m.layers {
        let (mut lo2, mut hi2) = (vec![0f32; l.out_dim], vec![0f32; l.out_dim]);
        let relu = matches!(l.activation, tabsmith_contracts::model::Activation::Relu);
        kernels::dense_bounds(&l.weight, &l.bias, &lo, &hi, relu, &mut lo2, &mut hi2);
        (lo, hi) = (lo2, hi2);
    }
    (lo, hi)
}

/// Farthest a candidate point goes along a feature without a bound, in standard deviations.
const FAR: f32 = 1e4;

/// The search over a network.
pub(crate) struct NetSearch<'a> {
    m: &'a DenseModel,
    domain: Domain<'a>,
    claim: Claim,
    p: Vec<f64>,
}

impl<'a> NetSearch<'a> {
    pub fn new(m: &'a DenseModel, domain: Domain<'a>, claim: Claim) -> Self {
        NetSearch {
            m,
            domain,
            claim,
            p: vec![0.0; m.config.n_outputs()],
        }
    }

    fn outputs(&self, x: &[f32]) -> Vec<f32> {
        let mut cur = x.to_vec();
        for l in &self.m.layers {
            let mut next = vec![0f32; l.out_dim];
            let relu = matches!(l.activation, tabsmith_contracts::model::Activation::Relu);
            kernels::dense(&l.weight, &l.bias, &cur, relu, &mut next);
            cur = next;
        }
        cur
    }

    /// The gradient of `dir · outputs` at `x`, through the ReLUs active there.
    fn gradient(&self, x: &[f32], dir: &[i8]) -> Vec<f64> {
        let mut inputs = vec![x.to_vec()];
        let mut active = Vec::new();
        for l in &self.m.layers {
            let mut out = vec![0f32; l.out_dim];
            kernels::dense(&l.weight, &l.bias, inputs.last().expect("one per layer"), false, &mut out);
            let relu = matches!(l.activation, tabsmith_contracts::model::Activation::Relu);
            active.push(out.iter().map(|&v| !relu || v > 0.0).collect::<Vec<bool>>());
            if relu {
                out.iter_mut().for_each(|v| *v = v.max(0.0));
            }
            inputs.push(out);
        }
        let mut g: Vec<f64> = dir.iter().map(|&d| f64::from(d)).collect();
        for (l, on) in self.m.layers.iter().zip(&active).rev() {
            let mut back = vec![0f64; l.in_dim];
            for o in 0..l.out_dim {
                if !on[o] || g[o] == 0.0 {
                    continue;
                }
                for i in 0..l.in_dim {
                    back[i] += f64::from(l.weight[o * l.in_dim + i]) * g[o];
                }
            }
            g = back;
        }
        g
    }

    /// For each category of the box, its possible values: `Some(slot)` or `None`.
    fn choices(&self, b: &Bx, c: usize) -> Vec<Option<usize>> {
        let [first, end] = self.domain.slots[c];
        let (first, end) = (first as usize, end as usize);
        if let Some(s) = (first..end).find(|&s| b.lo[s] > 0.0) {
            return vec![Some(s)];
        }
        let mut out: Vec<Option<usize>> = (first..end).filter(|&s| b.lo[s] <= 1.0 && 1.0 <= b.hi[s]).map(Some).collect();
        if b.none[c] {
            out.push(None);
        }
        out
    }

    /// A row of the box: numbers from `pick`, each category's value with the largest `g`
    /// (its first one without a gradient).
    fn point(&self, b: &Bx, g: Option<&[f64]>, pick: &dyn Fn(f32, f32, f64) -> f32) -> (Vec<f32>, Vec<Option<usize>>) {
        let mut x = vec![0f32; b.lo.len()];
        for j in 0..x.len() {
            if self.domain.category[j] == NUMERIC {
                x[j] = pick(b.lo[j], b.hi[j], g.map_or(0.0, |g| g[j]));
            }
        }
        let mut chosen = Vec::with_capacity(self.domain.slots.len());
        for c in 0..self.domain.slots.len() {
            let score = |ch: &Option<usize>| match (g, ch) {
                (Some(g), Some(s)) => g[*s],
                _ => 0.0,
            };
            let options = self.choices(b, c);
            let best = options.iter().copied().fold(None, |best: Option<Option<usize>>, ch| match best {
                Some(b) if score(&b) >= score(&ch) => Some(b),
                _ => Some(ch),
            });
            let ch = best.flatten();
            if let Some(s) = ch {
                x[s] = 1.0;
            }
            chosen.push(ch);
        }
        (x, chosen)
    }

    /// A numeric feature's split: a threshold with values of the box on both sides. A range
    /// across 0 splits at 0 (the training mean); a range far wider than its distance from 0
    /// splits at about twice that distance, so an unbounded feature is cut into pieces of
    /// growing size rather than halved from `f32::MAX` down.
    fn split_point(lo: f32, hi: f32) -> Option<f32> {
        if lo >= hi || lo.is_nan() || hi.is_nan() {
            return None;
        }
        let (a, z) = (f64::from(lo), f64::from(hi));
        let t = if a < 0.0 && z > 0.0 {
            0.0
        } else if a >= 0.0 && z > 2.0 * a + 2.0 {
            2.0 * a + 1.0
        } else if z <= 0.0 && a < 2.0 * z - 2.0 {
            2.0 * z - 1.0
        } else {
            a / 2.0 + z / 2.0
        };
        let t = (t as f32).clamp(lo, hi.next_down());
        (lo <= t && t < hi).then_some(t)
    }

    /// The split that the gradient says matters most: a number's sensitivity times its width,
    /// or the spread of a category's values.
    fn split_of(&self, b: &Bx, g: &[f64]) -> Option<(usize, f32)> {
        let mut best: Option<(f64, usize, f32)> = None;
        let mut consider = |score: f64, f: usize, t: f32| {
            if best.is_none_or(|(s, _, _)| score > s) {
                best = Some((score, f, t));
            }
        };
        for j in 0..b.lo.len() {
            if self.domain.category[j] != NUMERIC {
                continue;
            }
            if let Some(t) = Self::split_point(b.lo[j], b.hi[j]) {
                let width = f64::from(b.hi[j]) - f64::from(b.lo[j]);
                // Without a gradient, the widest feature; an unbounded one first.
                consider(g[j].abs() * width + width * 1e-12, j, t);
            }
        }
        for c in 0..self.domain.slots.len() {
            let options = self.choices(b, c);
            if options.len() < 2 {
                continue;
            }
            let score = |ch: &Option<usize>| ch.map_or(0.0, |s| g[s]);
            let (min, max) = options.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, z), ch| {
                (a.min(score(ch)), z.max(score(ch)))
            });
            let slot = options
                .iter()
                .flatten()
                .copied()
                .max_by(|&a, &c| g[a].total_cmp(&g[c]).then(c.cmp(&a)));
            if let Some(s) = slot {
                consider(max - min + 1e-12, s, 0.0);
            }
        }
        best.map(|(_, f, t)| (f, t))
    }

    /// Searches the region for a proof or a counterexample. `confirm` makes an input near a
    /// point of a box where the claim fails and predicts it.
    pub fn run<W>(&mut self, region: Bx, limits: &Limits, confirm: &mut Confirm<'_, W>) -> Result<(Found<W>, usize)> {
        let mut nodes = vec![Node::Open];
        let mut stack = vec![(region, 0u32)];
        let mut visited = 0usize;
        // Candidates stay within FAR standard deviations where the box allows.
        let mid = |lo: f32, hi: f32, _: f64| {
            let (a, z) = (lo.max(-FAR), hi.min(FAR));
            if a <= z {
                ((f64::from(a) + f64::from(z)) / 2.0) as f32
            } else {
                lo.max(hi.min(0.0))
            }
        };
        let corner = |lo: f32, hi: f32, g: f64| {
            let (a, z) = (lo.max(-FAR), hi.min(FAR));
            if a > z || g == 0.0 {
                mid(lo, hi, g)
            } else if g > 0.0 {
                z
            } else {
                a
            }
        };
        while let Some((b, id)) = stack.pop() {
            visited += 1;
            if let Some(why) = limits.reached(visited) {
                return Ok((Found::Undecided(why), visited - 1));
            }
            if b.is_empty(&self.domain) {
                nodes[id as usize] = Node::Close;
                continue;
            }
            let (lo_out, hi_out) = network_bounds(self.m, &b.lo, &b.hi);
            if self.claim.holds_within(&lo_out, &hi_out, &mut self.p) {
                nodes[id as usize] = Node::Close;
                continue;
            }
            let dir = directions(&self.claim, &lo_out, &hi_out);
            let (center, _) = self.point(&b, None, &mid);
            let g = self.gradient(&center, &dir);
            for (x, chosen) in [self.point(&b, None, &mid), self.point(&b, Some(&g), &corner)] {
                let z = self.outputs(&x);
                if !self.claim.holds_at(&z, &mut self.p)
                    && let Check::Fails(w) = confirm(&b, Some((&x, &chosen)))?
                {
                    return Ok((Found::Fails(w), visited));
                }
            }
            let Some((f, th)) = self.split_of(&b, &g) else {
                return Ok((
                    Found::Undecided("a box of single values stays open: interval bounds cannot settle it".to_string()),
                    visited,
                ));
            };
            let mut children = Vec::with_capacity(2);
            for left in [true, false] {
                let c = b.narrowed(&self.domain, f, th, left);
                let (lo_c, hi_c) = network_bounds(self.m, &c.lo, &c.hi);
                let cid = nodes.len() as u32;
                nodes.push(Node::Open);
                children.push((slack(&self.claim, &lo_c, &hi_c), c, cid));
            }
            nodes[id as usize] = Node::Split {
                feature: f as u32,
                threshold: th,
                left: children[0].2,
                right: children[1].2,
            };
            children.sort_by(|a, c| c.0.total_cmp(&a.0));
            for (_, c, cid) in children {
                stack.push((c, cid));
            }
        }
        let (steps, depth) = serialize(&nodes);
        Ok((Found::Proof { steps, depth }, visited))
    }
}

#[cfg(test)]
mod tests {
    use tabsmith_contracts::config::Task;
    use tabsmith_contracts::model::{ModelConfig, TreesSpec};

    use super::*;
    use crate::audit::{Certificate, FeatureDomain};
    use crate::model::Model;

    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }
    }

    /// Features 0 and 1 numeric; 2..5 a category of three slots; 5..7 one of two.
    const CATEGORY: [u32; 7] = [NUMERIC, NUMERIC, 0, 0, 0, 1, 1];
    const SLOTS: [[u32; 2]; 2] = [[2, 5], [5, 7]];

    fn grow(rng: &mut Rng, depth: usize, nodes: &mut Vec<(i32, f32, u32, u32, f32)>) -> u32 {
        let id = nodes.len();
        nodes.push((-1, 0.0, 0, 0, 0.0));
        if depth == 0 || rng.below(5) == 0 {
            // Leaves from a few values, so that sums tie.
            nodes[id].4 = (rng.below(9) as f32 - 4.0) * 0.125;
            return id as u32;
        }
        let f = rng.below(7) as usize;
        let t = if CATEGORY[f] == NUMERIC {
            (rng.below(9) as f32 - 4.0) * 0.5
        } else {
            [0.0, 0.5][rng.below(2) as usize]
        };
        let left = grow(rng, depth - 1, nodes);
        let right = grow(rng, depth - 1, nodes);
        nodes[id] = (f as i32, t, left, right, 0.0);
        id as u32
    }

    fn ensemble(rng: &mut Rng, k: usize) -> TreeEnsemble {
        let n_trees = k * (1 + rng.below(4) as usize);
        let mut nodes = Vec::new();
        let mut roots = Vec::new();
        for _ in 0..n_trees {
            let start = nodes.len();
            roots.push(start as u32);
            let mut tree = Vec::new();
            grow(rng, 4, &mut tree);
            for (f, t, l, r, v) in tree {
                let shift = |c: u32| if f >= 0 { c + start as u32 } else { 0 };
                nodes.push((f, t, shift(l), shift(r), v));
            }
        }
        let spec = TreesSpec {
            n_trees,
            n_nodes: nodes.len(),
            max_depth: 4,
            base_score: 0.25,
            n_outputs: k,
            base_scores: if k == 1 {
                Vec::new()
            } else {
                (0..k).map(|o| o as f64 * 0.125).collect()
            },
            monotone: Vec::new(),
            sum: Default::default(),
        };
        let task = if k == 1 { Task::Binary } else { Task::Multiclass };
        TreeEnsemble::new(
            ModelConfig::trees_for(7, task, Some(spec)),
            roots,
            nodes.iter().map(|n| n.0).collect(),
            nodes.iter().map(|n| n.1).collect(),
            nodes.iter().map(|n| n.2).collect(),
            nodes.iter().map(|n| n.3).collect(),
            nodes.iter().map(|n| n.4).collect(),
        )
        .unwrap()
    }

    /// A random region: numeric ranges, a subset of each category's values, `none` or not.
    fn region(rng: &mut Rng, domain: &Domain<'_>) -> Bx {
        let mut b = Bx {
            lo: vec![-f32::MAX, -f32::MAX, 0.0, 0.0, 0.0, 0.0, 0.0],
            hi: vec![f32::MAX, f32::MAX, 1.0, 1.0, 1.0, 1.0, 1.0],
            none: vec![rng.below(2) == 0, rng.below(2) == 0],
        };
        for j in 0..2 {
            if rng.below(2) == 0 {
                let (a, c) = ((rng.below(9) as f32 - 4.0) * 0.5, (rng.below(9) as f32 - 4.0) * 0.5);
                (b.lo[j], b.hi[j]) = (a.min(c), a.max(c) + 0.25);
            }
        }
        for j in 2..7 {
            if rng.below(4) == 0 {
                b.hi[j] = 0.0;
            }
        }
        for (c, &[first, end]) in domain.slots.iter().enumerate() {
            let open: Vec<usize> = (first as usize..end as usize).filter(|&j| b.hi[j] == 1.0).collect();
            if open.len() == 1 && !b.none[c] {
                rules::narrow(domain, &mut b.lo, &mut b.hi, &mut b.none, open[0], 0.0, false);
            }
        }
        b
    }

    fn claim(rng: &mut Rng, k: usize) -> Claim {
        let c = (rng.below(9) as f64 - 4.0) * 0.25;
        if k > 1 {
            let class = rng.below(k as u64) as usize;
            return if rng.below(2) == 0 {
                Claim::Class(class)
            } else {
                Claim::NotClass(class)
            };
        }
        let p = kernels::sigmoid(c);
        match rng.below(6) {
            0 => Claim::ProbabilityAtMost(p),
            1 => Claim::ProbabilityAtLeast(p),
            2 => Claim::Positive { threshold: p },
            3 => Claim::Negative { threshold: p },
            4 => Claim::ValueAtMost {
                limit: c * 3.0 + 10.0,
                mean: 10.0,
                std: 3.0,
            },
            _ => Claim::ValueAtLeast {
                limit: c * 3.0 + 10.0,
                mean: 10.0,
                std: 3.0,
            },
        }
    }

    /// Every row of the region, one per cell of the trees' thresholds.
    fn rows(m: &TreeEnsemble, domain: &Domain<'_>, b: &Bx) -> Vec<Vec<f32>> {
        let mut numbers: Vec<Vec<f32>> = Vec::new();
        for j in 0..2 {
            let mut v = vec![b.lo[j], b.hi[j]];
            for i in 0..m.feature.len() {
                if m.feature[i] == j as i32 {
                    v.extend([m.threshold[i], m.threshold[i].next_up()]);
                }
            }
            v.retain(|&x| b.lo[j] <= x && x <= b.hi[j]);
            v.sort_by(f32::total_cmp);
            v.dedup();
            numbers.push(v);
        }
        let mut values: Vec<Vec<Option<usize>>> = Vec::new();
        for (c, &[first, end]) in domain.slots.iter().enumerate() {
            let forced = (first as usize..end as usize).find(|&s| b.lo[s] > 0.0);
            let mut v: Vec<Option<usize>> = match forced {
                Some(s) => vec![Some(s)],
                None => (first as usize..end as usize).filter(|&s| b.hi[s] >= 1.0).map(Some).collect(),
            };
            if forced.is_none() && b.none[c] {
                v.push(None);
            }
            values.push(v);
        }
        let mut out = Vec::new();
        for &x0 in &numbers[0] {
            for &x1 in &numbers[1] {
                for &a in &values[0] {
                    for &c in &values[1] {
                        let mut row = vec![x0, x1, 0.0, 0.0, 0.0, 0.0, 0.0];
                        for s in [a, c].into_iter().flatten() {
                            row[s] = 1.0;
                        }
                        out.push(row);
                    }
                }
            }
        }
        out
    }

    fn holds_at(m: &TreeEnsemble, claim: &Claim, row: &[f32]) -> bool {
        let k = m.config.n_outputs();
        let mut z = vec![0f32; k];
        m.outputs_row(row, &mut z);
        claim.holds_at(&z, &mut vec![0f64; k])
    }

    #[test]
    fn the_tree_search_agrees_with_every_row_of_small_ensembles() {
        let domain = Domain {
            category: &CATEGORY,
            slots: &SLOTS,
        };
        let owned = FeatureDomain {
            category: CATEGORY.to_vec(),
            slots: SLOTS.to_vec(),
            none: true,
        };
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        let (mut proved, mut refuted) = (0, 0);
        for case in 0..400 {
            let k = if case % 3 == 0 { 3 } else { 1 };
            let m = ensemble(&mut rng, k);
            let b = region(&mut rng, &domain);
            let claim = claim(&mut rng, k);
            let all = rows(&m, &domain, &b);
            let broken = all.iter().find(|r| !holds_at(&m, &claim, r));
            let mut confirm = |bx: &Bx, _: Option<(&[f32], &[Option<usize>])>| -> Result<Check<Vec<f32>>> {
                // A row of the box: the lower ends, a possible slot or none per category.
                let mut row = bx.lo.clone();
                for (c, &[first, end]) in SLOTS.iter().enumerate() {
                    let slot = (first as usize..end as usize)
                        .find(|&s| bx.lo[s] > 0.0)
                        .or_else(|| (first as usize..end as usize).find(|&s| bx.hi[s] >= 1.0).filter(|_| !bx.none[c]))
                        .or_else(|| if bx.none[c] { None } else { Some(first as usize) });
                    for s in first as usize..end as usize {
                        row[s] = if Some(s) == slot { 1.0 } else { 0.0 };
                    }
                }
                Ok(if holds_at(&m, &claim, &row) {
                    Check::Holds
                } else {
                    Check::Fails(row)
                })
            };
            let base = m.base().to_vec();
            let limits = Limits {
                boxes: 1_000_000,
                deadline: None,
            };
            let (found, _) = TreeSearch::new(&m, &base, domain, claim)
                .run(b.clone(), &limits, &mut confirm)
                .unwrap();
            match found {
                Found::Proof { steps, depth } => {
                    assert!(broken.is_none(), "case {case}: proved, but {broken:?} breaks {claim:?}");
                    let proof = Certificate {
                        claim,
                        lo: b.lo.clone(),
                        hi: b.hi.clone(),
                        none: b.none.clone(),
                        steps,
                        depth,
                    };
                    assert!(
                        proof.check(&Model::Trees(m.clone()), &owned),
                        "case {case}: the proof does not check"
                    );
                    proved += 1;
                }
                Found::Fails(row) => {
                    assert!(broken.is_some(), "case {case}: refuted by {row:?}, but every row holds");
                    assert!(
                        row.iter().enumerate().all(|(j, &x)| b.lo[j] <= x && x <= b.hi[j]),
                        "case {case}: {row:?} outside"
                    );
                    assert!(
                        !rules::is_empty(&domain, &row, &row, &b.none),
                        "case {case}: {row:?} is not an input"
                    );
                    refuted += 1;
                }
                Found::Undecided(why) => panic!("case {case}: {why}"),
            }
        }
        assert!(proved > 80 && refuted > 80, "{proved} proved, {refuted} refuted");
    }

    #[test]
    fn kept_ranges_and_the_search_give_the_bits_of_the_kernel() {
        let domain = Domain {
            category: &CATEGORY,
            slots: &SLOTS,
        };
        let mut rng = Rng(0x1234_5678_9abc_def1);
        for case in 0..200 {
            let k = if case % 2 == 0 { 3 } else { 1 };
            let m = ensemble(&mut rng, k);
            let base = m.base().to_vec();
            let mut kept = Ranges::new(&m, 7);
            let mut search = TreeSearch::new(&m, &base, domain, Claim::NotClass(0));
            let mut b = region(&mut rng, &domain);
            for _ in 0..12 {
                let (mut lo, mut hi) = (vec![0f32; k], vec![0f32; k]);
                let mut reachable = vec![false; m.feature.len()];
                m.kernel(&base).bounds(&b.lo, &b.hi, &mut reachable, &mut lo, &mut hi);
                let r = search.ranges(&b);
                for (got, name) in [(kept.bounds(&m, &b.lo, &b.hi), "kept"), (search.bounds(&r), "search")] {
                    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
                    assert_eq!((bits(&got.0), bits(&got.1)), (bits(&lo), bits(&hi)), "case {case}, {name}");
                }
                // Narrow the box at a random split, as a walk does.
                let f = rng.below(7) as usize;
                let t = if CATEGORY[f] == NUMERIC {
                    (rng.below(9) as f32 - 4.0) * 0.5
                } else {
                    0.0
                };
                let next = b.narrowed(&domain, f, t, rng.below(2) == 0);
                if !next.is_empty(&domain) {
                    b = next;
                }
            }
        }
    }

    #[test]
    fn a_proof_is_written_left_part_first() {
        // Root splits 0 at 1.0; its left part splits 2 at 0.5; the rest close.
        let nodes = [
            Node::Split {
                feature: 0,
                threshold: 1.0,
                left: 2,
                right: 1,
            },
            Node::Close,
            Node::Split {
                feature: 2,
                threshold: 0.5,
                left: 3,
                right: 4,
            },
            Node::Close,
            Node::Close,
        ];
        let (steps, depth) = serialize(&nodes);
        assert_eq!(
            steps,
            vec![
                Step::Split {
                    feature: 0,
                    threshold: 1.0
                },
                Step::Split {
                    feature: 2,
                    threshold: 0.5
                },
                Step::Close,
                Step::Close,
                Step::Close
            ]
        );
        assert_eq!(depth, 2);
        assert_eq!(serialize(&[Node::Close]), (vec![Step::Close], 0));
    }

    #[test]
    fn a_split_point_leaves_values_on_both_sides() {
        for (lo, hi) in [
            (0.0f32, 1.0f32),
            (1.0, 1.0f32.next_up()),
            (-f32::MAX, f32::MAX),
            (3.0, f32::MAX),
            (-3.0, f32::MAX),
            (-f32::MAX, -2.0),
            (-f32::MAX, 2.0),
            (f32::MAX.next_down(), f32::MAX),
        ] {
            let t = NetSearch::split_point(lo, hi).unwrap();
            assert!(lo <= t && t < hi, "{lo} {hi}: {t}");
        }
        assert_eq!(NetSearch::split_point(-f32::MAX, f32::MAX), Some(0.0));
        assert_eq!(NetSearch::split_point(3.0, f32::MAX), Some(7.0));
        assert_eq!(NetSearch::split_point(1.0, 1.0), None);
        assert_eq!(NetSearch::split_point(-0.0, 0.0), None, "one number");
        assert_eq!(NetSearch::split_point(2.0, 1.0), None);
    }
}
