//! Numeric kernels. Each one fixes its order of operations, so a given input gives
//! the same bits wherever it runs; changing an order here changes every prediction.

use crate::InputError;

/// Dot product of two equally long slices: eight partial sums over chunks of eight,
/// combined pairwise, then the tail.
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
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

/// One dense layer: `out[o] = dot(weight[o], input) + bias[o]`, then ReLU when `relu`.
/// `weight` is row-major `[out.len(), input.len()]`.
pub fn dense(weight: &[f32], bias: &[f32], input: &[f32], relu: bool, out: &mut [f32]) {
    let n = input.len();
    for (o, slot) in out.iter_mut().enumerate() {
        let v = dot(&weight[o * n..(o + 1) * n], input) + bias[o];
        *slot = if relu { v.max(0.0) } else { v };
    }
}

/// A gradient-boosted tree ensemble in the `tabsmith.trees.v1` layout: every tree's
/// nodes in one set of arrays. Node `i` is a leaf when `feature[i] < 0`; otherwise a
/// row with `x[feature[i]] <= threshold[i]` continues at `left[i]`, others at `right[i]`.
/// Children always come after their parent, inside the same tree.
#[derive(Debug, Clone, Copy)]
pub struct Trees<'a> {
    /// Index of each tree's root node, ascending from 0.
    pub roots: &'a [u32],
    /// Split feature of each node, or -1 for a leaf.
    pub feature: &'a [i32],
    /// Split threshold of each node.
    pub threshold: &'a [f32],
    /// Left child of each split node.
    pub left: &'a [u32],
    /// Right child of each split node.
    pub right: &'a [u32],
    /// Value of each leaf, already scaled by the learning rate.
    pub value: &'a [f32],
    /// Starting score of each output.
    pub base: &'a [f64],
}

impl Trees<'_> {
    /// The leaf of tree `t` that `row` reaches.
    pub fn leaf(&self, t: usize, row: &[f32]) -> usize {
        let mut i = self.roots[t] as usize;
        while self.feature[i] >= 0 {
            i = if row[self.feature[i] as usize] <= self.threshold[i] {
                self.left[i]
            } else {
                self.right[i]
            } as usize;
        }
        i
    }

    /// The outputs of one row into `out`, one per output: the base score plus the
    /// leaf of every tree of that output (tree `t` belongs to output `t % out.len()`),
    /// summed in f64 in tree order and rounded to f32 at the end.
    pub fn outputs(&self, row: &[f32], out: &mut [f32]) {
        let k = out.len();
        for (o, slot) in out.iter_mut().enumerate() {
            let mut sum = self.base[o];
            let mut t = o;
            while t < self.roots.len() {
                sum += f64::from(self.value[self.leaf(t, row)]);
                t += k;
            }
            *slot = sum as f32;
        }
    }

    /// Bounds of [`outputs`](Self::outputs) over every row with `lo[j] <= row[j] <= hi[j]`
    /// for each feature `j`, into `lo_out` and `hi_out` (one entry per output). A NaN
    /// in `lo` or `hi` lets the feature be NaN, which every split sends right.
    ///
    /// Each tree contributes the smallest and the largest of the leaves the box can
    /// reach, and the sums run in the order `outputs` uses. Rounding is monotone, so
    /// the sum any row of the box produces lies in between: the bounds are sound, with
    /// no error margin, though not necessarily reached. A box that reaches no leaf of
    /// some tree (one with `lo > hi`) gives `+inf > -inf`.
    ///
    /// `reachable` is scratch space with one entry per node; it is left all `false`.
    /// `const`, so a compiled model can bound itself while it compiles.
    pub const fn bounds(&self, lo: &[f32], hi: &[f32], reachable: &mut [bool], lo_out: &mut [f32], hi_out: &mut [f32]) {
        let k = lo_out.len();
        let (n_trees, n_nodes) = (self.roots.len(), self.feature.len());
        let mut o = 0;
        while o < k {
            let (mut min_sum, mut max_sum) = (self.base[o], self.base[o]);
            let mut t = o;
            while t < n_trees {
                let start = self.roots[t] as usize;
                let end = if t + 1 < n_trees { self.roots[t + 1] as usize } else { n_nodes };
                let (mut min, mut max) = (f32::INFINITY, f32::NEG_INFINITY);
                // Children come after their parent, so one pass in index order visits
                // every reachable node after the split that reached it.
                reachable[start] = true;
                let mut i = start;
                while i < end {
                    if reachable[i] {
                        reachable[i] = false;
                        if self.feature[i] < 0 {
                            let v = self.value[i];
                            if v < min {
                                min = v;
                            }
                            if v > max {
                                max = v;
                            }
                        } else {
                            let f = self.feature[i] as usize;
                            if lo[f] <= self.threshold[i] {
                                reachable[self.left[i] as usize] = true;
                            }
                            if hi[f] > self.threshold[i] || hi[f].is_nan() {
                                reachable[self.right[i] as usize] = true;
                            }
                        }
                    }
                    i += 1;
                }
                min_sum += min as f64;
                max_sum += max as f64;
                t += k;
            }
            lo_out[o] = min_sum as f32;
            hi_out[o] = max_sum as f32;
            o += 1;
        }
    }
}

/// The unit roundoff of f32: the relative error of one rounded operation is at most this.
const F32_UNIT_ROUNDOFF: f64 = 5.960_464_477_539_063e-8;

/// Bounds of [`dense`] over every input with `lo_in[i] <= input[i] <= hi_in[i]`, into
/// `lo_out` and `hi_out` (one entry per output).
///
/// Interval arithmetic in f64, widened by the worst rounding error of the f32 kernel:
/// `n + 3` roundings of at most one unit roundoff each, relative to the sum of the
/// magnitudes of the products and the bias (this covers every summation order), then a
/// margin for the final rounding to f32. The bounds are sound but not tight. An
/// unbounded input gives unbounded outputs (ReLU keeps the lower bound at 0).
/// `const`, so a compiled model can bound itself while it compiles.
pub const fn dense_bounds(weight: &[f32], bias: &[f32], lo_in: &[f32], hi_in: &[f32], relu: bool, lo_out: &mut [f32], hi_out: &mut [f32]) {
    let n = lo_in.len();
    let mut o = 0;
    while o < lo_out.len() {
        let b = bias[o] as f64;
        let (mut lo, mut hi, mut magnitude) = (b, b, if b < 0.0 { -b } else { b });
        let mut i = 0;
        while i < n {
            let w = weight[o * n + i] as f64;
            if w != 0.0 {
                let (a, c) = (lo_in[i] as f64, hi_in[i] as f64);
                let (p, q) = if w > 0.0 { (w * a, w * c) } else { (w * c, w * a) };
                lo += p;
                hi += q;
                let (a, c) = (if a < 0.0 { -a } else { a }, if c < 0.0 { -c } else { c });
                magnitude += (if w < 0.0 { -w } else { w }) * (if a > c { a } else { c });
            }
            i += 1;
        }
        let error = magnitude * (n as f64 + 3.0) * F32_UNIT_ROUNDOFF;
        let (lo, hi) = (lo - error, hi + error);
        // Rounding to f32 moves a value by at most half an f32 ulp (2⁻²⁴ relative):
        // a 2⁻²³ margin keeps the rounded bound outside.
        let lo = lo - (if lo < 0.0 { -lo } else { lo }) * (2.0 * F32_UNIT_ROUNDOFF);
        let hi = hi + (if hi < 0.0 { -hi } else { hi }) * (2.0 * F32_UNIT_ROUNDOFF);
        let (mut lo, mut hi) = (lo as f32, hi as f32);
        if relu {
            if lo < 0.0 {
                lo = 0.0;
            }
            if hi < 0.0 {
                hi = 0.0;
            }
        }
        lo_out[o] = lo;
        hi_out[o] = hi;
        o += 1;
    }
}

/// One standardised numeric feature: `(x - mean) / std` in f64, rounded to f32.
pub fn standardize(x: f64, mean: f64, std: f64) -> f32 {
    ((x - mean) / std) as f32
}

/// One numeric input encoded as the transform encodes it. A missing value is imputed
/// with `impute`: `None`, a value equal to one of the bundle's numeric missing-value
/// tokens (such as -999), or NaN when `nan_is_missing`. The value is then standardised.
pub fn numeric(
    field: &'static str,
    value: Option<f64>,
    tokens: &[f64],
    nan_is_missing: bool,
    impute: f64,
    mean: f64,
    std: f64,
) -> Result<f32, InputError> {
    let x = match value {
        None => impute,
        Some(v) if tokens.contains(&v) => impute,
        Some(v) if v.is_nan() => {
            if nan_is_missing {
                impute
            } else {
                return Err(InputError::NotANumber { field });
            }
        }
        Some(v) if v.is_infinite() => return Err(InputError::Infinite { field }),
        Some(v) => v,
    };
    let z = standardize(x, mean, std);
    if z.is_finite() {
        Ok(z)
    } else {
        Err(InputError::OutOfRange { field })
    }
}

/// The logistic function in f64, in the form that cannot overflow.
pub fn sigmoid(z: f64) -> f64 {
    if z >= 0.0 {
        1.0 / (1.0 + libm::exp(-z))
    } else {
        let e = libm::exp(z);
        e / (1.0 + e)
    }
}

/// Softmax of `z` into `out` (the same length), in f64, shifted by the maximum.
pub fn softmax(z: &[f32], out: &mut [f64]) {
    let max = f64::from(z.iter().copied().fold(f32::NEG_INFINITY, f32::max));
    for (o, &v) in out.iter_mut().zip(z) {
        *o = libm::exp(f64::from(v) - max);
    }
    let sum: f64 = out.iter().sum();
    for o in out.iter_mut() {
        *o /= sum;
    }
}

/// Index of the largest value, the first one on a tie.
pub fn argmax(p: &[f64]) -> usize {
    let mut best = 0;
    for (k, &v) in p.iter().enumerate() {
        if v > p[best] {
            best = k;
        }
    }
    best
}

/// A regression output back in the target's units: `mean + std * z`.
pub fn unscale(z: f32, mean: f64, std: f64) -> f64 {
    mean + std * f64::from(z)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_has_a_fixed_order() {
        let a: [f32; 11] = core::array::from_fn(|i| 0.1 * i as f32);
        let b: [f32; 11] = core::array::from_fn(|i| 1.0 - 0.05 * i as f32);
        let naive: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        assert!((dot(&a, &b) - naive).abs() < 1e-5);
        assert_eq!(dot(&a, &b).to_bits(), dot(&a, &b).to_bits());
    }

    #[test]
    fn sigmoid_is_stable_and_symmetric() {
        assert_eq!(sigmoid(0.0), 0.5);
        assert!(sigmoid(-1000.0) >= 0.0 && sigmoid(1000.0) <= 1.0);
        assert!((sigmoid(2.0) + sigmoid(-2.0) - 1.0).abs() < 1e-15);
    }

    #[test]
    fn softmax_sums_to_one_and_argmax_takes_the_first_tie() {
        let mut p = [0.0; 3];
        softmax(&[1.0, 3.0, 3.0], &mut p);
        assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-15);
        assert_eq!(argmax(&p), 1);
    }

    /// A small deterministic generator for the bound tests (no proptest here).
    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            self.0 >> 11
        }

        fn unit(&mut self) -> f32 {
            (self.next() % 1_000_001) as f32 / 1_000_000.0
        }

        fn range(&mut self, lo: f32, hi: f32) -> f32 {
            lo + (hi - lo) * self.unit()
        }
    }

    /// Four trees of depth 2 on two features, two outputs: trees 0 and 2 belong to
    /// output 0, trees 1 and 3 to output 1.
    const FOUR_TREES: Trees<'static> = Trees {
        roots: &[0, 7, 14, 21],
        feature: &[
            0, 1, 1, -1, -1, -1, -1, 1, 0, 0, -1, -1, -1, -1, 0, 1, 1, -1, -1, -1, -1, 1, 0, 0, -1, -1, -1, -1,
        ],
        threshold: &[
            0.5, -0.2, 0.3, 0.0, 0.0, 0.0, 0.0, 0.1, 0.4, -0.4, 0.0, 0.0, 0.0, 0.0, -0.3, 0.0, 0.7, 0.0, 0.0, 0.0, 0.0, 0.6, 0.2, -0.8,
            0.0, 0.0, 0.0, 0.0,
        ],
        left: &[
            1, 3, 5, 0, 0, 0, 0, 8, 10, 12, 0, 0, 0, 0, 15, 17, 19, 0, 0, 0, 0, 22, 24, 26, 0, 0, 0, 0,
        ],
        right: &[
            2, 4, 6, 0, 0, 0, 0, 9, 11, 13, 0, 0, 0, 0, 16, 18, 20, 0, 0, 0, 0, 23, 25, 27, 0, 0, 0, 0,
        ],
        value: &[
            0.0, 0.0, 0.0, -1.0, 0.5, 0.25, 2.0, 0.0, 0.0, 0.0, 0.125, -0.75, 1.5, -2.0, 0.0, 0.0, 0.0, 0.3, -0.6, 0.9, -1.2, 0.0, 0.0,
            0.0, 0.7, -0.1, 1.1, -0.4,
        ],
        base: &[0.1, -0.2],
    };

    #[test]
    fn tree_bounds_contain_every_row_of_the_box_and_are_reached_by_some() {
        let mut rng = Lcg(7);
        for _ in 0..200 {
            let (a, b) = (rng.range(-1.0, 1.0), rng.range(-1.0, 1.0));
            let lo = [a.min(b), rng.range(-1.0, 0.0)];
            let hi = [a.max(b), lo[1] + rng.range(0.0, 1.5)];
            let (mut lo_out, mut hi_out, mut scratch) = ([0f32; 2], [0f32; 2], [false; 28]);
            FOUR_TREES.bounds(&lo, &hi, &mut scratch, &mut lo_out, &mut hi_out);
            assert!(scratch.iter().all(|r| !r), "scratch left clean");
            let mut out = [0f32; 2];
            let (mut seen_lo, mut seen_hi) = ([f32::INFINITY; 2], [f32::NEG_INFINITY; 2]);
            for _ in 0..300 {
                let row = [rng.range(lo[0], hi[0]), rng.range(lo[1], hi[1])];
                FOUR_TREES.outputs(&row, &mut out);
                for o in 0..2 {
                    assert!(
                        lo_out[o] <= out[o] && out[o] <= hi_out[o],
                        "{row:?}: {out:?} outside {lo_out:?}..{hi_out:?}"
                    );
                    seen_lo[o] = seen_lo[o].min(out[o]);
                    seen_hi[o] = seen_hi[o].max(out[o]);
                }
            }
            // The box's corners and the thresholds inside it reach the extreme leaves of
            // each tree, but not necessarily all at once: the bound is sound, not tight.
            for o in 0..2 {
                assert!(lo_out[o] <= seen_lo[o] && seen_hi[o] <= hi_out[o]);
            }
        }
    }

    #[test]
    fn tree_bounds_over_everything_are_the_sums_of_the_extreme_leaves() {
        let (mut lo_out, mut hi_out, mut scratch) = ([0f32; 2], [0f32; 2], [false; 28]);
        FOUR_TREES.bounds(&[f32::NEG_INFINITY; 2], &[f32::INFINITY; 2], &mut scratch, &mut lo_out, &mut hi_out);
        // Summed as `outputs` sums: the f64 base, then each f32 leaf widened to f64.
        let sum = |base: f64, a: f32, b: f32| (base + f64::from(a) + f64::from(b)) as f32;
        assert_eq!(lo_out, [sum(0.1, -1.0, -1.2), sum(-0.2, -2.0, -0.4)]);
        assert_eq!(hi_out, [sum(0.1, 2.0, 0.9), sum(-0.2, 1.5, 1.1)]);
        // NaN features go right at every split: the bounds over everything cover them.
        let mut out = [0f32; 2];
        FOUR_TREES.outputs(&[f32::NAN, f32::NAN], &mut out);
        assert!(lo_out[0] <= out[0] && out[0] <= hi_out[0]);
        // A NaN bound stands for "may be NaN": only the right children are reachable.
        FOUR_TREES.bounds(&[f32::NAN; 2], &[f32::NAN; 2], &mut scratch, &mut lo_out, &mut hi_out);
        assert_eq!([lo_out, hi_out], [out, out]);
    }

    #[test]
    fn dense_bounds_contain_every_input_of_the_box() {
        let mut rng = Lcg(11);
        for _ in 0..200 {
            let (n_in, n_out) = (1 + (rng.next() % 19) as usize, 1 + (rng.next() % 5) as usize);
            let weight: Vec<f32> = (0..n_in * n_out).map(|_| rng.range(-3.0, 3.0)).collect();
            let bias: Vec<f32> = (0..n_out).map(|_| rng.range(-1.0, 1.0)).collect();
            let relu = rng.next().is_multiple_of(2);
            let lo_in: Vec<f32> = (0..n_in).map(|_| rng.range(-100.0, 100.0)).collect();
            let hi_in: Vec<f32> = lo_in.iter().map(|&l| l + rng.range(0.0, 50.0)).collect();
            let (mut lo_out, mut hi_out) = (vec![0f32; n_out], vec![0f32; n_out]);
            dense_bounds(&weight, &bias, &lo_in, &hi_in, relu, &mut lo_out, &mut hi_out);
            let mut out = vec![0f32; n_out];
            for _ in 0..100 {
                let input: Vec<f32> = (0..n_in).map(|i| rng.range(lo_in[i], hi_in[i])).collect();
                dense(&weight, &bias, &input, relu, &mut out);
                for o in 0..n_out {
                    assert!(lo_out[o] <= out[o] && out[o] <= hi_out[o], "{out:?} outside {lo_out:?}..{hi_out:?}");
                }
            }
            // Tight to within the rounding margin: the corners of the box reach the
            // bounds, up to (n + 3) unit roundoffs of the magnitude of the terms.
            for o in 0..n_out {
                let corner = |high: bool| -> Vec<f32> {
                    (0..n_in)
                        .map(|i| if (weight[o * n_in + i] > 0.0) == high { hi_in[i] } else { lo_in[i] })
                        .collect()
                };
                let magnitude: f32 = (0..n_in)
                    .map(|i| weight[o * n_in + i].abs() * lo_in[i].abs().max(hi_in[i].abs()))
                    .sum::<f32>()
                    + bias[o].abs();
                let margin = 1e-5 * magnitude + 1e-6;
                dense(&weight, &bias, &corner(false), relu, &mut out);
                assert!(
                    out[o] - lo_out[o] <= margin,
                    "lower bound {} is loose: {} reached",
                    lo_out[o],
                    out[o]
                );
                dense(&weight, &bias, &corner(true), relu, &mut out);
                assert!(
                    hi_out[o] - out[o] <= margin,
                    "upper bound {} is loose: {} reached",
                    hi_out[o],
                    out[o]
                );
            }
        }
    }

    #[test]
    fn dense_bounds_of_an_unbounded_box_are_unbounded() {
        let (mut lo, mut hi) = ([0f32; 2], [0f32; 2]);
        dense_bounds(
            &[1.0, 0.0, -2.0, 0.0],
            &[0.5, 0.5],
            &[f32::NEG_INFINITY; 2],
            &[f32::INFINITY; 2],
            true,
            &mut lo,
            &mut hi,
        );
        assert_eq!(lo, [0.0, 0.0]);
        assert_eq!(hi, [f32::INFINITY, f32::INFINITY]);
        // A zero weight ignores its input, so an unbounded input it does not read is fine.
        dense_bounds(
            &[0.0, 1.0],
            &[0.5],
            &[f32::NEG_INFINITY, -1.0],
            &[f32::INFINITY, 1.0],
            false,
            &mut lo[..1],
            &mut hi[..1],
        );
        assert!(lo[0] < -0.5 && lo[0] > -0.5001 && hi[0] > 1.5 && hi[0] < 1.5001);
    }

    #[test]
    fn trees_sum_their_outputs_in_tree_order() {
        // Two outputs, two rounds: trees 0 and 2 belong to output 0, 1 and 3 to output 1.
        let trees = Trees {
            roots: &[0, 3, 4, 5],
            feature: &[0, -1, -1, -1, -1, -1],
            threshold: &[0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
            left: &[1, 0, 0, 0, 0, 0],
            right: &[2, 0, 0, 0, 0, 0],
            value: &[0.0, -1.0, 1.0, 0.25, 0.5, 0.125],
            base: &[0.1, 0.2],
        };
        let mut out = [0f32; 2];
        trees.outputs(&[0.0], &mut out);
        assert_eq!(out, [(0.1 - 1.0 + 0.5) as f32, (0.2 + 0.25 + 0.125) as f32]);
        trees.outputs(&[1.0], &mut out);
        assert_eq!(out[0], (0.1 + 1.0 + 0.5) as f32);
    }
}
