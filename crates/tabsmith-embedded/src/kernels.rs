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
