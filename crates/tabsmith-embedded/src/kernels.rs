//! Numeric kernels. Each one fixes its order of operations, so a given input gives
//! the same bits wherever it runs; changing an order here changes every prediction.
//!
//! Every kernel a prediction uses is a `const fn`, so a compiled model can run in the
//! compiler: rustc's constant evaluator computes IEEE 754 arithmetic in software, with
//! the same rounding as the hardware, and gives the same bits.
//!
//! The one thing IEEE 754 leaves to the processor is the sign and payload of a NaN that an
//! invalid operation creates, such as `inf - inf` in a dense layer: x86 gives `-NaN`, ARM
//! and the compiler `+NaN`. So every kernel that can return a NaN returns this one,
//! [`NAN_F32`] or [`NAN_F64`].

use crate::InputError;

/// The NaN every kernel returns: quiet, positive, no payload.
pub const NAN_F32: f32 = f32::from_bits(0x7fc0_0000);
/// The NaN every kernel returns in f64: quiet, positive, no payload.
pub const NAN_F64: f64 = f64::from_bits(0x7ff8_0000_0000_0000);

const fn one_nan_f64(v: f64) -> f64 {
    if v.is_nan() { NAN_F64 } else { v }
}

/// Dot product of two equally long slices: eight partial sums over chunks of eight,
/// combined pairwise, then the tail. `b` may be longer; its extra values are ignored.
pub const fn dot(a: &[f32], b: &[f32]) -> f32 {
    // `b` cut to `a`'s length, so that the compiler knows that both have as many chunks
    // and as long a tail, and checks no index in the loops.
    let (b, _) = b.split_at(a.len());
    let (a_chunks, a_tail) = a.as_chunks::<8>();
    let (b_chunks, b_tail) = b.as_chunks::<8>();
    let (b_tail, _) = b_tail.split_at(a_tail.len());
    let mut acc = [0f32; 8];
    let mut c = 0;
    while c < a_chunks.len() {
        let (x, y) = (&a_chunks[c], &b_chunks[c]);
        let mut k = 0;
        while k < 8 {
            acc[k] += x[k] * y[k];
            k += 1;
        }
        c += 1;
    }
    let mut tail = 0f32;
    let mut i = 0;
    while i < a_tail.len() {
        tail += a_tail[i] * b_tail[i];
        i += 1;
    }
    ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7])) + tail
}

/// One dense layer: `out[o] = dot(weight[o], input) + bias[o]`, then ReLU when `relu`.
/// `weight` is row-major `[out.len(), input.len()]`. The ReLU sends NaN to 0; without
/// it, a NaN comes out as [`NAN_F32`]. (`dot` never returns `-0.0`, so neither does the
/// sum here, and `max` has no zeros of two signs to choose between.)
pub const fn dense(weight: &[f32], bias: &[f32], input: &[f32], relu: bool, out: &mut [f32]) {
    let n = input.len();
    let mut rows = weight;
    let mut o = 0;
    while o < out.len() {
        let (row, rest) = rows.split_at(n);
        let v = dot(row, input) + bias[o];
        out[o] = if relu {
            v.max(0.0)
        } else if v.is_nan() {
            NAN_F32
        } else {
            v
        };
        rows = rest;
        o += 1;
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
    pub const fn leaf(&self, t: usize, row: &[f32]) -> usize {
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
    pub const fn outputs(&self, row: &[f32], out: &mut [f32]) {
        let k = out.len();
        let mut o = 0;
        while o < k {
            let mut sum = self.base[o];
            let mut t = o;
            while t < self.roots.len() {
                sum += self.value[self.leaf(t, row)] as f64;
                t += k;
            }
            out[o] = sum as f32;
            o += 1;
        }
    }

    /// The smallest and the largest leaf of tree `t` that a row of the box can reach,
    /// `[min, max]`. The box holds every row with `lo[j] <= row[j] <= hi[j]` for each
    /// feature `j`; where `lo[j]` or `hi[j]` is NaN, `row[j]` may also be NaN, which
    /// every split sends right. A box that reaches no leaf (one with `lo > hi`) gives
    /// `[inf, -inf]`.
    ///
    /// It only compares, so no rounding enters: no row of the box reaches a leaf outside
    /// the range. The Kani proofs that `model!` generates check this for every tree of a
    /// compiled model, every box and every row of it.
    ///
    /// `reachable` is scratch space with at least as many entries as tree `t` has
    /// nodes; it is left all `false`. `const`, like [`bounds`](Self::bounds), which
    /// adds these ranges up.
    pub const fn leaf_range(&self, t: usize, lo: &[f32], hi: &[f32], reachable: &mut [bool]) -> [f32; 2] {
        let start = self.roots[t] as usize;
        let end = if t + 1 < self.roots.len() {
            self.roots[t + 1] as usize
        } else {
            self.feature.len()
        };
        let (mut min, mut max) = (f32::INFINITY, f32::NEG_INFINITY);
        // Children come after their parent, inside the same tree, so one pass in index
        // order visits every reachable node after the split that reached it.
        reachable[0] = true;
        let mut i = start;
        while i < end {
            if reachable[i - start] {
                reachable[i - start] = false;
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
                        reachable[self.left[i] as usize - start] = true;
                    }
                    if hi[f] > self.threshold[i] || hi[f].is_nan() || lo[f].is_nan() {
                        reachable[self.right[i] as usize - start] = true;
                    }
                }
            }
            i += 1;
        }
        [min, max]
    }

    /// The outputs of one row into `out` as XGBoost computes them: each output starts at
    /// its base score rounded to `f32` and adds the leaf of every tree of that output in
    /// `f32`, in tree order, one rounding per tree. For models imported from XGBoost,
    /// whose predictions this reproduces bit for bit; Tabsmith's own trees add in `f64`
    /// ([`outputs`](Self::outputs)).
    pub const fn outputs_in_f32(&self, row: &[f32], out: &mut [f32]) {
        let k = out.len();
        let mut o = 0;
        while o < k {
            let mut sum = self.base[o] as f32;
            let mut t = o;
            while t < self.roots.len() {
                sum += self.value[self.leaf(t, row)];
                t += k;
            }
            out[o] = sum;
            o += 1;
        }
    }

    /// Bounds of [`outputs_in_f32`](Self::outputs_in_f32), as [`bounds`](Self::bounds) is
    /// of [`outputs`](Self::outputs): the leaf ranges added in `f32`, in the same order.
    /// Rounding an `f32` sum is monotone too, so no row of the box leaves them.
    pub const fn bounds_in_f32(&self, lo: &[f32], hi: &[f32], reachable: &mut [bool], lo_out: &mut [f32], hi_out: &mut [f32]) {
        let k = lo_out.len();
        let n_trees = self.roots.len();
        let mut o = 0;
        while o < k {
            let (mut min_sum, mut max_sum) = (self.base[o] as f32, self.base[o] as f32);
            let mut t = o;
            while t < n_trees {
                let [min, max] = self.leaf_range(t, lo, hi, reachable);
                min_sum += min;
                max_sum += max;
                t += k;
            }
            lo_out[o] = min_sum;
            hi_out[o] = max_sum;
            o += 1;
        }
    }

    /// Bounds of [`outputs`](Self::outputs) over every row of the box of
    /// [`leaf_range`](Self::leaf_range), into `lo_out` and `hi_out` (one entry per
    /// output).
    ///
    /// Each tree contributes its [`leaf_range`](Self::leaf_range), and the sums run in
    /// the order `outputs` uses. Adding a smaller leaf to a smaller sum never gives a
    /// larger sum, and rounding to `f32` keeps the order (this crate's Kani proofs check
    /// both), so the sum any row of the box produces lies in between: the bounds are
    /// sound, with no error margin, though not necessarily reached. A box that reaches no
    /// leaf of some tree (one with `lo > hi`) gives `+inf > -inf`.
    ///
    /// `reachable` is scratch space with at least as many entries as the largest tree
    /// has nodes (one per node of the ensemble is always enough); it is left all
    /// `false`. `const`, so a compiled model can bound itself while it compiles.
    pub const fn bounds(&self, lo: &[f32], hi: &[f32], reachable: &mut [bool], lo_out: &mut [f32], hi_out: &mut [f32]) {
        let k = lo_out.len();
        let n_trees = self.roots.len();
        let mut o = 0;
        while o < k {
            let (mut min_sum, mut max_sum) = (self.base[o], self.base[o]);
            let mut t = o;
            while t < n_trees {
                let [min, max] = self.leaf_range(t, lo, hi, reachable);
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
pub const fn standardize(x: f64, mean: f64, std: f64) -> f32 {
    ((x - mean) / std) as f32
}

/// One numeric input encoded as the transform encodes it. A missing value is imputed
/// with `impute`, or refused when `refuse_missing`: `None`, a value equal to one of the
/// bundle's numeric missing-value tokens (such as -999), or NaN when `nan_is_missing`.
/// The value is then standardised.
#[allow(clippy::too_many_arguments)]
pub const fn numeric(
    field: &'static str,
    value: Option<f64>,
    tokens: &[f64],
    nan_is_missing: bool,
    refuse_missing: bool,
    impute: f64,
    mean: f64,
    std: f64,
) -> Result<f32, InputError> {
    let missing = match value {
        None => true,
        Some(v) => {
            let mut i = 0;
            while i < tokens.len() && tokens[i] != v {
                i += 1;
            }
            i < tokens.len() || (v.is_nan() && nan_is_missing)
        }
    };
    if missing && refuse_missing {
        return Err(InputError::Missing { field });
    }
    let x = match value {
        Some(v) if !missing => {
            if v.is_nan() {
                return Err(InputError::NotANumber { field });
            } else if v.is_infinite() {
                return Err(InputError::Infinite { field });
            }
            v
        }
        _ => impute,
    };
    let z = standardize(x, mean, std);
    if z.is_finite() {
        Ok(z)
    } else {
        Err(InputError::OutOfRange { field })
    }
}

// exp, ported from libm 0.2.16 (`src/math/exp.rs`, and `scalbn` from
// `src/math/generic/scalbn.rs` for f64), operation for operation, as a `const fn`. libm's
// own notice, which its source keeps from FreeBSD's e_exp.c:
//
// ====================================================
// Copyright (C) 2004 by Sun Microsystems, Inc. All rights reserved.
//
// Permission to use, copy, modify, and distribute this
// software is freely granted, provided that this notice
// is preserved.
// ====================================================
//
// libm's `force_eval!` statements only raise floating-point exception flags, which Rust
// does not expose, so they are left out; every value is computed as there. The constants
// are written as the bits libm's comments give, which are what its decimals round to.
const EXP_HALF: [f64; 2] = [0.5, -0.5];
const EXP_LN2_HI: f64 = f64::from_bits(0x3fe62e42_fee00000); // 6.93147180369123816490e-01
const EXP_LN2_LO: f64 = f64::from_bits(0x3dea39ef_35793c76); // 1.90821492927058770002e-10
const EXP_INV_LN2: f64 = f64::from_bits(0x3ff71547_652b82fe); // 1.44269504088896338700e+00
const EXP_P1: f64 = f64::from_bits(0x3fc55555_5555553e); // 1.66666666666666019037e-01
const EXP_P2: f64 = f64::from_bits(0xbf66c16c_16bebd93); // -2.77777777770155933842e-03
const EXP_P3: f64 = f64::from_bits(0x3f11566a_af25de2c); // 6.61375632143793436117e-05
const EXP_P4: f64 = f64::from_bits(0xbebbbd41_c5d26bf1); // -1.65339022054652515390e-06
const EXP_P5: f64 = f64::from_bits(0x3e663769_72bea4d0); // 4.13813679705723846039e-08
const EXP_OVERFLOW: f64 = f64::from_bits(0x40862e42_fefa39ef); // 709.782712893383973096
const EXP_UNDERFLOW: f64 = f64::from_bits(0xc0874910_d52d3051); // -745.13321910194110842

/// `x * 2^n`, exact unless it overflows or underflows.
const fn scalbn(mut x: f64, mut n: i32) -> f64 {
    const EXP_MAX: i32 = 1023;
    const EXP_MIN: i32 = -1022;
    const SIG_TOTAL_BITS: i32 = 53;
    let f_exp_max = f64::from_bits(2046 << 52); // 2^1023
    let f_exp_min = f64::from_bits(1 << 52); // 2^-1022
    let f_pow_subnorm = f64::from_bits(((SIG_TOTAL_BITS + 1023) as u64) << 52); // 2^53
    if n > EXP_MAX {
        x *= f_exp_max;
        n -= EXP_MAX;
        if n > EXP_MAX {
            x *= f_exp_max;
            n -= EXP_MAX;
            if n > EXP_MAX {
                n = EXP_MAX;
            }
        }
    } else if n < EXP_MIN {
        let mul = f_exp_min * f_pow_subnorm;
        let add = -EXP_MIN - SIG_TOTAL_BITS;
        x *= mul;
        n += add;
        if n < EXP_MIN {
            x *= mul;
            n += add;
            if n < EXP_MIN {
                n = EXP_MIN;
            }
        }
    }
    x * f64::from_bits(((n + 1023) as u64) << 52)
}

/// The exponential function, `e^x`, in f64: libm's algorithm (FreeBSD's e_exp.c) as a
/// `const fn`, with libm's bits for every input. The error is under one unit in the last
/// place.
pub const fn exp(mut x: f64) -> f64 {
    let x1p1023 = f64::from_bits(0x7fe0000000000000); // 2^1023
    let mut hx = (x.to_bits() >> 32) as u32;
    let sign = (hx >> 31) as i32;
    hx &= 0x7fffffff; // high word of |x|
    if hx >= 0x4086232b {
        // |x| >= 708.39...
        if x.is_nan() {
            return x;
        }
        if x > EXP_OVERFLOW {
            // overflow, unless x is inf
            x *= x1p1023;
            return x;
        }
        if x < EXP_UNDERFLOW {
            return 0.;
        }
    }
    // Argument reduction: x = k * ln2 + r, |r| <= 0.5 * ln2, with r = hi - lo.
    let (hi, lo, k): (f64, f64, i32);
    if hx > 0x3fd62e42 {
        // |x| > 0.5 * ln2
        if hx >= 0x3ff0a2b2 {
            // |x| >= 1.5 * ln2
            k = (EXP_INV_LN2 * x + EXP_HALF[sign as usize]) as i32;
        } else {
            k = 1 - sign - sign;
        }
        hi = x - k as f64 * EXP_LN2_HI; // k * ln2hi is exact here
        lo = k as f64 * EXP_LN2_LO;
        x = hi - lo;
    } else if hx > 0x3e300000 {
        // |x| > 2^-28
        k = 0;
        hi = x;
        lo = 0.;
    } else {
        return 1. + x;
    }
    // x is now in the primary range.
    let xx = x * x;
    let c = x - xx * (EXP_P1 + xx * (EXP_P2 + xx * (EXP_P3 + xx * (EXP_P4 + xx * EXP_P5))));
    let y = 1. + (x * c / (2. - c) - lo + hi);
    if k == 0 { y } else { scalbn(y, k) }
}

/// The logistic function in f64, in the form that cannot overflow. NaN gives [`NAN_F64`].
pub const fn sigmoid(z: f64) -> f64 {
    let p = if z >= 0.0 {
        1.0 / (1.0 + exp(-z))
    } else {
        let e = exp(z);
        e / (1.0 + e)
    };
    one_nan_f64(p)
}

/// Softmax of `z` into `out` (the same length), in f64, shifted by the maximum. A NaN
/// logit makes every probability [`NAN_F64`].
pub const fn softmax(z: &[f32], out: &mut [f64]) {
    let mut max = f32::NEG_INFINITY;
    let mut i = 0;
    while i < z.len() {
        max = max.max(z[i]);
        i += 1;
    }
    let max = max as f64;
    let mut i = 0;
    while i < out.len() && i < z.len() {
        out[i] = exp(z[i] as f64 - max);
        i += 1;
    }
    // Added in order from -0.0, as `Iterator::sum` adds.
    let mut sum = -0.0;
    let mut i = 0;
    while i < out.len() {
        sum += out[i];
        i += 1;
    }
    let mut i = 0;
    while i < out.len() {
        out[i] = one_nan_f64(out[i] / sum);
        i += 1;
    }
}

/// Index of the largest value, the first one on a tie.
pub const fn argmax(p: &[f64]) -> usize {
    let mut best = 0;
    let mut k = 0;
    while k < p.len() {
        if p[k] > p[best] {
            best = k;
        }
        k += 1;
    }
    best
}

/// A regression output back in the target's units: `mean + std * z`. NaN gives
/// [`NAN_F64`].
pub const fn unscale(z: f32, mean: f64, std: f64) -> f64 {
    one_nan_f64(mean + std * z as f64)
}

/// Proofs for the Kani model checker of the rounding facts that make [`Trees::bounds`]
/// sound once every tree's leaf lies inside its [`Trees::leaf_range`] (which the proofs
/// `model!` generates check): `bounds` and `outputs` add the same way, a smaller leaf to a
/// smaller sum, so each sum stays below the other, and so does its rounding to `f32`.
/// These are facts of IEEE 754 arithmetic (every operation rounds its exact result, and
/// rounding is monotone), checked here bit for bit on the operations the two functions
/// use: an `f32` leaf widened and added to an `f64` sum. Comparing two sums one operand at
/// a time keeps the proofs within reach of the SAT solver.
///
/// `cargo kani -p tabsmith-embedded` runs them. The addition of two different leaves takes
/// 9 to 13 minutes and 5 to 10 GB, so `tools/verify.sh --all` runs them, not CI: they do not
/// depend on the code of this crate.
#[cfg(kani)]
mod proofs {
    /// A larger sum stays larger when the same leaf is added to both.
    #[kani::proof]
    fn a_larger_sum_stays_larger() {
        let (s, t): (f64, f64) = (kani::any(), kani::any());
        let a: f32 = kani::any();
        kani::assume(s <= t && a.is_finite());
        assert!(s + f64::from(a) <= t + f64::from(a));
    }

    /// A larger leaf gives a larger sum.
    #[kani::proof]
    fn a_larger_leaf_gives_a_larger_sum() {
        let s: f64 = kani::any();
        let (a, b): (f32, f32) = (kani::any(), kani::any());
        kani::assume(!s.is_nan() && a <= b && a.is_finite() && b.is_finite());
        assert!(s + f64::from(a) <= s + f64::from(b));
    }

    /// Rounding to `f32` keeps the order.
    #[kani::proof]
    fn rounding_to_f32_keeps_the_order() {
        let (s, t): (f64, f64) = (kani::any(), kani::any());
        kani::assume(s <= t);
        assert!((s as f32) <= (t as f32));
    }

    /// For `bounds_in_f32`: a larger `f32` sum stays larger when the same leaf is added.
    #[kani::proof]
    fn a_larger_f32_sum_stays_larger() {
        let (s, t, a): (f32, f32, f32) = (kani::any(), kani::any(), kani::any());
        kani::assume(s <= t && a.is_finite());
        assert!(s + a <= t + a);
    }

    /// For `bounds_in_f32`: a larger leaf gives a larger `f32` sum.
    #[kani::proof]
    fn a_larger_leaf_gives_a_larger_f32_sum() {
        let (s, a, b): (f32, f32, f32) = (kani::any(), kani::any(), kani::any());
        kani::assume(!s.is_nan() && a <= b && a.is_finite() && b.is_finite());
        assert!(s + a <= s + b);
    }
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

    fn same(a: f64, b: f64) -> bool {
        a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
    }

    #[test]
    fn exp_gives_the_bits_of_libm() {
        // Special values, and both sides of every branch of the algorithm.
        let edges = [
            0.0,
            -0.0,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
            f64::MAX,
            f64::MIN,
            f64::MIN_POSITIVE,
            5e-324,
            EXP_OVERFLOW,
            EXP_OVERFLOW.next_up(),
            EXP_UNDERFLOW,
            EXP_UNDERFLOW.next_down(),
            -708.3964185322641,
            708.3964185322641,
            3.725290298461914e-9, // 2^-28
            0.34657359027997264,  // 0.5 ln 2
            1.0397207708399179,   // 1.5 ln 2
            1.0,
            -1.0,
            0.5,
        ];
        for x in edges {
            for x in [x, x.next_up(), x.next_down(), -x] {
                assert!(
                    same(exp(x), libm::exp(x)),
                    "exp({x:e}): {:e} against libm {:e}",
                    exp(x),
                    libm::exp(x)
                );
            }
        }
        // Every 1024th f32 bit pattern, both signs: every exponent of every logit. (Every f32
        // and 3.2 billion random f64 are in the repository's report on compiled models.)
        for bits in (0..=u32::MAX).step_by(1024) {
            let x = f64::from(f32::from_bits(bits));
            assert!(same(exp(x), libm::exp(x)), "exp({x:e})");
        }
        let mut rng = Lcg(5);
        for _ in 0..200_000 {
            let raw = (rng.next() << 11) ^ rng.next();
            let x = match raw & 3 {
                0 => f64::from_bits(raw),
                1 => (raw >> 11) as f64 / (1u64 << 53) as f64 * 1500.0 - 750.0,
                _ => (raw >> 11) as f64 / (1u64 << 53) as f64 * 60.0 - 30.0,
            };
            assert!(same(exp(x), libm::exp(x)), "exp({x:e})");
        }
    }

    // What the compiler itself computes, with its own software floating point.
    const AT_COMPILE_TIME: [u64; 6] = [
        exp(-3.5).to_bits(),
        exp(0.125).to_bits(),
        exp(709.0).to_bits(),
        exp(-744.0).to_bits(),
        sigmoid(1.7).to_bits(),
        dot(
            &[0.5, -1.25, 3.0, 0.75, 2.0, -0.5, 1.5, 0.25, 4.0, -2.0],
            &[1.0, 2.0, -0.5, 0.25, 1.5, 3.0, -1.0, 0.5, 0.125, 2.5],
        )
        .to_bits() as u64,
    ];

    #[test]
    fn the_compiler_computes_the_bits_the_program_computes() {
        let a = std::hint::black_box([0.5f32, -1.25, 3.0, 0.75, 2.0, -0.5, 1.5, 0.25, 4.0, -2.0]);
        let b = std::hint::black_box([1.0f32, 2.0, -0.5, 0.25, 1.5, 3.0, -1.0, 0.5, 0.125, 2.5]);
        let run = [
            exp(std::hint::black_box(-3.5)).to_bits(),
            exp(std::hint::black_box(0.125)).to_bits(),
            exp(std::hint::black_box(709.0)).to_bits(),
            exp(std::hint::black_box(-744.0)).to_bits(),
            sigmoid(std::hint::black_box(1.7)).to_bits(),
            dot(&a, &b).to_bits() as u64,
        ];
        assert_eq!(AT_COMPILE_TIME, run);
        assert_eq!(AT_COMPILE_TIME[0], libm::exp(-3.5).to_bits());
    }

    #[test]
    fn every_nan_a_kernel_returns_has_the_same_bits() {
        // inf - inf in a layer: the processor picks the NaN (x86 a negative one); the
        // kernel returns its own, here and in the compiler.
        let nan_in_layer = |relu: bool| {
            let mut out = [1f32; 2];
            dense(
                &[1.0, 1.0, 2.0, -1.0],
                &[0.0, 0.5],
                &[f32::INFINITY, std::hint::black_box(f32::INFINITY)],
                relu,
                &mut out,
            );
            out
        };
        assert_eq!(nan_in_layer(false)[0], f32::INFINITY, "inf + inf is no NaN");
        assert_eq!(nan_in_layer(false)[1].to_bits(), NAN_F32.to_bits());
        assert_eq!(nan_in_layer(true)[1].to_bits(), 0f32.to_bits(), "the ReLU of NaN is 0");
        let negative_nan = f32::from_bits(0xffc0_0000);
        assert_eq!(sigmoid(f64::from(negative_nan)).to_bits(), NAN_F64.to_bits());
        assert_eq!(unscale(negative_nan, 3.0, 2.0).to_bits(), NAN_F64.to_bits());
        let mut p = [0f64; 3];
        softmax(&[1.0, f32::INFINITY, -2.0], &mut p);
        assert!(p.iter().all(|v| v.to_bits() == NAN_F64.to_bits()), "{p:?}");
        softmax(&[1.0, negative_nan, -2.0], &mut p);
        assert!(p.iter().all(|v| v.to_bits() == NAN_F64.to_bits()), "{p:?}");
        // What the compiler computes for the same layer.
        const IN_THE_COMPILER: [u32; 2] = {
            let mut out = [1f32; 2];
            dense(
                &[1.0, 1.0, 2.0, -1.0],
                &[0.0, 0.5],
                &[f32::INFINITY, f32::INFINITY],
                false,
                &mut out,
            );
            [out[0].to_bits(), out[1].to_bits()]
        };
        assert_eq!(IN_THE_COMPILER, [f32::INFINITY.to_bits(), NAN_F32.to_bits()]);
    }

    #[test]
    fn dot_ignores_the_extra_values_of_a_longer_second_slice() {
        let a = [1.0f32; 11];
        let mut b = [2.0f32; 14];
        assert_eq!(dot(&a, &b), 22.0);
        b[11] = f32::NAN;
        assert_eq!(dot(&a, &b), 22.0);
        assert_eq!(dot(&[], &b), 0.0);
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
        // A NaN lower bound alone admits NaN too, whatever the upper bound. Up to 0.4.0
        // it did not: with `hi` below a threshold, no leaf was reachable at all.
        FOUR_TREES.bounds(&[f32::NAN; 2], &[-10.0; 2], &mut scratch, &mut lo_out, &mut hi_out);
        assert_eq!([lo_out, hi_out], [out, out]);
        assert!(scratch.iter().all(|r| !r), "scratch left clean");
    }

    #[test]
    fn the_bounds_add_up_the_leaf_ranges_of_the_trees() {
        let (lo, hi) = ([-0.25, -0.5], [0.45, 0.2]);
        let mut scratch = [false; 7];
        let ranges: [[f32; 2]; 4] = core::array::from_fn(|t| FOUR_TREES.leaf_range(t, &lo, &hi, &mut scratch));
        assert!(scratch.iter().all(|r| !r), "scratch left clean");
        // Tree 0 splits feature 0 at 0.5 (the box stays left), then feature 1 at -0.2:
        // both of its left leaves, -1 and 0.5, are reachable.
        assert_eq!(ranges[0], [-1.0, 0.5]);
        let (mut lo_out, mut hi_out, mut all) = ([0f32; 2], [0f32; 2], [false; 28]);
        FOUR_TREES.bounds(&lo, &hi, &mut all, &mut lo_out, &mut hi_out);
        let sum = |base: f64, a: f32, b: f32| (base + f64::from(a) + f64::from(b)) as f32;
        assert_eq!(
            lo_out,
            [sum(0.1, ranges[0][0], ranges[2][0]), sum(-0.2, ranges[1][0], ranges[3][0])]
        );
        assert_eq!(
            hi_out,
            [sum(0.1, ranges[0][1], ranges[2][1]), sum(-0.2, ranges[1][1], ranges[3][1])]
        );
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
