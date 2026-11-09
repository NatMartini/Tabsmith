//! Checking a proof that a rule holds: for every input in a region, the prediction satisfies
//! a claim, such as "the probability is at most 0.9".
//!
//! A proof is a tree of splits. Each split cuts a box of feature values in two at
//! `x[feature] <= threshold`, as a tree of the model does, and each leaf of the proof closes
//! its box: the bounds of the model's outputs over the box satisfy the claim, or the box holds
//! no input the model accepts. The splits cover the region, so the claim holds for every input
//! in it. `tabsmith_runtime::audit` searches for such a proof; this module checks one, with
//! the functions that bound a model ([`Trees::bounds`](crate::kernels::Trees::bounds),
//! [`dense_bounds`](crate::kernels::dense_bounds)) and the functions that read its outputs
//! ([`sigmoid`](crate::kernels::sigmoid), [`softmax`](crate::kernels::softmax),
//! [`unscale`](crate::kernels::unscale)). Everything here is a `const fn`, so a model compiled
//! with `tabsmith_macros::model!(…, rule(…))` has the compiler check its proofs again.
//!
//! A box holds every row of features with `lo[j] <= x[j] <= hi[j]`, where the model accepts it:
//! a numeric feature is any number, and a category is a run of slots, each 0 or 1, with at most
//! one at 1. All slots at 0 is a value the model never saw in training, which only a bundle that
//! ignores those accepts; [`Walk`] and [`is_empty`] track it per category as `none`. A box that
//! only treated the slots as numbers would hold rows with two slots at 1, which no input
//! encodes to: bounds over it would be loose, and a search would report counterexamples that do
//! not exist.

use crate::kernels;

/// In [`Domain::category`]: a numeric feature.
pub const NUMERIC: u32 = u32::MAX;

/// How far one class's output must stay above another's before a box of outputs counts as
/// deciding the label of a multiclass model ([`Claim::holds_within`]). Above this margin, every
/// row's probabilities come out in the same order after rounding, whatever their scale.
pub const LABEL_MARGIN: f64 = 1e-6;

/// The rows of features a model accepts: which features are numbers and which are slots of a
/// category.
#[derive(Debug, Clone, Copy)]
pub struct Domain<'a> {
    /// For each feature: [`NUMERIC`], or the index in [`slots`](Self::slots) of the category
    /// it is a slot of.
    pub category: &'a [u32],
    /// For each category: its slots, the features `first..end`.
    pub slots: &'a [[u32; 2]],
}

/// One step of a proof, in the order of a walk that visits a split's left part before its
/// right part.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Step {
    /// Cut the box at `x[feature] <= threshold`: the steps for the left part follow, then
    /// those for the right part.
    Split {
        /// The feature the split reads.
        feature: u32,
        /// Rows with `x[feature] <= threshold` go left, the others right.
        threshold: f32,
    },
    /// The box is closed: its output bounds satisfy the claim, or it holds no input.
    Close,
}

/// What a prediction must satisfy, read from the model's outputs as `decode` reads them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Claim {
    /// Binary: the probability is at most this.
    ProbabilityAtMost(f64),
    /// Binary: the probability is at least this.
    ProbabilityAtLeast(f64),
    /// Binary: the positive label, a probability at or above the threshold.
    Positive {
        /// The bundle's decision threshold.
        threshold: f64,
    },
    /// Binary: the negative label, a probability below the threshold.
    Negative {
        /// The bundle's decision threshold.
        threshold: f64,
    },
    /// Regression: the value, `mean + std * output`, is at most `limit`.
    ValueAtMost {
        /// The largest value the claim allows.
        limit: f64,
        /// The training targets' mean.
        mean: f64,
        /// The training targets' standard deviation (positive).
        std: f64,
    },
    /// Regression: the value is at least `limit`.
    ValueAtLeast {
        /// The smallest value the claim allows.
        limit: f64,
        /// The training targets' mean.
        mean: f64,
        /// The training targets' standard deviation (positive).
        std: f64,
    },
    /// Multiclass: the label is the class with this index.
    Class(usize),
    /// Multiclass: the label is not the class with this index.
    NotClass(usize),
}

impl Claim {
    /// Whether the prediction for the outputs `z` satisfies the claim, computed as `decode`
    /// computes it. `p` is scratch space for a multiclass model's probabilities, as long as
    /// `z`; the other claims do not use it.
    // `!(p >= threshold)` is the negative label exactly as `decode` decides it, NaN included.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    pub const fn holds_at(&self, z: &[f32], p: &mut [f64]) -> bool {
        match *self {
            Claim::ProbabilityAtMost(c) => kernels::sigmoid(z[0] as f64) <= c,
            Claim::ProbabilityAtLeast(c) => kernels::sigmoid(z[0] as f64) >= c,
            Claim::Positive { threshold } => kernels::sigmoid(z[0] as f64) >= threshold,
            Claim::Negative { threshold } => !(kernels::sigmoid(z[0] as f64) >= threshold),
            Claim::ValueAtMost { limit, mean, std } => kernels::unscale(z[0], mean, std) <= limit,
            Claim::ValueAtLeast { limit, mean, std } => kernels::unscale(z[0], mean, std) >= limit,
            Claim::Class(k) => {
                kernels::softmax(z, p);
                kernels::argmax(p) == k
            }
            Claim::NotClass(k) => {
                kernels::softmax(z, p);
                kernels::argmax(p) != k
            }
        }
    }

    /// Whether the claim holds for every row of outputs `z` with `lo[o] <= z[o] <= hi[o]`.
    ///
    /// For a binary model and a regression this is exact: the computed logistic function
    /// never decreases from one `f32` to the next (the Tabsmith repository checks all 2^32),
    /// and `mean + std * z` rounds monotonically, so the bound on the side the claim limits
    /// decides. For a multiclass model it is sufficient, not necessary: when `lo` equals `hi`
    /// it is [`holds_at`](Self::holds_at); otherwise the class must stay [`LABEL_MARGIN`]
    /// above every other (or, for [`NotClass`](Self::NotClass), another class above it).
    /// Above that margin no rounding of the softmax reverses the order. Any NaN bound fails.
    pub const fn holds_within(&self, lo: &[f32], hi: &[f32], p: &mut [f64]) -> bool {
        let mut o = 0;
        while o < lo.len() {
            if lo[o].is_nan() || hi[o].is_nan() {
                return false;
            }
            o += 1;
        }
        match *self {
            Claim::ProbabilityAtMost(_) | Claim::Negative { .. } | Claim::ValueAtMost { .. } => self.holds_at(hi, p),
            Claim::ProbabilityAtLeast(_) | Claim::Positive { .. } | Claim::ValueAtLeast { .. } => self.holds_at(lo, p),
            Claim::Class(k) | Claim::NotClass(k) => {
                let mut same = true;
                let mut o = 0;
                while o < lo.len() {
                    if lo[o] != hi[o] {
                        same = false;
                    }
                    o += 1;
                }
                if same {
                    return self.holds_at(lo, p);
                }
                let class = matches!(*self, Claim::Class(_));
                let mut j = 0;
                while j < lo.len() {
                    if j != k {
                        let gap = if class {
                            lo[k] as f64 - hi[j] as f64
                        } else {
                            lo[j] as f64 - hi[k] as f64
                        };
                        let wide = gap >= LABEL_MARGIN;
                        if class && !wide {
                            return false;
                        }
                        if !class && wide {
                            return true;
                        }
                    }
                    j += 1;
                }
                class
            }
        }
    }
}

/// Narrows the box to the rows on one side of the split `x[feature] <= threshold`: the rows
/// that satisfy it when `left`, the others otherwise. A numeric feature keeps
/// `[lo, threshold]` or `[next_up(threshold), hi]`. A slot keeps the values among 0 and 1
/// that go that way; a slot at 1 sets its category's other slots to 0 and rules out `none`,
/// and a category left with one possible slot and no `none` gets that slot at 1.
pub const fn narrow(domain: &Domain<'_>, lo: &mut [f32], hi: &mut [f32], none: &mut [bool], feature: usize, threshold: f32, left: bool) {
    let c = domain.category[feature];
    if c == NUMERIC {
        if left {
            if threshold < hi[feature] {
                hi[feature] = threshold;
            }
        } else {
            let t = threshold.next_up();
            if t > lo[feature] {
                lo[feature] = t;
            }
        }
        return;
    }
    let c = c as usize;
    let (zero, one) = if left {
        (0.0 <= threshold, 1.0 <= threshold)
    } else {
        (0.0 > threshold, 1.0 > threshold)
    };
    if !one && hi[feature] > 0.0 {
        hi[feature] = 0.0;
    }
    if !zero && lo[feature] < 1.0 {
        lo[feature] = 1.0;
    }
    let [first, end] = domain.slots[c];
    let (first, end) = (first as usize, end as usize);
    if !zero && one {
        let mut s = first;
        while s < end {
            if s != feature && hi[s] > 0.0 {
                hi[s] = 0.0;
            }
            s += 1;
        }
        none[c] = false;
    } else if zero && !one && !none[c] {
        // The slot is 0: if one slot is left that can be 1, it is.
        let (mut open, mut last) = (0, 0);
        let mut s = first;
        while s < end {
            if lo[s] <= 1.0 && 1.0 <= hi[s] {
                open += 1;
                last = s;
            }
            s += 1;
        }
        if open == 1 && lo[last] < 1.0 {
            narrow(domain, lo, hi, none, last, 0.0, false);
        }
    }
}

/// Whether the box holds no row the model accepts: a feature with `lo > hi` (or a NaN bound),
/// a slot that can be neither 0 nor 1, or a category that can neither have exactly one slot
/// at 1 nor, where `none` allows it, all of them at 0.
// `!(lo <= hi)` counts a NaN bound as empty, which `lo > hi` would not.
#[allow(clippy::neg_cmp_op_on_partial_ord)]
pub const fn is_empty(domain: &Domain<'_>, lo: &[f32], hi: &[f32], none: &[bool]) -> bool {
    let mut j = 0;
    while j < lo.len() {
        if !(lo[j] <= hi[j]) {
            return true;
        }
        j += 1;
    }
    let mut c = 0;
    while c < domain.slots.len() {
        let [first, end] = domain.slots[c];
        let (mut can_be_one, mut must_be_one) = (0usize, 0usize);
        let mut s = first as usize;
        while s < end as usize {
            let (zero, one) = (lo[s] <= 0.0 && 0.0 <= hi[s], lo[s] <= 1.0 && 1.0 <= hi[s]);
            if !zero && !one {
                return true;
            }
            if one {
                can_be_one += 1;
            }
            if !zero {
                must_be_one += 1;
            }
            s += 1;
        }
        let holds_a_value = match must_be_one {
            0 => can_be_one > 0 || none[c],
            1 => true,
            _ => false,
        };
        if !holds_a_value {
            return true;
        }
        c += 1;
    }
    false
}

/// Why [`Walk::next`] refused a proof: the index of the step where it went wrong.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Malformed(pub usize);

/// A walk through a proof's steps that yields every box the proof closes, in order. The caller
/// keeps the box (`lo`, `hi`, `none`, starting as the rule's region) and checks each closed
/// box; the walk keeps the boxes of the right parts still to visit.
#[derive(Debug)]
pub struct Walk<'a> {
    domain: Domain<'a>,
    steps: &'a [Step],
    /// The next step to read.
    at: usize,
    /// Whether the last call closed a box, so the next call resumes at a right part.
    closed: bool,
    /// How many right parts wait.
    depth: usize,
    /// For each waiting right part: the box of its split, one row of features each.
    saved_lo: &'a mut [f32],
    saved_hi: &'a mut [f32],
    saved_none: &'a mut [bool],
    /// For each waiting right part: the index of its split's step.
    saved_step: &'a mut [usize],
}

impl<'a> Walk<'a> {
    /// A walk through `steps`. The saved boxes need room for as many splits as the deepest
    /// path of the proof has: `saved_step.len()` of them, each as wide as the box
    /// (`saved_lo` and `saved_hi`) and as the categories (`saved_none`).
    pub const fn new(
        domain: Domain<'a>,
        steps: &'a [Step],
        saved_lo: &'a mut [f32],
        saved_hi: &'a mut [f32],
        saved_none: &'a mut [bool],
        saved_step: &'a mut [usize],
    ) -> Self {
        Walk {
            domain,
            steps,
            at: 0,
            closed: false,
            depth: 0,
            saved_lo,
            saved_hi,
            saved_none,
            saved_step,
        }
    }

    /// Moves the box to the next box the proof closes: `Ok(true)` when there is one,
    /// `Ok(false)` when every box was visited and every step read, and an error for a proof
    /// that is not a well-formed tree of splits over this domain (a feature out of range, a
    /// threshold that is not finite, steps missing or left over, a path deeper than the room
    /// for saved boxes).
    pub const fn next(&mut self, lo: &mut [f32], hi: &mut [f32], none: &mut [bool]) -> Result<bool, Malformed> {
        let (n, g) = (lo.len(), none.len());
        if self.closed {
            self.closed = false;
            if self.depth == 0 {
                return if self.at == self.steps.len() {
                    Ok(false)
                } else {
                    Err(Malformed(self.at))
                };
            }
            self.depth -= 1;
            let d = self.depth;
            copy_f32(lo, self.saved_lo, d * n);
            copy_f32(hi, self.saved_hi, d * n);
            copy_bool(none, self.saved_none, d * g);
            if let Step::Split { feature, threshold } = self.steps[self.saved_step[d]] {
                narrow(&self.domain, lo, hi, none, feature as usize, threshold, false);
            }
        }
        loop {
            if self.at >= self.steps.len() {
                return Err(Malformed(self.at));
            }
            match self.steps[self.at] {
                Step::Close => {
                    self.at += 1;
                    self.closed = true;
                    return Ok(true);
                }
                Step::Split { feature, threshold } => {
                    let d = self.depth;
                    if feature as usize >= n
                        || !threshold.is_finite()
                        || d >= self.saved_step.len()
                        || (d + 1) * n > self.saved_lo.len()
                        || (d + 1) * n > self.saved_hi.len()
                        || (d + 1) * g > self.saved_none.len()
                    {
                        return Err(Malformed(self.at));
                    }
                    save_f32(self.saved_lo, d * n, lo);
                    save_f32(self.saved_hi, d * n, hi);
                    save_bool(self.saved_none, d * g, none);
                    self.saved_step[d] = self.at;
                    self.depth += 1;
                    narrow(&self.domain, lo, hi, none, feature as usize, threshold, true);
                    self.at += 1;
                }
            }
        }
    }
}

/// Proofs for the Kani model checker that a proof's splits cover its region: every input of a
/// box lies in the part of a split its own value selects, and that part is never taken for
/// empty. A walk visits both parts of every split, so every input of the region reaches a box
/// the proof closes. The domain has the shapes that matter: numbers, a category with a value it
/// never saw (`none`) and one without. `cargo kani -p tabsmith-embedded` runs them.
#[cfg(kani)]
mod proofs {
    use super::*;

    /// Features 0 and 1 numeric; 2..5 a category of three slots; 5..7 one of two.
    const CATEGORY: [u32; 7] = [NUMERIC, NUMERIC, 0, 0, 0, 1, 1];
    const SLOTS: [[u32; 2]; 2] = [[2, 5], [5, 7]];
    const DOMAIN: Domain<'static> = Domain {
        category: &CATEGORY,
        slots: &SLOTS,
    };

    /// Whether `row` is an input of the box: finite numbers, slots of 0 or 1 with at most one 1
    /// per category, all 0 only where `none` allows it, and every feature within its bounds.
    fn input_of(row: &[f32; 7], lo: &[f32; 7], hi: &[f32; 7], none: &[bool; 2]) -> bool {
        let mut ok = true;
        for j in 0..7 {
            ok &= lo[j] <= row[j] && row[j] <= hi[j];
            ok &= if CATEGORY[j] == NUMERIC {
                row[j].is_finite()
            } else {
                row[j] == 0.0 || row[j] == 1.0
            };
        }
        for (c, [first, end]) in SLOTS.iter().enumerate() {
            let ones = (*first as usize..*end as usize).filter(|&s| row[s] == 1.0).count();
            ok &= ones == 1 || (ones == 0 && none[c]);
        }
        ok
    }

    /// A split on any feature at any finite threshold keeps every input of any box in the part
    /// its value selects, and that part is not empty. The feature is symbolic, so the solver
    /// does not see the loops' bounds: 9 iterations cover the 7 features, and Kani checks that
    /// they do.
    #[kani::proof]
    #[kani::unwind(9)]
    fn a_split_keeps_every_input_in_its_part() {
        let (lo, hi, none): ([f32; 7], [f32; 7], [bool; 2]) = (kani::any(), kani::any(), kani::any());
        let row: [f32; 7] = kani::any();
        kani::assume(input_of(&row, &lo, &hi, &none));
        let feature: usize = kani::any();
        kani::assume(feature < 7);
        let threshold: f32 = kani::any();
        kani::assume(threshold.is_finite());
        let left = row[feature] <= threshold;
        let (mut l, mut h, mut n) = (lo, hi, none);
        narrow(&DOMAIN, &mut l, &mut h, &mut n, feature, threshold, left);
        assert!(input_of(&row, &l, &h, &n));
        assert!(!is_empty(&DOMAIN, &l, &h, &n));
    }

    /// A box that holds an input is never taken for empty, whatever its bounds.
    #[kani::proof]
    #[kani::unwind(9)]
    fn a_box_with_an_input_is_not_empty() {
        let (lo, hi, none): ([f32; 7], [f32; 7], [bool; 2]) = (kani::any(), kani::any(), kani::any());
        let row: [f32; 7] = kani::any();
        kani::assume(input_of(&row, &lo, &hi, &none));
        assert!(!is_empty(&DOMAIN, &lo, &hi, &none));
    }
}

const fn save_f32(to: &mut [f32], at: usize, from: &[f32]) {
    let mut i = 0;
    while i < from.len() {
        to[at + i] = from[i];
        i += 1;
    }
}

const fn save_bool(to: &mut [bool], at: usize, from: &[bool]) {
    let mut i = 0;
    while i < from.len() {
        to[at + i] = from[i];
        i += 1;
    }
}

const fn copy_f32(to: &mut [f32], from: &[f32], at: usize) {
    let mut i = 0;
    while i < to.len() {
        to[i] = from[at + i];
        i += 1;
    }
}

const fn copy_bool(to: &mut [bool], from: &[bool], at: usize) {
    let mut i = 0;
    while i < to.len() {
        to[i] = from[at + i];
        i += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Features: 0 numeric, 1..4 one category of three slots, 4 numeric.
    const CATEGORY: [u32; 5] = [NUMERIC, 0, 0, 0, NUMERIC];
    const SLOTS: [[u32; 2]; 1] = [[1, 4]];
    const DOMAIN: Domain<'static> = Domain {
        category: &CATEGORY,
        slots: &SLOTS,
    };

    fn region(none: bool) -> ([f32; 5], [f32; 5], [bool; 1]) {
        (
            [f32::NEG_INFINITY, 0.0, 0.0, 0.0, -1.0],
            [f32::INFINITY, 1.0, 1.0, 1.0, 1.0],
            [none],
        )
    }

    #[test]
    fn a_numeric_split_keeps_each_side_of_the_threshold() {
        let (mut lo, mut hi, mut none) = region(false);
        narrow(&DOMAIN, &mut lo, &mut hi, &mut none, 4, 0.25, true);
        assert_eq!((lo[4], hi[4]), (-1.0, 0.25));
        let (mut lo, mut hi, mut none) = region(false);
        narrow(&DOMAIN, &mut lo, &mut hi, &mut none, 4, 0.25, false);
        assert_eq!((lo[4], hi[4]), (0.25f32.next_up(), 1.0));
        // A threshold outside the box leaves it as it is on the side that holds it all.
        let (mut lo, mut hi, mut none) = region(false);
        narrow(&DOMAIN, &mut lo, &mut hi, &mut none, 4, 5.0, true);
        assert_eq!((lo[4], hi[4]), (-1.0, 1.0));
    }

    #[test]
    fn a_slot_at_one_sets_the_other_slots_of_its_category_to_zero() {
        let (mut lo, mut hi, mut none) = region(true);
        narrow(&DOMAIN, &mut lo, &mut hi, &mut none, 2, 0.0, false);
        assert_eq!(&lo[1..4], &[0.0, 1.0, 0.0]);
        assert_eq!(&hi[1..4], &[0.0, 1.0, 0.0]);
        assert!(!none[0]);
        assert!(!is_empty(&DOMAIN, &lo, &hi, &none));
        // The other side: slot 2 at 0, the rest untouched.
        let (mut lo, mut hi, mut none) = region(true);
        narrow(&DOMAIN, &mut lo, &mut hi, &mut none, 2, 0.0, true);
        assert_eq!(&hi[1..4], &[1.0, 0.0, 1.0]);
        assert!(none[0]);
    }

    #[test]
    fn the_last_possible_slot_is_at_one_unless_the_category_may_hold_none() {
        let (mut lo, mut hi, mut none) = region(false);
        narrow(&DOMAIN, &mut lo, &mut hi, &mut none, 1, 0.5, true);
        narrow(&DOMAIN, &mut lo, &mut hi, &mut none, 2, 0.5, true);
        assert_eq!((&lo[1..4], &hi[1..4]), (&[0.0, 0.0, 1.0][..], &[0.0, 0.0, 1.0][..]));
        // Then setting it to 0 too leaves no value at all.
        narrow(&DOMAIN, &mut lo, &mut hi, &mut none, 3, 0.5, true);
        assert!(is_empty(&DOMAIN, &lo, &hi, &none));

        let (mut lo, mut hi, mut none) = region(true);
        narrow(&DOMAIN, &mut lo, &mut hi, &mut none, 1, 0.5, true);
        narrow(&DOMAIN, &mut lo, &mut hi, &mut none, 2, 0.5, true);
        assert_eq!(&lo[1..4], &[0.0, 0.0, 0.0]);
        narrow(&DOMAIN, &mut lo, &mut hi, &mut none, 3, 0.5, true);
        assert!(!is_empty(&DOMAIN, &lo, &hi, &none), "all slots at 0 is a value it never saw");
    }

    #[test]
    fn two_slots_at_one_hold_no_input() {
        let (mut lo, hi, none) = region(false);
        lo[1] = 1.0;
        lo[3] = 1.0;
        assert!(is_empty(&DOMAIN, &lo, &hi, &none));
        let (mut lo, mut hi, none) = region(false);
        (lo[2], hi[2]) = (0.25, 0.75);
        assert!(is_empty(&DOMAIN, &lo, &hi, &none), "a slot is 0 or 1");
    }

    #[test]
    fn binary_and_regression_claims_read_the_bound_on_their_side() {
        let mut p = [0f64; 0];
        let at_most = Claim::ProbabilityAtMost(0.5);
        assert!(at_most.holds_within(&[-10.0], &[0.0], &mut p));
        // The probability stays 0.5 just above 0: e^-x rounds to 1 for tiny x.
        assert!(at_most.holds_within(&[-10.0], &[0.0f32.next_up()], &mut p));
        assert!(!at_most.holds_within(&[-10.0], &[0.001], &mut p));
        assert!(Claim::Negative { threshold: 0.5 }.holds_within(&[-3.0], &[-0.0001], &mut p));
        assert!(
            !Claim::Negative { threshold: 0.5 }.holds_within(&[-3.0], &[0.0], &mut p),
            "0.5 is positive"
        );
        assert!(Claim::Positive { threshold: 0.5 }.holds_within(&[0.0], &[3.0], &mut p));
        let value = Claim::ValueAtLeast {
            limit: 10.0,
            mean: 20.0,
            std: 4.0,
        };
        assert!(value.holds_within(&[-2.5], &[9.0], &mut p));
        assert!(!value.holds_within(&[-2.6], &[9.0], &mut p));
        assert!(!at_most.holds_within(&[f32::NAN], &[-1.0], &mut p), "a NaN bound decides nothing");
    }

    #[test]
    fn a_multiclass_label_needs_a_margin_unless_the_outputs_are_known() {
        let mut p = [0f64; 3];
        let class = Claim::Class(1);
        assert!(class.holds_within(&[0.0, 2.0, -1.0], &[1.0, 3.0, 1.0], &mut p));
        assert!(!class.holds_within(&[0.0, 2.0, -1.0], &[2.0, 3.0, 1.0], &mut p));
        // Equal bounds are one row: decided exactly, ties going to the first class.
        assert!(class.holds_within(&[1.0, 2.0, 2.0], &[1.0, 2.0, 2.0], &mut p));
        assert!(!Claim::Class(2).holds_within(&[1.0, 2.0, 2.0], &[1.0, 2.0, 2.0], &mut p));
        assert!(Claim::NotClass(0).holds_within(&[0.0, 2.0, -1.0], &[1.0, 3.0, 1.0], &mut p));
        assert!(!Claim::NotClass(1).holds_within(&[0.0, 2.0, -1.0], &[1.0, 3.0, 1.0], &mut p));
    }

    /// Splits feature 4 at 0, then the left part on slot 2: three boxes.
    const STEPS: [Step; 5] = [
        Step::Split {
            feature: 4,
            threshold: 0.0,
        },
        Step::Split {
            feature: 2,
            threshold: 0.0,
        },
        Step::Close,
        Step::Close,
        Step::Close,
    ];

    /// A closed box: its bounds and whether its category may hold a value never seen.
    type Closed = ([f32; 5], [f32; 5], bool);

    fn boxes(steps: &[Step], depth: usize) -> Result<Vec<Closed>, Malformed> {
        let (mut lo, mut hi, mut none) = region(true);
        let (mut slo, mut shi, mut snone, mut sstep) = (
            vec![0f32; 5 * depth],
            vec![0f32; 5 * depth],
            vec![false; depth],
            vec![0usize; depth],
        );
        let mut walk = Walk::new(DOMAIN, steps, &mut slo, &mut shi, &mut snone, &mut sstep);
        let mut out = Vec::new();
        while walk.next(&mut lo, &mut hi, &mut none)? {
            out.push((lo, hi, none[0]));
        }
        Ok(out)
    }

    #[test]
    fn a_walk_visits_the_left_part_then_the_right_part() {
        let b = boxes(&STEPS, 2).unwrap();
        assert_eq!(b.len(), 3);
        assert_eq!((b[0].0[4], b[0].1[4], b[0].1[2]), (-1.0, 0.0, 0.0));
        assert_eq!((b[1].0[4], b[1].1[4], b[1].0[2], b[1].1[1]), (-1.0, 0.0, 1.0, 0.0));
        assert!(!b[1].2, "slot 2 at 1: the category holds a value");
        assert_eq!((b[2].0[4], b[2].1[4], b[2].1[2]), (0.0f32.next_up(), 1.0, 1.0));
        assert!(b[2].2, "the right part of the first split is the region's again");
        assert_eq!(boxes(&[Step::Close], 0).unwrap().len(), 1);
    }

    #[test]
    fn a_walk_refuses_a_proof_that_is_not_a_tree_of_splits() {
        assert_eq!(boxes(&STEPS[..4], 2), Err(Malformed(4)), "a right part is missing");
        let mut more = STEPS.to_vec();
        more.push(Step::Close);
        assert_eq!(boxes(&more, 2), Err(Malformed(5)), "a step is left over");
        assert_eq!(boxes(&STEPS, 1), Err(Malformed(1)), "no room for the second split");
        let bad = [
            Step::Split {
                feature: 5,
                threshold: 0.0,
            },
            Step::Close,
            Step::Close,
        ];
        assert_eq!(boxes(&bad, 1), Err(Malformed(0)));
        let nan = [
            Step::Split {
                feature: 0,
                threshold: f32::NAN,
            },
            Step::Close,
            Step::Close,
        ];
        assert_eq!(boxes(&nan, 1), Err(Malformed(0)));
        assert_eq!(boxes(&[], 1), Err(Malformed(0)));
    }

    const fn walk_in_const() -> usize {
        let (mut lo, mut hi, mut none) = (
            [f32::NEG_INFINITY, 0.0, 0.0, 0.0, -1.0],
            [f32::INFINITY, 1.0, 1.0, 1.0, 1.0],
            [true],
        );
        let (mut slo, mut shi, mut snone, mut sstep) = ([0f32; 10], [0f32; 10], [false; 2], [0usize; 2]);
        let mut walk = Walk::new(DOMAIN, &STEPS, &mut slo, &mut shi, &mut snone, &mut sstep);
        let mut closed = 0;
        loop {
            match walk.next(&mut lo, &mut hi, &mut none) {
                Ok(true) => closed += 1,
                Ok(false) => return closed,
                Err(_) => panic!("malformed"),
            }
        }
    }

    #[test]
    fn the_compiler_walks_a_proof() {
        const CLOSED: usize = walk_in_const();
        assert_eq!(CLOSED, 3);
    }
}
