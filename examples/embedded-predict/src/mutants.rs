//! Proofs that must fail, so that the generated ones cannot pass by checking nothing.
//!
//! Each harness states a claim that is false because of one deliberate mistake, of the
//! kind the generated proofs exist to catch, and `#[kani::should_panic]` makes
//! `cargo kani` report it as successful only if Kani finds a counterexample. If a
//! change ever made the generated proofs vacuous (an assumption that excludes every
//! row, an assertion that cannot fail), these would stop failing and the run would
//! report them.

use super::defects::{N_FEATURES, TREE_FUNCTIONS, TREES};
use tabsmith_embedded::kernels::Trees;

/// The nodes of tree 0, the tree the mutants take.
const TREE_0_NODES: usize = TREES.roots[1] as usize;

/// The thresholds of the trees with the root threshold of tree 0 one unit in the last
/// place off: the smallest change a number written into the generated code can have.
const THRESHOLD_ONE_BIT_OFF: [f32; TREES.threshold.len()] = {
    let mut t = [0f32; TREES.threshold.len()];
    let mut i = 0;
    while i < t.len() {
        t[i] = TREES.threshold[i];
        i += 1;
    }
    t[0] = f32::from_bits(t[0].to_bits() + 1);
    t
};

const ONE_BIT_OFF: Trees<'static> = Trees {
    threshold: &THRESHOLD_ONE_BIT_OFF,
    ..TREES
};

/// The mistakes that [`leaf_range_with`] can make.
#[derive(Clone, Copy, PartialEq)]
enum Mistake {
    /// `lo < threshold` where `<=` belongs: a box whose lower bound is a threshold
    /// misses the left child, where a row at that threshold goes.
    StrictComparison,
    /// The rule of tabsmith-embedded 0.4.0: a NaN lower bound did not admit NaN.
    NanRuleOf040,
}

/// `tabsmith_embedded::kernels::Trees::leaf_range` for tree 0, with one mistake.
fn leaf_range_with(mistake: Mistake, t: &Trees<'_>, lo: &[f32], hi: &[f32], reachable: &mut [bool]) -> [f32; 2] {
    let (start, end) = (t.roots[0] as usize, t.roots[1] as usize);
    let (mut min, mut max) = (f32::INFINITY, f32::NEG_INFINITY);
    reachable[0] = true;
    for i in start..end {
        if !reachable[i - start] {
            continue;
        }
        if t.feature[i] < 0 {
            min = min.min(t.value[i]);
            max = max.max(t.value[i]);
            continue;
        }
        let f = t.feature[i] as usize;
        let left = if mistake == Mistake::StrictComparison {
            lo[f] < t.threshold[i]
        } else {
            lo[f] <= t.threshold[i]
        };
        let right = hi[f] > t.threshold[i] || hi[f].is_nan() || (mistake != Mistake::NanRuleOf040 && lo[f].is_nan());
        if left {
            reachable[t.left[i] as usize - start] = true;
        }
        if right {
            reachable[t.right[i] as usize - start] = true;
        }
    }
    [min, max]
}

/// A box and a row of it, as the generated range proofs take them.
fn any_row_of_any_box() -> ([f32; N_FEATURES], [f32; N_FEATURES], [f32; N_FEATURES]) {
    let (lo, hi, x): ([f32; N_FEATURES], [f32; N_FEATURES], [f32; N_FEATURES]) = (kani::any(), kani::any(), kani::any());
    for j in 0..N_FEATURES {
        kani::assume((lo[j] <= x[j] && x[j] <= hi[j]) || (x[j].is_nan() && (lo[j].is_nan() || hi[j].is_nan())));
    }
    (lo, hi, x)
}

/// The code of tree 0 against the walk through tree 1.
#[kani::proof]
#[kani::should_panic]
#[kani::unwind(12)]
fn mutant_another_tree() {
    let x: [f32; N_FEATURES] = kani::any();
    assert!(TREE_FUNCTIONS[0](&x).to_bits() == TREES.value[TREES.leaf(1, &x)].to_bits());
}

/// The code of tree 0 against its walk with one threshold one unit in the last place off.
#[kani::proof]
#[kani::should_panic]
#[kani::unwind(12)]
fn mutant_one_bit_off() {
    let x: [f32; N_FEATURES] = kani::any();
    assert!(TREE_FUNCTIONS[0](&x).to_bits() == ONE_BIT_OFF.value[ONE_BIT_OFF.leaf(0, &x)].to_bits());
}

/// A range analysis that sends rows at a threshold the wrong way misses a leaf.
#[kani::proof]
#[kani::should_panic]
#[kani::unwind(66)]
fn mutant_strict_comparison() {
    let (lo, hi, x) = any_row_of_any_box();
    let [min, max] = leaf_range_with(Mistake::StrictComparison, &TREES, &lo, &hi, &mut [false; TREE_0_NODES]);
    let leaf = TREE_FUNCTIONS[0](&x);
    assert!(min <= leaf && leaf <= max);
}

/// The range analysis of 0.4.0 misses the leaf of a NaN row when only `lo` is NaN.
#[kani::proof]
#[kani::should_panic]
#[kani::unwind(66)]
fn mutant_nan_rule_of_0_4_0() {
    let (lo, hi, x) = any_row_of_any_box();
    let [min, max] = leaf_range_with(Mistake::NanRuleOf040, &TREES, &lo, &hi, &mut [false; TREE_0_NODES]);
    let leaf = TREE_FUNCTIONS[0](&x);
    assert!(min <= leaf && leaf <= max);
}
