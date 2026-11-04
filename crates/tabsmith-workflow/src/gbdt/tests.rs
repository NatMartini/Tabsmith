//! Tests of the trees: above all, that nothing but the data can change the model.

use super::bins::*;
use super::grow::*;
use super::rows::*;
use super::*;
use tabsmith_contracts::config::TreeParams;

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

/// `n` rows of `dense` uniform features and `sparse` one-hot-like ones (1 in 5% of the
/// rows, never two on the same row, which makes bin 0 dominant and lets them share a pass).
fn table(n: usize, dense: usize, sparse: usize) -> Matrix {
    let mut next = uniform(0x9e37_79b9_7f4a_7c15);
    let d = dense + sparse;
    let mut data = Vec::with_capacity(n * d);
    for _ in 0..n {
        for _ in 0..dense {
            data.push(next() as f32);
        }
        let which = (next() / 0.05) as usize;
        for s in 0..sparse {
            data.push(if s == which { 1.0 } else { 0.0 });
        }
    }
    Matrix { rows: n, cols: d, data }
}

/// Stores gradients and hessians as the integers of a tree, and returns its scale.
fn set_gradients(rows: &Rows, gh: &[(f64, f64)]) -> Units {
    let bound_g = gh.iter().fold(0f64, |b, p| b.max(p.0.abs()));
    let bound_h = gh.iter().fold(0f64, |b, p| b.max(p.1));
    let units = Units {
        g: exponent_for(bound_g),
        h: exponent_for(bound_h),
    };
    let (mut total_g, mut total_h) = (0i64, 0i64);
    for (r, &(g, h)) in gh.iter().enumerate() {
        let (gi, hi) = (nearest(g * two_pow(units.g)), nearest(h * two_pow(units.h)).max(1));
        rows.gh[r].store(pack(gi, hi), Relaxed);
        total_g += i64::from(gi);
        total_h += i64::from(hi);
    }
    rows.total_g.store(total_g, Relaxed);
    rows.total_h.store(total_h, Relaxed);
    units
}

/// Grows one tree and returns the value of the leaf each row lands in.
#[allow(clippy::too_many_arguments)]
fn grow_one<'env>(
    pool: &Pool<'env>,
    rows: &'env Rows,
    gh: &[(f64, f64)],
    p: &'env TreeParams,
    l2: f64,
    shrinkage: f64,
    monotone: &'env [i8],
    forest: &mut Forest,
    free: &mut FreeHists,
) -> Vec<f64> {
    let units = set_gradients(rows, gh);
    let leaves = grow_tree(pool, rows, Growth::new(p, monotone, l2, shrinkage, units), forest, free);
    let mut moved = vec![0.0; rows.bins.rows];
    for leaf in leaves {
        for at in &rows.order[leaf.start..leaf.start + leaf.len] {
            moved[at.load(Relaxed) as usize] = f64::from(leaf.value);
        }
    }
    moved
}

/// One tree on one thread, for the tests about what a tree learns.
fn grow_alone(x: &Matrix, gh: &[(f64, f64)], p: &TreeParams, l2: f64, shrinkage: f64, monotone: &[i8], forest: &mut Forest) -> Vec<f64> {
    let rows = Rows::new(Bins::new(x, p.max_bins, true), p.max_leaves);
    let mut free = FreeHists::default();
    with_pool(1, |pool| grow_one(pool, &rows, gh, p, l2, shrinkage, monotone, forest, &mut free))
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

fn f32_bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn f64_bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn assert_same_forest(a: &Forest, b: &Forest) {
    assert_eq!(a.roots, b.roots);
    assert_eq!(a.feature, b.feature);
    assert_eq!(a.left, b.left);
    assert_eq!(a.right, b.right);
    assert_eq!(f32_bits(&a.threshold), f32_bits(&b.threshold));
    assert_eq!(f32_bits(&a.value), f32_bits(&b.value));
}

#[test]
fn the_model_does_not_depend_on_threads_or_passes() {
    // Large enough that nodes take every parallel path: histograms, partition, applying a tree.
    let (n, dense, sparse) = (150_000usize, 3usize, 3usize);
    let x = table(n, dense, sparse);
    assert!(n * (dense + sparse) >= MIN_SECTION_WORK && n > 2 * PIECE_ROWS);
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
    let train = |threads: usize, bundle: bool| {
        let rows = Rows::new(Bins::new(&x, params.max_bins, bundle), params.max_leaves);
        assert_eq!(rows.bins.passes.len() < dense + sparse, bundle, "bundling joins the sparse columns");
        with_pool(threads, |pool| {
            let mut forest = Forest::default();
            let mut free = FreeHists::default();
            let mut scores = vec![0.0; n];
            for _ in 0..3 {
                let gh: Vec<(f64, f64)> = scores.iter().zip(&target).map(|(s, y)| (s - y, 1.0)).collect();
                let moved = grow_one(pool, &rows, &gh, &params, 1.0, 0.1, &[], &mut forest, &mut free);
                for (s, m) in scores.iter_mut().zip(moved) {
                    *s += m;
                }
            }
            let mut applied = vec![0.0; n];
            for t in 0..forest.roots.len() {
                let nodes = forest.nodes(t);
                for r in 0..n {
                    applied[r] += f64::from(walk(&nodes, x.row(r)));
                }
            }
            (forest, scores, applied)
        })
    };
    let (reference, scores, applied) = train(1, false);
    assert_eq!(reference.roots.len(), 3);
    assert!(reference.feature.len() > 30, "the trees really split");
    assert_eq!(
        f64_bits(&scores),
        f64_bits(&applied),
        "the rows' leaves are the ones the walk reaches"
    );
    for (threads, bundle) in [(1, true), (2, false), (2, true), (7, false), (7, true)] {
        let (forest, s, a) = train(threads, bundle);
        assert_same_forest(&forest, &reference);
        assert_eq!(f64_bits(&s), f64_bits(&scores));
        assert_eq!(f64_bits(&a), f64_bits(&applied));
    }
}

#[test]
fn the_model_does_not_depend_on_the_order_of_the_rows() {
    let (n, d) = (20_000usize, 5usize);
    let x = table(n, 3, 2);
    let target: Vec<f64> = (0..n).map(|r| f64::from(x.row(r)[0]) * 3.0 + f64::from(x.row(r)[3])).collect();
    let reversed = Matrix {
        rows: n,
        cols: d,
        data: (0..n).rev().flat_map(|r| x.row(r).iter().copied()).collect(),
    };
    let reversed_target: Vec<f64> = target.iter().rev().copied().collect();
    let params = TreeParams {
        max_leaves: 15,
        max_depth: 6,
        min_samples_leaf: 20,
        max_bins: 255,
        monotone: Default::default(),
    };
    let train = |x: &Matrix, target: &[f64]| {
        let rows = Rows::new(Bins::new(x, params.max_bins, true), params.max_leaves);
        with_pool(2, |pool| {
            let mut forest = Forest::default();
            let mut free = FreeHists::default();
            let mut scores = vec![0.0; n];
            for _ in 0..3 {
                let gh: Vec<(f64, f64)> = scores.iter().zip(target).map(|(s, y)| (s - y, 1.0)).collect();
                let moved = grow_one(pool, &rows, &gh, &params, 1.0, 0.1, &[], &mut forest, &mut free);
                for (s, m) in scores.iter_mut().zip(moved) {
                    *s += m;
                }
            }
            forest
        })
    };
    assert_same_forest(&train(&x, &target), &train(&reversed, &reversed_target));
}

/// The sums of every (feature, bin) one row at a time, with no shortcut.
fn plain_histogram(bins: &Bins, rows: &[usize], gh: &[(i32, i32)]) -> Vec<Cell> {
    let mut hist = vec![Cell::default(); bins.slots()];
    for j in 0..bins.width {
        for &r in rows {
            let c = &mut hist[bins.offsets[j] + bins.column(j)[r] as usize];
            c.g += i64::from(gh[r].0);
            c.h += i64::from(gh[r].1);
            c.n += 1;
        }
    }
    hist
}

fn read_histogram(hist: &[Slot]) -> Vec<Cell> {
    hist.iter().map(Slot::get).collect()
}

#[test]
fn shortcuts_give_exactly_what_summing_every_row_gives() {
    // Two tables, gathered gradients, bundled columns, a dominant bin from the totals and the
    // parent minus its sibling: every cell equals the plain sum, bit for bit.
    let n = 20_000usize;
    let x = table(n, 2, 3);
    let mut next = uniform(7);
    let gh: Vec<(f64, f64)> = (0..n).map(|_| (next() - 0.5, next() * 0.25 + 1e-3)).collect();
    for threads in [1, 4] {
        let rows = Rows::new(Bins::new(&x, 64, true), 8);
        assert!(rows.bins.passes.iter().any(|p| matches!(p, Pass::Bundle { .. })));
        let units = set_gradients(&rows, &gh);
        let ints: Vec<(i32, i32)> = gh
            .iter()
            .map(|&(g, h)| (nearest(g * two_pow(units.g)), nearest(h * two_pow(units.h)).max(1)))
            .collect();
        let all: Vec<usize> = (0..n).collect();
        with_pool(threads, |pool| {
            let total = rows.begin_tree(pool);
            rows.fill_root(pool, total, 0);
            assert_eq!(read_histogram(rows.hist(0)), plain_histogram(&rows.bins, &all, &ints));

            let bin = rows.bins.thresholds[0].len() / 3;
            let left = rows.partition(pool, 0, bin, 0, n);
            assert!(left > 0 && left < n / 2, "the left side is the smaller one here");
            let in_left: Vec<usize> = rows.order[..left].iter().map(|r| r.load(Relaxed) as usize).collect();
            let in_right: Vec<usize> = rows.order[left..].iter().map(|r| r.load(Relaxed) as usize).collect();
            let left_hist = plain_histogram(&rows.bins, &in_left, &ints);
            let left_total = left_hist[rows.bins.offsets[0]..rows.bins.offsets[1]]
                .iter()
                .fold(Cell::default(), |mut t, c| {
                    t += *c;
                    t
                });
            rows.fill_node(pool, 0, left, left_total, 1);
            assert_eq!(read_histogram(rows.hist(1)), left_hist);
            rows.subtract(pool, 0, 1);
            assert_eq!(read_histogram(rows.hist(0)), plain_histogram(&rows.bins, &in_right, &ints));
        });
    }
}

#[test]
fn the_partition_keeps_the_order_on_both_sides() {
    // Below and above the size that runs on several threads.
    for n in [1000usize, 2 * PIECE_ROWS + 12_345] {
        let mut next = uniform(11);
        let col: Vec<u8> = (0..n).map(|_| (next() * 8.0) as u8).collect();
        let x = Matrix {
            rows: n,
            cols: 1,
            data: col.iter().map(|&c| f32::from(c)).collect(),
        };
        for threads in [1, 4] {
            let rows = Rows::new(Bins::new(&x, 255, true), 4);
            assert_eq!(rows.bins.column(0), col.as_slice(), "each value is its own bin");
            with_pool(threads, |pool| {
                rows.begin_tree(pool);
                let left = rows.partition(pool, 0, 3, 0, n);
                let expected: Vec<u32> = (0..n as u32)
                    .filter(|&r| col[r as usize] <= 3)
                    .chain((0..n as u32).filter(|&r| col[r as usize] > 3))
                    .collect();
                assert_eq!(left, (0..n).filter(|&r| col[r] <= 3).count());
                let order: Vec<u32> = rows.order.iter().map(|r| r.load(Relaxed)).collect();
                assert_eq!(order, expected);
            });
        }
    }
}

#[test]
fn binning_and_applying_a_tree_match_the_runtime() {
    let (n, d) = (40_000usize, 3usize);
    let x = table(n, d, 0);
    let mut next = uniform(3);
    let gh: Vec<(f64, f64)> = (0..n).map(|_| (next() - 0.5, next() * 0.25)).collect();
    let bins = Bins::new(&x, 64, true);
    for r in 0..n {
        for j in 0..d {
            let code = bins.thresholds[j].partition_point(|&t| t < x.data[r * d + j]) as u8;
            assert_eq!(bins.column(j)[r], code);
        }
    }
    // Applying a tree equals the runtime's walk, row by row.
    let mut forest = Forest::default();
    let moved = grow_alone(&x, &gh, &stump(), 1.0, 1.0, &[], &mut forest);
    let e = forest.ensemble(d, Task::Regression, &[0.0], 1).unwrap();
    let runtime = Model::Trees(e).outputs(&x).data;
    assert!(moved.iter().zip(&runtime).all(|(a, b)| (*a as f32).to_bits() == b.to_bits()));
}

fn thresholds(values: &[f32], max_bins: usize) -> Vec<f32> {
    thresholds_from_counts(values.len(), max_bins, |level, parents| count_cells(values, level, parents))
}

#[test]
fn thresholds_bin_values_like_the_runtime_compares_them() {
    let x = Matrix {
        rows: 6,
        cols: 1,
        data: vec![3.0, 1.0, 2.0, 2.0, 5.0, 1.0],
    };
    let bins = Bins::new(&x, 255, true);
    // bin k <=> x <= thresholds[k]; 5.0 is above every threshold. A value that repeats is
    // refined down to itself; the threshold of one that does not may lie above it.
    let t = &bins.thresholds[0];
    assert_eq!((t.len(), t[0], t[1]), (3, 1.0, 2.0));
    assert!(t[2] >= 3.0 && t[2] < 5.0, "{t:?}");
    assert_eq!(bins.codes, vec![2, 0, 1, 1, 3, 0]);
}

#[test]
fn bins_hold_about_their_share_of_the_rows() {
    let values: Vec<f32> = (0..1000).map(|i| i as f32).collect();
    let t = thresholds(&values, 4);
    assert!(t.len() <= 3 && t.len() >= 2, "{t:?}");
    let mut counts = vec![0usize; t.len() + 1];
    for &v in &values {
        counts[t.partition_point(|&t| t < v)] += 1;
    }
    assert!(counts[..t.len()].iter().all(|&c| (250..=500).contains(&c)), "{counts:?}");
}

#[test]
fn thresholds_handle_categories_signs_and_outliers() {
    // A one-hot column splits at exactly zero, whichever side is the larger.
    let mostly_zero: Vec<f32> = (0..1000).map(|i| if i % 50 == 0 { 1.0 } else { 0.0 }).collect();
    let mostly_one: Vec<f32> = mostly_zero.iter().map(|v| 1.0 - v).collect();
    assert_eq!(thresholds(&mostly_zero, 255), vec![0.0]);
    let t = thresholds(&mostly_one, 255);
    assert_eq!(t.len(), 1);
    assert!(t[0] >= 0.0 && t[0] < 1.0, "{t:?}");

    // Zero of either sign stays apart from the numbers next to it.
    let near_zero: Vec<f32> = [-2.0, -1e-40, -1e-45, -0.0, 0.0, 1e-45, 1e-40, 2.0]
        .iter()
        .flat_map(|&v| std::iter::repeat_n(v, 50))
        .collect();
    let t = thresholds(&near_zero, 255);
    let bin = |v: f32| t.partition_point(|&t| t < v);
    assert!(t.iter().all(|v| v.is_finite()));
    assert_eq!(bin(-0.0), bin(0.0));
    for pair in [
        (-2.0, -1e-40),
        (-1e-40, -1e-45),
        (-1e-45, 0.0),
        (0.0, 1e-45),
        (1e-45, 1e-40),
        (1e-40, 2.0),
    ] {
        assert!(bin(pair.0) < bin(pair.1), "{pair:?} in {t:?}");
    }

    // A far outlier does not squeeze the other values into one bin.
    let mut skewed: Vec<f32> = (0..10_000).map(|i| 0.001 * (i % 1000) as f32).collect();
    skewed.push(1e30);
    let t = thresholds(&skewed, 64);
    assert!(t.len() >= 32, "{} bins", t.len());
}

#[test]
fn thresholds_do_not_depend_on_where_the_rows_are() {
    // Rows held in three places, in another order, summing their counts level by level.
    let mut next = uniform(19);
    let values: Vec<f32> = (0..30_000).map(|_| ((next() - 0.5) * 1000.0).powi(3) as f32).collect();
    let reference = thresholds(&values, 255);
    let (a, rest) = values.split_at(1_000);
    let (b, c) = rest.split_at(20_000);
    let summed = thresholds_from_counts(values.len(), 255, |level, parents| {
        let mut sums = count_cells(c, level, parents);
        for part in [a, b] {
            for (s, v) in sums.iter_mut().zip(count_cells(part, level, parents)) {
                *s += v;
            }
        }
        sums
    });
    assert_eq!(f32_bits(&summed), f32_bits(&reference));
}

#[test]
fn the_scale_of_a_tree_fits_its_integers() {
    for bound in [0.0, 1e-300, 1e-12, 0.25, 1.0, 1.5, 2.0, 7.5, 1e12, 1e300] {
        let e = exponent_for(bound);
        assert!(e <= ROW_BITS, "{bound}");
        assert!(bound * two_pow(e) <= two_pow(ROW_BITS), "{bound} at 2^{e}");
        let row = nearest(bound * two_pow(e));
        assert!(i64::from(row).abs() <= 1 << ROW_BITS);
    }
    assert_eq!(two_pow(0), 1.0);
    assert_eq!(two_pow(-3), 0.125);
    assert_eq!((nearest(2.5), nearest(-2.5), nearest(0.4), nearest(-0.4)), (3, -3, 0, 0));
    let mut cell = Cell::default();
    add_row(&mut cell, pack(-5, 7));
    add_row(&mut cell, pack(2, 1 << 30));
    assert_eq!(
        cell,
        Cell {
            g: -3,
            h: 7 + (1 << 30),
            n: 2
        }
    );
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
    let gh: Vec<(f64, f64)> = y.iter().map(|&t| (0.5 - if t { 1.0 } else { 0.0 }, 0.25)).collect();
    let mut forest = Forest::default();
    let moved = grow_alone(&x, &gh, &stump(), 1.0, 1.0, &[], &mut forest);
    assert_eq!(forest.feature[0], 0, "splits on the informative feature");
    assert!(
        forest.threshold[0] >= 0.5 && forest.threshold[0] < 101.0 / 200.0,
        "{}",
        forest.threshold[0]
    );
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
    let gh: Vec<(f64, f64)> = y.iter().map(|v| (-v, 1.0)).collect();
    let mut forest = Forest::default();
    let moved = grow_alone(&x, &gh, &stump(), 0.0, 1.0, &[], &mut forest);
    assert!(moved.iter().zip(&y).all(|(&m, &t)| (m - t).abs() < 1e-6), "{moved:?}");
}

#[test]
fn saturated_rows_with_no_l2_stay_finite() {
    // Hessians at the floor next to large ones. With no L2 nothing may blow up.
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
    let moved = grow_alone(&x, &gh, &params, 0.0, 0.1, &[], &mut forest);
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
    let deep = TreeParams {
        max_leaves: 16,
        max_depth: 5,
        min_samples_leaf: 5,
        max_bins: 255,
        monotone: Default::default(),
    };
    let train = |monotone: &[i8]| {
        let rows = Rows::new(Bins::new(&x, deep.max_bins, true), deep.max_leaves);
        with_pool(1, |pool| {
            let mut forest = Forest::default();
            let mut free = FreeHists::default();
            let mut scores = vec![0.0; n];
            for _ in 0..30 {
                let gh: Vec<(f64, f64)> = scores.iter().zip(&target).map(|(s, y)| (s - y, 1.0)).collect();
                let moved = grow_one(pool, &rows, &gh, &deep, 0.0, 0.3, monotone, &mut forest, &mut free);
                for r in 0..n {
                    scores[r] += moved[r];
                }
            }
            (forest, scores)
        })
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
