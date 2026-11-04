//! Thresholds from counts, the bin of every row, and the passes that fill a histogram.

use rayon::prelude::*;
use tabsmith_runtime::Matrix;

/// A feature with at least this share of the rows in one bin can join a bundle.
const BUNDLE_SHARE: f64 = 0.5;
/// A feature is tried against at most this many bundles.
const MAX_BUNDLES_TRIED: usize = 64;

/// A finite f32 as a u32 in the same order; -0 and +0 map alike.
pub(super) fn ordered(x: f32) -> u32 {
    let bits = (x + 0.0).to_bits();
    if bits >> 31 == 0 { bits | 0x8000_0000 } else { !bits }
}

/// The f32 whose ordered form is `k`. The form of -0 would compare equal to +0, so it becomes the
/// negative number closest to zero: every negative value is at most that one, and zero is not.
pub(super) fn threshold_of(k: u32) -> f32 {
    let k = if k == 0x7fff_ffff { k - 1 } else { k };
    f32::from_bits(if k >> 31 == 1 { k & 0x7fff_ffff } else { !k })
}

/// How many of `values` fall in each of the 256 children of each cell in `parents`, at `level`
/// bits of the ordered form.
pub(super) fn count_cells(values: &[f32], level: u32, parents: &[u32]) -> Vec<u64> {
    let shift = 32 - level;
    let mut counts = vec![0u64; parents.len() * 256];
    for &x in values {
        let k = ordered(x);
        let parent = if level == 8 { 0 } else { k >> (shift + 8) };
        if let Ok(i) = parents.binary_search(&parent) {
            counts[i * 256 + ((k >> shift) & 255) as usize] += 1;
        }
    }
    counts
}

/// Up to `max_bins - 1` ascending thresholds for `total` values, from counts alone. The values fall
/// into 256 cells of their ordered form, and a cell with more than its share of the rows is refined
/// into 256 more, down to single values. Adjacent cells are then joined until each bin holds its
/// share. `counts_of(level, parents)` answers one refinement, as [`count_cells`] does; rows held in
/// several places can sum their counts, and get the same thresholds as one place holding them all.
///
/// A bin `k` holds the values `<= thresholds[k]` above the previous threshold, the last bin the
/// rest, so a split never leaves its right side empty.
pub(super) fn thresholds_from_counts(total: usize, max_bins: usize, mut counts_of: impl FnMut(u32, &[u32]) -> Vec<u64>) -> Vec<f32> {
    let share = total.div_ceil(max_bins).max(1) as u64;
    let mut parents = vec![0u32];
    let mut cells: Vec<(u32, u64)> = Vec::new();
    for level in [8u32, 16, 24, 32] {
        let shift = 32 - level;
        let counts = counts_of(level, &parents);
        let mut next = Vec::new();
        for (i, &parent) in parents.iter().enumerate() {
            for child in 0..256u32 {
                let count = counts[i * 256 + child as usize];
                if count == 0 {
                    continue;
                }
                let prefix = if level == 8 { child } else { (parent << 8) | child };
                if level < 32 && count > share {
                    next.push(prefix);
                } else {
                    let top = if shift == 0 {
                        prefix
                    } else {
                        (prefix << shift) | ((1u32 << shift) - 1)
                    };
                    cells.push((top, count));
                }
            }
        }
        parents = next;
        if parents.is_empty() {
            break;
        }
    }
    cells.sort_unstable();
    let keep_all = cells.len() <= max_bins;
    let mut thresholds = Vec::new();
    let mut held = 0;
    for (i, &(top, count)) in cells.iter().enumerate() {
        held += count;
        if (keep_all || held >= share) && i + 1 < cells.len() {
            thresholds.push(threshold_of(top));
            held = 0;
        }
    }
    thresholds
}

/// One loop over a node's rows, which fills the cells of one or more features.
pub(super) enum Pass {
    /// One feature: a row's code in [`Bins::column`] is its bin.
    Plain { feature: usize },
    /// Features that are never away from their dominant bin on the same row, such as the one-hot
    /// columns of a category. Code 0 means that all of them are at their dominant bin; the codes
    /// of a feature's other bins are consecutive, from `first_code`. A dominant bin's sums are
    /// what the other bins leave of the node's totals. Nothing to sum: no codes at all.
    Bundle { codes: Vec<u8>, features: Vec<BundledFeature> },
}

#[derive(Debug, Clone, Copy)]
pub(super) struct BundledFeature {
    pub(super) feature: usize,
    pub(super) dominant: u8,
    pub(super) first_code: u8,
}

/// Candidate split points of every feature, the bin of every training row, and the passes that
/// fill a histogram.
pub(super) struct Bins {
    pub(super) thresholds: Vec<Vec<f32>>,
    /// Bin of every training row, one column per feature.
    pub(super) codes: Vec<u8>,
    pub(super) rows: usize,
    pub(super) width: usize,
    /// First histogram slot of each feature, then the total: feature `j` has
    /// `thresholds[j].len() + 1` slots.
    pub(super) offsets: Vec<usize>,
    pub(super) passes: Vec<Pass>,
}

/// What [`Bins::new`] finds out about one feature.
pub(super) struct FeatureBins {
    pub(super) thresholds: Vec<f32>,
    pub(super) codes: Vec<u8>,
    pub(super) dominant: u8,
    /// Rows outside the dominant bin.
    pub(super) others: usize,
}

impl Bins {
    /// `bundle` is off only in tests, which check that it changes nothing but the time.
    pub(super) fn new(x: &Matrix, max_bins: usize, bundle: bool) -> Self {
        let (n, d) = (x.rows, x.cols);
        // Every feature's thresholds and codes depend on that feature's values alone.
        let features: Vec<FeatureBins> = (0..d)
            .into_par_iter()
            .map(|j| {
                let values: Vec<f32> = (0..n).map(|r| x.data[r * d + j]).collect();
                let thresholds = thresholds_from_counts(n, max_bins, |level, parents| count_cells(&values, level, parents));
                let mut counts = [0usize; 256];
                let codes: Vec<u8> = values
                    .iter()
                    .map(|&v| {
                        let code = thresholds.partition_point(|&t| t < v) as u8;
                        counts[code as usize] += 1;
                        code
                    })
                    .collect();
                let (dominant, most) = counts
                    .iter()
                    .enumerate()
                    .fold((0, 0), |b, (i, &c)| if c > b.1 { (i, c) } else { b });
                FeatureBins {
                    thresholds,
                    codes,
                    dominant: dominant as u8,
                    others: n - most,
                }
            })
            .collect();
        let passes = plan_passes(&features, n, bundle);
        let mut offsets = Vec::with_capacity(d + 1);
        let mut total = 0;
        for f in &features {
            offsets.push(total);
            total += f.thresholds.len() + 1;
        }
        offsets.push(total);
        let (mut thresholds, mut codes) = (Vec::with_capacity(d), Vec::with_capacity(n * d));
        for f in features {
            thresholds.push(f.thresholds);
            codes.extend_from_slice(&f.codes);
        }
        Self {
            thresholds,
            codes,
            rows: n,
            width: d,
            offsets,
            passes,
        }
    }

    /// The bin of every training row in feature `j`.
    pub(super) fn column(&self, j: usize) -> &[u8] {
        &self.codes[j * self.rows..(j + 1) * self.rows]
    }

    /// The histogram slots of all features.
    pub(super) fn slots(&self) -> usize {
        self.offsets[self.width]
    }

    /// The codes a pass reads, or nothing if it has no row to sum.
    pub(super) fn pass_codes<'a>(&'a self, pass: &'a Pass) -> Option<&'a [u8]> {
        match pass {
            Pass::Plain { feature } => Some(self.column(*feature)),
            Pass::Bundle { codes, .. } => (!codes.is_empty()).then_some(codes.as_slice()),
        }
    }
}

/// A bundle being built.
pub(super) struct Building {
    /// One bit per row: some feature of the bundle is away from its dominant bin.
    pub(super) taken: Vec<u64>,
    pub(super) codes: Vec<u8>,
    pub(super) used_codes: usize,
    pub(super) features: Vec<BundledFeature>,
}

/// One pass per feature, except that features with one dominant bin share a pass wherever none of
/// their other bins meet on a row. This only decides how cells get filled, not what they hold.
pub(super) fn plan_passes(features: &[FeatureBins], n: usize, bundle: bool) -> Vec<Pass> {
    let mut passes = Vec::new();
    let mut candidates = Vec::new();
    for (j, f) in features.iter().enumerate() {
        if bundle && (f.others as f64) <= (1.0 - BUNDLE_SHARE) * n as f64 {
            candidates.push(j);
        } else {
            passes.push(Pass::Plain { feature: j });
        }
    }
    // The fullest first, as they are the hardest to place.
    candidates.sort_by_key(|&j| (std::cmp::Reverse(features[j].others), j));
    let mut bundles: Vec<Building> = Vec::new();
    for j in candidates {
        let f = &features[j];
        let bins = f.thresholds.len() + 1;
        let away: Vec<usize> = (0..n).filter(|&r| f.codes[r] != f.dominant).collect();
        let fits = |b: &Building| b.used_codes + bins - 1 <= 255 && away.iter().all(|&r| b.taken[r / 64] >> (r % 64) & 1 == 0);
        let tried_from = bundles.len().saturating_sub(MAX_BUNDLES_TRIED);
        let at = match bundles[tried_from..].iter().position(fits) {
            Some(i) => tried_from + i,
            None => {
                bundles.push(Building {
                    taken: vec![0; n.div_ceil(64)],
                    codes: vec![0; n],
                    used_codes: 0,
                    features: Vec::new(),
                });
                bundles.len() - 1
            }
        };
        let b = &mut bundles[at];
        let first_code = b.used_codes + 1;
        let mut code_of = [0u8; 256];
        let mut code = first_code;
        for bin in 0..bins {
            if bin != f.dominant as usize {
                code_of[bin] = code as u8;
                code += 1;
            }
        }
        for &r in &away {
            b.codes[r] = code_of[f.codes[r] as usize];
            b.taken[r / 64] |= 1 << (r % 64);
        }
        b.used_codes = code - 1;
        b.features.push(BundledFeature {
            feature: j,
            dominant: f.dominant,
            first_code: first_code.min(255) as u8,
        });
    }
    for b in bundles {
        passes.push(match (b.used_codes, b.features.as_slice()) {
            // Alone, a feature's own column will do.
            (1.., [only]) => Pass::Plain { feature: only.feature },
            (0, _) => Pass::Bundle {
                codes: Vec::new(),
                features: b.features,
            },
            _ => Pass::Bundle {
                codes: b.codes,
                features: b.features,
            },
        });
    }
    passes
}
