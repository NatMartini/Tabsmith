//! The rows of the trainer and what a tree does with them: histograms, partitions, split search.
//! Everything that crosses threads here is an integer in an atomic, and a section's closure can
//! borrow only what outlives the pool, so the compiler sees that no data race is possible.

use std::ops::{AddAssign, Sub, SubAssign};
use std::sync::atomic::Ordering::Relaxed;
use std::sync::atomic::{AtomicI64, AtomicU32, AtomicU64, AtomicUsize};
use std::sync::{Mutex, MutexGuard, OnceLock, PoisonError};

use super::bins::{Bins, Pass};
use super::grow::{Growth, Split, feature_split};
use crate::pool::Pool;

/// Rows of one piece of work when a loop over rows runs on several threads.
pub(super) const PIECE_ROWS: usize = 1 << 12;
/// Below this many row visits a section stays on the calling thread: starting the workers costs
/// a microsecond or two, about as much as this work.
pub(super) const MIN_SECTION_WORK: usize = 1 << 13;
/// A histogram with fewer slots than this is searched and subtracted on the calling thread.
const PAR_MIN_SLOTS: usize = 1 << 15;
/// Passes over fewer rows than this use one table of cells; see [`fill_node`].
const TWO_TABLES_FROM: usize = 1 << 8;

pub(super) fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Sums over some rows: gradient, hessian, count. Integers at the scale of the tree.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct Cell {
    pub(super) g: i64,
    pub(super) h: i64,
    pub(super) n: i64,
}

impl AddAssign for Cell {
    #[inline(always)]
    fn add_assign(&mut self, other: Cell) {
        self.g += other.g;
        self.h += other.h;
        self.n += other.n;
    }
}

impl SubAssign for Cell {
    #[inline(always)]
    fn sub_assign(&mut self, other: Cell) {
        self.g -= other.g;
        self.h -= other.h;
        self.n -= other.n;
    }
}

impl Sub for Cell {
    type Output = Cell;

    #[inline(always)]
    fn sub(mut self, other: Cell) -> Cell {
        self -= other;
        self
    }
}

/// A row's gradient and hessian at the scale of the tree, in one word.
#[inline(always)]
pub(super) fn pack(g: i32, h: i32) -> u64 {
    u64::from(g as u32) | (u64::from(h as u32) << 32)
}

#[inline(always)]
pub(super) fn add_row(cell: &mut Cell, pair: u64) {
    cell.g += i64::from(pair as u32 as i32);
    cell.h += (pair >> 32) as i64;
    cell.n += 1;
}

/// A cell of a histogram that several threads fill. Every feature's slots are written by one item
/// of a section, and sections follow each other, so these are plain loads and stores in effect:
/// relaxed atomics compile to the same instructions on x86-64 and AArch64, and let the compiler
/// see that no data race is possible.
#[derive(Default)]
pub(super) struct Slot {
    pub(super) g: AtomicI64,
    pub(super) h: AtomicI64,
    pub(super) n: AtomicI64,
}

impl Slot {
    #[inline(always)]
    pub(super) fn get(&self) -> Cell {
        Cell {
            g: self.g.load(Relaxed),
            h: self.h.load(Relaxed),
            n: self.n.load(Relaxed),
        }
    }

    #[inline(always)]
    pub(super) fn set(&self, c: Cell) {
        self.g.store(c.g, Relaxed);
        self.h.store(c.h, Relaxed);
        self.n.store(c.n, Relaxed);
    }
}

/// Adds another table's cells to `cells`.
pub(super) fn merge(cells: &mut [Cell; 256], other: &[Cell; 256]) {
    for (a, b) in cells.iter_mut().zip(other) {
        *a += *b;
    }
}

/// One pass over all the rows, in order. Rows go to two tables in turn: most rows of a column hit
/// few cells, and a row that adds to the cell the row before it just stored to waits for that
/// store. Sums of integers can be merged afterwards without changing a bit.
pub(super) fn fill_root(codes: &[u8], gh: &[AtomicU64], cells: &mut [Cell; 256]) {
    let mut second = [Cell::default(); 256];
    let (code_pairs, last_code) = codes.as_chunks::<2>();
    let (gh_pairs, last_gh) = gh.as_chunks::<2>();
    for (c, p) in code_pairs.iter().zip(gh_pairs) {
        add_row(&mut cells[c[0] as usize], p[0].load(Relaxed));
        add_row(&mut second[c[1] as usize], p[1].load(Relaxed));
    }
    for (&c, p) in last_code.iter().zip(last_gh) {
        add_row(&mut cells[c as usize], p.load(Relaxed));
    }
    merge(cells, &second);
}

/// One pass over a node's rows, whose gradients are gathered next to each other. A short pass uses
/// one table: clearing and merging the second would cost more than it saves.
pub(super) fn fill_node(codes: &[u8], rows: &[AtomicU32], gathered: &[AtomicU64], cells: &mut [Cell; 256]) {
    if rows.len() < TWO_TABLES_FROM {
        for (r, p) in rows.iter().zip(gathered) {
            add_row(&mut cells[codes[r.load(Relaxed) as usize] as usize], p.load(Relaxed));
        }
        return;
    }
    let mut second = [Cell::default(); 256];
    let (row_pairs, last_row) = rows.as_chunks::<2>();
    let (gh_pairs, last_gh) = gathered.as_chunks::<2>();
    for (r, p) in row_pairs.iter().zip(gh_pairs) {
        add_row(&mut cells[codes[r[0].load(Relaxed) as usize] as usize], p[0].load(Relaxed));
        add_row(&mut second[codes[r[1].load(Relaxed) as usize] as usize], p[1].load(Relaxed));
    }
    for (r, p) in last_row.iter().zip(last_gh) {
        add_row(&mut cells[codes[r.load(Relaxed) as usize] as usize], p.load(Relaxed));
    }
    merge(cells, &second);
}

/// Writes a pass's cells into the histogram. `total` holds the sums of the node's rows.
pub(super) fn publish(bins: &Bins, pass: &Pass, cells: &[Cell; 256], total: Cell, hist: &[Slot]) {
    match pass {
        Pass::Plain { feature } => {
            let slots = &hist[bins.offsets[*feature]..bins.offsets[*feature + 1]];
            for (slot, cell) in slots.iter().zip(cells) {
                slot.set(*cell);
            }
        }
        Pass::Bundle { features, .. } => {
            for f in features {
                let slots = &hist[bins.offsets[f.feature]..bins.offsets[f.feature + 1]];
                let mut rest = total;
                let mut code = f.first_code as usize;
                for (bin, slot) in slots.iter().enumerate() {
                    if bin != f.dominant as usize {
                        let cell = cells[code];
                        slot.set(cell);
                        rest -= cell;
                        code += 1;
                    }
                }
                slots[f.dominant as usize].set(rest);
            }
        }
    }
}

/// The rows of the trainer and what a tree does with them, shared with the workers. Everything
/// here is allocated once and lives as long as the training.
pub(super) struct Rows {
    pub(super) bins: Bins,
    /// Gradient and hessian of every row, for the output whose tree is growing ([`pack`]).
    pub(super) gh: Vec<AtomicU64>,
    /// The gradients of the node being summed, next to each other.
    pub(super) gathered: Vec<AtomicU64>,
    /// The rows of the tree: every node is a range of this order.
    pub(super) order: Vec<AtomicU32>,
    pub(super) spare: Vec<AtomicU32>,
    pub(super) lefts: Vec<AtomicUsize>,
    /// Sums of all the rows' gradients and hessians.
    pub(super) total_g: AtomicI64,
    pub(super) total_h: AtomicI64,
    /// Histograms, made when first needed: a large tree of few leaves needs few.
    pub(super) hists: Vec<OnceLock<Box<[Slot]>>>,
    pub(super) best: Vec<Mutex<Option<Split>>>,
}

impl Rows {
    pub(super) fn new(bins: Bins, max_leaves: usize) -> Self {
        let n = bins.rows;
        let width = bins.width;
        Self {
            bins,
            gh: (0..n).map(|_| AtomicU64::new(0)).collect(),
            gathered: (0..n.div_ceil(2)).map(|_| AtomicU64::new(0)).collect(),
            order: (0..n).map(|_| AtomicU32::new(0)).collect(),
            spare: (0..n).map(|_| AtomicU32::new(0)).collect(),
            lefts: (0..n.div_ceil(PIECE_ROWS)).map(|_| AtomicUsize::new(0)).collect(),
            total_g: AtomicI64::new(0),
            total_h: AtomicI64::new(0),
            hists: (0..max_leaves + 2).map(|_| OnceLock::new()).collect(),
            best: (0..width).map(|_| Mutex::new(None)).collect(),
        }
    }

    pub(super) fn hist(&self, at: usize) -> &[Slot] {
        self.hists[at].get_or_init(|| (0..self.bins.slots()).map(|_| Slot::default()).collect())
    }

    /// Every row in the tree's order, and the sums of all rows as a cell.
    pub(super) fn begin_tree<'env>(&'env self, pool: &Pool<'env>) -> Cell {
        let n = self.bins.rows;
        section(pool, n, n.div_ceil(PIECE_ROWS), move |piece| {
            for r in piece_range(piece, n) {
                self.order[r].store(r as u32, Relaxed);
            }
        });
        Cell {
            g: self.total_g.load(Relaxed),
            h: self.total_h.load(Relaxed),
            n: n as i64,
        }
    }

    /// The histogram of all the rows, into `hists[at]`.
    pub(super) fn fill_root<'env>(&'env self, pool: &Pool<'env>, total: Cell, at: usize) {
        let passes = self.bins.passes.len();
        section(pool, self.bins.rows * passes, passes, move |j| {
            let pass = &self.bins.passes[j];
            let mut cells = [Cell::default(); 256];
            if let Some(codes) = self.bins.pass_codes(pass) {
                fill_root(codes, &self.gh, &mut cells);
            }
            publish(&self.bins, pass, &cells, total, self.hist(at));
        });
    }

    /// The histogram of the node at `start..start + len`, into `hists[at]`.
    pub(super) fn fill_node<'env>(&'env self, pool: &Pool<'env>, start: usize, len: usize, total: Cell, at: usize) {
        section(pool, len, len.div_ceil(PIECE_ROWS), move |piece| {
            for i in piece_range(piece, len) {
                let row = self.order[start + i].load(Relaxed) as usize;
                self.gathered[i].store(self.gh[row].load(Relaxed), Relaxed);
            }
        });
        let passes = self.bins.passes.len();
        section(pool, len * passes, passes, move |j| {
            let pass = &self.bins.passes[j];
            let mut cells = [Cell::default(); 256];
            if let Some(codes) = self.bins.pass_codes(pass) {
                fill_node(codes, &self.order[start..start + len], &self.gathered[..len], &mut cells);
            }
            publish(&self.bins, pass, &cells, total, self.hist(at));
        });
    }

    /// The histogram of the larger child: its parent's, in place, minus the smaller child's.
    pub(super) fn subtract<'env>(&'env self, pool: &Pool<'env>, parent: usize, smaller: usize) {
        let width = self.bins.width;
        let work = if self.bins.slots() < PAR_MIN_SLOTS { 0 } else { usize::MAX };
        section(pool, work, width, move |j| {
            let (a, b) = (self.bins.offsets[j], self.bins.offsets[j + 1]);
            for (p, s) in self.hist(parent)[a..b].iter().zip(&self.hist(smaller)[a..b]) {
                p.set(p.get() - s.get());
            }
        });
    }

    /// Moves the rows of `start..start + len` that go left (`bin <= split bin` in `feature`) to the
    /// front of the range, keeping the order on both sides, and returns how many there are.
    pub(super) fn partition<'env>(&'env self, pool: &Pool<'env>, feature: usize, bin: usize, start: usize, len: usize) -> usize {
        let codes = self.bins.column(feature);
        if len < 2 * PIECE_ROWS || pool.threads() == 1 {
            let (mut left, mut right) = (start, 0);
            for i in start..start + len {
                let row = self.order[i].load(Relaxed);
                let goes_left = usize::from(codes[row as usize] as usize <= bin);
                self.order[left].store(row, Relaxed);
                self.spare[right].store(row, Relaxed);
                left += goes_left;
                right += 1 - goes_left;
            }
            for i in 0..right {
                self.order[left + i].store(self.spare[i].load(Relaxed), Relaxed);
            }
            return left - start;
        }
        // Each piece splits itself into the spare order, then the two sides move into place.
        let pieces = len.div_ceil(PIECE_ROWS);
        pool.run(pieces, move |piece| {
            let span = piece_range(piece, len);
            let (from, to) = (start + span.start, start + span.end);
            let mut right_rows = [0u32; PIECE_ROWS];
            let (mut left, mut right) = (from, 0);
            for i in from..to {
                let row = self.order[i].load(Relaxed);
                let goes_left = usize::from(codes[row as usize] as usize <= bin);
                self.spare[left].store(row, Relaxed);
                right_rows[right] = row;
                left += goes_left;
                right += 1 - goes_left;
            }
            for (i, &row) in right_rows[..right].iter().enumerate() {
                self.spare[left + i].store(row, Relaxed);
            }
            self.lefts[piece].store(left - from, Relaxed);
        });
        let total_left: usize = self.lefts[..pieces].iter().map(|l| l.load(Relaxed)).sum();
        let mut moves = Vec::with_capacity(2 * pieces);
        let (mut at_left, mut at_right) = (start, start + total_left);
        for piece in 0..pieces {
            let span = piece_range(piece, len);
            let (from, count) = (start + span.start, span.len());
            let left = self.lefts[piece].load(Relaxed);
            moves.push((from, at_left, left));
            moves.push((from + left, at_right, count - left));
            at_left += left;
            at_right += count - left;
        }
        pool.run(moves.len(), move |m| {
            let (from, to, count) = moves[m];
            for i in 0..count {
                self.order[to + i].store(self.spare[from + i].load(Relaxed), Relaxed);
            }
        });
        total_left
    }

    /// The largest gain over all features' bins for a node with sums `total` and leaf bounds
    /// `[lower, upper]`, first in (feature, bin) order on ties.
    pub(super) fn best_split<'env>(
        &'env self,
        pool: &Pool<'env>,
        at: usize,
        total: Cell,
        grow: Growth<'env>,
        lower: f64,
        upper: f64,
    ) -> Option<Split> {
        let hist = self.hist(at);
        let width = self.bins.width;
        if self.bins.slots() < PAR_MIN_SLOTS {
            return (0..width).fold(None, |best: Option<Split>, j| {
                let s = feature_split(&self.bins, hist, j, total, &grow, lower, upper);
                pick_split(best, s)
            });
        }
        pool.run(width, move |j| {
            *lock(&self.best[j]) = feature_split(&self.bins, self.hist(at), j, total, &grow, lower, upper);
        });
        self.best.iter().fold(None, |best, s| pick_split(best, *lock(s)))
    }
}

/// The first feature wins a tie, as in one scan over all of them.
pub(super) fn pick_split(best: Option<Split>, s: Option<Split>) -> Option<Split> {
    match (best, s) {
        (Some(b), Some(s)) if s.gain > b.gain => Some(s),
        (None, s) => s,
        (best, _) => best,
    }
}

pub(super) fn piece_range(piece: usize, n: usize) -> std::ops::Range<usize> {
    piece * PIECE_ROWS..((piece + 1) * PIECE_ROWS).min(n)
}

/// Runs the items on the calling thread when the work is too small to be worth more.
pub(super) fn section<'env>(pool: &Pool<'env>, work: usize, items: usize, f: impl Fn(usize) + Send + Sync + 'env) {
    if work < MIN_SECTION_WORK {
        for i in 0..items {
            f(i);
        }
    } else {
        pool.run(items, f);
    }
}
