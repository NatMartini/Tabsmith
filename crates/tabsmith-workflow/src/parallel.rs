//! The thread pool of the parts of the workflow that run on several threads: the trees'
//! training, row fingerprints, permutation importance, per-row losses and metrics.
//!
//! Everything run here is a pure function of its index (a feature, a row, a leaf), and
//! the results are combined in index order, so nothing depends on the number of threads.
//! The pool belongs to this crate: an application's own rayon configuration is left alone.

use std::sync::OnceLock;

use rayon::prelude::*;
use rayon::{ThreadPool, ThreadPoolBuilder};

/// At most this many threads unless `RAYON_NUM_THREADS` says otherwise. A boosting round
/// has many short parallel sections, and memory bandwidth limits the long ones. Measured
/// on 16 cores with 32 logical CPUs: 12 threads train within about 9% of the best count
/// on Windows and in WSL; 32 threads gain 8% at most on Windows and lose 31% in WSL.
const MAX_THREADS: usize = 12;

/// Below this many rows, [`map_rows`] stays on the calling thread.
const MIN_ROWS: usize = 1 << 14;

/// The threads of the pool: `RAYON_NUM_THREADS` when it is set to a positive number,
/// otherwise the logical CPUs, up to [`MAX_THREADS`].
pub fn threads() -> usize {
    match std::env::var("RAYON_NUM_THREADS").ok().and_then(|v| v.trim().parse::<usize>().ok()) {
        Some(n) if n > 0 => n,
        _ => std::thread::available_parallelism().map_or(1, |n| n.get()).min(MAX_THREADS),
    }
}

/// Runs `f` in the pool, so that its parallel sections do not cross threads to start.
pub fn install<R: Send>(f: impl FnOnce() -> R + Send) -> R {
    static POOL: OnceLock<ThreadPool> = OnceLock::new();
    POOL.get_or_init(|| {
        ThreadPoolBuilder::new()
            .num_threads(threads())
            .thread_name(|i| format!("tabsmith-{i}"))
            .build()
            .expect("the operating system can start threads")
    })
    .install(f)
}

/// `f(state, r)` for every `r < n`, in order, with one `init()` state per piece of work.
/// Stays on the calling thread when `n` is too small to be worth waking others.
pub fn map_rows_with<S, T: Send>(n: usize, init: impl Fn() -> S + Send + Sync, f: impl Fn(&mut S, usize) -> T + Send + Sync) -> Vec<T> {
    if n < MIN_ROWS {
        let mut state = init();
        (0..n).map(|r| f(&mut state, r)).collect()
    } else {
        install(|| {
            (0..n)
                .into_par_iter()
                .with_min_len(MIN_ROWS / 8)
                .map_init(init, |s, r| f(s, r))
                .collect()
        })
    }
}

/// `f(r)` for every `r < n`, in order. See [`map_rows_with`].
pub fn map_rows<T: Send>(n: usize, f: impl Fn(usize) -> T + Send + Sync) -> Vec<T> {
    map_rows_with(n, || (), |_, r| f(r))
}

/// The mean of `f(r)` over `n` rows: `f` runs on several threads, the sum in row order,
/// so the result is the plain sum's, bit for bit, on any number of threads.
pub fn mean_in_row_order(n: usize, f: impl Fn(usize) -> f64 + Send + Sync) -> f64 {
    let mut sum = 0.0;
    for v in map_rows(n, f) {
        sum += v;
    }
    sum / n.max(1) as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_keep_their_order_on_both_paths() {
        for n in [0, 10, MIN_ROWS, 3 * MIN_ROWS + 5] {
            let v = map_rows(n, |r| r * 2);
            assert_eq!(v, (0..n).map(|r| r * 2).collect::<Vec<_>>());
        }
    }

    #[test]
    fn the_mean_is_the_plain_sums_bit_for_bit() {
        let f = |r: usize| ((r as f64 + 0.5).sin() * 1e8).fract();
        for n in [100, 2 * MIN_ROWS] {
            let mut sum = 0.0;
            for r in 0..n {
                sum += f(r);
            }
            assert_eq!(mean_in_row_order(n, f).to_bits(), (sum / n as f64).to_bits());
        }
    }
}
