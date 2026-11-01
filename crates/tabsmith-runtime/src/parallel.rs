//! A large batch on several threads. Every row is computed exactly as on one thread and
//! the results are joined in row order, so nothing depends on the number of threads.
//! Plain scoped threads: the runtime stays free of dependencies, and a batch is rare
//! enough, and long enough, for starting threads not to matter.

use std::ops::Range;

/// A batch with fewer rows stays on the calling thread.
pub(crate) const MIN_ROWS: usize = 4096;
/// Rows a thread gets at least, so that a small batch starts few threads.
const MIN_ROWS_PER_THREAD: usize = 1024;
/// Threads when the options leave the choice to the runtime.
const DEFAULT_MAX_THREADS: usize = 16;

/// The threads a batch may use, from `LoadOptions::threads` (0: the logical CPUs, up to
/// `DEFAULT_MAX_THREADS`). WebAssembly has no threads here.
pub(crate) fn resolve(requested: usize) -> usize {
    if cfg!(target_family = "wasm") {
        1
    } else if requested > 0 {
        requested
    } else {
        std::thread::available_parallelism().map_or(1, |n| n.get()).min(DEFAULT_MAX_THREADS)
    }
}

/// How many pieces a batch of `n` rows runs in.
pub(crate) fn pieces(n: usize, threads: usize) -> usize {
    if n < MIN_ROWS {
        1
    } else {
        threads.min(n / MIN_ROWS_PER_THREAD).max(1)
    }
}

/// `f` on `pieces` consecutive ranges of `0..n`, the results in order. The calling thread
/// takes the first range, and any range for which no thread could be started.
pub(crate) fn map_ranges<T: Send>(n: usize, pieces: usize, f: impl Fn(Range<usize>) -> T + Sync) -> Vec<T> {
    if pieces <= 1 {
        return vec![f(0..n)];
    }
    let per = n.div_ceil(pieces);
    let f = &f;
    std::thread::scope(|s| {
        let others: Vec<_> = (1..pieces)
            .map(|i| {
                let rows = (i * per).min(n)..((i + 1) * per).min(n);
                let job = rows.clone();
                std::thread::Builder::new().spawn_scoped(s, move || f(job)).map_err(|_| rows)
            })
            .collect();
        let mut out = Vec::with_capacity(pieces);
        out.push(f(0..per.min(n)));
        out.extend(others.into_iter().map(|started| match started {
            Ok(thread) => thread.join().unwrap_or_else(|panic| std::panic::resume_unwind(panic)),
            Err(rows) => f(rows),
        }));
        out
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_ranges_cover_the_rows_in_order() {
        for (n, parts) in [(0usize, 1usize), (10, 1), (10, 3), (10_000, 7), (10_001, 16), (7, 16)] {
            let ranges = map_ranges(n, parts, |r| r);
            let rows: Vec<usize> = ranges.into_iter().flatten().collect();
            assert_eq!(rows, (0..n).collect::<Vec<_>>(), "{n} rows in {parts} pieces");
        }
    }

    #[test]
    fn small_batches_stay_on_one_thread() {
        assert_eq!(pieces(MIN_ROWS - 1, 16), 1);
        assert_eq!(pieces(MIN_ROWS, 16), 4);
        assert_eq!(pieces(100_000, 16), 16);
        assert_eq!(pieces(100_000, 1), 1);
    }
}
