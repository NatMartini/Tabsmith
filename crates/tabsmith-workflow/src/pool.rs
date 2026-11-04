//! Workers that wait by spinning, for parallel sections that last microseconds.
//!
//! A boosting round runs thousands of short sections. Under a pool whose threads sleep between
//! sections, each section pays to wake them. Measured with rayon on 8 threads after an idle gap
//! of 100 µs: 20 to 40 µs per section on Windows, and 540 to 700 µs in WSL 2, where waking a
//! sleeping virtual CPU goes through the hypervisor. 100 µs of work then takes longer on 8 threads
//! than on one. A worker that spins between sections starts one in a microsecond or two on both
//! systems (`reports/benchmarks/`). So a worker here spins for about a millisecond, and only then
//! parks until the next section.
//!
//! There is no unsafe code. A section's closure can borrow only what outlives the pool (`'env`),
//! and what its items share with each other are atomics or locks.

use std::any::Any;
use std::hint::spin_loop;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::sync::atomic::Ordering::{AcqRel, Acquire, Relaxed, SeqCst};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, PoisonError};
use std::thread::Thread;

/// Calls of `spin_loop` before a waiting worker parks: about a millisecond on current CPUs.
const SPINS_BEFORE_PARKING: u32 = 1 << 15;
/// Calls of `spin_loop` between yields while the calling thread waits for a section's last item.
const SPINS_BEFORE_YIELDING: u32 = 1 << 12;

type Panic = Box<dyn Any + Send>;

/// One parallel section. Its counters live here, so a late worker cannot touch a newer section's.
struct Section<'env> {
    id: u32,
    items: usize,
    finished: AtomicUsize,
    panic: Mutex<Option<Panic>>,
    body: Box<dyn Fn(usize) + Send + Sync + 'env>,
}

/// Keeps the word that every worker reads apart from anything that is written more often.
#[repr(align(64))]
struct CacheLine<T>(T);

/// The calling thread and its workers. Made by [`with_pool`].
pub struct Pool<'env> {
    /// The current section's id and its next item, in one word: a worker that arrives late
    /// cannot take an item of a newer section and run it with an older closure.
    next: CacheLine<AtomicU64>,
    section: Mutex<Option<Arc<Section<'env>>>>,
    last_id: AtomicU32,
    parked: AtomicUsize,
    stop: AtomicBool,
    workers: OnceLock<Vec<Thread>>,
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

impl<'env> Pool<'env> {
    /// How many threads run a section's items, the calling one included.
    pub fn threads(&self) -> usize {
        self.workers.get().map_or(1, |w| w.len() + 1)
    }

    /// Runs `f(i)` once for every `i < items`, on the calling thread and the workers, in no
    /// particular order, and returns when all have finished. A panic in an item is raised here.
    pub fn run(&self, items: usize, f: impl Fn(usize) + Send + Sync + 'env) {
        if items <= 1 || self.threads() == 1 {
            for i in 0..items {
                f(i);
            }
            return;
        }
        let id = self.last_id.fetch_add(1, Relaxed).wrapping_add(1);
        let section = Arc::new(Section {
            id,
            items,
            finished: AtomicUsize::new(0),
            panic: Mutex::new(None),
            body: Box::new(f),
        });
        *lock(&self.section) = Some(section.clone());
        // With `parked` below, the two sides of a handshake: a worker that is about to park
        // either sees this store, or is counted by the load that follows it.
        self.next.0.store(u64::from(id) << 32, SeqCst);
        if self.parked.load(SeqCst) > 0 {
            self.wake_all();
        }
        self.help(&section);
        let mut spins = 0u32;
        while section.finished.load(Acquire) < items {
            spin_loop();
            spins += 1;
            if spins == SPINS_BEFORE_YIELDING {
                std::thread::yield_now();
                spins = 0;
            }
        }
        if let Some(payload) = lock(&section.panic).take() {
            resume_unwind(payload);
        }
    }

    fn wake_all(&self) {
        for worker in self.workers.get().into_iter().flatten() {
            worker.unpark();
        }
    }

    /// Takes items of `section` until none is left.
    fn help(&self, section: &Section<'env>) {
        loop {
            let current = self.next.0.load(Acquire);
            if (current >> 32) as u32 != section.id {
                return;
            }
            let item = current as u32 as usize;
            if item >= section.items {
                return;
            }
            if self.next.0.compare_exchange_weak(current, current + 1, AcqRel, Relaxed).is_ok() {
                if let Err(payload) = catch_unwind(AssertUnwindSafe(|| (section.body)(item))) {
                    *lock(&section.panic) = Some(payload);
                }
                section.finished.fetch_add(1, AcqRel);
            }
        }
    }

    /// What a worker thread does until the pool stops.
    fn work(&self) {
        let mut seen = 0u32;
        loop {
            let mut spins = 0u32;
            while (self.next.0.load(Acquire) >> 32) as u32 == seen {
                if self.stop.load(Relaxed) {
                    return;
                }
                spin_loop();
                spins += 1;
                if spins == SPINS_BEFORE_PARKING {
                    self.parked.fetch_add(1, SeqCst);
                    if (self.next.0.load(SeqCst) >> 32) as u32 == seen && !self.stop.load(SeqCst) {
                        std::thread::park();
                    }
                    self.parked.fetch_sub(1, SeqCst);
                    spins = 0;
                }
            }
            let section = lock(&self.section).clone();
            if let Some(section) = section {
                seen = section.id;
                self.help(&section);
            }
        }
    }
}

/// Stops the workers when the body returns or panics, so that the scope can join them.
struct StopOnDrop<'a, 'env>(&'a Pool<'env>);

impl Drop for StopOnDrop<'_, '_> {
    fn drop(&mut self) {
        self.0.stop.store(true, SeqCst);
        self.0.wake_all();
    }
}

/// Runs `body` with a pool of up to `threads` threads, the calling one included. If the system
/// cannot start a thread, the pool has fewer, which changes nothing but the time.
pub fn with_pool<'env, R>(threads: usize, body: impl FnOnce(&Pool<'env>) -> R) -> R {
    let pool = Pool {
        next: CacheLine(AtomicU64::new(0)),
        section: Mutex::new(None),
        last_id: AtomicU32::new(0),
        parked: AtomicUsize::new(0),
        stop: AtomicBool::new(false),
        workers: OnceLock::new(),
    };
    std::thread::scope(|s| {
        let pool = &pool;
        let workers: Vec<Thread> = (1..threads)
            .map_while(|i| {
                std::thread::Builder::new()
                    .name(format!("tabsmith-{i}"))
                    .spawn_scoped(s, move || pool.work())
                    .ok()
                    .map(|handle| handle.thread().clone())
            })
            .collect();
        pool.workers.set(workers).expect("set once, here");
        let _stop = StopOnDrop(pool);
        body(pool)
    })
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    #[test]
    fn every_item_runs_exactly_once() {
        for threads in [1, 2, 7] {
            let counts: Vec<AtomicUsize> = (0..1000).map(|_| AtomicUsize::new(0)).collect();
            with_pool(threads, |pool| {
                assert_eq!(pool.threads(), threads);
                for items in [0, 1, 2, 3, 10, 1000] {
                    pool.run(items, {
                        let counts = &counts;
                        move |i| {
                            counts[i].fetch_add(1, Relaxed);
                        }
                    });
                }
            });
            let expected = |i: usize| [1, 2, 3, 10, 1000].iter().filter(|&&items| i < items).count();
            for (i, c) in counts.iter().enumerate() {
                assert_eq!(c.load(Relaxed), expected(i), "item {i} on {threads} threads");
            }
        }
    }

    #[test]
    fn a_late_worker_never_mixes_two_sections() {
        // Many short sections in a row, each with its own closure and its own counters: if a
        // worker ran an item of a newer section with an older closure, a count would be off.
        with_pool(8, |pool| {
            for section in 0..20_000usize {
                let items = 2 + section % 5;
                let counts: Arc<Vec<AtomicUsize>> = Arc::new((0..items).map(|_| AtomicUsize::new(0)).collect());
                let in_section = counts.clone();
                pool.run(items, move |i| {
                    in_section[i].fetch_add(1, Relaxed);
                });
                assert!(counts.iter().all(|c| c.load(Relaxed) == 1), "section {section}");
            }
        });
    }

    #[test]
    fn workers_come_back_after_parking() {
        let total = AtomicUsize::new(0);
        with_pool(4, |pool| {
            for _ in 0..3 {
                pool.run(64, {
                    let total = &total;
                    move |_| {
                        total.fetch_add(1, Relaxed);
                    }
                });
                // Longer than any spin: the workers park.
                std::thread::sleep(Duration::from_millis(50));
            }
        });
        assert_eq!(total.load(Relaxed), 3 * 64);
    }

    #[test]
    fn a_panic_in_an_item_reaches_the_caller_and_the_workers_stop() {
        let outcome = catch_unwind(|| {
            with_pool(4, |pool| {
                pool.run(32, |i| {
                    assert!(i != 7, "item seven");
                });
            });
        });
        let payload = outcome.expect_err("the panic is raised on the calling thread");
        let message = payload.downcast_ref::<&str>().copied().unwrap_or("");
        assert!(message.contains("item seven"), "{message}");
    }
}
