//! What one parallel section costs when the threads have been idle for a while: rayon, whose
//! workers go to sleep within microseconds, against workers that spin for a bounded time.
//!
//!     poolbench <threads>

use std::hint::{black_box, spin_loop};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use rayon::prelude::*;

fn busy(d: Duration) {
    let t = Instant::now();
    while t.elapsed() < d {
        spin_loop();
    }
}

fn work(units: u64) -> u64 {
    let mut x = 0x9e37_79b9_7f4a_7c15u64;
    for _ in 0..units {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
    }
    black_box(x)
}

const REGIONS: usize = 2000;

fn calibrate() -> u64 {
    // Units of `work` that take about a microsecond.
    let t = Instant::now();
    work(10_000_000);
    (10_000_000.0 / (t.elapsed().as_secs_f64() * 1e6)) as u64
}

fn serial(gap: Duration, per_item: u64, items: usize) -> f64 {
    let t = Instant::now();
    for _ in 0..REGIONS {
        busy(gap);
        for _ in 0..items {
            work(per_item);
        }
    }
    t.elapsed().as_secs_f64()
}

fn rayon_regions(pool: &rayon::ThreadPool, gap: Duration, per_item: u64, items: usize) -> f64 {
    pool.install(|| {
        let t = Instant::now();
        for _ in 0..REGIONS {
            busy(gap);
            (0..items).into_par_iter().for_each(|_| {
                work(per_item);
            });
        }
        t.elapsed().as_secs_f64()
    })
}

fn spin_regions(threads: usize, gap: Duration, per_item: u64, items: usize, spin_limit: Duration) -> f64 {
    let generation = AtomicU64::new(0);
    let next = AtomicUsize::new(0);
    let done = AtomicUsize::new(0);
    std::thread::scope(|s| {
        for _ in 0..threads - 1 {
            s.spawn(|| {
                let mut seen = 0;
                loop {
                    let started = Instant::now();
                    let mut spins = 0u32;
                    let current = loop {
                        let g = generation.load(Ordering::Acquire);
                        if g != seen {
                            break g;
                        }
                        spin_loop();
                        spins += 1;
                        if spins % 1024 == 0 && started.elapsed() > spin_limit {
                            std::thread::yield_now();
                        }
                    };
                    seen = current;
                    if current == u64::MAX {
                        return;
                    }
                    loop {
                        let i = next.fetch_add(1, Ordering::AcqRel);
                        if i >= items {
                            break;
                        }
                        work(per_item);
                        done.fetch_add(1, Ordering::AcqRel);
                    }
                }
            });
        }
        let t = Instant::now();
        for g in 1..=REGIONS as u64 {
            busy(gap);
            next.store(0, Ordering::Release);
            done.store(0, Ordering::Release);
            generation.store(g, Ordering::Release);
            loop {
                let i = next.fetch_add(1, Ordering::AcqRel);
                if i >= items {
                    break;
                }
                work(per_item);
                done.fetch_add(1, Ordering::AcqRel);
            }
            while done.load(Ordering::Acquire) < items {
                spin_loop();
            }
        }
        let elapsed = t.elapsed().as_secs_f64();
        generation.store(u64::MAX, Ordering::Release);
        elapsed
    })
}

fn main() {
    let threads: usize = std::env::args().nth(1).unwrap().parse().unwrap();
    let per_us = calibrate();
    let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
    println!("threads {threads}; a parallel section of {threads} x items after an idle gap; microseconds per section");
    println!("gap_us work_us_serial | serial | rayon | spin | rayon_speedup spin_speedup | rayon_overhead_us spin_overhead_us");
    for gap_us in [0u64, 20, 100, 1000] {
        for work_us in [10u64, 100, 1000] {
            let gap = Duration::from_micros(gap_us);
            let items = threads * 4;
            let per_item = work_us * per_us / items as u64;
            let a = serial(gap, per_item, items) / REGIONS as f64 * 1e6;
            let b = rayon_regions(&pool, gap, per_item, items) / REGIONS as f64 * 1e6;
            let c = spin_regions(threads, gap, per_item, items, Duration::from_millis(5)) / REGIONS as f64 * 1e6;
            let ideal = gap_us as f64 + (a - gap_us as f64) / threads as f64;
            println!(
                "{gap_us:>5} {work_us:>5} | {a:>8.1} | {b:>8.1} | {c:>8.1} | {:>5.2} {:>5.2} | {:>7.1} {:>7.1}",
                (a - gap_us as f64) / (b - gap_us as f64).max(1e-9),
                (a - gap_us as f64) / (c - gap_us as f64).max(1e-9),
                b - ideal,
                c - ideal
            );
        }
    }
}
