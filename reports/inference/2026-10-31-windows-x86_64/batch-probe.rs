//! Where one batch prediction spends its time, and how it scales with threads.
//!     infer-probe <bundle-dir> <records.jsonl>
use std::hint::black_box;
use std::time::Instant;

use tabsmith_runtime::transform::apply_transform;
use tabsmith_runtime::{LoadOptions, Phase, Predictor, Record};

fn median_s(rounds: usize, mut f: impl FnMut()) -> f64 {
    let mut t = Vec::new();
    for _ in 0..rounds {
        let s = Instant::now();
        f();
        t.push(s.elapsed().as_secs_f64());
    }
    t.sort_by(f64::total_cmp);
    t[rounds / 2]
}

fn main() {
    let mut a = std::env::args().skip(1);
    let (bundle, input) = (a.next().unwrap(), a.next().unwrap());
    let load = |threads: usize| Predictor::load(&bundle, LoadOptions::cpu().with_threads(threads)).unwrap();
    let one = load(1);
    let base: Vec<Record> = std::fs::read_to_string(&input)
        .unwrap()
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| Record::from_json(&serde_json::from_str(l).unwrap()).unwrap())
        .collect();

    let records = &base;
    let n = records.len();
    let table = one.records_table(records, &[]).unwrap();
    let x = apply_transform(&one.bundle().transform, &table, Phase::Predict).unwrap();
    println!("{bundle}: {n} rows, {} features, default threads {}", x.cols, std::thread::available_parallelism().unwrap().get().min(16));

    println!("\nOne thread, by stage:");
    let line = |name: &str, s: f64| println!("  {name:32} {:>9.2} ms {:>12.0} rows/s {:>8.3} us/row", s * 1e3, n as f64 / s, s * 1e6 / n as f64);
    line("predict(records)", median_s(15, || {
        black_box(one.predict(black_box(records)).unwrap());
    }));
    line("records -> table", median_s(15, || {
        black_box(one.records_table(black_box(records), &[]).unwrap());
    }));
    line("table -> features (transform)", median_s(15, || {
        black_box(apply_transform(&one.bundle().transform, black_box(&table), Phase::Predict).unwrap());
    }));
    line("features -> outputs (model)", median_s(15, || {
        black_box(one.outputs_matrix(black_box(&x)));
    }));
    line("table -> predictions", median_s(15, || {
        black_box(one.predict_table(black_box(&table)).unwrap());
    }));

    for rows in [1024usize, 4096, 10_000, 100_000, 1_000_000] {
        let many: Vec<Record> = (0..rows).map(|i| base[i % n].clone()).collect();
        let table = one.records_table(&many, &[]).unwrap();
        let x = apply_transform(&one.bundle().transform, &table, Phase::Predict).unwrap();
        let rounds = if rows >= 1_000_000 { 3 } else if rows >= 100_000 { 7 } else { 21 };
        println!("\n{rows} rows, by threads (rows/s):\n  threads     records       table    features");
        for threads in [1usize, 2, 4, 6, 8, 12, 16, 24, 32] {
            let p = load(threads);
            let a = median_s(rounds, || {
                black_box(p.predict(black_box(&many)).unwrap());
            });
            let b = median_s(rounds, || {
                black_box(p.predict_table(black_box(&table)).unwrap());
            });
            let c = median_s(rounds, || {
                black_box(p.outputs_matrix(black_box(&x)));
            });
            let r = rows as f64;
            println!("  {threads:>7} {:>11.0} {:>11.0} {:>11.0}", r / a, r / b, r / c);
        }
    }
}
