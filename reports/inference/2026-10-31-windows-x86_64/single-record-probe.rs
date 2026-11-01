//! Median time of predict_one, over different records, as the README benchmark times it:
//! one call per clock reading.
//!     single-probe <bundle-dir> <records.jsonl> <iterations>
use std::hint::black_box;
use std::time::Instant;

use tabsmith_runtime::{LoadOptions, Predictor, Record};

fn main() {
    let mut a = std::env::args().skip(1);
    let (bundle, input) = (a.next().unwrap(), a.next().unwrap());
    let iterations: usize = a.next().unwrap().parse().unwrap();
    let p = Predictor::load(&bundle, LoadOptions::cpu()).unwrap();
    let records: Vec<Record> = std::fs::read_to_string(&input)
        .unwrap()
        .lines()
        .filter(|l| !l.trim().is_empty())
        .take(1000)
        .map(|l| Record::from_json(&serde_json::from_str(l).unwrap()).unwrap())
        .collect();
    let mut one = Vec::with_capacity(iterations);
    for i in 0..iterations {
        let r = &records[i % records.len()];
        let start = Instant::now();
        black_box(p.predict_one(black_box(r)).unwrap());
        one.push(start.elapsed().as_secs_f64() * 1e6);
    }
    one.sort_by(f64::total_cmp);
    let same = &records[0];
    let mut repeat = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        black_box(p.predict_one(black_box(same)).unwrap());
        repeat.push(start.elapsed().as_secs_f64() * 1e6);
    }
    repeat.sort_by(f64::total_cmp);
    let batch: Vec<Record> = records[..64.min(records.len())].to_vec();
    let mut small = Vec::new();
    for _ in 0..2000 {
        let start = Instant::now();
        black_box(p.predict(black_box(&batch)).unwrap());
        small.push(start.elapsed().as_secs_f64() * 1e6 / batch.len() as f64);
    }
    small.sort_by(f64::total_cmp);
    println!(
        "varied records p50 {:.3} us  p99 {:.3} us | same record p50 {:.3} us | batch of {} p50 {:.3} us/row",
        one[iterations / 2],
        one[iterations * 99 / 100],
        repeat[iterations / 2],
        batch.len(),
        small[small.len() / 2]
    );
}
