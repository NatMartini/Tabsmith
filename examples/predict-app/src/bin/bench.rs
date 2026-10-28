//! In-application prediction benchmark: bundle load time, single-record latency
//! and batch throughput, printed as one JSON object.
//!
//!     cargo run --release --bin bench -- <bundle-dir> <records.jsonl> [iterations]
use std::time::Instant;

use tabsmith_runtime::{LoadOptions, Predictor, Record};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let bundle = args.next().ok_or("usage: bench <bundle-dir> <records.jsonl> [iterations]")?;
    let input = args.next().ok_or("usage: bench <bundle-dir> <records.jsonl> [iterations]")?;
    let iterations: usize = args.next().map(|s| s.parse()).transpose()?.unwrap_or(20_000);

    let t = Instant::now();
    let predictor = Predictor::load(&bundle, LoadOptions::cpu())?;
    let load_ms = t.elapsed().as_secs_f64() * 1e3;

    let records: Vec<Record> = std::fs::read_to_string(&input)?
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| Ok(Record::from_json(&serde_json::from_str(l)?)?))
        .collect::<Result<_, Box<dyn std::error::Error>>>()?;

    // Single record at a time, as a request handler would call it.
    for r in records.iter().take(200) {
        std::hint::black_box(predictor.predict_one(r)?);
    }
    let mut us = Vec::with_capacity(iterations);
    for i in 0..iterations {
        let r = &records[i % records.len()];
        let t = Instant::now();
        std::hint::black_box(predictor.predict_one(r)?);
        us.push(t.elapsed().as_secs_f64() * 1e6);
    }
    us.sort_by(f64::total_cmp);
    let q = |f: f64| us[((us.len() - 1) as f64 * f).round() as usize];

    // Whole batch in one call.
    let rounds = 5;
    let t = Instant::now();
    for _ in 0..rounds {
        std::hint::black_box(predictor.predict(&records)?);
    }
    let batch_s = t.elapsed().as_secs_f64() / rounds as f64;

    println!(
        "{}",
        serde_json::json!({
            "bundle_id": predictor.bundle_id(),
            "load_ms": load_ms,
            "single_record_us": {"p50": q(0.5), "p95": q(0.95), "p99": q(0.99), "iterations": iterations},
            "batch": {"rows": records.len(), "seconds": batch_s, "rows_per_second": records.len() as f64 / batch_s},
        })
    );
    Ok(())
}
