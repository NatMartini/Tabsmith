//! Reads JSON records (one per line) on stdin and prints `label<TAB>probability`
//! (a regression bundle: the predicted value).
//!
//!     cargo run --release -- ../../my-project/bundles/run-002 < readings.jsonl
use std::io::BufRead;

use tabsmith_runtime::{LoadOptions, Predictor, Record};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let bundle = std::env::args().nth(1).ok_or("usage: predict-app <bundle-dir> < records.jsonl")?;
    let predictor = Predictor::load(&bundle, LoadOptions::cpu())?;
    eprintln!(
        "loaded bundle {} ({})",
        predictor.bundle_id(),
        predictor.manifest().evaluation_status
    );
    for line in std::io::stdin().lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let record = Record::from_json(&serde_json::from_str(&line)?)?;
        match predictor.predict_one(&record) {
            Ok(p) => match p.probability() {
                Some(q) => println!("{}\t{q:.6}", p.label),
                None => println!("{}", p.label),
            },
            Err(e) => eprintln!("{}", e.render()),
        }
    }
    Ok(())
}
