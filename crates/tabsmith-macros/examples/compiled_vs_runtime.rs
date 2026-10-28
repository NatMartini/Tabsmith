//! Compiled models next to the runtime, on the same bundles.
//!
//!     cargo run --release -p tabsmith-macros --example compiled_vs_runtime
//!
//! For each fixture bundle it prints:
//! - a hash of the predictions for 100,000 pseudo-random records, from the compiled
//!   model and from the runtime. The two are equal on one machine, and equal across
//!   operating systems, because the arithmetic, `exp` included, is pure Rust (the
//!   `portable` test checks the latter on every CI platform);
//! - the median time of one prediction: the runtime from a `Record`, the compiled model
//!   from raw values, and the compiled model from its typed `Input`.

use std::hint::black_box;
use std::time::Instant;

use tabsmith_embedded::FieldValue;
use tabsmith_runtime::{LoadOptions, Predictor};

#[path = "../tests/sample/mod.rs"]
mod sample;

use sample::{FNV_OFFSET, Rng, fnv, record, row, runtime_numbers};

tabsmith_macros::model!(binary_mlp, "../../fixtures/bundles/binary-mlp");
tabsmith_macros::model!(binary_gbdt, "../../fixtures/bundles/binary-gbdt");
tabsmith_macros::model!(multiclass_gbdt, "../../fixtures/bundles/multiclass-gbdt");
tabsmith_macros::model!(regression_gbdt, "../../fixtures/bundles/regression-gbdt");

fn median_ns(mut f: impl FnMut()) -> f64 {
    let mut times = Vec::with_capacity(51);
    for _ in 0..51 {
        let start = Instant::now();
        for _ in 0..2000 {
            f();
        }
        times.push(start.elapsed().as_nanos() as f64 / 2000.0);
    }
    times.sort_by(f64::total_cmp);
    times[25]
}

macro_rules! compare {
    ($m:ident, $bundle:literal, |$p:ident| $numbers:expr) => {{
        let dir = format!("{}/../../fixtures/bundles/{}", env!("CARGO_MANIFEST_DIR"), $bundle);
        let rt = Predictor::load(dir, LoadOptions::cpu()).unwrap();
        let mut rng = Rng(7);
        let (mut compiled, mut interpreted) = (FNV_OFFSET, FNV_OFFSET);
        for _ in 0..100_000 {
            let values = row(&$m::FIELDS, &mut rng);
            let values: [FieldValue<'_>; $m::N_FIELDS] = values.try_into().unwrap();
            let $p = $m::predict_values(&values).unwrap();
            for v in $numbers {
                fnv(&mut compiled, v.to_bits());
            }
            let p = rt.predict_one(&record(&$m::FIELDS, &values)).unwrap();
            for v in runtime_numbers(&p.outcome) {
                fnv(&mut interpreted, v.to_bits());
            }
        }
        let values: [FieldValue<'_>; $m::N_FIELDS] = $m::EXAMPLES[0].values();
        let rec = record(&$m::FIELDS, &values);
        let input = $m::EXAMPLES[0];
        let t_runtime = median_ns(|| {
            black_box(rt.predict_one(black_box(&rec)).unwrap());
        });
        let t_values = median_ns(|| {
            black_box($m::predict_values(black_box(&values)).unwrap());
        });
        let t_typed = median_ns(|| {
            black_box($m::predict(black_box(&input)).unwrap());
        });
        println!(
            "{:<18} compiled {compiled:016x}  runtime {interpreted:016x}  {}  | runtime {:>8.0} ns  compiled(values) {:>6.0} ns  compiled(Input) {:>6.0} ns",
            $bundle,
            if compiled == interpreted { "equal" } else { "DIFFERENT" },
            t_runtime,
            t_values,
            t_typed
        );
        assert_eq!(compiled, interpreted, "{}: compiled and runtime predictions differ", $bundle);
    }};
}

fn main() {
    println!("{} {}", std::env::consts::OS, std::env::consts::ARCH);
    compare!(binary_mlp, "binary-mlp", |p| [p.probability]);
    compare!(binary_gbdt, "binary-gbdt", |p| [p.probability]);
    compare!(multiclass_gbdt, "multiclass-gbdt", |p| p.probabilities);
    compare!(regression_gbdt, "regression-gbdt", |p| [p.value]);
}
