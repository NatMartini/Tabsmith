//! The kernels a const prediction rewrites, timed through compiled models: trees kept as
//! arrays (`Trees::outputs`, which the runtime calls too) and a dense network (`dot`, `dense`),
//! and the whole `predict`. Run it before and after the change, on the same machine.
// Covertype's bundle is not in the repository: train one on Covertype (reports/training/) and put
// it where the third `model!` path points, or remove that model and its line in `main`.
use std::hint::black_box;
use std::time::Instant;

tabsmith_macros::model!(regression, "../../../../fixtures/bundles/regression-gbdt", tree_arrays);
tabsmith_macros::model!(multiclass, "../../../../fixtures/bundles/multiclass-gbdt", tree_arrays);
tabsmith_macros::model!(covertype, "../../../../scratch/speed/runs/final-covertype/bundles/run-001", tree_arrays);
tabsmith_macros::model!(network, "../../../../fixtures/bundles/binary-mlp");
tabsmith_macros::model!(binary_code, "../../../../fixtures/bundles/binary-gbdt");
tabsmith_macros::model!(multiclass_code, "../../../../fixtures/bundles/multiclass-gbdt");
tabsmith_macros::model!(regression_code, "../../../../fixtures/bundles/regression-gbdt");

macro_rules! time_predict {
    ($m:ident) => {{
        let inputs = $m::EXAMPLES;
        let mut hash = 0u64;
        for input in &inputs {
            hash = hash.rotate_left(7) ^ format!("{:?}", $m::predict(input).unwrap()).len() as u64;
        }
        let t = median(|| {
            for _ in 0..20_000 {
                for input in &inputs {
                    black_box($m::predict(black_box(input)).unwrap());
                }
            }
        });
        println!(
            "{:<11} predict {:>8.1} ns per input",
            stringify!($m).trim_end_matches("_code"),
            t * 1e9 / (20_000 * inputs.len()) as f64
        );
        let _ = hash;
    }};
}

fn rows<const N: usize>(n: usize) -> Vec<[f32; N]> {
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    (0..n)
        .map(|_| {
            core::array::from_fn(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                ((state >> 40) as f32 / (1u64 << 24) as f32) * 4.0 - 2.0
            })
        })
        .collect()
}

fn median(mut f: impl FnMut()) -> f64 {
    let mut t = Vec::new();
    for _ in 0..21 {
        let s = Instant::now();
        f();
        t.push(s.elapsed().as_secs_f64());
    }
    t.sort_by(f64::total_cmp);
    t[10]
}

macro_rules! time_outputs {
    ($m:ident, $n:expr) => {{
        let xs = rows::<{ $m::N_FEATURES }>($n);
        let mut hash = 0u64;
        for x in &xs {
            for v in $m::outputs(x) {
                hash = hash.rotate_left(5) ^ u64::from(v.to_bits());
            }
        }
        let t = median(|| {
            for x in &xs {
                black_box($m::outputs(black_box(x)));
            }
        });
        println!("{:<11} outputs {:>8.1} ns per row  (hash {hash:016x})", stringify!($m), t * 1e9 / xs.len() as f64);
    }};
}

fn main() {
    time_outputs!(regression, 20_000);
    time_outputs!(multiclass, 20_000);
    time_outputs!(covertype, 2_000);
    time_outputs!(network, 20_000);
    let input = network::EXAMPLES[0];
    let t = median(|| {
        for _ in 0..100_000 {
            black_box(network::predict(black_box(&input)).unwrap());
        }
    });
    println!("network     predict {:>8.1} ns per input", t * 1e9 / 100_000.0);
    time_predict!(binary_code);
    time_predict!(multiclass_code);
    time_predict!(regression_code);
}
