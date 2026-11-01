//! Trees as arrays or as generated code, row by row or tree by tree, on real bundles and
//! real rows. One thread. Every line is checked bit for bit against the runtime.
//!     codegen-probe <work folder> <output.json>
use std::hint::black_box;
use std::time::Instant;

use tabsmith_runtime::transform::{Matrix, apply_transform};
use tabsmith_runtime::{LoadOptions, Phase, Predictor, Record};

tabsmith_macros::model!(defects_code, "<repo>/fixtures/bundles/binary-gbdt", tree_code);
tabsmith_macros::model!(defects_arrays, "<repo>/fixtures/bundles/binary-gbdt", tree_arrays);
tabsmith_macros::model!(adult_code, "../runs/final-adult/bundles/run-001", tree_code);
tabsmith_macros::model!(adult_arrays, "../runs/final-adult/bundles/run-001", tree_arrays);
tabsmith_macros::model!(california_code, "../runs/final-california/bundles/run-001", tree_code);
tabsmith_macros::model!(california_arrays, "../runs/final-california/bundles/run-001", tree_arrays);
tabsmith_macros::model!(covertype_code, "../runs/final-covertype/bundles/run-001", tree_code);
tabsmith_macros::model!(covertype_arrays, "../runs/final-covertype/bundles/run-001", tree_arrays);
tabsmith_macros::model!(mid_code, "../runs/mid-covertype/bundles/run-001", tree_code);
tabsmith_macros::model!(mid_arrays, "../runs/mid-covertype/bundles/run-001", tree_arrays);
tabsmith_macros::model!(big_code, "../runs/timeline4-covertype/bundles/run-001", tree_code);
tabsmith_macros::model!(big_arrays, "../runs/timeline4-covertype/bundles/run-001", tree_arrays);

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

fn load(bundle: &str, input: &str) -> (Predictor, Matrix) {
    let p = Predictor::load(bundle, LoadOptions::cpu().with_threads(1)).unwrap();
    let records: Vec<Record> = std::fs::read_to_string(input)
        .unwrap()
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| Record::from_json(&serde_json::from_str(l).unwrap()).unwrap())
        .collect();
    let table = p.records_table(&records, &[]).unwrap();
    let x = apply_transform(&p.bundle().transform, &table, Phase::Predict).unwrap();
    (p, x)
}

macro_rules! measure {
    ($name:literal, $code:ident, $arrays:ident, $bundle:expr, $input:expr, $rounds:literal, $results:ident) => {{
        use $arrays as a;
        use $code as c;
        let (p, x) = load(&$bundle, &$input);
        let n = x.rows;
        assert_eq!(x.cols, c::N_FEATURES);
        let rows: Vec<[f32; c::N_FEATURES]> = x.data.chunks_exact(x.cols).map(|r| r.try_into().unwrap()).collect();
        let want: Vec<u32> = p.outputs_matrix(&x).data.iter().map(|v| v.to_bits()).collect();
        let mut out = vec![[0f32; c::N_OUTPUTS]; n];
        println!("\n{}: {n} rows, {} trees, {} nodes, {} outputs, depth {}", $name, a::TREES.roots.len(), a::TREES.feature.len(), c::N_OUTPUTS, c::MAX_TREE_DEPTH);
        let mut lines = Vec::new();
        macro_rules! line {
            ($label:expr, $f:expr) => {{
                let s = median_s($rounds, || $f);
                let got: Vec<u32> = out.iter().flatten().map(|v| v.to_bits()).collect();
                assert!(got == want, "{} differs", $label);
                println!("{:<30} {:>9.3} us/row {:>11.0} rows/s", $label, s * 1e6 / n as f64, n as f64 / s);
                lines.push(serde_json::json!({"how": $label, "us_per_row": s * 1e6 / n as f64}));
            }};
        }
        let s = median_s($rounds, || {
            black_box(p.outputs_matrix(black_box(&x)));
        });
        println!("{:<30} {:>9.3} us/row {:>11.0} rows/s", "runtime, one thread", s * 1e6 / n as f64, n as f64 / s);
        lines.push(serde_json::json!({"how": "runtime, one thread", "us_per_row": s * 1e6 / n as f64}));
        line!("arrays, row by row", {
            for (r, o) in black_box(&rows).iter().zip(out.iter_mut()) {
                *o = a::outputs(r);
            }
        });
        line!("arrays, tree by tree", {
            const BLOCK: usize = 256;
            let t = &a::TREES;
            let k = c::N_OUTPUTS;
            let mut sums = vec![0f64; BLOCK * k];
            for (chunk, results) in black_box(&rows).chunks(BLOCK).zip(out.chunks_mut(BLOCK)) {
                for s in sums.chunks_exact_mut(k).take(chunk.len()) {
                    s.copy_from_slice(t.base);
                }
                for tree in 0..t.roots.len() {
                    for (row, s) in chunk.iter().zip(sums.chunks_exact_mut(k)) {
                        s[tree % k] += f64::from(t.value[t.leaf(tree, row)]);
                    }
                }
                for (s, r) in sums.chunks_exact(k).zip(results.iter_mut()) {
                    for o in 0..k {
                        r[o] = s[o] as f32;
                    }
                }
            }
        });
        line!("code, row by row", {
            for (r, o) in black_box(&rows).iter().zip(out.iter_mut()) {
                *o = c::outputs(r);
            }
        });
        line!("code, tree by tree", c::outputs_by_tree(black_box(&rows), &mut out));
        line!("code, outputs_batch", {
            black_box(c::outputs_batch(black_box(&rows), &mut out));
        });
        $results.push(serde_json::json!({
            "model": $name, "rows": n, "trees": a::TREES.roots.len(), "nodes": a::TREES.feature.len(),
            "outputs": c::N_OUTPUTS, "max_depth": c::MAX_TREE_DEPTH, "lines": lines,
        }));
    }};
}

fn main() {
    let mut a = std::env::args().skip(1);
    let (work, out_path) = (a.next().unwrap(), a.next().unwrap());
    let run = |name: &str| format!("{work}/scratch/speed/runs/{name}/bundles/run-001");
    let data = |name: &str| format!("{work}/scratch/speed/data/{name}.jsonl");
    let mut results = Vec::new();
    measure!("Fixture binary-gbdt", defects_code, defects_arrays, format!("{work}/fixtures/bundles/binary-gbdt"), data("binary-gbdt"), 21, results);
    measure!("Adult", adult_code, adult_arrays, run("final-adult"), data("adult"), 11, results);
    measure!("Covertype, 40 rounds", covertype_code, covertype_arrays, run("final-covertype"), data("covertype"), 11, results);
    measure!("California Housing", california_code, california_arrays, run("final-california"), data("california"), 11, results);
    measure!("Covertype, 143 rounds", mid_code, mid_arrays, run("mid-covertype"), data("covertype"), 9, results);
    measure!("Covertype, 400 rounds", big_code, big_arrays, run("timeline4-covertype"), data("covertype"), 5, results);
    std::fs::write(out_path, serde_json::to_string_pretty(&results).unwrap()).unwrap();
}
