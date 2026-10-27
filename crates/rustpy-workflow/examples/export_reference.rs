//! Exports what a run saw and produced so an independent reference implementation
//! (scikit-learn, see tools/reference/sklearn_reference.py) can check it:
//!
//! * raw training/validation rows (feature columns + target, as in the snapshot)
//! * the transformed matrices produced by RustPy's shared transform
//! * the run's kept weights, validation probabilities and metrics
//!
//!     cargo run -p rustpy-workflow --example export_reference -- <project> <run-id> <out-dir>

use std::fmt::Write as _;
use std::path::PathBuf;

use rustpy_contracts::data::SplitName;
use rustpy_runtime::ValueRef;
use rustpy_runtime::model::sigmoid;
use rustpy_workflow::Workspace;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let project = args.next().ok_or("project dir")?;
    let run_id = args.next().ok_or("run id")?;
    let out = PathBuf::from(args.next().ok_or("output dir")?);
    std::fs::create_dir_all(&out)?;
    let ws = Workspace::open(&project)?;
    let run = ws.load_run(&run_id)?;
    let p = ws.prepare_for_run(&run)?;
    let model = ws.run_weights(&run)?;
    let features: Vec<String> = p.fitted.input_schema.fields.iter().map(|f| f.name.clone()).collect();
    for (split, name) in [(SplitName::Train, "train"), (SplitName::Valid, "valid")] {
        let d = ws.split_data(&p, split)?;
        // Raw rows: features + target, missing values written as empty cells.
        let sub = p.table.take(&d.rows);
        let mut raw = String::new();
        let mut header: Vec<&str> = features.iter().map(String::as_str).collect();
        header.push("__target__");
        raw.push_str(&header.join(","));
        raw.push('\n');
        for i in 0..sub.n_rows() {
            let mut cells = Vec::new();
            for f in &features {
                cells.push(match sub.column(f).map(|c| c.get(i)) {
                    Some(ValueRef::Null) | None => String::new(),
                    Some(ValueRef::Float(x)) if x.is_nan() => "NaN".into(),
                    Some(ValueRef::Float(x)) => format!("{x:?}"),
                    Some(ValueRef::Int(x)) => x.to_string(),
                    Some(ValueRef::Bool(b)) => b.to_string(),
                    Some(ValueRef::Text(s)) => format!("\"{}\"", s.replace('"', "\"\"")),
                });
            }
            cells.push(u8::from(d.y[i]).to_string());
            raw.push_str(&cells.join(","));
            raw.push('\n');
        }
        std::fs::write(out.join(format!("raw_{name}.csv")), raw)?;
        // Transformed matrix.
        let mut x = String::new();
        x.push_str(
            &p.fitted
                .transform
                .output_names
                .iter()
                .map(|n| format!("\"{n}\""))
                .collect::<Vec<_>>()
                .join(","),
        );
        x.push('\n');
        for r in 0..d.x.rows {
            let row: Vec<String> = d.x.row(r).iter().map(|v| format!("{v:?}")).collect();
            let _ = writeln!(x, "{}", row.join(","));
        }
        std::fs::write(out.join(format!("x_{name}.csv")), x)?;
        let y: Vec<u8> = d.y.iter().map(|&b| u8::from(b)).collect();
        let probs: Vec<f64> = model.logits(&d.x).into_iter().map(|z| sigmoid(f64::from(z))).collect();
        std::fs::write(
            out.join(format!("rustpy_{name}.json")),
            serde_json::to_vec_pretty(&serde_json::json!({ "y": y, "probability": probs }))?,
        )?;
    }
    let layers: Vec<serde_json::Value> = model
        .layers
        .iter()
        .map(|l| serde_json::json!({"in": l.in_dim, "out": l.out_dim, "weight": l.weight, "bias": l.bias}))
        .collect();
    std::fs::write(
        out.join("rustpy_run.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "run_id": run.run_id,
            "candidate": run.candidate,
            "train_config": run.train_config,
            "threshold": run.threshold,
            "metrics": run.metrics,
            "layers": layers,
            "fitted": p.fitted.transform,
            "input_schema": p.fitted.input_schema,
        }))?,
    )?;
    println!("exported {} to {}", run.run_id, out.display());
    Ok(())
}
