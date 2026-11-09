//! `tabsmith audit`: rules over regions of inputs, each proved, refuted with an input, or
//! left undecided; and the stability of the labels of a file's records.

use std::path::{Path, PathBuf};

use tabsmith_contracts::{ErrorCode, Phase, Result, TabsmithError};
use tabsmith_runtime::audit::{Audit, Budget, Rule, Verdict};
use tabsmith_runtime::model::Model;
use tabsmith_runtime::{LoadOptions, Outcome, Prediction, Predictor, Record, Value, ValueRef};

use crate::output::print_json_status;

/// Exit code when a rule fails.
const FAILS: u8 = 8;
/// Exit code when no rule fails but one is undecided.
const UNDECIDED: u8 = 9;

/// The rules of `--rule` and of the lines of `--rules FILE`, in that order.
fn rules(inline: Vec<String>, file: Option<&Path>) -> Result<Vec<String>> {
    let mut out = inline;
    if let Some(f) = file {
        let text = std::fs::read_to_string(f).map_err(|e| TabsmithError::io(Phase::Audit, f, e))?;
        out.extend(
            text.lines()
                .map(str::trim)
                .filter(|l| !l.is_empty() && !l.starts_with('#'))
                .map(String::from),
        );
    }
    Ok(out)
}

fn boxes_text(n: usize) -> String {
    if n == 1 { "1 box".to_string() } else { format!("{n} boxes") }
}

fn shown(p: &Prediction) -> String {
    match &p.outcome {
        Outcome::Binary { probability, .. } => format!("label {:?}, probability {probability:.6}", p.label),
        Outcome::Multiclass { probability, .. } => format!("label {:?}, probability {probability:.6}", p.label),
        Outcome::Regression { value } => format!("value {value}"),
    }
}

/// The records of a file, as `tabsmith predict` reads them.
fn records(p: &Predictor, input: &Path) -> Result<Vec<Record>> {
    if !input.is_file() {
        return Err(TabsmithError::new(
            ErrorCode::DataSourceNotFound,
            Phase::Audit,
            format!("input file {} does not exist", input.display()),
        )
        .field("--input")
        .next("check the --input path (relative to the current directory)"));
    }
    let ext = input
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_lowercase)
        .unwrap_or_default();
    if matches!(ext.as_str(), "json" | "jsonl" | "ndjson") {
        let text = std::fs::read_to_string(input).map_err(|e| TabsmithError::io(Phase::Audit, input, e))?;
        let bad = |e: serde_json::Error| {
            TabsmithError::new(ErrorCode::DataReadFailed, Phase::Audit, format!("invalid JSON: {e}")).field(input.display().to_string())
        };
        // One record, an array of them, or `{"records": [...]}` like a bundle's examples.
        let values: Vec<serde_json::Value> = if ext == "json" {
            match serde_json::from_str(&text).map_err(bad)? {
                serde_json::Value::Array(a) => a,
                serde_json::Value::Object(mut o) if o.get("records").is_some_and(serde_json::Value::is_array) => {
                    match o.remove("records") {
                        Some(serde_json::Value::Array(a)) => a,
                        _ => unreachable!("checked to be an array"),
                    }
                }
                v => vec![v],
            }
        } else {
            text.lines()
                .filter(|l| !l.trim().is_empty())
                .map(|l| serde_json::from_str(l).map_err(bad))
                .collect::<Result<_>>()?
        };
        return values.iter().map(Record::from_json).collect();
    }
    let table = tabsmith_workflow::data::read_for_prediction(input, p.input_schema(), &p.bundle().transform, &[])
        .map_err(|e| e.in_phase(Phase::Audit))?;
    let names = table.names();
    Ok((0..table.n_rows())
        .map(|i| {
            let mut r = Record::new();
            for name in &names {
                let value = match table.column(name).expect("listed").get(i) {
                    ValueRef::Null => Value::Null,
                    ValueRef::Bool(b) => Value::Bool(b),
                    ValueRef::Int(v) => Value::Int(v),
                    ValueRef::Float(v) => Value::Float(v),
                    ValueRef::Text(s) => Value::Text(s.to_string()),
                };
                r.insert(*name, value);
            }
            r
        })
        .collect())
}

/// The numbers a witness changed from the record it started at.
fn moved(record: &Record, witness: &Record) -> String {
    let number = |v: Option<&Value>| match v {
        Some(Value::Float(x)) => Some(*x),
        Some(Value::Int(i)) => Some(*i as f64),
        _ => None,
    };
    let changes: Vec<String> = witness
        .names()
        .filter_map(|n| match (number(record.get(n)), number(witness.get(n))) {
            (Some(a), Some(b)) if a != b => Some(format!("{n} {a} → {b}")),
            _ => None,
        })
        .collect();
    if changes.is_empty() {
        "no number moved".to_string()
    } else {
        changes.join(", ")
    }
}

fn code_of(audits: &[&Audit]) -> u8 {
    if audits.iter().any(|a| a.result() == "fails") {
        FAILS
    } else if audits.iter().any(|a| a.result() == "undecided") {
        UNDECIDED
    } else {
        0
    }
}

pub fn run(
    bundle: &Path,
    inline: Vec<String>,
    file: Option<&Path>,
    stability: Option<(f64, PathBuf)>,
    boxes: usize,
    seconds: Option<f64>,
    json: bool,
) -> Result<u8> {
    let texts = rules(inline, file)?;
    if texts.is_empty() && stability.is_none() {
        return Err(TabsmithError::new(ErrorCode::RuleInvalid, Phase::Audit, "no rule to check")
            .expected(
                "--rule '<conditions> => <claim>' (repeatable), --rules <file> with one rule per line, or --stability <sd> --input <file>",
            )
            .next("tabsmith audit --rule '=> probability <= 0.99'  (see tabsmith audit --help)"));
    }
    // Every rule is read before any is searched: a typo fails fast.
    let parsed = texts.iter().map(|t| Rule::parse(t)).collect::<Result<Vec<_>>>()?;
    let p = Predictor::load(bundle, LoadOptions::cpu())?;
    let budget = Budget { boxes, seconds };
    if !json {
        let model = match &p.bundle().model {
            Model::Trees(m) => format!("{} trees, {} nodes: searched exactly", m.roots.len(), m.feature.len()),
            Model::Dense(m) => format!("a network of {} layers: searched with interval bounds", m.layers.len()),
        };
        println!("bundle {} at {} ({model})", p.bundle_id(), bundle.display());
    }
    let mut audits = Vec::with_capacity(parsed.len());
    for (i, rule) in parsed.iter().enumerate() {
        let a = p.audit(rule, &budget)?;
        if !json {
            println!("\nrule {}: {}", i + 1, a.rule);
            match &a.verdict {
                Verdict::Holds(proof) => println!(
                    "  holds: a proof of {}, checked again ({} visited, {:.2} s)",
                    boxes_text(proof.boxes()),
                    boxes_text(a.boxes),
                    a.seconds
                ),
                Verdict::Fails { witness, prediction } => {
                    println!("  fails: {} for the input", shown(prediction));
                    println!("    {}", serde_json::to_string(&witness.to_json()).unwrap_or_default());
                    println!("  ({} visited, {:.2} s)", boxes_text(a.boxes), a.seconds);
                }
                Verdict::Undecided(why) => println!("  undecided: {why} ({} visited, {:.2} s)", boxes_text(a.boxes), a.seconds),
            }
        }
        audits.push(a);
    }
    let mut rows = Vec::new();
    if let Some((sd, input)) = &stability {
        let records = records(&p, input)?;
        if !json {
            println!(
                "\nstability of {} records of {}: every number within {sd} sd of the record's, categories and missing values kept",
                records.len(),
                input.display()
            );
        }
        for (i, record) in records.iter().enumerate() {
            let rule = p.stability_rule(record, *sd).map_err(|e| e.field(format!("record {}", i + 1)))?;
            let label = match &rule.claim {
                tabsmith_runtime::audit::Then::Label(l) => l.clone(),
                _ => unreachable!("a stability rule claims a label"),
            };
            let a = p.audit_near(&rule, &budget, record)?;
            if !json {
                match &a.verdict {
                    Verdict::Holds(_) => {}
                    Verdict::Fails { witness, prediction } => println!(
                        "  record {}: {label:?} can change to {}: {}",
                        i + 1,
                        shown(prediction),
                        moved(record, witness)
                    ),
                    Verdict::Undecided(why) => println!("  record {}: {label:?} undecided: {why}", i + 1),
                }
            }
            rows.push((label, a));
        }
    }
    let all: Vec<&Audit> = audits.iter().chain(rows.iter().map(|(_, a)| a)).collect();
    let code = code_of(&all);
    let count = |list: &mut dyn Iterator<Item = &Audit>, r: &str| list.filter(|a| a.result() == r).count();
    if json {
        let mut result = serde_json::json!({
            "bundle_id": p.bundle_id(),
            "bundle": bundle,
            "rules": audits.iter().map(|a| a.to_json()).collect::<Vec<_>>(),
        });
        if let Some((sd, input)) = &stability {
            let records: Vec<serde_json::Value> = rows
                .iter()
                .enumerate()
                .map(|(i, (label, a))| {
                    let mut v = a.to_json();
                    v["record"] = serde_json::Value::from(i + 1);
                    v["label"] = serde_json::Value::from(label.as_str());
                    if let Some(o) = v.as_object_mut() {
                        o.remove("rule");
                    }
                    v
                })
                .collect();
            result["stability"] = serde_json::json!({"sd": sd, "input": input, "records": records});
        }
        print_json_status(&result, code == 0);
    } else {
        if !audits.is_empty() {
            let (h, f, u) = (
                count(&mut audits.iter(), "holds"),
                count(&mut audits.iter(), "fails"),
                count(&mut audits.iter(), "undecided"),
            );
            println!("\nrules: {h} hold, {f} fail, {u} undecided");
        }
        if stability.is_some() {
            let list = || rows.iter().map(|(_, a)| a);
            let (h, f, u) = (
                count(&mut list(), "holds"),
                count(&mut list(), "fails"),
                count(&mut list(), "undecided"),
            );
            println!("records: {h} keep their label, {f} can change it, {u} undecided");
        }
    }
    Ok(code)
}
