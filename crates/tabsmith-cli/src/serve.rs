//! `tabsmith serve`: a bundle behind a small HTTP/JSON API.
//!
//!   GET  /health    {"status":"ok","bundle_id":...}
//!   GET  /schema    input and output schema of the active bundle
//!   GET  /stats     the inputs predicted since the bundle was loaded, against the
//!                   training statistics the bundle carries (drift monitoring)
//!   POST /predict   one record (object), several (array) or {"records": [...]}
//!   POST /reload    {"bundle": "<dir>"} — only with --allow-reload, and only for a
//!                   bundle in the same folder as the served one; the new bundle
//!                   must load and self-check, otherwise the old one stays
//!
//! Errors are `{"ok": false, "error": {...}}` with the usual Tabsmith error fields.
//! There is no authentication and no request timeout: keep the default
//! `--host 127.0.0.1`, or put a reverse proxy in front that provides both.

use std::collections::BTreeMap;
use std::io::Read as _;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use tabsmith_contracts::transform::FeatureTransform;
use tabsmith_contracts::{ErrorCode, Phase, Result, TabsmithError};
use tabsmith_runtime::{LoadOptions, Outcome, Predictor, PredictorHandle, Record, Value};
use tiny_http::{Header, Method, Request, Response, Server};

/// What the server has seen since its bundle was loaded, field by field, against the
/// training statistics in the bundle's transform. A numeric field is tracked as its
/// standardised value `z = (x − mean) / std`, which was 0 ± 1 on the training rows; a
/// categorical field as the share of values never seen in training.
#[derive(Default)]
struct Monitor {
    requests: u64,
    records: u64,
    fields: BTreeMap<String, FieldStats>,
    /// Binary: the sum of the probabilities and the positives; multiclass: a count
    /// per class; regression: the sum of the values.
    outputs: Outputs,
}

#[derive(Default)]
enum Outputs {
    #[default]
    None,
    Binary {
        sum_probability: f64,
        positives: u64,
    },
    Multiclass(BTreeMap<String, u64>),
    Regression {
        sum: f64,
    },
}

enum FieldStats {
    Numeric { count: u64, missing: u64, sum_z: f64, sum_z2: f64 },
    Categorical { count: u64, missing: u64, unseen: u64 },
}

impl Monitor {
    /// Accounts one accepted request: every record was predicted.
    fn record(&mut self, p: &Predictor, records: &[Record], predictions: &[tabsmith_runtime::Prediction]) {
        self.requests += 1;
        self.records += records.len() as u64;
        for f in &p.bundle().transform.features {
            let stats = self.fields.entry(f.name().to_string()).or_insert_with(|| match f {
                FeatureTransform::Numeric { .. } => FieldStats::Numeric {
                    count: 0,
                    missing: 0,
                    sum_z: 0.0,
                    sum_z2: 0.0,
                },
                FeatureTransform::Categorical { .. } => FieldStats::Categorical {
                    count: 0,
                    missing: 0,
                    unseen: 0,
                },
            });
            for r in records {
                match (f, &mut *stats) {
                    (
                        FeatureTransform::Numeric { mean, std, .. },
                        FieldStats::Numeric {
                            count,
                            missing,
                            sum_z,
                            sum_z2,
                        },
                    ) => {
                        *count += 1;
                        let x = match r.get(f.name()) {
                            Some(Value::Float(v)) if v.is_finite() => Some(*v),
                            Some(Value::Int(v)) => Some(*v as f64),
                            _ => None,
                        };
                        match x {
                            Some(x) => {
                                let z = (x - mean) / std;
                                *sum_z += z;
                                *sum_z2 += z * z;
                            }
                            None => *missing += 1,
                        }
                    }
                    (FeatureTransform::Categorical { categories, .. }, FieldStats::Categorical { count, missing, unseen }) => {
                        *count += 1;
                        let text = match r.get(f.name()) {
                            Some(Value::Text(s)) => Some(s.clone()),
                            Some(Value::Bool(b)) => Some(b.to_string()),
                            Some(Value::Int(i)) => Some(i.to_string()),
                            Some(Value::Float(v)) if v.is_finite() => Some(v.to_string()),
                            _ => None,
                        };
                        match text {
                            Some(t) if categories.contains(&t) => {}
                            Some(_) => *unseen += 1,
                            None => *missing += 1,
                        }
                    }
                    _ => {}
                }
            }
        }
        for pred in predictions {
            match &pred.outcome {
                Outcome::Binary { probability, is_positive } => {
                    if let Outputs::None = self.outputs {
                        self.outputs = Outputs::Binary {
                            sum_probability: 0.0,
                            positives: 0,
                        };
                    }
                    if let Outputs::Binary {
                        sum_probability,
                        positives,
                    } = &mut self.outputs
                    {
                        *sum_probability += probability;
                        *positives += u64::from(*is_positive);
                    }
                }
                Outcome::Multiclass { .. } => {
                    if let Outputs::None = self.outputs {
                        self.outputs = Outputs::Multiclass(BTreeMap::new());
                    }
                    if let Outputs::Multiclass(counts) = &mut self.outputs {
                        *counts.entry(pred.label.clone()).or_default() += 1;
                    }
                }
                Outcome::Regression { value } => {
                    if let Outputs::None = self.outputs {
                        self.outputs = Outputs::Regression { sum: 0.0 };
                    }
                    if let Outputs::Regression { sum } = &mut self.outputs {
                        *sum += value;
                    }
                }
            }
        }
    }

    /// The report of `GET /stats`. A field is listed under `drift` when its inputs
    /// moved clearly away from the training rows: the mean of `z` further than 0.5
    /// from 0, its standard deviation outside 0.5 to 2, or more than 5 % of the
    /// categorical values unseen in training. These are signals to look at the data,
    /// not a verdict on the model.
    fn report(&self, bundle_id: &str) -> serde_json::Value {
        let mut fields = serde_json::Map::new();
        let mut drift = Vec::new();
        for (name, s) in &self.fields {
            let v = match s {
                FieldStats::Numeric {
                    count,
                    missing,
                    sum_z,
                    sum_z2,
                } => {
                    let n = (count - missing) as f64;
                    let (mean_z, std_z) = if n > 0.0 {
                        let mean = sum_z / n;
                        (Some(mean), Some((sum_z2 / n - mean * mean).max(0.0).sqrt()))
                    } else {
                        (None, None)
                    };
                    if mean_z.is_some_and(|m| m.abs() > 0.5) || std_z.is_some_and(|s| !(0.5..=2.0).contains(&s) && n >= 2.0) {
                        drift.push(name.clone());
                    }
                    serde_json::json!({
                        "kind": "numeric",
                        "count": count,
                        "missing": missing,
                        "mean_z": mean_z,
                        "std_z": std_z,
                    })
                }
                FieldStats::Categorical { count, missing, unseen } => {
                    let rate = if *count > 0 { *unseen as f64 / *count as f64 } else { 0.0 };
                    if rate > 0.05 {
                        drift.push(name.clone());
                    }
                    serde_json::json!({
                        "kind": "categorical",
                        "count": count,
                        "missing": missing,
                        "unseen": unseen,
                        "unseen_rate": rate,
                    })
                }
            };
            fields.insert(name.clone(), v);
        }
        let n = self.records.max(1) as f64;
        let outputs = match &self.outputs {
            Outputs::None => serde_json::Value::Null,
            Outputs::Binary {
                sum_probability,
                positives,
            } => serde_json::json!({
                "mean_probability": sum_probability / n,
                "positive_rate": *positives as f64 / n,
            }),
            Outputs::Multiclass(counts) => serde_json::json!({ "labels": counts }),
            Outputs::Regression { sum } => serde_json::json!({ "mean_value": sum / n }),
        };
        serde_json::json!({
            "bundle_id": bundle_id,
            "requests": self.requests,
            "records": self.records,
            "fields": fields,
            "outputs": outputs,
            "drift": drift,
            "rule": "numeric: |mean_z| > 0.5 or std_z outside 0.5..2; categorical: more than 5 % unseen",
        })
    }
}

/// Largest request body accepted.
const MAX_BODY: u64 = 16 * 1024 * 1024;

pub struct ServeOptions {
    pub bundle: PathBuf,
    pub host: String,
    pub port: u16,
    pub threads: usize,
    pub allow_reload: bool,
}

pub fn serve(opts: ServeOptions, stop: Arc<AtomicBool>) -> Result<()> {
    let handle = Arc::new(PredictorHandle::load(&opts.bundle, LoadOptions::cpu().with_self_check(true))?);
    // /reload may only load bundles next to the served one: a client must not make the
    // server read, or probe for, other directories.
    let served = opts
        .bundle
        .canonicalize()
        .map_err(|e| TabsmithError::io(Phase::Load, &opts.bundle, e))?;
    let reload_root = Arc::new(served.parent().map_or_else(|| served.clone(), Path::to_path_buf));
    let addr = format!("{}:{}", opts.host, opts.port);
    let server = Arc::new(Server::http(&addr).map_err(|e| {
        TabsmithError::new(ErrorCode::Io, Phase::Predict, format!("cannot listen on {addr}: {e}"))
            .field("--port")
            .next("choose another --port, or stop the process that uses it")
    })?);
    let bound = server.server_addr().to_ip().map(|a| a.to_string()).unwrap_or(addr);
    let current = handle.current();
    let monitor = Arc::new(Mutex::new(Monitor::default()));
    // The first stdout line is machine-readable (tests and scripts wait for it).
    println!("listening on http://{bound}");
    eprintln!(
        "serving bundle {} ({}) — POST /predict, GET /schema, GET /health, GET /stats{}",
        current.bundle_id(),
        opts.bundle.display(),
        if opts.allow_reload { ", POST /reload" } else { "" }
    );
    {
        let server = server.clone();
        let stop = stop.clone();
        std::thread::spawn(move || {
            while !stop.load(Ordering::SeqCst) {
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
            server.unblock();
        });
    }
    let workers: Vec<_> = (0..opts.threads.max(1))
        .map(|_| {
            let server = server.clone();
            let handle = handle.clone();
            let monitor = monitor.clone();
            let stop = stop.clone();
            let reload_root = opts.allow_reload.then(|| reload_root.clone());
            std::thread::spawn(move || {
                while !stop.load(Ordering::SeqCst) {
                    match server.recv() {
                        Ok(req) => handle_request(req, &handle, &monitor, reload_root.as_deref().map(PathBuf::as_path)),
                        Err(_) => break,
                    }
                }
                // Wake the other workers too.
                server.unblock();
            })
        })
        .collect();
    for w in workers {
        let _ = w.join();
    }
    eprintln!("stopped");
    Ok(())
}

/// `reload_root` is the folder `/reload` may load bundles from, `None` without --allow-reload.
fn handle_request(mut req: Request, handle: &PredictorHandle, monitor: &Mutex<Monitor>, reload_root: Option<&Path>) {
    let path = req.url().split('?').next().unwrap_or("").to_string();
    let method = req.method().clone();
    let (status, body) = match (method, path.as_str()) {
        (Method::Get, "/health") => {
            let p = handle.current();
            (200, serde_json::json!({"status": "ok", "bundle_id": p.bundle_id()}))
        }
        (Method::Get, "/stats") => {
            let p = handle.current();
            let report = monitor.lock().unwrap_or_else(|e| e.into_inner()).report(p.bundle_id());
            (200, report)
        }
        (Method::Get, "/schema") => {
            let p = handle.current();
            (
                200,
                serde_json::json!({
                    "bundle_id": p.bundle_id(),
                    "input": p.input_schema(),
                    "output": p.output_schema(),
                }),
            )
        }
        (Method::Post, "/predict") => match read_json(&mut req).and_then(|v| predict(handle, monitor, &v)) {
            Ok(v) => (200, v),
            Err(e) => error_body(e),
        },
        (Method::Post, "/reload") if reload_root.is_some() => {
            let root = reload_root.expect("checked by the guard");
            match read_json(&mut req).and_then(|v| reload(handle, root, &v)) {
                Ok(v) => {
                    // A new bundle has its own training statistics: start over.
                    if v["active_bundle_id"] != v["previous_bundle_id"] {
                        *monitor.lock().unwrap_or_else(|e| e.into_inner()) = Monitor::default();
                    }
                    (200, v)
                }
                Err(e) => error_body(e),
            }
        }
        (Method::Post, "/reload") => error_body(
            TabsmithError::new(ErrorCode::ConfigInvalid, Phase::Load, "reloading is disabled on this server")
                .next("restart with `tabsmith serve --allow-reload` to swap bundles at runtime"),
        ),
        (_, p) => (
            404,
            serde_json::json!({"ok": false, "error": {"message": format!("no route {p}; use POST /predict, GET /schema, GET /health, GET /stats")}}),
        ),
    };
    let json = serde_json::to_vec(&body).unwrap_or_default();
    let header = Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).expect("static header");
    let _ = req.respond(Response::from_data(json).with_status_code(status).with_header(header));
}

fn read_json(req: &mut Request) -> Result<serde_json::Value> {
    let mut body = Vec::new();
    req.as_reader().take(MAX_BODY + 1).read_to_end(&mut body).map_err(|e| {
        TabsmithError::new(
            ErrorCode::DataReadFailed,
            Phase::Predict,
            format!("cannot read the request body: {e}"),
        )
    })?;
    if body.len() as u64 > MAX_BODY {
        return Err(TabsmithError::new(
            ErrorCode::DataReadFailed,
            Phase::Predict,
            format!("request body larger than {} MB", MAX_BODY / (1024 * 1024)),
        )
        .next("send fewer records per request"));
    }
    serde_json::from_slice(&body).map_err(|e| {
        TabsmithError::new(
            ErrorCode::DataReadFailed,
            Phase::Predict,
            format!("the body is not valid JSON: {e}"),
        )
        .expected("a record object, an array of records, or {\"records\": [...]}")
    })
}

fn predict(handle: &PredictorHandle, monitor: &Mutex<Monitor>, v: &serde_json::Value) -> Result<serde_json::Value> {
    let items: Vec<&serde_json::Value> = match v {
        serde_json::Value::Array(a) => a.iter().collect(),
        serde_json::Value::Object(o) if o.contains_key("records") => match &o["records"] {
            serde_json::Value::Array(a) => a.iter().collect(),
            _ => {
                return Err(
                    TabsmithError::new(ErrorCode::SchemaTypeMismatch, Phase::Predict, "`records` must be an array").field("records"),
                );
            }
        },
        other => vec![other],
    };
    let records = items.into_iter().map(Record::from_json).collect::<Result<Vec<_>>>()?;
    // One predictor for the whole request, even if a reload happens meanwhile.
    let p = handle.current();
    let preds = p.predict(&records)?;
    monitor.lock().unwrap_or_else(|e| e.into_inner()).record(&p, &records, &preds);
    Ok(serde_json::json!({"bundle_id": p.bundle_id(), "predictions": preds}))
}

fn reload(handle: &PredictorHandle, root: &Path, v: &serde_json::Value) -> Result<serde_json::Value> {
    let dir = v.get("bundle").and_then(|b| b.as_str()).ok_or_else(|| {
        TabsmithError::new(ErrorCode::ConfigInvalid, Phase::Load, "missing \"bundle\"").expected("{\"bundle\": \"<bundle dir>\"}")
    })?;
    // A path that does not exist and one outside the folder get the same answer, so the
    // response tells a client nothing about the server's file system.
    let path = Path::new(dir)
        .canonicalize()
        .ok()
        .filter(|p| p.starts_with(root) && p != root)
        .ok_or_else(|| {
            TabsmithError::new(
                ErrorCode::ConfigInvalid,
                Phase::Load,
                format!("`{dir}` is not a bundle directory in this server's bundle folder"),
            )
            .field("bundle")
            .expected("a bundle directory next to the one being served")
            .next("pass the path of a bundle in the same folder, for example {\"bundle\": \"bundles/run-004\"}")
        })?;
    let report = handle.try_update(&path, LoadOptions::cpu().with_self_check(true))?;
    Ok(serde_json::to_value(report).unwrap_or_default())
}

fn error_body(e: TabsmithError) -> (u16, serde_json::Value) {
    let status = match e.code {
        ErrorCode::PredictorUpdateFailed => 409,
        ErrorCode::Io | ErrorCode::Internal => 500,
        _ => 400,
    };
    (status, serde_json::json!({"ok": false, "error": e}))
}
