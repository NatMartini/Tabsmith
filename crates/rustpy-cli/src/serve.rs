//! `rustpy serve`: a bundle behind a small HTTP/JSON API.
//!
//!   GET  /health    {"status":"ok","bundle_id":...}
//!   GET  /schema    input and output schema of the active bundle
//!   POST /predict   one record (object), several (array) or {"records": [...]}
//!   POST /reload    {"bundle": "<dir>"} — only with --allow-reload; the new
//!                   bundle must load and self-check, otherwise the old one stays
//!
//! Errors are `{"ok": false, "error": {...}}` with the usual RustPy error fields.

use std::io::Read as _;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use rustpy_contracts::{ErrorCode, Phase, Result, RustpyError};
use rustpy_runtime::{LoadOptions, PredictorHandle, Record};
use tiny_http::{Header, Method, Request, Response, Server};

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
    let addr = format!("{}:{}", opts.host, opts.port);
    let server = Arc::new(Server::http(&addr).map_err(|e| {
        RustpyError::new(ErrorCode::Io, Phase::Predict, format!("cannot listen on {addr}: {e}"))
            .field("--port")
            .next("choose another --port, or stop the process that uses it")
    })?);
    let bound = server.server_addr().to_ip().map(|a| a.to_string()).unwrap_or(addr);
    let current = handle.current();
    // The first stdout line is machine-readable (tests and scripts wait for it).
    println!("listening on http://{bound}");
    eprintln!(
        "serving bundle {} ({}) — POST /predict, GET /schema, GET /health{}",
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
            let stop = stop.clone();
            let allow_reload = opts.allow_reload;
            std::thread::spawn(move || {
                while !stop.load(Ordering::SeqCst) {
                    match server.recv() {
                        Ok(req) => handle_request(req, &handle, allow_reload),
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

fn handle_request(mut req: Request, handle: &PredictorHandle, allow_reload: bool) {
    let path = req.url().split('?').next().unwrap_or("").to_string();
    let method = req.method().clone();
    let (status, body) = match (method, path.as_str()) {
        (Method::Get, "/health") => {
            let p = handle.current();
            (200, serde_json::json!({"status": "ok", "bundle_id": p.bundle_id()}))
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
        (Method::Post, "/predict") => match read_json(&mut req).and_then(|v| predict(handle, &v)) {
            Ok(v) => (200, v),
            Err(e) => error_body(e),
        },
        (Method::Post, "/reload") if allow_reload => match read_json(&mut req).and_then(|v| reload(handle, &v)) {
            Ok(v) => (200, v),
            Err(e) => error_body(e),
        },
        (Method::Post, "/reload") => error_body(
            RustpyError::new(ErrorCode::ConfigInvalid, Phase::Load, "reloading is disabled on this server")
                .next("restart with `rustpy serve --allow-reload` to swap bundles at runtime"),
        ),
        (_, p) => (
            404,
            serde_json::json!({"ok": false, "error": {"message": format!("no route {p}; use POST /predict, GET /schema, GET /health")}}),
        ),
    };
    let json = serde_json::to_vec(&body).unwrap_or_default();
    let header = Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).expect("static header");
    let _ = req.respond(Response::from_data(json).with_status_code(status).with_header(header));
}

fn read_json(req: &mut Request) -> Result<serde_json::Value> {
    let mut body = Vec::new();
    req.as_reader()
        .take(MAX_BODY + 1)
        .read_to_end(&mut body)
        .map_err(|e| RustpyError::new(ErrorCode::DataReadFailed, Phase::Predict, format!("cannot read the request body: {e}")))?;
    if body.len() as u64 > MAX_BODY {
        return Err(RustpyError::new(
            ErrorCode::DataReadFailed,
            Phase::Predict,
            format!("request body larger than {} MB", MAX_BODY / (1024 * 1024)),
        )
        .next("send fewer records per request"));
    }
    serde_json::from_slice(&body).map_err(|e| {
        RustpyError::new(ErrorCode::DataReadFailed, Phase::Predict, format!("the body is not valid JSON: {e}"))
            .expected("a record object, an array of records, or {\"records\": [...]}")
    })
}

fn predict(handle: &PredictorHandle, v: &serde_json::Value) -> Result<serde_json::Value> {
    let items: Vec<&serde_json::Value> = match v {
        serde_json::Value::Array(a) => a.iter().collect(),
        serde_json::Value::Object(o) if o.contains_key("records") => match &o["records"] {
            serde_json::Value::Array(a) => a.iter().collect(),
            _ => {
                return Err(RustpyError::new(ErrorCode::SchemaTypeMismatch, Phase::Predict, "`records` must be an array")
                    .field("records"));
            }
        },
        other => vec![other],
    };
    let records = items.into_iter().map(Record::from_json).collect::<Result<Vec<_>>>()?;
    // One predictor for the whole request, even if a reload happens meanwhile.
    let p = handle.current();
    let preds = p.predict(&records)?;
    Ok(serde_json::json!({"bundle_id": p.bundle_id(), "predictions": preds}))
}

fn reload(handle: &PredictorHandle, v: &serde_json::Value) -> Result<serde_json::Value> {
    let dir = v.get("bundle").and_then(|b| b.as_str()).ok_or_else(|| {
        RustpyError::new(ErrorCode::ConfigInvalid, Phase::Load, "missing \"bundle\"")
            .expected("{\"bundle\": \"<bundle dir>\"}")
    })?;
    let report = handle.try_update(Path::new(dir), LoadOptions::cpu().with_self_check(true))?;
    Ok(serde_json::to_value(report).unwrap_or_default())
}

fn error_body(e: RustpyError) -> (u16, serde_json::Value) {
    let status = match e.code {
        ErrorCode::PredictorUpdateFailed => 409,
        ErrorCode::Io | ErrorCode::Internal => 500,
        _ => 400,
    };
    (status, serde_json::json!({"ok": false, "error": e}))
}
