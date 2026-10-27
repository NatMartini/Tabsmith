//! Human and JSON output helpers.

use rustpy_contracts::RustpyError;
use serde::Serialize;

pub fn print_error(e: &RustpyError, json: bool) {
    if json {
        println!("{}", serde_json::to_string_pretty(&serde_json::json!({ "ok": false, "error": e })).unwrap_or_default());
    } else {
        eprint!("{}", e.render());
    }
}

pub fn print_json<T: Serialize>(value: &T) {
    print_json_status(value, true);
}

/// JSON envelope; `ok` is false whenever the command exits non-zero.
pub fn print_json_status<T: Serialize>(value: &T, ok: bool) {
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({ "ok": ok, "result": value })).unwrap_or_default()
    );
}

/// Simple left-aligned text table.
pub struct Table {
    header: Vec<String>,
    rows: Vec<Vec<String>>,
}

impl Table {
    pub fn new(header: &[&str]) -> Self {
        Self {
            header: header.iter().map(|s| s.to_string()).collect(),
            rows: Vec::new(),
        }
    }

    pub fn row(&mut self, cells: Vec<String>) {
        self.rows.push(cells);
    }

    pub fn render(&self) -> String {
        let n = self.header.len();
        let mut widths: Vec<usize> = self.header.iter().map(|h| h.chars().count()).collect();
        for r in &self.rows {
            for (i, c) in r.iter().enumerate().take(n) {
                widths[i] = widths[i].max(c.chars().count());
            }
        }
        let line = |cells: &[String]| {
            let mut s = String::from("  ");
            for (i, c) in cells.iter().enumerate().take(n) {
                s.push_str(c);
                if i + 1 < n {
                    s.push_str(&" ".repeat(widths[i] - c.chars().count() + 2));
                }
            }
            s.trim_end().to_string()
        };
        let mut out = line(&self.header);
        out.push('\n');
        out.push_str(&line(&widths.iter().map(|w| "─".repeat(*w)).collect::<Vec<_>>()));
        out.push('\n');
        for r in &self.rows {
            out.push_str(&line(r));
            out.push('\n');
        }
        out
    }
}

pub fn f4(x: f64) -> String {
    format!("{x:.4}")
}

pub fn opt4(x: Option<f64>) -> String {
    x.map(f4).unwrap_or_else(|| "–".into())
}

pub fn duration(ms: f64) -> String {
    if ms < 1000.0 {
        format!("{ms:.0} ms")
    } else if ms < 60_000.0 {
        format!("{:.1} s", ms / 1000.0)
    } else {
        format!("{:.1} min", ms / 60_000.0)
    }
}
