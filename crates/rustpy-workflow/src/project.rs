//! Project directory layout, configuration loading and record storage.
//!
//! ```text
//! <project>/
//!   rustpy.toml                 the only user-edited file
//!   data/                       user data (suggested location)
//!   bundles/<run-id>/           immutable inference bundles
//!   .rustpy/
//!     snapshots/<id>/           snapshot.json (+ data copy)
//!     splits/<id>.json          split manifests
//!     transforms/<id>/          fitted transform + input/output schemas
//!     runs/<run-id>/            run.json, events.jsonl, checkpoints/, model/
//!     groups/<grp-id>.json      runs created by one `rustpy run`
//!     evaluations/<id>.json     evaluation records; ledger.jsonl = test-set use log
//!     reports/                  generated HTML/JSON reports
//! ```

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use rustpy_contracts::config::{ConfigLayer, ProjectFile, ResolvedConfig, ValueSource, parse_override, resolve};
use rustpy_contracts::{ErrorCode, Phase, Result, RustpyError};
use serde::Serialize;
use serde::de::DeserializeOwned;

pub const CONFIG_FILE: &str = "rustpy.toml";
pub const STATE_DIR: &str = ".rustpy";

/// How to resolve the configuration for one command.
#[derive(Debug, Clone, Default)]
pub struct ConfigRequest {
    pub profile: Option<String>,
    /// `key.path=value` overrides (CLI `--set`).
    pub overrides: Vec<String>,
    /// Pre-built override tables (SDK).
    pub override_tables: Vec<toml::Table>,
}

#[derive(Debug, Clone)]
pub struct Project {
    pub root: PathBuf,
}

/// [`Path::canonicalize`] without the `\\?\` prefix Windows adds to every
/// canonical path (`\\?\C:\data` becomes `C:\data`), so paths shown to users
/// and passed to other programs look like the paths they typed.
pub fn canonical(path: &Path) -> std::io::Result<PathBuf> {
    let p = path.canonicalize()?;
    #[cfg(windows)]
    {
        let text = p.to_string_lossy();
        if let Some(rest) = text.strip_prefix(r"\\?\") {
            let b = rest.as_bytes();
            // Only plain drive paths short enough for every Windows API.
            if b.len() >= 3 && b.len() < 260 && b[0].is_ascii_alphabetic() && b[1] == b':' && b[2] == b'\\' {
                return Ok(PathBuf::from(rest));
            }
        }
    }
    Ok(p)
}

impl Project {
    /// Opens the project whose `rustpy.toml` is in `dir` or the nearest parent.
    pub fn discover(dir: impl AsRef<Path>) -> Result<Self> {
        let start = dir.as_ref();
        let start = if start.is_absolute() {
            start.to_path_buf()
        } else {
            std::env::current_dir()
                .map_err(|e| RustpyError::io(Phase::Config, ".", e))?
                .join(start)
        };
        let mut cur: Option<&Path> = Some(&start);
        while let Some(d) = cur {
            if d.join(CONFIG_FILE).is_file() {
                return Self::open(d);
            }
            cur = d.parent();
        }
        Err(RustpyError::new(
            ErrorCode::ConfigNotFound,
            Phase::Config,
            format!("no {CONFIG_FILE} found in `{}` or any parent directory", start.display()),
        )
        .next("create a project with `rustpy init <name>` or pass --project <dir>"))
    }

    /// Opens a project at exactly `root`.
    pub fn open(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref();
        // Absolute, canonical root: relative paths in records and the config are
        // always resolved against it, never against the current directory.
        let root = canonical(root).unwrap_or_else(|_| root.to_path_buf());
        if !root.join(CONFIG_FILE).is_file() {
            return Err(RustpyError::new(
                ErrorCode::ConfigNotFound,
                Phase::Config,
                format!("`{}` has no {CONFIG_FILE}", root.display()),
            )
            .next("create a project with `rustpy init <name>`"));
        }
        Ok(Self { root })
    }

    pub fn config_path(&self) -> PathBuf {
        self.root.join(CONFIG_FILE)
    }

    pub fn state_dir(&self) -> PathBuf {
        self.root.join(STATE_DIR)
    }

    pub fn dir(&self, sub: &str) -> PathBuf {
        self.state_dir().join(sub)
    }

    pub fn bundles_dir(&self) -> PathBuf {
        self.root.join("bundles")
    }

    pub fn run_dir(&self, run_id: &str) -> PathBuf {
        self.dir("runs").join(run_id)
    }

    /// Path relative to the project root (for records), with `/` separators.
    pub fn rel(&self, path: &Path) -> String {
        path.strip_prefix(&self.root).unwrap_or(path).to_string_lossy().replace('\\', "/")
    }

    pub fn read_config_file(&self) -> Result<ProjectFile> {
        let path = self.config_path();
        let text = fs::read_to_string(&path).map_err(|e| RustpyError::io(Phase::Config, &path, e))?;
        ProjectFile::parse(&text, CONFIG_FILE)
    }

    /// Resolves defaults < rustpy.toml < profile < overrides.
    pub fn resolve_config(&self, req: &ConfigRequest) -> Result<ResolvedConfig> {
        let file = self.read_config_file()?;
        let mut layers = vec![file.project.clone()];
        if let Some(p) = &req.profile {
            layers.push(file.profile_layer(p)?);
        }
        for t in &req.override_tables {
            layers.push(ConfigLayer {
                source: ValueSource::Override,
                table: t.clone(),
            });
        }
        for o in &req.overrides {
            layers.push(ConfigLayer {
                source: ValueSource::Override,
                table: parse_override(o)?,
            });
        }
        resolve(&layers)
    }

    /// Allocates the next sequential id (`run-001`, `grp-001`, ...) atomically.
    pub fn allocate_id(&self, sub: &str, prefix: &str) -> Result<String> {
        let dir = self.dir(sub);
        fs::create_dir_all(&dir).map_err(|e| RustpyError::io(Phase::Internal, &dir, e))?;
        let mut max = 0u32;
        for entry in fs::read_dir(&dir).map_err(|e| RustpyError::io(Phase::Internal, &dir, e))? {
            let name = entry.map_err(|e| RustpyError::io(Phase::Internal, &dir, e))?.file_name();
            let name = name.to_string_lossy();
            let stem = name.trim_end_matches(".json").trim_end_matches(".lock");
            if let Some(n) = stem.strip_prefix(&format!("{prefix}-")).and_then(|n| n.parse::<u32>().ok()) {
                max = max.max(n);
            }
        }
        for n in (max + 1)..(max + 1000) {
            let id = format!("{prefix}-{n:03}");
            // Reserve with an exclusive marker file so concurrent commands never share an id.
            // Markers are never deleted, so an id can only be claimed once; an existing
            // record or directory is skipped as well.
            let marker = dir.join(format!("{id}.lock"));
            match fs::OpenOptions::new().write(true).create_new(true).open(&marker) {
                Ok(_) if dir.join(&id).exists() || dir.join(format!("{id}.json")).exists() => continue,
                Ok(_) => return Ok(id),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => return Err(RustpyError::io(Phase::Internal, &marker, e)),
            }
        }
        Err(RustpyError::internal(Phase::Internal, "could not allocate an id"))
    }
}

/// Writes `bytes` to `path` atomically (temp file in the same dir + fsync + rename).
pub fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    let dir = path.parent().unwrap_or(Path::new("."));
    fs::create_dir_all(dir).map_err(|e| RustpyError::io(Phase::Internal, dir, e))?;
    let tmp = dir.join(format!(
        ".{}.{}.tmp",
        path.file_name().and_then(|n| n.to_str()).unwrap_or("file"),
        std::process::id()
    ));
    {
        let mut f = fs::File::create(&tmp).map_err(|e| RustpyError::io(Phase::Internal, &tmp, e))?;
        f.write_all(bytes).map_err(|e| RustpyError::io(Phase::Internal, &tmp, e))?;
        f.sync_all().map_err(|e| RustpyError::io(Phase::Internal, &tmp, e))?;
    }
    fs::rename(&tmp, path).map_err(|e| RustpyError::io(Phase::Internal, path, e))
}

pub fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    let mut bytes = serde_json::to_vec_pretty(value)
        .map_err(|e| RustpyError::internal(Phase::Internal, format!("serialize {}: {e}", path.display())))?;
    bytes.push(b'\n');
    write_atomic(path, &bytes)
}

pub fn read_json<T: DeserializeOwned>(path: &Path, phase: Phase) -> Result<T> {
    let bytes = fs::read(path).map_err(|e| RustpyError::io(phase, path, e))?;
    serde_json::from_slice(&bytes).map_err(|e| {
        RustpyError::new(
            ErrorCode::Internal,
            phase,
            format!("record `{}` is unreadable: {e}", path.display()),
        )
        .field(path.display().to_string())
        .next("the project state was edited or damaged; restore it or re-run the stage")
    })
}

/// Appends one JSON line to a log file.
pub fn append_jsonl<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir).map_err(|e| RustpyError::io(Phase::Internal, dir, e))?;
    }
    let mut line = serde_json::to_vec(value).map_err(|e| RustpyError::internal(Phase::Internal, e.to_string()))?;
    line.push(b'\n');
    let mut f = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|e| RustpyError::io(Phase::Internal, path, e))?;
    f.write_all(&line).map_err(|e| RustpyError::io(Phase::Internal, path, e))?;
    f.sync_data().map_err(|e| RustpyError::io(Phase::Internal, path, e))
}

/// Levenshtein-based "did you mean" for names.
pub fn did_you_mean<'a>(name: &str, candidates: impl IntoIterator<Item = &'a str>) -> Option<String> {
    let mut best: Option<(usize, &str)> = None;
    for c in candidates {
        let d = levenshtein(&name.to_lowercase(), &c.to_lowercase());
        if d <= 2.max(name.len() / 4) && best.is_none_or(|(bd, _)| d < bd) {
            best = Some((d, c));
        }
    }
    best.map(|(_, c)| c.to_string())
}

fn levenshtein(a: &str, b: &str) -> usize {
    let b: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    for (i, ca) in a.chars().enumerate() {
        let mut cur = vec![i + 1];
        for (j, cb) in b.iter().enumerate() {
            let cost = usize::from(ca != *cb);
            cur.push((prev[j] + cost).min(prev[j + 1] + 1).min(cur[j] + 1));
        }
        prev = cur;
    }
    prev[b.len()]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn suggestions() {
        assert_eq!(did_you_mean("defekt", ["defect", "line"]), Some("defect".into()));
        assert_eq!(did_you_mean("zzz", ["defect", "line"]), None);
    }

    #[test]
    fn ids_are_sequential_and_exclusive() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join(CONFIG_FILE), "").unwrap();
        let p = Project::open(dir.path()).unwrap();
        assert_eq!(p.allocate_id("runs", "run").unwrap(), "run-001");
        assert_eq!(p.allocate_id("runs", "run").unwrap(), "run-002");
        assert_eq!(p.allocate_id("groups", "grp").unwrap(), "grp-001");
    }
}
