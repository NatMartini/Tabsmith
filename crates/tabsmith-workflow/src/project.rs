//! Project directory layout, configuration loading and record storage.
//!
//! ```text
//! <project>/
//!   tabsmith.toml                 the only user-edited file
//!   data/                       user data (suggested location)
//!   bundles/<run-id>/           immutable inference bundles
//!   .tabsmith/
//!     snapshots/<id>/           snapshot.json (+ data copy)
//!     splits/<id>.json          split manifests
//!     transforms/<id>/          fitted transform + input/output schemas
//!     runs/<run-id>/            run.json, events.jsonl, checkpoints/, model/
//!     groups/<grp-id>.json      runs created by one `tabsmith run`
//!     evaluations/<id>.json     evaluation records; ledger.jsonl = test-set use log
//!     reports/                  generated HTML/JSON reports
//! ```

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use serde::Serialize;
use serde::de::DeserializeOwned;
use tabsmith_contracts::config::{ConfigLayer, ProjectFile, ResolvedConfig, ValueSource, parse_override, resolve};
use tabsmith_contracts::{ErrorCode, Phase, Result, TabsmithError};

pub const CONFIG_FILE: &str = "tabsmith.toml";
pub const STATE_DIR: &str = ".tabsmith";

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
    /// Opens the project whose `tabsmith.toml` is in `dir` or the nearest parent.
    pub fn discover(dir: impl AsRef<Path>) -> Result<Self> {
        let start = dir.as_ref();
        let start = if start.is_absolute() {
            start.to_path_buf()
        } else {
            std::env::current_dir()
                .map_err(|e| TabsmithError::io(Phase::Config, ".", e))?
                .join(start)
        };
        let mut cur: Option<&Path> = Some(&start);
        while let Some(d) = cur {
            if d.join(CONFIG_FILE).is_file() {
                return Self::open(d);
            }
            cur = d.parent();
        }
        Err(TabsmithError::new(
            ErrorCode::ConfigNotFound,
            Phase::Config,
            format!("no {CONFIG_FILE} found in `{}` or any parent directory", start.display()),
        )
        .next("create a project with `tabsmith init <name>` or pass --project <dir>"))
    }

    /// Opens a project at exactly `root`.
    pub fn open(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref();
        // Absolute, canonical root: relative paths in records and the config are
        // always resolved against it, never against the current directory.
        let root = canonical(root).unwrap_or_else(|_| root.to_path_buf());
        if !root.join(CONFIG_FILE).is_file() {
            return Err(TabsmithError::new(
                ErrorCode::ConfigNotFound,
                Phase::Config,
                format!("`{}` has no {CONFIG_FILE}", root.display()),
            )
            .next("create a project with `tabsmith init <name>`"));
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
        let text = fs::read_to_string(&path).map_err(|e| TabsmithError::io(Phase::Config, &path, e))?;
        ProjectFile::parse(&text, CONFIG_FILE)
    }

    /// Resolves defaults < tabsmith.toml < profile < overrides.
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
        fs::create_dir_all(&dir).map_err(|e| TabsmithError::io(Phase::Internal, &dir, e))?;
        let mut max = 0u32;
        for entry in fs::read_dir(&dir).map_err(|e| TabsmithError::io(Phase::Internal, &dir, e))? {
            let name = entry.map_err(|e| TabsmithError::io(Phase::Internal, &dir, e))?.file_name();
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
                Err(e) => return Err(TabsmithError::io(Phase::Internal, &marker, e)),
            }
        }
        Err(TabsmithError::internal(Phase::Internal, "could not allocate an id"))
    }
}

/// Writes `bytes` to `path` atomically (temp file in the same dir + fsync + rename).
///
/// The temp file is created with `create_new`, which never follows a symbolic link: a
/// link planted at the predictable temp name cannot make the bytes land elsewhere. One
/// left by an interrupted run is removed first (a link goes, whatever it points at stays).
pub fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    let dir = path.parent().unwrap_or(Path::new("."));
    fs::create_dir_all(dir).map_err(|e| TabsmithError::io(Phase::Internal, dir, e))?;
    let tmp = dir.join(format!(
        ".{}.{}.tmp",
        path.file_name().and_then(|n| n.to_str()).unwrap_or("file"),
        std::process::id()
    ));
    let io = |e| TabsmithError::io(Phase::Internal, &tmp, e);
    let create = || fs::OpenOptions::new().write(true).create_new(true).open(&tmp);
    let mut f = match create() {
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
            fs::remove_file(&tmp).map_err(io)?;
            create().map_err(io)?
        }
        r => r.map_err(io)?,
    };
    f.write_all(bytes).map_err(io)?;
    f.sync_all().map_err(io)?;
    drop(f);
    fs::rename(&tmp, path).map_err(|e| TabsmithError::io(Phase::Internal, path, e))
}

/// Creates `dir` afresh for the files of a bundle, with the subdirectories `subs`.
/// Whatever an interrupted run left at `dir` is removed first (a link goes, whatever it
/// points at stays). The directories are made with `create_dir`, not `create_dir_all`:
/// it fails on a link planted at the path meanwhile instead of writing through it.
pub fn fresh_dir(dir: &Path, subs: &[&str], phase: Phase) -> Result<()> {
    let io = |e| TabsmithError::io(phase, dir, e);
    match fs::symlink_metadata(dir) {
        Ok(m) if m.is_dir() => fs::remove_dir_all(dir).map_err(io)?,
        Ok(_) => fs::remove_file(dir).or_else(|_| fs::remove_dir(dir)).map_err(io)?,
        Err(_) => {}
    }
    fs::create_dir(dir).map_err(io)?;
    for sub in subs {
        fs::create_dir(dir.join(sub)).map_err(io)?;
    }
    Ok(())
}

pub fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    let mut bytes = serde_json::to_vec_pretty(value)
        .map_err(|e| TabsmithError::internal(Phase::Internal, format!("serialize {}: {e}", path.display())))?;
    bytes.push(b'\n');
    write_atomic(path, &bytes)
}

pub fn read_json<T: DeserializeOwned>(path: &Path, phase: Phase) -> Result<T> {
    let bytes = fs::read(path).map_err(|e| TabsmithError::io(phase, path, e))?;
    serde_json::from_slice(&bytes).map_err(|e| {
        TabsmithError::new(
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
        fs::create_dir_all(dir).map_err(|e| TabsmithError::io(Phase::Internal, dir, e))?;
    }
    let mut line = serde_json::to_vec(value).map_err(|e| TabsmithError::internal(Phase::Internal, e.to_string()))?;
    line.push(b'\n');
    let mut f = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|e| TabsmithError::io(Phase::Internal, path, e))?;
    f.write_all(&line).map_err(|e| TabsmithError::io(Phase::Internal, path, e))?;
    f.sync_data().map_err(|e| TabsmithError::io(Phase::Internal, path, e))
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

    /// A link planted at the predictable temp name of a file or a directory is removed,
    /// never written through: what it points at keeps its content.
    #[cfg(unix)]
    #[test]
    fn a_planted_link_at_the_temp_name_is_not_written_through() {
        let dir = tempfile::tempdir().unwrap();
        let victim = dir.path().join("victim");
        fs::write(&victim, "keep").unwrap();
        let target = dir.path().join("out.json");
        let tmp = dir.path().join(format!(".out.json.{}.tmp", std::process::id()));
        std::os::unix::fs::symlink(&victim, &tmp).unwrap();
        write_atomic(&target, b"new").unwrap();
        assert_eq!(fs::read(&victim).unwrap(), b"keep");
        assert_eq!(fs::read(&target).unwrap(), b"new");
        assert!(!tmp.exists());

        let keep = dir.path().join("keep");
        fs::create_dir(&keep).unwrap();
        fs::write(keep.join("f"), "x").unwrap();
        let fresh = dir.path().join("fresh");
        std::os::unix::fs::symlink(&keep, &fresh).unwrap();
        fresh_dir(&fresh, &["examples"], Phase::Bundle).unwrap();
        assert!(keep.join("f").is_file(), "the link's target keeps its files");
        assert!(fresh.join("examples").is_dir() && !fs::symlink_metadata(&fresh).unwrap().file_type().is_symlink());
        assert!(!keep.join("examples").exists());
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
