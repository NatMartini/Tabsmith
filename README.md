# RustPy

**From a CSV/Parquet file to a verified model inside your Rust application — in Rust, without Python.**

[中文说明](README.zh-CN.md)

RustPy is a Rust AI toolkit organised around one complete job, not around a list
of algorithms. Release 0.1 implements the first job end to end:

> tabular binary classification: local CSV/Parquet → trustworthy baseline →
> small neural network → fair comparison → model bundle with its preprocessing →
> loaded by a Rust application → retrained on new data, with rollback.

Status: **internal CPU prototype**. Everything below runs and is tested on CPU
(Linux x86_64). GPU training, user studies and cross-vendor certification are
**not** done yet — see [what is not done](#what-is-not-done-yet).

## Five-minute tour

```text
cargo install --path crates/rustpy-cli        # or: cargo build --release -p rustpy-cli

rustpy init defect-model --sample-data         # project + 10,000 synthetic rows, bound to target `defect`
cd defect-model
rustpy plan                                    # columns, missing values, split, preprocessing, candidates, device
rustpy run                                     # trains `linear` and `small-mlp` on the same split + transform
rustpy compare                                 # validation-only comparison and a recommendation
rustpy evaluate --run run-002 --split test     # locks the final candidate; the test split is used once
rustpy bundle build --run run-002              # immutable bundle: weights + preprocessing + schemas + evidence
rustpy bundle check bundles/run-002            # verifies hashes, versions and example predictions here
rustpy predict --bundle bundles/run-002 --input new.csv --output preds.csv
rustpy report                                  # offline HTML/JSON report in .rustpy/reports/
```

With your own data:

```text
rustpy init churn && cd churn
rustpy data bind ../customers.csv --target churned --positive yes --group customer_id
rustpy plan
```

In your application (depends only on `rustpy-runtime`: no Burn, no Polars, no Python):

```rust
use rustpy_runtime::{LoadOptions, Predictor, Record};

let predictor = Predictor::load("bundles/run-002", LoadOptions::cpu())?;
let reading = Record::new()
    .with("line", "B")
    .with("vibration_rms", 1.42)
    .with("humidity", None::<f64>)     // missing values are fine; they are imputed as in training
    /* … the other input fields … */;
let p = predictor.predict_one(&reading)?;
println!("{} (p = {:.3})", p.label, p.probability);
```

`predict_serialize(&[my_struct])` accepts any `#[derive(Serialize)]` struct whose
field names match the bundle inputs; `PredictorHandle::try_update` swaps in a
new bundle only after it loads and self-checks, otherwise the old one keeps
serving. See [examples/predict-app](examples/predict-app).

## What "integrated" means here

| Convention | What you get | What happens on failure |
|---|---|---|
| One project config | `rustpy.toml`, profiles and `--set` resolve the same way for CLI and SDK; every effective value records its source (`rustpy config`) | Unknown keys and wrong types are errors with the location |
| One fitted preprocessing | Statistics learned on training rows only; the *same code* encodes training, evaluation, offline and in-app prediction | Missing fields, wrong types, unknown categories (if configured) fail before prediction with the field name |
| One traceable Run | Config hash, data snapshot, split, transform, device plan and receipt, history, metrics, artifacts with hashes | `failed`, `cancelled` and `interrupted` are distinct; checkpoints are kept |
| One deliverable Bundle | Weights, transform, input/output schemas, label mapping, evaluation evidence, synthetic examples, file hashes | A damaged or incomplete bundle never loads half-way |
| One execution plan | Requested vs. selected device, precision, copies, memory estimate — before training | No silent fallback: an unavailable device is an error |
| One diagnosis experience | Every error: stable code, phase, field, expected/actual, what was kept, the next command | Exit codes per error family; `--json` for automation |

## Workspace

| Crate | Role | Depends on |
|---|---|---|
| `rustpy-contracts` | Config layering, errors, schemas, Run/Bundle/Evaluation records | serde, toml, sha2 |
| `rustpy-runtime` | Bundle verification, the shared transform, FP32 model execution, `Predictor` | contracts, safetensors |
| `rustpy-workflow` | Snapshots, splits, fitting, training (Burn 0.22, CPU), evaluation, bundles, reports | Polars 0.55, Burn 0.22 |
| `rustpy-cli` | The `rustpy` binary | workflow, runtime |

More: [architecture](docs/architecture.md) · [configuration](docs/configuration.md) ·
[bundle format](docs/bundle-format.md) · [errors](docs/errors.md) ·
[implementation backlog](docs/implementation-backlog.md) ·
[acceptance report](reports/implementation-acceptance.md)

## Guarantees that are tested

- Preprocessing parity: raw rows → training-side probabilities equal app-side `Predictor` probabilities (max |Δp| < 1e-9); scikit-learn reproduces RustPy's transformed matrix (max |Δ| ≈ 2e-7, f32).
- No leakage by construction: split before fit; grouped split never puts one entity in two splits; time-dependent data is refused rather than split randomly.
- Exact resume (R2): a run that crashed in one process and was resumed in another produces **bit-identical** weights and history to an uninterrupted run (same binary, same CPU features).
- Test-set discipline: a test split is used once per split; reuse is refused unless explicitly allowed, and then recorded.
- Bundles are immutable, content-addressed and verified file by file.
- Metrics match scikit-learn to 1e-15; gradients match independent finite differences; the first Adam step matches the update rule.

## What is not done yet

- **GPU**: no CUDA or non-NVIDIA backend is compiled in or certified. `candidate_device = "auto-certified"` selects the CPU and says so; `"cuda"` is an error. Cross-vendor GPU support (plan items P2-04/P2-05) needs real hardware.
- **User validation**: no interviews or user trials have taken place (P0-02, P0-06, P3-03). The usability targets are untested.
- Time-based splits, regression/multiclass, tree models, CNN/sequence/text/LLM recipes are future task packages.

License: MIT OR Apache-2.0. The bundled sample dataset is synthetic (CC0).
