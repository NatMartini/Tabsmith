# Errors

[中文](#中文)

Every failure, in the CLI and the SDK, is one `TabsmithError`:

| Field | Meaning |
|---|---|
| `code` | Stable code (below); never reused for another meaning |
| `phase` | Stage: config, data, split, fit, plan, train, run (locating run records), resume, evaluate, compare, bundle, load, predict, audit, device, report |
| `message` | What happened |
| `run_id`, `field` | Where (run, column, config key, file) |
| `expected`, `actual` | The contract and what was found |
| `retained_artifacts` | What is still valid (checkpoint, previous bundle, evaluation) |
| `next_action` | The concrete next step, usually a command |
| `diagnostic_id` | Stable short id of this error content, for logs and reports |

Human output goes to stderr; with `--json` the CLI prints `{"ok": false, "error": {...}}`
on stdout. Example:

```text
error[E_DEVICE_UNSUPPORTED]: device `cuda` is not available: no cuda backend is compiled into this build and it is not certified for tabular-binary@0.1
  phase:    device
  field:    execution.candidate_device
  expected: cpu or auto-certified
  actual:   requested cuda; detected: Intel(R) Xeon(R) Processor @ 2.10GHz (cpu)
  next:     use candidate_device = "cpu" (or "auto-certified"); Tabsmith never falls back silently
  diagnostic: d-…
```

| Exit | Codes |
|---|---|
| 2 | `E_CONFIG_NOT_FOUND`, `E_CONFIG_UNKNOWN_FIELD`, `E_CONFIG_INVALID`, `E_RULE_INVALID` |
| 3 | `E_DATA_SOURCE_NOT_FOUND`, `E_DATA_SOURCE_CHANGED`, `E_DATA_READ_FAILED`, `E_DATA_UNSUPPORTED`, `E_SCHEMA_MISSING_COLUMN`, `E_SCHEMA_EXTRA_COLUMN`, `E_SCHEMA_TYPE_MISMATCH`, `E_INVALID_VALUE`, `E_UNKNOWN_CATEGORY`, `E_CARDINALITY_BUDGET`, `E_TARGET_INVALID`, `E_SPLIT_UNSUPPORTED`, `E_SPLIT_DEGENERATE`, `E_LEAKAGE_DETECTED` |
| 4 | `E_BUNDLE_NOT_FOUND`, `E_BUNDLE_MISSING_FILE`, `E_BUNDLE_CORRUPT`, `E_BUNDLE_INCOMPATIBLE`, `E_BUNDLE_EXISTS`, `E_PREDICTOR_UPDATE_FAILED`, `E_IMPORT_INVALID`, `E_IMPORT_UNSUPPORTED` |
| 5 | `E_DEVICE_UNSUPPORTED`, `E_RESOURCE_BUDGET_EXCEEDED` |
| 6 | `E_RUN_NOT_FOUND`, `E_RUN_STATE_INVALID`, `E_RESUME_INCOMPATIBLE`, `E_CHECKPOINT_MISSING`, `E_CHECKPOINT_CORRUPT`, `E_TRAINING_DIVERGED`, `E_INTERRUPTED` |
| 7 | `E_NOT_COMPARABLE`, `E_TEST_SET_REUSE`, `E_EVALUATION_MISSING` |
| 130 | `E_CANCELLED` (Ctrl-C; a checkpoint is written at the next step boundary) |
| 1 | `E_IO`, `E_INTERNAL` |

`tabsmith audit` also ends with 8 when a rule fails (an input of its region gets a
prediction that breaks it) and 9 when none fails but one is undecided. Those are answers,
not errors: `--json` reports them under `result.rules`. A rule that cannot be read, or
that does not fit the bundle (a field it lacks, a category it never saw when it refuses
unseen ones, a claim for another task), is `E_RULE_INVALID`.

`tabsmith import` reports a file that is not an XGBoost JSON or LightGBM text model, or one
whose trees do not hold together, as `E_IMPORT_INVALID`, and a model it cannot reproduce
(a categorical split, an objective with another link, a random forest, ...) as
`E_IMPORT_UNSUPPORTED`, with the reason. A record that lacks a value its field refuses to go
without, in an imported bundle whose splits send a missing value both ways, is
`E_INVALID_VALUE` ([docs/import.md](import.md)).

Target checks depend on the task (`E_TARGET_INVALID`). A binary target needs exactly
two labels. A multiclass target needs 3 to 100; with two, the error points to the binary
recipe. A regression target needs finite numbers with at least 10 distinct values; with
fewer, it points to the classification recipes. Missing targets are refused in every
task, never dropped silently.

Device checks (`E_DEVICE_UNSUPPORTED`) refuse three things:
- a backend that is not compiled in (GPU backends need the Cargo features `vulkan`/`cuda`);
- a GPU that is not detected;
- a precision the device would not honour. CUDA on compute capability 8.0+ computes f32
  matrix products in TF32, so `execution.precision = "fp32"` is refused there, with the
  advice to set `"tf32"` or use Vulkan.

Evaluation guards: `E_LEAKAGE_DETECTED` is raised when evaluated rows were used
to train or select the model (or a run it was warm-started from), and also when
that cannot be checked because the rows the model saw cannot be matched with the
current data (source run not in the project, input columns missing). The second
case can be accepted with `--allow-unverified-rows`; the evaluation then records
`unseen_rows_verified: null`. `E_TEST_SET_REUSE` is raised when another model of
the same data version already used the test rows (`--allow-test-reuse` records
the reuse). Models of another data version may share test rows with a note
(`test_shared_with`), which is what an old-vs-new comparison needs.

Run states: `planned`, `running`, `completed`, `failed`, `cancelled` (Ctrl-C),
`interrupted` (the process died; detected the next time the run is listed or
resumed). `tabsmith resume <run>` continues `cancelled`/`interrupted`/`failed`
runs exactly from the last checkpoint (R2) as a new run linked to its parent;
`--warm-start` starts a new training from a completed run's weights and is
labelled as such, never as a resume.

---

## 中文

CLI 和 SDK 中的每个失败都是同一种 `TabsmithError`，字段含义见上表：错误代码（稳定、不复用）、阶段、消息、位置（Run、列、配置键、文件）、期望值与实际值、仍然有效的产物（检查点、旧模型包、评估记录）、下一步操作（通常是一条命令）、诊断 id。

人类可读输出写到 stderr；使用 `--json` 时，CLI 在 stdout 输出 `{"ok": false, "error": {...}}`。退出码按错误类别划分（见上表）：2 配置与规则，3 数据/schema，4 模型包与导入，5 设备与资源，6 Run/续训，7 评估纪律，130 用户取消，1 I/O 与内部错误。`tabsmith audit` 在有规则不成立时以 8 结束，没有规则不成立但有规则未定时以 9 结束：这些是答案，不是错误。无法读懂或与模型包不符的规则（没有的字段、拒绝未见类别时训练中没见过的类别、属于其他任务的结论）报 `E_RULE_INVALID`。`tabsmith import` 遇到既不是 XGBoost JSON 也不是 LightGBM 文本模型的文件，或树结构不成立的文件，报 `E_IMPORT_INVALID`；遇到无法复现的模型（类别切分、其他链接函数的目标函数、随机森林等）报 `E_IMPORT_UNSUPPORTED` 并说明原因。导入的模型包中，某字段的切分把缺失值两边都送，该字段缺值的记录报 `E_INVALID_VALUE`（见 [docs/import.md](import.md)）。

目标列检查与任务有关（`E_TARGET_INVALID`）：二分类目标必须恰好有两个标签；多分类需要 3 到 100 个类别（只有两个时，错误会建议改用二分类 recipe）；回归目标必须是有限数值且至少有 10 个不同取值（更少时建议改用分类 recipe）。任何任务中缺失的目标值都会被拒绝，而不是被悄悄丢弃。

设备检查（`E_DEVICE_UNSUPPORTED`）拒绝三种情况：未编译进来的后端（GPU 后端需要 Cargo 特性 `vulkan`/`cuda`）、未探测到的 GPU，以及设备无法保证的精度——计算能力 8.0 及以上的 CUDA 以 TF32 计算 f32 矩阵乘法，因此在那里 `execution.precision = "fp32"` 会被拒绝，并建议设置 `"tf32"` 或改用 Vulkan。

评估保护：当被评估的行曾用于训练或选择该模型（或它 warm start 所依据的 Run）时，报 `E_LEAKAGE_DETECTED`；当无法检查这一点时也会报这个错误，即模型见过的行无法与当前数据匹配（来源 Run 不在本项目中、输入列缺失）。第二种情况可以用 `--allow-unverified-rows` 接受，评估记录中 `unseen_rows_verified` 为 null。当同一数据版本的另一个模型已经使用过这些测试行时，报 `E_TEST_SET_REUSE`（`--allow-test-reuse` 会记录这次复用）。不同数据版本的模型可以共享测试行，并附带说明（`test_shared_with`），这正是新旧模型对比所需要的。

Run 状态：`planned`、`running`、`completed`、`failed`、`cancelled`（Ctrl-C）、`interrupted`（进程意外结束，在下一次列出或恢复时检测到）。`tabsmith resume <run>` 从最近的检查点精确继续 `cancelled`/`interrupted`/`failed` 的 Run（R2），生成一个与父 Run 关联的新 Run；`--warm-start` 用已完成 Run 的权重开始新训练，并明确标注为 warm start，绝不冒充续训。
