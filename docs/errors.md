# Errors

[中文](#中文)

Every failure, in the CLI and the SDK, is one `RustpyError`:

| Field | Meaning |
|---|---|
| `code` | Stable code (below); never reused for another meaning |
| `phase` | Stage: config, data, split, fit, train, resume, evaluate, compare, bundle, load, predict, device, report |
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
  next:     use candidate_device = "cpu" (or "auto-certified"); RustPy never falls back silently
  diagnostic: d-…
```

| Exit | Codes |
|---|---|
| 2 | `E_CONFIG_NOT_FOUND`, `E_CONFIG_UNKNOWN_FIELD`, `E_CONFIG_INVALID` |
| 3 | `E_DATA_SOURCE_NOT_FOUND`, `E_DATA_SOURCE_CHANGED`, `E_DATA_READ_FAILED`, `E_DATA_UNSUPPORTED`, `E_SCHEMA_MISSING_COLUMN`, `E_SCHEMA_EXTRA_COLUMN`, `E_SCHEMA_TYPE_MISMATCH`, `E_INVALID_VALUE`, `E_UNKNOWN_CATEGORY`, `E_CARDINALITY_BUDGET`, `E_TARGET_INVALID`, `E_SPLIT_UNSUPPORTED`, `E_SPLIT_DEGENERATE`, `E_LEAKAGE_DETECTED` |
| 4 | `E_BUNDLE_NOT_FOUND`, `E_BUNDLE_MISSING_FILE`, `E_BUNDLE_CORRUPT`, `E_BUNDLE_INCOMPATIBLE`, `E_BUNDLE_EXISTS`, `E_PREDICTOR_UPDATE_FAILED` |
| 5 | `E_DEVICE_UNSUPPORTED`, `E_RESOURCE_BUDGET_EXCEEDED` |
| 6 | `E_RUN_NOT_FOUND`, `E_RUN_STATE_INVALID`, `E_RESUME_INCOMPATIBLE`, `E_CHECKPOINT_MISSING`, `E_CHECKPOINT_CORRUPT`, `E_TRAINING_DIVERGED`, `E_INTERRUPTED` |
| 7 | `E_NOT_COMPARABLE`, `E_TEST_SET_REUSE`, `E_EVALUATION_MISSING` |
| 130 | `E_CANCELLED` (Ctrl-C; a checkpoint is written at the next step boundary) |
| 1 | `E_IO`, `E_INTERNAL` |

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
resumed). `rustpy resume <run>` continues `cancelled`/`interrupted`/`failed`
runs exactly from the last checkpoint (R2) as a new run linked to its parent;
`--warm-start` starts a new training from a completed run's weights and is
labelled as such, never as a resume.

---

## 中文

CLI 和 SDK 中的每个失败都是同一种 `RustpyError`，字段含义见上表：错误代码（稳定、不复用）、阶段、消息、位置（Run、列、配置键、文件）、期望值与实际值、仍然有效的产物（检查点、旧模型包、评估记录）、下一步操作（通常是一条命令）、诊断 id。

人类可读输出写到 stderr；使用 `--json` 时，CLI 在 stdout 输出 `{"ok": false, "error": {...}}`。退出码按错误类别划分（见上表）：2 配置，3 数据/schema，4 模型包，5 设备与资源，6 Run/续训，7 评估纪律，130 用户取消，1 I/O 与内部错误。

评估保护：当被评估的行曾用于训练或选择该模型（或它 warm start 所依据的 Run）时，报 `E_LEAKAGE_DETECTED`；当无法检查这一点时也会报这个错误，即模型见过的行无法与当前数据匹配（来源 Run 不在本项目中、输入列缺失）。第二种情况可以用 `--allow-unverified-rows` 接受，评估记录中 `unseen_rows_verified` 为 null。当同一数据版本的另一个模型已经使用过这些测试行时，报 `E_TEST_SET_REUSE`（`--allow-test-reuse` 会记录这次复用）。不同数据版本的模型可以共享测试行，并附带说明（`test_shared_with`），这正是新旧模型对比所需要的。

Run 状态：`planned`、`running`、`completed`、`failed`、`cancelled`（Ctrl-C）、`interrupted`（进程意外结束，在下一次列出或恢复时检测到）。`rustpy resume <run>` 从最近的检查点精确继续 `cancelled`/`interrupted`/`failed` 的 Run（R2），生成一个与父 Run 关联的新 Run；`--warm-start` 用已完成 Run 的权重开始新训练，并明确标注为 warm start，绝不冒充续训。
