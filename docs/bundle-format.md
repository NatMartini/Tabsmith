# Bundle format (version 1)

[中文](#中文)

A bundle is an immutable directory created by `rustpy bundle build`. It contains
everything needed to predict and nothing else: no training data, no optimizer
state, no credentials, no code.

```text
bundles/run-002/
  manifest.json          format/recipe/runtime versions, provenance, sha256 of every file
  input.schema.json      named, typed input fields; missing/extra/unknown/NaN policies
  output.schema.json     target, negative/positive labels, probability meaning, threshold
  transform.json         frozen standard transform (rustpy.std-tabular.v2)
  model.config.json      engine-neutral architecture (rustpy.dense.v1)
  model.safetensors      F32 weights: layers.{i}.weight [out, in], layers.{i}.bias [out]
  evaluation.json        held-out test evaluation of the source run + validation metrics
  examples/input.json    synthetic records built from transform statistics (no training rows)
  examples/expected.json outputs produced at build time, tolerance 1e-6
```

Identity: `bundle_id = "b-" + sha256(lines + claims)[..16]`, where `lines` is
`"path\0sha256\n"` for every listed file sorted by path, and `claims` is the
canonical-JSON hash of every manifest field except `bundle_id`, `created_at` and
`files` (versions, provenance, evaluation status and metrics, threshold, device
list, ...). Changing a file, or editing a claim in `manifest.json` to hide a
change, makes the recorded id stop matching and the bundle is refused.

Loading (`Predictor::load`) is all-or-nothing:

1. `manifest.json` parses; `bundle_format == 1`; `min_runtime_version` ≤ runtime;
   transform and model formats are implemented; precision is FP32; the requested
   device is listed (only `cpu` in this release).
2. Every required file is declared, every declared path is a safe relative path,
   exists, and matches its SHA-256 and size. Files that are not listed, and
   symbolic links, are refused. Each file is read once, so what was hashed is
   what gets parsed. The recorded `bundle_id` must equal the recomputed one, and
   the manifest's evaluation claims must agree with `evaluation.json`.
3. Schemas, transform and model are mutually consistent (dimensions, field names,
   distinct labels); weights are finite and have the declared shapes.

`rustpy bundle check` / `Predictor::self_check` additionally run
`examples/input.json` and compare with `examples/expected.json`.

Status in `evaluation.json` / manifest `evaluation_status`:
`evaluated` (built after `rustpy evaluate --run <id> --split test`, or from a
run whose weights are identical to an evaluated run's on the same data version)
or `unevaluated-candidate` (built with `--allow-unevaluated`). `rustpy predict`
and `rustpy serve` without `--bundle` use the newest `evaluated` bundle of the
project, and a candidate only when no bundle was evaluated.

Updates: `PredictorHandle::try_update(dir, opts)` loads and self-checks the new
bundle in a separate instance; only then is it swapped in. Updates are
serialized, so each one is compared with the predictor it replaces. A breaking
contract change is refused unless `LoadOptions::allow_contract_change(true)`:
new required input fields, changed field kinds or types, stricter missing /
unknown-category policies, or a different target, label mapping or threshold.
On any failure the old predictor stays active and the error says so. The
returned `UpdateReport` lists `schema_changes` and `output_changes`, so even
allowed changes are visible. Rollback is `rollback_to(old_dir, opts)` — the old
bundle is unchanged on disk.

Version rules: `bundle_format` must equal 1; versions are strict
`major.minor.patch`; a bundle whose `min_runtime_version` is newer than the
runtime is refused with `E_BUNDLE_INCOMPATIBLE`. Unknown JSON fields in
any bundle file are refused (`deny_unknown_fields`) instead of being silently
ignored.

---

## 中文

模型包是 `rustpy bundle build` 生成的不可变目录。它包含预测所需的一切，且仅此而已：不含训练数据、优化器状态、凭据或代码。各文件含义见上方目录树。

身份：`bundle_id = "b-" + sha256(lines + claims)[..16]`。其中 `lines` 是所有已列出文件按路径排序后的 `"path\0sha256\n"`；`claims` 是 manifest 中除 `bundle_id`、`created_at`、`files` 之外所有字段（版本、来源、评估状态与指标、阈值、设备列表等）的规范 JSON 哈希。修改任何文件，或改动 `manifest.json` 中的声明来掩盖修改，都会导致记录的 id 与重新计算的 id 不一致，模型包会被拒绝加载。

加载（`Predictor::load`）要么完全成功，要么什么都不加载：

1. `manifest.json` 可解析；`bundle_format == 1`；`min_runtime_version` 不高于当前运行时；变换和模型格式已实现；精度为 FP32；请求的设备在列表中（本版本只有 `cpu`）。
2. 每个必需文件都已声明；每个声明的路径都是安全的相对路径、存在，且 SHA-256 与大小一致。未列出的文件和符号链接会被拒绝。每个文件只读取一次，因此校验哈希的内容就是被解析的内容。记录的 `bundle_id` 必须等于重新计算的值，manifest 中的评估声明必须与 `evaluation.json` 一致。
3. schema、变换和模型彼此一致（维度、字段名、标签不同）；权重为有限值且形状符合声明。

`rustpy bundle check` / `Predictor::self_check` 还会运行 `examples/input.json`，并与 `examples/expected.json` 比对。

`evaluation.json` 和 manifest 中的 `evaluation_status`：`evaluated`（在 `rustpy evaluate --run <id> --split test` 之后构建，或者来自一个与同一数据版本上已评估 Run 权重完全相同的 Run）或 `unevaluated-candidate`（使用 `--allow-unevaluated` 构建）。不带 `--bundle` 时，`rustpy predict` 和 `rustpy serve` 使用项目中最新的 `evaluated` 模型包；只有在没有任何已评估模型包时才使用候选包。

更新：`PredictorHandle::try_update(dir, opts)` 先在独立实例中加载并自检新包，通过后才切换。更新是串行的，因此每次都与它要替换的预测器比较。破坏性的契约变化默认被拒绝，除非设置 `LoadOptions::allow_contract_change(true)`：新增必需输入字段、字段种类或类型改变、更严格的缺失值/未知类别策略，或目标列、标签映射、阈值改变。任何失败都会让旧预测器继续生效，错误信息会说明这一点。返回的 `UpdateReport` 列出 `schema_changes` 和 `output_changes`，即使被允许的变化也清晰可见。回退使用 `rollback_to(old_dir, opts)`——旧包在磁盘上保持不变。

版本规则：`bundle_format` 必须等于 1；版本号必须是严格的 `major.minor.patch`；如果模型包的 `min_runtime_version` 高于当前运行时，会以 `E_BUNDLE_INCOMPATIBLE` 拒绝。所有模型包文件中的未知 JSON 字段都会被拒绝（`deny_unknown_fields`），而不是被静默忽略。
