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
  transform.json         frozen standard transform (rustpy.std-tabular.v1)
  model.config.json      engine-neutral architecture (rustpy.dense.v1)
  model.safetensors      F32 weights: layers.{i}.weight [out, in], layers.{i}.bias [out]
  evaluation.json        held-out test evaluation of the source run + validation metrics
  examples/input.json    synthetic records built from transform statistics (no training rows)
  examples/expected.json outputs produced at build time, tolerance 1e-6
```

Identity: `bundle_id = "b-" + sha256("path\0sha256\n" for every file, sorted)[..16]`.
The manifest itself is not part of the id; editing it to hide a changed file is
detected because the id no longer matches.

Loading (`Predictor::load`) is all-or-nothing:

1. `manifest.json` parses; `bundle_format == 1`; `min_runtime_version` ≤ runtime;
   transform and model formats are implemented; precision is FP32; the requested
   device is listed (only `cpu` in this release).
2. Every required file is declared, every declared path is a safe relative path,
   exists, and matches its SHA-256 and size.
3. Schemas, transform and model are mutually consistent (dimensions, field names,
   distinct labels); weights are finite and have the declared shapes.

`rustpy bundle check` / `Predictor::self_check` additionally run
`examples/input.json` and compare with `examples/expected.json`.

Status in `evaluation.json` / manifest `evaluation_status`:
`evaluated` (built after `rustpy evaluate --run <id> --split test`) or
`unevaluated-candidate` (built with `--allow-unevaluated`).

Updates: `PredictorHandle::try_update(dir)` loads and self-checks the new bundle
in a separate instance; only then is it swapped in. If it fails, or if it
requires input fields the current bundle did not (unless explicitly allowed),
the old predictor stays active and the error says so. Rollback is
`rollback_to(old_dir)` — the old bundle is unchanged on disk.

---

## 中文

模型包是 `rustpy bundle build` 生成的不可变目录。它包含预测所需的一切，且仅此而已：不含训练数据、优化器状态、凭据或代码。各文件含义见上方目录树。

身份：`bundle_id = "b-" + sha256(所有文件按路径排序后的 "path\0sha256\n")[..16]`。manifest 本身不计入 id；如果有人改了某个文件并修改 manifest 去掩盖，id 将不再匹配，因而会被发现。

加载（`Predictor::load`）要么完全成功，要么什么都不加载：

1. `manifest.json` 可解析；`bundle_format == 1`；`min_runtime_version` 不高于当前运行时；变换和模型格式已实现；精度为 FP32；请求的设备在列表中（本版本只有 `cpu`）。
2. 每个必需文件都已声明；每个声明的路径都是安全的相对路径、存在，且 SHA-256 与大小一致。
3. schema、变换和模型彼此一致（维度、字段名、标签不同）；权重为有限值且形状符合声明。

`rustpy bundle check` / `Predictor::self_check` 还会运行 `examples/input.json`，并与 `examples/expected.json` 比对。

`evaluation.json` 和 manifest 中的 `evaluation_status`：`evaluated`（在 `rustpy evaluate --run <id> --split test` 之后构建）或 `unevaluated-candidate`（使用 `--allow-unevaluated` 构建）。

更新：`PredictorHandle::try_update(dir)` 先在独立实例中加载并自检新包，通过后才切换。如果失败，或新包需要当前包不需要的输入字段（除非显式允许），旧预测器继续生效，错误信息会说明这一点。回退使用 `rollback_to(old_dir)`——旧包在磁盘上保持不变。
