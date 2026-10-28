# Contract walkthrough: a second task (local LLM generation) — P2-08

[中文](#中文)

Design review only; no engine is implemented. Question asked by the plan:
*can a very different task reuse the configuration, Run and Bundle contracts
without copying the product core?* The task used for the walkthrough is
"generate text locally with one specified model" (plan §12, "LLM生成").

## What carries over unchanged

| Contract | Why it already fits |
|---|---|
| Config layering (`ConfigLayer`, `resolve`, provenance, `--set`, profiles) | Generic over the table content; only the typed `ProjectConfig` is recipe-specific |
| `TabsmithError` (codes, phase, expected/actual, retained, next action, exit codes) | Task-neutral; new codes are additive (e.g. `E_TOKENIZER_MISMATCH`, `E_CONTEXT_BUDGET`) |
| `DevicePlan` / `DeviceReceipt`, "no silent fallback" | Exactly what LLM inference needs (VRAM budget, precision, quantization are extra fields) |
| Content-addressed artifacts, `FileEntry`, `compute_bundle_id`, all-or-nothing load, `self_check` with examples/expected | Identical mechanism; expected outputs become fixed-seed greedy generations or logits with a tolerance |
| Run lineage (R0/R1/R2/warm start), receipts, test-set ledger | Applies to fine-tuning; for inference-only tasks Runs are evaluation runs |

## What is recipe-specific today (must be generalised before a 2nd recipe)

1. `ProjectConfig` is a single struct with tabular sections (`data.target`, `preprocess.*`, `train.linear`). → Split into a common header (`schema_version`, `recipe`, `execution`) plus a recipe-owned section validated by the recipe.
2. `manifest::REQUIRED_FILES` is a constant list for tabular bundles. → Move the required-file list into the recipe; keep `manifest.json` common.
3. `FittedTransform` / `InputSchema` / `OutputSchema` model tabular rows. → For LLM generation the equivalent contract is: tokenizer file + revision hash, special tokens, chat template, truncation policy (input); generation config (max tokens, stop sequences, sampling) (output).
4. `ModelConfig` (`tabsmith.dense.v1`) and `DenseModel` are a format, not an abstraction. → Introduce a runtime trait per model format; an LLM bundle would declare e.g. `gguf` or `safetensors-llama` and reference an engine (one of llama.cpp / mistral.rs / Candle, chosen once per plan §2).
5. `BinaryMetrics` inside `EvaluationRecord`. → Make `metrics` a tagged enum per task (binary, generation quality/latency/throughput).
6. `self_contained = true` is assumed. → Multi-GB base weights need external references with hashes (`self_contained = false`, plan §7.1 already requires saying so).

## New contracts the LLM task needs

- Context and KV-cache budget in the DevicePlan; refuse instead of silently truncating.
- Streaming and cancellation semantics in the runtime API (`predict` is batch-only today).
- Exact model / tokenizer / quantization identity in the manifest.

## Conclusion

The cross-cutting machinery (errors, layering, provenance, hashing, device plan,
lineage, self-check) generalises. Five tabular-specific types listed above would
be copied if a second recipe were added now, which the plan's stop rule forbids
("第二任务若需要复制Run/Bundle/配置核心，暂停新增模型，先修架构"). Recommended
order before any second recipe: (1) recipe-owned config section, (2)
recipe-owned bundle file list, (3) tagged metrics, (4) runtime model-format
trait. None of these changes the bundle files of tabular-binary 0.1.

**Update (multiclass, regression and trees).** Two further *tabular* recipes and a
second model format were added later. They exercised items 3–5 and left items 1, 2
and 6 alone:
- **Item 5, tagged metrics: done.** `Metrics` is an enum internally tagged by
  `task` (binary, multiclass, regression), and `EvaluationRecord` and the bundle
  evidence carry it.
- **Item 4, model formats: done for two formats.** The runtime's `Model` is an enum
  over `tabsmith.dense.v1` and `tabsmith.trees.v1`, and each format checks its own
  consistency. A third format adds a variant; an engine-backed format (an LLM) would
  turn the enum into the trait this walkthrough proposed.
- **Item 3, output contract: generalised within tabular.** `OutputSchema` carries the
  task, the classes and the target scale, and the runtime's `Outcome` is an enum per
  task. The input side (rows of typed fields) is unchanged.
- **Items 1, 2 and 6** are untouched. All three tabular recipes share the config
  sections and the bundle file list, so a non-tabular recipe still needs the
  recipe-owned config section, the recipe-owned file list and external weights first.

---

## 中文

本文只做设计评审，不实现任何引擎。计划提出的问题是：*一个差异很大的任务，能否复用配置、Run 和 Bundle 契约，而不复制产品核心？* 走查用的任务是“用一个指定模型在本地生成文本”（计划第 12 节“LLM生成”）。

**可以原样沿用的**：配置分层（来源记录、`--set`、profile）、`TabsmithError`（代码、阶段、期望/实际、保留产物、下一步、退出码）、`DevicePlan`/`DeviceReceipt` 与“不静默回退”、按内容寻址的产物与全有或全无的加载和自检、Run 血缘（R0/R1/R2/warm start）、回执和测试集台账。

**目前专属于表格任务、在加入第二个 recipe 前必须泛化的**：
1. `ProjectConfig` 是带表格专用段的单一结构体 → 拆为公共头部加 recipe 自有、由 recipe 校验的配置段。
2. `manifest::REQUIRED_FILES` 是表格包的固定列表 → 由 recipe 提供必需文件列表。
3. `FittedTransform`/`InputSchema`/`OutputSchema` 面向表格行 → LLM 对应的是分词器文件与版本哈希、特殊 token、对话模板、截断策略（输入）以及生成配置（输出）。
4. `ModelConfig`（`tabsmith.dense.v1`）和 `DenseModel` 是具体格式而非抽象 → 为每种模型格式引入运行时 trait，并按计划第 2 节只选定一个推理引擎。
5. `EvaluationRecord` 内嵌 `BinaryMetrics` → 改为按任务区分的标签枚举。
6. 默认 `self_contained = true` → 数 GB 的基座权重需要带哈希的外部引用。

**LLM 任务需要的新契约**：在 DevicePlan 中声明上下文与 KV 缓存预算（超出时拒绝，而不是静默截断）；运行时 API 的流式输出与取消语义；在 manifest 中精确记录模型、分词器和量化方式。

**后续更新（多分类、回归与树模型）。** 之后又加入了两个*表格类* recipe 和第二种模型格式，它们用到了第 3–5 项，没有涉及第 1、2、6 项：
第 5 项（带标签的指标）已完成——`Metrics` 是按 `task` 标记的枚举；第 4 项（模型格式）对两种格式已完成——运行时的 `Model` 是
`tabsmith.dense.v1` 与 `tabsmith.trees.v1` 上的枚举，引入依赖外部引擎的格式（如 LLM）时再把它变成本文建议的 trait；第 3 项
在表格范围内完成了泛化——`OutputSchema` 带有任务、类别和目标尺度，运行时的 `Outcome` 按任务区分。第 1、2、6 项保持不变：
三个表格 recipe 共用配置段和模型包文件列表，因此非表格 recipe 仍需先完成这三项。

**结论**：横切机制可以泛化。上面列出的五个表格专用类型，如果现在就加入第二个 recipe 会被复制，这违反计划的停止规则。建议在任何第二个 recipe 之前依次完成：recipe 自有配置段、recipe 自有包文件列表、标签化指标、运行时模型格式 trait。这些改动都不会改变 tabular-binary 0.1 的包文件。
