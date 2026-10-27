# Implementation backlog

[中文](#中文)

Status of every task in the v2 task list (`执行路线与任务清单.md`) after the first
implementation pass, dated 2026-10-27. Status values are the four the task list
allows: **Not started**, **In progress**, **Waiting on external conditions**,
**Accepted**. "Accepted" here means *accepted by internal engineering evidence*
(tests, the acceptance script, reference comparisons) on one Linux x86_64 machine.
No task is marked accepted on the basis of user trials, interviews or GPU
hardware, because none of those happened.

Evidence for "Accepted" rows: the automated tests (`cargo test --workspace`,
67 tests) and the acceptance run in
[reports/implementation-acceptance.md](../reports/implementation-acceptance.md).

## P0 — is it worth building?

| ID | Task | Status | What exists / what is missing |
|---|---|---|---|
| P0-01 | User/task boundary and success criteria | In progress | The scope is frozen in code and docs: tabular binary classification, linear + small-MLP, CPU, explicit "not supported" list (README). Missing: user personas and a hypothesis register written from real users. |
| P0-02 | Observe real work, interviews | Waiting on external conditions | No interviews were held. Contacting external people needs separate authorization. |
| P0-03 | Best-alternative task comparison | In progress | Only the machine side was measured: RustPy's training loop vs. a plain Burn loop on the same data and objective (wrapper overhead). Human time with Burn + hand-written glue, or Python training + Rust serving, was not measured. |
| P0-04 | Success/failure journey prototype | Accepted | Replaced by the real implementation: commands, configuration, a bundle and 12 preset failures run end to end. Sample data is synthetic and labelled as such (`rustpy sample-data`). |
| P0-05 | Dependency and device risk probe | Waiting on external conditions | CPU: clean build timed, Burn 0.22.1 + Polars 0.55.2 pinned. CUDA and non-NVIDIA backends were not compiled or run: no such hardware here. |
| P0-06 | Direction decision | Waiting on external conditions | Needs P0-02 (≥3 people with the repeated problem, ≥2 trial partners). Not decided by default. |

## P1 — full CPU workflow

| ID | Task | Status | Evidence |
|---|---|---|---|
| P1-01 | 4-crate skeleton, pinned versions, config parsing | Accepted | `rustpy-contracts` / `-runtime` / `-workflow` / `-cli`; `Cargo.lock` committed; unknown keys rejected with location and suggestion; `rustpy config` shows the layer of every value. Tests: `unknown_field_is_reported_with_location`, `layering_and_sources`, `effective_values_cover_every_leaf`. |
| P1-02 | Input snapshot, schema, split | Accepted | CSV/Parquet; content-hashed snapshots; stratified, id-stable (`rustpy.split.v2-stable-hash`) and group splits; same path with new content creates a new snapshot. Tests: `id_based_splits_are_stable_across_data_versions`, `group_split_has_no_leakage`, `data_update_creates_new_snapshot_and_old_bundle_stays_usable`. |
| P1-03 | Reference values and tolerances before implementation | In progress | scikit-learn goldens exist (`fixtures/reference/goldens.json`, 3 tests) but were generated **after** the implementation, not frozen before it. The tolerances (1e-5 transform, 1e-12 metrics, 1 % objective gap) were chosen when the goldens were made. The linear defaults were tuned once afterwards (gap 0.15 %). This does not meet "frozen before implementation"; a fresh golden set should be frozen before the next change to preprocessing or training. |
| P1-04 | Shared fit/transform | Accepted | One `apply_transform` in `rustpy-runtime` is used by training, evaluation, CLI prediction and the app. Max difference vs. scikit-learn 2.1e-7; training-side vs. separate-process app probabilities < 1e-12. |
| P1-05 | Run states and immutable artifacts | Accepted | planned / running / completed / failed / cancelled / interrupted; `process.lock` liveness; atomic writes; content hashes in `run.json`. |
| P1-06 | CPU linear baseline and small MLP | Accepted | Burn (ndarray) only; gradient and Adam unit tests against finite differences and a hand-written update; linear objective within 0.15 % of the scikit-learn optimum (see P1-03 caveat). |
| P1-07 | Comparison protocol and test-set ledger | Accepted | Ranked by validation; test used once per data snapshot; other models of the same snapshot are refused (`E_TEST_SET_REUSE`); runs on different snapshots or protocols are marked not directly comparable; quality, latency and peak memory reported separately. |
| P1-08 | Minimal bundle format and checks | Accepted | [bundle-format.md](bundle-format.md); tampering, missing files, unlisted files, symlinks and edited claims are rejected (`bundle_tampering_is_rejected`). |
| P1-09 | Load in a new Rust process | Accepted | `examples/predict-app` depends only on `rustpy-runtime` (no Burn, no Polars, no Python); main program is 22 non-blank lines. |
| P1-10 | CLI end-to-end | Accepted | `tests/cli.rs` and the acceptance script drive init → bind → plan → run → compare → evaluate → bundle build/check → predict → report through the binary. |

## P2 — reliability and cross-vendor GPU

| ID | Task | Status | Evidence / gap |
|---|---|---|---|
| P2-01 | DevicePlan and receipt | In progress | CPU path accepted: the plan shows backend, precision, fallback (`error`, never silent), data placement, copy path, thread count, memory estimate vs. budget; the receipt records what ran and whether it matched. Detected GPUs are reported as `not_compiled`. GPU initialization and certification levels cannot be exercised without GPU backends and hardware. |
| P2-02 | R0/R1, caching and change branches | Accepted | R0 = new run; R1 = reuse of hash-compatible stages ("cached stages reused: …"); changed inputs invalidate stages; warm start is a new run with `parent_run_id`, never called R2. |
| P2-03 | CPU R2 exact resume | Accepted | Crash at step 130 and SIGINT mid-epoch, resumed in a new process: weights and history bitwise identical to an uninterrupted run (`resume_check.json`). Changed settings, data, platform or recipe are refused. |
| P2-04 | CUDA, same MLP | Waiting on external conditions | No NVIDIA GPU or CUDA toolchain in this environment. Not implemented. |
| P2-05 | Non-NVIDIA GPU, same MLP | Waiting on external conditions | No AMD/Intel/Apple GPU available. Not implemented. |
| P2-06 | Data update, candidate bundle, rollback | Accepted | New data → new snapshot and runs; old bundle evaluated on new data with seen rows excluded; `PredictorHandle::try_update` keeps the old predictor on failure and refuses contract breaks; `rollback_to` restores the old bundle (`hot_swap` example, step 28). |
| P2-07 | Ten failure classes and cancellation | Accepted | 12 preset failures, each with an error code, the offending field, expected/actual, retained artifacts and a next action; exit codes per family; SIGINT → `cancelled`, exit 130, resumable. OOM is covered by the memory budget check, not by provoking a real OOM. |
| P2-08 | Second-task contract walkthrough | Accepted (design only) | [extension-walkthrough-llm.md](extension-walkthrough-llm.md). Nothing implemented, by design. |

## P3 — independent users and release candidate

| ID | Task | Status | Evidence / gap |
|---|---|---|---|
| P3-01 | Install and runtime-only distribution | In progress | Linux x86_64 clean release build timed (CLI ≈ 5 min with dependencies already downloaded; app ≈ 24 s). No Windows build, no prebuilt binaries, no clean-runner download timing. |
| P3-02 | Local report and diagnostics | Accepted | `rustpy report` writes HTML + JSON: data issues, quality, resources, configuration differences between runs, and non-comparable runs flagged. |
| P3-03 | Independent user trials | Waiting on external conditions | No participants. No pass rate is claimed. |
| P3-04 | Upstream and bundle upgrade drill | In progress | Bundle-side rules exist and are tested (strict versions, `min_runtime_version`, unknown fields rejected). No real dependency upgrade (e.g. a new Burn release) was rehearsed. |
| P3-05 | Correctness, device and soak report | In progress | CPU correctness and wrapper overhead measured (acceptance report). No GPU paths, no 2-hour soak run. |
| P3-06 | Dual gate and scope decision | Waiting on external conditions | Needs P3-03, P3-04, P3-05. |

## P4 and later

Not started, by plan: one adjacent task (regression / multi-class) only after
binary classification is used repeatedly; CNN / sequence / BERT / LLM only with
real demand. The LLM contract walkthrough (P2-08) is the only work done there.

## What would change these statuses

1. Interviews and alternative-tool sessions (P0-02, P0-03, P0-06).
2. An NVIDIA machine and one AMD/Intel/Apple machine (P0-05, P2-01, P2-04, P2-05, P3-05).
3. A Windows runner and a release pipeline with prebuilt binaries (P3-01).
4. Five matching users for timed independent tasks (P3-03).
5. Re-freezing reference goldens before the next numeric change (P1-03).

---

## 中文

# 实施待办

本文记录 v2 任务清单（`执行路线与任务清单.md`）中每个任务在首轮实现后的状态，日期 2026-10-27。状态只使用任务清单允许的四种：**未开始**、**进行中**、**等待外部条件**、**已验收**。这里的“已验收”指*在一台 Linux x86_64 机器上，由内部工程证据验收*（自动测试、验收脚本、参考对照）。没有任何任务依据用户试验、访谈或 GPU 硬件被标为已验收，因为这些都没有进行。

“已验收”的证据：自动测试（`cargo test --workspace`，67 个测试）以及 [reports/implementation-acceptance.md](../reports/implementation-acceptance.md) 中的验收运行。

## P0 —— 是否值得构建

| ID | 任务 | 状态 | 已有 / 缺失 |
|---|---|---|---|
| P0-01 | 用户/任务边界与成功标准 | 进行中 | 范围已在代码和文档中冻结：表格二分类、线性 + 小 MLP、CPU，以及明确的“不支持”列表（README）。缺失：基于真实用户的用户画像和假设登记。 |
| P0-02 | 观察真实工作与访谈 | 等待外部条件 | 未进行任何访谈。联系外部人员需要另行授权。 |
| P0-03 | 最佳替代方案任务对照 | 进行中 | 只测了机器侧：同一数据和目标函数下，RustPy 训练循环对比纯 Burn 循环（封装开销）。未测量用 Burn + 手写胶水、或 Python 训练 + Rust 部署所需的人工时间。 |
| P0-04 | 成功与失败旅程原型 | 已验收 | 已被真实实现取代：命令、配置、模型包和 12 个预设故障均端到端运行。样例数据是合成的，并明确标注（`rustpy sample-data`）。 |
| P0-05 | 最小依赖与设备风险探针 | 等待外部条件 | CPU：已计时清洁构建，锁定 Burn 0.22.1 + Polars 0.55.2。CUDA 和非 NVIDIA 后端未编译、未运行：本环境没有此类硬件。 |
| P0-06 | 方向裁决 | 等待外部条件 | 依赖 P0-02（至少 3 人有重复问题、至少 2 个试用伙伴）。不会默认通过。 |

## P1 —— 完整 CPU 工作流

| ID | 任务 | 状态 | 证据 |
|---|---|---|---|
| P1-01 | 4-crate 骨架、锁版本、配置解析 | 已验收 | `rustpy-contracts` / `-runtime` / `-workflow` / `-cli`；提交了 `Cargo.lock`；未知键会被拒绝并给出位置和建议；`rustpy config` 显示每个值来自哪一层。测试：`unknown_field_is_reported_with_location`、`layering_and_sources`、`effective_values_cover_every_leaf`。 |
| P1-02 | 输入快照、schema 与 split | 已验收 | CSV/Parquet；基于内容哈希的快照；分层、基于 id 的稳定切分（`rustpy.split.v2-stable-hash`）和分组切分；同一路径换了内容会生成新快照。测试：`id_based_splits_are_stable_across_data_versions`、`group_split_has_no_leakage`、`data_update_creates_new_snapshot_and_old_bundle_stays_usable`。 |
| P1-03 | 实现前的参考值与容差 | 进行中 | 已有 scikit-learn goldens（`fixtures/reference/goldens.json`，3 个测试），但它们是在实现**之后**生成的，而非实现前冻结。容差（变换 1e-5、指标 1e-12、目标函数差距 1%）是在生成 goldens 时选定的。线性默认参数事后调过一次（差距 0.15%）。这不满足“实现前冻结”；下一次修改预处理或训练前，应先重新冻结一套 goldens。 |
| P1-04 | 共享 fit/transform | 已验收 | `rustpy-runtime` 中只有一个 `apply_transform`，训练、评估、CLI 预测和应用都使用它。与 scikit-learn 的最大差异 2.1e-7；训练侧与独立进程应用的概率差 < 1e-12。 |
| P1-05 | Run 状态与不可变产物 | 已验收 | planned / running / completed / failed / cancelled / interrupted；用 `process.lock` 判断存活；原子写入；`run.json` 中记录内容哈希。 |
| P1-06 | CPU 线性基线与小 MLP | 已验收 | 只用 Burn（ndarray）；梯度和 Adam 单元测试分别对照有限差分和手写更新；线性目标函数与 scikit-learn 最优值相差 0.15% 以内（见 P1-03 的说明）。 |
| P1-07 | 比较协议与测试集台账 | 已验收 | 按验证集排名；每个数据快照的测试集只用一次；同一快照的其他模型会被拒绝（`E_TEST_SET_REUSE`）；不同快照或协议的运行会被标为不可直接比较；质量、延迟和峰值内存分别报告。 |
| P1-08 | 最小模型包格式与校验 | 已验收 | [bundle-format.md](bundle-format.md)；篡改、缺文件、未列出的文件、符号链接和被修改的声明都会被拒绝（`bundle_tampering_is_rejected`）。 |
| P1-09 | 新 Rust 进程加载 | 已验收 | `examples/predict-app` 只依赖 `rustpy-runtime`（无 Burn、无 Polars、无 Python）；主程序 22 行非空代码。 |
| P1-10 | CLI 端到端 | 已验收 | `tests/cli.rs` 和验收脚本通过二进制驱动 init → bind → plan → run → compare → evaluate → bundle build/check → predict → report。 |

## P2 —— 可靠性与跨厂商 GPU

| ID | 任务 | 状态 | 证据 / 缺口 |
|---|---|---|---|
| P2-01 | DevicePlan 与执行回执 | 进行中 | CPU 路径已验收：计划显示后端、精度、回退策略（`error`，从不静默回退）、数据放置、复制路径、线程数、内存估计与预算；回执记录实际运行情况及是否与计划一致。检测到的 GPU 被报告为 `not_compiled`。没有 GPU 后端和硬件，无法验证 GPU 初始化和认证分级。 |
| P2-02 | R0/R1、缓存与变更分支 | 已验收 | R0 = 新运行；R1 = 复用哈希兼容的阶段（“cached stages reused: …”）；输入变化会使阶段失效；warm start 是带 `parent_run_id` 的新运行，从不称为 R2。 |
| P2-03 | CPU R2 完整恢复 | 已验收 | 在第 130 步崩溃，以及 epoch 中途 SIGINT，在新进程中恢复后：权重和历史与不中断运行逐位一致（`resume_check.json`）。设置、数据、平台或配方改变时拒绝恢复。 |
| P2-04 | CUDA 同一 MLP | 等待外部条件 | 本环境没有 NVIDIA GPU 或 CUDA 工具链。未实现。 |
| P2-05 | 非 NVIDIA GPU 同一 MLP | 等待外部条件 | 没有 AMD/Intel/Apple GPU。未实现。 |
| P2-06 | 数据更新、候选包与回退 | 已验收 | 新数据 → 新快照和新运行；在新数据上评估旧模型包时排除其见过的行；`PredictorHandle::try_update` 失败时保留旧预测器，并拒绝破坏契约的更新；`rollback_to` 恢复旧包（`hot_swap` 示例，第 28 步）。 |
| P2-07 | 10 类失败与取消 | 已验收 | 12 个预设故障，每个都有错误码、出错字段、期望值/实际值、保留的产物和下一步操作；按类别区分退出码；SIGINT → `cancelled`，退出码 130，可恢复。OOM 由内存预算检查覆盖，没有真的触发 OOM。 |
| P2-08 | 第二任务契约设计走查 | 已验收（仅设计） | [extension-walkthrough-llm.md](extension-walkthrough-llm.md)。按设计不做实现。 |

## P3 —— 独立用户与发布候选

| ID | 任务 | 状态 | 证据 / 缺口 |
|---|---|---|---|
| P3-01 | 安装与仅运行时分发 | 进行中 | 已计时 Linux x86_64 清洁 release 构建（依赖已下载时 CLI 约 5 分钟；应用约 24 秒）。没有 Windows 构建、预编译二进制，也没有在清洁 runner 上计时下载。 |
| P3-02 | 本地报告与诊断 | 已验收 | `rustpy report` 生成 HTML + JSON：数据问题、质量、资源、运行之间的配置差异，并标出不可比的运行。 |
| P3-03 | 用户独立任务试验 | 等待外部条件 | 没有参与者。不声称任何通过率。 |
| P3-04 | 上游与模型包升级演练 | 进行中 | 模型包侧规则已实现并测试（严格版本号、`min_runtime_version`、拒绝未知字段）。未演练真实的依赖升级（例如新版 Burn）。 |
| P3-05 | 正确性、设备、持续运行报告 | 进行中 | 已测量 CPU 正确性和封装开销（见验收报告）。没有 GPU 路径，没有 2 小时持续运行。 |
| P3-06 | 双门槛与范围裁决 | 等待外部条件 | 依赖 P3-03、P3-04、P3-05。 |

## P4 及以后

按计划未开始：只有在二分类被反复使用之后，才做一个相邻任务（回归/多分类）；CNN / 序列 / BERT / LLM 只在有真实需求时才做。这方面唯一完成的工作是 LLM 契约走查（P2-08）。

## 什么会改变这些状态

1. 访谈和替代工具对照（P0-02、P0-03、P0-06）。
2. 一台 NVIDIA 机器和一台 AMD/Intel/Apple 机器（P0-05、P2-01、P2-04、P2-05、P3-05）。
3. Windows runner 和带预编译二进制的发布流水线（P3-01）。
4. 五名画像匹配的用户进行计时独立任务（P3-03）。
5. 下一次数值相关修改前，重新冻结参考 goldens（P1-03）。
