# Implementation backlog

[中文](#中文)

Status of every task in the project's v2 task list (an internal planning
document that is not part of this repository; each row says what its task is),
dated 2026-10-27. Status values are the four the task list allows: **Not
started**, **In progress**, **Waiting on external conditions**, **Accepted**.
"Accepted" here means *accepted by internal engineering evidence* (tests, the
acceptance script, reference comparisons) on Linux x86_64, and for GPU rows by
certification on real hardware (an NVIDIA RTX 5090 and an AMD integrated GPU, on
Windows 11). No task is marked accepted on the basis of user trials or interviews,
because none happened.

Evidence for "Accepted" rows: the automated tests (`cargo test --workspace`:
140 tests and 6 doctests, passing on Linux x86_64 and Windows x86_64; GPU tests in
`tests/gpu.rs` with the `vulkan`/`cuda` features), the acceptance run in
[reports/implementation-acceptance.md](../reports/implementation-acceptance.md), the
GPU certifications in [reports/gpu/](../reports/gpu/) and the fuzzing run in
[reports/fuzz/](../reports/fuzz/).

## P0 — is it worth building?

| ID | Task | Status | What exists / what is missing |
|---|---|---|---|
| P0-01 | User/task boundary and success criteria | In progress | The scope is frozen in code and docs: supervised learning on one table (binary, multiclass, regression), linear + small-MLP + gradient-boosted trees, CPU with opt-in certified GPUs, explicit "not supported" list (README). Missing: user personas and a hypothesis register written from real users. |
| P0-02 | Observe real work, interviews | Waiting on external conditions | No interviews were held. Contacting external people needs separate authorization. |
| P0-03 | Best-alternative task comparison | In progress | Only the machine side was measured: Tabsmith's training loop vs. a plain Burn loop on the same data and objective (wrapper overhead). Human time with Burn + hand-written glue, or Python training + Rust serving, was not measured. |
| P0-04 | Success/failure journey prototype | Accepted | Replaced by the real implementation: commands, configuration, a bundle and 12 preset failures run end to end. Sample data is synthetic and labelled as such (`tabsmith sample-data`). |
| P0-05 | Dependency and device risk probe | Accepted | CPU: clean build timed, Burn 0.22.1 + Polars 0.55.2 pinned. GPU: a probe of the training loop on Burn's CUDA and Vulkan backends (RTX 5090, CUDA 12.8; AMD integrated GPU) found three risks, each now handled: autotuning makes CUDA nondeterministic across processes (autotune off), CUDA computes f32 matmuls in TF32 with no switch (explicit `precision = "tf32"`), and autotuning lost the device on the integrated GPU (autotune off). Findings in [design-decisions.md](design-decisions.md) D11; upstream reports drafted, not filed. |
| P0-06 | Direction decision | Waiting on external conditions | Needs P0-02 (≥3 people with the repeated problem, ≥2 trial partners). Not decided by default. |

## P1 — full CPU workflow

| ID | Task | Status | Evidence |
|---|---|---|---|
| P1-01 | 4-crate skeleton, pinned versions, config parsing | Accepted | `tabsmith-contracts` / `-runtime` / `-workflow` / `-cli`; `Cargo.lock` committed; unknown keys rejected with location and suggestion; `tabsmith config` shows the layer of every value. Tests: `unknown_field_is_reported_with_location`, `layering_and_sources`, `effective_values_cover_every_leaf`. |
| P1-02 | Input snapshot, schema, split | Accepted | CSV/Parquet; content-hashed snapshots; stratified, id-stable (`tabsmith.split.v4-stable-key`) and group splits; same path with new content creates a new snapshot. Tests: `id_based_splits_are_stable_across_data_versions`, `group_split_has_no_leakage`, `data_update_creates_new_snapshot_and_old_bundle_stays_usable`. |
| P1-03 | Reference values and tolerances before implementation | In progress | scikit-learn goldens exist (`fixtures/reference/goldens.json`, 3 tests) but were generated **after** the implementation, not frozen before it. The tolerances (1e-5 transform, 1e-12 metrics, 1 % objective gap) were chosen when the goldens were made. The linear defaults were tuned once afterwards (gap 0.15 %). This does not meet "frozen before implementation"; a fresh golden set should be frozen before the next change to preprocessing or training. |
| P1-04 | Shared fit/transform | Accepted | One `apply_transform` in `tabsmith-runtime` is used by training, evaluation, CLI prediction and the app. Max difference vs. scikit-learn 2.3e-7 in the acceptance run; training-side vs. separate-process app probabilities < 1e-12. |
| P1-05 | Run states and immutable artifacts | Accepted | planned / running / completed / failed / cancelled / interrupted; `process.lock` liveness; atomic writes; content hashes in `run.json`. |
| P1-06 | CPU linear baseline and small MLP | Accepted | Burn (Flex since the move off the deprecated NdArray backend; same results to ~1e-7); gradient and Adam unit tests against finite differences and a hand-written update; linear objective within 0.10 % of the scikit-learn optimum in the acceptance run (see P1-03 caveat). |
| P1-07 | Comparison protocol and test-set ledger | Accepted | Ranked by validation; test used once per data snapshot; other models of the same snapshot are refused (`E_TEST_SET_REUSE`); runs on different snapshots or protocols are marked not directly comparable; quality, latency and peak memory reported separately. |
| P1-08 | Minimal bundle format and checks | Accepted | [bundle-format.md](bundle-format.md); tampering, missing files, unlisted files, symlinks and edited claims are rejected (`bundle_tampering_is_rejected`). |
| P1-09 | Load in a new Rust process | Accepted | `examples/predict-app` depends only on `tabsmith-runtime` (no Burn, no Polars, no Python); main program is 22 non-blank lines. |
| P1-10 | CLI end-to-end | Accepted | `tests/cli.rs` and the acceptance script drive init → bind → plan → run → compare → evaluate → bundle build/check → predict → report through the binary. |

## P2 — reliability and cross-vendor GPU

| ID | Task | Status | Evidence / gap |
|---|---|---|---|
| P2-01 | DevicePlan and receipt | Accepted | The plan shows backend, precision, certification, fallback (`error`, never silent), data placement, copy path, thread count, memory estimate vs. budget; the receipt records what ran and whether it matched. GPUs are detected read-only (wgpu, cudarc, behind `catch_unwind`), initialised with a test product before training, and chosen by `auto-certified` only when certified and a cost model expects ≥ 2× speed-up. Exercised on an RTX 5090 (Vulkan, CUDA) and an AMD integrated GPU (Vulkan); not on Linux or macOS GPUs. |
| P2-02 | R0/R1, caching and change branches | Accepted | R0 = new run; R1 = reuse of hash-compatible stages ("cached stages reused: …"); changed inputs invalidate stages; warm start is a new run with `parent_run_id`, never called R2. |
| P2-03 | CPU R2 exact resume | Accepted | Crash at step 130 and SIGINT mid-epoch, resumed in a new process: weights and history bitwise identical to an uninterrupted run (`resume_check.json`). Changed settings, data, platform or recipe are refused. |
| P2-04 | CUDA, same MLP | Accepted (Windows, RTX 5090, TF32) | Feature `cuda`. Certified by `tools/gpu_certify.py` ([report](../reports/gpu/2026-10-27-rtx5090-windows-cuda-tf32/)): two processes train bit-identical weights, a crash-and-resume equals the uninterrupted run, validation metrics agree with the CPU within the TF32 tolerance, and the bundle passes the CPU runtime's self-check. Strict fp32 is not possible on CUDA (upstream has no switch; refused rather than lowered silently). |
| P2-05 | Non-NVIDIA GPU, same MLP | Accepted (Windows, Vulkan) | Feature `vulkan` (wgpu), fp32. Certified the same way on an AMD integrated GPU ([report](../reports/gpu/2026-10-27-radeon-igpu-windows-vulkan-fp32/)) and on the RTX 5090 through Vulkan ([report](../reports/gpu/2026-10-27-rtx5090-windows-vulkan-fp32/)). Not run: Intel, Apple (Metal), Linux GPU drivers. |
| P2-06 | Data update, candidate bundle, rollback | Accepted | New data → new snapshot and runs; old bundle evaluated on new data with seen rows excluded; `PredictorHandle::try_update` keeps the old predictor on failure and refuses contract breaks; `rollback_to` restores the old bundle (`hot_swap` example, step 28). |
| P2-07 | Ten failure classes and cancellation | Accepted | 12 preset failures, each with an error code, the offending field, expected/actual, retained artifacts and a next action; exit codes per family; SIGINT → `cancelled`, exit 130, resumable. OOM is covered by the memory budget check, not by provoking a real OOM. |
| P2-08 | Second-task contract walkthrough | Accepted (design only) | [extension-walkthrough-llm.md](extension-walkthrough-llm.md). Nothing implemented, by design. |

## P3 — independent users and release candidate

| ID | Task | Status | Evidence / gap |
|---|---|---|---|
| P3-01 | Install and runtime-only distribution | In progress | Linux x86_64 clean release build timed (see the acceptance report). The full test suite passes on Windows 11 x86_64 (run locally). CI (Linux, macOS, Windows), a release workflow with prebuilt binaries for five targets, `install.sh`, `install.ps1` and cargo-binstall metadata exist. The repository is public at https://github.com/NatMartini/Tabsmith since 2026-10-28; the first CI runs there are pending, and there is no release tag yet. Linux arm64 and macOS Intel binaries are built but never tested; no clean-runner download timing. |
| P3-02 | Local report and diagnostics | Accepted | `tabsmith report` writes HTML + JSON: data issues, quality, resources, configuration differences between runs, and non-comparable runs flagged. |
| P3-03 | Independent user trials | Waiting on external conditions | No participants. No pass rate is claimed. |
| P3-04 | Upstream and bundle upgrade drill | In progress | Bundle-side rules exist and are tested (strict versions, `min_runtime_version`, unknown fields rejected). No real dependency upgrade (e.g. a new Burn release) was rehearsed. |
| P3-05 | Correctness, device and soak report | In progress | CPU correctness and wrapper overhead measured (acceptance report); GPU paths certified with timings ([reports/gpu](../reports/gpu/)); the bundle loader fuzzed for 900 s with network and tree bundles of every task, no crash ([reports/fuzz](../reports/fuzz/)). No 2-hour soak run. |
| P3-06 | Dual gate and scope decision | Waiting on external conditions | Needs P3-03, P3-04, P3-05. |

## P4 and later

The plan put adjacent tasks after repeated use of binary classification. They were
built earlier, at the maintainer's decision and without the user evidence the plan
asked for. They are recorded here as engineering work, not as validated demand:

| ID | Task | Status | Evidence |
|---|---|---|---|
| P4-01 | Multiclass and regression recipes | Accepted (engineering) | `tabular-multiclass@0.1`, `tabular-regression@0.1`; task enums through targets, metrics, bundles and predictions; binary path unchanged (goldens and bitwise tests untouched). Tests: `tasks.rs`. |
| P4-02 | Gradient-boosted trees | Accepted (engineering) | `gbdt` candidate, `tabsmith.trees.v1`; deterministic, exact resume, engine Δ = 0. Tests: `gbdt.rs`; compared with scikit-learn's HistGradientBoosting in [reports/benchmarks](../reports/benchmarks/). |
| P4-03 | Runtime in WebAssembly | Accepted (engineering) | `examples/wasm-predict`; CI builds the runtime for `wasm32-unknown-unknown`. |
| P4-04 | Assurance: no `unsafe`, property tests, fuzzing | Accepted (engineering) | `unsafe_code = "forbid"`; `properties.rs`; `fuzz/` and [reports/fuzz](../reports/fuzz/). |
| P4-05 | Models compiled into Rust code (`no_std`) | In progress | `tabsmith-embedded` (kernels shared with the runtime) and `tabsmith-macros` (`model!`). Compiled and runtime predictions are bit-identical on seven bundles (`equivalence.rs`, 3,000 random records each). `portable.rs` records the prediction hashes on Windows x86_64; CI checks them on Linux and macOS too. `examples/embedded-predict` builds for `thumbv7em-none-eabihf` and `wasm32`. Six Kani proofs (prediction never panics, stage by stage) pass with Kani 0.68 on Linux, each under 2 GB of memory. CI: the tests pass on Linux, macOS and Windows, hashes included. Gap: the Kani job has not passed in CI yet; its first runs stopped on the old proof that walked all 14 trees at once, which needs more memory than a runner has. |

CNN / sequence / BERT / LLM only with real demand. The LLM contract walkthrough
(P2-08) is the only work done there; it now records what the tabular tasks
generalised.

## What would change these statuses

1. Interviews and alternative-tool sessions (P0-02, P0-03, P0-06).
2. GPUs on Linux and macOS, and Intel/Apple GPUs (P2-04 and P2-05 beyond Windows).
3. The first CI run and tagged release on GitHub, and test runners for Linux arm64 and macOS Intel (P3-01).
4. Five matching users for timed independent tasks (P3-03).
5. Re-freezing reference goldens before the next numeric change (P1-03).

---

## 中文

# 实施待办

本文记录项目 v2 任务清单（内部规划文档，不在本仓库中；每一行都说明了对应任务的内容）中每个任务的状态，日期 2026-10-27。状态只使用任务清单允许的四种：**未开始**、**进行中**、**等待外部条件**、**已验收**。这里的“已验收”指*在 Linux x86_64 上，由内部工程证据验收*（自动测试、验收脚本、参考对照）；GPU 条目则以真实硬件上的认证为依据。

“已验收”的证据：自动测试（`cargo test --workspace`：140 个测试和 6 个文档测试，在 Linux x86_64 和 Windows x86_64 上通过；GPU 测试 `tests/gpu.rs` 需启用 `vulkan`/`cuda` 特性）、[reports/implementation-acceptance.md](../reports/implementation-acceptance.md) 中的验收运行、[reports/gpu/](../reports/gpu/) 中的 GPU 认证，以及 [reports/fuzz/](../reports/fuzz/) 中的模糊测试。GPU 相关条目以真实硬件（NVIDIA RTX 5090 和 AMD 核显，Windows 11）上的认证为证据；没有任何条目基于用户试用或访谈验收，因为这些都没有发生。

## P0 —— 是否值得构建

| ID | 任务 | 状态 | 已有 / 缺失 |
|---|---|---|---|
| P0-01 | 用户/任务边界与成功标准 | 进行中 | 范围已在代码和文档中冻结：单表监督学习（二分类、多分类、回归）、线性 + 小 MLP + 梯度提升树、CPU 加可选的已认证 GPU，以及明确的“不支持”列表（README）。缺失：基于真实用户的用户画像和假设登记。 |
| P0-02 | 观察真实工作与访谈 | 等待外部条件 | 未进行任何访谈。联系外部人员需要另行授权。 |
| P0-03 | 最佳替代方案任务对照 | 进行中 | 只测了机器侧：同一数据和目标函数下，Tabsmith 训练循环对比纯 Burn 循环（封装开销）。未测量用 Burn + 手写胶水、或 Python 训练 + Rust 部署所需的人工时间。 |
| P0-04 | 成功与失败旅程原型 | 已验收 | 已被真实实现取代：命令、配置、模型包和 12 个预设故障均端到端运行。样例数据是合成的，并明确标注（`tabsmith sample-data`）。 |
| P0-05 | 最小依赖与设备风险探针 | 已验收 | CPU：已计时清洁构建，锁定 Burn 0.22.1 + Polars 0.55.2。GPU：在 Burn 的 CUDA 与 Vulkan 后端上探测训练循环（RTX 5090、CUDA 12.8；AMD 核显），发现三个风险并逐一处理：autotune 使 CUDA 跨进程不确定（关闭 autotune）；CUDA 以 TF32 计算 f32 矩阵乘法且无开关（显式 `precision = "tf32"`）；核显在 autotune 下设备丢失（关闭 autotune）。见 [design-decisions.md](design-decisions.md) D11；上游报告已起草、未提交。 |
| P0-06 | 方向裁决 | 等待外部条件 | 依赖 P0-02（至少 3 人有重复问题、至少 2 个试用伙伴）。不会默认通过。 |

## P1 —— 完整 CPU 工作流

| ID | 任务 | 状态 | 证据 |
|---|---|---|---|
| P1-01 | 4-crate 骨架、锁版本、配置解析 | 已验收 | `tabsmith-contracts` / `-runtime` / `-workflow` / `-cli`；提交了 `Cargo.lock`；未知键会被拒绝并给出位置和建议；`tabsmith config` 显示每个值来自哪一层。测试：`unknown_field_is_reported_with_location`、`layering_and_sources`、`effective_values_cover_every_leaf`。 |
| P1-02 | 输入快照、schema 与 split | 已验收 | CSV/Parquet；基于内容哈希的快照；分层、基于 id 的稳定切分（`tabsmith.split.v4-stable-key`）和分组切分；同一路径换了内容会生成新快照。测试：`id_based_splits_are_stable_across_data_versions`、`group_split_has_no_leakage`、`data_update_creates_new_snapshot_and_old_bundle_stays_usable`。 |
| P1-03 | 实现前的参考值与容差 | 进行中 | 已有 scikit-learn goldens（`fixtures/reference/goldens.json`，3 个测试），但它们是在实现**之后**生成的，而非实现前冻结。容差（变换 1e-5、指标 1e-12、目标函数差距 1%）是在生成 goldens 时选定的。线性默认参数事后调过一次（差距 0.15%）。这不满足“实现前冻结”；下一次修改预处理或训练前，应先重新冻结一套 goldens。 |
| P1-04 | 共享 fit/transform | 已验收 | `tabsmith-runtime` 中只有一个 `apply_transform`，训练、评估、CLI 预测和应用都使用它。与 scikit-learn 的最大差异为 2.3e-7（验收运行）；训练侧与独立进程应用的概率差 < 1e-12。 |
| P1-05 | Run 状态与不可变产物 | 已验收 | planned / running / completed / failed / cancelled / interrupted；用 `process.lock` 判断存活；原子写入；`run.json` 中记录内容哈希。 |
| P1-06 | CPU 线性基线与小 MLP | 已验收 | 只用 Burn（已从弃用的 NdArray 迁移到 Flex，结果相差约 1e-7）；梯度和 Adam 单元测试分别对照有限差分和手写更新；线性目标函数与 scikit-learn 最优值相差 0.10% 以内（验收运行）（见 P1-03 的说明）。 |
| P1-07 | 比较协议与测试集台账 | 已验收 | 按验证集排名；每个数据快照的测试集只用一次；同一快照的其他模型会被拒绝（`E_TEST_SET_REUSE`）；不同快照或协议的运行会被标为不可直接比较；质量、延迟和峰值内存分别报告。 |
| P1-08 | 最小模型包格式与校验 | 已验收 | [bundle-format.md](bundle-format.md)；篡改、缺文件、未列出的文件、符号链接和被修改的声明都会被拒绝（`bundle_tampering_is_rejected`）。 |
| P1-09 | 新 Rust 进程加载 | 已验收 | `examples/predict-app` 只依赖 `tabsmith-runtime`（无 Burn、无 Polars、无 Python）；主程序 22 行非空代码。 |
| P1-10 | CLI 端到端 | 已验收 | `tests/cli.rs` 和验收脚本通过二进制驱动 init → bind → plan → run → compare → evaluate → bundle build/check → predict → report。 |

## P2 —— 可靠性与跨厂商 GPU

| ID | 任务 | 状态 | 证据 / 缺口 |
|---|---|---|---|
| P2-01 | DevicePlan 与执行回执 | 已验收 | 计划显示后端、精度、认证状态、回退策略（`error`，从不静默回退）、数据放置、复制路径、线程数、内存估计与预算；回执记录实际运行情况及是否与计划一致。GPU 以只读方式探测（wgpu、cudarc，在 `catch_unwind` 内），训练前用一次测试乘法初始化；`auto-certified` 只在已认证且成本模型预计至少快一倍时选择 GPU。已在 RTX 5090（Vulkan、CUDA）和 AMD 核显（Vulkan）上验证；未在 Linux 或 macOS 的 GPU 上验证。 |
| P2-02 | R0/R1、缓存与变更分支 | 已验收 | R0 = 新运行；R1 = 复用哈希兼容的阶段（“cached stages reused: …”）；输入变化会使阶段失效；warm start 是带 `parent_run_id` 的新运行，从不称为 R2。 |
| P2-03 | CPU R2 完整恢复 | 已验收 | 在第 130 步崩溃，以及 epoch 中途 SIGINT，在新进程中恢复后：权重和历史与不中断运行逐位一致（`resume_check.json`）。设置、数据、平台或配方改变时拒绝恢复。 |
| P2-04 | CUDA 同一 MLP | 已验收（Windows、RTX 5090、TF32） | 特性 `cuda`。经 `tools/gpu_certify.py` 认证（[报告](../reports/gpu/2026-10-27-rtx5090-windows-cuda-tf32/)）：两个进程训练出逐位相同的权重，崩溃后续训与不中断训练相同，验证指标在 TF32 容差内与 CPU 一致，模型包通过 CPU 运行时自检。CUDA 无法严格按 fp32 训练（上游没有开关；会拒绝而不是悄悄降低精度）。 |
| P2-05 | 非 NVIDIA GPU 同一 MLP | 已验收（Windows、Vulkan） | 特性 `vulkan`（wgpu），fp32。以同样方式在 AMD 核显（[报告](../reports/gpu/2026-10-27-radeon-igpu-windows-vulkan-fp32/)）和通过 Vulkan 使用的 RTX 5090（[报告](../reports/gpu/2026-10-27-rtx5090-windows-vulkan-fp32/)）上认证。未运行：Intel、Apple（Metal）、Linux GPU 驱动。 |
| P2-06 | 数据更新、候选包与回退 | 已验收 | 新数据 → 新快照和新运行；在新数据上评估旧模型包时排除其见过的行；`PredictorHandle::try_update` 失败时保留旧预测器，并拒绝破坏契约的更新；`rollback_to` 恢复旧包（`hot_swap` 示例，第 28 步）。 |
| P2-07 | 10 类失败与取消 | 已验收 | 12 个预设故障，每个都有错误码、出错字段、期望值/实际值、保留的产物和下一步操作；按类别区分退出码；SIGINT → `cancelled`，退出码 130，可恢复。OOM 由内存预算检查覆盖，没有真的触发 OOM。 |
| P2-08 | 第二任务契约设计走查 | 已验收（仅设计） | [extension-walkthrough-llm.md](extension-walkthrough-llm.md)。按设计不做实现。 |

## P3 —— 独立用户与发布候选

| ID | 任务 | 状态 | 证据 / 缺口 |
|---|---|---|---|
| P3-01 | 安装与仅运行时分发 | 进行中 | 已计时 Linux x86_64 清洁 release 构建（见验收报告）。完整测试套件在 Windows 11 x86_64 上通过（本地运行）。CI（Linux、macOS、Windows）、为五个目标平台构建预编译二进制的发布流程、`install.sh`、`install.ps1` 和 cargo-binstall 元数据均已就绪。仓库自 2026-10-28 起公开在 https://github.com/NatMartini/Tabsmith ；那里的 CI 还在等待首次运行，也还没有发布标签。Linux arm64 和 macOS Intel 二进制只构建、从未测试；没有在清洁 runner 上计时下载。 |
| P3-02 | 本地报告与诊断 | 已验收 | `tabsmith report` 生成 HTML + JSON：数据问题、质量、资源、运行之间的配置差异，并标出不可比的运行。 |
| P3-03 | 用户独立任务试验 | 等待外部条件 | 没有参与者。不声称任何通过率。 |
| P3-04 | 上游与模型包升级演练 | 进行中 | 模型包侧规则已实现并测试（严格版本号、`min_runtime_version`、拒绝未知字段）。未演练真实的依赖升级（例如新版 Burn）。 |
| P3-05 | 正确性、设备、持续运行报告 | 进行中 | 已测量 CPU 正确性和封装开销（验收报告）；GPU 路径已认证并计时（[reports/gpu](../reports/gpu/)）；用三种任务的网络与树模型包对模型包加载器做了 900 秒模糊测试，无崩溃（[reports/fuzz](../reports/fuzz/)）。没有 2 小时持续运行。 |
| P3-06 | 双门槛与范围裁决 | 等待外部条件 | 依赖 P3-03、P3-04、P3-05。 |

## P4 及以后

计划把相邻任务放在二分类被反复使用之后。它们被提前实现了：这是维护者的决定，并没有计划所要求的用户证据，因此这里只记作工程工作，而不是已验证的需求——P4-01 多分类与回归 recipe、P4-02 梯度提升树、P4-03 WebAssembly 中的运行时、P4-04 保障措施（禁止 `unsafe`、性质测试、模糊测试），均为“已验收（工程）”，证据见上方英文表格。P4-05 编译进 Rust 代码的模型（`no_std`）为“进行中”：编译模型与运行时在 7 个模型包上逐位相同，Windows 上记录了预测哈希；6 个 Kani 证明（逐段证明预测不会 panic）已在 Linux 上用 Kani 0.68 通过，每个用的内存都不到 2 GB；CI 上的测试在 Linux、macOS 和 Windows 上都通过（包括预测哈希）；缺口是 kani 任务还没在 CI 上通过过：头几次运行停在旧的那个一次走完 14 棵树的证明上，它需要的内存超过 runner 的容量。CNN / 序列 / BERT / LLM 只在有真实需求时才做；这方面唯一完成的工作是 LLM 契约走查（P2-08），它现在也记录了表格任务泛化了哪些部分。

## 什么会改变这些状态

1. 访谈和替代工具对照（P0-02、P0-03、P0-06）。
2. Linux 和 macOS 上的 GPU，以及 Intel/Apple GPU（P2-04、P2-05 在 Windows 之外的部分）。
3. 在 GitHub 上首次运行 CI 并发布带标签的版本，以及 Linux arm64 和 macOS Intel 的测试 runner（P3-01）。
4. 五名画像匹配的用户进行计时独立任务（P3-03）。
5. 下一次数值相关修改前，重新冻结参考 goldens（P1-03）。
