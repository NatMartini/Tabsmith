# RustPy

**从一个 CSV/Parquet 文件，到 Rust 应用里一个经过验证的模型——全程 Rust，不需要 Python。**

[English](README.md)

RustPy 是一个围绕“完整任务”而不是“算法清单”组织的 Rust AI 工具包。0.1 版端到端实现了第一条任务：

> 表格二分类：本地 CSV/Parquet → 可信基线 → 小型神经网络 → 公平比较 → 带预处理的模型包 →
> Rust 应用加载 → 用新数据重训，并可回退。

状态：**内部 CPU 原型**。下面的内容都在 CPU（Linux x86_64）上运行并有测试。GPU 训练、用户研究和跨厂商认证**尚未完成**——见[尚未完成的部分](#尚未完成的部分)。

## 五分钟上手

```text
cargo install --path crates/rustpy-cli        # 或：cargo build --release -p rustpy-cli

rustpy init defect-model --sample-data         # 创建项目 + 1 万行合成数据，目标列 `defect` 已绑定
cd defect-model
rustpy plan                                    # 列、缺失值、切分、预处理、候选模型、设备
rustpy run                                     # 在同一切分和同一变换上训练 `linear` 和 `small-mlp`
rustpy compare                                 # 只用验证集比较，并给出推荐
rustpy evaluate --run run-002 --split test     # 锁定最终候选；测试集只使用一次
rustpy bundle build --run run-002              # 不可变模型包：权重 + 预处理 + schema + 评价证据
rustpy bundle check bundles/run-002            # 在本机校验哈希、版本和样例预测
rustpy sample-data new.csv --rows 200 --seed 7 # 生成一些新的合成数据用于预测
rustpy predict --bundle bundles/run-002 --input new.csv --output preds.csv
rustpy report                                  # 离线 HTML/JSON 报告，位于 .rustpy/reports/
```

使用自己的数据：

```text
rustpy init churn && cd churn
rustpy data bind ../customers.csv --target churned --positive yes --group customer_id
rustpy plan
```

`--group customer_id` 让同一客户的所有行留在同一个子集里。如果每一行都有唯一的键，请改用 `--id <列名>`：文件增长时各行保持原来的切分。没有 id 时，行按内容识别；只要旧行没有被修改、文件的列保持不变，这在不同数据版本之间同样有效。

### 新数据到达时

替换同一路径下的文件或在末尾追加数据，然后：

```text
rustpy run                                       # 新快照、新 Run；旧 Run 和旧模型包保持不变
rustpy compare                                   # 对新 Run 排名（以最新数据版本为准）
rustpy evaluate --run run-004 --split test       # 新候选在新测试行上的表现
rustpy evaluate --bundle bundles/run-002 --split test   # 旧模型在同一批行上的表现；跳过它训练时用过的行
rustpy bundle build --run run-004
rustpy predict --bundle bundles/run-004 --input new.csv --keep customer_id,month --output preds.csv
```

当输入中出现模型包从未见过的类别（新月份、新产品或新代码）时，`predict` 会发出警告；`--keep` 会把用于识别行的列原样复制到输出。在应用中，`PredictorHandle::try_update` 只有在新包加载并通过自检后才切换，`rollback_to` 可以切回旧包。

在应用中（只依赖 `rustpy-runtime`：没有 Burn、没有 Polars、没有 Python）：

```rust
use rustpy_runtime::{LoadOptions, Predictor, Record};

let predictor = Predictor::load("bundles/run-002", LoadOptions::cpu())?;
let reading = Record::new()
    .with("line", "B")
    .with("vibration_rms", 1.42)
    .with("humidity", None::<f64>)     // 缺失值可以传入，按训练时的方式填补
    /* … 其余输入字段 … */;
let p = predictor.predict_one(&reading)?;
println!("{} (p = {:.3})", p.label, p.probability);
```

错误是 `RustpyError` 值，包含错误码、字段、期望值/实际值和下一步建议；`e.render()`（以及 `{:?}`，因此 `fn main() -> Result<…>` 也一样）会打印多行报告。`p.unseen_categories` 列出该预测中模型包从未见过其取值的输入字段。

`predict_serialize(&[my_struct])` 接受任何字段名与模型包输入一致的 `#[derive(Serialize)]` 结构体；
`PredictorHandle::try_update` 只在新包加载并自检通过后才切换，否则旧模型继续服务。如果更新改变了输入或输出契约（新增必需字段、标签或阈值改变等），默认会被拒绝，除非传入 `LoadOptions::cpu().allow_contract_change(true)`；无论是否允许，返回的报告都会列出每一处变化。见 [examples/predict-app](examples/predict-app)。

## “高度集成”在这里的含义

| 约定 | 用户得到什么 | 失败时的表现 |
|---|---|---|
| 一个项目配置 | `rustpy.toml`、profile 和 `--set` 在 CLI 与 SDK 中按同一规则解析；每个生效值都记录来源（`rustpy config`） | 未知字段、类型错误会报错并给出位置 |
| 一份拟合后的预处理 | 统计量只从训练行学习；训练、评估、离线预测和应用内预测使用**同一份代码** | 缺字段、类型错误、（按配置）未知类别，在预测前带字段名报错 |
| 一个可追踪的 Run | 配置哈希、数据快照、切分、变换、设备计划与回执、训练历史、指标、带哈希的产物 | 区分 `failed`、`cancelled`、`interrupted`；保留检查点 |
| 一个可交付的 Bundle | 权重、变换、输入输出 schema、标签映射、评价证据、合成样例、文件哈希 | 损坏或不完整的包绝不会加载一半 |
| 一份执行计划 | 训练前就显示请求设备与实际设备、精度、数据复制路径、内存估算 | 没有静默回退：设备不可用就是错误 |
| 一套诊断体验 | 每个错误都有稳定代码、阶段、字段、期望/实际、保留了什么、下一条命令 | 按错误类别区分退出码；`--json` 便于自动化 |

## 工作区结构

| crate | 职责 | 依赖 |
|---|---|---|
| `rustpy-contracts` | 配置分层、错误、schema、Run/Bundle/评价记录 | serde、toml、sha2 |
| `rustpy-runtime` | Bundle 校验、共享变换、FP32 模型执行、`Predictor` | contracts、safetensors |
| `rustpy-workflow` | 快照、切分、拟合、训练（Burn 0.22，CPU）、评估、打包、报告 | Polars 0.55、Burn 0.22 |
| `rustpy-cli` | `rustpy` 命令行 | workflow、runtime |

更多文档：[架构](docs/architecture.md) · [配置](docs/configuration.md) · [模型包格式](docs/bundle-format.md) ·
[错误](docs/errors.md) · [实施待办](docs/implementation-backlog.md) · [验收报告](reports/implementation-acceptance.md)

## 有测试覆盖的保证

- 预处理一致：原始行 → 训练侧概率与应用侧 `Predictor` 概率相等（最大 |Δp| < 1e-12，在独立进程中检查）；scikit-learn 复现了 RustPy 的变换矩阵（最大 |Δ| ≈ 2e-7，f32 精度）。
- 结构上防泄漏：先切分再拟合；分组切分不会把同一实体放进两个子集；时间相关数据会被拒绝，而不是随机切分。
- 精确续训（R2）：在一个进程中崩溃、在另一个进程中恢复的 Run，得到与不中断训练**逐位相同**的权重和历史（同一二进制、同一 CPU 指令集）。
- 测试集纪律：一个数据快照的测试集只给一个模型使用；用它评估同一快照的另一个模型会被拒绝，除非显式允许，并且会被记录。在新数据上评估模型包时，会跳过其来源 Run 训练或选择时用过的行。
- 模型包不可变、按内容寻址，并逐文件校验。
- 指标与 scikit-learn 一致（冻结的 goldens，容差 1e-12，实测 ≤ 3e-16）；线性基线与 scikit-learn 在同一目标函数上的最优值相差 0.15% 以内；梯度与独立有限差分一致；第一步 Adam 更新符合更新公式。
- 每个 Run 都记录自己的峰值内存（Linux），比较表把它与质量、延迟并列显示。

## 尚未完成的部分

- **GPU**：没有编译或认证任何 CUDA 或非 NVIDIA 后端。`candidate_device = "auto-certified"` 会选择 CPU 并明确说明；`"cuda"` 会报错。跨厂商 GPU 支持（计划 P2-04/P2-05）需要真实硬件。
- **用户验证**：尚未进行访谈或用户试用（P0-02、P0-06、P3-03）。易用性目标未经检验。
- **平台**：只在 Linux x86_64 上构建和测试过。Windows、macOS 构建、预编译二进制和清洁机器安装计时尚未完成（P3-01）。
- **参考值**：scikit-learn goldens 是在实现之后生成的，而不是实现前冻结的（P1-03）；见[实施待办](docs/implementation-backlog.md)。
- 按时间切分、回归/多分类、树模型、CNN/序列/文本/LLM 等任务包属于后续工作。

许可：MIT OR Apache-2.0。附带的样例数据是合成数据（CC0）。
