# Contributing to Tabsmith

[中文](#中文)

Thank you for helping. Tabsmith is an early, CPU-only release; bug reports with a
reproduction, tests, and careful fixes are the most valuable contributions.

## Ground rules

- **No claims without evidence.** Every statement about behaviour, accuracy,
  speed or memory, in code comments, docs, the README, the changelog or a pull
  request, must point to a test, a command whose output anyone can reproduce, or
  a checked-in report. If you cannot show it, write it as a limitation or a
  question instead.
- **GPU paths are not certified.** No GPU backend is compiled in or tested. Do not
  describe GPU support as working, and keep the rule that an unavailable or
  uncertified device is an error, never a silent fallback to the CPU.
- Keep "what is not done yet" honest: when a change narrows or widens what works,
  update the documentation that lists it.
- User-facing documentation is written in English with a Chinese translation
  (`README.md` / `README.zh-CN.md`, and a `中文` section in other documents).
  If you cannot write one of the two languages, say so in the pull request and
  a maintainer will help.
- Be kind and constructive; see the [Code of Conduct](CODE_OF_CONDUCT.md).
  Report security problems privately, as described in [SECURITY.md](SECURITY.md).

## Development setup

1. Install Rust with [rustup](https://rustup.rs). The workspace needs Rust 1.95
   or newer (Burn 0.22.1 requires it); `tabsmith-contracts` and `tabsmith-runtime`
   alone build with Rust 1.88.
2. Clone the repository and build:

   ```sh
   cargo build --workspace
   ```

   The first build compiles Polars and Burn and takes several minutes. Debug
   builds optimise dependencies (`opt-level = 2`) so that training in tests is
   not painfully slow.
3. Try the CLI from the checkout: `cargo run -p tabsmith-cli -- --help`.

No Python is needed, except to regenerate the scikit-learn reference values
(see `tools/reference/README.md`); do not regenerate them to make a failing
test pass.

## Running the checks

Run these before opening a pull request; CI runs the same commands on Linux,
macOS and Windows. On Linux or macOS, `bash tools/verify.sh` runs every check of the
CI workflow in one go and prints a summary; `--all` adds the minimum Rust versions,
the GPU-feature lint, the fuzz target and the Kani proofs.

```sh
# All unit and integration tests of the workspace
cargo test --workspace

# The example application must build with tabsmith-runtime alone
cargo build --manifest-path examples/predict-app/Cargo.toml --all-targets

# ... and must not pull in Burn or Polars (this should print nothing)
cargo tree --manifest-path examples/predict-app/Cargo.toml -e normal --prefix none | grep -E '^(burn|polars)'
```

`examples/predict-app` is deliberately outside the workspace: its own
`Cargo.lock` shows exactly what an application that loads bundles depends on.
`examples/wasm-predict` (the runtime in a web page) is outside it too; the `wasm`
CI job builds it for `wasm32-unknown-unknown`:

```sh
rustup target add wasm32-unknown-unknown
cargo build -p tabsmith-runtime --target wasm32-unknown-unknown
cargo test --manifest-path examples/wasm-predict/Cargo.toml
cargo build --release --manifest-path examples/wasm-predict/Cargo.toml --target wasm32-unknown-unknown
```

Models compiled into code (`tabsmith-macros`) build without the standard library. The
`embedded` CI job builds `examples/embedded-predict`, also outside the workspace, for a
Cortex-M4F and for WebAssembly. The Kani proofs in that crate run in the `kani` job, or
locally on Linux or macOS (`cargo install --locked kani-verifier`, then `cargo kani` in
`examples/embedded-predict`):

```sh
rustup target add thumbv7em-none-eabihf
cargo build --release --manifest-path examples/embedded-predict/Cargo.toml --target thumbv7em-none-eabihf
cargo test --manifest-path examples/embedded-predict/Cargo.toml
```

Formatting and lints are checked by the `lint` CI job (Linux):

```sh
cargo fmt --all                                   # rustfmt.toml: max_width = 140
cargo fmt --manifest-path examples/predict-app/Cargo.toml
cargo clippy --workspace --all-targets -- -D warnings
cargo clippy --workspace --all-targets --features tabsmith-cli/gpu -- -D warnings   # GPU code, no GPU needed
cargo clippy --manifest-path examples/predict-app/Cargo.toml --all-targets -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps -p tabsmith-contracts -p tabsmith-embedded -p tabsmith-macros -p tabsmith-runtime -p tabsmith-workflow
```

Every workspace crate is compiled with `unsafe_code = "forbid"`; a change that needs
`unsafe` needs a design discussion first. The workspace allows two Clippy lints on purpose (see `[workspace.lints]` in the
root `Cargo.toml`); prefer fixing a warning over adding an `#[allow]`, and say
why next to any `#[allow]` you add.

## Adding tests

- **Unit tests** live next to the code in a `#[cfg(test)] mod tests` block.
- **Workflow integration tests** live in `crates/tabsmith-workflow/tests/`. Use a
  `tempfile::tempdir()` project and generate data with
  `tabsmith_workflow::sample::write_defects_csv` instead of adding data files.
  `review_regressions.rs` collects regression tests for fixed bugs; add yours
  there with a name that describes the behaviour, not the issue number.
- **CLI tests** live in `crates/tabsmith-cli/tests/cli.rs` and run the real
  `tabsmith` binary (`env!("CARGO_BIN_EXE_tabsmith")`) in a temporary directory,
  checking exit codes and `--json` output.
- **Reference values** (`fixtures/reference/`) are checked by
  `crates/tabsmith-workflow/tests/reference_goldens.rs`. Tolerances are part of the
  claim: do not loosen one without explaining why in the pull request.
- Tests must be deterministic (fixed seeds), must not use the network, and
  should keep data small so the suite stays fast. A bug fix comes with a test
  that fails without it.
- Prefer testing observable behaviour (files written, errors returned, exit
  codes) over private details.
- **Property tests** (`crates/tabsmith-workflow/tests/properties.rs`, proptest) state
  invariants over random inputs. Add one when an invariant, not an example, is the
  claim.
- **GPU code** compiles without a GPU (`--features tabsmith-cli/gpu`).
  `crates/tabsmith-workflow/tests/gpu.rs` runs when built with
  `--features vulkan,cuda` on a machine with a usable GPU. A GPU path enters
  `certified_devices.json` only with the evidence of `tools/gpu_certify.py` in
  `reports/gpu/`.
- **Fuzzing** the bundle loader is described in [fuzz/README.md](fuzz/README.md);
  a crash it finds becomes a regression test.

## Pull requests

- Keep each pull request focused on one change and explain what it changes and
  how you checked it (commands and their results).
- Add an entry under `## [Unreleased]` in [CHANGELOG.md](CHANGELOG.md) for
  user-visible changes.
- Breaking changes to the bundle format, the configuration or the error codes
  need a format-version bump and a note in the changelog.

## License of contributions

Tabsmith is dual-licensed under the [MIT license](LICENSE-MIT) and the
[Apache License, Version 2.0](LICENSE-APACHE). Unless you explicitly state
otherwise, any contribution intentionally submitted for inclusion in the work by
you, as defined in the Apache-2.0 license, shall be dual licensed as above,
without any additional terms or conditions.

## Releasing (maintainers)

1. Once, after the GitHub repository exists: run
   `tools/release/set-repo.sh OWNER/REPO` and commit the result. It fills in the
   repository in the install scripts, the cargo-binstall metadata, the Markdown
   links and `Cargo.toml`. The release workflow refuses to run until this is done.
2. Update `version` in `[workspace.package]` and the `version` of the internal
   crates in `[workspace.dependencies]` (root `Cargo.toml`), then move the
   `Unreleased` entries in `CHANGELOG.md` under the new version with today's date.
3. Commit, tag `vX.Y.Z` and push the tag. `.github/workflows/release.yml` builds
   the binaries for five targets and publishes the GitHub release with
   `install.sh`, `install.ps1` and a `.sha256` file per archive. Tags with a
   `-` (for example `v0.2.0-rc.1`) become pre-releases.
4. Publish the crates in dependency order: `tabsmith-contracts`, `tabsmith-embedded`,
   `tabsmith-runtime`, `tabsmith-macros`, `tabsmith-workflow`, `tabsmith-cli`
   (`cargo publish -p <crate>`). Check first
   with `cargo package --list -p <crate>`; the packages contain only `src/`,
   the crate README and the license files (warnings about skipped tests and
   examples are expected).

---

## 中文

感谢参与。Tabsmith 目前是早期的、仅支持 CPU 的版本；附带复现步骤的问题报告、测试和谨慎的修复最有价值。

### 基本规则

- **没有证据就不下结论。** 代码注释、文档、README、变更日志或 Pull Request 中关于行为、
  准确度、速度或内存的任何说法，都必须对应一个测试、一条任何人都能复现输出的命令，或一份已提交的报告。
  无法证明的内容请写成限制或待确认的问题。
- **GPU 路径未经认证。** 目前没有编译或测试任何 GPU 后端。不要把 GPU 支持描述为可用；
  并保持现有规则：设备不可用或未经认证时报错，绝不悄悄回退到 CPU。
- 如实维护“尚未完成”的内容：如果改动扩大或缩小了可用范围，请同步更新列出这些内容的文档。
- 面向用户的文档以英文撰写并附中文翻译（`README.md` / `README.zh-CN.md`，其他文档中附 `中文` 小节）。
  如果你只能写其中一种语言，请在 Pull Request 中说明，维护者会协助。
- 请友善、建设性地交流，见[行为准则](CODE_OF_CONDUCT.md)。安全问题请按 [SECURITY.md](SECURITY.md) 私下报告。

### 开发环境

1. 用 [rustup](https://rustup.rs) 安装 Rust。整个 workspace 需要 Rust 1.95 或更新版本
   （Burn 0.22.1 的要求）；单独构建 `tabsmith-contracts` 和 `tabsmith-runtime` 只需 Rust 1.88。
2. 克隆仓库并构建：`cargo build --workspace`。首次构建需要编译 Polars 和 Burn，耗时数分钟。
   调试构建会优化依赖（`opt-level = 2`），以免测试中的训练过慢。
3. 在仓库中试用 CLI：`cargo run -p tabsmith-cli -- --help`。

除了重新生成 scikit-learn 参考值（见 `tools/reference/README.md`）之外不需要 Python；
不要为了让失败的测试通过而重新生成参考值。

### 运行检查

提交 Pull Request 之前请运行以下命令；CI 会在 Linux、macOS 和 Windows 上运行相同的命令。在 Linux 或 macOS 上，
`bash tools/verify.sh` 一次跑完 CI 工作流的全部检查并打印汇总；加 `--all` 还会检查最低 Rust 版本、GPU 特性的 lint、
模糊测试目标和 Kani 证明：

```sh
cargo test --workspace
cargo build --manifest-path examples/predict-app/Cargo.toml --all-targets
cargo tree --manifest-path examples/predict-app/Cargo.toml -e normal --prefix none | grep -E '^(burn|polars)'   # 应当没有输出
```

`examples/predict-app` 有意放在 workspace 之外：它自己的 `Cargo.lock` 精确展示了一个只加载
bundle 的应用依赖了什么。

格式和 lint 由 CI 的 `lint` 任务（Linux）检查：

```sh
cargo fmt --all                                   # rustfmt.toml：max_width = 140
cargo fmt --manifest-path examples/predict-app/Cargo.toml
cargo clippy --workspace --all-targets -- -D warnings
cargo clippy --workspace --all-targets --features tabsmith-cli/gpu -- -D warnings   # GPU 代码，无需 GPU
cargo clippy --manifest-path examples/predict-app/Cargo.toml --all-targets -- -D warnings
```

全部 workspace crate 都以 `unsafe_code = "forbid"` 编译。`examples/wasm-predict`（网页中的运行时）同样在
workspace 之外，由 CI 的 `wasm` 任务构建为 `wasm32-unknown-unknown`。`examples/embedded-predict`（编译进代码的
模型，`no_std`）也在 workspace 之外，由 `embedded` 任务为 Cortex-M4F 和 WebAssembly 构建；其中的 Kani 证明由
`kani` 任务运行，也可以在 Linux 或 macOS 上本地运行（`cargo install --locked kani-verifier`，然后在该目录执行 `cargo kani`）。GPU 代码无需 GPU 即可编译；
`tests/gpu.rs` 只在启用 `vulkan,cuda` 特性且有可用 GPU 时运行；只有附带 `tools/gpu_certify.py`
证据（`reports/gpu/`）的 GPU 路径才能写入 `certified_devices.json`。模糊测试见 [fuzz/README.md](fuzz/README.md)。

workspace 有意放宽了两条 Clippy lint（见根目录 `Cargo.toml` 的 `[workspace.lints]`）；能修复警告就不要添加
`#[allow]`，确需添加时请在旁边写明原因。

### 添加测试

- **单元测试**放在代码旁的 `#[cfg(test)] mod tests` 中。
- **工作流集成测试**放在 `crates/tabsmith-workflow/tests/`。使用 `tempfile::tempdir()` 创建项目，
  用 `tabsmith_workflow::sample::write_defects_csv` 生成数据，而不是添加数据文件。
  `review_regressions.rs` 收集已修复缺陷的回归测试；请把你的测试加在那里，名称描述行为而不是问题编号。
- **CLI 测试**在 `crates/tabsmith-cli/tests/cli.rs`，在临时目录中运行真实的 `tabsmith` 二进制
  （`env!("CARGO_BIN_EXE_tabsmith")`），检查退出码和 `--json` 输出。
- **参考值**（`fixtures/reference/`）由 `crates/tabsmith-workflow/tests/reference_goldens.rs` 校验。
  容差本身就是结论的一部分：放宽容差必须在 Pull Request 中说明理由。
- 测试必须是确定性的（固定随机种子），不能访问网络，并尽量使用小数据以保持测试套件快速。修复缺陷时要附带一个没有该修复就会失败的测试。
- 优先测试可观察的行为（写出的文件、返回的错误、退出码），而不是内部细节。

### Pull Request

- 每个 Pull Request 只做一件事，并说明改了什么、如何验证（命令及结果）。
- 用户可见的改动请在 [CHANGELOG.md](CHANGELOG.md) 的 `## [Unreleased]` 下添加条目。
- 对 bundle 格式、配置或错误码的破坏性修改，需要提升格式版本并在变更日志中注明。

### 贡献的许可

Tabsmith 采用 [MIT](LICENSE-MIT) 与 [Apache-2.0](LICENSE-APACHE) 双许可。除非你明确声明，
你有意提交以纳入本项目的任何贡献（按 Apache-2.0 许可证的定义）均按上述双许可授权，不附加任何其他条款或条件。

### 发布（维护者）

1. GitHub 仓库建立后执行一次 `tools/release/set-repo.sh OWNER/REPO` 并提交结果。
   它会在安装脚本、cargo-binstall 元数据、Markdown 链接和 `Cargo.toml` 中填入仓库地址；未执行前发布流程会拒绝运行。
2. 更新根 `Cargo.toml` 中 `[workspace.package]` 的 `version` 以及 `[workspace.dependencies]` 中内部 crate 的 `version`，
   再把 `CHANGELOG.md` 中 `Unreleased` 的条目移到新版本下并写上日期。
3. 提交、打标签 `vX.Y.Z` 并推送标签。`.github/workflows/release.yml` 会为五个目标平台构建二进制，
   并发布 GitHub Release（附 `install.sh`、`install.ps1` 和每个压缩包的 `.sha256`）。带 `-` 的标签（如 `v0.2.0-rc.1`）会作为预发布。
4. 按依赖顺序发布 crate：`tabsmith-contracts`、`tabsmith-embedded`、`tabsmith-runtime`、`tabsmith-macros`、`tabsmith-workflow`、`tabsmith-cli`（`cargo publish -p <crate>`）。
   发布前可用 `cargo package --list -p <crate>` 检查；包中只含 `src/`、crate 的 README 和许可证文件（关于跳过测试和示例的警告属正常）。
