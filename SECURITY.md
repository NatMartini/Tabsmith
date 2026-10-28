# Security policy

[中文](#中文)

## Supported versions

Tabsmith is pre-1.0. Security fixes are made on the latest release only.

| Version | Supported |
|---|---|
| 0.1.x (latest) | yes |
| older | no |

## Reporting a vulnerability

Please **do not open a public issue** for a security problem. Report it privately
through GitHub's private vulnerability reporting:
[Report a vulnerability](https://github.com/NatMartini/Tabsmith/security/advisories/new)
(the "Security" tab of the repository).

Include the Tabsmith version (`tabsmith --version`), your platform, what an attacker
controls (a data file, a bundle, a configuration, ...), the steps to reproduce
and the impact you expect. A minimal bundle or data file that triggers the
problem helps most.

This is a small project maintained on a best-effort basis. We aim to acknowledge
a report within 7 days, agree on a fix and a disclosure date with you, and credit
you in the advisory unless you prefer otherwise.

## What to expect from Tabsmith

**Bundles are data, not code.** A bundle is a directory of JSON files and one
`safetensors` weight file. Loading a bundle (`Predictor::load`, `tabsmith predict`,
`tabsmith bundle check`) parses those files; it never executes code from the
bundle, loads native libraries or unpickles objects. A way to make Tabsmith execute
code or touch files outside the bundle directory by loading a bundle is a
vulnerability; please report it.

**Integrity checks.** Before a bundle is used, every file listed in
`manifest.json` is checked against its recorded SHA-256 and size, unlisted files
and symbolic links are refused, the recorded bundle id must match the recomputed
one, and the schemas, transform and weights must be consistent. A damaged,
incomplete or partly edited bundle is refused as a whole. See
`docs/bundle-format.md` for the full list. The loader is written without `unsafe`
code and is fuzzed with arbitrary file sets (`fuzz/`, results in `reports/fuzz/`);
a property test feeds it damaged bundles on every `cargo test`.

**What the checks do not do.** The hashes detect corruption and inconsistent
edits; they are not signatures. Anyone who can write a bundle can produce a
fully consistent one, so a bundle is only as trustworthy as its source. A bundle
from an untrusted source can make wrong predictions, and Tabsmith does not limit
the size of bundle files, so a hostile bundle can use a lot of memory or CPU
time while it is verified. Load bundles only from sources you trust, and keep
them where only trusted users can write.

**Input data.** CSV, Parquet and JSON files are parsed by Polars and serde_json.
Crashes, hangs or unbounded memory use on crafted input files are worth
reporting, even when the root cause is in a dependency.

**Dependencies.** CI runs `cargo audit` on every `Cargo.lock` of the repository. The
lockfiles of applications that load bundles (`examples/predict-app`,
`examples/wasm-predict`, `examples/embedded-predict`) have no open RustSec advisory.
The `tabsmith` tool depends on Polars 0.55 (its latest release), which reads
zstd-compressed Parquet pages and `.zst` CSV files with zstd 0.13. Two advisories
against that version apply in principle: RUSTSEC-2026-0391, where data of an
abandoned stream leaks into the next one on a reused context, and RUSTSEC-2026-0403,
where uninitialised memory is exposed when streaming decompression of a legacy frame
fails. Until Polars moves to zstd 0.14, do not train or predict on Parquet or `.zst`
files from untrusted sources with `tabsmith`. The same upgrade will clear the
unsoundness warnings that cargo-audit reports for zstd-safe 7.3. `tabsmith-runtime`
does not use Polars and is not affected. `.cargo/audit.toml` lists these exceptions with their reasons,
and the other flagged crates (`zip` 0.6, `quick-xml` 0.39), which appear in
`Cargo.lock` but are never built.

**`tabsmith serve`.** The HTTP server has no authentication, no TLS and no request
timeouts: a client that sends a request body slowly keeps one of the worker
threads (4 by default) busy until it finishes. It listens on `127.0.0.1` unless
`--host` says otherwise. To accept connections from other machines, put a reverse
proxy in front of it that provides TLS, authentication and timeouts. `POST /reload`
exists only with `--allow-reload`, and it loads only bundle directories in the folder
of the bundle being served. A path outside that folder gets the same answer as one
that does not exist, so a client cannot probe the server's file system.

**Release downloads.** `install.sh` and `install.ps1` download archives over
HTTPS and verify each against the `.sha256` file published in the same GitHub
release. This protects against corrupted downloads, not against a compromised
release; you can always build from source with `cargo install tabsmith-cli --locked`.

---

## 中文

### 支持的版本

Tabsmith 尚未到 1.0，只在最新版本上修复安全问题（目前为 0.1.x）。

### 报告漏洞

请**不要在公开 issue 中报告安全问题**。请通过 GitHub 的私密漏洞报告功能提交：
[Report a vulnerability](https://github.com/NatMartini/Tabsmith/security/advisories/new)（仓库的 “Security” 标签页）。

请附上 Tabsmith 版本（`tabsmith --version`）、平台、攻击者能控制什么（数据文件、bundle、配置等）、复现步骤和预期影响；
最好附一个能触发问题的最小 bundle 或数据文件。

本项目由维护者尽力维护。我们争取在 7 天内确认收到报告，与你商定修复和公开日期，并在公告中致谢（除非你不希望署名）。

### Tabsmith 的安全边界

**Bundle 是数据，不是代码。** bundle 是一个包含 JSON 文件和一个 `safetensors` 权重文件的目录。
加载 bundle（`Predictor::load`、`tabsmith predict`、`tabsmith bundle check`）只解析这些文件，
从不执行 bundle 中的代码、加载本地库或反序列化 pickle 对象。如果加载 bundle 能让 Tabsmith 执行代码或访问 bundle 目录之外的文件，
这就是漏洞，请报告。

**完整性检查。** 使用 bundle 之前，会对照 `manifest.json` 中记录的 SHA-256 和大小检查每个文件，
拒绝未列出的文件和符号链接，要求记录的 bundle id 与重新计算的一致，并检查 schema、变换和权重相互一致。
损坏、不完整或被部分修改的 bundle 会被整体拒绝。完整列表见 `docs/bundle-format.md`。加载器不含 `unsafe` 代码，并用任意文件集合做了模糊测试（`fuzz/`，结果见 `reports/fuzz/`）；每次 `cargo test` 还会用性质测试向它输入损坏的 bundle。

**检查做不到的事。** 哈希能发现损坏和不一致的修改，但它不是签名。任何能写 bundle 的人都能生成一个完全自洽的 bundle，
因此 bundle 的可信度取决于它的来源。来源不可信的 bundle 可能给出错误的预测；Tabsmith 也没有限制 bundle 文件的大小，
恶意 bundle 在校验时可能占用大量内存或 CPU。请只加载来源可信的 bundle，并把它们放在只有可信用户能写入的位置。

**输入数据。** CSV、Parquet 和 JSON 文件由 Polars 和 serde_json 解析。精心构造的输入文件导致的崩溃、卡死或无限制的内存占用值得报告，
即使根本原因在依赖库中。

**依赖。** CI 会对仓库里每个 `Cargo.lock` 运行 `cargo audit`。加载模型包的应用所用的锁文件（`examples/predict-app`、
`examples/wasm-predict`、`examples/embedded-predict`）没有未解决的 RustSec 公告。`tabsmith` 工具依赖 Polars 0.55（目前的最新版），
它用 zstd 0.13 读取 zstd 压缩的 Parquet 数据页和 `.zst` CSV 文件。针对这个版本有两条公告原则上适用：RUSTSEC-2026-0391
（复用解压上下文时，被放弃的流的数据会泄漏到下一个流）和 RUSTSEC-2026-0403（旧格式帧流式解压失败时暴露未初始化内存）。
在 Polars 升级到 zstd 0.14 之前，请不要用 `tabsmith` 在来源不可信的 Parquet 或 `.zst` 文件上训练或预测；同一次升级也会消除
cargo-audit 对 zstd-safe 7.3 报告的“不健全”警告。`tabsmith-runtime` 不使用 Polars，不受影响。`.cargo/audit.toml` 列出了这些例外及其理由，以及另外几个被标记的库（`zip` 0.6、`quick-xml` 0.39）：
它们出现在 `Cargo.lock` 中，但从不会被编译。

**`tabsmith serve`。** HTTP 服务没有身份认证、没有 TLS，也没有请求超时：慢速发送请求体的客户端会一直占着一个工作线程（默认 4 个），
直到发完为止。除非用 `--host` 另行指定，它只监听 `127.0.0.1`。要接受其他机器的连接，请在前面放一个提供 TLS、认证和超时的反向代理。
`POST /reload` 只在加了 `--allow-reload` 时存在，而且只能加载和当前模型包在同一目录下的模型包；目录之外的路径与不存在的路径得到相同的回答，
客户端无法借此探测服务器的文件系统。

**发布下载。** `install.sh` 和 `install.ps1` 通过 HTTPS 下载压缩包，并用同一 GitHub Release 中发布的 `.sha256` 文件校验。
这能防止下载损坏，但不能防止发布本身被篡改；你也可以随时用 `cargo install tabsmith-cli --locked` 从源码构建。
