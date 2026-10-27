# Security policy

[中文](#中文)

## Supported versions

RustPy is pre-1.0. Security fixes are made on the latest release only.

| Version | Supported |
|---|---|
| 0.1.x (latest) | yes |
| older | no |

## Reporting a vulnerability

Please **do not open a public issue** for a security problem. Report it privately
through GitHub's private vulnerability reporting:
[Report a vulnerability](https://github.com/__OWNER__/__REPO__/security/advisories/new)
(the "Security" tab of the repository).

Include the RustPy version (`rustpy --version`), your platform, what an attacker
controls (a data file, a bundle, a configuration, ...), the steps to reproduce
and the impact you expect. A minimal bundle or data file that triggers the
problem helps most.

This is a small project maintained on a best-effort basis. We aim to acknowledge
a report within 7 days, agree on a fix and a disclosure date with you, and credit
you in the advisory unless you prefer otherwise.

## What to expect from RustPy

**Bundles are data, not code.** A bundle is a directory of JSON files and one
`safetensors` weight file. Loading a bundle (`Predictor::load`, `rustpy predict`,
`rustpy bundle check`) parses those files; it never executes code from the
bundle, loads native libraries or unpickles objects. A way to make RustPy execute
code or touch files outside the bundle directory by loading a bundle is a
vulnerability; please report it.

**Integrity checks.** Before a bundle is used, every file listed in
`manifest.json` is checked against its recorded SHA-256 and size, unlisted files
and symbolic links are refused, the recorded bundle id must match the recomputed
one, and the schemas, transform and weights must be consistent. A damaged,
incomplete or partly edited bundle is refused as a whole. See
`docs/bundle-format.md` for the full list.

**What the checks do not do.** The hashes detect corruption and inconsistent
edits; they are not signatures. Anyone who can write a bundle can produce a
fully consistent one, so a bundle is only as trustworthy as its source. A bundle
from an untrusted source can make wrong predictions, and RustPy does not limit
the size of bundle files, so a hostile bundle can use a lot of memory or CPU
time while it is verified. Load bundles only from sources you trust, and keep
them where only trusted users can write.

**Input data.** CSV, Parquet and JSON files are parsed by Polars and serde_json.
Crashes, hangs or unbounded memory use on crafted input files are worth
reporting, even when the root cause is in a dependency.

**Release downloads.** `install.sh` and `install.ps1` download archives over
HTTPS and verify each against the `.sha256` file published in the same GitHub
release. This protects against corrupted downloads, not against a compromised
release; you can always build from source with `cargo install rustpy-cli --locked`.

---

## 中文

### 支持的版本

RustPy 尚未到 1.0，只在最新版本上修复安全问题（目前为 0.1.x）。

### 报告漏洞

请**不要在公开 issue 中报告安全问题**。请通过 GitHub 的私密漏洞报告功能提交：
[Report a vulnerability](https://github.com/__OWNER__/__REPO__/security/advisories/new)（仓库的 “Security” 标签页）。

请附上 RustPy 版本（`rustpy --version`）、平台、攻击者能控制什么（数据文件、bundle、配置等）、复现步骤和预期影响；
最好附一个能触发问题的最小 bundle 或数据文件。

本项目由维护者尽力维护。我们争取在 7 天内确认收到报告，与你商定修复和公开日期，并在公告中致谢（除非你不希望署名）。

### RustPy 的安全边界

**Bundle 是数据，不是代码。** bundle 是一个包含 JSON 文件和一个 `safetensors` 权重文件的目录。
加载 bundle（`Predictor::load`、`rustpy predict`、`rustpy bundle check`）只解析这些文件，
从不执行 bundle 中的代码、加载本地库或反序列化 pickle 对象。如果加载 bundle 能让 RustPy 执行代码或访问 bundle 目录之外的文件，
这就是漏洞，请报告。

**完整性检查。** 使用 bundle 之前，会对照 `manifest.json` 中记录的 SHA-256 和大小检查每个文件，
拒绝未列出的文件和符号链接，要求记录的 bundle id 与重新计算的一致，并检查 schema、变换和权重相互一致。
损坏、不完整或被部分修改的 bundle 会被整体拒绝。完整列表见 `docs/bundle-format.md`。

**检查做不到的事。** 哈希能发现损坏和不一致的修改，但它不是签名。任何能写 bundle 的人都能生成一个完全自洽的 bundle，
因此 bundle 的可信度取决于它的来源。来源不可信的 bundle 可能给出错误的预测；RustPy 也没有限制 bundle 文件的大小，
恶意 bundle 在校验时可能占用大量内存或 CPU。请只加载来源可信的 bundle，并把它们放在只有可信用户能写入的位置。

**输入数据。** CSV、Parquet 和 JSON 文件由 Polars 和 serde_json 解析。精心构造的输入文件导致的崩溃、卡死或无限制的内存占用值得报告，
即使根本原因在依赖库中。

**发布下载。** `install.sh` 和 `install.ps1` 通过 HTTPS 下载压缩包，并用同一 GitHub Release 中发布的 `.sha256` 文件校验。
这能防止下载损坏，但不能防止发布本身被篡改；你也可以随时用 `cargo install rustpy-cli --locked` 从源码构建。
