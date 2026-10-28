## What and why

<!-- What does this change, and why? Link the issue it addresses, if any. -->

## How it was checked

<!-- The commands you ran and their results. For claims about accuracy, speed or
memory, link the test or the reproducible command that shows them. -->

## Checklist

- [ ] `cargo test --workspace` passes
- [ ] `cargo build --manifest-path examples/predict-app/Cargo.toml --all-targets` passes (when `tabsmith-runtime` or `tabsmith-contracts` changed)
- [ ] New behaviour and fixed bugs are covered by tests
- [ ] Every claim in code, docs and this description is backed by a test, a reproducible command or a report
- [ ] Nothing describes GPU support as working; unavailable devices are still errors, never a silent CPU fallback
- [ ] User-facing docs updated in English and Chinese (or noted below what is missing)
- [ ] `CHANGELOG.md` updated under `## [Unreleased]` for user-visible changes
- [ ] No unrelated reformatting

<!-- 中文：请说明改了什么、为什么改、如何验证（命令及结果）。任何关于准确度、速度或内存的结论都需要有测试、可复现的命令或报告作为依据。 -->
