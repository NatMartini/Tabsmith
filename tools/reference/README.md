# Reference checks / 参考对照

English: RustPy is checked against scikit-learn (pinned versions are recorded in
the outputs).

* `make_goldens.py` freezes reference values for the fixture
  `fixtures/reference/defects_1500.csv` into `fixtures/reference/goldens.json`;
  `crates/rustpy-workflow/tests/reference_goldens.rs` checks every build against them
  (transform parity, metric parity, linear objective within 1% of the lbfgs optimum).
* `sklearn_reference.py` compares any exported run (see the script docstring).

Regenerate the goldens (only when the definition of a check changes on purpose):

```text
python -m venv .venv && .venv/bin/pip install scikit-learn==1.7.2 pandas numpy
rustpy init gp && cp fixtures/reference/defects_1500.csv gp/data/ && cd gp
rustpy data bind data/defects_1500.csv --target defect --group device_id --id unit_id
rustpy run --candidates linear && cd ..
cargo run -p rustpy-workflow --example export_reference -- gp run-001 export
.venv/bin/python tools/reference/make_goldens.py export fixtures/reference/goldens.json
```

Note: the goldens were generated after the first implementation (not frozen
before it, as plan task P1-03 asks); they guard against regressions from now on.

中文：RustPy 与 scikit-learn 做对照（输出中记录了固定的版本号）。`make_goldens.py` 把夹具
`fixtures/reference/defects_1500.csv` 的参考值冻结到 `fixtures/reference/goldens.json`，
`crates/rustpy-workflow/tests/reference_goldens.rs` 在每次构建时校验（变换一致、指标一致、
线性目标值不超过 lbfgs 最优值的 1%）。`sklearn_reference.py` 可对任意导出的 Run 做对照。
注意：这些参考值是在首次实现之后生成的（而不是计划 P1-03 要求的“实现前冻结”），从现在起用于防止回归。
