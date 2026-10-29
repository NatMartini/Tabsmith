# The Model Is a Rust Program: Compiled, Verified Tabular Models from a CSV to `no_std`

*Draft of a workshop-length paper (about 3,000 words). Every number below is reproduced
by a command or a report in the repository; the sections that cite `reports/` are the
evidence, this text is the argument.*

## Abstract

Machine-learning models usually live outside the type system of the program that uses
them: a file of weights, loaded at run time, whose input contract is a convention. We
present Tabsmith, a tool that trains a tabular model from a CSV file and compiles the
result into Rust code while the application builds. The compiled model has typed
inputs (a misspelled field or an unseen category does not compile), allocates nothing,
needs no standard library, and predicts bit for bit what the training tool's runtime
predicts, on every platform. Because the model is a Rust program made of constants,
the compiler can say more about it than a file of weights allows: `rustc`'s constant
evaluator bounds its outputs over every input or over a box of plausible inputs,
checks the monotone constraints its trees were trained under, counts the arithmetic of
one prediction, and the Kani model checker proves, from harnesses the macro generates,
that prediction never panics for any input. Around the compiled model, Tabsmith keeps
the discipline that makes the model worth compiling: leakage-free splits, one
preprocessing implementation, a test set used once, exact resume, conformal intervals,
and bundles that carry their own evidence. On a 10,000-row problem the trained models
match scikit-learn's quality; a new process predicts after 1 ms instead of 612 ms, in
12 MB instead of 161 MB, and the application ships 1.7 MB instead of 354 MB.

## 1. Introduction

A trained model is a function from a record to a prediction. In practice it is stored
as data (a pickle, a safetensors file, an ONNX graph) and interpreted at run time by
code that neither the compiler nor the type checker has seen. Three things follow.
The input contract is informal: the names, types and allowed categories of the fields
are a convention between the training script and the serving code, and a mismatch
appears at run time, often as a silently wrong prediction rather than an error. The
numerical behaviour is unverifiable: whether the interpreter can panic, or what range
the outputs can take, is not a question the toolchain can answer. And the deployment
drags a runtime along: a Python environment of hundreds of megabytes, or at least an
interpreter for the file format.

Rust programs that need a model today reach for a binding to Python (PyO3), an ONNX
runtime, or a Rust inference library. All three keep the model outside the type system.
We take the other road: the model becomes Rust code. A procedural macro reads a trained
bundle at compile time, verifies it exactly as the runtime would, and expands to a
module whose input is a struct with one typed field per model input, whose weights and
trees are constants, and whose `predict` calls the same numeric kernels the runtime
calls. The model is then subject to everything the Rust toolchain can do with a
program: it type-checks, it runs without `std` or an allocator on a microcontroller,
its constants can be evaluated at compile time, and it can be model-checked.

**Contributions.**

1. `model!`, a macro that compiles a verified model bundle into a typed `no_std` Rust
   module, bit-identical to the interpreting runtime on Linux, macOS, Windows,
   WebAssembly and Cortex-M (Section 4.1).
2. Compile-time properties of the compiled model, computed by the Rust constant
   evaluator: sound output bounds over any box of inputs (exact for trees, interval
   arithmetic widened for rounding for networks), a structural check of monotone
   constraints, and the worst-case arithmetic of a prediction (Section 4.2).
3. Generated Kani harnesses, one per stage of prediction, with loop bounds taken from
   the model's shape, so that every compiled model carries a machine-checked proof
   that prediction never panics (Section 4.3).
4. The workflow that produces models worth compiling: leakage-free and chronological
   splits, one preprocessing implementation, a test-set ledger, exact resume,
   deterministic trees that are the same on every operating system, conformal
   intervals and permutation importance carried in the bundle (Section 3).
5. Measurements against the standard Python stack on the same data, and the honest
   list of what is not done (Sections 5 and 7).

Tabsmith is open source (MIT or Apache-2.0) at <https://github.com/NatMartini/Tabsmith>.

## 2. Background

**Tabular learning.** Gradient-boosted trees and small networks remain the strongest
practical models for tables [8]. Both are cheap to evaluate: a prediction is a few
hundred comparisons or a few thousand multiply-adds. What costs money in production is
everything around the arithmetic: the Python process, the input checks, the
preprocessing that must match training exactly.

**Rust.** Rust gives memory safety without a garbage collector, a strong static type
system, `const` evaluation of a growing subset of the language (floating-point
arithmetic since 1.82), procedural macros that run arbitrary code at compile time, and
first-class support for targets without an operating system (`no_std`).

**Kani** [13] is a bit-precise model checker for Rust built on CBMC. A harness is a
function marked `#[kani::proof]` whose inputs are symbolic (`kani::any()`); Kani proves
that no assertion, panic, overflow or memory error can occur for any value of them,
within declared loop bounds.

**Split conformal prediction** [15] turns any point predictor into a set predictor with
a finite-sample coverage guarantee: on held-out calibration rows one computes a
nonconformity score (for example the absolute residual) and its (1 − α) quantile; the
interval "prediction ± quantile" then covers the truth with probability at least
1 − α on exchangeable rows.

## 3. The workflow around the model

A model is only worth compiling if it was trained honestly. Tabsmith's command line
does one job, `tabsmith train data.csv --target y`, and the job has fixed stages
(Figure 1 of the README): a snapshot of the data; a split that is stratified, grouped by
an entity column (`--group device_id`, so that no device is in both training and test),
or chronological (`--time`); preprocessing statistics fitted on the training rows only;
three candidates trained on the same features (a linear model, a small multilayer
perceptron, both with Burn [5], and Tabsmith's own histogram gradient-boosted trees);
a comparison on the validation split; one evaluation of the chosen model on the test
split; and a bundle.

Four design decisions matter for what follows (the full list is `docs/design-decisions.md`).

*One preprocessing implementation.* The fitted transform is applied by one function in
the runtime crate, at training, at evaluation, at command-line prediction and inside the
application. There is no second implementation to drift. Against scikit-learn's
`ColumnTransformer` on the same data the encoded features agree to 2.3 × 10⁻⁷
(`reports/implementation-acceptance.md`).

*The test split is used once.* Evaluating a model on the test rows is recorded in a
ledger keyed by the data version and the model's weight hash. A second model of the same
data version is refused unless the user allows it, and the reuse is then recorded in the
evaluation. When new data arrives, the previous bundle and the new model are scored on
the same rows that neither has seen.

*Determinism is a contract.* Randomness comes from a versioned PRNG; CPU training is
single-threaded; trees sum in a fixed order; checkpoints hold the optimizer state, so
that resuming after a crash gives the same weights, bit for bit, as an uninterrupted
run. The trees' training takes its transcendental functions (`exp`, `log`) from a pure
Rust `libm` [14] rather than the platform C library, so the same data gives the same
trees on every operating system (Section 5.4).

*Bundles carry their evidence.* A bundle is an immutable directory: schemas, the
transform, the model, the evaluation record, synthetic example records with the
predictions they must reproduce, and, when the project's data is available at build
time, a conformal calibration and the permutation importance of every input. Every
file is hashed; the bundle id is a hash of the files and of the manifest's claims, so a
claim cannot be edited to hide a change. The loader verifies all of it before loading
anything, and replays the examples (the self-check) on the target machine.

## 4. The model as a Rust program

### 4.1 Compilation

```rust
tabsmith_macros::model!(defects, "bundles/run-003");   // verified at compile time

let reading = defects::Input {
    line: Some(defects::Line::B),       // a misspelled field or category does not compile
    hours_in_service: Some(12_000.0),   // None is imputed as in training
    ..defects::EXAMPLES[0]
};
let p = defects::predict(&reading)?;    // p.label, p.probability
```

The macro loads the bundle with the runtime's own loader (every hash, every format
version, the self-check), so a damaged or tampered bundle is a compile error, and
`include_bytes!` makes cargo rebuild when a file of the bundle changes. It then
generates a module. `Input` has one field per feature the model uses: `Option<f64>` for
a number (a missing value is imputed as in training), an enum of the categories seen
in training for text, or `bool`. The variant for a category never seen in training
exists only when the bundle's policy accepts such values. `Prediction` holds a `Label`
and a probability, a `Class` and every class's probability, or a value. The three stages
of prediction are separate functions, `features`, `outputs` and `decode`, and `predict`
is their composition. The weights, or the trees' node arrays, are `const` arrays. For
trees, constant assertions check the structure at compile time: every split reads a real
feature and points forward inside its own tree, so a walk can neither leave the tree nor
loop.

The generated code calls the kernels of a small `no_std` crate (`tabsmith-embedded`,
about 200 lines of arithmetic) that the runtime calls too. Each kernel fixes its order
of operations, and `exp` comes from `libm`. That is why a compiled model and the runtime
produce identical bits, and why Linux, macOS, Windows, WebAssembly and a Cortex-M4F do:
the platform C libraries' `exp` differ in the last bit for 0.07 % of `f32` inputs between
glibc and the Microsoft C runtime (`crates/tabsmith-embedded/README.md`). The equivalence
is tested on seven bundles of every task and model kind with 3,000 random records each,
including missing values, unseen categories, NaN, infinities and wrong types, and a
portability test compares the hash of 20,000 predictions per bundle with the value
recorded on Windows; CI runs it on Linux and macOS. A compiled prediction takes 0.13 to
0.59 µs on a desktop CPU, 7 to 27 times less than the runtime's from a dynamic record.

### 4.2 What the compiler can say about the model

A model made of constants can be analysed by the compiler. The generated module exposes
three results of such analysis, all computed by `rustc`'s constant evaluator from the
generated arrays, so they cost nothing at run time and cannot drift from the weights.

*Output bounds.* `output_bounds(&lo, &hi)` returns, for every output, an interval that
the raw outputs cannot leave for any row of features with `lo[j] ≤ x[j] ≤ hi[j]`;
`OUTPUT_BOUNDS` is the unbounded box. For trees the analysis is exact in a precise
sense: each tree contributes the smallest and the largest leaf that the box can reach
(a one-pass reachability computation over the node arrays), and the contributions are
summed in the order inference sums them. Rounding to nearest is monotone, so the
rounded sum of any row's leaves lies between the rounded sums of the per-tree minima and
maxima: the bound is sound with no error margin, though not necessarily attained. For a
dense network, interval arithmetic runs through the layers and each layer's result is
widened by the worst rounding error of the kernel, (n + 3) unit roundoffs of the sum of
the magnitudes of the products and the bias, which covers every summation order; ReLU
is monotone and exact. `feature_box(sd)` builds the box "every numeric feature within
`sd` standard deviations of its training mean, every category slot between 0 and 1". A
binary tree model therefore knows `PROBABILITY_BOUNDS` before it runs; the embedded
example asserts at compile time that its defect probability is strictly inside (0, 1)
and that its network's logit is finite for plausible readings. The macro computes the
same bounds at expansion time with the same `const fn`s and emits an assertion that the
constant evaluator's result agrees bit for bit.

*Monotonicity.* A tree ensemble trained under a monotone constraint keeps it by
construction (Section 3; the method is XGBoost's: a split on the constrained feature is
taken only when the lesser side's leaf weight is at most the greater side's, and the two
subtrees' weights are then bounded at the midpoint). The constraint is recorded in the
bundle. The generated module checks the structural property at compile time, one
backward pass over the node arrays: at every split on a constrained feature, every leaf
of the lesser subtree is at most every leaf of the greater one. Each tree is then
monotone in the feature, and so is their sum, again because rounding is monotone. The
runtime's loader performs the same check, so a bundle cannot claim a constraint it does
not keep.

*Cost.* `WORST_CASE` counts the comparisons, multiply-adds, additions and exponentials
of one prediction. No loop bound of the kernels depends on the data, so the count is a
worst case for trees (the sum of the trees' depths) and exact for a network.

### 4.3 Proofs the macro writes

With the `kani` option, the module also carries a `proofs` submodule, compiled only
under `cfg(kani)`. It holds one harness per stage of prediction: `features` for any
`Input` (the input types derive `kani::Arbitrary`), the tree walk for any tree index and
any feature values (asserting that it ends at a leaf), or the dense layers for any
feature values, and `decode` for any raw outputs. Proving the stages apart keeps each
proof small: no array index depends on the network's arithmetic, so Kani slices the
arithmetic away, and `exp` is checked once. The loop bounds are derived from the model:
the deepest tree plus two, the widest layer plus two, the number of missing-value
tokens plus two. Nothing is written by hand, so every model compiled with the option
gets the same proofs.

For the two models of the embedded example (14 trees with 854 nodes; a 35-32-16-1
network), the six harnesses pass with Kani 0.68 (CBMC 6.11) in 72 s together, each
under 2 GB of memory (`reports/kani/`). Two findings on the way are recorded there: a
single proof of all 14 tree walks needs more than 14 GB of memory (every array index
depends on the comparisons before it, so nothing can be sliced away), which is why
`outputs` is proved one walk at a time; and CBMC's floating-point NaN check reports a
false positive in `libm`'s `exp(−∞)`, which divides a finite number by an infinity
(IEEE 754 defines the quotient as 0), so the proofs run with that check off and the
Rust panic checks on.

What the proofs say, exactly: for every value of every input type, including NaN,
infinities, every category and every missing value, prediction either returns a
prediction or returns an `InputError`; it never panics, never indexes out of bounds and
never overflows. They say nothing about accuracy. The bounds of Section 4.2 say
something about values; the conformal calibration of Section 3 says something about
error. The three are different claims, and the bundle keeps them apart.

## 5. Evaluation

### 5.1 Parity with scikit-learn

On the project's reference data, preprocessing agrees with scikit-learn's to 2.3 × 10⁻⁷,
the linear model reaches an objective within 0.10 % of scikit-learn's optimum for the
same objective, and the metrics agree to 10⁻¹²; CI guards these with frozen goldens
(`fixtures/reference/`, `reports/implementation-acceptance.md`).

### 5.2 Against the Python stack on the same problem

The README benchmark trains Tabsmith's three candidates and scikit-learn's
`LogisticRegression`, `MLPClassifier` and `HistGradientBoostingClassifier` on the same
10,000 rows, split, features and settings, on one machine
(`reports/benchmarks/2026-10-27-linux-x86_64-v2/`). Quality is on par: the trees match
`HistGradientBoostingClassifier` (ROC AUC 0.8188 against 0.8184) and the small MLP did
better than `MLPClassifier` here (0.8250 against 0.7928). Training takes about a second
or less on both sides. The difference is in shipping: a single-record prediction in an
application takes about 2.5 µs instead of 2–3 ms; a new process predicts after 1 ms
instead of 612 ms, in 12 MB instead of 161 MB; the application is a 1.7 MB binary
instead of a 354 MB Python environment. On three public datasets, one per task (UCI
Adult, California Housing, UCI Covertype; `reports/datasets/`), every candidate matches
its scikit-learn equivalent on the same rows and split to within the test split's
resolution; the trees agree to the third decimal. The clearest loss is there too: the
trees train 17 to 21 times slower than `HistGradientBoosting` on the larger problems,
because each leaf's histogram is rebuilt from its rows and the sums keep a fixed order
(Section 5.5). LightGBM and XGBoost, the faster trainers on large data, were not
measured.

### 5.3 Robustness

The bundle loader was fuzzed with libFuzzer for 5.3 million inputs without a crash
(`reports/fuzz/`). Every workspace crate forbids `unsafe`. Property tests cover
predictions, metrics, splits, epoch orders and damaged bundles; 179 automated tests and
7 doctests run on Linux, macOS and Windows in CI, including process-level tests that
drive the real binary and a crash-and-resume test that aborts the process.

### 5.4 The same trees on every operating system

Inference is bit-identical across platforms by construction (Section 4.1). For training
we claim it for the trees only. Burn's CPU backend computes the networks' matrix
products with platform-dependent code, so the networks' weights are reproducible across
processes on one machine but not across operating systems; the trees are plain Rust
whose only transcendental functions are `exp` and `log`, taken from `libm`. The
experiment trains the three candidates on the same 3,000 synthetic rows with the same
settings on Windows 11 and on Linux (WSL 2) and compares the SHA-256 of the weight files
(`reports/reproducibility/`). The data file is identical on both sides. The trees' weight
file has the same SHA-256 on Windows and on Linux, and all 39 epoch records of the
training history are identical to every digit; the linear model's and the network's
weight files differ, and their validation losses differ by up to 1.9 × 10⁻⁷ and
4.8 × 10⁻⁷. Both sides ran on the same x86-64 processor, so the experiment shows
independence from the operating system and its C library, not from the CPU
architecture, which is expected for the same reason but was not measured.

### 5.5 Scale

The trees build their bins and histograms and apply each grown tree on every core,
with sums in fixed chunks of 16,384 rows merged in order, so the result does not
depend on the core count. On 1,000,000 rows (a 172 MB CSV, 35 features) the whole
`tabsmith train` runs in 42 s on a 16-core desktop with a 1.6 GB working set; the 148
boosting rounds take 29.6 s, 0.2 s per round over 698,621 training rows, and the
validation ROC AUC is 0.840 (`reports/scaling/`). The data is held in memory, so ten
million rows would need about 16 GB; out-of-core training is future work.

## 6. Related work

Rust inference libraries (tract, candle's inference paths, ONNX runtimes with Rust
bindings) interpret a model file; they keep the model's contract and weights outside the
type system, which is the gap this work closes. Tools that generate C code from trees
(for example m2cgen or treelite) produce source, but the generated code is not analysed
by the compiler beyond type checking, and C offers neither the constant evaluation nor
the model checking used here. Formal verification of neural networks (Reluplex and its
successors) targets properties of the function; our proofs target the program: absence
of panics for every input, with value bounds from a sound static analysis rather than
an SMT query. Conformal prediction [15] is used as published; our contribution is only
to carry the calibration into the compiled model. Burn [5] and CubeCL provided the
training backends; three findings about their GPU paths (TF32 without an opt-out,
autotune nondeterminism across processes, a device loss on an integrated GPU) are
documented in `docs/design-decisions.md` and drafted as upstream reports.

## 7. Limitations

The proofs cover the two example models, not every bundle, until a user runs `cargo
kani` on their own compiled model; the macro generates the harnesses, the user runs
them. The output bounds of a network over all inputs are infinite; the useful bounds
need a box, and the box is the user's claim about the inputs. Conformal coverage is
calibrated on the validation split, which also selected the model, so it is approximate;
the exact guarantee needs rows the model never saw. The trees have no native categorical
splits and no sampling, and they train 17 to 21 times slower than scikit-learn's
`HistGradientBoosting` on the larger public datasets (each leaf's histogram is rebuilt
from its rows, and the sums keep an order that makes the trees the same on any core
count); LightGBM and XGBoost train faster still on large data. Training of the
networks is reproducible across processes, not across operating systems. GPU training
is certified only on the hardware it was run on. There has been no user study: every
number here is about the machine, not about the people who would use it. Per-prediction
attributions (TreeSHAP) are not implemented, because the tree format does not record
node coverage.

## 8. Conclusion

A model that is a Rust program is not only faster and smaller to ship. It is a program:
the type checker enforces its contract, the constant evaluator bounds its values and
checks its constraints, and a model checker proves that it cannot fail. None of this
needs a new language or a new verifier; it needs the model to be written in the
language's own terms, constants and functions, and a toolchain whose compile-time
evaluation and verification are good enough to reason about them. Rust's are.

## References

1. The Tabsmith repository: <https://github.com/NatMartini/Tabsmith>. The technical
   report (`docs/technical-report.md`) and the design decisions
   (`docs/design-decisions.md`) hold the full account; `reports/` the evidence.
5. N. Dilokthanakul et al. Burn: a flexible and comprehensive deep learning framework
   in Rust. <https://burn.dev>.
8. L. Grinsztajn, E. Oyallon, G. Varoquaux. Why do tree-based models still outperform
   deep learning on typical tabular data? *NeurIPS Datasets and Benchmarks*, 2022.
13. The Kani Rust Verifier, a bit-precise model checker for Rust.
    <https://github.com/model-checking/kani>
14. libm: a port of musl's libm to Rust. <https://crates.io/crates/libm>
15. V. Vovk, A. Gammerman, G. Shafer. *Algorithmic Learning in a Random World*.
    Springer, 2005; A. N. Angelopoulos, S. Bates. A gentle introduction to conformal
    prediction and distribution-free uncertainty quantification, 2021.

*Reference numbers follow the technical report's list, so that the two documents can be
read together.*
