//! Compiles a Tabsmith bundle into Rust code while the application builds.
//!
//! ```
//! tabsmith_macros::model!(defects, "../../fixtures/bundles/binary-mlp");
//!
//! let reading = defects::Input {
//!     line: Some(defects::Line::B),
//!     temp_mean: Some(71.4),
//!     ..defects::EXAMPLES[0]
//! };
//! let p = defects::predict(&reading)?;
//! println!("{} (p = {:.3})", p.label, p.probability);
//! # Ok::<(), tabsmith_embedded::InputError>(())
//! ```
//!
//! `model!(name, "path/to/bundle")` reads the bundle when the crate compiles (the path
//! is relative to the crate's `Cargo.toml`) and verifies it exactly as
//! `tabsmith_runtime::Predictor::load` does: every file's SHA-256, the format versions,
//! the consistency of schemas, transform and model, and the self-check. A bundle that
//! fails is a compile error. Changing a file of the bundle makes cargo recompile.
//!
//! It then generates a module `name` with:
//! - **`Input`**, a struct with one typed field per input the model uses:
//!   `Option<f64>` for numbers (missing values are imputed as in training), an enum of
//!   the training categories for text, or `bool`. A field
//!   the model does not know, a misspelled name or a text where a category belongs
//!   does not compile.
//! - **`predict(&Input) -> Result<Prediction, InputError>`**. `Prediction` holds a
//!   `Label` and the probability (binary), a `Class` and every class's probability
//!   (multiclass), or the value (regression).
//! - **`features`, `outputs`, `decode`**, the three stages of `predict`, and
//!   **`outputs_batch`**, the raw outputs of many rows of features.
//! - **`Input::from_values`**, which builds an input from raw [`FieldValue`]s with the
//!   rules the runtime applies to JSON records.
//! - **`EXAMPLES` and `self_check()`**: the bundle's example records, and a check that
//!   they predict, bit for bit, what the runtime predicted for them at compile time. The
//!   compiler already ran the same check while the crate compiled; `self_check` repeats
//!   it on the device.
//! - **What the compiler knows about the model.** `WORST_CASE` bounds the arithmetic
//!   of one prediction (no loop depends on the data). `OUTPUT_BOUNDS` bounds the raw
//!   outputs over every possible input, and `output_bounds(&lo, &hi)` over a box of
//!   feature values (`feature_box(sd)` builds one): the smallest and largest reachable
//!   leaf of every tree, or interval arithmetic through the layers widened for
//!   rounding. Both are `const` and evaluated by rustc. A binary model also gets
//!   `PROBABILITY_BOUNDS`, a regression `VALUE_BOUNDS`.
//! - **`Model`**, a unit struct implementing the `tabsmith_embedded::Model` trait, which
//!   the runtime's `Predictor` implements too.
//!
//! The generated code is `no_std`: the model's weights and trees are constants and
//! inference allocates nothing. A network calls the kernels of `tabsmith-embedded`, the
//! same functions the runtime calls. Each tree becomes a function of nested `if`s, with
//! the arithmetic of the kernels:
//!
//! ```ignore
//! const fn tree_0(x: &[f32; N_FEATURES]) -> f32 {
//!     if x[24] <= 0.2705937f32 {
//!         if x[23] <= 1.4444073f32 { /* ... */ } else { /* ... */ }
//!     } else {
//!         if x[21] <= -0.09460988f32 { /* ... */ } else { /* ... */ }
//!     }
//! }
//! ```
//!
//! So predictions are identical to the runtime's, and the same on every platform. The
//! crate that invokes `model!` must depend on `tabsmith-embedded`.
//!
//! On real models, code is 2 to 2.8 times faster than walking the trees' arrays, and it
//! takes a third less flash on a Cortex-M4F. Code takes longer to compile, so an ensemble
//! of more than 32,768 nodes stays arrays unless the invocation says `tree_code`:
//! `model!(name, "path", tree_code)`. The numbers are in `reports/inference/` of the
//! repository. `tree_arrays` keeps the arrays at any size, and a tree more than 64 levels
//! deep always does. A network ignores both options, so an invocation keeps compiling when
//! retraining picks another model kind.
//!
//! **The compiler can run the model.** `features`, `outputs`, `decode` and `predict` are
//! `const fn`s for every kind of model and either form of trees, and so are the kernels
//! they call, `exp` included. The compiler runs the bundle's examples through `predict`
//! while the crate compiles and compares the bits with what `tabsmith-runtime` predicted
//! for them: a model the compiler computes differently does not compile. An application
//! can predict in its own constants:
//!
//! ```
//! tabsmith_macros::model!(defects, "../../fixtures/bundles/binary-mlp");
//!
//! // Computed while this compiles: the network, its features and the logistic function.
//! const READING: defects::Input = defects::Input { temp_mean: Some(71.4), ..defects::EXAMPLES[0] };
//! const RISK: f64 = match defects::predict(&READING) {
//!     Ok(p) => p.probability,
//!     Err(_) => panic!("a reading the model refuses"),
//! };
//! const _: () = assert!(0.0 < RISK && RISK < 1.0);
//! # assert_eq!(RISK.to_bits(), defects::predict(&READING).unwrap().probability.to_bits());
//! ```
//!
//! **Rules the model must keep.** `rule(conditions => claim)` states what every input of a
//! region must get, and the module does not compile unless that is so:
//!
//! ```
//! tabsmith_macros::model!(
//!     defects,
//!     "../../fixtures/bundles/binary-gbdt",
//!     // Every reading with under 1,000 hours in service, the other numbers within two
//!     // standard deviations of their training means and any categories, stays at or below 0.82.
//!     rule(hours_in_service in 0..=1000, _ within 2 sd => probability <= 0.82),
//! );
//! assert_eq!(defects::RULES, ["hours_in_service in 0..=1000, _ within 2 sd => probability <= 0.82"]);
//! ```
//!
//! A condition limits one input field: `field in a..=b`, `field <= b`, `field >= a`,
//! `field == value`, `field in [value, …]` or `field within k sd`, and `_ within k sd` limits
//! every numeric field no other condition names. The claim is `probability <= p` or
//! `probability >= p` (binary), `label == "…"` or `label != "…"` (classification), `value <= v`
//! or `value >= v` (regression). The macro audits each rule as `tabsmith audit` does
//! (`tabsmith_runtime::audit`): a search over boxes of feature values, exact for trees, with
//! interval bounds for networks, in which a category's slots stay one-hot. A rule that fails is
//! a compile error that shows an input breaking it, as `tabsmith-runtime` predicts it; a rule
//! the search cannot settle within its budget of a million boxes is one too. A rule that holds
//! comes with a proof, a tree of splits whose every box the model's bounds close, and the
//! module makes the compiler check the proof again: it walks the splits with the module's own
//! `output_bounds` and the claim, in its own floating point. The compiler takes about 3 µs
//! per tree node (or network weight) per box, so a proof above a million of those, about 3 s,
//! is checked when the macro expands instead, and the module's documentation says which.
//! `RULES` lists the rules.
//!
//! "Every input" means every input `predict` accepts, realistic or not: any number for a
//! numeric field without a condition. A tree ensemble answers for those exactly; a network
//! extrapolates, so its rules need ranges such as `_ within 3 sd` to hold.
//!
//! `model!(name, "path", kani)` also makes the module carry its own proofs for the
//! [Kani](https://github.com/model-checking/kani) model checker: under `cfg(kani)`, the
//! input types derive `kani::Arbitrary` and a module `name::proofs` holds one harness
//! per stage of `predict` (features, outputs, decode), each proving that the stage never
//! panics for any value of its argument, with loop bounds taken from the model's shape.
//! Trees get two more harnesses per tree, which together make a chain from the
//! runtime to the bounds:
//! - `name_tree_<t>_is_its_walk` (generated code only): the code of tree `t` gives the
//!   same bits as the walk through its arrays, for every row of features, NaN and
//!   infinities included. The runtime walks the same arrays with the same kernel, and
//!   `outputs` adds the trees in the kernel's order, so the compiled model predicts what
//!   the runtime predicts for every input, not only for `EXAMPLES`.
//! - `name_tree_<t>_stays_in_its_range`: every row of every box of feature values reaches
//!   a leaf of tree `t` inside the range that `output_bounds` adds up for that box.
//!   Rounding never puts two sums in the wrong order (tabsmith-embedded proves that with
//!   Kani too), so `output_bounds` and `OUTPUT_BOUNDS` hold for every row.
//!
//! Trees kept as arrays get their code inside the proofs only, as the link between the
//! walk and its range. `cargo kani` runs every harness; nothing is written by hand. They
//! cover `predict`, not `outputs_batch`. A range proof takes seconds at any size, while a
//! walk proof grows with the whole ensemble (about 7 s per tree at 854 nodes, 32 s at
//! 3,610), so a large model is better proved a subset at a time (`cargo kani --harness`).
//!
//! These do not compile:
//!
//! ```compile_fail,E0308
//! tabsmith_macros::model!(defects, "../../fixtures/bundles/binary-mlp");
//! // text where the model takes one of its categories
//! let reading = defects::Input { line: Some("B"), ..defects::EXAMPLES[0] };
//! ```
//!
//! ```compile_fail,E0560
//! tabsmith_macros::model!(defects, "../../fixtures/bundles/binary-mlp");
//! // a field the model does not have
//! let reading = defects::Input { pressure_max: Some(1.0), ..defects::EXAMPLES[0] };
//! ```
//!
//! ```compile_fail,E0599
//! tabsmith_macros::model!(defects, "../../fixtures/bundles/binary-mlp");
//! // a label the model does not predict
//! let positive = defects::Label::Maybe;
//! ```
//!
//! ```compile_fail
//! // no bundle at this path
//! tabsmith_macros::model!(defects, "../../fixtures/bundles/no-such-bundle");
//! ```
//!
//! ```compile_fail
//! // an option that does not exist
//! tabsmith_macros::model!(defects, "../../fixtures/bundles/binary-gbdt", tree_cod);
//! ```
//!
//! ```compile_fail
//! // two forms of the same trees
//! tabsmith_macros::model!(defects, "../../fixtures/bundles/binary-gbdt", tree_code, tree_arrays);
//! ```
//!
//! ```compile_fail
//! // a rule the model breaks: the error shows an input of its region and its prediction
//! tabsmith_macros::model!(defects, "../../fixtures/bundles/binary-gbdt", rule(_ within 2 sd => probability <= 0.8));
//! ```
//!
//! ```compile_fail
//! // a rule about a field the model does not have
//! tabsmith_macros::model!(defects, "../../fixtures/bundles/binary-gbdt", rule(pressure_max <= 3 => probability <= 0.9));
//! ```
//!
//! ```compile_fail,E0080
//! tabsmith_macros::model!(defects, "../../fixtures/bundles/binary-mlp");
//! // a reading the model refuses, refused while the crate compiles
//! const RISK: f64 = match defects::predict(&defects::Input { temp_mean: Some(f64::INFINITY), ..defects::EXAMPLES[0] }) {
//!     Ok(p) => p.probability,
//!     Err(_) => panic!("a reading the model refuses"),
//! };
//! ```
//!
//! [`FieldValue`]: https://docs.rs/tabsmith-embedded/latest/tabsmith_embedded/enum.FieldValue.html

mod codegen;
mod names;

use proc_macro::TokenStream;

/// Compiles the bundle at `path` (relative to the invoking crate's `Cargo.toml`) into
/// a module `name`. See the crate documentation.
#[proc_macro]
pub fn model(input: TokenStream) -> TokenStream {
    let args = syn::parse_macro_input!(input as codegen::Args);
    match codegen::expand(&args) {
        Ok(tokens) => tokens.into(),
        Err(e) => e.to_compile_error().into(),
    }
}
