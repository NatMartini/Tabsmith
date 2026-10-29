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
//! - **`features`, `outputs`, `decode`**, the three stages of `predict`.
//! - **`Input::from_values`**, which builds an input from raw [`FieldValue`]s with the
//!   rules the runtime applies to JSON records.
//! - **`EXAMPLES` and `self_check()`**: the bundle's example records, and a check that
//!   they predict, bit for bit, what the runtime predicted for them at compile time.
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
//! inference allocates nothing. It calls the kernels of `tabsmith-embedded`, the
//! same functions the runtime calls, so predictions are identical to the runtime's,
//! and the same on every platform. The crate that invokes `model!` must depend on
//! `tabsmith-embedded`.
//!
//! `model!(name, "path", kani)` also makes the module carry its own proofs for the
//! [Kani](https://github.com/model-checking/kani) model checker: under `cfg(kani)`, the
//! input types derive `kani::Arbitrary` and a module `name::proofs` holds one harness
//! per stage of `predict` (features, outputs, decode), each proving that the stage never
//! panics for any value of its argument, with loop bounds taken from the model's shape.
//! `cargo kani` runs them; nothing is written by hand.
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
