//! The arithmetic of Tabsmith models, in `no_std` Rust.
//!
//! `tabsmith-runtime` and the models that `tabsmith_macros::model!` compiles into
//! an application's code both call the [`kernels`] of this crate, so the two compute
//! the same bits. The kernels allocate nothing, contain no `unsafe` code, and are
//! `const fn`s, so a compiled model can predict while it compiles. Their one
//! transcendental function, [`kernels::exp`], is the algorithm of the `libm` crate,
//! written in this crate, rather than the platform's C library, whose last bits differ
//! between operating systems: a model predicts the same values on Linux, Windows, macOS,
//! WebAssembly, a microcontroller and in the compiler. The crate has no dependencies.
//!
//! The crate also holds the few types that compiled models share: [`FieldValue`] for
//! raw inputs, [`InputError`] for refused inputs, [`SelfCheckError`], [`Work`] for a
//! model's cost, and the [`Model`] trait that the runtime's `Predictor` and every
//! compiled model implement. [`rules`] checks proofs that a model keeps a rule over a
//! region of inputs, as `const fn`s too, so the compiler can check them.
#![cfg_attr(not(test), no_std)]
#![warn(missing_docs)]

mod input;
pub mod kernels;
pub mod rules;

pub use input::{FieldInfo, FieldKind, FieldValue, InputError, SelfCheckError};

/// Upper bounds on the arithmetic of one prediction, counted from a model's shape.
/// No loop of the kernels depends on the data, so these are the worst case and, for
/// a dense network, the exact count. Encoding the input is a few comparisons per field
/// on top.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Work {
    /// Comparisons: the splits a walk through every tree can take (the sum of the
    /// trees' depths) and the ReLUs of the dense layers.
    pub comparisons: usize,
    /// Multiply-adds of the dense layers.
    pub multiply_adds: usize,
    /// Additions on their own: a leaf value per tree, a bias per dense output.
    pub additions: usize,
    /// Exponentials in `decode`: one for a binary model, one per class for a
    /// multiclass model, none for a regression.
    pub exponentials: usize,
}

/// Split conformal prediction, calibrated when a bundle was built: on rows like the
/// calibration rows, a regression interval `value ± quantile` holds the true value,
/// and a classification set (the labels with probability at least `1 − quantile`)
/// holds the true label, with probability about `1 − alpha`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Conformal {
    /// The target miscoverage (0.1: 90 % coverage).
    pub alpha: f64,
    /// The half-width of the interval (regression), or the threshold on `1 − p` (classification).
    pub quantile: f64,
}

/// Something that predicts one input at a time.
///
/// `tabsmith_runtime::Predictor` implements it for `Record`s, and every module that
/// `tabsmith_macros::model!` generates implements it for its typed `Input` (as the
/// unit struct `name::Model`), so an application can be written once for both:
///
/// ```
/// use tabsmith_embedded::Model;
///
/// fn score<M: Model>(model: &M, input: &M::Input) -> Result<M::Prediction, M::Error> {
///     model.predict(input)
/// }
/// ```
pub trait Model {
    /// What the model takes.
    type Input: ?Sized;
    /// What a prediction holds.
    type Prediction;
    /// Why an input is refused.
    type Error;

    /// Predicts one input.
    fn predict(&self, input: &Self::Input) -> Result<Self::Prediction, Self::Error>;
}
