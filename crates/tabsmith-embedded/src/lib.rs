//! The arithmetic of Tabsmith models, in `no_std` Rust.
//!
//! `tabsmith-runtime` and the models that `tabsmith_macros::model!` compiles into
//! an application's code both call the [`kernels`] of this crate, so the two compute
//! the same bits. The kernels allocate nothing, contain no `unsafe` code and take
//! their transcendental functions from [`libm`] (pure Rust) instead of the platform's
//! C library, whose last bits differ between operating systems: a model predicts the
//! same values on Linux, Windows, macOS, WebAssembly and a microcontroller.
//!
//! The crate also holds the few types that compiled models share: [`FieldValue`] for
//! raw inputs, [`InputError`] for refused inputs and [`SelfCheckError`].
#![cfg_attr(not(test), no_std)]
#![warn(missing_docs)]

mod input;
pub mod kernels;

pub use input::{FieldInfo, FieldKind, FieldValue, InputError, SelfCheckError};
