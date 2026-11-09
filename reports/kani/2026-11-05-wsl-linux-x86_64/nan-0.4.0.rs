// The check behind `nan-0.4.0.txt`: the bounds analysis of tabsmith-embedded 0.4.0 against its
// own documentation. To run it, make a crate with this file as src/lib.rs, put lines 34 to 145
// of crates/tabsmith-embedded/src/kernels.rs at tag v0.4.0 (the `Trees` type and its
// methods) into src/trees_v040.rs, add `[lints.rust] unexpected_cfgs = { level = "warn",
// check-cfg = ["cfg(kani)"] }` to its Cargo.toml, and run `cargo kani` and `cargo test`.

//! The bounds analysis of tabsmith-embedded 0.4.0 (`Trees`, copied verbatim from
//! `crates/tabsmith-embedded/src/kernels.rs` at v0.4.0), checked against its own
//! documentation: "A NaN in `lo` or `hi` lets the feature be NaN, which every split
//! sends right."
#![cfg_attr(not(test), no_std)]

mod v040 {
    include!("trees_v040.rs");
}
use v040::Trees;

/// One split on feature 0 at 10: a row with x <= 10 gets -1, any other row (NaN too) 1.
const ONE_SPLIT: Trees<'static> = Trees {
    roots: &[0],
    feature: &[0, -1, -1],
    threshold: &[10.0, 0.0, 0.0],
    left: &[1, 0, 0],
    right: &[2, 0, 0],
    value: &[0.0, -1.0, 1.0],
    base: &[0.0],
};

#[cfg(kani)]
mod proofs {
    use super::*;

    /// Any box, any row in it as the documentation defines it: the output lies inside
    /// the bounds of the box.
    #[kani::proof]
    #[kani::unwind(5)]
    fn the_documented_box_is_sound() {
        let (lo, hi, x): (f32, f32, f32) = (kani::any(), kani::any(), kani::any());
        kani::assume((lo <= x && x <= hi) || (x.is_nan() && (lo.is_nan() || hi.is_nan())));
        let (mut lo_out, mut hi_out, mut reachable) = ([0f32], [0f32], [false; 3]);
        ONE_SPLIT.bounds(&[lo], &[hi], &mut reachable, &mut lo_out, &mut hi_out);
        let mut out = [0f32];
        ONE_SPLIT.outputs(&[x], &mut out);
        assert!(lo_out[0] <= out[0] && out[0] <= hi_out[0]);
    }

    /// The same with NaN admitted only where `hi` is NaN.
    #[kani::proof]
    #[kani::unwind(5)]
    fn a_nan_upper_bound_is_sound() {
        let (lo, hi, x): (f32, f32, f32) = (kani::any(), kani::any(), kani::any());
        kani::assume((lo <= x && x <= hi) || (x.is_nan() && hi.is_nan()));
        let (mut lo_out, mut hi_out, mut reachable) = ([0f32], [0f32], [false; 3]);
        ONE_SPLIT.bounds(&[lo], &[hi], &mut reachable, &mut lo_out, &mut hi_out);
        let mut out = [0f32];
        ONE_SPLIT.outputs(&[x], &mut out);
        assert!(lo_out[0] <= out[0] && out[0] <= hi_out[0]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_nan_lower_bound_does_not_admit_nan() {
        let (mut lo_out, mut hi_out, mut reachable) = ([0f32], [0f32], [false; 3]);
        ONE_SPLIT.bounds(&[f32::NAN], &[5.0], &mut reachable, &mut lo_out, &mut hi_out);
        let mut out = [0f32];
        ONE_SPLIT.outputs(&[f32::NAN], &mut out);
        assert_eq!(out, [1.0]);
        assert_eq!((lo_out, hi_out), ([f32::INFINITY], [f32::NEG_INFINITY]));
    }
}
