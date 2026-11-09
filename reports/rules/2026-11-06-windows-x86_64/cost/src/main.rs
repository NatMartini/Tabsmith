//! The compiler's check of one proof: `cargo check` of this crate with and without the rule.
//! The rule's proof has 3,244 boxes over the 3,610 nodes of the regression fixture's trees, so
//! it is above `RULE_CHECK_MAX_WORK` and the macro skips the compiler's check by default; the
//! measurement set that limit to `usize::MAX` for the duration.
tabsmith_macros::model!(m, "../../../../fixtures/bundles/regression-gbdt", rule(_ within 1 sd => value <= 30139.57));

fn main() {
    println!("{:?}", m::RULES);
}
