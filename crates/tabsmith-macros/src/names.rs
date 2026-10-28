//! Rust identifiers for names that come from data: field names, categories, labels.

use std::collections::HashSet;

use proc_macro2::{Ident, Span};

/// `UpperCamelCase` for a type or variant: "night" -> `Night`, "v1.2" -> `V1_2`,
/// "low-risk" -> `LowRisk`, "3" -> `V3`.
pub fn camel(s: &str) -> String {
    let mut out = String::new();
    let mut prev_digit = false;
    for seg in s.split(|c: char| !c.is_alphanumeric()).filter(|w| !w.is_empty()) {
        let mut chars = seg.chars();
        let first = chars.next().expect("segments are not empty");
        if prev_digit && first.is_ascii_digit() {
            out.push('_');
        }
        out.extend(first.to_uppercase());
        out.push_str(chars.as_str());
        prev_digit = seg.ends_with(|c: char| c.is_ascii_digit());
    }
    if out.is_empty() {
        out.push_str("Empty");
    }
    if out.starts_with(|c: char| c.is_ascii_digit()) {
        out.insert(0, 'V');
    }
    out
}

/// A field name as a Rust field identifier: non-alphanumeric characters become `_`.
pub fn snake(s: &str) -> String {
    let mut out: String = s.chars().map(|c| if c.is_alphanumeric() { c } else { '_' }).collect();
    if out.is_empty() || out.starts_with(|c: char| c.is_ascii_digit()) {
        out.insert_str(0, "f_");
    }
    out
}

/// Hands out identifiers that are valid Rust and unique among those already given.
pub struct Namer {
    taken: HashSet<String>,
}

impl Namer {
    pub fn new(reserved: &[&str]) -> Self {
        Self {
            taken: reserved.iter().map(|s| s.to_string()).collect(),
        }
    }

    /// `candidate`, or `candidate_2`, `candidate_3`, ... if it is taken. Names that are
    /// not identifiers (keywords, unusual characters) become raw identifiers or are
    /// rewritten with `_`.
    pub fn ident(&mut self, candidate: &str) -> Ident {
        let mut base = candidate.to_string();
        if matches!(base.as_str(), "None" | "Some" | "Ok" | "Err") {
            base.push('_');
        }
        let mut n = 1;
        loop {
            let name = if n == 1 { base.clone() } else { format!("{base}_{n}") };
            let id = to_ident(&name);
            if self.taken.insert(id.to_string()) {
                return id;
            }
            n += 1;
        }
    }
}

fn to_ident(name: &str) -> Ident {
    if syn::parse_str::<Ident>(name).is_ok() {
        return Ident::new(name, Span::call_site());
    }
    // A keyword: `type` becomes `r#type`. A few keywords cannot be raw.
    if !matches!(name, "self" | "Self" | "super" | "crate" | "_") && syn::parse_str::<Ident>(&format!("r#{name}")).is_ok() {
        return Ident::new_raw(name, Span::call_site());
    }
    let ascii: String = name.chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '_' }).collect();
    let fallback = if ascii.starts_with(|c: char| c.is_ascii_alphabetic()) {
        format!("{ascii}_")
    } else {
        format!("V{ascii}_")
    };
    Ident::new(&fallback, Span::call_site())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn data_names_become_identifiers() {
        assert_eq!(camel("night"), "Night");
        assert_eq!(camel("v1.2"), "V1_2");
        assert_eq!(camel("low-risk"), "LowRisk");
        assert_eq!(camel("3"), "V3");
        assert_eq!(camel("--"), "Empty");
        assert_eq!(snake("temp mean"), "temp_mean");
        assert_eq!(snake("2nd"), "f_2nd");
        let mut n = Namer::new(&["Input"]);
        assert_eq!(n.ident("Input").to_string(), "Input_2");
        assert_eq!(n.ident("None").to_string(), "None_");
        assert_eq!(n.ident("type").to_string(), "r#type");
        assert_eq!(n.ident("Self").to_string(), "Self_");
    }
}
