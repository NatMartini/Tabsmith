//! The text of a rule, `conditions => claim`; the syntax is documented on [`Rule`].

use std::fmt;

use tabsmith_contracts::{ErrorCode, Phase, Result, TabsmithError};

/// A rule: every input that meets the conditions gets a prediction that meets the claim. Its
/// text is `conditions => claim`:
///
/// ```text
/// age in 17..=25, capital_gain == 0, workclass in ["Private", "Local-gov"] => probability <= 0.9
/// ```
///
/// A condition limits one input field: `name in a..=b`, `name <= b`, `name >= a`,
/// `name == value`, `name in [value, …]`, or `name within k sd` (within `k` standard deviations
/// of the training mean). `_ within k sd` limits every numeric field the other conditions do
/// not name. A name is an identifier or a quoted string; a value is a number (`inf` and `-inf`
/// included), a quoted string, `true`, `false` or `null` (a missing value). The claim is
/// `probability <= p`, `probability >= p`, `label == "…"`, `label != "…"`, `value <= v` or
/// `value >= v`. Without conditions, `=> claim` is a rule about every input. Strict comparisons
/// do not exist, and `a..b` is refused for `a..=b`. [`Display`](fmt::Display) writes the
/// canonical text, which `parse` reads back to the same rule.
#[derive(Debug, Clone, PartialEq)]
pub struct Rule {
    /// The limits on the input fields; a field without one takes any value the model accepts.
    pub conditions: Vec<Condition>,
    /// What the prediction must satisfy.
    pub claim: Then,
}

/// A limit on one input field.
#[derive(Debug, Clone, PartialEq)]
pub struct Condition {
    /// The field's name, as in the bundle's `input.schema.json`.
    pub field: String,
    /// What the field's value must be.
    pub test: Test,
}

/// What a field's value must be.
#[derive(Debug, Clone, PartialEq)]
pub enum Test {
    /// A number in `[lo, hi]`.
    Range(f64, f64),
    /// A number at most this.
    AtMost(f64),
    /// A number at least this.
    AtLeast(f64),
    /// This value.
    Equals(Literal),
    /// One of these values.
    OneOf(Vec<Literal>),
    /// A number within this many standard deviations of the training mean. On the field `_`,
    /// every numeric field that no other condition names.
    Within(f64),
}

/// The field name that stands for every numeric field no other condition names.
pub const OTHER_FIELDS: &str = "_";

/// A value in a rule.
#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    /// A number.
    Number(f64),
    /// Text.
    Text(String),
    /// A boolean.
    Bool(bool),
    /// A missing value.
    Null,
}

/// What the prediction must satisfy.
#[derive(Debug, Clone, PartialEq)]
pub enum Then {
    /// Binary: the probability is at most this.
    ProbabilityAtMost(f64),
    /// Binary: the probability is at least this.
    ProbabilityAtLeast(f64),
    /// Classification: the label is this.
    Label(String),
    /// Classification: the label is not this.
    NotLabel(String),
    /// Regression: the value is at most this.
    ValueAtMost(f64),
    /// Regression: the value is at least this.
    ValueAtLeast(f64),
}

pub(crate) fn invalid(text: &str, message: impl Into<String>) -> TabsmithError {
    TabsmithError::new(ErrorCode::RuleInvalid, Phase::Audit, message)
        .field(text.to_string())
        .expected("conditions => claim, e.g. `age in 17..=25, sex == \"Female\" => probability <= 0.9`")
        .next("fix the rule: `tabsmith audit --help` shows every form a condition and a claim can take")
}

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Name(String),
    Text(String),
    Number(f64),
    Symbol(&'static str),
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Token::Name(s) => write!(f, "`{s}`"),
            Token::Text(s) => write!(f, "{s:?}"),
            Token::Number(v) => write!(f, "`{v}`"),
            Token::Symbol(s) => write!(f, "`{s}`"),
        }
    }
}

const SYMBOLS: [&str; 15] = ["..=", "=>", "==", "!=", "<=", ">=", "..", ",", "[", "]", "<", ">", "=", "-", "+"];

fn tokens(text: &str) -> Result<Vec<(usize, Token)>> {
    let bytes = text.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i];
        let column = text[..i].chars().count() + 1;
        if c.is_ascii_whitespace() {
            i += 1;
        } else if c == b'"' {
            let mut s = String::new();
            let mut chars = text[i + 1..].char_indices();
            let mut end = None;
            while let Some((k, ch)) = chars.next() {
                match ch {
                    '"' => {
                        end = Some(i + 1 + k + 1);
                        break;
                    }
                    '\\' => match chars.next() {
                        Some((_, e @ ('"' | '\\'))) => s.push(e),
                        _ => return Err(invalid(text, format!("column {column}: a string may only escape `\"` and `\\`"))),
                    },
                    other => s.push(other),
                }
            }
            let Some(end) = end else {
                return Err(invalid(text, format!("column {column}: the string never ends")));
            };
            out.push((column, Token::Text(s)));
            i = end;
        } else if c.is_ascii_digit() || (c == b'.' && bytes.get(i + 1).is_some_and(u8::is_ascii_digit)) {
            let start = i;
            while i < bytes.len() && bytes[i].is_ascii_digit() {
                i += 1;
            }
            // A decimal point, unless it starts `..` or `..=`.
            if i < bytes.len() && bytes[i] == b'.' && bytes.get(i + 1) != Some(&b'.') {
                i += 1;
                while i < bytes.len() && bytes[i].is_ascii_digit() {
                    i += 1;
                }
            }
            if i < bytes.len() && (bytes[i] == b'e' || bytes[i] == b'E') {
                let mut j = i + 1;
                if j < bytes.len() && (bytes[j] == b'+' || bytes[j] == b'-') {
                    j += 1;
                }
                if j < bytes.len() && bytes[j].is_ascii_digit() {
                    i = j;
                    while i < bytes.len() && bytes[i].is_ascii_digit() {
                        i += 1;
                    }
                }
            }
            let v: f64 = text[start..i]
                .parse()
                .map_err(|_| invalid(text, format!("column {column}: `{}` is not a number", &text[start..i])))?;
            out.push((column, Token::Number(v)));
        } else if c.is_ascii_alphabetic() || c == b'_' {
            let start = i;
            while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                i += 1;
            }
            out.push((column, Token::Name(text[start..i].to_string())));
        } else if let Some(s) = SYMBOLS.iter().find(|s| text[i..].starts_with(**s)) {
            out.push((column, Token::Symbol(s)));
            i += s.len();
        } else {
            let ch = text[i..].chars().next().unwrap_or(' ');
            return Err(invalid(text, format!("column {column}: unexpected `{ch}`")));
        }
    }
    Ok(out)
}

struct Parser<'a> {
    text: &'a str,
    tokens: Vec<(usize, Token)>,
    at: usize,
}

impl Parser<'_> {
    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.at).map(|(_, t)| t)
    }

    fn fail(&self, wanted: &str) -> TabsmithError {
        match self.tokens.get(self.at) {
            Some((column, t)) => invalid(self.text, format!("column {column}: expected {wanted}, found {t}")),
            None => invalid(self.text, format!("expected {wanted} at the end of the rule")),
        }
    }

    fn symbol(&mut self, s: &str) -> bool {
        if self.peek()
            == Some(&Token::Symbol(match SYMBOLS.iter().find(|x| **x == s) {
                Some(x) => x,
                None => return false,
            }))
        {
            self.at += 1;
            true
        } else {
            false
        }
    }

    fn keyword(&mut self, k: &str) -> bool {
        if matches!(self.peek(), Some(Token::Name(n)) if n == k) {
            self.at += 1;
            true
        } else {
            false
        }
    }

    fn number(&mut self) -> Result<f64> {
        let sign = if self.symbol("-") {
            -1.0
        } else {
            self.symbol("+");
            1.0
        };
        match self.peek() {
            Some(Token::Number(v)) => {
                let v = *v;
                self.at += 1;
                Ok(sign * v)
            }
            Some(Token::Name(n)) if n == "inf" => {
                self.at += 1;
                Ok(sign * f64::INFINITY)
            }
            _ => Err(self.fail("a number")),
        }
    }

    fn literal(&mut self) -> Result<Literal> {
        match self.peek() {
            Some(Token::Text(s)) => {
                let s = s.clone();
                self.at += 1;
                Ok(Literal::Text(s))
            }
            Some(Token::Name(n)) if n == "true" || n == "false" => {
                let b = n == "true";
                self.at += 1;
                Ok(Literal::Bool(b))
            }
            Some(Token::Name(n)) if n == "null" => {
                self.at += 1;
                Ok(Literal::Null)
            }
            _ => self
                .number()
                .map(Literal::Number)
                .map_err(|_| self.fail("a value: a number, a quoted string, true, false or null")),
        }
    }

    fn name(&mut self) -> Result<String> {
        match self.peek() {
            Some(Token::Name(n) | Token::Text(n)) => {
                let n = n.clone();
                self.at += 1;
                Ok(n)
            }
            _ => Err(self.fail("a field name")),
        }
    }

    fn condition(&mut self) -> Result<Condition> {
        if self.peek() == Some(&Token::Text(OTHER_FIELDS.to_string())) {
            return Err(self.fail("a field name: `_` stands for the fields no other condition names"));
        }
        let field = self.name()?;
        let test = if self.keyword("within") {
            let k = self.number()?;
            if !self.keyword("sd") {
                return Err(self.fail("`sd` (standard deviations of the training mean)"));
            }
            if !(k >= 0.0 && k.is_finite()) {
                return Err(invalid(
                    self.text,
                    format!("`{field} within {k} sd` needs a finite number of standard deviations, 0 or more"),
                ));
            }
            Test::Within(k)
        } else if field == OTHER_FIELDS {
            return Err(self.fail("`within` (`_ within k sd` limits every numeric field no other condition names)"));
        } else if self.keyword("in") {
            if self.symbol("[") {
                let mut values = Vec::new();
                while !self.symbol("]") {
                    values.push(self.literal()?);
                    if !self.symbol(",") {
                        if self.symbol("]") {
                            break;
                        }
                        return Err(self.fail("`,` or `]`"));
                    }
                }
                if values.is_empty() {
                    return Err(invalid(self.text, format!("`{field} in []` allows no value")));
                }
                Test::OneOf(values)
            } else {
                let lo = self.number()?;
                if !self.symbol("..=") {
                    return Err(if self.peek() == Some(&Token::Symbol("..")) {
                        self.fail("`..=` (both ends of a range belong to it)")
                    } else {
                        self.fail("`..=`")
                    });
                }
                let hi = self.number()?;
                Test::Range(lo, hi)
            }
        } else if self.symbol("<=") {
            Test::AtMost(self.number()?)
        } else if self.symbol(">=") {
            Test::AtLeast(self.number()?)
        } else if self.symbol("==") {
            Test::Equals(self.literal()?)
        } else if matches!(self.peek(), Some(Token::Symbol("<" | ">" | "="))) {
            return Err(self.fail("`<=`, `>=` or `==` (a rule has no strict comparisons)"));
        } else {
            return Err(self.fail("`in`, `<=`, `>=`, `==` or `within`"));
        };
        let empty = match test {
            Test::Range(a, b) => a > b || a.is_nan() || b.is_nan(),
            Test::AtMost(v) | Test::AtLeast(v) | Test::Equals(Literal::Number(v)) => v.is_nan(),
            _ => false,
        };
        if empty {
            return Err(invalid(self.text, format!("the range of `{field}` holds no number")));
        }
        Ok(Condition { field, test })
    }

    fn claim(&mut self) -> Result<Then> {
        let subject = match self.peek() {
            Some(Token::Name(n)) if ["probability", "label", "value"].contains(&n.as_str()) => n.clone(),
            _ => return Err(self.fail("`probability`, `label` or `value`")),
        };
        self.at += 1;
        let claim = match subject.as_str() {
            "label" => {
                let equal = if self.symbol("==") {
                    true
                } else if self.symbol("!=") {
                    false
                } else {
                    return Err(self.fail("`==` or `!=`"));
                };
                let label = match self.literal()? {
                    Literal::Text(s) => s,
                    Literal::Bool(b) => b.to_string(),
                    Literal::Number(v) if v.fract() == 0.0 && v.abs() < 9.0e15 => format!("{}", v as i64),
                    Literal::Number(v) => v.to_string(),
                    Literal::Null => return Err(invalid(self.text, "a label is never null")),
                };
                if equal { Then::Label(label) } else { Then::NotLabel(label) }
            }
            _ => {
                let at_most = if self.symbol("<=") {
                    true
                } else if self.symbol(">=") {
                    false
                } else {
                    return Err(self.fail("`<=` or `>=`"));
                };
                let v = self.number()?;
                if !v.is_finite() {
                    return Err(invalid(self.text, format!("the claim on the {subject} needs a finite number")));
                }
                match (subject.as_str(), at_most) {
                    ("probability", true) => Then::ProbabilityAtMost(v),
                    ("probability", false) => Then::ProbabilityAtLeast(v),
                    (_, true) => Then::ValueAtMost(v),
                    (_, false) => Then::ValueAtLeast(v),
                }
            }
        };
        Ok(claim)
    }
}

impl Rule {
    /// Reads a rule from its text; see [`Rule`] for the syntax.
    pub fn parse(text: &str) -> Result<Rule> {
        let mut p = Parser {
            text,
            tokens: tokens(text)?,
            at: 0,
        };
        let mut conditions = Vec::new();
        while !p.symbol("=>") {
            conditions.push(p.condition()?);
            if !p.symbol(",") {
                if p.symbol("=>") {
                    break;
                }
                return Err(p.fail("`,` or `=>`"));
            }
        }
        let claim = p.claim()?;
        if p.at < p.tokens.len() {
            return Err(p.fail("the end of the rule"));
        }
        Ok(Rule { conditions, claim })
    }
}

fn number(f: &mut fmt::Formatter<'_>, v: f64) -> fmt::Result {
    if v == f64::INFINITY {
        f.write_str("inf")
    } else if v == f64::NEG_INFINITY {
        f.write_str("-inf")
    } else {
        write!(f, "{v}")
    }
}

fn name(f: &mut fmt::Formatter<'_>, n: &str) -> fmt::Result {
    let plain = n.chars().next().is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && n.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
        && !["in", "within", "sd", "true", "false", "null", "inf"].contains(&n);
    if plain { f.write_str(n) } else { text(f, n) }
}

fn text(f: &mut fmt::Formatter<'_>, s: &str) -> fmt::Result {
    write!(f, "\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
}

impl fmt::Display for Literal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Literal::Number(v) => number(f, *v),
            Literal::Text(s) => text(f, s),
            Literal::Bool(b) => write!(f, "{b}"),
            Literal::Null => f.write_str("null"),
        }
    }
}

/// The rule in its canonical form, which [`Rule::parse`] reads back to the same rule.
impl fmt::Display for Rule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, c) in self.conditions.iter().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            name(f, &c.field)?;
            match &c.test {
                Test::Within(k) => {
                    f.write_str(" within ")?;
                    number(f, *k)?;
                    f.write_str(" sd")?;
                }
                Test::Range(a, b) => {
                    f.write_str(" in ")?;
                    number(f, *a)?;
                    f.write_str("..=")?;
                    number(f, *b)?;
                }
                Test::AtMost(v) => {
                    f.write_str(" <= ")?;
                    number(f, *v)?;
                }
                Test::AtLeast(v) => {
                    f.write_str(" >= ")?;
                    number(f, *v)?;
                }
                Test::Equals(v) => write!(f, " == {v}")?,
                Test::OneOf(values) => {
                    f.write_str(" in [")?;
                    for (k, v) in values.iter().enumerate() {
                        if k > 0 {
                            f.write_str(", ")?;
                        }
                        write!(f, "{v}")?;
                    }
                    f.write_str("]")?;
                }
            }
        }
        f.write_str(if self.conditions.is_empty() { "=> " } else { " => " })?;
        match &self.claim {
            Then::ProbabilityAtMost(v) => {
                f.write_str("probability <= ")?;
                number(f, *v)
            }
            Then::ProbabilityAtLeast(v) => {
                f.write_str("probability >= ")?;
                number(f, *v)
            }
            Then::Label(l) => {
                f.write_str("label == ")?;
                text(f, l)
            }
            Then::NotLabel(l) => {
                f.write_str("label != ")?;
                text(f, l)
            }
            Then::ValueAtMost(v) => {
                f.write_str("value <= ")?;
                number(f, *v)
            }
            Then::ValueAtLeast(v) => {
                f.write_str("value >= ")?;
                number(f, *v)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_every_kind_of_condition_and_claim() {
        let r = Rule::parse(r#"age in 17..=25, capital_gain == 0, "native-country" in ["United-States", null], hours <= 1e2, x >= -inf => probability <= 0.9"#).unwrap();
        assert_eq!(r.conditions.len(), 5);
        assert_eq!(r.conditions[0].test, Test::Range(17.0, 25.0));
        assert_eq!(r.conditions[1].test, Test::Equals(Literal::Number(0.0)));
        assert_eq!(r.conditions[2].field, "native-country");
        assert_eq!(
            r.conditions[2].test,
            Test::OneOf(vec![Literal::Text("United-States".into()), Literal::Null])
        );
        assert_eq!(r.conditions[3].test, Test::AtMost(100.0));
        assert_eq!(r.conditions[4].test, Test::AtLeast(f64::NEG_INFINITY));
        assert_eq!(r.claim, Then::ProbabilityAtMost(0.9));
        assert_eq!(Rule::parse("=> label != \">50K\"").unwrap().claim, Then::NotLabel(">50K".into()));
        assert_eq!(Rule::parse("flag == true => label == 1").unwrap().claim, Then::Label("1".into()));
        assert_eq!(
            Rule::parse("a in -1.5..=.5 => value >= -3").unwrap().conditions[0].test,
            Test::Range(-1.5, 0.5)
        );
        let r = Rule::parse("temp within 2.5 sd, _ within 3 sd => value <= 1").unwrap();
        assert_eq!(r.conditions[0].test, Test::Within(2.5));
        assert_eq!(
            (r.conditions[1].field.as_str(), &r.conditions[1].test),
            (OTHER_FIELDS, &Test::Within(3.0))
        );
    }

    #[test]
    fn the_canonical_text_reads_back_to_the_same_rule() {
        for text in [
            r#"age in 17..=25, "capital-gain" == 0, sex in ["F", "a \"b\" \\ c", true, null] => label == "x""#,
            "=> value >= -1.25",
            "in_ == 3, t <= inf => probability >= 0.000001",
            r#"_ within 3 sd, "sd" within 0.5 sd, x in [1, 2.5] => label != "no""#,
        ] {
            let r = Rule::parse(text).unwrap();
            let again = Rule::parse(&r.to_string()).unwrap();
            assert_eq!(again, r, "{r}");
        }
        assert_eq!(
            Rule::parse("age in 17 ..= 25 ,x==1=>probability<=0.5").unwrap().to_string(),
            "age in 17..=25, x == 1 => probability <= 0.5"
        );
    }

    #[test]
    fn says_where_a_rule_goes_wrong() {
        let cases = [
            ("age in 17..25 => probability <= 0.5", "column 10: expected `..=` (both ends"),
            ("age < 3 => probability <= 0.5", "no strict comparisons"),
            ("age in 30..=20 => value <= 1", "holds no number"),
            ("age == 3", "expected `,` or `=>` at the end"),
            ("age == 3 => prob <= 1", "expected `probability`, `label` or `value`"),
            ("=> probability <= inf", "needs a finite number"),
            ("=> label == null", "never null"),
            ("x in [] => value <= 1", "allows no value"),
            ("x == \"open => value <= 1", "never ends"),
            ("x == 1 => value <= 1 extra", "expected the end of the rule"),
            ("x # 1 => value <= 1", "unexpected `#`"),
            ("_ == 3 => value <= 1", "expected `within`"),
            ("x within -1 sd => value <= 1", "0 or more"),
            ("x within 3 => value <= 1", "expected `sd`"),
            ("\"_\" within 3 sd => value <= 1", "stands for the fields"),
        ];
        for (text, wanted) in cases {
            let e = Rule::parse(text).unwrap_err();
            assert_eq!(e.code, ErrorCode::RuleInvalid);
            assert!(e.message.contains(wanted), "{text}: {}", e.message);
        }
    }
}
