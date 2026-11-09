//! Rules audited on the fixture bundles: every proof is checked again and holds on random
//! inputs of its region, every counterexample lies in its region and is predicted as reported.

use std::path::PathBuf;

use tabsmith_contracts::ErrorCode;
use tabsmith_contracts::transform::FeatureTransform;
use tabsmith_runtime::audit::{Budget, FeatureDomain, Literal, Rule, Test, Verdict};
use tabsmith_runtime::{LoadOptions, Predictor, Record, Value};

fn bundle(name: &str) -> Predictor {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/bundles").join(name);
    Predictor::load(dir, LoadOptions::cpu()).unwrap()
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// The raw range a rule allows a numeric field, `None` for a missing value.
fn range(p: &Predictor, rule: &Rule, field: &str) -> Option<(f64, f64)> {
    let (mean, std) = p
        .bundle()
        .transform
        .features
        .iter()
        .find_map(|f| match f {
            FeatureTransform::Numeric { name, mean, std, .. } if name == field => Some((*mean, *std)),
            _ => None,
        })
        .unwrap();
    let (mut a, mut b) = (f64::NEG_INFINITY, f64::INFINITY);
    let mut named = false;
    for c in rule.conditions.iter().filter(|c| c.field == field) {
        named = true;
        let (x, y) = match c.test {
            Test::Range(x, y) => (x, y),
            Test::AtMost(y) => (f64::NEG_INFINITY, y),
            Test::AtLeast(x) => (x, f64::INFINITY),
            Test::Equals(Literal::Number(x)) => (x, x),
            Test::Within(k) => (mean - k * std, mean + k * std),
            Test::Equals(Literal::Null) => return None,
            _ => unreachable!(),
        };
        (a, b) = (a.max(x), b.min(y));
    }
    if !named && let Some(c) = rule.conditions.iter().find(|c| c.field == "_") {
        let Test::Within(k) = c.test else { unreachable!() };
        (a, b) = (mean - k * std, mean + k * std);
    }
    Some((a, b))
}

/// The categories a rule allows a field, as text (`None`: missing), or all of them.
fn allowed(rule: &Rule, field: &str, categories: &[String]) -> Vec<Option<String>> {
    let text = |v: &Literal| match v {
        Literal::Text(s) => Some(s.clone()),
        Literal::Bool(b) => Some(b.to_string()),
        Literal::Number(x) => Some(x.to_string()),
        Literal::Null => None,
    };
    let mut out: Vec<Option<String>> = categories.iter().cloned().map(Some).collect();
    if rule.conditions.iter().any(|c| c.field == field) {
        out.push(None);
    }
    for c in rule.conditions.iter().filter(|c| c.field == field) {
        let set: Vec<Option<String>> = match &c.test {
            Test::Equals(v) => vec![text(v)],
            Test::OneOf(v) => v.iter().map(text).collect(),
            _ => unreachable!(),
        };
        out.retain(|v| set.contains(v));
    }
    out
}

/// Whether the record lies in the rule's region.
fn inside(p: &Predictor, rule: &Rule, r: &Record) -> bool {
    p.bundle().transform.features.iter().all(|f| match f {
        FeatureTransform::Numeric { name, .. } => match (range(p, rule, name), r.get(name)) {
            (None, Some(Value::Null)) => true,
            (Some((a, b)), Some(Value::Float(x))) => a <= *x && *x <= b,
            (Some((a, b)), Some(Value::Int(x))) => a <= *x as f64 && *x as f64 <= b,
            _ => false,
        },
        FeatureTransform::Categorical { name, categories, .. } => {
            if !rule.conditions.iter().any(|c| c.field == *name) {
                return true;
            }
            let value = match r.get(name) {
                Some(Value::Text(s)) => Some(s.clone()),
                Some(Value::Bool(b)) => Some(b.to_string()),
                _ => None,
            };
            allowed(rule, name, categories).contains(&value)
        }
    })
}

/// A random record of the rule's region: numbers uniform in their range (within 4 standard
/// deviations of the mean where it is unbounded), a random allowed category.
fn sample(p: &Predictor, rule: &Rule, rng: &mut Rng) -> Record {
    let b = p.bundle();
    let mut r = Record::from_json(&b.examples.records[0]).unwrap();
    for f in &b.transform.features {
        match f {
            FeatureTransform::Numeric { name, mean, std, .. } => match range(p, rule, name) {
                None => r.insert(name.clone(), Value::Null),
                Some((a, z)) => {
                    let (a, z) = (a.max(mean - 4.0 * std), z.min(mean + 4.0 * std));
                    let x = if a <= z { a + (z - a) * rng.unit() } else { a };
                    r.insert(name.clone(), x);
                }
            },
            FeatureTransform::Categorical { name, categories, .. } => {
                let options = allowed(rule, name, categories);
                let pick = options[(rng.next() % options.len() as u64) as usize].clone();
                let bool_field = b
                    .input_schema
                    .field(name)
                    .is_some_and(|f| f.raw_dtype == tabsmith_contracts::schema::RawDtype::Bool);
                let value = match pick {
                    None => Value::Null,
                    Some(s) if bool_field => Value::Bool(s == "true"),
                    Some(s) => Value::Text(s),
                };
                r.insert(name.clone(), value);
            }
        }
    }
    r
}

/// Audits each rule and checks the answer on its own terms. Returns the results in order.
fn audit(name: &str, rules: &[&str], boxes: usize) -> Vec<&'static str> {
    let p = bundle(name);
    let mut rng = Rng(0x2545_f491_4f6c_dd1d);
    let mut results = Vec::new();
    for text in rules {
        let rule = Rule::parse(text).unwrap();
        let a = p.audit(&rule, &Budget { boxes, seconds: None }).unwrap();
        match &a.verdict {
            Verdict::Holds(proof) => {
                assert!(
                    proof.check(&p.bundle().model, &FeatureDomain::of(&p.bundle().transform)),
                    "{name}: {text}"
                );
                for _ in 0..300 {
                    let r = sample(&p, &rule, &mut rng);
                    let pred = p.predict_one(&r).unwrap();
                    assert!(rule.claim.holds_for(&pred), "{name}: `{text}` holds, but {r:?} predicts {pred:?}");
                }
            }
            Verdict::Fails { witness, prediction } => {
                assert!(inside(&p, &rule, witness), "{name}: `{text}`: the witness {witness:?} lies outside");
                assert_eq!(&p.predict_one(witness).unwrap(), prediction, "{name}: {text}");
                assert!(!rule.claim.holds_for(prediction), "{name}: {text}");
            }
            Verdict::Undecided(_) => {}
        }
        results.push(a.result());
    }
    results
}

#[test]
fn rules_on_binary_trees() {
    let r = audit(
        "binary-gbdt",
        &[
            "=> probability <= 0.99",
            "=> probability <= 0.8",
            "hours_in_service in 0..=1000 => probability <= 0.9",
            "hours_in_service in 0..=1000 => probability <= 0.8",
            "line == \"A\", temp_mean <= 60 => label == \"0\"",
            "=> probability >= 0.01",
            "supplier in [\"S1\", \"S2\"], maintenance_flag == true => probability <= 0.7",
            "temp_mean within 1 sd, _ within 2 sd, line in [\"A\", \"B\"] => probability <= 0.75",
            "maintenance_flag == null => label == \"0\"",
        ],
        1_000_000,
    );
    assert_eq!(r, ["holds", "fails", "holds", "fails", "fails", "holds", "fails", "fails", "fails"]);
}

#[test]
fn the_largest_probability_is_found_exactly() {
    let p = bundle("binary-gbdt");
    let a = p.audit(&Rule::parse("=> probability <= 0.8").unwrap(), &Budget::default()).unwrap();
    let Verdict::Fails { prediction, .. } = a.verdict else {
        panic!("{a:?}")
    };
    // The largest probability any input gets: the next rule up holds.
    let top = prediction.probability().unwrap();
    assert!((top - 0.8426).abs() < 1e-4, "{top}");
    let at_top = Rule::parse(&format!("=> probability <= {top}")).unwrap();
    assert_eq!(p.audit(&at_top, &Budget::default()).unwrap().result(), "holds");
}

#[test]
fn rules_on_multiclass_and_regression_trees() {
    let r = audit(
        "multiclass-gbdt",
        &[
            "=> label != \"none\"",
            "temp_mean >= 80, vibration_rms <= 0.5 => label == \"overheat\"",
            "temp_mean <= 50 => label != \"overheat\"",
            "_ within 0.25 sd, line == \"A\", shift == \"day\", supplier == \"S1\", device_type == \"pump\", firmware == \"v1.3\", maintenance_flag == false => label == \"none\"",
        ],
        1_000_000,
    );
    assert_eq!(r[..3], ["fails", "fails", "fails"]);
    assert_ne!(r[3], "undecided");
    let r = audit(
        "regression-gbdt",
        &[
            "=> value <= 1000000",
            "=> value >= 0",
            "hours_in_service >= 15000 => value <= 5000",
            "device_type == \"motor\" => value >= 100",
            "_ within 1 sd => value <= 20000",
        ],
        1_000_000,
    );
    assert_eq!(r[..4], ["holds", "holds", "fails", "holds"]);
    assert_ne!(r[4], "undecided");
}

#[test]
fn rules_on_networks() {
    let r = audit(
        "regression-mlp",
        &[
            "_ within 2 sd => value <= 100000",
            "_ within 2 sd => value >= -1000000",
            "_ within 1 sd => value <= 30000",
        ],
        200_000,
    );
    assert_eq!(r, ["holds", "holds", "fails"]);
    let r = audit(
        "policies-linear",
        &[
            "temp in 50..=70, load in 0.5..=1.5 => probability <= 0.99",
            "mode == null, _ within 2 sd => label == \"ok\"",
            "zone == \"1\", temp <= 55, _ within 3 sd => label == \"ok\"",
        ],
        200_000,
    );
    assert_eq!(r, ["holds", "fails", "fails"]);
    let r = audit(
        "multiclass-linear",
        &["_ within 0.5 sd => label != \"bearing\"", "=> label != \"seal\""],
        200_000,
    );
    assert_eq!(r, ["fails", "fails"]);
    // A network with every field unbounded extrapolates: refuted far out, but refuted.
    let r = audit("binary-mlp", &["=> probability <= 0.999"], 200_000);
    assert_eq!(r, ["fails"]);
}

#[test]
fn the_same_rule_gets_the_same_answer() {
    let p = bundle("multiclass-gbdt");
    let rule = Rule::parse("_ within 1 sd, line == \"A\" => label != \"bearing\"").unwrap();
    let a = p.audit(&rule, &Budget::default()).unwrap();
    let b = p.audit(&rule, &Budget::default()).unwrap();
    assert_eq!((a.verdict, a.boxes), (b.verdict, b.boxes));
}

#[test]
fn a_rule_that_does_not_fit_the_bundle_is_refused() {
    let cases = [
        (
            "binary-gbdt",
            "pressure_max <= 3 => probability <= 0.9",
            "has no input field `pressure_max`",
        ),
        (
            "binary-gbdt",
            "calibration_offset <= 3 => probability <= 0.9",
            "does not use `calibration_offset`",
        ),
        (
            "binary-gbdt",
            "line == 3.5 => probability <= 0.9",
            "`line` holds text values, not 3.5",
        ),
        (
            "binary-gbdt",
            "maintenance_flag == \"yes\" => probability <= 0.9",
            "`maintenance_flag` holds bool values",
        ),
        (
            "policies-linear",
            "zone == \"9\" => probability <= 0.9",
            "refuses unseen categories",
        ),
        ("policies-linear", "flag == null => probability <= 0.9", "refuses missing values"),
        ("binary-gbdt", "=> value <= 3", "claims the probability or the label"),
        ("binary-gbdt", "=> label == \"yes\"", "labels are \"0\" and \"1\""),
        ("multiclass-gbdt", "=> probability <= 0.5", "claims a label"),
        ("regression-gbdt", "=> label == \"x\"", "claims the value"),
        ("binary-gbdt", "temp_mean == \"hot\" => probability <= 0.9", "is a number"),
        ("binary-gbdt", "line in 1..=2 => probability <= 0.9", "is a category"),
        (
            "binary-gbdt",
            "temp_mean <= 50, temp_mean >= 60 => probability <= 0.9",
            "allow no value together",
        ),
    ];
    for (name, text, wanted) in cases {
        let p = bundle(name);
        let e = p.audit(&Rule::parse(text).unwrap(), &Budget::default()).unwrap_err();
        assert_eq!(e.code, ErrorCode::RuleInvalid, "{text}");
        assert!(e.render().contains(wanted), "{text}: {}", e.render());
    }
}

#[test]
fn a_budget_leaves_a_rule_undecided() {
    let p = bundle("binary-mlp");
    let rule = Rule::parse("_ within 2 sd => probability <= 0.999").unwrap();
    let a = p.audit(&rule, &Budget { boxes: 50, seconds: None }).unwrap();
    assert_eq!(a.result(), "undecided");
    assert_eq!(a.boxes, 50);
    let v = a.to_json();
    assert_eq!(v["result"], "undecided");
    assert!(v["reason"].as_str().unwrap().contains("50 boxes"));
}
