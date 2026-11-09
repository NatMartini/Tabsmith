//! Rules a model keeps over a region of inputs, checked for every input of the region.
//!
//! ```no_run
//! use tabsmith_runtime::audit::{Budget, Rule, Verdict};
//! use tabsmith_runtime::{LoadOptions, Predictor};
//!
//! let predictor = Predictor::load("bundles/run-002", LoadOptions::cpu())?;
//! let rule = Rule::parse("hours_in_service in 0..=1000, line == \"A\" => probability <= 0.9")?;
//! match predictor.audit(&rule, &Budget::default())?.verdict {
//!     Verdict::Holds(proof) => println!("holds: a proof of {} boxes", proof.boxes()),
//!     Verdict::Fails { witness, prediction } => println!("fails for {:?}: {}", witness, prediction.label),
//!     Verdict::Undecided(why) => println!("undecided: {why}"),
//! }
//! # Ok::<(), tabsmith_runtime::TabsmithError>(())
//! ```
//!
//! A rule holds when every input the model accepts with the fields in the rule's ranges gets
//! a prediction that meets the claim. "Every input" means what `predict` accepts, not what is
//! realistic: any finite number for a numeric field (a missing value counts as its imputed
//! value), one category per categorical field (a missing value when the bundle has a slot for
//! it, and a value never seen in training when it ignores those). Fields without a condition
//! take any such value.
//!
//! The search cuts the rule's region into boxes of feature values, where a category's slots
//! stay one-hot, until the bounds of every box settle the claim, or it finds an input where
//! the claim fails. For a tree ensemble it is exact: a box where every tree reaches leaves of
//! one value is a single prediction, so the search ends with a proof or a counterexample
//! unless the [`Budget`] runs out. For a network it uses interval bounds, which can prove a
//! claim but not refute it, so it also predicts points of the boxes it visits: it ends with a
//! proof, a counterexample, or undecided.
//!
//! Both answers are checked on their own terms. A counterexample is an input record, predicted
//! by [`Predictor::predict_one`] like any other, and kept only when that prediction breaks the
//! claim. A proof is a [`Certificate`], checked again by [`Certificate::check`] with the walk of
//! `tabsmith_embedded::rules` and the model's bounds before it is reported; `model!(…, rule(…))`
//! makes the compiler check it once more.

mod parse;
mod region;
mod search;

use std::time::{Duration, Instant};

pub use parse::{Condition, Literal, Rule, Test, Then};
pub use region::FeatureDomain;
use region::Region;
use search::{Bx, Check, Found, Limits, NetSearch, TreeSearch};
use tabsmith_contracts::config::Task;
use tabsmith_contracts::schema::OutputSchema;
use tabsmith_contracts::{Phase, Result, TabsmithError};
pub use tabsmith_embedded::rules::{Claim, Step};
use tabsmith_embedded::rules::{Walk, is_empty};

use crate::model::Model;
use crate::{Prediction, Predictor, Record};

/// How much search a rule may take before it is reported undecided.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Budget {
    /// Boxes the search may visit. The search is deterministic, so with this limit alone the
    /// same rule gets the same answer on every machine.
    pub boxes: usize,
    /// A time limit in seconds, if any. The answer then depends on the machine's speed.
    pub seconds: Option<f64>,
}

impl Default for Budget {
    /// A million boxes, no time limit.
    fn default() -> Self {
        Budget {
            boxes: 1_000_000,
            seconds: None,
        }
    }
}

/// A proof that a rule holds, in the form `tabsmith_embedded::rules` checks: the region's box,
/// the splits that cut it, and the claim every closed box satisfies.
#[derive(Debug, Clone, PartialEq)]
pub struct Certificate {
    /// What every prediction of the region satisfies, read from the model's outputs.
    pub claim: Claim,
    /// The box of feature values the rule's region encodes to: lower bounds.
    pub lo: Vec<f32>,
    /// Upper bounds.
    pub hi: Vec<f32>,
    /// Per category: whether the region holds values the model never saw.
    pub none: Vec<bool>,
    /// The splits and closed boxes, in the order of a walk that visits left parts first.
    pub steps: Vec<Step>,
    /// The most splits on one path: the room for saved boxes a walk needs.
    pub depth: usize,
}

impl Certificate {
    /// The boxes the proof closes.
    pub fn boxes(&self) -> usize {
        self.steps.iter().filter(|s| matches!(s, Step::Close)).count()
    }

    /// Checks the proof again: a walk through its steps (`tabsmith_embedded::rules::Walk`)
    /// must cover the region with well-formed splits, and every box it closes must hold no
    /// input or have output bounds that satisfy the claim. A network's bounds are
    /// `kernels::dense_bounds` through the layers. A tree ensemble's are the sums that
    /// `kernels::Trees::bounds` adds, of each tree's `kernels::Trees::leaf_range`; from one
    /// closed box to the next, only the trees that read a feature whose bounds changed get
    /// their range again.
    pub fn check(&self, model: &Model, domain: &FeatureDomain) -> bool {
        let d = domain.as_domain();
        let (n, g) = (domain.category.len(), domain.slots.len());
        if self.lo.len() != n || self.hi.len() != n || self.none.len() != g {
            return false;
        }
        let (mut lo, mut hi, mut none) = (self.lo.clone(), self.hi.clone(), self.none.clone());
        let mut saved_lo = vec![0f32; n * self.depth];
        let mut saved_hi = vec![0f32; n * self.depth];
        let mut saved_none = vec![false; g * self.depth];
        let mut saved_step = vec![0usize; self.depth];
        let mut walk = Walk::new(d, &self.steps, &mut saved_lo, &mut saved_hi, &mut saved_none, &mut saved_step);
        let k = model.config().n_outputs();
        let mut p = vec![0f64; k];
        let mut trees = match model {
            Model::Trees(m) => Some(search::Ranges::new(m, n)),
            Model::Dense(_) => None,
        };
        loop {
            match walk.next(&mut lo, &mut hi, &mut none) {
                Ok(true) => {
                    if is_empty(&d, &lo, &hi, &none) {
                        continue;
                    }
                    let (lo_out, hi_out) = match (model, &mut trees) {
                        (Model::Trees(m), Some(r)) => r.bounds(m, &lo, &hi),
                        (Model::Dense(m), _) => search::network_bounds(m, &lo, &hi),
                        _ => unreachable!("ranges exist for trees"),
                    };
                    if !self.claim.holds_within(&lo_out, &hi_out, &mut p) {
                        return false;
                    }
                }
                Ok(false) => return true,
                Err(_) => return false,
            }
        }
    }
}

/// What the audit of a rule found.
#[derive(Debug, Clone, PartialEq)]
pub enum Verdict {
    /// Every input of the region gets a prediction that meets the claim.
    Holds(Certificate),
    /// An input of the region whose prediction breaks the claim, as `predict_one` predicted it.
    Fails {
        /// The input.
        witness: Record,
        /// Its prediction.
        prediction: Prediction,
    },
    /// The search ended without either answer; why.
    Undecided(String),
}

/// The audit of one rule.
#[derive(Debug, Clone, PartialEq)]
pub struct Audit {
    /// The rule, in its canonical text.
    pub rule: String,
    /// What the search found.
    pub verdict: Verdict,
    /// Boxes the search visited.
    pub boxes: usize,
    /// Time taken, the check of a proof included.
    pub seconds: f64,
}

impl Audit {
    /// `holds`, `fails` or `undecided`.
    pub fn result(&self) -> &'static str {
        match self.verdict {
            Verdict::Holds(_) => "holds",
            Verdict::Fails { .. } => "fails",
            Verdict::Undecided(_) => "undecided",
        }
    }

    /// The audit as JSON: the rule, the result, the boxes visited and the time, and the
    /// proof's size, the witness and its prediction, or the reason it is undecided.
    pub fn to_json(&self) -> serde_json::Value {
        let mut v = serde_json::json!({
            "rule": self.rule,
            "result": self.result(),
            "boxes_visited": self.boxes,
            "seconds": self.seconds,
        });
        match &self.verdict {
            Verdict::Holds(c) => {
                v["proof"] = serde_json::json!({"boxes": c.boxes(), "steps": c.steps.len(), "depth": c.depth});
            }
            Verdict::Fails { witness, prediction } => {
                v["witness"] = witness.to_json();
                v["prediction"] = serde_json::to_value(prediction).unwrap_or_default();
            }
            Verdict::Undecided(why) => v["reason"] = serde_json::Value::from(why.as_str()),
        }
        v
    }
}

impl Then {
    /// Whether a prediction meets the claim, read from what `predict` returns.
    pub fn holds_for(&self, p: &Prediction) -> bool {
        match self {
            Then::ProbabilityAtMost(c) => p.probability().is_some_and(|q| q <= *c),
            Then::ProbabilityAtLeast(c) => p.probability().is_some_and(|q| q >= *c),
            Then::Label(l) => p.label == *l,
            Then::NotLabel(l) => p.label != *l,
            Then::ValueAtMost(c) => p.value().is_some_and(|v| v <= *c),
            Then::ValueAtLeast(c) => p.value().is_some_and(|v| v >= *c),
        }
    }

    /// The claim on the model's outputs that means this for a bundle with output schema `o`.
    pub fn claim(&self, o: &OutputSchema, text: &str) -> Result<Claim> {
        let refuse = |m: String| parse::invalid(text, m);
        match (o.task, self) {
            (Task::Binary, Then::ProbabilityAtMost(c)) => Ok(Claim::ProbabilityAtMost(*c)),
            (Task::Binary, Then::ProbabilityAtLeast(c)) => Ok(Claim::ProbabilityAtLeast(*c)),
            (Task::Binary, Then::Label(l) | Then::NotLabel(l)) => {
                let positive = if *l == o.positive_label {
                    true
                } else if *l == o.negative_label {
                    false
                } else {
                    return Err(refuse(format!(
                        "the model's labels are {:?} and {:?}, not {l:?}",
                        o.negative_label, o.positive_label
                    )));
                };
                Ok(if positive == matches!(self, Then::Label(_)) {
                    Claim::Positive { threshold: o.threshold }
                } else {
                    Claim::Negative { threshold: o.threshold }
                })
            }
            (Task::Multiclass, Then::Label(l) | Then::NotLabel(l)) => {
                let k = o.classes.iter().position(|c| c == l).ok_or_else(|| {
                    let shown: Vec<String> = o.classes.iter().take(12).map(|c| format!("{c:?}")).collect();
                    refuse(format!("the model has no class {l:?}")).actual(format!("classes: {}", shown.join(", ")))
                })?;
                Ok(if matches!(self, Then::Label(_)) {
                    Claim::Class(k)
                } else {
                    Claim::NotClass(k)
                })
            }
            (Task::Regression, Then::ValueAtMost(v) | Then::ValueAtLeast(v)) => {
                let s = o
                    .target_scale
                    .ok_or_else(|| refuse("a regression bundle without its target scale".into()))?;
                Ok(if matches!(self, Then::ValueAtMost(_)) {
                    Claim::ValueAtMost {
                        limit: *v,
                        mean: s.mean,
                        std: s.std,
                    }
                } else {
                    Claim::ValueAtLeast {
                        limit: *v,
                        mean: s.mean,
                        std: s.std,
                    }
                })
            }
            (Task::Multiclass, _) => Err(refuse(
                "a multiclass rule claims a label: `label == \"…\"` or `label != \"…\"`".into(),
            )),
            (Task::Regression, _) => Err(refuse("a regression rule claims the value: `value <= v` or `value >= v`".into())),
            (Task::Binary, _) => Err(refuse(
                "a binary rule claims the probability or the label: `probability <= p`, `probability >= p`, `label == \"…\"`".into(),
            )),
        }
    }
}

impl Predictor {
    /// The rule that a record's label is stable: every input with the record's categories and
    /// missing values, and each of its numbers within `sd` standard deviations (of the field in
    /// training) of the record's, gets the label the record gets. A field that held whole
    /// numbers in training has the ends of its range rounded inward to whole numbers, so that
    /// a 0/1 indicator stays put unless `sd` reaches a whole step (a range can still hold
    /// fractions between its ends). A regression has no label to keep, so it is refused.
    pub fn stability_rule(&self, record: &Record, sd: f64) -> Result<Rule> {
        let b = self.bundle();
        if b.output_schema.task == Task::Regression {
            return Err(parse::invalid(
                "--stability",
                "stability is about a label, and a regression predicts a value",
            ));
        }
        if !(sd >= 0.0 && sd.is_finite()) {
            return Err(parse::invalid(
                "--stability",
                format!("{sd} standard deviations: give a finite number, 0 or more"),
            ));
        }
        if !b.transform.has_statistics() {
            return Err(parse::invalid(
                "--stability",
                "the bundle was imported and holds no training statistics, so it has no standard deviations to move by",
            ));
        }
        let label = self.predict_one(record)?.label;
        let tokens = crate::MissingTokens::new(&b.transform.missing_tokens);
        let mut conditions = Vec::new();
        for f in &b.transform.features {
            let name = f.name().to_string();
            let value = record.get(&name).map_or(crate::ValueRef::Null, crate::Value::as_ref);
            let whole = b
                .input_schema
                .field(&name)
                .is_some_and(|f| f.raw_dtype == tabsmith_contracts::schema::RawDtype::Int);
            let range = |x: f64, std: f64| {
                let (a, z) = (x - sd * std, x + sd * std);
                if whole && x.fract() == 0.0 {
                    Test::Range(a.ceil(), z.floor())
                } else {
                    Test::Range(a, z)
                }
            };
            let test = match f {
                tabsmith_contracts::transform::FeatureTransform::Numeric { std, .. } => match tokens.numeric(value) {
                    crate::ValueRef::Float(x) if x.is_finite() => range(x, *std),
                    crate::ValueRef::Int(i) => range(i as f64, *std),
                    _ => Test::Equals(Literal::Null),
                },
                tabsmith_contracts::transform::FeatureTransform::Categorical { .. } => Test::Equals(match tokens.categorical(value) {
                    crate::ValueRef::Text(s) => Literal::Text(s.to_string()),
                    crate::ValueRef::Bool(v) => Literal::Bool(v),
                    crate::ValueRef::Int(i) => Literal::Number(i as f64),
                    crate::ValueRef::Float(x) if !x.is_nan() => Literal::Number(x),
                    _ => Literal::Null,
                }),
            };
            conditions.push(Condition { field: name, test });
        }
        Ok(Rule {
            conditions,
            claim: Then::Label(label),
        })
    }

    /// Checks a rule for every input of its region; see the [module](crate::audit). An error
    /// means the rule does not fit the bundle (a field it lacks, a category it never saw, a
    /// claim for another task); a rule that fails is an [`Audit`] with a counterexample.
    pub fn audit(&self, rule: &Rule, budget: &Budget) -> Result<Audit> {
        self.audit_from(rule, budget, None)
    }

    /// [`audit`](Self::audit), with a counterexample made from `near`: it keeps every value of
    /// `near` the counterexample's box allows, and moves the others as little as the box
    /// allows. For a tree ensemble, so that a record's [`stability_rule`](Self::stability_rule)
    /// fails with the record changed only where the change matters.
    pub fn audit_near(&self, rule: &Rule, budget: &Budget, near: &Record) -> Result<Audit> {
        self.audit_from(rule, budget, Some(near))
    }

    fn audit_from(&self, rule: &Rule, budget: &Budget, near: Option<&Record>) -> Result<Audit> {
        let start = Instant::now();
        let text = rule.to_string();
        let b = self.bundle();
        let claim = rule.claim.claim(&b.output_schema, &text)?;
        let region = Region::of(self, rule, &text)?;
        let domain = FeatureDomain::of(&b.transform);
        let limits = Limits {
            boxes: budget.boxes,
            deadline: budget.seconds.map(|s| start + Duration::from_secs_f64(s.max(0.0))),
        };
        let whole = Bx {
            lo: region.lo.clone(),
            hi: region.hi.clone(),
            none: region.none.clone(),
        };
        let mut confirm = |bx: &Bx, point: Option<(&[f32], &[Option<usize>])>| -> Result<Check<(Record, Prediction)>> {
            // A network's point is where the claim fails, so it wins over `near`.
            let near = if point.is_some() { None } else { near };
            let made = region::record_in(
                self,
                &region,
                (&bx.lo, &bx.hi, &bx.none),
                point.map(|p| p.0),
                point.map(|p| p.1),
                near,
            )?;
            let Some((record, _)) = made else { return Ok(Check::NoInput) };
            let prediction = self.predict_one(&record)?;
            Ok(if rule.claim.holds_for(&prediction) {
                Check::Holds
            } else {
                Check::Fails((record, prediction))
            })
        };
        let (found, boxes) = match &b.model {
            Model::Trees(m) => TreeSearch::new(m, m.base(), domain.as_domain(), claim).run(whole, &limits, &mut confirm)?,
            Model::Dense(m) => NetSearch::new(m, domain.as_domain(), claim).run(whole, &limits, &mut confirm)?,
        };
        let verdict = match found {
            Found::Proof { steps, depth } => {
                let proof = Certificate {
                    claim,
                    lo: region.lo,
                    hi: region.hi,
                    none: region.none,
                    steps,
                    depth,
                };
                if !proof.check(&b.model, &domain) {
                    return Err(TabsmithError::internal(
                        Phase::Audit,
                        format!("the proof found for `{text}` does not check"),
                    ));
                }
                Verdict::Holds(proof)
            }
            Found::Fails((witness, prediction)) => Verdict::Fails { witness, prediction },
            Found::Undecided(why) => Verdict::Undecided(why),
        };
        Ok(Audit {
            rule: text,
            verdict,
            boxes,
            seconds: start.elapsed().as_secs_f64(),
        })
    }
}
