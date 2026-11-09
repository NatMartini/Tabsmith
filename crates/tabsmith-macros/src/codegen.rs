//! Reads a bundle with the runtime and generates the module that `model!` expands to.

use std::path::{Path, PathBuf};

use proc_macro2::{Ident, Literal, TokenStream};
use quote::{format_ident, quote};
use syn::LitStr;
use syn::parse::{Parse, ParseStream};
use tabsmith_contracts::config::{NanPolicy, Task, UnknownCategoryPolicy};
use tabsmith_contracts::model::{Activation, LeafSum};
use tabsmith_contracts::schema::RawDtype;
use tabsmith_contracts::transform::{FeatureTransform, MissingRule};
use tabsmith_embedded::rules::{Claim, NUMERIC, Step};
use tabsmith_embedded::{Work, kernels};
use tabsmith_runtime::audit::{Budget, Certificate, FeatureDomain, Rule, Verdict};
use tabsmith_runtime::model::Model;
use tabsmith_runtime::{LoadOptions, Outcome, Predictor, Record};

use crate::names::{Namer, camel, snake};

/// Trees with more nodes than this stay arrays unless the invocation asks for `tree_code`:
/// compile time grows with the generated code. Above it, the generated `outputs_batch` runs
/// tree by tree over blocks of rows, because one pass through all the code no longer fits
/// in the cache; below it, row by row is as fast (`reports/inference/`).
const TREE_CODE_MAX_NODES: usize = 32_768;
/// Trees deeper than this always stay arrays: each level nests one more `if`.
const TREE_CODE_MAX_DEPTH: usize = 64;

/// The most work the compiler is asked to do to check a rule's proof again: closed boxes
/// times the nodes of the trees (or the weights of the network) that bound each one. rustc's
/// constant evaluator takes about 3 µs per unit, so this is about 3 s (`reports/rules/`); a
/// larger proof is checked when the macro expands, natively, and the module's documentation
/// says so.
const RULE_CHECK_MAX_WORK: usize = 1_000_000;

/// `name, "path"` with the options `kani`, `tree_code` or `tree_arrays`, and `rule(…)`.
pub struct Args {
    name: Ident,
    path: LitStr,
    kani: bool,
    /// `Some(true)` for `tree_code`, `Some(false)` for `tree_arrays`.
    tree_code: Option<bool>,
    /// Each `rule(…)`: where it is, and its text.
    rules: Vec<(proc_macro2::Span, String)>,
}

impl Parse for Args {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let name: Ident = input.parse()?;
        input.parse::<syn::Token![,]>()?;
        let path: LitStr = input.parse()?;
        let (mut kani, mut tree_code, mut rules) = (false, None, Vec::new());
        while input.peek(syn::Token![,]) {
            input.parse::<syn::Token![,]>()?;
            if input.is_empty() {
                break;
            }
            let option: Ident = input.parse()?;
            if option == "kani" {
                kani = true;
            } else if option == "tree_code" || option == "tree_arrays" {
                if tree_code.is_some() {
                    return Err(syn::Error::new(option.span(), "`tree_code` and `tree_arrays` exclude each other"));
                }
                tree_code = Some(option == "tree_code");
            } else if option == "rule" {
                let content;
                syn::parenthesized!(content in input);
                let tokens: TokenStream = content.parse()?;
                rules.push((option.span(), tokens.to_string()));
            } else {
                return Err(syn::Error::new(
                    option.span(),
                    format!("unknown option `{option}`; the options are `kani`, `tree_code`, `tree_arrays` and `rule(…)`"),
                ));
            }
        }
        Ok(Args {
            name,
            path,
            kani,
            tree_code,
            rules,
        })
    }
}

pub fn expand(args: &Args) -> syn::Result<TokenStream> {
    let fail = |msg: String| syn::Error::new(args.path.span(), msg);
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").map_err(|_| fail("CARGO_MANIFEST_DIR is not set; build with cargo".into()))?;
    let dir = Path::new(&manifest_dir).join(args.path.value());
    let compiled = Compiled::load(&dir, &args.path.value()).map_err(|e| fail(format!("Tabsmith bundle `{}`:\n{e}", args.path.value())))?;
    let tree_code = compiled
        .tree_code(args.tree_code)
        .map_err(|e| fail(format!("Tabsmith bundle `{}`:\n{e}", args.path.value())))?;
    let mut proofs = Vec::with_capacity(args.rules.len());
    for (span, text) in &args.rules {
        proofs.push(compiled.prove(text).map_err(|e| syn::Error::new(*span, e))?);
    }
    Ok(compiled.generate(&args.name, args.kani, tree_code, &proofs))
}

/// A rule the macro proved: its canonical text and its proof.
pub(crate) struct Proved {
    text: String,
    proof: Certificate,
    /// Boxes the search visited.
    visited: usize,
}

/// One input field of the model.
struct Field {
    name: String,
    ident: Ident,
    spec: Spec,
}

enum Spec {
    Numeric {
        impute: f64,
        mean: f64,
        std: f64,
        offset: usize,
        /// The field refuses missing values (a model imported from XGBoost or LightGBM).
        refuse_missing: bool,
    },
    Text {
        ty: Ident,
        categories: Vec<String>,
        variants: Vec<Ident>,
        /// The variant for categories not seen in training (`unknown_category = "ignore"`).
        other: Option<(Ident, String)>,
        missing_slot: bool,
        nullable: bool,
        offset: usize,
    },
    Bool {
        categories: Vec<String>,
        missing_slot: bool,
        nullable: bool,
        offset: usize,
    },
}

enum Network {
    Dense(Vec<Layer>),
    Trees {
        roots: Vec<u32>,
        feature: Vec<i32>,
        threshold: Vec<f32>,
        left: Vec<u32>,
        right: Vec<u32>,
        value: Vec<f32>,
        base: Vec<f64>,
        /// One sign per feature: 1 increasing, -1 decreasing, 0 unconstrained.
        monotone: Vec<i8>,
        /// Whether the leaves are added in f32 (a model imported from XGBoost).
        sum_in_f32: bool,
    },
}

struct Layer {
    in_dim: usize,
    out_dim: usize,
    weight: Vec<f32>,
    bias: Vec<f32>,
    relu: bool,
}

enum Output {
    Binary {
        negative: String,
        positive: String,
        threshold: f64,
    },
    Multiclass {
        classes: Vec<String>,
        variants: Vec<Ident>,
    },
    Regression {
        mean: f64,
        std: f64,
    },
}

struct Compiled {
    path: String,
    files: Vec<PathBuf>,
    bundle_id: String,
    task: Task,
    target: String,
    summary: String,
    fields: Vec<Field>,
    dropped: Vec<String>,
    n_features: usize,
    n_outputs: usize,
    text_tokens: Vec<String>,
    numeric_tokens: Vec<f64>,
    nan_is_missing: bool,
    ignore_unknown: bool,
    network: Network,
    output: Output,
    examples: Vec<serde_json::Value>,
    /// The runtime's prediction for each example: the probability (binary), every
    /// class's probability (multiclass) or the value (regression).
    expected: Vec<Vec<f64>>,
    /// The bundle's conformal calibration, `(alpha, quantile)`, when it has one.
    conformal: Option<(f64, f64)>,
    /// Depth of the deepest tree; 0 for a network.
    max_tree_depth: usize,
    /// The depth of each tree, in order; empty for a network.
    tree_depths: Vec<usize>,
    /// Upper bounds on the arithmetic of one prediction.
    work: Work,
    /// Bounds of the raw outputs over every input, `[lo, hi]` per output.
    output_bounds: Vec<[f32; 2]>,
    /// Which features are numbers and which are slots of a category, for rules.
    domain: FeatureDomain,
    /// The loaded bundle, for rules.
    predictor: Predictor,
}

/// What the compiler will know about the model besides its weights: the depth of each
/// tree, the worst-case work of one prediction, and the bounds of its raw outputs over
/// every input. The bounds are computed here with the kernels the generated code calls,
/// and the generated code makes rustc's constant evaluator derive them again from the
/// constants; the two must agree bit for bit.
fn analyse(network: &Network, n_features: usize, n_outputs: usize, task: Task) -> (Vec<usize>, Work, Vec<[f32; 2]>) {
    let exponentials = match task {
        Task::Binary => 1,
        Task::Multiclass => n_outputs,
        Task::Regression => 0,
    };
    let (mut lo, mut hi) = (vec![f32::NEG_INFINITY; n_features], vec![f32::INFINITY; n_features]);
    match network {
        Network::Dense(layers) => {
            for l in layers {
                let (mut lo_out, mut hi_out) = (vec![0f32; l.out_dim], vec![0f32; l.out_dim]);
                kernels::dense_bounds(&l.weight, &l.bias, &lo, &hi, l.relu, &mut lo_out, &mut hi_out);
                (lo, hi) = (lo_out, hi_out);
            }
            let work = Work {
                comparisons: layers.iter().filter(|l| l.relu).map(|l| l.out_dim).sum(),
                multiply_adds: layers.iter().map(|l| l.in_dim * l.out_dim).sum(),
                additions: layers.iter().map(|l| l.out_dim).sum(),
                exponentials,
            };
            (Vec::new(), work, lo.into_iter().zip(hi).map(|(a, b)| [a, b]).collect())
        }
        Network::Trees {
            roots,
            feature,
            threshold,
            left,
            right,
            value,
            base,
            sum_in_f32,
            ..
        } => {
            // The runtime verified that children come after their parent, inside the
            // same tree, so one pass in index order gives every node's depth.
            let mut depths = Vec::with_capacity(roots.len());
            for t in 0..roots.len() {
                let start = roots[t] as usize;
                let end = roots.get(t + 1).map_or(feature.len(), |&r| r as usize);
                let mut depth = vec![0usize; end - start];
                let mut deepest = 0;
                for i in start..end {
                    let d = depth[i - start];
                    if feature[i] >= 0 {
                        depth[left[i] as usize - start] = d + 1;
                        depth[right[i] as usize - start] = d + 1;
                    } else {
                        deepest = deepest.max(d);
                    }
                }
                depths.push(deepest);
            }
            let trees = kernels::Trees {
                roots,
                feature,
                threshold,
                left,
                right,
                value,
                base,
            };
            let (mut lo_out, mut hi_out) = (vec![0f32; n_outputs], vec![0f32; n_outputs]);
            if *sum_in_f32 {
                trees.bounds_in_f32(&lo, &hi, &mut vec![false; feature.len()], &mut lo_out, &mut hi_out);
            } else {
                trees.bounds(&lo, &hi, &mut vec![false; feature.len()], &mut lo_out, &mut hi_out);
            }
            let work = Work {
                comparisons: depths.iter().sum(),
                multiply_adds: 0,
                additions: roots.len(),
                exponentials,
            };
            (depths, work, lo_out.into_iter().zip(hi_out).map(|(a, b)| [a, b]).collect())
        }
    }
}

impl Compiled {
    fn load(dir: &Path, path: &str) -> Result<Self, String> {
        // Exactly the runtime's verification, self-check included.
        let predictor = Predictor::load(dir, LoadOptions::cpu().with_self_check(true)).map_err(|e| e.render())?;
        let b = predictor.bundle();
        let records = b
            .examples
            .records
            .iter()
            .map(Record::from_json)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| e.render())?;
        let expected = predictor
            .predict(&records)
            .map_err(|e| e.render())?
            .into_iter()
            .map(|p| match p.outcome {
                Outcome::Binary { probability, .. } => vec![probability],
                Outcome::Multiclass { probabilities, .. } => probabilities,
                Outcome::Regression { value } => vec![value],
            })
            .collect();

        let mut files = vec![dir.join("manifest.json")];
        files.extend(b.manifest.files.iter().map(|f| dir.join(&f.path)));
        if let Some(f) = files.iter().find(|f| f.to_str().is_none()) {
            return Err(format!(
                "`{}` is not valid UTF-8, and `include_bytes!` needs the path as a string",
                f.display()
            ));
        }

        let t = &b.transform;
        let ignore_unknown = t.unknown_category == UnknownCategoryPolicy::Ignore;
        // `from_values` binds every field as a local: none may shadow its parameter or
        // the constant its arms read.
        let mut field_names = Namer::new(&["values", "MISSING_TOKENS"]);
        let mut type_names = Namer::new(&[
            "Input",
            "Prediction",
            "Label",
            "Class",
            "Model",
            "FieldValue",
            "FieldInfo",
            "FieldKind",
            "InputError",
            "SelfCheckError",
            "Option",
            "Result",
        ]);
        let mut fields = Vec::new();
        for feature in &t.features {
            let name = feature.name().to_string();
            let ident = field_names.ident(&snake(&name));
            let spec = match feature {
                FeatureTransform::Numeric {
                    impute,
                    mean,
                    std,
                    offset,
                    missing,
                    ..
                } => Spec::Numeric {
                    impute: *impute,
                    mean: *mean,
                    std: *std,
                    offset: *offset,
                    refuse_missing: *missing == MissingRule::Refuse,
                },
                FeatureTransform::Categorical {
                    categories,
                    missing_slot,
                    offset,
                    raw_dtype,
                    ..
                } => {
                    let raw = raw_dtype
                        .or_else(|| b.input_schema.field(&name).map(|f| f.raw_dtype))
                        .unwrap_or(RawDtype::Text);
                    // As when the bundle was fitted: a missing value is accepted when it
                    // has a slot, or when unknown categories are ignored.
                    let nullable = *missing_slot || ignore_unknown;
                    match raw {
                        RawDtype::Text => {
                            let ty = type_names.ident(&camel(&name));
                            let mut variant_names = Namer::new(&[]);
                            // The trained categories are named first, so one called "Other"
                            // keeps that name and the catch-all becomes `Other_2`.
                            let variants = categories.iter().map(|c| variant_names.ident(&camel(c))).collect();
                            let other = ignore_unknown.then(|| {
                                let mut shown = "(other)".to_string();
                                while categories.contains(&shown) {
                                    shown.insert(0, '~');
                                }
                                (variant_names.ident("Other"), shown)
                            });
                            Spec::Text {
                                ty,
                                categories: categories.clone(),
                                variants,
                                other,
                                missing_slot: *missing_slot,
                                nullable,
                                offset: *offset,
                            }
                        }
                        RawDtype::Bool => Spec::Bool {
                            categories: categories.clone(),
                            missing_slot: *missing_slot,
                            nullable,
                            offset: *offset,
                        },
                        // Tabsmith reads every declared category as text: numbers only
                        // appear here in a bundle it did not make.
                        RawDtype::Int | RawDtype::Float => {
                            return Err(format!(
                                "`{name}` is a category coded by {} values; Tabsmith reads categories as text, \
                                 and `model!` supports text and boolean categories",
                                raw.as_str()
                            ));
                        }
                    }
                }
            };
            fields.push(Field { name, ident, spec });
        }

        let config = b.model.config();
        let network = match &b.model {
            Model::Dense(d) => Network::Dense(
                d.layers
                    .iter()
                    .map(|l| Layer {
                        in_dim: l.in_dim,
                        out_dim: l.out_dim,
                        weight: l.weight.clone(),
                        bias: l.bias.clone(),
                        relu: matches!(l.activation, Activation::Relu),
                    })
                    .collect(),
            ),
            Model::Trees(e) => {
                let spec = config.trees.as_ref().ok_or("tree model without its ensemble size")?;
                Network::Trees {
                    roots: e.roots.clone(),
                    feature: e.feature.clone(),
                    threshold: e.threshold.clone(),
                    left: e.left.clone(),
                    right: e.right.clone(),
                    value: e.value.clone(),
                    base: if spec.n_outputs == 1 {
                        vec![spec.base_score]
                    } else {
                        spec.base_scores.clone()
                    },
                    monotone: if spec.monotone.is_empty() {
                        vec![0; config.input_dim]
                    } else {
                        spec.monotone.clone()
                    },
                    sum_in_f32: spec.sum == LeafSum::F32,
                }
            }
        };
        let summary = match &network {
            Network::Dense(layers) => {
                let dims: Vec<String> = std::iter::once(config.input_dim)
                    .chain(layers.iter().map(|l| l.out_dim))
                    .map(|d| d.to_string())
                    .collect();
                let params: usize = layers.iter().map(|l| l.weight.len() + l.bias.len()).sum();
                format!("{}, dense {} ({params} parameters)", config.family, dims.join(" → "))
            }
            Network::Trees { roots, feature, .. } => format!("{}, {} trees with {} nodes", config.family, roots.len(), feature.len()),
        };

        let o = &b.output_schema;
        let output = match o.task {
            Task::Binary => Output::Binary {
                negative: o.negative_label.clone(),
                positive: o.positive_label.clone(),
                threshold: o.threshold,
            },
            Task::Multiclass => {
                let mut names = Namer::new(&[]);
                Output::Multiclass {
                    variants: o.classes.iter().map(|c| names.ident(&camel(c))).collect(),
                    classes: o.classes.clone(),
                }
            }
            Task::Regression => {
                let scale = o.target_scale.ok_or("a regression bundle without its target scale")?;
                Output::Regression {
                    mean: scale.mean,
                    std: scale.std,
                }
            }
        };

        let (tree_depths, work, output_bounds) = analyse(&network, t.output_dim, config.n_outputs(), o.task);
        let max_tree_depth = tree_depths.iter().copied().max().unwrap_or(0);
        Ok(Compiled {
            path: path.to_string(),
            files,
            bundle_id: b.manifest.bundle_id.clone(),
            task: o.task,
            target: o.target.clone(),
            summary,
            fields,
            dropped: t.dropped.iter().map(|d| d.name.clone()).collect(),
            n_features: t.output_dim,
            n_outputs: config.n_outputs(),
            conformal: b.conformal.as_ref().map(|c| (c.alpha, c.quantile)),
            max_tree_depth,
            tree_depths,
            work,
            output_bounds,
            // The runtime's missing-value tokens (`MissingTokens::new`): every token is a
            // text token, and those that parse as a number also match numbers.
            text_tokens: t.missing_tokens.clone(),
            numeric_tokens: t
                .missing_tokens
                .iter()
                .filter_map(|s| s.trim().parse::<f64>().ok())
                .filter(|f| !f.is_nan())
                .collect(),
            nan_is_missing: t.nan == NanPolicy::Missing,
            ignore_unknown,
            network,
            output,
            examples: b.examples.records.clone(),
            expected,
            domain: FeatureDomain::of(t),
            predictor: predictor.clone(),
        })
    }

    /// Audits a rule as `tabsmith audit` does with its default budget: a proof, or the reason
    /// the rule cannot be part of the module.
    fn prove(&self, text: &str) -> Result<Proved, String> {
        let rule = Rule::parse(text).map_err(|e| e.render())?;
        let audit = self.predictor.audit(&rule, &Budget::default()).map_err(|e| e.render())?;
        match audit.verdict {
            Verdict::Holds(proof) => Ok(Proved {
                text: audit.rule,
                proof,
                visited: audit.boxes,
            }),
            Verdict::Fails { witness, prediction } => {
                let got = match &prediction.outcome {
                    Outcome::Binary { probability, .. } | Outcome::Multiclass { probability, .. } => {
                        format!("label {:?} with probability {probability}", prediction.label)
                    }
                    Outcome::Regression { value } => format!("the value {value}"),
                };
                Err(format!(
                    "the rule `{}` does not hold: tabsmith-runtime predicts {got} for this input of its region:\n{}",
                    audit.rule,
                    serde_json::to_string_pretty(&witness.to_json()).unwrap_or_default()
                ))
            }
            Verdict::Undecided(why) => Err(format!(
                "the rule `{}` is undecided: {why} ({} boxes visited). Narrow its region, for example with \
                 `_ within 3 sd`, or check it with `tabsmith audit --boxes <more>`",
                audit.rule, audit.boxes
            )),
        }
    }

    /// Whether the trees become code: what the invocation asks for, otherwise as long as
    /// they stay cheap to compile. A network has no trees, so the options change nothing
    /// for it, and an invocation keeps compiling when retraining picks another model kind.
    fn tree_code(&self, requested: Option<bool>) -> Result<bool, String> {
        let Network::Trees { feature, .. } = &self.network else {
            return Ok(false);
        };
        let shallow = self.max_tree_depth <= TREE_CODE_MAX_DEPTH;
        match requested {
            Some(true) if !shallow => Err(format!(
                "`tree_code` stops at trees {TREE_CODE_MAX_DEPTH} levels deep and this bundle has one of {}; use `tree_arrays`",
                self.max_tree_depth
            )),
            Some(code) => Ok(code),
            None => Ok(shallow && feature.len() <= TREE_CODE_MAX_NODES),
        }
    }

    fn generate(&self, name: &Ident, kani: bool, tree_code: bool, rules: &[Proved]) -> TokenStream {
        let kani_derive = if kani {
            quote!(#[cfg_attr(kani, derive(kani::Arbitrary))])
        } else {
            quote!()
        };
        let track = self.files.iter().map(|p| {
            let path = p.to_str().expect("checked to be UTF-8 when loaded");
            quote!(
                const _: &[u8] = include_bytes!(#path);
            )
        });
        let bundle_id = &self.bundle_id;
        let task = self.task.as_str();
        let target = &self.target;
        let n_fields = self.fields.len();
        let n_features = self.n_features;
        let n_outputs = self.n_outputs;
        let text_tokens = &self.text_tokens;
        let n_text_tokens = text_tokens.len();
        let numeric_tokens: Vec<TokenStream> = self.numeric_tokens.iter().map(|&v| f64_lit(v)).collect();
        let n_numeric_tokens = numeric_tokens.len();

        let mut enums = Vec::new();
        let mut input_fields = Vec::new();
        let mut encode = Vec::new();
        let mut from_values = Vec::new();
        let mut values = Vec::new();
        let mut infos = Vec::new();
        for (i, f) in self.fields.iter().enumerate() {
            let (decl, ty) = self.field_type(f, &kani_derive);
            enums.extend(decl);
            let doc = self.field_doc(f);
            let ident = &f.ident;
            input_fields.push(quote!(#[doc = #doc] pub #ident: #ty,));
            encode.push(self.encode(f));
            let parse = self.parse_value(f, i);
            from_values.push(quote!(let #ident = #parse;));
            values.push(self.to_value(f));
            infos.push(self.info(f));
        }
        let idents: Vec<&Ident> = self.fields.iter().map(|f| &f.ident).collect();

        let (output_types, decode) = self.output();
        let (model_consts, outputs_body, batch_body) = self.network(tree_code);
        let outputs_doc = match &self.network {
            Network::Trees { .. } if tree_code => {
                "The model's raw outputs for one row of features: every tree is a function of nested `if`s."
            }
            Network::Trees { .. } => "The model's raw outputs for one row of features: a walk through every tree's arrays.",
            Network::Dense(_) => "The model's raw outputs for one row of features: the dense layers.",
        };
        let examples: Vec<TokenStream> = self.examples.iter().map(|e| self.example(e)).collect();
        let n_examples = examples.len();
        let self_check = self.self_check();
        let analysis = self.analysis();
        let conformal = self.conformal();
        let model_impl = self.model_impl();
        let proofs = if kani { self.proofs(name, tree_code) } else { quote!() };
        let (rules_code, rules_doc) = self.rules(rules);

        let dropped = if self.dropped.is_empty() {
            String::new()
        } else {
            format!(
                "\n\nFields the model does not use (dropped when it was fitted) are not part of [`Input`]: {}.",
                self.dropped.iter().map(|d| format!("`{d}`")).collect::<Vec<_>>().join(", ")
            )
        };
        let module_doc = format!(
            "The Tabsmith bundle `{bundle_id}` (`{}`), compiled into Rust code: a {task} model of `{target}` ({}), \
             {n_fields} input fields → {n_features} features → {n_outputs} output(s).{dropped}\n\n\
             Generated by `tabsmith_macros::model!`. The bundle was verified when this crate was compiled, and \
             [`predict`] computes exactly what `tabsmith-runtime` computes for the same input, bit for bit. \
             It is a `const fn`: the compiler ran it on the bundle's examples while this crate compiled and \
             got the runtime's bits, and [`self_check`] repeats that on the device. `no_std`, no allocation.\n\n\
             The compiler also knows the model's cost ([`WORST_CASE`]) and the bounds of its raw outputs \
             ([`OUTPUT_BOUNDS`] for every input, [`output_bounds`] for a box of feature values){}; \
             [`Model`] implements the `tabsmith_embedded::Model` trait.{rules_doc}",
            self.path,
            self.summary,
            if kani {
                ", and the module carries its own proofs for the Kani model checker ([`proofs`])"
            } else {
                ""
            }
        );

        quote! {
            #[doc = #module_doc]
            #[allow(clippy::all, clippy::pedantic, non_camel_case_types, non_snake_case, dead_code, unused_variables)]
            pub mod #name {
                use ::tabsmith_embedded::kernels;
                pub use ::tabsmith_embedded::{FieldInfo, FieldKind, FieldValue, InputError, SelfCheckError};

                #(#track)*

                /// The id of the bundle this module was compiled from.
                pub const BUNDLE_ID: &str = #bundle_id;
                /// `binary`, `multiclass` or `regression`.
                pub const TASK: &str = #task;
                /// The column the model predicts.
                pub const TARGET: &str = #target;
                /// Number of input fields.
                pub const N_FIELDS: usize = #n_fields;
                /// Number of features the model reads (the width of [`features`]).
                pub const N_FEATURES: usize = #n_features;
                /// Number of model outputs (logits or the standardised value).
                pub const N_OUTPUTS: usize = #n_outputs;
                /// The input fields, in the order of [`Input::from_values`] and [`Input::values`].
                pub const FIELDS: [FieldInfo; N_FIELDS] = [#(#infos),*];
                /// Text values that mean "missing" (the project's `data.null_values`).
                pub const MISSING_TOKENS: [&str; #n_text_tokens] = [#(#text_tokens),*];
                const NUMERIC_TOKENS: [f64; #n_numeric_tokens] = [#(#numeric_tokens),*];

                #(#enums)*

                /// One record, with the types the model's training data had.
                #[derive(Debug, Clone, Copy, PartialEq)]
                #kani_derive
                pub struct Input {
                    #(#input_fields)*
                }

                impl Input {
                    /// An input from raw values in the order of [`FIELDS`], with the rules the
                    /// runtime applies to JSON records: a missing-value token or `Missing` is a
                    /// missing value, an integer is accepted for a number, and a value of another
                    /// type is refused.
                    pub fn from_values(values: &[FieldValue<'_>; N_FIELDS]) -> ::core::result::Result<Input, InputError> {
                        #(#from_values)*
                        Ok(Input { #(#idents),* })
                    }

                    /// The input as raw values, in the order of [`FIELDS`].
                    pub fn values(&self) -> [FieldValue<'static>; N_FIELDS] {
                        [#(#values),*]
                    }
                }

                #output_types

                /// The model's features for one input: the frozen transform of the bundle.
                pub const fn features(input: &Input) -> ::core::result::Result<[f32; N_FEATURES], InputError> {
                    let mut x = [0f32; N_FEATURES];
                    #(#encode)*
                    Ok(x)
                }

                #model_consts

                #[doc = #outputs_doc]
                pub const fn outputs(x: &[f32; N_FEATURES]) -> [f32; N_OUTPUTS] {
                    #outputs_body
                }

                /// The raw outputs of many rows into `out`, bit for bit what [`outputs`] gives
                /// for each row. It stops at the end of the shorter slice and returns the
                /// number of rows it computed.
                pub fn outputs_batch(x: &[[f32; N_FEATURES]], out: &mut [[f32; N_OUTPUTS]]) -> usize {
                    #batch_body
                    x.len().min(out.len())
                }

                #decode

                /// Predicts one input. `const`, so the compiler can predict while it compiles:
                /// `const P: Prediction = match predict(&INPUT) { Ok(p) => p, Err(_) => panic!() };`
                pub const fn predict(input: &Input) -> ::core::result::Result<Prediction, InputError> {
                    match features(input) {
                        Ok(x) => Ok(decode(&outputs(&x))),
                        Err(e) => Err(e),
                    }
                }

                /// Predicts one row of raw values, in the order of [`FIELDS`].
                pub fn predict_values(values: &[FieldValue<'_>; N_FIELDS]) -> ::core::result::Result<Prediction, InputError> {
                    predict(&Input::from_values(values)?)
                }

                /// The bundle's example records (`examples/input.json`): synthetic values built
                /// from the training statistics, never training rows.
                pub const EXAMPLES: [Input; #n_examples] = [#(#examples),*];

                #self_check

                #analysis

                #conformal

                #model_impl

                #rules_code

                #proofs
            }
        }
    }

    /// `RULES`, the domain the proofs cut, and for every proof small enough a constant that
    /// makes the compiler walk it: each box it closes must hold no input, or have
    /// [`output_bounds`] that satisfy the claim. Also the paragraph of the module's
    /// documentation that lists the rules.
    fn rules(&self, proved: &[Proved]) -> (TokenStream, String) {
        if proved.is_empty() {
            return (quote!(), String::new());
        }
        let r = quote!(::tabsmith_embedded::rules);
        let per_box = match &self.network {
            Network::Trees { feature, .. } => feature.len(),
            Network::Dense(layers) => layers.iter().map(|l| l.weight.len() + l.bias.len()).sum(),
        };
        let mut checks = Vec::new();
        let mut lines = Vec::new();
        for p in proved {
            let boxes = p.proof.boxes();
            let by_compiler = boxes.saturating_mul(per_box.max(1)) <= RULE_CHECK_MAX_WORK;
            lines.push(format!(
                "- `{}`: a proof of {boxes} box{} ({} visited){}",
                p.text,
                if boxes == 1 { "" } else { "es" },
                p.visited,
                if by_compiler {
                    ", which the compiler checks again while this crate compiles"
                } else {
                    ", checked again when the macro expanded: too large for the compiler"
                }
            ));
            if by_compiler {
                checks.push(self.rule_check(p));
            }
        }
        let texts: Vec<&str> = proved.iter().map(|p| p.text.as_str()).collect();
        let n = texts.len();
        let category = self.domain.category.iter().map(|&c| {
            if c == NUMERIC {
                quote!(#r::NUMERIC)
            } else {
                let c = Literal::u32_unsuffixed(c);
                quote!(#c)
            }
        });
        let g = self.domain.slots.len();
        let slots = self.domain.slots.iter().map(|[a, b]| {
            let (a, b) = (Literal::u32_unsuffixed(*a), Literal::u32_unsuffixed(*b));
            quote!([#a, #b])
        });
        let doc = format!(
            "\n\n**Rules.** Every input of each rule's region gets a prediction that meets the rule's claim: \
             the macro proved it when it expanded (`tabsmith_runtime::audit`, as `tabsmith audit` does), and a \
             rule that fails or stays undecided does not compile. A proof cuts the region into boxes of feature \
             values, and the bounds of every box settle the claim.\n{}",
            lines.join("\n")
        );
        let code = quote! {
            /// The rules this module was compiled with, in their canonical text: each holds for every
            /// input of its region (see the module's documentation).
            pub const RULES: [&str; #n] = [#(#texts),*];

            /// Which features are numbers and which are slots of a category, for the proofs of [`RULES`].
            const RULE_CATEGORY: [u32; N_FEATURES] = [#(#category),*];
            const RULE_SLOTS: [[u32; 2]; #g] = [#(#slots),*];
            const RULE_DOMAIN: #r::Domain<'static> = #r::Domain {
                category: &RULE_CATEGORY,
                slots: &RULE_SLOTS,
            };

            #(#checks)*
        };
        (code, doc)
    }

    /// The constant that makes the compiler check one rule's proof.
    fn rule_check(&self, p: &Proved) -> TokenStream {
        let r = quote!(::tabsmith_embedded::rules);
        let steps = p.proof.steps.iter().map(|s| match *s {
            Step::Split { feature, threshold } => {
                let (f, t) = (Literal::u32_unsuffixed(feature), f32_lit(threshold));
                quote!(#r::Step::Split { feature: #f, threshold: #t })
            }
            Step::Close => quote!(#r::Step::Close),
        });
        let n_steps = p.proof.steps.len();
        let lo = p.proof.lo.iter().map(|&v| f32_lit(v));
        let hi = p.proof.hi.iter().map(|&v| f32_lit(v));
        let none = &p.proof.none;
        let g = none.len();
        let depth = p.proof.depth;
        let claim = match p.proof.claim {
            Claim::ProbabilityAtMost(v) => {
                let v = f64_lit(v);
                quote!(#r::Claim::ProbabilityAtMost(#v))
            }
            Claim::ProbabilityAtLeast(v) => {
                let v = f64_lit(v);
                quote!(#r::Claim::ProbabilityAtLeast(#v))
            }
            Claim::Positive { threshold } => {
                let t = f64_lit(threshold);
                quote!(#r::Claim::Positive { threshold: #t })
            }
            Claim::Negative { threshold } => {
                let t = f64_lit(threshold);
                quote!(#r::Claim::Negative { threshold: #t })
            }
            Claim::ValueAtMost { limit, mean, std } | Claim::ValueAtLeast { limit, mean, std } => {
                let (l, m, s) = (f64_lit(limit), f64_lit(mean), f64_lit(std));
                if matches!(p.proof.claim, Claim::ValueAtMost { .. }) {
                    quote!(#r::Claim::ValueAtMost { limit: #l, mean: #m, std: #s })
                } else {
                    quote!(#r::Claim::ValueAtLeast { limit: #l, mean: #m, std: #s })
                }
            }
            Claim::Class(k) => {
                let k = Literal::usize_unsuffixed(k);
                quote!(#r::Claim::Class(#k))
            }
            Claim::NotClass(k) => {
                let k = Literal::usize_unsuffixed(k);
                quote!(#r::Claim::NotClass(#k))
            }
        };
        let open = format!(
            "the compiler finds a box of the proof of the rule `{}` whose bounds do not settle it",
            p.text
        );
        let malformed = format!("the proof of the rule `{}` is not a tree of splits", p.text);
        quote! {
            // The compiler walks the rule's proof with its own floating point: every box it closes
            // holds no input, or has output bounds that satisfy the claim.
            #[allow(long_running_const_eval)]
            const _: () = {
                const STEPS: [#r::Step; #n_steps] = [#(#steps),*];
                let mut lo: [f32; N_FEATURES] = [#(#lo),*];
                let mut hi: [f32; N_FEATURES] = [#(#hi),*];
                let mut none: [bool; #g] = [#(#none),*];
                let mut saved_lo = [0f32; N_FEATURES * #depth];
                let mut saved_hi = [0f32; N_FEATURES * #depth];
                let mut saved_none = [false; #g * #depth];
                let mut saved_step = [0usize; #depth];
                let mut walk = #r::Walk::new(RULE_DOMAIN, &STEPS, &mut saved_lo, &mut saved_hi, &mut saved_none, &mut saved_step);
                let claim = #claim;
                let mut p = [0f64; N_OUTPUTS];
                loop {
                    match walk.next(&mut lo, &mut hi, &mut none) {
                        Ok(true) => {
                            if !#r::is_empty(&RULE_DOMAIN, &lo, &hi, &none) {
                                let b = output_bounds(&lo, &hi);
                                let (mut lo_out, mut hi_out) = ([0f32; N_OUTPUTS], [0f32; N_OUTPUTS]);
                                let mut o = 0;
                                while o < N_OUTPUTS {
                                    lo_out[o] = b[o][0];
                                    hi_out[o] = b[o][1];
                                    o += 1;
                                }
                                assert!(claim.holds_within(&lo_out, &hi_out, &mut p), #open);
                            }
                        }
                        Ok(false) => break,
                        Err(_) => panic!(#malformed),
                    }
                }
            };
        }
    }

    /// `WORST_CASE`, `output_bounds`, `feature_box`, `OUTPUT_BOUNDS` and the bounds
    /// derived from them for the task.
    fn analysis(&self) -> TokenStream {
        let Work {
            comparisons,
            multiply_adds,
            additions,
            exponentials,
        } = self.work;
        let n_outputs = self.n_outputs;
        let pack = |lo: TokenStream, hi: TokenStream| {
            quote! {
                let mut out = [[0f32; 2]; N_OUTPUTS];
                let mut o = 0;
                while o < N_OUTPUTS {
                    out[o] = [#lo[o], #hi[o]];
                    o += 1;
                }
                out
            }
        };
        let (work_doc, bounds_fn) = match &self.network {
            Network::Dense(layers) => {
                let mut steps = Vec::new();
                for (i, l) in layers.iter().enumerate() {
                    let (w, b) = (format_ident!("L{i}_WEIGHT"), format_ident!("L{i}_BIAS"));
                    let (lo, hi) = (format_ident!("lo{i}"), format_ident!("hi{i}"));
                    let (lo_in, hi_in) = if i == 0 {
                        (quote!(lo), quote!(hi))
                    } else {
                        let (p, q) = (format_ident!("lo{}", i - 1), format_ident!("hi{}", i - 1));
                        (quote!(&#p), quote!(&#q))
                    };
                    let (width, relu) = (l.out_dim, l.relu);
                    steps.push(quote! {
                        let (mut #lo, mut #hi) = ([0f32; #width], [0f32; #width]);
                        kernels::dense_bounds(&#w, &#b, #lo_in, #hi_in, #relu, &mut #lo, &mut #hi);
                    });
                }
                let last = layers.len() - 1;
                let (lo, hi) = (format_ident!("lo{last}"), format_ident!("hi{last}"));
                let packed = pack(quote!(#lo), quote!(#hi));
                let doc = "Upper bounds on the arithmetic of one prediction (`outputs` and `decode`), from the \
                           model's shape; for a dense network they are the exact count. No loop depends on the data.";
                let f = quote! {
                    /// Bounds of the raw [`outputs`] over every row of features with `lo[j] <= x[j] <= hi[j]`,
                    /// `[lo, hi]` per output: interval arithmetic through the layers, widened by the worst
                    /// rounding error of the kernels. Sound (no row of the box leaves the bounds) but not
                    /// tight. An unbounded feature gives unbounded outputs; [`feature_box`] builds a bounded
                    /// box. `const`, so `const B: [[f32; 2]; N_OUTPUTS] = output_bounds(&LO, &HI);` costs
                    /// nothing at run time.
                    pub const fn output_bounds(lo: &[f32; N_FEATURES], hi: &[f32; N_FEATURES]) -> [[f32; 2]; N_OUTPUTS] {
                        #(#steps)*
                        #packed
                    }
                };
                (doc, f)
            }
            Network::Trees { feature, sum_in_f32, .. } => {
                let n_nodes = feature.len();
                let depth = self.max_tree_depth;
                let packed = pack(quote!(lo_out), quote!(hi_out));
                let bounds = if *sum_in_f32 { quote!(bounds_in_f32) } else { quote!(bounds) };
                let doc = "Upper bounds on the arithmetic of one prediction (`outputs` and `decode`), from the \
                           model's shape: the comparisons are the sum of the trees' depths. No loop depends on the data.";
                let f = quote! {
                    /// Depth of the deepest tree: the most comparisons one walk can take.
                    pub const MAX_TREE_DEPTH: usize = #depth;

                    /// Bounds of the raw [`outputs`] over every row of features with `lo[j] <= x[j] <= hi[j]`,
                    /// `[lo, hi]` per output (a NaN bound lets the feature be NaN). Each tree contributes the
                    /// smallest and the largest leaf the box can reach, summed as `outputs` sums; rounding is
                    /// monotone, so the bounds are sound with no error margin, though not necessarily reached.
                    /// `const`, so `const B: [[f32; 2]; N_OUTPUTS] = output_bounds(&LO, &HI);` costs nothing
                    /// at run time.
                    pub const fn output_bounds(lo: &[f32; N_FEATURES], hi: &[f32; N_FEATURES]) -> [[f32; 2]; N_OUTPUTS] {
                        let mut reachable = [false; #n_nodes];
                        let (mut lo_out, mut hi_out) = ([0f32; N_OUTPUTS], [0f32; N_OUTPUTS]);
                        TREES.#bounds(lo, hi, &mut reachable, &mut lo_out, &mut hi_out);
                        #packed
                    }
                };
                (doc, f)
            }
        };
        let numeric_offsets: Vec<usize> = self
            .fields
            .iter()
            .filter_map(|f| match f.spec {
                Spec::Numeric { offset, .. } => Some(offset),
                _ => None,
            })
            .collect();
        let box_fn = if numeric_offsets.is_empty() {
            quote! {
                /// A box of feature values for [`output_bounds`]: every category slot anywhere between 0
                /// and 1 (this model has no numeric field, so `sd` is unused). `[lo, hi]`.
                pub const fn feature_box(sd: f32) -> [[f32; N_FEATURES]; 2] {
                    let _ = sd;
                    [[0f32; N_FEATURES], [1f32; N_FEATURES]]
                }
            }
        } else {
            quote! {
                /// A box of feature values for [`output_bounds`]: every numeric feature within `sd`
                /// standard deviations of its training mean (the features are standardised), every
                /// category slot anywhere between 0 and 1. `[lo, hi]`.
                pub const fn feature_box(sd: f32) -> [[f32; N_FEATURES]; 2] {
                    let mut lo = [0f32; N_FEATURES];
                    let mut hi = [1f32; N_FEATURES];
                    #(lo[#numeric_offsets] = -sd; hi[#numeric_offsets] = sd;)*
                    [lo, hi]
                }
            }
        };
        let expected = self.output_bounds.iter().map(|[a, b]| {
            let (a, b) = (Literal::u32_unsuffixed(a.to_bits()), Literal::u32_unsuffixed(b.to_bits()));
            quote!([#a, #b])
        });
        let shown: Vec<String> = self
            .output_bounds
            .iter()
            .map(|[a, b]| format!("[{}, {}]", short(f64::from(*a)), short(f64::from(*b))))
            .collect();
        let bounds_doc = format!(
            "Bounds of the raw [`outputs`] over every possible input, `[lo, hi]` per output: \
             [`output_bounds`] of the unbounded box, evaluated by the compiler. For this model: {}.",
            shown.join(", ")
        );
        // The derived bounds come from the kernels `decode` calls, evaluated by the compiler,
        // and must agree with the same kernels run when the macro expanded.
        let derived = match &self.output {
            Output::Binary { .. } => {
                let [lo, hi] = self.output_bounds[0];
                let (p, q) = (kernels::sigmoid(f64::from(lo)), kernels::sigmoid(f64::from(hi)));
                let (pb, qb) = (Literal::u64_unsuffixed(p.to_bits()), Literal::u64_unsuffixed(q.to_bits()));
                let doc = format!(
                    "Bounds of the probability over every possible input: the logistic function of \
                     [`OUTPUT_BOUNDS`] ({} to {}), as [`decode`] computes it. The computed logistic function \
                     never decreases from one `f32` logit to the next, which the Tabsmith repository checks for \
                     all 2^32 of them, so no input leaves these bounds.",
                    short(p),
                    short(q)
                );
                quote! {
                    #[doc = #doc]
                    pub const PROBABILITY_BOUNDS: [f64; 2] =
                        [kernels::sigmoid(OUTPUT_BOUNDS[0][0] as f64), kernels::sigmoid(OUTPUT_BOUNDS[0][1] as f64)];

                    const _: () = assert!(
                        PROBABILITY_BOUNDS[0].to_bits() == #pb && PROBABILITY_BOUNDS[1].to_bits() == #qb,
                        "the probability bounds differ from the ones computed when the macro expanded"
                    );
                }
            }
            Output::Regression { mean, std } => {
                let [lo, hi] = self.output_bounds[0];
                let (p, q) = (kernels::unscale(lo, *mean, *std), kernels::unscale(hi, *mean, *std));
                let (pb, qb) = (Literal::u64_unsuffixed(p.to_bits()), Literal::u64_unsuffixed(q.to_bits()));
                let doc = format!(
                    "Bounds of the predicted value over every possible input, in the target's units: \
                     [`OUTPUT_BOUNDS`] unscaled ({} to {}), as [`decode`] computes it. Every operation rounds \
                     monotonically, so no input leaves these bounds.",
                    short(p),
                    short(q)
                );
                quote! {
                    #[doc = #doc]
                    pub const VALUE_BOUNDS: [f64; 2] = [
                        kernels::unscale(OUTPUT_BOUNDS[0][0], TARGET_MEAN, TARGET_STD),
                        kernels::unscale(OUTPUT_BOUNDS[0][1], TARGET_MEAN, TARGET_STD),
                    ];

                    const _: () = assert!(
                        VALUE_BOUNDS[0].to_bits() == #pb && VALUE_BOUNDS[1].to_bits() == #qb,
                        "the value bounds differ from the ones computed when the macro expanded"
                    );
                }
            }
            Output::Multiclass { .. } => quote!(),
        };
        quote! {
            #[doc = #work_doc]
            pub const WORST_CASE: ::tabsmith_embedded::Work = ::tabsmith_embedded::Work {
                comparisons: #comparisons,
                multiply_adds: #multiply_adds,
                additions: #additions,
                exponentials: #exponentials,
            };

            #bounds_fn

            #box_fn

            #[doc = #bounds_doc]
            pub const OUTPUT_BOUNDS: [[f32; 2]; N_OUTPUTS] =
                output_bounds(&[f32::NEG_INFINITY; N_FEATURES], &[f32::INFINITY; N_FEATURES]);

            // The host that expanded `model!` computed the same bounds with the same kernels:
            // the constant evaluator must agree bit for bit.
            const _: () = {
                const EXPECTED: [[u32; 2]; #n_outputs] = [#(#expected),*];
                let mut o = 0;
                while o < N_OUTPUTS {
                    assert!(
                        OUTPUT_BOUNDS[o][0].to_bits() == EXPECTED[o][0] && OUTPUT_BOUNDS[o][1].to_bits() == EXPECTED[o][1],
                        "the output bounds differ from the ones computed when the macro expanded"
                    );
                    o += 1;
                }
            };

            #derived
        }
    }

    /// `CONFORMAL`, and the interval or prediction set it gives a prediction.
    fn conformal(&self) -> TokenStream {
        let value = match self.conformal {
            Some((alpha, quantile)) => {
                let (a, q) = (f64_lit(alpha), f64_lit(quantile));
                quote!(Some(::tabsmith_embedded::Conformal { alpha: #a, quantile: #q }))
            }
            None => quote!(None),
        };
        let doc = format!(
            "The bundle's split conformal calibration (`conformal.json`), or `None` when it has none{}. \
             On rows distributed like the calibration rows (the validation split, which also selected the \
             model, so this is approximate), the interval or set below holds the truth with probability \
             about `1 − alpha`.",
            match self.conformal {
                Some((alpha, quantile)) => format!(": here alpha {} and quantile {}", short(alpha), short(quantile)),
                None => String::new(),
            }
        );
        let per_task = match &self.output {
            Output::Regression { .. } => quote! {
                /// `[value − quantile, value + quantile]`, in the target's units; `None` without a calibration.
                pub const fn interval(p: &Prediction) -> Option<[f64; 2]> {
                    match CONFORMAL {
                        Some(c) => Some([p.value - c.quantile, p.value + c.quantile]),
                        None => None,
                    }
                }
            },
            Output::Binary { .. } => quote! {
                /// Whether `[Label::Negative, Label::Positive]` are in the prediction set: a label is in
                /// it when its probability is at least `1 − quantile`. The set can be empty or hold
                /// both. `None` without a calibration.
                pub const fn prediction_set(p: &Prediction) -> Option<[bool; 2]> {
                    match CONFORMAL {
                        Some(c) => Some([1.0 - p.probability >= 1.0 - c.quantile, p.probability >= 1.0 - c.quantile]),
                        None => None,
                    }
                }
            },
            Output::Multiclass { .. } => quote! {
                /// Whether each class of [`Class::ALL`] is in the prediction set: a class is in it when
                /// its probability is at least `1 − quantile`. `None` without a calibration.
                pub const fn prediction_set(p: &Prediction) -> Option<[bool; N_OUTPUTS]> {
                    match CONFORMAL {
                        Some(c) => {
                            let mut set = [false; N_OUTPUTS];
                            let mut k = 0;
                            while k < N_OUTPUTS {
                                set[k] = p.probabilities[k] >= 1.0 - c.quantile;
                                k += 1;
                            }
                            Some(set)
                        }
                        None => None,
                    }
                }
            },
        };
        quote! {
            #[doc = #doc]
            pub const CONFORMAL: Option<::tabsmith_embedded::Conformal> = #value;

            #per_task
        }
    }

    fn model_impl(&self) -> TokenStream {
        quote! {
            /// [`predict`] as a value: the unit struct implements `tabsmith_embedded::Model`, so
            /// code written against the trait takes this model or the runtime's `Predictor` alike.
            #[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
            pub struct Model;

            impl ::tabsmith_embedded::Model for Model {
                type Input = Input;
                type Prediction = Prediction;
                type Error = InputError;

                fn predict(&self, input: &Input) -> ::core::result::Result<Prediction, InputError> {
                    predict(input)
                }
            }
        }
    }

    /// The harnesses for the Kani model checker: prediction never panics, for every
    /// possible input, proved stage by stage with loop bounds from the model's shape.
    /// For trees also, tree by tree, that the generated code is the walk over the arrays
    /// and that every row of a box stays inside the range `output_bounds` adds up.
    fn proofs(&self, name: &Ident, tree_code: bool) -> TokenStream {
        let features = format_ident!("{name}_features_are_total");
        let decode = format_ident!("{name}_decode_is_total");
        // `kernels::numeric` scans the numeric missing-value tokens; nothing else in
        // `features` loops.
        let u_features = Literal::usize_unsuffixed(self.numeric_tokens.len() + 2);
        let chain = self.tree_proofs(name, tree_code);
        let outputs = match &self.network {
            Network::Trees { .. } if tree_code => {
                let total = format_ident!("{name}_outputs_are_total");
                quote! {
                    /// Every tree at once, for any feature values: comparisons with an array
                    /// at indices the compiler has checked, and additions. Generated code has
                    /// no loop to bound and no index that depends on the data.
                    #[kani::proof]
                    fn #total() {
                        let x: [f32; N_FEATURES] = kani::any();
                        let _ = outputs(&x);
                    }
                }
            }
            Network::Trees { .. } => {
                let walk = format_ident!("{name}_tree_walk_is_total");
                let u = Literal::usize_unsuffixed(self.max_tree_depth + 2);
                quote! {
                    /// Walking any tree with any feature values stays inside the arrays and ends at
                    /// a leaf. `outputs` adds one walk per tree to each output's starting score:
                    /// `value` has one entry per node, and the compiler checks that there is one
                    /// starting score per output, so with every walk in range the sum cannot panic
                    /// either. (All the walks in one proof need more memory than a CI runner has.)
                    #[kani::proof]
                    #[kani::unwind(#u)]
                    fn #walk() {
                        let t: usize = kani::any();
                        kani::assume(t < TREES.roots.len());
                        let x: [f32; N_FEATURES] = kani::any();
                        let leaf = TREES.leaf(t, &x);
                        assert!(TREES.feature[leaf] < 0);
                    }
                }
            }
            Network::Dense(layers) => {
                let layers_total = format_ident!("{name}_layers_are_total");
                // The loops of `dense` and `dot`: the outputs of a layer, the chunks of
                // eight of its inputs, the eight of a chunk, and the tail.
                let longest = layers
                    .iter()
                    .map(|l| l.out_dim.max(l.in_dim / 8).max(8).max(l.in_dim % 8))
                    .max()
                    .unwrap_or(0);
                let u = Literal::usize_unsuffixed(longest + 2);
                quote! {
                    /// The dense layers stay inside their weight arrays, for any feature values.
                    /// No check depends on the arithmetic, which Kani therefore slices away.
                    #[kani::proof]
                    #[kani::unwind(#u)]
                    fn #layers_total() {
                        let x: [f32; N_FEATURES] = kani::any();
                        let _ = outputs(&x);
                    }
                }
            }
        };
        let u_decode = if self.task == Task::Multiclass {
            let u = Literal::usize_unsuffixed(self.n_outputs + 2);
            quote!(#[kani::unwind(#u)])
        } else {
            quote!()
        };
        let trees_doc = match &self.network {
            Network::Dense(_) => String::new(),
            Network::Trees { .. } => format!(
                "\n\nTrees get two more proofs each, small because each takes one tree. \
                 `{name}_tree_<t>_is_its_walk`: the code of tree `t` gives the same bits as the walk through \
                 the arrays, the walk the runtime takes, for every row of features (NaN and infinities \
                 included). `{name}_tree_<t>_stays_in_its_range`: every row of every box of feature values \
                 reaches a leaf of that code inside `TREES.leaf_range(t, ..)`, the range that [`output_bounds`] \
                 adds up for the box. Each output adds its trees in the order of the runtime's kernel, so the \
                 model predicts what the runtime predicts for every input, not only for [`EXAMPLES`]. Rounded \
                 addition never puts two sums in the wrong order, which tabsmith-embedded proves with Kani too, \
                 so `output_bounds` holds for every row of its box.{}",
                if tree_code {
                    ""
                } else {
                    " These trees are kept as arrays: their code exists only in this module, as the link between \
                     the walk and its range. A tree too deep for code has neither proof."
                }
            ),
        };
        let doc = format!(
            "Proofs for the [Kani](https://github.com/model-checking/kani) model checker (`cargo kani`): \
             prediction never panics, for every possible input. `predict` is `decode(&outputs(&features(input)?))`, \
             and each stage is proved total for every value of its argument's type, a superset of what the \
             stage before it can return: any number (NaN and infinities included), any category, any missing \
             value. Proving the stages apart keeps each proof small. The loop bounds come from the model's \
             shape.{trees_doc}\n\nHarness names start with `{name}_`, so several models can share a crate; \
             `examples/embedded-predict` in the Tabsmith repository shows the recommended settings."
        );
        quote! {
            #[doc = #doc]
            #[cfg(kani)]
            pub mod proofs {
                use super::*;

                /// Encoding any input returns features or an `InputError`.
                #[kani::proof]
                #[kani::unwind(#u_features)]
                fn #features() {
                    let input: Input = kani::any();
                    let _ = features(&input);
                }

                #outputs

                /// The decision for any model output: the logistic function or softmax (and
                /// its `exp`), or the unscaling, never panics.
                #[kani::proof]
                #u_decode
                fn #decode() {
                    let z: [f32; N_OUTPUTS] = kani::any();
                    let _ = decode(&z);
                }

                #chain
            }
        }
    }

    /// The proofs of each tree: that its code gives the same bits as the walk over the
    /// arrays, and that every row of a box reaches a leaf inside the tree's `leaf_range` for
    /// that box. Each proof takes one tree with a constant index, so the proofs stay small:
    /// with a symbolic tree, the range proof of the 14 trees of `binary-gbdt` ran out of
    /// 14 GB. The range is proved on the code, where every index is a constant: on the
    /// walk, whose indices depend on the comparisons, one tree took 82 s and over 7 GB
    /// instead of 5 s and 130 MB. Trees kept as arrays therefore get their code here too,
    /// for the proofs only, and the walk inherits the range through the first proof. A tree
    /// too deep for code (`TREE_CODE_MAX_DEPTH`) gets neither proof.
    fn tree_proofs(&self, name: &Ident, tree_code: bool) -> TokenStream {
        let Network::Trees {
            roots,
            feature,
            threshold,
            left,
            right,
            value,
            base,
            ..
        } = &self.network
        else {
            return quote!();
        };
        let trees = kernels::Trees {
            roots,
            feature,
            threshold,
            left,
            right,
            value,
            base,
        };
        let mut proofs = Vec::new();
        for (t, &depth) in self.tree_depths.iter().enumerate() {
            if depth > TREE_CODE_MAX_DEPTH {
                continue;
            }
            let start = roots[t] as usize;
            let end = roots.get(t + 1).map_or(feature.len(), |&r| r as usize);
            let nodes = end - start;
            let (index, function) = (Literal::usize_unsuffixed(t), format_ident!("tree_{t}"));
            if !tree_code {
                // The same function `model!` generates for trees as code.
                let body = tree_expr(&trees, start);
                proofs.push(quote! {
                    const fn #function(x: &[f32; N_FEATURES]) -> f32 {
                        #body
                    }
                });
            }
            let walk = format_ident!("{name}_tree_{t}_is_its_walk");
            let u_walk = Literal::usize_unsuffixed(depth + 2);
            let walk_doc = if tree_code {
                format!(
                    "Tree {t} as generated code gives the same bits as the walk through its arrays, the walk the \
                     runtime takes, for every row of features."
                )
            } else {
                format!(
                    "The code of tree {t}, generated for these proofs, gives the same bits as the walk through its \
                     arrays, the walk `outputs` and the runtime take, for every row of features."
                )
            };
            proofs.push(quote! {
                #[doc = #walk_doc]
                #[kani::proof]
                #[kani::unwind(#u_walk)]
                fn #walk() {
                    let x: [f32; N_FEATURES] = kani::any();
                    assert!(#function(&x).to_bits() == TREES.value[TREES.leaf(#index, &x)].to_bits());
                }
            });
            // The loops: the features of the box and the nodes of the tree.
            let u_range = Literal::usize_unsuffixed(nodes.max(self.n_features) + 2);
            let n_nodes = Literal::usize_unsuffixed(nodes);
            let range = format_ident!("{name}_tree_{t}_stays_in_its_range");
            let range_doc = format!("Every row of every box reaches a leaf of tree {t} inside `TREES.leaf_range({t}, ..)` of the box.");
            proofs.push(quote! {
                #[doc = #range_doc]
                #[kani::proof]
                #[kani::unwind(#u_range)]
                fn #range() {
                    let (lo, hi, x) = any_row_of_any_box();
                    let [min, max] = TREES.leaf_range(#index, &lo, &hi, &mut [false; #n_nodes]);
                    let leaf = #function(&x);
                    assert!(min <= leaf && leaf <= max);
                }
            });
        }
        quote! {
            /// Any box of feature values and any row of it: `lo[j] <= x[j] <= hi[j]` for every
            /// feature `j`, where `x[j]` may also be NaN if `lo[j]` or `hi[j]` is NaN. Empty boxes
            /// (`lo > hi`) have no row.
            fn any_row_of_any_box() -> ([f32; N_FEATURES], [f32; N_FEATURES], [f32; N_FEATURES]) {
                let (lo, hi, x): ([f32; N_FEATURES], [f32; N_FEATURES], [f32; N_FEATURES]) = (kani::any(), kani::any(), kani::any());
                let mut j = 0;
                while j < N_FEATURES {
                    kani::assume((lo[j] <= x[j] && x[j] <= hi[j]) || (x[j].is_nan() && (lo[j].is_nan() || hi[j].is_nan())));
                    j += 1;
                }
                (lo, hi, x)
            }

            #(#proofs)*
        }
    }

    /// The field's type, and the enum it needs if any.
    fn field_type(&self, f: &Field, kani_derive: &TokenStream) -> (Option<TokenStream>, TokenStream) {
        match &f.spec {
            Spec::Numeric { .. } => (None, quote!(Option<f64>)),
            Spec::Text {
                ty,
                categories,
                variants,
                other,
                nullable,
                ..
            } => {
                let n = categories.len();
                let all: Vec<TokenStream> = variants.iter().map(|v| quote!(#ty::#v)).collect();
                let docs: Vec<String> = categories.iter().map(|c| format!("`{c}`")).collect();
                let mut as_str: Vec<TokenStream> = variants.iter().zip(categories).map(|(v, c)| quote!(#ty::#v => #c,)).collect();
                let from: Vec<TokenStream> = variants.iter().zip(categories).map(|(v, c)| quote!(#c => Some(#ty::#v),)).collect();
                let other_variant = other.as_ref().map(|(v, shown)| {
                    as_str.push(quote!(#ty::#v => #shown,));
                    quote! {
                        /// A value not seen in training: encoded as no category, as the runtime
                        /// does with `unknown_category = "ignore"`.
                        #v,
                    }
                });
                let field = &f.name;
                let doc = format!("The categories of `{field}` seen in training.");
                let decl = quote! {
                    #[doc = #doc]
                    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
                    #kani_derive
                    pub enum #ty {
                        #(#[doc = #docs] #variants,)*
                        #other_variant
                    }

                    impl #ty {
                        /// Every category seen in training, in training order.
                        pub const ALL: [#ty; #n] = [#(#all),*];

                        /// The category as it appears in the data.
                        pub const fn as_str(self) -> &'static str {
                            match self {
                                #(#as_str)*
                            }
                        }

                        /// The category of a text value; `None` for one not seen in training.
                        pub fn from_category(s: &str) -> Option<#ty> {
                            match s {
                                #(#from)*
                                _ => None,
                            }
                        }
                    }

                    impl ::core::fmt::Display for #ty {
                        fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                            f.write_str(self.as_str())
                        }
                    }
                };
                let t = if *nullable { quote!(Option<#ty>) } else { quote!(#ty) };
                (Some(decl), t)
            }
            Spec::Bool { nullable, .. } => (None, if *nullable { quote!(Option<bool>) } else { quote!(bool) }),
        }
    }

    fn field_doc(&self, f: &Field) -> String {
        let name = &f.name;
        let missing = |missing_slot: bool, nullable: bool| match (missing_slot, nullable) {
            (true, _) => "A missing value has its own slot (it was seen in training).",
            (false, true) => "A missing value is encoded as no category.",
            (false, false) => "Never missing: the model has no slot for a missing value.",
        };
        match &f.spec {
            Spec::Numeric {
                refuse_missing: true,
                mean,
                std,
                ..
            } => format!(
                "`{name}`: a number, standardised with the mean {} and standard deviation {}. A missing value \
                 is refused: the model's trees send it different ways at different splits.",
                short(*mean),
                short(*std),
            ),
            Spec::Numeric { impute, mean, std, .. } => format!(
                "`{name}`: a number, standardised with the training mean {} and standard deviation {}. \
                 A missing value is imputed with {}.",
                short(*mean),
                short(*std),
                short(*impute)
            ),
            Spec::Text {
                categories,
                missing_slot,
                nullable,
                ..
            } => {
                let shown: Vec<&str> = categories.iter().take(12).map(String::as_str).collect();
                let more = if categories.len() > 12 { ", …" } else { "" };
                format!(
                    "`{name}`: one of {} categories ({}{more}). {}",
                    categories.len(),
                    shown.join(", "),
                    missing(*missing_slot, *nullable)
                )
            }
            Spec::Bool {
                categories,
                missing_slot,
                nullable,
                ..
            } => format!(
                "`{name}`: a boolean category (seen in training: {}). {}",
                categories.join(", "),
                missing(*missing_slot, *nullable)
            ),
        }
    }

    /// What to do with a value the model has no slot for.
    fn unknown(&self, field: &str) -> TokenStream {
        if self.ignore_unknown {
            quote!({})
        } else {
            quote!(return Err(InputError::UnknownCategory { field: #field }))
        }
    }

    fn encode(&self, f: &Field) -> TokenStream {
        let ident = &f.ident;
        let field = f.name.as_str();
        let slot = |offset: usize, i: usize| {
            let s = offset + i;
            quote!(x[#s] = 1.0)
        };
        let missing = |missing_slot: bool, offset: usize, n: usize| {
            if missing_slot {
                let s = slot(offset, n);
                quote!(None => #s,)
            } else {
                quote!(None => {})
            }
        };
        match &f.spec {
            Spec::Numeric {
                impute,
                mean,
                std,
                offset,
                refuse_missing,
            } => {
                let (impute, mean, std) = (f64_lit(*impute), f64_lit(*mean), f64_lit(*std));
                let nan = self.nan_is_missing;
                quote! {
                    x[#offset] = match kernels::numeric(#field, input.#ident, &NUMERIC_TOKENS, #nan, #refuse_missing, #impute, #mean, #std) {
                        Ok(z) => z,
                        Err(e) => return Err(e),
                    };
                }
            }
            Spec::Text {
                ty,
                variants,
                other,
                missing_slot,
                nullable,
                offset,
                categories,
                ..
            } => {
                let wrap = |p: TokenStream| if *nullable { quote!(Some(#p)) } else { p };
                let mut arms: Vec<TokenStream> = variants
                    .iter()
                    .enumerate()
                    .map(|(i, v)| {
                        let pat = wrap(quote!(#ty::#v));
                        let s = slot(*offset, i);
                        quote!(#pat => #s,)
                    })
                    .collect();
                if let Some((v, _)) = other {
                    let pat = wrap(quote!(#ty::#v));
                    arms.push(quote!(#pat => {}));
                }
                if *nullable {
                    arms.push(missing(*missing_slot, *offset, categories.len()));
                }
                quote!(match input.#ident { #(#arms)* })
            }
            Spec::Bool {
                categories,
                missing_slot,
                nullable,
                offset,
            } => {
                let arm = |b: bool| {
                    let pat = if *nullable { quote!(Some(#b)) } else { quote!(#b) };
                    let body = match categories.iter().position(|c| c == if b { "true" } else { "false" }) {
                        Some(i) => slot(*offset, i),
                        None => self.unknown(field),
                    };
                    quote!(#pat => #body,)
                };
                let (f_arm, t_arm) = (arm(false), arm(true));
                let none = if *nullable {
                    missing(*missing_slot, *offset, categories.len())
                } else {
                    quote!()
                };
                quote!(match input.#ident { #f_arm #t_arm #none })
            }
        }
    }

    /// The expression that turns `values[i]` into the field's value.
    fn parse_value(&self, f: &Field, i: usize) -> TokenStream {
        let field = f.name.as_str();
        let mismatch = |expected: &str| quote!(return Err(InputError::TypeMismatch { field: #field, expected: #expected }));
        // A missing value: `None`, or refused when the field cannot be missing.
        let none = |nullable: bool| {
            if nullable {
                quote!(None)
            } else {
                quote!(return Err(InputError::UnknownCategory { field: #field }))
            }
        };
        let some = |nullable: bool, v: TokenStream| if nullable { quote!(Some(#v)) } else { v };
        match &f.spec {
            Spec::Numeric { .. } => {
                let m = mismatch("number");
                quote! {
                    match values[#i] {
                        FieldValue::Missing => None,
                        FieldValue::Text(s) if MISSING_TOKENS.contains(&s) => None,
                        FieldValue::Int(v) => Some(v as f64),
                        FieldValue::Float(v) => Some(v),
                        _ => #m,
                    }
                }
            }
            Spec::Text { ty, nullable, other, .. } => {
                let none = none(*nullable);
                let known = some(*nullable, quote!(c));
                let unknown = match other {
                    Some((v, _)) => some(*nullable, quote!(#ty::#v)),
                    None => quote!(return Err(InputError::UnknownCategory { field: #field })),
                };
                let m = mismatch("text");
                quote! {
                    match values[#i] {
                        FieldValue::Missing => #none,
                        FieldValue::Text(s) if MISSING_TOKENS.contains(&s) => #none,
                        FieldValue::Text(s) => match #ty::from_category(s) {
                            Some(c) => #known,
                            None => #unknown,
                        },
                        _ => #m,
                    }
                }
            }
            Spec::Bool { nullable, .. } => {
                let none = none(*nullable);
                let b = some(*nullable, quote!(b));
                let m = mismatch("bool");
                quote! {
                    match values[#i] {
                        FieldValue::Missing => #none,
                        FieldValue::Text(s) if MISSING_TOKENS.contains(&s) => #none,
                        FieldValue::Bool(b) => #b,
                        _ => #m,
                    }
                }
            }
        }
    }

    fn to_value(&self, f: &Field) -> TokenStream {
        let ident = &f.ident;
        match &f.spec {
            Spec::Numeric { .. } => quote!(match self.#ident { Some(v) => FieldValue::Float(v), None => FieldValue::Missing }),
            Spec::Text { nullable: true, .. } => {
                quote!(match self.#ident { Some(c) => FieldValue::Text(c.as_str()), None => FieldValue::Missing })
            }
            Spec::Text { nullable: false, .. } => quote!(FieldValue::Text(self.#ident.as_str())),
            Spec::Bool { nullable: true, .. } => quote!(match self.#ident { Some(b) => FieldValue::Bool(b), None => FieldValue::Missing }),
            Spec::Bool { nullable: false, .. } => quote!(FieldValue::Bool(self.#ident)),
        }
    }

    fn info(&self, f: &Field) -> TokenStream {
        let name = f.name.as_str();
        let kind = match &f.spec {
            Spec::Numeric { mean, std, .. } => {
                let (mean, std) = (f64_lit(*mean), f64_lit(*std));
                quote!(FieldKind::Numeric { mean: #mean, std: #std })
            }
            Spec::Text { categories, .. } => quote!(FieldKind::Text { categories: &[#(#categories),*] }),
            Spec::Bool { .. } => quote!(FieldKind::Bool),
        };
        quote!(FieldInfo { name: #name, kind: #kind })
    }

    /// One example record as an `Input` expression.
    fn example(&self, record: &serde_json::Value) -> TokenStream {
        let inits = self.fields.iter().map(|f| {
            let ident = &f.ident;
            let v = record.get(&f.name).unwrap_or(&serde_json::Value::Null);
            let value = match &f.spec {
                Spec::Numeric { .. } => match v.as_f64() {
                    Some(x) => {
                        let x = f64_lit(v.as_i64().map_or(x, |i| i as f64));
                        quote!(Some(#x))
                    }
                    None => quote!(None),
                },
                Spec::Text {
                    ty,
                    categories,
                    variants,
                    other,
                    nullable,
                    ..
                } => {
                    let c = v.as_str().map(|s| match categories.iter().position(|c| c == s) {
                        Some(k) => {
                            let v = &variants[k];
                            quote!(#ty::#v)
                        }
                        None => {
                            let (v, _) = other.as_ref().expect("examples only hold categories seen in training");
                            quote!(#ty::#v)
                        }
                    });
                    match (c, nullable) {
                        (Some(c), true) => quote!(Some(#c)),
                        (Some(c), false) => c,
                        (None, _) => quote!(None),
                    }
                }
                Spec::Bool { nullable, .. } => match (v.as_bool(), nullable) {
                    (Some(b), true) => quote!(Some(#b)),
                    (Some(b), false) => quote!(#b),
                    (None, _) => quote!(None),
                },
            };
            quote!(#ident: #value)
        });
        quote!(Input { #(#inits),* })
    }

    fn output(&self) -> (TokenStream, TokenStream) {
        match &self.output {
            Output::Binary {
                negative,
                positive,
                threshold,
            } => {
                let t = f64_lit(*threshold);
                let doc_t = format!("Decision threshold on the probability of `{positive}`: {}.", short(*threshold));
                let doc_neg = format!("`{negative}`: the negative label.");
                let doc_pos = format!("`{positive}`: the positive label, the one the model detects.");
                let types = quote! {
                    #[doc = #doc_t]
                    pub const THRESHOLD: f64 = #t;

                    /// The decision at [`THRESHOLD`].
                    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
                    pub enum Label {
                        #[doc = #doc_neg]
                        Negative,
                        #[doc = #doc_pos]
                        Positive,
                    }

                    impl Label {
                        /// The label as it appears in the data.
                        pub const fn as_str(self) -> &'static str {
                            match self {
                                Label::Negative => #negative,
                                Label::Positive => #positive,
                            }
                        }
                    }

                    impl ::core::fmt::Display for Label {
                        fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                            f.write_str(self.as_str())
                        }
                    }

                    /// A prediction.
                    #[derive(Debug, Clone, Copy, PartialEq)]
                    pub struct Prediction {
                        /// [`Label::Positive`] when `probability >= THRESHOLD`.
                        pub label: Label,
                        /// Probability of the positive label.
                        pub probability: f64,
                    }
                };
                let decode = quote! {
                    /// The prediction for one row of model outputs (one logit).
                    pub const fn decode(z: &[f32; N_OUTPUTS]) -> Prediction {
                        let probability = kernels::sigmoid(z[0] as f64);
                        let label = if probability >= THRESHOLD { Label::Positive } else { Label::Negative };
                        Prediction { label, probability }
                    }
                };
                (types, decode)
            }
            Output::Multiclass { classes, variants } => {
                let n = classes.len();
                let docs: Vec<String> = classes.iter().map(|c| format!("`{c}`")).collect();
                let all: Vec<TokenStream> = variants.iter().map(|v| quote!(Class::#v)).collect();
                let as_str: Vec<TokenStream> = variants.iter().zip(classes).map(|(v, c)| quote!(Class::#v => #c,)).collect();
                let types = quote! {
                    /// The classes, in the order of the model's outputs.
                    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
                    pub enum Class {
                        #(#[doc = #docs] #variants,)*
                    }

                    impl Class {
                        /// Every class, in the order of the model's outputs.
                        pub const ALL: [Class; #n] = [#(#all),*];

                        /// The class as it appears in the data.
                        pub const fn as_str(self) -> &'static str {
                            match self {
                                #(#as_str)*
                            }
                        }
                    }

                    impl ::core::fmt::Display for Class {
                        fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                            f.write_str(self.as_str())
                        }
                    }

                    /// A prediction.
                    #[derive(Debug, Clone, Copy, PartialEq)]
                    pub struct Prediction {
                        /// The most probable class (the first one on a tie).
                        pub class: Class,
                        /// Its probability.
                        pub probability: f64,
                        /// Every class's probability, in the order of [`Class::ALL`].
                        pub probabilities: [f64; N_OUTPUTS],
                    }
                };
                let decode = quote! {
                    /// The prediction for one row of model outputs (one logit per class).
                    pub const fn decode(z: &[f32; N_OUTPUTS]) -> Prediction {
                        let mut probabilities = [0f64; N_OUTPUTS];
                        kernels::softmax(z, &mut probabilities);
                        let best = kernels::argmax(&probabilities);
                        Prediction { class: Class::ALL[best], probability: probabilities[best], probabilities }
                    }
                };
                (types, decode)
            }
            Output::Regression { mean, std } => {
                let (m, s) = (f64_lit(*mean), f64_lit(*std));
                let doc = format!(
                    "The model learned `{}` standardised: the value is `{} + {} × output`.",
                    self.target,
                    short(*mean),
                    short(*std)
                );
                let types = quote! {
                    const TARGET_MEAN: f64 = #m;
                    const TARGET_STD: f64 = #s;

                    #[doc = #doc]
                    #[derive(Debug, Clone, Copy, PartialEq)]
                    pub struct Prediction {
                        /// The predicted value, in the target's units.
                        pub value: f64,
                    }
                };
                let decode = quote! {
                    /// The prediction for one row of model outputs (the standardised value).
                    pub const fn decode(z: &[f32; N_OUTPUTS]) -> Prediction {
                        Prediction { value: kernels::unscale(z[0], TARGET_MEAN, TARGET_STD) }
                    }
                };
                (types, decode)
            }
        }
    }

    /// The model's constants, the body of `outputs` and the body of `outputs_batch`.
    fn network(&self, tree_code: bool) -> (TokenStream, TokenStream, TokenStream) {
        let row_by_row = quote! {
            for (row, o) in x.iter().zip(out.iter_mut()) {
                *o = outputs(row);
            }
        };
        match &self.network {
            Network::Dense(layers) => {
                let mut consts = Vec::new();
                let mut body = Vec::new();
                let last = layers.len() - 1;
                for (i, l) in layers.iter().enumerate() {
                    let w = format_ident!("L{i}_WEIGHT");
                    let b = format_ident!("L{i}_BIAS");
                    let (n_w, n_b) = (l.weight.len(), l.bias.len());
                    let weights = l.weight.iter().map(|&v| f32_lit(v));
                    let biases = l.bias.iter().map(|&v| f32_lit(v));
                    let doc = format!("Layer {i}: {} → {}, row-major `[out, in]`.", l.in_dim, l.out_dim);
                    consts.push(quote! {
                        #[doc = #doc]
                        const #w: [f32; #n_w] = [#(#weights),*];
                        const #b: [f32; #n_b] = [#(#biases),*];
                    });
                    let input = if i == 0 {
                        quote!(x)
                    } else {
                        let h = format_ident!("h{}", i - 1);
                        quote!(&#h)
                    };
                    let out = if i == last { format_ident!("out") } else { format_ident!("h{i}") };
                    let width = l.out_dim;
                    let relu = l.relu;
                    body.push(quote! {
                        let mut #out = [0f32; #width];
                        kernels::dense(&#w, &#b, #input, #relu, &mut #out);
                    });
                }
                (quote!(#(#consts)*), quote!(#(#body)* out), row_by_row)
            }
            Network::Trees {
                roots,
                feature,
                threshold,
                left,
                right,
                value,
                base,
                monotone,
                sum_in_f32,
            } => {
                let sum_in_f32 = *sum_in_f32;
                let (n_trees, n_nodes, n_base) = (roots.len(), feature.len(), base.len());
                let generated = tree_code.then(|| {
                    let trees = kernels::Trees {
                        roots,
                        feature,
                        threshold,
                        left,
                        right,
                        value,
                        base,
                    };
                    generated_trees(&trees, self.n_outputs, sum_in_f32)
                });
                let trees_doc = if tree_code {
                    "The same trees as arrays, for [`output_bounds`] and the compiler's checks. \
                     [`outputs`] runs the functions generated from them."
                } else {
                    "The ensemble that [`outputs`] evaluates. `TREES.leaf(t, &x)` is the walk \
                     through tree `t`; the Kani proofs check one walk at a time."
                };
                let constrained: Vec<String> = self
                    .fields
                    .iter()
                    .filter_map(|f| match f.spec {
                        Spec::Numeric { offset, .. } if monotone[offset] != 0 => Some(format!(
                            "{} in `{}`",
                            if monotone[offset] > 0 { "increasing" } else { "decreasing" },
                            f.name
                        )),
                        _ => None,
                    })
                    .collect();
                let monotone_doc = format!(
                    "Monotone constraints the trees were trained under, one per feature: 1 (the output never \
                     decreases in the feature), -1 (never increases) or 0. This model: {}. The compiler checks \
                     that every tree keeps them: at each split on a constrained feature, every leaf on the lesser \
                     side is at most every leaf on the greater side, so the sum of the trees is monotone too.",
                    if constrained.is_empty() {
                        "none".to_string()
                    } else {
                        constrained.join(", ")
                    }
                );
                let monotone = monotone.iter().map(|&v| Literal::i8_unsuffixed(v));
                let roots = roots.iter().map(|&v| Literal::u32_unsuffixed(v));
                let feature = feature.iter().map(|&v| Literal::i32_unsuffixed(v));
                let threshold = threshold.iter().map(|&v| f32_lit(v));
                let left = left.iter().map(|&v| Literal::u32_unsuffixed(v));
                let right = right.iter().map(|&v| Literal::u32_unsuffixed(v));
                let value = value.iter().map(|&v| f32_lit(v));
                let base = base.iter().map(|&v| f64_lit(v));
                let consts = quote! {
                    const TREE_ROOTS: [u32; #n_trees] = [#(#roots),*];
                    const TREE_FEATURE: [i32; #n_nodes] = [#(#feature),*];
                    const TREE_THRESHOLD: [f32; #n_nodes] = [#(#threshold),*];
                    const TREE_LEFT: [u32; #n_nodes] = [#(#left),*];
                    const TREE_RIGHT: [u32; #n_nodes] = [#(#right),*];
                    const TREE_VALUE: [f32; #n_nodes] = [#(#value),*];
                    const TREE_BASE: [f64; #n_base] = [#(#base),*];
                    /// The starting scores rounded to f32, where the leaves are added in f32.
                    const TREE_BASE_F32: [f32; #n_base] = {
                        let mut b = [0f32; #n_base];
                        let mut o = 0;
                        while o < #n_base {
                            b[o] = TREE_BASE[o] as f32;
                            o += 1;
                        }
                        b
                    };

                    #[doc = #monotone_doc]
                    pub const MONOTONE: [i8; N_FEATURES] = [#(#monotone),*];

                    // Children come after their parent, so one pass backwards gives every
                    // subtree's smallest and largest leaf, and checks each constrained split.
                    const _: () = {
                        let (mut lo, mut hi) = ([0f32; #n_nodes], [0f32; #n_nodes]);
                        let mut i = #n_nodes;
                        while i > 0 {
                            i -= 1;
                            if TREE_FEATURE[i] < 0 {
                                lo[i] = TREE_VALUE[i];
                                hi[i] = TREE_VALUE[i];
                            } else {
                                let (l, r) = (TREE_LEFT[i] as usize, TREE_RIGHT[i] as usize);
                                lo[i] = if lo[l] < lo[r] { lo[l] } else { lo[r] };
                                hi[i] = if hi[l] > hi[r] { hi[l] } else { hi[r] };
                                let c = MONOTONE[TREE_FEATURE[i] as usize];
                                assert!(c <= 0 || hi[l] <= lo[r], "a tree breaks an increasing monotone constraint");
                                assert!(c >= 0 || lo[l] >= hi[r], "a tree breaks a decreasing monotone constraint");
                            }
                        }
                    };

                    #[doc = #trees_doc]
                    pub(crate) const TREES: kernels::Trees<'static> = kernels::Trees {
                        roots: &TREE_ROOTS,
                        feature: &TREE_FEATURE,
                        threshold: &TREE_THRESHOLD,
                        left: &TREE_LEFT,
                        right: &TREE_RIGHT,
                        value: &TREE_VALUE,
                        base: &TREE_BASE,
                    };

                    // The trees' structure, checked by the compiler: one starting score per
                    // output, the first root is node 0, roots ascend, and every split reads a
                    // real feature and points forward to a node of its own tree. Inference can
                    // then neither leave a tree nor loop.
                    const _: () = {
                        assert!(TREE_BASE.len() == N_OUTPUTS, "one starting score per output");
                        assert!(TREE_ROOTS[0] == 0, "the first tree starts at node 0");
                        let mut t = 0;
                        while t < TREE_ROOTS.len() {
                            let end = if t + 1 < TREE_ROOTS.len() { TREE_ROOTS[t + 1] as usize } else { TREE_FEATURE.len() };
                            assert!((TREE_ROOTS[t] as usize) < end, "a tree has no nodes");
                            let mut i = TREE_ROOTS[t] as usize;
                            while i < end {
                                if TREE_FEATURE[i] >= 0 {
                                    assert!((TREE_FEATURE[i] as usize) < N_FEATURES, "a split reads a feature the model does not have");
                                    assert!(TREE_LEFT[i] as usize > i && (TREE_LEFT[i] as usize) < end, "a left child outside its tree");
                                    assert!(TREE_RIGHT[i] as usize > i && (TREE_RIGHT[i] as usize) < end, "a right child outside its tree");
                                }
                                i += 1;
                            }
                            t += 1;
                        }
                    };
                };
                let Some((functions, body)) = generated else {
                    let call = if sum_in_f32 { quote!(outputs_in_f32) } else { quote!(outputs) };
                    let body = quote! {
                        let mut out = [0f32; N_OUTPUTS];
                        TREES.#call(x, &mut out);
                        out
                    };
                    return (consts, body, row_by_row);
                };
                let batch = if n_nodes > TREE_CODE_MAX_NODES {
                    quote!(outputs_by_tree(x, out);)
                } else {
                    row_by_row
                };
                (quote!(#consts #functions), body, batch)
            }
        }
    }

    fn self_check(&self) -> TokenStream {
        let check = match self.task {
            Task::Binary | Task::Regression => {
                let field = if self.task == Task::Binary {
                    quote!(probability)
                } else {
                    quote!(value)
                };
                let bits = self.expected.iter().map(|e| Literal::u64_unsuffixed(e[0].to_bits()));
                let n = self.expected.len();
                quote! {
                    const EXPECTED: [u64; #n] = [#(#bits),*];

                    const fn same_as_runtime(i: usize, p: &Prediction) -> bool {
                        p.#field.to_bits() == EXPECTED[i]
                    }
                }
            }
            Task::Multiclass => {
                let n = self.expected.len();
                let rows = self.expected.iter().map(|e| {
                    let bits = e.iter().map(|v| Literal::u64_unsuffixed(v.to_bits()));
                    quote!([#(#bits),*])
                });
                quote! {
                    const EXPECTED: [[u64; N_OUTPUTS]; #n] = [#(#rows),*];

                    const fn same_as_runtime(i: usize, p: &Prediction) -> bool {
                        let mut k = 0;
                        while k < N_OUTPUTS {
                            if p.probabilities[k].to_bits() != EXPECTED[i][k] {
                                return false;
                            }
                            k += 1;
                        }
                        true
                    }
                }
            }
        };
        quote! {
            #check

            // The compiler predicts every example with its own floating point, and must get
            // the bits tabsmith-runtime predicted when the macro expanded: otherwise the crate
            // does not compile.
            #[allow(long_running_const_eval)]
            const _: () = {
                let mut i = 0;
                while i < EXAMPLES.len() {
                    match predict(&EXAMPLES[i]) {
                        Ok(p) => assert!(
                            same_as_runtime(i, &p),
                            "the compiler predicts an example of the bundle differently from tabsmith-runtime"
                        ),
                        Err(_) => panic!("the compiler refuses an example of the bundle that tabsmith-runtime predicts"),
                    }
                    i += 1;
                }
            };

            /// Predicts [`EXAMPLES`] and compares every result, bit for bit, with what
            /// `tabsmith-runtime` predicted for them when this crate was compiled. The compiler
            /// already did the same while it compiled; run this once on the target (a power-on
            /// self test): it fails on a platform whose floating point differs, for example one
            /// that flushes subnormals to zero. The examples pass through `black_box`, so that
            /// the device computes them rather than the optimiser.
            pub fn self_check() -> ::core::result::Result<(), SelfCheckError> {
                for (i, input) in EXAMPLES.iter().enumerate() {
                    let same = match predict(::core::hint::black_box(input)) {
                        Ok(p) => same_as_runtime(i, &p),
                        Err(_) => false,
                    };
                    if !same {
                        return Err(SelfCheckError { example: i });
                    }
                }
                Ok(())
            }
        }
    }
}

/// Node `i` of a tree as an expression: a leaf's value, or a comparison and two subtrees.
/// A row goes left when `x[feature] <= threshold`, as in `kernels::Trees`, so NaN goes right.
/// The numbers are finite (`TreeEnsemble::new` checks) and written in decimal, so that the
/// code reads as the tree it is. That is exact: Rust prints the shortest decimal that reads
/// back as the same bits, and rustc reads it back correctly rounded. `tests/tree_code.rs`
/// checks every number of its bundles at its exact value.
fn tree_expr(t: &kernels::Trees<'_>, i: usize) -> TokenStream {
    if t.feature[i] < 0 {
        let value = Literal::f32_suffixed(t.value[i]);
        return quote!(#value);
    }
    let f = Literal::usize_unsuffixed(t.feature[i] as usize);
    let threshold = Literal::f32_suffixed(t.threshold[i]);
    let (left, right) = (tree_expr(t, t.left[i] as usize), tree_expr(t, t.right[i] as usize));
    quote!(if x[#f] <= #threshold { #left } else { #right })
}

/// One function per tree, the body of `outputs` that adds them up, and `outputs_by_tree`
/// for many rows. Each output adds its trees in order, in f64, and is rounded to f32 at
/// the end, as in `kernels::Trees::outputs`, or in f32 when `sum_in_f32`, as in
/// `kernels::Trees::outputs_in_f32`: the same bits.
fn generated_trees(t: &kernels::Trees<'_>, n_outputs: usize, sum_in_f32: bool) -> (TokenStream, TokenStream) {
    let n_trees = t.roots.len();
    let names: Vec<Ident> = (0..n_trees).map(|i| format_ident!("tree_{i}")).collect();
    let functions = t.roots.iter().zip(&names).map(|(&root, name)| {
        let body = tree_expr(t, root as usize);
        quote! {
            #[inline]
            const fn #name(x: &[f32; N_FEATURES]) -> f32 {
                #body
            }
        }
    });
    let additions = names.iter().enumerate().map(|(i, name)| {
        let o = Literal::usize_unsuffixed(i % n_outputs);
        if sum_in_f32 {
            quote!(sums[#o] += #name(x);)
        } else {
            quote!(sums[#o] += #name(x) as f64;)
        }
    });
    let results = (0..n_outputs).map(|o| {
        let o = Literal::usize_unsuffixed(o);
        if sum_in_f32 { quote!(sums[#o]) } else { quote!(sums[#o] as f32) }
    });
    let (start, widen, round) = if sum_in_f32 {
        (quote!(TREE_BASE_F32), quote!(), quote!())
    } else {
        (quote!(TREE_BASE), quote!(as f64), quote!(as f32))
    };
    let body = quote! {
        let mut sums = #start;
        #(#additions)*
        [#(#results),*]
    };
    let functions = quote! {
        #(#functions)*

        /// Every tree, for [`outputs_by_tree`].
        pub(crate) const TREE_FUNCTIONS: [fn(&[f32; N_FEATURES]) -> f32; #n_trees] = [#(#names),*];

        /// [`outputs_batch`] tree by tree over blocks of rows: one tree's code stays in the
        /// cache while it runs. Each output still adds its trees in order.
        pub(crate) fn outputs_by_tree(x: &[[f32; N_FEATURES]], out: &mut [[f32; N_OUTPUTS]]) {
            const BLOCK: usize = 256;
            for (rows, results) in x.chunks(BLOCK).zip(out.chunks_mut(BLOCK)) {
                let mut sums = [#start; BLOCK];
                for (t, tree) in TREE_FUNCTIONS.iter().enumerate() {
                    for (row, s) in rows.iter().zip(sums.iter_mut()) {
                        s[t % N_OUTPUTS] += tree(row) #widen;
                    }
                }
                for (s, r) in sums[..rows.len()].iter().zip(results.iter_mut()) {
                    for k in 0..N_OUTPUTS {
                        r[k] = s[k] #round;
                    }
                }
            }
        }
    };
    (functions, body)
}

fn f32_lit(v: f32) -> TokenStream {
    let bits = Literal::u32_unsuffixed(v.to_bits());
    quote!(f32::from_bits(#bits))
}

fn f64_lit(v: f64) -> TokenStream {
    let bits = Literal::u64_unsuffixed(v.to_bits());
    quote!(f64::from_bits(#bits))
}

/// At most four decimals, for documentation.
fn short(v: f64) -> String {
    let s = format!("{v:.4}");
    let s = s.trim_end_matches('0').trim_end_matches('.');
    if s.is_empty() || s == "-" { "0".to_string() } else { s.to_string() }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURES: [&str; 7] = [
        "binary-mlp",
        "binary-gbdt",
        "multiclass-linear",
        "multiclass-gbdt",
        "regression-mlp",
        "regression-gbdt",
        "policies-linear",
    ];

    fn fixture(name: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/bundles").join(name)
    }

    fn copy_dir(from: &Path, to: &Path) {
        std::fs::create_dir_all(to).unwrap();
        for entry in std::fs::read_dir(from).unwrap() {
            let entry = entry.unwrap();
            let target = to.join(entry.file_name());
            if entry.file_type().unwrap().is_dir() {
                copy_dir(&entry.path(), &target);
            } else {
                std::fs::copy(entry.path(), target).unwrap();
            }
        }
    }

    #[test]
    fn every_fixture_generates_well_formed_code() {
        for name in FIXTURES {
            let compiled = Compiled::load(&fixture(name), name).unwrap();
            for kani in [false, true] {
                for requested in [None, Some(true), Some(false)] {
                    let tree_code = compiled.tree_code(requested).unwrap();
                    let tokens = compiled.generate(&format_ident!("m"), kani, tree_code, &[]);
                    syn::parse2::<syn::File>(tokens).unwrap_or_else(|e| panic!("{name}: {e}"));
                }
            }
        }
    }

    #[test]
    fn small_trees_become_code_and_deep_ones_stay_arrays() {
        let mut compiled = Compiled::load(&fixture("binary-gbdt"), "binary-gbdt").unwrap();
        assert_eq!(compiled.tree_code(None), Ok(true));
        assert_eq!(compiled.tree_code(Some(true)), Ok(true));
        assert_eq!(compiled.tree_code(Some(false)), Ok(false));

        let code = compiled.generate(&format_ident!("m"), false, true, &[]).to_string();
        assert!(code.contains("const fn tree_0"), "one function per tree");
        assert!(code.contains("pub const fn outputs"), "the compiler can run the model");
        assert!(!code.contains("TREES . outputs"), "the arrays are not walked");
        let arrays = compiled.generate(&format_ident!("m"), false, false, &[]).to_string();
        assert!(!arrays.contains("const fn tree_0"));
        assert!(arrays.contains("TREES . outputs"));
        // Walking the arrays is a `const fn` too, and so is the whole prediction.
        for both in [&code, &arrays] {
            for f in ["features", "outputs", "decode", "predict"] {
                assert!(both.contains(&format!("pub const fn {f} (")), "{f}");
            }
        }

        compiled.max_tree_depth = TREE_CODE_MAX_DEPTH + 1;
        assert_eq!(compiled.tree_code(None), Ok(false));
        assert_eq!(compiled.tree_code(Some(false)), Ok(false));
        assert!(compiled.tree_code(Some(true)).unwrap_err().contains("tree_arrays"));
    }

    #[test]
    fn every_tree_gets_its_two_proofs_in_both_forms() {
        let mut compiled = Compiled::load(&fixture("binary-gbdt"), "binary-gbdt").unwrap();
        let count = |code: &str, needle: &str| code.matches(needle).count();
        for tree_code in [true, false] {
            let code = compiled.generate(&format_ident!("m"), true, tree_code, &[]).to_string();
            for t in 0..14 {
                assert_eq!(count(&code, &format!("fn m_tree_{t}_is_its_walk ")), 1, "tree {t}");
                assert_eq!(count(&code, &format!("fn m_tree_{t}_stays_in_its_range ")), 1, "tree {t}");
                // The code of each tree exists once: in the module, or for arrays in the proofs.
                assert_eq!(count(&code, &format!("const fn tree_{t} ")), 1, "tree {t}");
            }
            assert_eq!(count(&code, "_is_its_walk "), 14);
            let without = compiled.generate(&format_ident!("m"), false, tree_code, &[]).to_string();
            assert_eq!(count(&without, "_is_its_walk "), 0, "no proofs without `kani`");
            assert_eq!(count(&without, "const fn tree_0 "), usize::from(tree_code));
        }
        // A tree too deep for code has neither proof; the others keep theirs.
        compiled.tree_depths[3] = TREE_CODE_MAX_DEPTH + 1;
        let code = compiled.generate(&format_ident!("m"), true, false, &[]).to_string();
        assert_eq!(count(&code, "fn m_tree_3_is_its_walk "), 0);
        assert_eq!(count(&code, "fn m_tree_3_stays_in_its_range "), 0);
        assert_eq!(count(&code, "_stays_in_its_range "), 13);
        // A network has no trees to prove.
        let network = Compiled::load(&fixture("binary-mlp"), "binary-mlp").unwrap();
        let code = network.generate(&format_ident!("m"), true, false, &[]).to_string();
        assert_eq!(count(&code, "_is_its_walk "), 0);
        assert_eq!(count(&code, "any_row_of_any_box"), 0);
    }

    #[test]
    fn a_network_ignores_the_tree_options() {
        let compiled = Compiled::load(&fixture("binary-mlp"), "binary-mlp").unwrap();
        for requested in [None, Some(true), Some(false)] {
            assert_eq!(compiled.tree_code(requested), Ok(false));
        }
    }

    #[test]
    fn a_tampered_bundle_is_refused_at_compile_time() {
        let tmp = tempfile::tempdir().unwrap();
        copy_dir(&fixture("binary-mlp"), tmp.path());
        let weights = tmp.path().join("model.safetensors");
        let mut bytes = std::fs::read(&weights).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 1;
        std::fs::write(&weights, bytes).unwrap();
        let error = Compiled::load(tmp.path(), "tampered").err().expect("refused");
        assert!(error.contains("E_BUNDLE_CORRUPT"), "{error}");
    }
}
