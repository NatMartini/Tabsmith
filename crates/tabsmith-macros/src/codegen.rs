//! Reads a bundle with the runtime and generates the module that `model!` expands to.

use std::path::{Path, PathBuf};

use proc_macro2::{Ident, Literal, TokenStream};
use quote::{format_ident, quote};
use syn::LitStr;
use syn::parse::{Parse, ParseStream};
use tabsmith_contracts::config::{NanPolicy, Task, UnknownCategoryPolicy};
use tabsmith_contracts::model::Activation;
use tabsmith_contracts::schema::RawDtype;
use tabsmith_contracts::transform::FeatureTransform;
use tabsmith_runtime::model::Model;
use tabsmith_runtime::{LoadOptions, Outcome, Predictor, Record};

use crate::names::{Namer, camel, snake};

/// `name, "path"` with an optional `, kani`.
pub struct Args {
    name: Ident,
    path: LitStr,
    kani: bool,
}

impl Parse for Args {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let name: Ident = input.parse()?;
        input.parse::<syn::Token![,]>()?;
        let path: LitStr = input.parse()?;
        let mut kani = false;
        while input.peek(syn::Token![,]) {
            input.parse::<syn::Token![,]>()?;
            if input.is_empty() {
                break;
            }
            let option: Ident = input.parse()?;
            if option == "kani" {
                kani = true;
            } else {
                return Err(syn::Error::new(
                    option.span(),
                    format!("unknown option `{option}`; the one option is `kani`"),
                ));
            }
        }
        Ok(Args { name, path, kani })
    }
}

pub fn expand(args: &Args) -> syn::Result<TokenStream> {
    let fail = |msg: String| syn::Error::new(args.path.span(), msg);
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").map_err(|_| fail("CARGO_MANIFEST_DIR is not set; build with cargo".into()))?;
    let dir = Path::new(&manifest_dir).join(args.path.value());
    let compiled = Compiled::load(&dir, &args.path.value()).map_err(|e| fail(format!("Tabsmith bundle `{}`:\n{e}", args.path.value())))?;
    Ok(compiled.generate(&args.name, args.kani))
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
                    impute, mean, std, offset, ..
                } => Spec::Numeric {
                    impute: *impute,
                    mean: *mean,
                    std: *std,
                    offset: *offset,
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
        })
    }

    fn generate(&self, name: &Ident, kani: bool) -> TokenStream {
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
        let (model_consts, outputs_body) = self.network();
        let examples: Vec<TokenStream> = self.examples.iter().map(|e| self.example(e)).collect();
        let n_examples = examples.len();
        let self_check = self.self_check();

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
             [`predict`] computes exactly what `tabsmith-runtime` computes for the same input, bit for bit; \
             [`self_check`] confirms it on the bundle's examples. `no_std`, no allocation.",
            self.path, self.summary
        );

        quote! {
            #[doc = #module_doc]
            #[allow(clippy::all, clippy::pedantic, non_camel_case_types, non_snake_case, dead_code)]
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
                pub fn features(input: &Input) -> ::core::result::Result<[f32; N_FEATURES], InputError> {
                    let mut x = [0f32; N_FEATURES];
                    #(#encode)*
                    Ok(x)
                }

                #model_consts

                /// The model's raw outputs for one row of features.
                pub fn outputs(x: &[f32; N_FEATURES]) -> [f32; N_OUTPUTS] {
                    #outputs_body
                }

                #decode

                /// Predicts one input.
                pub fn predict(input: &Input) -> ::core::result::Result<Prediction, InputError> {
                    Ok(decode(&outputs(&features(input)?)))
                }

                /// Predicts one row of raw values, in the order of [`FIELDS`].
                pub fn predict_values(values: &[FieldValue<'_>; N_FIELDS]) -> ::core::result::Result<Prediction, InputError> {
                    predict(&Input::from_values(values)?)
                }

                /// The bundle's example records (`examples/input.json`): synthetic values built
                /// from the training statistics, never training rows.
                pub const EXAMPLES: [Input; #n_examples] = [#(#examples),*];

                #self_check
            }
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
                impute, mean, std, offset, ..
            } => {
                let (impute, mean, std) = (f64_lit(*impute), f64_lit(*mean), f64_lit(*std));
                let nan = self.nan_is_missing;
                quote! {
                    x[#offset] = kernels::numeric(#field, input.#ident, &NUMERIC_TOKENS, #nan, #impute, #mean, #std)?;
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
                    pub fn decode(z: &[f32; N_OUTPUTS]) -> Prediction {
                        let probability = kernels::sigmoid(f64::from(z[0]));
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
                    pub fn decode(z: &[f32; N_OUTPUTS]) -> Prediction {
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
                    pub fn decode(z: &[f32; N_OUTPUTS]) -> Prediction {
                        Prediction { value: kernels::unscale(z[0], TARGET_MEAN, TARGET_STD) }
                    }
                };
                (types, decode)
            }
        }
    }

    fn network(&self) -> (TokenStream, TokenStream) {
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
                (quote!(#(#consts)*), quote!(#(#body)* out))
            }
            Network::Trees {
                roots,
                feature,
                threshold,
                left,
                right,
                value,
                base,
            } => {
                let (n_trees, n_nodes, n_base) = (roots.len(), feature.len(), base.len());
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

                    /// The ensemble that [`outputs`] evaluates. `TREES.leaf(t, &x)` is the walk
                    /// through tree `t`; the Kani proofs check one walk at a time.
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
                let body = quote! {
                    let mut out = [0f32; N_OUTPUTS];
                    TREES.outputs(x, &mut out);
                    out
                };
                (consts, body)
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

                    fn same_as_runtime(i: usize, p: &Prediction) -> bool {
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

                    fn same_as_runtime(i: usize, p: &Prediction) -> bool {
                        p.probabilities.iter().zip(EXPECTED[i].iter()).all(|(a, b)| a.to_bits() == *b)
                    }
                }
            }
        };
        quote! {
            #check

            /// Predicts [`EXAMPLES`] and compares every result, bit for bit, with what
            /// `tabsmith-runtime` predicted for them when this crate was compiled. Run it
            /// once on the target (a power-on self test): it fails on a platform whose
            /// floating point differs, for example one that flushes subnormals to zero.
            pub fn self_check() -> ::core::result::Result<(), SelfCheckError> {
                for (i, input) in EXAMPLES.iter().enumerate() {
                    let same = match predict(input) {
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
                let tokens = compiled.generate(&format_ident!("m"), kani);
                syn::parse2::<syn::File>(tokens).unwrap_or_else(|e| panic!("{name}: {e}"));
            }
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
