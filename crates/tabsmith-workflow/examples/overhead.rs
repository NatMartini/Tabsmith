//! Wrapper overhead: Tabsmith's training loop vs. a plain Burn loop on the same
//! data, model shape, optimizer and number of optimizer steps.
//!
//!     cargo run --release -p tabsmith-workflow --example overhead -- <project> direct|tabsmith [epochs]
//!
//! Run each mode in its own process so the peak RSS (VmHWM) is per mode.
//! The Tabsmith mode includes what the product adds per epoch: validation through
//! the runtime forward pass, early-stopping bookkeeping and an fsynced checkpoint.
//! Both modes optimise the same objective with the same shuffled minibatches.

use std::time::Instant;

use burn::module::Module;
use burn::nn::{Linear, LinearConfig, Relu};
use burn::optim::{AdamConfig, GradientsParams};
use burn::tensor::activation::log_sigmoid;
use burn::tensor::{Device, Tensor, TensorData};
use tabsmith_contracts::config::{Candidate, SelectionMetric};
use tabsmith_contracts::data::SplitName;
use tabsmith_workflow::train::{ResumeKey, RunControl, StartFrom, TrainInputs, model_config, train};
use tabsmith_workflow::{ConfigRequest, Workspace};

#[derive(Module, Debug)]
struct PlainMlp {
    l1: Linear,
    l2: Linear,
    l3: Linear,
    act: Relu,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let project = args.next().ok_or("project")?;
    let mode = args.next().ok_or("direct|tabsmith")?;
    let epochs: usize = args.next().map(|e| e.parse()).transpose()?.unwrap_or(30);
    let ws = Workspace::open(&project)?;
    let resolved = ws.resolve(&ConfigRequest::default())?;
    let p = ws.prepare(resolved, false)?;
    let tr = ws.split_data(&p, SplitName::Train)?;
    let va = ws.split_data(&p, SplitName::Valid)?;
    let mut cfg = p.resolved.config.train.small_mlp.clone();
    cfg.epochs = epochs;
    cfg.early_stopping_patience = 0;
    assert_eq!(cfg.hidden.len(), 2, "benchmark expects the default two hidden layers");
    let (n, d) = (tr.x.rows, tr.x.cols);
    let bs = cfg.batch_size;
    let steps = n.div_ceil(bs) * epochs;
    let t0 = Instant::now();
    match mode.as_str() {
        "direct" => {
            let device = Device::flex().autodiff();
            device.seed(7);
            let mut model = PlainMlp {
                l1: LinearConfig::new(d, cfg.hidden[0]).init(&device),
                l2: LinearConfig::new(cfg.hidden[0], cfg.hidden[1]).init(&device),
                l3: LinearConfig::new(cfg.hidden[1], 1).init(&device),
                act: Relu::new(),
            };
            let mut optim = AdamConfig::new().with_epsilon(1e-8).init();
            // Same objective as Tabsmith: class-weighted BCE + l2/2 * sum(W^2), shuffled minibatches.
            let ty = tr.y.binary().ok_or("the overhead benchmark needs a binary project")?;
            let pos = ty.iter().filter(|&&t| t).count() as f32;
            let (w_pos, w_neg) = (n as f32 / (2.0 * pos), n as f32 / (2.0 * (n as f32 - pos)));
            let l2 = cfg.l2 as f32;
            for epoch in 0..epochs {
                let perm = tabsmith_workflow::rng::epoch_permutation(7, epoch, n);
                for b in 0..n.div_ceil(bs) {
                    let rows = &perm[b * bs..((b + 1) * bs).min(n)];
                    let m = rows.len();
                    let mut xb = Vec::with_capacity(m * d);
                    let (mut yv, mut wv) = (Vec::with_capacity(m), Vec::with_capacity(m));
                    for &r in rows {
                        xb.extend_from_slice(tr.x.row(r));
                        yv.push(if ty[r] { 1f32 } else { 0.0 });
                        wv.push(if ty[r] { w_pos } else { w_neg });
                    }
                    let x = Tensor::<2>::from_data(TensorData::new(xb, [m, d]), &device);
                    let y = Tensor::<1>::from_data(TensorData::new(yv, [m]), &device);
                    let w = Tensor::<1>::from_data(TensorData::new(wv, [m]), &device);
                    let z: Tensor<1> = model
                        .l3
                        .forward(model.act.forward(model.l2.forward(model.act.forward(model.l1.forward(x)))))
                        .reshape([m]);
                    let data = ((y.neg().add_scalar(1.0) * z.clone() - log_sigmoid(z)) * w).mean();
                    let reg = model.l1.weight.val().powf_scalar(2.0).sum()
                        + model.l2.weight.val().powf_scalar(2.0).sum()
                        + model.l3.weight.val().powf_scalar(2.0).sum();
                    let loss = data + reg.mul_scalar(l2 / 2.0);
                    let grads = GradientsParams::from_grads(loss.backward(), &model);
                    model = optim.step(cfg.learning_rate, model, grads);
                }
            }
        }
        "tabsmith" | "tabsmith-final-checkpoint-only" => {
            if mode == "tabsmith-final-checkpoint-only" {
                cfg.checkpoint_every_epochs = epochs;
            }
            let dir = std::env::temp_dir().join(format!("tabsmith-overhead-{}", std::process::id()));
            let key = ResumeKey {
                snapshot_id: p.snapshot.snapshot_id.clone(),
                split_id: p.split.split_id.clone(),
                transform_id: p.fitted.transform_id.clone(),
                candidate: Candidate::SmallMlp,
                train_config: cfg.clone(),
                seed: 7,
                threshold: 0.5,
                model_config: model_config(Candidate::SmallMlp, d, &cfg),
                engine: tabsmith_workflow::device::ENGINE.into(),
                tabsmith_version: tabsmith_contracts::TABSMITH_VERSION.into(),
                n_train: n,
                platform: tabsmith_workflow::train::platform_fingerprint(),
                recipe_constants: tabsmith_workflow::train::RECIPE_CONSTANTS.to_string(),
            };
            let inputs = TrainInputs {
                candidate: Candidate::SmallMlp,
                cfg: &cfg,
                seed: 7,
                metric: SelectionMetric::BalancedAccuracy,
                max_train_seconds: 0,
                output: &p.fitted.output_schema,
                x_train: &tr.x,
                y_train: &tr.y,
                x_valid: &va.x,
                y_valid: &va.y,
                checkpoint_dir: dir.clone(),
                key,
                device: tabsmith_workflow::device::EngineDevice::Cpu,
            };
            train(&inputs, &RunControl::default(), StartFrom::Scratch, &mut |_| {})?;
            let _ = std::fs::remove_dir_all(dir);
        }
        other => return Err(format!("unknown mode {other}").into()),
    }
    let secs = t0.elapsed().as_secs_f64();
    let rss = tabsmith_workflow::device::peak_rss_mb().unwrap_or(f64::NAN);
    println!(
        "{}",
        serde_json::json!({"mode": mode, "epochs": epochs, "steps": steps, "train_rows": n, "features": d,
                           "train_seconds": secs, "peak_rss_mb": rss})
    );
    Ok(())
}
