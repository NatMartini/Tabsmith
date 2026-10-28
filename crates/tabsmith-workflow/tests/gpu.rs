//! GPU training (features `vulkan` and `cuda`), run when a usable GPU is present:
//! two trainings in one process give the same weights, and validation metrics stay
//! close to the CPU's. Determinism across processes and exact resume are checked
//! by `tools/gpu_certify.py`, which needs several processes.
#![cfg(any(feature = "vulkan", feature = "cuda"))]

use std::path::Path;

use tabsmith_contracts::config::Candidate;
use tabsmith_contracts::run::RunRecord;
use tabsmith_workflow::device::{EngineDevice, gpu_candidates};
use tabsmith_workflow::pipeline::RunOptions;
use tabsmith_workflow::sample::{SampleVariant, write_defects_csv};
use tabsmith_workflow::setup::{BindRequest, init_project};
use tabsmith_workflow::{ConfigRequest, Workspace};

fn project(dir: &Path) -> Workspace {
    init_project(dir, "tabular-binary", false).unwrap();
    write_defects_csv(&dir.join("data/defects.csv"), 3000, 11, SampleVariant::Base).unwrap();
    let ws = Workspace::open(dir).unwrap();
    ws.bind_data(&BindRequest {
        file: dir.join("data/defects.csv"),
        target: "defect".into(),
        positive: Some("1".into()),
        group: Some("device_id".into()),
        id: Some("unit_id".into()),
        ..Default::default()
    })
    .unwrap();
    ws
}

fn train(ws: &Workspace, device: &str, gpu: usize, precision: &str) -> RunRecord {
    let config = ConfigRequest {
        profile: None,
        overrides: vec![
            format!("execution.candidate_device=\"{device}\""),
            format!("execution.gpu={gpu}"),
            format!("execution.precision=\"{precision}\""),
            "train.small-mlp.epochs=4".into(),
        ],
        override_tables: vec![],
    };
    let opts = RunOptions {
        candidates: Some(vec![Candidate::SmallMlp]),
        ..Default::default()
    };
    ws.run(&config, &opts, &mut |_| {}).unwrap().runs.remove(0)
}

fn weights(ws: &Workspace, run: &RunRecord) -> Vec<u8> {
    std::fs::read(ws.project.run_dir(&run.run_id).join("model/model.safetensors")).unwrap()
}

#[test]
fn every_usable_gpu_trains_deterministically_and_close_to_the_cpu() {
    let (gpus, problems) = gpu_candidates();
    if gpus.is_empty() {
        eprintln!("no usable GPU, skipped: {problems:?}");
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let ws = project(tmp.path());
    let cpu = train(&ws, "cpu", 0, "fp32");
    let (mut vulkan, mut cuda) = (0, 0);
    for g in &gpus {
        let (device, index, precision, tolerance) = match &g.engine {
            EngineDevice::Vulkan { .. } => ("vulkan", &mut vulkan, "fp32", 0.01),
            EngineDevice::Cuda { .. } => ("cuda", &mut cuda, "tf32", 0.03),
            EngineDevice::Cpu | EngineDevice::Trees => continue,
        };
        let a = train(&ws, device, *index, precision);
        let b = train(&ws, device, *index, precision);
        *index += 1;
        assert_eq!(a.device_plan.device, g.engine.label());
        assert_eq!(a.device_receipt.as_ref().map(|r| r.matches_plan), Some(true));
        assert!(weights(&ws, &a) == weights(&ws, &b), "{}: two trainings differ", g.engine.label());
        assert_eq!(a.history, b.history);
        let balanced_accuracy = |r: &RunRecord| r.metrics.as_ref().unwrap().valid.get("balanced_accuracy").unwrap();
        let (ga, ca) = (balanced_accuracy(&a), balanced_accuracy(&cpu));
        assert!(
            (ga - ca).abs() <= tolerance,
            "{}: balanced accuracy {ga} vs CPU {ca}",
            g.engine.label()
        );
        eprintln!(
            "{}: balanced accuracy {ga:.4} (CPU {ca:.4}), {:.1} ms per step",
            g.engine.label(),
            a.timings.train_ms / a.steps_done.max(1) as f64
        );
    }
}
