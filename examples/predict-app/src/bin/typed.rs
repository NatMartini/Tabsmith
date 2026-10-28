//! Predicting from your own typed struct (fields named like the bundle's inputs).
//!
//!     cargo run --release --bin typed -- <bundle-dir>
use serde::Serialize;
use tabsmith_runtime::{LoadOptions, Predictor};

/// One sensor reading of the synthetic equipment-defect demo dataset.
#[derive(Serialize)]
struct Reading<'a> {
    line: &'a str,
    shift: &'a str,
    supplier: &'a str,
    device_type: &'a str,
    firmware: &'a str,
    hours_in_service: i64,
    cycles: i64,
    temp_mean: f64,
    temp_std: f64,
    temp_max: f64,
    vibration_rms: f64,
    vibration_peak: f64,
    pressure_mean: f64,
    pressure_std: f64,
    current_mean: f64,
    current_std: f64,
    humidity: Option<f64>,
    voltage_drop: Option<f64>,
    ambient_noise_db: f64,
    maintenance_flag: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let bundle = std::env::args().nth(1).ok_or("usage: typed <bundle-dir>")?;
    let predictor = Predictor::load(&bundle, LoadOptions::cpu())?;
    let calm = Reading {
        line: "A",
        shift: "day",
        supplier: "S1",
        device_type: "pump",
        firmware: "v2.0",
        hours_in_service: 1_200,
        cycles: 110,
        temp_mean: 58.0,
        temp_std: 1.5,
        temp_max: 62.0,
        vibration_rms: 0.8,
        vibration_peak: 2.3,
        pressure_mean: 4.1,
        pressure_std: 0.3,
        current_mean: 10.5,
        current_std: 0.7,
        humidity: Some(44.0),
        voltage_drop: Some(0.8),
        ambient_noise_db: 50.0,
        maintenance_flag: true,
    };
    let hot = Reading {
        line: "D",
        supplier: "S4",
        hours_in_service: 18_500,
        temp_mean: 74.0,
        temp_std: 4.1,
        temp_max: 86.0,
        vibration_rms: 1.9,
        vibration_peak: 5.8,
        humidity: None,
        maintenance_flag: false,
        ..calm
    };
    for (name, p) in ["calm", "hot"].iter().zip(predictor.predict_serialize(&[calm, hot])?) {
        match p.probability() {
            Some(q) => println!("{name}: {} (p = {q:.3})", p.label),
            None => println!("{name}: {}", p.label),
        }
    }
    Ok(())
}
