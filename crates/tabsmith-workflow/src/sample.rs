//! Deterministic synthetic "equipment defect" dataset for demos, tests and
//! usability trials. Generated data; free to use (CC0).
//!
//! Rows are readings of units on devices; several readings share a device, and a
//! hidden per-device effect makes `split = "group"` the right choice when
//! `device_id` is declared. The label depends non-linearly on a few features
//! (including an interaction), so the small MLP has something to gain over the
//! linear baseline.
//!
//! The same readings also come with a multiclass target (`failure_mode`: none,
//! bearing, overheat or seal) and a regression target (`remaining_hours` of
//! service), for the other two recipes.

use std::fmt::Write as _;
use std::path::Path;

use tabsmith_contracts::config::Task;
use tabsmith_contracts::{Phase, Result, TabsmithError};

use crate::rng::SplitMix64;

/// Variant of the generated distribution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SampleVariant {
    /// The baseline distribution.
    Base,
    /// A later data delivery: higher vibration, a new supplier `S7`, more humidity gaps.
    Shifted,
}

const LINES: [&str; 4] = ["A", "B", "C", "D"];
const SHIFTS: [&str; 3] = ["day", "night", "swing"];
const TYPES: [&str; 3] = ["pump", "valve", "motor"];
const FIRMWARE: [&str; 3] = ["v1.2", "v1.3", "v2.0"];

fn pick<'a>(rng: &mut SplitMix64, items: &[&'a str]) -> &'a str {
    items[rng.below(items.len() as u64) as usize]
}

fn fmt_f(x: f64) -> String {
    format!("{:.4}", x)
}

/// The target column of the sample data of each task.
pub fn sample_target(task: Task) -> &'static str {
    match task {
        Task::Binary => "defect",
        Task::Multiclass => "failure_mode",
        Task::Regression => "remaining_hours",
    }
}

/// Generates the CSV text of the binary sample (target `defect`).
pub fn defects_csv(rows: usize, seed: u64, variant: SampleVariant) -> String {
    sample_csv(rows, seed, variant, Task::Binary)
}

/// Generates the CSV text with the target of `task` as its last column. The
/// binary sample draws exactly as earlier versions, so its rows never change.
pub fn sample_csv(rows: usize, seed: u64, variant: SampleVariant, task: Task) -> String {
    let mut rng = SplitMix64::derive(seed, if variant == SampleVariant::Base { "defects" } else { "defects-v2" });
    let n_devices = (rows / 25).max(8);
    let device_effect: Vec<f64> = (0..n_devices).map(|_| rng.normal() * 0.8).collect();
    let device_type: Vec<&str> = (0..n_devices).map(|_| pick(&mut rng, &TYPES)).collect();
    let device_line: Vec<&str> = (0..n_devices).map(|_| pick(&mut rng, &LINES)).collect();
    let suppliers: Vec<&str> = match variant {
        SampleVariant::Base => vec!["S1", "S2", "S3", "S4", "S5", "S6"],
        SampleVariant::Shifted => vec!["S1", "S2", "S3", "S4", "S5", "S6", "S7"],
    };
    let vib_shift = if variant == SampleVariant::Shifted { 0.25 } else { 0.0 };
    let mut out = String::with_capacity(rows * 220);
    out.push_str(
        "unit_id,device_id,line,shift,supplier,device_type,firmware,hours_in_service,cycles,\
temp_mean,temp_std,temp_max,vibration_rms,vibration_peak,pressure_mean,pressure_std,\
current_mean,current_std,humidity,voltage_drop,ambient_noise_db,calibration_offset,\
maintenance_flag,inspector_note,",
    );
    out.push_str(sample_target(task));
    out.push('\n');
    // Unit ids differ between variants and seeds, so a "new delivery" generated
    // with another seed never reuses the ids of the default sample (seed 42).
    let offset =
        if variant == SampleVariant::Base { 0 } else { 500_000 } + if seed == 42 { 0 } else { (seed as usize % 1000 + 1) * 1_000_000 };
    for i in 0..rows {
        let dev = rng.below(n_devices as u64) as usize;
        let line = device_line[dev];
        let shift = pick(&mut rng, &SHIFTS);
        let supplier = suppliers[rng.below(suppliers.len() as u64) as usize];
        let dtype = device_type[dev];
        let firmware = pick(&mut rng, &FIRMWARE);
        let hours = (rng.next_f64() * 20_000.0) as i64;
        let cycles = (hours as f64 * (0.8 + 0.4 * rng.next_f64())) as i64 / 10;
        let z_hours = (hours as f64 - 10_000.0) / 5_773.0;
        let temp_mean = 62.0 + 6.0 * rng.normal() + 2.0 * z_hours;
        let temp_std = (2.0 + 0.8 * rng.normal()).abs() + 0.1;
        let temp_max = temp_mean + 2.5 * temp_std + 1.5 * rng.normal().abs();
        let vib_rms = (1.0 + vib_shift + 0.35 * rng.normal() + 0.15 * device_effect[dev]).max(0.05);
        let vib_peak = vib_rms * (2.8 + 0.4 * rng.normal().abs());
        let pressure_mean = 4.2 + 0.5 * rng.normal();
        let pressure_std = (0.3 + 0.1 * rng.normal()).abs();
        let current_mean = 11.0 + 1.5 * rng.normal() + if dtype == "motor" { 2.0 } else { 0.0 };
        let current_std = (0.8 + 0.3 * rng.normal()).abs();
        let humidity_gap = if variant == SampleVariant::Shifted { 0.12 } else { 0.05 };
        let humidity = if rng.next_f64() < humidity_gap {
            None
        } else {
            Some(45.0 + 12.0 * rng.normal())
        };
        let voltage_drop = if rng.next_f64() < 0.03 {
            f64::NAN
        } else {
            0.8 + 0.2 * rng.normal()
        };
        let noise = 50.0 + 5.0 * rng.normal();
        let maintenance = rng.next_f64() < 0.3;
        let z_vib = (vib_rms - 1.0) / 0.35;
        let z_temp = (temp_mean - 62.0) / 6.0;
        let z_tmax = (temp_max - 68.0) / 7.0;
        let z_pressure = (pressure_mean - 4.2) / 0.5;
        let target = match task {
            Task::Binary => String::new(),
            Task::Multiclass => {
                // Each failure mode has its own drivers; the mode is drawn from their softmax.
                let scores = [
                    1.6 + if maintenance { 0.6 } else { 0.0 },
                    -0.6 + 1.4 * z_vib + 0.5 * device_effect[dev] + if dtype == "motor" { 0.5 } else { 0.0 },
                    -0.9 + 1.3 * z_tmax + 0.4 * z_hours + if temp_std > 3.2 { 0.8 } else { 0.0 },
                    -1.1 + 1.1 * z_pressure
                        + if supplier == "S4" || supplier == "S5" { 0.7 } else { 0.0 }
                        + if dtype == "valve" { 0.6 } else { 0.0 },
                ];
                let e: Vec<f64> = scores.iter().map(|s| (s + 0.3 * rng.normal()).exp()).collect();
                let mut u = rng.next_f64() * e.iter().sum::<f64>();
                let mut k = 0;
                while k + 1 < e.len() && u >= e[k] {
                    u -= e[k];
                    k += 1;
                }
                ["none", "bearing", "overheat", "seal"][k].to_string()
            }
            Task::Regression => {
                // Remaining service life: shorter with age, vibration, heat and wear.
                let wear = 2600.0 * z_vib
                    + 1500.0 * z_tmax
                    + 1200.0 * (z_vib * z_temp).max(0.0)
                    + if supplier == "S4" || supplier == "S5" { 900.0 } else { 0.0 }
                    + if maintenance { -1100.0 } else { 0.0 }
                    + 800.0 * device_effect[dev];
                let remaining = 30_000.0 - hours as f64 - wear + 1_500.0 * rng.normal();
                format!("{:.1}", remaining.max(0.0))
            }
        };
        let mut logit = -2.9
            + 0.9 * z_vib
            + 0.5 * z_tmax
            + 0.35 * z_hours
            + 0.6 * z_vib * z_temp
            + if temp_std > 3.2 { 0.9 } else { 0.0 }
            + if supplier == "S4" || supplier == "S5" { 0.6 } else { 0.0 }
            + if supplier == "S7" { 0.8 } else { 0.0 }
            + if line == "D" { 0.4 } else { 0.0 }
            + if maintenance { -0.5 } else { 0.0 }
            + 0.7 * device_effect[dev];
        if firmware == "v2.0" {
            logit -= 0.3;
        }
        let target = if task == Task::Binary {
            logit += 0.4 * rng.normal();
            let p = 1.0 / (1.0 + (-logit).exp());
            u8::from(rng.next_f64() < p).to_string()
        } else {
            target
        };
        let note = format!("inspected by tech {} ref {:08x}", rng.below(60), rng.next_u64() as u32);
        let _ = writeln!(
            out,
            "U{:07},D{:04},{line},{shift},{supplier},{dtype},{firmware},{hours},{cycles},{},{},{},{},{},{},{},{},{},{},{},{},0.0,{maintenance},{note},{target}",
            i + 1 + offset,
            dev + 1,
            fmt_f(temp_mean),
            fmt_f(temp_std),
            fmt_f(temp_max),
            fmt_f(vib_rms),
            fmt_f(vib_peak),
            fmt_f(pressure_mean),
            fmt_f(pressure_std),
            fmt_f(current_mean),
            fmt_f(current_std),
            humidity.map(fmt_f).unwrap_or_default(),
            if voltage_drop.is_nan() {
                "NaN".to_string()
            } else {
                fmt_f(voltage_drop)
            },
            fmt_f(noise),
        );
    }
    out
}

/// Writes the binary sample CSV to `path`.
pub fn write_defects_csv(path: &Path, rows: usize, seed: u64, variant: SampleVariant) -> Result<()> {
    write_sample_csv(path, rows, seed, variant, Task::Binary)
}

/// Writes the sample CSV of `task` to `path`.
pub fn write_sample_csv(path: &Path, rows: usize, seed: u64, variant: SampleVariant, task: Task) -> Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| TabsmithError::io(Phase::Data, dir, e))?;
    }
    std::fs::write(path, sample_csv(rows, seed, variant, task)).map_err(|e| TabsmithError::io(Phase::Data, path, e))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_and_well_formed() {
        let a = defects_csv(500, 42, SampleVariant::Base);
        assert_eq!(a, defects_csv(500, 42, SampleVariant::Base));
        let lines: Vec<&str> = a.lines().collect();
        assert_eq!(lines.len(), 501);
        let width = lines[0].split(',').count();
        assert!(lines.iter().all(|l| l.split(',').count() == width));
        let positives = lines[1..].iter().filter(|l| l.ends_with(",1")).count();
        assert!(positives > 25 && positives < 200, "positives = {positives}");
        assert!(defects_csv(200, 42, SampleVariant::Shifted).contains(",S7,"));
    }
}
