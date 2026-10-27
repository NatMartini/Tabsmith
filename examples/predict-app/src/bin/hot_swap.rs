//! Safe model updates: a new bundle is loaded and self-checked in a separate
//! instance before it replaces the active one; on failure the old one keeps serving.
//!
//!     cargo run --release --bin hot_swap -- <old-bundle> <new-bundle> [<broken-bundle>]
use rustpy_runtime::{LoadOptions, PredictorHandle};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let (old, new) = (args.next().ok_or("old bundle")?, args.next().ok_or("new bundle")?);
    let broken = args.next();
    let handle = PredictorHandle::load(&old, LoadOptions::cpu())?;
    println!("active: {}", handle.current().bundle_id());
    if let Some(broken) = broken {
        match handle.try_update(&broken, LoadOptions::cpu()) {
            Ok(r) => println!("unexpected: updated to {}", r.active_bundle_id),
            Err(e) => {
                println!("update refused: {} ({})", e.message, e.code);
                println!("still active: {}", handle.current().bundle_id());
            }
        }
    }
    let report = handle.try_update(&new, LoadOptions::cpu())?;
    println!(
        "updated {} -> {} ({} examples re-checked)",
        report.previous_bundle_id, report.active_bundle_id, report.examples_checked
    );
    for change in report.schema_changes.iter().chain(&report.output_changes) {
        println!("  contract change: {change}");
    }
    let back = handle.rollback_to(&old, LoadOptions::cpu())?;
    println!("rolled back to {}", back.active_bundle_id);
    Ok(())
}
