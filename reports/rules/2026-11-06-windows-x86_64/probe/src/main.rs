//! rules-probe <bundle> <rules file> [boxes]: audits each rule with a release build.
//! rules-probe stability <bundle> <records.jsonl> <sd,sd,...> <rows>: label stability of rows.
use tabsmith_runtime::audit::{Budget, FeatureDomain, Rule, Verdict};
use tabsmith_runtime::{LoadOptions, Predictor, Record};

fn stability(bundle: &str, input: &str, sds: &str, rows: usize) {
    let p = Predictor::load(bundle, LoadOptions::cpu()).unwrap();
    let all: Vec<Record> = std::fs::read_to_string(input).unwrap().lines().filter(|l| !l.trim().is_empty()).take(2000)
        .map(|l| Record::from_json(&serde_json::from_str(l).unwrap()).unwrap()).collect();
    let rows = rows.min(all.len());
    println!("\n{bundle}: stability of {rows} rows (every {}th of the first {})", all.len() / rows, all.len());
    for sd in sds.split(',').map(|s| s.parse::<f64>().unwrap()) {
        let (mut holds, mut fails, mut open, mut largest, mut total_boxes) = (0, 0, 0, 0, 0usize);
        let t = std::time::Instant::now();
        for i in 0..rows {
            let r = &all[i * all.len() / rows];
            let rule = p.stability_rule(r, sd).unwrap();
            let a = p.audit_near(&rule, &Budget { boxes: 1_000_000, seconds: None }, r).unwrap();
            total_boxes += a.boxes;
            match a.verdict {
                Verdict::Holds(c) => { holds += 1; largest = largest.max(c.boxes()); }
                Verdict::Fails { .. } => fails += 1,
                Verdict::Undecided(_) => open += 1,
            }
        }
        let ms = t.elapsed().as_secs_f64() * 1e3 / rows as f64;
        println!("  within {sd} sd: {holds} keep their label (largest proof {largest} boxes), {fails} can change it, {open} undecided; {:.0} boxes visited, {ms:.3} ms per row", total_boxes as f64 / rows as f64);
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args[0] == "stability" {
        return stability(&args[1], &args[2], &args[3], args[4].parse().unwrap());
    }
    let (bundle, file) = (&args[0], &args[1]);
    let boxes = args.get(2).map_or(1_000_000, |b| b.parse().unwrap());
    let p = Predictor::load(bundle, LoadOptions::cpu()).unwrap();
    println!("\n{bundle}");
    for line in std::fs::read_to_string(file).unwrap().lines().map(str::trim).filter(|l| !l.is_empty() && !l.starts_with('#')) {
        let rule = Rule::parse(line).unwrap();
        let t = std::time::Instant::now();
        let audit = p.audit(&rule, &Budget { boxes, seconds: None }).unwrap();
        let total = t.elapsed().as_secs_f64();
        let what = match &audit.verdict {
            Verdict::Holds(c) => {
                let t = std::time::Instant::now();
                let ok = c.check(&p.bundle().model, &FeatureDomain::of(&p.bundle().transform));
                format!("holds: {} boxes in the proof, {} steps, depth {}; checked again {ok} in {:.2} s", c.boxes(), c.steps.len(), c.depth, t.elapsed().as_secs_f64())
            }
            Verdict::Fails { witness, prediction } => format!("fails: {} p={:?} v={:?}\n      {}", prediction.label, prediction.probability(), prediction.value(), serde_json::to_string(&witness.to_json()).unwrap()),
            Verdict::Undecided(why) => format!("undecided: {why}"),
        };
        println!("  {}\n    {what}\n    {} visited, {:.2} s in all", audit.rule, audit.boxes, total);
    }
}
