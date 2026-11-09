//! Compares the platform's logf (glibc on Linux, the UCRT on Windows) with a port of
//! musl/Arm's logf (with and without fused multiply-adds) and with the correctly rounded
//! value, for every positive finite f32. XGBoost turns a binary model's base_score into
//! its starting margin with `-logf(1.0f / p - 1.0f)`, so its logf decides that margin.
//! Build: rustc -O logf_probe.rs. Writes nothing.

const TAB: [(u64, u64); 16] = [
    (0x3ff661ec79f8f3be, 0xbfd57bf7808caade),
    (0x3ff571ed4aaf883d, 0xbfd2bef0a7c06ddb),
    (0x3ff49539f0f010b0, 0xbfd01eae7f513a67),
    (0x3ff3c995b0b80385, 0xbfcb31d8a68224e9),
    (0x3ff30d190c8864a5, 0xbfc6574f0ac07758),
    (0x3ff25e227b0b8ea0, 0xbfc1aa2bc79c8100),
    (0x3ff1bb4a4a1a343f, 0xbfba4e76ce8c0e5e),
    (0x3ff12358f08ae5ba, 0xbfb1973c5a611ccc),
    (0x3ff0953f419900a7, 0xbfa252f438e10c1e),
    (0x3ff0000000000000, 0x0000000000000000),
    (0x3fee608cfd9a47ac, 0x3faaa5aa5df25984),
    (0x3feca4b31f026aa0, 0x3fbc5e53aa362eb4),
    (0x3feb2036576afce6, 0x3fc526e57720db08),
    (0x3fe9c2d163a1aa2d, 0x3fcbc2860d224770),
    (0x3fe886e6037841ed, 0x3fd1058bc8a07ee1),
    (0x3fe767dcf5534862, 0x3fd4043057b6ee09),
];
const LN2: u64 = 0x3fe62e42fefa39ef;
const POLY: [u64; 3] = [0xbfd00ea348b88334, 0x3fd5575b0be00b6a, 0xbfdffffef20a4123];
const OFF: u32 = 0x3f330000;

fn port(x: f32, fused: bool) -> f32 {
    let mut ix = x.to_bits();
    if ix == 0x3f800000 {
        return 0.0;
    }
    if ix.wrapping_sub(0x0080_0000) >= 0x7f80_0000 - 0x0080_0000 {
        if ix.wrapping_mul(2) == 0 {
            return f32::NEG_INFINITY;
        }
        if ix == 0x7f80_0000 {
            return x;
        }
        if ix & 0x8000_0000 != 0 || ix.wrapping_mul(2) >= 0xff00_0000 {
            return f32::NAN;
        }
        ix = (x * f32::from_bits(0x4b00_0000)).to_bits();
        ix = ix.wrapping_sub(23 << 23);
    }
    let tmp = ix.wrapping_sub(OFF);
    let i = ((tmp >> (23 - 4)) % 16) as usize;
    let k = (tmp as i32) >> 23;
    let iz = ix.wrapping_sub(tmp & 0xff80_0000);
    let (invc, logc) = (f64::from_bits(TAB[i].0), f64::from_bits(TAB[i].1));
    let z = f32::from_bits(iz) as f64;
    let (a0, a1, a2) = (f64::from_bits(POLY[0]), f64::from_bits(POLY[1]), f64::from_bits(POLY[2]));
    let ln2 = f64::from_bits(LN2);
    let y = if fused {
        let r = z.mul_add(invc, -1.0);
        let y0 = (k as f64).mul_add(ln2, logc);
        let r2 = r * r;
        let y = a1.mul_add(r, a2);
        let y = a0.mul_add(r2, y);
        y.mul_add(r2, y0 + r)
    } else {
        let r = z * invc - 1.0;
        let y0 = logc + k as f64 * ln2;
        let r2 = r * r;
        let y = a1 * r + a2;
        let y = a0 * r2 + y;
        y * r2 + (y0 + r)
    };
    y as f32
}

fn main() {
    let threads = 16u32;
    let last = 0x7f7f_ffffu32;
    let chunk = last / threads + 1;
    let handles: Vec<_> = (0..threads)
        .map(|t| {
            std::thread::spawn(move || {
                // platform vs plain port, vs fused port, vs correctly rounded; plain vs fused;
                // plain vs correctly rounded; fused vs correctly rounded
                let mut n = [0u64; 6];
                let mut first: Vec<(u32, [u32; 4])> = Vec::new();
                let lo = (t * chunk).max(1);
                let hi = ((t + 1) * chunk - 1).min(last);
                for bits in lo..=hi {
                    let x = f32::from_bits(bits);
                    let p = x.ln();
                    let a = port(x, false);
                    let b = port(x, true);
                    let c = (x as f64).ln() as f32;
                    let d = [p != a, p != b, p != c, a != b, a != c, b != c];
                    for (k, &diff) in d.iter().enumerate() {
                        n[k] += diff as u64;
                    }
                    if (d[0] || d[1]) && first.len() < 4 {
                        first.push((bits, [p.to_bits(), a.to_bits(), b.to_bits(), c.to_bits()]));
                    }
                }
                (n, first)
            })
        })
        .collect();
    let mut n = [0u64; 6];
    let mut first = Vec::new();
    for h in handles {
        let (m, f) = h.join().unwrap();
        for k in 0..6 {
            n[k] += m[k];
        }
        first.extend(f);
    }
    let names = [
        "platform != port",
        "platform != port with FMA",
        "platform != correctly rounded",
        "port != port with FMA",
        "port != correctly rounded",
        "port with FMA != correctly rounded",
    ];
    println!("positive finite f32 inputs: {}", last);
    for (name, v) in names.iter().zip(n) {
        println!("{name:36} {v}");
    }
    for (bits, r) in first.iter().take(8) {
        println!("x = {:e} ({bits:#010x}): platform {:#010x} port {:#010x} fma {:#010x} cr {:#010x}", f32::from_bits(*bits), r[0], r[1], r[2], r[3]);
    }
    // XGBoost's own steps for a few base scores.
    for p in [0.127f32, 0.5, 0.3, 0.01, 0.9, 0.333] {
        let b = 1.0f32 / p - 1.0f32;
        println!(
            "p = {p}: -logf(1/p - 1): platform {:#010x} port {:#010x} fma {:#010x} cr {:#010x}",
            (-b.ln()).to_bits(),
            (-port(b, false)).to_bits(),
            (-port(b, true)).to_bits(),
            (-((b as f64).ln() as f32)).to_bits()
        );
    }
}
