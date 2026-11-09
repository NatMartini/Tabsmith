//! Evidence for a const prediction, before it goes into tabsmith-embedded:
//! 1. `exp` as a `const fn`, ported operation for operation from libm 0.2.16 (FreeBSD's
//!    e_exp.c): the same bits as `libm::exp` for every input that is an `f32` (every input of
//!    a binary model's logistic function), and for billions of random `f64`;
//! 2. the logistic function built on it: the kernel's bits for every `f32` logit, and never
//!    smaller for a larger logit, which makes `PROBABILITY_BOUNDS` sound;
//! 3. `dot` as a `const fn`: the kernel's bits, and its speed at the lengths of real layers.
use std::hint::black_box;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::time::Instant;

use tabsmith_embedded::kernels;

const HALF: [f64; 2] = [0.5, -0.5];
const LN2HI: f64 = 6.93147180369123816490e-01;
const LN2LO: f64 = 1.90821492927058770002e-10;
const INVLN2: f64 = 1.44269504088896338700e+00;
const P1: f64 = 1.66666666666666019037e-01;
const P2: f64 = -2.77777777770155933842e-03;
const P3: f64 = 6.61375632143793436117e-05;
const P4: f64 = -1.65339022054652515390e-06;
const P5: f64 = 4.13813679705723846039e-08;

const fn scalbn(mut x: f64, mut n: i32) -> f64 {
    const EXP_MAX: i32 = 1023;
    const EXP_MIN: i32 = -1022;
    const SIG_TOTAL: i32 = 53;
    let f_exp_max = f64::from_bits(2046u64 << 52);
    let f_exp_min = f64::from_bits(1u64 << 52);
    let f_pow_subnorm = f64::from_bits(((SIG_TOTAL as u64) + 1023) << 52);
    if n > EXP_MAX {
        x *= f_exp_max;
        n -= EXP_MAX;
        if n > EXP_MAX {
            x *= f_exp_max;
            n -= EXP_MAX;
            if n > EXP_MAX {
                n = EXP_MAX;
            }
        }
    } else if n < EXP_MIN {
        let mul = f_exp_min * f_pow_subnorm;
        let add = -EXP_MIN - SIG_TOTAL;
        x *= mul;
        n += add;
        if n < EXP_MIN {
            x *= mul;
            n += add;
            if n < EXP_MIN {
                n = EXP_MIN;
            }
        }
    }
    x * f64::from_bits(((n + 1023) as u64) << 52)
}

const fn exp(mut x: f64) -> f64 {
    let x1p1023 = f64::from_bits(0x7fe0000000000000);
    let mut hx = (x.to_bits() >> 32) as u32;
    let sign = (hx >> 31) as i32;
    hx &= 0x7fffffff;
    if hx >= 0x4086232b {
        if x.is_nan() {
            return x;
        }
        if x > 709.782712893383973096 {
            x *= x1p1023;
            return x;
        }
        if x < -745.13321910194110842 {
            return 0.;
        }
    }
    let (hi, lo, k): (f64, f64, i32);
    if hx > 0x3fd62e42 {
        if hx >= 0x3ff0a2b2 {
            k = (INVLN2 * x + HALF[sign as usize]) as i32;
        } else {
            k = 1 - sign - sign;
        }
        hi = x - k as f64 * LN2HI;
        lo = k as f64 * LN2LO;
        x = hi - lo;
    } else if hx > 0x3e300000 {
        k = 0;
        hi = x;
        lo = 0.;
    } else {
        return 1. + x;
    }
    let xx = x * x;
    let c = x - xx * (P1 + xx * (P2 + xx * (P3 + xx * (P4 + xx * P5))));
    let y = 1. + (x * c / (2. - c) - lo + hi);
    if k == 0 { y } else { scalbn(y, k) }
}

const fn sigmoid(z: f64) -> f64 {
    if z >= 0.0 {
        1.0 / (1.0 + exp(-z))
    } else {
        let e = exp(z);
        e / (1.0 + e)
    }
}

/// The kernel's order of operations, with chunks the compiler knows are eight long.
const fn dot_as_chunks(a: &[f32], b: &[f32]) -> f32 {
    let (xa, ta) = a.as_chunks::<8>();
    let (xb, tb) = b.as_chunks::<8>();
    let mut acc = [0f32; 8];
    let mut c = 0;
    while c < xa.len() {
        let (x, y) = (&xa[c], &xb[c]);
        acc[0] += x[0] * y[0];
        acc[1] += x[1] * y[1];
        acc[2] += x[2] * y[2];
        acc[3] += x[3] * y[3];
        acc[4] += x[4] * y[4];
        acc[5] += x[5] * y[5];
        acc[6] += x[6] * y[6];
        acc[7] += x[7] * y[7];
        c += 1;
    }
    let mut tail = 0f32;
    let mut i = 0;
    while i < ta.len() {
        tail += ta[i] * tb[i];
        i += 1;
    }
    ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7])) + tail
}

/// The same, with the second slice cut to the first's length first, so that the compiler
/// knows both have as many chunks.
const fn dot_as_chunks_cut(a: &[f32], b: &[f32]) -> f32 {
    let (b, _) = b.split_at(a.len());
    let (xa, ta) = a.as_chunks::<8>();
    let (xb, tb) = b.as_chunks::<8>();
    let mut acc = [0f32; 8];
    let mut c = 0;
    while c < xa.len() {
        let (x, y) = (&xa[c], &xb[c]);
        let mut k = 0;
        while k < 8 {
            acc[k] += x[k] * y[k];
            k += 1;
        }
        c += 1;
    }
    let mut tail = 0f32;
    let mut i = 0;
    while i < ta.len() {
        tail += ta[i] * tb[i];
        i += 1;
    }
    ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7])) + tail
}

/// Cut, and the tail of the second slice cut to the first's, so that no index is checked.
const fn dot_cut_tail(a: &[f32], b: &[f32]) -> f32 {
    let (b, _) = b.split_at(a.len());
    let (xa, ta) = a.as_chunks::<8>();
    let (xb, tb) = b.as_chunks::<8>();
    let (tb, _) = tb.split_at(ta.len());
    let mut acc = [0f32; 8];
    let mut c = 0;
    while c < xa.len() {
        let (x, y) = (&xa[c], &xb[c]);
        let mut k = 0;
        while k < 8 {
            acc[k] += x[k] * y[k];
            k += 1;
        }
        c += 1;
    }
    let mut tail = 0f32;
    let mut i = 0;
    while i < ta.len() {
        tail += ta[i] * tb[i];
        i += 1;
    }
    ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7])) + tail
}

/// Cut, and the tail walked with `split_first` on both.
const fn dot_cut_split_tail(a: &[f32], b: &[f32]) -> f32 {
    let (b, _) = b.split_at(a.len());
    let (xa, mut ta) = a.as_chunks::<8>();
    let (xb, mut tb) = b.as_chunks::<8>();
    let mut acc = [0f32; 8];
    let mut c = 0;
    while c < xa.len() {
        let (x, y) = (&xa[c], &xb[c]);
        let mut k = 0;
        while k < 8 {
            acc[k] += x[k] * y[k];
            k += 1;
        }
        c += 1;
    }
    let mut tail = 0f32;
    while let (Some((x, na)), Some((y, nb))) = (ta.split_first(), tb.split_first()) {
        tail += *x * *y;
        ta = na;
        tb = nb;
    }
    ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7])) + tail
}

/// The trial of 2026-11-01.
const fn dot_split_first_chunk(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let (mut ra, mut rb) = (a, b);
    while let (Some((x, na)), Some((y, nb))) = (ra.split_first_chunk::<8>(), rb.split_first_chunk::<8>()) {
        let mut k = 0;
        while k < 8 {
            acc[k] += x[k] * y[k];
            k += 1;
        }
        ra = na;
        rb = nb;
    }
    let mut tail = 0f32;
    while let (Some((x, na)), Some((y, nb))) = (ra.split_first(), rb.split_first()) {
        tail += *x * *y;
        ra = na;
        rb = nb;
    }
    ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7])) + tail
}

// rustc's own floating point, on a few inputs.
const SOME: [f64; 12] = [0.0, -0.0, 1.0, -1.0, 0.5, 20.0, -20.0, 700.0, -740.0, 1e-9, 709.7, -745.0];
const AT_COMPILE_TIME: [u64; 12] = {
    let mut out = [0u64; 12];
    let mut i = 0;
    while i < 12 {
        out[i] = exp(SOME[i]).to_bits();
        i += 1;
    }
    out
};

fn threads() -> usize {
    std::thread::available_parallelism().map_or(8, |n| n.get())
}

/// The `f32` values in increasing order: key 0 is -NaN with the largest payload, the last key
/// +NaN; -0 comes just before +0.
fn f32_of_key(key: u32) -> f32 {
    f32::from_bits(if key & 0x8000_0000 != 0 { key & 0x7fff_ffff } else { !key })
}

fn exhaustive_f32() {
    let t0 = Instant::now();
    let n = threads();
    let (mismatch_exp, mismatch_sigmoid, not_monotone, checked) =
        (AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0));
    let span = (1u64 << 32) / n as u64 + 1;
    std::thread::scope(|s| {
        for part in 0..n as u64 {
            let (mismatch_exp, mismatch_sigmoid, not_monotone, checked) = (&mismatch_exp, &mismatch_sigmoid, &not_monotone, &checked);
            s.spawn(move || {
                let start = part * span;
                let end = ((part + 1) * span).min(1u64 << 32);
                // The element before the part, so that every consecutive pair is compared once.
                let mut prev: Option<f64> = if start > 0 {
                    let mut k = start - 1;
                    loop {
                        let z = f32_of_key(k as u32);
                        if !z.is_nan() {
                            break Some(kernels::sigmoid(f64::from(z)));
                        }
                        if k == 0 {
                            break None;
                        }
                        k -= 1;
                    }
                } else {
                    None
                };
                let (mut me, mut ms, mut nm, mut c) = (0u64, 0u64, 0u64, 0u64);
                for key in start..end {
                    let z = f32_of_key(key as u32);
                    let x = f64::from(z);
                    for x in [x, -x] {
                        let (a, b) = (kernels::exp(x), libm::exp(x));
                        if a.to_bits() != b.to_bits() {
                            me += 1;
                            if me <= 3 {
                                eprintln!("exp({x:e}) = {a:e}, libm {b:e}");
                            }
                        }
                    }
                    let p = kernels::sigmoid(x);
                    if p.to_bits() != libm_sigmoid(x).to_bits() && !(p.is_nan() && libm_sigmoid(x).is_nan()) {
                        ms += 1;
                    }
                    if !z.is_nan() {
                        if let Some(q) = prev {
                            if p < q {
                                nm += 1;
                                if nm <= 3 {
                                    eprintln!("sigmoid falls at {z:e}: {q:e} then {p:e}");
                                }
                            }
                        }
                        prev = Some(p);
                    }
                    c += 1;
                }
                mismatch_exp.fetch_add(me, Relaxed);
                mismatch_sigmoid.fetch_add(ms, Relaxed);
                not_monotone.fetch_add(nm, Relaxed);
                checked.fetch_add(c, Relaxed);
            });
        }
    });
    println!(
        "every f32 x (and -x): {} values, exp differs from libm at {}, the logistic function differs from 0.4.0's (on libm) at {}, \
         sigmoid falls between consecutive f32 at {} ({:.1} s, {n} threads)",
        checked.load(Relaxed),
        mismatch_exp.load(Relaxed),
        mismatch_sigmoid.load(Relaxed),
        not_monotone.load(Relaxed),
        t0.elapsed().as_secs_f64()
    );
}

fn random_f64(per_thread: u64) {
    let t0 = Instant::now();
    let n = threads();
    let (mismatch, checked) = (AtomicU64::new(0), AtomicU64::new(0));
    std::thread::scope(|s| {
        for part in 0..n as u64 {
            let (mismatch, checked) = (&mismatch, &checked);
            s.spawn(move || {
                let mut state = 0x9E37_79B9_7F4A_7C15u64 ^ (part.wrapping_mul(0xD1B5_4A32_D192_ED03) | 1);
                let mut next = move || {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    state
                };
                let (mut m, mut c) = (0u64, 0u64);
                for _ in 0..per_thread {
                    let r = next();
                    let u = (r >> 11) as f64 / (1u64 << 53) as f64;
                    let x = match r & 3 {
                        0 => f64::from_bits(next()),
                        1 => u * 1500.0 - 750.0,
                        2 => u * 60.0 - 30.0,
                        // a softmax input: the difference of two f32 logits
                        _ => f64::from(f32::from_bits(next() as u32)) - f64::from(f32::from_bits((next() >> 32) as u32)),
                    };
                    let (a, b) = (kernels::exp(x), libm::exp(x));
                    if a.to_bits() != b.to_bits() && !(a.is_nan() && b.is_nan()) {
                        m += 1;
                        if m <= 3 {
                            eprintln!("exp({x:e}) = {a:e}, libm {b:e}");
                        }
                    }
                    c += 1;
                }
                mismatch.fetch_add(m, Relaxed);
                checked.fetch_add(c, Relaxed);
            });
        }
    });
    println!(
        "random f64: {} values, exp differs from libm at {} ({:.1} s)",
        checked.load(Relaxed),
        mismatch.load(Relaxed),
        t0.elapsed().as_secs_f64()
    );
}

fn median(mut f: impl FnMut()) -> f64 {
    let mut t = Vec::new();
    for _ in 0..31 {
        let s = Instant::now();
        f();
        t.push(s.elapsed().as_secs_f64());
    }
    t.sort_by(f64::total_cmp);
    t[15]
}

fn dots() {
    type Dot = fn(&[f32], &[f32]) -> f32;
    let variants: [(&str, Dot); 6] = [
        ("kernel", kernels::dot),
        ("as_chunks", dot_as_chunks),
        ("cut", dot_as_chunks_cut),
        ("cut+tail", dot_cut_tail),
        ("cut+split tail", dot_cut_split_tail),
        ("split_first_chunk", dot_split_first_chunk),
    ];
    for len in [16usize, 32, 35, 39, 64, 128, 512] {
        let a: Vec<f32> = (0..len).map(|i| i as f32 * 0.01 - 0.3).collect();
        let b: Vec<f32> = (0..len).map(|i| 1.0 - i as f32 * 0.003).collect();
        let want = kernels::dot(&a, &b).to_bits();
        let rounds = 4_000_000 / len;
        let mut line = format!("dot of {len:>3}:");
        for (name, f) in variants {
            assert_eq!(f(&a, &b).to_bits(), want, "{name} at {len}");
            let t = median(|| {
                let mut s = 0f32;
                for _ in 0..rounds {
                    s += f(black_box(&a), black_box(&b));
                }
                black_box(s);
            });
            line += &format!("  {name} {:.2} ns", t * 1e9 / rounds as f64);
        }
        println!("{line}");
    }
}

fn main() {
    for (i, &x) in SOME.iter().enumerate() {
        assert_eq!(AT_COMPILE_TIME[i], libm::exp(x).to_bits(), "rustc's exp at {x}");
    }
    println!("rustc's constant evaluator gives libm's bits on {} inputs", SOME.len());
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "dot") {
        dots();
    }
    if args.iter().any(|a| a == "exhaustive") {
        exhaustive_f32();
    }
    if let Some(i) = args.iter().position(|a| a == "random") {
        let per_thread: u64 = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(10_000_000);
        random_f64(per_thread);
    }
}

/// The logistic function of tabsmith-embedded 0.4.0, on `libm::exp`.
fn libm_sigmoid(z: f64) -> f64 {
    if z >= 0.0 {
        1.0 / (1.0 + libm::exp(-z))
    } else {
        let e = libm::exp(z);
        e / (1.0 + e)
    }
}
