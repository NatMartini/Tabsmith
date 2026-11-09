//! The single-precision natural logarithm of glibc (2.28 and later) and musl, bit for bit.
//!
//! XGBoost turns a binary model's `base_score` (a probability) into its starting margin
//! with `-logf(1.0f / p - 1.0f)`, using the C library's `logf`. C libraries round `logf`
//! differently: glibc's differs from the correctly rounded value on 416,909 of the
//! 2,139,095,039 positive finite f32 inputs, and the Windows CRT's differs from glibc's on
//! 501,924. The import takes the margin XGBoost computes on Linux, with this port, so an
//! imported bundle starts from the same bits on every platform.
//!
//! A port of Arm's optimized-routines `logf`, as in musl (`src/math/logf.c`,
//! `src/math/logf_data.c`) and glibc (`sysdeps/ieee754/flt-32/e_logf.c`). It equals glibc
//! 2.43's `logf` on every positive finite f32 input, computed with and without fused
//! multiply-adds (`reports/import/`).

// Copyright (c) 2017-2018, Arm Limited.
// SPDX-License-Identifier: MIT
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this
// software and associated documentation files (the "Software"), to deal in the Software
// without restriction, including without limitation the rights to use, copy, modify, merge,
// publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons
// to whom the Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all copies or
// substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
// PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
// FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
// OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.

/// `(1/c, log(c))` for 16 points `c` spread over `[0x1.66p-1, 0x1.66p0]` (`__logf_data.tab`).
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
/// `ln(2)` (`__logf_data.ln2`).
const LN2: u64 = 0x3fe62e42fefa39ef;
/// The polynomial for `log1p(r)` (`__logf_data.poly`).
const POLY: [u64; 3] = [0xbfd00ea348b88334, 0x3fd5575b0be00b6a, 0xbfdffffef20a4123];
/// The table's subintervals start here (`OFF`).
const OFF: u32 = 0x3f33_0000;

/// `logf(x)` as glibc and musl compute it.
pub fn logf(x: f32) -> f32 {
    let mut ix = x.to_bits();
    if ix == 0x3f80_0000 {
        return 0.0;
    }
    if ix.wrapping_sub(0x0080_0000) >= 0x7f80_0000 - 0x0080_0000 {
        // Zero, subnormal, negative, infinite or NaN.
        if ix.wrapping_mul(2) == 0 {
            return f32::NEG_INFINITY;
        }
        if ix == 0x7f80_0000 {
            return x;
        }
        if ix & 0x8000_0000 != 0 || ix.wrapping_mul(2) >= 0xff00_0000 {
            return f32::NAN;
        }
        // Subnormal: scale by 2^23 and take 23 off the exponent.
        ix = (x * f32::from_bits(0x4b00_0000)).to_bits();
        ix = ix.wrapping_sub(23 << 23);
    }
    // x = 2^k z, z in [OFF, 2 OFF]; c is near the centre of z's subinterval.
    let tmp = ix.wrapping_sub(OFF);
    let i = ((tmp >> (23 - 4)) % 16) as usize;
    let k = (tmp as i32) >> 23;
    let iz = ix.wrapping_sub(tmp & 0xff80_0000);
    let (invc, logc) = (f64::from_bits(TAB[i].0), f64::from_bits(TAB[i].1));
    let z = f64::from(f32::from_bits(iz));
    let [a0, a1, a2] = POLY.map(f64::from_bits);
    // log(x) = log1p(z/c - 1) + log(c) + k ln 2
    let r = z * invc - 1.0;
    let y0 = logc + f64::from(k) * f64::from_bits(LN2);
    let r2 = r * r;
    let y = a1 * r + a2;
    let y = a0 * r2 + y;
    let y = y * r2 + (y0 + r);
    y as f32
}

#[cfg(test)]
mod tests {
    use super::logf;

    /// Inputs where glibc's `logf` is not the correctly rounded value (nor the Windows
    /// CRT's), with glibc 2.43's result.
    #[test]
    fn rounds_like_glibc_where_others_differ() {
        for (x, glibc, correctly_rounded) in [
            (0x0000_2447u32, 0xc2bc_48fdu32, 0xc2bc_48feu32),
            (0x0002_092c, 0xc2b6_f491, 0xc2b6_f492),
            (0x07f9_68be, 0xc299_ee4b, 0xc299_ee4c),
            (0x07fb_408e, 0xc299_ea87, 0xc299_ea86),
        ] {
            let x = f32::from_bits(x);
            assert_eq!(logf(x).to_bits(), glibc, "logf({x:e})");
            assert_eq!(((f64::from(x)).ln() as f32).to_bits(), correctly_rounded, "ln({x:e})");
        }
    }

    /// XGBoost's starting margin for a few base scores, as glibc gives it.
    #[test]
    fn starting_margins_of_base_scores() {
        for (p, margin) in [
            (0.127f32, 0xbff6_c076u32),
            (0.5, 0x8000_0000),
            (0.3, 0xbf58_e883),
            (0.01, 0xc093_0b39),
            (0.9, 0x400c_9f52),
        ] {
            assert_eq!((-logf(1.0f32 / p - 1.0f32)).to_bits(), margin, "p = {p}");
        }
    }

    #[test]
    fn edges() {
        assert_eq!(logf(1.0).to_bits(), 0);
        assert_eq!(logf(0.0), f32::NEG_INFINITY);
        assert_eq!(logf(f32::INFINITY), f32::INFINITY);
        assert!(logf(-1.0).is_nan() && logf(f32::NAN).is_nan());
        assert!((logf(std::f32::consts::E) - 1.0).abs() <= f32::EPSILON);
    }
}
