//! Ternary packing (`thrush-ternary-v2`): 5 base-3 digits per byte, least significant digit first,
//! `trit = digit - 1`. Scales: one f16 per row per `group_size` input columns.

use half::f16;

/// Unpack one packed row into `out` (length `in_features`), values in {-1, 0, +1}.
pub fn unpack_row(packed: &[u8], in_features: usize, out: &mut [i8]) {
    debug_assert_eq!(packed.len(), in_features.div_ceil(5));
    debug_assert_eq!(out.len(), in_features);
    let full = in_features / 5;
    for (i, &b) in packed[..full].iter().enumerate() {
        let mut v = b;
        let o = &mut out[i * 5..i * 5 + 5];
        for slot in o.iter_mut() {
            *slot = (v % 3) as i8 - 1;
            v /= 3;
        }
    }
    let rem = in_features - full * 5;
    if rem > 0 {
        let mut v = packed[full];
        for slot in out[full * 5..].iter_mut() {
            *slot = (v % 3) as i8 - 1;
            v /= 3;
        }
    }
}

/// Pack a row of trits (inverse of [`unpack_row`]); pad digits are 0.
pub fn pack_row(trits: &[i8], out: &mut [u8]) {
    debug_assert_eq!(out.len(), trits.len().div_ceil(5));
    for (i, chunk) in trits.chunks(5).enumerate() {
        let mut v = 0u8;
        let mut p = 1u8;
        for &t in chunk {
            v += ((t + 1) as u8) * p;
            p = p.wrapping_mul(3);
        }
        out[i] = v;
    }
}

/// Dequantize one row: `w[c] = scales[c / group] * trit[c]` (f32).
pub fn dequant_row(
    packed: &[u8],
    scales_f16: &[u8],
    in_features: usize,
    group_size: usize,
    out: &mut [f32],
) {
    let mut trits = vec![0i8; in_features];
    unpack_row(packed, in_features, &mut trits);
    let n_groups = in_features / group_size;
    debug_assert_eq!(scales_f16.len(), n_groups * 2);
    for g in 0..n_groups {
        let s = f16::from_le_bytes([scales_f16[2 * g], scales_f16[2 * g + 1]]).to_f32();
        for c in g * group_size..(g + 1) * group_size {
            out[c] = s * trits[c] as f32;
        }
    }
}

/// Dequantize a whole `[out_features, in_features]` matrix to row-major f32.
pub fn dequant_matrix(
    qweight: &[u8],
    scales: &[u8],
    out_features: usize,
    in_features: usize,
    group_size: usize,
) -> Vec<f32> {
    let rb = in_features.div_ceil(5);
    let sb = (in_features / group_size) * 2;
    let mut w = vec![0f32; out_features * in_features];
    for r in 0..out_features {
        dequant_row(
            &qweight[r * rb..(r + 1) * rb],
            &scales[r * sb..(r + 1) * sb],
            in_features,
            group_size,
            &mut w[r * in_features..(r + 1) * in_features],
        );
    }
    w
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_ragged_widths() {
        for n in [1usize, 4, 5, 6, 127, 128, 1024, 4096] {
            let trits: Vec<i8> = (0..n).map(|i| ((i * 7 + 3) % 3) as i8 - 1).collect();
            let mut packed = vec![0u8; n.div_ceil(5)];
            pack_row(&trits, &mut packed);
            assert!(packed.iter().all(|&b| b < 243));
            let mut back = vec![0i8; n];
            unpack_row(&packed, n, &mut back);
            assert_eq!(back, trits, "n={n}");
        }
    }

    #[test]
    fn lsd_first() {
        // digits (1,2,0,1,2) → 1 + 2·3 + 0·9 + 1·27 + 2·81 = 196
        let mut out = [0i8; 5];
        unpack_row(&[196], 5, &mut out);
        assert_eq!(out, [0, 1, -1, 0, 1]);
    }
}
