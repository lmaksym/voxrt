//! Test helpers: minimal .npy reader and golden-set paths.
#![allow(dead_code)]
use std::path::{Path, PathBuf};

pub fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}
pub fn golden_dir() -> PathBuf {
    root().join("golden")
}
pub fn golden_ids() -> Vec<String> {
    let p = root().join("reference-py/golden_ids.txt");
    match std::fs::read_to_string(p) {
        Ok(s) => s.split_whitespace().map(String::from).collect(),
        Err(_) => Vec::new(),
    }
}
pub fn have_golden() -> bool {
    let ids = golden_ids();
    !ids.is_empty() && golden_dir().join(&ids[0]).join("features.npy").exists()
}

pub struct Npy {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

/// Reads little-endian f32/f64/i32/i64/i8/u8 .npy (C order) into f32.
pub fn read_npy(path: impl AsRef<Path>) -> Npy {
    let bytes =
        std::fs::read(path.as_ref()).unwrap_or_else(|e| panic!("{}: {e}", path.as_ref().display()));
    assert_eq!(&bytes[..6], b"\x93NUMPY");
    let (major, _minor) = (bytes[6], bytes[7]);
    let (hlen, start) = if major == 1 {
        (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10)
    } else {
        (
            u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
            12,
        )
    };
    let header = std::str::from_utf8(&bytes[start..start + hlen]).unwrap();
    let descr = header
        .split("'descr':")
        .nth(1)
        .unwrap()
        .trim()
        .trim_start_matches('\'')
        .split('\'')
        .next()
        .unwrap()
        .to_string();
    assert!(header.contains("'fortran_order': False"), "{header}");
    let shape_s = header
        .split("'shape':")
        .nth(1)
        .unwrap()
        .trim()
        .trim_start_matches('(')
        .split(')')
        .next()
        .unwrap();
    let shape: Vec<usize> = shape_s
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.parse().unwrap())
        .collect();
    let body = &bytes[start + hlen..];
    let n: usize = shape
        .iter()
        .product::<usize>()
        .max(if shape.is_empty() { 1 } else { 0 });
    let data: Vec<f32> = match descr.as_str() {
        "<f4" => body
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect(),
        "<f8" => body
            .as_chunks::<8>()
            .0
            .iter()
            .map(|c| f64::from_le_bytes(*c) as f32)
            .collect(),
        "<i4" => body
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| i32::from_le_bytes(*c) as f32)
            .collect(),
        "<i8" => body
            .as_chunks::<8>()
            .0
            .iter()
            .map(|c| i64::from_le_bytes(*c) as f32)
            .collect(),
        "|i1" => body.iter().map(|&b| b as i8 as f32).collect(),
        "|u1" => body.iter().map(|&b| b as f32).collect(),
        d => panic!("unsupported dtype {d}"),
    };
    assert_eq!(data.len(), n, "{}", path.as_ref().display());
    Npy { shape, data }
}

pub fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}
