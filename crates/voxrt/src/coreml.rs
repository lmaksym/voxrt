//! Apple Neural Engine encoder (feature `coreml`, `VOXRT_DEVICE=ane`): the 24 encoder layers as a Core ML model,
//! converted ahead of time by `tools/coreml/convert_encoder.py` into `<model>/coreml/`. The Neural Engine compiles
//! fixed shapes only, so the bundle holds one function per length bucket (`t64`, `t128`, …) over one shared copy of
//! the int8 weights; an input is padded to the smallest bucket that fits, with a key mask (softmax) and a row mask
//! (depthwise conv) keeping padding out of the valid frames. The Neural Engine computes in fp16, so the output is
//! close to, not equal to, the f32 reference.
//!
//! Routing: only buckets up to `VOXRT_ANE_MAX_FRAMES` (default 256 frames ≈ 20 s) are loaded; longer inputs return
//! `None` and the caller runs them on the GPU. Measured on an M4 Max, the Neural Engine is faster than Metal up to
//! ≈ 20 s and slower beyond (padding to the bucket and the T² attention); on Macs with a smaller GPU the crossover
//! moves up, so the limit is a setting.
//!
//! Core ML compiles each function for the Neural Engine on its first load (seconds to tens of seconds per bucket) and
//! caches the result per `.mlmodelc` and executable; later loads take ≈ 0.4 s per bucket.
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::sync::Mutex;

use half::f16;
use objc2::rc::Retained;
use objc2::runtime::{AnyObject, ProtocolObject};
use objc2::AnyThread;
use objc2_core_ml::{
    MLComputeUnits, MLDictionaryFeatureProvider, MLFeatureProvider, MLFeatureValue, MLModel,
    MLModelConfiguration, MLMultiArray, MLMultiArrayDataType,
};
use objc2_foundation::{NSArray, NSDictionary, NSNumber, NSObjectProtocol, NSString, NSURL};

use crate::{Error, Result};

/// Padding keys get this additive score (fp16-safe "minus infinity").
const MASKED: f32 = -30_000.0;
/// Default routing limit in encoder frames (see the module docs).
const DEFAULT_MAX_FRAMES: usize = 256;

struct Bucket {
    t: usize,
    model: Retained<MLModel>,
    output: Mutex<Option<Retained<NSString>>>,
}

pub struct AneEncoder {
    buckets: Vec<Bucket>, // ascending t
    d: usize,
    pub dir: PathBuf,
}

// Core ML models may be used from any thread; predictions are serialised per bucket by the caller's usage pattern
// (one segment at a time per transcriber) and Core ML itself is thread-safe for prediction.
unsafe impl Send for AneEncoder {}
unsafe impl Sync for AneEncoder {}

#[derive(serde::Deserialize)]
struct Manifest {
    buckets: Vec<usize>,
    d_model: usize,
}

fn err(what: &str, e: impl std::fmt::Display) -> Error {
    Error::Format(format!("core ml: {what}: {e}"))
}

impl AneEncoder {
    /// The bundle directory for a model: `VOXRT_COREML_DIR`, else `<model>/coreml`.
    pub fn bundle_dir(model_dir: &Path) -> PathBuf {
        std::env::var_os("VOXRT_COREML_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| model_dir.join("coreml"))
    }

    /// Load the buckets of `<dir>/encoder.mlmodelc` (listed in `<dir>/encoder.json`) up to `VOXRT_ANE_MAX_FRAMES`.
    pub fn load(dir: &Path) -> Result<Self> {
        let max = std::env::var("VOXRT_ANE_MAX_FRAMES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(DEFAULT_MAX_FRAMES);
        Self::load_upto(dir, max)
    }

    /// Load the buckets up to `max_frames` (at least the smallest one).
    pub fn load_upto(dir: &Path, max_frames: usize) -> Result<Self> {
        let json = dir.join("encoder.json");
        if !json.exists() {
            return Err(err(
                "no bundle",
                format!(
                    "{} (build it with tools/coreml/convert_encoder.py)",
                    dir.display()
                ),
            ));
        }
        let manifest: Manifest = serde_json::from_slice(&std::fs::read(json)?)?;
        let compiled = dir.join("encoder.mlmodelc");
        if !compiled.is_dir() {
            return Err(err("missing", compiled.display()));
        }
        // multi-function models need macOS 15 / iOS 18 (`MLModelConfiguration.functionName`)
        if !unsafe { MLModelConfiguration::new() }.respondsToSelector(objc2::sel!(setFunctionName:))
        {
            return Err(err("unsupported", "needs macOS 15 / iOS 18"));
        }
        let url =
            NSURL::from_directory_path(&compiled).ok_or_else(|| err("path", compiled.display()))?;
        let mut buckets = Vec::new();
        let mut ts = manifest.buckets.clone();
        ts.sort_unstable();
        let first = ts.first().copied().unwrap_or(0);
        for t in ts.into_iter().filter(|&t| t <= max_frames || t == first) {
            let model = unsafe {
                let cfg = MLModelConfiguration::new();
                cfg.setComputeUnits(MLComputeUnits::CPUAndNeuralEngine);
                cfg.setFunctionName(Some(&NSString::from_str(&format!("t{t}"))));
                MLModel::modelWithContentsOfURL_configuration_error(&url, &cfg)
            }
            .map_err(|e| err(&format!("load t{t}"), e.localizedDescription()))?;
            buckets.push(Bucket {
                t,
                model,
                output: Mutex::new(None),
            });
        }
        if buckets.is_empty() {
            return Err(err("bundle", "no buckets"));
        }
        Ok(Self {
            buckets,
            d: manifest.d_model,
            dir: dir.to_path_buf(),
        })
    }

    /// The largest input (encoder frames) routed to the Neural Engine.
    pub fn max_frames(&self) -> usize {
        self.buckets.last().map_or(0, |b| b.t)
    }

    /// Run every bucket once on silence (first predictions allocate; also wakes the Neural Engine).
    pub fn warm_up(&self) {
        for b in &self.buckets {
            let _ = self.run(b, &vec![0f32; self.d], 1);
        }
    }

    /// Encoder layers for the subsampling output `x` `[t, d]` → `[t, d]`, or `None` when `t` exceeds every bucket
    /// (or Core ML fails, which is reported once on stderr).
    pub fn forward(&self, x: &[f32], t: usize) -> Option<Vec<f32>> {
        assert_eq!(x.len(), t * self.d);
        let b = self.buckets.iter().find(|b| b.t >= t)?;
        match self.run(b, x, t) {
            Ok(y) => Some(y),
            Err(e) => {
                eprintln!("voxrt: {e}");
                None
            }
        }
    }

    fn run(&self, b: &Bucket, x: &[f32], t: usize) -> Result<Vec<f32>> {
        let (d, tb) = (self.d, b.t);
        let xa = array(&[tb, d])?;
        let km = array(&[1, 1, tb])?;
        let rm = array(&[tb, 1])?;
        unsafe {
            fill(&xa, |o| {
                for (o, v) in o.iter_mut().zip(x) {
                    *o = f16::from_f32(*v);
                }
                o[t * d..].fill(f16::ZERO);
            });
            fill(&km, |o| {
                o[..t].fill(f16::ZERO);
                o[t..].fill(f16::from_f32(MASKED));
            });
            fill(&rm, |o| {
                o[..t].fill(f16::ONE);
                o[t..].fill(f16::ZERO);
            });
        }
        let keys = [ns("x"), ns("key_mask"), ns("row_mask")];
        let vals: Vec<Retained<AnyObject>> = [&xa, &km, &rm]
            .into_iter()
            .map(|a| unsafe {
                Retained::into_super(Retained::into_super(
                    MLFeatureValue::featureValueWithMultiArray(a),
                ))
            })
            .collect();
        let dict = NSDictionary::from_retained_objects(
            &keys.iter().map(|k| &**k).collect::<Vec<_>>(),
            &vals,
        );
        let out = unsafe {
            let input = MLDictionaryFeatureProvider::initWithDictionary_error(
                MLDictionaryFeatureProvider::alloc(),
                &dict,
            )
            .map_err(|e| err("input", e.localizedDescription()))?;
            b.model
                .predictionFromFeatures_error(ProtocolObject::from_ref(&*input))
                .map_err(|e| err("predict", e.localizedDescription()))?
        };
        let name = {
            let mut g = b.output.lock().unwrap();
            g.get_or_insert_with(|| unsafe { out.featureNames().anyObject().expect("one output") })
                .clone()
        };
        let arr = unsafe { out.featureValueForName(&name) }
            .and_then(|v| unsafe { v.multiArrayValue() })
            .ok_or_else(|| err("output", "not a multi-array"))?;
        read_rows(&arr, t, d)
    }
}

fn ns(s: &str) -> Retained<NSString> {
    NSString::from_str(s)
}

fn array(shape: &[usize]) -> Result<Retained<MLMultiArray>> {
    let dims: Vec<Retained<NSNumber>> = shape.iter().map(|&n| NSNumber::new_usize(n)).collect();
    let shape = NSArray::from_retained_slice(&dims);
    unsafe {
        MLMultiArray::initWithShape_dataType_error(
            MLMultiArray::alloc(),
            &shape,
            MLMultiArrayDataType::Float16,
        )
    }
    .map_err(|e| err("array", e.localizedDescription()))
}

/// Fill a freshly created (contiguous) fp16 array.
unsafe fn fill(a: &MLMultiArray, f: impl Fn(&mut [f16]) + Clone) {
    let n: usize = unsafe { a.count() } as usize;
    let block = block2::StackBlock::new(
        move |p: NonNull<std::ffi::c_void>, _len: isize, _s: NonNull<NSArray<NSNumber>>| {
            let s = unsafe { std::slice::from_raw_parts_mut(p.as_ptr() as *mut f16, n) };
            f(s);
        },
    );
    unsafe { a.getMutableBytesWithHandler(&block) };
}

/// The first `t` rows of a `[T, d]` output (fp16 or fp32, any strides) as f32.
fn read_rows(a: &MLMultiArray, t: usize, d: usize) -> Result<Vec<f32>> {
    let strides: Vec<usize> = unsafe { a.strides() }
        .iter()
        .map(|n| n.as_usize())
        .collect();
    let dtype = unsafe { a.dataType() };
    if strides.len() != 2 {
        return Err(err("output", format!("rank {}", strides.len())));
    }
    let (rs, cs) = (strides[0], strides[1]);
    let mut y = vec![0f32; t * d];
    let yp = y.as_mut_ptr();
    let block = block2::StackBlock::new(move |p: NonNull<std::ffi::c_void>, _len: isize| {
        let y = unsafe { std::slice::from_raw_parts_mut(yp, t * d) };
        if dtype == MLMultiArrayDataType::Float16 {
            let src = p.as_ptr() as *const f16;
            for r in 0..t {
                for c in 0..d {
                    y[r * d + c] = unsafe { *src.add(r * rs + c * cs) }.to_f32();
                }
            }
        } else {
            let src = p.as_ptr() as *const f32;
            for r in 0..t {
                for c in 0..d {
                    y[r * d + c] = unsafe { *src.add(r * rs + c * cs) };
                }
            }
        }
    });
    unsafe { a.getBytesWithHandler(&block) };
    Ok(y)
}
