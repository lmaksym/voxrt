//! Apple GPU encoder (feature `metal`, opt-in with `VOXRT_DEVICE=gpu`): the 24 FastConformer layers run as Metal
//! compute kernels; subsampling, VAD and the decoder stay on the CPU. Ternary weights stay 2-bit on the GPU and are
//! decoded inside the GEMM; activations are f32. Output tracks the CPU f32 reference closely but not bit for bit
//! (different summation order), so this is a separate, opt-in mode.
use std::ptr::NonNull;
use std::sync::Mutex;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary,
    MTLResourceOptions, MTLSize,
};

use crate::encoder::rel_pos_table;
use crate::format::ModelFile;
use crate::tensor::load_f32_shaped;
use crate::trits::unpack_row;
use crate::{Error, Result};

type Buf = Retained<ProtocolObject<dyn MTLBuffer>>;
type Pso = Retained<ProtocolObject<dyn MTLComputePipelineState>>;
type Enc = ProtocolObject<dyn MTLComputeCommandEncoder>;
/// Receives each encoding step: its kind (for the profiler) and the closure that encodes it.
type StepSink<'a> = dyn FnMut(usize, &dyn Fn(&Enc)) + 'a;

/// Split-K policy (measured with `tests/metal_bench.rs`): the K = 4096 projections (feed-forward down) run as 4 K
/// slices plus a reduction at every length; for K = 1024 one slice is as fast or faster.
const SPLITS: usize = 4;
/// Step kinds reported by `VOXRT_PROFILE_GPU`.
const STEP_KINDS: [&str; 6] = [
    "layernorm",
    "ternary gemm",
    "attention gemm",
    "softmax",
    "conv",
    "subsampling",
];
const SUB: usize = 5;

fn gpu_err(what: &str, e: impl std::fmt::Display) -> Error {
    Error::Format(format!("metal: {what}: {e}"))
}

#[repr(C)]
struct TGemmP {
    m: u32,
    n: u32,
    k: u32,
    g: u32,
    mode: u32,
    alpha: f32,
    kchunk: u32,
}

#[repr(C)]
struct BGemmP {
    m: u32,
    n: u32,
    k: u32,
    lda: u32,
    ldb: u32,
    ldc: u32,
    sa: u32,
    sb: u32,
    sc: u32,
    trans_b: u32,
    alpha: f32,
}

#[repr(C)]
struct SoftP {
    t: u32,
    np: u32,
    scale: f32,
}

#[repr(C)]
struct SubP {
    t_in: u32,
    f_in: u32,
    t_out: u32,
    f_out: u32,
    c: u32,
    k: u32,
    s: u32,
    pad: u32,
}

#[repr(C)]
struct BiasP {
    rows: u32,
    n: u32,
    relu: u32,
}

#[repr(C)]
struct ConvP {
    t: u32,
    d: u32,
    k: u32,
}

/// Device, queue and compiled kernels.
struct Gpu {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    lib: Retained<ProtocolObject<dyn MTLLibrary>>,
    tgemm_small: Pso, // 16 × 64 tiles (≤ 64 tokens)
    tgemm_large: Pso, // 32 × 64 tiles
    tgemm_reduce: Pso,
    bgemm: Pso,
    layernorm: Pso,
    qbias: Pso,
    relsoftmax: Pso,
    conv: Pso,
    sub_conv0_dw: Pso,
    sub_dw: Pso,
    bias_act: Pso,
}

impl Gpu {
    fn new() -> Result<Self> {
        let device =
            MTLCreateSystemDefaultDevice().ok_or_else(|| gpu_err("device", "no Metal device"))?;
        let queue = device
            .newCommandQueue()
            .ok_or_else(|| gpu_err("queue", "newCommandQueue failed"))?;
        let t0 = std::time::Instant::now();
        let src = NSString::from_str(include_str!("kernels.metal"));
        let lib = device
            .newLibraryWithSource_options_error(&src, None)
            .map_err(|e| gpu_err("compile", e.localizedDescription()))?;
        if std::env::var_os("VOXRT_PROFILE").is_some() {
            eprintln!(
                "metal: shader library compiled in {:.3}s",
                t0.elapsed().as_secs_f64()
            );
        }
        let pso = |name: &str| pipeline(&device, &lib, name);
        Ok(Self {
            tgemm_small: pso("tgemm_16")?,
            tgemm_large: pso("tgemm_32")?,
            tgemm_reduce: pso("tgemm_reduce")?,
            bgemm: pso("bgemm")?,
            layernorm: pso("layernorm")?,
            qbias: pso("qbias")?,
            relsoftmax: pso("relsoftmax")?,
            conv: pso("glu_dwconv_silu")?,
            sub_conv0_dw: pso("sub_conv0_dw")?,
            sub_dw: pso("sub_dw")?,
            bias_act: pso("bias_act")?,
            device,
            queue,
            lib,
        })
    }

    fn buf(&self, bytes: usize) -> Buf {
        self.device
            .newBufferWithLength_options(bytes.max(16), MTLResourceOptions::StorageModeShared)
            .expect("metal buffer allocation")
    }

    fn upload<T: Copy>(&self, data: &[T]) -> Buf {
        let b = self.buf(std::mem::size_of_val(data));
        write(&b, data);
        b
    }
}

fn pipeline(
    device: &ProtocolObject<dyn MTLDevice>,
    lib: &ProtocolObject<dyn MTLLibrary>,
    name: &str,
) -> Result<Pso> {
    let f = lib
        .newFunctionWithName(&NSString::from_str(name))
        .ok_or_else(|| gpu_err("function", name))?;
    device
        .newComputePipelineStateWithFunction_error(&f)
        .map_err(|e| gpu_err(name, e.localizedDescription()))
}

/// GEMM micro-benchmark (development): seconds per ternary GEMM with kernel `name` (output tile `bm × bn`) on
/// `[m, k] · [n, k]ᵀ`, split over `splits` K slices (plus the reduction), averaged over `reps` in one command buffer.
#[allow(clippy::too_many_arguments)]
pub fn bench_tgemm(
    name: &str,
    bm: usize,
    bn: usize,
    threads: usize,
    m: usize,
    n: usize,
    k: usize,
    splits: usize,
    reps: usize,
) -> Result<f64> {
    let gpu = Gpu::new()?;
    let pso = pipeline(&gpu.device, &gpu.lib, name)?;
    let codes: Vec<u8> = (0..n * k / 4)
        .map(|i| (i * 37 % 255) as u8 & 0b1010_1010)
        .collect();
    let scales = vec![0.02f32; n * k / 128];
    let x: Vec<f32> = (0..m * k)
        .map(|i| ((i * 13 % 101) as f32 - 50.0) * 0.01)
        .collect();
    let l = TLin {
        w: gpu.upload(&codes),
        s: gpu.upload(&scales),
        out: n,
        inp: k,
        group: 128,
    };
    let (xb, yb, pb) = (
        gpu.upload(&x),
        gpu.buf(m * n * 4),
        gpu.buf(splits * m * n * 4),
    );
    let p = TGemmP {
        m: m as u32,
        n: n as u32,
        k: k as u32,
        g: 128,
        mode: 0,
        alpha: 0.0,
        kchunk: (k / splits) as u32,
    };
    let sp = splits as u32;
    let run = |reps: usize| {
        let cb = gpu.queue.commandBuffer().expect("cb");
        let enc = cb.computeCommandEncoder().expect("enc");
        for _ in 0..reps {
            bind(&enc, &pso, &[(&xb, 0), (&l.w, 0), (&l.s, 0), (&yb, 0)], &p);
            // SAFETY: partial-sum buffer sized splits × m × n floats.
            unsafe { enc.setBuffer_offset_atIndex(Some(&pb), 0, 5) };
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                size(n.div_ceil(bn), m.div_ceil(bm), splits),
                size(threads, 1, 1),
            );
            if splits > 1 {
                bind(&enc, &gpu.tgemm_reduce, &[(&pb, 0), (&yb, 0)], &p);
                // SAFETY: plain u32 constant.
                unsafe { enc.setBytes_length_atIndex(NonNull::from(&sp).cast(), 4, 3) };
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    size(n.div_ceil(256), m, 1),
                    size(256, 1, 1),
                );
            }
        }
        enc.endEncoding();
        cb.commit();
        cb.waitUntilCompleted();
    };
    run(3);
    let t0 = std::time::Instant::now();
    run(reps);
    Ok(t0.elapsed().as_secs_f64() / reps as f64)
}

fn write<T: Copy>(b: &Buf, data: &[T]) {
    let n = std::mem::size_of_val(data);
    assert!(n <= b.length());
    // SAFETY: shared-storage buffer of at least `n` bytes; no GPU work is in flight on it (callers serialise).
    unsafe {
        std::ptr::copy_nonoverlapping(
            data.as_ptr() as *const u8,
            b.contents().as_ptr() as *mut u8,
            n,
        )
    };
}

fn read(b: &Buf, out: &mut [f32]) {
    let n = std::mem::size_of_val(out);
    assert!(n <= b.length());
    // SAFETY: as above; the command buffer has completed.
    unsafe {
        std::ptr::copy_nonoverlapping(
            b.contents().as_ptr() as *const u8,
            out.as_mut_ptr() as *mut u8,
            n,
        )
    };
}

fn size(w: usize, h: usize, d: usize) -> MTLSize {
    MTLSize {
        width: w,
        height: h,
        depth: d,
    }
}

/// Bind `buffers` (with byte offsets) at indices 0.., then `params` right after.
fn bind<P>(enc: &Enc, pso: &Pso, buffers: &[(&Buf, usize)], params: &P) {
    enc.setComputePipelineState(pso);
    for (i, (b, off)) in buffers.iter().enumerate() {
        // SAFETY: valid buffer and in-bounds offset (callers size every buffer for the dispatch).
        unsafe { enc.setBuffer_offset_atIndex(Some(b), *off, i) };
    }
    // SAFETY: `params` is a plain #[repr(C)] value matching the kernel's constant struct.
    unsafe {
        enc.setBytes_length_atIndex(
            NonNull::from(params).cast(),
            std::mem::size_of::<P>(),
            buffers.len(),
        )
    };
}

/// Ternary linear on the GPU: 2-bit codes `[out, in/4]` (code = trit + 1) and f32 scales `[out, in/group]`.
struct TLin {
    w: Buf,
    s: Buf,
    out: usize,
    inp: usize,
    group: usize,
}

impl TLin {
    fn load(gpu: &Gpu, m: &ModelFile, names: &[&str]) -> Result<Self> {
        let (mut codes, mut scales) = (Vec::<u8>::new(), Vec::<f32>::new());
        let (mut out, mut inp, mut group) = (0, 0, 0);
        for name in names {
            let q = m
                .ternary
                .quantized_modules
                .iter()
                .find(|q| q.name == *name)
                .ok_or_else(|| Error::Format(format!("{name} is not a ternary module")))?;
            inp = q.in_features;
            group = q.group_size;
            out += q.out_features;
            let (_, qw) = m.tensor(&format!("{name}.qweight"))?;
            let (_, sc) = m.tensor(&format!("{name}.scales"))?;
            let rb = inp.div_ceil(5);
            let mut row = vec![0i8; inp];
            for r in 0..q.out_features {
                unpack_row(&qw[r * rb..(r + 1) * rb], inp, &mut row);
                for quad in row.chunks_exact(4) {
                    let mut b = 0u8;
                    for (j, &t) in quad.iter().enumerate() {
                        b |= ((t + 1) as u8) << (2 * j);
                    }
                    codes.push(b);
                }
            }
            scales.extend(
                sc.chunks_exact(2)
                    .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32()),
            );
        }
        if inp % 32 != 0 || out % 64 != 0 || group % 32 != 0 {
            return Err(Error::Format(format!(
                "metal: unsupported linear shape {out}x{inp} g{group}"
            )));
        }
        Ok(Self {
            w: gpu.upload(&codes),
            s: gpu.upload(&scales),
            out,
            inp,
            group,
        })
    }
}

struct LayerNorm {
    g: Buf,
    b: Buf,
}

struct Layer {
    norms: [LayerNorm; 5], // ff1, attn, conv, ff2, out
    ff1: (TLin, TLin),
    ff2: (TLin, TLin),
    qkv: TLin,
    pos: TLin,
    o: TLin,
    bias_u: Buf,
    bias_v: Buf,
    pw1: TLin,
    pw2: TLin,
    dw: Buf,
    dw_bias: Buf,
    k: usize,
}

/// Subsampling weights on the GPU (same layouts as the CPU module: taps [k*k, C], pointwise [C_out, C_in], linear
/// [d, F3*C] with channels-last columns).
struct Sub {
    w0: Buf,
    b0: Buf,
    wd: [Buf; 2],
    bd: [Buf; 2],
    pw: [Buf; 2],
    pb: [Buf; 2],
    lw: Buf,
    lb: Buf,
    c: usize,
    k: usize,
    s: usize,
    n_mels: usize,
}

/// Subsampling scratch (grown on demand): features, stage-1 depthwise output, pointwise outputs.
struct SubWs {
    feats: Buf,
    y1: Buf,
    x2: Buf,
    y2: Buf,
    x3: Buf,
}

/// Per-length scratch buffers (grown on demand).
struct Ws {
    t: usize,
    x: [Buf; 2],
    h: Buf,
    u: Buf,
    qkv: Buf,
    pos: Buf,
    relk: Buf,
    qu: Buf,
    qv: Buf,
    s1: Buf,
    s2: Buf,
    p: Buf,
    ctx: Buf,
    a: Buf,
    partial: Buf, // split-K partial sums
}

pub struct GpuEncoder {
    gpu: Gpu,
    sub: Sub,
    sub_ws: Mutex<Option<(usize, SubWs)>>,
    layers: Vec<Layer>,
    d: usize,
    heads: usize,
    ff: usize,
    ws: Mutex<Option<Ws>>,
}

// SAFETY: Metal devices, queues, pipeline states and buffers are thread-safe objects; all use of the mutable
// workspace and of command encoding goes through `ws`'s mutex.
unsafe impl Send for GpuEncoder {}
unsafe impl Sync for GpuEncoder {}

impl GpuEncoder {
    pub fn load(m: &ModelFile) -> Result<Self> {
        let gpu = Gpu::new()?;
        let e = &m.config.encoder_config;
        let (d, heads, k) = (e.hidden_size, e.num_attention_heads, e.conv_kernel_size);
        let mut layers = Vec::with_capacity(e.num_hidden_layers);
        for i in 0..e.num_hidden_layers {
            let p = format!("encoder.layers.{i}");
            let ln = |name: &str| -> Result<LayerNorm> {
                Ok(LayerNorm {
                    g: gpu.upload(&load_f32_shaped(m, &format!("{p}.{name}.weight"), &[d])?),
                    b: gpu.upload(&load_f32_shaped(m, &format!("{p}.{name}.bias"), &[d])?),
                })
            };
            let lin = |names: &[&str]| -> Result<TLin> {
                let full: Vec<String> = names.iter().map(|n| format!("{p}.{n}")).collect();
                let refs: Vec<&str> = full.iter().map(String::as_str).collect();
                TLin::load(&gpu, m, &refs)
            };
            // depthwise conv with BatchNorm(eval) folded in, tap-major [k, d] (as on the CPU)
            let c = format!("{p}.conv");
            let dw = load_f32_shaped(m, &format!("{c}.depthwise_conv.weight"), &[d, 1, k])?;
            let gamma = load_f32_shaped(m, &format!("{c}.norm.weight"), &[d])?;
            let beta = load_f32_shaped(m, &format!("{c}.norm.bias"), &[d])?;
            let mean = load_f32_shaped(m, &format!("{c}.norm.running_mean"), &[d])?;
            let var = load_f32_shaped(m, &format!("{c}.norm.running_var"), &[d])?;
            let mut dw_t = vec![0f32; k * d];
            let mut dw_bias = vec![0f32; d];
            for ch in 0..d {
                let s = gamma[ch] / (var[ch] + 1e-5).sqrt();
                for kk in 0..k {
                    dw_t[kk * d + ch] = dw[ch * k + kk] * s;
                }
                dw_bias[ch] = beta[ch] - mean[ch] * s;
            }
            let a = format!("{p}.self_attn");
            layers.push(Layer {
                norms: [
                    ln("norm_feed_forward1")?,
                    ln("norm_self_att")?,
                    ln("norm_conv")?,
                    ln("norm_feed_forward2")?,
                    ln("norm_out")?,
                ],
                ff1: (
                    lin(&["feed_forward1.linear1"])?,
                    lin(&["feed_forward1.linear2"])?,
                ),
                ff2: (
                    lin(&["feed_forward2.linear1"])?,
                    lin(&["feed_forward2.linear2"])?,
                ),
                qkv: lin(&["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"])?,
                pos: lin(&["self_attn.relative_k_proj"])?,
                o: lin(&["self_attn.o_proj"])?,
                bias_u: gpu.upload(&load_f32_shaped(
                    m,
                    &format!("{a}.bias_u"),
                    &[heads, d / heads],
                )?),
                bias_v: gpu.upload(&load_f32_shaped(
                    m,
                    &format!("{a}.bias_v"),
                    &[heads, d / heads],
                )?),
                pw1: lin(&["conv.pointwise_conv1"])?,
                pw2: lin(&["conv.pointwise_conv2"])?,
                dw: gpu.upload(&dw_t),
                dw_bias: gpu.upload(&dw_bias),
                k,
            });
        }
        let ff = layers[0].ff1.0.out;
        let cpu_sub = crate::subsampling::Subsampling::load(m)?;
        let sub = Sub {
            w0: gpu.upload(&cpu_sub.conv0_w),
            b0: gpu.upload(&cpu_sub.conv0_b),
            wd: [gpu.upload(&cpu_sub.dw_w[0]), gpu.upload(&cpu_sub.dw_w[1])],
            bd: [gpu.upload(&cpu_sub.dw_b[0]), gpu.upload(&cpu_sub.dw_b[1])],
            pw: [gpu.upload(&cpu_sub.pw_w[0]), gpu.upload(&cpu_sub.pw_w[1])],
            pb: [gpu.upload(&cpu_sub.pw_b[0]), gpu.upload(&cpu_sub.pw_b[1])],
            lw: gpu.upload(&cpu_sub.lin_w),
            lb: gpu.upload(&cpu_sub.lin_b),
            c: cpu_sub.channels,
            k: cpu_sub.kernel,
            s: cpu_sub.stride,
            n_mels: cpu_sub.n_mels,
        };
        Ok(Self {
            gpu,
            sub,
            sub_ws: Mutex::new(None),
            layers,
            d,
            heads,
            ff,
            ws: Mutex::new(None),
        })
    }

    fn workspace(&self, t: usize) -> Ws {
        let (g, d, h) = (&self.gpu, self.d, self.heads);
        let np = 2 * t - 1;
        let f = std::mem::size_of::<f32>();
        Ws {
            t,
            x: [g.buf(t * d * f), g.buf(t * d * f)],
            h: g.buf(t * d * f),
            u: g.buf(t * self.ff.max(2 * d) * f),
            qkv: g.buf(t * 3 * d * f),
            pos: g.buf(np * d * f),
            relk: g.buf(np * d * f),
            qu: g.buf(t * d * f),
            qv: g.buf(t * d * f),
            s1: g.buf(h * t * t * f),
            s2: g.buf(h * t * np * f),
            p: g.buf(h * t * t * f),
            ctx: g.buf(t * d * f),
            a: g.buf(t * d * f),
            partial: g.buf(SPLITS * t * d * f),
        }
    }

    /// Encoder layers on `x` (`[t, d]`, the subsampling output, computed elsewhere) → `[t, d]`.
    pub fn forward(&self, x: &[f32], t: usize) -> Vec<f32> {
        assert_eq!(x.len(), t * self.d);
        self.run(t, Some(x), None)
    }

    /// The whole encoder on the GPU: log-mel `feats` `[t_in, n_mels]` → subsampling → layers → `([t_e, d], t_e)`.
    pub fn forward_feats(&self, feats: &[f32], t_in: usize) -> (Vec<f32>, usize) {
        let te = self.sub_grids(t_in)[3].0;
        (self.run(te, None, Some((feats, t_in))), te)
    }

    /// Prepare buffers and pipelines for inputs up to `t` encoder frames by running the encoder once on silence
    /// (moves first-call latency to load time; also wakes the GPU, e.g. when recording starts).
    pub fn warm_up(&self, t: usize) {
        let t_in = t * 8;
        let feats = vec![0f32; t_in * self.sub.n_mels];
        let _ = self.forward_feats(&feats, t_in);
    }

    /// (t, f) grids of the subsampling stages: input, conv0 output, depthwise 1, depthwise 2.
    fn sub_grids(&self, t_in: usize) -> [(usize, usize); 4] {
        let (k, s) = (self.sub.k, self.sub.s);
        let pad = (k - 1) / 2;
        let next = |(t, f): (usize, usize)| {
            (
                crate::subsampling::out_len(t, k, s, pad),
                crate::subsampling::out_len(f, k, s, pad),
            )
        };
        let g0 = (t_in, self.sub.n_mels);
        let g1 = next(g0);
        let g2 = next(g1);
        [g0, g1, g2, next(g2)]
    }

    /// Subsampling only (log-mel `feats` `[t_in, n_mels]` → `([t_e, d], t_e)`): the VAD head's input in long-form
    /// segmentation.
    pub fn subsample(&self, feats: &[f32], t_in: usize) -> (Vec<f32>, usize) {
        let te = self.sub_grids(t_in)[3].0;
        (self.run_with(te, None, Some((feats, t_in)), false), te)
    }

    fn run(&self, t: usize, x: Option<&[f32]>, feats: Option<(&[f32], usize)>) -> Vec<f32> {
        self.run_with(t, x, feats, true)
    }

    fn run_with(
        &self,
        t: usize,
        x: Option<&[f32]>,
        feats: Option<(&[f32], usize)>,
        with_layers: bool,
    ) -> Vec<f32> {
        let d = self.d;
        let layers: &[Layer] = if with_layers { &self.layers } else { &[] };
        let mut guard = self.ws.lock().unwrap();
        if guard.as_ref().is_none_or(|w| w.t < t) {
            *guard = Some(self.workspace((t + t / 4).max(64))); // grow in 25 % steps
        }
        let ws = guard.as_ref().unwrap();
        let mut sub_guard = self.sub_ws.lock().unwrap();
        if let Some((f, t_in)) = feats {
            let g = self.sub_grids(t_in);
            // stage-1 buffers hold [t2·f2, C], stage-2 buffers [t3·f3, C] (4× smaller); grow by 25 % steps
            let need = (g[2].0 * g[2].1 * self.sub.c).max(1);
            if sub_guard.as_ref().is_none_or(|(cap, _)| *cap < need) {
                let n = need + need / 4;
                let n2 = n.div_ceil(4) + 4 * self.sub.c * g[3].1;
                let b = |floats: usize| self.gpu.buf(floats * 4);
                *sub_guard = Some((
                    n,
                    SubWs {
                        feats: b(f.len()),
                        y1: b(n),
                        x2: b(n),
                        y2: b(n2),
                        x3: b(n2),
                    },
                ));
            }
            let sw = &mut sub_guard.as_mut().unwrap().1;
            if sw.feats.length() < std::mem::size_of_val(f) {
                sw.feats = self.gpu.buf(std::mem::size_of_val(f));
            }
            write(&sw.feats, f);
        } else if let Some(x) = x {
            write(&ws.x[0], x);
        }
        write(&ws.pos, &rel_pos_table(t, d));
        let sub = feats.map(|(_, t_in)| (self.sub_grids(t_in), &sub_guard.as_ref().unwrap().1));
        let mut cur = 0;
        if std::env::var_os("VOXRT_PROFILE_GPU").is_some() {
            // one command buffer per step, GPU time per step kind (diagnostics; much slower overall)
            let mut gpu_s = [0f64; STEP_KINDS.len()];
            let mut sink = |kind: usize, f: &dyn Fn(&Enc)| {
                let cb = self
                    .gpu
                    .queue
                    .commandBuffer()
                    .expect("metal command buffer");
                let enc = cb.computeCommandEncoder().expect("metal compute encoder");
                f(&enc);
                enc.endEncoding();
                cb.commit();
                cb.waitUntilCompleted();
                gpu_s[kind] += cb.GPUEndTime() - cb.GPUStartTime();
            };
            if let Some((g, sw)) = sub {
                self.encode_subsampling(&g, sw, ws, &mut sink);
            }
            for l in layers {
                self.encode_layer(l, ws, cur, t, &mut sink);
                cur ^= 1;
            }
            let total: f64 = gpu_s.iter().sum();
            let parts: Vec<String> = STEP_KINDS
                .iter()
                .zip(&gpu_s)
                .map(|(k, s)| format!("{k} {:.1} ms", s * 1e3))
                .collect();
            eprintln!(
                "gpu profile (T_e={t}): {:.1} ms GPU · {}",
                total * 1e3,
                parts.join(" · ")
            );
        } else {
            let cb = self
                .gpu
                .queue
                .commandBuffer()
                .expect("metal command buffer");
            let enc = cb.computeCommandEncoder().expect("metal compute encoder");
            if let Some((g, sw)) = sub {
                self.encode_subsampling(&g, sw, ws, &mut |_, f| f(&enc));
            }
            for l in layers {
                self.encode_layer(l, ws, cur, t, &mut |_, f| f(&enc));
                cur ^= 1;
            }
            enc.endEncoding();
            cb.commit();
            cb.waitUntilCompleted();
            if let Some(e) = cb.error() {
                panic!("metal: command buffer failed: {}", e.localizedDescription());
            }
        }
        let mut out = vec![0f32; t * d];
        read(&ws.x[cur], &mut out);
        out
    }

    /// Subsampling into `ws.x[0]`: fused conv0+ReLU+depthwise → pointwise+ReLU → depthwise → pointwise+ReLU →
    /// linear (+bias), channels-last as on the CPU.
    fn encode_subsampling(
        &self,
        g: &[(usize, usize); 4],
        sw: &SubWs,
        ws: &Ws,
        step: &mut StepSink,
    ) {
        let (c, k, s, d) = (self.sub.c, self.sub.k, self.sub.s, self.d);
        let pad = (k - 1) / 2;
        let u = |v: usize| v as u32;
        let grid3 = |(t, f): (usize, usize)| size(c.div_ceil(64), f, t);
        let stage = |gi: (usize, usize), go: (usize, usize)| SubP {
            t_in: u(gi.0),
            f_in: u(gi.1),
            t_out: u(go.0),
            f_out: u(go.1),
            c: u(c),
            k: u(k),
            s: u(s),
            pad: u(pad),
        };
        let sp1 = stage(g[1], g[2]);
        let in0 = [u(g[0].0), u(g[0].1)];
        step(SUB, &|e| {
            let bufs = [
                (&sw.feats, 0),
                (&self.sub.w0, 0),
                (&self.sub.b0, 0),
                (&self.sub.wd[0], 0),
                (&self.sub.bd[0], 0),
                (&sw.y1, 0),
            ];
            bind(e, &self.gpu.sub_conv0_dw, &bufs, &sp1);
            // SAFETY: plain [u32; 2] constant at slot 7.
            unsafe { e.setBytes_length_atIndex(NonNull::from(&in0).cast(), 8, 7) };
            e.dispatchThreadgroups_threadsPerThreadgroup(grid3(g[2]), size(64, 1, 1));
        });
        let pw = |e: &Enc, x: &Buf, y: &Buf, i: usize, rows: usize| {
            let p = BGemmP {
                m: u(rows),
                n: u(c),
                k: u(c),
                lda: u(c),
                ldb: u(c),
                ldc: u(c),
                sa: 0,
                sb: 0,
                sc: 0,
                trans_b: 1,
                alpha: 1.0,
            };
            self.bgemm(e, (x, 0), (&self.sub.pw[i], 0), (y, 0), p, 1);
            let bp = BiasP {
                rows: u(rows),
                n: u(c),
                relu: 1,
            };
            bind(e, &self.gpu.bias_act, &[(y, 0), (&self.sub.pb[i], 0)], &bp);
            e.dispatchThreadgroups_threadsPerThreadgroup(
                size(c.div_ceil(256), rows, 1),
                size(256, 1, 1),
            );
        };
        step(SUB, &|e| pw(e, &sw.y1, &sw.x2, 0, g[2].0 * g[2].1));
        let sp2 = stage(g[2], g[3]);
        step(SUB, &|e| {
            bind(
                e,
                &self.gpu.sub_dw,
                &[
                    (&sw.x2, 0),
                    (&self.sub.wd[1], 0),
                    (&self.sub.bd[1], 0),
                    (&sw.y2, 0),
                ],
                &sp2,
            );
            e.dispatchThreadgroups_threadsPerThreadgroup(grid3(g[3]), size(64, 1, 1));
        });
        step(SUB, &|e| pw(e, &sw.y2, &sw.x3, 1, g[3].0 * g[3].1));
        let (te, flat) = (g[3].0, g[3].1 * c);
        step(SUB, &|e| {
            let p = BGemmP {
                m: u(te),
                n: u(d),
                k: u(flat),
                lda: u(flat),
                ldb: u(flat),
                ldc: u(d),
                sa: 0,
                sb: 0,
                sc: 0,
                trans_b: 1,
                alpha: 1.0,
            };
            self.bgemm(e, (&sw.x3, 0), (&self.sub.lw, 0), (&ws.x[0], 0), p, 1);
            let bp = BiasP {
                rows: u(te),
                n: u(d),
                relu: 0,
            };
            bind(
                e,
                &self.gpu.bias_act,
                &[(&ws.x[0], 0), (&self.sub.lb, 0)],
                &bp,
            );
            e.dispatchThreadgroups_threadsPerThreadgroup(
                size(d.div_ceil(256), te, 1),
                size(256, 1, 1),
            );
        });
    }

    fn ln(&self, enc: &Enc, n: &LayerNorm, x: &Buf, y: &Buf, t: usize) {
        let d = self.d as u32;
        bind(
            enc,
            &self.gpu.layernorm,
            &[(x, 0), (&n.g, 0), (&n.b, 0), (y, 0)],
            &d,
        );
        enc.dispatchThreadgroups_threadsPerThreadgroup(size(t, 1, 1), size(256, 1, 1));
    }

    /// Ternary GEMM with the policy measured by `tests/metal_bench.rs`: 16 × 64 tiles up to 64 tokens, 32 × 64 above,
    /// split-K (plus a reduction) for the K = 4096 projections.
    #[allow(clippy::too_many_arguments)]
    fn tgemm(
        &self,
        enc: &Enc,
        ws: &Ws,
        l: &TLin,
        x: &Buf,
        y: &Buf,
        t: usize,
        mode: u32,
        alpha: f32,
    ) {
        let (pso, bm) = if t <= 64 {
            (&self.gpu.tgemm_small, 16)
        } else {
            (&self.gpu.tgemm_large, 32)
        };
        let splits = if l.inp >= 4096 && l.out <= self.d {
            SPLITS
        } else {
            1
        };
        let p = TGemmP {
            m: t as u32,
            n: l.out as u32,
            k: l.inp as u32,
            g: l.group as u32,
            mode,
            alpha,
            kchunk: (l.inp / splits) as u32,
        };
        bind(enc, pso, &[(x, 0), (&l.w, 0), (&l.s, 0), (y, 0)], &p);
        // SAFETY: the partial buffer holds SPLITS × t × d floats and splitting happens only for outputs ≤ d (with one
        // slice the kernel never touches it).
        unsafe { enc.setBuffer_offset_atIndex(Some(&ws.partial), 0, 5) };
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            size(l.out / 64, t.div_ceil(bm), splits),
            size(128, 1, 1),
        );
        if splits > 1 {
            let sp = splits as u32;
            bind(enc, &self.gpu.tgemm_reduce, &[(&ws.partial, 0), (y, 0)], &p);
            // SAFETY: plain u32 constant at slot 3.
            unsafe { enc.setBytes_length_atIndex(NonNull::from(&sp).cast(), 4, 3) };
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                size(l.out.div_ceil(256), t, 1),
                size(256, 1, 1),
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn bgemm(
        &self,
        enc: &Enc,
        a: (&Buf, usize),
        b: (&Buf, usize),
        c: (&Buf, usize),
        p: BGemmP,
        batch: usize,
    ) {
        let f = std::mem::size_of::<f32>();
        let grid = size(
            (p.n as usize).div_ceil(32),
            (p.m as usize).div_ceil(32),
            batch,
        );
        bind(
            enc,
            &self.gpu.bgemm,
            &[(a.0, a.1 * f), (b.0, b.1 * f), (c.0, c.1 * f)],
            &p,
        );
        enc.dispatchThreadgroups_threadsPerThreadgroup(grid, size(128, 1, 1));
    }

    /// Encode one layer as a sequence of steps; `step(kind, f)` runs `f` on an encoder (all steps share one in normal
    /// mode; the profiler gives each its own command buffer).
    fn encode_layer(&self, l: &Layer, ws: &Ws, cur: usize, t: usize, step: &mut StepSink) {
        let (d, h) = (self.d, self.heads);
        let dh = d / h;
        let np = 2 * t - 1;
        let (x, xn) = (&ws.x[cur], &ws.x[cur ^ 1]);
        let (du, tu, npu, dhu) = (d as u32, t as u32, np as u32, dh as u32);
        const LN: usize = 0;
        const GEMM: usize = 1;
        const ATTN: usize = 2;
        const SOFT: usize = 3;
        const CONV: usize = 4;
        // x += ½ · W2(silu(W1(LN(x))))
        step(LN, &|e| self.ln(e, &l.norms[0], x, &ws.h, t));
        step(GEMM, &|e| {
            self.tgemm(e, ws, &l.ff1.0, &ws.h, &ws.u, t, 1, 0.0)
        });
        step(GEMM, &|e| self.tgemm(e, ws, &l.ff1.1, &ws.u, x, t, 2, 0.5));
        // relative-position self-attention
        step(LN, &|e| self.ln(e, &l.norms[1], x, &ws.h, t));
        step(GEMM, &|e| {
            self.tgemm(e, ws, &l.qkv, &ws.h, &ws.qkv, t, 0, 0.0)
        });
        step(GEMM, &|e| {
            self.tgemm(e, ws, &l.pos, &ws.pos, &ws.relk, np, 0, 0.0)
        });
        step(ATTN, &|e| {
            bind(
                e,
                &self.gpu.qbias,
                &[
                    (&ws.qkv, 0),
                    (&l.bias_u, 0),
                    (&l.bias_v, 0),
                    (&ws.qu, 0),
                    (&ws.qv, 0),
                ],
                &du,
            );
            e.dispatchThreadgroups_threadsPerThreadgroup(
                size(d.div_ceil(256), t, 1),
                size(256, 1, 1),
            );
        });
        // S1[h] = (q + bias_u) · kᵀ
        step(ATTN, &|e| {
            self.bgemm(
                e,
                (&ws.qu, 0),
                (&ws.qkv, d),
                (&ws.s1, 0),
                BGemmP {
                    m: tu,
                    n: tu,
                    k: dhu,
                    lda: du,
                    ldb: 3 * du,
                    ldc: tu,
                    sa: dhu,
                    sb: dhu,
                    sc: tu * tu,
                    trans_b: 1,
                    alpha: 1.0,
                },
                h,
            )
        });
        // S2[h] = (q + bias_v) · relkᵀ  (all 2T-1 positions; the band is picked in the softmax)
        step(ATTN, &|e| {
            self.bgemm(
                e,
                (&ws.qv, 0),
                (&ws.relk, 0),
                (&ws.s2, 0),
                BGemmP {
                    m: tu,
                    n: npu,
                    k: dhu,
                    lda: du,
                    ldb: du,
                    ldc: npu,
                    sa: dhu,
                    sb: dhu,
                    sc: tu * npu,
                    trans_b: 1,
                    alpha: 1.0,
                },
                h,
            )
        });
        step(SOFT, &|e| {
            let sp = SoftP {
                t: tu,
                np: npu,
                scale: 1.0 / (dh as f32).sqrt(),
            };
            bind(
                e,
                &self.gpu.relsoftmax,
                &[(&ws.s1, 0), (&ws.s2, 0), (&ws.p, 0)],
                &sp,
            );
            e.dispatchThreadgroups_threadsPerThreadgroup(size(t, h, 1), size(256, 1, 1));
        });
        // ctx[:, h] = P[h] · v[h]
        step(ATTN, &|e| {
            self.bgemm(
                e,
                (&ws.p, 0),
                (&ws.qkv, 2 * d),
                (&ws.ctx, 0),
                BGemmP {
                    m: tu,
                    n: dhu,
                    k: tu,
                    lda: tu,
                    ldb: 3 * du,
                    ldc: du,
                    sa: tu * tu,
                    sb: dhu,
                    sc: dhu,
                    trans_b: 0,
                    alpha: 1.0,
                },
                h,
            )
        });
        step(GEMM, &|e| self.tgemm(e, ws, &l.o, &ws.ctx, x, t, 2, 1.0));
        // convolution module: x += pw2(silu(bn(dw(glu(pw1(LN(x)))))))
        step(LN, &|e| self.ln(e, &l.norms[2], x, &ws.h, t));
        step(GEMM, &|e| {
            self.tgemm(e, ws, &l.pw1, &ws.h, &ws.u, t, 0, 0.0)
        });
        step(CONV, &|e| {
            let cp = ConvP {
                t: tu,
                d: du,
                k: l.k as u32,
            };
            bind(
                e,
                &self.gpu.conv,
                &[(&ws.u, 0), (&l.dw, 0), (&l.dw_bias, 0), (&ws.a, 0)],
                &cp,
            );
            e.dispatchThreadgroups_threadsPerThreadgroup(
                size(d.div_ceil(256), t, 1),
                size(256, 1, 1),
            );
        });
        step(GEMM, &|e| self.tgemm(e, ws, &l.pw2, &ws.a, x, t, 2, 1.0));
        // x += ½ · FF2(LN(x)); then the output LayerNorm into the other buffer
        step(LN, &|e| self.ln(e, &l.norms[3], x, &ws.h, t));
        step(GEMM, &|e| {
            self.tgemm(e, ws, &l.ff2.0, &ws.h, &ws.u, t, 1, 0.0)
        });
        step(GEMM, &|e| self.tgemm(e, ws, &l.ff2.1, &ws.u, x, t, 2, 0.5));
        step(LN, &|e| self.ln(e, &l.norms[4], x, xn, t));
    }
}
