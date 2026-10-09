// Metal compute kernels for the FastConformer encoder (feature `metal`). Activations are f32; ternary weights stay
// 2-bit on the GPU (code = trit + 1, four per byte, row-major [N, K/4]) with f32 scales per (row, group).
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// ---------------------------------------------------------------------------------------------------------------
// Ternary GEMM: Y[m, n] (=|+=) X[m, :] · W[n, :]ᵀ with f32 accumulators in simdgroup 8×8 tiles; W is decoded
// (trit · scale) into threadgroup memory. Split-K: the grid's z dimension partitions K in `kchunk` slices; with more
// than one slice each writes its raw partial sums to P[z] and `tgemm_reduce` applies the epilogue.
struct TGemmP {
    uint M, N, K, G;  // rows, outputs, inputs, group size (multiple of BK)
    uint mode;        // 0: store, 1: store SiLU, 2: y += alpha · v
    float alpha;
    uint kchunk;      // K per z slice (multiple of BK)
};

static inline void tg_store(device float* y, float v, uint mode, float alpha) {
    if (mode == 1) *y = v / (1.0f + exp(-v));
    else if (mode == 2) *y = fma(alpha, v, *y);
    else *y = v;
}

// Operand pipeline: vectorised loads, conflict-free decode. X tile loaded as float4; each of 128 threads decodes
// a 32-bit word (16 trits) of one weight row (thread n = tid % 64 owns row n, (tid / 64) picks the 16-wide k half),
// writing along n so a simdgroup's stores hit consecutive banks. BM ∈ {16, 32}, BN = 64, BK ∈ {16, 32}, 4 simdgroups.
template <typename T, uint BM, uint BK>
kernel void tgemm_t(device const float* X [[buffer(0)]],
                     device const uchar* W [[buffer(1)]],
                     device const float* S [[buffer(2)]],
                     device float* Y [[buffer(3)]],
                     constant TGemmP& p [[buffer(4)]],
                     device float* P [[buffer(5)]],
                     uint3 tg [[threadgroup_position_in_grid]],
                     uint3 ntg [[threadgroups_per_grid]],
                     uint tid [[thread_index_in_threadgroup]],
                     uint sg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint BN = 64, FR = BM / 8, FC = 2;
    constexpr uint OPS = (BM * BK + BK * BN) * sizeof(T), OUT = BM * BN * sizeof(float);
    threadgroup uchar smem[OPS > OUT ? OPS : OUT];
    threadgroup T* Xs = (threadgroup T*)smem;              // [m][k]
    threadgroup T* Ws = (threadgroup T*)smem + BM * BK;    // [k][n]
    threadgroup float* Cs = (threadgroup float*)smem;
    const uint m0 = tg.y * BM, n0 = tg.x * BN;
    const uint kq = p.K / 4, ng = p.K / p.G;
    const uint kb = tg.z * p.kchunk, ke = min(p.K, kb + p.kchunk);
    const uint scol = sg * 16;
    const uint wn = tid % 64, wh = tid / 64;  // decode: row, 16-wide k half
    device const uchar* wrow = W + (n0 + wn) * kq;
    device const float* srow = S + (n0 + wn) * ng;
    simdgroup_float8x8 acc[FR][FC];
    for (uint i = 0; i < FR; i++)
        for (uint j = 0; j < FC; j++) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint k0 = kb; k0 < ke; k0 += BK) {
        for (uint e = tid; e < BM * BK / 4; e += 128) {
            uint r = (e * 4) / BK, c = (e * 4) % BK, gm = m0 + r;
            float4 v = gm < p.M ? *(device const float4*)(X + gm * p.K + k0 + c) : float4(0.0f);
            threadgroup T* d = Xs + r * BK + c;
            d[0] = T(v.x); d[1] = T(v.y); d[2] = T(v.z); d[3] = T(v.w);
        }
        if (wh < BK / 16) {
            uint q = *(device const uint*)(wrow + k0 / 4 + wh * 4);
            T sc = T(srow[k0 / p.G]);
            for (uint j = 0; j < 16; j++) Ws[(wh * 16 + j) * BN + wn] = (T((q >> (2 * j)) & 3u) - T(1)) * sc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < BK; kk += 8) {
            simdgroup_matrix<T, 8, 8> a[FR], b[FC];
            for (uint i = 0; i < FR; i++) simdgroup_load(a[i], Xs + i * 8 * BK + kk, BK);
            for (uint j = 0; j < FC; j++) simdgroup_load(b[j], Ws + kk * BN + scol + j * 8, BN);
            for (uint i = 0; i < FR; i++)
                for (uint j = 0; j < FC; j++) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint i = 0; i < FR; i++)
        for (uint j = 0; j < FC; j++) simdgroup_store(acc[i][j], Cs + i * 8 * BN + scol + j * 8, BN);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const bool split = ntg.z > 1;
    for (uint e = tid; e < BM * BN; e += 128) {
        uint r = e / BN, c = e % BN, gm = m0 + r;
        if (gm >= p.M) continue;
        if (split) P[(tg.z * p.M + gm) * p.N + n0 + c] = Cs[e];
        else tg_store(Y + gm * p.N + n0 + c, Cs[e], p.mode, p.alpha);
    }
}
// Tiles measured with tests/metal_bench.rs (16 rows up to 64 tokens, 32 above; K step 16). Lost: K step 32 or 8,
// half staging, half partial sums, 64 × 64 / 32 × 128 tiles with 4 or 8 simdgroups, byte-wise strided decode.
#define TGEMM(NAME, T, BM, BK)                                                                               \
    template [[host_name(NAME)]] kernel void tgemm_t<T, BM, BK>(                                             \
        device const float*, device const uchar*, device const float*, device float*, constant TGemmP&,     \
        device float*, uint3, uint3, uint, uint);
TGEMM("tgemm_16", float, 16, 16)
TGEMM("tgemm_32", float, 32, 16)

// Split-K reduction: Y[m, n] = epilogue(Σ_z P[z, m, n]).
kernel void tgemm_reduce(device const float* P [[buffer(0)]],
                         device float* Y [[buffer(1)]],
                         constant TGemmP& p [[buffer(2)]],
                         constant uint& splits [[buffer(3)]],
                         uint2 gid [[thread_position_in_grid]]) {
    uint n = gid.x, m = gid.y;
    if (n >= p.N || m >= p.M) return;
    float v = 0.0f;
    for (uint z = 0; z < splits; z++) v += P[(z * p.M + m) * p.N + n];
    tg_store(Y + m * p.N + n, v, p.mode, p.alpha);
}

// ---------------------------------------------------------------------------------------------------------------
// Batched f32 GEMM for attention: C[b] = alpha · A[b] · op(B[b]); op(B) = Bᵀ when transB (B stored [n][k]), else B
// ([k][n]). Arbitrary M, N, K (zero-padded tiles). Tile 32 × 32 × 16, four simdgroups of 16 × 16.
struct BGemmP {
    uint M, N, K, lda, ldb, ldc, sa, sb, sc, transB;
    float alpha;
};

constant constexpr uint BM = 32, BN = 32, BK = 16;

kernel void bgemm(device const float* A [[buffer(0)]],
                  device const float* B [[buffer(1)]],
                  device float* C [[buffer(2)]],
                  constant BGemmP& p [[buffer(3)]],
                  uint3 tg [[threadgroup_position_in_grid]],
                  uint tid [[thread_index_in_threadgroup]],
                  uint sg [[simdgroup_index_in_threadgroup]]) {
    // operand tiles, then the output tile in the same memory (occupancy is bounded by threadgroup memory)
    threadgroup float smem[(BM * BK + BK * BN) > BM * BN ? (BM * BK + BK * BN) : BM * BN];
    threadgroup float* As = smem;
    threadgroup float* Bs = smem + BM * BK;  // [k][n]
    threadgroup float* Cs = smem;
    const uint m0 = tg.y * BM, n0 = tg.x * BN, bt = tg.z;
    A += bt * p.sa;
    B += bt * p.sb;
    C += bt * p.sc;
    const uint sr = (sg / 2) * 16, scol = (sg % 2) * 16;
    simdgroup_float8x8 acc[2][2];
    for (uint i = 0; i < 2; i++)
        for (uint j = 0; j < 2; j++) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint k0 = 0; k0 < p.K; k0 += BK) {
        for (uint e = tid; e < BM * BK; e += 128) {
            uint r = e / BK, c = e % BK, gm = m0 + r, gk = k0 + c;
            As[e] = (gm < p.M && gk < p.K) ? A[gm * p.lda + gk] : 0.0f;
        }
        for (uint e = tid; e < BK * BN; e += 128) {
            uint kk, n;
            if (p.transB) { n = e / BK; kk = e % BK; }  // coalesced along k for [n][k] storage
            else { kk = e / BN; n = e % BN; }
            uint gk = k0 + kk, gn = n0 + n;
            float v = 0.0f;
            if (gk < p.K && gn < p.N) v = p.transB ? B[gn * p.ldb + gk] : B[gk * p.ldb + gn];
            Bs[kk * BN + n] = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < BK; kk += 8) {
            simdgroup_float8x8 a[2], b[2];
            for (uint i = 0; i < 2; i++) simdgroup_load(a[i], As + (sr + i * 8) * BK + kk, BK);
            for (uint j = 0; j < 2; j++) simdgroup_load(b[j], Bs + kk * BN + scol + j * 8, BN);
            for (uint i = 0; i < 2; i++)
                for (uint j = 0; j < 2; j++) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint i = 0; i < 2; i++)
        for (uint j = 0; j < 2; j++) simdgroup_store(acc[i][j], Cs + (sr + i * 8) * BN + scol + j * 8, BN);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = tid; e < BM * BN; e += 128) {
        uint r = e / BN, c = e % BN, gm = m0 + r, gn = n0 + c;
        if (gm < p.M && gn < p.N) C[gm * p.ldc + gn] = p.alpha * Cs[e];
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Row reductions over a threadgroup of 256 threads (8 simdgroups of 32).
static float tg_sum(float v, threadgroup float* sh, uint tid, uint lane, uint sg) {
    v = simd_sum(v);
    if (lane == 0) sh[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float t = tid < 8 ? sh[tid] : 0.0f;
    t = simd_sum(t);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) sh[0] = t;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float r = sh[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return r;
}

static float tg_max(float v, threadgroup float* sh, uint tid, uint lane, uint sg) {
    v = simd_max(v);
    if (lane == 0) sh[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float t = tid < 8 ? sh[tid] : -INFINITY;
    t = simd_max(t);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) sh[0] = t;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float r = sh[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return r;
}

// LayerNorm (eps 1e-5, biased variance), one threadgroup of 256 per row.
kernel void layernorm(device const float* X [[buffer(0)]],
                      device const float* g [[buffer(1)]],
                      device const float* b [[buffer(2)]],
                      device float* Y [[buffer(3)]],
                      constant uint& D [[buffer(4)]],
                      uint row [[threadgroup_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float sh[8];
    device const float* x = X + row * D;
    float s = 0.0f;
    for (uint c = tid; c < D; c += 256) s += x[c];
    float mean = tg_sum(s, sh, tid, lane, sg) / float(D);
    float q = 0.0f;
    for (uint c = tid; c < D; c += 256) { float d = x[c] - mean; q += d * d; }
    float inv = rsqrt(tg_sum(q, sh, tid, lane, sg) / float(D) + 1e-5f);
    device float* y = Y + row * D;
    for (uint c = tid; c < D; c += 256) y[c] = (x[c] - mean) * inv * g[c] + b[c];
}

// q + bias_u and q + bias_v from the fused qkv rows: qu, qv [T, D].
kernel void qbias(device const float* QKV [[buffer(0)]],
                  device const float* bu [[buffer(1)]],
                  device const float* bv [[buffer(2)]],
                  device float* QU [[buffer(3)]],
                  device float* QV [[buffer(4)]],
                  constant uint& D [[buffer(5)]],
                  uint2 gid [[thread_position_in_grid]]) {
    uint c = gid.x, i = gid.y;
    if (c >= D) return;
    float q = QKV[i * 3 * D + c];
    QU[i * D + c] = q + bu[c];
    QV[i * D + c] = q + bv[c];
}

// Relative-position attention scores → probabilities, per (head h, query i):
// s[j] = (S1[h,i,j] + S2[h,i,T-1-i+j]) · scale; P[h,i,:] = softmax(s).
struct SoftP {
    uint T, NP;
    float scale;
};

kernel void relsoftmax(device const float* S1 [[buffer(0)]],
                       device const float* S2 [[buffer(1)]],
                       device float* P [[buffer(2)]],
                       constant SoftP& p [[buffer(3)]],
                       uint2 tg [[threadgroup_position_in_grid]],
                       uint tid [[thread_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]],
                       uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float sh[8];
    const uint i = tg.x, h = tg.y, T = p.T;
    device const float* s1 = S1 + (h * T + i) * T;
    device const float* s2 = S2 + (h * T + i) * p.NP + (T - 1 - i);
    device float* pr = P + (h * T + i) * T;
    float mx = -INFINITY;
    for (uint j = tid; j < T; j += 256) {
        float v = (s1[j] + s2[j]) * p.scale;
        pr[j] = v;
        mx = max(mx, v);
    }
    mx = tg_max(mx, sh, tid, lane, sg);
    float sum = 0.0f;
    for (uint j = tid; j < T; j += 256) {
        float e = exp(pr[j] - mx);
        pr[j] = e;
        sum += e;
    }
    float inv = 1.0f / tg_sum(sum, sh, tid, lane, sg);
    for (uint j = tid; j < T; j += 256) pr[j] *= inv;
}

// Convolution module middle: GLU over channels → depthwise conv (k taps, BatchNorm folded) → SiLU.
// G [T, 2D] (pointwise_conv1 output), w tap-major [k, D], bias [D] → Out [T, D].
struct ConvP {
    uint T, D, K;
};

kernel void glu_dwconv_silu(device const float* G [[buffer(0)]],
                            device const float* w [[buffer(1)]],
                            device const float* bias [[buffer(2)]],
                            device float* Out [[buffer(3)]],
                            constant ConvP& p [[buffer(4)]],
                            uint2 gid [[thread_position_in_grid]]) {
    uint c = gid.x, i = gid.y;
    if (c >= p.D || i >= p.T) return;
    int pad = int(p.K - 1) / 2;
    float acc = bias[c];
    for (uint kk = 0; kk < p.K; kk++) {
        int j = int(i) + int(kk) - pad;
        if (j < 0 || j >= int(p.T)) continue;
        float a = G[uint(j) * 2 * p.D + c];
        float gate = G[uint(j) * 2 * p.D + p.D + c];
        acc = fma(w[kk * p.D + c], a / (1.0f + exp(-gate)), acc);
    }
    Out[i * p.D + c] = acc / (1.0f + exp(-acc));
}

// ---------------------------------------------------------------------------------------------------------------
// Subsampling (dw-striding Conv2d, 8× in time), channels-last [T, F, C] as on the CPU.
struct SubP {
    uint t_in, f_in;    // input grid of this stage
    uint t_out, f_out;  // output grid
    uint C, K, S, pad;
};

// conv0 (1 → C, k×k, stride S) + ReLU fused into the first depthwise conv (k×k, stride S): the conv0 outputs a
// depthwise tap needs are recomputed on the fly, so the large conv0 activation is never stored.
// feats [t0, f0] → Y [t_out, f_out, C]; `p.t_in, p.f_in` describe the conv0 output grid (t1, f1).
kernel void sub_conv0_dw(device const float* feats [[buffer(0)]],
                         device const float* w0 [[buffer(1)]],    // [k*k, C]
                         device const float* b0 [[buffer(2)]],    // [C]
                         device const float* wd [[buffer(3)]],    // [k*k, C]
                         device const float* bd [[buffer(4)]],    // [C]
                         device float* Y [[buffer(5)]],
                         constant SubP& p [[buffer(6)]],
                         constant uint2& in0 [[buffer(7)]],       // (t0, f0): the feature grid
                         uint3 gid [[thread_position_in_grid]]) {
    uint ch = gid.x, of = gid.y, ot = gid.z;
    if (ch >= p.C || of >= p.f_out || ot >= p.t_out) return;
    const int K = int(p.K), S = int(p.S), pad = int(p.pad);
    float acc = bd[ch];
    for (int kt = 0; kt < K; kt++) {
        int it = int(ot) * S + kt - pad;  // conv0 output row
        if (it < 0 || it >= int(p.t_in)) continue;
        for (int kf = 0; kf < K; kf++) {
            int jf = int(of) * S + kf - pad;  // conv0 output column
            if (jf < 0 || jf >= int(p.f_in)) continue;
            float v = b0[ch];
            for (int ut = 0; ut < K; ut++) {
                int xt = it * S + ut - pad;
                if (xt < 0 || xt >= int(in0.x)) continue;
                for (int uf = 0; uf < K; uf++) {
                    int xf = jf * S + uf - pad;
                    if (xf < 0 || xf >= int(in0.y)) continue;
                    v = fma(w0[(ut * K + uf) * p.C + ch], feats[uint(xt) * in0.y + uint(xf)], v);
                }
            }
            acc = fma(wd[(kt * K + kf) * p.C + ch], max(v, 0.0f), acc);
        }
    }
    Y[(ot * p.f_out + of) * p.C + ch] = acc;
}

// Depthwise k×k stride-S conv (no activation): X [t_in, f_in, C] → Y [t_out, f_out, C].
kernel void sub_dw(device const float* X [[buffer(0)]],
                   device const float* w [[buffer(1)]],
                   device const float* b [[buffer(2)]],
                   device float* Y [[buffer(3)]],
                   constant SubP& p [[buffer(4)]],
                   uint3 gid [[thread_position_in_grid]]) {
    uint ch = gid.x, of = gid.y, ot = gid.z;
    if (ch >= p.C || of >= p.f_out || ot >= p.t_out) return;
    const int K = int(p.K), S = int(p.S), pad = int(p.pad);
    float acc = b[ch];
    for (int kt = 0; kt < K; kt++) {
        int it = int(ot) * S + kt - pad;
        if (it < 0 || it >= int(p.t_in)) continue;
        for (int kf = 0; kf < K; kf++) {
            int i_f = int(of) * S + kf - pad;
            if (i_f < 0 || i_f >= int(p.f_in)) continue;
            acc = fma(w[(kt * K + kf) * p.C + ch], X[(uint(it) * p.f_in + uint(i_f)) * p.C + ch], acc);
        }
    }
    Y[(ot * p.f_out + of) * p.C + ch] = acc;
}

// Y[r, c] = act(Y[r, c] + b[c]) over [rows, N]; act = ReLU when relu != 0.
struct BiasP {
    uint rows, N, relu;
};
kernel void bias_act(device float* Y [[buffer(0)]],
                     device const float* b [[buffer(1)]],
                     constant BiasP& p [[buffer(2)]],
                     uint2 gid [[thread_position_in_grid]]) {
    uint c = gid.x, r = gid.y;
    if (c >= p.N || r >= p.rows) return;
    float v = Y[r * p.N + c] + b[c];
    Y[r * p.N + c] = p.relu ? max(v, 0.0f) : v;
}
