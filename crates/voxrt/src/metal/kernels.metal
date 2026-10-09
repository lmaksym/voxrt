// Metal compute kernels for the FastConformer encoder (feature `metal`). Activations are f32; ternary weights stay
// 2-bit on the GPU (code = trit + 1, four per byte, row-major [N, K/4]) with f32 scales per (row, group).
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// ---------------------------------------------------------------------------------------------------------------
// Ternary GEMM: Y[m, n] (=|+=) X[m, :] · W[n, :]ᵀ, as a template over the staging type T (float or half; the
// accumulators are always f32), the tile BM × BN and the k step BK. Four simdgroups split the BN columns; each owns
// BM/8 × BN/32 simdgroup 8×8 accumulators. W is decoded (trit · scale) into threadgroup memory.
// Split-K: the grid's z dimension partitions K in `kchunk` slices; with more than one slice each writes its raw
// partial sums to P[z] and `tgemm_reduce` applies the epilogue.
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

template <typename T, uint BM, uint BN, uint BK, uint SGR, uint SGC>
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
    // SGR × SGC simdgroups (32 · SGR · SGC threads); simdgroup (sg / SGC, sg % SGC) owns a (BM/SGR) × (BN/SGC) block
    constexpr uint FR = BM / (8 * SGR), FC = BN / (8 * SGC), NT = 32 * SGR * SGC;
    // operand tiles during the k loop, the f32 output tile afterwards: one aliased allocation (occupancy is bounded
    // by threadgroup memory)
    constexpr uint OPS = (BM * BK + BK * BN) * sizeof(T), OUT = BM * BN * sizeof(float);
    threadgroup uchar smem[OPS > OUT ? OPS : OUT];
    threadgroup T* Xs = (threadgroup T*)smem;              // [m][k]
    threadgroup T* Ws = (threadgroup T*)smem + BM * BK;    // [k][n]
    threadgroup float* Cs = (threadgroup float*)smem;
    const uint m0 = tg.y * BM, n0 = tg.x * BN;
    const uint kq = p.K / 4, ng = p.K / p.G;
    const uint kb = tg.z * p.kchunk, ke = min(p.K, kb + p.kchunk);
    const uint sr = (sg / SGC) * (BM / SGR), scol = (sg % SGC) * (BN / SGC);
    simdgroup_float8x8 acc[FR][FC];
    for (uint i = 0; i < FR; i++)
        for (uint j = 0; j < FC; j++) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint k0 = kb; k0 < ke; k0 += BK) {
        for (uint e = tid; e < BM * BK; e += NT) {
            uint r = e / BK, c = e % BK, gm = m0 + r;
            Xs[e] = T(gm < p.M ? X[gm * p.K + k0 + c] : 0.0f);
        }
        for (uint e = tid; e < BN * (BK / 4); e += NT) {
            uint n = e / (BK / 4), b = e % (BK / 4);
            uchar q = W[(n0 + n) * kq + k0 / 4 + b];
            T sc = T(S[(n0 + n) * ng + k0 / p.G]);
            for (uint j = 0; j < 4; j++) Ws[(b * 4 + j) * BN + n] = (T((q >> (2 * j)) & 3) - T(1)) * sc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < BK; kk += 8) {
            simdgroup_matrix<T, 8, 8> a[FR], b[FC];
            for (uint i = 0; i < FR; i++) simdgroup_load(a[i], Xs + (sr + i * 8) * BK + kk, BK);
            for (uint j = 0; j < FC; j++) simdgroup_load(b[j], Ws + kk * BN + scol + j * 8, BN);
            for (uint i = 0; i < FR; i++)
                for (uint j = 0; j < FC; j++) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint i = 0; i < FR; i++)
        for (uint j = 0; j < FC; j++) simdgroup_store(acc[i][j], Cs + (sr + i * 8) * BN + scol + j * 8, BN);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const bool split = ntg.z > 1;
    for (uint e = tid; e < BM * BN; e += NT) {
        uint r = e / BN, c = e % BN, gm = m0 + r;
        if (gm >= p.M) continue;
        if (split) P[(tg.z * p.M + gm) * p.N + n0 + c] = Cs[e];
        else tg_store(Y + gm * p.N + n0 + c, Cs[e], p.mode, p.alpha);
    }
}

#define TGEMM(NAME, T, BM, BN, BK, SGR, SGC)                                                              \
    template [[host_name(NAME)]] kernel void tgemm_t<T, BM, BN, BK, SGR, SGC>(                              \
        device const float*, device const uchar*, device const float*, device float*, constant TGemmP&,     \
        device float*, uint3, uint3, uint, uint);
// The two tiles the encoder uses (16 × 64 up to 64 tokens, 32 × 64 above). Measured alternatives that lost:
// K step 32 or 8, half staging, 64 × 64 and 32 × 128 tiles with 4 or 8 simdgroups (tests/metal_bench.rs).
TGEMM("tgemm_f_16_64_16", float, 16, 64, 16, 1, 4)
TGEMM("tgemm_f_32_64_16", float, 32, 64, 16, 1, 4)

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
