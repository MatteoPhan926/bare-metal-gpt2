// kernels_naive.cu — STAGE 1 naive fp16 kernels (the GPU baseline; correctness before speed).
//
// Every kernel: fp16 storage (matches the HF fp16 oracle's per-op rounding) + fp32 accumulation in
// the reductions (matmul dot, LayerNorm mean/var, softmax) — mirrors HF fp16 (cuBLAS fp32-accum,
// fp32 LN/softmax stats). NO shared-memory tiling, NO coalescing tricks: correctness + a baseline
// number. The director map says decode is BW-bound; naive uncoalesced reads waste BW, so this
// baseline is expected to sit WELL BELOW the ~941/1004 tok/s ceiling (above it would be a bug).
//
// gelu_new tanh; LayerNorm biased var, eps inside sqrt (1e-5); attn scale 1/sqrt(head_dim); causal.

#include "kernels.cuh"
#include "common.cuh"
#include <math.h>

// ---------------- kernels ----------------

__global__ void k_embed(half *x, const half *wte, const half *wpe, const int *ids, int T, int E) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= T * E) return;
    int t = idx / E, e = idx - t * E;
    float v = __half2float(wte[(size_t)ids[t] * E + e]) + __half2float(wpe[(size_t)t * E + e]);
    x[idx] = __float2half(v);
}

// one block per row; blockDim threads cooperatively reduce over E. fp32 stats -> fp16 out.
__global__ void k_layernorm(half *out, const half *x, const half *g, const half *b, int M, int E) {
    int m = blockIdx.x; if (m >= M) return;
    const half *xr = x + (size_t)m * E; half *o = out + (size_t)m * E;
    __shared__ float red[256];
    int tid = threadIdx.x, nt = blockDim.x;
    float s = 0.f; for (int e = tid; e < E; e += nt) s += __half2float(xr[e]);
    red[tid] = s; __syncthreads();
    for (int st = nt/2; st > 0; st >>= 1) { if (tid < st) red[tid] += red[tid+st]; __syncthreads(); }
    float mean = red[0] / E; __syncthreads();
    float v = 0.f; for (int e = tid; e < E; e += nt) { float d = __half2float(xr[e]) - mean; v += d*d; }
    red[tid] = v; __syncthreads();
    for (int st = nt/2; st > 0; st >>= 1) { if (tid < st) red[tid] += red[tid+st]; __syncthreads(); }
    float inv = 1.0f / sqrtf(red[0] / E + (float)GPT2_LN_EPS);
    for (int e = tid; e < E; e += nt)
        o[e] = __float2half((__half2float(xr[e]) - mean) * inv * __half2float(g[e]) + __half2float(b[e]));
}

// C[M,N] = A[M,K] · W[N,K]^T + bias[N].  one thread per output element (naive; uncoalesced W reads).
__global__ void k_matmul(half *C, const half *A, const half *W, const half *bias, int M, int N, int K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;                       // M*N <= 512*50257 < INT_MAX here
    int m = idx / N, n = idx - m * N;
    const half *a = A + (size_t)m * K;
    const half *w = W + (size_t)n * K;
    float acc = bias ? __half2float(bias[n]) : 0.f;
    for (int k = 0; k < K; k++) acc += __half2float(a[k]) * __half2float(w[k]);
    C[idx] = __float2half(acc);
}

// causal multi-head self-attention. one thread per (head, query). scores/softmax/context in fp32.
// qkv[T, 3E] laid out [q(E) | k(E) | v(E)]; att[T, E] concatenated head contexts.
__global__ void k_attention(half *att, const half *qkv, int T, int E, int H, int D) {
    int hi = blockIdx.x * blockDim.x + threadIdx.x;
    if (hi >= H * T) return;
    int h = hi / T, i = hi - h * T;                 // head h, query i
    const int QK = 3 * E;
    const half *qi = qkv + (size_t)i * QK + 0 * E + h * D;
    float scale = 1.0f / sqrtf((float)D);
    // pass 1: max score over keys j<=i
    float mx = -1e30f;
    for (int j = 0; j <= i; j++) {
        const half *kj = qkv + (size_t)j * QK + 1 * E + h * D;
        float d = 0.f; for (int x = 0; x < D; x++) d += __half2float(qi[x]) * __half2float(kj[x]);
        d *= scale; if (d > mx) mx = d;
    }
    // pass 2: softmax numerator + accumulate context (recompute d — naive, avoids storing scores)
    float sum = 0.f, acc[GPT2_HEAD_DIM];
    for (int x = 0; x < D; x++) acc[x] = 0.f;
    for (int j = 0; j <= i; j++) {
        const half *kj = qkv + (size_t)j * QK + 1 * E + h * D;
        float d = 0.f; for (int x = 0; x < D; x++) d += __half2float(qi[x]) * __half2float(kj[x]);
        float e = expf(d * scale - mx); sum += e;
        const half *vj = qkv + (size_t)j * QK + 2 * E + h * D;
        for (int x = 0; x < D; x++) acc[x] += e * __half2float(vj[x]);
    }
    float inv = 1.0f / sum; half *o = att + (size_t)i * E + h * D;
    for (int x = 0; x < D; x++) o[x] = __float2half(acc[x] * inv);
}

__global__ void k_gelu(half *x, size_t n) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float v = __half2float(x[idx]);
    float inner = 0.7978845608028654f * (v + 0.044715f * v * v * v);  // sqrt(2/pi)*(x+0.044715 x^3)
    x[idx] = __float2half(0.5f * v * (1.0f + tanhf(inner)));
}

__global__ void k_add(half *x, const half *y, size_t n) {           // residual x += y
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    x[idx] = __float2half(__half2float(x[idx]) + __half2float(y[idx]));
}

// ---------------- naive backend launchers ----------------
static void naive_embed(half *x, const half *wte, const half *wpe, const int *ids, int T, int E) {
    int nt = 256; k_embed<<<CEIL_DIV(T*E, nt), nt>>>(x, wte, wpe, ids, T, E);
}
static void naive_layernorm(half *out, const half *x, const half *g, const half *b, int M, int E) {
    k_layernorm<<<M, 256>>>(out, x, g, b, M, E);
}
static void naive_matmul(half *C, const half *A, const half *W, const half *bias, int M, int N, int K) {
    int nt = 256; long tot = (long)M * N; k_matmul<<<CEIL_DIV(tot, nt), nt>>>(C, A, W, bias, M, N, K);
}
static void naive_attention(half *att, const half *qkv, int T, int E, int H, int D) {
    int nt = 128; k_attention<<<CEIL_DIV(H*T, nt), nt>>>(att, qkv, T, E, H, D);
}
static void naive_gelu(half *x, size_t n) { int nt = 256; k_gelu<<<CEIL_DIV(n, (size_t)nt), nt>>>(x, n); }
static void naive_add (half *x, const half *y, size_t n) { int nt = 256; k_add<<<CEIL_DIV(n, (size_t)nt), nt>>>(x, y, n); }

const GPT2Backend GPT2_BACKEND_NAIVE = {
    "naive", naive_embed, naive_layernorm, naive_matmul, naive_attention, naive_gelu, naive_add
};
