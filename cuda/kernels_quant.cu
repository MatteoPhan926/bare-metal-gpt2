// kernels_quant.cu — STAGE 4: weight-only INT8 matmul (the main DECODE lever, ROOFLINE director map).
//
// WHAT THIS IS (and, importantly, what it is NOT):
//   The pre-registered scheme (QUALITY_GATES §2) is symmetric, per-channel, **weight-only** INT8:
//   weights are int8, ACTIVATIONS STAY fp16. So the inner product is fp16 x int8 -> fp32 on CUDA
//   cores. It therefore CANNOT use dp4a or the INT8 tensor cores — those compute int8 x int8 -> int32
//   and would require quantizing the activations too (W8A8), which is a different, un-preregistered
//   scheme. The int->float convert is not a "slow fallback"; it is the scheme.
//
//   This matters for how the number is read: INT8 here buys **traffic**, not math. ROOFLINE §3:
//   "Quantization buys decode speed by moving fewer bytes (~2x per halving), *not* faster math."
//   The math ceiling stays the CUDA-core one, exactly as for the Stage 2/3 GEMMs: MEASURED (ROOFLINE
//   §6b) achievable CUDA-core GEMM = 10.10 TFLOP/s, fp32 FMA pipe = 15.64 TFLOP/s (bug line).
//   The DECODE ceiling halves (124 MB vs 248 MB -> ~1876 copy / ~2000 read tok/s).
//
// WHY THE SCALE LEAVES THE INNER LOOP (this is what "per-channel over rows" buys):
//   C[m,n] = sum_k A[m,k]*W[n,k] + b[n],  W[n,k] ~= q[n,k]*s[n]   with s indexed by the OUTPUT row n.
//   s[n] is constant across k, so:  C[m,n] ~= s[n] * (sum_k A[m,k]*q[n,k]) + b[n].
//   The k-loop accumulates a scale-free dot product; s[n] is applied ONCE per output element. Had we
//   quantized per-column (over k) the scale would sit inside the sum and cost a multiply per k.
//
//   Stage 3a's lesson applies directly: the tiled GEMM re-stages each W tile ceil(M/16) times, so any
//   per-staging work is paid ceil(M/16) times. Dequant here is ONE int8->float convert per staged
//   element (no multiply — the scale is hoisted), against a HALVED load width. That is the whole bet.
//
// The kernel is otherwise k_matmul_tiled byte-for-byte: same 16x16 tiling, same k order, same fp32
// accumulation, same bias-added-last. So a gate/timing delta localizes to the quantization.

#include "kernels.cuh"
#include "kvcache.cuh"     // STAGE 5: gpt2_gemv_fp16 / gpt2_gemv_int8 for the decode path
#include "common.cuh"
#include <cstdio>
#include <cstring>

#define TILE 16

// C[M,N] = A[M,K] · (q[N,K] * s[N])^T + bias[N].   A fp16, q int8 (row-major, row n contiguous in k).
__global__ void k_matmul_int8_tiled(half *C, const half *A, const int8_t *Wq, const float *Ws,
                                    const half *bias, int M, int N, int K) {
    __shared__ float As[TILE][TILE + 1];
    __shared__ float Bs[TILE][TILE + 1];
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;                 // m (output row)
    int col = blockIdx.x * TILE + tx;                 // n (output col)
    int nB  = blockIdx.x * TILE + ty;                 // W row this thread stages into Bs
    float acc = 0.f;                                  // scale-free: sum_k A[m,k] * q[n,k]
    for (int k0 = 0; k0 < K; k0 += TILE) {
        int k = k0 + tx;                              // coalesced: adjacent tx -> adjacent k
        As[ty][tx] = (row < M && k < K) ? __half2float(A[(size_t)row * K + k]) : 0.f;
        Bs[ty][tx] = (nB  < N && k < K) ? (float)Wq[(size_t)nB * K + k] : 0.f;   // int8 -> float, no scale
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < TILE; kk++) acc += As[ty][kk] * Bs[tx][kk];
        __syncthreads();
    }
    if (row < M && col < N) {                         // scale ONCE, then bias last (as in k_matmul_tiled)
        float v = acc * Ws[col];
        C[(size_t)row * N + col] = __float2half(bias ? v + __half2float(bias[col]) : v);
    }
}

static void int8_matmul(half *C, const half *A, const int8_t *Wq, const float *Ws, const half *bias,
                        int M, int N, int K) {
    dim3 blk(TILE, TILE);
    dim3 grd(CEIL_DIV(N, TILE), CEIL_DIV(M, TILE));
    k_matmul_int8_tiled<<<grd, blk>>>(C, A, Wq, Ws, bias, M, N, K);
}

// ---------------- packed-weight loading (mirrors tools/quantize.py's layout) ----------------

#define MAGIC_Q8 0x38515047      // 'GPQ8'

// Tensor order is FIXED and shared with quantize.py: wte, then per layer {qkv, attn_proj, fc, proj}.
struct QShape { int N, K; };
static const int N_QT = 1 + 4 * GPT2_N_LAYER;

static void q_shapes(QShape *sh) {
    sh[0] = {GPT2_VOCAB, GPT2_N_EMBD};                       // tied head
    for (int L = 0; L < GPT2_N_LAYER; L++) {
        sh[1 + 4*L + 0] = {GPT2_QKV_DIM, GPT2_N_EMBD};
        sh[1 + 4*L + 1] = {GPT2_N_EMBD,  GPT2_N_EMBD};
        sh[1 + 4*L + 2] = {GPT2_FFN_DIM, GPT2_N_EMBD};
        sh[1 + 4*L + 3] = {GPT2_N_EMBD,  GPT2_FFN_DIM};
    }
}

int gpt2_quant_load_upload(const char *path, GPT2WeightsGPU *gpu, GPT2QWeightsGPU *q) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[quant] cannot open %s (run tools/quantize.py first)\n", path); return 1; }
    int hdr[16];
    if (fread(hdr, sizeof(int), 16, f) != 16) { fprintf(stderr, "[quant] short header\n"); fclose(f); return 1; }
    if (hdr[0] != MAGIC_Q8) { fprintf(stderr, "[quant] bad magic 0x%08x\n", hdr[0]); fclose(f); return 1; }
    if (hdr[1] != 2)        { fprintf(stderr, "[quant] version %d != 2\n", hdr[1]); fclose(f); return 1; }
    if (hdr[2] != GPT2_N_LAYER || hdr[3] != GPT2_N_EMBD || hdr[4] != GPT2_VOCAB ||
        hdr[5] != GPT2_FFN_DIM || hdr[6] != GPT2_QKV_DIM || hdr[7] != N_QT) {
        fprintf(stderr, "[quant] dim mismatch vs config.h\n"); fclose(f); return 1;
    }

    // Per-tensor precision flags (the pre-registered kill-test): 0 = this tensor stays fp16, and no
    // int8/scale bytes exist for it. Its GPT2QW keeps q=NULL, so gpt2_matmul_dispatch routes that ONE
    // matmul back to the fp16 tiled GEMM. Nothing else changes.
    int flag[N_QT];
    if (fread(flag, sizeof(int), N_QT, f) != (size_t)N_QT) { fprintf(stderr,"[quant] short flags\n"); fclose(f); return 1; }

    QShape sh[N_QT]; q_shapes(sh);
    size_t nq = 0, ns = 0;
    for (int i = 0; i < N_QT; i++) if (flag[i]) { nq += (size_t)sh[i].N * sh[i].K; ns += sh[i].N; }

    std::vector<int8_t> hq(nq ? nq : 1); std::vector<float> hs(ns ? ns : 1);
    size_t oq = 0, os_ = 0;
    for (int i = 0; i < N_QT; i++) {                          // body: per QUANTIZED tensor, q[N*K] then s[N]
        if (!flag[i]) continue;
        size_t n = (size_t)sh[i].N * sh[i].K;
        if (fread(hq.data() + oq, 1, n, f) != n)                       { fprintf(stderr,"[quant] short q %d\n",i); fclose(f); return 1; }
        if (fread(hs.data() + os_, sizeof(float), sh[i].N, f) != (size_t)sh[i].N) { fprintf(stderr,"[quant] short s %d\n",i); fclose(f); return 1; }
        oq += n; os_ += sh[i].N;
    }
    long extra = 0; { long cur = ftell(f); fseek(f, 0, SEEK_END); extra = ftell(f) - cur; }
    fclose(f);
    if (extra != 0) { fprintf(stderr, "[quant] %ld trailing bytes -> layout mismatch\n", extra); return 1; }

    q->nq = nq; q->ns = ns;
    q->qdata = dmalloc<int8_t>(nq ? nq : 1); q->sdata = dmalloc<float>(ns ? ns : 1);
    if (nq) h2d(q->qdata, hq.data(), nq);
    if (ns) h2d(q->sdata, hs.data(), ns);

    GPT2QW *dst[N_QT];                                        // same fixed order
    dst[0] = &q->wte;
    for (int L = 0; L < GPT2_N_LAYER; L++) {
        dst[1 + 4*L + 0] = &q->layers[L].qkv_w;
        dst[1 + 4*L + 1] = &q->layers[L].attn_proj_w;
        dst[1 + 4*L + 2] = &q->layers[L].fc_w;
        dst[1 + 4*L + 3] = &q->layers[L].proj_w;
    }
    oq = 0; os_ = 0;
    size_t kept_bytes = 0;
    for (int i = 0; i < N_QT; i++) {
        if (!flag[i]) { dst[i]->q = nullptr; dst[i]->s = nullptr;   // -> fp16 fallback for this tensor
                        kept_bytes += 2 * (size_t)sh[i].N * sh[i].K; continue; }
        dst[i]->q = q->qdata + oq; dst[i]->s = q->sdata + os_;
        oq += (size_t)sh[i].N * sh[i].K; os_ += sh[i].N;
    }
    gpu->q = q;
    CUDA_CHECK(cudaGetLastError());
    int n_kept = 0; for (int i = 0; i < N_QT; i++) n_kept += !flag[i];
    // Streamed per decode step = int8 payload + per-channel scales (a quantized GEMV reads one float
    // per output row) + any kept-fp16 tensor. Kill-test build: 84.93 + 0.33 + 77.19 -> 162.46 MB.
    size_t streamed = nq + ns * sizeof(float) + kept_bytes;
    q->streamed_bytes = streamed;
    printf("[quant] %s: %d/%d tensors int8 (%.1f MB + %.2f MB scales) + %d kept fp16 (%.1f MB)\n",
           path, N_QT - n_kept, N_QT, nq / 1e6, ns * sizeof(float) / 1e6, n_kept, kept_bytes / 1e6);
    printf("[quant] streamed weight bytes = %.1f MB  -> decode ceilings [copy %.0f / read %.0f / theo %.0f tok/s]\n",
           streamed / 1e6, 233.4e3 / (streamed / 1e6), 248.9e3 / (streamed / 1e6), 256.0e3 / (streamed / 1e6));
    return 0;
}

void gpt2_quant_free(GPT2QWeightsGPU *q) {
    if (q && q->qdata) { cudaFree(q->qdata); q->qdata = nullptr; }
    if (q && q->sdata) { cudaFree(q->sdata); q->sdata = nullptr; }
}

// The DEFAULT is the kill-test build (tied head kept fp16) because that is the one that PASSES
// QUALITY_GATES §1 and §2. The pure all-49-tensor build (weights/gpt2_124m_int8.bin) is the
// pre-registered scheme and is reproducible via GPT2_INT8_WEIGHTS, but it FAILS gate (c)
// (max KL 0.257 >> 0.02) and the Δppl bound (+0.549 > +0.3) — see BENCHMARKS.md Stage 4. Defaulting
// to it would let an ungated artifact be benchmarked by accident.
#define GPT2_INT8_DEFAULT "weights/gpt2_124m_int8_kt.bin"

int gpt2_quant_attach_if_needed(const GPT2Backend *be, GPT2WeightsGPU *gpu, GPT2QWeightsGPU *q) {
    memset(q, 0, sizeof(*q));
    if (!be->matmul_q) return 0;                              // fp16 backend: nothing to attach
    const char *p = getenv("GPT2_INT8_WEIGHTS");
    return gpt2_quant_load_upload(p ? p : GPT2_INT8_DEFAULT, gpu, q);
}

// Stage 4 backend = the VALIDATED Stage-3b flash backend with ONLY the weight matmul swapped for the
// INT8 one. embed/LayerNorm/flash-attention/gelu/add and the whole forward are reused byte-for-byte,
// so every gate and timing delta localizes to the quantized GEMM.
//
// Composed from GPT2_BACKEND_NAIVE (constant-initialized) + the exported gpt2_matmul_tiled + the flash
// attention pointer, rather than by copying GPT2_BACKEND_FLASH: that object is dynamically initialized
// in another TU and cross-TU dynamic-init order is unspecified (the Stage-3 static-init trap).
//
// .matmul is kept pointing at the fp16 tiled GEMM on purpose: it is the fallback whenever the packed
// weights are absent (gpt2_matmul_dispatch checks). gpt2_quant_attach_if_needed() fails loudly if the
// int8 file is missing, so a run can never silently benchmark fp16 while claiming INT8.
// STAGE 5: the INT8 backend also gets the M=1 GEMVs. .gemv_q handles the 48 quantized matmuls; .gemv
// (fp16) is what a tensor kept fp16 by Stage 4's kill-test -- the tied head -- falls back to, because
// its GPT2QW.q is NULL. So "INT8 decode" honestly means "INT8 blocks + fp16 head", as gated.
static GPT2Backend make_int8_backend() {
    GPT2Backend b = GPT2_BACKEND_NAIVE;
    b.name      = "int8";
    b.matmul    = gpt2_matmul_tiled;        // prefill == Stage 2
    b.attention = gpt2_attention_flash;     // prefill == Stage 3b
    b.matmul_q  = int8_matmul;              // prefill == Stage 4
    b.gemv      = gpt2_gemv_fp16;           // decode, kept-fp16 tensors == Stage 5
    b.gemv_q    = gpt2_gemv_int8;           // decode, quantized tensors == Stage 5
    return b;
}
const GPT2Backend GPT2_BACKEND_INT8 = make_int8_backend();
