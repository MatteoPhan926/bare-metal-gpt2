// kvcache.cu — STAGE 5: KV cache, memory planner, true M=1 GEMV, and the decode step.
//
// See kvcache.cuh for why the GEMV lives here and why a KV cache alone cannot surface the INT8 win.
//
// Numerics: every kernel is fp16 storage + fp32 accumulation, the SAME regime as Stages 1-4, so the
// decode path is comparable to the fp16 oracle at the same 1e-2/KL-0.02 gates (QUALITY_GATES §1 + A1).
//
// NOTE on exactness: cached decode can NOT be bit-identical to the no-KV recompute path, and it is a
// bug to claim otherwise. In the recompute path a position's K/V are produced by a GEMM at M=ctx; in
// the cached path they were produced at M=1 (GEMV) when that position was decoded. Different reduction
// orders -> different fp16 rounding. The gate is therefore "equal within fp16 noise" (max|dlogit| and
// KL), which is what the correctness harness checks, plus the per-layer localizer.

#include "kvcache.cuh"
#include "common.cuh"
#include <math.h>
#include <cstdio>
#include <cstring>
#include <math_constants.h>

#define E_  GPT2_N_EMBD
#define H_  GPT2_N_HEAD
#define D_  GPT2_HEAD_DIM

// ============================ memory planner ============================
//
// One contiguous arena for all layers' K and V. Sized for the full context so no reallocation can
// happen mid-decode (a realloc would move the cache under a running kernel and is a classic silent bug).
int gpt2_kv_alloc(GPT2KVCache *kv, int maxT) {
    kv->maxT = maxT; kv->nLayer = GPT2_N_LAYER; kv->nHead = H_; kv->headDim = D_; kv->len = 0;
    kv->bytes = (size_t)2 * GPT2_N_LAYER * H_ * maxT * D_ * sizeof(half);
    size_t freeB=0, totB=0; CUDA_CHECK(cudaMemGetInfo(&freeB, &totB));
    if (kv->bytes + (64u<<20) > freeB) {                 // keep 64 MiB of headroom for scratch/logits
        fprintf(stderr, "[kv] plan needs %.1f MB but only %.1f MB free -> refusing to allocate\n",
                kv->bytes/1e6, freeB/1e6);
        return 1;
    }
    kv->data = dmalloc<half>(kv->bytes / sizeof(half));
    printf("[kv] plan: %d layers x 2 (K,V) x %d heads x %d pos x %d dim x fp16 = %.1f MB\n",
           GPT2_N_LAYER, H_, maxT, D_, kv->bytes/1e6);
    printf("[kv] arena = ONE contiguous alloc; VRAM free %.0f/%.0f MB -> headroom %.0f MB after cache\n",
           freeB/1e6, totB/1e6, (freeB - kv->bytes)/1e6);
    printf("[kv] decode KV traffic = %.1f KB per token of context (2*L*H*D*2B); at ctx=512 -> %.1f MB/step\n",
           2.0*GPT2_N_LAYER*H_*D_*sizeof(half)/1e3, gpt2_kv_step_bytes(kv, 512)/1e6);
    return 0;
}
void gpt2_kv_free(GPT2KVCache *kv){ if(kv && kv->data){ cudaFree(kv->data); kv->data=nullptr; } }
void gpt2_kv_reset(GPT2KVCache *kv){ if(kv) kv->len = 0; }

double gpt2_kv_step_bytes(const GPT2KVCache *kv, int len) {
    return 2.0 * kv->nLayer * kv->nHead * (double)len * kv->headDim * sizeof(half);
}

// ============================ cache write kernels ============================

// Decode append: one position, all heads. qkv is [3E] = [q|k|v]. Strided across heads by maxT*D.
__global__ void k_kv_append(half *K, half *V, const half *qkv, int pos, int maxT) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;         // 0..E-1
    if (i >= E_) return;
    int h = i / D_, d = i - h*D_;
    size_t off = (size_t)h*maxT*D_ + (size_t)pos*D_ + d;
    K[off] = qkv[1*E_ + i];
    V[off] = qkv[2*E_ + i];
}

// Prefill scatter: T positions at once. qkv is [T, 3E].
__global__ void k_kv_scatter(half *K, half *V, const half *qkv, int T, int maxT) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;       // 0..T*E-1
    if (idx >= T*E_) return;
    int t = idx / E_, i = idx - t*E_;
    int h = i / D_, d = i - h*D_;
    size_t off = (size_t)h*maxT*D_ + (size_t)t*D_ + d;
    const half *row = qkv + (size_t)t*3*E_;
    K[off] = row[1*E_ + i];
    V[off] = row[2*E_ + i];
}

void gpt2_kv_scatter_prefill(GPT2KVCache *kv, int L, const half *qkv, int T) {
    int nt = 256;
    k_kv_scatter<<<CEIL_DIV(T*E_, nt), nt>>>(gpt2_kv_K(kv,L), gpt2_kv_V(kv,L), qkv, T, kv->maxT);
}

// ============================ M=1 attention over the cache ============================
//
// One block per head, 64 threads (= D). Reads the contiguous slab K[h][0..len) -- the reason for the
// head-major layout. Scores live in dynamic shared memory (len floats, <= 4 KB at ctx 1024); they are
// never written to global memory (the flash lesson, at M=1 there is only ONE query row anyway).
__global__ void k_attn_decode(half *att, const half *q, const half *K, const half *V,
                              int len, int maxT) {
    const int h = blockIdx.x, tid = threadIdx.x, nt = blockDim.x;
    extern __shared__ float sh[];
    float *qs = sh;                 // [D_]
    float *sc = sh + D_;            // [len]

    for (int d = tid; d < D_; d += nt) qs[d] = __half2float(q[h*D_ + d]);
    __syncthreads();

    const half *Kh = K + (size_t)h*maxT*D_;
    const half *Vh = V + (size_t)h*maxT*D_;
    const float scale = 1.0f / sqrtf((float)D_);          // identical expression to the other kernels

    __shared__ float red[64];
    // pass 1: scores + max
    float m = -CUDART_INF_F;
    for (int j = tid; j < len; j += nt) {
        const half *kj = Kh + (size_t)j*D_;
        float dot = 0.f;
        #pragma unroll
        for (int d = 0; d < D_; d++) dot += qs[d] * __half2float(kj[d]);
        float s = dot * scale; sc[j] = s; m = fmaxf(m, s);
    }
    red[tid] = m; __syncthreads();
    for (int st = nt/2; st; st >>= 1){ if(tid<st) red[tid]=fmaxf(red[tid],red[tid+st]); __syncthreads(); }
    m = red[0]; __syncthreads();

    // pass 2: exp + sum
    float l = 0.f;
    for (int j = tid; j < len; j += nt) { float p = __expf(sc[j] - m); sc[j] = p; l += p; }
    red[tid] = l; __syncthreads();
    for (int st = nt/2; st; st >>= 1){ if(tid<st) red[tid]+=red[tid+st]; __syncthreads(); }
    const float inv = 1.0f / red[0]; __syncthreads();

    // pass 3: context. thread d owns output dim d; V reads are 64 contiguous halves per j.
    for (int d = tid; d < D_; d += nt) {
        float a = 0.f;
        for (int j = 0; j < len; j++) a += sc[j] * __half2float(Vh[(size_t)j*D_ + d]);
        att[h*D_ + d] = __float2half(a * inv);
    }
}

// ============================ embed for one token at `pos` ============================
__global__ void k_embed_one(half *x, const half *wte, const half *wpe, int token, int pos) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= E_) return;
    x[i] = __float2half(__half2float(wte[(size_t)token*E_ + i]) + __half2float(wpe[(size_t)pos*E_ + i]));
}

// ============================ the TRUE M=1 GEMV ============================
//
// One WARP per output row n. The warp's 32 lanes stride the K-dim with VECTORISED loads, so row n of W
// is read exactly once, coalesced (32 lanes x 4 B = 128 B per instruction), and NOTHING is discarded.
// A[1,K] is staged once in shared memory and reused by all warps in the block.
//
// This is the whole point of Stage 5's speed story: the tiled GEMM at M=1 executes 16x the necessary
// FLOPs (it computes a full 16-row tile and masks 15 rows at the write). That made it compute-bound,
// which is why Stage 4's INT8 byte-halving bought exactly 1.00x. Here the kernel is weight-traffic
// bound by construction, so halving the bytes can finally show up.
#define GV_WARPS 4

__global__ void k_gemv_fp16(half *C, const half *A, const half *W, const half *bias, int N, int K) {
    extern __shared__ float As[];
    const int tid = threadIdx.x, nt = blockDim.x;
    for (int k = tid; k < K; k += nt) As[k] = __half2float(A[k]);
    __syncthreads();

    const int warp = tid >> 5, lane = tid & 31;
    const int n = blockIdx.x*GV_WARPS + warp;
    if (n >= N) return;

    const __half2 *w2 = (const __half2 *)(W + (size_t)n*K);   // K even (768/2304/3072/50257-row K=768)
    const int K2 = K >> 1;
    float acc = 0.f;
    for (int k2 = lane; k2 < K2; k2 += 32) {
        float2 wf = __half22float2(w2[k2]);
        acc += As[2*k2] * wf.x + As[2*k2 + 1] * wf.y;
    }
    #pragma unroll
    for (int off = 16; off; off >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, off);
    if (lane == 0) C[n] = __float2half(bias ? acc + __half2float(bias[n]) : acc);
}

// Weight-only INT8: q int8 rows + per-channel scale s[n]. The scale is applied ONCE, outside the
// k-loop (that is what per-channel-over-rows buys). char4 loads keep the warp at 128 B/instruction.
__global__ void k_gemv_int8(half *C, const half *A, const int8_t *Wq, const float *Ws,
                            const half *bias, int N, int K) {
    extern __shared__ float As[];
    const int tid = threadIdx.x, nt = blockDim.x;
    for (int k = tid; k < K; k += nt) As[k] = __half2float(A[k]);
    __syncthreads();

    const int warp = tid >> 5, lane = tid & 31;
    const int n = blockIdx.x*GV_WARPS + warp;
    if (n >= N) return;

    const char4 *w4 = (const char4 *)(Wq + (size_t)n*K);      // K % 4 == 0 for every weight here
    const int K4 = K >> 2;
    float acc = 0.f;                                          // scale-free dot
    for (int k4 = lane; k4 < K4; k4 += 32) {
        char4 v = w4[k4]; const int k = k4 << 2;
        acc += As[k]*(float)v.x + As[k+1]*(float)v.y + As[k+2]*(float)v.z + As[k+3]*(float)v.w;
    }
    #pragma unroll
    for (int off = 16; off; off >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, off);
    if (lane == 0) {
        float r = acc * Ws[n];
        C[n] = __float2half(bias ? r + __half2float(bias[n]) : r);
    }
}

// ---- decode-kernel launchers (used by the decode step AND, identically, by profile_decode.cu) ----
void gpt2_embed_one(half *x, const half *wte, const half *wpe, int token, int pos) {
    int nt = 256; k_embed_one<<<CEIL_DIV(E_, nt), nt>>>(x, wte, wpe, token, pos);
}
void gpt2_kv_append_one(GPT2KVCache *kv, int L, const half *qkv, int pos) {
    int nt = 256; k_kv_append<<<CEIL_DIV(E_, nt), nt>>>(gpt2_kv_K(kv,L), gpt2_kv_V(kv,L), qkv, pos, kv->maxT);
}
void gpt2_attn_decode(half *att, const half *qkv, const GPT2KVCache *kv, int L, int len) {
    k_attn_decode<<<H_, 64, (D_ + len)*sizeof(float)>>>(
        att, qkv, gpt2_kv_K((GPT2KVCache*)kv,L), gpt2_kv_V((GPT2KVCache*)kv,L), len, kv->maxT);
}

void gpt2_gemv_fp16(half *C, const half *A, const half *W, const half *bias, int N, int K) {
    k_gemv_fp16<<<CEIL_DIV(N, GV_WARPS), 32*GV_WARPS, K*sizeof(float)>>>(C, A, W, bias, N, K);
}
void gpt2_gemv_int8(half *C, const half *A, const int8_t *Wq, const float *Ws, const half *bias,
                    int N, int K) {
    k_gemv_int8<<<CEIL_DIV(N, GV_WARPS), 32*GV_WARPS, K*sizeof(float)>>>(C, A, Wq, Ws, bias, N, K);
}

// The ONE decode-path decision point. Mirrors gpt2_matmul_dispatch.
//   .gemv_q + quantized tensor -> INT8 GEMV        (Stage 5's INT8 decode)
//   .gemv                      -> fp16 GEMV        (Stage 5's fp16 decode; also the fallback for a
//                                                   tensor kept fp16 by Stage 4's kill-test, e.g. the head)
//   neither                    -> this backend's M=1 MATMUL  (naive / tiled -> director-map row 2)
void gpt2_gemv_dispatch(const GPT2Backend *be, half *C, const half *A,
                        const half *Wh, const GPT2QW *Wq, const half *bias, int N, int K) {
    if (be->gemv_q && Wq && Wq->q) be->gemv_q(C, A, Wq->q, Wq->s, bias, N, K);
    else if (be->gemv)             be->gemv  (C, A, Wh, bias, N, K);
    else                           gpt2_matmul_dispatch(be, C, A, Wh, Wq, bias, 1, N, K);
}

// ============================ the decode step ============================
void gpt2_decode_step_cuda(const GPT2Backend *be, const GPT2WeightsGPU *w, GPT2KVCache *kv,
                           int token, int pos, GPT2ScratchGPU *s, half *logits, half *caps_blocks) {
    const int F = GPT2_FFN_DIM, Q = GPT2_QKV_DIM, V = GPT2_VOCAB;
    const GPT2QWeightsGPU *qw = w->q;

    gpt2_embed_one(s->x, w->wte, w->wpe, token, pos);
    if (caps_blocks) CUDA_CHECK(cudaMemcpyAsync(caps_blocks + (size_t)(GPT2_N_LAYER+1)*E_, s->x,
                                                E_*sizeof(half), cudaMemcpyDeviceToDevice));

    for (int L = 0; L < GPT2_N_LAYER; L++) {
        const GPT2LayerGPU  *ly = &w->layers[L];
        const GPT2QLayerGPU *qy = qw ? &qw->layers[L] : nullptr;

        be->layernorm(s->ln, s->x, ly->ln1_g, ly->ln1_b, 1, E_);
        gpt2_gemv_dispatch(be, s->qkv, s->ln, ly->qkv_w, qy?&qy->qkv_w:nullptr, ly->qkv_b, Q, E_);

        // append this position's K/V, THEN attend over [0, pos] inclusive (len = pos+1)
        gpt2_kv_append_one(kv, L, s->qkv, pos);
        const int len = pos + 1;
        gpt2_attn_decode(s->att, s->qkv, kv, L, len);

        gpt2_gemv_dispatch(be, s->ao, s->att, ly->attn_proj_w, qy?&qy->attn_proj_w:nullptr, ly->attn_proj_b, E_, E_);
        be->add(s->x, s->ao, (size_t)E_);
        be->layernorm(s->ln, s->x, ly->ln2_g, ly->ln2_b, 1, E_);
        gpt2_gemv_dispatch(be, s->fc, s->ln, ly->fc_w, qy?&qy->fc_w:nullptr, ly->fc_b, F, E_);
        be->gelu(s->fc, (size_t)F);
        gpt2_gemv_dispatch(be, s->ff, s->fc, ly->proj_w, qy?&qy->proj_w:nullptr, ly->proj_b, E_, F);
        be->add(s->x, s->ff, (size_t)E_);
        if (caps_blocks) CUDA_CHECK(cudaMemcpyAsync(caps_blocks + (size_t)L*E_, s->x,
                                                    E_*sizeof(half), cudaMemcpyDeviceToDevice));
    }
    be->layernorm(s->ln, s->x, w->lnf_g, w->lnf_b, 1, E_);
    if (caps_blocks) CUDA_CHECK(cudaMemcpyAsync(caps_blocks + (size_t)GPT2_N_LAYER*E_, s->ln,
                                                E_*sizeof(half), cudaMemcpyDeviceToDevice));
    if (logits) gpt2_gemv_dispatch(be, logits, s->ln, w->wte, qw?&qw->wte:nullptr, nullptr, V, E_);
    kv->len = pos + 1;
}

void gpt2_prefill_fill_cache(const GPT2Backend *be, const GPT2WeightsGPU *w, GPT2KVCache *kv,
                             const int *d_ids, int T, GPT2ScratchGPU *s, GPT2CapsGPU *caps) {
    GPT2CapsGPU c; if (caps) c = *caps; else memset(&c, 0, sizeof(c));
    c.kv = (struct GPT2KVCache *)kv;
    gpt2_forward_cuda(be, w, d_ids, T, s, &c);
    kv->len = T;
}

// ---------------- Stage 5 backend: flash + the true fp16 M=1 GEMV ----------------
// Composed from GPT2_BACKEND_NAIVE (constant-initialized) + exported function pointers, never by
// copying a dynamically-initialized backend object from another TU (the Stage-3 static-init trap).
static GPT2Backend make_gemv_backend() {
    GPT2Backend b = GPT2_BACKEND_NAIVE;
    b.name      = "gemv";
    b.matmul    = gpt2_matmul_tiled;        // prefill == Stage 2
    b.attention = gpt2_attention_flash;     // prefill == Stage 3b
    b.gemv      = gpt2_gemv_fp16;           // decode  == Stage 5 (the only new thing)
    return b;
}
const GPT2Backend GPT2_BACKEND_GEMV = make_gemv_backend();
