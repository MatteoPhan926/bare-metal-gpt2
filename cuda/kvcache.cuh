// kvcache.cuh — STAGE 5: KV cache + memory planner + the TRUE M=1 decode path.
//
// This is the stage where a **true (M=1) decode** first EXISTS in this engine. Before it, the
// "decode" harness full-recomputed the growing sequence, so its GEMMs ran at M = context length --
// the prefill-shaped, reuse-bound regime (BENCH_PROTOCOL §3, ROOFLINE §5 annotations). Consequently
// TWO director-map rows were untestable and are first falsifiable here:
//   row 2 "tiled GEMM -> decode ~flat"      (is the tiling win really absent at M=1?)
//   row 4 "INT8 -> decode high"             (does halving weight bytes finally buy speed?)
//
// THE LOAD-BEARING DESIGN POINT (from Stage 4's measurement):
//   A KV cache ALONE will not surface the INT8 decode win. The 16x16 tiled GEMM at M=1 computes 16
//   output rows and discards 15, so it is saturated on wasted arithmetic (860.6 GFLOP/s = 98.8% of its
//   own M=512 throughput) while pulling only 23% of achieved BW. Measured proof: head M=1 (1.4351 ms)
//   == head M=16 (1.4346 ms). A byte-halving cannot show up against a compute-bound kernel.
//   => Stage 5 must ALSO introduce a true M=1-shaped GEMV: one warp per output row, every weight byte
//      read exactly once, no discarded rows. Only then is the traffic saving expressible.
//   The GEMV kernels live in kvcache.cu (not a new file) because they exist ONLY for the decode path
//   that this stage creates; BUILD_PLAN's Stage-5 manifest is cuda/kvcache.{cu,cuh}.

#ifndef GPT2_KVCACHE_CUH
#define GPT2_KVCACHE_CUH

#include <cuda_fp16.h>
#include "config.h"
#include "kernels.cuh"

// ---- the cache ------------------------------------------------------------------------------
// ONE contiguous allocation (the "memory planner": a single arena, no per-step malloc, no
// fragmentation, and a layout chosen to minimise decode traffic).
//
// Layout, per layer L:   K = base + (2L+0)*H*maxT*D      V = base + (2L+1)*H*maxT*D
// and within each:       [head][pos][dim]   (head-major, then position, then head_dim)
//
// WHY [H][T][D] and not [T][E]:  the decode attention reads, for ONE head, the slab K[h][0..len][0..D)
// -- contiguous. With [T][E] the same read strides by E=768 halves per position. Head-major makes the
// dominant decode read sequential; the append (one position, all heads) becomes the strided side, and
// it moves 12x64 halves per layer instead of 12x64x(len+1). Trade the rare stride for the hot one.
// Named (not an anonymous typedef) so kernels.cuh can forward-declare it for GPT2CapsGPU::kv.
typedef struct GPT2KVCache {
    half  *data;                 // owns the whole arena (device)
    size_t bytes;
    int    maxT, nLayer, nHead, headDim;
    int    len;                  // positions currently cached [0, len)
} GPT2KVCache;

int  gpt2_kv_alloc(GPT2KVCache *kv, int maxT);   // 0 on success; prints the plan + VRAM headroom
// Scatter a prefill's [T,3E] qkv into layer L of the cache (called by gpt2_forward_cuda via caps->kv).
void gpt2_kv_scatter_prefill(GPT2KVCache *kv, int L, const half *qkv, int T);

// The three decode-only kernels, exported as launchers so a profiler drives EXACTLY the code path
// gpt2_decode_step_cuda drives (rather than re-declaring __global__s across translation units).
void gpt2_embed_one     (half *x, const half *wte, const half *wpe, int token, int pos);
void gpt2_kv_append_one (GPT2KVCache *kv, int L, const half *qkv, int pos);   // appends this step's K,V
void gpt2_attn_decode   (half *att, const half *qkv, const GPT2KVCache *kv, int L, int len);
void gpt2_kv_free(GPT2KVCache *kv);
void gpt2_kv_reset(GPT2KVCache *kv);             // len = 0 (no realloc, no memset needed)

// bytes of K+V read per decode step at a given context length (the decode traffic the cache ADDS)
double gpt2_kv_step_bytes(const GPT2KVCache *kv, int len);

__host__ __device__ static inline half *gpt2_kv_K(const GPT2KVCache *kv, int L) {
    return kv->data + (size_t)(2*L + 0) * kv->nHead * kv->maxT * kv->headDim;
}
__host__ __device__ static inline half *gpt2_kv_V(const GPT2KVCache *kv, int L) {
    return kv->data + (size_t)(2*L + 1) * kv->nHead * kv->maxT * kv->headDim;
}

// ---- the decode step (TRUE M=1) -------------------------------------------------------------
// Consumes ONE token at position `pos`, appends its K/V to the cache, and writes logits[V].
// Requires kv->len == pos on entry; leaves kv->len == pos+1.
// `caps` (optional) captures this step's activations for gate (a); it must hold (N_LAYER+2)*E halves:
//    [0 .. N_LAYER-1] per-block hidden   [N_LAYER] final_ln   [N_LAYER+1] embed
#define GPT2_DECODE_CAPS_ROWS (GPT2_N_LAYER + 2)
void gpt2_decode_step_cuda(const GPT2Backend *be, const GPT2WeightsGPU *w, GPT2KVCache *kv,
                           int token, int pos, GPT2ScratchGPU *s, half *logits,
                           half *caps_blocks);

// Fill the cache from a prefill forward. Pass `caps.kv = &kv` to gpt2_forward_cuda instead when you
// want the normal forward to populate it; this wrapper is the convenience form.
void gpt2_prefill_fill_cache(const GPT2Backend *be, const GPT2WeightsGPU *w, GPT2KVCache *kv,
                             const int *d_ids, int T, GPT2ScratchGPU *s, GPT2CapsGPU *caps);

// ---- the M=1 GEMV (the reason Stage 5 can express the INT8 win at all) -----------------------
// C[1,N] = A[1,K] * W[N,K]^T + bias[N].  One warp per output row n: the warp's 32 lanes stride the
// K-dim with VECTORISED loads (half2 / char4), so W is read EXACTLY ONCE, fully coalesced, with zero
// discarded rows. Contrast the 16x16 tiled GEMM at M=1, which computes 16 rows and throws 15 away.
void gpt2_gemv_fp16(half *C, const half *A, const half *W, const half *bias, int N, int K);
void gpt2_gemv_int8(half *C, const half *A, const int8_t *Wq, const float *Ws, const half *bias,
                    int N, int K);

// The ONE fp16-vs-INT8-vs-fallback decision point for decode. A backend with .gemv == NULL falls back
// to its M=1 *matmul* -- which is exactly how "tiled GEMM at true M=1" (director-map row 2) is measured.
void gpt2_gemv_dispatch(const GPT2Backend *be, half *C, const half *A,
                        const half *Wh, const GPT2QW *Wq, const half *bias, int N, int K);

#endif // GPT2_KVCACHE_CUH
