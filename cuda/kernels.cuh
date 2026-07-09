// kernels.cuh — STAGE 1 backend interface: fp16 device weights, reusable scratch, capture buffers,
// and the SWAPPABLE kernel backend (Stage 1 = naive; later stages register tiled/fused/quant).
#ifndef GPT2_KERNELS_CUH
#define GPT2_KERNELS_CUH

#include <cuda_fp16.h>
#include <stdint.h>
#include "config.h"

// weights.h is C (compiled as C in weights.c) -> give its decls C linkage in these C++ TUs.
#ifdef __cplusplus
extern "C" {
#endif
#include "weights.h"      // GPT2Weights (fp32 master, host)
#ifdef __cplusplus
}
#endif

// ---- fp16 device weights: a deterministic .half() of the fp32 master (matches HF fp16 oracle) ----
typedef struct {
    half *ln1_g,*ln1_b, *qkv_w,*qkv_b, *attn_proj_w,*attn_proj_b,
         *ln2_g,*ln2_b, *fc_w,*fc_b, *proj_w,*proj_b;
} GPT2LayerGPU;

// ---- STAGE 4: one quantized weight tensor. W[N,K] ~= q[n,k] * s[n]  (symmetric, PER-CHANNEL over
// the output rows n, weight-only: activations stay fp16). Attached ALONGSIDE the fp16 tensors; the
// fp16 copies stay resident but are not streamed by a quantized matmul (only embed still reads wte).
typedef struct { const int8_t *q; const float *s; } GPT2QW;   // q [N,K] row-major, s [N]

typedef struct { GPT2QW qkv_w, attn_proj_w, fc_w, proj_w; } GPT2QLayerGPU;

typedef struct {
    int8_t *qdata;               // owns the packed int8 body (device)
    float  *sdata;               // owns all per-channel scales (device)
    size_t  nq, ns;
    size_t  streamed_bytes;      // int8 bytes + fp16 bytes of any kept-fp16 tensor == the decode denominator
    GPT2QW  wte;                 // the tied output head (quantized); the EMBED lookup still uses fp16 wte
    GPT2QLayerGPU layers[GPT2_N_LAYER];
} GPT2QWeightsGPU;

typedef struct {
    half  *data;                 // owns the whole fp16 body (device)
    size_t n;
    half  *wte, *wpe;            // wte is also the tied output head
    GPT2LayerGPU layers[GPT2_N_LAYER];
    half  *lnf_g, *lnf_b;
    const GPT2QWeightsGPU *q;    // STAGE 4: NULL unless quantized weights were uploaded
} GPT2WeightsGPU;

void gpt2_upload_fp16(const GPT2Weights *cpu, GPT2WeightsGPU *gpu);   // host fp32 -> device fp16 (RN)
void gpt2_free_gpu(GPT2WeightsGPU *gpu);

// ---- reusable device activation scratch (allocate once for max T, reuse every forward) ----
typedef struct { half *x,*ln,*qkv,*att,*ao,*fc,*ff; int maxT; } GPT2ScratchGPU;
void gpt2_scratch_alloc(GPT2ScratchGPU *s, int maxT);
void gpt2_scratch_free(GPT2ScratchGPU *s);

// ---- optional per-stage capture (device buffers; any NULL field is skipped) ----
//   embed [T*E]   blocks [N_LAYER*T*E] (block L at +L*T*E)   final_ln [T*E]
//   logits [T*V] if logits_all else [V] (last position)
//   kv (STAGE 5): if non-NULL, the forward SCATTERS each layer's K/V into the cache as it computes
//       them. Capturing the cache IS a capture, so it belongs here -- this keeps ONE forward loop
//       rather than a near-duplicate "prefill_with_cache" that could silently drift from it.
struct GPT2KVCache;
typedef struct { half *embed,*blocks,*final_ln,*logits; int logits_all; struct GPT2KVCache *kv; } GPT2CapsGPU;

// ---- swappable kernel backend ----
typedef struct {
    const char *name;
    void (*embed)    (half *x, const half *wte, const half *wpe, const int *ids, int T, int E);
    void (*layernorm)(half *out, const half *x, const half *g, const half *b, int M, int E);
    void (*matmul)   (half *C, const half *A, const half *W, const half *bias, int M, int N, int K); // C=A[M,K]·W[N,K]^T+b
    void (*attention)(half *att, const half *qkv, int T, int E, int H, int D);                        // causal MHA
    void (*gelu)     (half *x, size_t n);                                                             // gelu_new, in-place
    void (*add)      (half *x, const half *y, size_t n);                                              // residual x += y
    // STAGE 4 (optional; NULL on every fp16 backend). Same GEMM contract, but W arrives as int8 rows +
    // per-row scales. When non-NULL AND quantized weights are attached, the forward calls this INSTEAD
    // of .matmul at all five weight-matmul sites. Leaving it NULL keeps stages 1-3 byte-identical.
    void (*matmul_q) (half *C, const half *A, const int8_t *Wq, const float *Ws, const half *bias,
                      int M, int N, int K);
    // STAGE 5 (optional; NULL => the decode path falls back to this backend's M=1 *matmul*, which is
    // exactly how "tiled GEMM at true M=1" is measured for director-map row 2). C[1,N] = A[1,K]*W^T+b.
    void (*gemv)     (half *C, const half *A, const half *W, const half *bias, int N, int K);
    void (*gemv_q)   (half *C, const half *A, const int8_t *Wq, const float *Ws, const half *bias,
                      int N, int K);
} GPT2Backend;

extern const GPT2Backend GPT2_BACKEND_NAIVE;   // constant-initialized (safe to read from any TU's init)
extern const GPT2Backend GPT2_BACKEND_TILED;   // STAGE 2: naive backend with the tiled/SMEM matmul swapped in
extern const GPT2Backend GPT2_BACKEND_FLASH;   // STAGE 3b: tiled backend with flash-style attention swapped in
extern const GPT2Backend GPT2_BACKEND_INT8;    // STAGE 4: flash backend with the weight-only INT8 matmul
extern const GPT2Backend GPT2_BACKEND_GEMV;    // STAGE 5: flash backend + the true M=1 fp16 GEMV (decode)

// Stage 2's tiled GEMM, exported so later backends can compose from GPT2_BACKEND_NAIVE + this, instead
// of reading the dynamically-initialized GPT2_BACKEND_TILED across TUs (unspecified init order).
void gpt2_matmul_tiled(half *C, const half *A, const half *W, const half *bias, int M, int N, int K);
// Likewise Stage 3b's flash attention (Stage 4's backend = flash + the INT8 matmul).
void gpt2_attention_flash(half *att, const half *qkv, int T, int E, int H, int D);

// Backend lookup by name (env GPT2_BACKEND); NULL/unknown -> naive. One definition, all harnesses.
const GPT2Backend *gpt2_backend_by_name(const char *name);

// The ONE place that decides fp16-vs-INT8 for a weight matmul. Used by the forward and by the A/B
// harness, so an isolated-op timing dispatches exactly as the whole forward does.
void gpt2_matmul_dispatch(const GPT2Backend *be, half *C, const half *A,
                          const half *Wh, const GPT2QW *Wq, const half *bias, int M, int N, int K);

// ---- STAGE 4 quantized-weight loading ----
// Load tools/quantize.py's packed file, upload, and attach to `gpu->q`. Returns 0 on success.
int  gpt2_quant_load_upload(const char *path, GPT2WeightsGPU *gpu, GPT2QWeightsGPU *q);
void gpt2_quant_free(GPT2QWeightsGPU *q);
// Harness convenience: loads+attaches iff `be` actually has a quantized matmul; else a no-op.
int  gpt2_quant_attach_if_needed(const GPT2Backend *be, GPT2WeightsGPU *gpu, GPT2QWeightsGPU *q);

// ---- forward pass (drives the backend); fills the non-NULL fields of caps ----
void gpt2_forward_cuda(const GPT2Backend *be, const GPT2WeightsGPU *w, const int *d_ids, int T,
                       GPT2ScratchGPU *s, GPT2CapsGPU *caps);

#endif // GPT2_KERNELS_CUH
