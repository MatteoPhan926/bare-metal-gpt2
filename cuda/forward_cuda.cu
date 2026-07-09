// forward_cuda.cu — STAGE 1: fp16 weight upload, reusable scratch, and the swappable forward pass.
//
// Weights: fp32 master (host) -> fp16 device via __float2half (round-to-nearest-even == torch .half()),
// so the GPU weights are byte-identical to what HF fp16 saw (QUALITY_GATES same-precision principle).
// The forward mirrors forward_cpu.c exactly, but each op runs through the backend's kernels.

#include "kernels.cuh"
#include "kvcache.cuh"     // STAGE 5: caps->kv scatter
#include "common.cuh"
#include <vector>
#include <cstring>

// Resolved at call time (not static-init time), so reading the dynamically-initialized TILED/FLASH
// backend objects here is safe.
const GPT2Backend *gpt2_backend_by_name(const char *name) {
    if (!name) return &GPT2_BACKEND_NAIVE;
    if (!strcmp(name, "tiled")) return &GPT2_BACKEND_TILED;
    if (!strcmp(name, "flash")) return &GPT2_BACKEND_FLASH;
    if (!strcmp(name, "int8"))  return &GPT2_BACKEND_INT8;
    if (!strcmp(name, "gemv"))  return &GPT2_BACKEND_GEMV;
    return &GPT2_BACKEND_NAIVE;
}

// STAGE 4. The single fp16-vs-INT8 decision point. A backend without .matmul_q, or a run without
// quantized weights attached (Wq->q == NULL), takes the fp16 path unchanged — so stages 1-3 are
// bit-for-bit unaffected by this hook.
void gpt2_matmul_dispatch(const GPT2Backend *be, half *C, const half *A,
                          const half *Wh, const GPT2QW *Wq, const half *bias, int M, int N, int K) {
    if (be->matmul_q && Wq && Wq->q) be->matmul_q(C, A, Wq->q, Wq->s, bias, M, N, K);
    else                             be->matmul  (C, A, Wh, bias, M, N, K);
}

void gpt2_upload_fp16(const GPT2Weights *cpu, GPT2WeightsGPU *gpu) {
    size_t n = cpu->n_floats; gpu->n = n; gpu->q = nullptr;   // Stage 4 mirrors attached separately
    std::vector<half> stg(n);
    for (size_t i = 0; i < n; i++) stg[i] = __float2half(cpu->data[i]);   // == torch .half() (RN)
    gpu->data = dmalloc<half>(n);
    h2d(gpu->data, stg.data(), n);
    // device pointers = same offsets the CPU loader computed into cpu->data
    #define OFF(p) (gpu->data + ((p) - cpu->data))
    gpu->wte = OFF(cpu->wte); gpu->wpe = OFF(cpu->wpe);
    for (int L = 0; L < GPT2_N_LAYER; L++) {
        const GPT2Layer *c = &cpu->layers[L]; GPT2LayerGPU *g = &gpu->layers[L];
        g->ln1_g = OFF(c->ln1_g); g->ln1_b = OFF(c->ln1_b);
        g->qkv_w = OFF(c->qkv_w); g->qkv_b = OFF(c->qkv_b);
        g->attn_proj_w = OFF(c->attn_proj_w); g->attn_proj_b = OFF(c->attn_proj_b);
        g->ln2_g = OFF(c->ln2_g); g->ln2_b = OFF(c->ln2_b);
        g->fc_w = OFF(c->fc_w); g->fc_b = OFF(c->fc_b);
        g->proj_w = OFF(c->proj_w); g->proj_b = OFF(c->proj_b);
    }
    gpu->lnf_g = OFF(cpu->lnf_g); gpu->lnf_b = OFF(cpu->lnf_b);
    #undef OFF
    CUDA_CHECK(cudaGetLastError());
}
void gpt2_free_gpu(GPT2WeightsGPU *g) { if (g && g->data) { cudaFree(g->data); g->data = nullptr; } }

void gpt2_scratch_alloc(GPT2ScratchGPU *s, int maxT) {
    s->maxT = maxT;
    const int E = GPT2_N_EMBD, Q = GPT2_QKV_DIM, F = GPT2_FFN_DIM;
    s->x  = dmalloc<half>((size_t)maxT * E);
    s->ln = dmalloc<half>((size_t)maxT * E);
    s->qkv= dmalloc<half>((size_t)maxT * Q);
    s->att= dmalloc<half>((size_t)maxT * E);
    s->ao = dmalloc<half>((size_t)maxT * E);
    s->fc = dmalloc<half>((size_t)maxT * F);
    s->ff = dmalloc<half>((size_t)maxT * E);
}
void gpt2_scratch_free(GPT2ScratchGPU *s) {
    cudaFree(s->x); cudaFree(s->ln); cudaFree(s->qkv); cudaFree(s->att);
    cudaFree(s->ao); cudaFree(s->fc); cudaFree(s->ff);
}

static void capcpy(half *dst, const half *src, size_t n) {
    CUDA_CHECK(cudaMemcpy(dst, src, n * sizeof(half), cudaMemcpyDeviceToDevice));
}

void gpt2_forward_cuda(const GPT2Backend *be, const GPT2WeightsGPU *w, const int *d_ids, int T,
                       GPT2ScratchGPU *s, GPT2CapsGPU *caps) {
    const int E = GPT2_N_EMBD, Q = GPT2_QKV_DIM, F = GPT2_FFN_DIM, V = GPT2_VOCAB,
              H = GPT2_N_HEAD, D = GPT2_HEAD_DIM;

    // Quantized mirrors of the five weight matmuls, when attached (Stage 4). The EMBED lookup below
    // deliberately keeps the fp16 wte: it is a 1-row-per-token gather, not a matmul, and quantizing it
    // would perturb the embedding itself rather than the tied output head.
    const GPT2QWeightsGPU *qw = w->q;

    be->embed(s->x, w->wte, w->wpe, d_ids, T, E);
    if (caps && caps->embed) capcpy(caps->embed, s->x, (size_t)T * E);

    for (int L = 0; L < GPT2_N_LAYER; L++) {
        const GPT2LayerGPU  *ly = &w->layers[L];
        const GPT2QLayerGPU *qy = qw ? &qw->layers[L] : nullptr;
        be->layernorm(s->ln, s->x, ly->ln1_g, ly->ln1_b, T, E);
        gpt2_matmul_dispatch(be, s->qkv, s->ln,  ly->qkv_w,       qy?&qy->qkv_w:nullptr,       ly->qkv_b,       T, Q, E);
        // STAGE 5: a prefill fills the KV cache from the SAME qkv the attention consumes -- one loop,
        // no near-duplicate "prefill_with_cache" to drift out of sync.
        if (caps && caps->kv) gpt2_kv_scatter_prefill((GPT2KVCache *)caps->kv, L, s->qkv, T);
        be->attention(s->att, s->qkv, T, E, H, D);
        gpt2_matmul_dispatch(be, s->ao,  s->att, ly->attn_proj_w, qy?&qy->attn_proj_w:nullptr, ly->attn_proj_b, T, E, E);
        be->add(s->x, s->ao, (size_t)T * E);
        be->layernorm(s->ln, s->x, ly->ln2_g, ly->ln2_b, T, E);
        gpt2_matmul_dispatch(be, s->fc,  s->ln,  ly->fc_w,        qy?&qy->fc_w:nullptr,        ly->fc_b,        T, F, E);
        be->gelu(s->fc, (size_t)T * F);
        gpt2_matmul_dispatch(be, s->ff,  s->fc,  ly->proj_w,      qy?&qy->proj_w:nullptr,      ly->proj_b,      T, E, F);
        be->add(s->x, s->ff, (size_t)T * E);
        if (caps && caps->blocks) capcpy(caps->blocks + (size_t)L * T * E, s->x, (size_t)T * E);
    }

    be->layernorm(s->ln, s->x, w->lnf_g, w->lnf_b, T, E);
    if (caps && caps->final_ln) capcpy(caps->final_ln, s->ln, (size_t)T * E);
    if (caps && caps->logits) {
        const GPT2QW *qh = qw ? &qw->wte : nullptr;                       // tied head
        if (caps->logits_all) gpt2_matmul_dispatch(be, caps->logits, s->ln, w->wte, qh, nullptr, T, V, E);
        else gpt2_matmul_dispatch(be, caps->logits, s->ln + (size_t)(T-1)*E, w->wte, qh, nullptr, 1, V, E);
    }
}
