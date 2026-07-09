// forward_cpu.c — STAGE 0 : pure-C fp32 GPT-2-124M forward pass (correctness reference).
//
// Matches HF `gpt2` fp32 exactly:
//   x = wte[id] + wpe[pos]
//   per block (pre-LN):  a = LN1(x); x += c_proj(attn(c_attn(a)));  m = LN2(x); x += mlp_cproj(gelu_new(c_fc(m)))
//   final:               h = LN_f(x);  logits = h · wteᵀ            (tied output head, no bias)
//
// Conventions (mirror model/weights.h): every linear weight is stored [out,in], so ONE matmul form
//   C[t,n] = sum_k A[t,k] * W[n,k] + bias[n]     (== HF Conv1D's  x @ weight + bias  after transpose)
//
// TRAP checklist honored here:
//   * gelu_new = 0.5 x (1 + tanh( sqrt(2/pi) (x + 0.044715 x^3) ))   — tanh approx, NOT erf.
//   * LayerNorm uses BIASED variance (÷E) and eps INSIDE the sqrt: 1/sqrt(var+eps).  eps = 1e-5.
//   * attention scale = 1/sqrt(head_dim) = 0.125, applied to the q·k score. causal: query i sees j<=i.
//   * qkv split is [q|k|v], each N_EMBD wide; head h occupies [h*head_dim, (h+1)*head_dim) within each.
//
// float storage (matches the fp32 oracle's per-op rounding) + double accumulation (removes OUR
// reduction error, leaving only HF's fp32 rounding as the diff).

#include "forward_cpu.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define E    GPT2_N_EMBD     // 768
#define H    GPT2_N_HEAD     // 12
#define D    GPT2_HEAD_DIM   // 64
#define QKV  GPT2_QKV_DIM    // 2304
#define FF   GPT2_FFN_DIM    // 3072
#define V    GPT2_VOCAB      // 50257

// C[M,N] = A[M,K] @ W[N,K]^T + bias[N]   (W row-major [N,K]; bias may be NULL).
// Parallel over the flattened M*N output space; each element is a full independent dot product,
// so accumulation order is fixed and thread-count-invariant. (M*N <= 512*50257 < INT_MAX here.)
static void matmul(float *C, const float *A, const float *W, const float *bias,
                   int M, int N, int K) {
    const int MN = M * N;
    int idx;   // OpenMP 2.0 (MSVC) requires the loop var declared outside the for-init
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (idx = 0; idx < MN; idx++) {
        int m = idx / N, n = idx - m * N;
        const float *a  = A + (size_t)m * K;
        const float *wn = W + (size_t)n * K;
        double acc = bias ? (double)bias[n] : 0.0;
        for (int k = 0; k < K; k++) acc += (double)a[k] * (double)wn[k];
        C[idx] = (float)acc;
    }
}

// out[M,E] = LayerNorm(x[M,E]; g,b)  row-wise. Biased variance, eps inside sqrt (HF semantics).
static void layernorm(float *out, const float *x, const float *g, const float *b, int M) {
    int m;
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (m = 0; m < M; m++) {
        const float *xr = x + (size_t)m * E;
        double mean = 0.0;
        for (int e = 0; e < E; e++) mean += (double)xr[e];
        mean /= E;
        double var = 0.0;
        for (int e = 0; e < E; e++) { double d = (double)xr[e] - mean; var += d * d; }
        var /= E;
        double inv = 1.0 / sqrt(var + (double)GPT2_LN_EPS);
        float *o = out + (size_t)m * E;
        for (int e = 0; e < E; e++)
            o[e] = (float)(((double)xr[e] - mean) * inv * (double)g[e] + (double)b[e]);
    }
}

// gelu_new (tanh approximation) — GPT-2's activation. double math, cast back to float.
static inline float gelu_new(float xf) {
    double x = (double)xf;
    double inner = 0.7978845608028654 * (x + 0.044715 * x * x * x); // sqrt(2/pi) * (x + 0.044715 x^3)
    return (float)(0.5 * x * (1.0 + tanh(inner)));
}

// Causal multi-head self-attention.
//   qkv[T, 3E] laid out [q(E) | k(E) | v(E)] per row; att[T, E] receives concatenated head contexts.
// Parallel over (head, query) pairs — each writes a disjoint att[i, h*D..) slice.
static void attention(float *att, const float *qkv, int T) {
    const double scale = 1.0 / sqrt((double)D);
    const int HT = H * T;
    int hi;
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (hi = 0; hi < HT; hi++) {
        int h = hi / T, i = hi - h * T;                 // head h, query position i
        const float *qi = qkv + (size_t)i * QKV + 0 * E + h * D;
        double sc[GPT2_N_CTX];                          // scores over keys j=0..i  (fixed-size, thread-local)
        double maxv = -1e300;
        for (int j = 0; j <= i; j++) {
            const float *kj = qkv + (size_t)j * QKV + 1 * E + h * D;
            double d = 0.0;
            for (int x = 0; x < D; x++) d += (double)qi[x] * (double)kj[x];
            d *= scale;
            sc[j] = d;
            if (d > maxv) maxv = d;
        }
        double sum = 0.0;
        for (int j = 0; j <= i; j++) { sc[j] = exp(sc[j] - maxv); sum += sc[j]; }
        double inv = 1.0 / sum;
        double acc[GPT2_HEAD_DIM];
        for (int x = 0; x < D; x++) acc[x] = 0.0;
        for (int j = 0; j <= i; j++) {                  // context = sum_j softmax_j * v_j
            const float *vj = qkv + (size_t)j * QKV + 2 * E + h * D;
            double p = sc[j];
            for (int x = 0; x < D; x++) acc[x] += p * (double)vj[x];
        }
        float *o = att + (size_t)i * E + h * D;
        for (int x = 0; x < D; x++) o[x] = (float)(acc[x] * inv);
    }
}

void gpt2_forward_cpu(const GPT2Weights *w, const int *ids, int T, GPT2Caps *caps) {
    float *x   = (float*)malloc((size_t)T * E   * sizeof(float));   // residual stream (in/out)
    float *ln  = (float*)malloc((size_t)T * E   * sizeof(float));   // LN output (reused: attn then mlp)
    float *qkv = (float*)malloc((size_t)T * QKV * sizeof(float));
    float *att = (float*)malloc((size_t)T * E   * sizeof(float));   // attention context
    float *ao  = (float*)malloc((size_t)T * E   * sizeof(float));   // attn c_proj output
    float *fc  = (float*)malloc((size_t)T * FF  * sizeof(float));   // mlp c_fc output (pre/post gelu)
    float *ff  = (float*)malloc((size_t)T * E   * sizeof(float));   // mlp c_proj output

    // --- embeddings: token + learned positional ---
    for (int t = 0; t < T; t++) {
        const float *we = w->wte + (size_t)ids[t] * E;
        const float *pe = w->wpe + (size_t)t * E;
        float *xt = x + (size_t)t * E;
        for (int e = 0; e < E; e++) xt[e] = we[e] + pe[e];
    }
    if (caps && caps->embed) memcpy(caps->embed, x, (size_t)T * E * sizeof(float));

    // --- 12 transformer blocks (pre-LN) ---
    for (int L = 0; L < GPT2_N_LAYER; L++) {
        const GPT2Layer *ly = &w->layers[L];
        // attention sub-block
        layernorm(ln, x, ly->ln1_g, ly->ln1_b, T);
        matmul(qkv, ln, ly->qkv_w, ly->qkv_b, T, QKV, E);
        attention(att, qkv, T);
        matmul(ao, att, ly->attn_proj_w, ly->attn_proj_b, T, E, E);
        for (size_t i = 0; i < (size_t)T * E; i++) x[i] += ao[i];         // residual
        // mlp sub-block
        layernorm(ln, x, ly->ln2_g, ly->ln2_b, T);
        matmul(fc, ln, ly->fc_w, ly->fc_b, T, FF, E);
        for (size_t i = 0; i < (size_t)T * FF; i++) fc[i] = gelu_new(fc[i]);
        matmul(ff, fc, ly->proj_w, ly->proj_b, T, E, FF);
        for (size_t i = 0; i < (size_t)T * E; i++) x[i] += ff[i];         // residual
        if (caps && caps->blocks)
            memcpy(caps->blocks + (size_t)L * T * E, x, (size_t)T * E * sizeof(float));
    }

    // --- final LayerNorm + tied-head logits ---
    layernorm(ln, x, w->lnf_g, w->lnf_b, T);
    if (caps && caps->final_ln) memcpy(caps->final_ln, ln, (size_t)T * E * sizeof(float));
    if (caps && caps->logits) {
        if (caps->logits_all)
            matmul(caps->logits, ln, w->wte, NULL, T, V, E);              // [T, V]
        else
            matmul(caps->logits, ln + (size_t)(T - 1) * E, w->wte, NULL, 1, V, E); // last pos [V]
    }

    free(x); free(ln); free(qkv); free(att); free(ao); free(fc); free(ff);
}
