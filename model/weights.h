// weights.h — load the fp32 raw weight file (tools/export_gpt2.py format) into typed pointers.
// Layout MIRRORS export_gpt2.py exactly. All linear weights are [out, in] (transposed from HF
// Conv1D) so the engine uses ONE matmul form: y[n] = sum_k x[k]*W[n,k] + b[n]  (C = A * B^T).
#ifndef GPT2_WEIGHTS_H
#define GPT2_WEIGHTS_H
#include <stddef.h>
#include "config.h"

typedef struct {
    float *ln1_g, *ln1_b;              // [N_EMBD]
    float *qkv_w,  *qkv_b;             // qkv_w [QKV_DIM, N_EMBD]   qkv_b [QKV_DIM]
    float *attn_proj_w, *attn_proj_b;  // [N_EMBD, N_EMBD] , [N_EMBD]
    float *ln2_g, *ln2_b;              // [N_EMBD]
    float *fc_w,   *fc_b;              // fc_w [FFN_DIM, N_EMBD]     fc_b [FFN_DIM]
    float *proj_w, *proj_b;            // proj_w [N_EMBD, FFN_DIM]   proj_b [N_EMBD]
} GPT2Layer;

typedef struct {
    float *data;                       // owns the whole body buffer (free this)
    size_t n_floats;
    float *wte;                        // [VOCAB, N_EMBD]  (also the tied output head)
    float *wpe;                        // [N_CTX, N_EMBD]
    GPT2Layer layers[GPT2_N_LAYER];
    float *lnf_g, *lnf_b;              // [N_EMBD]
} GPT2Weights;

// returns 0 on success, nonzero on error (prints reason). Validates magic + dims + file size.
int  gpt2_load_weights(const char *path, GPT2Weights *w);
void gpt2_free_weights(GPT2Weights *w);

#endif // GPT2_WEIGHTS_H
