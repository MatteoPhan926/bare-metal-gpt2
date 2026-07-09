// config.h — LOCKED GPT-2-124M dimensions (ROOFLINE §2). Single source of truth.
// Included by both C (cpu/) and CUDA (cuda/) — keep plain-C compatible.
//
// GPT-2 small (HF id "gpt2"):
//   n_layer=12  n_head=12  n_embd=768  vocab=50257  ctx=1024
//   activation = gelu_new (tanh approx, NOT erf)   norm = pre-LN LayerNorm (eps 1e-5)
//   learned positional embeddings (wpe)            tied input/output embeddings (wte is the head)
#ifndef GPT2_CONFIG_H
#define GPT2_CONFIG_H

#define GPT2_N_LAYER    12
#define GPT2_N_HEAD     12
#define GPT2_N_EMBD     768
#define GPT2_VOCAB      50257
#define GPT2_N_CTX      1024
#define GPT2_HEAD_DIM   (GPT2_N_EMBD / GPT2_N_HEAD)   // 64
#define GPT2_FFN_DIM    (4 * GPT2_N_EMBD)             // 3072
#define GPT2_QKV_DIM    (3 * GPT2_N_EMBD)             // 2304 (c_attn output)
#define GPT2_LN_EPS     1e-5f

// ---- Parameter-count contract (export_gpt2.py must reproduce EXACTLY; else a layout bug) ----
//   wte            50257 x 768                       = 38,597,376
//   wpe             1024 x 768                       =    786,432
//   per block (x12):
//     ln_1 (g,b)                                     =      1,536
//     attn.c_attn  W[768x2304] + b[2304]             =  1,771,776
//     attn.c_proj  W[768x768]  + b[768]              =    590,592
//     ln_2 (g,b)                                     =      1,536
//     mlp.c_fc     W[768x3072] + b[3072]             =  2,362,368
//     mlp.c_proj   W[3072x768] + b[768]              =  2,360,064
//     block total                                    =  7,087,872
//   12 blocks                                        = 85,054,464
//   ln_f (g,b)                                       =      1,536
//   ----------------------------------------------------------------
//   TOTAL (tied head, counted once)                  = 124,439,808  (~124.44M)
#define GPT2_TOTAL_PARAMS 124439808L

// Weight-streamed-per-decode-token (ROOFLINE §2): 12 blocks (85.05M) + tied head (38.60M) ~= 124M.
// wpe is a 1-row lookup per token -> negligible traffic.

// TRAP (export): HF stores linear weights as Conv1D [in,out], NOT Linear [out,out]. Transpose on export.
// TRAP (activation): gelu_new = 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3))). Match exactly.
#endif // GPT2_CONFIG_H
