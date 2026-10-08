#ifndef GPT2_ATTENTION_V4_CUH
#define GPT2_ATTENTION_V4_CUH
#include "kvcache.cuh"
// Same six-argument ABI and dynamic shared-memory layout as k_attn_decode.
// Decode only, D=64, 1<=len<=maxT<=1024; no layout/precision changes.
__global__ void k_attn_decode_v4(half *, const half *, const half *, const half *, int, int);
void gpt2_attn_decode_v4(half *out, const half *q, const GPT2KVCache *kv, int layer, int len);
#endif
