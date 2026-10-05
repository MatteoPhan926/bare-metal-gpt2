#ifndef GPT2_ATTENTION_ORDERED_CUH
#define GPT2_ATTENTION_ORDERED_CUH
#include "kvcache.cuh"
__global__ void k_attn_decode_ordered4(half *,const half *,const half *,const half *,int,int);
void gpt2_attn_decode_ordered4(half *,const half *,const GPT2KVCache *,int,int);
#endif
