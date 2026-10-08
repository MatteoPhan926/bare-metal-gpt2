// Phase 2.2: change ONLY the serial value reduction. Keep QK/softmax's
// 64 logical workers and reduction order, including the inefficient K loads.
#include "attention_v4.cuh"
#include <math_constants.h>
#include <cassert>

static_assert(GPT2_HEAD_DIM == 64, "V4 kernel assumes 64 dimensions");
__global__ void k_attn_decode_v4(half *att, const half *q, const half *K, const half *V,
                               int len, int maxT) {
    const int h=blockIdx.x, tid=threadIdx.x;
    extern __shared__ float sh[];
    float *qs=sh, *sc=sh+64;
    __shared__ float red[64], partial[4][64];
    if(tid<64) qs[tid]=__half2float(q[h*64+tid]);
    __syncthreads();
    const half *Kh=K+(size_t)h*maxT*64;
    const half *Vh=V+(size_t)h*maxT*64;
    const float scale=1.0f/sqrtf(64.0f);

    // All 256 threads reach every barrier, even while only 64 do QK/softmax.
    float m=-CUDART_INF_F;
    if(tid<64) {
        for(int j=tid;j<len;j+=64) {
            float dot=0.0f;
            #pragma unroll
            for(int d=0;d<64;d++) dot+=qs[d]*__half2float(Kh[(size_t)j*64+d]);
            float s=dot*scale; sc[j]=s; m=fmaxf(m,s);
        }
        red[tid]=m;
    }
    __syncthreads();
    for(int st=32;st;st>>=1) {
        if(tid<st) red[tid]=fmaxf(red[tid],red[tid+st]);
        __syncthreads();
    }
    m=red[0]; __syncthreads();
    if(tid<64) {
        float l=0.0f;
        for(int j=tid;j<len;j+=64) { float p=__expf(sc[j]-m); sc[j]=p; l+=p; }
        red[tid]=l;
    }
    __syncthreads();
    for(int st=32;st;st>>=1) {
        if(tid<st) red[tid]+=red[tid+st];
        __syncthreads();
    }
    const float inv=1.0f/red[0]; __syncthreads();

    // The intervention: four independent contiguous portions of the same V
    // sum, then three additions. Empty ranges (len<4) contribute zero.
    const int group=tid/64, d=tid%64;
    float a=0.0f;
    for(int j=len*group/4;j<len*(group+1)/4;j++)
        a+=sc[j]*__half2float(Vh[(size_t)j*64+d]);
    partial[group][d]=a; __syncthreads();
    if(tid<64) {
        float sum=((partial[0][tid]+partial[1][tid])+partial[2][tid])+partial[3][tid];
        att[h*64+tid]=__float2half(sum*inv);
    }
}

void gpt2_attn_decode_v4(half *out,const half *q,const GPT2KVCache *kv,int layer,int len) {
    assert(len>=1 && len<=kv->maxT && kv->maxT<=GPT2_N_CTX);
    assert(layer>=0 && layer<GPT2_N_LAYER && kv->headDim==64);
    k_attn_decode_v4<<<GPT2_N_HEAD,256,(64+len)*sizeof(float)>>>(
        out,q,gpt2_kv_K(kv,layer),gpt2_kv_V(kv,layer),len,kv->maxT);
}
