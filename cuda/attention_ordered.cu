// Value-loop replacement after V4 reassociation failed its extra numerical gate.
// QK, softmax and the sequence of fp32 value FMAs match k_attn_decode.
#include "attention_ordered.cuh"
#include <math_constants.h>
#include <cassert>

static_assert(GPT2_HEAD_DIM==64,"ordered kernel assumes D=64");
__global__ void k_attn_decode_ordered4(half *att,const half *q,const half *K,const half *V,
                                     int len,int maxT) {
    const int h=blockIdx.x,tid=threadIdx.x,nt=blockDim.x;
    extern __shared__ float sh[];
    float *qs=sh,*sc=sh+64;
    for(int d=tid;d<64;d+=nt) qs[d]=__half2float(q[h*64+d]);
    __syncthreads();
    const half *Kh=K+(size_t)h*maxT*64,*Vh=V+(size_t)h*maxT*64;
    const float scale=1.0f/sqrtf(64.0f);
    __shared__ float red[64];
    float m=-CUDART_INF_F;
    for(int j=tid;j<len;j+=nt) {
        float dot=0.0f;
        #pragma unroll
        for(int d=0;d<64;d++) dot+=qs[d]*__half2float(Kh[(size_t)j*64+d]);
        float s=dot*scale;sc[j]=s;m=fmaxf(m,s);
    }
    red[tid]=m;__syncthreads();
    for(int st=nt/2;st;st>>=1){if(tid<st)red[tid]=fmaxf(red[tid],red[tid+st]);__syncthreads();}
    m=red[0];__syncthreads();
    float l=0.0f;
    for(int j=tid;j<len;j+=nt){float p=__expf(sc[j]-m);sc[j]=p;l+=p;}
    red[tid]=l;__syncthreads();
    for(int st=nt/2;st;st>>=1){if(tid<st)red[tid]+=red[tid+st];__syncthreads();}
    const float inv=1.0f/red[0];__syncthreads();

    for(int d=tid;d<64;d+=nt) {
        float a=0.0f; int j=0;
        // Independent memory operations can be outstanding together. Unlike
        // partial sums, these FMAs still visit j=0,1,...,len-1 in exact order.
        for(;j+3<len;j+=4) {
            float v0=__half2float(Vh[(size_t)(j+0)*64+d]);
            float v1=__half2float(Vh[(size_t)(j+1)*64+d]);
            float v2=__half2float(Vh[(size_t)(j+2)*64+d]);
            float v3=__half2float(Vh[(size_t)(j+3)*64+d]);
            float p0=sc[j+0],p1=sc[j+1],p2=sc[j+2],p3=sc[j+3];
            a+=p0*v0; a+=p1*v1; a+=p2*v2; a+=p3*v3;
        }
        for(;j<len;j++) a+=sc[j]*__half2float(Vh[(size_t)j*64+d]);
        att[h*64+d]=__float2half(a*inv);
    }
}
void gpt2_attn_decode_ordered4(half *out,const half *q,const GPT2KVCache *kv,int layer,int len) {
    assert(len>=1 && len<=kv->maxT && kv->maxT<=GPT2_N_CTX && kv->headDim==64);
    assert(layer>=0 && layer<GPT2_N_LAYER);
    k_attn_decode_ordered4<<<GPT2_N_HEAD,64,(64+len)*sizeof(float)>>>(
        out,q,gpt2_kv_K(kv,layer),gpt2_kv_V(kv,layer),len,kv->maxT);
}
