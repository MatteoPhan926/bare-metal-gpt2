// kv_gate.cu — STAGE 5 correctness gate (QUALITY_GATES §1 + A1) for the TRUE M=1 KV-cache decode path.
//
// A KV cache is a prime silent-bug site: an off-by-one in the append, a wrong head stride, or attending
// over [0,pos) instead of [0,pos] all produce output that still *looks* like text. So this harness
// checks the decode path four ways, cheapest-localizer first:
//
//   [EQ]     cached decode  ==  no-KV recompute, position by position (max |dlogit|, max KL, top-1).
//            NOT bit-identical by construction: in the recompute path a position's K/V come from a GEMM
//            at M=ctx; in the cached path they were produced at M=1 (GEMV) when that position was
//            decoded. Different reduction order -> different fp16 rounding. The bar is fp16 noise.
//   gate (a) per-layer rel_err of a DECODE-ONLY run (28 steps from pos 0, cache filled entirely by the
//            M=1 path) vs the HF fp16 oracle. This is the bug localizer: a KV indexing error shows up
//            as one layer blowing past 1e-2 while its predecessors pass.
//   gate (b) greedy token match, N=128, teacher-forced, A1 margin rule.
//   gate (c) PRIMARY max KL < 0.02 over the 512-position eval window, decoded ONE TOKEN AT A TIME.
//            512 sequential cache appends + attends: the deepest exercise of the cache in this repo.
//
// Run from repo root:  kv_gate.exe            env: GPT2_BACKEND=gemv|int8|tiled|naive|flash

#include "kernels.cuh"
#include "kvcache.cuh"
#include "common.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <algorithm>

#define E  GPT2_N_EMBD
#define V  GPT2_VOCAB
#define NL GPT2_N_LAYER

static char *read_text(const char *p){ FILE*f=fopen(p,"rb"); if(!f){fprintf(stderr,"[kv] open %s\n",p);exit(1);}
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET); char*b=(char*)malloc(n+1);
    if(fread(b,1,n,f)!=(size_t)n){exit(1);} b[n]=0; fclose(f); return b; }
static int parse_int_array(const char*buf,const char*key,int*out,int maxn){
    const char*p=strstr(buf,key); if(!p){fprintf(stderr,"[kv] key %s\n",key);exit(1);}
    p=strchr(p,'['); p++; int n=0;
    while(*p&&*p!=']'){ while(*p&&*p!=']'&&*p!='-'&&(*p<'0'||*p>'9'))p++; if(!*p||*p==']')break;
        char*e; long v=strtol(p,&e,10); if(e==p){p++;continue;} if(n<maxn)out[n]=(int)v; n++; p=e; }
    return n; }
static float *load_f16(const char*path,size_t n){ FILE*f=fopen(path,"rb"); if(!f){fprintf(stderr,"[kv] open %s\n",path);exit(1);}
    std::vector<half> raw(n); if(fread(raw.data(),sizeof(half),n,f)!=n){fprintf(stderr,"[kv] short %s\n",path);exit(1);} fclose(f);
    float*o=(float*)malloc(n*sizeof(float)); for(size_t i=0;i<n;i++)o[i]=__half2float(raw[i]); return o; }
static int32_t *load_i32(const char*p,size_t n){ FILE*f=fopen(p,"rb"); int32_t*q=(int32_t*)malloc(n*4);
    if(fread(q,4,n,f)!=n){exit(1);} fclose(f); return q; }
static float *load_f32(const char*p,size_t n){ FILE*f=fopen(p,"rb"); float*q=(float*)malloc(n*4);
    if(fread(q,4,n,f)!=n){exit(1);} fclose(f); return q; }
static float *dl_f16(const half*d,size_t n){ std::vector<half> h(n); d2h(h.data(),d,n);
    float*o=(float*)malloc(n*sizeof(float)); for(size_t i=0;i<n;i++)o[i]=__half2float(h[i]); return o; }
static double rel_err(const float*a,const float*b,size_t n){ double md=0,mr=0;
    for(size_t i=0;i<n;i++){ double d=fabs((double)a[i]-(double)b[i]); if(d>md)md=d;
        double r=fabs((double)b[i]); if(r>mr)mr=r; } return md/(mr+1e-9); }
static int argmax(const float*v,int n){ int m=0; for(int i=1;i<n;i++) if(v[i]>v[m])m=i; return m; }
static double kl_row(const float*ref,const float*ours,int n){ double mr=-1e300,mo=-1e300;
    for(int i=0;i<n;i++){ if(ref[i]>mr)mr=ref[i]; if(ours[i]>mo)mo=ours[i]; }
    double sr=0,so=0; for(int i=0;i<n;i++){ sr+=exp((double)ref[i]-mr); so+=exp((double)ours[i]-mo); }
    double lsr=mr+log(sr),lso=mo+log(so),kl=0;
    for(int i=0;i<n;i++){ double lpr=(double)ref[i]-lsr,lpo=(double)ours[i]-lso; kl+=exp(lpr)*(lpr-lpo); }
    return kl; }
static float fp16_ulp(float x){ float a=fabsf(x); if(a<6.104e-5f) return 5.96e-8f; int e; frexpf(a,&e); return ldexpf(1.0f,e-11); }

// Decode `n` tokens one at a time from position 0 (cache filled ENTIRELY by the M=1 path).
// d_logits_all: [n, V].  d_caps (optional): [(NL+2), n, E] laid out row-major per capture index.
static void kv_decode_seq(const GPT2Backend *be, const GPT2WeightsGPU *w, GPT2KVCache *kv,
                          const int *ids, int n, GPT2ScratchGPU *s, half *d_logits_all, half *d_caps) {
    gpt2_kv_reset(kv);
    half *step_caps = d_caps ? dmalloc<half>((size_t)GPT2_DECODE_CAPS_ROWS*E) : nullptr;
    for (int t = 0; t < n; t++) {
        gpt2_decode_step_cuda(be, w, kv, ids[t], t, s, d_logits_all + (size_t)t*V, step_caps);
        if (d_caps) for (int r = 0; r < GPT2_DECODE_CAPS_ROWS; r++)
            CUDA_CHECK(cudaMemcpy(d_caps + ((size_t)r*n + t)*E, step_caps + (size_t)r*E,
                                  E*sizeof(half), cudaMemcpyDeviceToDevice));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    if (step_caps) cudaFree(step_caps);
}

int main(int argc, char **argv) {
    const char *wpath = argc>1? argv[1] : "weights/gpt2_124m_fp32.bin";
    const char *refdir= argc>2? argv[2] : "refdumps/fp16";
    const char *meta  = argc>3? argv[3] : "refdumps/meta.json";
    char path[512];

    const GPT2Backend *be = gpt2_backend_by_name(getenv("GPT2_BACKEND"));
    printf("==== STAGE 5 KV-cache decode gate: %s backend vs HF fp16 oracle (QUALITY_GATES §1 + A1) ====\n", be->name);
    printf("decode path: %s\n", be->gemv_q ? "INT8 M=1 GEMV (blocks) + fp16 M=1 GEMV (kept-fp16 head)"
                              : be->gemv   ? "fp16 M=1 GEMV"
                                           : "FALLBACK: this backend's M=1 MATMUL (no GEMV) -- director-map row 2 probe");

    GPT2Weights wcpu;
    if (gpt2_load_weights(wpath,&wcpu)) { fprintf(stderr,"weight load failed\n"); return 1; }
    GPT2WeightsGPU w; gpt2_upload_fp16(&wcpu,&w);
    gpt2_free_weights(&wcpu);
    GPT2QWeightsGPU qw;
    if (gpt2_quant_attach_if_needed(be,&w,&qw)) { fprintf(stderr,"int8 weight load failed\n"); return 1; }

    GPT2KVCache kv;
    if (gpt2_kv_alloc(&kv, GPT2_N_CTX)) return 1;

    char *mbuf = read_text(meta);
    static int prompt_ids[GPT2_N_CTX], eval_ids[GPT2_N_CTX];
    int Tp = parse_int_array(mbuf,"\"prompt_ids\"",prompt_ids,GPT2_N_CTX);
    int Ne = parse_int_array(mbuf,"\"eval_ids\"",eval_ids,GPT2_N_CTX);
    free(mbuf);
    printf("[meta] prompt=%d eval=%d\n\n", Tp, Ne);

    GPT2ScratchGPU s; gpt2_scratch_alloc(&s, GPT2_N_CTX);
    int *d_ids = dmalloc<int>(GPT2_N_CTX);
    int passEQ=1, passA=1, passB=1, passC=1;

    // ================= [EQ] cached decode vs no-KV recompute, position by position =================
    printf("---- [EQ] cached decode  vs  no-KV recompute  (must agree within fp16 noise) ----\n");
    {
        const int n = Tp;                                  // the frozen 28-token prompt
        half *d_kv  = dmalloc<half>((size_t)n*V);
        half *d_rec = dmalloc<half>((size_t)n*V);
        kv_decode_seq(be,&w,&kv,prompt_ids,n,&s,d_kv,nullptr);
        h2d(d_ids,prompt_ids,n);
        GPT2CapsGPU c; memset(&c,0,sizeof(c)); c.logits=d_rec; c.logits_all=1;
        gpt2_forward_cuda(be,&w,d_ids,n,&s,&c);            // caps.kv == NULL -> plain recompute
        CUDA_CHECK(cudaDeviceSynchronize());
        float *A=dl_f16(d_kv,(size_t)n*V), *B=dl_f16(d_rec,(size_t)n*V);
        double mx=0, maxkl=0; int agree=0;
        for(int t=0;t<n;t++){ const float*a=A+(size_t)t*V,*b=B+(size_t)t*V;
            for(int v=0;v<V;v++){ double d=fabs((double)a[v]-(double)b[v]); if(d>mx)mx=d; }
            double kl=kl_row(b,a,V); if(kl>maxkl)maxkl=kl;
            if(argmax(a,V)==argmax(b,V)) agree++; }
        // fp16 ulp at GPT-2 logit magnitude (|logit| ~ 100-300) is 0.06-0.25; a few ulp is noise.
        printf("  max |logit_kv - logit_recompute| = %.4f   (fp16 ulp @|logit|=128 is 0.125)\n", mx);
        printf("  max KL(recompute || kv)          = %.3e   (threshold < 0.02)  %s\n", maxkl, maxkl<0.02?"OK":"** FAIL **");
        printf("  argmax agreement                 = %d/%d\n", agree, n);
        if(maxkl>=0.02) passEQ=0;
        free(A); free(B); cudaFree(d_kv); cudaFree(d_rec);
    }
    printf("  => [EQ]: %s\n\n", passEQ?"PASS":"FAIL");

    // ================= gate (a): per-layer rel_err of a DECODE-ONLY run =================
    printf("---- gate (a): per-layer rel_err, 28 DECODE steps from pos 0 (cache built by the M=1 path) ----\n");
    {
        const int n = Tp;
        half *d_lg  = dmalloc<half>((size_t)n*V);
        half *d_caps= dmalloc<half>((size_t)GPT2_DECODE_CAPS_ROWS*n*E);
        kv_decode_seq(be,&w,&kv,prompt_ids,n,&s,d_lg,d_caps);
        const double THR=1e-2;
        auto cmp=[&](const char*name,int row){
            snprintf(path,sizeof(path),"%s/%s.bin",refdir,name);
            float*m=dl_f16(d_caps+(size_t)row*n*E,(size_t)n*E), *r=load_f16(path,(size_t)n*E);
            double e=rel_err(m,r,(size_t)n*E);
            printf("  %-10s rel_err = %.3e  %s\n",name,e,e<=THR?"OK":"** FAIL (bug localized to this layer) **");
            if(e>THR) passA=0; free(m); free(r); };
        cmp("embed", NL+1);
        for(int L=0;L<NL;L++){ char nm[16]; snprintf(nm,sizeof(nm),"block_%d",L); cmp(nm,L); }
        cmp("final_ln", NL);
        cudaFree(d_lg); cudaFree(d_caps);
    }
    printf("  => gate (a): %s\n\n", passA?"PASS":"FAIL");

    // ================= gate (b): greedy N=128, teacher-forced, A1 margin rule =================
    printf("---- gate (b): greedy N=128 teacher-forced via KV decode (bug iff margin >= 3*fp16_ulp — A1) ----\n");
    {
        snprintf(path,sizeof(path),"%s/greedy_ids.bin",refdir);
        FILE*f=fopen(path,"rb"); fseek(f,0,SEEK_END); int Ng=(int)(ftell(f)/4); fclose(f);
        int32_t *gids=load_i32(path,Ng);
        snprintf(path,sizeof(path),"%s/greedy_margin.bin",refdir); float *gmar=load_f32(path,Ng);
        int slen=Tp+Ng-1;
        static int seq[GPT2_N_CTX]; memcpy(seq,prompt_ids,(size_t)Tp*sizeof(int));
        for(int t=0;t<Ng-1;t++) seq[Tp+t]=gids[t];
        half *d_lg=dmalloc<half>((size_t)slen*V);
        kv_decode_seq(be,&w,&kv,seq,slen,&s,d_lg,nullptr);
        float *lg=dl_f16(d_lg,(size_t)slen*V);
        int match=0,tol=0,bug=0;
        for(int t=0;t<Ng;t++){
            const float*row=lg+(size_t)(Tp-1+t)*V;
            int pred=argmax(row,V);
            if(pred==gids[t]){match++;continue;}
            float mag=fabsf(row[pred]), thr=3.0f*fp16_ulp(mag);
            if(gmar[t]<thr){ tol++;
                printf("    pos %3d: pred %d != ref %d (margin %.4f < 3*ulp %.4f @|logit|~%.0f -> tolerated near-tie)\n",t,pred,gids[t],gmar[t],thr,mag); }
            else { bug++; passB=0;
                printf("    pos %3d: pred %d != ref %d (margin %.4f >= 3*ulp %.4f @|logit|~%.0f -> ** BUG **)\n",t,pred,gids[t],gmar[t],thr,mag); }
        }
        printf("  teacher-forced (A1): %d/%d match, %d tolerated near-tie, %d bug\n",match,Ng,tol,bug);
        free(lg); free(gids); free(gmar); cudaFree(d_lg);
    }
    printf("  => gate (b): %s\n\n", passB?"PASS":"FAIL");

    // ================= gate (c): 512 eval positions, decoded ONE TOKEN AT A TIME =================
    printf("---- gate (c): eval distribution, 512 sequential decode steps (PRIMARY max KL < 0.02 — A1) ----\n");
    {
        half *d_lg=dmalloc<half>((size_t)Ne*V);
        kv_decode_seq(be,&w,&kv,eval_ids,Ne,&s,d_lg,nullptr);
        float *mine=dl_f16(d_lg,(size_t)Ne*V);
        snprintf(path,sizeof(path),"%s/eval_logits.bin",refdir); float *r=load_f16(path,(size_t)Ne*V);
        int agree=0; double maxkl=0;
        for(int q=0;q<Ne;q++){ const float*ro=r+(size_t)q*V,*oo=mine+(size_t)q*V;
            if(argmax(ro,V)==argmax(oo,V)) agree++;
            double kl=kl_row(ro,oo,V); if(kl>maxkl)maxkl=kl; }
        printf("  [PRIMARY] max KL(ref||ours) = %.3e   (threshold < 0.02)  %s\n",maxkl,maxkl<0.02?"OK":"** FAIL **");
        printf("  [diag]    top-1 agreement  = %d/%d (%.2f%%)\n",agree,Ne,100.0*agree/Ne);
        if(maxkl>=0.02) passC=0;
        free(mine); free(r); cudaFree(d_lg);
    }
    printf("  => gate (c): %s\n\n", passC?"PASS":"FAIL");

    int ok = passEQ&&passA&&passB&&passC;
    printf("==== VERDICT [%s, KV-cache decode]: EQ=%s (a)=%s (b)=%s (c)=%s -> %s ====\n",
           be->name,passEQ?"PASS":"FAIL",passA?"PASS":"FAIL",passB?"PASS":"FAIL",passC?"PASS":"FAIL",
           ok?"ALL PASS":"FAILED");
    gpt2_scratch_free(&s); cudaFree(d_ids); gpt2_kv_free(&kv); gpt2_free_gpu(&w); gpt2_quant_free(&qw);
    return ok?0:1;
}
