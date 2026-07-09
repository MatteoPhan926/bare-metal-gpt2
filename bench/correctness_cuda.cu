// correctness_cuda.cu — STAGE 1 GPU gate harness (QUALITY_GATES §1) for the naive fp16 backend.
//
// Same structure as bench/correctness.c, but runs the CUDA forward and compares vs the HF *fp16*
// oracle (refdumps/fp16) at the fp16 tolerance 1e-2/layer. Then prefill/decode timing via the
// SYNC-ENFORCING helper in common.cuh, with the first GPU decode number checked against the
// measured bug-line (ROOFLINE §6: >1032 tok/s theo = measurement bug; ~941 copy / ~1004 read).
//
// Prefill and decode reported SEPARATELY. This stage establishes the GPU baseline only — no
// optimization payoff is claimed. Run from repo root:
//   correctness_cuda.exe [all|quick] [weights.bin] [refdir=refdumps/fp16] [meta.json]

#include "kernels.cuh"
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

// ---------- host IO / math helpers ----------
static char *read_text(const char *path) {
    FILE *f = fopen(path, "rb"); if (!f) { fprintf(stderr,"[gate] cannot open %s\n",path); exit(1); }
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
    char *b=(char*)malloc(n+1); if(fread(b,1,n,f)!=(size_t)n){fprintf(stderr,"short read %s\n",path);exit(1);}
    b[n]=0; fclose(f); return b;
}
static int parse_int_array(const char *buf, const char *key, int *out, int maxn) {
    const char *p=strstr(buf,key); if(!p){fprintf(stderr,"[gate] key %s missing\n",key);exit(1);}
    p=strchr(p,'['); if(!p){fprintf(stderr,"[gate] no [ after %s\n",key);exit(1);} p++;
    int n=0;
    while(*p&&*p!=']'){
        while(*p&&*p!=']'&&*p!='-'&&(*p<'0'||*p>'9')) p++;
        if(!*p||*p==']') break;
        char*end; long v=strtol(p,&end,10); if(end==p){p++;continue;}
        if(n<maxn) out[n]=(int)v; n++; p=end;
    }
    return n;
}
static float *load_f16(const char *path, size_t n) {           // fp16 file -> host fp32
    FILE *f=fopen(path,"rb"); if(!f){fprintf(stderr,"[gate] cannot open %s\n",path);exit(1);}
    std::vector<half> raw(n);
    if(fread(raw.data(),sizeof(half),n,f)!=n){fprintf(stderr,"[gate] short read %s (want %zu f16)\n",path,n);exit(1);}
    fclose(f);
    float *o=(float*)malloc(n*sizeof(float));
    for(size_t i=0;i<n;i++) o[i]=__half2float(raw[i]);
    return o;
}
static int32_t *load_i32(const char *path, size_t n) {
    FILE *f=fopen(path,"rb"); if(!f){fprintf(stderr,"[gate] cannot open %s\n",path);exit(1);}
    int32_t *p=(int32_t*)malloc(n*sizeof(int32_t));
    if(fread(p,sizeof(int32_t),n,f)!=n){fprintf(stderr,"[gate] short read %s\n",path);exit(1);} fclose(f); return p;
}
static float *load_f32(const char *path, size_t n) {
    FILE *f=fopen(path,"rb"); if(!f){fprintf(stderr,"[gate] cannot open %s\n",path);exit(1);}
    float *p=(float*)malloc(n*sizeof(float));
    if(fread(p,sizeof(float),n,f)!=n){fprintf(stderr,"[gate] short read %s\n",path);exit(1);} fclose(f); return p;
}
static float *dl_f16(const half *d, size_t n) {                // device fp16 -> host fp32
    std::vector<half> h(n); d2h(h.data(), d, n);
    float *o=(float*)malloc(n*sizeof(float));
    for(size_t i=0;i<n;i++) o[i]=__half2float(h[i]);
    return o;
}
static double rel_err(const float *a, const float *b, size_t n) {
    double md=0,mr=0;
    for(size_t i=0;i<n;i++){ double d=fabs((double)a[i]-(double)b[i]); if(d>md)md=d;
                             double r=fabs((double)b[i]); if(r>mr)mr=r; }
    return md/(mr+1e-9);
}
static int argmax(const float *v,int n){ int m=0; for(int i=1;i<n;i++) if(v[i]>v[m]) m=i; return m; }
static double kl_row(const float *ref,const float *ours,int n){ // KL(softmax(ref)||softmax(ours))
    double mr=-1e300,mo=-1e300;
    for(int i=0;i<n;i++){ if(ref[i]>mr)mr=ref[i]; if(ours[i]>mo)mo=ours[i]; }
    double sr=0,so=0; for(int i=0;i<n;i++){ sr+=exp((double)ref[i]-mr); so+=exp((double)ours[i]-mo); }
    double lsr=mr+log(sr),lso=mo+log(so),kl=0;
    for(int i=0;i<n;i++){ double lpr=(double)ref[i]-lsr,lpo=(double)ours[i]-lso; kl+=exp(lpr)*(lpr-lpo); }
    return kl;
}
static int cmp_d(const void*a,const void*b){ double x=*(const double*)a,y=*(const double*)b; return (x>y)-(x<y); }
static double median_d(double*v,int n){ qsort(v,n,sizeof(double),cmp_d); return n%2? v[n/2]:0.5*(v[n/2-1]+v[n/2]); }
// fp16 ulp at magnitude |x| (QUALITY_GATES Amendment A1): normalized fp16 has 10 mantissa bits.
static float fp16_ulp(float x){ float a=fabsf(x); if(a<6.104e-5f) return 5.96e-8f; int e; frexpf(a,&e); return ldexpf(1.0f, e-11); }
// prefill forward FLOPs at seq len T: 12 layers × [QKV, attn-proj, FC, FFN-proj] matmuls + causal
// attention (QK^T + softmax·V) + last-token logits head (logits_all=0 in the timed prefill). Divisor
// for GFLOP/s and the compute bug-line. (naive@512 → ~109 GFLOP/s ≈ 0.35% of 31.5 TFLOP/s, as recorded.)
static double prefill_flops(int T){                                // d-prefixed: E/V are #defines above
    double dE=GPT2_N_EMBD, dQ=GPT2_QKV_DIM, dF=GPT2_FFN_DIM, dV=GPT2_VOCAB, dL=GPT2_N_LAYER;
    double mm   = 2.0 * dL * (dE*dQ + dE*dE + dE*dF + dF*dE) * (double)T;   // 4 matmuls / layer / token
    double attn = 2.0 * dL * dE * (double)T * (T + 1.0);                    // causal QK^T + softmax·V
    double head = 2.0 * dV * dE;                                            // logits for the last token only
    return mm + attn + head;
}

int main(int argc, char **argv) {
    const char *mode  = argc>1? argv[1] : "all";
    const char *wpath = argc>2? argv[2] : "weights/gpt2_124m_fp32.bin";
    const char *refdir= argc>3? argv[3] : "refdumps/fp16";
    const char *meta  = argc>4? argv[4] : "refdumps/meta.json";
    int quick = strcmp(mode,"quick")==0;
    char path[512];

    cudaDeviceProp p; CUDA_CHECK(cudaGetDeviceProperties(&p,0));
    // backend selectable via env GPT2_BACKEND=naive|tiled|flash (default naive); positional args unchanged.
    const GPT2Backend *be = gpt2_backend_by_name(getenv("GPT2_BACKEND"));
    printf("==== GPU gate: %s fp16 backend vs HF fp16 oracle (QUALITY_GATES §1 + A1) ====\n", be->name);
    printf("device: %s  sm_%d%d  %d SM  L2=%dMiB  smclk(max)=%.0fMHz memclk=%.0fMHz\n",
           p.name,p.major,p.minor,p.multiProcessorCount,p.l2CacheSize/(1024*1024),
           p.clockRate/1000.0,p.memoryClockRate/1000.0);
    printf("backend: %s | fp16 storage, fp32 accumulation | weights = __float2half(fp32 master)\n\n",
           be->name);

    // ---- weights: load fp32 master, upload fp16, free host fp32 ----
    GPT2Weights wcpu;
    if (gpt2_load_weights(wpath,&wcpu)) { fprintf(stderr,"weight load failed\n"); return 1; }
    GPT2WeightsGPU w; gpt2_upload_fp16(&wcpu,&w);
    gpt2_free_weights(&wcpu);
    // STAGE 4: attach packed INT8 weights iff the backend has a quantized matmul (else a no-op).
    // Fails loudly rather than silently running fp16 under an "int8" label.
    GPT2QWeightsGPU qw;
    if (gpt2_quant_attach_if_needed(be,&w,&qw)) { fprintf(stderr,"int8 weight load failed\n"); return 1; }
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---- token ids ----
    char *mbuf=read_text(meta);
    static int prompt_ids[GPT2_N_CTX], eval_ids[GPT2_N_CTX];
    int Tp=parse_int_array(mbuf,"\"prompt_ids\"",prompt_ids,GPT2_N_CTX);
    int Ne=parse_int_array(mbuf,"\"eval_ids\"",eval_ids,GPT2_N_CTX);
    free(mbuf);
    printf("[meta] prompt=%d tokens  eval=%d tokens\n\n",Tp,Ne);

    GPT2ScratchGPU s; gpt2_scratch_alloc(&s, GPT2_N_CTX);
    int *d_ids = dmalloc<int>(GPT2_N_CTX);
    int passA=1,passB=1,passC=1;

    // =================== GATE (a): per-layer rel_err (threshold fp16 <= 1e-2) ===================
    printf("---- gate (a): per-layer rel_err  (threshold fp16 <= 1e-2) ----\n");
    {
        GPT2CapsGPU c; memset(&c,0,sizeof(c));
        c.embed=dmalloc<half>((size_t)Tp*E); c.blocks=dmalloc<half>((size_t)NL*Tp*E); c.final_ln=dmalloc<half>((size_t)Tp*E);
        h2d(d_ids, prompt_ids, Tp);
        gpt2_forward_cuda(be,&w,d_ids,Tp,&s,&c);
        CUDA_CHECK(cudaDeviceSynchronize());
        const double THR=1e-2;
        snprintf(path,sizeof(path),"%s/embed.bin",refdir);
        { float*m=dl_f16(c.embed,(size_t)Tp*E),*r=load_f16(path,(size_t)Tp*E); double e=rel_err(m,r,(size_t)Tp*E);
          printf("  %-10s rel_err = %.3e  %s\n","embed",e,e<=THR?"OK":"** FAIL **"); if(e>THR)passA=0; free(m);free(r); }
        for(int L=0;L<NL;L++){
            snprintf(path,sizeof(path),"%s/block_%d.bin",refdir,L);
            float*m=dl_f16(c.blocks+(size_t)L*Tp*E,(size_t)Tp*E),*r=load_f16(path,(size_t)Tp*E);
            double e=rel_err(m,r,(size_t)Tp*E); char nm[16]; snprintf(nm,sizeof(nm),"block_%d",L);
            printf("  %-10s rel_err = %.3e  %s\n",nm,e,e<=THR?"OK":"** FAIL (bug localized to this layer) **");
            if(e>THR)passA=0; free(m);free(r);
        }
        snprintf(path,sizeof(path),"%s/final_ln.bin",refdir);
        { float*m=dl_f16(c.final_ln,(size_t)Tp*E),*r=load_f16(path,(size_t)Tp*E); double e=rel_err(m,r,(size_t)Tp*E);
          printf("  %-10s rel_err = %.3e  %s\n","final_ln",e,e<=THR?"OK":"** FAIL **"); if(e>THR)passA=0; free(m);free(r); }
        cudaFree(c.embed); cudaFree(c.blocks); cudaFree(c.final_ln);
    }
    printf("  => gate (a): %s\n\n", passA?"PASS":"FAIL");

    if (quick) { printf("[quick] skipping gates (b),(c) and timing.\n"); return passA?0:1; }

    // =================== GATE (b): greedy token match, teacher-forced ===================
    printf("---- gate (b): greedy token match, N=128 (teacher-forced; bug iff margin >= 3*fp16_ulp — Amendment A1) ----\n");
    {
        snprintf(path,sizeof(path),"%s/greedy_ids.bin",refdir);
        FILE*f=fopen(path,"rb"); fseek(f,0,SEEK_END); int Ng=(int)(ftell(f)/4); fclose(f);
        int32_t *gids=load_i32(path,Ng);
        snprintf(path,sizeof(path),"%s/greedy_margin.bin",refdir); float *gmar=load_f32(path,Ng);
        int slen=Tp+Ng-1;
        static int seq[GPT2_N_CTX]; memcpy(seq,prompt_ids,(size_t)Tp*sizeof(int));
        for(int t=0;t<Ng-1;t++) seq[Tp+t]=gids[t];
        h2d(d_ids,seq,slen);
        GPT2CapsGPU c; memset(&c,0,sizeof(c)); c.logits=dmalloc<half>((size_t)slen*V); c.logits_all=1;
        gpt2_forward_cuda(be,&w,d_ids,slen,&s,&c);
        CUDA_CHECK(cudaDeviceSynchronize());
        float *lg=dl_f16(c.logits,(size_t)slen*V);
        int match=0,tol=0,bug=0;
        for(int t=0;t<Ng;t++){
            const float *row=lg+(size_t)(Tp-1+t)*V;
            int pred=argmax(row,V);
            if(pred==gids[t]){ match++; continue; }
            float mag=fabsf(row[pred]), thr=3.0f*fp16_ulp(mag);   // Amendment A1: bug iff ref margin >= 3*fp16_ulp(|logit|)
            if(gmar[t]<thr){ tol++;
                printf("    pos %3d: pred %d != ref %d  (margin %.4f < 3*ulp %.4f @|logit|~%.0f -> tolerated near-tie)\n",t,pred,gids[t],gmar[t],thr,mag); }
            else { bug++; passB=0;
                printf("    pos %3d: pred %d != ref %d  (margin %.4f >= 3*ulp %.4f @|logit|~%.0f -> ** BUG **)\n",t,pred,gids[t],gmar[t],thr,mag); }
        }
        printf("  teacher-forced (A1 margin rule): %d/%d match, %d tolerated near-tie, %d bug\n",match,Ng,tol,bug);
        free(lg); free(gids); free(gmar); cudaFree(c.logits);
    }
    printf("  => gate (b): %s\n\n", passB?"PASS":"FAIL");

    // =================== GATE (c): eval-window distribution agreement ===================
    printf("---- gate (c): eval distribution agreement  (PRIMARY max KL < 0.02; top-1 = diagnostic — Amendment A1) ----\n");
    {
        h2d(d_ids,eval_ids,Ne);
        GPT2CapsGPU c; memset(&c,0,sizeof(c)); c.logits=dmalloc<half>((size_t)Ne*V); c.logits_all=1;
        gpt2_forward_cuda(be,&w,d_ids,Ne,&s,&c);
        CUDA_CHECK(cudaDeviceSynchronize());
        float *mine=dl_f16(c.logits,(size_t)Ne*V);
        snprintf(path,sizeof(path),"%s/eval_logits.bin",refdir); float *r=load_f16(path,(size_t)Ne*V);
        int agree=0; double maxkl=0;
        for(int q=0;q<Ne;q++){ const float*ro=r+(size_t)q*V,*oo=mine+(size_t)q*V;
            if(argmax(ro,V)==argmax(oo,V)) agree++;
            double kl=kl_row(ro,oo,V); if(kl>maxkl)maxkl=kl; }
        double ap=100.0*agree/Ne;
        printf("  [PRIMARY] max KL(ref||ours) = %.3e   (threshold < 0.02)   %s\n", maxkl, maxkl<0.02?"OK":"** FAIL **");
        printf("  [diag]    top-1 agreement  = %d/%d (%.2f%%)   (reported, not pass/fail per Amendment A1)\n", agree,Ne,ap);
        if(maxkl>=0.02) passC=0;
        free(mine); free(r); cudaFree(c.logits);
    }
    printf("  => gate (c): %s\n\n", passC?"PASS":"FAIL");

    // =================== GPU baseline timing (prefill and decode SEPARATELY) ===================
    printf("---- GPU timing (%s backend; sync-enforced timer; median + min-max) ----\n", be->name);
    half *d_logits = dmalloc<half>(V);
    // prefill @P : full forward, last-position logits only
    int Ps[2]={128,512};
    for(int pi=0;pi<2;pi++){
        int P=Ps[pi]; h2d(d_ids,eval_ids,P);
        GPT2CapsGPU cp; memset(&cp,0,sizeof(cp)); cp.logits=d_logits; cp.logits_all=0;
        BenchStats st=cuda_bench([&](){ gpt2_forward_cuda(be,&w,d_ids,P,&s,&cp); },10,30);
        double gf = prefill_flops(P)/(st.median*1e-3)/1e9;                 // GFLOP/s (whole forward)
        const double TENSOR_PEAK=31500.0;                                  // 31.5 TFLOP/s microbench (TENSOR-core, fp32-acc)
        printf("  prefill @P=%-3d : %.3f ms median  (%.3f - %.3f, N=%d)  %.1f GFLOP/s = %.2f%% of 31.5TF tensor peak%s\n",
               P,st.median,st.min,st.max,st.n, gf, 100.0*gf/TENSOR_PEAK,
               gf>TENSOR_PEAK ? "  *** ABOVE COMPUTE CEILING -> MEASUREMENT BUG ***":"");
    }
    // no-KV decode: free-run greedy from the prompt, time each full-recompute step
    {
        snprintf(path,sizeof(path),"%s/greedy_ids.bin",refdir);
        FILE*f=fopen(path,"rb"); fseek(f,0,SEEK_END); int Ng=(int)(ftell(f)/4); fclose(f); int32_t*gids=load_i32(path,Ng);
        int Ndec=Ng;                                   // 128 steps, ctx Tp -> Tp+Ndec
        std::vector<int> seq(prompt_ids,prompt_ids+Tp);
        GPT2CapsGPU cp; memset(&cp,0,sizeof(cp)); cp.logits=d_logits; cp.logits_all=0;
        std::vector<double> stepms; int lead=0,diverged=-1;
        for(int step=0;step<Ndec;step++){
            int slen=Tp+step; h2d(d_ids,seq.data(),(size_t)slen);
            double ms=cuda_time_once_ms([&](){ gpt2_forward_cuda(be,&w,d_ids,slen,&s,&cp); });
            std::vector<half> lg(V); d2h(lg.data(),d_logits,V);
            int tok=0; float best=-1e30f; for(int v=0;v<V;v++){ float x=__half2float(lg[v]); if(x>best){best=x;tok=v;} }
            if(step>0) stepms.push_back(ms);           // step 0 includes any first-launch cost
            if(diverged<0){ if(tok==gids[step]) lead=step+1; else diverged=step; }
            seq.push_back(tok);
        }
        int lo=5, cnt=(int)stepms.size()-lo; std::vector<double> tail(stepms.begin()+lo,stepms.end());
        double med=median_d(tail.data(),cnt);
        std::sort(tail.begin(),tail.end());
        double tps=1000.0/med;
        printf("  no-KV decode  : %.3f ms/tok median  (%.3f – %.3f, N=%d) @ ctx %d->%d  -> %.1f tok/s\n",
               med,tail.front(),tail.back(),cnt,Tp+lo+1,Tp+Ndec-1,tps);
        printf("  free-run greedy vs fp16 reference: %d/%d leading tokens match%s\n",
               lead,Ndec,diverged<0?" (full)":"");
        // BUG-LINE CHECK (ROOFLINE §6): a decode number above the theoretical ceiling = measurement bug.
        // The ceiling = weight_bytes / BW, so it is derived from the bytes this build ACTUALLY streams
        // (fp16 248 MB; pure INT8 123.5 MB; a mixed kill-test build lands in between) — never hardcoded.
        const double MB = w.q ? w.q->streamed_bytes/1e6 : 2.0*124.44e6/1e6;   // fp16: 124.44M params x 2 B
        const double C_COPY=233.4e3/MB, C_READ=248.9e3/MB, C_THEO=256.0e3/MB;
        printf("  bug-line check (%.1f MB streamed): decode %.1f tok/s vs ceilings [copy %.0f / read %.0f / theo %.0f]\n",
               MB,tps,C_COPY,C_READ,C_THEO);
        if(tps>C_THEO)      printf("  *** ABOVE THEORETICAL CEILING -> MEASUREMENT BUG (likely missing sync). Fix it; do not publish it. ***\n");
        else if(tps>C_READ) printf("  *** above achieved-read BW ceiling -> suspicious; investigate. ***\n");
        else                printf("  OK: below the achieved-BW ceiling (as expected for this backend).\n");
        free(gids);
    }
    printf("\n");

    int ok=passA&&passB&&passC;
    printf("==== VERDICT [%s backend]: gate(a)=%s gate(b)=%s gate(c)=%s -> %s ====\n",
           be->name,passA?"PASS":"FAIL",passB?"PASS":"FAIL",passC?"PASS":"FAIL",ok?"ALL PASS":"FAILED");
    gpt2_scratch_free(&s); cudaFree(d_ids); cudaFree(d_logits); gpt2_free_gpu(&w);
    return ok?0:1;
}
