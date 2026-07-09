// bench_decode.cu — STAGE 5 speed harness: the FIRST true (M=1) decode this engine has ever had.
//
// Everything here answers one of three questions, and each is a pre-registered director-map row:
//   (1) KV vs no-KV recompute (fp16)     -> the KV-cache decode win. Expected LARGE.
//   (2) naive vs tiled MATMUL at M=1     -> row 2, "tiled GEMM -> decode ~flat". First falsifiable here.
//   (3) fp16 GEMV vs INT8 GEMV           -> row 4, "INT8 -> decode high".        First falsifiable here.
//
// Honest comparison rule: both sides must answer "what does it cost to produce ONE MORE token at a
// context of `ctx`?".  KV  = one decode step with a cache of length ctx.
//                      no-KV = one FULL forward over ctx+1 tokens (that is literally what the pre-Stage-5
//                              harness did per token). Timing only the KV step against a *prefill* would
//                              be the classic "count cached tokens as generated" cheat (BENCH_PROTOCOL §7).
//
// Interleaved A/B so both sides see the same thermal state within microseconds; sync-enforced timer;
// median + [min,max]; never best-of-N.
//
// Run from repo root:  bench_decode.exe [iters]     env: GPT2_INT8_WEIGHTS (defaults to the gated build)

#include "kernels.cuh"
#include "kvcache.cuh"
#include "common.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <functional>

#define E GPT2_N_EMBD
#define V GPT2_VOCAB
#define NL GPT2_N_LAYER

static char *read_text(const char*p){ FILE*f=fopen(p,"rb"); if(!f){fprintf(stderr,"open %s\n",p);exit(1);}
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET); char*b=(char*)malloc(n+1);
    if(fread(b,1,n,f)!=(size_t)n)exit(1); b[n]=0; fclose(f); return b; }
static int parse_int_array(const char*buf,const char*key,int*out,int maxn){
    const char*p=strstr(buf,key); if(!p)exit(1); p=strchr(p,'['); p++; int n=0;
    while(*p&&*p!=']'){ while(*p&&*p!=']'&&*p!='-'&&(*p<'0'||*p>'9'))p++; if(!*p||*p==']')break;
        char*e; long v=strtol(p,&e,10); if(e==p){p++;continue;} if(n<maxn)out[n]=(int)v; n++; p=e; }
    return n; }

struct Stat { double med, lo, hi; };
static Stat stat_of(std::vector<double> v){ std::sort(v.begin(),v.end()); size_t n=v.size();
    Stat s; s.lo=v.front(); s.hi=v.back(); s.med = n%2? v[n/2] : 0.5*(v[n/2-1]+v[n/2]); return s; }

static void report(const char *what,const char *na,const char *nb,Stat a,Stat b){
    double ratio=a.med/b.med;
    bool overlap = !(b.hi<a.lo || a.hi<b.lo);
    printf("  %-40s %-7s %9.4f ms [%.4f-%.4f]  (%7.1f tok/s)\n", what, na, a.med,a.lo,a.hi, 1000.0/a.med);
    printf("  %-40s %-7s %9.4f ms [%.4f-%.4f]  (%7.1f tok/s)\n", "",   nb, b.med,b.lo,b.hi, 1000.0/b.med);
    printf("  %-40s speedup B vs A: %.3fx   %s\n\n", "", ratio,
           overlap ? "** [min,max] OVERLAP -> NOT above noise **"
                   : (ratio>1.0 ? "disjoint -> above noise (B faster)" : "disjoint -> above noise (B SLOWER)"));
}

// weight bytes each build streams per decode token (Stage 4's kill-test build keeps the head fp16)
static double weight_MB(const GPT2WeightsGPU &w){
    return w.q ? w.q->streamed_bytes/1e6 : 124439808.0*2/1e6;
}

int main(int argc,char**argv){
    const int iters = argc>1? atoi(argv[1]) : 40;
    cudaDeviceProp p; CUDA_CHECK(cudaGetDeviceProperties(&p,0));
    printf("==== STAGE 5: TRUE (M=1) DECODE — the first one this engine has had ====\n");
    printf("device %s  sm_%d%d  %d SM   iters=%d, interleaved A/B, sync-enforced, median [min-max]\n\n",
           p.name,p.major,p.minor,p.multiProcessorCount,iters);

    GPT2Weights wcpu;
    if(gpt2_load_weights("weights/gpt2_124m_fp32.bin",&wcpu)){ fprintf(stderr,"weight load failed\n"); return 1; }
    GPT2WeightsGPU w;  gpt2_upload_fp16(&wcpu,&w);   // fp16 blob (used by naive/tiled/flash/gemv)
    gpt2_free_weights(&wcpu);

    const GPT2Backend *NAIVE=gpt2_backend_by_name("naive");
    const GPT2Backend *TILED=gpt2_backend_by_name("tiled");
    const GPT2Backend *FLASH=gpt2_backend_by_name("flash");
    const GPT2Backend *GEMV =gpt2_backend_by_name("gemv");
    const GPT2Backend *INT8 =gpt2_backend_by_name("int8");

    GPT2QWeightsGPU qw;                                  // attach once; fp16 backends ignore it
    if(gpt2_quant_attach_if_needed(INT8,&w,&qw)){ fprintf(stderr,"int8 weight load failed\n"); return 1; }
    const double W_INT8_MB = weight_MB(w);               // with q attached
    const double W_FP16_MB = 124439808.0*2/1e6;
    printf("[bytes] fp16 weights %.1f MB/token   INT8(kill-test) weights %.1f MB/token\n\n", W_FP16_MB, W_INT8_MB);

    GPT2KVCache kv; if(gpt2_kv_alloc(&kv,GPT2_N_CTX)) return 1;
    printf("\n");

    char *mbuf=read_text("refdumps/meta.json");
    static int eval_ids[GPT2_N_CTX];
    parse_int_array(mbuf,"\"eval_ids\"",eval_ids,GPT2_N_CTX); free(mbuf);

    GPT2ScratchGPU s; gpt2_scratch_alloc(&s,GPT2_N_CTX);
    int *d_ids=dmalloc<int>(GPT2_N_CTX); h2d(d_ids,eval_ids,GPT2_N_CTX);
    half *d_logits=dmalloc<half>(V);

    // Fill the cache once with `ctx` real tokens; each timed step then re-decodes position `ctx`.
    auto fill=[&](int ctx){ gpt2_kv_reset(&kv);
        GPT2CapsGPU c; memset(&c,0,sizeof(c));
        gpt2_prefill_fill_cache(FLASH,&w,&kv,d_ids,ctx,&s,&c); CUDA_CHECK(cudaDeviceSynchronize()); };

    auto kv_step=[&](const GPT2Backend*be,int ctx){ kv.len=ctx;
        gpt2_decode_step_cuda(be,&w,&kv,eval_ids[ctx],ctx,&s,d_logits,nullptr); };
    auto recompute_step=[&](const GPT2Backend*be,int ctx){
        GPT2CapsGPU c; memset(&c,0,sizeof(c)); c.logits=d_logits; c.logits_all=0;
        gpt2_forward_cuda(be,&w,d_ids,ctx+1,&s,&c); };

    // sustained warmup to boost-clock steady state
    { cudaEvent_t s0,s1; CUDA_CHECK(cudaEventCreate(&s0)); CUDA_CHECK(cudaEventCreate(&s1));
      fill(512); float wm=0; CUDA_CHECK(cudaEventRecord(s0));
      do { for(int r=0;r<20;r++){ kv_step(GEMV,512); kv_step(INT8,512); }
           CUDA_CHECK(cudaEventRecord(s1)); CUDA_CHECK(cudaEventSynchronize(s1));
           CUDA_CHECK(cudaEventElapsedTime(&wm,s0,s1)); } while(wm<1500.f);
      cudaEventDestroy(s0); cudaEventDestroy(s1); }

    // `batch` timed calls per sample, then divide -> the reported number is still ms PER TOKEN.
    // A 2 ms decode step launches 135 kernels, so a single Windows host hiccup inflates one sample by
    // ~30% and destroys min-max disjointness. Batching amortises host jitter across `batch` steps; it
    // does NOT change what is measured (BENCH_PROTOCOL §6: "spread wide -> increase N, don't pick the
    // good one"). batch=1 for the slow no-KV side, where a step is 110-221 ms and jitter is irrelevant.
    auto ab=[&](const char*what,const char*na,const char*nb,
                std::function<void()> fa, std::function<void()> fb, int batch){
        for(int i=0;i<5;i++){ fa(); fb(); }
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<double> va,vb;
        for(int i=0;i<iters;i++){
            va.push_back(cuda_time_once_ms([&]{ for(int r=0;r<batch;r++) fa(); })/batch);
            vb.push_back(cuda_time_once_ms([&]{ for(int r=0;r<batch;r++) fb(); })/batch);
        }
        CUDA_CHECK(cudaGetLastError());
        report(what,na,nb,stat_of(va),stat_of(vb));
    };

    // ---------------- (1) THE KV-CACHE WIN: KV vs no-KV recompute, fp16 ----------------
    printf("---- (1) KV cache vs no-KV recompute (fp16). Cost of ONE MORE token at ctx. ----\n");
    printf("     A = full forward over ctx+1 tokens (what the pre-Stage-5 harness did per token)\n");
    printf("     B = one M=1 decode step against a cache of length ctx\n\n");
    for(int ctx : {128, 512, 1023}){
        fill(ctx);
        char what[80]; snprintf(what,sizeof(what),"decode @ ctx=%d (fp16)",ctx);
        ab(what,"no-KV","KV",[&]{ recompute_step(FLASH,ctx); },[&]{ kv_step(GEMV,ctx); }, 1);
    }

    // ---------------- (2) DIRECTOR-MAP ROW 2: tiled GEMM at TRUE M=1 ----------------
    printf("---- (2) row 2 probe: naive vs tiled MATMUL at TRUE M=1 (both KV; .gemv==NULL -> M=1 matmul) ----\n");
    for(int ctx : {128, 512}){
        fill(ctx);
        char what[80]; snprintf(what,sizeof(what),"KV decode @ ctx=%d, M=1 matmul",ctx);
        ab(what,"naive","tiled",[&]{ kv_step(NAIVE,ctx); },[&]{ kv_step(TILED,ctx); }, 4);
    }
    printf("---- (2b) the GEMV kernel itself: tiled M=1 matmul vs true M=1 GEMV (both KV, both fp16) ----\n");
    for(int ctx : {128, 512}){
        fill(ctx);
        char what[80]; snprintf(what,sizeof(what),"KV decode @ ctx=%d",ctx);
        ab(what,"tiled","gemv",[&]{ kv_step(TILED,ctx); },[&]{ kv_step(GEMV,ctx); }, 4);
    }

    // ---------------- (3) DIRECTOR-MAP ROW 4: INT8 at TRUE M=1 ----------------
    printf("---- (3) row 4 probe: fp16 GEMV vs INT8 GEMV at TRUE M=1 (both KV). THE INT8 decode payoff. ----\n");
    for(int ctx : {128, 512, 1023}){
        fill(ctx);
        char what[80]; snprintf(what,sizeof(what),"KV decode @ ctx=%d",ctx);
        ab(what,"gemv","int8",[&]{ kv_step(GEMV,ctx); },[&]{ kv_step(INT8,ctx); }, 16);
    }

    // ---------------- ceilings (ROOFLINE §3: decode ceiling degrades with ctx) ----------------
    printf("---- decode vs its ROOFLINE ceiling (weights + KV traffic; copy BW 233.4 GB/s) ----\n");
    printf("  %-6s %-6s %10s %10s %10s %10s\n","backend","ctx","ms/tok","tok/s","ceil tok/s","%% of ceil");
    for(int ctx : {128, 512, 1023}){
        fill(ctx);
        for(int b=0;b<2;b++){
            const GPT2Backend *be = b? INT8 : GEMV;
            double wMB = b? W_INT8_MB : W_FP16_MB;
            for(int i=0;i<5;i++) kv_step(be,ctx);
            CUDA_CHECK(cudaDeviceSynchronize());
            std::vector<double> t; for(int i=0;i<iters;i++) t.push_back(cuda_time_once_ms([&]{ for(int r=0;r<16;r++) kv_step(be,ctx); })/16.0);
            Stat st=stat_of(t);
            double kvMB = gpt2_kv_step_bytes(&kv,ctx)/1e6;
            double ceil_tok = 233.4e3/(wMB+kvMB);
            double tps=1000.0/st.med;
            printf("  %-6s %-6d %10.4f %10.1f %10.0f %9.1f%%  (weights %.1f + KV %.1f MB)%s\n",
                   be->name,ctx,st.med,tps,ceil_tok,100.0*tps/ceil_tok,wMB,kvMB,
                   tps>ceil_tok? "  *** ABOVE CEILING -> MEASUREMENT BUG ***":"");
        }
    }

    gpt2_scratch_free(&s); cudaFree(d_ids); cudaFree(d_logits);
    gpt2_kv_free(&kv); gpt2_free_gpu(&w); gpt2_quant_free(&qw);
    return 0;
}
