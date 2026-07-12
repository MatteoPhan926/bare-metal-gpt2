// profile_decode.cu — STAGE 5 measure-before-optimize (DESIGN.md §9.1).
//
// The A/B said INT8 gives only ~1.1x at true M=1, NOT above noise -- even though it streams 162.5 MB
// against fp16's 248.9 MB (1.53x fewer weight bytes). Either (i) the GEMV is not weight-traffic-bound,
// or (ii) something else dominates the step. Guessing is forbidden; this profiles it.
//
// Two instruments:
//   [A] per-op CUDA-event attribution of ONE decode step, with the sum-of-parts vs whole validity check
//       (per-op syncs serialize, so SUM runs slightly high -> use it for RELATIVE SHARE only, never as
//       a latency). Same guard as bench/profile_forward.cu.
//   [B] the isolated GEMVs, fp16 vs INT8, with ACHIEVED weight bandwidth. This is the direct test of
//       "does the M=1 GEMV express the byte halving?" -- if a GEMV is at ~BW, halving bytes must ~halve
//       its time; if it is far below BW, the byte saving cannot appear.
//
// Run from repo root:  profile_decode.exe [ctx]

#include "kernels.cuh"
#include "kvcache.cuh"
#include "common.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>

#define E GPT2_N_EMBD
#define V GPT2_VOCAB
#define NL GPT2_N_LAYER

enum { OP_EMBED=0, OP_LN, OP_QKV, OP_APPEND, OP_ATTN, OP_APROJ, OP_ADD, OP_FC, OP_GELU, OP_FPROJ, OP_LOGITS, N_OP };
static const char *OP_NAME[N_OP] = { "embed","layernorm","gemv_qkv","kv_append","attn_decode",
                                     "gemv_attnproj","add","gemv_fc","gelu","gemv_ffnproj","gemv_logits" };
static const char *OP_SHAPE[N_OP] = { "1 row gather","25 calls, M=1","N=2304 K=768 (x12)","1 pos, all heads (x12)",
    "M=1 over ctx keys (x12)","N=768 K=768 (x12)","24 calls","N=3072 K=768 (x12)","3072 elem (x12)",
    "N=768 K=3072 (x12)","N=50257 K=768 (tied head)" };
// weight bytes touched per decode step, per op, at fp16 (x2 for halves). INT8 halves the quantized ones.
static double op_weight_bytes(int op, int q8){
    const double L=NL;
    double h = q8 ? 1.0 : 2.0;                       // quantized tensors: 1 B/elem under INT8
    switch(op){
        case OP_QKV:    return L*2304.0*768*h;
        case OP_APROJ:  return L*768.0*768*h;
        case OP_FC:     return L*3072.0*768*h;
        case OP_FPROJ:  return L*768.0*3072*h;
        case OP_LOGITS: return 50257.0*768*2.0;      // head kept fp16 by Stage 4's kill-test
        default: return 0.0;
    }
}

struct OpTimer {
    cudaEvent_t a,b;
    OpTimer(){ CUDA_CHECK(cudaEventCreate(&a)); CUDA_CHECK(cudaEventCreate(&b)); }
    ~OpTimer(){ cudaEventDestroy(a); cudaEventDestroy(b); }
    template<class F> double ms(F launch){
        CUDA_CHECK(cudaEventRecord(a)); launch(); CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(b)); CUDA_CHECK(cudaEventSynchronize(b));
        float t=0; CUDA_CHECK(cudaEventElapsedTime(&t,a,b)); return (double)t;
    }
};

// Mirrors gpt2_decode_step_cuda op-for-op, driving the SAME exported launchers. Any drift would show
// up in the sum-vs-whole check.
static void step_timed(OpTimer &tm, const GPT2Backend *be, const GPT2WeightsGPU *w, GPT2KVCache *kv,
                       int token, int pos, GPT2ScratchGPU *s, half *logits, double *acc){
    const int F=GPT2_FFN_DIM, Q=GPT2_QKV_DIM;
    const GPT2QWeightsGPU *qw = w->q;
    acc[OP_EMBED] += tm.ms([&]{ gpt2_embed_one(s->x, w->wte, w->wpe, token, pos); });
    for(int L=0;L<NL;L++){
        const GPT2LayerGPU *ly=&w->layers[L];
        const GPT2QLayerGPU *qy = qw? &qw->layers[L] : nullptr;
        acc[OP_LN]     += tm.ms([&]{ be->layernorm(s->ln,s->x,ly->ln1_g,ly->ln1_b,1,E); });
        acc[OP_QKV]    += tm.ms([&]{ gpt2_gemv_dispatch(be,s->qkv,s->ln,ly->qkv_w,qy?&qy->qkv_w:nullptr,ly->qkv_b,Q,E); });
        acc[OP_APPEND] += tm.ms([&]{ gpt2_kv_append_one(kv,L,s->qkv,pos); });
        const int len=pos+1;
        acc[OP_ATTN]   += tm.ms([&]{ gpt2_attn_decode(s->att,s->qkv,kv,L,len); });
        acc[OP_APROJ]  += tm.ms([&]{ gpt2_gemv_dispatch(be,s->ao,s->att,ly->attn_proj_w,qy?&qy->attn_proj_w:nullptr,ly->attn_proj_b,E,E); });
        acc[OP_ADD]    += tm.ms([&]{ be->add(s->x,s->ao,(size_t)E); });
        acc[OP_LN]     += tm.ms([&]{ be->layernorm(s->ln,s->x,ly->ln2_g,ly->ln2_b,1,E); });
        acc[OP_FC]     += tm.ms([&]{ gpt2_gemv_dispatch(be,s->fc,s->ln,ly->fc_w,qy?&qy->fc_w:nullptr,ly->fc_b,F,E); });
        acc[OP_GELU]   += tm.ms([&]{ be->gelu(s->fc,(size_t)F); });
        acc[OP_FPROJ]  += tm.ms([&]{ gpt2_gemv_dispatch(be,s->ff,s->fc,ly->proj_w,qy?&qy->proj_w:nullptr,ly->proj_b,E,F); });
        acc[OP_ADD]    += tm.ms([&]{ be->add(s->x,s->ff,(size_t)E); });
    }
    acc[OP_LN]     += tm.ms([&]{ be->layernorm(s->ln,s->x,w->lnf_g,w->lnf_b,1,E); });
    acc[OP_LOGITS] += tm.ms([&]{ gpt2_gemv_dispatch(be,logits,s->ln,w->wte,qw?&qw->wte:nullptr,nullptr,V,E); });
}

static double median_of(std::vector<double> v){ std::sort(v.begin(),v.end()); size_t n=v.size();
    return n? (n%2? v[n/2] : 0.5*(v[n/2-1]+v[n/2])) : 0.0; }

static char *read_text(const char*p){ FILE*f=fopen(p,"rb"); fseek(f,0,SEEK_END); long n=ftell(f);
    fseek(f,0,SEEK_SET); char*b=(char*)malloc(n+1); if(fread(b,1,n,f)!=(size_t)n)exit(1); b[n]=0; fclose(f); return b; }
static int parse_int_array(const char*buf,const char*key,int*out,int maxn){
    const char*p=strstr(buf,key); p=strchr(p,'['); p++; int n=0;
    while(*p&&*p!=']'){ while(*p&&*p!=']'&&*p!='-'&&(*p<'0'||*p>'9'))p++; if(!*p||*p==']')break;
        char*e; long v=strtol(p,&e,10); if(e==p){p++;continue;} if(n<maxn)out[n]=(int)v; n++; p=e; } return n; }

int main(int argc,char**argv){
    const int ctx = argc>1? atoi(argv[1]) : 512;
    const int N = 30, WARM = 10;

    GPT2Weights wcpu; if(gpt2_load_weights("weights/gpt2_124m_fp32.bin",&wcpu)) return 1;
    GPT2WeightsGPU w; gpt2_upload_fp16(&wcpu,&w); gpt2_free_weights(&wcpu);
    const GPT2Backend *FLASH=gpt2_backend_by_name("flash");
    const GPT2Backend *GEMV =gpt2_backend_by_name("gemv");
    const GPT2Backend *INT8 =gpt2_backend_by_name("int8");
    GPT2QWeightsGPU qw; if(gpt2_quant_attach_if_needed(INT8,&w,&qw)) return 1;

    GPT2KVCache kv; if(gpt2_kv_alloc(&kv,GPT2_N_CTX)) return 1;
    char *mbuf=read_text("refdumps/meta.json"); static int ids[GPT2_N_CTX];
    parse_int_array(mbuf,"\"eval_ids\"",ids,GPT2_N_CTX); free(mbuf);
    GPT2ScratchGPU s; gpt2_scratch_alloc(&s,GPT2_N_CTX);
    int *d_ids=dmalloc<int>(GPT2_N_CTX); h2d(d_ids,ids,GPT2_N_CTX);
    half *d_logits=dmalloc<half>(V);

    gpt2_kv_reset(&kv);
    { GPT2CapsGPU c; memset(&c,0,sizeof(c)); gpt2_prefill_fill_cache(FLASH,&w,&kv,d_ids,ctx,&s,&c); }
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("==== [A] per-op attribution of ONE decode step @ ctx=%d (N=%d, warmup=%d) ====\n", ctx,N,WARM);
    printf("     per-op syncs SERIALIZE -> SUM(per-op) > true step latency. Use for RELATIVE SHARE only.\n\n");

    for(int b=0;b<2;b++){
        const GPT2Backend *be = b? INT8 : GEMV;
        OpTimer tm;
        for(int i=0;i<WARM;i++){ kv.len=ctx; gpt2_decode_step_cuda(be,&w,&kv,ids[ctx],ctx,&s,d_logits,nullptr); }
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<std::vector<double>> per(N, std::vector<double>(N_OP,0.0));
        for(int i=0;i<N;i++){ kv.len=ctx; step_timed(tm,be,&w,&kv,ids[ctx],ctx,&s,d_logits,per[i].data()); }
        double med[N_OP], sum=0;
        for(int o=0;o<N_OP;o++){ std::vector<double> v; for(int i=0;i<N;i++) v.push_back(per[i][o]);
            med[o]=median_of(v); sum+=med[o]; }
        // uninstrumented whole step
        std::vector<double> whole;
        for(int i=0;i<N;i++) whole.push_back(cuda_time_once_ms([&]{ kv.len=ctx;
            gpt2_decode_step_cuda(be,&w,&kv,ids[ctx],ctx,&s,d_logits,nullptr); }));
        double wmed=median_of(whole);

        printf("---- backend = %s ----\n", be->name);
        printf("  %-15s %10s %8s  %10s %10s   %s\n","op","ms","%%","W bytes MB","GB/s","shape");
        int idx[N_OP]; for(int o=0;o<N_OP;o++) idx[o]=o;
        std::sort(idx,idx+N_OP,[&](int x,int y){ return med[x]>med[y]; });
        for(int k=0;k<N_OP;k++){ int o=idx[k];
            double wb = op_weight_bytes(o, be->gemv_q!=nullptr);
            printf("  %-15s %10.4f %7.2f%%  %10.1f %10.1f   %s\n", OP_NAME[o], med[o], 100.0*med[o]/sum,
                   wb/1e6, wb? (wb/(med[o]*1e-3)/1e9) : 0.0, OP_SHAPE[o]);
        }
        printf("  %-15s %10.4f %7.2f%%\n","SUM(per-op)",sum,100.0);
        printf("  [validity] whole step (uninstrumented) = %.4f ms ; SUM/whole = %.3fx  (%s)\n",
               wmed, sum/wmed, sum/wmed<1.35? "serialization overhead only -> shares trustworthy"
                                            : "** large serialization -> shares distorted **");
        // Launch inventory of gpt2_decode_step_cuda (cuda/kvcache.cu):
        //   embed_one (1)
        //   x NL layers, 11 each: ln1, gemv qkv, kv_append, attn_decode, gemv attnproj, add,
        //                         ln2, gemv fc, gelu, gemv ffnproj, add
        //   final ln_f (1) + tied-head gemv (1)
        // [CORRECTED 2026-07-10] this printed `1 + NL*11 + 1` = 134, which counted only ONE of the two
        // trailing kernels. The true count is 1 + 12*11 + 2 = 135.
        printf("  [launch]   %d kernel launches per step ; whole-step - SUM(GEMV+attn+append) leaves the rest\n\n",
               1 + NL*11 + 2);
    }

    // ---------------- [B] isolated GEMVs: does the byte halving appear? ----------------
    printf("==== [B] isolated M=1 GEMV, fp16 vs INT8, with ACHIEVED weight bandwidth ====\n");
    printf("     If a GEMV is weight-BW-bound, halving its bytes must ~halve its time.\n\n");
    printf("  %-28s %10s %10s %10s   %10s %10s %10s   %8s\n",
           "op (M=1)","fp16 ms","fp16 MB","fp16 GB/s","int8 ms","int8 MB","int8 GB/s","speedup");
    struct Case { const char*name; int N_,K_; const half*Wh; const GPT2QW*Wq; const half*bias; half*out; };
    const GPT2LayerGPU *ly=&w.layers[0]; const GPT2QLayerGPU *qy=&qw.layers[0];
    half *big=dmalloc<half>(V);
    Case cs[] = {
        {"qkv     N=2304 K=768", GPT2_QKV_DIM, E, ly->qkv_w, &qy->qkv_w, ly->qkv_b, s.qkv},
        {"attnproj N=768 K=768", E, E, ly->attn_proj_w, &qy->attn_proj_w, ly->attn_proj_b, s.ao},
        {"fc      N=3072 K=768", GPT2_FFN_DIM, E, ly->fc_w, &qy->fc_w, ly->fc_b, s.fc},
        {"ffnproj N=768 K=3072", E, GPT2_FFN_DIM, ly->proj_w, &qy->proj_w, ly->proj_b, s.ff},
        {"head    N=50257 K=768", V, E, w.wte, &qw.wte, nullptr, big},
    };
    double head_fp16_ms = 0.0;                 // [B]'s tied-head median; the shared term in [C]'s ceiling
    for(auto &c : cs){
        auto f16=[&]{ gpt2_gemv_fp16(c.out,s.ln,c.Wh,c.bias,c.N_,c.K_); };
        auto i8 =[&]{ if(c.Wq->q) gpt2_gemv_int8(c.out,s.ln,c.Wq->q,c.Wq->s,c.bias,c.N_,c.K_);
                      else        gpt2_gemv_fp16(c.out,s.ln,c.Wh,c.bias,c.N_,c.K_); };
        for(int i=0;i<20;i++){ f16(); i8(); }
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<double> a,b;
        for(int i=0;i<200;i++){ a.push_back(cuda_time_once_ms(f16)); b.push_back(cuda_time_once_ms(i8)); }
        double ma=median_of(a), mb=median_of(b);
        if (c.N_ == V) head_fp16_ms = ma;      // head: fp16 in BOTH builds (kill-test), 77.2 MB >> L2 ->
                                               // a true DRAM figure from THIS session
        double B16=(double)c.N_*c.K_*2.0, B8 = c.Wq->q? (double)c.N_*c.K_ : B16;
        printf("  %-28s %10.4f %10.1f %10.1f   %10.4f %10.1f %10.1f   %7.3fx%s\n", c.name,
               ma,B16/1e6,B16/(ma*1e-3)/1e9, mb,B8/1e6,B8/(mb*1e-3)/1e9, ma/mb,
               c.Wq->q? "":"   (kept fp16 -> identical kernel)");
    }
    printf("\n  achieved copy BW (ROOFLINE §6) = 233.4 GB/s ; read BW = 248.9 GB/s\n");
    printf("  CAVEAT: the four per-LAYER rows above are L2-INFLATED (some read >233 GB/s). Timing the same\n");
    printf("  1.2-4.7 MB weight 200x in a loop keeps it in the 32 MiB L2. The head (77.2 MB >> L2) is the\n");
    printf("  only row here whose GB/s is a true DRAM figure. Section [C] fixes this.\n\n");

    // ---------------- [C] all 48 block GEMVs back-to-back = one step's worth of block weights ----------------
    // This is what the real step streams: 12 layers x {qkv, attnproj, fc, ffnproj}. Together they are
    // 169.9 MB (fp16) / 84.9 MB (int8) -- both >> 32 MiB L2, so each GEMV evicts its predecessors and the
    // achieved GB/s is a DRAM figure. This bounds how much INT8 can possibly save in a decode step.
    printf("==== [C] one step's 48 BLOCK GEMVs back-to-back (no L2 inflation: 169.9 / 84.9 MB >> 32 MiB L2) ====\n\n");
    {
        auto blocks=[&](const GPT2Backend *be){
            for(int L=0;L<NL;L++){
                const GPT2LayerGPU *ly=&w.layers[L];
                const GPT2QLayerGPU *qy = w.q? &w.q->layers[L] : nullptr;
                gpt2_gemv_dispatch(be,s.qkv,s.ln,ly->qkv_w,qy?&qy->qkv_w:nullptr,ly->qkv_b,GPT2_QKV_DIM,E);
                gpt2_gemv_dispatch(be,s.ao,s.att,ly->attn_proj_w,qy?&qy->attn_proj_w:nullptr,ly->attn_proj_b,E,E);
                gpt2_gemv_dispatch(be,s.fc,s.ln,ly->fc_w,qy?&qy->fc_w:nullptr,ly->fc_b,GPT2_FFN_DIM,E);
                gpt2_gemv_dispatch(be,s.ff,s.fc,ly->proj_w,qy?&qy->proj_w:nullptr,ly->proj_b,E,GPT2_FFN_DIM);
            }
        };
        for(int i=0;i<20;i++){ blocks(GEMV); blocks(INT8); }
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<double> a,b;
        for(int i=0;i<100;i++){ a.push_back(cuda_time_once_ms([&]{ blocks(GEMV); }));
                                b.push_back(cuda_time_once_ms([&]{ blocks(INT8); })); }
        double ma=median_of(a), mb=median_of(b);
        const double MB16=169.87, MB8=84.93;      // 12*(2304+768+3072+768*4... ) -- from quantize.py's count
        printf("  fp16 48 block GEMVs : %.4f ms  -> %.1f MB at %.1f GB/s\n", ma, MB16, MB16/(ma*1e-3)/1e3);
        printf("  INT8 48 block GEMVs : %.4f ms  -> %.1f MB at %.1f GB/s\n", mb, MB8,  MB8 /(mb*1e-3)/1e3);
        printf("  speedup = %.3fx   (a pure byte-halving would give 2.00x; the shortfall is the per-kernel\n"
               "            latency floor -- 48 launches of 1.2-4.7 MB each, see attnproj at 8.2 us both ways)\n\n", ma/mb);
        // What that implies for the whole step, given the head is fp16 in BOTH builds. The head term is
        // [B]'s median from THIS run: a hardcoded constant would mix perf states across sessions.
        double head = head_fp16_ms;
        printf("  IMPLIED CEILING on INT8's whole-step win, with the head kept fp16 (Stage 4's kill-test):\n");
        printf("    fp16 step >= blocks %.4f + head %.4f = %.4f ms ;  INT8 step >= %.4f + %.4f = %.4f ms\n",
               ma, head, ma+head, mb, head, mb+head);
        printf("    => weight-side ceiling on the speedup = %.3fx, BEFORE attention/LN/add/gelu and the\n"
               "       135-launch overhead dilute it further. THAT is why row 4 cannot reach ~1.5x here.\n\n",
               (ma+head)/(mb+head));
    }

    cudaFree(big); gpt2_scratch_free(&s); cudaFree(d_ids); cudaFree(d_logits);
    gpt2_kv_free(&kv); gpt2_free_gpu(&w); gpt2_quant_free(&qw);
    return 0;
}
