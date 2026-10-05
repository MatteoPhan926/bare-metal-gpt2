#ifndef GPT2_PHASE2_COMMON_CUH
#define GPT2_PHASE2_COMMON_CUH
#include "common.cuh"
#include "decode_graph.cuh"
#include <chrono>
#include <cmath>
#include <cstring>
#include <string>

using WallClock = std::chrono::steady_clock;
static double wall_ms(WallClock::time_point a) {
    return std::chrono::duration<double,std::milli>(WallClock::now()-a).count();
}
static void require(bool ok, const char *message) {
    if (!ok) { fprintf(stderr,"FAIL: %s\n",message); exit(1); }
}
static std::vector<int> meta_ids(const char *key) {
    FILE *f=fopen("refdumps/meta.json","rb"); require(f!=nullptr,"missing meta.json");
    fseek(f,0,SEEK_END); long n=ftell(f); rewind(f);
    std::string text(n,'\0'); require(fread(&text[0],1,n,f)==(size_t)n,"short meta"); fclose(f);
    std::string quoted="\""+std::string(key)+"\"";
    size_t k=text.find(quoted); require(k!=std::string::npos,"missing token key");
    size_t b=text.find('[',k), e=text.find(']',b);
    require(b!=std::string::npos && e!=std::string::npos,"malformed token array");
    std::vector<int> ids;
    const char *p=text.c_str()+b+1, *end=text.c_str()+e;
    while (p<end) {
        if (*p==',' || *p==' ' || *p=='\r' || *p=='\n' || *p=='\t') { ++p; continue; }
        char *next=nullptr; long v=strtol(p,&next,10);
        require(next>p && next<=end && v>=0 && v<GPT2_VOCAB,"invalid token array");
        ids.push_back((int)v); p=next;
    }
    require(!ids.empty() && ids.size()<=GPT2_N_CTX,"token count outside context");
    return ids;
}
static std::vector<int> context_ids() {
    FILE *f=fopen("refdumps/wikitext2_val_ids.bin","rb");
    require(f!=nullptr,"need tools/dump_wikitext_ids.py output; no silent fallback");
    std::vector<int> ids(GPT2_N_CTX);
    require(fread(ids.data(),sizeof(int),ids.size(),f)==ids.size(),"need 1024 real context IDs");
    fclose(f);
    auto frozen=meta_ids("eval_ids");
    require(frozen.size()==512 && std::equal(frozen.begin(),frozen.end(),ids.begin()),"WikiText/oracle input drift");
    for (int token:ids) require(token>=0 && token<GPT2_VOCAB,"out-of-vocabulary input");
    return ids;
}
struct Phase2Model {
    GPT2WeightsGPU w{}; GPT2QWeightsGPU qw{}; const GPT2Backend *be;
    explicit Phase2Model(const char *name) {
        require(!strcmp(name,"gemv") || !strcmp(name,"int8"),"expected gemv or int8 backend");
        be=gpt2_backend_by_name(name);
        GPT2Weights cpu{};
        require(!gpt2_load_weights("weights/gpt2_124m_fp32.bin",&cpu),"weight load");
        gpt2_upload_fp16(&cpu,&w); gpt2_free_weights(&cpu);
        require(!gpt2_quant_attach_if_needed(be,&w,&qw),"quantized weight load");
    }
    ~Phase2Model(){ gpt2_quant_free(&qw); gpt2_free_gpu(&w); }
};
struct Phase2State {
    GPT2KVCache kv{}; GPT2ScratchGPU s{};
    half *logits, *caps; int *d_ids;
    GPT2DecodeGraph *graph=nullptr;
    explicit Phase2State(bool captures=false) {
        require(!gpt2_kv_alloc(&kv,GPT2_N_CTX),"cache allocation");
        gpt2_scratch_alloc(&s,GPT2_N_CTX);
        logits=dmalloc<half>(GPT2_VOCAB); d_ids=dmalloc<int>(GPT2_N_CTX);
        caps=captures ? dmalloc<half>((size_t)GPT2_DECODE_CAPS_ROWS*GPT2_N_EMBD) : nullptr;
        CUDA_CHECK(cudaMemset(kv.data,0x7e,kv.bytes)); // poison uncached slots
    }
    void fill(Phase2Model &m,const std::vector<int> &ids,int n) {
        require(n>0 && n<=(int)ids.size() && n<=s.maxT,"invalid prefill length");
        h2d(d_ids,ids.data(),n); GPT2CapsGPU c{}; c.logits=logits;
        gpt2_prefill_fill_cache(m.be,&m.w,&kv,d_ids,n,&s,&c);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    void step(Phase2Model &m,int token,int pos,bool replay) {
        require(pos==kv.len && pos>=0 && pos<kv.maxT && token>=0 && token<GPT2_VOCAB,"decode bounds");
        if (replay) CUDA_CHECK(gpt2_decode_graph_step(graph,token,pos));
        else gpt2_decode_step_cuda(m.be,&m.w,&kv,token,pos,&s,logits,caps);
    }
    ~Phase2State() {
        gpt2_decode_graph_destroy(graph); CUDA_CHECK(cudaDeviceSynchronize());
        cudaFree(logits); cudaFree(caps); cudaFree(d_ids);
        gpt2_scratch_free(&s); gpt2_kv_free(&kv);
    }
};
static int host_argmax(const half *row) {
    int best=0;
    for(int i=0;i<GPT2_VOCAB;i++) {
        float v=__half2float(row[i]); require(std::isfinite(v),"nonfinite logits");
        if(v>__half2float(row[best])) best=i;
    }
    return best; // ascending scan -> lowest token ID for ties
}
#endif
