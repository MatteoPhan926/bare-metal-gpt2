// Optional external baseline; never linked into the bare-metal engine.
// Compile against the owner's existing llama.cpp checkout, no changes to it.
#include "llama.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
using Clock=std::chrono::steady_clock;
static void check(bool v,const char *s){if(!v){fprintf(stderr,"FAIL: %s\n",s);exit(1);}}
struct Row{int ctx,pos,rep,token; double ms;};
int main(int argc,char **argv){
    check(argc>=2,"usage: llama_phase2 model-f16.gguf [N=256]");
    int n=argc>2?atoi(argv[2]):256; check(n>=256 && n%16==0,"N >=256 and multiple of 16");
    FILE *f=fopen("refdumps/wikitext2_val_ids.bin","rb"); check(f!=nullptr,"token file");
    std::vector<llama_token> ids(1024); check(fread(ids.data(),4,1024,f)==1024,"1024 tokens"); fclose(f);
    // Match llama-bench: debug logging per replay is not inference work.
    llama_log_set([](ggml_log_level level,const char *text,void *) {
        if(level==GGML_LOG_LEVEL_ERROR || level==GGML_LOG_LEVEL_WARN) fputs(text,stderr);
    },nullptr);
    llama_backend_init();
    auto mp=llama_model_default_params(); mp.n_gpu_layers=99;
    auto *model=llama_model_load_from_file(argv[1],mp); check(model!=nullptr,"model load");
    check(llama_vocab_n_tokens(llama_model_get_vocab(model))==50257,"GPT-2 vocab");
    auto cp=llama_context_default_params(); cp.n_ctx=1024; cp.n_batch=1024; cp.n_ubatch=1024;
    cp.n_threads=1; cp.n_threads_batch=1; cp.type_k=GGML_TYPE_F16; cp.type_v=GGML_TYPE_F16;
    cp.offload_kqv=true; cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED;
    auto *ctx=llama_init_from_model(model,cp); check(ctx!=nullptr,"context creation");
    // Same sustained clock warmup as the engine, not merely 21 short steps.
    auto warm=Clock::now();
    do {
        llama_memory_clear(llama_get_memory(ctx),true);
        check(llama_decode(ctx,llama_batch_get_one(ids.data(),512))==0,"warm prefill");
        llama_synchronize(ctx);
        for(int p=512;p<544;p++) {
            check(llama_decode(ctx,llama_batch_get_one(&ids[p],1))==0,"warm decode");
            llama_synchronize(ctx);
        }
    } while(std::chrono::duration<double>(Clock::now()-warm).count()<1.5);
    std::vector<Row> rows;
    for(int start:{128,512,1007}) for(int rep=-1;rep<n/16;rep++){
        llama_memory_clear(llama_get_memory(ctx),true);
        check(llama_decode(ctx,llama_batch_get_one(ids.data(),start-5))==0,"prefill");
        llama_synchronize(ctx);
        for(int pos=start-5;pos<start+16;pos++){
            auto begin=Clock::now();
            check(llama_decode(ctx,llama_batch_get_one(&ids[pos],1))==0,"decode");
            llama_synchronize(ctx);
            float *logits=llama_get_logits_ith(ctx,-1); check(logits!=nullptr,"host logits");
            double ms=std::chrono::duration<double,std::milli>(Clock::now()-begin).count();
            // Validate outside timing; no sampling, same fixed teacher-forced IDs.
            for(int j=0;j<50257;j++) check(std::isfinite(logits[j]),"nonfinite logits");
            if(rep>=0 && pos>=start) rows.push_back({start,pos,rep,ids[pos],ms});
        }
    }
    puts("sample,experiment,policy,ctx,pos,rep,token,event_ms,wall_ms,enqueue_ms,update_ms");
    for(auto &r:rows) printf("sample,forward_host,llama_f16,%d,%d,%d,%d,0,%.9f,0,0\n",r.ctx,r.pos,r.rep,r.token,r.ms);
    llama_free(ctx); llama_model_free(model); llama_backend_free();
}
