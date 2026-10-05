// Exact-equivalence gate in addition to unchanged HF tolerances in kv_gate_graph.
#include "phase2_common.cuh"
#include <thread>

static void same(const half *a,const half *b,size_t n,bool finite,const char *what) {
    std::vector<half> x(n),y(n); d2h(x.data(),a,n); d2h(y.data(),b,n);
    require(!memcmp(x.data(),y.data(),n*sizeof(half)),what);
    if(finite) for(size_t i=0;i<n;++i)
        require(std::isfinite(__half2float(x[i])) && std::isfinite(__half2float(y[i])),"nonfinite state");
}
int main(int argc,char **argv) {
    Phase2Model model(argc>1?argv[1]:"gemv"); auto ids=context_ids();
    Phase2State eager(true),graph(true);
    GPT2GraphSetup setup{};
    CUDA_CHECK(gpt2_decode_graph_create(&graph.graph,model.be,&model.w,&graph.kv,&graph.s,graph.logits,graph.caps,&setup));
    require(graph.kv.len==0,"capture changed host cache length");
    same(eager.kv.data,graph.kv.data,eager.kv.bytes/sizeof(half),false,"capture executed/cache changed");
    require(setup.kernel_nodes==135 && setup.copy_nodes==14 && setup.updated_nodes==25,"graph topology");
    require(gpt2_decode_graph_launch(graph.graph)==cudaErrorInvalidValue,"unprepared launch accepted");
    cudaError_t foreign=cudaSuccess;
    std::thread other([&]{foreign=gpt2_decode_graph_step(graph.graph,0,0);}); other.join();
    require(foreign==cudaErrorInvalidValue,"cross-thread graph use accepted");
    graph.kv.maxT=512;
    require(gpt2_decode_graph_step(graph.graph,0,0)==cudaErrorInvalidValue,"changed cache layout accepted");
    graph.kv.maxT=1024;
    for(auto bad : std::vector<std::pair<int,int>>{{-1,0},{GPT2_VOCAB,0},{0,-1},{0,1024},{0,1}})
        require(gpt2_decode_graph_step(graph.graph,bad.first,bad.second)==cudaErrorInvalidValue,"invalid input accepted");
    for(int t=0;t<GPT2_N_CTX;++t) {
        eager.step(model,ids[t],t,false); graph.step(model,ids[t],t,true);
        same(eager.logits,graph.logits,GPT2_VOCAB,true,"sequential logits mismatch");
        same(eager.caps,graph.caps,(size_t)GPT2_DECODE_CAPS_ROWS*GPT2_N_EMBD,true,"layer mismatch");
        require(eager.kv.len==t+1 && graph.kv.len==t+1,"cache length drift");
        if(t==0 || t==63 || t==64 || t==65 || t==127 || t==128 || t==129 ||
           t==511 || t==512 || t==513 || t==1023)
            same(eager.kv.data,graph.kv.data,eager.kv.bytes/sizeof(half),false,"KV contents/untouched slots mismatch");
    }
    require(gpt2_decode_graph_step(graph.graph,0,1024)==cudaErrorInvalidValue,"context overflow accepted");
    printf("PASS: 1024 sequential positions; bit-exact finite logits/layers; KV and boundary checks\n");

    // Reuse this SAME graph after reset and changed inputs; stale length/token bugs
    // must not pass by repeatedly replaying the captured (token=0,pos=0) step.
    std::reverse(ids.begin(),ids.end());
    for(int prefix : {1,63,128,512,1007}) {
        gpt2_kv_reset(&eager.kv); gpt2_kv_reset(&graph.kv);
        eager.fill(model,ids,prefix); graph.fill(model,ids,prefix);
        for(int t=prefix;t<std::min(prefix+17,GPT2_N_CTX);++t) {
            eager.step(model,ids[t],t,false); graph.step(model,ids[t],t,true);
            same(eager.logits,graph.logits,GPT2_VOCAB,true,"prefill/reset logits mismatch");
        }
        same(eager.kv.data,graph.kv.data,eager.kv.bytes/sizeof(half),false,"prefill/reset cache mismatch");
    }
    printf("PASS: prefill-to-decode; graph reuse; changed tokens; lengths 1/63/128/512/1007\n");
    auto prompt=meta_ids("prompt_ids");
    eager.fill(model,prompt,(int)prompt.size()); graph.fill(model,prompt,(int)prompt.size());
    std::vector<half> a(GPT2_VOCAB),b(GPT2_VOCAB);
    for(int i=0;i<128;++i) {
        d2h(a.data(),eager.logits,a.size()); d2h(b.data(),graph.logits,b.size());
        int token=host_argmax(a.data()); require(token==host_argmax(b.data()),"free-run token mismatch");
        int pos=(int)prompt.size()+i;
        eager.step(model,token,pos,false); graph.step(model,token,pos,true);
        same(eager.logits,graph.logits,GPT2_VOCAB,true,"free-run logits mismatch");
    }
    printf("PASS: 128 free-running greedy steps; independent caches; exact logits/tokens\n");
    // Also gate the exact no-diagnostic graph topology used for speed.
    gpt2_decode_graph_destroy(graph.graph); graph.graph=nullptr;
    CUDA_CHECK(gpt2_decode_graph_create(&graph.graph,model.be,&model.w,&graph.kv,&graph.s,graph.logits,nullptr,&setup));
    require(setup.copy_nodes==0,"speed graph contains diagnostic copies");
    for(int t=graph.kv.len;t<512;++t) {
        eager.step(model,ids[t],t,false); graph.step(model,ids[t],t,true);
        same(eager.logits,graph.logits,GPT2_VOCAB,true,"speed graph mismatch");
    }
    printf("ALL PASS backend=%s; no thresholds changed\n",model.be->name);
    return 0;
}
