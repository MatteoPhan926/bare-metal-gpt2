// Phase 2: unchanged kernels, explicit timing boundaries, every sample retained.
#include "phase2_common.cuh"
#include <cuda_profiler_api.h>
#include <functional>

struct Sample {
    std::string experiment, policy;
    int ctx, pos, rep, token;
    double event, wall, enqueue, update;
};
static std::vector<Sample> samples;
struct Timer {
    cudaEvent_t a,b;
    Timer(){ CUDA_CHECK(cudaEventCreate(&a)); CUDA_CHECK(cudaEventCreate(&b)); }
    ~Timer(){ cudaEventDestroy(a); cudaEventDestroy(b); }
    float elapsed(){ float ms; CUDA_CHECK(cudaEventElapsedTime(&ms,a,b)); return ms; }
};
static const char *policy(int p){ return p==0 ? "ordinary" : p==1 ? "graph" : "graph_fixed_diagnostic"; }
static Sample timed(Phase2Model &m,Phase2State &s,Timer &t,int p,int token,int pos,
                    const char *experiment,int ctx,int rep,half *host=nullptr,bool sampling=true) {
    auto begin=WallClock::now();
    CUDA_CHECK(cudaEventRecord(t.a));
    auto submit=WallClock::now();
    double update=0;
    if(p==1) {
        CUDA_CHECK(gpt2_decode_graph_prepare(s.graph,token,pos));
        update=wall_ms(submit);
        CUDA_CHECK(gpt2_decode_graph_launch(s.graph));
    } else if(p==2) CUDA_CHECK(gpt2_decode_graph_launch(s.graph));
    else s.step(m,token,pos,false);
    double enqueue=wall_ms(submit);
    CUDA_CHECK(cudaEventRecord(t.b));
    if(host) {
        CUDA_CHECK(cudaMemcpyAsync(host,s.logits,GPT2_VOCAB*sizeof(half),cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
        if(sampling) token=host_argmax(host);
    } else CUDA_CHECK(cudaEventSynchronize(t.b));
    double wall=wall_ms(begin);
    return {experiment,policy(p),ctx,pos,rep,token,t.elapsed(),wall,enqueue,update};
}
static void setup(Phase2Model &m,Phase2State &s,GPT2GraphSetup *t=nullptr) {
    CUDA_CHECK(gpt2_decode_graph_create(&s.graph,m.be,&m.w,&s.kv,&s.s,s.logits,nullptr,t));
}
static void output() {
    puts("sample,experiment,policy,ctx,pos,rep,token,event_ms,wall_ms,enqueue_ms,update_ms");
    for(size_t i=0;i<samples.size();i++) {
        auto &s=samples[i];
        printf("sample,%s,%s,%d,%d,%d,%d,%.9f,%.9f,%.9f,%.9f\n",s.experiment.c_str(),
            s.policy.c_str(),s.ctx,s.pos,s.rep,s.token,s.event,s.wall,s.enqueue,s.update);
    }
}
int main(int argc,char **argv) {
    const char *backend=argc>1?argv[1]:"gemv";
    const char *mode=argc>2?argv[2]:"all";
    int n=argc>3?atoi(argv[3]):256;
    require(n>=16 && n%16==0,"sample count must be a positive multiple of 16 (publication >=256)");
#ifdef GPT2_LEGACY_BENCH
    const int policies=1;
    require(!strcmp(mode,"all") || !strcmp(mode,"fixed"),"legacy build supports all/fixed only");
    puts("metadata,stream,legacy");
#else
    const int policies=2;
    puts("metadata,stream,per-thread");
#endif
    Phase2Model m(backend);
    auto ids=context_ids();
    Phase2State a,b;
    Phase2State *states[2]={&a,&b};
    Timer timer;
    if(policies==2) {
        GPT2GraphSetup t{}; auto begin=WallClock::now(); setup(m,b,&t);
        samples.push_back({"setup","process_cold",128,128,-1,0,t.capture_ms,wall_ms(begin),t.instantiate_ms,t.upload_ms});
    }
    printf("metadata,backend,%s\nmetadata,samples_per_condition,%d\n",backend,n);
    // Sustained boost warmup plus ten condition-specific warmups below.
    a.fill(m,ids,512);
    auto warm=WallClock::now();
    do {
        for(int j=0;j<20;j++){ a.kv.len=512; a.step(m,ids[512],512,false); }
        CUDA_CHECK(cudaDeviceSynchronize());
    } while(wall_ms(warm)<1500);

    if(!strcmp(mode,"profile_ordinary") || !strcmp(mode,"profile_graph")) {
        int ctx=argc>4?atoi(argv[4]):128;
        require(ctx>0 && ctx<=1023,"profile ctx out of range");
        int p=!strcmp(mode,"profile_graph"); auto &s=*states[p];
        s.fill(m,ids,ctx);
        for(int i=0;i<10;i++){ s.kv.len=ctx; s.step(m,ids[ctx],ctx,p!=0); }
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaProfilerStart());
        for(int i=0;i<n;i++) {
            s.kv.len=ctx;
            samples.push_back(timed(m,s,timer,p,ids[ctx],ctx,"profile",ctx,i));
        }
        CUDA_CHECK(cudaProfilerStop()); output(); return 0;
    }
    bool all=!strcmp(mode,"all");
    require(all || !strcmp(mode,"fixed") || !strcmp(mode,"growing") || !strcmp(mode,"setup"),"unknown mode");
    if(all || !strcmp(mode,"fixed")) for(int ctx:{128,512,1023}) {
        for(int p=0;p<policies;p++) states[p]->fill(m,ids,ctx);
        int total=policies==2?3:1;
        for(int i=-10;i<n;i++) {
            for(int k=0;k<total;k++) {
                // Reverse A/B on alternate pairs. Diagnostic fixed graph is separate.
                int p=k<2 && policies==2 && (i&1) ? 1-k : k;
                auto &s=*states[p?1:0]; s.kv.len=ctx;
                // p=1 executes first on the first (-10) pair and has prepared
                // these exact parameters. p=2 reuses them without any update.
                auto row=timed(m,s,timer,p,ids[ctx],ctx,"fixed_single",ctx,i);
                if(i>=0) samples.push_back(row);
            }
        }
        // Legacy-style batches: 50 INDEPENDENT BATCH averages, not 800 samples.
        for(int i=-10;i<50;i++) for(int k=0;k<policies;k++) {
            int p=policies==2 && (i&1)?1-k:k; auto &s=*states[p];
            auto begin=WallClock::now(); CUDA_CHECK(cudaEventRecord(timer.a));
            auto submit=WallClock::now();
            for(int j=0;j<16;j++){ s.kv.len=ctx; s.step(m,ids[ctx],ctx,p!=0); }
            double enqueue=wall_ms(submit)/16;
            CUDA_CHECK(cudaEventRecord(timer.b)); CUDA_CHECK(cudaEventSynchronize(timer.b));
            double wall=wall_ms(begin)/16;
            if(i>=0) samples.push_back({"fixed_batch16",policy(p),ctx,ctx,i,ids[ctx],timer.elapsed()/16,wall,enqueue,0});
        }
    }
    if(all || !strcmp(mode,"growing")) {
        half *host; CUDA_CHECK(cudaHostAlloc(&host,GPT2_VOCAB*sizeof(half),cudaHostAllocDefault));
        for(int workload:{0,1,2}) for(int ctx:{128,512,1007}) {
            bool sampling=workload==2;
            for(int rep=-1;rep<n/16;rep++) {
                std::vector<int> trajectory[2];
                for(int k=0;k<policies;k++) {
                    int p=policies==2 && (rep&1)?1-k:k; auto &s=*states[p];
                    s.fill(m,ids,ctx-5);
                    CUDA_CHECK(cudaMemcpy(host,s.logits,GPT2_VOCAB*sizeof(half),cudaMemcpyDeviceToHost));
                    int token=sampling?host_argmax(host):ids[ctx-5];
                    for(int pos=ctx-5;pos<ctx+16;pos++) {
                        auto row=timed(m,s,timer,p,token,pos,sampling?"generation":workload==1?"forward_host":"advancing_forward",ctx,rep,
                                       workload?host:nullptr,sampling);
                        if(sampling) token=row.token;
                        else token=ids[std::min(pos+1,1023)];
                        trajectory[p].push_back(token);
                        if(rep>=0 && pos>=ctx) samples.push_back(row);
                    }
                }
                if(policies==2) require(trajectory[0]==trajectory[1],"timed generation trajectories differ");
            }
        }
        cudaFreeHost(host);
    }
    if(policies==2 && (all || !strcmp(mode,"setup"))) {
        if(!strcmp(mode,"setup")) b.fill(m,ids,128); // first replay needs a valid cache
        // The initial b graph was already instantiated. First setup here is labelled
        // first of this series, NOT process-cold. Process-cold setup is recorded above.
        for(int i=0;i<31;i++) {
            gpt2_decode_graph_destroy(b.graph); b.graph=nullptr;
            GPT2GraphSetup t{}; auto begin=WallClock::now(); setup(m,b,&t);
            double total=wall_ms(begin);
            samples.push_back({"setup",i==0?"first_series":"warm",128,128,i,0,t.capture_ms,total,t.instantiate_ms,t.upload_ms});
            b.kv.len=128;
            samples.push_back(timed(m,b,timer,1,ids[128],128,"first_replay",128,i));
        }
    }
    // Prefill and unchanged isolated head: paired labels share the SAME function.
    // No graph is executed. Noise/control, not an optimization measurement.
    if(all) {
        for(int i=-10;i<30;i++) for(int k=0;k<policies;k++) {
            int p=policies==2 && (i&1)?1-k:k; auto &s=*states[p];
            CUDA_CHECK(cudaEventRecord(timer.a)); auto begin=WallClock::now();
            s.fill(m,ids,512);
            CUDA_CHECK(cudaEventRecord(timer.b)); CUDA_CHECK(cudaEventSynchronize(timer.b));
            double wall=wall_ms(begin);
            if(i>=0) samples.push_back({"prefill_control",policy(p),512,512,i,0,timer.elapsed(),wall,0,0});
            CUDA_CHECK(cudaEventRecord(timer.a)); begin=WallClock::now();
            m.be->gemv(s.logits,s.s.ln,m.w.wte,nullptr,GPT2_VOCAB,GPT2_N_EMBD);
            CUDA_CHECK(cudaEventRecord(timer.b)); CUDA_CHECK(cudaEventSynchronize(timer.b));
            wall=wall_ms(begin);
            if(i>=0) samples.push_back({"head_control",policy(p),512,512,i,0,timer.elapsed(),wall,0,0});
        }
    }
    output();
    return 0;
}
