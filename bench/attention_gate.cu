// Phase 2.2 correctness/localization. No numerical threshold is relaxed.
#include "phase2_common.cuh"
#include "attention_v4.cuh"
#include "attention_ordered.cuh"
#include <limits>
static bool reassociate=false;

static double relative(const half *x,const half *y,size_t n) {
    double diff=0,scale=0;
    for(size_t i=0;i<n;i++) {
        double a=__half2float(x[i]),b=__half2float(y[i]);
        require(std::isfinite(a) && std::isfinite(b),"nonfinite comparison");
        diff=std::max(diff,fabs(a-b)); scale=std::max(scale,fabs(b));
    }
    return diff/(scale+1e-9);
}
static void isolated() {
    constexpr int H=GPT2_N_HEAD,D=64,T=1024;
    GPT2KVCache kv{}; require(!gpt2_kv_alloc(&kv,T),"isolated cache");
    half *q=dmalloc<half>(H*D),*a=dmalloc<half>(H*D),*b=dmalloc<half>(H*D);
    std::vector<half> K(H*T*D),V(K.size()),Q(H*D),A(H*D),B(H*D);
    for(size_t i=0;i<K.size();i++) {
        K[i]=__float2half(sinf((float)i*0.1231f));
        V[i]=__float2half(cosf((float)i*0.0917f));
    }
    double worst=0,cross=0;
    for(int pattern=0;pattern<3;pattern++) {
        for(int i=0;i<H*D;i++) Q[i]=__float2half(pattern==2?0.0f:
            sinf(i*0.317f)*(pattern==1?80.0f:1.0f));
        h2d(q,Q.data(),Q.size());
        CUDA_CHECK(cudaMemset(kv.data,0x7e,kv.bytes));
        std::vector<double> scores(H*T);
        for(int h=0;h<H;h++) for(int j=0;j<T;j++) {
            double dot=0;
            for(int d=0;d<D;d++) dot+=(double)__half2float(Q[h*D+d])*__half2float(K[(h*T+j)*D+d]);
            scores[h*T+j]=dot/8.0;
        }
        for(int len=1;len<=T;len++) {
            // Only the newly valid row is copied. Every future row stays NaN.
            for(int h=0;h<H;h++) {
                size_t offset=(h*T+len-1)*D;
                h2d(gpt2_kv_K(&kv,0)+offset,K.data()+offset,D);
                h2d(gpt2_kv_V(&kv,0)+offset,V.data()+offset,D);
            }
            gpt2_attn_decode(a,q,&kv,0,len);
            if(reassociate) gpt2_attn_decode_v4(b,q,&kv,0,len);
            else gpt2_attn_decode_ordered4(b,q,&kv,0,len);
            d2h(A.data(),a,A.size()); d2h(B.data(),b,B.size());
            double err=0,mag=0;
            for(int h=0;h<H;h++) {
                double m=-std::numeric_limits<double>::infinity();
                for(int j=0;j<len;j++) m=std::max(m,scores[h*T+j]);
                double sum=0,ref[D]={};
                for(int j=0;j<len;j++) {
                    double p=exp(scores[h*T+j]-m); sum+=p;
                    for(int d=0;d<D;d++) ref[d]+=p*__half2float(V[(h*T+j)*D+d]);
                }
                for(int d=0;d<D;d++) {
                    double r=ref[d]/sum,bv=__half2float(B[h*D+d]);
                    require(std::isfinite(bv),"future poison read or nonfinite attention");
                    err=std::max(err,fabs(bv-r)); mag=std::max(mag,fabs(r));
                }
            }
            worst=std::max(worst,err/(mag+1e-9));
            cross=std::max(cross,relative(B.data(),A.data(),A.size()));
            if(!reassociate) require(!memcmp(A.data(),B.data(),A.size()*2),"ordered isolated kernel not bit-exact");
            require(worst<=1e-2 && cross<=1e-2,"isolated attention numerical gate");
        }
    }
    printf("PASS isolated: 3 patterns x all 1024 lengths, poisoned future; CPU-double rel=%.9g, original rel=%.9g\n",worst,cross);
    cudaFree(q); cudaFree(a); cudaFree(b); gpt2_kv_free(&kv);
}

static double max_layer=0,max_logits=0,max_kl=0;
static void compare(Phase2State &a,Phase2State &b,Phase2State &speed,bool caps) {
    std::vector<half> x(GPT2_VOCAB),y(x.size()),z(x.size());
    d2h(x.data(),a.logits,x.size()); d2h(y.data(),b.logits,y.size()); d2h(z.data(),speed.logits,z.size());
    require(!memcmp(y.data(),z.data(),y.size()*sizeof(half)),"V4 diagnostic vs speed graph differs");
    if(!reassociate) require(!memcmp(x.data(),y.data(),x.size()*2),"ordered logits not bit-exact");
    max_logits=std::max(max_logits,relative(y.data(),x.data(),x.size()));
    double mx=-1e300,my=mx,sx=0,sy=0,kl=0;
    for(size_t i=0;i<x.size();i++){mx=std::max(mx,(double)__half2float(x[i]));my=std::max(my,(double)__half2float(y[i]));}
    for(size_t i=0;i<x.size();i++){sx+=exp(__half2float(x[i])-mx);sy+=exp(__half2float(y[i])-my);}
    double lx=mx+log(sx),ly=my+log(sy);
    for(size_t i=0;i<x.size();i++) kl+=exp(__half2float(x[i])-lx)*(__half2float(x[i])-lx-__half2float(y[i])+ly);
    if(!(std::isfinite(kl) && kl<0.02 && max_logits<=1e-2))
        printf("distribution_failure,pos=%d,KL=%.12g,max_logit_relative=%.12g\n",a.kv.len-1,kl,max_logits);
    require(std::isfinite(kl) && kl<0.02 && max_logits<=1e-2,"cross-policy distribution gate");
    max_kl=std::max(max_kl,kl);
    if(caps) {
        x.resize(GPT2_DECODE_CAPS_ROWS*GPT2_N_EMBD);y.resize(x.size());
        d2h(x.data(),a.caps,x.size());d2h(y.data(),b.caps,y.size());
        if(!reassociate) require(!memcmp(x.data(),y.data(),x.size()*2),"ordered layers not bit-exact");
        for(int r=0;r<GPT2_DECODE_CAPS_ROWS;r++)
            max_layer=std::max(max_layer,relative(y.data()+r*GPT2_N_EMBD,x.data()+r*GPT2_N_EMBD,GPT2_N_EMBD));
        require(max_layer<=1e-2,"cross-policy layer gate");
    }
}
static void cache_check(Phase2State &a,Phase2State &b,Phase2State &c,int len,bool poison) {
    std::vector<half> x(a.kv.bytes/2),y(x.size()),z(x.size());
    d2h(x.data(),a.kv.data,x.size());d2h(y.data(),b.kv.data,y.size());d2h(z.data(),c.kv.data,z.size());
    require(!memcmp(y.data(),z.data(),y.size()*2),"V4 captures/no-captures KV mismatch");
    double worst=0;
    for(int slab=0;slab<2*GPT2_N_LAYER*GPT2_N_HEAD;slab++) {
        size_t off=(size_t)slab*1024*64;
        worst=std::max(worst,relative(y.data()+off,x.data()+off,len*64));
        if(poison) for(size_t j=off+len*64;j<off+1024*64;j++) {
            unsigned short bits; memcpy(&bits,&y[j],2);
            require(bits==0x7e7e,"future KV overwritten");
        }
    }
    require(worst<=1e-2,"cross-policy KV error");
    printf("cache,len=%d,rel=%.9g,poison=%d\n",len,worst,(int)poison);
}
int main(int argc,char **argv) {
    reassociate=argc>2 && !strcmp(argv[2],"v4");
    printf("attention policy: %s\n",reassociate?"v4 (rejected experiment)":"ordered4");
    if(argc>1 && !strcmp(argv[1],"isolated")){isolated();return 0;}
    Phase2Model model(argc>1?argv[1]:"gemv");auto ids=context_ids();
    Phase2State a(true),b(true),c;
    GPT2GraphSetup t{};
    auto attention=reassociate?GPT2GraphAttention::V4:GPT2GraphAttention::Ordered4;
    CUDA_CHECK(gpt2_decode_graph_create(&a.graph,model.be,&model.w,&a.kv,&a.s,a.logits,a.caps));
    CUDA_CHECK(gpt2_decode_graph_create(&b.graph,model.be,&model.w,&b.kv,&b.s,b.logits,b.caps,&t,attention));
    require(t.kernel_nodes==135 && t.copy_nodes==14 && t.updated_nodes==25,"V4 capture topology");
    CUDA_CHECK(gpt2_decode_graph_create(&c.graph,model.be,&model.w,&c.kv,&c.s,c.logits,nullptr,&t,attention));
    require(t.kernel_nodes==135 && t.copy_nodes==0 && t.updated_nodes==25,"V4 speed topology");
    cache_check(a,b,c,0,true);
    require(gpt2_decode_graph_launch(b.graph)==cudaErrorInvalidValue,"unprepared V4 accepted");
    for(auto bad:std::vector<std::pair<int,int>>{{-1,0},{GPT2_VOCAB,0},{0,-1},{0,1024},{0,1}})
        require(gpt2_decode_graph_step(b.graph,bad.first,bad.second)==cudaErrorInvalidValue,"invalid V4 input accepted");
    for(int pos=0;pos<1024;pos++) {
        for(auto *s:{&a,&b,&c}) s->step(model,ids[pos],pos,true);
        compare(a,b,c,true);
        require(a.kv.len==pos+1 && b.kv.len==pos+1 && c.kv.len==pos+1,"length drift");
        if(pos==0 || pos==2 || pos==63 || pos==64 || pos==127 || pos==255 || pos==511 || pos==512 || pos==1023)
            cache_check(a,b,c,pos+1,true);
    }
    std::reverse(ids.begin(),ids.end());
    for(int prefix:{1,63,128,512,1007}) {
        for(auto *s:{&a,&b,&c}) {gpt2_kv_reset(&s->kv);s->fill(model,ids,prefix);}
        for(int pos=prefix;pos<std::min(prefix+17,1024);pos++) {
            for(auto *s:{&a,&b,&c}) s->step(model,ids[pos],pos,true);
            compare(a,b,c,true);
        }
        cache_check(a,b,c,std::min(prefix+17,1024),false);
    }
    printf("ALL PASS backend=%s, all positions+reset+prefill+speed topology; max layer=%.9g, logit=%.9g, KL=%.9g\n",
           model.be->name,max_layer,max_logits,max_kl);
}
