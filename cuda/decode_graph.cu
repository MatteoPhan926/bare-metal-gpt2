// Capture the original 135 kernels. No duplicated forward loop or altered math.
#include "decode_graph.cuh"
#include "common.cuh"
#include <chrono>
#include <thread>

// These three existing kernels alone have changing scalar launch parameters.
// Names/signatures are checked against captured function identities, not node order.
extern __global__ void k_embed_one(half *, const half *, const half *, int, int);
extern __global__ void k_kv_append(half *, half *, const half *, int, int);
extern __global__ void k_attn_decode(half *, const half *, const half *, const half *, int, int);

using GraphClock = std::chrono::steady_clock;
static double elapsed_ms(GraphClock::time_point a) {
    return std::chrono::duration<double, std::milli>(GraphClock::now()-a).count();
}

struct DynamicNode {
    cudaGraphNode_t node;
    cudaKernelNodeParams params;
    int kind, nptr;
    void *ptr[4];
    int scalar[2];
    void *args[6];
};
struct GPT2DecodeGraph {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    GPT2KVCache *kv;
    half *cache_address;
    int maxT, device, pos = -1;
    bool prepared = false;
    std::thread::id thread;
    // Fixed storage: params.kernelParams points into these nodes, never relocates.
    DynamicNode dynamic[1 + 2*GPT2_N_LAYER];
};

static bool owner_ok(const GPT2DecodeGraph *g) {
    if (!g || g->thread != std::this_thread::get_id()) return false;
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess || device != g->device) return false;
    return g->kv->data == g->cache_address && g->kv->maxT == g->maxT &&
        g->kv->nLayer == GPT2_N_LAYER && g->kv->nHead == GPT2_N_HEAD &&
        g->kv->headDim == GPT2_HEAD_DIM;
}

cudaError_t gpt2_decode_graph_create(GPT2DecodeGraph **out, const GPT2Backend *be,
    const GPT2WeightsGPU *w, GPT2KVCache *kv, GPT2ScratchGPU *s,
    half *logits, half *caps, GPT2GraphSetup *setup) {
    if (!out) return cudaErrorInvalidValue;
    *out = nullptr;
    if (!be || !be->gemv || !w || !w->data || !kv || !kv->data || !s ||
        !s->x || s->maxT < 1 || !logits || kv->maxT < 1 || kv->maxT > GPT2_N_CTX ||
        kv->nLayer != GPT2_N_LAYER || kv->nHead != GPT2_N_HEAD || kv->headDim != GPT2_HEAD_DIM)
        return cudaErrorInvalidValue;
    auto *g = new GPT2DecodeGraph;
    g->kv = kv; g->cache_address = kv->data; g->maxT = kv->maxT;
    g->thread = std::this_thread::get_id();
    CUDA_CHECK(cudaGetDevice(&g->device));
    GPT2GraphSetup timing{};
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
    auto t = GraphClock::now();
    CUDA_CHECK(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
    int saved_len = kv->len;
    gpt2_decode_step_cuda(be, w, kv, 0, 0, s, logits, caps);
    kv->len = saved_len; // host statements run during capture; GPU work does not.
    CUDA_CHECK(cudaStreamEndCapture(cudaStreamPerThread, &g->graph));
    timing.capture_ms = elapsed_ms(t);

    size_t n = 0;
    CUDA_CHECK(cudaGraphGetNodes(g->graph, nullptr, &n));
    std::vector<cudaGraphNode_t> nodes(n);
    CUDA_CHECK(cudaGraphGetNodes(g->graph, nodes.data(), &n));
    int counts[3] = {0,0,0}, dyn = 0;
    for (auto node : nodes) {
        cudaGraphNodeType type;
        CUDA_CHECK(cudaGraphNodeGetType(node, &type));
        if (type == cudaGraphNodeTypeMemcpy) { ++timing.copy_nodes; continue; }
        if (type != cudaGraphNodeTypeKernel) {
            gpt2_decode_graph_destroy(g); return cudaErrorInvalidValue;
        }
        ++timing.kernel_nodes;
        cudaKernelNodeParams p{};
        CUDA_CHECK(cudaGraphKernelNodeGetParams(node, &p));
        int kind = p.func == (void*)k_embed_one ? 0 :
                   p.func == (void*)k_kv_append ? 1 :
                   p.func == (void*)k_attn_decode ? 2 : -1;
        if (kind < 0) continue;
        if (dyn >= 1+2*GPT2_N_LAYER || !p.kernelParams || p.extra) {
            gpt2_decode_graph_destroy(g); return cudaErrorInvalidValue;
        }
        auto &d = g->dynamic[dyn++];
        d.node = node; d.params = p; d.kind = kind; d.nptr = kind == 2 ? 4 : 3;
        for (int j=0; j<d.nptr; ++j) {
            d.ptr[j] = *static_cast<void**>(p.kernelParams[j]);
            d.args[j] = &d.ptr[j];
        }
        for (int j=0; j<2; ++j) {
            d.scalar[j] = *static_cast<int*>(p.kernelParams[d.nptr+j]);
            d.args[d.nptr+j] = &d.scalar[j];
        }
        d.params.kernelParams = d.args;
        ++counts[kind];
    }
    if (counts[0]!=1 || counts[1]!=GPT2_N_LAYER || counts[2]!=GPT2_N_LAYER ||
        timing.kernel_nodes!=135 || timing.copy_nodes!=(caps ? GPT2_DECODE_CAPS_ROWS : 0)) {
        fprintf(stderr, "[graph] unexpected topology; refusing stale node mapping\n");
        gpt2_decode_graph_destroy(g); return cudaErrorInvalidValue;
    }
    timing.updated_nodes = dyn;
    t = GraphClock::now();
    CUDA_CHECK(cudaGraphInstantiate(&g->exec, g->graph, 0));
    timing.instantiate_ms = elapsed_ms(t);
    t = GraphClock::now();
    CUDA_CHECK(cudaGraphUpload(g->exec, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
    timing.upload_ms = elapsed_ms(t);
    if (setup) *setup = timing;
    *out = g;
    return cudaSuccess;
}

cudaError_t gpt2_decode_graph_prepare(GPT2DecodeGraph *g, int token, int pos) {
    if (!owner_ok(g)) return cudaErrorInvalidValue;
    g->prepared = false;
    if (token < 0 || token >= GPT2_VOCAB || pos < 0 || pos >= g->maxT || g->kv->len != pos)
        return cudaErrorInvalidValue;
    // Always update all 25 nodes, even for fixed-context timing. This prevents
    // fixed-position benchmarks from hiding work needed by real generation.
    for (auto &d : g->dynamic) {
        if (d.kind == 0) { d.scalar[0] = token; d.scalar[1] = pos; }
        else if (d.kind == 1) d.scalar[0] = pos;
        else {
            d.scalar[0] = pos+1;
            d.params.sharedMemBytes = (GPT2_HEAD_DIM+pos+1)*sizeof(float);
        }
        cudaError_t e = cudaGraphExecKernelNodeSetParams(g->exec, d.node, &d.params);
        if (e != cudaSuccess) return e;
    }
    g->pos = pos; g->prepared = true;
    return cudaSuccess;
}

cudaError_t gpt2_decode_graph_launch(GPT2DecodeGraph *g) {
    if (!owner_ok(g) || !g->prepared || g->kv->len != g->pos) return cudaErrorInvalidValue;
    cudaError_t e = cudaGraphLaunch(g->exec, cudaStreamPerThread);
    if (e == cudaSuccess) g->kv->len = g->pos+1;
    return e;
}
cudaError_t gpt2_decode_graph_step(GPT2DecodeGraph *g, int token, int pos) {
    cudaError_t e = gpt2_decode_graph_prepare(g, token, pos);
    return e == cudaSuccess ? gpt2_decode_graph_launch(g) : e;
}
void gpt2_decode_graph_destroy(GPT2DecodeGraph *g) {
    if (!g) return;
    // Destruction is deliberately synchronized; never included in replay timing.
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
    if (g->exec) CUDA_CHECK(cudaGraphExecDestroy(g->exec));
    if (g->graph) CUDA_CHECK(cudaGraphDestroy(g->graph));
    delete g;
}
