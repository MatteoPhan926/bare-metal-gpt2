// Optional Phase-2 execution policy; arithmetic remains in kvcache.cu.
#ifndef GPT2_DECODE_GRAPH_CUH
#define GPT2_DECODE_GRAPH_CUH
#include "kvcache.cuh"
#include <cuda_runtime.h>

struct GPT2DecodeGraph;
struct GPT2GraphSetup {
    double capture_ms, instantiate_ms, upload_ms;
    size_t kernel_nodes, copy_nodes, updated_nodes;
};

// All model TUs must use --default-stream per-thread. Borrowed weights, scratch,
// cache, logits and optional capture buffer must stay alive at fixed addresses.
// One graph per host thread/device/cache; destroy before freeing any buffer.
// Capture does not execute the step or change kv.len. logits is required.
cudaError_t gpt2_decode_graph_create(GPT2DecodeGraph **out, const GPT2Backend *be,
    const GPT2WeightsGPU *w, GPT2KVCache *kv, GPT2ScratchGPU *s,
    half *logits, half *caps, GPT2GraphSetup *setup = nullptr);
cudaError_t gpt2_decode_graph_step(GPT2DecodeGraph *g, int token, int pos);

// Split only for attribution: production calls step (prepare + launch).
// launch uses the last prepared token/position; it checks kv.len == position.
cudaError_t gpt2_decode_graph_prepare(GPT2DecodeGraph *g, int token, int pos);
cudaError_t gpt2_decode_graph_launch(GPT2DecodeGraph *g);
void gpt2_decode_graph_destroy(GPT2DecodeGraph *g);
#endif
