// forward_cpu.h — STAGE 0 : pure-C fp32 GPT-2 forward pass (the correctness reference).
//
// This is the golden reference every GPU kernel is (indirectly) judged against: it embodies the
// HF gpt2 fp32 math exactly (pre-LN, gelu_new tanh, tied head), with float activations but
// DOUBLE accumulation in the reductions (matmul / LayerNorm / softmax) so the only diff vs the
// HF fp32 oracle is HF's own fp32 rounding — keeping us tightly inside the QUALITY_GATES §1(a)
// 1e-4 per-layer band, not fighting it.
//
// Zero deps (pure C99 + libm). OpenMP pragmas parallelize over *independent output elements*
// (no cross-thread reduction) so results are bit-identical with or without -fopenmp/ /openmp.
#ifndef GPT2_FORWARD_CPU_H
#define GPT2_FORWARD_CPU_H

#include "config.h"
#include "weights.h"

// Optional per-stage capture (any field NULL is skipped). Sizes are for a forward over T tokens:
//   embed     [T * N_EMBD]            hidden after wte+wpe                    (gate a)
//   blocks    [N_LAYER * T * N_EMBD]  hidden after each block; block L at blocks + L*T*N_EMBD (gate a)
//   final_ln  [T * N_EMBD]            hidden after ln_f                       (gate a)
//   logits    [T * VOCAB] if logits_all, else [VOCAB] for the LAST position  (gates b, c)
typedef struct {
    float *embed;
    float *blocks;
    float *final_ln;
    float *logits;
    int    logits_all;   // 1 -> logits for every position; 0 -> last position only
} GPT2Caps;

// Forward over ids[0..T-1] (1 <= T <= N_CTX). Fills the non-NULL fields of caps.
void gpt2_forward_cpu(const GPT2Weights *w, const int *ids, int T, GPT2Caps *caps);

#endif // GPT2_FORWARD_CPU_H
