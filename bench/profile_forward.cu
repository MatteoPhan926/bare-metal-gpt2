// profile_forward.cu — STAGE 3 measure-before-optimize (DESIGN.md §9.1). Answers ONE question with a
// MEASUREMENT instead of a guess: in the Stage-2 tiled forward, WHERE does the time go?
//
// Method. Mirrors cuda/forward_cuda.cu op-for-op, but brackets EVERY backend call in its own
// CUDA-event pair with a sync after (same discipline as common.cuh's cuda_time_once_ms: we time
// EXECUTION, never launch). Per-op ms are accumulated per category per iteration; we report the
// MEDIAN over N iterations (never best-of-N) plus min-max, and each op's %-of-attributed-time.
//
// TWO HONESTY CAVEATS, both enforced below rather than left to the reader:
//
//  (1) SERIALIZATION. Syncing after every op removes the launch/execute pipelining that the real
//      forward enjoys, so SUM(per-op) runs HIGH vs the true forward latency. The per-op split is
//      therefore valid for RELATIVE SHARE ONLY. The summed number is NOT a forward latency and is
//      never reported as one.
//  (2) VALIDITY CHECK. We independently time the *unmodified* forward (cuda_bench, no inner syncs)
//      and print sum-of-parts vs whole. If they diverge materially, time is unattributed (launch
//      gaps / overlap) and the split is NOT trustworthy — the tool says so instead of us using it.
//
// ncu (sectors/request, memory-vs-compute) stays PENDING on ERR_NVGPUCTRPERM; it would refine WHY a
// kernel is slow. This tool answers WHERE, which is what picks the Stage-3 target.
//
// Run from repo root:  profile_forward.exe [weights.bin] [meta.json]     (env GPT2_BACKEND=naive|tiled)

#include "kernels.cuh"
#include "common.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>

#define E  GPT2_N_EMBD
#define V  GPT2_VOCAB
#define NL GPT2_N_LAYER

// ---------- meta.json token ids (same parse as bench/correctness_cuda.cu) ----------
static char *read_text(const char *path) {
    FILE *f = fopen(path, "rb"); if (!f) { fprintf(stderr, "[prof] cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char *b = (char *)malloc(n + 1);
    if (fread(b, 1, n, f) != (size_t)n) { fprintf(stderr, "short read %s\n", path); exit(1); }
    b[n] = 0; fclose(f); return b;
}
static int parse_int_array(const char *buf, const char *key, int *out, int maxn) {
    const char *p = strstr(buf, key); if (!p) { fprintf(stderr, "[prof] key %s missing\n", key); exit(1); }
    p = strchr(p, '['); if (!p) { fprintf(stderr, "[prof] no [ after %s\n", key); exit(1); } p++;
    int n = 0;
    while (*p && *p != ']') {
        while (*p && *p != ']' && *p != '-' && (*p < '0' || *p > '9')) p++;
        if (!*p || *p == ']') break;
        char *end; long v = strtol(p, &end, 10); if (end == p) { p++; continue; }
        if (n < maxn) out[n] = (int)v; n++; p = end;
    }
    return n;
}

// ---------- op categories ----------
enum { OP_EMBED = 0, OP_LN, OP_QKV, OP_ATTN, OP_APROJ, OP_FC, OP_GELU, OP_FPROJ, OP_ADD, OP_LOGITS, N_OP };
static const char *OP_NAME[N_OP] = {
    "embed", "layernorm", "matmul_qkv", "attention", "matmul_attnproj",
    "matmul_fc", "gelu", "matmul_ffnproj", "add", "matmul_logits(M=1)"
};
// what each op is, for the report (shape at seq len T)
static const char *OP_SHAPE[N_OP] = {
    "T*E lookup", "25 calls: 2/layer + final_ln", "M=T N=2304 K=768", "causal MHA, O(T^2) score work",
    "M=T N=768 K=768", "M=T N=3072 K=768", "T*3072 elementwise", "M=T N=768 K=3072",
    "24 calls: residual", "M=1 N=50257 K=768 (tied head)"
};

// Reusable event pair: created ONCE so no allocation lands inside a timed region.
struct OpTimer {
    cudaEvent_t a, b;
    OpTimer()  { CUDA_CHECK(cudaEventCreate(&a)); CUDA_CHECK(cudaEventCreate(&b)); }
    ~OpTimer() { cudaEventDestroy(a); cudaEventDestroy(b); }
    template <class F> double ms(F launch) {
        CUDA_CHECK(cudaEventRecord(a));
        launch();
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(b));
        CUDA_CHECK(cudaEventSynchronize(b));           // real completion, not launch
        float t = 0; CUDA_CHECK(cudaEventElapsedTime(&t, a, b));
        return (double)t;
    }
};

// One instrumented forward. Mirrors gpt2_forward_cuda EXACTLY (same ops, same order, same buffers);
// the ONLY difference is the per-op event bracket. acc[] gains this pass's per-category ms.
static void forward_timed(OpTimer &tm, const GPT2Backend *be, const GPT2WeightsGPU *w,
                          const int *d_ids, int T, GPT2ScratchGPU *s, half *d_logits, double *acc) {
    const int Q = GPT2_QKV_DIM, F = GPT2_FFN_DIM, H = GPT2_N_HEAD, D = GPT2_HEAD_DIM;

    // The five weight matmuls go through gpt2_matmul_dispatch, exactly as gpt2_forward_cuda does, so a
    // quantized backend is profiled on its INT8 path rather than silently on the fp16 fallback.
    const GPT2QWeightsGPU *qw = w->q;

    acc[OP_EMBED] += tm.ms([&] { be->embed(s->x, w->wte, w->wpe, d_ids, T, E); });
    for (int L = 0; L < NL; L++) {
        const GPT2LayerGPU  *ly = &w->layers[L];
        const GPT2QLayerGPU *qy = qw ? &qw->layers[L] : nullptr;
        acc[OP_LN]    += tm.ms([&] { be->layernorm(s->ln, s->x, ly->ln1_g, ly->ln1_b, T, E); });
        acc[OP_QKV]   += tm.ms([&] { gpt2_matmul_dispatch(be, s->qkv, s->ln, ly->qkv_w, qy?&qy->qkv_w:nullptr, ly->qkv_b, T, Q, E); });
        acc[OP_ATTN]  += tm.ms([&] { be->attention(s->att, s->qkv, T, E, H, D); });
        acc[OP_APROJ] += tm.ms([&] { gpt2_matmul_dispatch(be, s->ao, s->att, ly->attn_proj_w, qy?&qy->attn_proj_w:nullptr, ly->attn_proj_b, T, E, E); });
        acc[OP_ADD]   += tm.ms([&] { be->add(s->x, s->ao, (size_t)T * E); });
        acc[OP_LN]    += tm.ms([&] { be->layernorm(s->ln, s->x, ly->ln2_g, ly->ln2_b, T, E); });
        acc[OP_FC]    += tm.ms([&] { gpt2_matmul_dispatch(be, s->fc, s->ln, ly->fc_w, qy?&qy->fc_w:nullptr, ly->fc_b, T, F, E); });
        acc[OP_GELU]  += tm.ms([&] { be->gelu(s->fc, (size_t)T * F); });
        acc[OP_FPROJ] += tm.ms([&] { gpt2_matmul_dispatch(be, s->ff, s->fc, ly->proj_w, qy?&qy->proj_w:nullptr, ly->proj_b, T, E, F); });
        acc[OP_ADD]   += tm.ms([&] { be->add(s->x, s->ff, (size_t)T * E); });
    }
    acc[OP_LN]     += tm.ms([&] { be->layernorm(s->ln, s->x, w->lnf_g, w->lnf_b, T, E); });
    acc[OP_LOGITS] += tm.ms([&] { gpt2_matmul_dispatch(be, d_logits, s->ln + (size_t)(T-1)*E, w->wte,
                                                       qw?&qw->wte:nullptr, nullptr, 1, V, E); });
}

static double median_of(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return n ? (n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2])) : 0.0;
}

// number of calls per category in one forward (for the per-call column)
static int op_calls(int op) {
    switch (op) {
        case OP_EMBED: case OP_LOGITS: return 1;
        case OP_LN:  return 2 * NL + 1;
        case OP_ADD: return 2 * NL;
        default:     return NL;
    }
}

static void run_at_T(OpTimer &tm, const GPT2Backend *be, const GPT2WeightsGPU *w, const int *d_ids,
                     int T, GPT2ScratchGPU *s, half *d_logits, int warmup, int iters, const char *label) {
    // ---- per-op attribution: median over `iters` instrumented forwards ----
    std::vector<std::vector<double>> per_op(N_OP);
    std::vector<double> sums;
    {
        double junk[N_OP];
        for (int i = 0; i < warmup; i++) { memset(junk, 0, sizeof(junk)); forward_timed(tm, be, w, d_ids, T, s, d_logits, junk); }
    }
    for (int i = 0; i < iters; i++) {
        double acc[N_OP]; memset(acc, 0, sizeof(acc));
        forward_timed(tm, be, w, d_ids, T, s, d_logits, acc);
        double tot = 0;
        for (int o = 0; o < N_OP; o++) { per_op[o].push_back(acc[o]); tot += acc[o]; }
        sums.push_back(tot);
    }

    // ---- independent whole-forward timing: the UNINSTRUMENTED path, no inner syncs ----
    GPT2CapsGPU cp; memset(&cp, 0, sizeof(cp)); cp.logits = d_logits; cp.logits_all = 0;
    BenchStats whole = cuda_bench([&] { gpt2_forward_cuda(be, w, d_ids, T, s, &cp); }, warmup, iters);

    double sum_med = median_of(sums);
    double attributed = 0;
    for (int o = 0; o < N_OP; o++) attributed += median_of(per_op[o]);

    printf("\n================ %s  (T=%d, backend=%s, warmup=%d, N=%d) ================\n",
           label, T, be->name, warmup, iters);
    printf("  %-19s %5s  %10s  %10s  %8s   %s\n", "op", "calls", "median ms", "ms/call", "% attrib", "shape / note");
    printf("  ------------------------------------------------------------------------------------------------\n");

    // sort ops by median descending for readability, but keep a stable index list
    int order[N_OP]; for (int o = 0; o < N_OP; o++) order[o] = o;
    std::sort(order, order + N_OP, [&](int x, int y) { return median_of(per_op[x]) > median_of(per_op[y]); });

    for (int i = 0; i < N_OP; i++) {
        int o = order[i];
        double m = median_of(per_op[o]);
        int c = op_calls(o);
        printf("  %-19s %5d  %10.4f  %10.4f  %7.2f%%   %s\n",
               OP_NAME[o], c, m, m / c, 100.0 * m / attributed, OP_SHAPE[o]);
    }
    printf("  ------------------------------------------------------------------------------------------------\n");
    printf("  %-19s %5s  %10.4f  %10s  %7.2f%%\n", "SUM(per-op)", "", sum_med, "", 100.0);

    // ---- VALIDITY CHECK (the honesty guard) ----
    double ratio = sum_med / whole.median;
    printf("\n  [validity] whole-forward (uninstrumented, cuda_bench): %.4f ms median (%.4f-%.4f, N=%d)\n",
           whole.median, whole.min, whole.max, whole.n);
    printf("  [validity] SUM(per-op) / whole = %.4f/%.4f = %.3fx  (serialization overhead: %+.2f%%)\n",
           sum_med, whole.median, ratio, 100.0 * (ratio - 1.0));
    if (ratio < 0.97)
        printf("  [validity] ** SUM < WHOLE by >3%%: UNATTRIBUTED TIME EXISTS -> per-op split NOT trustworthy. **\n");
    else if (ratio > 1.25)
        printf("  [validity] ** SUM > WHOLE by >25%%: serialization dominates -> shares are distorted; treat as weak evidence. **\n");
    else
        printf("  [validity] OK: sum-of-parts accounts for the whole (within per-op sync overhead).\n"
               "             -> relative shares are trustworthy. SUM is NOT a forward latency (serialized); use %s.\n",
               "the whole-forward number for latency");
}

int main(int argc, char **argv) {
    const char *wpath = argc > 1 ? argv[1] : "weights/gpt2_124m_fp32.bin";
    const char *meta  = argc > 2 ? argv[2] : "refdumps/meta.json";

    const GPT2Backend *be = gpt2_backend_by_name(getenv("GPT2_BACKEND"));
    cudaDeviceProp p; CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
    printf("==== per-op forward attribution (measure-before-optimize, DESIGN.md §9.1) ====\n");
    printf("device: %s  sm_%d%d  %d SM   backend: %s\n", p.name, p.major, p.minor, p.multiProcessorCount, be->name);
    printf("timing: every op bracketed by its own CUDA-event pair + sync AFTER (execution, not launch).\n");
    printf("        per-op syncs SERIALIZE the forward -> SUM(per-op) > true forward latency.\n");
    printf("        => use the split for RELATIVE SHARE only; latency comes from the uninstrumented run.\n");

    GPT2Weights wcpu;
    if (gpt2_load_weights(wpath, &wcpu)) { fprintf(stderr, "weight load failed\n"); return 1; }
    GPT2WeightsGPU w; gpt2_upload_fp16(&wcpu, &w);
    gpt2_free_weights(&wcpu);
    GPT2QWeightsGPU qw;                                  // STAGE 4: no-op unless the backend is quantized
    if (gpt2_quant_attach_if_needed(be, &w, &qw)) { fprintf(stderr, "int8 weight load failed\n"); return 1; }
    CUDA_CHECK(cudaDeviceSynchronize());

    char *mbuf = read_text(meta);
    static int eval_ids[GPT2_N_CTX];
    int Ne = parse_int_array(mbuf, "\"eval_ids\"", eval_ids, GPT2_N_CTX);
    free(mbuf);
    if (Ne < 512) { fprintf(stderr, "[prof] need >=512 eval ids, got %d\n", Ne); return 1; }

    GPT2ScratchGPU s; gpt2_scratch_alloc(&s, GPT2_N_CTX);
    int *d_ids = dmalloc<int>(GPT2_N_CTX);
    h2d(d_ids, eval_ids, GPT2_N_CTX >= 512 ? 512 : Ne);
    half *d_logits = dmalloc<half>(V);

    OpTimer tm;
    // P=512 is the headline: Stage 2's whole-forward GFLOP/s FELL from 619 (@128) to 497 (@512),
    // and the O(T^2) attention is the only op whose cost grows superlinearly in T. Confirm or refute.
    run_at_T(tm, be, &w, d_ids, 512, &s, d_logits, 10, 30, "PREFILL @P=512");
    run_at_T(tm, be, &w, d_ids, 128, &s, d_logits, 10, 30, "PREFILL @P=128");
    // recompute-decode's steady-state shape: the no-KV harness runs full forwards at M=ctx (34..161).
    run_at_T(tm, be, &w, d_ids, 161, &s, d_logits, 10, 30, "no-KV RECOMPUTE-decode step @ctx=161 (M>>1, prefill-shaped)");

    gpt2_scratch_free(&s); cudaFree(d_ids); cudaFree(d_logits); gpt2_free_gpu(&w);
    return 0;
}
