// ab_forward.cu — STAGE 3 KILL-TEST harness (CLAUDE.md §5, §9.3).
//
// Answers exactly one question per run: does backend B beat backend A by MORE THAN THE NOISE?
// If not, the fusion is REVERTED — "should be faster" is not "is faster".
//
// Design choices, each aimed at a specific way this measurement could lie:
//   * INTERLEAVED A/B (a_i, b_i, a_{i+1}, b_{i+1}, ...): both backends see the same clock/thermal
//     state within microseconds of each other, so a laptop downclock mid-run cannot masquerade as a
//     speedup. (Same trick as the Stage-2 M-sweep.)
//   * SUSTAINED WARMUP to boost-clock steady state before any timing (clock-lock needs admin here).
//   * cuda_time_once_ms per iteration: sync-enforced, so we time EXECUTION, not launch.
//   * MEDIAN + min-max, never best-of-N. Spread is printed so "above noise" is checkable, not asserted:
//     we report the ratio of medians AND whether A's and B's [min,max] intervals overlap.
//   * Both a WHOLE-FORWARD timing (what the stage claims) and an ISOLATED-OP timing (where the claim
//     comes from), so a whole-forward win can be attributed rather than assumed.
//
// Run from repo root:  ab_forward.exe <backendA> <backendB> [iters]
//   e.g. ab_forward.exe tiled flash 50

#include "kernels.cuh"
#include "common.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cmath>

#define E  GPT2_N_EMBD
#define V  GPT2_VOCAB

static char *read_text(const char *path) {
    FILE *f = fopen(path, "rb"); if (!f) { fprintf(stderr, "[ab] cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char *b = (char *)malloc(n + 1);
    if (fread(b, 1, n, f) != (size_t)n) { fprintf(stderr, "short read %s\n", path); exit(1); }
    b[n] = 0; fclose(f); return b;
}
static int parse_int_array(const char *buf, const char *key, int *out, int maxn) {
    const char *p = strstr(buf, key); if (!p) { fprintf(stderr, "[ab] key %s missing\n", key); exit(1); }
    p = strchr(p, '['); if (!p) exit(1); p++;
    int n = 0;
    while (*p && *p != ']') {
        while (*p && *p != ']' && *p != '-' && (*p < '0' || *p > '9')) p++;
        if (!*p || *p == ']') break;
        char *end; long v = strtol(p, &end, 10); if (end == p) { p++; continue; }
        if (n < maxn) out[n] = (int)v; n++; p = end;
    }
    return n;
}
struct Stat { double med, lo, hi; };
static Stat stat_of(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    Stat s; s.lo = v.front(); s.hi = v.back();
    s.med = n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
    return s;
}

// Print one A/B comparison with an explicit above-noise verdict.
static void report(const char *what, const char *na, const char *nb, Stat a, Stat b) {
    double ratio = a.med / b.med;                       // >1 => B faster
    double spread_a = 100.0 * (a.hi - a.lo) / a.med;
    double spread_b = 100.0 * (b.hi - b.lo) / b.med;
    bool overlap = !(b.hi < a.lo || a.hi < b.lo);       // do the [min,max] intervals touch at all?
    printf("  %-34s  %-6s %9.4f ms [%.4f-%.4f] (+-%.2f%%)\n", what, na, a.med, a.lo, a.hi, spread_a);
    printf("  %-34s  %-6s %9.4f ms [%.4f-%.4f] (+-%.2f%%)\n", "", nb, b.med, b.lo, b.hi, spread_b);
    printf("  %-34s  %s %.3fx   %s\n", "", "speedup B vs A:", ratio,
           overlap ? "** [min,max] OVERLAP -> NOT above noise **"
                   : (ratio > 1.0 ? "disjoint ranges -> above noise (B faster)"
                                  : "disjoint ranges -> above noise (B SLOWER)"));
    printf("\n");
}

int main(int argc, char **argv) {
    const char *na = argc > 1 ? argv[1] : "tiled";
    const char *nb = argc > 2 ? argv[2] : "flash";
    const int iters = argc > 3 ? atoi(argv[3]) : 50;
    const GPT2Backend *A = gpt2_backend_by_name(na);
    const GPT2Backend *B = gpt2_backend_by_name(nb);
    if (A == B) { fprintf(stderr, "[ab] A and B resolve to the same backend (%s)\n", A->name); return 1; }
    na = A->name; nb = B->name;

    cudaDeviceProp p; CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
    printf("==== A/B kill-test: %s (A) vs %s (B) ====\n", na, nb);
    printf("device: %s  sm_%d%d  %d SM\n", p.name, p.major, p.minor, p.multiProcessorCount);
    printf("method: interleaved A/B (same thermal state), sync-enforced timer, median + min-max,\n");
    printf("        iters=%d, sustained warmup to boost clocks. Never best-of-N.\n", iters);
    printf("verdict rule (CLAUDE.md §5): no speedup above noise -> REVERT the fusion.\n");

    GPT2Weights wcpu;
    if (gpt2_load_weights("weights/gpt2_124m_fp32.bin", &wcpu)) { fprintf(stderr, "weight load failed\n"); return 1; }
    GPT2WeightsGPU w; gpt2_upload_fp16(&wcpu, &w);
    gpt2_free_weights(&wcpu);
    // Attach packed INT8 weights if EITHER backend needs them. Both share `w`; the fp16 backend has
    // .matmul_q == NULL, so gpt2_matmul_dispatch keeps it on the fp16 path regardless.
    GPT2QWeightsGPU qw;
    if (gpt2_quant_attach_if_needed(A, &w, &qw) || (!A->matmul_q && gpt2_quant_attach_if_needed(B, &w, &qw))) {
        fprintf(stderr, "int8 weight load failed\n"); return 1;
    }

    char *mbuf = read_text("refdumps/meta.json");
    static int eval_ids[GPT2_N_CTX];
    parse_int_array(mbuf, "\"eval_ids\"", eval_ids, GPT2_N_CTX);
    free(mbuf);

    GPT2ScratchGPU s; gpt2_scratch_alloc(&s, GPT2_N_CTX);
    int *d_ids = dmalloc<int>(GPT2_N_CTX);
    h2d(d_ids, eval_ids, 512);
    half *d_logits = dmalloc<half>(V);
    half *d_logits_all = dmalloc<half>((size_t)512 * V);   // scratch for the isolated head matmul (M<=512)
    GPT2CapsGPU cp; memset(&cp, 0, sizeof(cp)); cp.logits = d_logits; cp.logits_all = 0;

    // ---- sustained warmup (~1.5 s) so both backends time at a settled boost clock ----
    {
        cudaEvent_t s0, s1; CUDA_CHECK(cudaEventCreate(&s0)); CUDA_CHECK(cudaEventCreate(&s1));
        float warmed = 0.f; CUDA_CHECK(cudaEventRecord(s0));
        do {
            for (int r = 0; r < 3; r++) {
                gpt2_forward_cuda(A, &w, d_ids, 256, &s, &cp);
                gpt2_forward_cuda(B, &w, d_ids, 256, &s, &cp);
            }
            CUDA_CHECK(cudaEventRecord(s1)); CUDA_CHECK(cudaEventSynchronize(s1));
            CUDA_CHECK(cudaEventElapsedTime(&warmed, s0, s1));
        } while (warmed < 1500.f);
        CUDA_CHECK(cudaEventDestroy(s0)); CUDA_CHECK(cudaEventDestroy(s1));
    }

    // ---- WHOLE FORWARD, interleaved, at the three shapes that matter ----
    const int Ts[] = {128, 512, 161};
    const char *lbl[] = {"whole forward @P=128", "whole forward @P=512",
                         "whole forward @ctx=161 (recompute-dec)"};
    printf("\n---- whole forward (the stage claim) ----\n\n");
    for (int ti = 0; ti < 3; ti++) {
        int T = Ts[ti];
        auto fa = [&] { gpt2_forward_cuda(A, &w, d_ids, T, &s, &cp); };
        auto fb = [&] { gpt2_forward_cuda(B, &w, d_ids, T, &s, &cp); };
        for (int i = 0; i < 5; i++) { fa(); fb(); }                  // per-shape rewarm
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<double> va, vb;
        for (int i = 0; i < iters; i++) { va.push_back(cuda_time_once_ms(fa));   // adjacent in time
                                          vb.push_back(cuda_time_once_ms(fb)); }
        CUDA_CHECK(cudaGetLastError());
        report(lbl[ti], na, nb, stat_of(va), stat_of(vb));
    }

    // ---- ISOLATED ATTENTION (where a flash win must come from, if the whole-forward win is real) ----
    // qkv scratch is already populated by the warmup forwards; values are irrelevant to timing here,
    // and both backends read the identical buffer.
    printf("---- isolated attention op (attribution of the above) ----\n\n");
    for (int ti = 0; ti < 3; ti++) {
        int T = Ts[ti];
        auto fa = [&] { A->attention(s.att, s.qkv, T, E, GPT2_N_HEAD, GPT2_HEAD_DIM); };
        auto fb = [&] { B->attention(s.att, s.qkv, T, E, GPT2_N_HEAD, GPT2_HEAD_DIM); };
        for (int i = 0; i < 5; i++) { fa(); fb(); }
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<double> va, vb;
        for (int i = 0; i < iters; i++) { va.push_back(cuda_time_once_ms(fa));
                                          vb.push_back(cuda_time_once_ms(fb)); }
        CUDA_CHECK(cudaGetLastError());
        char what[64]; snprintf(what, sizeof(what), "attention T=%d (1 call)", T);
        report(what, na, nb, stat_of(va), stat_of(vb));
    }

    // ---- ISOLATED WEIGHT MATMUL (where an INT8 win must come from) ----
    // Dispatched through gpt2_matmul_dispatch, so an isolated op takes exactly the fp16/INT8 path the
    // whole forward takes. Two shapes with very different weight-reuse factors:
    //   qkv  [M,2304]x[2304,768] -- each W element re-staged ceil(M/16)x -> reuse hides weight traffic
    //   head [M,50257]x[50257,768] -- 77 MB fp16 >> 32 MiB L2, and at M=1 there is NO reuse at all:
    //       the true GEMV shape of a KV-cache decode step (Stage 5). This is where halving the bytes
    //       should show up first, and it is the honest preview of the Stage-5 decode payoff.
    printf("---- isolated weight matmul (attribution; INT8 buys traffic, not math) ----\n\n");
    {
        const GPT2LayerGPU  *ly = &w.layers[0];
        const GPT2QLayerGPU *qy = w.q ? &w.q->layers[0] : nullptr;
        const GPT2QW        *qh = w.q ? &w.q->wte       : nullptr;
        // M=16 vs M=1 is the FALSIFIABLE test of "compute-bound on wasted work": a 16x16 tile computes
        // 16 output rows whether or not M=1 needs them, so if arithmetic (not weight traffic) sets the
        // time, M=1 and M=16 must cost the SAME. If instead the kernel were weight-BW-bound at M=1,
        // INT8 (half the bytes) would show ~2x there. Both predictions are checked below.
        const int NM = 5;
        const int Ms[NM] = {128, 512, 161, 16, 1};
        for (int mi = 0; mi < NM; mi++) {
            int M = Ms[mi];
            auto fa = [&] { gpt2_matmul_dispatch(A, s.qkv, s.ln, ly->qkv_w, qy?&qy->qkv_w:nullptr, ly->qkv_b, M, GPT2_QKV_DIM, E); };
            auto fb = [&] { gpt2_matmul_dispatch(B, s.qkv, s.ln, ly->qkv_w, qy?&qy->qkv_w:nullptr, ly->qkv_b, M, GPT2_QKV_DIM, E); };
            for (int i = 0; i < 5; i++) { fa(); fb(); }
            CUDA_CHECK(cudaDeviceSynchronize());
            std::vector<double> va, vb;
            for (int i = 0; i < iters; i++) { va.push_back(cuda_time_once_ms(fa));
                                              vb.push_back(cuda_time_once_ms(fb)); }
            CUDA_CHECK(cudaGetLastError());
            char what[64]; snprintf(what, sizeof(what), "qkv matmul M=%d (N=2304 K=768)", M);
            report(what, na, nb, stat_of(va), stat_of(vb));
        }
        for (int mi = 0; mi < NM; mi++) {
            int M = Ms[mi];
            auto fa = [&] { gpt2_matmul_dispatch(A, d_logits_all, s.ln, w.wte, qh, nullptr, M, V, E); };
            auto fb = [&] { gpt2_matmul_dispatch(B, d_logits_all, s.ln, w.wte, qh, nullptr, M, V, E); };
            for (int i = 0; i < 5; i++) { fa(); fb(); }
            CUDA_CHECK(cudaDeviceSynchronize());
            std::vector<double> va, vb;
            for (int i = 0; i < iters; i++) { va.push_back(cuda_time_once_ms(fa));
                                              vb.push_back(cuda_time_once_ms(fb)); }
            CUDA_CHECK(cudaGetLastError());
            char what[64]; snprintf(what, sizeof(what), "logits head M=%d (N=50257 K=768)", M);
            report(what, na, nb, stat_of(va), stat_of(vb));
        }
    }

    gpt2_scratch_free(&s); cudaFree(d_ids); cudaFree(d_logits); cudaFree(d_logits_all); gpt2_free_gpu(&w);
    gpt2_quant_free(&qw);
    return 0;
}
