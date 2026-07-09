// profile_matmul.cu — STAGE 2 diagnostic. Isolates the matmul kernel to MEASURE (not infer) where the
// tiled backend's speedup comes from. Two modes:
//
//   profile_matmul.exe [N K]        -> PROFILE mode: exactly ONE launch of naive k_matmul then ONE of
//                                       tiled k_matmul_tiled at M=1, so an ncu run reports 2 clean
//                                       kernels for the global-load sectors/request access-pattern check.
//   profile_matmul.exe bw [N K]     -> BW / M-SWEEP mode (no admin): warmup + INTERLEAVED timed loop of
//                                       each kernel via the verified cuda_time_once_ms, across a range of
//                                       M. Shows the tiled/naive advantage as a function of M.
//
// WHY the sweep: the Stage-2 harness "decode" has NO KV cache yet (that is Stage 5) — it FULL-RECOMPUTES
// the growing sequence each step, so its GEMMs run at M = context length (~34..161), NOT M=1. Only a
// KV-cached decode (or the logits head) is a true M=1 GEMV. Tiling-for-REUSE scales with M (each W tile
// is reused across M rows); at M=1 there is zero reuse, so only coalescing remains. The sweep separates
// the two: ratio ~1x at M=1 (coalescing only) rising toward the recompute-decode ratio at M~128-161.
// Default shape is qkv (N=2304, K=768); pass N K to override (e.g. 3072 768 for c_fc). Inputs are zeroed
// (values are irrelevant to the access pattern / timing; CUDA cores issue every load anyway).

#include "kernels.cuh"
#include "common.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>

static double median_of(std::vector<double> v) {           // median + spread, never best-of-N (BENCH_PROTOCOL)
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return n ? (n % 2 ? v[n/2] : 0.5*(v[n/2-1] + v[n/2])) : 0.0;
}

int main(int argc, char **argv) {
    bool bw = (argc > 1 && strcmp(argv[1], "bw") == 0);     // "bw": no-admin M-sweep; else ncu profile (M=1)
    int sh  = bw ? 1 : 0;
    int N = argc > 1 + sh ? atoi(argv[1 + sh]) : 2304;      // output cols (qkv projection by default)
    int K = argc > 2 + sh ? atoi(argv[2 + sh]) : 768;       // contraction dim

    const int maxM = bw ? 512 : 1;
    half *A = dmalloc<half>((size_t)maxM * K), *W = dmalloc<half>((size_t)N * K),
         *C = dmalloc<half>((size_t)maxM * N), *bias = dmalloc<half>(N);
    CUDA_CHECK(cudaMemset(A, 0, (size_t)maxM * K * sizeof(half)));
    CUDA_CHECK(cudaMemset(W, 0, (size_t)N * K * sizeof(half)));
    CUDA_CHECK(cudaMemset(C, 0, (size_t)maxM * N * sizeof(half)));
    CUDA_CHECK(cudaMemset(bias, 0, (size_t)N * sizeof(half)));
    CUDA_CHECK(cudaDeviceSynchronize());

    if (!bw) {
        // ---- ncu PROFILE mode: one launch each at M=1 so the ncu report is exactly 2 kernels ----
        GPT2_BACKEND_NAIVE.matmul(C, A, W, bias, 1, N, K);   // -> k_matmul       (strided W, uncoalesced)
        GPT2_BACKEND_TILED.matmul(C, A, W, bias, 1, N, K);   // -> k_matmul_tiled (contiguous W, coalesced)
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        printf("profiled decode-shaped GEMV  M=1 N=%d K=%d  (naive k_matmul + tiled k_matmul_tiled)\n", N, K);
        cudaFree(A); cudaFree(W); cudaFree(C); cudaFree(bias);
        return 0;
    }

    // ---- no-admin M-SWEEP: tiled/naive advantage vs M on this GEMM shape ----
    // Clock-lock needs admin (denied on this laptop), so: (1) a ~1.2s sustained warmup boosts and holds
    // the clock past the idle->boost ramp (the #1 laptop repro trap), and (2) timing is INTERLEAVED
    // (naive_i then tiled_i, adjacent) so both kernels see the same clock state -> the RATIO is robust
    // even if absolute GB/s is L2-fed/overhead-bound at tiny M (W=N*K*2 fits the 32MB L2).
    const int Ms[] = {1, 4, 16, 32, 64, 128, 161, 512};
    const int nM = (int)(sizeof(Ms)/sizeof(Ms[0]));
    {   // ~1.2s sustained warmup at a mid M to stabilize the boost clock
        cudaEvent_t s0, s1; CUDA_CHECK(cudaEventCreate(&s0)); CUDA_CHECK(cudaEventCreate(&s1));
        float warmed = 0.f; CUDA_CHECK(cudaEventRecord(s0));
        do {
            for (int r = 0; r < 40; r++) { GPT2_BACKEND_NAIVE.matmul(C, A, W, bias, 128, N, K);
                                           GPT2_BACKEND_TILED.matmul(C, A, W, bias, 128, N, K); }
            CUDA_CHECK(cudaEventRecord(s1)); CUDA_CHECK(cudaEventSynchronize(s1));
            CUDA_CHECK(cudaEventElapsedTime(&warmed, s0, s1));
        } while (warmed < 1200.f);
        CUDA_CHECK(cudaEventDestroy(s0)); CUDA_CHECK(cudaEventDestroy(s1));
    }

    printf("no-admin matmul M-sweep  (qkv shape N=%d K=%d)  interleaved naive/tiled, clocks warmed\n", N, K);
    printf("M=1 is the only true GEMV (KV-cached decode / logits head); the Stage-2 no-KV \"decode\"\n");
    printf("full-recomputes the sequence -> its GEMMs run at M=ctx(~34..161), the tiling-for-reuse regime.\n\n");
    const int ITERS = 100;
    for (int mi = 0; mi < nM; mi++) {
        int Mv = Ms[mi];
        auto na = [&]{ GPT2_BACKEND_NAIVE.matmul(C, A, W, bias, Mv, N, K); };
        auto ti = [&]{ GPT2_BACKEND_TILED.matmul(C, A, W, bias, Mv, N, K); };
        for (int i = 0; i < 5; i++) { na(); ti(); }         // short per-M rewarm
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<double> tn, tt;
        for (int i = 0; i < ITERS; i++) { tn.push_back(cuda_time_once_ms(na));   // INTERLEAVED: adjacent in time
                                          tt.push_back(cuda_time_once_ms(ti)); }
        CUDA_CHECK(cudaGetLastError());
        double mn = median_of(tn), mt = median_of(tt);
        std::sort(tn.begin(), tn.end()); std::sort(tt.begin(), tt.end());
        printf("  M=%-4d  naive %8.4f ms [%.4f-%.4f]  tiled %8.4f ms [%.4f-%.4f]  tiled/naive %5.2fx\n",
               Mv, mn, tn.front(), tn.back(), mt, tt.front(), tt.back(), mn/mt);
        if (Mv == 1) {                                       // the user-requested useful-W effective BW at M=1
            double useful = (double)N * K * 2.0;             // fp16 weight, read once = minimum GEMV traffic
            printf("         [M=1 useful-W BW] naive %.1f GB/s  tiled %.1f GB/s  (useful_W=%.2f MB; L2-fed, overhead-bound)\n",
                   useful/(mn*1e-3)/1e9, useful/(mt*1e-3)/1e9, useful/1e6);
        }
    }

    cudaFree(A); cudaFree(W); cudaFree(C); cudaFree(bias);
    return 0;
}
