// common.cuh — STAGE 1 CUDA infra: error checking, a SYNC-ENFORCING timing helper, device buffers.
//
// The timing helper exists to make BENCH_PROTOCOL §7 bug #1 (timing an async launch instead of
// execution) hard to write by accident: it syncs the device BEFORE the timed region and
// cudaEventSynchronize's AFTER every iteration, so the elapsed time is real GPU execution.
// (The #1 source of fake CUDA speedups — CLAUDE.md build traps.)
#ifndef GPT2_COMMON_CUH
#define GPT2_COMMON_CUH

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CUDA_CHECK(x) do { cudaError_t e_=(x); if(e_!=cudaSuccess){ \
    fprintf(stderr,"CUDA error %s at %s:%d\n",cudaGetErrorString(e_),__FILE__,__LINE__); exit(1);} } while(0)

#define CEIL_DIV(a,b) (((a)+(b)-1)/(b))

struct BenchStats { double median, min, max; int n; };

// Bench a callable that enqueues GPU work on the default stream. warmup iters are discarded; then
// each timed iter is bracketed by event records with a sync AFTER -> we time EXECUTION, not launch.
// Never best-of-N: returns median + min/max (BENCH_PROTOCOL §6).
template<class F>
static BenchStats cuda_bench(F launch, int warmup, int iters) {
    for (int i = 0; i < warmup; i++) launch();
    CUDA_CHECK(cudaDeviceSynchronize());                 // BEFORE: device idle -> next event a is a true start
    cudaEvent_t a, b; CUDA_CHECK(cudaEventCreate(&a)); CUDA_CHECK(cudaEventCreate(&b));
    std::vector<double> ms;
    for (int i = 0; i < iters; i++) {
        CUDA_CHECK(cudaEventRecord(a));
        launch();
        CUDA_CHECK(cudaGetLastError());                  // catch launch-config errors at the launch site, not downstream
        CUDA_CHECK(cudaEventRecord(b));
        CUDA_CHECK(cudaEventSynchronize(b));             // AFTER: wait for real completion
        float t = 0; CUDA_CHECK(cudaEventElapsedTime(&t, a, b));
        ms.push_back((double)t);
    }
    CUDA_CHECK(cudaEventDestroy(a)); CUDA_CHECK(cudaEventDestroy(b));
    std::sort(ms.begin(), ms.end());
    BenchStats s; s.n = iters; s.min = ms.front(); s.max = ms.back();
    s.median = iters % 2 ? ms[iters/2] : 0.5*(ms[iters/2-1] + ms[iters/2]);
    return s;
}

// Time ONE launch (already-warmed) with a sync after — for the growing-context decode loop where
// each step has a different T (so a fixed-iters bench doesn't fit). Still syncs -> real execution.
template<class F>
static double cuda_time_once_ms(F launch) {
    cudaEvent_t a, b; CUDA_CHECK(cudaEventCreate(&a)); CUDA_CHECK(cudaEventCreate(&b));
    CUDA_CHECK(cudaEventRecord(a));
    launch();
    CUDA_CHECK(cudaGetLastError());                  // catch launch-config errors at the launch site, not downstream
    CUDA_CHECK(cudaEventRecord(b));
    CUDA_CHECK(cudaEventSynchronize(b));
    float t = 0; CUDA_CHECK(cudaEventElapsedTime(&t, a, b));
    CUDA_CHECK(cudaEventDestroy(a)); CUDA_CHECK(cudaEventDestroy(b));
    return (double)t;
}

template<class T> static T*   dmalloc(size_t n)                 { T* p=nullptr; CUDA_CHECK(cudaMalloc(&p, n*sizeof(T))); return p; }
template<class T> static void h2d(T* d, const T* h, size_t n)   { CUDA_CHECK(cudaMemcpy(d, h, n*sizeof(T), cudaMemcpyHostToDevice)); }
template<class T> static void d2h(T* h, const T* d, size_t n)   { CUDA_CHECK(cudaMemcpy(h, d, n*sizeof(T), cudaMemcpyDeviceToHost)); }

#endif // GPT2_COMMON_CUH
