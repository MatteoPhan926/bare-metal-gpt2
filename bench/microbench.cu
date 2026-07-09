// microbench.cu — Phase -1 : pin the roofline denominators (DO FIRST)
//
// Measures, on THIS machine (RTX 4060 Laptop, sm_89, 105W):
//   1. Achieved DRAM copy bandwidth   (float4 grid-stride, read+write = 2N)  -> decode denominator
//   2. Achieved DRAM read bandwidth   (grid-stride reduction, read = 1N)     -> cross-check (weight streaming)
//   3. Achieved FP16 GEMM FLOP/s      (cuBLAS, fp16 in / fp32 accum, TENSOR) -> bug line + WMMA headroom
//   4. Achieved CUDA-CORE FLOP/s      (NO tensor cores)                      -> prefill denominator for
//                                                                               the kernels this engine
//                                                                               actually ships (no WMMA)
//   5. Ridge points = (3)/(1) for a future WMMA kernel, (4)/(1) for the shipped kernels
//
// WHY (4) EXISTS (added 2026-07-09). Every GEMM this engine ships -- Stage 2 tiled, Stage 3b flash,
// Stage 4 INT8 -- runs on CUDA cores; none uses WMMA, and weight-only INT8 cannot reach dp4a/INT8
// tensor cores at all (its activations stay fp16). So (3) is the wrong efficiency denominator for them.
// The CUDA-core peak was previously DERIVED from the clock (24 SM x 128 lanes x 2 FLOP x 2.61 GHz =
// 16.0 TFLOP/s). G2 says microbench, do not assume -- an assumed denominator is exactly what §6 avoided
// for the tensor peak. So it is measured here, two independent ways:
//   (4a) fp32 FMA issue rate, register-resident, zero memory traffic -> the HARDWARE ceiling.
//   (4b) cuBLAS GEMM with tensor cores DISABLED (CUBLAS_COMPUTE_*_PEDANTIC + CUBLAS_PEDANTIC_MATH)
//        -> the ACHIEVABLE non-tensor GEMM ceiling. This is the exact analog of how (3) pins 31.5,
//        and it -- not (4a) -- is the honest denominator for a %-of-roofline claim.
//
// Honesty rules baked in:
//   - buffers (256 MiB) >> L2 (32 MiB)  => we measure DRAM, not cache.
//   - CUDA-event timing WITH sync; warmup discarded; MEDIAN + min/max over many iters (never best-of-N).
//   - copy BW > 256 GB/s (theoretical) is flagged as a MEASUREMENT BUG, not a result.
//   - cuBLAS is a *measurement instrument* to establish the achievable compute ceiling
//     (the honest prefill denominator); the model engine's GEMMs are written from scratch elsewhere.
//
// Build:  nvcc -O3 -arch=sm_89 microbench.cu -o microbench.exe -lcublas
// Run  :  microbench.exe [all|bw|gemm]     (default: all)

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

#define CUDA_CHECK(x) do { cudaError_t e_=(x); if(e_!=cudaSuccess){ \
    printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1);} } while(0)
#define CUBLAS_CHECK(x) do { cublasStatus_t s_=(x); if(s_!=CUBLAS_STATUS_SUCCESS){ \
    printf("cuBLAS error %d at %s:%d\n", (int)s_, __FILE__, __LINE__); exit(1);} } while(0)

static double median(std::vector<double>& v){ std::sort(v.begin(), v.end()); size_t n=v.size();
    return n? (n%2? v[n/2] : 0.5*(v[n/2-1]+v[n/2])) : 0.0; }

// ---------------- bandwidth kernels ----------------
__global__ void copy_kernel(const float4* __restrict__ src, float4* __restrict__ dst, size_t n4){
    size_t i = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x*blockDim.x;
    for(; i<n4; i+=stride) dst[i] = src[i];
}
// read-only: reduce, dead-store guarded by a magic value so the compiler can't elide the loads
__global__ void read_kernel(const float4* __restrict__ src, size_t n4, float magic, float* out){
    size_t i = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x*blockDim.x;
    float acc = 0.f;
    for(; i<n4; i+=stride){ float4 v = src[i]; acc += v.x+v.y+v.z+v.w; }
    if(acc==magic) out[(size_t)blockIdx.x*blockDim.x+threadIdx.x] = acc; // never true -> no real write traffic
}
__global__ void fill_kernel(float* p, size_t n, float val){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x, s=(size_t)gridDim.x*blockDim.x;
    for(;i<n;i+=s) p[i]=val;
}

static void bench_bandwidth(){
    const size_t BYTES = (size_t)256*1024*1024;          // 256 MiB per buffer  (>> 32 MiB L2)
    const size_t N     = BYTES/sizeof(float);            // floats
    const size_t N4    = N/4;                            // float4 elements
    const int WARM=10, ITERS=50;
    const int block=256, grid=4096;

    float *src=nullptr,*dst=nullptr,*sink=nullptr;
    CUDA_CHECK(cudaMalloc(&src, BYTES));
    CUDA_CHECK(cudaMalloc(&dst, BYTES));
    CUDA_CHECK(cudaMalloc(&sink, (size_t)grid*block*sizeof(float)));
    fill_kernel<<<grid,block>>>(src, N, 1.0f);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t a,b; CUDA_CHECK(cudaEventCreate(&a)); CUDA_CHECK(cudaEventCreate(&b));

    // ---- copy (2N traffic) ----
    for(int i=0;i<WARM;i++) copy_kernel<<<grid,block>>>((float4*)src,(float4*)dst,N4);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<double> gbps_copy;
    for(int i=0;i<ITERS;i++){
        CUDA_CHECK(cudaEventRecord(a));
        copy_kernel<<<grid,block>>>((float4*)src,(float4*)dst,N4);
        CUDA_CHECK(cudaEventRecord(b));
        CUDA_CHECK(cudaEventSynchronize(b));
        float ms=0; CUDA_CHECK(cudaEventElapsedTime(&ms,a,b));
        double bytes = 2.0*BYTES;                        // read src + write dst
        gbps_copy.push_back(bytes/(ms*1e-3)/1e9);
    }
    std::sort(gbps_copy.begin(),gbps_copy.end());
    double cmed=median(gbps_copy), cmin=gbps_copy.front(), cmax=gbps_copy.back();

    // ---- read-only (1N traffic) ----
    for(int i=0;i<WARM;i++) read_kernel<<<grid,block>>>((float4*)src,N4,-123456.789f,sink);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<double> gbps_read;
    for(int i=0;i<ITERS;i++){
        CUDA_CHECK(cudaEventRecord(a));
        read_kernel<<<grid,block>>>((float4*)src,N4,-123456.789f,sink);
        CUDA_CHECK(cudaEventRecord(b));
        CUDA_CHECK(cudaEventSynchronize(b));
        float ms=0; CUDA_CHECK(cudaEventElapsedTime(&ms,a,b));
        double bytes = 1.0*BYTES;                        // read only
        gbps_read.push_back(bytes/(ms*1e-3)/1e9);
    }
    std::sort(gbps_read.begin(),gbps_read.end());
    double rmed=median(gbps_read), rmin=gbps_read.front(), rmax=gbps_read.back();

    const double THEO=256.0;
    printf("\n==== BANDWIDTH (buffer=256MiB/ea, block=%d grid=%d, WARM=%d ITERS=%d) ====\n",block,grid,WARM,ITERS);
    printf("COPY (2N r+w) : median %.1f  min %.1f  max %.1f GB/s   (%.1f%% of %.0f theo)\n",
           cmed,cmin,cmax, 100.0*cmed/THEO, THEO);
    printf("READ (1N)     : median %.1f  min %.1f  max %.1f GB/s   (%.1f%% of %.0f theo)\n",
           rmed,rmin,rmax, 100.0*rmed/THEO, THEO);
    if(cmed>THEO || rmed>THEO) printf("*** WARNING: measured BW > 256 GB/s theoretical => MEASUREMENT BUG (likely L2 hit). ***\n");
    printf("ACHIEVED_BW_COPY_GBPS=%.1f\nACHIEVED_BW_READ_GBPS=%.1f\n", cmed, rmed);

    cudaEventDestroy(a); cudaEventDestroy(b);
    cudaFree(src); cudaFree(dst); cudaFree(sink);
}

static double bench_gemm_one(cublasHandle_t h, int M,int N,int K,int WARM,int ITERS,
                             cublasComputeType_t ctype, double* out_tflops){
    __half *dA,*dB,*dC;
    CUDA_CHECK(cudaMalloc(&dA,(size_t)M*K*sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&dB,(size_t)K*N*sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&dC,(size_t)M*N*sizeof(__half)));
    // init to small values (content irrelevant to throughput)
    { std::vector<__half> tmp((size_t)M*K, __float2half(0.01f));
      CUDA_CHECK(cudaMemcpy(dA,tmp.data(),tmp.size()*sizeof(__half),cudaMemcpyHostToDevice)); }
    { std::vector<__half> tmp((size_t)K*N, __float2half(0.01f));
      CUDA_CHECK(cudaMemcpy(dB,tmp.data(),tmp.size()*sizeof(__half),cudaMemcpyHostToDevice)); }
    // alpha/beta type follows compute type: 16F -> __half, 32F -> float
    __half ah=__float2half(1.f), bh=__float2half(0.f); float af=1.f, bf=0.f;
    const void *alpha = (ctype==CUBLAS_COMPUTE_16F)?(const void*)&ah:(const void*)&af;
    const void *beta  = (ctype==CUBLAS_COMPUTE_16F)?(const void*)&bh:(const void*)&bf;
    // C(MxN) = A(MxK) * B(KxN), row-major emulated via column-major swap: compute C^T = B^T * A^T
    auto run=[&](){ CUBLAS_CHECK(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K, alpha, dB, CUDA_R_16F, N, dA, CUDA_R_16F, K, beta, dC, CUDA_R_16F, N,
        ctype, CUBLAS_GEMM_DEFAULT)); };
    for(int i=0;i<WARM;i++) run();
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t a,b; CUDA_CHECK(cudaEventCreate(&a)); CUDA_CHECK(cudaEventCreate(&b));
    std::vector<double> tf;
    double flops = 2.0*M*N*K;
    for(int i=0;i<ITERS;i++){
        CUDA_CHECK(cudaEventRecord(a));
        run();
        CUDA_CHECK(cudaEventRecord(b));
        CUDA_CHECK(cudaEventSynchronize(b));
        float ms=0; CUDA_CHECK(cudaEventElapsedTime(&ms,a,b));
        tf.push_back(flops/(ms*1e-3)/1e12);
    }
    double med=median(tf);
    std::sort(tf.begin(),tf.end());
    printf("  GEMM %5d^3 : median %.2f  min %.2f  max %.2f TFLOP/s\n", M, med, tf.front(), tf.back());
    cudaEventDestroy(a); cudaEventDestroy(b);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    *out_tflops = med;
    return med;
}

// ================= (4) CUDA-CORE (NON-tensor-core) COMPUTE PEAK =================
//
// (4a) fp32 FMA issue rate. Register-resident: no loads, no stores, no shared memory -> the ONLY thing
// this can measure is the FMA pipe. FMA_ILP independent dependency chains per thread hide the ~4-cycle
// FMA latency, so we measure ISSUE throughput, not latency. `a`,`b` come from the host so nothing is
// constant-folded; the accumulate is consumed by a store that provably never executes, so ptxas cannot
// dead-code the loop (verified: the kernel's SASS retains iters*FMA_ILP FFMAs).
#define FMA_ILP 32

// NOTE (2026-07-09): a clock64()-based "FLOP/SM/cycle" cross-check was written, run, and REMOVED. It
// reported 374 FLOP/SM/cycle against a hardware ceiling of 256 (146% of issue width) -- impossible. The
// cause is that clock64()'s tick rate is NOT calibrated on this device: it implied a 1739 MHz SM clock
// while the same kernel sustained 15.63 TFLOP/s, which 128 lanes x 24 SM cannot reach below ~2.55 GHz.
// (2610/1739 = 1.501, i.e. the counter appears to tick at ~2/3 of the SM clock here.) Rather than
// publish a number from an uncalibrated counter, issue width is computed below from instruments whose
// calibration IS established: CUDA events (wall time) and nvidia-smi (SM clock).
__global__ void k_fma_peak(float *out, int iters, float a, float b){
    float x[FMA_ILP];
    #pragma unroll
    for(int i=0;i<FMA_ILP;i++) x[i] = a + (float)i;
    for(int it=0; it<iters; ++it){
        #pragma unroll
        for(int i=0;i<FMA_ILP;i++) x[i] = fmaf(x[i], b, a);   // 1 FMA = 2 FLOP
    }
    float s = 0.f;
    #pragma unroll
    for(int i=0;i<FMA_ILP;i++) s += x[i];
    if(s == -1.0f) out[blockIdx.x*blockDim.x+threadIdx.x] = s;  // unreachable: keeps the FMAs alive
}

static double bench_fma_peak(const cudaDeviceProp& p){
    const int block  = 256;
    const int blocks = p.multiProcessorCount * 8;
    const int iters  = 40000;
    const int ITERS  = 50;
    float *sink; CUDA_CHECK(cudaMalloc(&sink, (size_t)blocks*block*sizeof(float)));
    auto run=[&](){ k_fma_peak<<<blocks,block>>>(sink, iters, 1.0000001f, 0.9999999f); };

    int active=0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, k_fma_peak, block, 0));
    printf("  [occupancy] %d resident blocks/SM x %d thr = %d thr/SM (device max %d)\n",
           active, block, active*block, p.maxThreadsPerMultiProcessor);

    // SUSTAINED WARMUP to thermal/clock steady state (BENCH_PROTOCOL §4: a cold-burst clock is not a
    // result). Without this the medians are BIMODAL across runs -- 13.45 TF at the sagged clock vs
    // 15.65 at boost -- because a 100%-FP32 kernel is exactly what makes an unlocked laptop clock move.
    {
        cudaEvent_t s0,s1; CUDA_CHECK(cudaEventCreate(&s0)); CUDA_CHECK(cudaEventCreate(&s1));
        float warmed=0.f; CUDA_CHECK(cudaEventRecord(s0));
        do { for(int r=0;r<5;r++) run();
             CUDA_CHECK(cudaEventRecord(s1)); CUDA_CHECK(cudaEventSynchronize(s1));
             CUDA_CHECK(cudaEventElapsedTime(&warmed,s0,s1));
        } while(warmed < 2500.f);
        CUDA_CHECK(cudaEventDestroy(s0)); CUDA_CHECK(cudaEventDestroy(s1));
    }

    cudaEvent_t a,b; CUDA_CHECK(cudaEventCreate(&a)); CUDA_CHECK(cudaEventCreate(&b));
    const double flops = 2.0*(double)FMA_ILP*(double)iters*(double)blocks*(double)block;
    std::vector<double> tf;
    for(int i=0;i<ITERS;i++){
        CUDA_CHECK(cudaEventRecord(a)); run();
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(b)); CUDA_CHECK(cudaEventSynchronize(b));
        float ms=0; CUDA_CHECK(cudaEventElapsedTime(&ms,a,b));
        tf.push_back(flops/(ms*1e-3)/1e12);
    }
    double med=median(tf); std::sort(tf.begin(),tf.end());
    printf("  fp32 FMA peak (regs only, ILP=%d, %d blk x %d thr, sustained): median %.2f  min %.2f  max %.2f TFLOP/s\n",
           FMA_ILP, blocks, block, med, tf.front(), tf.back());

    // Issue width from calibrated instruments only: wall TFLOP/s vs (SM x 128 lanes x 2 x SM_clock).
    // Report against BOTH the boost clock and the clock actually sustained here (record via nvidia-smi).
    for(double ghz : {2.610, 2.595, 2.250}){
        double ceil_tf = p.multiProcessorCount * 128.0 * 2.0 * ghz*1e9 / 1e12;
        printf("     vs 128-lane ceiling @ %.3f GHz = %5.2f TFLOP/s -> %.1f%% of issue width\n",
               ghz, ceil_tf, 100.0*med/ceil_tf);
    }
    double assumed = p.multiProcessorCount * 128.0 * 2.0 * 2.610e9 / 1e12;   // what the docs ASSUMED
    printf("  [vs assumed] docs assumed %.2f TFLOP/s (24x128x2x2.61GHz) -> measured/assumed = %.3f %s\n",
           assumed, med/assumed, (med < 0.95*assumed ? "** the assumption was OPTIMISTIC **" : "-> consistent"));
    cudaEventDestroy(a); cudaEventDestroy(b); cudaFree(sink);
    return med;
}

// (4b) achievable non-tensor GEMM. PEDANTIC compute types + CUBLAS_PEDANTIC_MATH forbid tensor cores.
// Returns 0.0 (and says so) if cuBLAS refuses the combination, rather than silently reporting a
// tensor-core number under a non-tensor label.
static double bench_gemm_pedantic(cublasHandle_t h, int S, cudaDataType_t abType,
                                  cublasComputeType_t ctype, const char* label){
    const int M=S,N=S,K=S, WARM=10, ITERS=40;
    size_t esz = (abType==CUDA_R_16F)? sizeof(__half) : sizeof(float);
    void *dA,*dB,*dC;
    CUDA_CHECK(cudaMalloc(&dA,(size_t)M*K*esz));
    CUDA_CHECK(cudaMalloc(&dB,(size_t)K*N*esz));
    CUDA_CHECK(cudaMalloc(&dC,(size_t)M*N*esz));
    CUDA_CHECK(cudaMemset(dA,0,(size_t)M*K*esz));
    CUDA_CHECK(cudaMemset(dB,0,(size_t)K*N*esz));
    float af=1.f, bf=0.f;

    auto once=[&](){ return cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
        &af, dB, abType, N, dA, abType, K, &bf, dC, abType, N, ctype, CUBLAS_GEMM_DEFAULT); };

    if(once()!=CUBLAS_STATUS_SUCCESS){
        printf("  %-34s : NOT SUPPORTED by cuBLAS on this device -> skipped (no number claimed)\n", label);
        cudaFree(dA);cudaFree(dB);cudaFree(dC); return 0.0;
    }
    for(int i=0;i<WARM;i++) once();
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t a,b; CUDA_CHECK(cudaEventCreate(&a)); CUDA_CHECK(cudaEventCreate(&b));
    double flops = 2.0*M*N*K;
    std::vector<double> tf;
    for(int i=0;i<ITERS;i++){
        CUDA_CHECK(cudaEventRecord(a)); once();
        CUDA_CHECK(cudaEventRecord(b)); CUDA_CHECK(cudaEventSynchronize(b));
        float ms=0; CUDA_CHECK(cudaEventElapsedTime(&ms,a,b));
        tf.push_back(flops/(ms*1e-3)/1e12);
    }
    double med=median(tf); std::sort(tf.begin(),tf.end());
    printf("  %-34s : median %.2f  min %.2f  max %.2f TFLOP/s   (%d^3)\n",
           label, med, tf.front(), tf.back(), S);
    cudaEventDestroy(a); cudaEventDestroy(b); cudaFree(dA);cudaFree(dB);cudaFree(dC);
    return med;
}

static double bench_cudacore(const cudaDeviceProp& p){
    printf("\n==== CUDA-CORE COMPUTE PEAK (NO tensor cores) -> prefill denominator for the shipped kernels ====\n");
    printf("-- (4a) hardware fp32 FMA ceiling: register-resident, zero memory traffic --\n");
    double fma = bench_fma_peak(p);

    printf("-- (4b) ACHIEVABLE non-tensor GEMM (cuBLAS, PEDANTIC compute type + CUBLAS_PEDANTIC_MATH) --\n");
    cublasHandle_t h; CUBLAS_CHECK(cublasCreate(&h));
    CUBLAS_CHECK(cublasSetMathMode(h, CUBLAS_PEDANTIC_MATH));   // belt+braces: forbid tensor-core paths
    double g16=0, g32=0;
    for(int s: {4096, 8192}){
        double v = bench_gemm_pedantic(h, s, CUDA_R_16F, CUBLAS_COMPUTE_32F_PEDANTIC, "fp16 in, fp32 acc, PEDANTIC");
        if(v>g16) g16=v;
    }
    for(int s: {4096, 8192}){
        double v = bench_gemm_pedantic(h, s, CUDA_R_32F, CUBLAS_COMPUTE_32F_PEDANTIC, "fp32 SGEMM, PEDANTIC");
        if(v>g32) g32=v;
    }
    cublasDestroy(h);

    double achievable = (g16>0? g16 : g32);
    printf("\nMEASURED_CUDACORE_FMA_PEAK_TFLOPS=%.2f\n", fma);
    printf("MEASURED_CUDACORE_GEMM_TFLOPS_F16IN=%.2f\n", g16);
    printf("MEASURED_CUDACORE_GEMM_TFLOPS_F32=%.2f\n", g32);
    if(achievable>0) printf("  GEMM/FMA efficiency = %.1f%%  (a real GEMM never reaches the raw FMA pipe)\n",
                            100.0*achievable/fma);
    printf("  SANITY: the non-tensor GEMM must be well BELOW the tensor GEMM (31.5 TF, §6).\n"
           "          If it is NOT, PEDANTIC failed to disable tensor cores -> the number is a lie.\n");
    return fma;
}

static double bench_gemm(){
    cublasHandle_t h; CUBLAS_CHECK(cublasCreate(&h));
    printf("\n==== FP16 GEMM (cuBLAS GemmEx, fp16 in, TENSOR op) ====\n");
    printf("-- FP32 accumulate (CUBLAS_COMPUTE_32F) : the tensor-core ceiling -> BUG LINE + WMMA headroom --\n");
    printf("   (NOT the efficiency denominator for this engine's kernels -- they use no WMMA; see mode 'cudacore')\n");
    int sizes[] = {1024,2048,4096,8192};
    double best32=0;
    for(int s: sizes){
        int iters = (s>=4096)?60:100;
        double tf; bench_gemm_one(h, s,s,s, 10, iters, CUBLAS_COMPUTE_32F, &tf);
        if(tf>best32) best32=tf;
    }
    printf("-- FP16 accumulate (CUBLAS_COMPUTE_16F) : documents headroom (worse numerics; not used by model) --\n");
    double best16=0;
    for(int s: {4096,8192}){
        double tf; bench_gemm_one(h, s,s,s, 10, 60, CUBLAS_COMPUTE_16F, &tf);
        if(tf>best16) best16=tf;
    }
    printf("ACHIEVED_FP16_TFLOPS_F32ACC=%.2f\n", best32);
    printf("ACHIEVED_FP16_TFLOPS_F16ACC=%.2f\n", best16);
    printf("F16ACC/F32ACC ratio = %.2fx\n", best16/best32);
    cublasDestroy(h);
    return best32;
}

int main(int argc,char**argv){
    const char* mode = (argc>1)? argv[1] : "all";
    cudaDeviceProp p; CUDA_CHECK(cudaGetDeviceProperties(&p,0));
    printf("Device: %s  sm_%d%d  %d SM  L2=%dMiB  memclk=%.0fMHz bus=%d-bit\n",
           p.name,p.major,p.minor,p.multiProcessorCount,p.l2CacheSize/(1024*1024),
           p.memoryClockRate/1000.0,p.memoryBusWidth);
    double theo_bw = 2.0*p.memoryClockRate*1000.0*(p.memoryBusWidth/8.0)/1e9;
    printf("Theoretical BW: %.1f GB/s\n", theo_bw);

    double bw=0, tf=0;
    bool doBW  = (strcmp(mode,"all")==0)||(strcmp(mode,"bw")==0);
    bool doGEMM= (strcmp(mode,"all")==0)||(strcmp(mode,"gemm")==0);
    bool doCC  = (strcmp(mode,"all")==0)||(strcmp(mode,"cudacore")==0);
    if(doBW)  bench_bandwidth();
    if(doGEMM) tf = bench_gemm();
    if(doCC)  bench_cudacore(p);

    if(strcmp(mode,"all")==0){
        // ridge point uses COPY bandwidth (conservative: read+write achievable BW)
        // recompute copy bw quickly-not; instead re-run is wasteful. We printed it above.
        printf("\n(Ridge points = TFLOPS / achieved_copy_BW, reported by the driver script that parses the\n"
               " ACHIEVED_*/MEASURED_* lines. TWO ridges: tensor (for a future WMMA kernel) and CUDA-core\n"
               " (which governs every kernel this engine currently ships).)\n");
    }
    printf("\nDONE mode=%s\n", mode);
    return 0;
}
