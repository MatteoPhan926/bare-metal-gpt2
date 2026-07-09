// kernels_tiled.cu — STAGE 2: tiled / shared-memory GEMM (the PREFILL lever, ROOFLINE director map).
//
// Replaces the naive one-thread-per-output matmul with a classic shared-memory tiled GEMM: each
// TILE×TILE thread block computes a TILE×TILE output tile, streaming TILE-wide K-slices of A and W
// through shared memory so every global element is reused TILE times → ~TILE× less GDDR traffic on
// the weight reads that the naive kernel wastes (adjacent naive threads read different W rows,
// uncoalesced). This is a CUDA-CORE GEMM: fp16 storage + fp32 accumulation, IDENTICAL numeric regime
// to the naive kernel / HF fp16 (NO tensor cores — WMMA is a LATER lever, not Stage 2).
//
// Same GEMM contract as naive:  C[M,N] = A[M,K] · W[N,K]^T + bias[N]   (W row n contiguous in k).
// Stage 2 changes EXACTLY ONE kernel; the other five ops are reused from the naive backend verbatim
// (built by copying GPT2_BACKEND_NAIVE and overriding only .matmul), so any gate delta localizes here.

#include "kernels.cuh"
#include "common.cuh"

#define TILE 16

// block (bx,by) owns C rows [by*TILE, +TILE) × cols [bx*TILE, +TILE); thread (tx,ty) owns one element.
// As holds A[row][k0+..] indexed [ty][k]; Bs holds W[col][k0+..] indexed [n_local=tx][k]. The +1 pad
// makes the strided Bs[tx][kk] read (tx varies within a warp) bank-conflict-free. Summation walks
// k = 0,1,2,... in the SAME order as the naive loop, so the fp32 partial sums are bit-identical to
// naive — only the bias (added last here vs first in naive) differs, ≪ the fp16 gate tolerance.
__global__ void k_matmul_tiled(half *C, const half *A, const half *W, const half *bias,
                               int M, int N, int K) {
    __shared__ float As[TILE][TILE + 1];
    __shared__ float Bs[TILE][TILE + 1];
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;                 // m (output row)
    int col = blockIdx.x * TILE + tx;                 // n (output col)
    int nB  = blockIdx.x * TILE + ty;                 // W row this thread stages into Bs
    float acc = 0.f;
    for (int k0 = 0; k0 < K; k0 += TILE) {
        int k = k0 + tx;                              // coalesced: adjacent tx -> adjacent k
        As[ty][tx] = (row < M && k < K) ? __half2float(A[(size_t)row * K + k]) : 0.f;
        Bs[ty][tx] = (nB  < N && k < K) ? __half2float(W[(size_t)nB  * K + k]) : 0.f;
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < TILE; kk++) acc += As[ty][kk] * Bs[tx][kk];
        __syncthreads();
    }
    if (row < M && col < N)
        C[(size_t)row * N + col] = __float2half(bias ? acc + __half2float(bias[col]) : acc);
}

// Exported (not static) so later stages can compose a backend from the naive struct + this matmul
// WITHOUT reading GPT2_BACKEND_TILED, which is dynamically initialized — see make_tiled_backend().
void gpt2_matmul_tiled(half *C, const half *A, const half *W, const half *bias,
                       int M, int N, int K) {
    dim3 blk(TILE, TILE);
    dim3 grd(CEIL_DIV(N, TILE), CEIL_DIV(M, TILE));
    k_matmul_tiled<<<grd, blk>>>(C, A, W, bias, M, N, K);
}

// Stage 2 backend = naive backend with ONLY .matmul swapped for the tiled GEMM. Copying the naive
// struct (constant-initialized in its TU, so ready before this dynamic init) reuses embed/LayerNorm/
// attention/gelu/add byte-for-byte — Stage 2 is the tiled GEMM and nothing else.
//
// NOTE for later stages: this object is DYNAMICALLY initialized, so another TU must not read it in
// its own dynamic initializer (unspecified order across TUs). Compose from GPT2_BACKEND_NAIVE (which
// IS constant-initialized) plus gpt2_matmul_tiled instead.
static GPT2Backend make_tiled_backend() {
    GPT2Backend b = GPT2_BACKEND_NAIVE;
    b.name   = "tiled";
    b.matmul = gpt2_matmul_tiled;
    return b;
}
const GPT2Backend GPT2_BACKEND_TILED = make_tiled_backend();
