// kernels_fused.cu — STAGE 3: fused kernels.
//
//   (3b) flash-style causal attention — online softmax, NO materialized [T,T] score matrix.
//   (3a) fused LayerNorm+matmul                                        [next; kill-tested separately]
//
// WHY 3b FIRST (measured, not assumed — DESIGN.md §9.1, bench/profile_forward.cu):
//   prefill@512, tiled backend, per-op medians (N=30), sum-of-parts/whole = 1.005x (validated split):
//     attention 83.15 ms = 44.80% of the forward   <-- the single largest op, and the only one whose
//     matmul_ffnproj 17.96% | matmul_fc 17.84% |       share GROWS with T (23.51% @128 -> 44.80% @512):
//     matmul_qkv 13.42% | attn_proj 4.59% |            the O(T^2)-vs-O(T) signature.
//     layernorm 0.28% | add 0.17% | gelu 0.16%     <-- 3a's Amdahl ceiling is <1% at prefill.
//
// WHAT MAKES THE NAIVE ATTENTION SLOW (both fixed here):
//   1. TWO passes over K, recomputing every q·k dot (pass 1 = running max; pass 2 = softmax numerator).
//      Online softmax folds the max into a single pass -> half the dot-product work, half the K reads.
//   2. `float acc[GPT2_HEAD_DIM]` per thread, dynamically indexed -> it cannot live in registers and is
//      placed in LOCAL memory, touched on every accumulate. MEASURED (nvcc -Xptxas -v, sm_89):
//        k_attention       256 bytes stack frame, 40 regs      <- 256 B == float acc[64], per thread
//        k_attention_flash   0 bytes stack frame, 40 regs, 19744 B smem
//      (ptxas reports 0 "spill" bytes for the naive kernel because acc is a *declared* local array
//       rather than a register spill — same DRAM-backed local space, different label.)
//      Here each thread owns exactly TWO head dims (D=64 over a 32-lane warp) -> acc is 2 registers.
//
// THE FLASH TILING (BR=8 queries/block, BC=32 keys/tile):
//   grid (ceil(T/BR), H); block (32, BR) = 256 threads = BR warps. Warp w owns query i = i0+w; lane j
//   owns key j0+j inside the tile, and head dims {lane, lane+32} of the output accumulator.
//   K and V tiles are staged in shared memory ONCE per block and reused by all BR warps (BR x fewer
//   global reads of K/V than a one-block-per-query scheme). Scores never leave shared/registers: at
//   most BR x BC = 256 of them exist at any instant, vs the naive T x T = 262,144 @512.
//   Row stride D+1 makes the strided Ks[lane][d] / Vs[j][lane] reads bank-conflict-free.
//
// ONLINE SOFTMAX (the numerically subtle part — gate (a) is the localizer if the rescale is wrong):
//   Keep running (m, l, acc). For each tile: m_new = max(m, tile_max); alpha = exp(m - m_new);
//   l = l*alpha + sum_j p_j ; acc = acc*alpha + sum_j p_j * v_j , with p_j = exp(s_j - m_new).
//   Masked lanes get s_j = -inf -> p_j = exp(-inf) = 0 exactly (no NaN). A tile is SKIPPED entirely
//   when j0 > i, so m_new is always finite when the rescale runs -> exp(-inf - (-inf)) never happens.
//   fp16 in, fp32 accumulate — the SAME numeric regime as the naive kernel and HF fp16.
//
// Stage 3 changes EXACTLY ONE kernel at a time. The flash backend = the Stage-2 tiled backend with
// ONLY .attention overridden, so every gate/timing delta localizes to attention.

#include "kernels.cuh"
#include "common.cuh"
#include <math.h>
#include <math_constants.h>   // CUDART_INF_F: an explicit +inf. MSVC's INFINITY macro is an overflowing
                              // literal ((float)1e300) -> compiles to inf, but warns. The -inf sentinel
                              // is load-bearing for the causal mask (exp(-inf - m) must be exactly 0),
                              // so it is spelled out rather than left to an overflow.

// ---------------- (3b) flash-style causal attention ----------------

#define FA_BR 8      // queries per block (one warp each)
#define FA_BC 32     // keys per tile (one per lane)
#define FA_D  64     // head dim; == GPT2_HEAD_DIM (static_assert below)

static_assert(GPT2_HEAD_DIM == FA_D, "flash attention specialized for head_dim=64 (2 dims/lane)");

__global__ void k_attention_flash(half *att, const half *qkv, int T, int E, int H, int D) {
    const int lane = threadIdx.x;              // 0..31 : key j0+lane, and head dims {lane, lane+32}
    const int w    = threadIdx.y;              // 0..BR-1 : which query this warp owns
    const int h    = blockIdx.y;
    const int i0   = blockIdx.x * FA_BR;
    const int i    = i0 + w;                   // this warp's query (warp-uniform)
    const int QK   = 3 * E;                    // qkv row stride: [q(E) | k(E) | v(E)]

    __shared__ float Ks[FA_BC][FA_D + 1];      // +1 -> conflict-free Ks[lane][d]
    __shared__ float Vs[FA_BC][FA_D + 1];
    __shared__ float qs[FA_BR][FA_D + 1];
    __shared__ float ps[FA_BR][FA_BC];         // this tile's softmax numerators, per warp

    if (i < T) {                               // 32 lanes x 2 dims = 64 = D
        const half *qi = qkv + (size_t)i * QK + 0 * E + h * D;
        qs[w][lane]      = __half2float(qi[lane]);
        qs[w][lane + 32] = __half2float(qi[lane + 32]);
    }

    const float scale = 1.0f / sqrtf((float)D);          // identical expression to the naive kernel
    float m = -CUDART_INF_F, l = 0.f, acc0 = 0.f, acc1 = 0.f;

    const int tid  = w * 32 + lane;                       // 0..255, for the cooperative tile load
    const int last = min(i0 + FA_BR - 1, T - 1);          // block-uniform: largest query here

    for (int j0 = 0; j0 <= last; j0 += FA_BC) {
        __syncthreads();                                  // protect Ks/Vs/qs from the previous iter's readers
        for (int idx = tid; idx < FA_BC * FA_D; idx += FA_BR * 32) {
            int jj = idx / FA_D, d = idx - jj * FA_D;     // consecutive tid -> consecutive d -> coalesced
            int j  = j0 + jj;
            float kv = 0.f, vv = 0.f;
            if (j < T) {
                const half *kj = qkv + (size_t)j * QK + 1 * E + h * D;
                const half *vj = qkv + (size_t)j * QK + 2 * E + h * D;
                kv = __half2float(kj[d]); vv = __half2float(vj[d]);
            }
            Ks[jj][d] = kv; Vs[jj][d] = vv;
        }
        __syncthreads();

        if (i < T && j0 <= i) {                           // warp-uniform -> full-mask shuffles are safe
            const int j = j0 + lane;
            float s = -CUDART_INF_F;
            if (j <= i && j < T) {                        // causal mask + tail guard
                float dot = 0.f;
                #pragma unroll
                for (int d = 0; d < FA_D; d++) dot += qs[w][d] * Ks[lane][d];
                s = dot * scale;
            }
            float tmax = s;                               // warp-reduce max over this tile's keys
            #pragma unroll
            for (int off = 16; off; off >>= 1) tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffffu, tmax, off));

            const float mn = fmaxf(m, tmax);              // finite: lane 0 has j=j0 <= i
            const float p  = expf(s - mn);                // s=-inf -> p=0 exactly
            ps[w][lane] = p;
            float lsum = p;                               // warp-reduce sum over this tile's keys
            #pragma unroll
            for (int off = 16; off; off >>= 1) lsum += __shfl_xor_sync(0xffffffffu, lsum, off);

            const float alpha = expf(m - mn);             // m=-inf on the first tile -> alpha=0
            m = mn;
            l = l * alpha + lsum;

            __syncwarp();                                 // ps[w][*] written by all lanes, now read by all
            float a0 = acc0 * alpha, a1 = acc1 * alpha;
            #pragma unroll
            for (int jj = 0; jj < FA_BC; jj++) {
                const float pj = ps[w][jj];               // 0 for masked / out-of-range keys
                a0 += pj * Vs[jj][lane];
                a1 += pj * Vs[jj][lane + 32];
            }
            acc0 = a0; acc1 = a1;
        }
    }

    if (i < T) {
        const float inv = 1.0f / l;
        half *o = att + (size_t)i * E + h * D;
        o[lane]      = __float2half(acc0 * inv);
        o[lane + 32] = __float2half(acc1 * inv);
    }
}

// Exported (not static) for the same reason gpt2_matmul_tiled is: later backends compose from
// GPT2_BACKEND_NAIVE + these function pointers instead of reading a dynamically-initialized backend
// object from another TU (unspecified cross-TU init order).
void gpt2_attention_flash(half *att, const half *qkv, int T, int E, int H, int D) {
    dim3 blk(32, FA_BR);
    dim3 grd(CEIL_DIV(T, FA_BR), H);
    k_attention_flash<<<grd, blk>>>(att, qkv, T, E, H, D);
}

// Stage 3b backend = the Stage-2 TILED backend with ONLY .attention swapped. Every other op (including
// the tiled GEMM) is reused byte-for-byte, so any gate or timing delta localizes to attention.
//
// Built from GPT2_BACKEND_NAIVE + gpt2_matmul_tiled rather than from GPT2_BACKEND_TILED: the latter is
// dynamically initialized in its own TU and cross-TU dynamic-init order is unspecified. NAIVE is
// constant-initialized (all its members are address constants), so it is safe to read here.
static GPT2Backend make_flash_backend() {
    GPT2Backend b = GPT2_BACKEND_NAIVE;
    b.name      = "flash";
    b.matmul    = gpt2_matmul_tiled;      // == what GPT2_BACKEND_TILED installs
    b.attention = gpt2_attention_flash;
    return b;
}
const GPT2Backend GPT2_BACKEND_FLASH = make_flash_backend();

