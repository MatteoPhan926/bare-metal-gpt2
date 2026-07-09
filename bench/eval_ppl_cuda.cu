// eval_ppl_cuda.cu — STAGE 4 QUALITY GATE (QUALITY_GATES §2): WikiText-2 perplexity of OUR engine.
//
// QUALITY_GATES §2 says the baseline is "**your own fp16 kernel's** PPL on this exact set/harness —
// not a published number". tools/eval_ppl.py scores the *HF* model and produced the frozen fp16
// number 25.57. This binary scores OUR engine's logits under the IDENTICAL convention, so it can
// report BOTH deltas:
//     (primary, per §2)  Δppl = ppl_int8(ours) − ppl_fp16(ours)      <- harness bias cancels exactly
//     (the frozen literal) Δppl = ppl_int8(ours) − 25.57
// Running it first on the fp16 (flash) backend VALIDATES the replication: it must land next to 25.57,
// or the harness is wrong and no Δ from it means anything.
//
// EXACT replication of eval_ppl.py's loop (window=1024, stride=512), including HF's own quirk that the
// first window's `trg` (1024) exceeds the number of shifted positions it actually scores (1023):
//
//   nll_sum, n_tokens, prev_end = 0, 0, 0
//   for begin in range(0, seq_len, 512):
//       end = min(begin+1024, seq_len);  trg = end - prev_end
//       inp = ids[begin:end];  tgt = inp.clone();  tgt[:, :-trg] = -100     # empty slice if trg >= L
//       loss = model(inp, labels=tgt).loss    # CE over the SHIFTED, unmasked positions, mean
//       nll_sum += loss*trg;  n_tokens += trg;  prev_end = end
//       if end == seq_len: break
//   ppl = exp(nll_sum / n_tokens)
//
// HF shifts internally: logits[i] predicts labels[i+1]. With the first (L-trg) labels masked, the
// scored logit rows are i in [max(0, L-trg-1), L-2]. We therefore compute the LM head for ONLY those
// rows — mathematically identical to computing all L and masking, at ~half the head GEMM.
// Modern transformers upcasts logits to fp32 before the loss, so the CE here is fp32 over fp16 logits.
//
// Run from repo root:   eval_ppl_cuda.exe [ids.bin] [limit_tokens]     env: GPT2_BACKEND=flash|int8

#include "kernels.cuh"
#include "common.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

#define E GPT2_N_EMBD
#define V GPT2_VOCAB

static const int WINDOW = 1024, STRIDE = 512;    // pre-registered (QUALITY_GATES §2). NOT knobs.

// One block per scored row. Two-pass fp32 log-softmax; writes -log p(target) for that row.
__global__ void k_row_nll(float *nll, const half *logits, const int *targets, int V_, int rows) {
    int r = blockIdx.x; if (r >= rows) return;
    const half *lg = logits + (size_t)r * V_;
    __shared__ float red[256];
    int tid = threadIdx.x, nt = blockDim.x;

    float mx = -1e30f;
    for (int v = tid; v < V_; v += nt) mx = fmaxf(mx, __half2float(lg[v]));
    red[tid] = mx; __syncthreads();
    for (int s = nt/2; s > 0; s >>= 1) { if (tid < s) red[tid] = fmaxf(red[tid], red[tid+s]); __syncthreads(); }
    float m = red[0]; __syncthreads();

    float sum = 0.f;
    for (int v = tid; v < V_; v += nt) sum += __expf(__half2float(lg[v]) - m);
    red[tid] = sum; __syncthreads();
    for (int s = nt/2; s > 0; s >>= 1) { if (tid < s) red[tid] += red[tid+s]; __syncthreads(); }

    if (tid == 0) {
        float lse = m + logf(red[0]);                       // log sum exp
        nll[r] = lse - __half2float(lg[targets[r]]);        // -log p(target)
    }
}

int main(int argc, char **argv) {
    const char *idspath = argc > 1 ? argv[1] : "refdumps/wikitext2_val_ids.bin";
    const long  limit   = argc > 2 ? atol(argv[2]) : 0;     // 0 = all tokens (the gate); else a smoke test

    const GPT2Backend *be = gpt2_backend_by_name(getenv("GPT2_BACKEND"));

    FILE *f = fopen(idspath, "rb");
    if (!f) { fprintf(stderr, "[ppl] cannot open %s (run tools/dump_wikitext_ids.py)\n", idspath); return 1; }
    fseek(f, 0, SEEK_END); long ntok = ftell(f) / 4; fseek(f, 0, SEEK_SET);
    std::vector<int> ids(ntok);
    if (fread(ids.data(), 4, ntok, f) != (size_t)ntok) { fprintf(stderr, "[ppl] short read\n"); return 1; }
    fclose(f);
    long seq_len = (limit > 0 && limit < ntok) ? limit : ntok;

    printf("==== PPL (QUALITY_GATES §2) : %s backend, WikiText-2-raw val ====\n", be->name);
    printf("tokens=%ld  window=%d  stride=%d  %s\n", seq_len, WINDOW, STRIDE,
           limit ? "[LIMITED -> smoke test, NOT the gate]" : "[full set -> the gate]");

    GPT2Weights wcpu;
    if (gpt2_load_weights("weights/gpt2_124m_fp32.bin", &wcpu)) { fprintf(stderr, "weight load failed\n"); return 1; }
    GPT2WeightsGPU w; gpt2_upload_fp16(&wcpu, &w);
    gpt2_free_weights(&wcpu);
    GPT2QWeightsGPU qw;
    if (gpt2_quant_attach_if_needed(be, &w, &qw)) { fprintf(stderr, "int8 weight load failed\n"); return 1; }
    printf("weights: fp16 blob + %s\n\n", w.q ? "INT8 packed (matmuls take the quantized path)" : "no quant");

    GPT2ScratchGPU s; gpt2_scratch_alloc(&s, GPT2_N_CTX);
    int   *d_ids  = dmalloc<int>(WINDOW);
    int   *d_tgt  = dmalloc<int>(WINDOW);
    half  *d_lg   = dmalloc<half>((size_t)WINDOW * V);      // logits for the scored rows only
    float *d_nll  = dmalloc<float>(WINDOW);
    std::vector<float> h_nll(WINDOW);
    std::vector<int>   h_tgt(WINDOW);

    double nll_sum = 0.0; long n_tokens = 0, prev_end = 0; int nwin = 0;

    for (long begin = 0; begin < seq_len; begin += STRIDE) {
        long end = begin + WINDOW; if (end > seq_len) end = seq_len;
        long trg = end - prev_end;                          // newly-scored tokens this window
        int  L   = (int)(end - begin);

        // scored logit rows: i in [i0, L-2]  (i0 = max(0, L-trg-1)); row i predicts ids[begin+i+1]
        long i0l = (long)L - trg - 1; int i0 = (int)(i0l > 0 ? i0l : 0);
        int rows = (L - 2) - i0 + 1;
        if (rows <= 0) { prev_end = end; if (end == seq_len) break; continue; }

        h2d(d_ids, ids.data() + begin, (size_t)L);
        gpt2_forward_cuda(be, &w, d_ids, L, &s, nullptr);   // s.ln holds final_ln after the call

        gpt2_matmul_dispatch(be, d_lg, s.ln + (size_t)i0 * E, w.wte, w.q ? &w.q->wte : nullptr,
                             nullptr, rows, V, E);

        for (int r = 0; r < rows; r++) h_tgt[r] = ids[begin + i0 + r + 1];
        h2d(d_tgt, h_tgt.data(), (size_t)rows);
        k_row_nll<<<rows, 256>>>(d_nll, d_lg, d_tgt, V, rows);
        CUDA_CHECK(cudaGetLastError());
        d2h(h_nll.data(), d_nll, (size_t)rows);

        double win = 0.0;                                    // HF's CE = MEAN over the scored rows
        for (int r = 0; r < rows; r++) win += (double)h_nll[r];
        double loss = win / rows;

        nll_sum += loss * (double)trg;                       // ... then weighted by trg (HF's convention)
        n_tokens += trg;
        prev_end = end;
        if (++nwin % 50 == 0) { printf("  [%3d windows] running ppl = %.4f\n", nwin, exp(nll_sum/n_tokens)); fflush(stdout); }
        if (end == seq_len) break;
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    double mean_nll = nll_sum / n_tokens, ppl = exp(mean_nll);
    printf("\n[ppl] windows=%d  scored_tokens=%ld  mean_nll=%.4f\n", nwin, n_tokens, mean_nll);
    printf("[ppl] PPL(%s, ours) = %.4f\n", be->name, ppl);
    printf("PPL_%s=%.4f\n", be->name, ppl);

    gpt2_scratch_free(&s);
    cudaFree(d_ids); cudaFree(d_tgt); cudaFree(d_lg); cudaFree(d_nll);
    gpt2_free_gpu(&w); gpt2_quant_free(&qw);
    return 0;
}
