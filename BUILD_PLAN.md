# BUILD_PLAN.md — File manifest & build order (self-driving)
### GPT-2-124M · C → CUDA · companion to DESIGN.md + ROOFLINE.md + BENCH_PROTOCOL.md + QUALITY_GATES.md

> **How to use.** Build in the order below. Each file lists **objective / done-contract / governing
> gate / known trap**. The order *is* the anti-drift discipline made executable: the reference oracle
> and the microbench come **before any kernel**, because nothing can be validated or
> plausibility-checked without them.

---

## Suggested tree
```
inference-engine/
├── DESIGN.md · ROOFLINE.md · BENCH_PROTOCOL.md · QUALITY_GATES.md · BUILD_PLAN.md
├── BENCHMARKS.md            # results, filled one template block per validated run
├── tools/
│   ├── export_gpt2.py       # HF gpt2 → raw fp32 weight file + config
│   ├── reference.py         # HF oracle dumps (per-layer states, logits, greedy tokens, margins)
│   ├── eval_ppl.py          # WikiText-2 perplexity (fp16 baseline + quantized)
│   └── quantize.py          # fp16 → INT8/INT4 packed weights + scales
├── model/
│   ├── config.h             # locked dims (ROOFLINE §2)
│   ├── weights.{c,h}        # load raw weights into structs; memory layout
│   └── tokenizer.{c,h}      # GPT-2 BPE encode/decode
├── cpu/
│   └── forward_cpu.c        # STAGE 0 — pure-C fp32 forward (the correctness reference)
├── cuda/
│   ├── common.cuh           # CUDA_CHECK, sync-enforcing timing helper, device buffers
│   ├── kernels_naive.cu     # STAGE 1
│   ├── kernels_tiled.cu     # STAGE 2
│   ├── kernels_fused.cu     # STAGE 3
│   ├── kernels_quant.cu     # STAGE 4
│   ├── kvcache.{cu,cuh}     # STAGE 5
│   └── forward_cuda.cu      # orchestrates forward pass; swappable kernel backend
├── bench/
│   ├── microbench.cu        # copy-BW + big-GEMM FLOP/s  (DO FIRST)
│   ├── correctness.{c,cu}   # runs QUALITY_GATES §1 vs reference dumps
│   └── benchmark.{c,cu}     # the BENCH_PROTOCOL prefill/decode harness
└── Makefile
```

---

## Build order — do NOT reorder the first three phases

### Phase −1 · Pin the denominators — DO FIRST
**`bench/microbench.cu`**
- *Objective:* copy kernel → achieved memory BW; big square GEMM (fp16, fp32-accum) → achieved FLOP/s; ridge = FLOP/s ÷ BW.
- *Done:* stable medians (many runs); BW ≤ 256 GB/s (if > 256 → bug); write results into ROOFLINE.md, replacing the provisional numbers.
- *Gate:* ROOFLINE §6. **Unblocks every ceiling** — until it exists, no speedup is final.

### Phase 0 · Scaffolding that makes everything checkable
**`tools/export_gpt2.py`**
- *Objective:* HF `gpt2` → raw fp32 weight file + config; fixed binary layout (define the order once).
- *Done:* file loads in C at matching offsets; param count = 124M.
- *Trap:* HF GPT-2 stores linear weights as **Conv1D** (shape `[in,out]`), not Linear (`[out,in]`) → **transpose on export**. Tied embeddings: output head = token_emb, **store once**.

**`tools/reference.py`**
- *Objective:* fixed prompt + seed → dump HF oracle in **fp32 AND fp16**: per-layer hidden states, final logits, greedy tokens (N=128), per-position top-2 margin.
- *Done:* dumps exist for both precisions; these **are** the correctness ground truth.
- *Gate:* feeds QUALITY_GATES §1.

**`tools/eval_ppl.py`**
- *Objective:* WikiText-2 (raw, val), window 1024 / stride 512 → PPL; run on fp16 = quality baseline; reusable for quantized weights.
- *Done:* fp16 PPL ≈ 29–30 (else harness bug).
- *Gate:* QUALITY_GATES §2.

**`model/config.h`** — locked dims (ROOFLINE §2): 12 / 12 / 768 / 50257 / 1024, `gelu_new`, pre-LN, learned pos, tied emb.

**`model/weights.{c,h}` · `model/tokenizer.{c,h}`**
- *Objective:* load raw weights into structs (define layout); GPT-2 BPE encode/decode.
- *Done:* tokenizer round-trips **byte-identically** to HF on a test string.
- *Trap:* tokenizer must match HF exactly, or logits diverge from token 0.

### STAGE 0 · Pure-C forward (the correctness reference)
**`cpu/forward_cpu.c`**
- *Objective:* fp32 forward, zero deps. `embed+pos → 12×[pre-LN, QKV, causal attn+softmax, proj, +res, LN, fc→GELU→proj, +res] → final LN → logits = h·token_embᵀ`.
- *Done:* gate (a) rel_err ≤ 1e-4 vs HF fp32; gate (b) greedy tokens match exactly. **Becomes the reference for all GPU kernels.** Also emits CPU baseline tok/s (slow — fine).
- *Gate:* QUALITY_GATES §1.
- *Trap:* GPT-2 GELU = **tanh approximation (`gelu_new`)**, not erf GELU. Match it exactly.

### STAGE 1 · Naive CUDA port
**`cuda/common.cuh`**
- *Objective:* `CUDA_CHECK` macro; a timing helper that **always** syncs before+after (makes a benchmark-without-sync hard to write — enforces BENCH_PROTOCOL §7 bug #1 at infra level); device buffer helpers.

**`cuda/kernels_naive.cu` · `cuda/forward_cuda.cu`**
- *Objective:* naive GEMM (no shared mem), elementwise (GELU, residual), LayerNorm, naive softmax, embed lookup; `forward_cuda` orchestrates a **swappable** kernel backend.
- *Done:* re-validate on GPU vs HF fp16 — gate (a) ≤ 1e-2/layer, gate (b) token-match w/ margin rule, gate (c). Then it's the **GPU baseline**.
- *Gate:* QUALITY_GATES §1 (re-run — a port is a prime place for silent bugs, DESIGN.md G4.1).

### STAGE 2 · Tiled / shared-memory GEMM
**`cuda/kernels_tiled.cu`**
- *Objective:* tiled GEMM with shared-memory reuse.
- *Done:* correctness re-pass; measured **prefill** speedup vs naive; report achieved % of compute roofline (uses microbench FLOP/s).
- *Gate:* QUALITY_GATES §1 + BENCH_PROTOCOL. *Expected (ROOFLINE director map):* prefill win, **decode ~flat**. Decode jumping a lot → suspect a measurement bug or an unrelated change.

### STAGE 3 · Fused kernels
**`cuda/kernels_fused.cu`**
- *Objective:* (3a) fused LN+matmul; (3b) flash-style attention (online softmax, no full score matrix).
- *Done:* correctness re-pass; measured traffic/speedup; **kill-test** — no speedup → drop the fusion (DESIGN.md §5). Note which regime each helps.
- *Gate:* QUALITY_GATES §1 + BENCH_PROTOCOL.

### STAGE 4 · Quantized matmul (the main decode lever)
**`tools/quantize.py`**
- *Objective:* fp16 → INT8 (symmetric, per-channel, weight-only) [+ optional INT4]: packed weights + scales.
- *Gate:* scheme pre-registered in QUALITY_GATES §2.

**`cuda/kernels_quant.cu`**
- *Objective:* INT8 (then optional INT4) matmul consuming packed weights + scales.
- *Done:* correctness (dequant path sane) + **quality gate** (Δppl ≤ +0.3 INT8, QUALITY_GATES §2) + measured **decode** speedup (expected high — fewer bytes).
- *Gate:* QUALITY_GATES §1 & §2 + BENCH_PROTOCOL.

### STAGE 5 · KV cache + planner
**`cuda/kvcache.{cu,cuh}`**
- *Objective:* store/append past K/V; simple memory planner (layout fits 8 GB, minimizes traffic).
- *Done:* correctness (cached outputs == recompute) + measured **decode** speedup (**no-KV vs KV**, same protocol — expected essential). Report decode **@ stated ctx**.
- *Gate:* QUALITY_GATES §1 + BENCH_PROTOCOL §3.

### Harness · build early, run continuously
**`bench/correctness.{c,cu}`** — loads reference dumps, runs a backend, applies gates (a)(b)(c), prints pass/fail + localization. **Run after every kernel change.**
**`bench/benchmark.{c,cu}`** — implements BENCH_PROTOCOL exactly (warmup, prefill per P, decode per ctx w/ discard + median±spread, clock/thermal record, emits the BENCHMARKS.md template block); compares vs PyTorch eager / llama.cpp at **matched precision**.

---

## Definition of done (per stage)
A stage is done when, **in this order:** correctness gates pass (QUALITY_GATES §1) → measured
before/after speedup, apples-to-apples, reproducible (BENCH_PROTOCOL) → number is **≤ its ROOFLINE
ceiling** and consistent with the regime → one template block written to BENCHMARKS.md. **Not before.**

## Escalate and stop — do not paper over these
- A measured number **above** a ROOFLINE ceiling (a bug is hiding).
- A number whose **regime contradicts the director map** (e.g., tiled GEMM "speeds up decode" a lot).
- A **quality-gate failure** the kill-test options don't recover.

Otherwise the loop is: implement, profile, check against these docs, record.
