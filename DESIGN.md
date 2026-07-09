# DESIGN.md — Mini Inference Engine (GPT-2-124M · C → CUDA)
### Design log: the method, decided before the code

> **Locked selections.** G1's "pick one model" was resolved to **GPT-2-124M** at Phase 0, and every
> artifact since assumes it (`model/config.h`, `tools/export_gpt2.py`). The `[VERIFY]` tags in §3 that
> concern the roofline denominators are **discharged** — Phase −1 measured them; see ROOFLINE §6 and
> BENCHMARKS "Phase −1". They are annotated `[VERIFIED]` in place below.

> **What this file is.** The design log for a from-scratch LLM inference engine: the place where method
> was reasoned out *before* code was written, and the anchor that kept the scope from drifting (scope
> creep is the failure mode here — see §2). It carries a deliberate character: serious, precise, no
> over-claiming, pre-registered, validate-before-you-claim, treat-your-own-numbers-as-leads. Those
> habits are aimed at the failure modes of **performance engineering** specifically. In research the
> trap is manufacturing novelty; here the trap is a **fast-but-wrong kernel** and a **dishonest
> benchmark**. This document's whole job is to make neither happen.

> **The character, stated once.** A modest, correct, reproducible number beats an impressive,
> unvalidated one — always, and especially for the actual goal (a portfolio a systems interviewer
> trusts). "Should be faster" is not "is faster." A kernel is not done when it runs; it is done when
> it is *validated* and its speedup is *measured, reproducible, and consistent with the roofline*.

---

## §0. The discipline

**Design and execution are kept apart, deliberately.** The optimization ladder, the roofline analysis,
and the pre-registered thresholds are all decided *before* the corresponding kernel is written — and
written down here, where they can't quietly move afterwards. Implementation then follows the ladder;
it does not get to redesign it to make a number look better.

**The loop:** every measured number is checked back against the roofline (plausible? apples-to-apples?
within its ceiling?) *before* it is trusted. A number is a **lead, not a fact**, until it survives
that check.

**The three firewalls (no exceptions).**
1. **Correctness before speed.** No kernel's tokens/s is reported until it passes the reference-
   correctness gate (§6). **A fast wrong kernel is worse than a slow right one** — it is deceptive.
2. **Benchmark honesty.** Pre-register the protocol. Report **median + spread**, not best-of-N. State
   **exactly** what is included/excluded (tokenization? H2D copy? warmup?). Compare **apples-to-
   apples** (same precision, seq len, batch; warmed up; clocks noted). **Report prefill and decode
   separately** (different regimes — see §3 keystone).
3. **Roofline as plausibility check + director.** Compute the theoretical ceiling for each kernel.
   A measured number **above** the ceiling means a measurement bug (fix it — it is not a result). The
   **gap** to the ceiling tells you which optimization is worth doing (and which is pointless).

**Launder nothing.** Any speedup or quality claim stays a `[CONJECTURE]` until a validated,
reproducible measurement backs it. Blog-post/paper tricks are **leads to verify on your hardware**,
not facts — a trick that wins on an A100 may do nothing on a 4060 (different ridge point).

---

## §1. The locked problem (one sentence)

A **single-model, forward-inference-only** LLM engine (**GPT-2-124M**, locked at Phase 0),
implemented from scratch in **pure C** (llama2.c-style), then ported to **CUDA** and
**progressively optimized** along a fixed ladder (§3), **benchmarked honestly** against PyTorch eager
and llama.cpp on **one RTX 4060 Laptop**, with a **roofline analysis at each stage**. It is a
**performance-engineering flagship** — not a framework, not a serving system, not a research project.

---

## §2. What it IS / What it is NOT (anti-drift spine)

**IS:** forward-pass inference; one model; one GPU; the specific optimization ladder; honest
benchmarks + roofline; a story about **efficient ML systems**.

**IS NOT:**
- **NOT** training / backprop / fine-tuning. Forward inference only.
- **NOT** a general framework, serving system, or batching/scheduling engine.
- **NOT** multi-GPU / distributed.
- **NOT** exotic or large models. One small model, locked.
- **NOT** "expand the OS kernel." That was the low-signal trap (a toy kernel needs a year to
  impress). This project is the **C/POSIX skills redirected to ML systems** — that redirection *is*
  the point.
- **NOT** a llama.cpp clone. llama.cpp is a **baseline to compare to and learn from**, not to
  reproduce.
- **NOT** SOTA-kernel-chasing beyond the ladder. The ladder is the scope; exotic kernels are future
  work.

---

## §3. HARD GROUND (constraints that channel the work; exacts flagged `[VERIFY]`)

**G1. Target model.** ~~SmolLM-135M or GPT-2-124M — **pick ONE, lock it**~~ → **LOCKED: GPT-2-124M.**
`[VERIFIED 2026-07-08]` param count **124,439,808** exact vs `model/config.h` (→ **248.9 MB** fp16, not
the ~250–270 MB placeholder); config 12/12/768/50257/1024; tokenizer = GPT-2 BPE, pre-tokenized HF ids
feed every gate; reference weights = HF `gpt2` via `tools/export_gpt2.py` (sha256 recorded in BENCHMARKS).

**G2. Target hardware (the roofline anchors).** RTX 4060 Laptop (AD107, **sm_89**, 8188 MiB GDDR6,
24 SM, L2 = 32 MiB). `[VERIFIED 2026-07-08 by bench/microbench.cu — ROOFLINE §6]`:
- memory bandwidth: **233.4 GB/s** achieved copy (2N r+w) · **248.9 GB/s** achieved read (1N) · 256 GB/s theoretical.
- fp16 tensor throughput: **31.51 TFLOP/s** fp32-accum (60.98 fp16-accum, 1.94×) — *tensor cores*.
- CUDA-core (no tensor cores) `[VERIFIED 2026-07-09 — ROOFLINE §6b]`: fp32 **FMA pipe 15.64 TFLOP/s**
  (97.5% of the 128-lane issue width) and **achievable GEMM 10.10 TFLOP/s** (fp16 in / fp32 acc, cuBLAS
  PEDANTIC). The **10.10** is the denominator for the engine's *actual* GEMMs, which use **no WMMA**.
  > This was previously **derived from the clock** (24 × 128 × 2 × 2.61 GHz = 16.0 TFLOP/s) — an assumed
  > denominator, which is exactly what G2 forbids. Measured, it is optimistic by 1.55× against the
  > achievable GEMM. Ridge corrected accordingly (G7): **43.3, not 69.**

The microbenchmark, not the datasheet, is the source. Both were run; they agree to within 9% on BW.

**G3. THE keystone fact (directs the entire strategy — the "physics" of this project).**
Autoregressive **DECODE (batch=1)** is **memory-bandwidth-bound, not compute-bound**: each generated
token streams ~all weights through the bus once, and the arithmetic-per-byte is tiny. Consequences,
all load-bearing:
- **Decode speed ceiling ≈ weight_bytes / bandwidth.** `[VERIFIED]` **248.9 MB / 233.4 GB/s ≈ 1.07 ms/token
  ≈ ~938 tok/s** in fp16 at achieved copy BW (~1000 at read BW; **1029 at theoretical 256 GB/s = the
  measurement-bug line**). The old "~270 MB → ~950 tok/s" was a placeholder; the true param count is
  124.44M → 248.9 MB. Exact ceilings per precision: ROOFLINE §6.
- **Quantization speeds up decode by moving FEWER BYTES** (~2× per halving of weight bytes: INT8 →
  ~1889 tok/s ceiling, INT4 → ~3778 at copy BW), **not** by faster math.
  > **`[MEASURED, Stage 4]` — the byte saving is necessary but NOT sufficient.** It only becomes speed
  > once the kernel reading the weights is actually traffic-bound. The 16×16 tiled GEMM at M=1 computes
  > 16 output rows and discards 15, so it is saturated on wasted arithmetic (860.6 GFLOP/s = 98.8% of its
  > M=512 throughput) while pulling only 23% of achieved BW. Weight-only INT8 therefore measured **1.00×**
  > at every shape. See BENCHMARKS Stage 4.
- **Tiled/compute-optimized GEMM matters mainly for PREFILL** (compute-bound at longer seq), much
  less for pure decode. **Prefill and decode are different regimes — measure and report them
  separately, always.**
This fact is what prevents cargo-cult optimization: before optimizing a kernel, know whether its
regime is memory- or compute-bound, or the effort is wasted.

**G4. The optimization ladder (locked sequence; each stage gated by correctness + a *measured*
speedup — no stage is "done" on faith).**
0. **Pure-C forward pass** (llama2.c-style), fp32 → establishes the correctness reference and a CPU
   baseline.
1. **Naive CUDA port** (naive GEMM, one block per output tile) → correctness **re-validated** on GPU
   (a port is a prime place for silent bugs).
2. **Tiled / shared-memory GEMM** → measured speedup; report achieved % of the compute roofline.
3. **Fused kernels:** RMSNorm+matmul (LayerNorm for GPT-2); **attention with online softmax**
   (flash-style — avoid materializing the full score matrix; this is a memory-traffic win).
4. **INT8** (then optionally **INT4**) quantized matmul → measured speedup **+ a quality gate**
   (perplexity delta within a pre-registered bound, §5).
5. **KV cache + a simple memory planner** → decode speedup (avoids recomputing past K/V; trades
   memory for compute — essential for decode).

**G5. Correctness reference.** PyTorch/HuggingFace running the **identical** model/config. This is
ground truth for every kernel (§6).

**G6. Benchmark baselines.** PyTorch eager **and** llama.cpp — same model, same precision where
comparable, same seq/batch, warmed up, same GPU. Apples-to-apples or the number is meaningless.

**G7. Durable systems fundamentals (fuel — reason from these; they don't expire).** Memory hierarchy
(registers → shared/SMEM → L2 → GDDR/HBM), each ~10× slower and larger than the last → tiling/reuse
wins by cutting traffic to the slow level. Roofline model: a kernel is memory-bound left of the ridge
point (arithmetic intensity = FLOP/byte < ridge), compute-bound right of it; optimize for the side
you're on. Occupancy, coalesced access, bank conflicts, warp divergence are the usual CUDA levers —
but **each only matters if it moves the measured bottleneck** (§9.1). `[VERIFIED 2026-07-08/09]` ridge point
= peak-FLOP/s ÷ bandwidth = **135 FLOP/byte** for a *tensor-core* kernel (31.5 TF ÷ 233.4 GB/s), and
**43 FLOP/byte** for the *CUDA-core* kernels this engine actually ships (measured achievable 10.10 TF ÷
233.4 GB/s; the CUDA-core FMA pipe, 15.64 TF, gives 67 and is a bug line). Decode's AI ≈ 1–2 FLOP/byte sits
far left of ALL THREE → memory-bound regardless (ROOFLINE §4, §6, §6b).

---

## §4. OPEN QUESTIONS (questions to be answered by measurement, not pre-decided)

- **Where is the ACTUAL bottleneck at each stage?** Profile first; never guess. Per the roofline, is
  the current stage memory- or compute-bound? (Answer changes which optimization is worth doing.)
- **Which optimizations give a REAL speedup on THIS GPU** vs cargo-cult from A100 blogs? The ridge
  point and cache sizes differ; each trick must be measured here.
- **Fused-kernel design:** which fusions actually cut GDDR traffic enough to matter — and for decode
  or prefill?
- **Quantization design:** symmetric/asymmetric? per-tensor/per-channel? which layers tolerate INT4?
  where is the quality/speed frontier on this model?
- **Memory planner:** the minimal KV-cache + activation layout that fits 8 GB and minimizes traffic?

## §5. DESIGN BETS (labeled `[CONJECTURE]`; each kill-test is a MEASUREMENT, not an argument)

*(Status pointers added 2026-07-09 doc pass. The bets and their bounds are UNCHANGED — only the outcome
of each kill-test is recorded, which is the whole point of pre-registering them.)*

- **[CONJECTURE]** Fused RMSNorm+matmul beats separate kernels. *Kill-test:* measure both (validated),
  before/after. No speedup → drop the fusion.
  → **`[FALSIFIED for prefill, Stage 3a]`** an above-noise **2.9% SLOWDOWN** (0.971× at three shapes,
    disjoint ranges) → **REVERTED** (`git show fb5ad32`). A fusion pays only when the producer's output is
    consumed **once**; the tiled GEMM re-stages A per column-tile (144× for qkv), so it re-normalized 144×.
    **Still OPEN for true M=1 decode**, where LN's share is far larger — untestable until Stage 5.
- **[CONJECTURE]** INT8 holds quality within Δperplexity ≤ **(pre-register, e.g. +0.3)** vs fp16.
  *Kill-test:* measure perplexity on held-out text. Exceeds the bound → per-channel, or keep fp16 for
  sensitive layers.
  → **`[FALSIFIED as stated, then RECOVERED by the pre-registered kill-test, Stage 4]`** pure per-channel
    weight-only INT8 (all 49 matmul weights): **Δppl +0.549** and gate (c) max KL **0.257** → fails.
    Localized: the **tied output head** alone causes ~all of it. Keeping the head fp16 (48/49 int8):
    **Δppl +0.027**, KL 0.0150 → passes. Full INT8 *including the head* is **not a valid quality point**
    for this model. Bound never re-tuned.
- **[CONJECTURE]** Tiled GEMM reaches **X%** of the compute roofline for prefill. *Kill-test:* measure
  achieved FLOP/s vs the ceiling (G7).
  → **`[MEASURED]`** prefill@512 = **856 GFLOP/s** = **5.35%** of the CUDA-core 16.0 TF peak (the right
    denominator — no WMMA), = 2.72% of the 31.5 TF *tensor* peak, which is the WMMA headroom, not this
    kernel's efficiency. Stage 2 = 4.5–5.6× prefill; Stage 3b flash added 1.72× @512.
- **[CONJECTURE]** KV cache gives ≥ **Y×** decode speedup. *Kill-test:* measure decode tok/s with and
  without.
  → **`[OPEN — Stage 5]`** and note Stage 4's finding: the KV cache alone will not surface the INT8 decode
    win unless the M=1 path uses a **GEMV-shaped kernel** instead of the 16×16 tiled GEMM.
> **No stated speedup is a result until a validated, reproducible measurement backs it.**

---

## §6. The central validation gates (make-or-break, purely empirical)

- **CORRECTNESS gate (every kernel, before any timing):** output matches the reference —
  (a) **layer-by-layer** logits within tolerance (localizes bugs — don't only check the final
  logits), (b) **greedy-decoded tokens match** the reference for a fixed prompt/seed, (c) for
  quantized kernels, **perplexity within the pre-registered bound**. Fails → it is not a working
  kernel, regardless of speed.
- **HONEST-SPEEDUP gate (every stage):** tokens/s as **median + spread**, warmed up, clocks noted,
  full config + seed recorded, **reproducible**; **apples-to-apples** vs the baseline; **consistent
  with the roofline** (above-ceiling = a measurement bug to fix, not a result to publish). **Prefill
  and decode reported separately.**

---

## §7. Scope / validity (claim only within)

**In:** forward inference, batch=1 (small batch if time permits), the locked model, the RTX 4060, the
ladder. **Deferred (future work, no claim):** training, multi-GPU, serving/batching, speculative
decoding, other architectures, production robustness, other GPUs.

## §8. Goal alignment (scope/finishing only — NOT a steer on benchmark honesty)

Narrative: **efficient ML systems.** A from-scratch, honestly-benchmarked inference engine with
roofline analysis maps directly to **Qualcomm** (QNN / on-device inference runtimes) and **NVIDIA**
(CUDA kernels, TensorRT-adjacent) and makes the Infrastructure track a coherent story. **Guardrail:**
these goals decide *what to build and finish* — they must **never** inflate a benchmark, excuse an
unfair comparison, or lower the correctness bar. The credential is only worth anything if the numbers
are true; a systems interviewer will probe exactly the methodology, so honesty *is* the moat.

## §9. Process rules

1. **Measure before you optimize.** Profile/roofline first; optimize the **measured** bottleneck, not
   a guessed one. (This rule exists because guessing the bottleneck is the most common way to waste a
   week.)
2. **Correctness before speed.** Validate, then time. Every kernel, every time.
3. **One optimization at a time**, with a measured before/after. **Revert** anything that doesn't show
   a real, validated speedup — even if it "should" help.
4. **Commit; don't thrash.** Follow the ladder; don't jump to exotic kernels to chase a number.
5. **Treat your own numbers — and every trick you read — as LEADS to verify:** against the roofline,
   against the reference, on **this** hardware.
6. **Reproducibility is non-negotiable:** every benchmark = a script + recorded config + clocks +
   seed. If you can't reproduce it, you don't have it.
7. **Every number is checked against the roofline before it is believed.** Measure and profile on the
   target hardware; interpret sceptically, away from the keyboard, before publishing a claim.

---

## §10. Pointers

- Keep a short `BENCHMARKS.md`: for each stage, the config, clocks, median±spread tok/s (prefill and
  decode), the roofline ceiling, and % of ceiling achieved. This *is* the portfolio artifact.
- Optional companion `perf_store.md` (a "fuel inventory" like the sibling project's physics store):
  the durable systems facts (memory hierarchy numbers, roofline formulas, common CUDA pitfalls) to
  reason from without re-looking-them-up — build it only if the reasoning starts repeating.
