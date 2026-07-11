# ROOFLINE.md — Pre-registered ceilings & plausibility oracle
### GPT-2-124M · RTX 4060 Laptop (105W) · companion to DESIGN.md

> **What this file is.** Written *before any kernel exists*, so that every number measured later has a
> ceiling to be checked against. A measured number **above** a ceiling here is a
> **measurement bug, not a result** (DESIGN.md §0, firewall 3). Ceilings are theoretical (100% of the
> stated bound); real kernels reach a *fraction* of them — that fraction is stated per regime. This
> file holds the *ceilings and the plan*; `BENCHMARKS.md` holds the *measured evidence*.

---

## 1. Locked hardware anchors (G2)

| Quantity | Value | Status |
|---|---|---|
| GPU | RTX 4060 Laptop (AD107, **sm_89**), 24 SM, 96 4th-gen tensor cores | confirmed |
| Power limit | **105 W** (`nvidia-smi` max_limit, this machine) | confirmed |
| VRAM | 8188 MiB (~8 GB) | confirmed |
| Memory bus | 128-bit, GDDR6 @ 16 Gbps effective | confirmed (spec) |
| **Memory bandwidth (theoretical)** | **256 GB/s** = 16e9 × 128 / 8 | confirmed — the **measurement-BUG LINE** (§6), *not* the denominator |
| Memory bandwidth (achieved) | **233 GB/s** copy(2N r+w) · **249 GB/s** read(1N) | ✅ **MEASURED** 2026-07-08 → **decode denominator** — see §6 |
| FP16 tensor peak (fp32-accum) | **31.5 TFLOP/s** (fp16-accum 61.0 = 1.94×) | ✅ **MEASURED** 2026-07-08 → prefill **bug line** + WMMA headroom — see §6 |
| CUDA-core fp32 FMA pipe | **15.64 TFLOP/s** | ✅ **MEASURED** 2026-07-09 → CUDA-core **bug line** (97.5% of the 128-lane issue width @2610 MHz) — see §6b |
| CUDA-core GEMM, achievable | **10.10 TFLOP/s** (fp16 in, fp32 acc, tensor cores disabled) | ✅ **MEASURED** 2026-07-09 → **prefill denominator for the shipped, non-WMMA kernels** — see §6b |
| L2 cache | 32 MiB | confirmed — runtime API (`cudaDeviceProp.l2CacheSize`) |

> **Which compute denominator? (doc pass 2026-07-09 — this was internally contradictory.)** Every GEMM this
> engine ships (Stage 2 tiled, Stage 3b flash, Stage 4 INT8) runs on **CUDA cores with NO tensor cores** —
> and weight-only INT8 cannot reach `dp4a`/INT8 tensor cores at all, since its activations stay fp16.
> So **31.5 TFLOP/s is a bug line** (nothing may exceed it) **and a statement of the WMMA headroom left on
> the table**, not an efficiency yardstick. The efficiency denominator is the **achievable non-tensor GEMM,
> 10.10 TFLOP/s** — the exact analog of how §6 pins 31.5 with cuBLAS. BENCHMARKS Stages 1–4 report against
> all three ceilings, labelled.
>
> ⚠ **Correction (2026-07-09, second pass).** An earlier revision of this table asserted a CUDA-core peak of
> **16.0 TFLOP/s "derived from clock"** (24 × 128 × 2 × 2.61 GHz) and a ridge of ≈69 from it. That violated
> **G2 (microbench, do not assume)** — the very thing §6 avoided for the tensor peak — so it was **measured**
> (§6b). Outcome: the clock-derived 16.04 is a *hardware ceiling the pipe does not reach* (measured FMA pipe
> **15.64**, 97.5% of it), and the **achievable GEMM is 10.10**, far below both. The ridge that governs this
> engine is therefore **43, not 69**. The old 69 was numerically wrong; it happened to sit near the FMA-pipe
> ridge (67.0), which is a bug line, not a roofline denominator.

**Load-bearing note on the power limit.** GDDR6 clock is set by the memory chips, **not** by TGP.
So 105 W (vs the part's 115 W ceiling) throttles the **core/tensor clock → prefill/compute**, and
leaves **memory bandwidth → decode essentially intact**. Expect the power limit to cost prefill
FLOP/s, not decode tok/s. This is *why* the decode-throughput headline is robust on this machine.

---

## 2. Locked model (G1)

GPT-2 small, HF id `gpt2`. n_layer=12, n_head=12, n_embd=768, vocab=50257, ctx=1024, GELU,
pre-LN LayerNorm, learned positional embeddings, **tied** input/output embeddings.
Params ≈ 124M → fp16 ≈ 248 MB. Tokenizer: GPT-2 BPE (50257), must be byte-identical to the oracle.

**Weight bytes streamed per decode token** (the number the decode ceiling divides):
all matmul weights (~85M) + the **output head** (= tied token embedding, 50257×768 ≈ 38.6M, streamed
in full every token to produce logits) ≈ **124M params**. Input/position embeddings are 1-row
lookups → negligible.
> **TRAP:** excluding "the embedding" under-counts by ~77 MB (fp16) and yields a falsely optimistic
> ceiling. The tied embedding *is* the output head and *is* streamed every token. Count it.

---

## 3. Decode ceiling — memory-bound regime (batch=1)

`decode ceiling = weight_bytes / bandwidth`. At **theoretical 256 GB/s**, short context (KV ≈ 0):

| Precision | Weight bytes | ms/token (ceiling) | tok/s (ceiling) |
|---|---|---|---|
| fp32 | 496 MB | 1.94 | ~516 |
| fp16 | 248 MB | 0.97 | ~1032 |
| INT8 | 124 MB | 0.48 | ~2065 |
| INT4 | 62 MB | 0.24 | ~4129 |

**Reading these honestly:**
- These are **hard ceilings** at 100% of theoretical BW, ignoring KV-cache reads, activation traffic,
  and kernel-launch overhead. A real kernel that *beats* one of these has a measurement bug — fix it.
- Realistic decode ≈ **0.55–0.80×** the ceiling. *(Pre-measurement this bullet assumed "achieved BW
  ~200 GB/s"; ✅ **MEASURED 2026-07-08: 233.4 copy / 248.9 read** — see §6. Recomputing against the
  measured copy BW: fp16 realistic ≈ **~515–750 tok/s**, not 1032.)*
- Quantization buys decode speed by **moving fewer bytes** (~2× per halving), *not* faster math —
  the entire reason INT8/INT4 is on the ladder for decode.
- KV-cache reads grow with context. At full 1024 ctx, KV ≈ 38 MB fp16 (2·12·1024·768·2), up to
  **~+15%** traffic on top of weights → **the decode ceiling degrades with context length.** Always
  report decode tok/s *with the context length stated*.

---

## 4. Prefill regime — compute-bound at non-trivial prompt length

Prefill reuses each weight across all T prompt tokens → arithmetic intensity ≈ T (FLOP/byte).

There are **three** ridge points on this GPU, one per compute ceiling, and picking the wrong one is the
easiest way to misread a prefill number. All three denominators are measured (§6, §6b):

| Compute ceiling | TFLOP/s | Ridge (FLOP/byte) | Governs |
|---|---|---|---|
| **CUDA-core GEMM, achievable** | **10.10** | **43.3** | **every kernel this engine ships** (no WMMA) |
| CUDA-core FMA pipe | 15.64 | 67.0 | a bug line for a CUDA-core kernel, not a denominator |
| Tensor-core GEMM | 31.51 | 135.0 | a future WMMA kernel; also the hard bug line |

`ridge = peak_FLOP/s ÷ achieved_copy_BW`, e.g. `10.10 TFLOP/s ÷ 233.4 GB/s = 43.3 FLOP/byte`.

**Consequences:**
- **The ridge that governs this engine is 43.3.** Its GEMMs use no tensor cores, so the prefill
  compute-bound crossover sits at **T ≳ ~43 tokens**, not 135. Corroborated empirically: whole-forward
  throughput is already on a compute plateau at T=128 (798 GFLOP/s) and barely rises by T=512 (856) —
  consistent with a crossover near 43, and **inconsistent** with 135, which would place T=128 in the
  memory-bound regime. (The plateau alone is indirect — occupancy and launch overhead can flatten a curve
  too — so §6b's direct peak measurement is what settles it.)
- **135 is the tensor-core ridge**, quoted throughout the early design work because a WMMA kernel was the
  assumed endpoint. It remains correct *for that kernel*, and remains the hard bug line: nothing on this
  GPU may exceed 31.51 TFLOP/s. It is not the efficiency denominator for anything actually built here.
- Decode's arithmetic intensity is **1–2 FLOP/byte** → **20–135× to the left of every one of the three
  ridges**. Memory-bound is not marginal, it is overwhelming. No amount of GEMM cleverness speeds up decode.
- Short prompts sit nearer memory-bound; long prompts are genuinely compute-bound, which is where a
  tiled-GEMM's %-of-roofline actually matters.

> **A correction worth keeping visible.** An earlier revision of this section asserted a CUDA-core peak of
> 16.0 TFLOP/s *derived from the clock* (24 SM × 128 lanes × 2 × 2.61 GHz) and a ridge of ≈69 from it. That
> violated the project's own rule — microbench, don't assume. Measured (§6b), the achievable CUDA-core GEMM
> is **10.10 TFLOP/s**, so the governing ridge is **43.3**. The 69 was simply wrong; it happened to sit near
> the FMA-pipe ridge (67.0), which is a bug line, not a denominator. Coincidence, not corroboration.

---

## 5. The director map — which stage moves which regime (spend effort here)

| Ladder stage | Primary regime moved | Expected decode payoff | Expected prefill payoff |
|---|---|---|---|
| 0. Pure-C fp32 | (CPU baseline) | — | — |
| 1. Naive CUDA | correctness re-gate | GPU baseline | GPU baseline |
| 2. Tiled/SMEM GEMM | **prefill (compute)** | **low** (decode is BW-bound) | **high** — pays off here |
| 3a. Fused LN+matmul | traffic | modest (fewer round-trips) | modest |
| 3b. Flash-style attention | traffic (no score matrix) | helps at long ctx | helps at long ctx |
| 4. INT8 / INT4 | **decode (fewer bytes)** | **high** — the main decode lever | some |
| 5. KV cache + planner | **decode (no recompute)** | **high** — essential for decode | n/a (prefill fills cache) |

> **The single most important line:** stage 2 (tiled GEMM) is a **prefill win and a near-non-event for
> decode**; stages **4 and 5** are where decode tok/s actually moves. If the headline is decode
> throughput, do not over-invest in stage 2 expecting decode gains — the roofline says they won't come.

> **Annotation — Stages 3 & 4 measured (director-map calibration).**
> - **Row 3b (flash) — CONFIRMED and sharpened.** "Helps at long ctx": the win grows with T, 1.29× @128 →
>   **1.72× @512**; isolated attention 14.5×. Attention's share of prefill@512 fell 44.80% → 5.27%.
> - **Row 3a (fused LN+matmul) — FALSIFIED for prefill, REVERTED.** Predicted "modest"; measured an
>   above-noise **2.9% slowdown**. The honest prefill prediction was ≈0 (LayerNorm is 0.28% of prefill@512),
>   and negative once the fusion adds per-consumption work.
> - **Row 3a for true M=1 decode — `[CLOSED at Stage 5, 2026-07-11]`.** This line used to say the conjecture
>   "stays OPEN for true M=1 decode (LN's share is far larger there)". Stage 5 built true M=1 decode and
>   **redirected the premise rather than confirming it.** (i) LN's M=1 share is **not measurable with this
>   engine's tooling**: `profile_decode`'s per-op split trips its own validity guard (sum-of-parts/whole =
>   **1.51×**; each per-op sync adds ~7.7 µs to a ~2 ms, 135-launch step), so those shares are **not used**,
>   and no fused-LN claim may rest on them. (ii) What *is* measured says the M=1 lever is elsewhere: the
>   decode gap to llama.cpp is **0.3831 ms of fixed per-step overhead — the whole gap at ctx=128, 73% of it
>   at ctx=1023** — against **135 kernel launches**, several 1-block kernels (an M=1 LayerNorm occupies 1 of
>   24 SMs), where llama.cpp captures the step in a **CUDA graph**. **The lever is the launch count, not the
>   activation round-trip** — and CUDA-graph capture is deliberately not built (DESIGN §2), which is why the
>   gap can be *attributed* to it. **No fused-LN decode speedup is claimed; none was measured.**
>   Full outcome recorded at the bet itself, DESIGN §5.

> ## ⛔ SUPERSEDED — the annotation immediately below is the **Stage-4-era** verdict, kept as the record of
> ## what was known then. **Stage 5 made row 4 testable and REFUTED it** (1.02–1.16×, bounded at 1.278× by
> ## the fp16 tied head). **The standing verdict is the "✅ RESOLVED" block further down; read that one.**
> *Kept, not deleted: "untested, not refuted" was the honest call at Stage 4, and the reasoning below is
> what correctly predicted that a KV cache alone would not surface the win — which is why Stage 5 shipped a
> GEMV. Deleting it would erase a prediction that came true.*

> **Annotation — ROW 4 (INT8 → decode "high"), as measured at Stage 4.**
> *Deliberately mirrors the row-2 annotation below, because it is the same structural situation.*
>
> **"High decode payoff" is UNTESTED, not refuted `[as of Stage 4 — now REFUTED, see the RESOLVED block]`.**
> As with row 2, the cell describes a **true (M=1)
> decode, which does not exist in this engine until the KV cache (Stage 5)**. Stage 4 could therefore only
> measure INT8 on (i) **prefill** and (ii) the **no-KV recompute-decode (M≫1)** harness — both
> reuse/compute-shaped, neither weight-traffic-bound — plus (iii) the one M=1-shaped op that does exist,
> the **tied-head GEMV**. All three measured **1.00×**.
>
> **(iii) is the load-bearing finding.** The M=1 head GEMV streams 77.2 MB fp16 → 38.6 MB int8 and *still*
> shows 1.002×, because the 16×16 tiled GEMM at M=1 **computes 16 output rows and discards 15**. Two
> falsifiable predictions were tested: if weight-BW-bound, INT8 gives ~2× → **measured 1.002×, rejected**;
> if compute-bound on the discarded rows, then **M=1 and M=16 cost the same** → **measured 1.4351 vs
> 1.4346 ms, confirmed**. Corroborating: at M=1 it executes **860.6 GFLOP/s = 98.8% of its own M=512
> throughput** while pulling only **53.8 GB/s = 23% of achieved copy BW**. Same arithmetic, half the bytes,
> same time. **A byte-halving cannot surface against a compute-bound kernel.**
>
> **Therefore: INT8's decode payoff becomes expressible ONLY when paired with a true M=1-shaped GEMV kernel
> (one output row per unit of work, no 16-row tile) AND the KV cache — i.e. Stage 5.** A KV cache alone will
> not surface it. Row 4 and row 2 both become **falsifiable for the first time in Stage 5**, and are to be
> tested there.

> ## ✅ RESOLVED — Stage 5 measured 2026-07-09. True (M=1) decode now exists; both rows are settled.
>
> **Row 5 (KV cache → decode "high", essential) — CONFIRMED, emphatically.** Cost of one more token:
> ctx=128 **14.68×**, ctx=512 **55.98×**, ctx=1023 **98.33×** (30.6→2.09 ms, 110.2→1.97 ms, 221.4→2.25 ms;
> [min,max] disjoint). The win grows with ctx exactly as O(T) recompute vs O(1) append predicts.
>
> **Row 2 (tiled GEMM → decode "low") — CONFIRMED.** With the KV cache and no GEMV, the decode path uses the
> tiled GEMM at true M=1: naive→tiled is **1.396× @ctx=128, 1.379× @ctx=512** (above noise), against **5.57×
> at prefill@128**. The tiling-for-**reuse** win is absent — as the roofline requires, since at M=1 there is
> only one row to reuse. So it is not literally "flat", but the cell said *low*, and low is what it is.
>
> ⚠ **Mechanism corrected 2026-07-10** (pre-public audit; BENCHMARKS "Stage 5 (2)"). This paragraph used to
> attribute the ~1.4× to coalescing "amplified by the tied head's 50257 rows." **The head does the opposite.**
> At M=1 on the head shape the 16×16 tiled GEMM is **2.00× slower** than naive (ncu: 962.7 → 1927.5 µs, both
> reading the same 77.3 MB from DRAM) — it computes 16 output rows and masks 15, so it burns 16× the FLOPs on
> the model's largest tensor. Decomposed at ctx=128: the 48 block matmuls give **1.662×** (that *is* coalescing),
> the tied head gives **0.586×**, and the two compose to the measured **1.396×**. The head is the **brake**.
> The verdict is untouched — the point of row 2 was that *reuse* cannot pay at M=1, and it does not.
> This is the same tile-waste that Stage 4 measured (head M=1 ≡ head M=16 in time) and that forced Stage 5's
> dedicated M=1 GEMV.
>
> **Row 4 (INT8 → decode "high", the main decode lever) — REFUTED for this model and this build.** With the
> KV cache AND the true M=1 GEMV — i.e. with every precondition this annotation demanded — INT8 buys
> **1.018× @128, 1.119× @512, 1.163× @1023**, above noise only at the longest context. The cause is measured,
> not inferred, and it is *structural*, not a kernel defect:
> 1. **The M=1 GEMV is not the problem.** On the one tensor large enough to be DRAM-bound (the 77.2 MB tied
>    head) it achieves **243.2 GB/s = 104% of copy BW, 98% of read BW.** It is at peak.
> 2. **The tied head is 31% of the fp16 weight bytes and must stay fp16 in BOTH builds** — Stage 4's
>    pre-registered quality kill-test forbids quantizing it (alone it caused KL 0.257 and Δppl +0.549). It
>    costs 0.317 ms in every step, identically. *Unhalvable, by the quality gate.*
> 3. **The 48 quantizable block matmuls are 1.2–4.7 MB each**, so at M=1 they sit near a per-kernel latency
>    floor (`attnproj` takes 8.2 µs whether fp16 or INT8). Streamed back-to-back (169.9 vs 84.9 MB, both ≫ the
>    32 MiB L2), they give **1.405×, not 2.00×**.
>
> ⇒ **Weight-side ceiling on the whole-step speedup = 1.278×**, before attention/LayerNorm/add/GELU and 135
> kernel launches per step dilute it further. The measured 1.02–1.16× is exactly consistent with that ceiling.
> Had the head been quantizable the ceiling would be 1.516× — still not "high" at batch=1 on a 124M model.
>
> **The honest reading of row 4:** quantization's decode lever is real (162.1 vs 248.9 MB streamed) but its
> size is set by *what fraction of the weight bytes you are allowed to quantize* and by *whether the tensors
> are large enough to be bandwidth-bound at M=1*. On GPT-2-124M, the answer to both is unfavourable. On a
> larger model — bigger per-layer matrices, an output head that is a smaller share of total weights — the
> same kernels would land much closer to the roofline. **This is a property of the model, not of the engine,
> and the row's "high" is a fair prediction for the models it was written about, just not for this one.**
>
> **INT4 caveat (recorded now, so it is not discovered by re-tuning a bound later).** Stage 4's shipped
> kill-test build (tied head kept fp16) passes gate (c) at **max KL 1.495e-2 against the 0.02 bound — a
> margin of only 1.34×**, where fp16 had 10×. The 48 quantized *block* matmuls consume nearly the whole KL
> budget on their own (blocks-only KL 1.283e-2). **INT4 therefore has no headroom on this model** without
> further mixed precision. QUALITY_GATES' INT4 bound (Δppl ≤ +1.0) is **pre-registered and LOCKED**: an INT4
> attempt that breaches it is an honest negative result, **not** an invitation to re-tune the bound.

> **Annotation — Stage 2 measured (director-map calibration).**
> The "Stage 2 → decode **low/~flat**" cell concerns a **true (M=1) decode**, which **does not exist in
> this engine until the KV cache (Stage 5)**. With no KV cache, the Stage-1/2 "decode" harness
> FULL-RECOMPUTES the growing sequence, so its GEMMs run at **M = context length (~34–161)** — the
> tiling-for-**reuse**, prefill-shaped regime. Stage 2 therefore measured a **5.4× "recompute-decode"
> speedup that is a GEMM reuse win — NOT a decode result and NOT a regime violation** (isolated M-sweep:
> M=1 → **1.18×**, coalescing-only; M=16–161 → 6.5–7.4×, reuse — see BENCHMARKS Stage 2). So "decode
> ~flat" is **not yet falsifiable and was not tested** here. **Roofline physics & strategy UNCHANGED:**
> tiled recompute-decode = 36.1 tok/s = **3.8% of the 941 tok/s copy-BW ceiling**; **INT8 (S4)** + **KV
> cache (S5)** remain the real decode levers, and **Stage 5 is where "decode ~flat" first becomes testable.**

---

## 6. Microbench — MEASURED denominators (✅ 2026-07-08)

> Status: **DONE.** `bench/microbench.cu` on this machine. The provisional denominators above are now
> replaced by measurements; the evidence is here. Method + clock state below so the numbers are
> reproducible and honest (BENCH_PROTOCOL §4, §7).

**Method.** Buffers **256 MiB/ea ≫ 32 MiB L2** (so DRAM, not cache). CUDA-event timing with sync;
**warmup 10**, then **median + min/max** over 50 iters (BW) / 60–100 iters (GEMM) — never best-of-N.
GEMM = `cublasGemmEx`, fp16 in, tensor op, square M=N=K sweep {1024,2048,4096,8192}, peak at large
size (cuBLAS = measurement instrument for the *achievable* compute ceiling; model GEMMs are from-scratch).

| Quantity | Median | Spread (min–max) | % of 256 theo | Notes |
|---|---|---|---|---|
| Copy BW (2N read+write) | **233.4 GB/s** | 230.7 – 234.6 | 91.2% | STREAM-convention; conservative decode denom |
| Read BW (1N read-only) | **248.9 GB/s** | 246.4 – 249.7 | 97.2% | closer to decode's weight-streaming pattern (upper bracket) |
| FP16 GEMM, **fp32-accum** | **31.51 TFLOP/s** | 31.41 – 31.56 @8192³ | — | **tensor-core ceiling**: the hard bug line, and the WMMA headroom left on the table. *Not* the prefill denominator for this engine — see §6b |
| FP16 GEMM, fp16-accum | 60.98 TFLOP/s | 60.73 – 61.34 @8192³ | — | **1.94×** faster → confirms consumer-Ada half-rate fp32-accum; worse numerics, **not** used by model |
| **Tensor-core ridge** (31.5 / 233) | **135 FLOP/byte** | (≈127 vs read-BW) | — | the ridge a **future WMMA** kernel would face. The ridge governing the kernels actually shipped is **43.3** (§6b) |

**Clock / thermal state during the runs** (nvidia-smi @100 ms): mem clock **8000 MHz throughout**
(memory intact, as §1 predicts). fp32-accum GEMM: SM **2610 MHz, ~66 W, 55 °C** — *not* power- or
thermal-throttled (66 W ≪ 105 W cap), so 31.5 is a clean tensor-throughput ceiling. fp16-accum:
2595 MHz, ~94 W, 60 °C (2× math → ~2× power, still < cap).

**Sanity checks (all pass).** (a) Every BW < 256 theoretical → no L2-cache artifact. (b) fp16-accum/
fp32-accum = 1.94× ≈ the known 2× consumer-Ada penalty → the GEMM path is behaving as physics
predicts. (c) fp32-accum median reproduced bit-identically (31.51) across two separate runs.

**Decode ceilings recomputed at the measured achieved BW** (hard ceiling = weight_bytes ÷ BW; a real
kernel reaches a *fraction* — launch overhead, KV & activation traffic, imperfect overlap):

| Precision | Weight bytes | @256 theo (bug line) | @233 copy BW | @249 read BW |
|---|---|---|---|---|
| fp32 | 496 MB | ~516 tok/s | ~470 | ~502 |
| **fp16** | 248 MB | ~1032 | **~941** | ~1004 |
| INT8 | 124 MB | ~2065 | ~1883 | ~2008 |
| INT4 | 62 MB | ~4129 | ~3765 | ~4015 |

> Denominators are now **measured, not provisional** — speedups may be reported as final against them.
> The **theoretical-256** column remains the hard **measurement-bug line** (a measured decode number
> above it = a bug); the **233 copy-BW** column is the realistic hard ceiling to target.

> **Rounding footnote (doc pass 2026-07-09).** The table above divides by a rounded **248 MB**. The exact
> weight bytes are 124,439,808 × 2 = **248.9 MB**, giving fp16 **938 copy / 1000 read / 1029 theo** (vs the
> 941/1004/1032 quoted); INT8 **1876/2000/2057**; INT4 **3751/4000/4114**. The published figures are
> therefore ~0.3–0.4% **optimistic**, i.e. the bug line is very slightly *permissive*, never falsely
> tripping. Every stage to date sits 20–100× below these ceilings, so no verdict anywhere changes. The
> quoted numbers are kept as-written so the committed BENCHMARKS blocks stay internally consistent; use the
> exact column for any new claim near a ceiling.
>
> **`[Stage 4]`** A build that keeps some tensors fp16 streams neither 248.9 nor 124.4 MB. Ceilings must be
> derived from the bytes a build *actually* streams — `bench/correctness_cuda.cu` now computes them from
> `GPT2QWeightsGPU::streamed_bytes` rather than hardcoding. The shipped INT8 build (tied head kept fp16)
> streams **162.1 MB** → **1440 copy / 1535 read / 1579 theo**.

---

## 6b. CUDA-core compute peak — MEASURED (✅ 2026-07-09) · the prefill denominator that was being *assumed*

> **Why this section exists.** §6 pins the **tensor-core** GEMM peak (31.5 TFLOP/s). But **no kernel this
> engine ships uses tensor cores** — Stage 2 tiled, Stage 3b flash and Stage 4 INT8 are all CUDA-core
> GEMMs, and weight-only INT8 cannot reach `dp4a` at all (activations stay fp16). Their efficiency
> denominator was, until now, a **clock-derived assumption** (24 SM × 128 × 2 × 2.61 GHz = 16.0 TFLOP/s).
> **G2 says microbench, do not assume.** So it is measured here, the same way §6 measured 31.5.

**Method.** `bench/microbench.cu cudacore`. Two independent instruments:
- **(4a) fp32 FMA pipe** — `k_fma_peak`: 32 independent FMA dependency chains per thread (hides the ~4-cycle
  FMA latency → measures *issue* throughput, not latency), **register-resident: zero loads, zero stores**.
  Verified in SASS with `cuobjdump -sass`: **160 FFMA, 0 LDG, 1 (unreachable) STG** — the FMAs are not
  dead-coded and no memory traffic contaminates the number. **Sustained warmup to clock steady state**
  before timing (BENCH_PROTOCOL §4), then median over 50; never best-of-N.
- **(4b) achievable non-tensor GEMM** — cuBLAS `GemmEx` with `CUBLAS_COMPUTE_32F_PEDANTIC` +
  `cublasSetMathMode(CUBLAS_PEDANTIC_MATH)`, which forbid tensor-core paths. Same numerics as the engine
  (fp16 in, fp32 accumulate). This is the exact analog of §6's instrument, and it — not (4a) — is the honest
  denominator for a %-of-roofline claim.

| Quantity | Median | Spread | Notes |
|---|---|---|---|
| **CUDA-core fp32 FMA pipe** | **15.64 TFLOP/s** | 15.57 – 15.65 (N=50) | = **97.5%** of the 128-lane issue width @ 2610 MHz → **CUDA-core BUG LINE** |
| **CUDA-core GEMM, fp16 in / fp32 acc, PEDANTIC** | **10.10 TFLOP/s** | 10.09 – 10.10 @8192³ | **the prefill denominator**; 64.6% of the raw FMA pipe (a real GEMM never reaches it) |
| CUDA-core GEMM, fp32 SGEMM, PEDANTIC | 9.81 TFLOP/s | 9.73 – 9.88 @8192³ | cross-check: same pipe, same ballpark |

**Clock / thermal during the runs** (nvidia-smi sampled *during the FMA phase*): SM **2610 MHz** (sagging to
2595 late), mem 8000 MHz, **60 → 86 W**, **59 → 64 °C** — not throttled (≪ 105 W cap).

**Sanity checks (all pass).** (a) The non-tensor GEMM (10.10) is **3.1× below** the tensor GEMM (31.51) →
`PEDANTIC` really did disable tensor cores; had it not, the two would coincide and the label would be a lie.
(b) The FMA pipe reaches 97.5% of the 128-lane × 2-FLOP × 2.61 GHz issue width → the lane model is confirmed
and the remaining 2.5% is loop overhead (~3 non-FFMA instructions per 160 FFMA). (c) GEMM/FMA = 64.6%, a
plausible GEMM efficiency. (d) Every measured number is **below** the corresponding ceiling.

**Ridge points, both regimes** (÷ the measured 233.4 GB/s copy BW):

| Compute ceiling | TFLOP/s | Ridge (FLOP/byte) | Governs |
|---|---|---|---|
| CUDA-core GEMM, achievable | 10.10 | **43.3** | **every kernel this engine currently ships** |
| CUDA-core FMA pipe | 15.64 | 67.0 | bug line for a CUDA-core kernel |
| Tensor-core GEMM (§6) | 31.51 | 135.0 | a future WMMA kernel |

> **What this changed.** The previously *assumed* 16.0 TFLOP/s was optimistic as an achievable peak by
> **1.55×** (the achievable GEMM is 10.10), and the ridge it implied (≈69) was wrong: **the governing ridge
> is 43**. Decode's AI (1–2 FLOP/byte) sits far left of all three ridges, so **every decode verdict in this
> file is unchanged** — the memory-bound conclusion only gets stronger. What improves is the *prefill*
> efficiency statement: Stage 3b/4 prefill@512 = 856 GFLOP/s is **8.5% of the achievable CUDA-core GEMM
> ceiling** (not the 2.7%-of-tensor that understated it by 3.1×, nor the 5.35% the assumed 16.0 gave).

---

## 7. How to check a measured number against this file (the loop)

For every measurement:
- **Above** the relevant ceiling → measurement bug. Fix it; it is not a result. (Usual causes: timing
  excludes a real cost, no warmup, wrong precision, counting cached tokens as generated.)
- **Below** ceiling but < ~0.5× → profile; there is headroom the roofline says exists.
- **0.55–0.85×** of an honest ceiling, reproducible, apples-to-apples → believable. Record in
  `BENCHMARKS.md` with config + clocks + seed + median±spread, **prefill and decode separately**.
