# BENCHMARKS.md — measured results (the portfolio artifact)
### GPT-2-124M · RTX 4060 Laptop (105 W) · one validated block per run

> One block per **validated** run (BENCH_PROTOCOL §8).
> A number appears here **only** after: correctness gate passed (QUALITY_GATES §1) → measured
> apples-to-apples (BENCH_PROTOCOL) → checked ≤ its ROOFLINE ceiling. Prefill and decode always
> separate. Median + spread, never best-of-N.


---

## Results at a glance

### Table 1 — PRIMARY: three engines, one model, matched precision

GPT-2-124M · RTX 4060 Laptop (sm_89, 105 W) · same weights, same GPU, same session, clocks recorded.
**Precision is matched**: all three run fp16 / F16. Decode is **true M=1** in all three (each has a KV
cache), so it is a like-for-like comparison. PyTorch prefill is measured on the transformer trunk only,
because HF's `model(ids)` also runs the LM head over all P positions (+39.5 GFLOP at P=512) and ours
does not. Medians, never best-of-N.

| | **this engine** (fp16) | **llama.cpp** F16 (CUDA) | **PyTorch** fp16 eager |
|---|---|---|---|
| prefill @ P=128 | 27.548 ms | **2.732 ms** | 9.441 ms |
| prefill @ P=512 | 107.140 ms | **7.422 ms** | 9.395 ms |
| decode @ ctx=128 | 511.7 tok/s | **636.4 tok/s** | 107.4 tok/s |
| decode @ ctx=512 | 503.8 tok/s | **604.0 tok/s** | 107.9 tok/s |
| decode @ ctx=1023 | 448.6 tok/s | **586.3 tok/s** | 106.4 tok/s |

**Decode: 76–83% of llama.cpp, and 4.2–4.8× faster than PyTorch eager.**
**Prefill: 10.1× / 14.4× slower than llama.cpp** — and that gap factors exactly into two measured terms:

```
prefill gap @512 = 14.44x = 3.12x  (tensor cores / WMMA: 31.51 TF tensor ceiling / 10.10 TF CUDA-core ceiling)
                          x 4.63x  (GEMM maturity: llama.cpp reaches 39.3% of its ceiling; we reach 8.5% of ours)
```

The ceiling factor is an independent measurement; the second is the residual, equal to the ratio of how
much of its own ceiling each engine reaches. Neither is fusion: attention is 5.3% of our prefill, and the
~123 launches in a prefill forward cost well under 1% of it. See "External baseline" below.

No number from any engine exceeds its roofline ceiling. PyTorch eager is **not** a compute baseline at
this model size — its trunk prefill is 9.441 ms at P=128 and 9.395 ms at P=512, identical for 4× the work,
because it is host-dispatch-bound.

### Table 2 — SECONDARY: our optimization ladder (this engine only)

*Not a cross-engine comparison.* This is the from-scratch improvement path, stage by stage, on the same
GPU and the same gates. "Decode" before Stage 5 is a **no-KV full recompute** (M≫1, prefill-shaped) —
a true M=1 decode does not exist in this engine until the KV cache lands, which is exactly why the
early decode column is labelled, not celebrated.

| Stage | What it adds | prefill @512 | decode | note |
|---|---|---|---|---|
| 0 | pure-C fp32 reference | 3347 ms | ~1.6 tok/s (no-KV) | correctness oracle; CPU, not roofline-checked |
| 1 | naive CUDA fp16 | 840.0 ms | 6.7 tok/s (no-KV) | GPU baseline; uncoalesced by design |
| 2 | tiled / shared-memory GEMM | 184.8 ms (**4.55×**) | 36.1 tok/s (no-KV) | the "decode" win is GEMM reuse, **not** a decode result |
| 3b | flash attention (online softmax) | 107.3 ms (**1.72×**) | 46.7 tok/s (no-KV) | attention 44.8% → 5.3% of prefill |
| 3a | fused LayerNorm+matmul | — | — | **reverted**: measured 2.9% *slowdown*, above noise |
| 4 | weight-only INT8 | 107.1 ms (1.00×) | — | both gates pass via the pre-registered kill-test; **zero speedup, honestly** |
| 5 | KV cache + true M=1 GEMV | 107.1 ms | **511.7 tok/s** @ctx=128 | **14.7–98.3×** over no-KV recompute |

Stage 5 fp16 decode reaches 511.7 tok/s = **56% of the 920 tok/s copy-bandwidth ceiling** at ctx=128 —
inside the band the roofline calls believable, and confirmed independently by llama.cpp landing at 68%
of the same ceiling with CUDA graphs and fused kernels.

---

## Doc pass — record cleanliness (2026-07-09, after Stage 4; **no number was re-measured or changed**)

*A sweep of the four governing docs for pre-measurement residue and internal contradictions, now that the
denominators are measured. Every stage re-gates against the HF oracle, so no unsanitized number can survive
a stage; this pass fixes what the DOCS say, not what the engine did. Fixed in place, listed here.*

| # | Where | Was | Now |
|---|---|---|---|
| 1 | QUALITY_GATES §1(b),(c) | still stated the pre-A1 constants (`margin ≥ 0.05` = bug; `top-1 ≥ 99%` = pass/fail) — **self-contradictory** with §1.1 | marked SUPERSEDED for fp16/quant by A1; still binding for the fp32 path. Bug-catchers (a ≤1e-2, KL <0.02) untouched |
| 2 | ROOFLINE §1 | 256 GB/s labelled "decode denominator"; 31.5 TF labelled "prefill denom" | 256 = **bug line**; 233/249 achieved = decode denominator; 31.5 TF = bug line + WMMA headroom; **16.0 TF CUDA-core** added as the real prefill denominator for the shipped non-WMMA kernels |
| 3 | ROOFLINE §4 | ridge 135 presented as *the* ridge | 135 is the **tensor-core** ridge; the shipped CUDA-core kernels face their own. *(This row first said "≈69" from the assumed 16.0 TF — **superseded by item 12: the measured ridge is 43.**)* |
| 4 | ROOFLINE §3 | "achieved BW ~200 GB/s" (pre-measurement estimate) | recomputed at the measured 233.4 → realistic fp16 decode ≈ 515–750 tok/s |
| 5 | ROOFLINE §6 | ceilings divided by a rounded 248 MB | footnote: exact is 248.9 MB → fp16 938/1000/1029; published figures are ~0.3–0.4% **optimistic** (bug line slightly permissive, never falsely tripping). Quoted values kept so committed blocks stay consistent |
| 6 | ROOFLINE §5 | director map rows 3a/3b/4 unannotated vs their measured outcomes | annotation added (3b confirmed; 3a falsified+reverted; **row 4 "high decode payoff" = untested, not refuted**) |
| 7 | QUALITY_GATES §2 | "sanity anchor ≈29–30" fires falsely against our measured 25.57 | stride caveat recorded: anchor is a stride=1024 figure (30.18); the gate is stride=512 (25.57). Same harness, both run |
| 8 | QUALITY_GATES §2/§3 | kill-test + the 2 knobs written as still-to-do | knobs marked FROZEN (Phase 0, pre-Stage-4); kill-test marked EXECUTED with its measured outcome |
| 9 | DESIGN.md §3 G1/G2/G3/G7 | 4 open `[VERIFY]` tags; "SmolLM **or** GPT-2"; "~270 MB → ~950 tok/s" placeholder | model LOCKED = GPT-2-124M (124,439,808 params → 248.9 MB); tags discharged to `[VERIFIED]` with the measured values |
| 10 | DESIGN.md §5 | 4 open `[CONJECTURE]`s, 3 of them already kill-tested | outcome pointers added (bets and bounds unchanged) |
| 11 | BENCH_PROTOCOL §3 | "no-KV decode" — too weak, twice near-misread | sharpened to **"no-KV recompute-decode (M≫1)"** with the consequence spelled out |
| **12** | **ROOFLINE §1/§4 + CLAUDE §3 + Stage 3b/4 blocks** | **the CUDA-core peak 16.0 TF was DERIVED from clock** (G2: "microbench, do not assume") and ridge ≈69 rested on it | **MEASURED** (Phase −1b / ROOFLINE §6b): FMA pipe **15.64 TF**, achievable GEMM **10.10 TF**. **16.0 was optimistic by 1.55×; the governing ridge is 43, not 69.** Prefill efficiency restated (S3b/S4 @512 = **8.5%**, not 5.35%). No kernel re-run; no measured GFLOP/s changed. Caught before Stage 5; a hard blocker until measured. |

> **Item 12 note — the doc pass introduced the defect it later caught.** The 16.0 TF figure entered the docs
> *in this very pass* (item 2), correctly labelled "derived from clock (not microbenched)", and was then used
> as a denominator two lines later. Labelling an assumption does not license using it as a measurement. The
> brain caught it before Stage 5; it is now measured, and the label removed by making it true.

**Confirmations requested with this pass:**
- **Gate-(b) t=28 `|top-1 logit| = 135` is MEASURED, not carried over.** `bench/correctness_cuda.cu:185`
  computes `mag = fabsf(row[pred])` from the live logit row each run; re-printed 2026-07-09:
  `pos 28: pred 1663 != ref 7866 (margin 0.1250 < 3*ulp 0.3750 @|logit|~135 -> tolerated near-tie)`.
  Independent of the `%.0f` rounding: the printed `3*ulp = 0.3750` ⟹ `fp16_ulp = 0.1250 = ldexp(1, e−11)`
  ⟹ `e = 8` ⟹ `|logit| ∈ [2⁷, 2⁸) = [128, 256)`. **≥ 128, i.e. 2× the |logit|=64 pivot** below which the
  0.125 margin would have been a BUG. **Stage-1 gate (b) verdict is robust; no mis-tolerated flip.**
- **Amendment A1 lives in QUALITY_GATES.md §1.1** (not only in BENCHMARKS.md) — dated 2026-07-08,
  with rationale and an explicit **UNCHANGED bug-catchers** clause (gate (a) ≤ 1e-2,
  gate (c) KL < 0.02). Behavior was already correct at Stage 4: INT8's max KL **0.257 was NOT A1-tolerated**;
  it failed gate (c) and forced the kill-test. This pass adds the supersession pointers from §1(b)/§1(c)
  **to** §1.1, which were missing.

---

## Environment (reproducibility anchor — recorded once, re-note if it changes)

| Field | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU (AD107, **sm_89**), 24 SM, 8188 MiB, L2 = 32 MiB* |
| Power cap | 105 W (`nvidia-smi` max) |
| Driver | 561.09 |
| CUDA toolkit | 12.6 (nvcc V12.6.20) |
| Host compiler | MSVC 14.44 (VS 2022 BuildTools), `vcvars64` |
| Build flags | `nvcc -O3 -arch=sm_89` |
| OS | Windows 11 (26200) |
| PyTorch / transformers | torch 2.12.1+cu126 · transformers 5.13.0 · datasets 5.0.0 (venv on E:; Python 3.14.4) — **installed**; Phase 0 oracle + fp16 PPL generated |
| Clock policy | not locked yet; **clock + thermal recorded per run** via `nvidia-smi -lms 100` |

\* L2 = 32 MiB from the runtime API (`cudaDeviceProp.l2CacheSize`); ROOFLINE §1 corrected to match (was 24 MB). Not load-bearing (256 MiB microbench buffers ≫ 32 MiB either way).

---

## Phase −1 — Microbench (denominators) · ✅ VALIDATED 2026-07-08

*Not a model stage; this pins the ROOFLINE denominators every later number is checked against.
Full evidence + method + clock state written into ROOFLINE.md §6.*

```
artifact:         bench/microbench.cu  (nvcc -O3 -arch=sm_89 -lcublas)
method:           buffers 256 MiB/ea (>> 32 MiB L2); CUDA-event timing w/ sync;
                  warmup=10; median + min/max over 50 (BW) / 60–100 (GEMM) iters; never best-of-N.
environment:      sm_clk=2610 MHz (GEMM, fp32-acc)  mem_clk=8000 MHz  power=66 W  temp=55 °C
                  thermal=steady (not throttled: 66 W << 105 W cap)  driver=561.09  cuda=12.6

achieved copy BW (2N r+w):   233.4 GB/s   median  (230.7–234.6)   = 91.2% of 256 theo   [decode denom, conservative]
achieved read BW (1N):       248.9 GB/s   median  (246.4–249.7)   = 97.2% of 256 theo   [decode denom, upper bracket]
FP16 GEMM peak fp32-accum:    31.51 TFLOP/s median (31.41–31.56 @8192³)                 [PREFILL denom]
FP16 GEMM peak fp16-accum:    60.98 TFLOP/s median (60.73–61.34 @8192³)  = 1.94×         [headroom only; not model numerics]
ridge point (31.5/233):       135 FLOP/byte   (≈127 vs read BW)

sanity:           all BW < 256 theoretical (no L2 artifact); 1.94× ratio = known consumer-Ada
                  fp32-accum half-rate; fp32-accum median reproduced identically across 2 runs.
verdict:          denominators now MEASURED (were provisional). Decode is memory-bound by ~65–135×
                  (ridge 135 vs decode AI ~1–2 FLOP/byte) — regime verdicts in ROOFLINE hold.
```

**Recomputed decode ceilings (measured BW):** fp16 ~941 tok/s (copy) / ~1004 (read); INT8 ~1883/2008;
INT4 ~3765/4015. Theoretical-256 column stays the **bug line** (a decode number above it = a bug).

---

## Phase −1b — CUDA-core compute peak (the denominator that was being ASSUMED) · ✅ VALIDATED 2026-07-09

*Phase −1 pinned the **tensor-core** peak (31.5 TFLOP/s). But no kernel this engine ships uses tensor cores,
so 31.5 is a bug line, not an efficiency denominator. Stages 2–4 had been quoting a **clock-derived**
CUDA-core peak (24 SM × 128 × 2 × 2.61 GHz = 16.0 TFLOP/s) — an **assumed** denominator, which is exactly
what DESIGN.md G2 forbids and what Phase −1 refused to do for 31.5. Flagged before Stage 5; measured here. Full evidence + method + clock state in ROOFLINE §6b.*

```
artifact:         bench/microbench.cu  (new mode: `microbench.exe cudacore`)  nvcc -O3 -arch=sm_89 -lcublas
method:           (4a) fp32 FMA pipe -- k_fma_peak: 32 independent FMA chains/thread (hides ~4-cyc FMA
                       latency -> measures ISSUE rate); register-resident, ZERO memory traffic.
                       SASS-verified (cuobjdump -sass): 160 FFMA, 0 LDG, 1 unreachable STG -> the FMAs are
                       not dead-coded and no load/store contaminates the number.
                       SUSTAINED warmup (2.5 s) to clock steady state, then median over 50. Never best-of-N.
                  (4b) achievable non-tensor GEMM -- cuBLAS GemmEx, CUBLAS_COMPUTE_32F_PEDANTIC +
                       cublasSetMathMode(CUBLAS_PEDANTIC_MATH) -> tensor-core paths FORBIDDEN. Same numerics
                       as the engine (fp16 in, fp32 accumulate). This is the exact analog of Phase -1's
                       instrument for 31.5, and it -- not (4a) -- is the honest efficiency denominator.
environment:      sm_clk 2610 MHz (-> 2595 late), mem 8000 MHz, 60->86 W, 59->64 C, NOT throttled (<<105 W)
                  driver 561.09, CUDA 12.6.  Clocks sampled DURING the FMA phase, not across the whole run.

  CUDA-core fp32 FMA pipe          : 15.64 TFLOP/s median (15.57 - 15.65, N=50)  -> CUDA-core BUG LINE
                                      = 97.5% of the 128-lane issue width @2610 MHz
  CUDA-core GEMM (fp16 in/f32 acc) : 10.10 TFLOP/s median (10.09 - 10.10 @8192^3) -> PREFILL DENOMINATOR
  CUDA-core GEMM (fp32 SGEMM)      :  9.81 TFLOP/s median ( 9.73 -  9.88 @8192^3) -> cross-check, same pipe

sanity:           (a) non-tensor GEMM 10.10 is 3.1x BELOW the tensor GEMM 31.51 -> PEDANTIC really disabled
                      tensor cores. Had they coincided, the "non-tensor" label would have been a lie.
                  (b) FMA pipe = 97.5% of 128 lanes x 2 FLOP x 2.610 GHz -> the lane model is confirmed;
                      the missing 2.5% is loop overhead (~3 non-FFMA instrs per 160 FFMA).
                  (c) GEMM/FMA = 64.6% -- a real GEMM never reaches the raw FMA pipe. Plausible.
                  (d) every number below its ceiling.

ridge points (/ the measured 233.4 GB/s copy BW):
  CUDA-core GEMM achievable  10.10 TF -> ridge  43.3 FLOP/byte  <- GOVERNS EVERY KERNEL THIS ENGINE SHIPS
  CUDA-core FMA pipe         15.64 TF -> ridge  67.0            <- bug line for a CUDA-core kernel
  Tensor-core GEMM (§6)      31.51 TF -> ridge 135.0            <- a future WMMA kernel

WHAT THIS CHANGED (and what it did not):
  * The assumed 16.0 TF was OPTIMISTIC by 1.55x vs the achievable GEMM (10.10). The ridge it implied,
    "~69", was WRONG -> the governing ridge is 43. (69 happened to sit near the FMA-pipe ridge 67.0,
    which is a bug line, not a roofline denominator. Coincidence, not corroboration.)
  * A clock64()-based "FLOP/SM/cycle" cross-check was written, run, and DISCARDED: it reported 374
    FLOP/SM/cycle against a hardware ceiling of 256 (146% of issue width -- impossible), because
    clock64()'s tick rate is not calibrated on this device (it implied 1739 MHz while the same kernel
    sustained 15.63 TF, which 128 lanes x 24 SM cannot reach below ~2.55 GHz; 2610/1739 = 1.501).
    Rather than publish a number from an uncalibrated counter, issue width is computed from CUDA events
    + nvidia-smi only. The dead end is recorded, in the kernel comment and here, rather than quietly cut.
  * DECODE VERDICTS UNCHANGED. Decode AI (1-2 FLOP/byte) is far left of all three ridges; the
    memory-bound conclusion only strengthens. Nothing in Stages 0-4 crosses a ceiling under any reading.
  * PREFILL EFFICIENCY RESTATED (no kernel re-run, same measured GFLOP/s, honest denominator):
        stage                     GFLOP/s   % achievable CUDA-core GEMM   % FMA pipe   % tensor (bug line)
        S1 naive     @512           109         1.08%                       0.70%        0.35%
        S2 tiled     @512           497         4.92%                       3.18%        1.58%
        S3b flash    @512           856         8.48%                       5.47%        2.72%
        S3b isolated attention      839         8.31%                       5.36%        2.66%
        S4 int8      @512           856         8.47%                       5.47%        2.72%
```

---

## Phase 0 — Scaffolding (reference/eval infra) · ✅ VALIDATED 2026-07-08

*Not a perf stage; establishes the correctness oracle + quality baseline every kernel is judged against.*

```
weights:       tools/export_gpt2.py -> weights/gpt2_124m_fp32.bin (497.8 MB, 148 tensors, fp32)
               param count = 124,439,808 (EXACT vs config.h)   sha256(body)=03cdb9f90df10be3...
               layout: all linear weights transposed HF Conv1D[in,out] -> [out,in]  (one matmul form C=A*B^T)
               transpose correctness vs HF: PASS (c_attn/c_fc/c_proj == HF.weight.T); wte tied-head == HF: PASS
               C loader model/weights.{c,h}: loads + validates magic/dims/size; values byte-match Python: PASS
oracle:        tools/reference.py -> refdumps/{fp32(CPU),fp16(CUDA)}/  (gates a/b/c ground truth)
               per-block hidden [28,768]x14, greedy ids+margins [128], eval logits [512,50257]
               FROZEN correctness prompt (QUALITY_GATES §3 knob #2): the GPT-2 "unicorn" prompt (28 tokens)
               fp32 greedy sanity: "...the unicorns were able to survive in the wild for up to a year, and..."  (correct GPT-2 behavior)
               last-layer capture CORRECTED (HF has long recorded POST-ln_f as hidden_states[-1] — not a 5.x quirk):
               block_11 = pre-ln_f raw block output (ln_f forward-hook input); final_ln = single ln_f (hook output).
               dump-time self-check final_ln == ln_f(block_11): rel=0.0 -> PASS (fp32 & fp16). (Prior dump mislabeled
               block_11 as post-ln_f and final_ln as double-ln_f; found by STAGE-0 gate (a); re-dumped.)
quality base:  tools/eval_ppl.py -> PPL(fp16) = 25.57   [WikiText-2-raw val, 249,749 tok, window=1024 stride=512]
               HARNESS CONFIRMED: the SAME harness at stride=1024 (non-overlap) -> PPL(fp16)=30.18, inside the
               QUALITY_GATES "~29-30" anchor -> the 25.57 is a stride CONVENTION difference, NOT a bug. The
               pre-registered stride=512 (more overlap = more context per scored token) legitimately yields 25.57
               (matches HF's own perplexity tutorial). fp16 baseline for the stage-4 Δppl gate (INT8 ≤ +0.3, same harness).
env:           venv on E: (C: was full: 0.4 GB free) — torch 2.12.1+cu126, transformers 5.13.0, datasets 5.0.0
```

**INT8 Δppl bound (QUALITY_GATES §3 knob #1), frozen now before stage 4:** ppl_int8 − 25.57 ≤ **+0.3**.

---

## Ladder stages (BENCH_PROTOCOL §8 template — one block each, filled as they pass)

_Stage 0 (pure-C fp32 reference) → Stage 5 (KV cache) blocks appended below as each is validated._

### Stage 0 — pure-C fp32 forward (correctness reference + CPU baseline) · ✅ VALIDATED 2026-07-08

*The from-scratch fp32 forward that every GPU kernel is validated against. **Correctness is the deliverable**;
CPU throughput is an informational baseline — a different device from the GPU ROOFLINE ceilings, so it is
**not** roofline-checked (those ceilings gate stages 1–5). Engine FROZEN after this stage.*

```
stage:            0 — pure-C fp32 forward  (cpu/forward_cpu.c, bench/correctness.c)
kernel/precision: pure-C fp32; float storage + double accumulation (matmul / LayerNorm / softmax)
                  gelu_new (tanh); pre-LN (biased var, eps-in-sqrt, 1e-5); attn scale 1/sqrt(64); tied head
config:           correctness prompt = frozen "unicorn" (28 tok); gate(a) P=28; gate(b) greedy N=128;
                  gate(c) eval P=512 (WikiText-2 val ids); sampling = greedy/argmax (deterministic)
environment:      CPU AMD Ryzen 7 7840H (8C/16T); MSVC 14.44 /O2 /std:c11 /openmp /fp:precise; 16 threads
                  (CPU baseline -> GPU clock/thermal N/A; driver/CUDA unused this stage)
included:         tokenization=n (pre-tokenized IDs, refdumps/meta.json)   weight_load=n

CORRECTNESS (QUALITY_GATES §1 — vs HF fp32 oracle refdumps/fp32; the gate that defines "done"):
  gate(a) per-layer rel_err <= 1e-4 : PASS  embed 0.0 ; block_0..11 8.0e-8..2.2e-7 ; final_ln 4.58e-7
  gate(b) greedy match N=128        : PASS  teacher-forced 128/128, 0 bug ; free-run greedy 128/128 (full)
  gate(c) top-1 >=99% & KL < 0.02   : PASS  top-1 512/512 (100.00%) ; max KL(ref||ours) 6.48e-10

CPU baseline (informational; median + min–max, warmup done; NOT roofline-checked — CPU != GPU ceilings):
  prefill  @P=128 :  915.4 ms median  (777.7 – 1059.1, N=10)
  prefill  @P=512 : 3347.1 ms median  (3310.6 – 3992.3, N=5)
  no-KV decode    :  631.4 ms/tok median (233.0 – 1101.3, N=123) @ ctx 33->155  ->  ~1.6 tok/s
baseline:         n/a — this IS the from-scratch baseline (PyTorch eager / llama.cpp comparison begins at GPU stages)
notes:            no-KV decode per-token grows ~linearly with ctx (wide spread expected; the stage-5 KV win is
                  measured no-KV vs KV under this same protocol). CPU ~26 GFLOP/s (scalar double-accum),
                  internally consistent prefill<->decode. Engine now FROZEN as the fp32 reference for stages 1–5.
```

### Stage 1 — naive CUDA port (GPU baseline) · ✅ VALIDATED 2026-07-08

*First GPU stage: correctness RE-gated on GPU vs the HF **fp16** oracle (a port is a prime place for
silent bugs). Establishes the GPU baseline ONLY — no optimization payoff claimed. Naive kernels are
uncoalesced → BW/latency-starved (40 W draw); tiled GEMM (stage 2) is the prefill lever, INT8 (4) +
KV-cache (5) the decode levers (director map).*

```
stage:            1 — naive CUDA fp16  (cuda/kernels_naive.cu, cuda/forward_cuda.cu; swappable backend)
kernel/precision: naive fp16 (one thread / output element; no shared mem, no coalescing) + fp32 accumulation
                  weights = __float2half(fp32 master) == torch .half(); gelu_new; causal attn scale 1/sqrt(64)
config:           gate(a) P=28 ; gate(b) greedy N=128 teacher-forced ; gate(c) eval P=512 ; sampling greedy/argmax
environment:      RTX 4060 Laptop (AD107, sm_89), 24 SM, L2=32 MiB ; nvcc -O3 -arch=sm_89 (CUDA 12.6, driver 561.09)
                  clocks NOT locked -> sustained SM 2610 MHz, mem 8000 MHz, 40 W, 52 C, 100% util (NOT throttled)
included:         tokenization=n (pre-tokenized IDs)   weight_load=n (fp16 H2D upload excluded)

CORRECTNESS (QUALITY_GATES §1 + Amendment A1; vs HF fp16 oracle refdumps/fp16):
  gate(a) per-layer rel_err <= 1e-2 : PASS  embed 0.0 ; block_0..11 2.1e-5..7.2e-4 ; final_ln 1.07e-3
  gate(b) greedy N=128 (A1: bug iff margin >= 3*fp16_ulp) : PASS  127/128, 1 tolerated near-tie, 0 bug
                                      (t=28: pred 1663 != ref 7866; ref margin 0.1250 < 3*fp16_ulp 0.3750 @ measured
                                       |top-1 logit| = 135, fp16 bin [128,256) -> tolerated near-tie. 135 is well above
                                       the |logit|=64 pivot below which 3*ulp 0.094 < the 0.125 margin would be a BUG.)
  gate(c) PRIMARY max KL < 0.02     : PASS  max KL 1.95e-3     [diagnostic: top-1 506/512 = 98.83%]

GPU baseline (naive; sync-enforced timer; median + min-max; prefill and decode SEPARATE):
  prefill  @P=128 : 197.8 ms median  (197.8 - 197.9, N=30)
  prefill  @P=512 : 840.0 ms median  (839.8 - 840.1, N=30)
  no-KV decode    : 149.0 ms/tok median (57.3 - 244.6, N=122) @ ctx 34->155  ->  6.7 tok/s
roofline check:   decode 6.7 tok/s << bug-line [copy 941 / read 1004 / theo 1032 tok/s] -> below ceiling, OK.
                  prefill @512 840 ms ~ 0.3% of the 31.5 TFLOP/s TENSOR peak (naive; no tensor cores) ->
                  large prefill headroom = exactly where stage-2 tiling pays off. Regime matches the director map.
                  [doc pass 2026-07-09: 31.5 TF is the TENSOR peak = bug line + WMMA headroom. The efficiency
                   denominator for these CUDA-core kernels is the MEASURED 10.10 TFLOP/s achievable CUDA-core
                   GEMM peak (ROOFLINE §6b) -> 1.1%. Neither reading changes the verdict (huge headroom).
                   See ROOFLINE §1 "Which denominator?" and §6b.]
baseline:         PyTorch eager / llama.cpp deferred (this is the from-scratch GPU baseline to improve against).
notes:            fp16 argmax gates recalibrated per Amendment A1 (KL primary; margin >= 3*fp16_ulp). Kernel sits AT the
                  fp16 noise floor: my fp16<->HF-fp16 top-1 (98.83% = 6/512 flips) is statistically indistinguishable
                  from HF's OWN fp16<->fp32 top-1 (98.44% = 8/512 flips) -- 6 vs 8 flips is within Poisson noise
                  (sigma ~2.6 on ~7 flips), NOT "better than". max KL 1.95e-3 is the actual evidence of faithfulness.
                  fp32 CPU engine unchanged. free-run greedy diverges at the t=28 tolerated near-tie (expected cascade).
```

<!-- gates feed pre-tokenized HF IDs (refdumps/meta.json) so tokenizer coverage never confounds kernel
     correctness (mirrors the "remove the confound" philosophy). From-scratch C tokenizer = later Phase-0
     completion item; it gates nothing and is EXCLUDED from all benchmarks (BENCH_PROTOCOL §2).
     next: Stage 2 cuda/kernels_tiled.cu (tiled/shared-mem GEMM; prefill win expected, decode ~flat per director map) -->

### Stage 2 — tiled / shared-memory GEMM (the prefill lever) · ✅ VALIDATED 2026-07-09

*Swaps EXACTLY ONE kernel — the matmul — for a classic 16×16 shared-memory tiled GEMM (CUDA cores;
fp16 storage + fp32 accumulation; **no tensor cores** — WMMA is a later lever). embed/LayerNorm/
attention/gelu/add are reused byte-for-byte from the naive backend, so every gate/timing delta
localizes to the GEMM. Correctness re-gated vs the HF fp16 oracle; then measured apples-to-apples vs
Stage-1 naive, back-to-back, same clocks. Director map: prefill win — CONFIRMED. There is **NO true
(M=1) decode result here** — this engine has no KV cache until Stage 5 (see notes + ROOFLINE §5).*

```
stage:            2 — tiled fp16 GEMM  (cuda/kernels_tiled.cu ; GPT2_BACKEND=tiled ; matmul-only swap)
kernel/precision: 16×16 SMEM-tiled GEMM (one TILE×TILE block/output tile; each A/W tile reused TILE=16x
                  via shared mem). fp16 storage + fp32 accumulation -> SAME numeric regime as naive/HF
                  fp16; k walked in naive order (partial sums bit-identical); bias added last (vs first
                  in naive) << fp16 gate tol. CUDA cores only (NO WMMA).
config:           gate(a) P=28 ; gate(b) greedy N=128 teacher-forced ; gate(c) eval P=512 ; greedy/argmax
environment:      RTX 4060 Laptop (AD107, sm_89), 24 SM, L2=32 MiB ; nvcc -O3 -arch=sm_89 (CUDA 12.6, drv 561.09)
                  clocks NOT locked -> sustained (nvidia-smi dmon during the TIMED region): SM 2610 MHz,
                  mem 8000 MHz ; naive run 41 W / 53-56 C, tiled run 46-51 W / 55-57 C -- NOT throttled (<<105 W)
included:         tokenization=n (pre-tokenized IDs)   weight_load=n (fp16 H2D upload excluded)

CORRECTNESS (QUALITY_GATES §1 + Amendment A1 ; vs HF fp16 oracle refdumps/fp16 ; TILED backend):
  gate(a) per-layer rel_err <= 1e-2 : PASS  embed 0.0 ; block_0..11 2.1e-5..7.2e-4 ; final_ln 1.07e-3
  gate(b) greedy N=128 (A1)         : PASS  128/128 match, 0 tolerated, 0 bug
                                      (tiled's bias-added-last nudges the Stage-1 t=28 near-tie back to a
                                       match; both sit AT the fp16 noise floor -- NOT "more correct")
  gate(c) PRIMARY max KL < 0.02     : PASS  max KL 1.95e-3   [diagnostic: top-1 506/512 = 98.83%]

MEASURED -- tiled vs Stage-1 naive, same session/clocks ; sync-enforced timer ; median (min-max) ;
            warmup 10, N=30 (prefill). Prefill and (recompute-)decode SEPARATE:
                               naive (S1)               tiled (S2)             speedup
  prefill @P=128         : 199.174 (199.07-199.23)   35.742 (35.64-35.78)      5.57x   [tiled 619.0 GFLOP/s]
  prefill @P=512         : 840.109 (839.9-840.3)    184.785 (184.6-184.9)      4.55x   [tiled 497.3 GFLOP/s]
  no-KV RECOMPUTE-decode : 149.188 ms/tok            27.694 ms/tok             5.39x   6.7 -> 36.1 tok/s
    (M>>1, prefill-shaped)   (57.5-244.8, N=122)       (13.8-44.3, N=122)               @ ctx 34->155
  free-run greedy vs fp16 ref: naive 28/128 (t=28 near-tie cascade) ; tiled 128/128 (FULL)

ROOFLINE CHECK:   prefill = the on-lever win (director map §5: Stage 2 = prefill). GFLOP/s here is a
                  CUDA-CORE GEMM measured vs the 31.5 TFLOP/s TENSOR-core ceiling -> the 1.6-2.0% is the
                  tensor-core headroom WMMA (a later lever) would tap, NOT this kernel's compute
                  efficiency. Whole-forward GFLOP/s FALLS @512 (497) vs @128 (619): attention is still
                  the naive O(T^2) kernel (Stage 2 didn't touch it) and grows with T -- flash = Stage 3.
                  recompute-decode 36.1 tok/s << bug-line [copy 941 / read 1004 / theo 1032] = 3.8% of
                  the 941 copy-BW ceiling -> far below, OK (recompute != KV decode -- see notes).

MECHANISM -- where the win comes from (isolated qkv matmul N=2304 K=768 ; M-sweep ; INTERLEAVED
             naive/tiled ; clocks warmed 2610 MHz ; median over 100):
    M         naive ms      tiled ms    tiled/naive
     1        0.0881        0.0748        1.18x   <- TRUE GEMV (KV-decode / logits-head shape): zero reuse
     4        0.1700        0.0758        2.24x      possible -> the 1.18x is the COALESCING residual alone
    16        0.4956        0.0768        6.45x      (naive strided-W vs tiled contiguous-W), overhead-bound.
    64        1.9763        0.2744        7.20x   <- tiled FLAT M=1->16 (one 16-row tile reads W once) then
   161        5.0043        0.7270        6.88x      ~ceil(M/16); naive proportional to M (re-reads all W
   512       15.7655        2.0808        7.58x      per row) -> TILING-FOR-REUSE, the lever; scales with M.
  (M=1 useful-W BW: naive 40.2 / tiled 47.3 GB/s -- 3.5 MB L2-resident, overhead-bound; not a DRAM figure)

baseline:         PyTorch eager / llama.cpp deferred (still improving the from-scratch GPU baseline).
notes:            The ~5.4x "decode" is the M>>1 GEMM REUSE win diluted by the unchanged attention/LN/
                  gelu + the single true M=1 op (the tied logits head). This engine has NO KV cache
                  until Stage 5, so the harness "decode" FULL-RECOMPUTES the growing sequence -> its
                  GEMMs run at M=ctx(34-161), the reuse regime. Hence it is prefill-shaped: a GEMM win
                  there is EXPECTED, NOT a director-map violation, and NOT a true (M=1) decode result.
                  ncu global-load sectors/request (to quantify the 1.18x M=1 coalescing residual;
                  expect naive >> tiled) is PENDING GPU perf-counter access (ERR_NVGPUCTRPERM) -- it
                  refines, does not change, the reuse conclusion; not a blocker (brain). Stage 2 =
                  ONE kernel swapped; correctness/quality preserved at the fp16 noise floor.
```

### Stage 3b — flash-style attention (online softmax, no score matrix) · ✅ VALIDATED 2026-07-09

*Swaps EXACTLY ONE kernel — attention — for a flash-style kernel with an online softmax. Target chosen
by MEASUREMENT, not assumption (DESIGN.md §9.1): a per-op attribution of the Stage-2 forward
(`bench/profile_forward.cu`) put attention at **44.80% of prefill@512**, the only op whose share GROWS
with T. Everything else (tiled GEMM, LN, gelu, add, embed) is reused byte-for-byte from Stage 2, so
every gate/timing delta localizes to attention. ncu remains blocked (ERR_NVGPUCTRPERM, needs an
elevated registry write + reboot); the WHERE was established without it, and the mechanism is pinned by
`ptxas -v` instead.*

```
stage:            3b — flash attention  (cuda/kernels_fused.cu ; GPT2_BACKEND=flash ; attention-only swap)
kernel/precision: online-softmax causal MHA. BR=8 queries/block (1 warp each), BC=32 keys/tile; K,V
                  staged in SMEM once per block and reused by all 8 warps; each lane owns 2 head dims.
                  Running (m,l,acc) with rescale alpha=exp(m-m_new). fp16 storage + fp32 accumulation ->
                  SAME numeric regime as naive/tiled/HF fp16. CUDA cores only (NO WMMA, NO tensor cores).
                  Scores never materialized: <= BR*BC = 256 live at once vs the naive T*T = 262,144 @512.
config:           gate(a) P=28 ; gate(b) greedy N=128 teacher-forced ; gate(c) eval P=512 ; greedy/argmax
                  (gate (b) T=155 and gate (c) T=512 both exercise the MULTI-TILE rescale path, j0=0,32,64,...)
environment:      RTX 4060 Laptop (AD107, sm_89), 24 SM, L2=32 MiB ; nvcc -O3 -arch=sm_89 (CUDA 12.6, drv 561.09)
                  clocks NOT locked -> sustained (nvidia-smi dmon during the TIMED region): SM 2610->2595 MHz,
                  mem 8000 MHz, 45-58 W, 53-59 C -- NOT throttled (<< 105 W cap)
included:         tokenization=n (pre-tokenized IDs)   weight_load=n (fp16 H2D upload excluded)

CORRECTNESS (QUALITY_GATES §1 + Amendment A1 ; vs HF fp16 oracle refdumps/fp16 ; FLASH backend):
  gate(a) per-layer rel_err <= 1e-2 : PASS  embed 0.0 ; block_0 3.285e-4 ; block_1 2.092e-4 ;
                                      block_2..10 2.107e-5..4.197e-5 ; block_11 7.194e-4 ; final_ln 1.073e-3
                                      (worst layer is 14x inside the threshold; the early-block bump vs tiled
                                       is the online-softmax summation order, not a rescale bug)
  gate(b) greedy N=128 (A1)         : PASS  127/128 match, 1 tolerated near-tie, 0 bug
                                      (t=28: pred 1663 != ref 7866 ; ref margin 0.1250 < 3*fp16_ulp 0.3750
                                       @ |top-1 logit| = 135 -> tolerated. This is the SAME near-tie Stage-1
                                       naive hit; Stage-2 tiled matched it only because bias-added-last
                                       happened to nudge it. All three sit AT the fp16 noise floor.)
  gate(c) PRIMARY max KL < 0.02     : PASS  max KL 1.951e-3   [diagnostic: top-1 506/512 = 98.83%]

MEASURED -- flash vs Stage-2 tiled, back-to-back same session/clocks ; sync-enforced timer ;
            median (min-max) ; warmup 10, N=30 (prefill). Prefill and (recompute-)decode SEPARATE:
                               tiled (S2)                flash (S3b)            speedup
  prefill @P=128         :  35.714 (35.573-35.796)    27.719 (27.640-27.779)     1.29x   [flash 798.2 GFLOP/s]
  prefill @P=512         : 184.781 (184.641-184.941) 107.341 (107.299-107.975)   1.72x   [flash 856.1 GFLOP/s]
  no-KV RECOMPUTE-decode :  27.640 ms/tok             21.393 ms/tok              1.29x   36.2 -> 46.7 tok/s
    (M>>1, prefill-shaped)   (13.87-44.29, N=122)       (11.71-34.30, N=122)              @ ctx 34->155
  free-run greedy vs fp16 ref: tiled 128/128 ; flash 28/128 (diverges at the t=28 tolerated near-tie,
    exactly as Stage-1 naive did). NOT a regression: gate (b) is the teacher-forced gate; free-run is a
    diagnostic that cascades from any single fp16-unresolvable flip.

KILL-TEST (DESIGN.md §5 ; bench/ab_forward.cu ; INTERLEAVED A/B so both backends see the same thermal
           state within microseconds ; 50 iters ; median [min-max]):
  whole forward @P=128   : tiled  35.6659 [35.6076-35.7990]  flash  27.6972 [27.6316-27.7412]  1.288x
  whole forward @P=512   : tiled 184.7173 [184.5811-184.8351] flash 107.2620 [107.1596-107.4340] 1.722x
  whole forward @ctx=161 : tiled  48.0865 [47.7829-48.1413]  flash  37.4518 [37.2070-37.5255]   1.284x
    -> [min,max] ranges DISJOINT at all three shapes: the win is above noise. KEEP.
  isolated attention     : T=128  0.7199 -> 0.0512 ms  = 14.06x
                           T=512  6.9627 -> 0.4808 ms  = 14.48x
                           T=161  0.9626 -> 0.0758 ms  = 12.70x   (flash spreads are wide in % because the
                           kernel is now only 50-480 us; the ranges are still disjoint from tiled's)

MECHANISM -- the win is ATTRIBUTED, not inferred (per-op re-profile + ptxas, no ncu needed):
  per-op share of prefill@512    attention 83.150 ms (44.80%)  ->  5.696 ms (5.27%)
  whole forward (uninstrumented)           184.750 ms          ->  107.334 ms
    delta_attention = 77.45 ms  ~=  delta_whole = 77.42 ms  -> the whole-forward win IS the attention win;
    nothing else moved. (sum-of-parts/whole = 1.005x tiled, 1.007x flash -> no unattributed time.)
  ptxas -v (sm_89):  k_attention        256 B stack frame, 40 regs   <- float acc[64] per thread, dynamically
                                                                        indexed -> LOCAL memory every accumulate
                     k_attention_flash    0 B stack frame, 40 regs, 19744 B smem
  two causes, both real: (1) the naive kernel makes TWO passes over K, recomputing every q.k dot (once for
  the running max, once for the softmax numerator); online softmax does ONE. (2) the acc[64] local-memory
  array is gone -- each thread now owns 2 head dims in 2 registers.

ROOFLINE CHECK:   prefill @512 = 856.1 GFLOP/s = 2.72% of the 31.5 TFLOP/s TENSOR-core ceiling -> below.
                  (As in Stage 2, the 2.7% is the tensor-core headroom WMMA would tap, not this CUDA-core
                  kernel's efficiency.) Isolated attention @512 does 2*E*T*(T+1) = 4.034e8 FLOP in 0.4808 ms
                  = 839 GFLOP/s. [CORRECTED 2026-07-09: this line originally divided by a CLOCK-DERIVED
                  16.0 TFLOP/s "CUDA-core peak" -> 5.2%. That denominator was assumed, never microbenched
                  (violating G2). MEASURED (ROOFLINE §6b): achievable CUDA-core GEMM = 10.10 TFLOP/s, FMA
                  pipe = 15.64. So: 839 GFLOP/s = 8.3% of the achievable CUDA-core GEMM ceiling (2.7% of
                  the 31.5 TF tensor bug line). The right denominator, since flash uses no tensor cores.]
                  recompute-decode 46.7 tok/s = 5.0% of the 941 tok/s copy-BW ceiling, << bug-line
                  [copy 941 / read 1004 / theo 1032]. NO number is above any ceiling.
                  The whole-forward GFLOP/s now RISES with T (798 @128 -> 856 @512) where Stage 2's FELL
                  (619 -> 497): the O(T^2) attention drag that Stage 2 flagged is removed.
regime check:     director map §5 row 3b = "traffic (no score matrix); helps at long ctx" -> CONFIRMED and
                  sharpened: the win grows with T (1.29x @128 -> 1.72x @512). The recompute-decode 1.29x is
                  again the M>>1 prefill-shaped regime (no KV cache until Stage 5), NOT a true M=1 decode.

baseline:         PyTorch eager / llama.cpp deferred (still improving the from-scratch GPU baseline).
notes:            ncu still blocked (ERR_NVGPUCTRPERM; needs elevated `reg add ... RmProfilingAdminOnly=0`
                  + reboot). It was NOT needed to choose or validate this kernel: the target came from
                  event-based per-op attribution (sum-of-parts/whole = 1.005x) and the mechanism from
                  ptxas -v. ncu would add sectors/request and dram__bytes -- refinement, not the verdict.
```

### Stage 3a — fused LayerNorm+matmul · ❌ KILL-TESTED → **REVERTED** (measured slowdown) · 2026-07-09

*A NEGATIVE RESULT, recorded as one. DESIGN.md §5 lists "fused RMSNorm/LN+matmul beats separate
kernels" as a `[CONJECTURE]` whose kill-test is a MEASUREMENT, not an argument. It was implemented,
gated, measured — and it made the forward **slower**, above noise, at every shape. Per §5 / §9.3 it is
reverted: the engine ships Stage 3b (flash) only. The code is preserved in git history for
reproducibility — `git show fb5ad32` recovers the kernel and its harness.*

```
stage:            3a — fused LN+matmul  (cuda/kernels_fused.cu ; GPT2_BACKEND=fused ; ln_matmul hook)
what was built:   a [M]-sized LN-stats prologue (k_ln_stats -> mean, rstd) + a tiled GEMM that applies
                  the normalization/scale/shift ON THE FLY while staging each A tile
                  (k_matmul_lnA_tiled). The [T,E] normalized-activation buffer is never written to
                  global memory. Fused only where LN feeds a matmul directly (ln_1->qkv, ln_2->fc);
                  ln_f stays separate because gate (a) captures it.
                  Backend = the VALIDATED flash backend + ONLY the ln_matmul hook -> the A/B delta
                  localizes to the fusion.
what was NOT built (and why): the textbook single-kernel fusion recomputes the LN reduction inside each
                  GEMM block. The tiled GEMM launches gridDim.x = N/16 column-blocks per row-strip (144
                  for qkv, 192 for fc), so each block would re-read its 16 A-rows twice -> ~226 MB of
                  redundant reads @M=512 against the ~1.6 MB the fusion is trying to save. A guaranteed
                  loss; not a fair test of the idea.
numerics held fixed ON PURPOSE: As[][] stores __half2float(__float2half(a)) -- the same fp16 rounding the
                  separate LN kernel applies to its output. Confirmed by gate (a) coming back BIT-IDENTICAL
                  to flash's. So the kill-test measures the fusion, not a precision change.

PRE-REGISTERED EXPECTATION (from Stage 3b's per-op attribution, written down BEFORE measuring):
  layernorm = 0.28% of prefill@512; traffic saved = one [T,E] fp16 write = 0.79 MB = ~3.4 us at 233 GB/s.
  Amdahl ceiling therefore < 1%. Expected: unresolvable, or a small loss. (It was a loss.)

CORRECTNESS (QUALITY_GATES §1 + A1 ; FUSED backend): gates still PASS -- the kernel is CORRECT, just slower.
  gate(a) PASS  embed 0.0 ; block_0 3.285e-4 ... block_11 7.194e-4 ; final_ln 1.073e-3  (== flash, bit-identical)
  gate(b) PASS  127/128, 1 tolerated near-tie (t=28), 0 bug
  gate(c) PASS  max KL 1.951e-3  [diag: top-1 506/512 = 98.83%]

KILL-TEST (bench/ab_forward.cu ; INTERLEAVED flash vs fused ; 50 iters ; median [min-max]):
  whole forward @P=128   : flash  27.7028 [27.6070-27.8374]   fused  28.5245 [28.4221-28.6177]   0.971x
  whole forward @P=512   : flash 107.3121 [107.2343-107.9611] fused 110.5587 [110.4814-111.2463] 0.971x
  whole forward @ctx=161 : flash  37.4584 [37.3881-37.5286]   fused  38.5648 [38.5147-38.6335]   0.971x
    -> [min,max] DISJOINT at all three shapes, and the SAME 0.971x three times: an above-noise SLOWDOWN
       of 2.9%, not a wash. VERDICT: REVERT.
  isolated LN+qkv pair   : T=128 0.5376 -> 0.5663 ms (0.949x) ; T=512 2.0981 -> 2.2139 (0.948x) ;
                           T=161 0.7330 -> 0.7721 (0.949x)  -> the fused pair itself is ~5.3% slower
  CONTROL (must not move): isolated attention 1.000x / 0.998x / 1.007x, [min,max] OVERLAPPING at all three
                           -> only the LN+matmul path changed. The 2.9% is attributable to the fusion.

WHY IT LOSES (the mechanism, consistent with the numbers):
  the tiled GEMM re-stages each A element once per column-tile -- N/16 = 144 times for qkv, 192 for fc.
  The fused kernel RE-NORMALIZES on every staging: extra mean/rstd/g/bn loads plus sub-mul-mul-add and an
  fp16 round-trip, inside the GEMM's innermost staging loop. The separate LayerNorm normalizes each
  element exactly ONCE. That redundant arithmetic (~5.3% of the pair) dwarfs the ~3.4 us of write traffic
  the fusion removes. It is the same redundancy trap as the rejected single-kernel variant, at lower
  amplitude. A fusion only pays when the fused producer's output is consumed ONCE.

roofline/director-map: consistent. §5 row 3a predicted "modest" for both regimes; the measured LN share
                  (0.28% of prefill) says the honest prefill prediction is "≈0, and negative if the fusion
                  adds per-consumption work". NOT a director-map violation. The regime where LN is a large
                  share is TRUE (M=1) decode, which does not exist in this engine until the KV cache
                  (Stage 5) -- so 3a is untested there and its conjecture stays OPEN for that regime,
                  explicitly NOT resolved by this prefill-shaped measurement.
verdict:          REVERTED. Engine = Stage 3b (flash) + Stage 2 (tiled GEMM). No speedup claimed, none taken.
```

### Stage 4 — weight-only INT8 quantized matmul · ✅ VALIDATED 2026-07-09 (both gates) · **zero speedup, and that is the result**

*The first stage with TWO gates. Correctness (§1+A1) AND quality (§2, Δppl ≤ +0.3, frozen before the
stage). The pre-registered scheme **FAILED both**; the **pre-registered kill-test** recovered both. No
threshold was moved. Speed: **no change above noise in ANY regime measurable today** — and the reason
is measured, not guessed. The engine ships Stage 2 (tiled GEMM) + Stage 3b (flash) + Stage 4 (INT8).*

```
stage:            4 — weight-only INT8  (tools/quantize.py, cuda/kernels_quant.cu ; GPT2_BACKEND=int8)
kernel/precision: symmetric, PER-CHANNEL (over output rows), WEIGHT-ONLY INT8; activations stay fp16.
                  s[n] = max_k|W[n,k]|/127 ; q[n,k] = clip(round(W/s),-127,127) ; W ~= q*s.
                  Per-channel over ROWS is what lets the scale LEAVE the k-sum:
                      C[m,n] = s[n] * (sum_k A[m,k]*q[n,k]) + b[n]
                  so the k-loop accumulates a scale-free fp32 dot and s[n] is applied ONCE per output.
                  Kernel = k_matmul_tiled byte-for-byte with Bs staged from int8 (one int->float cvt,
                  no multiply) -> every gate/timing delta localizes to the quantization.
                  Quantized from the fp16 VIEW of the fp32 master, so Δppl isolates quantization with
                  fp16 rounding held fixed.

  *** NO dp4a / NO INT8 tensor cores -- and that is not a fallback, it IS the pre-registered scheme. ***
  Those paths compute int8 x int8 -> int32. Weight-only INT8 keeps activations fp16, so the inner
  product is fp16 x int8 -> fp32 on CUDA cores. The correct MATH ceiling therefore stays the CUDA-core
  one, exactly as for the Stage 2/3 GEMMs -- NOT 31.5 TF (tensor). [MEASURED 2026-07-09, ROOFLINE §6b:
  achievable CUDA-core GEMM 10.10 TFLOP/s; FMA pipe 15.64 TFLOP/s. This block originally said "~16.0
  TFLOP/s", which was derived from clock, not microbenched.]
  INT8 buys TRAFFIC, not math (ROOFLINE §3, verbatim: "fewer bytes ... *not* faster math").

config:           gate(a) P=28 ; gate(b) greedy N=128 teacher-forced ; gate(c) eval P=512 ; greedy/argmax
                  gate §2: WikiText-2-raw val, 249,749 tok, window=1024 stride=512 (pre-registered)
environment:      RTX 4060 Laptop (AD107, sm_89), 24 SM, L2=32 MiB ; nvcc -O3 -arch=sm_89 (CUDA 12.6, drv 561.09)
                  clocks NOT locked -> sustained (nvidia-smi during the TIMED region): SM 2610 MHz,
                  mem 8000 MHz, 54-57 W, 59-61 C, 100% util -- NOT throttled (<< 105 W cap)
included:         tokenization=n (pre-tokenized IDs)   weight_load=n (fp16 + int8 H2D upload excluded)

---------------- WHAT WAS QUANTIZED ----------------
  49 tensors = the ~124M params ROOFLINE §2 counts as decode traffic:
    the TIED OUTPUT HEAD (wte) + per layer {c_attn.w, attn c_proj.w, mlp c_fc.w, mlp c_proj.w}
  NOT quantized (kept fp16; none is a matmul weight): all biases, ln_1/ln_2/ln_f, wpe, and
    wte-as-EMBEDDING (a row gather, not a matmul -- the fp16 wte stays resident for the embed lookup;
    only the HEAD reads the int8 copy). The fp16 blob is RESIDENT but not STREAMED by a quantized
    matmul; traffic is what is READ, not what is allocated.

============ GATE 1 of 2 — CORRECTNESS (QUALITY_GATES §1 + A1, vs HF fp16 oracle) ============

*** THE PRE-REGISTERED SCHEME (all 49 tensors int8) FAILS. Reported, not hidden. ***
  gate(a) PASS  embed 0.0 ; block_0 6.991e-3 ; block_11 7.554e-3 ; final_ln 8.584e-3   (worst 1.16x inside 1e-2)
  gate(b) FAIL  122/128, 4 tolerated near-tie, **2 BUG** (A1 margin rule, NOT tolerated away):
                  pos 12: margin 0.3125 >= 3*ulp 0.1875 @|logit|~123 -> BUG
                  pos 21: margin 0.4375 >= 3*ulp 0.1875 @|logit|~104 -> BUG
  gate(c) FAIL  max KL(ref||ours) = **2.573e-1** (threshold 0.02 -> 12.9x OVER)  [diag top-1 429/512 = 83.79%]

IS THE KERNEL WRONG, OR IS THE ERROR THE QUANTIZATION? (these demand opposite responses; separate first)
  tools/dequant_check.py writes an fp32 weight file with the 49 tensors REPLACED by their dequantized
  values q*s, then runs the *unmodified, Stage-3-validated fp16 flash GEMM* on it. Result:
      gate(a) block_0..block_10 rel_err match the INT8 backend to 4 significant figures
              (6.991e-3, 2.510e-3, 6.088e-4, 9.542e-4, 1.449e-3, ... 1.343e-3)
      gate(c) max KL 2.597e-1 (vs INT8's 2.573e-1) ; top-1 428/512 (vs 429/512) ; gate(b) 2 bug (vs 2)
  => k_matmul_int8_tiled computes exactly what "fp16 GEMM on dequantized weights" computes.
     THE DEQUANT PATH IS SANE. The failure is quantization error, not a kernel bug.
  (Two asymmetries, both explained: (i) embed rel_err 1.557e-3 here vs 0.0 for the INT8 backend --
   this file also dequantizes wte for the EMBED lookup, which the INT8 backend deliberately does not.
   (ii) the INT8 kernel is measurably MORE accurate than the fp16-dequant path -- final_ln 8.584e-3 vs
   1.019e-2 -- because it applies the exact fp32 scale AFTER fp32 accumulation instead of rounding q*s
   to fp16 first. The fp16-dequant blocks-only variant actually FAILS gate (a) at 1.019e-2 where the
   real INT8 kernel passes at 8.584e-3.)

LOCALIZATION (same weights-perturbation trick, per tensor group -- zero kernel changes):
    dequantized subset      gate(a) final_ln   gate(b)    max KL      top-1
    head only (tied wte)      5.633e-3         3 bug    **2.557e-1**  83.79%
    blocks only (48 matmuls)  1.019e-2         0 bug      1.283e-2 OK 96.88%
    mlp.c_proj only (12)      6.438e-3         0 bug      4.896e-3 OK 97.46%
    all 49 (= INT8 backend)   6.974e-3         2 bug      2.573e-1    83.59%
  => THE TIED OUTPUT HEAD CARRIES ESSENTIALLY ALL THE DAMAGE (head-only 0.2557 ~= all 0.2573; the 48
     block matmuls together only reach 0.0128, INSIDE the bound).
  MECHANISM (structural, not incidental): every block matmul feeds a LayerNorm downstream that rescales
  its perturbation away. `final_ln -> logits` is the ONE matmul whose output goes straight into a
  softmax. An absolute logit error of order 1, against top-2 margins frequently < 1, flips argmax and
  blows up KL. (Same reason llama.cpp keeps output.weight at higher precision than the body.)
  Corroborating: mlp.c_proj rows carry ~2x the per-row rms quant error (0.020 vs 0.009 -- K=3072 rows
  with heavier outliers) yet cost only KL 0.0049. Sensitivity is about POSITION, not per-row error.

*** PRE-REGISTERED KILL-TEST APPLIED (QUALITY_GATES §2 / DESIGN.md §5, in the prescribed order) ***
  "exceeds bound -> go per-channel (if not already), or keep sensitive layers ... in fp16."
  Already per-channel. So: KEEP THE TIED HEAD IN fp16 (a "high-activation-range layer" by measurement,
  not by assumption). NO THRESHOLD WAS MOVED. Implemented as a per-tensor flag in the packed file: a
  flag=0 tensor emits no int8 bytes and its GPT2QW keeps q=NULL, so gpt2_matmul_dispatch routes that
  ONE matmul back to the fp16 tiled GEMM. 48/49 tensors int8.

  GATE 1, KILL-TEST BUILD (weights/gpt2_124m_int8_kt.bin -- the SHIPPED default):
  gate(a) per-layer rel_err <= 1e-2 : PASS  embed 0.0 ; block_0 6.991e-3 ; block_1 2.510e-3 ;
                                     block_2..10 6.088e-4..1.449e-3 ; block_11 7.554e-3 ; final_ln 8.584e-3
                                     (worst 8.584e-3 = 1.16x inside the threshold; ~8x looser than
                                      flash's 1.073e-3 -- expected, INT8 error >> fp16 rounding)
  gate(b) greedy N=128 (A1)         : PASS  127/128, 1 tolerated near-tie, 0 bug -- IDENTICAL to flash
                                     (t=28: margin 0.1250 < 3*fp16_ulp 0.3750 @|logit|=135)
  gate(c) PRIMARY max KL < 0.02     : PASS  max KL 1.495e-2   [diag: top-1 498/512 = 97.27%]
                                     NOTE the margin is only 1.34x (flash had 10x): the 48 quantized
                                     block matmuls consume most of the KL budget. INT4 has no room here.
  free-run greedy vs fp16 ref: 28/128 -- identical to flash (same t=28 tolerated near-tie cascade).

============ GATE 2 of 2 — QUALITY (QUALITY_GATES §2 ; bound FROZEN before the stage: Δppl <= +0.3) ============

HARNESS VALIDATED FIRST (a Δ from an unvalidated harness means nothing):
  bench/eval_ppl_cuda.cu replicates tools/eval_ppl.py's loop EXACTLY (window=1024, stride=512, HF's
  shift, HF's -100 masking, and HF's own quirk that window 0 weights 1023 scored rows by trg=1024).
  tools/dump_wikitext_ids.py re-emits the same token ids (249,749 -- matches eval_ppl.py's count).
  Cross-check on the first 20,000 tokens:  ours 22.0100  vs  HF eval_ppl.py 22.0109   (mean_nll both
  3.0915). Δ = 0.0009 = 0.004%. The replication is faithful.

  ppl_fp16 (OUR flash engine, full 249,749 tok) = 25.5681
  ppl_fp16 (HF eval_ppl.py, the FROZEN baseline) = 25.57
  => QUALITY_GATES §2's "your own fp16 kernel's PPL" and the frozen literal AGREE to 0.002. No ambiguity
     about which baseline to subtract; both deltas below are reported anyway.

    build                                    PPL       Δ vs frozen 25.57   Δ vs own fp16 25.5681   bound
    our fp16 (flash)                        25.5681          --                   --                --
    PURE INT8 (pre-registered, all 49)      26.1187      **+0.549**            +0.551          ** FAIL ** (1.8x over)
    KILL-TEST INT8 (head fp16, 48/49)       25.5969        +0.027               +0.029           PASS (11x margin)

  The pure scheme fails §2 by 1.8x -- the SAME cause as its gate-(c) failure, the tied head. The
  pre-registered kill-test recovers it with an 11x margin. Per §2: full weight-only INT8 including the
  tied head is **NOT a valid quality point for this model**; the honest degradation is recorded here
  rather than hidden, and the bound was NOT re-tuned after the fact.

============ SPEED (only after BOTH gates pass) ============

KILL-TEST / A-B (bench/ab_forward.cu ; INTERLEAVED flash vs int8 so both see the same thermal state
                 within microseconds ; 50 iters ; median [min-max] ; sync-enforced timer):

  KILL-TEST BUILD (shipped, 162.1 MB streamed):
    whole forward @P=128   : flash  27.7878 [27.6859-27.8610]  int8  27.7862 [27.7115-27.8313]  1.000x  OVERLAP
    whole forward @P=512   : flash 107.9178 [107.8385-107.9675] int8 107.9362 [107.8292-107.9931] 1.000x OVERLAP
    whole forward @ctx=161 : flash  37.4574 [37.3729-37.5040]  int8  37.3745 [37.2859-37.4979]  1.002x  OVERLAP
  PURE BUILD (123.5 MB streamed -- the maximal traffic win, ungated):
    whole forward @P=128 1.000x ; @P=512 1.000x ; @ctx=161 1.003x  -- ALL [min,max] OVERLAP

  => NO speedup above noise, in any regime this engine can measure today. Reported as measured.

  correctness_cuda timing (N=30, warmup 10, median (min-max)); prefill and recompute-decode SEPARATE:
                             flash (S3b)            int8 kill-test (S4)     int8 pure
    prefill @P=128    :  27.697 (27.60-27.78)   27.749 (27.70-27.80)    27.618 (27.56-27.66)
    prefill @P=512    : 107.290 (107.20-107.37) 107.360 (107.30-107.42) 107.140 (107.09-107.21)
    no-KV RECOMPUTE-decode  21.259 ms/tok        21.270 ms/tok           21.141 ms/tok
      (M>>1, prefill-shaped)  -> 47.0 tok/s        -> 47.0 tok/s           -> 47.3 tok/s   @ ctx 34->155

  ISOLATED WEIGHT MATMUL (where an INT8 win MUST come from if it exists; 50 iters interleaved):
    qkv   M=128 1.001x | M=512 1.000x | M=161 1.001x | M=16 1.000x | M=1 1.014x   -- all OVERLAP
    head  M=128 1.001x | M=512 1.001x | M=161 1.001x | M=16 1.001x | M=1 1.002x   -- all OVERLAP
  CONTROL (must not move): isolated attention 1.000x / 0.997x / 1.000x, ranges OVERLAP. Only the matmul
                           path was touched, and it did not move either.

WHY THERE IS NO WIN -- MEASURED, NOT ASSUMED (this is the load-bearing finding of Stage 4):

  The M=1 tied-head GEMV is the ONE true-decode-shaped op that exists pre-Stage-5, and it streams
  77.2 MB fp16 -> 38.6 MB int8. It still shows 1.002x. Two falsifiable predictions distinguish
  "bandwidth-bound" from "compute-bound on wasted work". Both were tested:

    (1) If BW-bound at M=1, INT8 (half the bytes) gives ~2x.        MEASURED: 1.002x.   REJECTED.
    (2) If compute-bound on the 16x16 tile's WASTED rows, then M=1 and M=16 cost the SAME (the tile
        computes 16 output rows either way; at M=1 fifteen are discarded at the write).
                                                                     MEASURED: head M=1 1.4351 ms
                                                                               head M=16 1.4346 ms
                                                                     -> identical (0.03%). CONFIRMED.
    Corroboration, head M=1 (fp16): executes 2*16*V*E = 1.235e9 FLOP in 1.4351 ms = **860.6 GFLOP/s**,
      which is **98.8%** of the SAME kernel's fully-utilised M=512 throughput (871.5 GFLOP/s).
      Useful throughput = 860.6/16 = 53.8 GFLOP/s. Weight BW = 53.8 GB/s = **23% of the 233 GB/s
      achieved copy BW** -> nowhere near BW-bound.
    INT8 head M=1: 862.1 GFLOP/s (same compute), 26.9 GB/s (half the bytes), same time.
    => SAME arithmetic, HALF the bytes, SAME time. The 16x16 tiled GEMM at M=1 is saturated on
       arithmetic it throws away, so halving weight bytes cannot show up. This is a KERNEL limitation,
       not a roofline one, and Stage 5 must pair the KV cache with an M=1-shaped GEMV kernel for the
       INT8 traffic win to become expressible AT ALL.

  In the M>>1 regimes (prefill, and the no-KV recompute-decode, which is prefill-shaped) the tiled GEMM
  re-stages each W tile ceil(M/16) times into SMEM and the weights are largely L2-resident (qkv W is
  3.5 MB << 32 MiB L2), so weight traffic was never the constraint there either. INT8 halves a cost the
  measured bottleneck does not pay.

ATTRIBUTION (bench/profile_forward.cu, int8 kill-test, prefill@512, N=30):
  sum-of-parts / whole = 108.7215/107.9393 = 1.007x -> split trustworthy, no unattributed time.
  Every per-op median is within ~0.5% of flash's: ffnproj 33.50 (flash 33.33) ; fc 33.32 (33.12) ;
  qkv 25.05 (24.91) ; attention 5.750 (5.696) ; logits(M=1) 1.439 (1.429) ; layernorm 0.507 (0.523).
  Nothing moved -- consistent with the whole-forward 1.000x.
  (Proof the INT8 path really ran, independent of timing: gate (c) KL and PPL both changed markedly
   -- 1.951e-3 -> 1.495e-2 and 25.5681 -> 25.5969. Only the quantized kernel can produce those logits.)

ROOFLINE CHECK:   prefill @512 = 855.9 GFLOP/s. Correct denominator for a CUDA-core, weight-only-INT8
                  GEMM = the MEASURED achievable CUDA-core GEMM peak **10.10 TFLOP/s** (ROOFLINE §6b)
                  -> **8.47%**. (Against the 31.5 TF TENSOR peak it is 2.72%, but no tensor core is used
                  -- and dp4a is unavailable to weight-only INT8, so 31.5 TF is only a BUG-LINE here, not
                  an efficiency yardstick. The CUDA-core FMA pipe, 15.64 TF, is the CUDA-core bug line.)
                  [CORRECTED 2026-07-09: originally "5.35% of 16.0 TFLOP/s", a clock-DERIVED peak that
                   G2 required to be microbenched. Now measured; 16.0 was optimistic by 1.55x.]
                  DECODE ceilings are derived from the bytes each build ACTUALLY streams, never hardcoded:
                    fp16          248.9 MB -> copy  938 / read 1000 / theo 1029 tok/s
                    INT8 pure     123.5 MB -> copy 1889 / read 2015 / theo 2072 tok/s
                    INT8 kill-test 162.1 MB -> copy 1440 / read 1535 / theo 1579 tok/s  [the shipped build]
                  measured no-KV recompute-decode 47.0 tok/s = **3.3%** of its 1440 copy-BW ceiling.
                  NO number is above any ceiling, in any build. No STOP condition triggered.

regime check:     director map §5 row 4 = "INT8 -> decode (fewer bytes) -> **high**, the main decode lever".
                  MEASURED: no decode payoff. This is **NOT** a director-map contradiction, for exactly
                  the reason Stage 2's "decode ~flat" was not one (ROOFLINE §5 annotation):
                  **true (M=1) decode does not exist in this engine until the KV cache (Stage 5).** Every
                  "decode" number here is the M>>1 full-recompute harness, which is prefill-shaped and
                  reuse-bound, not weight-traffic-bound. Row 4's payoff is therefore **UNTESTED, not
                  refuted** -- and Stage 4 sharpens the prediction: the payoff will not appear in Stage 5
                  either unless the M=1 kernel stops discarding 15/16 of its arithmetic. **ROOFLINE §5 row 4 therefore warrants the same annotation row 2 received.**

baseline:         PyTorch eager / llama.cpp deferred. NOTE BENCH_PROTOCOL §5: an INT8 engine compares to
                  llama.cpp **Q8_0**, never to PyTorch fp16 -- comparing across precisions is a category
                  error, not a result.
verdict:          KEPT. Stage 4 is a LADDER RUNG (BUILD_PLAN), not a discretionary fusion like 3a: it
                  passes both gates, it halves the block weights' bytes, and it is the artifact Stage 5
                  needs. It is NOT reverted-as-a-non-win because it is not a slowdown (1.000x, ranges
                  overlap) -- but **no speedup is claimed, because none was measured.** The pure
                  pre-registered scheme is recorded as an honest quality failure; the shipped default
                  (weights/gpt2_124m_int8_kt.bin) is the gated kill-test build.
```

### Stage 5 — KV cache + memory planner + the TRUE M=1 GEMV (the LAST rung) · ✅ VALIDATED 2026-07-09

*The stage where a **true (M=1) decode first EXISTS** in this engine. Before it, "decode" full-recomputed
the growing sequence (M = ctx, prefill-shaped). Two director-map rows were therefore untestable and are
**resolved here**: row 2 ("tiled GEMM → decode ~flat/low") and row 4 ("INT8 → decode high").*

*Stage 4's finding forced the design: a KV cache **alone** cannot surface the INT8 win, because the 16×16
tiled GEMM at M=1 computes 16 output rows and discards 15 (measured: head M=1 == head M=16 in time). So
Stage 5 ships **three** new kernels — KV append/scatter, an M=1 attention over the cache, and a **true
M=1 GEMV** (one warp per output row, W read exactly once, nothing discarded).*

```
stage:            5 — KV cache + planner + M=1 GEMV  (cuda/kvcache.{cu,cuh} ; GPT2_BACKEND=gemv|int8)
kernels added:    k_kv_append / k_kv_scatter  (decode append; prefill fill, via caps->kv in the ONE forward loop)
                  k_attn_decode               (M=1 causal attention over the cache; scores never leave SMEM)
                  k_gemv_fp16 / k_gemv_int8   (TRUE M=1 GEMV: 1 warp per output row, half2/char4 loads,
                                               W read exactly once, coalesced, ZERO discarded rows)
memory planner:   ONE contiguous arena, 12 layers x 2 (K,V) x 12 heads x 1024 pos x 64 dim x fp16 = 37.7 MB.
                  Sized for full context up front -> no realloc can move the cache under a running kernel.
                  Layout [layer][K|V][head][pos][dim]: decode attention reads the CONTIGUOUS slab
                  K[h][0..len) -- with a [pos][E] layout that read strides by E=768 halves per position.
                  Trade the rare strided write (1 pos, all heads) for the hot sequential read.
                  VRAM after cache: 7068 MB free of 8585. KV traffic = 36.9 KB per token of context.
config:           gate(a) 28 DECODE steps from pos 0 ; gate(b) greedy N=128 teacher-forced ; gate(c) 512
                  sequential decode steps ; [EQ] cached vs recompute ; greedy/argmax
environment:      RTX 4060 Laptop (sm_89) ; nvcc -O3 -arch=sm_89 (CUDA 12.6, drv 561.09)
                  clocks NOT locked -> sustained SM 2610 MHz, mem 8000 MHz, 55 W, 59-61 C -- NOT throttled
included:         tokenization=n   weight_load=n   (KV-cache alloc excluded; it is one-time setup)

CORRECTNESS (bench/kv_gate.cu ; a KV cache is a prime silent-bug site, so FOUR checks, localizer first):

  [EQ] cached decode == no-KV recompute, position by position. NOT bit-identical BY CONSTRUCTION: in the
       recompute path a position's K/V come from a GEMM at M=ctx; in the cached path they were produced at
       M=1 (GEMV) when that position was decoded. Different reduction order -> different fp16 rounding.
         fp16 : max |logit_kv - logit_recompute| = 0.1875 (= 1.5 fp16 ulp @|logit|=128) ; max KL 4.883e-4 ; argmax 28/28
         INT8 : max |dlogit| = 0.2500 ; max KL 4.687e-4 ; argmax 28/28                       -> PASS both

  fp16 KV decode (GPT2_BACKEND=gemv), vs HF fp16 oracle, A1:
    gate(a) 28 decode steps from pos 0 (cache built ENTIRELY by the M=1 path -- the strictest test):
            embed 0.0 ; block_0 3.285e-4 ; block_11 7.194e-4 ; final_ln 1.073e-3   -> PASS
            (these REPRODUCE the flash *prefill* per-layer numbers -> the cache + GEMV introduce no drift)
    gate(b) 127/128, 1 tolerated near-tie (t=28, margin 0.1250 < 3*ulp 0.3750 @|logit|=135), 0 bug -> PASS
    gate(c) 512 SEQUENTIAL decode steps: max KL 1.953e-3 (< 0.02) [diag top-1 505/512 = 98.63%]  -> PASS

  INT8 KV decode (GPT2_BACKEND=int8 ; blocks INT8 GEMV + tied head fp16 GEMV, per Stage 4's kill-test):
    gate(a) block_0 6.991e-3 ; block_11 7.194e-3 ; final_ln 9.388e-3   -> PASS (worst 1.06x inside 1e-2)
    gate(b) 127/128, 1 tolerated, 0 bug                                -> PASS
    gate(c) max KL 1.490e-2 (< 0.02) [diag top-1 497/512 = 97.07%]     -> PASS
            (== Stage 4's prefill KL 1.495e-2: the decode path adds no quantization error)

SPEED (bench/bench_decode.cu ; INTERLEAVED A/B ; sync-enforced ; median [min-max] ; iters=50 ;
       samples BATCHED over 4-16 steps then divided -- a 2 ms / 134-launch step is sensitive to host
       jitter; batching amortises it and does NOT change what is measured (BENCH_PROTOCOL §6: widen N,
       never pick the good one). Both sides answer "what does ONE MORE token at ctx cost?":
         no-KV = a full forward over ctx+1 tokens (literally what the pre-Stage-5 harness did per token)
         KV    = one M=1 decode step against a cache of length ctx
       Timing the KV step against a *prefill* instead would be the "count cached tokens as generated"
       cheat (BENCH_PROTOCOL §7). It is not done here.)

  (1) *** THE KV-CACHE WIN (fp16) ***                    no-KV            KV            speedup
      decode @ ctx=128    :  30.6196 ms (32.7 tok/s)  ->  2.0854 ms (479.5 tok/s)     14.68x
      decode @ ctx=512    : 110.2321 ms ( 9.1 tok/s)  ->  1.9692 ms (507.8 tok/s)     55.98x
      decode @ ctx=1023   : 221.3612 ms ( 4.5 tok/s)  ->  2.2513 ms (444.2 tok/s)     98.33x
      [min,max] disjoint at all three. The win GROWS with ctx exactly as O(T) recompute vs O(1) append
      predicts. This is the essential-for-decode result the director map called for.

  (2) *** DIRECTOR-MAP ROW 2 RESOLVED *** naive vs tiled MATMUL at TRUE M=1 (both KV; .gemv==NULL)
      ctx=128 : naive 8.0494 ms (124.2 tok/s) -> tiled 5.7649 ms (173.5 tok/s)  1.396x  disjoint
      ctx=512 : naive 8.3009 ms (120.5 tok/s) -> tiled 6.0198 ms (166.1 tok/s)  1.379x  disjoint
      VERDICT: row 2's "decode payoff LOW" is **CONFIRMED**. 1.38-1.40x at true M=1 against **5.57x at
      prefill@128** -- the tiling-for-REUSE win is absent, as the roofline said it must be (there is only
      one row to reuse). The residual 1.4x is COALESCING (tiled reads W contiguously, naive strided),
      matching Stage 2's isolated M=1 prediction of 1.18x, amplified here by the tied head's 50257 rows.
      Not literally "flat" -- but the prediction was "low", and low is what it is.

  (2b) the GEMV kernel itself: tiled M=1 matmul vs TRUE M=1 GEMV (both KV, both fp16)
      ctx=128 : tiled 5.7601 ms (173.6 tok/s) -> gemv 2.0865 ms (479.3 tok/s)   2.761x  disjoint
      ctx=512 : tiled 6.0143 ms (166.3 tok/s) -> gemv 2.0102 ms (497.5 tok/s)   2.992x  disjoint
      The 16x16 tile at M=1 discards 15/16 of its rows; the GEMV discards none. ~2.9x, as predicted.

  (3) *** DIRECTOR-MAP ROW 4 RESOLVED *** fp16 GEMV vs INT8 GEMV at TRUE M=1 (both KV)
      ctx=128 : gemv 1.6430 ms (608.6 tok/s) -> int8 1.6140 ms (619.6 tok/s)  1.018x  ** OVERLAP **
      ctx=512 : gemv 1.9391 ms (515.7 tok/s) -> int8 1.7336 ms (576.8 tok/s)  1.119x  ** OVERLAP **
      ctx=1023: gemv 2.1639 ms (462.1 tok/s) -> int8 1.8612 ms (537.3 tok/s)  1.163x  disjoint (above noise)
      VERDICT: row 4's "**high** -- the main decode lever" is **REFUTED for this model and this build.**
      INT8 at true M=1 buys 1.02-1.16x, above noise only at ctx=1023. The cause is MEASURED, not guessed:

MEASURE-BEFORE-OPTIMIZE (§9.1 ; bench/profile_decode.cu):
  [A] per-op attribution of one decode step: SUM/whole = 1.51x -> the harness's own validity guard says
      **the per-op shares are DISTORTED** (134 tiny kernels; each per-op sync adds ~7.7 us). They are NOT
      used. The guard firing is the point: an un-guarded per-op split here would have been wrong.
  [B] isolated M=1 GEMVs, achieved weight bandwidth:
        head  N=50257 K=768 : 77.2 MB in 0.3174 ms = **243.2 GB/s** = 104% of copy BW, 98% of READ BW.
              -> the M=1 GEMV kernel is genuinely DRAM-bound and essentially at peak. It is the fast path.
        per-layer GEMVs read 345-438 "GB/s" -- ABOVE DRAM BW, i.e. **L2-INFLATED**: timing the same
        1.2-4.7 MB weight 200x keeps it in the 32 MiB L2. Reported, and NOT used. attnproj takes 8.2 us
        whether fp16 or INT8 -> a per-kernel LATENCY FLOOR, not a byte cost.
  [C] the 48 block GEMVs back-to-back = one step's worth (169.9 / 84.9 MB, both >> 32 MiB L2 -> no inflation):
        fp16 0.9764 ms = 174.0 GB/s      INT8 0.6948 ms = 122.2 GB/s      **1.405x** (not 2.00x)
      the shortfall from 2x is the latency floor across 48 launches of 1.2-4.7 MB each.
  WHY ROW 4 CANNOT REACH "HIGH" HERE -- the decode step decomposes (measured, ctx=512):
        fp16 step 1.939 ms = blocks 0.976 + head 0.317 + everything-else 0.645
        INT8 step 1.734 ms = blocks 0.695 + head 0.317 + everything-else 0.721
      1. The **tied head is 77.2 MB = 31% of the fp16 weight bytes and is kept fp16 in BOTH builds** --
         Stage 4's pre-registered quality kill-test forbids quantizing it (it alone caused KL 0.257 and
         Dppl +0.549). It contributes 0.317 ms to every step, identically. Unhalvable, by the QUALITY gate.
      2. The 48 quantizable block matmuls are 1.2-4.7 MB each -> at M=1 they sit near a per-kernel latency
         floor, so halving their bytes gives 1.405x, not 2x.
      => **weight-side ceiling on the whole-step speedup = 1.278x**, BEFORE attention/LN/add/gelu and the
         134-launch overhead dilute it. Measured 1.02-1.16x is consistent with that ceiling.
      => If the head could also be INT8 (it cannot, at this quality bar) the weight-side ceiling would be
         1.516x. Even then, "high" would be a stretch at batch=1 on a 124M model.
  KV reads are NOT dominating: 18.9 MB/step at ctx=512 = 7% of the 247.1 MB of weights (37.7 MB = 13% at
  ctx=1023). Attention grows with ctx, which is why the INT8 ratio improves with ctx (more of the step is
  non-weight work at low ctx... and at high ctx the KV traffic evicts weights from L2, exposing more DRAM).

ROOFLINE CHECK (ceilings derived from the bytes each build ACTUALLY streams, weights + KV, copy BW 233.4):
  backend ctx    ms/tok   tok/s   ceiling   % of ceiling
  gemv    128    1.7584   568.7      920      61.8%      (weights 248.9 + KV  4.7 MB)
  int8    128    1.7803   561.7     1399      40.2%      (weights 162.1 + KV  4.7 MB)
  gemv    512    1.9278   518.7      872      59.5%
  int8    512    1.7427   573.8     1290      44.5%
  gemv   1023    2.1940   455.8      814      56.0%
  int8   1023    2.0043   498.9     1168      42.7%
  NO number above any ceiling, in any build. fp16 decode peaks at 608.6 tok/s (ctx=128) << the 938 tok/s
  weights-only copy-BW ceiling and << the 1029 theoretical bug-line. fp16 sits at 56-62% of its ceiling --
  inside ROOFLINE §7's "0.55-0.85x, reproducible -> believable" band. INT8 sits at only 40-45% of its
  (higher) ceiling, which is precisely the statement that it is NOT weight-traffic-bound: the fp16 head
  plus the small-tensor latency floor keep it off its roofline.

regime check:     row 5 ("KV cache -> decode HIGH, essential") **CONFIRMED**, 14.7x-98.3x, growing with ctx.
                  row 2 ("tiled -> decode low")                **CONFIRMED** at true M=1: 1.38-1.40x vs
                                                               5.57x at prefill; the residual is coalescing.
                  row 4 ("INT8 -> decode high")                **REFUTED for this model/build**: 1.02-1.16x,
                                                               bounded at 1.278x by the fp16 tied head that
                                                               the quality gate requires. Reported as measured.

baseline:         PyTorch eager / llama.cpp deferred. BENCH_PROTOCOL §5: an INT8 engine compares to
                  llama.cpp Q8_0, never to PyTorch fp16.
notes:            The GEMV is the fast path and is at 98% of read BW on the one tensor large enough to be
                  DRAM-bound. The remaining decode overhead is 134 kernel launches per step plus tiny,
                  under-parallel kernels (a M=1 LayerNorm is ONE block on a 24-SM GPU). Fusing them, or
                  CUDA-graphing the step, is the identified next lever -- it is NOT in BUILD_PLAN's Stage-5
                  manifest and was NOT attempted, per "don't add anything on DESIGN.md §2's IS-NOT list".
verdict:          KEPT, both gates and the equivalence check pass. The ladder is complete.
```

---

## External baseline — llama.cpp (CUDA) and PyTorch eager · ✅ MEASURED 2026-07-09

*NO kernel changed. The engine is FROZEN at Stage 5 (`81bd251`); no correctness re-gate was needed or run.
This section is pure external anchoring: apples-to-apples at matched precision, plus an attribution of the
gap to the specific levers this engine deliberately did not build.*

```
baselines:        llama.cpp b1-259f2e2, CUDA backend, built from source (Ninja/MSVC, -DGGML_CUDA=ON,
                  -DCMAKE_CUDA_ARCHITECTURES=89).  PyTorch 2.12.1+cu126 eager.
model:            IDENTICAL weights. gpt2-f16.gguf converted from HF openai-community/gpt2 ;
                  llama-bench reports "gpt2 0.1B F16, 239.08 MiB, 124.44 M params" -- our exact param count.
environment:      RTX 4060 Laptop (sm_89), driver 561.09, CUDA 12.6.  All three engines measured in the
                  SAME session. Ours: SM 2595-2610 MHz, mem 8000 MHz, 60 C, not throttled.
included:         tokenization=n  weight_load=n   (all three)

*** [VERIFY] RESOLVED: BENCH_PROTOCOL §5 asked whether mainline llama.cpp still runs GPT-2. It does --
    but the converter has a REAL BUG, and it had to be fixed locally before any number existed: ***
    conversion/gpt2.py:27 intends to DROP the legacy causal-mask buffers ("we don't need these") but
    falls through to super().modify_tensors(), which calls map_tensor_name() on the very tensor it means
    to drop -> ValueError: Can not map tensor 'h.0.attn.bias'. One-line local patch: honour the comment
    and return without yielding. Recorded, not silent. Upstream-reportable.
    SANITY (before trusting any timing): llama-completion, greedy ->
      "The capital of France is Paris. The capital of Germany is Berlin. The capital of the United
       States is Washington. The capital of the United Kingdom is London."  -> the GGUF is correct.
    (An earlier garbage output through llama-cli was a FRONT-END artifact -- that binary is now a chat
     UI and injected a ChatML template, which GPT-2 tokenised literally. Not a model defect.)

METHOD:           llama-bench reports mean+-stddev; BENCH_PROTOCOL §6 mandates MEDIAN. So -o json was
                  used and the MEDIAN was recomputed from the raw per-repetition samples_ns (r=25).
                  Decode is measured at a STATED context via -d/--n-depth with a SHORT generation
                  (-n 16), so the context stays pinned near the stated ctx instead of drifting over 128
                  tokens. PyTorch: torch.cuda.synchronize() around every timed region, TF32 DISABLED,
                  eager (NOT torch.compile), median of 100 after 20 warmup.
                  DECODE is true M=1 for all three (all have a KV cache) -> apples-to-apples.
                  PREFILL for PyTorch is measured on the TRANSFORMER TRUNK ONLY: HF's model(ids) runs the
                  LM head over ALL P positions (+39.5 GFLOP at P=512), which our engine does not.

=================== PRIMARY: fp16 vs F16 (MATCHED PRECISION) ===================

PREFILL (single forward; logits for the last token only; FLOPs = 22.12 G @128, 91.89 G @512)
                        ours (fp16)           llama.cpp (F16)        PyTorch eager (fp16, trunk)
  P=128            27.548 ms (0.803 TF)    2.732 ms ( 8.098 TF)     9.441 ms
  P=512           107.140 ms (0.858 TF)    7.422 ms (12.381 TF)     9.395 ms
  -> llama.cpp is 10.08x (P=128) / 14.44x (P=512) FASTER than us.  PyTorch 2.92x / 11.40x faster.

DECODE (TRUE M=1, one token against a KV cache of the stated ctx; per-token median)
                        ours (fp16)            llama.cpp (F16)        PyTorch eager (fp16)
  ctx=128         1.9544 ms (511.7 t/s)   1.5713 ms (636.4 t/s)   9.3116 ms (107.4 t/s)
  ctx=512         1.9849 ms (503.8 t/s)   1.6557 ms (604.0 t/s)   9.2657 ms (107.9 t/s)
  ctx=1023*       2.2292 ms (448.6 t/s)   1.7055 ms (586.3 t/s)   9.3943 ms (106.4 t/s)
    (* llama.cpp measured at d=1007: d=1023 plus 16 generated would exceed GPT-2's 1024 context.)
  -> WE ARE AT 80.4% / 83.4% / 76.5% OF llama.cpp.   We are 4.2-4.8x FASTER than PyTorch eager.

BUG-LINE (ROOFLINE §6): fp16 weights 248.9 MB -> copy-BW ceiling ~938 tok/s, theoretical 1029.
  llama.cpp F16 peaks at 636.4 tok/s -- BELOW the ceiling. Nothing to dispute. (Q8_0 at 925.3 tok/s is
  also below ITS ceiling: 134.9 MB streamed -> ~1730 tok/s copy-BW.) No baseline number is above a ceiling.

=================== GAP ATTRIBUTION (the actual deliverable) ===================

PREFILL @512 -- the 14.44x decomposes EXACTLY into two MEASURED factors:
    (a) CEILING (tensor cores / WMMA)      3.12x   = 31.51 TF (measured tensor GEMM, ROOFLINE §6)
                                                   / 10.10 TF (measured achievable CUDA-core GEMM, §6b)
        This engine uses NO tensor cores, by design. It is capped at 10.10 TF; llama.cpp is not.
    (b) EFFICIENCY WITHIN THE CEILING      4.63x   = llama.cpp reaches 39.29% of ITS ceiling;
                                                     we reach 8.49% of OURS.
        A textbook 16x16 SMEM tile vs a register-tiled, double-buffered, vectorised GEMM.
    check: 3.12 x 4.63 = 14.44x, the measured ratio.
    HONEST READING OF THIS PARTITION: only (a) is an INDEPENDENT measurement -- it is the ratio of two
    microbenchmarked peaks (§6, §6b). (b) is the RESIDUAL, gap/(a), which happens to equal the ratio of
    each engine's fraction-of-its-own-ceiling. So (a)x(b)=gap is an identity, not a prediction that
    checks out. What the partition buys is the split itself: how much of the gap is a ceiling we cannot
    reach vs. how much is kernel quality inside the ceiling we can.

  *** FUSION IS NOT THE PREFILL STORY. *** Attention is only 5.3% of our prefill after Stage 3b, and the
  ~123 kernel launches in a prefill forward cost well under 1% of a 107 ms forward -- bounded above by the
  per-op profile's sum-of-parts landing within 0.7% of the uninstrumented forward (Stage 3b). (A per-launch
  cost is NOT separately measured here; no hard percentage is claimed.) The prefill gap is WMMA plus raw
  GEMM microarchitecture -- nothing else is large enough to matter.

DECODE -- 1.24x at ctx=128, rising to 1.31x at ctx=1023. At short context the gap is almost ENTIRELY the
fixed per-step overhead (attention is negligible there). The 73/27 split below is the ctx=1023 decomposition:
    fixed per-step overhead   0.3831 ms  (= the WHOLE gap at ctx=128, where attention is negligible)
                                          our step is 134 kernel launches with several 1-block kernels
                                          (an M=1 LayerNorm occupies 1 of 24 SMs); llama.cpp captures the
                                          step in a CUDA graph and fuses.
    attention ctx-scaling     0.1406 ms  (ours +0.2748 ms from ctx 128->1023; llama.cpp only +0.1342 ms)
                                          -> our 3-pass k_attn_decode vs their flash-decode kernel.
    share of the ctx=1023 gap:  73% fixed overhead  |  27% attention scaling.
  Achieved weight bandwidth, decode @ctx=128:  ours 127.4 GB/s (54.6% of copy BW)
                                               llama.cpp 159.6 GB/s (68.4%)
  Both are far from the bus: at M=1 a 124M model is launch/latency-bound, not byte-bound (Stage 5).

  *** PyTorch eager is NOT a compute baseline at this model size. *** Its trunk prefill is 9.441 ms at
  P=128 and 9.395 ms at P=512 -- IDENTICAL for 4x the work -- and its decode is flat at ~9.3 ms across
  ctx=128..1023. It is host-dispatch-bound: the Python loop's ~9.4 ms of launches hides the GPU work.
  That is why beating it on decode by 4.2-4.8x is table stakes, and why llama.cpp is the real signal
  (DESIGN.md §8 / BENCH_PROTOCOL §5 framing: beating eager is table stakes; approaching llama.cpp is the
  actual signal).

=================== SECONDARY: our INT8 vs llama.cpp Q8_0 -- NOT clean apples-to-apples ===================

*** PRECISION-SCHEME MISMATCH, STATED UP FRONT. This is CONTEXT, NOT A HEADLINE. ***
  ours   : MIXED. 48 block matmuls INT8 (symmetric, per-channel) + the tied head kept fp16 (forced by the
           Stage-4 quality kill-test) + fp16 activations.        -> 162.1 MB streamed/token.
  Q8_0   : UNIFORM int8 over all weights, llama.cpp's own block-scaled scheme (8.67 bits/weight).
                                                                 -> 128.64 MiB = 134.9 MB streamed/token.
  Different schemes, different byte counts, different quality points. No Delta-ppl was measured for Q8_0
  here, so the QUALITY axis is NOT matched either. Read it as context only.

                        ours INT8 (mixed)      llama.cpp Q8_0 (uniform)
  decode ctx=128    1.8358 ms (544.7 t/s)   1.0808 ms (925.3 t/s)
  decode ctx=512    1.8406 ms (543.3 t/s)   1.0962 ms (912.2 t/s)
  decode ctx=1023*  2.0087 ms (497.8 t/s)   1.1492 ms (870.1 t/s)
  prefill P=512     107.1 ms (INT8 = 1.00x on prefill, Stage 4)   6.242 ms
  -> 1.70x / 1.68x / 1.75x in llama.cpp's favour on decode. Of that, ~1.20x is simply BYTES
     (162.1 vs 134.9 MB) -- and our extra bytes are the fp16 tied head that our OWN quality gate
     requires (Stage 4: quantizing it cost KL 0.257 and Delta-ppl +0.549, both gate failures). The
     residual ~1.41x is the same CUDA-graph/fusion advantage seen in the fp16 comparison.

=================== VERDICT ===================
  * DECODE, matched precision: we are at 76-83% of llama.cpp, and 4.2-4.8x faster than PyTorch eager.
    For a from-scratch engine with no tensor cores, no CUDA graphs and no kernel fusion, that is what the
    roofline predicts: decode is bandwidth/latency-bound, and those levers buy little there.
  * PREFILL, matched precision: we are 10-14x slower, and the gap is FULLY accounted for by two measured
    factors -- 3.12x we cannot close without WMMA, and 4.63x of GEMM microarchitecture we did not build.
    Nothing is unexplained. Prefill is where tensor cores live, and this ladder never claimed them.
  * The CHARACTERISED gap, not the raw ratio, is the deliverable. Both are recorded.
notes:            The llama.cpp converter patch (conversion/gpt2.py bias-skip) is REQUIRED to reproduce;
                  it is a genuine upstream bug, not a local workaround for our convenience.
                  Nothing in this section touched cuda/. The engine is unchanged and remains FROZEN.
                  Reproduce: tools/bench_pytorch.py ; llama.cpp build + llama-bench -o json (medians
                  recomputed from samples_ns, since llama-bench prints mean+-stddev, not median).
```
