# A GPT-2 inference engine, written from scratch in C and CUDA

This is a single-model, forward-inference-only engine for **GPT-2-124M**: first a pure-C fp32 forward
pass with no dependencies, then a CUDA port, then five stages of optimization applied one at a time.
Every stage is gated against a HuggingFace reference before it is timed, every number is checked against
a roofline ceiling measured on the actual GPU, and every stage that failed to pay for itself is still in
the repository, with the measurement that killed it.

There is no novelty claim here. The claim is that the numbers are true, and that I can tell you exactly
where each one comes from.

---

## Phase 2

The fixed Phase-1 ladder is now open to measured follow-up experiments. Start with
[PHASE2_AUDIT.md](PHASE2_AUDIT.md), the immutable [pre-registration](PHASE2_PLAN.md),
and [PHASE2_RESULTS.md](PHASE2_RESULTS.md). The first experiment is optional CUDA
Graph decode: the original 135 arithmetic kernels, with 25 changing node parameters.
Phase-1 backends and build scripts remain available.

**Validated on the target GPU:** across three independent runs, graph replay
reduces complete fp16 generation latency by **22–30% at short context** and
**10–12% at long context**, with bit-exact eager/graph results and unchanged HF
gates. The short llama.cpp median gap is closed in the new controlled harness;
about **56%** of the long-context gap is removed, subject to the documented
cross-engine arithmetic differences. Prefill is unchanged. Build and reproduce
with [PHASE2_REPRO.md](PHASE2_REPRO.md); graph execution is opt-in.

**Phase 2.2: useful negative result.** A value-only attention experiment did
not improve decode. Four partial sums failed an additional preregistered
numerical bound; an order-preserving four-load variant passed bit-exact/HF
checks but made long GPU decode **30% slower (fp16), 36% slower (mixed INT8)**.
The compiler already overlaps sixteen value loads in the original kernel.
Source-correlated profiles instead put ~64% of baseline attention samples in
the strided key-dot-product phase. The accepted graph baseline is unchanged;
no llama.cpp gap was closed. See [PHASE22_RESULTS.md](PHASE22_RESULTS.md),
[audit](PHASE22_AUDIT.md), and [reproduction](PHASE22_REPRO.md).

## Where Phase 1 landed (historical snapshot)

Same GPU (RTX 4060 Laptop, sm_89), nominal fp16/F16, medians not best-of-N.
These historical instruments have different boundaries: PyTorch prefill excludes
the head, our decode repeats a fixed position without sampling, and llama.cpp
advances a short window. F16 GGUF also retains some fp32 tensors. Do not treat
this table as an exactly matched modern benchmark; Phase 2 supplies new controls.

| | **this engine** (fp16) | **llama.cpp** F16 (CUDA) | **PyTorch** fp16 eager |
|---|---|---|---|
| prefill @ P=128 | 27.548 ms | **2.732 ms** | 9.441 ms |
| prefill @ P=512 | 107.140 ms | **7.422 ms** | 9.395 ms |
| decode @ ctx=128 | 511.7 tok/s | **636.4 tok/s** | 107.4 tok/s |
| decode @ ctx=512 | 503.8 tok/s | **604.0 tok/s** | 107.9 tok/s |
| decode @ ctx=1023 | 448.6 tok/s | **586.3 tok/s** | 106.4 tok/s |

**Decode runs at 76–83% of llama.cpp and 4.2–4.8× faster than PyTorch eager. Prefill is 10–14× slower
than llama.cpp.**

At P=512 the historical 14.44× ratio can be expressed using measured ceilings:

```
14.44x  =  3.12x   ratio of compute ceilings (not a measured WMMA speedup)
                   31.51 TFLOP/s measured tensor-GEMM ceiling
                 / 10.10 TFLOP/s measured achievable CUDA-core GEMM ceiling
        x  4.63x   residual efficiency ratio (not an isolated maturity experiment)
                   llama.cpp reaches 39.3% of its ceiling; this engine reaches 8.5% of its own
```

This identity motivates a prefill GEMM ladder; it does not causally allocate the
gap. No tensor-core intervention or register-tiling ablation was performed in
Phase 1. The shipped prefill GEMM remains a 16×16 shared-memory tile.

The historical per-op profile assigns 5.3% of prefill to attention. Its sum being
within 0.7% of forward time does **not** bound launch overhead: both event-timed
quantities include scheduling gaps. One unsuccessful LayerNorm fusion does not
prove that every possible fusion would lose.

The historical decode gap motivated the Phase-2 graph experiment. The previous
"73% fixed overhead / 27% attention" split was inferred from cross-engine timing
differences, not measured by removing launches. See the new intervention and
timeline evidence in [PHASE2_RESULTS.md](PHASE2_RESULTS.md).

One number is worth reading twice: **PyTorch eager's trunk prefill is 9.441 ms at P=128 and 9.395 ms at
P=512.** Identical, for four times the work. It is host-dispatch-bound at this model size, which is why
beating it on decode is table stakes and why llama.cpp is the only baseline that actually says anything.

---

## The ladder

Each stage changes one thing, re-passes the correctness gates, and is then timed against the previous
stage back-to-back on the same thermal state. A stage that doesn't earn its keep gets reverted.

**Stage 0 — pure-C fp32 forward.** No dependencies, double-accumulated. This becomes the correctness
reference for everything after it. Prefill @512: 3347 ms. It is slow and it is *right*, which is the
only ordering that works.

**Stage 1 — naive CUDA port.** One thread per output element, no shared memory. Re-gated against the
HF **fp16** oracle rather than the fp32 one, because comparing across precisions hides bugs in rounding.
Prefill @512: 840 ms.

**Stage 2 — tiled shared-memory GEMM.** 4.55× on prefill @512. It also appeared to make "decode" 5.4×
faster, which was the most instructive moment in the project: without a KV cache the decode harness
recomputes the whole sequence every step, so its GEMMs run at M = context length, not M = 1. That is
prefill in disguise. The win was real and the label was wrong.

**Stage 3b — flash attention.** Online softmax, no materialized score matrix. 1.72× on prefill @512;
attention falls from 44.8% of the forward to 5.3%. The target was chosen by profiling first, not by
assuming.

**Stage 3a — fused LayerNorm + matmul. Reverted.** It is correct, it passes every gate, and it is
**2.9% slower**, identically at three shapes, with disjoint min/max ranges. The tiled GEMM re-stages each
A tile once per column tile — 144× for the QKV shape — so an on-the-fly fusion recomputes the LayerNorm
144 times to save one activation round-trip. This fusion lost its reuse advantage.
The cited implementation commit is absent from the public clone's history;
the recorded negative result is preserved in BENCHMARKS.md.

**Stage 4 — weight-only INT8.** The pre-registered scheme (symmetric, per-channel, all 49 matmul weights)
**failed both gates**: max KL 0.257 against a 0.02 bound, and Δppl +0.549 against a +0.3 bound frozen
before the stage. Before touching anything I checked whether the kernel was wrong or the quantization was:
running the *already-validated fp16 GEMM* on dequantized weights reproduced the failure to four significant
figures, which exonerated the kernel. Perturbing one tensor group at a time located the damage in the
**tied output head** — the one matmul whose output goes straight into a softmax, with no downstream
LayerNorm to rescale the error away. The pre-registered kill-test (keep the sensitive layer in fp16)
recovers both gates: Δppl **+0.027**. The bound was never re-tuned.

INT8 then delivered **zero speedup**. Also honest, also measured, and explained below. BENCHMARKS.md also
records an INT8-vs-llama.cpp-Q8_0 decode comparison, deliberately not headlined here: the schemes don't
match. Ours is mixed precision — 48 block matmuls in INT8 with the tied head held in fp16 because the
quality gate demands it — while Q8_0 is uniform int8 over every weight. Different byte counts, different
quality points, and no Δppl measured for Q8_0. It is context, not a result.

**Stage 5 — KV cache, memory planner, and a true M=1 GEMV.** Decode goes from full recompute to a single
cached step: **14.7× at ctx=128, 56.0× at ctx=512, 98.3× at ctx=1023**. The GEMV matters as much as the
cache: the 16×16 tiled GEMM at M=1 computes sixteen output rows and throws fifteen away, so it is saturated
on arithmetic it discards. Replacing it with a one-warp-per-output-row GEMV is worth 2.8–3.0× on the
decode step by itself.

---

## What this project actually taught me

**A kernel can be compute-bound on work it throws away.** Stage 4's INT8 halves the weight bytes and buys
exactly 1.00×, at every shape. The reason took two falsifiable predictions to pin down: if the M=1 head
GEMV were bandwidth-bound, halving its bytes would give ~2× (measured 1.002×, rejected); if it were
compute-bound on the tile's discarded rows, then M=1 and M=16 would cost the same (measured 1.4351 ms vs
1.4346 ms, confirmed). It runs at 860.6 GFLOP/s — 98.8% of the *same kernel's* M=512 throughput — while
pulling 23% of achieved bandwidth. Fewer bytes cannot help a kernel that isn't waiting on bytes.

**INT8's decode payoff is bounded by the quality gate, not by the hardware.** Even after Stage 5 gave it
a KV cache and a proper GEMV, INT8 buys 1.02–1.16×. Two structural reasons, both measured: the tied head
is 31% of the weight bytes and *must* stay fp16 or the quality gate fails, and the 48 quantizable block
matmuls are 1.2–4.7 MB each, small enough to sit on a per-kernel latency floor at M=1. The weight-side
ceiling on the whole-step speedup is 1.278×. The director map predicted "high"; the honest answer is that
it is high for the models that prediction was written about, and GPT-2-124M is too small. That is a
property of the model, not a failure of the engine.

**At batch=1 a 124M model is launch-bound, not bandwidth-bound** — and llama.cpp independently confirms it.
Our decode step reaches 127.4 GB/s (54.6% of copy bandwidth); llama.cpp, with CUDA graphs and fused kernels,
reaches 159.6 GB/s (68.4%). Both are far from the bus. The roofline said decode was memory-bound; at this
model size, on this GPU, the thing it is actually waiting on is kernel launches.

**I checked my own denominators, and one of them was wrong.** For several stages the CUDA-core compute
ceiling was *derived from the clock* (24 SM × 128 lanes × 2 × 2.61 GHz = 16.0 TFLOP/s) and then used as the
denominator for "% of roofline". Measuring it (`microbench.exe cudacore`) showed the achievable non-tensor
GEMM is **10.10 TFLOP/s** — the assumption was optimistic by 1.55×, and the ridge point it implied (≈69) was
simply wrong. The real one is 43.3. Labelling a number "derived, not measured" does not license using it as
a measurement.

That correction is in the git history rather than quietly folded in, along with a `clock64()`-based
cross-check that reported 146% of the hardware's issue width — impossible — and was therefore discarded
instead of published.

---

## Reproducing it

**Build the engine** (Windows, MSVC + CUDA 12.6; the MinGW `gcc` on PATH is 32-bit and cannot hold the
498 MB fp32 model). Three scripts, each callable from the repo root; together they build every binary
named below:

```bat
build_cuda.bat      :: Stages 1-5: correctness_cuda, kv_gate, bench_decode, profile_decode,
                    ::            profile_forward, ab_forward, eval_ppl_cuda, microbench
build.bat           :: Stage 0: the pure-C fp32 reference (bench\correctness.exe)
build_profile.bat   :: the isolated matmul harness the ncu section drives (bench\profile_matmul.exe)
```

**Run the gates** (correctness before speed, every time):

```bat
set GPT2_BACKEND=flash
bench\correctness_cuda.exe all      :: gates (a) (b) (c) vs the HF fp16 oracle
bench\kv_gate.exe                   :: Stage-5 KV-cache decode gates + cached-vs-recompute equivalence
```

`GPT2_BACKEND` selects the kernel path: `naive` | `tiled` | `flash` | `int8` | `gemv`. All six shipped
paths (those five plus the pure-C fp32 reference) pass the gates.

**Benchmark:**

```bat
bench\bench_decode.exe 50           :: true M=1 decode; KV vs no-KV; naive/tiled/gemv/int8  (50 = ITERS, see below)
bench\bench_decode.exe 256          :: the same, at BENCH_PROTOCOL §6's N >= 256 median samples
bench\profile_decode.exe 512        :: per-op attribution + achieved bandwidth per GEMV
bench\microbench.exe cudacore       :: the measured CUDA-core ceiling behind every "% of roofline"
bench\profile_matmul.exe bw 50257 768  :: the M-sweep: why tiling loses at M=1 on the tied head
```

**`bench_decode`'s argument is `iters`, not decode steps.** At batch=16, `50`
means 50 independent batch averages over 800 forwards, **not 800 independent
per-token samples**. Batching changes the latency distribution and is not a
substitute for the protocol's 256 per-token samples. The Phase-2 harness retains
both instruments, labels them separately, and adds actual host-sampled generation.

**Reproducing the llama.cpp baseline** — three traps, all of which cost me time:

1. Mainline's HF→GGUF converter is **broken for GPT-2**. `conversion/gpt2.py` intends to drop the legacy
   causal-mask buffers, but hands them to `map_tensor_name()` instead, raising
   `Can not map tensor 'h.0.attn.bias'`. The one-line fix is at `tools/llamacpp_gpt2_convert.patch`.
2. **Use `llama-completion`, not `llama-cli`.** `llama-cli` is now a chat front-end and injects a ChatML
   template; GPT-2 tokenizes `<|im_start|>` literally and emits garbage. That is a front-end artifact, not
   a broken model — `llama-completion` gives *"Paris … Berlin … Washington … London"*.
3. **`llama-bench` prints mean ± stddev, not median.** Use `-o json` and recompute the median from
   `samples_ns`. Measure decode at a stated context with `-d/--n-depth` and a short `-n`, or the context
   drifts across the generation.

```bat
python tools\bench_pytorch.py       :: the PyTorch eager baseline (TF32 off, sync-bracketed)
```

---

## The documents

The methodology is the point, so it is written down and it was written down *first*.

- **[DESIGN.md](DESIGN.md)** — the design log: scope, the three firewalls, the fixed ladder, and the
  conjectures with the kill-test that would falsify each. Written before any kernel existed.
- **[ROOFLINE.md](ROOFLINE.md)** — the measured ceilings, the three ridge points and which one governs
  what, and the director map saying which stage should move which regime. Also where two of its own
  predictions were later refuted.
- **[QUALITY_GATES.md](QUALITY_GATES.md)** — what "correct" and "quality-preserving" mean, pre-registered,
  including the Δppl bound that INT8 breached and the kill-test that recovered it.
- **[BENCH_PROTOCOL.md](BENCH_PROTOCOL.md)** — how every number is measured, including the checklist of
  ways a benchmark silently lies.
- **[BENCHMARKS.md](BENCHMARKS.md)** — every validated run, with clocks, spread, and the ceiling it was
  checked against. Including the stages that failed.
- **[BUILD_PLAN.md](BUILD_PLAN.md)** — file-by-file build order and the trap attached to each.

---

## Scope

Forward inference, batch=1, one model, one GPU. Not a framework, serving system,
multi-GPU runtime, or llama.cpp clone. Phase 2 tests graph capture; tensor cores
and mature prefill GEMM remain unimplemented hypotheses. Their absence alone
does not quantify how much of a measured gap they would close.

**INT4 is not attempted, and the reason is measured rather than declined.** Its bound is pre-registered and
**LOCKED** (Δppl ≤ +1.0, [QUALITY_GATES.md](QUALITY_GATES.md) §2), but the shipped INT8 build clears the KL
gate with a margin of only **1.34×** (max KL 1.495e-2 against the 0.02 bound, where fp16 had 10×), and the
48 quantized *block* matmuls consume nearly all of that budget on their own (blocks-only KL 1.283e-2). So
INT4 has **essentially no headroom on this model** without further mixed precision, and would very likely
fail the gate. The bound stays locked anyway: if an INT4 attempt breaches it, that is an honest negative
result to report — not an invitation to re-tune the bound afterwards, exactly as INT8's breached +0.3 was
kept.

Hardware: RTX 4060 Laptop (AD107, sm_89, 24 SM, 8 GB GDDR6, 105 W). Measured: 233.4 GB/s copy bandwidth,
248.9 GB/s read, 31.51 TFLOP/s tensor GEMM, 10.10 TFLOP/s achievable CUDA-core GEMM.
