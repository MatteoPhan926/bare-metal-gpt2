# Phase 2 audit — 2026-10-05

Starting revision: `b56e5b8e37c7b492925c7a793726652e24d2bf8a` (fresh clone;
no pre-existing workspace changes). Read the seven governing documents, all
`cuda/`, `bench/`, `cpu/`, `model/`, `tools/`, the recorded Nsight exports and
the twelve published commits. This is an independent audit, not a new endorsement
of the Phase-1 explanations.

## What the code does

GPT-2-124M, batch 1, 12 blocks, E=768, H=12, D=64, context at most 1024.
The C reference stores fp32 and accumulates reductions in double. CUDA stores
fp16 and accumulates in fp32. Weights are transposed to [out,in]; the embedding
is also the output head. Backend selection swaps function pointers; unknown
names silently select naive. There is no tokenizer or inference CLI.

Prefill: embed -> 12 x [LN, QKV GEMM, causal attention, projection, residual,
LN, FC, GELU, projection, residual] -> final LN -> optional logits. Tiled GEMM
is a 16x16, one-output-per-thread CUDA-core kernel, with padded float shared
memory, two barriers per K tile, no register tile or pipeline. Flash attention
uses 8 query warps / 32 keys per tile and online softmax. Neither uses WMMA.

Cached decode: embed -> 12 x [LN, QKV GEMV, KV append, attention, projection
GEMV, residual, LN, FC GEMV, GELU, projection GEMV, residual] -> final LN ->
head GEMV. Exactly **135 kernel launches**, plus optional diagnostic copies.
The GEMV uses four warps/block, half2 loads and fp32 warp reductions. Attention
uses only 12 blocks of 64 threads, a strided score pass, a shared-memory
softmax, and a serial loop over keys for each output dimension.

Weights, seven scratch allocations, logits and the 37.75 MB [layer][K/V][head]
[position][dimension] cache have stable addresses. No decode allocation or
synchronization occurs without diagnostic captures. Host `kv.len` is bookkeeping:
the kernels receive explicit position/length, and the documented `len == pos`
contract is **not checked**. Token and position are embed arguments; position
is an argument to 12 append kernels; length and dynamic shared-memory size change
on 12 attention launches. All launchers currently use the legacy default stream.
Sampling belongs to harnesses, not to this decode function.

Weight-only INT8 quantizes 48 block matrices; the sensitive tied head stays fp16.
It uses int-to-float conversion, not INT8 tensor cores. No INT4 implementation.

## Evidence and limits

* Historical gates and timings are detailed, with direct Nsight counters for
  head GEMV bandwidth and naive/tiled coalescing. Head traffic ~77.2 MB at
  ~243 GB/s is strong evidence of a bandwidth-limited **head**, not the whole step.
* Prefill ~107 ms at P=512 and decode ~1.95/1.98/2.23 ms at ctx=128/512/1023
  are recorded. The GPU and CUDA 12.6/driver 561.09 are present locally. Original
  weights/oracles exist at `E:/engine`; llama.cpp revision `259f2e2` and F16 GGUF
  exist at `E:/llamacpp`. Verify identities before reuse; rebuild our baseline.
* The "73% fixed launch overhead / 27% attention" claim is a **hypothesis**.
  It assigns the entire short-context cross-engine gap to launches and subtracts
  that from the long-context gap. Different kernels, fusion, timing boundaries
  and slightly different contexts are confounds. No graph intervention or
  unperturbed timeline established the partition.
* The 14.44 = 3.12 x 4.63 prefill partition is an arithmetic identity; only the
  two microbenchmarked ceilings are independent. It does not measure either
  the payoff of WMMA at GPT-2 shapes or the contribution of register tiling.
* Sum(per-op)/whole ~1.007 does **not** upper-bound launch overhead at 0.7%:
  both timings contain launches. Decode per-op instrumentation itself inflated
  latency ~1.51x. Use a timeline or low-interference controls for attribution.
* `bench_decode` is repeated, fixed-position **forward-only** replay, not greedy
  generation. It resets host length, overwrites the last KV slot, excludes
  sampling and amortizes event costs over batches. This contradicts protocol
  sections 2/3. Fifty batch averages are fifty samples, not 800 independent
  per-token latency samples. Keep these as labelled microbenchmarks and add
  an actual advancing-context generation measurement; retain median + spread.
* `meta.json` has **512** eval IDs; `bench_decode`/`profile_decode` use a static
  1024-entry array, zero-filled past 511. Their long-context inputs are partly
  token 0. INT8 timing also fills its cache with fp16 prefill. New comparisons
  must state inputs and fill each backend's own cache.
* External comparison is approximate: llama.cpp used 16 steps (1007..1022
  at the long context), ours repeated position 1023; PyTorch uses wall time,
  ours events. PyTorch's "trunk" omits even the last-token head. Its cache crop
  catches and ignores every exception. Raw external samples are not shipped.
* Quality gates have no explicit nonfinite rejection (NaNs can evade max-error
  comparisons). The oracle tool can silently replace WikiText with repeated
  prompt tokens if a download fails. A1 really did amend thresholds after
  observation; Phase 2 will use the existing amended bounds without changing them,
  and require exact graph/eager equivalence plus explicit finite checks.
* Empirical cuBLAS/copy peaks are reference measurements, not immutable physical
  caps; low arithmetic intensity alone does not establish the realized bottleneck.
  The fp16 byte estimate includes the entire position embedding although only
  one row is read. Cache hits also prevent logical bytes from being a strict
  DRAM bound. Any surprising ceiling crossing requires counters, not celebration.
* Public history starts with an integrated release: cited commits `fb5ad32`
  (reverted fusion) and `81bd251` (Stage 5) are absent. The claimed recoverability
  of that negative implementation from **this** clone is false. Preserve the
  recorded negative result, but distinguish it from reproducible source history.
* Stale items: DESIGN's KV bet still says OPEN; BUILD_PLAN lists absent tokenizer,
  Makefile and benchmark files; kernels_fused says LN fusion is "next"; several
  comments say Nsight remains blocked although counters were later obtained.
  Build scripts assume one MSVC install. Phase-1 numbers vary substantially even
  within the ledger (short decode 1.64..2.09 ms); same-session A/B is essential.

## Candidate ranking before implementation

Scores are ordinal 1..5 (larger complexity is worse). Payoff refers to an
achievable first experiment, not the full theoretical gap. Score = product / cost.

| Rank | Candidate | Payoff | Diagnosis | Isolation | Learning | Cost | Score |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | A: decode CUDA Graph | 3 | 4 | 5 | 5 | 2 | 150 |
| 2 | B: register-tiled prefill GEMM ladder | 5 | 5 | 4 | 5 | 4 | 125 |
| 3 | C: from-scratch WMMA prefill | 5 | 4 | 3 | 5 | 5 | 60 |
| 4 | D: long-context decode attention | 2 | 3 | 4 | 4 | 3 | 32 |
| 5 | E: quantization beyond INT8 | 1 | 2 | 2 | 3 | 4 | 3 |

**Choose A first.** The kernel count is measured directly from execution structure;
the removable time is uncertain and experimentally isolatable. B has the largest
absolute latency opportunity and is the likely next prefill project. C also changes
reduction order and must earn the unchanged gates. D is visibly under-parallel but
its end-to-end value should be assessed after A. E lacks new quality evidence;
the present INT8 KL margin is too small to justify reopening it now.
