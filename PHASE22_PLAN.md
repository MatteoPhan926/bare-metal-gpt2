# Phase 2.2 preregistration — four-way value reduction

Frozen before implementing or timing the new kernel, 2026-10-05.
Baseline: accepted Phase-2.1 commit `9c8624e`. Target: RTX 4060 Laptop,
sm_89, 24 SMs. This document is not to be rewritten after results.

## Mechanism and competing explanation

The remaining context-dependent cost is primarily attention, not graph
submission: saved graph traces have ~11–12 us internal gaps, nearly constant
GEMV time, and attention grows from 0.129 to 0.702 ms/step. These are profiler
observations, not a measurement of which attention phase is limiting.

**Bet:** the serial `sum_j probability[j] * V[j,d]` in each of only 64
threads/head is a material latency/parallelism bottleneck. Divide that sum
into four contiguous, non-overlapping ranges, with four 64-thread groups
in the SAME block/head, then combine four fp32 partial sums. Retain the
original 64-thread QK computation and softmax reduction order. All threads
participate in block barriers. No new global scratch or kernel launches.

**Serious competitor:** strided key loads and their instruction/memory
latency dominate attention; alternatively increasing KV traffic/cache
pressure dominates the whole forward pass. The V-only change leaves key
accesses, logical KV bytes, graph topology and matrix kernels unchanged.
It may fail despite faster value work. Twelve head blocks still underfill
24 SMs; this experiment deliberately does not solve every attention flaw.

## Predictions, controls, acceptance

- At context ~1024, attention latency should fall at least 25%; expected
  whole GPU-resident forward reduction 8–22%, complete generation 5–20%.
  These ranges are predictions, not claims. Medium contexts should improve
  less in absolute time; short contexts may gain little or regress slightly.
- GPU event ms/token, host-visible-forward wall ms/token, complete greedy
  wall ms/token, and reciprocal tok/s are final metrics. Profile per-step
  attention, GEMV, other kernels and internal gaps separately.
- Expected unchanged: 135 kernels and 25 scalar-node updates/token, KV
  allocation/layout/logical bytes, QK/softmax arithmetic, prefill, head GEMV,
  weight values/precision, host sampling and D2H boundary. Profiler actual
  traffic can differ from logical bytes and must not be assumed invariant.
- **Falsifiable prediction:** >=8% long-context GPU-forward improvement in
  each of three process runs, >=5% long-context generation-wall improvement
  in their central process medians, with >=25% attention-time reduction and
  approximately flat GEMV time. A kernel-only win is insufficient.
- **Kill/revert:** any original numerical gate failure rejects the kernel
  before speed claims. If long GPU-forward gains are <8%, or short-context
  latency regresses >5% consistently, do not promote it. Preserve the
  opt-in experiment and negative result; leave the accepted default intact.
  If timing changes cannot be attributed to attention rather than clocks,
  host noise or other kernels, report inconclusive and investigate controls.

## Execution protocol

1. Recheck the unmodified accepted executable on real WikiText contexts.
2. Add opt-in graph attention policy `v4`; original remains default. Match
   graph topology, buffers and changing-position updates. No other kernels
   or optimizations change. Keep historical build/benchmark working.
3. Before performance: isolated attention versus double-precision CPU
   softmax, finite/boundary/poison checks; existing HF gate (a) <=1e-2,
   (b) A1 three-ulp near-tie rule, (c) KL <0.02, for fp16 and mixed INT8.
   Cross-policy sequential logits/layers/KV must also satisfy <=1e-2
   normalized max error and KL <0.02, including all 1024 positions, resets,
   prefill transitions and no-diagnostic topology. Changed reduction order
   need not be bit-identical. Exact graph-baseline tests remain unchanged.
4. Isolated attention checks use lengths 1..1024 including uneven partitions;
   require normalized max error <=1e-2 versus CPU. Poison future positions.
   An isolated paired microbenchmark is explanatory, not the headline.
5. Whole-engine paired original/v4 graph runs: three processes per precision,
   256 individual samples/condition. Retain Phase-2.1 windows 128..143,
   512..527, 1007..1022; five discarded preceding decode steps, 16 repeats
   plus one warm repeat, alternating A/B order. GPU-forward and host-logits
   workloads use identical teacher-forced inputs. Free greedy trajectories
   are retained and any divergence disclosed, not silently forced to match.
   Fixed-context diagnostics and 30 paired prefill/head controls supplement.
6. Warm >=1.5 s, telemetry every 200 ms, serial GPU workloads, no overlapping
   builds. Save every sample, receipt and failure; median + IQR and process
   spread, never best-of-N. Profile outside headline timing. If NCU locks
   clocks, disclose and use paired counters, not its latency as throughput.
7. Compare against matched local llama.cpp F16 where informative using the
   existing host-logits boundary and rounded-to-half GGUF. No new INT8/Q8
   quality-equivalence claim. Historical results are labelled historical.

## Scope limit

Do not add split-K blocks, coalesced-QK rewrites, new cache layouts, sampling
changes or GEMV tuning in this experiment. If V4 loses, that is useful
evidence for choosing the next intervention, not permission to stack fixes.
