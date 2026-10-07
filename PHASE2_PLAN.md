# Phase 2 pre-registration — decode CUDA Graph, 2026-10-05

Written after the source/history audit and **before implementing or timing graphs**.
Starting revision: `b56e5b8`. Do not edit this hypothesis after seeing results;
append outcomes to BENCHMARKS.md and PHASE2_RESULTS.md.

1. **Mechanism.** Host submission and GPU scheduling gaps between 135 small
   kernels materially inflate short-context decode. Capturing the existing
   kernels into one reusable graph should reduce that overhead with identical
   arithmetic, traffic, kernel count and outputs.
2. **Strong competitor.** Most residual time is actual GEMV/attention execution,
   device scheduling, or WDDM synchronization. Twenty-five dynamic node updates
   and host sampling may eat the graph benefit. A graph cannot remove those kernels.
3. **Metrics expected to move.** Forward-only event and synchronized wall ms/token,
   whole autoregressive wall ms/token (including logits D2H + CPU argmax), reciprocal
   tok/s, host submission duration and timeline gaps. Report short (128), medium
   (512), long (1023 fixed / 1007..1022 advancing) context separately. Measure
   capture, instantiation, upload/first replay and amortization separately.
4. **Controls expected not to move.** Bit patterns of per-layer states, logits and
   KV data vs ordinary decode on the same inputs; kernel inventory; weight format
   and numerical gates; ordinary prefill and isolated head GEMV throughput.
   No quantization, GEMM, attention, fusion or GPU sampling change in this bet.
5. **Bounded expectation.** Predict **0.15..0.45 ms/token** short-context forward
   savings (roughly 8..23% of the historical 1.95 ms baseline). Absolute savings
   should be broadly fixed as context grows, with smaller relative improvement.
   This is a prediction, not a claimed speedup. Do not transplant old timings
   into a new-session denominator.
6. **Falsifiable prediction.** On matched, same-session forward workloads, dynamic
   graph replay saves at least **0.19 ms/token at ctx near 128**, half the historical
   0.3831 ms short-context gap. If less, the strong Phase-1 "most of the gap is
   removable launch overhead" prediction fails for this implementation. Only a
   new matched llama.cpp run determines the actual fraction of its gap closed.
7. **Kill-test.** Compare ordinary decode, graph replay with all necessary updates,
   and diagnostic fixed-parameter graph replay. If dynamic replay saves <0.10 ms
   at short context, inspect host update cost and unperturbed timeline GPU work/gaps.
   If fixed-parameter replay helps but dynamic does not, updates are the culprit;
   if neither helps, actual kernel/device cost competes with the hypothesis.
   Never headline a fixed-position graph with no updates as autoregressive speed.
8. **Acceptance/revert.** Any nonfinite output, graph/eager bit mismatch, failure
   of existing gates, unsafe cache boundary, or unaccounted ceiling crossing blocks
   acceptance. Accept as opt-in only if >=5% synchronized whole-generation wall
   improvement is repeated in three independent runs, without >3% long-context
   regression. Otherwise keep ordinary decode as the usable path, disable/revert
   the optimization and preserve the experimental implementation/evidence as a
   documented negative result. A correct but unmeasured path is
   `IMPLEMENTED_UNMEASURED`, never an accepted speed result.

## Minimal implementation and invariants

Capture **the existing decode function** on CUDA's per-thread default stream,
enabled for all translation units in a separate Phase-2 build. Keep the ordinary
path selectable in that build, and preserve a separately built legacy-stream
Phase-1 binary to check for a stream-policy confound. No arithmetic kernel edits.
NVIDIA CUDA 12.6 permits capture on `cudaStreamPerThread`, not `cudaStreamLegacy`:
[programming guide](https://docs.nvidia.com/cuda/archive/12.6.0/cuda-c-programming-guide/index.html#creating-a-graph-using-stream-capture).

All weight/scratch/logit/cache addresses remain fixed for the graph lifetime.
Capture does not execute, and the host length mutation during capture is restored.
One graph covers all positions: update embed token/position, 12 KV append positions,
and 12 attention lengths + dynamic shared-memory sizes (25 kernel nodes).
No new device input copy, graph variants, padding of attention work or graph
recapture per token. Check node types/counts and reject a changed topology.
The graph is owned by one host thread/device; synchronize before destruction and
never free or move borrowed allocations until it is destroyed. Position must
equal cache length and remain in [0,1023]; token must be in vocabulary.

Host argmax stays outside capture. Measure the complete launch-update/replay,
output transfer, synchronization, and sampling loop separately from the
forward-only mechanism experiment. Do not call CUDA events kernel execution time:
they include device idle gaps between event markers.

## Verification and measurement order

* Rebuild unmodified Phase 1; verify weight/oracle hashes; run existing HF gates
  for fp16 GEMV and mixed INT8; retain stdout, exit codes and environment.
* Add graph API and fail-closed tests. Exact eager/graph equality across 1024
  successive positions, boundary lengths (63/64/65,127/128/129,511/512/513,1023),
  reset/reuse with changed tokens, prefill then decode, independent caches,
  invalid tokens/positions and free-running greedy. Test fp16 and existing INT8.
  Apply the unchanged HF gates through the graph path as well.
* Time only after passing gates. Save **all raw samples**, not just summaries.
  Interleave A/B and reverse order on alternating pairs. Ten warmups, discard
  first five generation steps, >=256 timed per-token samples per condition;
  >=30 prefill samples. Fixed-position batches remain a separately labelled
  legacy instrument; report number of batches and calls, plus single-step samples.
* For growing context, use repeatable short windows and record the exact range.
  At the context limit use windows ending at 1023; a 256-token continuation from
  1023 is impossible. Timing reset/prefill is excluded from steady-state decode.
* Capture/instantiate at least 30 graphs sequentially (destroy each); report cold
  setup separately from warm median/spread. Include update cost and first launch.
* Record clocks, memory clock, temperature and power **during** runs, with source,
  compiler, weight and reference identities. GPU timing experiments run serially.
  Re-measure bandwidth; use counters to investigate any implausible result.
* Match llama.cpp F16 context window, batch=1, logits work and sampling exclusion
  for the forward-only comparison; retain its raw `samples_ns`, medians + spread.
  PyTorch eager is informative context, with its boundary differences disclosed.
  Do not compare mixed INT8 to Q8_0 as a matched quality point.
* Use Nsight Systems if available to compare kernel work and device gaps; Nsight
  Compute for a suspected residual, not serialized per-op event shares. Report
  instrumentation distortion; unprofiled timings decide the speed claim.
