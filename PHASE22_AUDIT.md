# Phase 2.2 audit — before the intervention

Source of truth: accepted `9c8624e`, source and raw `docs/phase2` artifacts,
including timeline exports, NCU counters and excluded runs; original
QUALITY_GATES and amended measurement protocol. Hardware is available:
RTX 4060 Laptop/sm_89. Phase-2.1 remains the comparison baseline.

## Critical path reconstructed

Each step captures/replays the original decode: embed; 12 repetitions of
LN, QKV GEMV, KV append, attention, projection GEMV, add, LN, expansion GEMV,
GELU, contraction GEMV, add; final LN and tied-head GEMV. Exactly 135 kernels.
Only embed token/position, 12 append offsets, and 12 attention lengths/shared
memory sizes change: 25 node updates. Borrowed weights, seven scratch buffers,
logits and a 37.75 MB [layer,K/V,head,position,64] cache have fixed addresses.
Host argmax and logits transfer/synchronization are outside the GPU graph.

The graph already removed most *internal* scheduling gaps: saved medians
~12 us short and ~11 us long. It did not remove CPU sampling, D2H, WDDM
submission or kernel work. The no-update diagnostic saved only ~6–20 us.

## Evidence and limitations

- Saved fp16 graph traces: attention 0.129 -> 0.702 ms from ctx128 to1023;
  GEMV 1.137 -> 1.119 ms. Thus attention explains the observed context slope,
  but GEMV remains the largest total cost (~58% at long context).
- One long-attention NCU profile: 12 blocks x64 threads, 46 registers/thread,
  ~4.13% active-warps metric, 41.5 GB/s DRAM, ~17 L1 sectors/load request.
  NCU ran at ~1.919 GHz, not the ~2.595 GHz benchmark clock. Low occupancy
  does not by itself prove the limiting dependency or predict a speedup.
- Reading a contiguous head slab is **not** the same as coalesced key loads:
  neighboring threads read keys 128 bytes apart. Values are coalesced, but
  each output dimension performs a serial length-sized fp32 sum. Only 12
  blocks cover a 24-SM device. These are three distinct possible costs.
- Graph timings include GPU idle gaps; profiled kernel sums exclude them.
  Profile-start stalls and clock changes prohibit treating traces as exact
  decomposition of uninstrumented end-to-end measurements.
- Host greedy work is measurable (~0.15–0.4 ms), not “negligible” as the
  historical protocol said. Retain GPU, host-logits, and generation boundaries.
- The long KV arena can exceed L2 capacity; extra traffic is a plausible
  competitor. Flat GEMV time weakens, but does not eliminate, that explanation.
- Phase-2.1 exact equality gates were justified by unchanged arithmetic.
  They cannot be reused as cross-kernel bit-equality gates when reduction
  order changes. The numerical HF thresholds must stay unchanged instead.
- External comparison is nominal F16: weight values matched by a local GGUF
  copy, but intermediate precision and D2H element sizes differ. No equivalence
  between our mixed INT8 and llama Q8_0. Excluded runs remain excluded.

## Decision

Graph-update tuning has little remaining payoff; sampling is a fixed wall
cost, not the GPU context slope; GEMV is large but nearly context-invariant.
Attention is therefore worth testing. A full split-context attention rewrite
would simultaneously change occupancy, key coalescing and reduction order.
First test only four-way parallel value accumulation within each head block.
It is a smaller discriminator between serial value work and strided-key/memory
cost. Keep QK, softmax and all surrounding engine work unchanged.

No performance conclusion about that intervention exists at preregistration.
See PHASE22_PLAN.md for the frozen predictions and rejection criteria.
