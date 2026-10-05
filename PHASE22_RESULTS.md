# Phase 2.2 — negative result, sharper decode diagnosis

**Outcome B. No optimization promoted; the accepted Phase-2.1 graph remains
the default.** The specific “value loads lack useful parallelism” hypothesis
was falsified. The compiler already issues sixteen value loads ahead in the
baseline's hot loop. Explicit four-load grouping reduced that to four, added
instructions, and made long decode materially slower. Source-correlated
counter evidence now favors the strided **QK/key-read phase**, not the value
sum, as the largest attention subproblem. That next intervention is untested.

This is not a claim that values are free, or that all value-loop optimizations
must lose. Nor does a failed four-load implementation disprove every split-K
attention design. The useful result is the measured rejection of this narrow
mechanism, with a more discriminating next question.

## Before implementation

The [audit](PHASE22_AUDIT.md) reconstructed 135 graph kernels and 25 changing
nodes. Saved Phase-2.1 traces implicated attention in context scaling while
GEMV remained the largest total cost. They did **not** identify which attention
phase dominated. The initial [preregistration](PHASE22_PLAN.md), commit
`19fa7d1`, chose a value-only intervention instead of the previously suggested
all-at-once partitioned attention rewrite. Competing explanations were strided
keys, traffic/cache pressure, and insufficient parallelism across head blocks.

The unmodified accepted binary was rechecked first: fp16 graph GPU-forward
medians were **1.380 / 1.633 / 1.961 ms** at the three windows, reproducing
Phase-2.1. Raw samples: `docs/phase22/baseline_recheck`.

## Two implementations of one focused experiment

1. **V4 partial sums:** four 64-thread groups compute contiguous portions of
   the value sum in a 256-thread head block, then combine them. QK/softmax
   retain the original 64-worker arithmetic. This changes fp32 association.
   It passed the isolated CPU-double check and original fp16 HF gates, but
   failed the **additional preregistered** cross-policy relative-logit bound:
   position 173, **0.02041785 > 0.01**, despite KL **1.8243e-5**. This is not an
   HF-gate failure or proof of a mathematical indexing bug. We kept the stricter
   registered constraint and rejected it **before performance timing**.
   Code and failed runs are preserved in `a23dc14`; no V4 speed claim exists.
2. **Ordered4 replacement:** load four successive values/probabilities ahead,
   then execute FMAs in the original order, in the original 64-thread block.
   The [repair preregistration](PHASE22_PLAN_ORDERED.md), commit `3ca7eef`,
   preceded its implementation/timing. It retained the performance bar and
   **added bit-exact correctness**, rather than weakening any gate. QK,
   softmax, cache layout, precision, graphs, sampling and surrounding kernels
   remain unchanged. This implementation passed correctness but lost on speed.

Both remain explicit research policies in the separate Phase-2.2 build.
`Original` remains the graph API default. No split-K, cache-layout change,
GEMV tuning, new quantization, or prefill optimization was stacked on top.

## Correctness

The ordered-load path passes:

- Three isolated attention input patterns (normal, peaked, uniform), **every
  length 1..1024**, future cache entries poisoned; bit-identical to original.
  Worst normalized error versus CPU-double softmax: **7.487e-4**.
- Both precisions, all 1024 sequential positions: **bit-identical logits and
  captured layers**, zero observed cache error, correct untouched slots;
  reset/reuse, prefixes 1/63/128/512/1007, and diagnostic/no-diagnostic graphs.
- Original eager/graph exact gates, including 128 free-running greedy steps,
  invalid-input/thread/layout checks, and full no-capture sequences.
- Unchanged HF gates for original **and** ordered4: fp16 worst layer
  **1.073e-3**, eval max KL **1.953e-3**, top1 505/512; mixed INT8 worst layer
  **9.388e-3**, KL **1.490e-2**, top1 497/512. Both 127/128 teacher-forced tokens,
  one permitted A1 near-tie, zero bugs. No threshold changed.

All timed free-greedy trajectories matched across policies. Quantization and
prefill arithmetic did not change; no new perplexity or quality point claimed.
Nine gate receipts live in `docs/phase22/ordered`; five statistics tests pass.

## Whole-engine result: a repeatable loss

Measured locally 2026-10-05 UTC / 2026-10-06 Bangkok. RTX 4060 Laptop, sm_89,
24 SM, driver 561.09, CUDA 12.6.20, batch=1, same weights/IDs as Phase 2.1.
Three processes per precision, **256 individual samples/condition/process**,
alternating A/B, five discarded decode steps, 16 repeated 16-token windows
plus warm repeat. Setup/load/tokenization/prefill excluded from decode.
No overlapping GPU benchmarks or compilation. GPU clocks unlocked: run medians
2595–2610 MHz SM / 8000 MHz memory; load-proxy temperature medians 55–60 C.
Raw telemetry retains transient clock excursions; it is not per-event alignment.

Below: **median of process medians [minimum, maximum process median]**, ms/token.
Every individual sample, per-process IQR, and all-sample extrema are retained in
[aggregate.json](docs/phase22/ordered/aggregate.json) and adjacent run summaries.
These are regressions, not best-of-N or selected favorable runs.

| GPU-resident forward; window | Original graph | Ordered4 graph | Latency increase |
|---|---:|---:|---:|
| fp16; 128..143 | 1.374 [1.369,1.378] | 1.451 [1.448,1.451] | 5.6% |
| fp16; 512..527 | 1.628 [1.625,1.628] | 1.928 [1.927,1.930] | 18.4% |
| fp16; 1007..1022 | 1.955 [1.954,1.965] | 2.544 [2.541,2.564] | 30.1% |
| mixed INT8; 128..143 | 1.074 [1.072,1.075] | 1.151 [1.148,1.153] | 7.1% |
| mixed INT8; 512..527 | 1.329 [1.328,1.343] | 1.629 [1.627,1.645] | 22.6% |
| mixed INT8; 1007..1022 | 1.657 [1.656,1.662] | 2.252 [2.243,2.253] | 35.9% |

Complete generation includes node updates, graph replay, D2H, synchronization,
finite check and CPU argmax. CPU work has substantial variance; it is not called
negligible or removed from the final decision:

| Long-window generation | Original ms/token | Ordered4 ms/token | Original / Ordered4 tok/s |
|---|---:|---:|---:|
| fp16 | 2.187 [2.185,2.275] | 2.795 [2.737,2.820] | 457 / 358 |
| mixed INT8 | 1.944 [1.868,1.981] | 2.565 [2.482,2.709] | 515 / 390 |

Long generation latency worsens in **every** process: fp16 **24.0–27.8%**,
INT8 **32.0–36.8%**. Short GPU-forward latency also violates the >5% regression
kill criterion consistently. All registered end-to-end acceptance checks fail.
Head control remains ~0.345 ms; P512 prefill remains ~108.0 ms (30 samples per
label/process), with within-process paired prefill medians differing <0.03 ms.

## What the profiler and generated code actually show

New Nsight Systems node traces, same executable, 16 steps per condition:

| GPU component, ms/step | Original ctx128 | Ordered4 ctx128 | Original ctx1023 | Ordered4 ctx1023 |
|---|---:|---:|---:|---:|
| Attention (12 kernels) | 0.1138 | 0.1879 | 0.7028 | 1.2980 |
| GEMV (49 kernels) | 1.1195 | 1.1196 | 1.1185 | 1.1182 |
| Internal graph gaps | 0.01097 | 0.01087 | 0.01106 | 0.01114 |

The **+0.595 ms** long attention penalty accounts for the unprofiled
**+0.589 ms** fp16 forward penalty. No GEMV/cache-pressure or host-update
explanation is needed for the regression. Profiler start stalls/instrumentation
remain disclosed; those times are explanatory, not the throughput headline.

Paired long-attention Nsight Compute at ~1.919 GHz (not boost timing):

| Counter | Original | Ordered4 |
|---|---:|---:|
| Grid / block | 12 / 64 | 12 / 64 |
| Registers/thread; dynamic + static shared | 46; 4352 + 256 B | same |
| Active-SM warp occupancy metric | 4.169% | 4.169% |
| Executed warp instructions | 184,968 | 259,560 (+40.3%) |
| Global load instructions | 49,176 | 49,176 |
| L1 load sectors | 835,632 | 835,632 |
| DRAM bytes read | 3.160 MB | 3.167 MB |
| Local load/store sectors (spills) | 0 / 0 | 0 / 0 |
| Kernel duration (diagnostic only) | 75.456 us | 133.024 us |

Thus increased register pressure, spills, occupancy loss or a large increase
in DRAM bytes **do not explain the slowdown**. The generated hot value loop
does: original has **16 LDG instructions each executed 1536 times** (24 warps x
64 iterations); ordered4 has **4 each executed 6144 times** (24 x256).
The scalar-looking original was already unrolled/prefetched sixteen ways by
nvcc. Explicit four-way source grouping made the executed loop *less* parallel
and increased address/control work. FFMA, half conversion and load counts are
unchanged. Value-loop long-scoreboard samples grow **423 -> 1600**, whereas
QK long-scoreboard samples stay **832 -> 830**.

Baseline source-correlated PC samples: **QK dot 1055/1653 = 63.8%**, value loop
525/1653 = 31.8%, other 4.4%. These are **sampling proportions, not exact phase
durations or a claimed 63.8% Amdahl speedup**. The key dot accounts for 786,432
theoretical global sectors versus 49,152 for values; 737,280 key sectors are
classified excessive by the source counter. This is transaction inefficiency
at the cache/instruction level, **not 16x DRAM traffic**: total DRAM reads remain
near the logical ~3.146 MB single-layer KV size. Both paths read far below
saturated device bandwidth; neither changed logical model bytes.

[source_attribution.json](docs/phase22/source_attribution.json) is regenerated
by `tools/phase22_source_summary.py`, deduplicating inlined instruction addresses
and checking their executed-count sum against the whole-kernel counter. Raw
source CSVs, counters and disassembly are retained. The QK-focused competitor
now has stronger direct evidence than the initial value-first assumption.

## External baseline and what remains

Three fresh local llama.cpp runs use the same matched-value F16 GGUF,
revision `259f2e2a531af9ed3efa7f66adaa5eb5b53da95f`, contexts/IDs/batch/warmup,
host-visible logits without sampling. No external source/model was modified.
Arithmetic still differs: ours fp16 intermediates/logits versus llama fp32;
D2H is 100,514 versus 201,028 bytes. This is not bit-matched precision.

| Host-logits median ms/token; window | Original | Ordered4 | llama F16 |
|---|---:|---:|---:|
| 128..143 | 1.429 | 1.512 | 1.672 |
| 512..527 | 1.684 | 1.987 | 1.757 |
| 1007..1022 | 2.026 | 2.617 | 1.795 |

**Gap closed by Phase 2.2: zero.** The accepted long median gap remains
**0.230 ms** in these fresh measurements. Promoting ordered4 would enlarge it
to 0.821 ms. Cross-process spreads/IQRs are in the aggregate; historical
Phase-2.1 ratios are not rewritten using these newer measurements. No new
INT8/Q8_0 equivalence or PyTorch ratio is claimed.

The retained engine is still dominated overall by GEMV (~58% of long GPU
kernel time); attention (~36%) explains almost all context growth. Within
attention, **strided key-load latency now has the strongest subphase evidence**.
Host sampling adds variable fixed wall cost but does not explain the GPU slope.
No empirical/physical bandwidth ceiling was exceeded: original long fp16
logical weight+KV bytes/time is ~146 GB/s, below the previously remeasured
233.5 GB/s copy and 249.4 GB/s read references. This is a sanity check, not a
fresh DRAM-bandwidth measurement or attribution of all latency to bytes.

**Single best next experiment:** coalesce/stage only the key tile, preserving
the per-key fp32 dot-product accumulation order and the existing value loop.
Test whether lower key transaction/scoreboard cost materially lowers long graph
decode. Leave grid decomposition, value arithmetic, GEMV and prefill unchanged.
Do not simply hand-unroll the value loop further or assume a split-K rewrite
will preserve the numerical constraints.

## Retention / reproduction

[PHASE22_REPRO.md](PHASE22_REPRO.md) gives exact commands. All complete registered
runs enter the aggregate. The server restart killed the first fp16 run-3 attempt;
its incomplete directory is retained as `fp16_run3_interrupted/EXCLUDED.md` and
only that missing process was replaced. The failed V4 gates and a Windows
forward-slash batch-launch failure are also retained; none is a speed denominator.
The inherited setup `process_cold` label in the Phase-2.2 harness is not true
process-cold because an original graph already exists; no cold-start claim is made.
Large binary profiler reports remain local; portable CSV/JSON exports are in git.

Final evidence verification checks all **27,648** registered growing-window
samples, matching trajectories, one benchmark executable hash, unchanged
kernel/input hashes across gates and speed runs, both immutable preregistrations,
and the unchanged accepted attention source. The initial verification utility
mistook Windows cp1252 decoding of UTF-8 punctuation for a plan change; its
failed receipt is retained, explicit UTF-8 decoding passes, and git confirms
neither plan was edited. The unchanged Phase-2.1 build script also still builds.
