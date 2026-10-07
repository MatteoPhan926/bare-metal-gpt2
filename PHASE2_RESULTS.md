# Phase 2, iteration 1 — accepted opt-in decode graph

2026-10-05, RTX 4060 Laptop / sm_89. **MEASURED_VALIDATED**, not
IMPLEMENTED_UNMEASURED. Ordinary execution remains the default.

## Before implementation

The [audit](PHASE2_AUDIT.md) selected CUDA Graph capture ahead of prefill GEMM,
WMMA, long-context attention and further quantization. Stable allocations and
only 25 changing kernel-node parameters made a clean scheduling intervention
possible. Phase 1's 135 launches were real; its “73% overhead / 27% attention”
allocation of the llama.cpp gap was an inference, not an intervention.

[PHASE2_PLAN.md](PHASE2_PLAN.md) was committed as `7018406` **before code or graph
timings** and has not been rewritten. It predicted 0.15..0.45 ms short-context
savings, required at least 0.19 ms to support the strong mechanism claim, and
required >=5% whole-generation improvement in three runs without >3% long-context
regression. Original repository revision: `b56e5b8`.

## What changed and what did not

`cuda/decode_graph.cu` captures the existing decode function: **135 original
kernels, no arithmetic edits**. Each replay updates token/position, 12 KV-write
positions, and 12 attention lengths/shared-memory sizes. One graph covers all
1024 positions; no recapture, padding, alternate attention, device argmax, input
copy kernel, or quantization change. Weights, scratch, logits and KV addresses
are borrowed and stable. Capture restores host length and does not execute.
Topology, ownership and bounds checks fail closed. Setup/destruction synchronize.

`build_phase2.bat` selects a capturable per-thread default stream for every TU;
ordinary and graph paths coexist in the **same binary**. The legacy-stream
Phase-1 build is preserved. `GPT2_DECODE=ordinary|graph` selects the gate's policy;
the graph API and new speed harness expose both. Implementation/harness snapshot:
`a56faba`; subsequent harness-only fixes and extended tests are recorded in git.

## Correctness before speed

Both original fp16 GEMV and mixed INT8 passed the original gates before timing.
Graph/eager tests pass bit-exact finite logits and layer states across 1024
positions, cache boundaries, changed token sequences, reset/reuse, prefill to
decode, independent caches, and 128 free-running greedy steps. The no-diagnostic
speed graph is separately gated. Invalid tokens/positions, changed cache layout
and cross-thread use are rejected. Capture leaves poisoned cache bytes unchanged.

| Existing HF gate | Ordinary fp16 = graph fp16 | Ordinary INT8 = graph INT8 |
|---|---:|---:|
| KV/recompute max logit difference | 0.1875 | 0.2500 |
| KV/recompute max KL | 0.0004883 | 0.0004687 |
| Worst layer relative error | 0.001073 | 0.009388 |
| Teacher-forced greedy | 127/128, one A1 near-tie, zero bugs | same |
| 512-token max KL | 0.001953 | 0.01490 |

No tolerance was loosened. Nonfinite values now explicitly fail relative-error
and KL checks. INT8 remains 48 quantized block matrices plus an fp16 head; no new
quality point is proposed. Perplexity was not remeasured: weights/arithmetic are
unchanged, and exact equivalence plus the existing HF distribution gate is the
intervention's correctness test. Historical perplexity remains historical.
See [baseline fp16](docs/phase2/baseline_gate_fp16/stdout.txt),
[baseline INT8](docs/phase2/baseline_gate_int8/stdout.txt),
[graph HF fp16](docs/phase2/hf_graph_fp16/stdout.txt),
[graph HF INT8](docs/phase2/hf_graph_int8/stdout.txt), and final full-context
[fp16](docs/phase2/verified_exact_fp16/stdout.txt) /
[INT8](docs/phase2/verified_exact_int8/stdout.txt) speed-graph tests. The four
`verified_hf_*` runs re-pass both policies after the final build; the CPU-only
statistical tests also pass (3 tests).

## Measurement boundaries and environment

Three independent processes per backend, interleaved ordinary/graph and reversed
order on alternate pairs. Each condition has **256 individual per-token samples**,
plus separately labelled **50 batches of 16** at fixed positions. Growing windows
are 128..143, 512..527 and 1007..1022, after five discarded steps, with 16 repeated
windows. Warmup includes 1.5 seconds sustained GPU work; one full window is also
discarded. Inputs are 1024 actual WikiText IDs, validated against the frozen 512-ID
oracle. No seed/random token selection. Greedy paths must produce identical tokens.

Forward-only events include device idle/submission gaps, not just kernel work.
Whole-generation wall time includes parameter updates, event calls, logits D2H,
stream synchronization, finite checking and CPU argmax. Load, prefill/reset,
setup and tokenization are excluded from steady-state decode. No per-token file
I/O occurs inside measurement. All samples, including outliers, are retained.

Windows/WDDM, driver **561.09**, CUDA **12.6.20**, MSVC Build Tools 2022, `-O3
-std=c++17 -arch=sm_89`. Under load, engine runs held **2595 MHz** median SM clock
(range 2595..2610), **8000 MHz** memory, median temperatures **61..65 C**, median
power **54.6..57.2 W**. Clocks were not locked. External warmed runs held
2595..2610/8000 MHz. Telemetry is every 200 ms over the whole process, not perfectly
aligned per-token counters. Receipts pin source/executable/input hashes and commands.

## Whole-generation result (the acceptance metric)

Entries are **median of three process medians [minimum, maximum of all 768
samples]**, milliseconds/token. Tok/s is the reciprocal of the displayed median.
The last column is the range of within-run median latency reductions, not a ratio
constructed by choosing favorable runs. Wide/overlapping tails are not hidden.

| Backend; context window | Ordinary ms/token | Graph ms/token | Ordinary → graph tok/s | Reduction in runs 1..3 |
|---|---:|---:|---:|---:|
| fp16; 128..143 | 2.176 [1.694,3.731] | 1.565 [1.506,2.426] | 460 → 639 | 22.3..29.9% |
| fp16; 512..527 | 2.119 [1.966,3.401] | 1.926 [1.758,2.699] | 472 → 519 | 8.9..18.6% |
| fp16; 1007..1022 | 2.398 [2.283,3.860] | 2.148 [2.086,3.013] | 417 → 466 | 10.4..11.9% |
| mixed INT8; 128..143 | 2.020 [1.489,3.850] | 1.291 [1.237,1.938] | 495 → 775 | 35.6..36.7% |
| mixed INT8; 512..527 | 2.141 [1.678,3.303] | 1.564 [1.489,2.772] | 467 → 639 | 25.1..29.9% |
| mixed INT8; 1007..1022 | 2.239 [1.992,4.056] | 1.901 [1.794,2.704] | 447 → 526 | 11.7..18.9% |

**All acceptance conditions pass in every run.** This is a median improvement,
not a claim that every token is faster or that jitter disappeared. CPU sampling
is measurable, not “negligible”; its cost/variance remains outside the graph.

Unprofiled advancing-forward **event** medians (same aggregation; no sampling):

| Backend | 128..143 ordinary → graph | 512..527 ordinary → graph | 1007..1022 ordinary → graph |
|---|---:|---:|---:|
| fp16 | 1.845 → 1.377 ms | 1.872 → 1.637 ms | 2.186 → 1.959 ms |
| mixed INT8 | 1.721 → 1.089 ms | 1.728 → 1.332 ms | 1.905 → 1.661 ms |

Every per-run median, IQR, min/max, host enqueue duration and update duration is
in [aggregate.json](docs/phase2/aggregate.json) and each run's `summary.json` and
`stdout.txt`: [fp16 1](docs/phase2/fp16_run1/summary.json),
[2](docs/phase2/fp16_run2/summary.json), [3](docs/phase2/fp16_run3/summary.json);
[INT8 1](docs/phase2/int8_run1/summary.json),
[2](docs/phase2/int8_run2/summary.json), [3](docs/phase2/int8_run3/summary.json).

Controls: ordinary prefill P512 stays ~**107.9 ms** in both policy-labelled
buffers (paired median differences <0.02%); isolated fp16 head stays ~**0.34 ms**.
No prefill claim changed. Legacy-stream fixed-position event medians are
2.029/1.881/2.180 ms at 128/512/1023; per-thread ordinary medians span
1.905..1.954 / 1.847..1.883 / 2.172..2.206 ms. Stream/host variance exists,
especially short context; **only the same-binary A/B isolates the graph effect**.
[Legacy control](docs/phase2/legacy_fp16/summary.json).

## Did the mechanism survive?

**Yes, the removable scheduling-overhead hypothesis survives.** Fp16 short-window
event savings are **0.479 / 0.544 / 0.330 ms**, all above the preregistered 0.19-ms
threshold. Fixed-position single-step savings are 0.543..0.599 ms. Two advancing
runs exceed the predicted 0.45-ms upper expectation: the bet underestimated those
runs. Mixed INT8 saves 0.586..0.649 ms in the advancing short window.

**The approximately constant absolute-savings prediction does not survive.**
Long-window fp16 savings are only 0.199..0.233 ms. Host submission overlaps device
execution; it is not a fixed additive tax that can be subtracted from every
context. Ordinary host enqueue is ~1.46 ms at fixed ctx128, versus ~0.045 ms for
graph; this does **not** imply a 1.42-ms latency saving because the work overlaps.

The kill-test does not trigger. Fixed-parameter graph replay is only ~6..20 us
below fully updated replay at short context. The 25 host updates cost ~6..15 us
in those conditions; node-update work is not the residual bottleneck. No claim
uses the no-update diagnostic as valid autoregressive throughput.

### Timeline evidence, not headline timings

Nsight Systems, 16 steps each, verifies **2160 kernels = 16×135** in both policies.
Graph traces contain 16 replays and 400 node updates. Median internal gap time:

| Context | Ordinary kernel sum | Ordinary internal gaps | Graph kernel sum | Graph internal gaps |
|---|---:|---:|---:|---:|
| 128 | 1.347 ms | 0.537 ms | 1.393 ms | 0.012 ms |
| 1023 | 2.054 ms | 0.251 ms | 1.934 ms | 0.011 ms |

Gaps exclude before-first/after-last kernel work. Profiled spans are ~1.888/1.405
ms at short and 2.309/1.945 ms at long. Kernel times change with tracing/clock
state even though arithmetic is identical; **do not derive an exact unprofiled
overhead percentage from these sums**. Nsight initialization itself stalls and
can change clocks. CPU context-switch tracing was unavailable without elevated
privileges; host scheduling vs driver behavior is not fully disambiguated.
[Short timeline](docs/phase2/profile/graph_128/timeline.json),
[long timeline](docs/phase2/profile/graph_1023/timeline.json); all intervals retained.

### Setup and amortization

Fp16 warm setup (30 per process): capture **0.0716 [0.0544,0.4929] ms**,
instantiate **0.1368 [0.1051,0.4977]**, upload+sync **0.0739 [0.0513,0.1301]**,
total **0.3225 [0.2350,1.2859]** including node discovery. Component medians do not
sum to the total median. First graph setup after model/CUDA initialization:
**3.47..4.05 ms** across processes. Mixed INT8: warm total **0.3391
[0.2506,0.5855] ms**, first setup **2.98..4.40 ms**.

First post-upload fp16 replay is ~1.36 ms event time, close to warmed ctx128
replay. Using observed short-forward savings, first setup needs roughly **7..13
tokens** to amortize; warm setup roughly one token. This is a ratio estimate,
not a separately measured cold-request TTFT. Tiny generations can lose when
setup is included, hence opt-in, stable lifetimes and ordinary fallback.

## How much of the llama.cpp gap closed?

New external runs use the same real IDs, windows, batch=1, ctx allocation=1024,
last-position logits, 1.5-s sustained warmup and no sampling. All 148 tensors
were checked against our export. A **separate** GGUF copy rounds the 907,776
original fp32 values to our engine's half values; every weight value then matches.
The original GGUF and external checkout were not changed. F16 K/V, full offload,
one CPU thread, flash disabled, existing CUDA Graph support. Revision/DLL/input
hashes are retained in [weight verification](docs/phase2/llama_weight_match/stdout.txt).

Important limit: llama uses fp32 intermediate/logit storage; ours uses fp16.
Host output rows are 201,028 vs 100,514 bytes. Thus this is **matched weight
values and workload, not bit-matched arithmetic or identical transfer bytes**.
Both sides include full host-visible logits, excluding sampling. Our CUDA event
calls add a small extra cost. The causal result is still the engine's A/B, not
the cross-engine ratio. No mixed-INT8/Q8_0 equality is claimed.

Milliseconds/token, median of run medians **[range of run medians]**; all
per-token extremes are retained in the linked summaries:

| Window | Ordinary fp16 | Graph fp16 | llama, matched-value F16 |
|---|---:|---:|---:|
| 128..143 | 1.901 [1.866,1.980] | 1.426 [1.422,1.444] | 1.681 [1.667,1.791] |
| 512..527 | 1.936 [1.930,1.963] | 1.682 [1.675,1.695] | 1.824 [1.754,1.835] |
| 1007..1022 | 2.256 [2.226,2.259] | 2.008 [2.002,2.019] | 1.816 [1.784,1.825] |

On these central estimates the **short and medium median gaps are eliminated**;
graph is 0.255/0.141 ms ahead. This is not proof of universal superiority over
llama: native-F16 control medians are **1.664/1.751/1.787 ms**, and Windows host
variation is substantial, especially at medium context. Long context retains a
**0.192-ms gap**: **~56%** of the new 0.440-ms ordinary gap is removed. This is
not 56% of the old Phase-1 number. Do not headline “216% gap closure” for short
context; a crossed baseline is better stated as no remaining median gap.
[Warmed llama 1](docs/phase2/llama_warm1/summary.json),
[2](docs/phase2/llama_warm2/summary.json), [3](docs/phase2/llama_warm3/summary.json),
[native control](docs/phase2/llama_native_warm/summary.json).

Historical PyTorch eager results remain contextual only: its trunk prefill
omits the head and the old crop handler can silently fail. It was not rerun or
used to manufacture a new speedup ratio; it does not isolate this graph bet.

## What is dominant now; the single next experiment

Graph decode is mostly GPU work. GEMV is the largest class (~82% short / 58%
long in the diagnostic timeline). Attention rises from **~0.129 to ~0.702 ms**,
~9% to ~36%, and accounts for nearly all graph latency growth with context.
An individual long-context launch has **12 blocks × 64 threads**, **46 registers/
thread**, **4.13% active-SM warp occupancy**, and **41.48 GB/s DRAM read rate**.
L1 load sectors/request = **16.99**. Source and counters agree: strided K loads,
serial V reduction, insufficient parallelism. NCU's 38 replay passes and 1.919-GHz
clock make its 76.192-us duration diagnostic only. [Counters](docs/phase2/ncu_attention_metrics/selected_metrics.json).

The new bandwidth probe gives copy **233.5 [229.5,234.4] GB/s** and read **249.4
[243.0,249.9] GB/s**. Graph throughput is below even the conservative logical
bytes/copy-BW reference; nothing exceeds a physical or measured ceiling. Low
arithmetic intensity does not mean the underfilled attention kernel saturates
memory bandwidth. See [ROOFLINE §8](ROOFLINE.md#8-phase-2-launch-overhead-removed-attention-is-not-a-saturated-bandwidth-kernel).

**Single best next experiment:** under the accepted graph policy, replace only
decode attention's serial context reduction with a partitioned-key/two-stage
reduction. Preregister its numerical reduction-order risk; test the unchanged
HF gates, kernel counters and end-to-end long-window latency. A 2× attention
improvement would offer ~0.35 ms of *profile-based potential*, not a prediction
of achieved speedup. It directly tests the remaining decode-scaling mechanism.
This promotes candidate D because its diagnosis is now counter-backed and the
narrow experiment has value on every subsequent long-context token.
Do not stack it with GEMV or prefill changes. Mature prefill GEMM remains the
largest separate absolute-latency project (~108 ms), untouched this iteration;
WMMA payoff remains unmeasured and INT4 has no new quality justification.

## Audit trail and reproduction

Exact commands/API requirements: [PHASE2_REPRO.md](PHASE2_REPRO.md). The raw
directory is [docs/phase2](docs/phase2). `phase2_report.py` aggregates **all three
runs**, asserts the acceptance thresholds, and never selects a fastest run.
Large binary profiler reports stay local; portable kernel intervals/counters
are committed. No benchmark was run concurrently with another GPU benchmark.

Known non-headline runs are explicitly retained, not erased: original
`baseline_decode` overlapped CPU compilation; initial `llama_run1` had per-replay
debug logging; `llama_matched1..3` lacked equal sustained warmup and were replaced
as a group by `llama_warm1..3`. The first NCU command used an incorrect executable
path and did not run; `ncu_attention_long_retry` succeeded. One intermediate link
attempt encountered a gate executable still running; the subsequent complete
build succeeded. None is used as a favorable timing denominator. Final tests
also extend the no-diagnostic graph to all 1024 positions and initialize cache
before standalone setup/first-replay diagnostics; the measured `all` loop and
the graph implementation itself are unchanged by those harness fixes.
