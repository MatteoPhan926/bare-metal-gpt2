# Reproduce Phase 2.2

Run from the repository root, `E:\Downloads\metal_gpt` on the measured machine.
Use CUDA 12.6 / sm_89 and VS2022 Build Tools, the existing weights/reference
dumps, and no concurrent GPU workload or compilation during timing. The
original Phase-2.1 build/binaries remain separate. All output directories below
must be new; no tool overwrites earlier evidence.

```powershell
.\build_phase22.bat
if ($LASTEXITCODE -ne 0) { throw 'Build failed' }
powershell -NoProfile -ExecutionPolicy Bypass -File tools/phase22_suite.ps1 -Stage gates -Output docs/phase22/my_run
powershell -NoProfile -ExecutionPolicy Bypass -File tools/phase22_suite.ps1 -Stage bench -Output docs/phase22/my_run -LlamaDir E:\llamacpp
python tools/phase22_report.py docs/phase22/my_run
python tools/phase22_verify.py docs/phase22/my_run
powershell -NoProfile -ExecutionPolicy Bypass -File tools/phase22_profile.ps1 -Output docs/phase22/my_profile
powershell -NoProfile -ExecutionPolicy Bypass -File tools/phase22_ncu.ps1 -Output docs/phase22/my_counters
python tools/phase22_source_summary.py docs/phase22/my_counters
```

Omit `-LlamaDir` when that local baseline is unavailable. With it, the suite
uses the existing `bench/llama_phase2.exe` and matched-value
`weights/gpt2_llama_matched.gguf`; building/verifying those is documented in
[PHASE2_REPRO.md](PHASE2_REPRO.md). Do not substitute Q8_0 for mixed INT8.
No new PyTorch measurements are needed to isolate this same-engine change.

The suite checks nine correctness receipts and rejects changed C/CUDA sources
before benchmarking. Both kernels run in one executable; three processes per
precision, N=256 individual samples per condition, paired/reversed policy
order, warmup, all samples and 200-ms telemetry. `phase2_summary.py` also works
on any individual directory. Report median/IQR and all-process spread.

## Policies and timing boundaries

`GPT2GraphAttention::Original` is still the graph API default.
`GPT2GraphAttention::Ordered4` selects the order-preserving value-load loop.
It is **correct but rejected for measured performance regression**, not a
recommended execution policy.
`GPT2GraphAttention::V4` retains the rejected reassociation experiment for
research reproduction, **not a validated engine policy**. New policies require
the Phase-2.2 build; old builds return `cudaErrorNotSupported` for them.

The benchmark executable compares original versus ordered4 graphs:

```powershell
bench/bench_graph_phase22.exe gemv all 256
bench/bench_graph_phase22.exe int8 all 256
```

The shared Phase-2.1 harness retains its historical profile mode names:
`profile_ordinary` now means the **original graph** in this Phase-2.2 binary;
`profile_graph` means the **ordered4 graph**. The Phase-2.1 binary retains its
original meanings. CSV policies explicitly name `original` and `ordered4`.

GPU-forward event time excludes host-logit transfer/sampling. `forward_host`
wall time includes transfer/synchronization but not argmax. `generation` wall
time includes those and CPU greedy argmax. Teacher-forced forward IDs match;
free-greedy outputs are retained in CSV and trajectory differences printed.
Fixed-context/no-update graphs and batch16 averages are diagnostic only.

The inherited `setup,process_cold` label in Phase-2.2 denotes the first
**ordered4** graph after an original graph was already instantiated, not true
process-cold graph initialization. Do not use it for a cold-start claim.
Warm setup and first replay remain separate; setup is not this experiment's
endpoint. Prefill/head controls execute identical functions on both labels.

## Correctness and failed-attempt reproduction

```powershell
# Three input patterns, every length 1..1024, CPU-double oracle, poison checks:
bench/attention_gate_phase22.exe isolated
# All model positions, prefill/reset and capture/no-capture graphs:
bench/attention_gate_phase22.exe gemv
bench/attention_gate_phase22.exe int8
# Expected nonzero exit: preserve the rejected V4 cross-policy failure:
bench/attention_gate_phase22.exe gemv v4
# Optional reproduction of its isolated kernel check:
bench/attention_gate_phase22.exe isolated v4
```

HF gates use `GPT2_DECODE=graph`, `GPT2_BACKEND=gemv|int8`, and
`GPT2_ATTENTION=original|ordered4|v4` with `bench/kv_gate_phase22.exe`.
The suite restores the caller's environment. Numerical thresholds and A1
are unchanged. Ordered4 additionally must be bit-identical to original.

## Attention counters (not end-to-end speed)

```powershell
$Ncu='C:\Program Files\NVIDIA Corporation\Nsight Compute 2024.3.0\target\windows-desktop-win7-x64\ncu.exe'
python tools/phase2_run.py docs/phase22/my_ncu_original -- $Ncu --profile-from-start off --kernel-name regex:k_attn_decode --launch-count 1 --set full --target-processes all --export docs/phase22/my_ncu_original/attention bench/bench_graph_phase22.exe gemv profile_ordinary 16 1023
python tools/phase2_run.py docs/phase22/my_ncu_ordered4 -- $Ncu --profile-from-start off --kernel-name regex:k_attn_decode --launch-count 1 --set full --target-processes all --export docs/phase22/my_ncu_ordered4/attention bench/bench_graph_phase22.exe gemv profile_graph 16 1023
```

Run serially after headline measurements. NCU replay/clock control and Nsight
Systems graph tracing perturb timing. Keep reports locally and portable CSV/
JSON exports in git. Do not treat profiler wall time as throughput.
