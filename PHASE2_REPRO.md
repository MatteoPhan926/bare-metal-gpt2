# Reproducing the Phase-2 graph experiment

Run from the repository root, on the RTX 4060 Laptop / sm_89. CUDA 12.6 and
VS 2022 Build Tools were used. No runtime cuBLAS call was added to the engine.
The original `build_cuda.bat` stays legacy-stream/ordinary. The separate
`build_phase2.bat` compiles **every** model TU with `--default-stream per-thread`.
Do not mix stream policies across translation units. Build with no benchmarks
running, and run GPU experiments serially.

## Inputs and builds (PowerShell)

Use the Phase-1 export/reference instructions in README. Required artifacts are
`weights/gpt2_124m_fp32.bin` and its JSON manifest,
`weights/gpt2_124m_int8_kt.bin`, `refdumps/meta.json`, the associated fp16 oracle
dumps, and `refdumps/wikitext2_val_ids.bin` (at least 1024 IDs). The latter must
start with the exact 512 frozen evaluation IDs; there is no zero-padding or
fallback prompt. The run wrapper verifies the fp32 manifest hash and records
all reference/weight hashes. Do not regenerate a different oracle to pass a test.

```powershell
# Optional if MSVC is installed elsewhere:
# $env:GPT2_VCVARS = 'C:\path\to\VC\Auxiliary\Build\vcvars64.bat'
.\build_cuda.bat
if ($LASTEXITCODE -ne 0) { throw 'Phase-1 build failed' }
.\build_phase2.bat
if ($LASTEXITCODE -ne 0) { throw 'Phase-2 build failed' }
```

Exact equivalence plus the unchanged HF gates:

```powershell
python tools/phase2_run.py docs/phase2/my_exact_fp16 -- bench/graph_gate.exe gemv
python tools/phase2_run.py docs/phase2/my_exact_int8 -- bench/graph_gate.exe int8
$env:GPT2_DECODE = 'graph' # ordinary remains available in the same binary
$env:GPT2_BACKEND = 'gemv'
python tools/phase2_run.py docs/phase2/my_hf_fp16 -- bench/kv_gate_graph.exe
$env:GPT2_BACKEND = 'int8'
python tools/phase2_run.py docs/phase2/my_hf_int8 -- bench/kv_gate_graph.exe
Remove-Item Env:GPT2_DECODE
Remove-Item Env:GPT2_BACKEND
```

Stop on any nonzero exit. The suite below enforces this automatically. Output
directories must be new: the wrapper refuses to overwrite a previous run.

## Optional external baseline, exact input and weight values

Use an existing CUDA llama.cpp build. This iteration used revision
`259f2e2a531af9ed3efa7f66adaa5eb5b53da95f`, with the already documented converter
patch. No external source/binary was modified. The original F16 GGUF stores
907,776 values in fp32. The verifier can create a **new** GGUF with these rounded
through fp16, matching every engine weight value, while retaining fp32 storage
where llama requires it. It checks all 148 tensors against our export before
creating the copy; the source GGUF is never overwritten.

```powershell
$env:LLAMA_DIR = 'E:\llamacpp' # substitute your existing checkout
# Python below needs NumPy; imports gguf-py from LLAMA_DIR.
python tools/phase2_verify_gguf.py $env:LLAMA_DIR "$env:LLAMA_DIR/gpt2-f16.gguf" weights/gpt2_llama_matched.gguf
.\build_llama_phase2.bat
if ($LASTEXITCODE -ne 0) { throw 'External harness build failed' }
```

The custom harness uses the same 1024 WikiText IDs, batch=1 decode, full GPU
offload, context allocation 1024, F16 K/V, flash attention disabled, one CPU
thread, and one full host-visible logits row per step. Decode windows and
discard counts match ours. Graph reuse is enabled by the existing llama build.
No sampling is timed on either side of this comparison. Arithmetic is **not
bit-matched**: llama uses fp32 intermediates/logits; our engine stores fp16
intermediates/logits, with fp32 accumulation. We transfer 100,514 logit bytes
versus llama's 201,028. Report this limitation; a single cross-engine ratio does
not isolate graph overhead. The native unrounded GGUF is a separate control.

## Full serial suite

```powershell
.\tools\phase2_suite.ps1 -Output docs/phase2/my_suite -LlamaDir $env:LLAMA_DIR
# Omit -LlamaDir if llama.cpp is unavailable; this does not block the causal A/B.
```

This runs both exact gates, both HF gates, bandwidth, legacy-stream control,
then three independent processes for each backend, interspersed with llama
when supplied. Each process reverses ordinary/graph order on alternate pairs.
Run receipt, stdout (all CSV samples), stderr and 200-ms GPU telemetry are saved.
Record any unrelated GPU workloads or build overlap; do not silently discard
outliers or choose a favorable run. Large ranges remain in the results.

Individual benchmark: `bench/bench_graph.exe gemv all 256` (or `int8`). Modes:
`fixed`, `growing`, `setup`, `all`, `profile_ordinary`, `profile_graph`.
Publication uses N>=256, a multiple of 16; smaller N is diagnostic only.
The no-update `graph_fixed_diagnostic` is **not** usable autoregressive speed.

`tools/phase2_summary.py <run-directory>` writes medians, min/max and quartiles
for every condition, never best-of-N. Decode CSV columns are event/wall/enqueue/
update milliseconds. **Setup rows reuse those columns** for capture/total/
instantiate/upload respectively. Total setup additionally includes node
discovery and parameter copying. `process_cold` is first graph setup, *after*
model upload/CUDA initialization, not process startup. First replay is timed
after explicit upload, separately from steady state. `fixed_batch16` contains
50 batch averages (800 calls), not 800 independent samples.

## Profiling, separate from speed runs

```powershell
.\tools\phase2_profile.ps1 -Output docs/phase2/my_profile
# Override -Nsys if installed elsewhere.
```

Sixteen diagnostic steps at ctx128/1023, ordinary and dynamic graph; CUDA
Profiler API brackets only the warmed steps. Node-level tracing can perturb
execution. `phase2_timeline.py` verifies exactly 135 serialized kernels per
step, retains all intervals in `kernels.csv`, and reports summed kernel time,
first-to-last span and internal gaps. It does not count synchronization wait
as device work. Raw `.nsys-rep`/SQLite files stay local; CSV/JSON are portable.
Compare profiled event times with unprofiled samples before interpreting them.

Optional long-context attention counters (diagnostic, never a throughput run):

```powershell
& 'C:\Program Files\NVIDIA Corporation\Nsight Compute 2024.3.0\target\windows-desktop-win7-x64\ncu.exe' --profile-from-start off --kernel-name 'regex:k_attn_decode' --launch-count 1 --set full --target-processes all --export docs/phase2/attention_long bench/bench_graph.exe gemv profile_ordinary 16 1023
```

## API ownership

`cuda/decode_graph.cuh` exposes create/step/destroy; prepare/launch are split
only for measurement attribution. One graph borrows immutable weight, scratch,
cache, logit and optional diagnostic-buffer addresses. Destroy it before
releasing any buffer, on its owning host thread/device. No host sampling is
captured. `step(token,pos)` checks token/context bounds and `pos == kv.len`,
updates embedding, append and attention nodes (including shared memory), then
launches. Reset changes `kv.len`; no recapture is needed. A topology change
fails closed rather than silently reusing a stale node map. Runtime CUDA
failures abort through the existing `CUDA_CHECK` convention.

Graph setup is excluded from steady-state speed and must be amortized for a
short generation. Keep ordinary execution when capture cannot be amortized or
borrowed addresses cannot remain stable. No other optimization is stacked here.
