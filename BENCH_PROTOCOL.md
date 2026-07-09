# BENCH_PROTOCOL.md — Pre-registered benchmark protocol
### GPT-2-124M · RTX 4060 Laptop (105W) · companion to DESIGN.md + ROOFLINE.md

> **What this file is.** Pre-registers **how** every number is measured, *before* anything is measured —
> so the protocol cannot drift toward flattering numbers (DESIGN.md §0, firewall 2). Three files, three
> jobs: **this file = how to measure**; **ROOFLINE.md = the ceiling to check against**;
> **BENCHMARKS.md = the recorded results**, one block per validated run.

---

## 1. Metrics we report (locked)

- **Prefill:** latency (ms) for a **fixed** prompt length P — reported *per P*. Optionally
  prefill tok/s = P / latency (but the latency is the honest primary; "prefill tok/s" flatters).
- **Decode:** steady-state per-token time, batch=1 → tok/s, **reported at a stated context length**.
- **TTFT** (optional): prefill_latency + first decode step.
- **Weight-load time:** reported **once, separately, EXCLUDED from throughput.**
- **Always prefill and decode separately.** Never a single blended "tok/s" — they are different
  regimes (ROOFLINE §3–4).

---

## 2. Timing boundary — state included/excluded on every number

**Included in throughput:** the forward-pass kernels for the tokens in question.

**Excluded from throughput** (reported separately if at all):
- **Tokenization** (BPE encode/decode) — CPU string work, not the engine.
- **Weight load / weight H2D** — one-time setup.
- **Warmup iterations.**

**Included in the decode loop** (they happen per token): single-token embed lookup, per-step kernels,
sampling.

**Sampling:** **greedy/argmax** for all speed runs — deterministic, negligible time, and it enables
the token-match correctness gate (DESIGN.md §6). Sampling-based generation is not timed.

---

## 3. Isolating prefill vs decode (methodology)

**Prefill:** one forward pass over P tokens; device-sync around it; repeat ~30×; report median±spread.
Fixed **P ∈ {128, 512}** *(suggested defaults — override if the portfolio story needs other lengths).*

**Decode:** after prefill, generate **N tokens** autoregressively; **discard the first ~5 steps**
(allocation / cold), take the **median per-token time** over the rest; tok/s = 1 / median.
**N = 256** *(suggested).*
- **Report decode at a STATED ctx position** — KV traffic grows with context (ROOFLINE §3).
  e.g. "decode @ ctx=512", not a bare tok/s.
- **Before ladder stage 5 (no KV cache):** decode recomputes all past K/V → artificially slow.
  This is **EXPECTED** — label it **"no-KV decode."** The stage-5 win is measured as **no-KV vs KV**
  under this same protocol, not against an external number.
  > **`[Sharpened after Stage 2]`** The label is too weak and has caused two near-misreadings.
  > Without a KV cache the harness **full-recomputes the whole sequence every step**, so its GEMMs run at
  > **M = context length (34–161), not M = 1.** That is the *prefill-shaped, reuse-bound* regime. Use
  > **"no-KV recompute-decode (M≫1)"**. Consequences, both load-bearing: (i) a GEMM/reuse win here is
  > EXPECTED and is **not** a director-map violation (Stage 2's 5.4×); (ii) a **true (M=1) decode does not
  > exist in this engine until Stage 5**, so any row of the director map predicting a *decode* payoff —
  > row 2's "flat", row 4's "high" — is **untestable, hence untested, before then.** Never report this
  > number as decode tok/s without the label.

---

## 4. Warmup + clocks — laptop-specific, load-bearing

- **Warmup:** discard the first **~10 iterations** (JIT, cold cache, allocator, clock spin-up).
- **Clocks:** prefer **locking** (`nvidia-smi --lock-gpu-clocks=<val>` if permitted) for
  reproducibility; **else record the sustained clock** during the run.
- **LAPTOP THERMAL (this machine):** a long run downclocks as the chip heats. Either **lock clocks**,
  or run to **thermal steady-state and report the *sustained* number**, not the cold-start burst.
  A cold-start burst number is not reproducible → **not a result.** Record thermal state in
  BENCHMARKS.md.

---

## 5. Apples-to-apples with baselines (the honesty crux)

Same model, same seq len, same batch (=1), same generated-token count, same GPU, all warmed,
clocks noted.

**Precision matching — the trap and the rule:**

| Ours | Compare against | **NOT** against |
|---|---|---|
| fp16 | PyTorch fp16 eager; llama.cpp **F16** | llama.cpp Q4 / Q8 |
| INT8 | llama.cpp **Q8_0** (closest) | llama.cpp Q4; PyTorch fp16 |
| INT4 | llama.cpp **Q4_0** (closest) | anything higher-precision |

- **Comparing across precisions is a category error, not a result.** A fp16-vs-Q4 "win" means nothing.
- **PyTorch:** **eager mode** = the "beat a naive framework" bar (table stakes). `torch.compile`, if
  run, is a **separate, stronger** baseline — state which. Disable/note **TF32**. Put
  **`torch.cuda.synchronize()` around every timed region** — else you time async *launch*, not
  *execution* (the #1 CUDA benchmark bug, source of fake 100× numbers).
- **llama.cpp:** state build (CUDA backend) + quant + same GPU.
  > **`[VERIFY RESOLVED]`** Mainline llama.cpp *does* run GPT-2 (`b1-259f2e2`, CUDA backend), but the
  > HF→GGUF converter has a real bug: `conversion/gpt2.py` means to drop the legacy causal-mask buffers
  > and instead hands them to `map_tensor_name()`, raising `Can not map tensor 'h.0.attn.bias'`. The
  > one-line fix is kept at `tools/llamacpp_gpt2_convert.patch`. Two further traps: `llama-cli` is now a
  > chat front-end that injects a ChatML template (use **`llama-completion`** for a raw greedy sanity
  > check), and `llama-bench` prints **mean ± stddev**, not the median this protocol requires — use
  > `-o json` and recompute the median from `samples_ns`. Results in BENCHMARKS.md.
- **Framing:** beating PyTorch eager is table stakes; *approaching* llama.cpp is the real signal.
  Position claims accordingly — never imply parity you didn't measure at matched precision.

---

## 6. Statistics — report format (locked)

- **Median + spread** (IQR or min–max). **NEVER best-of-N** (cherry-picking).
- **Fixed prompt + fixed seed**, both recorded.
- **N:** enough for a stable median (decode: N ≥ 256 per-token samples; prefill: ≥ 30 runs).
  If spread is wide → **increase N, don't pick the good one.**
- **Every number ships with its full config:** model, precision, P, ctx, batch, N, seed, GPU clock,
  thermal state, and whether tokenization/load are included. **No number without its config.**

---

## 7. Measurement-bug checklist — run before believing ANY number

These silently **inflate** numbers. A number that trips one of these is a **bug, not a result**:

- [ ] No device sync (`cudaDeviceSynchronize` / `torch.cuda.synchronize`) around timing → timing async
      *launch*, not *execution*. **(most common)**
- [ ] No warmup → cold-start (alloc / JIT / cold cache / clock spin-up) pollutes the number.
- [ ] Counting prefill / cached tokens as "generated" → inflates decode tok/s.
- [ ] Best-of-N instead of median.
- [ ] Mismatched precision vs baseline.
- [ ] Cold-start burst clock on a laptop reported as sustained → not reproducible.
- [ ] Number **above** the ROOFLINE.md ceiling → a real bug is hiding. **Fix it; don't publish it.**

---

## 8. Reporting template — paste one block per validated run into BENCHMARKS.md

```
stage:            e.g. "2 — tiled GEMM"
kernel/precision: e.g. "tiled fp16 (fp32 accum)"
config:           P=512  ctx=512  batch=1  N=256  seed=1234
environment:      clock=<MHz> (locked? y/n)  thermal=<steady/burst>  driver=<ver>
included:         tokenization=n  weight_load=n
prefill:          latency <ms> median ± <spread>   (ceiling: <ms> from ROOFLINE)
decode @ ctx=512: <tok/s> median ± <spread>        (ceiling: <tok/s>; % of ceiling: <..>)
baseline:         <name> <precision> → prefill <ms> / decode <tok/s>   (matched precision: y)
notes:            anything that would change how the number is read
```
