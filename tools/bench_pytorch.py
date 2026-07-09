#!/usr/bin/env python
r"""bench_pytorch.py — EXTERNAL BASELINE: PyTorch fp16 eager, same GPU / same model / same shapes.

BENCH_PROTOCOL §5 compliance (the honesty crux):
  * MATCHED PRECISION. This script runs fp16 only. Our fp16 engine is compared to this; our INT8 build
    is NEVER compared to it (comparing across precisions is a category error, not a result).
  * eager mode, NOT torch.compile -> the "beat a naive framework" bar (table stakes). torch.compile
    would be a separate, stronger baseline and is not run here.
  * TF32 is DISABLED explicitly. Left on, matmuls would silently use tensor cores at reduced precision
    and the comparison would be neither fp16 nor honest.
  * torch.cuda.synchronize() brackets EVERY timed region (BENCH_PROTOCOL §7 bug #1).
  * median + min/max over N runs after warmup. NEVER best-of-N.
  * PREFILL = one forward over P tokens, no cache.
    DECODE  = ONE token appended to a KV cache of length ctx == exactly what our Stage-5 decode step
              does. Both engines have a KV cache, so this is true M=1 apples-to-apples.

Run:
  set HF_HOME=E:\gpt2_cache\hf
  .venv\Scripts\python.exe tools\bench_pytorch.py
"""
import argparse, json, statistics, time
import numpy as np


def sync_time(fn, n, warmup, torch):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()          # time EXECUTION, not launch
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return statistics.median(ts), ts[0], ts[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()

    import torch
    from transformers import GPT2LMHeadModel

    # TF32 off: otherwise fp32 matmuls silently use tensor cores; and cudnn autotune off for determinism.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.fp32_precision = "ieee" if hasattr(torch.backends.cuda.matmul, "fp32_precision") else None

    dev = "cuda"
    model = GPT2LMHeadModel.from_pretrained(args.model).half().eval().to(dev)
    print(f"[env] torch {torch.__version__}  device {torch.cuda.get_device_name(0)}")
    print(f"[env] dtype fp16, eager (NOT torch.compile), TF32 disabled")
    print(f"[env] warmup={args.warmup} iters={args.iters}  median + [min,max], never best-of-N\n")

    g = torch.Generator(device=dev).manual_seed(1234)
    out = {"prefill": {}, "decode": {}}

    # ---------------- PREFILL: one forward over P tokens, no cache ----------------
    # APPLES-TO-APPLES TRAP: HF's model(ids) runs the LM head over ALL P positions
    # (2*P*50257*768 FLOP = 39.5 GFLOP at P=512). Our engine's timed prefill computes logits for the
    # LAST token only. So `full` is NOT comparable to ours; `trunk` (transformer only, no head) plus
    # our own 1-row head is. Both are measured and both are reported.
    print("---- PREFILL (one forward over P tokens; use_cache=False) ----")
    print("    trunk = transformer only (no LM head)  <- comparable to our engine's timed prefill")
    print("    full  = trunk + LM head over ALL P positions (what model(ids) does)  <- NOT comparable\n")
    for P in (128, 512):
        ids = torch.randint(0, 50257, (1, P), device=dev, generator=g)
        def f_full():
            with torch.no_grad():
                model(ids, use_cache=False)
        def f_trunk():
            with torch.no_grad():
                model.transformer(ids, use_cache=False)
        mt, lt, ht = sync_time(f_trunk, args.iters, args.warmup, torch)
        mf, lf, hf = sync_time(f_full,  args.iters, args.warmup, torch)
        out["prefill"][P] = {"trunk": (mt, lt, ht), "full": (mf, lf, hf)}
        print(f"  prefill @P={P:<4d}: trunk {mt:8.3f} ms [{lt:.3f}-{ht:.3f}]   full {mf:8.3f} ms [{lf:.3f}-{hf:.3f}]")

    # ---------------- DECODE: ONE token against a cache of length ctx ----------------
    print("\n---- DECODE (true M=1: one token appended to a KV cache of length ctx) ----")
    for ctx in (128, 512, 1023):
        ids = torch.randint(0, 50257, (1, ctx), device=dev, generator=g)
        with torch.no_grad():
            pre = model(ids, use_cache=True)
        past = pre.past_key_values
        nxt = torch.randint(0, 50257, (1, 1), device=dev, generator=g)

        def f():
            with torch.no_grad():
                # pass a fresh copy of the cache object each call is unnecessary: HF appends to a copy
                # only when it must grow; we re-slice below to keep length fixed at ctx.
                model(nxt, past_key_values=past, use_cache=True)

        # keep the cache length pinned at ctx: crop after each call so the Nth timed step is identical
        # to the 1st (otherwise we would be timing a cache that grows under us).
        def f_pinned():
            with torch.no_grad():
                model(nxt, past_key_values=past, use_cache=True)
            try:
                past.crop(ctx)
            except Exception:
                pass

        med, lo, hi = sync_time(f_pinned, args.iters, args.warmup, torch)
        out["decode"][ctx] = (med, lo, hi)
        print(f"  decode @ctx={ctx:<4d}: {med:8.4f} ms/token median  [{lo:.4f} - {hi:.4f}]  -> {1000.0/med:7.1f} tok/s")

    # ---------------- bug-line ----------------
    print("\n---- ROOFLINE bug-line (ROOFLINE §6): fp16 weights 248.9 MB ----")
    for ctx, (med, _, _) in out["decode"].items():
        tps = 1000.0 / med
        kv_mb = 2 * 12 * 12 * ctx * 64 * 2 / 1e6
        ceil_copy = 233.4e3 / (248.9 + kv_mb)
        flag = "  *** ABOVE COPY-BW CEILING -> suspect ***" if tps > ceil_copy else ""
        print(f"  ctx={ctx:<4d} {tps:7.1f} tok/s   ceiling(+KV) {ceil_copy:6.0f}   {100*tps/ceil_copy:5.1f}% of ceiling{flag}")

    json.dump({k: {str(a): b for a, b in v.items()} for k, v in out.items()},
              open("bench/pytorch_baseline.json", "w"), indent=1)
    print("\n[ok] wrote bench/pytorch_baseline.json")


if __name__ == "__main__":
    main()
