#!/usr/bin/env python
r"""eval_ppl.py  (Phase 0) — WikiText-2 perplexity harness (QUALITY_GATES §2).

Canonical HF sliding-window PPL: window=1024, stride=512, on WikiText-2-raw validation.
Phase-0 job: establish the fp16 baseline PPL and CONFIRM THE HARNESS (must land ≈29–30 for
GPT-2-124M; if far off, the harness is buggy independent of any quantization).

Later (stage 4): the SAME harness scores the quantized engine's logits — Δppl vs this fp16
baseline is the quality gate (INT8 ≤ +0.3, INT4 ≤ +1.0). Do NOT compare to a published number.

Run:
  set HF_HOME=E:\gpt2_cache\hf
  .venv\Scripts\python.exe tools\eval_ppl.py --dtype fp16
"""
import argparse, math
import numpy as np

WINDOW, STRIDE = 1024, 512

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    ap.add_argument("--limit", type=int, default=0, help="cap tokens for a quick smoke (0=all)")
    ap.add_argument("--window", type=int, default=1024, help="sliding-window length (pre-registered 1024)")
    ap.add_argument("--stride", type=int, default=512,
                    help="stride: 512 = pre-registered overlap (the gate); 1024 = non-overlap convention")
    args = ap.parse_args()
    WINDOW, STRIDE = args.window, args.stride   # defaults mirror the pre-registered 1024/512; --stride is a one-off knob
    import torch
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    from datasets import load_dataset

    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    device = "cuda" if (torch.cuda.is_available() and args.dtype == "fp16") else "cpu"
    tok = GPT2TokenizerFast.from_pretrained(args.model)
    model = GPT2LMHeadModel.from_pretrained(args.model).to(device)   # default fp32
    model = (model.half() if dtype == torch.float16 else model.float()).eval()

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tok(text, return_tensors="pt").input_ids.to(device)
    if args.limit:
        ids = ids[:, :args.limit]
    seq_len = ids.size(1)
    print(f"[ppl] dtype={args.dtype} device={device}  tokens={seq_len}  window={WINDOW} stride={STRIDE}")

    nll_sum, n_tokens, prev_end = 0.0, 0, 0
    for begin in range(0, seq_len, STRIDE):
        end = min(begin + WINDOW, seq_len)
        trg = end - prev_end                       # newly-scored tokens this window
        inp = ids[:, begin:end]
        tgt = inp.clone(); tgt[:, :-trg] = -100     # only score the new tokens
        with torch.no_grad():
            loss = model(inp, labels=tgt).loss      # mean NLL over scored tokens
        nll_sum += float(loss) * trg
        n_tokens += trg
        prev_end = end
        if end == seq_len:
            break
    ppl = math.exp(nll_sum / n_tokens)
    print(f"[ppl] scored_tokens={n_tokens}  mean_nll={nll_sum/n_tokens:.4f}")
    print(f"[ppl] PPL({args.dtype}) = {ppl:.3f}   (sanity: GPT-2-124M WikiText-2 ~= 29-30)")
    print(f"PPL_{args.dtype.upper()}={ppl:.4f}")

if __name__ == "__main__":
    main()
