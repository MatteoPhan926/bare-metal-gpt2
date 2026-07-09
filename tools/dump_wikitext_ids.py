#!/usr/bin/env python
r"""dump_wikitext_ids.py  (STAGE 4) — dump the exact token ids eval_ppl.py scores.

Why this exists: eval_ppl.py runs the *HF* model and produced the frozen fp16 baseline PPL = 25.57.
The Stage-4 quality gate needs the SAME text, SAME tokenization, SAME windowing scored against OUR
engine's logits. Rather than edit the frozen baseline harness, this script re-emits only its inputs
(the token ids); bench/eval_ppl_cuda.cu then replicates its windowing/masking loop on the GPU.

The three lines below are copied VERBATIM from eval_ppl.py — if they ever drift, the Δppl stops
being apples-to-apples. The token-count assert is the tripwire (eval_ppl.py reported 249,749).

Run:
  set HF_HOME=E:\gpt2_cache\hf
  .venv\Scripts\python.exe tools\dump_wikitext_ids.py --out refdumps\wikitext2_val_ids.bin
"""
import argparse
import numpy as np

EXPECTED_TOKENS = 249_749          # as printed by eval_ppl.py when it measured PPL(fp16)=25.57


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="refdumps/wikitext2_val_ids.bin")
    ap.add_argument("--model", default="gpt2")
    args = ap.parse_args()

    from transformers import GPT2TokenizerFast
    from datasets import load_dataset

    tok = GPT2TokenizerFast.from_pretrained(args.model)
    # --- verbatim from eval_ppl.py ---
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tok(text, return_tensors="pt").input_ids
    # --- end verbatim ---

    n = ids.size(1)
    print(f"[ids] tokens = {n:,}  (eval_ppl.py reported {EXPECTED_TOKENS:,})")
    assert n == EXPECTED_TOKENS, "token count drifted -> the Δppl would not be apples-to-apples"

    a = ids[0].numpy().astype(np.int32)
    a.tofile(args.out)
    print(f"[ids] wrote {args.out}  ({a.nbytes/1e6:.1f} MB, int32)")


if __name__ == "__main__":
    main()
