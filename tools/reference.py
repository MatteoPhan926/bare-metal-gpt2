#!/usr/bin/env python
r"""reference.py  (Phase 0) — HF gpt2 oracle dumps = the correctness ground truth.

Produces, for BOTH precisions (QUALITY_GATES §1 governing principle):
  fp32 oracle on CPU   (compared against the fp32 C engine)
  fp16 oracle on CUDA  (compared against the fp16 CUDA engine)

Dumps (raw little-endian, row-major) under refdumps/{fp32,fp16}/ :
  embed.bin        [Tp, 768]      hidden after wte+wpe            (gate a)
  block_{i}.bin    [Tp, 768]      hidden after block i (0..11)    (gate a, bug localizer)
  final_ln.bin     [Tp, 768]      hidden after ln_f               (gate a)
  greedy_ids.bin   [Ngen] int32   greedy continuation             (gate b)
  greedy_margin.bin[Ngen] f32     top1-top2 logit margin/step     (gate b tolerance rule)
  eval_logits.bin  [Neval, vocab] final logits over eval window   (gate c: top-1 agree + KL)
meta.json: prompt, ids, dims, dtype per file, offsets.

FROZEN correctness prompt (QUALITY_GATES §3) — do not change once kernels are gated against it.
Run:
  set HF_HOME=E:\gpt2_cache\hf
  .venv\Scripts\python.exe tools\reference.py --out refdumps
"""
import argparse, json, os
import numpy as np

# ---- FROZEN inputs (QUALITY_GATES §3 knob #2) ----
FIXED_PROMPT = ("In a shocking finding, scientist discovered a herd of unicorns living in a remote, "
                "previously unexplored valley, in the Andes Mountains.")
N_GEN   = 128     # greedy tokens (gate b)
N_EVAL  = 512     # eval positions (gate c)
VOCAB   = 50257

def dump(path, arr):
    arr = np.ascontiguousarray(arr)
    arr.tofile(path)
    return {"shape": list(arr.shape), "dtype": str(arr.dtype), "bytes": arr.nbytes}

def run_precision(model_name, prompt_ids, eval_ids, device, dtype, outdir):
    import torch
    from transformers import GPT2LMHeadModel
    os.makedirs(outdir, exist_ok=True)
    m = GPT2LMHeadModel.from_pretrained(model_name).to(device)   # default fp32
    m = (m.half() if dtype == torch.float16 else m.float()).eval()
    npdt = np.float16 if dtype == torch.float16 else np.float32
    files = {}

    # ---- forward over prompt, capture per-block hidden states (gate a) ----
    # HF GPT2Model has ALWAYS recorded the POST-ln_f tensor as hidden_states[-1] (it IS last_hidden_state;
    # ln_f is applied before the final hidden state is appended) -- long-standing behavior, not a 5.x quirk.
    # So hs[i+1] is the pre-ln_f block output only for i=0..10; hs[12] is already ln_f'd. Capture the
    # raw block-11 output and the single ln_f explicitly via a forward hook on ln_f:
    #   block_11 = ln_f INPUT  (pre-ln_f raw block-11 output, consistent with block_0..10)
    #   final_ln = ln_f OUTPUT (single ln_f — NOT ln_f(hs[-1]), which would double-normalize).
    ids = torch.tensor([prompt_ids], device=device)
    capt = {}
    h_pre  = m.transformer.ln_f.register_forward_pre_hook(lambda mod, inp: capt.__setitem__("pre",  inp[0].detach()))
    h_post = m.transformer.ln_f.register_forward_hook(lambda mod, inp, out: capt.__setitem__("post", out.detach()))
    with torch.no_grad():
        out = m(ids, output_hidden_states=True)
        hs = out.hidden_states                 # [0]=embed, [i+1]=output of block i (pre-ln_f) for i=0..10
    h_pre.remove(); h_post.remove()
    pre_lnf, post_lnf = capt["pre"], capt["post"]
    files["embed"] = dump(os.path.join(outdir, "embed.bin"),
                          hs[0][0].to(torch.float32 if dtype==torch.float32 else dtype).cpu().numpy().astype(npdt))
    for i in range(11):                        # blocks 0..10 = hs[1..11] (pre-ln_f, correct as-is)
        files[f"block_{i}"] = dump(os.path.join(outdir, f"block_{i}.bin"),
                                   hs[i+1][0].cpu().numpy().astype(npdt))
    files["block_11"] = dump(os.path.join(outdir, "block_11.bin"),   # raw block-11 output (pre-ln_f)
                             pre_lnf[0].cpu().numpy().astype(npdt))
    files["final_ln"] = dump(os.path.join(outdir, "final_ln.bin"),   # ln_f(block_11), single
                             post_lnf[0].cpu().numpy().astype(npdt))
    # dump-time self-check: the dumped final_ln must equal ln_f(dumped block_11).
    with torch.no_grad():
        chk = m.transformer.ln_f(pre_lnf)
    rel = float((post_lnf - chk).abs().max() / (post_lnf.abs().max() + 1e-9))
    tol = 1e-2 if dtype == torch.float16 else 1e-5
    pname = "fp16" if dtype == torch.float16 else "fp32"
    print(f"[ref:{pname}] self-check final_ln == ln_f(block_11): rel={rel:.2e} tol={tol:.0e} -> "
          f"{'PASS' if rel < tol else 'FAIL'}")

    # ---- greedy decode N_GEN (gate b) + per-step top-2 margin ----
    seq = list(prompt_ids)
    greedy, margins = [], []
    with torch.no_grad():
        cur = torch.tensor([seq], device=device)
        for _ in range(N_GEN):
            logits = m(cur).logits[0, -1].float()     # [vocab]
            top2 = torch.topk(logits, 2).values
            margins.append(float(top2[0] - top2[1]))
            nxt = int(torch.argmax(logits))
            greedy.append(nxt)
            seq.append(nxt)
            cur = torch.tensor([seq], device=device)
    files["greedy_ids"]    = dump(os.path.join(outdir, "greedy_ids.bin"),   np.array(greedy, np.int32))
    files["greedy_margin"] = dump(os.path.join(outdir, "greedy_margin.bin"),np.array(margins, np.float32))

    # ---- eval window: final logits over N_EVAL positions (gate c) ----
    ev = torch.tensor([eval_ids], device=device)
    with torch.no_grad():
        elog = m(ev).logits[0].to(dtype).cpu().numpy().astype(npdt)   # [N_EVAL, vocab]
    files["eval_logits"] = dump(os.path.join(outdir, "eval_logits.bin"), elog)
    return files

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="refdumps")
    ap.add_argument("--model", default="gpt2")
    args = ap.parse_args()
    import torch
    from transformers import GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained(args.model)
    prompt_ids = tok(FIXED_PROMPT)["input_ids"]
    print(f"[ref] prompt = {len(prompt_ids)} tokens: {prompt_ids}")

    # eval window: first N_EVAL tokens of WikiText-2 val (same corpus as eval_ppl.py)
    try:
        from datasets import load_dataset
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation")
        text = "\n\n".join(t for t in ds["text"] if t.strip())
        eval_ids = tok(text)["input_ids"][:N_EVAL]
    except Exception as e:
        print(f"[ref] WARN wikitext load failed ({e}); using prompt-repeat fallback for eval window")
        eval_ids = (prompt_ids * (N_EVAL // len(prompt_ids) + 1))[:N_EVAL]
    assert len(eval_ids) == N_EVAL, f"eval window {len(eval_ids)} != {N_EVAL}"

    os.makedirs(args.out, exist_ok=True)
    meta = {"prompt": FIXED_PROMPT, "prompt_ids": prompt_ids, "n_prompt": len(prompt_ids),
            "n_gen": N_GEN, "n_eval": N_EVAL, "eval_ids": eval_ids, "vocab": VOCAB, "precisions": {}}

    meta["precisions"]["fp32"] = run_precision(args.model, prompt_ids, eval_ids,
                                               "cpu", torch.float32, os.path.join(args.out, "fp32"))
    print("[ref] fp32 (CPU) dumps done")
    if torch.cuda.is_available():
        meta["precisions"]["fp16"] = run_precision(args.model, prompt_ids, eval_ids,
                                                   "cuda", torch.float16, os.path.join(args.out, "fp16"))
        print("[ref] fp16 (CUDA) dumps done")
    else:
        print("[ref] WARN no CUDA -> skipped fp16 oracle")

    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    # human-readable greedy for eyeballing
    print("[ref] fp32 greedy continuation:")
    import torch as _t
    g = np.fromfile(os.path.join(args.out, "fp32", "greedy_ids.bin"), np.int32)
    print("   " + tok.decode(g.tolist()).replace("\n", " "))
    print(f"[ref] wrote {args.out}/meta.json")

if __name__ == "__main__":
    main()
