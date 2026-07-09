#!/usr/bin/env python
r"""dequant_check.py  (STAGE 4) — is the INT8 GEMM *correct*, or is the error *quantization*?

Those two demand opposite responses (fix the kernel vs. report honest degradation), so they must be
separated before any verdict. This script writes a normal fp32 weight file in which the quantized
tensors are replaced by their DEQUANTIZED values q[n,k]*s[n] (everything else byte-identical).

Then:
    GPT2_BACKEND=flash  correctness_cuda.exe all weights/gpt2_124m_deq_fp32.bin
runs the *fp16* engine on the dequantized weights. If the INT8 backend's gate numbers match this to
fp16 noise, then k_matmul_int8_tiled computes exactly what "fp16 GEMM on dequantized weights" computes
-> the dequant path is sane and every remaining gate delta is quantization error, not a kernel bug.

--subset also lets the same trick LOCALIZE the error to a tensor group, with no kernel change at all,
because quantization is just a perturbation of the weights:
    all     : every quantized tensor  (== the INT8 backend)
    head    : only the tied output head (wte)
    blocks  : only the 48 per-layer matmul weights
    mlpproj : only the 12 mlp.c_proj weights (the ones with ~2x the per-row rms error)

Run:
  .venv\Scripts\python.exe tools\dequant_check.py --subset all --out weights\gpt2_124m_deq_fp32.bin
"""
import argparse, json, shutil
import numpy as np

N_LAYER, N_EMBD, VOCAB = 12, 768, 50257
FFN, QKV = 4 * N_EMBD, 3 * N_EMBD


def qnames(subset):
    head = ["wte"]
    blocks, mlpproj = [], []
    for L in range(N_LAYER):
        blocks += [f"h{L}.attn.c_attn.w", f"h{L}.attn.c_proj.w", f"h{L}.mlp.c_fc.w", f"h{L}.mlp.c_proj.w"]
        mlpproj += [f"h{L}.mlp.c_proj.w"]
    return {"all": head + blocks, "head": head, "blocks": blocks, "mlpproj": mlpproj}[subset]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="weights/gpt2_124m_fp32.bin")
    ap.add_argument("--out", default="weights/gpt2_124m_deq_fp32.bin")
    ap.add_argument("--subset", default="all", choices=["all", "head", "blocks", "mlpproj"])
    args = ap.parse_args()

    man = json.load(open(args.src + ".json"))
    tmap = {t["name"]: t for t in man["tensors"]}
    want = set(qnames(args.subset))

    shutil.copyfile(args.src, args.out)                     # start from a byte-identical copy
    shutil.copyfile(args.src + ".json", args.out + ".json")

    raw = open(args.src, "rb").read()
    n_done, worst = 0, ("", 0.0)
    with open(args.out, "r+b") as fp:
        for name in sorted(want):
            t = tmap[name]
            shape = t["shape"]
            w = np.frombuffer(raw, dtype=np.float32, count=int(np.prod(shape)), offset=t["offset"]).reshape(shape)
            w16 = w.astype(np.float16).astype(np.float32)    # the fp16 values the engine runs
            absmax = np.abs(w16).max(axis=1)
            s = np.where(absmax == 0.0, 1.0, absmax / 127.0).astype(np.float32)
            q = np.clip(np.rint(w16 / s[:, None]), -127, 127).astype(np.int8)
            deq = (q.astype(np.float32) * s[:, None]).astype(np.float32)
            fp.seek(t["offset"]); fp.write(deq.tobytes())
            n_done += 1
            r = float(np.abs(deq - w16).max() / (np.abs(w16).max() + 1e-9))
            if r > worst[1]:
                worst = (name, r)
    print(f"[deq] subset={args.subset}: replaced {n_done} tensors with q*s in {args.out}")
    print(f"[deq] worst per-tensor rel(maxnorm) = {worst[0]} {worst[1]:.5f}")
    print(f"[deq] now run:  set GPT2_BACKEND=flash && bench\\correctness_cuda.exe all {args.out}")


if __name__ == "__main__":
    main()
