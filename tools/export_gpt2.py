#!/usr/bin/env python
r"""export_gpt2.py  (Phase 0)

HF `gpt2` (124M) -> single fp32 raw weight file + JSON manifest, in a FIXED binary layout.

Layout convention (LOCKED — mirrored in model/weights.h):
  * ALL block linear weights are TRANSPOSED from HF Conv1D [in,out] to [out,in].
    => the whole engine uses ONE matmul form  C = A * B^T :  out[n] = sum_k A[k]*B[n,k].
  * wte is stored ONCE as [vocab, n_embd]; it is BOTH the token embedding (row lookup)
    AND the tied output head (consumed in the same C=A*B^T form). Not duplicated.
  * fp32 master. fp16 GPU weights are a deterministic .half() cast at load (matches HF fp16 oracle).

Binary file:
  header: 16 x int32 little-endian  (magic, version, dims, dtype, reserved)
  body  : tensors in the exact order emitted below, float32, row-major, tightly packed.

Run (from repo root, venv on E:):
  set HF_HOME=E:\gpt2_cache\hf
  .venv\Scripts\python.exe tools\export_gpt2.py --out weights\gpt2_124m_fp32.bin
"""
import argparse, json, os, struct, sys, hashlib
import numpy as np

MAGIC   = 0x47505432   # 'GPT2'
VERSION = 1
# locked dims (must match model/config.h)
N_LAYER, N_HEAD, N_EMBD, VOCAB, N_CTX = 12, 12, 768, 50257, 1024
FFN, QKV = 4*N_EMBD, 3*N_EMBD           # 3072, 2304
EXPECTED_PARAMS = 124_439_808

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="weights/gpt2_124m_fp32.bin")
    ap.add_argument("--model", default="gpt2")
    args = ap.parse_args()

    import torch
    from transformers import GPT2LMHeadModel
    print(f"[export] loading {args.model} (fp32) ; HF_HOME={os.environ.get('HF_HOME')}")
    model = GPT2LMHeadModel.from_pretrained(args.model).float().eval()  # default load = fp32
    sd = model.state_dict()
    cfg = model.config
    assert (cfg.n_layer, cfg.n_head, cfg.n_embd, cfg.vocab_size, cfg.n_ctx) == \
           (N_LAYER, N_HEAD, N_EMBD, VOCAB, N_CTX), f"config mismatch: {cfg}"

    # tied-embedding sanity: lm_head.weight is the same data as wte.weight
    wte = sd["transformer.wte.weight"]
    lm  = sd["lm_head.weight"]
    assert torch.equal(wte, lm), "lm_head is NOT tied to wte — layout assumption broken"
    print("[export] tied-embedding check OK (lm_head == wte)")

    def f32(t):  # -> contiguous float32 numpy
        return np.ascontiguousarray(t.detach().cpu().float().numpy(), dtype=np.float32)
    def linT(name):  # Conv1D [in,out] -> [out,in]
        w = sd[name]                      # [in,out]
        return f32(w.t().contiguous())    # [out,in]

    tensors = []  # (name, shape, ndarray)
    def add(name, arr, shape):
        assert arr.shape == tuple(shape), f"{name}: {arr.shape} != {shape}"
        tensors.append((name, list(shape), arr))

    add("wte", f32(wte), [VOCAB, N_EMBD])
    add("wpe", f32(sd["transformer.wpe.weight"]), [N_CTX, N_EMBD])
    for L in range(N_LAYER):
        p = f"transformer.h.{L}."
        add(f"h{L}.ln_1.g", f32(sd[p+"ln_1.weight"]), [N_EMBD])
        add(f"h{L}.ln_1.b", f32(sd[p+"ln_1.bias"]),   [N_EMBD])
        add(f"h{L}.attn.c_attn.w", linT(p+"attn.c_attn.weight"), [QKV, N_EMBD])
        add(f"h{L}.attn.c_attn.b", f32(sd[p+"attn.c_attn.bias"]), [QKV])
        add(f"h{L}.attn.c_proj.w", linT(p+"attn.c_proj.weight"), [N_EMBD, N_EMBD])
        add(f"h{L}.attn.c_proj.b", f32(sd[p+"attn.c_proj.bias"]), [N_EMBD])
        add(f"h{L}.ln_2.g", f32(sd[p+"ln_2.weight"]), [N_EMBD])
        add(f"h{L}.ln_2.b", f32(sd[p+"ln_2.bias"]),   [N_EMBD])
        add(f"h{L}.mlp.c_fc.w",   linT(p+"mlp.c_fc.weight"),   [FFN, N_EMBD])
        add(f"h{L}.mlp.c_fc.b",   f32(sd[p+"mlp.c_fc.bias"]),  [FFN])
        add(f"h{L}.mlp.c_proj.w", linT(p+"mlp.c_proj.weight"), [N_EMBD, FFN])
        add(f"h{L}.mlp.c_proj.b", f32(sd[p+"mlp.c_proj.bias"]),[N_EMBD])
    add("ln_f.g", f32(sd["transformer.ln_f.weight"]), [N_EMBD])
    add("ln_f.b", f32(sd["transformer.ln_f.bias"]),   [N_EMBD])

    total = sum(int(np.prod(s)) for _, s, _ in tensors)
    print(f"[export] param count = {total:,}  (expected {EXPECTED_PARAMS:,})")
    assert total == EXPECTED_PARAMS, "PARAM COUNT MISMATCH -> layout bug"

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    header = struct.pack("<16i", MAGIC, VERSION, N_LAYER, N_HEAD, N_EMBD, VOCAB,
                         N_CTX, FFN, QKV, 0, 0, 0, 0, 0, 0, 0)  # dtype=0 fp32
    manifest = {"magic": MAGIC, "version": VERSION,
                "dims": {"n_layer":N_LAYER,"n_head":N_HEAD,"n_embd":N_EMBD,
                         "vocab":VOCAB,"n_ctx":N_CTX,"ffn":FFN,"qkv":QKV},
                "dtype":"fp32","header_bytes":len(header),"total_params":total,
                "tensors":[]}
    h = hashlib.sha256()
    off = len(header)
    with open(args.out, "wb") as fp:
        fp.write(header)
        for name, shape, arr in tensors:
            b = arr.tobytes()
            fp.write(b); h.update(b)
            manifest["tensors"].append({"name":name,"shape":shape,
                                        "offset":off,"bytes":len(b)})
            off += len(b)
    manifest["file_bytes"] = off
    manifest["sha256"] = h.hexdigest()
    with open(args.out + ".json", "w") as fp:
        json.dump(manifest, fp, indent=1)
    print(f"[export] wrote {args.out}  ({off/1e6:.1f} MB, {len(tensors)} tensors)")
    print(f"[export] sha256(body)={manifest['sha256'][:16]}...  manifest -> {args.out}.json")

if __name__ == "__main__":
    main()
