#!/usr/bin/env python
r"""quantize.py  (STAGE 4) — fp16 -> INT8 packed weights + scales.

SCHEME (pre-registered in QUALITY_GATES §2 — NOT a knob):
  symmetric, PER-CHANNEL, WEIGHT-ONLY INT8.

  For each quantized weight W[N,K] (rows = output channels, the engine's [out,in] layout):
      s[n] = max_k |W[n,k]| / 127            (per output channel n; symmetric -> no zero-point)
      q[n,k] = clip(round(W[n,k] / s[n]), -127, 127)          # int8, |q| <= 127
      W[n,k] ~= q[n,k] * s[n]

  Per-CHANNEL over rows is the axis that lets the scale leave the k-sum:
      C[m,n] = sum_k A[m,k] * W[n,k] + b[n]  ~=  s[n] * (sum_k A[m,k] * q[n,k]) + b[n]
  so the GPU kernel accumulates a scale-free dot product and applies s[n] ONCE at the end.
  (-127 not -128: symmetric range, so the negative tail cannot alias to a magnitude the positive
  side cannot represent.)

QUANTIZED (the 5 weight-matmul kinds = the ~124M params ROOFLINE §2 counts as decode traffic):
    wte (the TIED OUTPUT HEAD)  +  per layer: c_attn.w, c_proj.w (attn), c_fc.w, c_proj.w (mlp)
NOT QUANTIZED (kept fp16; ~0.1 MB, and none of them is a matmul weight):
    all biases, ln_1/ln_2/ln_f gains+biases, wpe, and wte-as-EMBEDDING (a row gather, not a matmul —
    the engine keeps the fp16 wte resident for the embed lookup; only the HEAD reads the int8 copy).

SOURCE PRECISION: we quantize the fp16 *values* the engine actually runs (fp32 master -> .half() ->
back to fp32), not the fp32 master. So Δppl measures quantization alone, with fp16 rounding held fixed.

--fp16-keep implements the PRE-REGISTERED kill-test (QUALITY_GATES §2 / CLAUDE.md §5): "exceeds bound
-> go per-channel (if not already), or keep sensitive layers (first block, final block, or
high-activation-range layers) in fp16." It does NOT touch any threshold. A tensor named here is left
un-quantized: the file carries a flag=0 for it, no int8 bytes are emitted, and the engine's
gpt2_matmul_dispatch falls back to the fp16 tiled GEMM for that tensor alone.

  Groups:  head (the tied output head, wte) | block0 | block11 | mlpproj (all 12 mlp.c_proj) | none

Binary layout (little-endian), mirrored by gpt2_quant_load_upload() in cuda/kernels_quant.cu:
  header: 16 x int32  (magic 'GQ8\0', version=2, n_layer, n_embd, vocab, ffn, qkv, n_tensors, 0...)
  flags : n_tensors x int32   (1 = quantized, 0 = keep fp16 -> no q/s bytes for it)
  body  : for each QUANTIZED tensor, in the FIXED order below:  int8 q[N*K]  then  float32 s[N]
  order : wte, then for L in 0..11: qkv_w, attn_proj_w, fc_w, proj_w

Run (from repo root):
  .venv\Scripts\python.exe tools\quantize.py --out weights\gpt2_124m_int8.bin
  .venv\Scripts\python.exe tools\quantize.py --fp16-keep head --out weights\gpt2_124m_int8_kt.bin
"""
import argparse, json, struct, os, hashlib
import numpy as np

MAGIC_IN  = 0x47505432          # 'GPT2' (export_gpt2.py)
MAGIC_OUT = 0x38515047          # 'GPQ8'
VERSION   = 2                   # v2 adds the per-tensor fp16-keep flags
N_LAYER, N_HEAD, N_EMBD, VOCAB, N_CTX = 12, 12, 768, 50257, 1024
FFN, QKV = 4 * N_EMBD, 3 * N_EMBD

# Achieved-BW denominators (ROOFLINE §6), used to report the decode ceiling this mix implies.
BW_COPY_GBS, BW_READ_GBS, BW_THEO_GBS = 233.4, 248.9, 256.0


def quantize_rows(w32):
    """w32: [N,K] float32 (already fp16-rounded). -> q int8 [N,K], s float32 [N], and error stats."""
    absmax = np.abs(w32).max(axis=1)                       # [N]
    dead = absmax == 0.0                                   # an all-zero row: keep s=1, q=0 (exact)
    s = np.where(dead, 1.0, absmax / 127.0).astype(np.float32)
    q = np.rint(w32 / s[:, None])
    q = np.clip(q, -127, 127).astype(np.int8)
    deq = q.astype(np.float32) * s[:, None]
    err = np.abs(deq - w32)
    denom = np.abs(w32).max() + 1e-9
    return q, s, {
        "max_abs_err": float(err.max()),
        "rel_err_maxnorm": float(err.max() / denom),       # same shape as the gate-(a) metric
        "rms_rel_err": float(np.sqrt((err ** 2).mean()) / (np.sqrt((w32 ** 2).mean()) + 1e-12)),
        "dead_rows": int(dead.sum()),
    }


def keep_set(spec):
    """--fp16-keep groups -> the set of tensor names left in fp16 (the pre-registered kill-test)."""
    keep = set()
    for g in [x.strip() for x in spec.split(",") if x.strip() and x.strip() != "none"]:
        if   g == "head":    keep.add("wte")
        elif g == "block0":  keep |= {f"h0.attn.c_attn.w", "h0.attn.c_proj.w", "h0.mlp.c_fc.w", "h0.mlp.c_proj.w"}
        elif g == "block11": keep |= {f"h11.attn.c_attn.w", "h11.attn.c_proj.w", "h11.mlp.c_fc.w", "h11.mlp.c_proj.w"}
        elif g == "mlpproj": keep |= {f"h{L}.mlp.c_proj.w" for L in range(N_LAYER)}
        else: raise SystemExit(f"unknown --fp16-keep group: {g}")
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="weights/gpt2_124m_fp32.bin")
    ap.add_argument("--out", default="weights/gpt2_124m_int8.bin")
    ap.add_argument("--fp16-keep", default="none",
                    help="comma list of groups kept in fp16 (kill-test): head,block0,block11,mlpproj,none")
    args = ap.parse_args()
    keep = keep_set(args.fp16_keep)

    manifest_in = json.load(open(args.inp + ".json"))
    tmap = {t["name"]: t for t in manifest_in["tensors"]}
    raw = open(args.inp, "rb").read()

    def get(name, shape):
        t = tmap[name]
        a = np.frombuffer(raw, dtype=np.float32, count=int(np.prod(shape)), offset=t["offset"])
        assert list(shape) == t["shape"], f"{name}: {shape} != {t['shape']}"
        return a.reshape(shape)

    # The engine runs fp16 weights (__float2half of the fp32 master == torch .half()). Quantize THOSE.
    def fp16_view(a):
        return a.astype(np.float16).astype(np.float32)

    order = [("wte", [VOCAB, N_EMBD])]
    for L in range(N_LAYER):
        order += [(f"h{L}.attn.c_attn.w", [QKV, N_EMBD]),
                  (f"h{L}.attn.c_proj.w", [N_EMBD, N_EMBD]),
                  (f"h{L}.mlp.c_fc.w",    [FFN, N_EMBD]),
                  (f"h{L}.mlp.c_proj.w",  [N_EMBD, FFN])]

    print(f"[quant] scheme = symmetric, per-channel (rows), weight-only INT8")
    print(f"[quant] source = fp16 view of {args.inp}  ({len(order)} tensors)")
    print(f"[quant] fp16-keep = {args.fp16_keep}" + (f"  -> {sorted(keep)}" if keep else ""))

    blobs, flags, meta, worst = [], [], [], ("", 0.0)
    q_params = kept_params = 0
    for name, shape in order:
        n_el = int(np.prod(shape))
        if name in keep:                                    # kill-test: this tensor stays fp16
            blobs.append(None); flags.append(0); kept_params += n_el
            meta.append({"name": name, "shape": shape, "quantized": False})
            print(f"  {name:24s} {str(shape):16s} -- KEPT fp16 (kill-test)")
            continue
        w = fp16_view(get(name, shape))
        q, s, st = quantize_rows(w)
        blobs.append((q, s)); flags.append(1); q_params += n_el
        meta.append({"name": name, "shape": shape, "quantized": True, **st})
        if st["rms_rel_err"] > worst[1]:
            worst = (name, st["rms_rel_err"])
        print(f"  {name:24s} {str(shape):16s} rms_rel={st['rms_rel_err']:.5f} "
              f"rel_maxnorm={st['rel_err_maxnorm']:.5f} dead={st['dead_rows']}")

    print(f"[quant] worst rms_rel_err: {worst[0]} = {worst[1]:.5f}" if worst[0] else "[quant] nothing quantized")
    # Weight bytes STREAMED per decode token (ROOFLINE §2 counts exactly these tensors).
    streamed = q_params * 1 + kept_params * 2
    fp16_bytes = (q_params + kept_params) * 2
    print(f"[quant] quantized params = {q_params:,}   kept-fp16 params = {kept_params:,}")
    print(f"[quant] streamed weight bytes = {streamed/1e6:.1f} MB  (fp16 would be {fp16_bytes/1e6:.1f} MB"
          f" -> {fp16_bytes/streamed:.2f}x fewer bytes)")
    print(f"[quant] implied decode ceiling: {1e3*BW_COPY_GBS/(streamed/1e6):.0f} tok/s (copy BW) / "
          f"{1e3*BW_READ_GBS/(streamed/1e6):.0f} (read BW) / {1e3*BW_THEO_GBS/(streamed/1e6):.0f} (theo BUG-LINE)")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    header = struct.pack("<16i", MAGIC_OUT, VERSION, N_LAYER, N_EMBD, VOCAB, FFN, QKV,
                         len(order), 0, 0, 0, 0, 0, 0, 0, 0)
    h = hashlib.sha256()
    with open(args.out, "wb") as fp:
        fp.write(header)
        fp.write(np.array(flags, dtype=np.int32).tobytes())
        for b in blobs:
            if b is None: continue                          # kept fp16: no bytes in the packed file
            q, s = b
            t = q.tobytes(); fp.write(t); h.update(t)
            t = s.tobytes(); fp.write(t); h.update(t)
        size = fp.tell()
    json.dump({"magic": MAGIC_OUT, "version": VERSION, "scheme": "symmetric/per-channel/weight-only-int8",
               "fp16_keep": args.fp16_keep, "source": args.inp,
               "source_precision": "fp16 view of the fp32 master",
               "quantized_params": int(q_params), "kept_fp16_params": int(kept_params),
               "streamed_weight_bytes": int(streamed), "file_bytes": size,
               "sha256": h.hexdigest(), "tensors": meta},
              open(args.out + ".json", "w"), indent=1)
    print(f"[quant] wrote {args.out} ({size/1e6:.1f} MB)  sha256(body)={h.hexdigest()[:16]}...")


if __name__ == "__main__":
    main()
