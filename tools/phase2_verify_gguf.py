"""Prove external F16 model provenance; expose (do not hide) scalar precision.

python tools/phase2_verify_gguf.py /path/to/llama.cpp /path/to/gpt2-f16.gguf [new-rounded.gguf]
Optional third path creates a NEW copy with fp32 scalars rounded to engine-half
values, still stored as fp32 for llama compatibility. Originals never modified.
Requires numpy and llama.cpp's gguf-py.
"""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import shutil
import sys
import numpy as np

llama, model = map(Path, sys.argv[1:3])
sys.path.insert(0, str(llama / "gguf-py"))
from gguf import GGUFReader

manifest = json.loads(Path("weights/gpt2_124m_fp32.bin.json").read_text())
reader = GGUFReader(str(model))
tensors = {t.name: t for t in reader.tensors}
mapping = {"wte": "token_embd.weight", "wpe": "position_embd.weight",
           "ln_f.g": "output_norm.weight", "ln_f.b": "output_norm.bias"}
layer_map = {"ln_1.g":"attn_norm.weight", "ln_1.b":"attn_norm.bias",
             "ln_2.g":"ffn_norm.weight", "ln_2.b":"ffn_norm.bias",
             "attn.c_attn.w":"attn_qkv.weight", "attn.c_attn.b":"attn_qkv.bias",
             "attn.c_proj.w":"attn_output.weight", "attn.c_proj.b":"attn_output.bias",
             "mlp.c_fc.w":"ffn_up.weight", "mlp.c_fc.b":"ffn_up.bias",
             "mlp.c_proj.w":"ffn_down.weight", "mlp.c_proj.b":"ffn_down.bias"}
counts = {}; used=set(); rounded_differences=0
for item in manifest["tensors"]:
    name = item["name"]
    if name not in mapping:
        m=re.fullmatch(r"h(\d+)\.(.+)",name)
        target=f"blk.{m[1]}.{layer_map[m[2]]}"
    else:
        target=mapping[name]
    tensor=tensors[target]; used.add(target)
    source=np.memmap("weights/gpt2_124m_fp32.bin",mode="r",dtype=np.float32,
                     offset=item["offset"],shape=tuple(item["shape"]))
    assert tensor.data.shape==source.shape, (target,tensor.data.shape,source.shape)
    assert tensor.data.dtype in (np.float16,np.float32), target
    assert np.array_equal(tensor.data, source.astype(tensor.data.dtype)), target
    dtype=str(tensor.data.dtype); counts[dtype]=counts.get(dtype,0)+source.size
    if tensor.data.dtype==np.float32:
        rounded_differences+=int(np.count_nonzero(source!=source.astype(np.float16).astype(np.float32)))
assert used==set(tensors), set(tensors)-used
def sha(path):
    with open(path,"rb") as f: return hashlib.file_digest(f,"sha256").hexdigest()
rounded=None
if len(sys.argv)>3:
    destination=Path(sys.argv[3])
    with open(model,"rb") as src,open(destination,"xb") as dst:
        shutil.copyfileobj(src,dst)
    copied=GGUFReader(str(destination),mode="r+")
    for t in copied.tensors:
        if t.data.dtype==np.float32:
            t.data[:]=t.data.astype(np.float16).astype(np.float32)
        assert np.array_equal(t.data.astype(np.float32),
                              tensors[t.name].data.astype(np.float16).astype(np.float32)),t.name
    copied.data.flush()
    rounded=dict(path=str(destination),sha256=sha(destination),
                 all_weight_values_equal_engine_half=True)
print(json.dumps(dict(status="PASS", tensors=len(used), scalars_by_storage=counts,
    rounded_copy=rounded,
    fp32_scalars_differing_from_engine_half_rounding=rounded_differences,
    gguf_sha256=sha(model), engine_fp32_sha256=sha("weights/gpt2_124m_fp32.bin"),
    llama_revision=subprocess.check_output(["git","-C",str(llama),"rev-parse","HEAD"],text=True).strip(),
    llama_diff=subprocess.check_output(["git","-C",str(llama),"diff","--stat"],text=True),
    runtime_sha256={p.name:sha(p) for p in (llama/"build/bin").glob("*.dll")
                    if p.name.startswith(("llama", "ggml"))}),indent=2))
