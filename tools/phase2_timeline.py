"""Nsight node timeline attribution. Profiled timings are NOT speed headlines.

Retains every kernel interval and step decomposition as portable CSV; the large
SQLite/.nsys-rep files can remain local. Never confuses API waiting with GPU work.
"""
import collections
import csv
import json
from pathlib import Path
import sqlite3
import statistics as st
import sys

path=Path(sys.argv[1]); db=sqlite3.connect(f"file:{path.as_posix()}?mode=ro",uri=True)
rows=db.execute("""SELECT k.start,k.end,s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k
    JOIN StringIds s ON k.demangledName=s.id ORDER BY k.start""").fetchall()
assert rows and len(rows)%135==0, f"expected 135 kernels per step, got {len(rows)}"
steps=[]; kinds=collections.defaultdict(float)
for i in range(0,len(rows),135):
    group=rows[i:i+135]
    assert "k_embed_one" in group[0][2] and "k_gemv_fp16" in group[-1][2],"misaligned step"
    assert all(group[j][0]>=group[j-1][1] for j in range(1,135)),"unexpected kernel overlap"
    busy=sum(b-a for a,b,_ in group)/1e6
    span=(group[-1][1]-group[0][0])/1e6
    steps.append(dict(step=i//135,kernel_ms=busy,span_ms=span,internal_gap_ms=span-busy))
    for a,b,name in group: kinds[name]+=(b-a)/1e6
with path.with_name("kernels.csv").open("w",newline="") as f:
    w=csv.writer(f);w.writerow(("start_ns","end_ns","name"));w.writerows(rows)
with path.with_name("steps.csv").open("w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(steps[0]));w.writeheader();w.writerows(steps)
def stat(xs):
    return dict(median=st.median(xs),min=min(xs),max=max(xs))
result=dict(n_steps=len(steps),n_kernels=len(rows),
    timing={key:stat([r[key] for r in steps]) for key in steps[0] if key!="step"},
    kernel_mean_ms_per_step={k:v/len(steps) for k,v in kinds.items()})
apis=db.execute("""SELECT s.value,COUNT(*),SUM(r.end-r.start)/1e6
    FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON r.nameId=s.id GROUP BY s.value""").fetchall()
result["api_count_total_ms"]={name:dict(count=n,total_ms=t) for name,n,t in apis}
path.with_name("timeline.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result,indent=2))
