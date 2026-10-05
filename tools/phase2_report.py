"""Aggregate ALL three registered runs, not the fastest one. Run after suite.

python tools/phase2_report.py docs/phase2
Writes aggregate.json; spread includes both run medians and all sample extrema.
"""
import csv
import json
from pathlib import Path
import statistics as st
import sys
from phase2_summary import summarize

root=Path(sys.argv[1]); result={}
for backend in ("fp16","int8","llama_matched"):
    data=[]
    for i in range(1,4):
        directory=root/(f"llama_warm{i}" if backend=="llama_matched" else f"{backend}_run{i}")
        receipt=json.loads((directory/"receipt.json").read_text())
        assert receipt["returncode"]==0, directory
        assert not (directory/"EXCLUDED.md").exists(),directory
        rows=summarize(directory)
        (directory/"summary.json").write_text(json.dumps(rows,indent=2)+"\n")
        data.append({(r["experiment"],r["policy"],r["ctx"]):r for r in rows})
    assert all(d.keys()==data[0].keys() for d in data)
    aggregate=[]
    for key in data[0]:
        group=[d[key] for d in data]
        row=dict(experiment=key[0],policy=key[1],ctx=key[2],n_each=[r["n"] for r in group])
        for metric in ("event_ms","wall_ms","enqueue_ms","update_ms"):
            medians=[r[metric]["median"] for r in group]
            row[metric]=dict(median_of_medians=st.median(medians),run_medians=medians,
                all_sample_min=min(r[metric]["min"] for r in group),
                all_sample_max=max(r[metric]["max"] for r in group))
        aggregate.append(row)
    result[backend]=aggregate
    if backend!="llama_matched":
        for ctx in (128,512,1007):
            percentages=[100*(1-d[("generation","graph",ctx)]["wall_ms"]["median"]/
                                d[("generation","ordinary",ctx)]["wall_ms"]["median"]) for d in data]
            assert min(percentages)>=5, (backend,ctx,"acceptance failed",percentages)
            print(backend,ctx,"generation latency reductions %",percentages)
        for metric in ("advancing_forward","fixed_single"):
            savings=[d[(metric,"ordinary",128)]["event_ms"]["median"]-
                     d[(metric,"graph",128)]["event_ms"]["median"] for d in data]
            print(backend,metric,"short event savings ms",savings)

def get(backend,exp,policy,ctx):
    return next(r for r in result[backend] if (r["experiment"],r["policy"],r["ctx"])==(exp,policy,ctx))
gaps=[]
for ctx in (128,512,1007):
    a=get("fp16","forward_host","ordinary",ctx)["wall_ms"]["median_of_medians"]
    b=get("fp16","forward_host","graph",ctx)["wall_ms"]["median_of_medians"]
    c=get("llama_matched","forward_host","llama_f16",ctx)["wall_ms"]["median_of_medians"]
    row=dict(ctx=ctx,ordinary_ms=a,graph_ms=b,llama_ms=c,
             old_gap_ms=a-c,remaining_gap_ms=b-c,removed_fraction=(a-b)/(a-c),
             graph_tps=1000/b,llama_tps=1000/c)
    gaps.append(row); print("Matched host-logits gap",row)
result["matched_gaps"]=gaps
telemetry={}
for d in root.iterdir():
    if not d.is_dir() or not (d/"telemetry.csv").exists():continue
    rows=list(csv.DictReader((d/"telemetry.csv").open()))
    if not rows:continue
    # This is a load proxy, not exact alignment with each timed GPU event.
    # Terminating the telemetry subprocess can leave a partial final CSV line.
    # Account for it explicitly; this NEVER filters benchmark latency samples.
    complete=[r for r in rows if r.get(" utilization.gpu [%]")]
    partial=len(rows)-len(complete)
    rows=[r for r in complete if float(r[" utilization.gpu [%]"].strip().split()[0])>=50]
    stats={}
    for field in (" clocks.current.sm [MHz]"," clocks.current.memory [MHz]"," temperature.gpu"," power.draw [W]"):
        if rows and field in rows[0]:
            xs=[float(r[field].strip().split()[0]) for r in rows]
            stats[field.strip()]=dict(median=st.median(xs),min=min(xs),max=max(xs))
    stats["partial_telemetry_rows"]=partial
    telemetry[d.name]=stats
result["telemetry_util_ge50"]=telemetry
(root/"aggregate.json").write_text(json.dumps(result,indent=2)+"\n")
