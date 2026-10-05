"""Summarize all retained samples: median, min/max, quartiles; never best-of-N.

python tools/phase2_summary.py docs/phase2/fp16_run1 [more run directories]
Setup rows use event=capture, wall=total, enqueue=instantiate, update=upload.
Batch16 rows are batch averages, NOT independent individual-token samples.
"""
import csv
import io
import json
from pathlib import Path
import statistics as st
import sys


def summarize(directory):
    text = (directory / "stdout.txt").read_text()
    lines = [line for line in text.splitlines() if line.startswith("sample,")]
    if not lines:
        raise ValueError(f"No sample rows in {directory}")
    groups = {}
    for row in csv.DictReader(io.StringIO("\n".join(lines))):
        key = (row["experiment"], row["policy"], int(row["ctx"]))
        groups.setdefault(key, []).append(row)
    result = []
    for (experiment, policy, ctx), rows in groups.items():
        item = dict(experiment=experiment, policy=policy, ctx=ctx, n=len(rows))
        for metric in ("event_ms", "wall_ms", "enqueue_ms", "update_ms"):
            values = [float(r[metric]) for r in rows]
            q = st.quantiles(values, n=4) if len(values)>1 else values*3
            item[metric] = dict(median=st.median(values), min=min(values), max=max(values), q1=q[0], q3=q[2])
        result.append(item)
    return result


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        directory = Path(arg)
        result = summarize(directory)
        (directory / "summary.json").write_text(json.dumps(result, indent=2)+"\n")
        print(f"\n{directory}: median [min,max] milliseconds; n is rows, not launches")
        for r in result:
            e,w = r["event_ms"],r["wall_ms"]
            print(f'{r["experiment"]:20s} {r["policy"]:24s} ctx={r["ctx"]:4d} n={r["n"]:4d}'
                  f' event={e["median"]:.4f} [{e["min"]:.4f},{e["max"]:.4f}]'
                  f' wall={w["median"]:.4f} [{w["min"]:.4f},{w["max"]:.4f}]')
