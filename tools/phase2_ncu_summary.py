"""Extract selected counters WITH UNITS from the saved raw Nsight Compute CSV."""
import csv
import json
from pathlib import Path
import sys
path=Path(sys.argv[1]); rows=list(csv.DictReader(path.open()))
assert len(rows)==2,"expected units row and one kernel"
units,data=rows
names=("Block Size","Grid Size","gpu__time_duration.sum","sm__cycles_elapsed.avg.per_second",
       "launch__registers_per_thread","launch__shared_mem_per_block",
       "launch__shared_mem_per_block_static","launch__shared_mem_per_block_dynamic",
       "launch__waves_per_multiprocessor","sm__warps_active.avg.pct_of_peak_sustained_active",
       "sm__throughput.avg.pct_of_peak_sustained_elapsed","dram__bytes_read.sum",
       "dram__bytes_read.sum.per_second","l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
       "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum")
result={k:dict(value=data[k],unit=units[k]) for k in names}
path.with_name("selected_metrics.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result,indent=2))
