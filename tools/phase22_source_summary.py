"""Correlate saved NCU instruction counters, without double-counting inlining.

python tools/phase22_source_summary.py docs/phase22
Line mappings are tied to the saved source receipts; samples are NOT exact
phase durations. Instruction counts include all warps in the selected kernel.
"""
import collections
import csv
import json
from pathlib import Path
import sys


def analyze(root, policy):
    main = 'kvcache.cu' if policy == 'original' else 'attention_ordered.cu'
    instructions={}; header=None; file=''; line=None
    path=root/f'ncu_{policy}_source/stdout.txt'
    for row in csv.reader(path.open()):
        if row and row[0]=='File Path': file=row[1]; line=None
        elif row and row[0]=='Line No': header=row
        elif header and len(row)>3:
            if row[0]: line=int(row[0])
            if row[2].startswith('0x'):
                record=dict(zip(header,row)) | {'file':file,'line':line}
                address=int(row[2],16)
                if address not in instructions or file.endswith('/'+main):
                    instructions[address]=record
    assert instructions
    def num(row,key):
        v=row[key]
        return 0 if v in ('','-') else float(v.replace(',',''))
    def phase(row):
        if not row['file'].endswith('/'+main): return 'other_or_inline'
        line=row['line']
        if line==(109 if policy=='original' else 22): return 'qk_dot'
        if (policy=='original' and line==126) or (policy=='ordered4' and 37<=line<=46): return 'value_loop'
        return 'other_or_inline'
    fields=('Warp Stall Sampling (All Samples)','stall_long_sb','stall_short_sb',
            'stall_no_inst','Instructions Executed','L1 Tag Requests Global',
            'L2 Theoretical Sectors Global','L2 Theoretical Sectors Global Excessive')
    groups={}; opcodes=collections.Counter(); value_loads=collections.Counter()
    for address,row in sorted(instructions.items()):
        group=groups.setdefault(phase(row),collections.Counter())
        for field in fields: group[field]+=num(row,field)
        parts=row['Source'].strip().split()
        op=parts[1] if parts[0].startswith('@') else parts[0]
        executions=int(num(row,'Instructions Executed'))
        opcodes[op]+=executions
        if phase(row)=='value_loop' and op.startswith('LDG') and executions:
            value_loads[executions]+=1
    # Cross-check deduplication against the independent whole-kernel counter.
    metrics=list(csv.DictReader((root/f'ncu_{policy}_metrics/stdout.txt').open()))[1]
    total=sum(opcodes.values())
    assert total==int(float(metrics['smsp__inst_executed.sum'].replace(',',''))), (policy,total)
    samples=sum(g['Warp Stall Sampling (All Samples)'] for g in groups.values())
    for g in groups.values(): g['sample_fraction']=g['Warp Stall Sampling (All Samples)']/samples
    return dict(unique_pcs=len(instructions),sample_count=samples,instructions=total,
                phases=groups,opcode_executions=dict(opcodes.most_common()),
                value_load_pc_count_by_executions=value_loads,
                caveat='PC samples are not exact phase times; inline PCs deduplicated by address; source line mappings fixed to receipts')


if __name__=='__main__':
    root=Path(sys.argv[1])
    result={p:analyze(root,p) for p in ('original','ordered4')}
    (root/'source_attribution.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
