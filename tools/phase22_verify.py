"""Fail closed on incomplete runs, code/input drift or unmatched timed workloads."""
import csv
import io
import json
from pathlib import Path
import subprocess
import sys

root=Path(sys.argv[1] if len(sys.argv)>1 else 'docs/phase22/ordered')
gates=['isolated','exact_original_gemv','exact_original_int8','cross_policy_gemv',
       'cross_policy_int8','hf_gemv_original','hf_gemv_ordered4',
       'hf_int8_original','hf_int8_ordered4']
bench=[f'{b}_run{i}' for b in ('fp16','int8') for i in range(1,4)]
receipts={n:json.loads((root/n/'receipt.json').read_text()) for n in gates+bench}
for name,r in receipts.items():
    assert r['returncode']==0, name
    for source in ('cuda/attention_ordered.cu','cuda/attention_v4.cu',
                   'cuda/kvcache.cu','cuda/decode_graph.cu','bench/bench_graph.cu'):
        assert r['source_sha256'][source]==receipts['isolated']['source_sha256'][source], (name,source)
    assert r['input_sha256']==receipts['isolated']['input_sha256'], name
for name in bench:
    lines=[s for s in (root/name/'stdout.txt').read_text().splitlines() if s.startswith('sample,')]
    rows=list(csv.DictReader(io.StringIO('\n'.join(lines))))
    for exp in ('advancing_forward','forward_host','generation'):
        for ctx in (128,512,1007):
            groups=[]
            for policy in ('original','ordered4'):
                group=[r for r in rows if r['experiment']==exp and r['policy']==policy and int(r['ctx'])==ctx]
                assert len(group)==256,(name,exp,ctx,policy)
                assert {(int(r['rep']),int(r['pos'])) for r in group}=={
                    (rep,pos) for rep in range(16) for pos in range(ctx,ctx+16)}
                assert all(float(r['event_ms'])>0 and float(r['wall_ms'])>0 for r in group)
                groups.append({(r['rep'],r['pos']):r['token'] for r in group})
            assert groups[0]==groups[1], (name,exp,ctx,'trajectory differs')
    assert 'trajectory_difference' not in (root/name/'stdout.txt').read_text()
for name,commit in (('PHASE22_PLAN.md','19fa7d1'),('PHASE22_PLAN_ORDERED.md','3ca7eef')):
    frozen=subprocess.check_output(['git','show',f'{commit}:{name}']).decode()
    assert Path(name).read_text(encoding='utf-8')==frozen, 'preregistration modified'
old_kernel=subprocess.check_output(['git','show','9c8624e:cuda/kvcache.cu'])
assert Path('cuda/kvcache.cu').read_text(encoding='utf-8')==old_kernel.decode(), 'accepted kernel changed'
result=dict(status='PASS',gates=len(gates),benchmark_processes=len(bench),
            decode_samples=6*3*3*2*256,all_trajectories_match=True,
            preregistrations_unchanged=True,accepted_kernel_unchanged=True,
            benchmark_executable_sha256=sorted({receipts[n]['executable_sha256'] for n in bench}))
print(json.dumps(result,indent=2))
