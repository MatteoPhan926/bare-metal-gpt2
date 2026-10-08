"""Summarize every registered process, including failed acceptance predictions.

python tools/phase22_report.py docs/phase22/ordered
No latency outlier filtering; failures are reported, never discarded.
"""
import csv
import json
from pathlib import Path
import statistics as st
import sys
from phase2_summary import summarize


def load_run(path):
    receipt = json.loads((path / 'receipt.json').read_text())
    if receipt['returncode'] != 0 or (path / 'EXCLUDED.md').exists():
        raise ValueError(f'Incomplete/excluded registered run: {path}')
    rows = summarize(path)
    (path / 'summary.json').write_text(json.dumps(rows, indent=2)+'\n')
    return {(r['experiment'], r['policy'], r['ctx']): r for r in rows}, receipt


def aggregate(data):
    if not all(d.keys() == data[0].keys() for d in data):
        raise ValueError('Process conditions differ')
    out = []
    for key in data[0]:
        rows = [d[key] for d in data]
        r = dict(experiment=key[0], policy=key[1], ctx=key[2], n_each=[a['n'] for a in rows])
        for metric in ('event_ms', 'wall_ms', 'enqueue_ms', 'update_ms'):
            xs = [a[metric]['median'] for a in rows]
            r[metric] = dict(median_of_medians=st.median(xs), run_medians=xs,
                             run_iqrs=[[a[metric]['q1'], a[metric]['q3']] for a in rows],
                             all_sample_min=min(a[metric]['min'] for a in rows),
                             all_sample_max=max(a[metric]['max'] for a in rows))
        out.append(r)
    return out


def main(root):
    result, claims = {}, {}
    for backend in ('fp16', 'int8'):
        loaded = [load_run(root/f'{backend}_run{i}') for i in range(1, 4)]
        data = [d for d, _ in loaded]
        for _, receipt in loaded:
            for source in ('cuda/kvcache.cu', 'cuda/attention_ordered.cu',
                           'cuda/decode_graph.cu', 'bench/bench_graph.cu'):
                if receipt['source_sha256'][source] != loaded[0][1]['source_sha256'][source]:
                    raise ValueError(f'Source drift: {source}')
        result[backend] = aggregate(data)
        report = []
        for exp, metric in (('advancing_forward', 'event_ms'),
                            ('forward_host', 'wall_ms'), ('generation', 'wall_ms')):
            for ctx in (128, 512, 1007):
                a = [d[(exp, 'original', ctx)][metric]['median'] for d in data]
                b = [d[(exp, 'ordered4', ctx)][metric]['median'] for d in data]
                assert all(d[(exp, p, ctx)]['n'] == 256 for d in data for p in ('original', 'ordered4'))
                delta = [100*(1-y/x) for x, y in zip(a, b)]
                row = dict(experiment=exp, metric=metric, ctx=ctx,
                           original_ms=st.median(a), ordered4_ms=st.median(b),
                           original_tps=1000/st.median(a), ordered4_tps=1000/st.median(b),
                           reduction_pct=100*(1-st.median(b)/st.median(a)),
                           paired_process_reductions_pct=delta)
                report.append(row)
                print(backend, exp, ctx, json.dumps(row))
        result[backend+'_comparisons'] = report
        long_gpu = next(r for r in report if r['experiment']=='advancing_forward' and r['ctx']==1007)
        long_wall = next(r for r in report if r['experiment']=='generation' and r['ctx']==1007)
        short = [r for r in report if r['ctx']==128]
        claims[backend] = dict(long_gpu_each_ge8=min(long_gpu['paired_process_reductions_pct'])>=8,
                               long_generation_central_ge5=long_wall['reduction_pct']>=5,
                               short_no_consistent_gt5_regression=all(
                                   not all(v < -5 for v in r['paired_process_reductions_pct']) for r in short))
    if (root/'llama_warm1').exists():
        result['llama_matched'] = aggregate([load_run(root/f'llama_warm{i}')[0] for i in range(1,4)])
        gaps=[]
        for ctx in (128,512,1007):
            ours=next(r for r in result['fp16_comparisons'] if r['experiment']=='forward_host' and r['ctx']==ctx)
            llama=next(r for r in result['llama_matched'] if r['ctx']==ctx)['wall_ms']['median_of_medians']
            a,b=ours['original_ms'],ours['ordered4_ms']
            gaps.append(dict(ctx=ctx,original_ms=a,ordered4_ms=b,llama_ms=llama,
                             old_gap_ms=a-llama,new_gap_ms=b-llama,
                             removed_fraction=(a-b)/(a-llama) if a>llama else None))
        result['matched_gaps']=gaps
        print('matched gaps',json.dumps(gaps))
    result['acceptance_end_to_end']=claims
    result['telemetry_util_ge50']={}
    for directory in root.iterdir():
        if not directory.is_dir() or not (directory/'telemetry.csv').exists(): continue
        rows=list(csv.DictReader((directory/'telemetry.csv').open()))
        complete=[r for r in rows if r.get(' utilization.gpu [%]')]
        active=[r for r in complete if float(r[' utilization.gpu [%]'].split()[0])>=50]
        stats={'partial_rows':len(rows)-len(complete)}
        for field in (' clocks.current.sm [MHz]',' clocks.current.memory [MHz]',' temperature.gpu',' power.draw [W]'):
            xs=[float(r[field].split()[0]) for r in active if field in r]
            if xs: stats[field.strip()]={'median':st.median(xs),'min':min(xs),'max':max(xs)}
        result['telemetry_util_ge50'][directory.name]=stats
    (root/'aggregate.json').write_text(json.dumps(result,indent=2)+'\n')
    print('End-to-end criteria (profile attribution checked separately):',json.dumps(claims))


if __name__ == '__main__':
    main(Path(sys.argv[1]))
