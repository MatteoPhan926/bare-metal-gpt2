# Nsight Compute exports

Authentic `ncu` output, Nsight Compute 2024.3.0, NVIDIA GeForce RTX 4060 Laptop GPU (CC 8.9), 2026-07-09.
Full discussion: `BENCHMARKS.md` → *"ncu — the two owed direct measurements"*.

The `.ncu-rep` binary reports are gitignored (local only). These text files are the durable record.
Regeneration commands are in that BENCHMARKS.md section.

| file | contents |
|---|---|
| `head_gemv_sol.txt` | `k_gemv_fp16` head GEMV (N=50257, K=768) — GPU Speed Of Light + Memory Workload Analysis |
| `m1_coalescing.txt` | `k_matmul` (naive) vs `k_matmul_tiled` at M=1, N=2304, K=768 — Memory Workload Analysis |

## Derived: global-load sectors per request

Sectors/request is **not** a stock ncu section metric on this version; it is derived from the raw counters
(`ncu --import <rep> --page raw --csv`):

```
sectors/request    = l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum
                   / l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum

global-load eff.   = smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.ratio / 32
```

```
                                  sectors    requests   sec/req   bytes/sector    eff
  M=1, N=2304, K=768 (qkv)
    k_matmul       (naive)      1,824,917     110,664    16.491      2.002 B     6.26%
    k_matmul_tiled                117,644      62,352     1.887     16.495 B    51.55%

  M=1, N=50257, K=768 (head)
    k_matmul       (naive)                              16.486      2.002 B     6.26%
    k_matmul_tiled                                       1.887     16.494 B    51.54%
    k_gemv_fp16    (Stage 5)                             3.333     32.000 B   100.00%
```

Ideal for a 32-lane warp of contiguous fp16 is 2.0 sectors/request at 32 B/sector. Naive's **2.002 bytes
per 32-byte sector is exactly one fp16 per sector** — its `W` stride is `K*2 = 1536 B`, far wider than a
sector, so every lane's load lands in a sector of its own. `k_gemv_fp16` uses `__half2` vector loads
(32 lanes × 4 B = 128 B = 4 sectors/request ideal) and hits 100% efficiency.
