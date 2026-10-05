# Same 135-node attribution as Phase 2.1. Mode names are retained by the shared
# harness: profile_ordinary selects Original GRAPH, profile_graph selects V4 GRAPH.
param([string]$Output='docs/phase22/profile',
      [string]$Nsys='C:/Program Files/NVIDIA Corporation/Nsight Systems 2024.4.2/target-windows-x64/nsys.exe')
$ErrorActionPreference='Stop'
foreach ($Context in @(128,1023)) {
    foreach ($Policy in @('original','v4')) {
        $Mode=if ($Policy -eq 'original') {'profile_ordinary'} else {'profile_graph'}
        $Run="$Output/${Policy}_$Context"
        & python tools/phase2_run.py $Run -- $Nsys profile --trace=cuda --sample=none --cuda-graph-trace=node --capture-range=cudaProfilerApi --capture-range-end=stop --output "$Run/trace" bench/bench_graph_phase22.exe gemv $Mode 16 $Context
        if ($LASTEXITCODE -ne 0) { throw "Nsight profile failed: $Run" }
        & $Nsys export --type=sqlite --output "$Run/trace.sqlite" "$Run/trace.nsys-rep"
        if ($LASTEXITCODE -ne 0) { throw "Nsight export failed: $Run" }
        & python tools/phase2_timeline.py "$Run/trace.sqlite"
        if ($LASTEXITCODE -ne 0) { throw "Timeline validation failed: $Run" }
    }
}
