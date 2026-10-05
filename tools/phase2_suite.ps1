# Serial GPU runs only. Requires .\build_phase2.bat and original input artifacts.
param([string]$Output = 'docs/phase2/reproduction', [string]$LlamaDir = '',
      [string]$LlamaModel = 'weights/gpt2_llama_matched.gguf')
$ErrorActionPreference = 'Stop'
function Run-Recorded([string]$Name, [string[]]$Command) {
    & python tools/phase2_run.py "$Output/$Name" -- @Command
    if ($LASTEXITCODE -ne 0) { throw "Failed: $Name" }
}
Run-Recorded 'exact_fp16' @('bench/graph_gate.exe','gemv')
Run-Recorded 'exact_int8' @('bench/graph_gate.exe','int8')
$env:GPT2_DECODE = 'graph'
foreach ($Backend in @('gemv','int8')) {
    $env:GPT2_BACKEND = $Backend
    Run-Recorded "hf_$Backend" @('bench/kv_gate_graph.exe')
}
Remove-Item Env:GPT2_DECODE
Remove-Item Env:GPT2_BACKEND
Run-Recorded 'bandwidth' @('bench/microbench.exe','bw')
Run-Recorded 'legacy_fp16' @('bench/bench_graph_legacy.exe','gemv','fixed','256')
foreach ($Iteration in 1..3) {
    Run-Recorded "fp16_run$Iteration" @('bench/bench_graph.exe','gemv','all','256')
    if ($LlamaDir) {
        $SavedPath = $env:PATH
        try {
            $env:PATH = "$LlamaDir/build/bin;$SavedPath"
            Run-Recorded "llama_warm$Iteration" @('bench/llama_phase2.exe',$LlamaModel,'256')
        } finally { $env:PATH = $SavedPath }
    }
    Run-Recorded "int8_run$Iteration" @('bench/bench_graph.exe','int8','all','256')
}
Get-ChildItem -LiteralPath $Output -Directory | ForEach-Object {
    if (Select-String -LiteralPath "$($_.FullName)/stdout.txt" -Pattern '^sample,' -Quiet) {
        & python tools/phase2_summary.py $_.FullName
        if ($LASTEXITCODE -ne 0) { throw "Summary failed: $_" }
    }
}
