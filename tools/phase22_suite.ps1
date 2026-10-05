param([ValidateSet('gates','bench')][string]$Stage='gates',
      [string]$Output='docs/phase22/ordered', [string]$LlamaDir='')
$ErrorActionPreference='Stop'
function Run-Saved([string]$Name, [string[]]$Command) {
    & python tools/phase2_run.py "$Output/$Name" -- @Command
    if ($LASTEXITCODE -ne 0) { throw "Failed: $Name (raw evidence retained)" }
}
if ($Stage -eq 'gates') {
    Run-Saved 'isolated' @('bench/attention_gate_phase22.exe','isolated')
    foreach ($Backend in @('gemv','int8')) {
        Run-Saved "exact_original_$Backend" @('bench/graph_gate_phase22.exe',$Backend)
        Run-Saved "cross_policy_$Backend" @('bench/attention_gate_phase22.exe',$Backend)
    }
    $PreviousBackend=$env:GPT2_BACKEND; $PreviousDecode=$env:GPT2_DECODE; $PreviousAttention=$env:GPT2_ATTENTION
    try {
        $env:GPT2_DECODE='graph'
        foreach ($Backend in @('gemv','int8')) {
            $env:GPT2_BACKEND=$Backend
            foreach ($Attention in @('original','ordered4')) {
                $env:GPT2_ATTENTION=$Attention
                Run-Saved "hf_${Backend}_$Attention" @('bench/kv_gate_phase22.exe')
            }
        }
    } finally {
        $env:GPT2_BACKEND=$PreviousBackend; $env:GPT2_DECODE=$PreviousDecode; $env:GPT2_ATTENTION=$PreviousAttention
    }
} else {
    # Do not allow publication timings without the recorded correctness gates.
    foreach ($Name in @('isolated','exact_original_gemv','exact_original_int8',
                        'cross_policy_gemv','cross_policy_int8','hf_gemv_original',
                        'hf_gemv_ordered4','hf_int8_original','hf_int8_ordered4')) {
        $Receipt=Get-Content "$Output/$Name/receipt.json" -Raw | ConvertFrom-Json
        if ($Receipt.returncode -ne 0) { throw "Correctness gate missing/failed: $Name" }
        foreach ($File in $Receipt.source_sha256.psobject.Properties) {
            if ($File.Name -match '\.(cu|cuh|c|h)$') {
                $Actual=(Get-FileHash -Algorithm SHA256 -LiteralPath $File.Name).Hash.ToLower()
                if ($Actual -ne $File.Value) { throw "Source changed since correctness gate: $($File.Name)" }
            }
        }
    }
    for ($Run=1; $Run -le 3; $Run++) {
        # Swap precision order across processes; policy order also alternates inside each run.
        $Backends=if ($Run -eq 2) { @('int8','gemv') } else { @('gemv','int8') }
        foreach ($Backend in $Backends) {
            $Label=if ($Backend -eq 'gemv') {'fp16'} else {'int8'}
            Run-Saved "${Label}_run$Run" @('bench/bench_graph_phase22.exe',$Backend,'all','256')
        }
        if ($LlamaDir) {
            $PreviousPath=$env:PATH
            $env:PATH="$LlamaDir/build/bin;$env:PATH"
            try {
            Run-Saved "llama_warm$Run" @('bench/llama_phase2.exe','weights/gpt2_llama_matched.gguf','256')
            } finally { $env:PATH=$PreviousPath }
        }
    }
}
