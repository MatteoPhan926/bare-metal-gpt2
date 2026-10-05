param([string]$Output='docs/phase22',
      [string]$Ncu='C:/Program Files/NVIDIA Corporation/Nsight Compute 2024.3.0/target/windows-desktop-win7-x64/ncu.exe')
$ErrorActionPreference='Stop'
foreach ($Policy in @('original','ordered4')) {
    $Mode=if ($Policy -eq 'original') {'profile_ordinary'} else {'profile_graph'}
    $Run="$Output/ncu_$Policy"
    & python tools/phase2_run.py $Run -- $Ncu --profile-from-start off --kernel-name regex:k_attn_decode --launch-count 1 --set full --import-source yes --target-processes all --export "$Run/attention" bench/bench_graph_phase22.exe gemv $Mode 16 1023
    if ($LASTEXITCODE -ne 0) { throw "NCU failed: $Run" }
    & python tools/phase2_run.py "${Run}_metrics" -- $Ncu --import "$Run/attention.ncu-rep" --page raw --csv
    if ($LASTEXITCODE -ne 0) { throw "NCU raw export failed" }
    & python tools/phase2_ncu_summary.py "${Run}_metrics/stdout.txt"
    if ($LASTEXITCODE -ne 0) { throw "NCU counter parse failed" }
    & python tools/phase2_run.py "${Run}_source" -- $Ncu --import "$Run/attention.ncu-rep" --page source --print-source cuda,sass --csv
    if ($LASTEXITCODE -ne 0) { throw "NCU source export failed" }
}
