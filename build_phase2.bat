@echo off
setlocal
REM All TUs use the capturable per-thread stream; Phase-1 scripts remain unchanged.
if defined GPT2_VCVARS (
  call "%GPT2_VCVARS%" >NUL 2>&1
) else (
  call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >NUL 2>&1
)
if errorlevel 1 ( echo vcvars64 failed: set GPT2_VCVARS to your vcvars64.bat & exit /b 1 )
set SRC=cuda\kernels_naive.cu cuda\kernels_tiled.cu cuda\kernels_fused.cu cuda\kernels_quant.cu cuda\kvcache.cu cuda\forward_cuda.cu cuda\decode_graph.cu model\weights.c
set FLAGS=-O3 -std=c++17 -arch=sm_89 --default-stream per-thread -DGPT2_ENABLE_GRAPHS -I model -I cuda
for %%T in (graph_gate bench_graph) do (
  nvcc %FLAGS% %SRC% bench\%%T.cu -o bench\%%T.exe
  if errorlevel 1 exit /b 1
)
nvcc %FLAGS% %SRC% bench\kv_gate.cu -o bench\kv_gate_graph.exe
if errorlevel 1 exit /b 1
REM Same new harness/input bytes, original legacy-stream policy, ordinary only.
nvcc -O3 -std=c++17 -arch=sm_89 --default-stream legacy -DGPT2_LEGACY_BENCH -I model -I cuda %SRC% bench\bench_graph.cu -o bench\bench_graph_legacy.exe
if errorlevel 1 exit /b 1
echo PHASE 2 BUILD OK
