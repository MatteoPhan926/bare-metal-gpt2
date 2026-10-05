@echo off
setlocal EnableDelayedExpansion
if defined GPT2_VCVARS (
  call "%GPT2_VCVARS%" >NUL 2>&1
) else (
  call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >NUL 2>&1
)
if errorlevel 1 exit /b 1
if not exist build\phase22 mkdir build\phase22
set FLAGS=-O3 -lineinfo -std=c++17 -arch=sm_89 --default-stream per-thread -DGPT2_ENABLE_GRAPHS -DGPT2_ENABLE_ATTN_V4 -DGPT2_PHASE22 -I model -I cuda
set OBJ=
for %%S in (cuda\kernels_naive.cu cuda\kernels_tiled.cu cuda\kernels_fused.cu cuda\kernels_quant.cu cuda\kvcache.cu cuda\forward_cuda.cu cuda\decode_graph.cu cuda\attention_v4.cu model\weights.c) do (
  nvcc %FLAGS% -c %%S -o build\phase22\%%~nS.obj
  if errorlevel 1 exit /b 1
  set OBJ=!OBJ! build\phase22\%%~nS.obj
)
for %%T in (bench_graph kv_gate graph_gate attention_gate) do (
  nvcc %FLAGS% !OBJ! bench\%%T.cu -o bench\%%T_phase22.exe
  if errorlevel 1 exit /b 1
)
echo PHASE 2.2 BUILD OK
