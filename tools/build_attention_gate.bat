@echo off
setlocal EnableDelayedExpansion
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >NUL 2>&1
if errorlevel 1 exit /b 1
set OBJ=
for %%S in (kernels_naive kernels_tiled kernels_fused kernels_quant kvcache forward_cuda decode_graph attention_v4 weights) do set OBJ=!OBJ! build\phase22\%%S.obj
nvcc -O3 -lineinfo -std=c++17 -arch=sm_89 --default-stream per-thread -DGPT2_ENABLE_GRAPHS -DGPT2_ENABLE_ATTN_V4 -DGPT2_PHASE22 -I model -I cuda !OBJ! bench\attention_gate.cu -o bench\attention_gate_phase22.exe
exit /b %errorlevel%
