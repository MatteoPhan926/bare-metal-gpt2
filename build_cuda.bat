@echo off
REM build_cuda.bat - STAGE 1..5 CUDA build (naive + tiled + flash + int8 + gemv/KV backends).
setlocal
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >NUL 2>&1
if errorlevel 1 ( echo vcvars64 failed & exit /b 1 )

set SRC=cuda\kernels_naive.cu cuda\kernels_tiled.cu cuda\kernels_fused.cu cuda\kernels_quant.cu cuda\kvcache.cu cuda\forward_cuda.cu model\weights.c

nvcc -O3 -arch=sm_89 -I model -I cuda %SRC% bench\correctness_cuda.cu -o bench\correctness_cuda.exe
if errorlevel 1 ( echo CUDA BUILD FAILED - correctness_cuda & exit /b 1 )
nvcc -O3 -arch=sm_89 -I model -I cuda %SRC% bench\profile_forward.cu -o bench\profile_forward.exe
if errorlevel 1 ( echo CUDA BUILD FAILED - profile_forward & exit /b 1 )
nvcc -O3 -arch=sm_89 -I model -I cuda %SRC% bench\ab_forward.cu -o bench\ab_forward.exe
if errorlevel 1 ( echo CUDA BUILD FAILED - ab_forward & exit /b 1 )
nvcc -O3 -arch=sm_89 -I model -I cuda %SRC% bench\eval_ppl_cuda.cu -o bench\eval_ppl_cuda.exe
if errorlevel 1 ( echo CUDA BUILD FAILED - eval_ppl_cuda & exit /b 1 )
nvcc -O3 -arch=sm_89 -I model -I cuda %SRC% bench\kv_gate.cu -o bench\kv_gate.exe
if errorlevel 1 ( echo CUDA BUILD FAILED - kv_gate & exit /b 1 )
nvcc -O3 -arch=sm_89 -I model -I cuda %SRC% bench\bench_decode.cu -o bench\bench_decode.exe
if errorlevel 1 ( echo CUDA BUILD FAILED - bench_decode & exit /b 1 )
nvcc -O3 -arch=sm_89 -I model -I cuda %SRC% bench\profile_decode.cu -o bench\profile_decode.exe
if errorlevel 1 ( echo CUDA BUILD FAILED - profile_decode & exit /b 1 )

REM microbench is standalone (no model sources) and is the ONLY target needing cuBLAS: it uses
REM cublasGemmEx as the measurement instrument for the achievable compute ceilings (ROOFLINE 6/6b).
nvcc -O3 -arch=sm_89 -I model -I cuda bench\microbench.cu -o bench\microbench.exe -lcublas
if errorlevel 1 ( echo CUDA BUILD FAILED - microbench & exit /b 1 )
echo CUDA BUILD OK
