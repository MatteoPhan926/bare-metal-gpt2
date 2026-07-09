@echo off
REM build_profile.bat — STAGE 2 diagnostic build: isolated naive+tiled matmul GEMV profiling harness.
setlocal
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
if errorlevel 1 ( echo vcvars64 failed & exit /b 1 )

nvcc -O3 -arch=sm_89 -I model -I cuda ^
     cuda\kernels_naive.cu cuda\kernels_tiled.cu bench\profile_matmul.cu ^
     -o bench\profile_matmul.exe
if errorlevel 1 ( echo PROFILE BUILD FAILED & exit /b 1 )
echo PROFILE BUILD OK -^> bench\profile_matmul.exe
