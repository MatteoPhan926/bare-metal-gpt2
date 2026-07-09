@echo off
REM build.bat — STAGE 0 (pure-C fp32) build via MSVC x64 under vcvars64.
REM Mirrors the recorded env: MSVC 14.44 (VS2022 BuildTools). cl /openmp = threads only (OpenMP 2.0);
REM the numerics stay /fp:precise (IEEE, no reassociation) so the 1e-4 gate isn't fought by fast-math.
setlocal
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
if errorlevel 1 ( echo vcvars64 failed & exit /b 1 )
if not exist build mkdir build

cl /nologo /O2 /std:c11 /openmp /fp:precise /D_CRT_SECURE_NO_WARNINGS ^
   /I model /I cpu ^
   cpu\forward_cpu.c model\weights.c bench\correctness.c ^
   /Fo"build\\" /Fe"bench\correctness.exe"
if errorlevel 1 ( echo BUILD FAILED & exit /b 1 )
echo BUILD OK -^> bench\correctness.exe
