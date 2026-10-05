@echo off
setlocal
if not defined LLAMA_DIR ( echo Set LLAMA_DIR to your existing llama.cpp checkout & exit /b 1 )
if defined GPT2_VCVARS (
  call "%GPT2_VCVARS%" >NUL 2>&1
) else (
  call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >NUL 2>&1
)
if errorlevel 1 exit /b 1
cl /nologo /O2 /EHsc /std:c++17 /I "%LLAMA_DIR%\include" /I "%LLAMA_DIR%\ggml\include" bench\llama_phase2.cpp /Febench\llama_phase2.exe /Fobench\llama_phase2.obj /link "%LLAMA_DIR%\build\src\llama.lib"
exit /b %errorlevel%
