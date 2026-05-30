@echo off
REM Launch HairFastGAN with the MSVC build environment so torch can JIT-compile
REM the StyleGAN2 / inplace_abn CUDA ops on first run. VS is located via vswhere
REM so this works regardless of the Visual Studio install path/edition.
for /f "usebackq tokens=*" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -property installationPath`) do set "VSPATH=%%i"
if exist "%VSPATH%\VC\Auxiliary\Build\vcvars64.bat" call "%VSPATH%\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cd /d "%~dp0"
if "%CUDA_HOME%"=="" set "CUDA_HOME=%CUDA_PATH%"
REM Compile only for the local GPU arch if set (e.g. 8.9 for RTX 40-series) for speed.
if "%TORCH_CUDA_ARCH_LIST%"=="" set "TORCH_CUDA_ARCH_LIST=8.9"
set PYTHONUNBUFFERED=1
python -u run_hairfast.py %*
