@echo off
rem Build dist\CT-Reg-MR.exe (single file, no console) with PyInstaller.
rem Uses the "pytorch" conda env by default; pass another python.exe as the first argument.
setlocal
set PY=%~1
if "%PY%"=="" set PY=%LOCALAPPDATA%\anaconda3\envs\pytorch\python.exe
cd /d "%~dp0.."
"%PY%" -m PyInstaller --noconfirm --clean --distpath dist --workpath build packaging\ctmr_viewer.spec
if errorlevel 1 exit /b 1
echo.
echo Built: %CD%\dist\CT-Reg-MR.exe
