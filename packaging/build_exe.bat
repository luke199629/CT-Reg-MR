@echo off
rem Build the CT/MR Registration Viewer with PyInstaller.
rem
rem   packaging\build_exe.bat                    standard: dist\CT-Reg-MR.exe (single file)
rem                                              viewer + rigid + export + elastix deformable (CPU)
rem   packaging\build_exe.bat full               full: dist\CT-Reg-MR-full\CT-Reg-MR-full.exe (folder)
rem                                              adds AI/GPU deformable: ConvexAdam, multiGradICON
rem   packaging\build_exe.bat [full] C:\path\to\python.exe   use a specific interpreter
rem
rem The interpreter's environment must contain everything the build includes:
rem   standard: pyinstaller pyside6 pydicom numpy scipy simpleitk itk-elastix
rem   full:     the above + torch (CUDA build recommended) convexAdam unigradicon
setlocal
rem Capture this script's folder before SHIFT (which also shifts %0).
set HERE=%~dp0
set MODE=
if /i "%~1"=="full" (set MODE=full& shift)
set PY=%~1
if "%PY%"=="" set PY=%LOCALAPPDATA%\anaconda3\envs\pytorch\python.exe
cd /d "%HERE%.."
set CTMR_BUILD=%MODE%
"%PY%" -m PyInstaller --noconfirm --clean --distpath dist --workpath build\%MODE%pyi packaging\ctmr_viewer.spec
if errorlevel 1 exit /b 1
echo.
if /i "%MODE%"=="full" (
  echo Built: %CD%\dist\CT-Reg-MR-full\CT-Reg-MR-full.exe  ^(keep the whole folder together^)
) else (
  echo Built: %CD%\dist\CT-Reg-MR.exe
)
