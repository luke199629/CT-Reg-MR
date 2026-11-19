# PyInstaller spec: single-file, windowed CT/MR Registration Viewer.
# Build with  packaging\build_exe.bat  (or: python -m PyInstaller packaging/ctmr_viewer.spec)
import os

from PyInstaller.utils.hooks import collect_submodules

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))

# pydicom loads pixel decoders / data dictionaries dynamically.
hidden = collect_submodules("pydicom") + collect_submodules("SimpleITK")

# The build env carries a large ML stack the viewer never imports; keep it out explicitly.
excludes = [
    "torch", "torchvision", "torchaudio", "triton", "tensorflow", "jax",
    "matplotlib", "pandas", "IPython", "jupyter", "notebook", "sklearn", "cv2",
    "PIL", "tkinter", "PyQt5", "PyQt6", "sympy", "numba", "pytest",
    "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.Qt3DCore",
    "PySide6.QtQuick", "PySide6.QtQml", "PySide6.QtMultimedia", "PySide6.QtCharts",
    "PySide6.QtDataVisualization", "PySide6.QtPdf", "PySide6.QtNetwork",
]

a = Analysis(
    [os.path.join(ROOT, "run_viewer.py")],
    pathex=[ROOT],
    hiddenimports=hidden,
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    name="CT-Reg-MR",
    console=False,          # GUI app, no console window
    upx=False,
    icon=os.path.join(ROOT, "packaging", "icon.ico") if os.path.exists(
        os.path.join(ROOT, "packaging", "icon.ico")) else None,
)
