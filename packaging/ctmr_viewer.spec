# PyInstaller spec for the CT/MR Registration Viewer.
#
#   Standard build (default) -> dist\CT-Reg-MR.exe   single file, no console
#       viewer, rigid registration, export, deformable registration with elastix (CPU).
#       NO AI/GPU deformable methods (ConvexAdam, multiGradICON).
#
#   Full build (CTMR_BUILD=full) -> dist\CT-Reg-MR-full\CT-Reg-MR-full.exe   folder build
#       everything above + ConvexAdam and multiGradICON (PyTorch, CUDA if available).
#       Several GB, so it is a folder (one-dir) build rather than a single file.
#
# Use packaging\build_exe.bat [full] [python.exe] rather than calling this directly.
import os

from PyInstaller.utils.hooks import collect_all, collect_submodules

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))
FULL = os.environ.get("CTMR_BUILD", "").lower() == "full"
NAME = "CT-Reg-MR-full" if FULL else "CT-Reg-MR"

# pydicom loads pixel decoders / data dictionaries dynamically.
hidden = collect_submodules("pydicom") + collect_submodules("SimpleITK")
datas, binaries = [], []

# ITK (elastix) imports its ~100 compiled wrapper modules lazily by name, so list them all.
hidden += collect_submodules("itk")

if FULL:
    for pkg in ("convexAdam", "unigradicon", "icon_registration", "footsteps", "nibabel"):
        d, b, h = collect_all(pkg)
        datas += d; binaries += b; hidden += h

# The build env carries a large ML stack the viewer never needs; keep it out explicitly.
excludes = [
    "tensorflow", "jax", "pandas", "IPython", "jupyter", "notebook", "sklearn",
    "cv2", "tkinter", "PyQt5", "PyQt6", "numba", "pytest",
    "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.Qt3DCore",
    "PySide6.QtQuick", "PySide6.QtQml", "PySide6.QtMultimedia", "PySide6.QtCharts",
    "PySide6.QtDataVisualization", "PySide6.QtPdf", "PySide6.QtNetwork",
]
if not FULL:
    # AI/GPU deformable back-ends: Python environment or full build only. Excluding them makes
    # the GUI show these methods as "not installed".
    excludes += ["torch", "torchvision", "torchaudio", "triton", "convexAdam", "unigradicon",
                 "icon_registration", "nibabel", "footsteps", "girder_client",
                 # needed only by the AI back-ends (torch -> sympy; icon_registration -> matplotlib, PIL)
                 "sympy", "matplotlib", "PIL"]

icon = os.path.join(ROOT, "packaging", "icon.ico")
icon = icon if os.path.exists(icon) else None

a = Analysis(
    [os.path.join(ROOT, "run_viewer.py")],
    pathex=[ROOT],
    hiddenimports=hidden,
    datas=datas,
    binaries=binaries,
    excludes=excludes,
    noarchive=False,
    # ITK loads its wrapper modules from .py files by path (itk.support.base), so ship sources too.
    module_collection_mode={"itk": "pyz+py"},
)
if not FULL:
    # A hook copies torch's dist-info metadata even though torch is excluded; ITK checks that
    # metadata to decide whether torch is importable, then fails. Drop the orphaned metadata.
    import re
    a.datas = [d for d in a.datas if not re.match(r"(torch|triton|torchvision)-[^\\/]*\.dist-info", d[0])]
pyz = PYZ(a.pure)

if FULL:
    exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name=NAME, console=False, upx=False, icon=icon)
    coll = COLLECT(exe, a.binaries, a.datas, upx=False, name=NAME)
else:
    exe = EXE(pyz, a.scripts, a.binaries, a.datas, name=NAME, console=False, upx=False, icon=icon)
