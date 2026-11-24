# CT ⟷ MR Registration Viewer (PySide6)

View a CT and an MR DICOM series overlaid in axial / sagittal / coronal planes and apply the
rigid/affine matrix from a DICOM Spatial Registration (REG) object.

## Run

```
C:\Users\kl300\AppData\Local\anaconda3\envs\pytorch\python.exe run_viewer.py
C:\Users\kl300\AppData\Local\anaconda3\envs\pytorch\python.exe run_viewer.py --folder sample_data
C:\Users\kl300\AppData\Local\anaconda3\envs\pytorch\python.exe run_viewer.py --ct CT_DIR --mr MR_DIR --reg REG.dcm
```

### Standalone executable (Windows)

Two builds exist. **The released `CT-Reg-MR.exe` is the standard build.**

| | Standard (`CT-Reg-MR.exe`) | Full (`CT-Reg-MR-full\`) |
|---|---|---|
| Viewer, overlay, rigid registration, export, unload | ✓ | ✓ |
| Deformable: **elastix** (B-spline + mutual information, CPU, classical — no AI) | ✓ | ✓ |
| Deformable: **ConvexAdam** (GPU) | ✗ | ✓ |
| Deformable: **multiGradICON** (AI: pretrained neural network, GPU) | ✗ | ✓ |
| Form | single ~300 MB .exe | folder of several GB (keep it together) |
| Published as a GitHub release | yes | no — build it yourself (below) |

> **The standard build does not include AI / deep-learning deformable image registration**
> (multiGradICON) or the GPU method ConvexAdam. They need PyTorch + CUDA, which is several GB
> — too large for a single-file exe and above GitHub's 2 GB release-file limit. In the standard
> exe these methods are listed as "not installed". Use the full build or run from Python to get them.

Both accept the same `--folder / --ct / --mr / --reg` arguments; you can also drag a data folder
onto the exe. The single-file exe takes a few seconds to start while it unpacks.

#### Build it yourself

From a clone of this repository, in a Command Prompt, with Python 3.10+ on the PATH:

```
python -m venv .venv
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python -m pip install -r requirements.txt pyinstaller
packaging\build_exe.bat full .venv\Scripts\python.exe
```

That produces the **full** build: `dist\CT-Reg-MR-full\CT-Reg-MR-full.exe` (keep the whole
`CT-Reg-MR-full` folder together; copy the folder to move it). The `cu128` index gives a CUDA build
of PyTorch for NVIDIA GPUs; use `https://download.pytorch.org/whl/cpu` instead for a CPU-only
machine (the GPU methods then run on the CPU, slowly).

For the **standard** build, skip the torch line and run `packaging\build_exe.bat` (optionally with
the interpreter path as its argument) → `dist\CT-Reg-MR.exe`.

Run without building — all methods, from Python (install the CUDA torch line above first for GPU speed;
otherwise pip installs a CPU-only torch):

```
python -m pip install -r requirements.txt
python run_viewer.py
```

multiGradICON downloads its pretrained weights (~280 MB, from the uniGradICON GitHub release) on
first use into `%LOCALAPPDATA%\ctmr_reg\weights`. The GPU methods use CUDA when available.

Synthetic test data with a known transform: `python tools/make_sample_data.py sample_data`,
then verify with `python tests/test_registration.py`, `python tests/test_export.py` and
`python tests/test_manual.py` (drives the viewer's manual registration with simulated mouse drags).
For deformable registration, `python tools/make_sample_data.py sample_data_deform --deform` adds a
known smooth non-rigid bump on top of the rigid transform; `python tests/test_deformable.py`
scores each method's displacement field against the ground truth.

## Features
- **Loading**: CT folder, MR folder, REG file — or one folder with everything (auto-detected by
  Modality / SOP Class, recursive). If a folder holds several series you are asked which one.
  *Unload all* (`Ctrl+W`) clears CT, MR, REG, the matrix and deformable results to start the next
  case (it asks first if there are deformable results; window/level and display settings stay).
- **Overlay slider** (bottom): 0 = CT only … 100 = MR only. `Space` flips the emphasis.
  Modes: gray blend, color fusion (MR hot), magenta/green, checkerboard.
- **Layout**: 2×2 — axial (top-left), coordinates panel (top-right), coronal (bottom-left),
  sagittal (bottom-right). Cells are resizable by dragging the dividers. The coordinates panel
  shows the crosshair and mouse position in fixed-image mm, CT and MR patient mm, voxel
  indices and values, and has a "Go to x, y, z" entry.
- **Views**: axial, sagittal, coronal on an axis-aligned grid of the fixed image (correct for
  oblique MR). Radiological orientation (patient right on screen left).
- **Registration**
  - *Matrix maps*: whether the 4×4 matrix maps MR → CT or CT → MR patient coordinates.
    DICOM REG semantics are "registered FoR → reference FoR", which the loader resolves using
    the Frame of Reference UIDs, so with a well-formed REG the default is already correct.
  - *Fixed image*: CT (MR resampled onto CT) or MR (CT resampled onto MR).
  - *Invert matrix*, *Reset to file*, *Identity*; the matrix table is editable; save/load as text.
  - `R` toggles registration on/off, `I` inverts.
  - *Manual registration* (`M` toggles manual mode): in any view, left-drag moves the moving
    image and Ctrl+left-drag rotates it about the crosshair; Shift+click sets the crosshair
    (rotation centre). Buttons nudge along / about the fixed image's x, y, z axes by the chosen
    step (mm / °). Every change goes into the matrix table, keeping its *Matrix maps* direction,
    so it can be saved, exported or used as the starting point for deformable registration.
    A displayed deformable result switches back to the rigid view first.
  - Undo / redo (`Ctrl+Z` / `Ctrl+Y`) covers manual moves, table edits and the matrix buttons.
- **Deformable registration** (panel under *Registration*), starting from the rigid matrix in effect:
  - *Direction*: deform MR → CT (CT fixed) or CT → MR (MR fixed).
  - *Method*:
    - **elastix** — B-spline, Advanced Mattes mutual information + bending-energy penalty (CPU).
    - **ConvexAdam** — MIND-SSC features, coupled convex search + Adam (Siebert & Heinrich), GPU.
    - **multiGradICON** — pretrained multimodal network + instance optimisation with MIND, GPU.
  - Runs on an isotropic working grid (default 2 mm) in the fixed image's orientation, in the
    background (the window stays usable). Method parameters are editable; hover for hints.
  - *Show*: switch the views between *Rigid only* and any result; the fixed image follows the
    result. The coordinates panel then shows the local displacement u and the deformed mapping.
  - Each result reports run time, |u| statistics and the Jacobian determinant range / % folding.
  - *Overlay on views*: colour the displayed result's **Jacobian determinant** (blue < 1
    compression, white = no change, red > 1 expansion, magenta = folding J ≤ 0) or its
    **displacement magnitude |u|** over all three views, with opacity, range (or *Auto*, from
    the 1st–99th percentile) and a colour bar. "Transparent where unchanged" keeps the anatomy
    visible. `J` cycles the overlay; the coordinates panel shows the Jacobian under the
    crosshair and mouse, flagging folding.
  - Convention: a fixed-space point x maps to the moving point `T_rigid(x + u(x))` (ITK
    "fixed → moving" displacement, mm, LPS).
- **Export** (`Ctrl+E` or the *Export (DICOM / RAW)…* button) — tick any combination of:
  - the fixed image, the moving image on its own grid, the moving image rigidly registered onto
    the fixed grid (current matrix);
  - for every deformable result: the deformed moving image (fixed native grid), the displacement
    field (vector .mhd/.raw), the Jacobian determinant, |u|, and the two registration inputs on the
    working grid (fixed, and moving after rigid only).
  - *DICOM*: a new classic series (`<prefix>_…_DICOM/`) with the fixed image's geometry and Frame of
    Reference, patient/study tags from the fixed series, acquisition tags from the source series,
    `ImageType = DERIVED\SECONDARY`, and the matrix recorded in `DerivationDescription`.
  - *RAW*: `<name>.raw` (little-endian, x fastest, then y, then z) plus a `<name>.mhd` MetaImage
    header with size, spacing, origin and direction (LPS), which opens in 3D Slicer, ITK-SNAP or
    Fiji. Data type float32, int16 or uint16.
  - The rigid matrix used is saved alongside as a `.txt` file. Displacement fields are RAW/.mhd only
    (they load in 3D Slicer as a grid transform).
- **Navigation**: mouse wheel / ↑↓ / sliders = scroll slices; left click/drag = crosshair (all views
  follow); Ctrl+wheel = zoom; right or middle drag = pan; right double-click = reset view.
  The status bar shows the mm position and CT HU / MR values under the cursor.

## Limitations
- When *loading* a DICOM REG, only matrix (rigid/affine) registrations are used; a deformable REG's
  vector grid is ignored (only its pre/post matrices are applied, with a warning). Deformable
  results are exported as .mhd displacement fields, not as DICOM Deformable Registration objects.
- Deformable registration of CT and MR is not validated for clinical use. Check results
  (Jacobian / folding, landmarks, contours) before relying on them — see AAPM TG-132.
- The standard .exe includes elastix but **not** the AI/GPU deformable methods (multiGradICON,
  ConvexAdam); see *Standalone executable* for the full build.
- Enhanced multi-frame DICOM is not supported (classic one-file-per-slice series only).
