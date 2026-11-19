# CT ⟷ MR Registration Viewer (PySide6)

View a CT and an MR DICOM series overlaid in axial / sagittal / coronal planes and apply the
rigid/affine matrix from a DICOM Spatial Registration (REG) object.

## Run

```
C:\Users\kl300\AppData\Local\anaconda3\envs\pytorch\python.exe run_viewer.py
C:\Users\kl300\AppData\Local\anaconda3\envs\pytorch\python.exe run_viewer.py --folder sample_data
C:\Users\kl300\AppData\Local\anaconda3\envs\pytorch\python.exe run_viewer.py --ct CT_DIR --mr MR_DIR --reg REG.dcm
```

Synthetic test data with a known transform: `python tools/make_sample_data.py sample_data`,
then verify with `python tests/test_registration.py` and `python tests/test_export.py`.

## Features
- **Loading**: CT folder, MR folder, REG file — or one folder with everything (auto-detected by
  Modality / SOP Class, recursive). If a folder holds several series you are asked which one.
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
- **Export** (`Ctrl+E` or the *Export (DICOM / RAW)…* button):
  - *Image*: registered moving image (resampled onto the fixed image's native voxel grid with the
    matrix currently in effect), fixed + registered moving, the fixed image alone, or the moving
    image on its original grid.
  - *DICOM*: a new classic series (`<prefix>_…_DICOM/`) with the fixed image's geometry and Frame of
    Reference, patient/study tags from the fixed series, acquisition tags from the source series,
    `ImageType = DERIVED\SECONDARY`, and the matrix recorded in `DerivationDescription`.
  - *RAW*: `<name>.raw` (little-endian, x fastest, then y, then z) plus a `<name>.mhd` MetaImage
    header with size, spacing, origin and direction (LPS), which opens in 3D Slicer, ITK-SNAP or
    Fiji. Data type float32, int16 or uint16.
  - For registered exports the matrix used is saved as `<prefix>_matrix.txt`.
- **Navigation**: mouse wheel / ↑↓ / sliders = scroll slices; left click/drag = crosshair (all views
  follow); Ctrl+wheel = zoom; right or middle drag = pan; right double-click = reset view.
  The status bar shows the mm position and CT HU / MR values under the cursor.

## Limitations
- Only matrix (rigid/affine) registrations are used; a deformable REG's vector grid is ignored
  (only its pre/post matrices are applied, with a warning).
- Enhanced multi-frame DICOM is not supported (classic one-file-per-slice series only).
